"""Lossy compression ON TOP OF a KV baseline (F16, Q8_0, Q4_0): size vs base and error (CPU, real KV).

Error metric: relative Frobenius error vs the ORIGINAL F16 values (so base and base+ours are on one
scale), mean over analysed layers. KL/top-1 need a model run with a custom KV type and are not measured.
Variants (bits per element include all scales, mins, outlier side lists):
  F16 base:  drop2 / drop4  (truncate 2 / 4 mantissa bits)                      14 / 12 b
  Q8_0 base: q7       (codes rounded to even = 7-bit codes, same scale)          7 + 0.5 = 7.5 b
             q6_out1  (1% largest |x| of the layer kept exact as side list [16-bit idx + f16 = 32 b each];
                       dense part requantized to 6-bit symmetric per 32)        6.5 + 0.32 b
             q8_chanK (K only: per-channel scale over 32 tokens instead of per-32-values row blocks) 8.5 b
  Q4_0 base: q4_out1  (1% outliers side list + dense q4_0)                       4.5 + 0.32 b
             q3_out1  (1% outliers + dense 3-bit symmetric per 32)               3.5 + 0.32 b
             kivi4    (K per-channel asym 4-bit over 32 tokens, V per-token asym 4-bit)  5.0 b
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def sym(x, bits):
    b = x.reshape(-1, 32)
    qmax = 2 ** (bits - 1) - 1
    d = (np.abs(b).max(1, keepdims=True) / qmax).astype(np.float16).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d > 0, np.clip(np.round(b / d), -qmax - 1, qmax), 0)
    return (q * d).reshape(x.shape)


def q8_codes(x):
    b = x.reshape(-1, 32)
    d = (np.abs(b).max(1, keepdims=True) / 127).astype(np.float16).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d > 0, np.round(b / d), 0)
    return q, d


def q4_0(x):
    b = x.reshape(-1, 32)
    i = np.abs(b).argmax(1)
    mx = b[np.arange(len(b)), i][:, None]
    d = (mx / -8.0).astype(np.float16).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d != 0, np.clip(np.floor(b / d + 8.5), 0, 15), 8)
    return ((q - 8) * d).reshape(x.shape)


def asym(b, bits):
    lo, hi = b.min(1, keepdims=True), b.max(1, keepdims=True)
    lo16 = lo.astype(np.float16).astype(np.float32)
    d = ((hi - lo) / (2 ** bits - 1)).astype(np.float16).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d > 0, np.clip(np.round((b - lo16) / d), 0, 2 ** bits - 1), 0)
    return q * d + lo16


def per_channel(x, fn):
    T, C = x.shape
    Tt = T - T % 32
    y = x.copy()
    g = x[:Tt].reshape(Tt // 32, 32, C).transpose(0, 2, 1).reshape(-1, 32)
    y[:Tt] = fn(g).reshape(Tt // 32, C, 32).transpose(0, 2, 1).reshape(Tt, C)
    return y


def with_outliers(x, dense_fn, p=1.0):
    thr = np.percentile(np.abs(x), 100 - p)
    m = np.abs(x) > thr
    y = dense_fn(np.where(m, 0, x))
    return np.where(m, x, y)


def drop(x16, k):
    w = x16.view(np.uint16)
    return (w & np.uint16(0xFFFF ^ ((1 << k) - 1))).view(np.float16).astype(np.float32)


def rel(x, y):
    return float(np.linalg.norm(x - y) / np.linalg.norm(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--layer-stride", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit("refusing to overwrite")
    raw = np.memmap(a.state, dtype=np.uint8, mode="r")
    _, _, _, T = struct.unpack_from("<IiII", raw, 0)
    off = 16 + 12 * T
    _, L = struct.unpack_from("<II", raw, off)
    off += 8
    acc = {}
    for comp in ("K", "V"):
        for li in range(L):
            _, rb = struct.unpack_from("<iQ", raw, off)
            off += 12
            x16 = np.frombuffer(raw[off:off + T * rb], dtype=np.float16).reshape(T, rb // 2)
            off += T * rb
            if li % a.layer_stride:
                continue
            x = x16.astype(np.float32)
            q, d = q8_codes(x)
            base8 = (q * d).reshape(x.shape)
            v = {
                ("F16", "base"): (x, 16.0), ("F16", "drop2"): (drop(x16, 2), 14.0), ("F16", "drop4"): (drop(x16, 4), 12.0),
                ("Q8_0", "base"): (base8, 8.5),
                ("Q8_0", "q7 (codes to 7 bit)"): (((np.round(q / 2) * 2).clip(-127, 127) * d).reshape(x.shape), 7.5),
                ("Q8_0", "q6 + 1% outliers"): (with_outliers(x, lambda z: sym(z, 6)), 6.5 + 0.32),
                ("Q4_0", "base"): (q4_0(x), 4.5),
                ("Q4_0", "q4 + 1% outliers"): (with_outliers(x, q4_0), 4.5 + 0.32),
                ("Q4_0", "q3 + 1% outliers"): (with_outliers(x, lambda z: sym(z, 3)), 3.5 + 0.32),
            }
            if comp == "K":
                v[("Q8_0", "per-channel scale (K)")] = (per_channel(x, lambda g: sym(g, 8)), 8.5)
                v[("Q4_0", "kivi4: per-channel K")] = (per_channel(x, lambda g: asym(g, 4)), 5.0)
            else:
                v[("Q4_0", "kivi4: per-token V")] = (asym(x.reshape(-1, 32), 4).reshape(x.shape), 5.0)
            for (base, name), (y, bits) in v.items():
                r = acc.setdefault(base, {}).setdefault(name, {}).setdefault(comp, {"bits": bits, "err": []})
                r["err"].append(rel(x, y))
    out = {"tokens": T, "layer_stride": a.layer_stride, "metric": "relative Frobenius error vs original F16", "bases": {}}
    for base, d in acc.items():
        for name, cs in d.items():
            for comp, r in cs.items():
                be = np.mean(acc[base]["base"][comp]["err"])
                out["bases"].setdefault(base, {}).setdefault(name, {})[comp] = {
                    "bits_per_element": r["bits"], "size_vs_base": r["bits"] / acc[base]["base"][comp]["bits"],
                    "rel_err": float(np.mean(r["err"])), "err_vs_base_err": (float(np.mean(r["err"])) / be) if be > 0 else None}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("| base | variant | comp | bits | size vs base | rel err | err vs base err |\n|---|---|---|---|---|---|---|")
    for base, d in out["bases"].items():
        for name, cs in d.items():
            for comp, r in cs.items():
                evb = "-" if r["err_vs_base_err"] is None else f"{r['err_vs_base_err']:.2f}x"
                print(f"| {base} | {name} | {comp} | {r['bits_per_element']:.2f} | {r['size_vs_base']:.3f} | {r['rel_err']:.4f} | {evb} |")


if __name__ == "__main__":
    main()
