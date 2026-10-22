"""Offline reconstruction error of lossy KV formats on a real F16 KV capture (CPU, read-only).

Formats (bits/element include scale/min storage as f16 unless noted):
  q8_0     32-elem blocks along the row, absmax scale            8.5 b
  q5_sym   5-bit symmetric absmax per 32 (close to, not identical to, llama.cpp q5_0)  5.5 b
  q4_0     llama.cpp 4-bit symmetric (max-magnitude / -8)        4.5 b
  q4_1     4-bit asymmetric min/scale per 32                     5.0 b
  q3_sym   3-bit symmetric per 32 (not a llama.cpp KV type)      3.5 b
  kivi4    4-bit asymmetric: K per-channel over groups of 32 tokens, V per-token over 32 dims  5.0 b
  kivi2    same with 2 bits                                      3.0 b
  lowrank  rank-r SVD of each layer's T x 1024 matrix, factors f16 (bits depend on r and T)
Metric: relative error ||X - X^||_F / ||X||_F and SNR in dB, per K and V, averaged over layers.
This is a PROXY: it does not measure KL or task accuracy; only native llama.cpp types get KL elsewhere.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def blocks(x, n=32):
    return x.reshape(-1, n)


def q_sym(x, bits):
    b = blocks(x)
    qmax = 2 ** (bits - 1) - 1
    d = np.abs(b).max(1, keepdims=True) / qmax
    d = d.astype(np.float16).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d > 0, np.round(b / d), 0)
    return (np.clip(q, -qmax - 1, qmax) * d).reshape(x.shape)


def q4_0(x):
    b = blocks(x)
    i = np.abs(b).argmax(1)
    mx = b[np.arange(len(b)), i][:, None]
    d = (mx / -8.0).astype(np.float16).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d != 0, np.clip(np.floor(b / d + 8.5), 0, 15), 8)
    return ((q - 8) * d).reshape(x.shape)


def q_asym(b, bits):
    lo, hi = b.min(1, keepdims=True), b.max(1, keepdims=True)
    lo16 = lo.astype(np.float16).astype(np.float32)
    d = ((hi - lo) / (2 ** bits - 1)).astype(np.float16).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d > 0, np.clip(np.round((b - lo16) / d), 0, 2 ** bits - 1), 0)
    return q * d + lo16


def kivi(x, bits, per_channel):
    T, C = x.shape
    Tt = T - T % 32
    y = x.copy()
    if per_channel:  # groups of 32 tokens, one scale per channel
        g = x[:Tt].reshape(Tt // 32, 32, C).transpose(0, 2, 1).reshape(-1, 32)
        y[:Tt] = q_asym(g, bits).reshape(Tt // 32, C, 32).transpose(0, 2, 1).reshape(Tt, C)
    else:  # per token, groups of 32 channels
        y = q_asym(x.reshape(-1, 32), bits).reshape(T, C)
    return y  # tail tokens (T % 32) kept exact, as in KIVI's residual window


def err(x, y):
    e = np.linalg.norm(x - y) / np.linalg.norm(x)
    return float(e), float(-20 * np.log10(e)) if e > 0 else float("inf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--ranks", default="128,256,512")
    ap.add_argument("--layer-stride", type=int, default=1, help="analyse every n-th layer (K and V)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite")
    raw = np.memmap(args.state, dtype=np.uint8, mode="r")
    _, _, _, T = struct.unpack_from("<IiII", raw, 0)
    off = 16 + 12 * T
    _, L = struct.unpack_from("<II", raw, off)
    off += 8
    ranks = [int(r) for r in args.ranks.split(",")]
    acc = {}
    for comp in ("K", "V"):
        for li in range(L):
            _, rb = struct.unpack_from("<iQ", raw, off)
            off += 12
            x = np.frombuffer(raw[off:off + T * rb], dtype=np.float16).reshape(T, rb // 2).astype(np.float32)
            off += T * rb
            if li % args.layer_stride:
                continue
            C = x.shape[1]
            fmts = {"q8_0": (q_sym(x, 8), 8.5), "q5_sym": (q_sym(x, 5), 5.5), "q4_0": (q4_0(x), 4.5),
                    "q4_1": (q_asym(x.reshape(-1, 32), 4).reshape(x.shape), 5.0), "q3_sym": (q_sym(x, 3), 3.5),
                    "kivi4": (kivi(x, 4, comp == "K"), 5.0), "kivi2": (kivi(x, 2, comp == "K"), 3.0)}
            u, s, vt = np.linalg.svd(x, full_matrices=False)
            for r in ranks:
                fmts[f"lowrank{r}"] = ((u[:, :r] * s[:r]) @ vt[:r], 16.0 * r * (T + C) / (T * C))
            for name, (y, bits) in fmts.items():
                e, snr = err(x, y)
                a = acc.setdefault(comp, {}).setdefault(name, {"bits": bits, "rel_err": [], "snr_db": []})
                a["rel_err"].append(e)
                a["snr_db"].append(snr)
    out = {"tokens": T, "layers": L, "metric": "relative Frobenius error, mean over layers", "formats": {}}
    for comp, d in acc.items():
        for name, a in d.items():
            out["formats"].setdefault(name, {})[comp] = {
                "bits_per_element": a["bits"], "rel_err_mean": float(np.mean(a["rel_err"])),
                "rel_err_max_layer": float(np.max(a["rel_err"])), "snr_db_mean": float(np.mean(a["snr_db"]))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("| format | bits/elt | K rel.err | K SNR dB | V rel.err | V SNR dB |")
    print("|---|---|---|---|---|---|")
    for name, d in out["formats"].items():
        print(f"| {name} | {d['K']['bits_per_element']:.2f} | {d['K']['rel_err_mean']:.4f} | {d['K']['snr_db_mean']:.1f} "
              f"| {d['V']['rel_err_mean']:.4f} | {d['V']['snr_db_mean']:.1f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
