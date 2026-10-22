"""CPU-only lossless-ceiling estimates for a real llama.cpp KV cache capture (read-only).

Input: native sequence-state export of Qwen3-4B Q4_K_M (llama.cpp ed7ac35, F16 KV, FA on, seq0),
layout verified by size: header <IiII, n_tokens x <iIi, <II (v_trans, n_layer), then for each of
2*n_layer payloads (K layers 0..L-1, then V layers 0..L-1): <iQ (dtype, row_bytes) + n_tokens*row_bytes.

For K and V separately we report empirical entropy (bits per element) under several simple models:
  f16_word        order-0 over the 16-bit pattern, per layer
  f16_fields      H(sign)+H(exponent)+H(mantissa) per layer (fields treated independently)
  f16_exp|chan    H(exponent | head,dim) + H(sign | head,dim) + H(mantissa) per layer
  f16_delta_t     order-0 over (word[t] - word[t-1]) mod 2^16 per channel, per layer
  q8_codes        offline Q8_0 (llama.cpp rule, 32-element blocks along the row) int8 codes, order-0
  q8_codes|chan   same, conditioned on channel (Miller-Madow corrected)
  q4_codes        offline Q4_0 4-bit codes, order-0
Ideal size ratio = bits / native bits (16, 8.5 for Q8_0 incl. f16 scale per 32, 4.5 for Q4_0).
Also: singular-value energy of each layer's T x (heads*dim) matrix (lossy low-rank potential).
Caveats: one capture, one text, T tokens; order-0/conditional estimates exclude table/metadata cost;
per-channel histograms over T samples are biased low for large alphabets (reported, corrected for q8).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def H(counts: np.ndarray) -> float:
    c = counts[counts > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def H_cond(values: np.ndarray, alphabet: int, miller_madow: bool = False) -> float:
    """values: (T, C) small-int symbols; returns mean over channels of H(X | channel)."""
    T, C = values.shape
    total = 0.0
    for c in range(C):
        cnt = np.bincount(values[:, c], minlength=alphabet)
        h = H(cnt)
        if miller_madow:
            h += (np.count_nonzero(cnt) - 1) / (2 * T * np.log(2))
        total += h
    return total / C


def q8_0(x: np.ndarray):
    b = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    with np.errstate(divide="ignore", invalid="ignore"):
        idv = np.where(d > 0, 1.0 / d, 0.0)
    q = np.round(b * idv[:, None]).astype(np.int8)  # ggml uses roundf (half away from zero); ties are rare
    return q.reshape(x.shape), d.astype(np.float16)


def q4_0(x: np.ndarray):
    b = x.reshape(-1, 32).astype(np.float32)
    idx = np.abs(b).argmax(axis=1)
    mx = b[np.arange(len(b)), idx]
    d = mx / -8.0
    with np.errstate(divide="ignore", invalid="ignore"):
        idv = np.where(d != 0, 1.0 / d, 0.0)
    q = np.clip(np.floor(b * idv[:, None] + 8.5), 0, 15).astype(np.uint8)
    return q.reshape(x.shape)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--n-layer", type=int, default=36)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--svd-layers", default="0,9,18,27,35")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite")
    raw = np.memmap(args.state, dtype=np.uint8, mode="r")
    magic, _, nseq, T = struct.unpack_from("<IiII", raw, 0)
    off = 16 + 12 * T
    v_trans, n_layer = struct.unpack_from("<II", raw, off)
    off += 8
    assert n_layer == args.n_layer and v_trans == 0, (n_layer, v_trans)
    row = args.heads * args.dim
    layers = {"K": [], "V": []}
    for comp in ("K", "V"):
        for _ in range(n_layer):
            dtype, row_bytes = struct.unpack_from("<iQ", raw, off)
            off += 12
            assert dtype == 1 and row_bytes == row * 2, (dtype, row_bytes)
            n = T * row_bytes
            layers[comp].append(np.frombuffer(raw[off:off + n], dtype=np.uint16).reshape(T, row))
            off += n
    assert off == raw.size, (off, raw.size)
    svd_layers = {int(x) for x in args.svd_layers.split(",")}
    out = {"state": str(args.state), "state_sha256": hashlib.sha256(bytes(raw)).hexdigest(),
           "tokens": T, "n_layer": n_layer, "row_elements": row, "components": {}}
    for comp, mats in layers.items():
        acc = {k: [] for k in ("f16_word", "f16_fields", "f16_exp|chan", "f16_delta_t",
                               "q8_codes", "q8_codes|chan", "q8_scale_f16", "q4_codes")}
        svd = {}
        for li, w in enumerate(mats):
            sign = (w >> 15).astype(np.uint8)
            exp = ((w >> 10) & 0x1F).astype(np.uint8)
            man = (w & 0x3FF).astype(np.uint16)
            acc["f16_word"].append(H(np.bincount(w.ravel(), minlength=65536)))
            acc["f16_fields"].append(H(np.bincount(sign.ravel(), minlength=2)) +
                                     H(np.bincount(exp.ravel(), minlength=32)) +
                                     H(np.bincount(man.ravel(), minlength=1024)))
            acc["f16_exp|chan"].append(H_cond(exp, 32) + H_cond(sign, 2) +
                                       H(np.bincount(man.ravel(), minlength=1024)))
            delta = (w[1:].astype(np.int32) - w[:-1].astype(np.int32)) & 0xFFFF
            acc["f16_delta_t"].append(H(np.bincount(delta.ravel(), minlength=65536)))
            x = w.view(np.float16).astype(np.float32)
            q, d = q8_0(x)
            qs = (q.astype(np.int16) + 128).astype(np.uint16)
            acc["q8_codes"].append(H(np.bincount(qs.ravel(), minlength=256)))
            acc["q8_codes|chan"].append(H_cond(qs, 256, miller_madow=True))
            acc["q8_scale_f16"].append(H(np.bincount(d.view(np.uint16).ravel(), minlength=65536)))
            acc["q4_codes"].append(H(np.bincount(q4_0(x).ravel(), minlength=16)))
            if li in svd_layers:
                s = np.linalg.svd(x - x.mean(axis=0, keepdims=True), compute_uv=False)
                e = np.cumsum(s ** 2) / np.sum(s ** 2)
                svd[li] = {f"rank_for_{p}": int(np.searchsorted(e, p / 100) + 1) for p in (90, 99, 99.9)}
        mean = {k: float(np.mean(v)) for k, v in acc.items()}
        res = {"mean_bits_per_element": mean, "per_layer": acc, "svd_energy_rank": svd,
               "ratios": {
                   "f16_word": mean["f16_word"] / 16, "f16_fields": mean["f16_fields"] / 16,
                   "f16_exp|chan": mean["f16_exp|chan"] / 16, "f16_delta_t": mean["f16_delta_t"] / 16,
                   "q8_0_order0": (mean["q8_codes"] + mean["q8_scale_f16"] / 32) / 8.5,
                   "q8_0_chan": (mean["q8_codes|chan"] + mean["q8_scale_f16"] / 32) / 8.5,
                   "q4_0_codes_only_order0": (mean["q4_codes"] + 16 / 32) / 4.5}}
        out["components"][comp] = res
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({c: {"bits": {k: round(v, 3) for k, v in r["mean_bits_per_element"].items()},
                          "ratios": {k: round(v, 4) for k, v in r["ratios"].items()},
                          "svd": r["svd_energy_rank"]} for c, r in out["components"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
