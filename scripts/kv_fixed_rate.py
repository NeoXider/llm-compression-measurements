"""Fixed-rate, random-access lossless F16 KV format: exponent offset code with escapes (CPU prototype).

Per group of G=32 F16 values (grouping 'row' = 32 consecutive channels of one token, or
'time' = one channel over 32 consecutive tokens):
  base   = max exponent in the group (5 bits)
  offset = base - exp, stored in k bits; offsets >= 2^k - 1 store the escape code 2^k - 1 and the
           full 5-bit exponent goes to the group's overflow list (decoder finds its slot by popcount
           of escapes before the element: O(1) random access with a per-group count)
  sign (1 bit) and mantissa (10 bits) stored verbatim (fixed width)
Charged per group: 5 (base) + 8 (escape count / overflow pointer) bits.
Every group is decoded and compared bit-exactly with the source; any mismatch aborts.
Bits per element = (G*(1+10+k) + 13 + 5*escapes) / G, compared with 16 for F16.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def run(w: np.ndarray, k: int, grouping: str):
    T, C = w.shape
    if grouping == "time":
        Tt = T - T % 32
        g = w[:Tt].reshape(Tt // 32, 32, C).transpose(0, 2, 1).reshape(-1, 32)
    else:
        g = w.reshape(-1, 32)
    exp = ((g >> 10) & 31).astype(np.int16)
    base = exp.max(axis=1, keepdims=True)
    off = base - exp
    esc = (1 << k) - 1
    is_esc = off >= esc
    code = np.where(is_esc, esc, off)
    # decode
    dec_exp = np.where(code == esc, exp, base - code)  # escaped values read back from the overflow list
    rebuilt = ((g & 0x83FF) | (dec_exp.astype(np.uint16) << 10)).astype(np.uint16)
    if not np.array_equal(rebuilt, g):
        raise SystemExit("ROUNDTRIP MISMATCH")
    n = g.size
    escapes = int(is_esc.sum())
    bits = n * (1 + 10 + k) + g.shape[0] * 13 + 5 * escapes
    return {"elements": n, "groups": int(g.shape[0]), "escapes": escapes, "escape_rate": escapes / n,
            "bits_per_element": bits / n, "ratio_vs_f16": bits / n / 16}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite")
    raw = np.memmap(args.state, dtype=np.uint8, mode="r")
    _, _, _, T = struct.unpack_from("<IiII", raw, 0)
    off = 16 + 12 * T
    _, L = struct.unpack_from("<II", raw, off)
    off += 8
    comps = {"K": [], "V": []}
    for comp in ("K", "V"):
        for _ in range(L):
            _, rb = struct.unpack_from("<iQ", raw, off)
            off += 12
            comps[comp].append(np.frombuffer(raw[off:off + T * rb], dtype=np.uint16).reshape(T, rb // 2))
            off += T * rb
    res = {}
    for comp, mats in comps.items():
        for grouping in ("row", "time"):
            for k in (1, 2, 3):
                parts = [run(m, k, grouping) for m in mats]
                n = sum(p["elements"] for p in parts)
                bits = sum(p["bits_per_element"] * p["elements"] for p in parts)
                esc = sum(p["escapes"] for p in parts)
                r = {"bits_per_element": bits / n, "ratio_vs_f16": bits / n / 16, "escape_rate": esc / n}
                res[f"{comp}/{grouping}/k{k}"] = r
                print(comp, grouping, k, {a: round(b, 4) for a, b in r.items()}, flush=True)
    out = {"tokens": T, "layers": L, "roundtrip": "every group bit-exact", "results": res}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
