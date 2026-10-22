"""E3: order-0 entropy of int4 weight codes under different quantization granularities (CPU, same weights).

Source: BF16 GGUF (Qwen3.5-2B). For each 2-D weight tensor (excluding embedding/output):
  per_tensor_sym : one scale per tensor, q = round(w / s) clipped to [-8, 7] (s = max|w| / 7)
  per_row_sym    : one scale per output row
  group128_asym  : groups of 128 along the input dim, asym 4-bit with f16 scale + zero (AWQ/GPTQ-like RTN layout,
                   WITHOUT the activation-aware / Hessian steps of AWQ / GPTQ)
  group32_sym    : llama.cpp Q4_0-like (scale per 32)
Reported per scheme: H(codes) order-0 per tensor (bits per weight), the 4-bit native code size, the ratio
H/4, and the full ratio including scales/zeros stored raw. Comparable to Tan et al.'s "int4" (per-tensor)
vs group formats.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def H(c):
    c = c[c > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--gguf-root", type=Path, default=Path("<llama.cpp-checkout>/gguf-py"))
    ap.add_argument("--max-tensors", type=int, default=40)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.gguf_root))
    from gguf import GGUFReader
    r = GGUFReader(str(a.model), "r")
    acc = {}
    n_t = 0
    for t in r.tensors:
        if t.tensor_type.name != "BF16" or len(t.shape) != 2 or "embd" in t.name or "output" in t.name:
            continue
        cols = int(t.shape[0])
        if cols % 128:
            continue
        w = (np.asarray(t.data).view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(-1, cols)
        schemes = {}
        s = np.abs(w).max() / 7
        schemes["per_tensor_sym"] = (np.clip(np.round(w / s), -8, 7), 16 / w.size)
        s = np.abs(w).max(1, keepdims=True) / 7
        s[s == 0] = 1
        schemes["per_row_sym"] = (np.clip(np.round(w / s), -8, 7), 16 / cols)
        g = w.reshape(-1, 128)
        lo, hi = g.min(1, keepdims=True), g.max(1, keepdims=True)
        d = (hi - lo) / 15
        d[d == 0] = 1
        schemes["group128_asym"] = (np.clip(np.round((g - lo) / d), 0, 15), 32 / 128)
        g = w.reshape(-1, 32)
        s = np.abs(g).max(1, keepdims=True) / 7
        s[s == 0] = 1
        schemes["group32_sym"] = (np.clip(np.round(g / s), -8, 7), 16 / 32)
        for name, (q, side) in schemes.items():
            q = q.astype(np.int64).ravel() + 8 * (name != "group128_asym")
            h = H(np.bincount(q, minlength=16))
            e = acc.setdefault(name, {"n": 0, "H": 0.0, "side": side})
            e["H"] += h * q.size
            e["n"] += q.size
        n_t += 1
        if n_t >= a.max_tensors:
            break
    out = {"tensors": n_t, "schemes": {}}
    for name, e in acc.items():
        h = e["H"] / e["n"]
        out["schemes"][name] = {"H_codes_bits": h, "codes_ratio": h / 4, "side_bits_per_weight": e["side"],
                                "total_ratio_incl_side": (h + e["side"]) / (4 + e["side"])}
        print(name, json.dumps({k: round(v, 4) for k, v in out["schemes"][name].items()}))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
