"""Per-field order-0 entropy of quantized GGUF tensors for Q8_0, Q2_K..Q6_K (CPU, read-only).

Extends q4k_code_entropy.py. Fields per block (llama.cpp ggml-common.h layouts):
  Q8_0  34 B/32 : d f16 | 32 x int8 codes                                  (codes exact)
  Q4_K 144 B/256: d, dmin f16 | 12 B scales/mins | 128 B 4-bit codes        (codes exact)
  Q5_K 176 B/256: d, dmin | 12 B scales | 32 B qh (high bits) | 128 B qs    (high bit and nibble as
                  independent fields: an upper bound on the joint 5-bit code entropy)
  Q6_K 210 B/256: 128 B ql | 64 B qh | 16 int8 scales | d                   (6-bit codes exact)
  Q3_K 110 B/256: 32 B hmask | 64 B qs (2-bit) | 12 B scales | d            (low 2 bits and high bit
                  independent: upper bound)
  Q2_K  84 B/256: 16 B scales|mins (nibbles) | 64 B qs (2-bit) | d, dmin    (codes exact)
Reports native vs ideal order-0 bytes per type and overall; caveats as in q4k_code_entropy.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def H(counts):
    c = counts[counts > 0].astype(np.float64)
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def f(values, alphabet, bits):
    v = np.ascontiguousarray(values).ravel()
    h = H(np.bincount(v.astype(np.int64), minlength=alphabet))
    return {"symbols": int(v.size), "native_bits": bits, "entropy_bits": h,
            "native_bytes": v.size * bits / 8, "ideal_bytes": v.size * h / 8}


def u16(b):
    return np.ascontiguousarray(b).view(np.uint16)


def nibbles(b):
    return np.concatenate([(b & 15).ravel(), (b >> 4).ravel()])


def crumbs(b):
    return np.concatenate([((b >> s) & 3).ravel() for s in (0, 2, 4, 6)])


def bits(b):
    return np.concatenate([((b >> s) & 1).ravel() for s in range(8)])


LAYOUT = {
    "Q8_0": (34, lambda b: {"d": f(u16(b[:, 0:2]), 65536, 16), "codes8": f(b[:, 2:34], 256, 8)}),
    "Q4_K": (144, lambda b: {"d": f(u16(b[:, 0:2]), 65536, 16), "dmin": f(u16(b[:, 2:4]), 65536, 16),
                             "scales": f(b[:, 4:16], 256, 8), "codes4": f(nibbles(b[:, 16:144]), 16, 4)}),
    "Q5_K": (176, lambda b: {"d": f(u16(b[:, 0:2]), 65536, 16), "dmin": f(u16(b[:, 2:4]), 65536, 16),
                             "scales": f(b[:, 4:16], 256, 8), "highbit": f(bits(b[:, 16:48]), 2, 1),
                             "codes4": f(nibbles(b[:, 48:176]), 16, 4)}),
    "Q3_K": (110, lambda b: {"highbit": f(bits(b[:, 0:32]), 2, 1), "codes2": f(crumbs(b[:, 32:96]), 4, 2),
                             "scales": f(b[:, 96:108], 256, 8), "d": f(u16(b[:, 108:110]), 65536, 16)}),
    "Q2_K": (84, lambda b: {"scales": f(nibbles(b[:, 0:16]), 16, 4), "codes2": f(crumbs(b[:, 16:80]), 4, 2),
                            "d": f(u16(b[:, 80:82]), 65536, 16), "dmin": f(u16(b[:, 82:84]), 65536, 16)}),
}


def q6k(b):
    ql = b[:, 0:128].astype(np.uint16)
    qh = b[:, 128:192].astype(np.uint16)
    codes = []
    for half in range(2):
        lo, hi, h = ql[:, half * 64:half * 64 + 32], ql[:, half * 64 + 32:half * 64 + 64], qh[:, half * 32:half * 32 + 32]
        codes += [(lo & 15) | ((h & 3) << 4), (hi & 15) | (((h >> 2) & 3) << 4),
                  (lo >> 4) | (((h >> 4) & 3) << 4), (hi >> 4) | (((h >> 6) & 3) << 4)]
    return {"codes6": f(np.concatenate([c.ravel() for c in codes]), 64, 6),
            "scales": f(b[:, 192:208], 256, 8), "d": f(u16(b[:, 208:210]), 65536, 16)}


LAYOUT["Q6_K"] = (210, q6k)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, action="append", required=True)
    ap.add_argument("--gguf-root", type=Path, default=Path("<llama.cpp-checkout>/gguf-py"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite")
    sys.path.insert(0, str(args.gguf_root))
    from gguf import GGUFReader  # noqa: E402
    out = {}
    for model in args.model:
        reader = GGUFReader(str(model), "r")
        agg, other = {}, 0
        for t in reader.tensors:
            q = t.tensor_type.name
            if q not in LAYOUT:
                other += int(t.n_bytes)
                continue
            size, fn = LAYOUT[q]
            blocks = np.asarray(t.data).view(np.uint8).reshape(-1, size)
            fields = fn(blocks)
            nat = sum(v["native_bytes"] for v in fields.values())
            if abs(nat - t.n_bytes) > 0.5:
                raise SystemExit(f"accounting mismatch {t.name} {q}: {nat} vs {t.n_bytes}")
            a = agg.setdefault(q, {"tensors": 0, "native_bytes": 0.0, "ideal_bytes": 0.0, "fields": {}})
            a["tensors"] += 1
            for k, v in fields.items():
                a["native_bytes"] += v["native_bytes"]
                a["ideal_bytes"] += v["ideal_bytes"]
                fa = a["fields"].setdefault(k, {"native_bytes": 0.0, "ideal_bytes": 0.0})
                fa["native_bytes"] += v["native_bytes"]
                fa["ideal_bytes"] += v["ideal_bytes"]
        for a in agg.values():
            a["ratio"] = a["ideal_bytes"] / a["native_bytes"]
            for fa in a["fields"].values():
                fa["ratio"] = fa["ideal_bytes"] / fa["native_bytes"]
        nat = sum(a["native_bytes"] for a in agg.values())
        ide = sum(a["ideal_bytes"] for a in agg.values())
        out[model.name] = {"file_bytes": model.stat().st_size, "analysed_native_bytes": nat, "analysed_ideal_bytes": ide,
                           "analysed_ratio": ide / nat, "unanalysed_bytes": other,
                           "whole_ratio": (ide + other) / (nat + other), "by_type": agg}
        print(model.name, round(ide / nat, 4), {q: (a["tensors"], round(a["ratio"], 4)) for q, a in agg.items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
