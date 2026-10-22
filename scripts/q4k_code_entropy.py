"""CPU-only, read-only empirical entropy of quantized weight fields in a GGUF file.

Purpose: estimate how much room an additional *lossless* codec has on top of
native Q4_K / Q6_K blocks (llama.cpp k-quants). Computes per-tensor order-0
(memoryless) empirical entropy of each block field and reports the ideal
order-0 coded size versus the native size.

Caveats (also printed into the JSON):
- Order-0 entropy ignores context; context-modelling coders can go below it,
  so this is NOT a hard lower bound for every lossless coder.
- Ideal sizes exclude codec metadata, alignment, random-access indexing and
  per-tensor model/table cost, which all make real codecs larger.
- No GPU, no model modification; the GGUF is opened read-only (memmap).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def entropy_bits(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log2(p)).sum())


def field_stats(values: np.ndarray, alphabet: int, native_bits: int) -> dict:
    counts = np.bincount(values.ravel(), minlength=alphabet)
    h = entropy_bits(counts)
    n = int(values.size)
    return {"symbols": n, "native_bits_per_symbol": native_bits, "entropy_bits_per_symbol": h,
            "native_bytes": n * native_bits / 8.0, "ideal_order0_bytes": n * h / 8.0}


def q4k_fields(blocks: np.ndarray) -> dict:
    # block_q4_K: d f16[2B], dmin f16[2B], scales[12], qs[128]  -> 144 B
    d = blocks[:, 0:2].copy().view(np.uint16)
    dmin = blocks[:, 2:4].copy().view(np.uint16)
    scales = blocks[:, 4:16]
    qs = blocks[:, 16:144]
    nibbles = np.concatenate([(qs & 0x0F).ravel(), (qs >> 4).ravel()])
    return {"d_f16": field_stats(d, 65536, 16), "dmin_f16": field_stats(dmin, 65536, 16),
            "scales_bytes": field_stats(scales, 256, 8), "quant_codes_4bit": field_stats(nibbles, 16, 4)}


def q6k_fields(blocks: np.ndarray) -> dict:
    # block_q6_K: ql[128], qh[64], scales int8[16], d f16[2B] -> 210 B
    ql = blocks[:, 0:128].astype(np.uint16)
    qh = blocks[:, 128:192].astype(np.uint16)
    scales = blocks[:, 192:208]
    d = blocks[:, 208:210].copy().view(np.uint16)
    codes = []
    for half in range(2):
        l_lo = ql[:, half * 64: half * 64 + 32]
        l_hi = ql[:, half * 64 + 32: half * 64 + 64]
        h = qh[:, half * 32: half * 32 + 32]
        codes.append((l_lo & 0xF) | (((h >> 0) & 3) << 4))
        codes.append((l_hi & 0xF) | (((h >> 2) & 3) << 4))
        codes.append((l_lo >> 4) | (((h >> 4) & 3) << 4))
        codes.append((l_hi >> 4) | (((h >> 6) & 3) << 4))
    q = np.concatenate([c.ravel() for c in codes])
    return {"d_f16": field_stats(d, 65536, 16), "scales_bytes": field_stats(scales, 256, 8),
            "quant_codes_6bit": field_stats(q, 64, 6)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--gguf-root", type=Path, default=Path("<llama.cpp-checkout>/gguf-py"))
    ap.add_argument("--expect-sha256", default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite existing output")
    sys.path.insert(0, str(args.gguf_root))
    from gguf import GGUFReader  # noqa: E402

    sha = hashlib.sha256()
    with args.model.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            sha.update(chunk)
    digest = sha.hexdigest()
    if args.expect_sha256 and digest != args.expect_sha256:
        raise SystemExit(f"sha mismatch: {digest}")

    reader = GGUFReader(str(args.model), "r")
    per_tensor = []
    agg = {}
    other_bytes = 0
    for t in reader.tensors:
        qtype = t.tensor_type.name
        raw = np.asarray(t.data)
        nbytes = int(t.n_bytes)
        if qtype == "Q4_K":
            blocks = raw.reshape(-1, 144) if raw.dtype == np.uint8 else raw.view(np.uint8).reshape(-1, 144)
            fields = q4k_fields(blocks)
        elif qtype == "Q6_K":
            blocks = raw.reshape(-1, 210) if raw.dtype == np.uint8 else raw.view(np.uint8).reshape(-1, 210)
            fields = q6k_fields(blocks)
        else:
            other_bytes += nbytes
            per_tensor.append({"name": t.name, "type": qtype, "bytes": nbytes, "analysed": False})
            continue
        native = sum(v["native_bytes"] for v in fields.values())
        if abs(native - nbytes) > 0.5:
            raise SystemExit(f"field accounting mismatch for {t.name}: {native} vs {nbytes}")
        ideal = sum(v["ideal_order0_bytes"] for v in fields.values())
        per_tensor.append({"name": t.name, "type": qtype, "bytes": nbytes, "analysed": True,
                           "ideal_order0_bytes": ideal, "ratio": ideal / nbytes, "fields": fields})
        a = agg.setdefault(qtype, {"tensors": 0, "native_bytes": 0.0, "ideal_order0_bytes": 0.0, "fields": {}})
        a["tensors"] += 1
        a["native_bytes"] += native
        a["ideal_order0_bytes"] += ideal
        for k, v in fields.items():
            fa = a["fields"].setdefault(k, {"native_bytes": 0.0, "ideal_order0_bytes": 0.0, "symbols": 0})
            fa["native_bytes"] += v["native_bytes"]
            fa["ideal_order0_bytes"] += v["ideal_order0_bytes"]
            fa["symbols"] += v["symbols"]
    for a in agg.values():
        a["ratio"] = a["ideal_order0_bytes"] / a["native_bytes"]
        for fa in a["fields"].values():
            fa["ratio"] = fa["ideal_order0_bytes"] / fa["native_bytes"]
            fa["mean_entropy_bits_per_symbol_weighted"] = None
    analysed_native = sum(a["native_bytes"] for a in agg.values())
    analysed_ideal = sum(a["ideal_order0_bytes"] for a in agg.values())
    result = {
        "schema": "q4k-code-entropy-v1",
        "model": {"path": str(args.model), "sha256": digest, "file_bytes": args.model.stat().st_size},
        "method": "per-tensor order-0 empirical entropy of each k-quant block field; ideal size = sum(n*H)/8",
        "caveats": [
            "order-0 entropy is not a bound for context-modelling coders",
            "ideal sizes exclude codec metadata, alignment, random-access index, table cost",
            "CPU-only read-only analysis; says nothing about GPU decode speed or VRAM residency",
        ],
        "totals": {"analysed_native_bytes": analysed_native, "analysed_ideal_order0_bytes": analysed_ideal,
                   "analysed_ratio": analysed_ideal / analysed_native,
                   "unanalysed_tensor_bytes": other_bytes,
                   "whole_tensor_payload_ratio": (analysed_ideal + other_bytes) / (analysed_native + other_bytes)},
        "by_type": agg,
        "per_tensor": per_tensor,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"totals": result["totals"],
                      "by_type": {k: {"tensors": v["tensors"], "ratio": v["ratio"],
                                      "fields": {fk: round(fv["ratio"], 6) for fk, fv in v["fields"].items()}}
                                  for k, v in agg.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
