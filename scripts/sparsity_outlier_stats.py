"""Statistics only (no method implemented): zeros/trivial blocks and outlier clipping on real data.

Weights (Qwen3-4B Q4_K_M GGUF, Q4_K tensors): share of zero 4-bit codes, share of 256-weight blocks
whose dequantized values are all equal (trivial), share of 32-weight sub-blocks with a single code.
KV (native F16 capture): share of exact zeros and of |x| below 1e-3 of the layer's max.
Outlier clipping ("dense-and-sparse" idea): per group of 32 values (KV rows; dequantized Q4_K weights),
clip the top/bottom p% of magnitudes (p = 0.1, 1), keep the clipped values as a sparse list
(index 16 b + value 16 b each), and report: dynamic range shrink (max|x| ratio), MSE of a symmetric
4-bit absmax quantizer on the dense part vs no clipping, and the side-list cost in bits/element.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def q4_err(x):
    d = np.abs(x).max(1, keepdims=True) / 7
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d > 0, np.clip(np.round(x / d), -8, 7), 0)
    return float(np.mean((x - q * d) ** 2))


def clip_stats(x, p):
    """x: (groups, 32). Clip values whose |x| is above the (100-p) percentile of the whole tensor."""
    thr = np.percentile(np.abs(x), 100 - p)
    mask = np.abs(x) > thr
    dense = np.where(mask, 0, x)
    base_err = q4_err(x)
    clip_err = float(np.mean(np.where(mask, 0, (dense - dense)) ** 2))  # outliers exact in side list
    dense_err = q4_err(dense)
    return {"p_percent": p, "outlier_share": float(mask.mean()),
            "mean_group_range_ratio": float(np.mean(np.abs(dense).max(1) / np.maximum(np.abs(x).max(1), 1e-30))),
            "q4_mse_no_clip": base_err, "q4_mse_dense_after_clip": dense_err + clip_err,
            "mse_ratio": (dense_err + clip_err) / base_err if base_err else None,
            "side_list_bits_per_element": float(mask.mean()) * 32}


def q4k_dequant(b):
    d = np.ascontiguousarray(b[:, 0:2]).view(np.float16).astype(np.float32)
    dmin = np.ascontiguousarray(b[:, 2:4]).view(np.float16).astype(np.float32)
    s = b[:, 4:16].astype(np.uint8)
    sc = np.zeros((len(b), 8), np.float32)
    mn = np.zeros((len(b), 8), np.float32)
    for j in range(8):
        if j < 4:
            sc[:, j] = s[:, j] & 63
            mn[:, j] = s[:, j + 4] & 63
        else:
            sc[:, j] = (s[:, j + 4] & 0xF) | ((s[:, j - 4] >> 6) << 4)
            mn[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    qs = b[:, 16:144]
    out = np.empty((len(b), 256), np.float32)
    codes = np.empty((len(b), 256), np.uint8)
    for pair in range(4):
        q = qs[:, pair * 32:(pair + 1) * 32]
        for half, c in ((0, q & 15), (1, q >> 4)):
            j = pair * 2 + half
            codes[:, j * 32:(j + 1) * 32] = c
            out[:, j * 32:(j + 1) * 32] = d * sc[:, j:j + 1] * c - dmin * mn[:, j:j + 1]
    return out, codes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--gguf-root", type=Path, default=Path("<llama.cpp-checkout>/gguf-py"))
    ap.add_argument("--max-blocks-per-tensor", type=int, default=20000)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite")
    sys.path.insert(0, str(args.gguf_root))
    from gguf import GGUFReader  # noqa: E402
    rng = np.random.default_rng(20261021)
    reader = GGUFReader(str(args.model), "r")
    zero_codes = total_codes = trivial_blocks = total_blocks = single_sub = total_sub = 0
    wvals = []
    for t in reader.tensors:
        if t.tensor_type.name != "Q4_K":
            continue
        b = np.asarray(t.data).view(np.uint8).reshape(-1, 144)
        if len(b) > args.max_blocks_per_tensor:
            b = b[np.sort(rng.choice(len(b), args.max_blocks_per_tensor, replace=False))]
        deq, codes = q4k_dequant(b)
        zero_codes += int((codes == 0).sum())
        total_codes += codes.size
        trivial_blocks += int((deq.max(1) == deq.min(1)).sum())
        total_blocks += len(b)
        sub = codes.reshape(-1, 32)
        single_sub += int((sub.max(1) == sub.min(1)).sum())
        total_sub += len(sub)
        wvals.append(deq.reshape(-1, 32)[rng.choice(len(deq) * 8, min(2000, len(deq) * 8), replace=False)])
    W = np.concatenate(wvals)
    raw = np.memmap(args.state, dtype=np.uint8, mode="r")
    _, _, _, T = struct.unpack_from("<IiII", raw, 0)
    off = 16 + 12 * T
    _, L = struct.unpack_from("<II", raw, off)
    off += 8
    kv = {"K": [], "V": []}
    zeros = {"K": 0, "V": 0}
    tiny = {"K": 0, "V": 0}
    count = {"K": 0, "V": 0}
    for comp in ("K", "V"):
        for _ in range(L):
            _, rb = struct.unpack_from("<iQ", raw, off)
            off += 12
            x = np.frombuffer(raw[off:off + T * rb], dtype=np.float16).astype(np.float32)
            off += T * rb
            zeros[comp] += int((x == 0).sum())
            tiny[comp] += int((np.abs(x) < 1e-3 * np.abs(x).max()).sum())
            count[comp] += x.size
            g = x.reshape(-1, 32)
            kv[comp].append(g[rng.choice(len(g), 4000, replace=False)])
    out = {"weights_q4k": {"blocks_sampled": total_blocks, "zero_code_share": zero_codes / total_codes,
                           "trivial_block_share": trivial_blocks / total_blocks,
                           "single_code_subblock_share": single_sub / total_sub,
                           "clipping": [clip_stats(W, p) for p in (0.1, 1.0)]},
           "kv_f16": {c: {"exact_zero_share": zeros[c] / count[c], "below_1e-3_of_layer_max_share": tiny[c] / count[c],
                          "clipping": [clip_stats(np.concatenate(kv[c]), p) for p in (0.1, 1.0)]} for c in ("K", "V")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
