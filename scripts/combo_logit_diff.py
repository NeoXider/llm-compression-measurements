"""Compare two raw float32 logit dumps of llama-cq-logits (same prompt, same vocabulary).
Prints: positions, max |diff|, number of positions that differ at all, mean KL(base || other) over positions,
top-1 agreement, and whether the greedy continuations (OUT.bin.tokens.txt) are identical.
Usage: python combo_logit_diff.py BASE.bin OTHER.bin LABEL
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

N_VOCAB = {248320: "qwen3.5", 151936: "qwen3"}


def main() -> int:
    base, other, label = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    a = np.memmap(base, dtype=np.float32, mode="r")
    b = np.memmap(other, dtype=np.float32, mode="r")
    if a.size != b.size:
        print(f"{label}: size mismatch {a.size} vs {b.size}")
        return 0
    nv = next((v for v in N_VOCAB if a.size % v == 0), None)
    if nv is None:
        print(f"{label}: unknown vocabulary")
        return 0
    a = a.reshape(-1, nv)
    b = b.reshape(-1, nv)
    n = a.shape[0]
    n_gen = 64
    n_prompt = n - n_gen
    maxd, ndiff, kl, top = 0.0, 0, 0.0, 0
    kl_p, top_p, maxd_p = 0.0, 0, 0.0
    for i in range(0, n, 64):
        x = np.asarray(a[i:i + 64], dtype=np.float64)
        y = np.asarray(b[i:i + 64], dtype=np.float64)
        d = np.abs(x - y)
        maxd = max(maxd, float(d.max()))
        ndiff += int((d.max(axis=1) > 0).sum())
        lx = x - x.max(axis=1, keepdims=True)
        lx -= np.log(np.exp(lx).sum(axis=1, keepdims=True))
        ly = y - y.max(axis=1, keepdims=True)
        ly -= np.log(np.exp(ly).sum(axis=1, keepdims=True))
        klrow = (np.exp(lx) * (lx - ly)).sum(axis=1)
        agree = x.argmax(axis=1) == y.argmax(axis=1)
        kl += float(klrow.sum())
        top += int(agree.sum())
        m = np.arange(i, i + x.shape[0]) < n_prompt
        kl_p += float(klrow[m].sum())
        top_p += int(agree[m].sum())
        if m.any():
            maxd_p = max(maxd_p, float(d[m].max()))
    ta = Path(str(base) + ".tokens.txt")
    tb = Path(str(other) + ".tokens.txt")
    same_gen = ta.exists() and tb.exists() and ta.read_text() == tb.read_text()
    first_div = None
    if ta.exists() and tb.exists():
        ga, gb = ta.read_text().split(), tb.read_text().split()
        first_div = next((k for k, (u, v) in enumerate(zip(ga, gb)) if u != v), None)
    print(f"{label}: positions {n}, positions differing {ndiff}, max|diff| {maxd:.6g}, mean KL {kl / n:.3e}, "
          f"top-1 agree {top}/{n}, greedy continuation identical: {same_gen}, first divergent generated token: {first_div}; "
          f"PROMPT positions only ({n_prompt}): max|diff| {maxd_p:.6g}, mean KL {kl_p / max(n_prompt, 1):.3e}, top-1 agree {top_p}/{n_prompt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
