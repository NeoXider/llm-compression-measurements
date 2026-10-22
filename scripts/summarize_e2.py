"""E2 summary: llama.cpp PR builds vs their base builds, ABBA (base, pr, pr, base), llama-bench jsonl."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

A = Path(sys.argv[1])
data = {}
for f in sorted(A.glob("*.jsonl")):
    pair, kv, d, i, which = f.stem.split("-")
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            test = "prefill" if r.get("n_gen", 0) == 0 else "decode"
            data.setdefault((pair, kv, int(d[1:]), test, which), []).extend(r.get("samples_ts", []))
rows = []
md = ["| PR | KV | depth | test | base tok/s (n) | PR tok/s (n) | PR / base | base CV % | PR CV % |", "|---|---|---|---|---|---|---|---|---|"]
for (pair, kv, d, test) in sorted({k[:4] for k in data}, key=lambda k: (k[0], ["f16", "q8_0", "q4_0"].index(k[1]), k[2], k[3])):
    b, p = np.array(data.get((pair, kv, d, test, "base"), [])), np.array(data.get((pair, kv, d, test, "pr"), []))
    if not len(b) or not len(p):
        continue
    row = {"pr": pair, "kv": kv, "depth": d, "test": test, "base": float(np.median(b)), "prv": float(np.median(p)),
           "ratio": float(np.median(p) / np.median(b)), "n": [len(b), len(p)],
           "cv": [float(100 * b.std() / b.mean()), float(100 * p.std() / p.mean())]}
    rows.append(row)
    md.append(f"| #{pair} | {kv} | {d} | {test} | {row['base']:.1f} ({len(b)}) | {row['prv']:.1f} ({len(p)}) | {row['ratio']:.3f} | {row['cv'][0]:.1f} | {row['cv'][1]:.1f} |")
# quantized vs f16 within each build
md += ["", "Quantized KV vs F16 KV within each build (decode):", "", "| PR | build | depth | q8_0 / f16 | q4_0 / f16 |", "|---|---|---|---|---|"]
for pair in sorted({r["pr"] for r in rows}):
    for which in ("base", "pr"):
        for d in sorted({r["depth"] for r in rows if r["pr"] == pair}):
            med = {kv: np.median(data.get((pair, kv, d, "decode", which), [np.nan])) for kv in ("f16", "q8_0", "q4_0")}
            md.append(f"| #{pair} | {which} | {d} | {med['q8_0'] / med['f16']:.3f} | {med['q4_0'] / med['f16']:.3f} |")
(A / "summary.md").write_text("\n".join(md), encoding="utf-8")
(A / "summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
print("\n".join(md))
