"""Summarize llama-bench jsonl files from gpu_speed_abba.sh (read-only).

For every (kv, depth, test = prefill pp512 or decode tg64) collects all per-repetition samples
(samples_ts) over all ABBA blocks, and reports median, p5/p95, min/max and the number of samples.
Speed ratios vs F16 at the same depth use medians. Window label comes from preflight.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()
    data = {}
    for f in sorted(args.dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip().startswith("{"):
                continue
            r = json.loads(line)
            test = "prefill" if r.get("n_gen", 0) == 0 else "decode"
            key = (r["type_k"], int(r.get("n_depth", 0)), test)
            data.setdefault(key, []).extend(r.get("samples_ts", []))
    pre = json.loads((args.dir / "preflight.json").read_text(encoding="utf-8"))
    out = {"window": pre["verdict"], "problems": pre["problems"], "rows": []}
    med = {}
    for (kv, d, test), s in sorted(data.items(), key=lambda x: (x[0][2], x[0][1], x[0][0])):
        a = np.array(s)
        row = {"kv": kv, "depth": d, "test": test, "n": len(s), "median_tok_s": float(np.median(a)),
               "p5": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
               "min": float(a.min()), "max": float(a.max()),
               "cv_percent": float(100 * a.std() / a.mean())}
        med[(kv, d, test)] = row["median_tok_s"]
        out["rows"].append(row)
    for row in out["rows"]:
        base = med.get(("f16", row["depth"], row["test"]))
        row["ratio_vs_f16"] = row["median_tok_s"] / base if base else None
    (args.dir / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"window: {out['window']}")
    print("| test | depth | KV | n | median tok/s | p5–p95 | CV % | vs F16 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in out["rows"]:
        print(f"| {r['test']} | {r['depth']} | {r['kv']} | {r['n']} | {r['median_tok_s']:.1f} | "
              f"{r['p5']:.1f}–{r['p95']:.1f} | {r['cv_percent']:.1f} | {r['ratio_vs_f16']:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
