"""Summarize a fork speed matrix (work/kv_speed.sh): median decode/prefill tok/s per config and depth, ratio to F16.

Usage: python summarize_fork_speed.py <artifacts/fork_speed_TAG> [<more dirs> ...]
Files are named NNN-<bin>-<kv>-g<gqa>-rot<on|off>.jsonl (llama-bench -o jsonl). Writes summary.md/.json in the first dir.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path


def main() -> int:
    dirs = [Path(d) for d in sys.argv[1:]]
    data: dict[tuple, list[float]] = {}
    for d in dirs:
        for f in sorted(d.glob("*.jsonl")):
            parts = f.stem.split("-", 1)[1]                    # bin-kv-gX-rotY
            bin_, rest = parts.split("-", 1)
            kv, g, rot = rest.rsplit("-", 2)
            cfg = f"{bin_}:{kv}:{g[1:]}:{rot[3:]}"
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.startswith("{"):
                    continue
                r = json.loads(line)
                test = "prefill" if r.get("n_gen", 0) == 0 else "decode"
                data.setdefault((cfg, int(r["n_depth"]), test), []).extend(r.get("samples_ts", []))
    cfgs = sorted({k[0] for k in data}, key=lambda c: (c.split(":")[0] != "fork", c))
    depths = sorted({k[1] for k in data})
    ref = "fork:f16:def:on"
    rows, md = [], []
    for test in ("decode", "prefill"):
        md += [f"### {test} (median tok/s; ratio to {ref}; n samples)", "",
               "| config | " + " | ".join(f"d={d}" for d in depths) + " |", "|---|" + "---|" * len(depths)]
        for c in cfgs:
            cells = []
            for d in depths:
                v = data.get((c, d, test))
                r0 = data.get((ref, d, test))
                if not v:
                    cells.append("—")
                    continue
                m = st.median(v)
                ratio = m / st.median(r0) if r0 else float("nan")
                cv = 100 * st.pstdev(v) / st.mean(v) if len(v) > 1 else 0.0
                cells.append(f"{m:.1f} ({ratio:.3f}; n={len(v)}; cv {cv:.1f}%)")
                rows.append({"config": c, "depth": d, "test": test, "median": m, "ratio_to_f16": ratio, "n": len(v), "cv_pct": cv})
            md.append(f"| {c} | " + " | ".join(cells) + " |")
        md.append("")
    (dirs[0] / "summary.md").write_text("\n".join(md), encoding="utf-8")
    (dirs[0] / "summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
