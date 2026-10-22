"""Summarize the weights-fork variant B series (artifacts/cqw_speed/<tag>/, artifacts/cqw_quality/<tag>/) into
JSON + Markdown. Decode = llama-bench tg64, prefill = pp512, both at the fill depth; 12 samples per cell
(3 per llama-bench run x 4 runs in forward/backward order). Speed ratios are ratios of means; the spread is the
min..max of per-run ratios against the base run of the same block and direction is not used (runs are not paired).

Usage: python cqw_summarize.py --speed TAG [TAG ...] [--quality TAG ...] --out PREFIX
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

P = Path(r"/path/to/repo")
PAT = {
    "ppl_q": r"Mean PPL\(Q\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)",
    "ppl_ratio": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)",
    "kld_mean": r"Mean\s+KLD:\s*([-\d.]+)\s*±\s*([\d.]+)",
    "kld_p99": r"99\.0%\s+KLD:\s*([-\d.]+)",
    "same_top_pct": r"Same top p:\s*([\d.]+)\s*±\s*([\d.]+)",
}


def speed(tag: str) -> dict:
    d = P / "artifacts" / "cqw_speed" / tag
    out: dict = {"tag": tag, "models": {}}
    for f in sorted((d / "bench").glob("*.jsonl")):
        m = re.match(r"\d+-d(\d+)-b(\d)-(fwd|bwd)-(.+)\.jsonl", f.name)
        if not m:
            continue
        depth, label = int(m.group(1)), m.group(4)
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            kind = "decode" if r["n_gen"] > 0 else "prefill"
            cell = out["models"].setdefault(label, {}).setdefault(str(depth), {}).setdefault(kind, [])
            cell.extend(r["samples_ts"])
            out["models"][label].setdefault("model_size", r["model_size"])
    for label, md in out["models"].items():
        for depth, cells in list(md.items()):
            if not isinstance(cells, dict):
                continue
            for kind in ("decode", "prefill"):
                s = cells.get(kind, [])
                if s:
                    cells[kind + "_mean"] = st.mean(s)
                    cells[kind + "_median"] = st.median(s)
                    cells[kind + "_min"] = min(s)
                    cells[kind + "_max"] = max(s)
                    cells[kind + "_cv"] = st.pstdev(s) / st.mean(s)
                    cells[kind + "_n"] = len(s)
            v = d / "vram" / f"{label}_d{depth}.json"
            if v.exists():
                cells["vram_peak_mib"] = json.loads(v.read_text(encoding="utf-8"))["peak_mib"]
        loads = []
        for r in range(2, 5):
            t = d / "load" / f"{label}_r{r}.txt"
            if t.exists():
                mm = re.search(r"load_ms ([\d.]+)", t.read_text())
                if mm:
                    loads.append(float(mm.group(1)))
        if loads:
            md["load_ms_warm_median"] = st.median(loads)
        t1 = d / "load" / f"{label}_r1.txt"
        if t1.exists():
            mm = re.search(r"load_ms ([\d.]+)", t1.read_text())
            md["load_ms_first"] = float(mm.group(1)) if mm else None
    return out


def quality(tag: str) -> dict:
    d = P / "artifacts" / "cqw_quality" / tag
    out: dict = {"tag": tag, "kl": {}, "tasks": {}}
    for f in sorted(d.glob("kl_*.log")):
        text = f.read_text(encoding="utf-8", errors="replace")
        row = {}
        for k, pat in PAT.items():
            m = re.search(pat, text)
            row[k] = [float(g) for g in m.groups()] if m else None
        out["kl"][f.stem] = row
    for t in sorted(d.glob("tasks_*/summary.json")):
        s = json.loads(t.read_text(encoding="utf-8"))
        for label, c in s["configs"].items():
            out["tasks"][label] = {"pass": c["pass"], "n": c["n"]}
        # identical outputs vs the group base
        recs = {}
        for rf in t.parent.glob("records_*.json"):
            recs[rf.stem[len("records_"):]] = {r["case_id"]: r.get("output") for r in json.loads(rf.read_text(encoding="utf-8"))}
        base = t.parent.name[len("tasks_"):]
        if base in recs:
            for label, outs in recs.items():
                same = sum(1 for k, v in outs.items() if recs[base].get(k) == v and v is not None)
                out["tasks"][label]["identical_outputs_vs_base"] = same
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", nargs="*", default=[])
    ap.add_argument("--quality", nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = {"speed": [speed(t) for t in a.speed], "quality": [quality(t) for t in a.quality]}
    a.out.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    lines = []
    for s in res["speed"]:
        lines.append(f"## speed {s['tag']}\n")
        lines.append("| model | size MiB | load ms (warm) | depth | decode tok/s median [min-max] (mean, CV) | prefill tok/s median (CV) | VRAM peak MiB |")
        lines.append("|---|---|---|---|---|---|---|")
        for label, md in s["models"].items():
            for depth in sorted(k for k in md if k.isdigit()):
                c = md[depth]
                lines.append(f"| {label} | {md['model_size']/2**20:.0f} | {md.get('load_ms_warm_median', float('nan')):.0f} | {depth} | "
                             f"{c.get('decode_median', float('nan')):.1f} [{c.get('decode_min', float('nan')):.0f}-{c.get('decode_max', float('nan')):.0f}] ({c.get('decode_mean', float('nan')):.1f}, {100*c.get('decode_cv', 0):.1f}%, n={c.get('decode_n', 0)}) | "
                             f"{c.get('prefill_median', float('nan')):.0f} ({100*c.get('prefill_cv', 0):.1f}%) | {c.get('vram_peak_mib') or float('nan'):.0f} |")
        lines.append("")
    for q in res["quality"]:
        lines.append(f"## quality {q['tag']}\n")
        lines.append("| run | mean KLD | p99 KLD | same top-1 % | PPL ratio |")
        lines.append("|---|---|---|---|---|")
        for k, r in q["kl"].items():
            f = lambda v, i=0: (f"{v[i]:.5f}" if v else "n/a")
            lines.append(f"| {k} | {f(r['kld_mean'])} | {f(r['kld_p99'])} | {r['same_top_pct'][0] if r['same_top_pct'] else 'n/a'} | {f(r['ppl_ratio'])} |")
        lines.append("")
        lines.append("| model | tasks pass / n | identical outputs vs group base |")
        lines.append("|---|---|---|")
        for k, r in q["tasks"].items():
            lines.append(f"| {k} | {r['pass']} / {r['n']} | {r.get('identical_outputs_vs_base', 'n/a')} |")
        lines.append("")
    a.out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
