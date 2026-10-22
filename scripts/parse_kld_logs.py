"""Parse llama-perplexity --kl-divergence logs into one JSON/Markdown table (read-only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

PATTERNS = {
    "ppl_q": r"Mean PPL\(Q\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)",
    "ppl_base": r"Mean PPL\(base\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)",
    "ppl_ratio": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)",
    "kld_mean": r"Mean\s+KLD:\s*([-\d.]+)\s*±\s*([\d.]+)",
    "kld_median": r"Median\s+KLD:\s*([-\d.]+)",
    "kld_p99": r"99\.0%\s+KLD:\s*([-\d.]+)",
    "kld_max": r"Maximum KLD:\s*([-\d.]+)",
    "same_top_pct": r"Same top p:\s*([\d.]+)\s*±\s*([\d.]+)",
    "rms_dp_pct": r"RMS Δp\s*:\s*([\d.]+)\s*±\s*([\d.]+)",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--n-params", type=int, default=None, help="for bits per weight over all tensors")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for label in args.labels:
        log = args.work / f"kld_{label}.log"
        text = log.read_text(encoding="utf-8", errors="replace")
        row = {"label": label, "log": str(log), "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
        for key, pat in PATTERNS.items():
            m = re.search(pat, text)
            row[key] = [float(g) for g in m.groups()] if m else None
        model = next(args.work.glob(f"*-{label}.gguf"))
        row["file_bytes"] = model.stat().st_size
        q = re.search(r"quant size\s*=\s*([\d.]+) MiB \(([\d.]+) BPW\)",
                      (args.work / f"quantize_{label}.log").read_text(encoding="utf-8", errors="replace")) \
            if (args.work / f"quantize_{label}.log").exists() else None
        row["llama_quant_size_mib"], row["llama_bpw"] = (float(q.group(1)), float(q.group(2))) if q else (None, None)
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("| config | file MiB | bpw | PPL | PPL/PPL(Q8) | mean KLD | median KLD | p99 KLD | same top-1 % |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        f = lambda v, i=0, d=4: "n/a" if v is None else f"{v[i]:.{d}f}"
        print(f"| {r['label']} | {r['file_bytes'] / 2**20:.1f} | {r['llama_bpw'] or 'n/a'} | {f(r['ppl_q'], d=3)} ± {f(r['ppl_q'], 1, 3)} "
              f"| {f(r['ppl_ratio'])} ± {f(r['ppl_ratio'], 1)} | {f(r['kld_mean'], d=5)} ± {f(r['kld_mean'], 1, 5)} "
              f"| {f(r['kld_median'], d=5)} | {f(r['kld_p99'], d=5)} | {f(r['same_top_pct'], d=2)} ± {f(r['same_top_pct'], 1, 2)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
