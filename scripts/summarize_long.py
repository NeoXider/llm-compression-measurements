"""Summarize the long-context batch (artifacts/gpu_long_20261021) into summary.md / summary.json."""
from __future__ import annotations

import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

A = Path(sys.argv[1])
out = {}
md = []

# memory
rows = json.loads((A / "memory_filled/rows.json").read_text())
md.append("## Filled-context per-process VRAM (MiB)\n\n| ctx | prompt tokens | KV | exit | after init | sampled peak | KV buffer | compute buffer | prefill s (diag) |\n|---|---|---|---|---|---|---|---|---|")
for r in rows:
    md.append(f"| {r['ctx']} | {r.get('prompt_tokens')} | {r['kv']} | {r['exit']} | {r.get('local_after_init_mib')} | {r.get('local_sampled_peak_mib')} | "
              f"{r.get('kv_buffer_mib')} | {r.get('compute_buffer_mib')} | {None if r.get('prompt_ms_diag') is None else round(r['prompt_ms_diag'] / 1000, 2)} |")
out["memory"] = rows

# KL at 32K
P = {"ppl": r"Mean PPL\(Q\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)", "ratio": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)",
     "kld": r"Mean\s+KLD:\s*([-\d.]+)\s*±\s*([\d.]+)", "p99": r"99\.0%\s+KLD:\s*([-\d.]+)", "top": r"Same top p:\s*([\d.]+)\s*±\s*([\d.]+)"}
md.append("\n## KV KL vs F16 KV at context 32768 (1 chunk, doubled corpus)\n\n| KV | PPL | PPL ratio | mean KLD | p99 KLD | same top-1 % |\n|---|---|---|---|---|---|")
base = (A / "kld_base_f16_c32768.log").read_text(encoding="utf-8", errors="replace")
m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", base)
md.append(f"| f16 (ref) | {m.group(1) if m else 'n/a'} | 1 | 0 | 0 | 100 |")
out["kl32k"] = {}
for kv in ("q8_0", "q4_0"):
    t = (A / f"kld_{kv}_c32768.log").read_text(encoding="utf-8", errors="replace")
    g = {k: (re.search(p, t).groups() if re.search(p, t) else None) for k, p in P.items()}
    out["kl32k"][kv] = g
    f = lambda k, i=0: g[k][i] if g[k] else "n/a"
    md.append(f"| {kv} | {f('ppl')} | {f('ratio')} ± {f('ratio', 1)} | {f('kld')} ± {f('kld', 1)} | {f('p99')} | {f('top')} ± {f('top', 1)} |")

# NIAH
nj = A / "niah/niah.json"
if nj.exists():
    n = json.loads(nj.read_text())
    out["niah"] = n
    md.append("\n## Needle in a haystack (5 depths each)\n\n| KV | length | passed | prompt tokens | errors |\n|---|---|---|---|---|")
    for kv in ("f16", "q8_0", "q4_0"):
        for L in sorted({r["length"] for r in n}):
            rs = [r for r in n if r["kv"] == kv and r["length"] == L]
            if rs:
                md.append(f"| {kv} | {L} | {sum(r['pass'] for r in rs)}/{len(rs)} | {sorted({r.get('prompt_tokens') for r in rs if r.get('prompt_tokens')})} | {sum(1 for r in rs if r.get('error'))} |")

# speed
data = {}
for fpath in sorted((A / "speed").glob("*.jsonl")):
    for line in fpath.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith("{"):
            r = json.loads(line)
            key = (r["type_k"], int(r.get("n_depth", 0)), "prefill" if r.get("n_gen", 0) == 0 else "decode")
            data.setdefault(key, []).extend(r.get("samples_ts", []))
errs = {}
for e in sorted((A / "speed").glob("*.err")):
    t = e.read_text(encoding="utf-8", errors="replace")
    if re.search(r"out of memory|failed to allocate|cudaMalloc failed|error", t, re.I):
        errs[e.name] = [l for l in t.splitlines() if re.search(r"memory|alloc|error", l, re.I)][-3:]
med = {k: float(np.median(v)) for k, v in data.items() if v}
md.append("\n## llama-bench speed (median tok/s; p5-p95; n samples; ratio vs F16)\n\n| test | depth | KV | n | median | p5-p95 | CV % | vs F16 |\n|---|---|---|---|---|---|---|---|")
out["speed"] = []
for (kv, d, test), v in sorted(data.items(), key=lambda x: (x[0][2], x[0][1], ["f16", "q8_0", "q4_0"].index(x[0][0]))):
    a = np.array(v)
    b = med.get(("f16", d, test))
    row = {"kv": kv, "depth": d, "test": test, "n": len(v), "median": float(np.median(a)), "p5": float(np.percentile(a, 5)),
           "p95": float(np.percentile(a, 95)), "cv": float(100 * a.std() / a.mean()), "ratio": (float(np.median(a)) / b) if b else None}
    out["speed"].append(row)
    ratio_txt = "n/a (no F16)" if row["ratio"] is None else f"{row['ratio']:.3f}"
    md.append(f"| {test} | {d} | {kv} | {row['n']} | {row['median']:.1f} | {row['p5']:.1f}-{row['p95']:.1f} | {row['cv']:.1f} | {ratio_txt} |")
out["speed_errors"] = errs
if errs:
    md.append("\nErrors in speed runs (e.g. out of memory):\n")
    for k, v in errs.items():
        md.append(f"- {k}: " + " / ".join(v))

# PoC-B
pb = [json.loads(l) for l in (A / "poc_b.jsonl").read_text().splitlines() if l.startswith("{\"T\"")]
out["pocb"] = pb
md.append("\n## PoC-B read microbenchmark (median ms; F16-time / format-time)\n\n| T | F16 ms (GB/s) | Q8_0 | Q4_0 | EXP3 | EXP3 ratio | bit-exact |\n|---|---|---|---|---|---|---|")
for T in sorted({r["T"] for r in pb}):
    rr = {r["format"]: r for r in pb if r["T"] == T}
    f = rr["f16"]["median_ms"]
    md.append(f"| {T} | {f:.4f} ({rr['f16']['GBps_stored']}) | {rr['q8_0']['median_ms']:.4f} / {f / rr['q8_0']['median_ms']:.2f}x | "
              f"{rr['q4_0']['median_ms']:.4f} / {f / rr['q4_0']['median_ms']:.2f}x | {rr['exp3']['median_ms']:.4f} / {f / rr['exp3']['median_ms']:.2f}x | "
              f"{rr['exp3']['ratio_vs_f16']} | {rr['exp3'].get('bit_exact')} |")

# telemetry
tel = list(csv.DictReader(open(A / "telemetry_1s.csv")))
pw = [float(r["power.draw"]) for r in tel]
md.append(f"\nTelemetry: {len(tel)} s, power median {st.median(pw):.0f} W max {max(pw):.0f} W, max temp "
          f"{max(float(r['temperature.gpu']) for r in tel):.0f} C, throttle codes {sorted(set(r['clocks_throttle_reasons.active'] for r in tel))}")
(A / "summary.md").write_text("\n".join(md), encoding="utf-8")
(A / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("\n".join(md))
