"""Summarize Stage 1 (CQ_KV_EMU) results: KL/PPL/top-1 vs F16 and vs own base at 4K/32K, tasks, needle."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

A = Path(sys.argv[1])
P = {"ppl": r"Mean PPL\(Q\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)", "ratio": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)",
     "kld": r"Mean\s+KLD:\s*([-\d.]+)\s*±\s*([\d.]+)", "p99": r"99\.0%\s+KLD:\s*([-\d.]+)", "top": r"Same top p:\s*([\d.]+)\s*±\s*([\d.]+)"}


def parse(name):
    f = A / f"{name}.log"
    if not f.exists():
        return None
    t = f.read_text(encoding="utf-8", errors="replace")
    r = {k: (re.search(p, t).groups() if re.search(p, t) else None) for k, p in P.items()}
    m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", t)
    r["final_ppl"] = m.groups() if m else None
    return r


def signp(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


out = {"kl": {}, "tasks": {}, "niah": {}}
md = ["## Stage 1: quality of our lossy KV variants (CQ_KV_EMU; Qwen3-4B Q4_K_M)\n"]
for ctx in (4096, 32768):
    md.append(f"\n### context {ctx}\n\n| variant | base | KL vs F16 KV | top-1 vs F16 | KL vs own base | top-1 vs own base | PPL |\n|---|---|---|---|---|---|---|")
    rows = [("f16drop2", "F16", None), ("f16drop4", "F16", None), ("q8_0", "Q8_0 (emulated base)", None), ("q8_7bit", "Q8_0", "vsQ8"),
            ("q8_6bit_out1", "Q8_0", "vsQ8"), ("q4_0", "Q4_0 (emulated base)", None), ("q4_out1", "Q4_0", "vsQ4"), ("q3_out1", "Q4_0", "vsQ4"),
            ("real_q8_0", "real llama.cpp q8_0 type", None), ("real_q4_0", "real llama.cpp q4_0 type", None)]
    for e, base, rel in rows:
        a = parse(f"kl_vsF16_{e}_c{ctx}")
        b = parse(f"kl_{rel}_{e}_c{ctx}") if rel else None
        out["kl"][f"{e}_c{ctx}"] = {"vsF16": a, "vsBase": b}
        g = lambda r, k, i=0: (r[k][i] if r and r.get(k) else "n/a")
        md.append(f"| {e} | {base} | {g(a,'kld')} ± {g(a,'kld',1)} | {g(a,'top')}% | {g(b,'kld') if b else '—'} | {g(b,'top') + '%' if b else '—'} | {g(a,'ppl')} |")
    for bn in ("base_f16", "base_q8emu", "base_q4emu"):
        r = parse(f"{bn}_c{ctx}")
        md.append(f"\n{bn} c{ctx}: PPL {r['final_ppl'] if r else 'n/a'}")
# tasks
ref = None
md.append("\n### Tasks (39, GPU, ctx 16384) and needle (5 depths at 4K and 32K)\n\n| mode | pass | lost vs F16 | gained | sign p | needle 4K | needle 32K |\n|---|---|---|---|---|---|---|")
recs = {}
for d in sorted(A.glob("tasks_*")):
    e = d.name[len("tasks_"):]
    f = next(d.glob("records_*.json"), None)
    if f:
        recs[e] = {r["case_id"]: r for r in json.loads(f.read_text(encoding="utf-8"))}
ref = recs.get("none")
for e in ["none", "f16drop2", "f16drop4", "q8_0", "q8_7bit", "q8_6bit_out1", "q4_0", "q4_out1", "q3_out1"]:
    if e not in recs:
        continue
    r = recs[e]
    P_ = lambda x: x["outcome"] == "pass"
    lost = sum(1 for k in r if ref and P_(ref[k]) and not P_(r[k]))
    gain = sum(1 for k in r if ref and P_(r[k]) and not P_(ref[k]))
    nj = A / f"niah_{e}" / "niah.json"
    n4 = n32 = "n/a"
    if nj.exists():
        n = json.loads(nj.read_text())
        n4 = f"{sum(x['pass'] for x in n if x['length'] == 4096)}/{sum(1 for x in n if x['length'] == 4096)}"
        n32 = f"{sum(x['pass'] for x in n if x['length'] == 32768)}/{sum(1 for x in n if x['length'] == 32768)}"
    out["tasks"][e] = {"pass": sum(P_(x) for x in r.values()), "lost": lost, "gained": gain, "p": signp(lost, gain), "niah4k": n4, "niah32k": n32}
    md.append(f"| {e} | {out['tasks'][e]['pass']}/39 | {lost} | {gain} | {signp(lost, gain):.2f} | {n4} | {n32} |")
(A / "summary.md").write_text("\n".join(md), encoding="utf-8")
(A / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("\n".join(md))
