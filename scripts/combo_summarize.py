"""Summarize the combined-fork A-D series (artifacts/combo/) into JSON + Markdown tables.

Speed: llama-bench decode = tg64 tok/s, prefill = pp512 tok/s, both after DEPTH filled tokens; medians over all samples
(REPS per run x runs in ABBA order). VRAM: peak per-process dedicated usage (WDDM) of one llama-bench run at that depth;
"pool" = same with CQ_FA_SCRATCH_POOL=1. Load: llama-cq-logits --load-only, median of runs 2..4 (warm OS cache).
Quality: llama-perplexity KL at context 4096 vs row A of the pair and vs BF16 weights + F16 KV; 39-case tasks.

Usage: python combo_summarize.py   (writes artifacts/combo/summary.json and summary.md)
"""
from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

P = Path(r"/path/to/repo")
C = P / "artifacts" / "combo"

# row roles per group: label -> (row, description, kind)   kind: base | lossless | lossy | kernel
ROLES = {
    "bf16_f16": [
        ("BF16_f16", "A", "BF16 weights + F16 KV", "base"),
        ("W4_f16", "B", "W4 weights (lossless, fixed-rate) + F16 KV", "lossless"),
        ("H_f16", "B", "H weights (lossless, Huffman) + F16 KV", "lossless"),
        ("W3_f16", "B", "W3 weights (lossy, 3-bit window) + F16 KV", "lossy"),
        ("BF16ffnQ8_f16", "B", "FFN in Q8_0, rest BF16 (lossy recipe) + F16 KV", "lossy"),
        ("BF16_f16e3", "C", "BF16 weights + f16e3 KV (lossless)", "lossless"),
        ("BF16_f16d4", "C", "BF16 weights + f16d4 KV (lossy, 12 bit)", "lossy"),
        ("W4_f16e3", "D", "W4 weights + f16e3 KV (both lossless)", "lossless"),
        ("H_f16e3", "D", "H weights + f16e3 KV (both lossless)", "lossless"),
        ("W3_f16d4", "D", "W3 weights + f16d4 KV (both lossy)", "lossy"),
        ("BF16ffnQ8_f16d4", "D", "FFN-Q8_0 recipe + f16d4 KV (both lossy)", "lossy"),
    ],
    "q8_q8": [
        ("Q8_0_q8_0", "A", "Q8_0 weights + Q8_0 KV", "base"),
        ("Q8ffnQ6K_q8_0", "B", "FFN in Q6_K (lossy recipe) + Q8_0 KV", "lossy"),
        ("Q8_0_q8_0_gqa", "C", "Q8_0 weights + Q8_0 KV, our GQA decode kernel (same bytes)", "kernel"),
        ("Q8ffnQ6K_q8_0_gqa", "D", "FFN-Q6_K recipe + Q8_0 KV with our GQA kernel", "lossy"),
    ],
    "q4km_q4": [
        ("Q4_K_M_q4_0", "A", "Q4_K_M weights + Q4_0 KV", "base"),
        ("Q4embQ4K_q4_0", "B", "embedding+output in Q4_K (lossy recipe) + Q4_0 KV", "lossy"),
        ("Q4imat_q4_0", "B", "Q4_K_M with imatrix (same size) + Q4_0 KV", "lossy"),
        ("Q4_K_M_q4o", "C", "Q4_K_M weights + q4o KV (lossy, 5.0 bit, +11% KV)", "lossy"),
        ("Q4_K_M_q4_0_gqa", "C", "Q4_K_M weights + Q4_0 KV, our GQA decode kernel (same bytes)", "kernel"),
        ("Q4embQ4K_q4o", "D", "emb-Q4_K recipe + q4o KV", "lossy"),
        ("Q4embQ4K_q4_0_gqa", "D", "emb-Q4_K recipe + Q4_0 KV with our GQA kernel", "lossy"),
    ],
    "q4b_kv": [
        ("4B_f16", "A(F16)", "Qwen3-4B Q4_K_M + F16 KV", "base"),
        ("4B_f16e3", "C", "+ f16e3 KV (lossless)", "lossless"),
        ("4B_f16d4", "C", "+ f16d4 KV (lossy, 12 bit)", "lossy"),
        ("4B_q8_0", "A(Q8_0)", "+ Q8_0 KV (stock kernel)", "base"),
        ("4B_q8_0_gqa", "C", "+ Q8_0 KV, our GQA kernel", "kernel"),
        ("4B_q4_0", "A(Q4_0)", "+ Q4_0 KV (stock kernel)", "base"),
        ("4B_q4_0_gqa", "C", "+ Q4_0 KV, our GQA kernel", "kernel"),
        ("4B_q4o", "C", "+ q4o KV (lossy, 5.0 bit)", "lossy"),
    ],
}
ROLES["bf16_f16_rerun"] = [r for r in ROLES["bf16_f16"] if r[0] in (
    "BF16_f16", "W4_f16", "W3_f16", "BF16_f16e3", "BF16_f16d4", "W4_f16e3", "W3_f16d4")]
ROLES["bf16_f16_rerun"].insert(1, ("BF16_f16_gqa", "C", "BF16 weights + F16 KV, our GQA decode kernel (same bytes)", "kernel"))
ROLES["q4km_q4_rerun"] = [r for r in ROLES["q4km_q4"] if r[0] != "Q4imat_q4_0"]
# base row for ratios inside q4b_kv: each KV family against its own base
Q4B_BASE = {"4B_f16": "4B_f16", "4B_f16e3": "4B_f16", "4B_f16d4": "4B_f16", "4B_q8_0": "4B_q8_0", "4B_q8_0_gqa": "4B_q8_0",
            "4B_q4_0": "4B_q4_0", "4B_q4_0_gqa": "4B_q4_0", "4B_q4o": "4B_q4_0"}

# KV bytes per value (packed formats incl. block headers); f16e3 excludes its overflow pool (16 MiB default)
KV_BITS = {"f16": 16.0, "f16e3": 15.25, "f16d4": 12.0, "q8_0": 8.5, "q4_0": 4.5, "q4o": 5.0}
# KV geometry: (layers with KV, KV heads, head dim)
GEOM = {"2B": (6, 2, 256), "4B": (36, 8, 128)}

KL_PAT = {
    "kld_mean": r"Mean\s+KLD:\s*([-\d.]+)\s*±\s*([\d.]+)",
    "kld_p99": r"99\.0%\s+KLD:\s*([-\d.]+)",
    "same_top_pct": r"Same top p:\s*([\d.]+)\s*±\s*([\d.]+)",
    "ppl_ratio": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([\d.]+)\s*±\s*([\d.]+)",
}


def kv_of(label: str) -> str:
    for k in ("f16e3", "f16d4", "q8_0", "q4_0", "q4o", "f16"):
        if f"_{k}_" in label + "_":
            return k
    raise ValueError(label)


def kv_mib(label: str, ctx: int) -> float:
    lay, heads, dim = GEOM["4B" if label.startswith("4B") else "2B"]
    return lay * heads * dim * 2 * ctx * KV_BITS[kv_of(label)] / 8 / 2**20


def stats(s: list[float]) -> dict:
    return {"median": st.median(s), "mean": st.mean(s), "min": min(s), "max": max(s),
            "cv": st.pstdev(s) / st.mean(s) if len(s) > 1 else 0.0, "n": len(s)}


def speed(tag: str) -> dict:
    d = C / tag
    out: dict = {}
    for f in sorted((d / "bench").glob("*.jsonl")):
        m = re.match(r"\d+-d(\d+)-b(\d)-(fwd|bwd)-(.+)\.jsonl", f.name)
        if not m:
            continue
        depth, label = m.group(1), m.group(4)
        row = out.setdefault(label, {"depths": {}})
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            kind = "decode" if r["n_gen"] > 0 else "prefill"
            row["depths"].setdefault(depth, {}).setdefault(kind + "_samples", []).extend(r["samples_ts"])
            row["depths"][depth].setdefault(kind + "_run_medians", []).append(st.median(r["samples_ts"]))
            row["model_size_mib"] = r["model_size"] / 2**20
    for label, row in out.items():
        for depth, cell in row["depths"].items():
            for kind in ("decode", "prefill"):
                s = cell.get(kind + "_samples")
                if s:
                    cell[kind] = stats(s)
                    # interference from other desktop GPU/CPU users only slows a run down: the fastest run is a
                    # second, one-sided estimate (reported next to the median)
                    cell[kind]["best_run_median"] = max(cell[kind + "_run_medians"])
            for suffix, key in (("", "vram_mib"), ("_pool", "vram_pool_mib")):
                v = d / "vram" / f"{label}_d{depth}{suffix}.json"
                if v.exists():
                    j = json.loads(v.read_text(encoding="utf-8"))
                    cell[key] = j["peak_mib"] if j.get("exit") == 0 else None
                    cell[key + "_exit"] = j.get("exit")
            for suffix, key in (("", "vram_dxgi_mib"), ("_pool", "vram_dxgi_pool_mib")):
                e = d / "vram_report" / f"{label}_d{depth}{suffix}.err"
                if e.exists():
                    txt = e.read_text(encoding="utf-8", errors="replace")
                    vals = [float(x) for x in re.findall(r"local_mib=([\d.]+)", txt)]
                    pool = [int(x) for x in re.findall(r"f16e3_pool_used_blocks=(\d+)", txt)]
                    cell[key] = max(vals) if vals else None
                    if pool:
                        cell[key.replace("mib", "f16e3_pool_used_blocks_max")] = max(pool)
            cell["kv_mib_analytic"] = kv_mib(label, int(depth) + 512 + 64)
        mname = label[: label.index("_" + kv_of(label))]
        loads = []
        for r in (2, 3, 4):
            t = d / "load" / f"{mname}_r{r}.txt"
            if t.exists():
                mm = re.search(r"load_ms ([\d.]+)", t.read_text())
                if mm:
                    loads.append(float(mm.group(1)))
        if loads:
            row["load_ms_warm_median"] = st.median(loads)
            row["load_ms_warm_all"] = loads
    return out


def quality() -> dict:
    d = C / "quality"
    out: dict = {"kl": {}, "tasks": {}}
    for f in sorted(d.glob("kl_*.log")):
        text = f.read_text(encoding="utf-8", errors="replace")
        row = {}
        for k, pat in KL_PAT.items():
            m = re.search(pat, text)
            row[k] = [float(g) for g in m.groups()] if m else None
        out["kl"][f.stem[3:]] = row
    for t in sorted(d.glob("tasks_*/summary.json")):
        s = json.loads(t.read_text(encoding="utf-8"))
        recs = {}
        for rf in t.parent.glob("records_*.json"):
            recs[rf.stem[len("records_"):]] = {r["case_id"]: r.get("output") for r in json.loads(rf.read_text(encoding="utf-8"))}
        for label, c in s["configs"].items():
            out["tasks"][f"{t.parent.name[6:]}:{label}"] = {"pass": c["pass"], "n": c["n"], "outputs": recs.get(label, {})}
    return out


def task_key(label: str) -> str | None:
    """map a matrix label to the task-run key 'kv[_gqa]:MODEL'"""
    kv = kv_of(label)
    model = label[: label.index("_" + kv)]
    g = "_gqa" if label.endswith("_gqa") else ""
    return f"{kv}{g}:{model}"


def main() -> int:
    res: dict = {"groups": {}, "quality": quality()}
    q = res["quality"]
    base_tasks = {"bf16_f16": "f16:BF16", "q8_q8": "q8_0:Q8_0", "q4km_q4": "q4_0:Q4_K_M"}
    lines = ["# Combined fork (cq/combined) A-D matrix - generated by scripts/combo_summarize.py", ""]
    for tag, roles in ROLES.items():
        if not (C / tag).exists():
            continue
        sp = speed(tag)
        res["groups"][tag] = {"speed": sp}
        depths = sorted({int(x) for r in sp.values() for x in r["depths"]})
        lines.append(f"## {tag}\n")
        lines.append("| Depth | Row | Config | kind | Model MiB (vs A) | KV MiB analytic | VRAM MiB (Δ vs A) | VRAM with pooled scratch MiB | Decode tok/s (vs A) [min..max] CV n; best run (vs A best) | Prefill tok/s (vs A) | Load ms | KL vs A | top-1 vs A % | KL vs BF16+F16 | tasks/39 (same outputs as A) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for depth in depths:
            for label, row, desc, kind in roles:
                if label not in sp or str(depth) not in sp[label]["depths"]:
                    lines.append(f"| {depth} | {row} | {label} | {kind} | not run | | | | | | | | | | |")
                    continue
                base = Q4B_BASE.get(label, roles[0][0])
                b = sp.get(base, {}).get("depths", {}).get(str(depth), {})
                c = sp[label]["depths"][str(depth)]
                bm = sp.get(base, {}).get("model_size_mib")
                def rat(x, y):
                    return f"{x / y:.3f}" if x and y else "n/a"
                dec, pre = c.get("decode"), c.get("prefill")
                bd, bp = b.get("decode"), b.get("prefill")
                # primary: fork DXGI report (deterministic, after every decode); fallback: ~1 s WDDM sampling peak
                vr = c.get("vram_dxgi_mib") or c.get("vram_mib")
                bvr = b.get("vram_dxgi_mib") or b.get("vram_mib")
                c["vram_primary_mib"] = vr
                c["vram_primary_src"] = "dxgi" if c.get("vram_dxgi_mib") else "wddm"
                c["vram_pool_primary_mib"] = c.get("vram_dxgi_pool_mib") or c.get("vram_pool_mib")
                kl_a = q["kl"].get(f"{label}_vs_{roles[0][0]}") if tag != "q4b_kv" else None
                kl_r = q["kl"].get(f"{label}_vs_BF16_f16") if tag != "q4b_kv" else None
                if label == "BF16_f16":
                    kl_r = None
                tk = task_key(label) if tag != "q4b_kv" else None
                tr = q["tasks"].get(tk) if tk else None
                tasks_s = ""
                if tr:
                    bt = q["tasks"].get(base_tasks.get(tag, ""), {})
                    same = sum(1 for k, v in tr["outputs"].items() if v is not None and bt.get("outputs", {}).get(k) == v)
                    tasks_s = f"{tr['pass']} ({same}/{tr['n']})"
                    tr_s = {"pass": tr["pass"], "n": tr["n"], "same_outputs_as_A": same}
                    c["tasks"] = tr_s
                c["kl_vs_A"] = kl_a
                c["kl_vs_ref"] = kl_r
                dec_s = (f"{dec['median']:.1f} ({rat(dec['median'], bd and bd['median'])}) [{dec['min']:.0f}..{dec['max']:.0f}] "
                         f"{100 * dec['cv']:.1f}% {dec['n']}; {dec['best_run_median']:.1f} "
                         f"({rat(dec['best_run_median'], bd and bd['best_run_median'])})") if dec else "failed"
                pre_s = f"{pre['median']:.0f} ({rat(pre['median'], bp and bp['median'])})" if pre else "failed"
                if kl_a and kl_a["kld_mean"]:
                    kla_s, top_s = f"{kl_a['kld_mean'][0]:.5f}", f"{kl_a['same_top_pct'][0]:.2f}"
                else:
                    kla_s, top_s = ("0 (A)" if label == roles[0][0] else "n/a"), ""
                klr_s = f"{kl_r['kld_mean'][0]:.5f}" if kl_r and kl_r["kld_mean"] else ""
                pool = c["vram_pool_primary_mib"]
                lines.append(
                    f"| {depth} | {row} | {label} | {kind} | {sp[label]['model_size_mib']:.0f} ({rat(sp[label]['model_size_mib'], bm)}) | "
                    f"{c['kv_mib_analytic']:.0f} | {'n/a' if vr is None else round(vr)} ({'' if (vr is None or bvr is None) else f'{vr - bvr:+.0f}'}) | {'' if pool is None else round(pool)} | "
                    f"{dec_s} | {pre_s} | {sp[label].get('load_ms_warm_median', float('nan')):.0f} | {kla_s} | {top_s} | {klr_s} | {tasks_s} |")
        lines.append("")
    # quality-only listing
    lines.append("## KL runs (context 4096)\n")
    lines.append("| run | mean KLD | p99 KLD | same top-1 % | PPL ratio |")
    lines.append("|---|---|---|---|---|")
    for k, r in q["kl"].items():
        f = lambda v: (f"{v[0]:.5f}" if v else "n/a")
        lines.append(f"| {k} | {f(r['kld_mean'])} | {f(r['kld_p99'])} | {f(r['same_top_pct'])} | {f(r['ppl_ratio'])} |")
    lines.append("")
    lines.append("## Tasks\n")
    for k, r in q["tasks"].items():
        lines.append(f"- {k}: {r['pass']}/{r['n']}")
    for t in res["quality"]["tasks"].values():
        t.pop("outputs", None)
    (C / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (C / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {C / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
