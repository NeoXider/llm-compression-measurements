"""Markdown tables for RESULTS.md R8x from artifacts/combo/summary.json (one table per pair, rows A-D, three depths).
A row: absolute values; other rows: ratio to A (speed), delta to A (VRAM MiB). Decode cells: median ratio, and in
brackets the ratio of the fastest runs (one-sided estimate under desktop interference).
Usage: python combo_results_tables.py [TAG ...]  -> prints Markdown (UTF-8) to artifacts/combo/results_tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from combo_summarize import ROLES, Q4B_BASE  # noqa: E402

P = Path(r"/path/to/repo")
S = json.loads((P / "artifacts" / "combo" / "summary.json").read_text(encoding="utf-8"))
DEPTHS = ["4096", "32768", "122880"]


def fmt_ratio(x, y, nd=2):
    return f"{x / y:.{nd}f}" if x and y else "–"


def main() -> int:
    tags = sys.argv[1:] or list(ROLES)
    out = []
    kl = S["quality"]["kl"]
    for tag in tags:
        g = S["groups"].get(tag)
        if not g:
            continue
        sp = g["speed"]
        roles = ROLES[tag]
        out.append(f"### {tag}\n")
        out.append("| Row | Config | Weights MiB | Load ms | VRAM MiB 4K / 32K / 120K | Decode 4K / 32K / 120K | Prefill 4K / 32K / 120K | KL vs A (same top-1) | KL vs BF16+F16 | Tasks /39 (outputs = A) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for label, row, desc, kind in roles:
            if label not in sp:
                out.append(f"| {row} | {desc} | not run | | | | | | | |")
                continue
            base = Q4B_BASE.get(label, roles[0][0])
            is_base = label == base
            r = sp[label]
            b = sp.get(base, {})
            msz = f"{r['model_size_mib']:.0f}" + ("" if is_base else f" ({fmt_ratio(r['model_size_mib'], b.get('model_size_mib'), 3)})")
            load = f"{r.get('load_ms_warm_median', float('nan')):.0f}"
            v, dec, pre = [], [], []
            for d in DEPTHS:
                c = r["depths"].get(d, {})
                cb = b.get("depths", {}).get(d, {})
                vr, bvr = c.get("vram_primary_mib"), cb.get("vram_primary_mib")
                if is_base:
                    v.append(f"{vr:.0f}" if vr else "–")
                else:
                    v.append(f"{vr - bvr:+.0f}" if vr and bvr else "–")
                cd, bd = c.get("decode"), cb.get("decode")
                cp, bp = c.get("prefill"), cb.get("prefill")
                if is_base:
                    dec.append(f"{cd['median']:.1f}" if cd else "–")
                    pre.append(f"{cp['median']:.0f}" if cp else "–")
                else:
                    dec.append(f"{fmt_ratio(cd and cd['median'], bd and bd['median'])} ({fmt_ratio(cd and cd['best_run_median'], bd and bd['best_run_median'])})" if cd else "–")
                    pre.append(fmt_ratio(cp and cp["median"], bp and bp["median"]))
            ka = kl.get(f"{label}_vs_{roles[0][0]}") if tag != "q4b_kv" else None
            kr = kl.get(f"{label}_vs_BF16_f16") if tag != "q4b_kv" else None
            ka_s = "—" if is_base else (f"{abs(ka['kld_mean'][0]):.5f} ({ka['same_top_pct'][0]:.1f}%)" if ka and ka.get("kld_mean") else "n/m")
            kr_s = "ref" if label == "BF16_f16" else (f"{abs(kr['kld_mean'][0]):.5f} ({kr['same_top_pct'][0]:.1f}%)" if kr and kr.get("kld_mean") else "n/m")
            t = r["depths"].get("4096", {}).get("tasks")
            t_s = f"{t['pass']} ({t['same_outputs_as_A']}/{t['n']})" if t else "n/m"
            unit = " tok/s" if is_base else " ×A"
            out.append(f"| {row} | {desc} | {msz} | {load} | {' / '.join(v)}{'' if is_base else ' Δ'} | {' / '.join(dec)}{unit} | "
                       f"{' / '.join(pre)}{unit} | {ka_s} | {kr_s} | {t_s} |")
        out.append("")
    (P / "artifacts" / "combo" / "results_tables.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote results_tables.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
