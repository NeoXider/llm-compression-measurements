"""Figures for the combined-fork A-D matrix (R80-R86). Reads artifacts/combo/summary.json (scripts/combo_summarize.py).
Colours as everywhere in the paper: grey = base (row A), green = ours lossless, orange = ours lossy; hatched green = our
decode kernel on an unchanged format (same bytes, not bit-identical). Arrows mark the "better" direction.
Writes figures/fig80..fig84 (PNG + SVG). English labels only.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

P = Path(r"/path/to/repo")
S = json.loads((P / "artifacts" / "combo" / "summary.json").read_text(encoding="utf-8"))
OUT = P / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
BASE, LL, LY, G = "#8c8c8c", "#1b9e77", "#e66101", "#2e7d32"
COL = {"base": BASE, "lossless": LL, "lossy": LY, "kernel": LL}
HATCH = {"kernel": "///"}

# rows shown per pair: (label, short name, kind)
PAIRS = {
    "bf16_f16": ("BF16 weights + F16 KV", [
        ("BF16_f16", "A  BF16 + F16", "base"),
        ("W4_f16", "B  W4 + F16", "lossless"),
        ("H_f16", "B  H + F16", "lossless"),
        ("W3_f16", "B  W3 + F16", "lossy"),
        ("BF16ffnQ8_f16", "B  FFN-Q8_0 + F16", "lossy"),
        ("BF16_f16e3", "C  BF16 + f16e3", "lossless"),
        ("BF16_f16d4", "C  BF16 + f16d4", "lossy"),
        ("W4_f16e3", "D  W4 + f16e3", "lossless"),
        ("H_f16e3", "D  H + f16e3", "lossless"),
        ("W3_f16d4", "D  W3 + f16d4", "lossy"),
        ("BF16ffnQ8_f16d4", "D  FFN-Q8_0 + f16d4", "lossy"),
    ]),
    "q8_q8": ("Q8_0 weights + Q8_0 KV", [
        ("Q8_0_q8_0", "A  Q8_0 + q8_0", "base"),
        ("Q8ffnQ6K_q8_0", "B  FFN-Q6_K + q8_0", "lossy"),
        ("Q8_0_q8_0_gqa", "C  Q8_0 + q8_0, GQA kernel", "kernel"),
        ("Q8ffnQ6K_q8_0_gqa", "D  FFN-Q6_K + q8_0, GQA kernel", "lossy"),
    ]),
    "q4km_q4": ("Q4_K_M weights + Q4_0 KV", [
        ("Q4_K_M_q4_0", "A  Q4_K_M + q4_0", "base"),
        ("Q4embQ4K_q4_0", "B  emb-Q4_K + q4_0", "lossy"),
        ("Q4imat_q4_0", "B  imatrix + q4_0", "lossy"),
        ("Q4_K_M_q4o", "C  Q4_K_M + q4o", "lossy"),
        ("Q4_K_M_q4_0_gqa", "C  Q4_K_M + q4_0, GQA kernel", "kernel"),
        ("Q4embQ4K_q4o", "D  emb-Q4_K + q4o", "lossy"),
        ("Q4embQ4K_q4_0_gqa", "D  emb-Q4_K + q4_0, GQA kernel", "lossy"),
    ]),
}
DEPTHS = ["4096", "32768", "122880"]
DNAME = {"4096": "4K", "32768": "32K", "122880": "120K"}


def legend(fig, y=0.0):
    fig.legend(handles=[Patch(color=BASE, label="Base (row A)"), Patch(color=LL, label="Ours, lossless"),
                        Patch(color=LY, label="Ours, lossy"),
                        Patch(facecolor=LL, hatch="///", edgecolor="white", label="Our decode kernel, same format")],
               loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, y))


def better(ax, direction):
    """direction: 'right' (larger is better) or 'left' (smaller is better) for horizontal bars"""
    if direction == "right":
        ax.annotate("", xy=(0.98, 1.02), xytext=(0.80, 1.02), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=G, lw=1.8))
        ax.text(0.79, 1.02, "better", transform=ax.transAxes, ha="right", va="center", fontsize=7, color=G, fontweight="bold")
    else:
        ax.annotate("", xy=(0.02, 1.02), xytext=(0.20, 1.02), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=G, lw=1.8))
        ax.text(0.21, 1.02, "better", transform=ax.transAxes, ha="left", va="center", fontsize=7, color=G, fontweight="bold")


RERUN = {"bf16_f16": "bf16_f16_rerun", "q4km_q4": "q4km_q4_rerun"}


def cell(tag, label, depth, speed=False):
    """speed=True: prefer the 18-sample --prio 2 rerun (R86) when it has this config and depth"""
    if speed and tag in RERUN:
        c = S["groups"].get(RERUN[tag], {}).get("speed", {}).get(label, {}).get("depths", {}).get(depth)
        b = S["groups"].get(RERUN[tag], {}).get("speed", {}).get(PAIRS[tag][1][0][0], {}).get("depths", {}).get(depth)
        if c and b:
            return c
    return S["groups"].get(tag, {}).get("speed", {}).get(label, {}).get("depths", {}).get(depth, {})


def ratio_grid(metric, fname, title, xlabel):
    fig, axs = plt.subplots(3, 3, figsize=(13, 10.5), sharey="row")
    for r, (tag, (pname, rows)) in enumerate(PAIRS.items()):
        for c, depth in enumerate(DEPTHS):
            ax = axs[r][c]
            base = cell(tag, rows[0][0], depth, True).get(metric)
            ys, names = [], []
            for i, (label, name, kind) in enumerate(rows):
                v = cell(tag, label, depth, True).get(metric)
                same_src = cell(tag, label, depth, True) is not cell(tag, label, depth)
                base_l = cell(tag, rows[0][0], depth, same_src).get(metric)
                y = len(rows) - 1 - i
                ys.append(y)
                names.append(name)
                if not v or not base:
                    ax.text(0.02, y, "not measured", va="center", fontsize=7, color="#666")
                    continue
                if not base_l:
                    ax.text(0.02, y, "not measured", va="center", fontsize=7, color="#666")
                    continue
                rv = v["median"] / base_l["median"]
                lo, hi = v["min"] / base_l["median"], v["max"] / base_l["median"]
                noisy = v.get("cv", 0) > 0.08 or base_l.get("cv", 0) > 0.08
                ax.barh(y, rv, color=COL[kind], hatch=HATCH.get(kind), edgecolor="white", height=0.7, alpha=0.45 if noisy else 1.0)
                ax.errorbar(rv, y, xerr=[[max(rv - lo, 0)], [max(hi - rv, 0)]], color="#333", lw=0.7, capsize=1.5)
                ax.text(rv + 0.02, y + 0.22, f"{rv:.2f}" + (" †" if noisy else ""), va="center", fontsize=6.5)
            ax.axvline(1.0, color="#444", lw=0.8, ls="--")
            ax.set_yticks(ys)
            ax.set_yticklabels(names, fontsize=7)
            ax.set_xlim(0, 1.45)
            if base:
                ax.set_title(f"{pname}, depth {DNAME[depth]}\nA = {base['median']:.0f} tok/s", fontsize=8)
            if r == 2:
                ax.set_xlabel(xlabel)
            better(ax, "right")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    legend(fig)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{fname}.{ext}")
    plt.close(fig)


def vram_fig():
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (tag, (pname, rows)) in zip(axs, PAIRS.items()):
        ys, names = [], []
        for i, (label, name, kind) in enumerate(rows):
            y = len(rows) - 1 - i
            ys.append(y)
            names.append(name)
            spd = S["groups"].get(tag, {}).get("speed", {}).get(label, {})
            c = spd.get("depths", {}).get("122880", {})
            v = c.get("vram_primary_mib")
            m = spd.get("model_size_mib")
            if not v:
                ax.text(10, y, "not measured", va="center", fontsize=7, color="#666")
                continue
            ax.barh(y, v, color=COL[kind], hatch=HATCH.get(kind), edgecolor="white", height=0.7)
            if m:
                ax.barh(y, m, color="black", alpha=0.18, height=0.7)
            ax.text(v + 40, y, f"{v:,.0f}", va="center", fontsize=6.5)
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Process VRAM at a filled 120K context, MiB\n(dark part = model tensors)")
        ax.set_title(pname, fontsize=8.5)
        better(ax, "left")
    fig.suptitle("Combined fork, Qwen3.5-2B: process VRAM with 120K tokens filled (rows A-D)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    legend(fig)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fig82_combo_vram_120k.{ext}")
    plt.close(fig)


def quality_fig():
    kl = S["quality"]["kl"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (tag, (pname, rows)) in zip(axs, PAIRS.items()):
        ys, names = [], []
        for i, (label, name, kind) in enumerate(rows):
            y = len(rows) - 1 - i
            ys.append(y)
            names.append(name)
            if label == "BF16_f16":
                ax.text(1.2e-5, y, "reference", va="center", fontsize=7, color="#333")
                continue
            r = kl.get(f"{label}_vs_BF16_f16")
            if not r or not r.get("kld_mean"):
                ax.text(1.2e-5, y, "not measured", va="center", fontsize=7, color="#666")
                continue
            if abs(r["kld_mean"][0]) < 5e-6:
                ax.text(1.3e-5, y, "0 (same logits as A in this test)", va="center", fontsize=6.5, color=LL, fontweight="bold")
                continue
            v = r["kld_mean"][0]
            ax.barh(y, v, color=COL[kind], hatch=HATCH.get(kind), edgecolor="white", height=0.7)
            ax.text(v * 1.15, y, f"{v:.4f}", va="center", fontsize=6.5)
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 1)
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("KL vs BF16+F16 KV (ctx 4096), log")
        ax.set_title(pname, fontsize=8.5, pad=16)
        better(ax, "left")
    fig.suptitle("Combined fork, Qwen3.5-2B: output quality of rows A-D against the best available model", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    legend(fig)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fig83_combo_quality_kl.{ext}")
    plt.close(fig)


def fig_4b():
    tag = "q4b_kv"
    g = S["groups"].get(tag)
    if not g:
        return
    sp = g["speed"]
    lines = [("4B_f16", "F16 (base)", BASE, "-", None), ("4B_q8_0", "q8_0, stock kernel (base)", BASE, "--", None),
             ("4B_q4_0", "q4_0, stock kernel (base)", BASE, ":", None),
             ("4B_f16e3", "f16e3 (ours, lossless*)", LL, "-", "o"), ("4B_q8_0_gqa", "q8_0, our GQA kernel", LL, "--", "s"),
             ("4B_q4_0_gqa", "q4_0, our GQA kernel", LL, ":", "^"),
             ("4B_f16d4", "f16d4 (ours, lossy 12 bit)", LY, "-", "o"), ("4B_q4o", "q4o (ours, lossy 5 bit)", LY, ":", "D")]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3))
    xs = [4, 32, 120]
    for label, name, colr, ls, mk in lines:
        d = sp.get(label, {}).get("depths", {})
        for ax, metric in ((axs[0], "decode"), (axs[1], "prefill")):
            y = [d.get(k, {}).get(metric, {}).get("median") for k in DEPTHS]
            if any(v is None for v in y):
                continue
            ax.plot(xs, y, color=colr, ls=ls, marker=mk or "o", ms=4, label=name)
    for ax, t in ((axs[0], "Decode (tg64) after N filled tokens"), (axs[1], "Prefill (pp512) after N filled tokens")):
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(["4K", "32K", "120K"])
        ax.set_ylabel("tokens/s")
        ax.set_title(t, fontsize=8.5)
        ax.annotate("", xy=(0.95, 0.98), xytext=(0.95, 0.80), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=G, lw=1.8))
        ax.text(0.93, 0.89, "better", transform=ax.transAxes, fontsize=7, color=G, fontweight="bold", ha="right")
    axs[0].legend(fontsize=6.5, frameon=False)
    ax = axs[2]
    names, vals, cols, hatches = [], [], [], []
    for label, name, colr, ls, mk in lines:
        v = sp.get(label, {}).get("depths", {}).get("122880", {}).get("vram_primary_mib")
        if v:
            names.append(name)
            vals.append(v)
            cols.append(colr)
            hatches.append("///" if "GQA" in name else None)
    y = list(range(len(names)))[::-1]
    for yi, v, c, h in zip(y, vals, cols, hatches):
        ax.barh(yi, v, color=c, hatch=h, edgecolor="white", height=0.7)
        ax.text(v + 150, yi, f"{v:,.0f}", va="center", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Process VRAM, 120K filled, MiB")
    ax.set_title("Memory at 120K", fontsize=8.5, pad=16)
    better(ax, "left")
    fig.suptitle("Combined fork, Qwen3-4B Q4_K_M (KV in all 36 layers): KV types and decode kernels\n* f16e3 is exact only until its 16 MiB overflow pool is full (about 8K tokens here, R81)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fig84_combo_qwen3_4b_kv.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    ratio_grid("decode", "fig80_combo_decode_ad", "Combined fork, Qwen3.5-2B: decode speed of rows A-D relative to row A of the same pair "
               "(median; 4K/32K core rows from the 18-sample rerun; whiskers = min..max;\n"
               "† pale = cell CV > 8% from desktop interference, use with care)", "Decode tok/s relative to A")
    ratio_grid("prefill", "fig81_combo_prefill_ad", "Combined fork, Qwen3.5-2B: prefill speed (pp512 after N filled tokens) relative to row A",
               "Prefill tok/s relative to A")
    vram_fig()
    quality_fig()
    fig_4b()
    print("ok")
