# mf_common.py -- shared helpers and data loaders for make_more_figures.py (and mf_*.py parts).
# Palette (FIXED): gray #8c8c8c = base (no compression), green #1b9e77 = ours LOSSLESS, orange #e66101 = ours LOSSY.
# Hatch = CPU-only prototype.  "n/m" = not measured.  Green arrow "better" on every data axis.
# Every number is copied from RESULTS.md (R-numbers), COMPARISON_TABLE.md, PAPER_V1_EN.md or artifacts/.
import os, sys, json, statistics as st, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FIG = os.environ.get("MF_FIG_DIR") or os.path.join(ROOT, "figures"); ART = os.path.join(ROOT, "artifacts")
# MF_NO_RU=1: English-only figures for the public release (Russian subtitle and Russian text fragments removed)
NO_RU = os.environ.get("MF_NO_RU") == "1"
os.makedirs(FIG, exist_ok=True)
BASE, LL, LY = "#8c8c8c", "#1b9e77", "#e66101"
LLL = "#8fd3bd"          # light green (fused / secondary lossless)
AMBER = "#e6ab02"        # only for traffic-light "doubtful"
G = "#2e7d32"            # the green "better" arrow
GR = {"F16": "#4d4d4d", "Q8_0": "#8c8c8c", "Q4_0": "#bdbdbd"}   # the three bases = three grays
INK = "#222222"; MUTED = "#555555"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110, "savefig.dpi": 150, "svg.fonttype": "path",
                     "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "legend.fontsize": 10, "hatch.linewidth": 1.0})
STAR_NOTE = "* analytic projection, to be validated on this hardware (not a measurement)"


def better(ax, d, at=None, L=0.16, label="better", fs=11, lab_right=False):
    """green 'better' arrow in axes fractions. d = direction of improvement."""
    vec = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0),
           "dl": (-.7, -.7), "ul": (-.7, .7), "ur": (.7, .7), "dr": (.7, -.7)}[d]
    if at is None:
        at = {"up": (0.965, 0.70), "down": (0.965, 0.30), "left": (0.30, 0.04), "right": (0.70, 0.04),
              "dl": (0.30, 0.30), "ul": (0.30, 0.70), "ur": (0.70, 0.70), "dr": (0.70, 0.30)}[d]
    x0, y0 = at; x1, y1 = x0 + vec[0] * L, y0 + vec[1] * L
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=G, lw=2.6, mutation_scale=16), zorder=20, annotation_clip=False)
    if d in ("up", "down"):
        ax.text(x0 + (0.015 if lab_right else -0.015), (y0 + y1) / 2, label, transform=ax.transAxes, ha="left" if lab_right else "right", va="center", fontsize=fs, color=G,
                fontweight="bold", rotation=90, zorder=20)
    elif d in ("left", "right"):
        ax.text((x0 + x1) / 2, y0 + 0.025, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=fs, color=G,
                fontweight="bold", zorder=20)
    else:
        ax.text(max(x0, x1) + 0.015, (y0 + y1) / 2, label, transform=ax.transAxes,
                ha="left", va="center", fontsize=fs, color=G, fontweight="bold", zorder=20)


def _strip_ru(fig):
    """English-only mode: drop ' / <Russian>' fragments and lines that still contain Cyrillic."""
    import re
    from matplotlib.text import Text
    cyr = re.compile(r"")
    for tx in fig.findobj(Text):
        s = tx.get_text()
        if not cyr.search(s):
            continue
        out = []
        for line in s.split("\n"):
            line = re.sub(r"", "", line)
            if not cyr.search(line):
                out.append(line)
        tx.set_text("\n".join(out))


def finish(fig, name, en, ru, note=None, top=0.87, bottom=0.07, left=0.0, right=1.0, tl=True, hpad=1.2, wpad=1.2):
    if tl:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout(rect=(left, bottom, right, top), h_pad=hpad, w_pad=wpad)
    fig.suptitle(en, fontsize=17, fontweight="bold", y=0.995, color=INK)
    if not NO_RU:
        fig.text(0.5, top + 0.035, ru, ha="center", va="center", fontsize=12.5, color=MUTED, style="italic")
    else:
        _strip_ru(fig)
    if note:
        fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=9.5, color=MUTED)
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", pad_inches=0.3, facecolor="white"); fig.savefig(os.path.join(FIG, name + ".svg"), bbox_inches="tight", pad_inches=0.3)
    plt.close(fig); print("saved", name)


def legend_fig(fig, items, y=0.045, ncol=None, fs=10.5):
    """items: (label, color[, hatch[, kind]]) kind in box/line/dash"""
    hs = []
    for it in items:
        lab, col = it[0], it[1]
        hatch = it[2] if len(it) > 2 else None
        kind = it[3] if len(it) > 3 else "box"
        if kind == "box":
            hs.append(Patch(facecolor=col, edgecolor="white" if hatch else col, hatch=hatch, label=lab))
        elif kind == "line":
            hs.append(Line2D([0], [0], color=col, lw=2.5, label=lab))
        else:
            hs.append(Line2D([0], [0], color=col, lw=2, ls="--", label=lab))
    fig.legend(handles=hs, loc="lower center", ncol=ncol or len(hs), frameon=False, fontsize=fs, bbox_to_anchor=(0.5, y))


def std_legend(fig, hatch=True, extra=None, y=0.04):
    it = [("Base (no compression)", BASE), ("Ours, LOSSLESS", LL), ("Ours, LOSSY", LY)]
    if hatch: it.append(("hatched = CPU prototype only", "#bbbbbb", "////"))
    if extra: it += extra
    legend_fig(fig, it, y=y)


CTX_X = [4096, 32768, 65536, 122880]; CTX_L = ["4K", "32K", "64K", "120K"]


def ctx_axis(ax, ticks=CTX_X, labels=CTX_L):
    ax.set_xscale("log", base=2); ax.set_xticks(ticks); ax.set_xticklabels(labels); ax.minorticks_off()


def jl(path):
    out = []
    for l in open(path, encoding="utf-8"):
        l = l.strip()
        if l.startswith("{"): out.append(json.loads(l))
    return out


POC_B = [r for r in jl(os.path.join(ART, "gpu_long_20261021", "poc_b.jsonl")) if "format" in r]      # R26 (near-clean)
POC_C = jl(os.path.join(ART, "gpu_bases_20261021", "poc_c.jsonl"))                                       # R33
KVQ = json.load(open(os.path.join(ART, "kv_quant_lossless_20261021", "results.json")))["results"]        # R33
KVE = json.load(open(os.path.join(ART, "kv_entropy_20261021", "kv_entropy.json")))                      # R9


def pocb(fmt, T, key): return [r for r in POC_B if r["format"] == fmt and r["T"] == T][0][key]
def pocc(fmt, T, key): return [r for r in POC_C if r["format"].startswith(fmt) and r["T"] == T][0][key]


def box(ax, x, y, w, h, text, fc, ec=None, fs=11, tc="white", bold=False, hatch=None, lw=1.2, round_=0.04, va="center", ha="center"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={round_}", fc=fc, ec=ec or fc, lw=lw, hatch=hatch)
    ax.add_patch(p)
    if text:
        tx = x + w / 2 if ha == "center" else x + 0.08
        ax.text(tx, y + h / 2, text, ha=ha, va=va, fontsize=fs, color=tc, fontweight="bold" if bold else "normal")
    return p


def arrow(ax, x0, y0, x1, y1, color=INK, lw=2.2, style="-|>", ms=18, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def nm_text(ax, x, y, fs=10, rot=90, color="#666666", txt="n/m"):
    ax.text(x, y, txt, ha="center", va="bottom", fontsize=fs, color=color, rotation=rot)


def heat(ax, M, rowlabels, collabels, cmap, norm, fmt="{:.2f}", fs=13, header_rows=None, header_text=None,
         nm_hatch="....", rowcolors=None, cellfs=None, gap_rows=()):
    """M: 2D list; nan -> n/m cell. header_rows: indices drawn as text band (header_text[i])."""
    M = np.array(M, float); nr, nc = M.shape
    header_rows = header_rows or []
    for i in range(nr):
        for j in range(nc):
            if i in header_rows:
                continue
            v = M[i, j]
            if np.isnan(v):
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc="#efefef", ec="white", lw=2, hatch=nm_hatch))
                ax.text(j, i, "n/m", ha="center", va="center", fontsize=fs - 1, color="#777")
            else:
                c = cmap(norm(v))
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc=c, ec="white", lw=2))
                lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=cellfs or fs, color="white" if lum < 0.5 else INK, fontweight="bold")
    for i in header_rows:
        ax.add_patch(Rectangle((-.5, i - .5), nc, 1, fc="#f4f4f4", ec="white", lw=2))
        ax.text((nc - 1) / 2, i, header_text[i], ha="center", va="center", fontsize=fs - 2, color=MUTED, fontweight="bold")
    ax.set_xlim(-.5, nc - .5); ax.set_ylim(nr - .5, -.5)
    ax.set_xticks(range(nc)); ax.set_xticklabels(collabels, fontsize=fs); ax.xaxis.tick_top()
    ax.set_yticks(range(nr)); ax.set_yticklabels(rowlabels, fontsize=fs - 1)
    if rowcolors:
        for t, c in zip(ax.get_yticklabels(), rowcolors):
            t.set_color(c)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)


def cb_better(cb, direction="up", label="better"):
    """green 'better' arrow beside a colorbar. direction = where the BETTER values are on the bar."""
    ax = cb.ax
    y0, y1 = (0.30, 0.72) if direction == "up" else (0.72, 0.30)
    ax.annotate("", xy=(8.2, y1), xytext=(8.2, y0), xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=G, lw=2.6, mutation_scale=16), annotation_clip=False)
    ax.text(9.3, 0.51, label, transform=ax.transAxes, rotation=90, ha="center", va="center", color=G, fontweight="bold", fontsize=11)


# red-free diverging map: orange (bad) - white - green (good)
CMAP_GOOD_UP = LinearSegmentedColormap.from_list("ogw", [LY, "#f6d3b8", "#ffffff", "#bfe6d9", LL])
CMAP_GOOD_DOWN = LinearSegmentedColormap.from_list("gwo", [LL, "#bfe6d9", "#ffffff", "#f6d3b8", LY])
