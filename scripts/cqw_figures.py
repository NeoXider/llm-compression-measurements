"""Figures for the weights fork (R60+). Palette as in make_three_types.py: gray = base, green = ours lossless,
orange = ours lossy, green arrow = better. Reads artifacts/cqw_speed/summary_*.json, artifacts/cqw_quality/summary_*.json
and artifacts/cqw_kbench/<tag>/kbench_b1_base.txt. Writes figures/fig40..fig44 (png + svg).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

P = Path(r"/path/to/repo")
FIG = P / "figures"
BASE, LL, LY, G = "#8c8c8c", "#1b9e77", "#e66101", "#2e7d32"
KIND = {  # label -> (kind, display name)
    "BF16": ("base", "BF16 (base)"), "H": ("ll", "H: exponent Huffman"), "W4": ("ll", "W4: 4-bit exp. window"),
    "W3": ("ly", "W3: 3-bit exp. window"), "BF16ffnQ8": ("ly", "FFN Q8_0, rest BF16"),
    "Q8_0": ("base", "Q8_0 (base)"), "Q8ffnQ6K": ("ly", "FFN Q6_K"), "Q8embQ6K": ("ly", "emb/output Q6_K"),
    "Q4_K_M": ("base", "Q4_K_M (base)"), "Q4embQ4K": ("ly", "emb/output Q4_K"), "Q4imat": ("ly", "imatrix (ref.)"),
    "Q4imatFfnIQ3S": ("ly", "imatrix + FFN up/gate IQ3_S"),
}
COL = {"base": BASE, "ll": LL, "ly": LY}
HATCH = {"W4": "//", "BF16ffnQ8": "..", "Q8ffnQ6K": "..", "Q4embQ4K": "..", "Q4imat": "xx"}


def better(ax, direction="up", x=0.97):
    if direction == "up":
        ax.annotate("", xy=(x, 0.95), xytext=(x, 0.78), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=G, lw=2))
        ax.text(x - 0.03, 0.865, "better", transform=ax.transAxes, ha="right", va="center", fontsize=8, color=G, fontweight="bold")
    elif direction == "down":
        ax.annotate("", xy=(x, 0.74), xytext=(x, 0.92), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=G, lw=2))
        ax.text(x - 0.03, 0.83, "better", transform=ax.transAxes, ha="right", va="center", fontsize=8, color=G, fontweight="bold")
    else:  # up-left
        ax.annotate("", xy=(0.05, 0.95), xytext=(0.20, 0.80), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=G, lw=2))
        ax.text(0.21, 0.80, "better", transform=ax.transAxes, ha="left", va="top", fontsize=8, color=G, fontweight="bold")


def legend(fig, y=0.0):
    fig.legend(handles=[Patch(color=BASE, label="base (no compression)"), Patch(color=LL, label="ours, lossless"),
                        Patch(color=LY, label="ours, lossy")], loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, y))


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def kbench(tag_h: str, tag_w: str):
    rows = {}
    for tag, types in ((tag_w, ("bf16", "cq_bf16w4", "cq_bf16w3")), (tag_h, ("bf16", "cq_bf16h"))):
        for line in (P / "artifacts" / "cqw_kbench" / tag / "kbench_b1_base.txt").read_text().splitlines()[1:]:
            f = line.split()
            if f[1] in types:
                rows.setdefault(f[0], {}).setdefault(f[1], float(f[6]))
    names = {"token_embd.weight": "output head\n2048x248320", "blk.5.attn_qkv.weight": "attn_qkv\n2048x6144",
             "blk.5.ffn_up.weight": "ffn_up\n2048x6144", "blk.5.ffn_down.weight": "ffn_down\n6144x2048",
             "blk.5.ssm_out.weight": "ssm_out\n2048x2048"}
    order = [k for k in names if k in rows]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.2
    for i, (t, c, lab) in enumerate((("bf16", BASE, "BF16"), ("cq_bf16h", LL, "H"), ("cq_bf16w4", LL, "W4"), ("cq_bf16w3", LY, "W3"))):
        xs = [j + (i - 1.5)*w for j in range(len(order))]
        ys = [rows[k]["bf16"]/rows[k][t] for k in order]
        ax.bar(xs, ys, w, color=c, edgecolor="white", hatch="//" if t == "cq_bf16w4" else None)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.02, f"{y:.2f}", ha="center", fontsize=6.5, rotation=90)
        ax.text(xs[0], 0.04, lab, ha="center", fontsize=6.5, color="white", rotation=90)
    ax.axhline(1.0, color="#444", lw=1, ls="--")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[k] for k in order], fontsize=7.5)
    ax.set_ylabel("GEMV speed vs BF16 (batch 1)")
    ax.set_ylim(0, 1.7)
    ax.set_title("Fused-decode GEMV per tensor (Qwen3.5-2B, RTX 3090); H/W4 bit-identical to BF16", fontsize=9)
    better(ax, "up")
    legend(fig, -0.06)
    save(fig, "fig60_cqw_gemv_per_tensor")


def group_speed(summary: Path):
    data = json.loads(summary.read_text(encoding="utf-8"))
    models = {}
    for s in data["speed"]:
        for label, md in s["models"].items():
            m = models.setdefault(label, {"size": md["model_size"]/2**20})
            for k, v in md.items():
                if k.isdigit():
                    m.setdefault("depth", {}).setdefault(int(k), {}).update(v)
    return models


def fig_size_speed(groups: dict, name: str):
    fig, axs = plt.subplots(1, len(groups), figsize=(4.2*len(groups), 3.6), squeeze=False)
    for ax, (gname, (models, depth)) in zip(axs[0], groups.items()):
        for label, m in models.items():
            kind, disp = KIND.get(label, ("ly", label))
            d = m.get("depth", {}).get(depth)
            if not d or "decode_mean" not in d:
                continue
            ax.scatter(m["size"], d["decode_mean"], s=60, color=COL[kind], edgecolor="k", zorder=3)
            ax.annotate(disp, (m["size"], d["decode_mean"]), textcoords="offset points", xytext=(5, 4), fontsize=7)
        ax.set_xlabel("model tensors, MiB")
        ax.set_ylabel(f"decode tok/s at depth {depth}")
        ax.set_title(gname, fontsize=9)
        ax.grid(alpha=0.3)
        better(ax, "upleft")
    legend(fig, -0.08)
    save(fig, name)


def fig_depth(models: dict, title: str, name: str):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    labels = list(models)
    depths = sorted({d for m in models.values() for d in m.get("depth", {})})
    w = 0.8/len(labels)
    for ax, key, ylab in ((axs[0], "decode_median", "decode tok/s (median of 12)"), (axs[1], "prefill_median", "prefill tok/s, pp512 (median of 12)")):
        for i, label in enumerate(labels):
            kind, disp = KIND.get(label, ("ly", label))
            xs = [j + (i - (len(labels) - 1)/2)*w for j in range(len(depths))]
            ys = [models[label]["depth"].get(d, {}).get(key, 0) for d in depths]
            ax.bar(xs, ys, w, color=COL[kind], edgecolor="white", label=disp, hatch=HATCH.get(label))
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels([f"{d//1024}K" for d in depths])
        ax.set_xlabel("filled context depth")
        ax.set_ylabel(ylab)
        better(ax, "up")
    axs[0].legend(fontsize=6.5, ncol=1, loc="lower left", framealpha=0.9)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    legend(fig, -0.09)
    save(fig, name)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "kbench"):
        kbench(sys.argv[2] if len(sys.argv) > 2 else "final", sys.argv[3] if len(sys.argv) > 3 else "final")
