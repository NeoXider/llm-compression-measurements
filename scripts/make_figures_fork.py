"""Figures for the forked-llama.cpp results (R43-R47): decode speed vs F16, KV quality vs bits, lazy KV memory.

Palette: grey = base formats, green = ours lossless, orange = ours lossy, blue = ours kernel only (same bytes).
Reads artifacts/fork_speed_a/summary.json, artifacts/fork_speed_b/summary.json (120K, optional),
artifacts/fork_s2_20261022/fill_memory/rows.json, artifacts/fork_s3_20261022/fill_memory_lazy_pooled/rows.json.
Writes figures/fig60_fork_decode_speed.png, fig61_fork_kv_quality.png, fig62_fork_lazy_memory.png. Run from the paper root.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
A = Path("artifacts")
OUT = Path(os.environ.get("FIG_DIR", "figures"))
OUT.mkdir(exist_ok=True)
GREY, GREY2, GREEN, ORANGE, BLUE, G = "#8c8c8c", "#bdbdbd", "#1b9e77", "#e66101", "#2a6fb0", "#2e7d32"


def better(ax, up=True, x=0.97):
    y0, y1 = (0.80, 0.95) if up else (0.95, 0.80)
    ax.annotate("", xy=(x, y1), xytext=(x, y0), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=G, lw=2))
    ax.text(x - 0.015, (y0 + y1) / 2, "better", transform=ax.transAxes, ha="right", va="center", color=G, fontsize=8,
            fontweight="bold", rotation=90)


# ---------- fig60: decode speed relative to F16 KV ----------
rows = []
for d in ("fork_speed_a", "fork_speed_b"):
    f = A / d / "summary.json"
    if f.exists():
        rows += json.loads(f.read_text(encoding="utf-8"))
cfgs = [("fork:q8_0:def:on", "q8_0 stock", GREY), ("pr:q8_0:def:on", "q8_0 PR #30202", GREY2),
        ("fork:q8_0:1:on", "q8_0 + our GQA kernel", BLUE), ("fork:q4_0:def:on", "q4_0 stock", GREY),
        ("pr:q4_0:def:on", "q4_0 PR #30202", GREY2), ("fork:q4_0:1:on", "q4_0 + our GQA kernel", BLUE),
        ("fork:f16d4:def:on", "ours f16d4 (12 bit, lossy)", ORANGE), ("fork:q4o:def:on", "ours q4o (5 bit, lossy)", ORANGE),
        ("fork:f16e3:def:on", "ours f16e3 (lossless)", GREEN)]
depths = sorted({r["depth"] for r in rows if r["test"] == "decode"})
fig, ax = plt.subplots(figsize=(11, 4.2))
w = 0.8 / max(1, len(cfgs))
for i, (c, lab, col) in enumerate(cfgs):
    ys = []
    for d in depths:
        m = [r["ratio_to_f16"] for r in rows if r["config"] == c and r["depth"] == d and r["test"] == "decode"]
        ys.append(m[0] if m else float("nan"))
    xs = [k + (i - len(cfgs) / 2 + 0.5) * w for k in range(len(depths))]
    ax.bar(xs, ys, w, color=col, label=lab, edgecolor="white", hatch="//" if "PR" in lab else None)
    for x, y in zip(xs, ys):
        if y == y:
            ax.text(x, y + 0.01, f"{y:.2f}", ha="center", fontsize=6, rotation=90)
ax.axhline(1.0, color="black", lw=1)
ax.text(len(depths) - 0.55, 1.01, "F16 KV, stock kernel", fontsize=8)
ax.set_xticks(range(len(depths)), [f"{d // 1024}K" for d in depths])
ax.set_ylabel("decode speed relative to F16 KV")
ax.set_xlabel("filled context")
ax.set_ylim(0, 1.35)
ax.legend(fontsize=7, ncol=3, loc="upper left")
better(ax)
ax.set_title("Real llama.cpp decode, Qwen3-4B, RTX 3090 (R46/R47)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig60_fork_decode_speed.png")
plt.close(fig)

# ---------- fig61: KV quality (KL vs F16, fusions off) vs bits per value ----------
pts = [  # bits, KL, label, colour, real?
    (16, 0.0, "F16 (base)", GREY, True), (8.5, 0.00349, "q8_0", GREY, True), (4.5, 0.0677, "q4_0 (rot on)", GREY, True),
    (15.25, 0.0, "f16e3 lossless", GREEN, True), (12, 0.00279, "f16d4", ORANGE, True), (5.0, 0.0328, "q4o", ORANGE, True),
    (14, 0.00286, "drop 2 bits (emu)", ORANGE, False), (6.8, 0.00456, "q8 6-bit+1% (emu)", ORANGE, False),
    (7.75, 0.0337, "q8 7-bit (emu)", ORANGE, False), (4.3, 0.2363, "q3+1% (emu)", ORANGE, False)]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
floor = 0.0024
for b, k, lab, col, real in pts:
    kk = max(k, 1e-4)
    ax.scatter([b], [kk], s=60 if real else 35, color=col, marker="o" if real else "^", edgecolor="black", lw=0.5, zorder=3)
    ax.annotate(lab, (b, kk), textcoords="offset points", xytext=(5, 3), fontsize=7)
ax.axhspan(1e-4, floor, color="#eeeeee", zorder=0)
ax.text(16.3, floor * 0.6, "numerics noise floor (R44)", fontsize=7, ha="right", color="#555555")
ax.set_yscale("log")
ax.set_ylim(8e-5, 0.5)
ax.set_xlim(3.5, 17)
ax.set_xlabel("bits per value")
ax.set_ylabel("KL vs F16 KV (4K, fusions off; 0 plotted at 1e-4)")
ax.set_title("KV quality vs size, Qwen3-4B (R45). Circles: real types; triangles: emulated", fontsize=9)
ax.annotate("", xy=(0.08, 0.08), xytext=(0.22, 0.22), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=G, lw=2))
ax.text(0.16, 0.12, "better", transform=ax.transAxes, color=G, fontsize=8, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig61_fork_kv_quality.png")
plt.close(fig)

# ---------- fig62: process VRAM vs filled tokens, eager vs lazy (pooled scratch) ----------
eager = json.loads((A / "fork_s2_20261022/fill_memory/rows.json").read_text(encoding="utf-8"))
lazy = json.loads((A / "fork_s3_20261022/fill_memory_lazy_pooled/rows.json").read_text(encoding="utf-8"))
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for kv, col, ls in (("f16", GREY, "-"), ("q8_0", GREY2, "-"), ("f16d4", ORANGE, "-"), ("q4_0", "#666666", ":")):
    e = sorted((r["tokens_total"], r["local_mib_last"]) for r in eager if r["kv"] == kv and r["mode"] == "eager")
    z = sorted((r["tokens_total"], r["local_mib_last"]) for r in lazy if r["kv"] == kv)
    ax.plot([t for t, _ in e], [m / 1024 for _, m in e], ls="--", color=col, lw=1)
    ax.plot([t for t, _ in z], [m / 1024 for _, m in z], ls=ls, marker="o", color=col, lw=2, label=f"{kv} lazy")
ax.text(1500, 20.4, "eager allocation (dashed): reserved for -c 122880", fontsize=7)
ax.set_xscale("log")
ax.set_xlabel("filled tokens (of a 122,880-token context)")
ax.set_ylabel("process VRAM, GiB (DXGI)")
ax.set_title("KV memory follows the filled context with the lazy buffer (R43/R44)", fontsize=10)
ax.legend(fontsize=8)
better(ax, up=False, x=0.05)
fig.tight_layout()
fig.savefig(OUT / "fig62_fork_lazy_memory.png")
plt.close(fig)
print("ok")
