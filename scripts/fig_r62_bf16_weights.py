"""Real-inference results R62 (RESULTS.md): BF16 base vs our weight formats, Qwen3.5-2B, RTX 3090.

Three panels: tensor size, process VRAM at 4K fill, decode speed at 4K / 32K / 120K fill.
Colours: gray = regular BF16, green = ours lossless, orange = ours lossy. Labels show the delta vs BF16.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRAY, GREEN, ORANGE = "#8c8c8c", "#2ca02c", "#ff7f0e"
rows = [
    # label, colour, size MiB, VRAM MiB @4K, decode tok/s @4K, @32K, @120K
    ("BF16\nregular",         GRAY,   3706, 4442, 158.9, 144.9, 104.9),
    ("Huffman\nlossless",     GREEN,  2579, 3366,  95.7,  91.5,  74.3),
    ("W4\nlossless",          GREEN,  2813, 3602, 165.7, 150.9, 103.1),
    ("W3\nlossy",             ORANGE, 2578, 3334, 177.9, 160.0, 119.2),
    ("FFN in Q8_0\nlossy",    ORANGE, 2855, 3622, 183.4, 166.1, 124.4),
]
base = rows[0]


def pct(v, b):
    d = (v / b - 1) * 100
    return ("%+.0f%%" % d) if abs(d) >= 0.5 else "0%"


fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), gridspec_kw={"width_ratios": [1, 1, 1.9]})
labels = [r[0] for r in rows]
x = list(range(len(rows)))

# panel 1: size
ax = axes[0]
vals = [r[2] for r in rows]
ax.bar(x, vals, color=[r[1] for r in rows], width=0.7)
for i, v in enumerate(vals):
    txt = "%d MiB" % v + ("" if i == 0 else "\n%s" % pct(v, vals[0]))
    ax.text(i, v + 40, txt, ha="center", va="bottom", fontsize=9, fontweight="bold" if i else None)
ax.set_title("Model tensors, MiB\n(lower is better)", fontsize=11)
ax.set_ylim(0, 5300)

# panel 2: VRAM
ax = axes[1]
vals = [r[3] for r in rows]
ax.bar(x, vals, color=[r[1] for r in rows], width=0.7)
for i, v in enumerate(vals):
    txt = "%d MiB" % v + ("" if i == 0 else "\n%+d MiB" % (v - vals[0]))
    ax.text(i, v + 50, txt, ha="center", va="bottom", fontsize=9, fontweight="bold" if i else None)
ax.set_title("Process VRAM at 4K context, MiB\n(lower is better)", fontsize=11)
ax.set_ylim(0, 6400)

# panel 3: decode at three depths
ax = axes[2]
w = 0.26
for j, (name, alpha) in enumerate((("4K", 1.0), ("32K", 0.7), ("120K", 0.45))):
    vals = [r[4 + j] for r in rows]
    xs = [i + (j - 1) * w for i in x]
    ax.bar(xs, vals, width=w, color=[r[1] for r in rows], alpha=alpha, edgecolor="white")
    for xi, v, b in zip(xs, vals, [base[4 + j]] * len(rows)):
        lab = "%.0f" % v + ("" if v == b else "\n%s" % pct(v, b))
        ax.text(xi, v + 2, lab, ha="center", va="bottom", fontsize=7.5)
ax.set_title("Decode speed, tokens/s, by filled context 4K | 32K | 120K\n(higher is better; 120K cells are noisy, see min..max in R62)", fontsize=11)
ax.set_ylim(0, 250)

for a in axes:
    a.set_xticks(x)
    a.set_xticklabels(labels, fontsize=8.5)
    a.spines[["top", "right"]].set_visible(False)

# "better" arrows (top-right corner, above the bars)
for a, up in ((axes[0], False), (axes[1], False), (axes[2], True)):
    y0, y1 = (0.97, 0.82) if not up else (0.82, 0.97)
    a.annotate("", xy=(0.97, y1), xytext=(0.97, y0), xycoords="axes fraction",
               arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.8))
    a.text(0.93, 0.895, "better", transform=a.transAxes, ha="right", va="center", fontsize=10, color="#555")

from matplotlib.patches import Patch
axes[2].legend(handles=[Patch(color="#555", alpha=1.0, label="4K filled"), Patch(color="#555", alpha=0.7, label="32K filled"), Patch(color="#555", alpha=0.45, label="120K filled")], loc="upper left", frameon=False, fontsize=9, ncol=3)
fig.legend(handles=[Patch(color=GRAY, label="regular BF16 (base)"),
                    Patch(color=GREEN, label="ours, lossless"),
                    Patch(color=ORANGE, label="ours, lossy (mixed precision / 3-bit exponent group)")],
           loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.03))
fig.suptitle("Compressed weights in real llama.cpp inference vs regular BF16 (Qwen3.5-2B, RTX 3090)", fontsize=13, fontweight="bold")
fig.text(0.5, 0.008, "Medians of 12 samples per cell (llama-bench, KV F16). Dirty window: desktop background 4-28% GPU. Source: RESULTS.md R62.",
         ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=(0, 0.09, 1, 0.94))
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "fig60_bf16_weights_real.png")
fig.savefig(out, dpi=150)
print(out)
