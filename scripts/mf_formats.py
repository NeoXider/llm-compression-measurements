# mf_formats.py -- figures 11-13: F16 bit layout, Q8_0/Q4_0 blocks, EXP3 format.
from mf_common import *


def fig11_fp16_bits():
    fig = plt.figure(figsize=(14, 9.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.5, 0.9], hspace=0.5, wspace=0.28)
    ax = fig.add_subplot(gs[0, :]); ax.set_xlim(-0.2, 16.2); ax.set_ylim(0, 2.9); ax.axis("off")
    for a, b, c, t in [(0, 1, BASE, "S"), (1, 6, LL, "E"), (6, 16, LY, "M")]:
        for i in range(a, b):
            ax.add_patch(Rectangle((i + 0.03, 0.55), 0.94, 0.8, fc=c, ec="white", lw=1.5))
            ax.text(i + .5, .95, t, ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    labs = [(0, 1, "", "#666666", 0.45), (1, 6, "", LL, 4.1),
            (6, 16, "", LY, 11)]
    for a, b, t, c, xc in labs:
        ax.annotate("", xy=(b - 0.05, 1.5), xytext=(a + 0.05, 1.5), arrowprops=dict(arrowstyle="-", color=BASE if a == 0 else c, lw=3))
        ax.text(xc, 1.62, t, ha="center", va="bottom", fontsize=12 if a else 11, color=c, fontweight="bold")
    ax.text(0.5, 0.33, "bit 15", ha="center", fontsize=9, color=MUTED); ax.text(15.5, 0.33, "bit 0", ha="center", fontsize=9, color=MUTED)
    ax.text(8, 0.02, "one F16 number = 16 bits  (every K and V value in the KV cache looks like this)", ha="center", fontsize=11.5, color=INK)
    ax = fig.add_subplot(gs[1, 0])
    K = [1.00, 2.89, 9.97]; V = [1.00, 3.31, 9.97]; nominal = [1, 5, 10]
    cols = [BASE, LL, LY]; y = np.arange(3)[::-1]
    ax.barh(y, nominal, 0.8, color="#eeeeee", edgecolor="#bbbbbb")
    ax.barh(y + 0.19, K, 0.34, color=cols); ax.barh(y - 0.19, V, 0.34, color=cols, alpha=0.5)
    for yi, k, v in zip(y, K, V):
        ax.text(max(k, v) + 0.2, yi + 0.19, f"K {k:.2f}", va="center", fontsize=11)
        ax.text(max(k, v) + 0.2, yi - 0.19, f"V {v:.2f}", va="center", fontsize=11)
    ax.set_yticks(y); ax.set_yticklabels(["sign (1 bit)", "exponent (5 bits)", "mantissa (10 bits)"]); ax.set_xlim(0, 12.8)
    ax.set_xlabel("bits per value  (grey = stored; colour = really needed = entropy)")
    ax.set_title("How many bits does each part REALLY need? (R9)", fontsize=12.5)
    better(ax, "left", at=(0.97, 0.86), L=0.2, label="better = fewer bits")
    ax = fig.add_subplot(gs[1, 1])
    for yi, e in enumerate([3.31, 2.89]):      # V at bottom, K on top
        used = [1.0, e, 9.97]; free = 16 - sum(used); left = 0
        for u, c in zip(used, cols):
            ax.barh(yi, u, 0.55, left=left, color=c, edgecolor="white"); left += u
        ax.barh(yi, free, 0.55, left=left, color="white", edgecolor=LL, hatch="////", lw=1.5)
        ax.text(left + free / 2, yi + 0.42, f"free {free:.2f}", ha="center", va="bottom", fontsize=9.5, color=LL, fontweight="bold")
        ax.text(16.3, yi, f"{sum(used):.2f}\nof 16", va="center", fontsize=11, fontweight="bold")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["V", "K"]); ax.set_xlim(0, 18.2); ax.set_ylim(-0.5, 1.9)
    ax.set_xlabel("bits of one 16-bit word   (green hatched = removable with NO loss)")
    ax.set_title("Order-0 entropy estimate: only the exponent has slack", fontsize=12.5)
    better(ax, "left", at=(0.97, 0.93), L=0.2, label="better = fewer bits")
    ax = fig.add_subplot(gs[2, :]); ax.axis("off")
    ax.text(0.0, 1.0, "", fontsize=13, fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(0.0, 0.72, "SIGN  (1.00 of 1 bit): nothing to gain.", fontsize=11.5, color="#555", transform=ax.transAxes, va="top")
    ax.text(0.0, 0.52, "EXPONENT  (2.9-3.3 of 5 bits): lossless gain. Order-0 entropy estimate 13.85 / 13.63 bits of 16 (K / V);\n   bit-exact CPU prototypes reach -13.6% (zlib) and -15.3% (LZMA).",
            fontsize=11.5, color=LL, transform=ax.transAxes, va="top", fontweight="bold")
    ax.text(0.0, 0.14, "MANTISSA  (9.97 of 10 bits): incompressible without loss. Only LOSSY:\n   drop 2 bits = 0.875 size, error 0.13%;  drop 4 bits = 0.75-0.80 size, error 0.62%.",
            fontsize=11.5, color=LY, transform=ax.transAxes, va="top", fontweight="bold")
    finish(fig, "fig11_fp16_bit_layout", "Inside one F16 number: where the redundancy is",
           "",
           note="Source: R9 (field entropies, all layers pooled; one capture, one text, T = 2304 tokens), R10, R33. K = keys, V = values.", top=0.87, bottom=0.04)


def fig12_q_blocks():
    fig = plt.figure(figsize=(14, 8.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.38, wspace=0.28)
    for gi, (nm_, desc, tot) in enumerate([("Q8_0", "32 x 8 bit = 32 B", "34 bytes for 32 values = 8.5 bits per value"),
                                           ("Q4_0", "32 x 4 bit = 16 B", "18 bytes for 32 values = 4.5 bits per value")]):
        ax = fig.add_subplot(gs[0, gi]); ax.set_xlim(0, 10); ax.set_ylim(0, 3.1); ax.axis("off")
        ax.text(0, 3.0, f"{nm_} block = 32 values", fontsize=14, fontweight="bold", va="top")
        box(ax, 0.0, 1.5, 1.6, 0.9, "scale\nf16 (2 B)", "#6b6b6b", fs=10, bold=True)
        box(ax, 1.75, 1.5, 8.2, 0.9, f"32 integer codes ({desc})", BASE, fs=10, bold=True)
        ax.text(0, 1.3, tot, fontsize=11.5, va="top", color=INK)
        ax.text(0, 0.88, "value = scale x code   (rounding = the loss of the base)", fontsize=11, va="top", color=MUTED)
        ax.text(0, 0.45, "Ours (green) re-codes these bytes without changing any value.", fontsize=10.5, va="top", color=LL)
    ax = fig.add_subplot(gs[1, :])
    rows = [("Q8_0 codes  (order-0)", 8, 7.263, 7.591), ("Q8_0 codes, conditioned on channel", 8, 6.670, 7.422),
            ("Q8_0 scale (f16, 16 bits)", 16, 11.685, 11.050), ("Q4_0 codes  (order-0)", 4, 3.376, 3.680)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, nom, k, v) in zip(y, rows):
        ax.barh(yi, nom, 0.84, color="#eeeeee", edgecolor="#bbbbbb")
        ax.barh(yi + .2, k, 0.36, color=LL); ax.barh(yi - .2, v, 0.36, color=LL, alpha=0.5)
        ax.text(max(k, v) + 0.2, yi + .2, f"K {k:.2f}", va="center", fontsize=11)
        ax.text(max(k, v) + 0.2, yi - .2, f"V {v:.2f}", va="center", fontsize=11)
        ax.text(16.6, yi, f"stored: {nom} bits", va="center", fontsize=10.5, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=11.5); ax.set_xlim(0, 21.5); ax.set_xticks(range(0, 17, 2))
    ax.set_xlabel("bits per symbol  (grey = stored, green = needed; dark = K, light = V)")
    ax.set_title("How much slack is in the quantized codes? (offline requantization of the same real KV, R9)", fontsize=12.5)
    ax.text(0.99, 0.0, "Q4_0 scale entropy: n/m (estimate keeps it raw)", transform=ax.transAxes, ha="right", fontsize=10, color="#666")
    better(ax, "left", at=(0.62, 0.17), L=0.2, label="better = fewer bits")
    finish(fig, "fig12_q8_q4_blocks_entropy", "Quantized KV blocks: a scale plus small integer codes",
           "",
           note="Source: R9 / artifacts/kv_entropy_20261021/kv_entropy.json (mean over 36 layers; Qwen3-4B, T = 2304). Whole-stream ratios: Q8_0 K 0.897 (0.828 by channel), Q4_0 K 0.861.",
           top=0.87, bottom=0.05)


def fig13_exp3():
    fig = plt.figure(figsize=(15, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.3], hspace=0.35, wspace=0.28)
    ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, 16); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.text(0, 4.15, "EXP3 = our fixed-width lossless format for F16: every value has the SAME width, so the GPU can jump to any value",
            fontsize=12.5, fontweight="bold", va="top")
    box(ax, 0, 2.3, 2.3, 0.9, "group base\nexponent (5 bit)", "#6b6b6b", fs=10.5, bold=True)
    ax.text(1.15, 2.2, "one per 32 values", ha="center", va="top", fontsize=9.5, color=MUTED)
    ax.text(2.6, 2.75, "+", fontsize=22, ha="center", va="center")
    box(ax, 3.0, 2.3, 0.9, 0.9, "sign\n1", BASE, fs=10, bold=True)
    box(ax, 3.95, 2.3, 1.7, 0.9, "offset\n3 bit", LL, fs=10.5, bold=True)
    box(ax, 5.7, 2.3, 4.4, 0.9, "mantissa 10 bit (verbatim)", LY, fs=10.5, bold=True)
    ax.text(6.55, 2.2, "one value = 14 bits  (instead of 16)", ha="center", va="top", fontsize=10.5, color=INK)
    ax.text(10.4, 2.75, "x 32 values", fontsize=11, ha="left", va="center", color=MUTED)
    ax.text(0, 1.65, "real exponent = base + offset      (offset 0...6 = normal value)", fontsize=11.5, color=INK, va="top")
    box(ax, 0, 0.5, 1.5, 0.6, "offset = 7", LL, fs=11, bold=True)
    arrow(ax, 1.55, 0.8, 2.1, 0.8, color=LL)
    box(ax, 2.15, 0.5, 7.4, 0.6, "ESCAPE: the exact exponent goes to a per-row overflow list", "white", ec=LL, fs=10.5, tc=LL, bold=True)
    ax.text(2.15, 0.35, "(found on the GPU by a warp prefix count)", fontsize=10, va="top", color=MUTED)
    ax.text(10.4, 1.25, "size = 14 + 5/32 bits per value + escapes\n= 0.93-0.96 of F16 on real data (GPU layout)", fontsize=11, va="top", color=LL, fontweight="bold")
    esc = {T: pocb("exp3", T, "escapes") / (pocb("exp3", T, "bytes") / pocb("exp3", T, "ratio_vs_f16") / 2) for T in (4096, 32768, 122880)}
    ax = fig.add_subplot(gs[1, 0])
    cats = [("K, 32 ch\nof 1\ntoken", 6.1, 1), ("K, 1 ch\nover 32\ntokens", 1.8, 1), ("V, 32 ch\nof 1\ntoken", 2.5, 1), ("V, 1 ch\nover 32\ntokens", 2.3, 1),
            ("GPU row\n4K", 100 * esc[4096], 0), ("GPU row\n32K", 100 * esc[32768], 0), ("GPU row\n120K", 100 * esc[122880], 0)]
    x = np.arange(len(cats))
    for xi, (lab, v, cpu) in zip(x, cats):
        ax.bar(xi, v, 0.7, color=LL, hatch="////" if cpu else None, edgecolor="white")
        ax.text(xi, v + 0.2, f"{v:.1f}%", ha="center", fontsize=10.5)
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cats], fontsize=9.2); ax.set_ylabel("escape share of values (%)"); ax.set_ylim(0, 13.5)
    ax.set_title("How often a value needs the escape (offset 7)", fontsize=12)
    better(ax, "down", L=0.2, at=(0.965, 0.9))
    ax = fig.add_subplot(gs[1, 1])
    sz = [("K, 32 ch\nof 1\ntoken", 0.920, 1), ("K, 1 ch\nover 32\ntokens", 0.906, 1), ("V, 32 ch\nof 1\ntoken", 0.908, 1), ("V, 1 ch\nover 32\ntokens", 0.907, 1),
          ("GPU row\n4K", pocb("exp3", 4096, "ratio_vs_f16"), 0), ("GPU row\n32K", pocb("exp3", 32768, "ratio_vs_f16"), 0),
          ("GPU row\n120K", pocb("exp3", 122880, "ratio_vs_f16"), 0)]
    for xi, (lab, v, cpu) in zip(x, sz):
        ax.bar(xi, v, 0.7, color=LL, hatch="////" if cpu else None, edgecolor="white")
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=10.5)
    ax.axhline(1.0, color=GR["F16"], ls="--", lw=1.8)
    ax.text(1.01, 1.0, "F16 = 1.00\n(base)", ha="left", va="center", transform=ax.get_yaxis_transform(), fontsize=10, color=GR["F16"])
    ax.axhline(0.9, color="#999999", ls=":", lw=1.5)
    ax.text(1.01, 0.9, "10% goal", ha="left", va="center", transform=ax.get_yaxis_transform(), fontsize=10, color="#777")
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in sz], fontsize=9.2); ax.set_ylim(0.8, 1.09); ax.set_ylabel("size relative to F16")
    ax.set_title("Resulting size (k = 3 offset bits)", fontsize=12)
    better(ax, "down", L=0.12, at=(0.93, 0.97))
    legend_fig(fig, [("Ours, LOSSLESS (GPU data)", LL), ("hatched = CPU estimate (R11)", "#bbbbbb", "////")], y=0.04)
    finish(fig, "fig13_exp3_format_escapes", "EXP3: a fixed-width lossless format and its escapes",
           "",
           note="Source: R11 (CPU, k=3), R22/R26 (GPU, real KV; escape share derived = escapes / elements from poc_b.jsonl), PAPER section 3.",
           top=0.88, bottom=0.13)
