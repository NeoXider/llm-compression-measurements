# mf_info.py -- figure 27 (dashboard), 28 (pipeline infographic), 29 (verdict map).
from mf_common import *
from mf_kv import VRAM, DEC_R, VR_CTX

HAT = "////"


def fig27_dashboard():
    fig, axs = plt.subplots(2, 3, figsize=(21, 11.5))
    # A VRAM
    ax = axs[0, 0]
    cx = [4096, 32768, 65536, 122880]
    for k, mk in (("F16", "o"), ("Q8_0", "s"), ("Q4_0", "^")):
        v = [VRAM[k][i] for i in (0, 2, 3, 4)]
        ax.plot(cx, v, marker=mk, color=GR[k], mec=INK, ms=8, lw=2.8)
        ax.annotate(f"{k} KV  {v[-1]}", (cx[-1], v[-1]), textcoords="offset points", xytext=(-6, 8 if k != "Q4_0" else -14), ha="right", fontsize=10, fontweight="bold")
    ax.axhline(24576, color="#999999", ls=":", lw=1.6); ax.text(4300, 23300, "24 GiB card", fontsize=9.5, color="#666")
    ctx_axis(ax); ax.set_ylim(0, 26000); ax.set_ylabel("process VRAM, MiB")
    ax.set_title("A. VRAM: quantized KV saves up to -60% (R30)", fontsize=12)
    better(ax, "down", at=(0.25, 0.80), L=0.14)
    # B speed
    ax = axs[0, 1]
    for k, mk in (("F16", "o"), ("Q8_0", "s"), ("Q4_0", "^")):
        ax.plot(cx, DEC_R[k], marker=mk, color=GR[k], mec=INK, ms=8, lw=2.8)
    ax.plot(cx, [0.73, 0.68, 0.65, 0.65], "D--", color=LL, mec=INK, ms=8, lw=2.4)
    ax.text(cx[-1], 0.665, "ours EXP3 lossless\n(read microbench)", color=LL, fontsize=9.5, ha="right", va="bottom", fontweight="bold")
    ax.text(cx[-1], 0.575, "Q8_0 / Q4_0 decode", color="#666", fontsize=9.5, ha="right", va="top", fontweight="bold")
    ax.text(cx[0], 1.02, "F16 = 1.00", color=GR["F16"], fontsize=9.5, fontweight="bold")
    ctx_axis(ax); ax.set_ylim(0.5, 1.12); ax.set_ylabel("speed relative to F16 KV")
    ax.set_title("B. Speed: fewer bytes did NOT mean faster (R25, R26)", fontsize=12)
    better(ax, "up", at=(0.985, 0.84), L=0.14)
    # C KV size on bases
    ax = axs[0, 2]
    groups = ["F16 base", "Q8_0 base", "Q4_0 base"]
    ll = [0.847, 0.9115, 0.8295]; ly = [0.797, 0.802, 0.849]
    x = np.arange(3); w = 0.26
    ax.bar(x - w, [1, 1, 1], w * 0.95, color=BASE, edgecolor="white"); ax.bar(x, ll, w * 0.95, color=LL, edgecolor="white", hatch=HAT); ax.bar(x + w, ly, w * 0.95, color=LY, edgecolor="white", hatch=HAT)
    for xi, a, b in zip(x, ll, ly):
        ax.text(xi, a + 0.01, f"{a:.2f}", ha="center", fontsize=10); ax.text(xi + w, b + 0.01, f"{b:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(0, 1.45); ax.set_ylabel("size relative to own base")
    ax.set_title("C. KV size on top of each base (R33)", fontsize=12)
    ax.set_xlabel("green: planes+LZMA (CPU); orange: drop-4 / 6-bit+1% / 3-bit+1% (error proxy)", fontsize=9)
    better(ax, "down", at=(0.985, 0.98), L=0.14)
    # D KL
    ax = axs[1, 0]
    x = np.arange(2); w = 0.3
    ax.bar(x - w / 2, [0.00268, 0.0694], w * 0.95, color=[GR["Q8_0"], GR["Q4_0"]], edgecolor=INK, lw=0.7)
    ax.bar(x + w / 2, [0.00128, 0.03487], w * 0.95, color=[GR["Q8_0"], GR["Q4_0"]], edgecolor=INK, lw=0.7, hatch="\\\\\\")
    for xi, a, b in zip(x, [0.00268, 0.0694], [0.00128, 0.03487]):
        ax.text(xi - w / 2, a * 1.3, f"{a:.4f}" if a < .01 else f"{a:.3f}", ha="center", fontsize=9.5); ax.text(xi + w / 2, b * 1.3, f"{b:.4f}" if b < .01 else f"{b:.3f}", ha="center", fontsize=9.5)
    ax.axhspan(4.5e-4, 8e-4, color=LL, alpha=0.9); ax.text(1.4, 6.2e-4, "ours lossless = 0", ha="right", va="center", fontsize=10, color="white", fontweight="bold")
    ax.set_yscale("log"); ax.set_ylim(4.5e-4, 0.4); ax.set_xticks(x); ax.set_xticklabels(["Q8_0 KV", "Q4_0 KV"]); ax.set_xlim(-0.6, 1.5)
    ax.set_ylabel("mean KL vs F16 KV (log)"); ax.set_title("D. Quality: KL at 4K (solid) and 32K (hatched) (R19, R29)", fontsize=12)
    ax.text(0.02, 0.96, "our lossy KV variants: KL in R36 (exact emulation)", transform=ax.transAxes, va="top", fontsize=9.5, color=MUTED)
    better(ax, "down", at=(0.965, 0.55), L=0.16)
    # E weights recipes
    ax = axs[1, 1]
    rec = [("BF16 FFN->Q8", 0.770, 1.195, 1), ("Q8 FFN->Q6K", 0.889, 1.055, 1), ("Q4KM emb->Q4K", 0.899, 1.110, 1), ("Q4KM FFN up/gate->Q3K", 0.936, 0.982, 1)]
    for n, s, v, _ in rec:
        ax.scatter([s], [v], s=190, color=LY, edgecolor=INK, zorder=5)
        ax.annotate(n, (s, v), textcoords="offset points", xytext=(8, 5 if v >= 1 else -16), fontsize=9.5, color=LY, fontweight="bold")
    for s, n in ((0.666, "BF16 Huffman"), (0.707, "BF16 planes"), (0.944, "Q8 planes"), (0.978, "Q4KM planes")):
        ax.scatter([s], [0.84], s=120, marker="D", color=LL, edgecolor="white", hatch=HAT, zorder=4)
    ax.text(0.5, 0.815, "green: lossless re-codings, speed n/m (no GPU decoder)", color=LL, fontsize=9.5, ha="left", va="top")
    ax.axhline(1, color=BASE, ls="--"); ax.axvline(1, color=BASE, ls="--")
    ax.set_xlim(0.6, 1.05); ax.set_ylim(0.78, 1.28); ax.set_xlabel("size vs own base"); ax.set_ylabel("decode speed vs own base")
    ax.set_title("E. Weights: our recipes run in llama.cpp (R34)", fontsize=12)
    better(ax, "ul", at=(0.30, 0.45), L=0.12)
    # F weights lossless
    ax = axs[1, 2]
    wl = [("BF16\nHuffman", 0.666, 0), ("BF16\nplanes+zlib", 0.707, 1), ("Q8_0\nplanes+zlib", 0.944, 1), ("Q4_K_M\nplanes+zlib", 0.978, 1), ("Q4_K_M\nCQW2 (GPU)", 1.006, 0)]
    for i, (n, v, c) in enumerate(wl):
        ax.bar(i, v, 0.65, color=LL, edgecolor="white", hatch=HAT if c or i == 0 else None); ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=10)
    ax.axhline(1.0, color=GR["F16"], ls="--", lw=1.8)
    ax.set_xticks(range(5)); ax.set_xticklabels([w[0] for w in wl], fontsize=9); ax.set_ylim(0, 1.4); ax.set_ylabel("size relative to own base")
    ax.set_title("F. Lossless weights: BF16 -33%, quantized only -2..-6% (R3, R34)", fontsize=12)
    better(ax, "down", at=(0.985, 0.98), L=0.13)
    legend_fig(fig, [("Base (no compression)", BASE), ("Ours LOSSLESS", LL), ("Ours LOSSY", LY), ("hatched = CPU prototype only", "#bbbbbb", HAT)], y=0.03)
    finish(fig, "fig27_dashboard", "The whole study on one page",
           "",
           note="Qwen3-4B Q4_K_M on RTX 3090 (A-D), Qwen3.5-2B (E-F). Numbers: R3, R19, R25, R26, R29, R30, R33, R34 (see FIGURES_INDEX.md). n/m = not measured.", top=0.89, bottom=0.07)


def fig28_pipeline_infographic():
    fig, ax = plt.subplots(figsize=(21, 9.2)); ax.set_xlim(0, 21); ax.set_ylim(0, 9.2); ax.axis("off")
    stages = [
        ("1  DATA", "", "#6b6b6b",
         ["real Qwen3-4B KV capture:", "2304 tokens, 36 layers x 8 heads x 128", "GGUF weights: Qwen3-4B Q4_K_M,", "Qwen3.5-2B BF16 ... Q2_K", "fixed 95.7 kB text corpus", "frozen 39-case task suite"],
         ["We take REAL numbers,", "not made-up ones."]),
        ("2  ANALYZE + CODE", "", LL,
         ["entropy of every field (R4, R9, R16)", "LOSSLESS: byte planes + zlib / LZMA,", "Huffman, fixed-rate EXP3, CQW2 palette", "LOSSY: drop mantissa bits, outliers,", "per-channel scales, mixed-precision recipes"],
         ["Every lossless codec is checked", "bit for bit (all pages exact)."]),
        ("3  MEASURE ON GPU", "", "#4d4d4d",
         ["RTX 3090: per-process VRAM (DXGI)", "ABBA-interleaved decode speed to 120K", "PoC-A lazy pool, PoC-B read kernel", "KL, 39 tasks, needle-in-a-haystack", "clean-window rule: no foreign GPU jobs"],
         ["Same GPU, same order, repeated:", "ratios inside one run."]),
        ("4  VERDICT", "", LY,
         ["VRAM: quantized KV -28% / -44% (32K)", "speed: decode SLOWER (0.63x at 120K)", "lossless: -15% CPU, GPU format -4..-7%", "weights recipes: +5..+20% speed", "servers: projections only (*)"],
         ["Fewer bytes did not mean", "more speed; capacity gains are real."]),
    ]
    w = 4.55; gap = 0.65; x0 = 0.3
    for i, (title, ru, col, lines, foot) in enumerate(stages):
        x = x0 + i * (w + gap)
        box(ax, x, 6.5, w, 1.4, "", col)
        ax.text(x + w / 2, 7.45, title, ha="center", va="center", fontsize=17, color="white", fontweight="bold")
        ax.text(x + w / 2, 6.9, ru, ha="center", va="center", fontsize=12.5, color="white", style="italic")
        box(ax, x, 2.35, w, 3.95, "", "#f6f6f6", ec=col, lw=2.2)
        for j, l in enumerate(lines):
            ax.text(x + 0.2, 5.9 - j * 0.6, l, fontsize=11.2, color=INK, va="center")
        ax.text(x + w / 2, 1.85, "\n".join(foot), ha="center", va="center", fontsize=11, color=col if col != "#4d4d4d" else INK, fontweight="bold")
        if i < 3:
            arrow(ax, x + w + 0.05, 7.2, x + w + gap - 0.05, 7.2, color=INK, lw=3.2, ms=26)
    ax.text(10.5, 8.95, "From data to verdict: how this study was done", ha="center", fontsize=20, fontweight="bold", color=INK)
    ax.text(10.5, 8.5, "", ha="center", fontsize=13, color=MUTED, style="italic")
    # bottom strip: legend of statuses
    box(ax, 0.25, 0.25, 20.5, 0.95, "", "#fafafa", ec="#cccccc")
    ax.text(0.5, 0.72, "Rules:", fontsize=11, fontweight="bold", va="center")
    ax.text(1.6, 0.72, "colours: gray = base, green = ours lossless, orange = ours lossy;  hatch = CPU prototype only;  n/m = not measured;  * = analytic projection, not a measurement.", fontsize=11, va="center")
    finish(fig, "fig28_pipeline_infographic", "", "", note=None, top=0.99, bottom=0.0, tl=False)


def fig29_verdict_map():
    ST = {"good": (LL, "worked"), "part": (AMBER, "partly"), "no": (LY, "did not work"), "nm": (BASE, "not measured")}
    rows = [
        ("good", "llama.cpp quantized KV saves VRAM: -28% / -44% at 32K, -38% / -60% at 120K", "R30"),
        ("good", "Q8_0 KV keeps quality: KL 0.0013-0.0042, tasks 25 = 25 of F16", "R12, R19, R20, R29"),
        ("part", "Q4_0 KV: KL 0.035-0.069, tasks 21 vs 25 (5 lost, not significant)", "R19, R20, R29"),
        ("no", "Quantized KV makes decode faster (0.88x at 4K ... 0.63x at 120K; PR #30202: Q8_0 0.86x at 120K)", "R25, R39"),
        ("good", "Lazy KV pool: process memory follows filled tokens (PoC, not in llama.cpp yet)", "R23"),
        ("part", "Lossless F16 KV: -13.6% (zlib) / -15.3% (LZMA), bit-exact, CPU variable-rate only", "R10"),
        ("no", "GPU-addressable lossless EXP3: only -4..-7% and reads 27-35% slower", "R22, R26"),
        ("part", "Lossless on top of Q8_0 / Q4_0 KV: K -11% / -21% (CPU variable-rate)", "R33"),
        ("no", "Fixed-rate lossless on Q8_0 / Q4_0 KV: LARGER than the base (+6..+25%)", "R33"),
        ("good", "Lossy F16 drop-4 bits: size 0.80, read 0.85-0.98x, KL 0.0027 = Q8_0 quality, tasks 25/39", "R33, R36"),
        ("part", "Per-channel K scales (KIVI-style): K error 0.32-0.43x of base (proxy only)", "R13, R33"),
        ("part", "Needle-in-a-haystack to 94K: all KV types pass (test does not separate them)", "R27, R32"),
        ("no", "Weights: lossless on quantized k-quants (headroom 2-7%; CQW2 palette +0.6% larger)", "R3, R4, R16"),
        ("part", "Weights: lossless BF16 -33% (Huffman exponent) - size only, no GPU decoder", "R34"),
        ("good", "Weights: BF16 + FFN->Q8_0 recipe = +16-18% decode (clean rerun) at the same KL and tasks", "R34, R37"),
        ("no", "Weights: Q4_K_M + FFN up/gate->Q3_K recipe = slower (0.95-0.98x), KL x2.1", "R34"),
        ("nm", "Fused attention decoder for lossless KV (condition 'faster than Q8' not met)", "R22"),
        ("good", "Q4_0 + 1% exact outliers beats plain Q4_0 (KL 0.0063 vs 0.0080); real q4_0 path costs 9x the format", "R36"),
        ("nm", "Lossless weights + KV in one llama.cpp run (combo)", "R35"),
        ("nm", "Speed on A100 / H100 / H200 / B200 / L40S / RTX 5090: analytic projection (*) only", "R24, R31"),
    ]
    fig, ax = plt.subplots(figsize=(17, 13.5)); ax.set_xlim(0, 17); ax.set_ylim(len(rows) + 0.2, -1.2); ax.axis("off")
    for i, (s, t, r) in enumerate(rows):
        col, lab = ST[s]
        ax.add_patch(Rectangle((0.0, i - 0.42), 16.9, 0.84, fc="#f7f7f7", ec="white"))
        ax.add_patch(Circle((0.55, i), 0.3, fc=col, ec="white", lw=2))
        ax.text(1.05, i, lab, fontsize=10.5, va="center", color=col if s != "part" else "#8a6a00", fontweight="bold")
        ax.text(2.9, i, t, fontsize=11.5, va="center", color=INK)
        ax.text(16.8, i, r, fontsize=10, va="center", ha="right", color=MUTED)
    ax.text(0.2, -0.85, "status", fontsize=11, fontweight="bold", color=MUTED); ax.text(2.9, -0.85, "what we tested", fontsize=11, fontweight="bold", color=MUTED); ax.text(16.8, -0.85, "source", fontsize=11, fontweight="bold", color=MUTED, ha="right")
    legend_fig(fig, [("worked", LL), ("partly (CPU only / proxy only / weak test)", AMBER), ("did not work (negative result)", LY), ("not measured", BASE)], y=0.045, ncol=4)
    finish(fig, "fig29_verdict_map", "Verdict map: what worked, what did not, what is not measured",
           "",
           note="All statuses from RESULTS.md / COMPARISON_TABLE.md (R-numbers at the right). 'Partly' = a real result with a scope limit named in the row. Qwen3-4B Q4_K_M and Qwen3.5-2B on one RTX 3090.", top=0.90, bottom=0.09, tl=False)
