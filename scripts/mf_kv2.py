# mf_kv2.py -- extra KV figures 30-42 (quality ladders, lossless codecs, compute buffer, real-KV structure, PoC-B, KV size).
from mf_common import *

HAT = "////"


# ================================================================ KL ladder of KV types
def fig30_kv_kl_ladder():
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 7), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    types = [("Q8_0", 8.5, .00419, .00083), ("q5_1", 6.0, .01354, .00053), ("q5_0", 5.5, .01679, .00087), ("q4_1", 5.0, .05431, .00195),
             ("Q4_0", 4.5, .05945, .00189), ("iq4_nl", 4.5, .09831, .00387)]
    cols = [GR["Q8_0"], BASE, BASE, BASE, GR["Q4_0"], BASE]
    for i, (n, b, k, e) in enumerate(types):
        ax.bar(i, k, 0.7, color=cols[i], edgecolor=INK, lw=0.8, yerr=e, capsize=4, error_kw=dict(ecolor=INK, lw=1.3))
        ax.text(i, k * 1.35 + 0.0005, f"{k:.3f}", ha="center", fontsize=10.5, fontweight="bold")
    ax.axhspan(4.5e-4, 8e-4, color=LL, alpha=0.9); ax.text(5.45, 6.2e-4, "ours LOSSLESS = 0 (bit-exact)", ha="right", va="center", fontsize=10.5, color="white", fontweight="bold")
    ax.set_yscale("log"); ax.set_ylim(4.5e-4, 0.4)
    ax.set_xticks(range(6)); ax.set_xticklabels([f"{n}\n{b} bit" for n, b, _, _ in types])
    ax.set_ylabel("mean KL divergence vs F16 KV (log scale)")
    ax.set_title("Context 2K, CPU, 6 chunks (R12)", fontsize=12.5)
    ax.text(0.02, 0.97, "step between 5-bit (~0.014-0.017)\nand 4-bit (~0.054-0.098)", transform=ax.transAxes, va="top", fontsize=10.5, color=MUTED)
    better(ax, "down", at=(0.05, 0.76), L=0.16)
    ax = axs[1]
    ctxs = ["2K\n(CPU, R12)", "4K\n(GPU, R19)", "32K\n(GPU, R29)"]
    q8 = [(.00419, .00083), (.00268, .00007), (.00128, .00012)]; q4 = [(.05945, .00189), (.06939, .00336), (.03487, .00171)]
    x = np.arange(3); w = 0.34
    for off, data, col, nm_ in ((-w / 2, q8, GR["Q8_0"], "Q8_0 KV"), (w / 2, q4, GR["Q4_0"], "Q4_0 KV")):
        ax.bar(x + off, [d[0] for d in data], w * 0.92, color=col, edgecolor=INK, lw=0.8, yerr=[d[1] for d in data], capsize=3, label=nm_, error_kw=dict(ecolor=INK, lw=1.2))
        for xi, d in zip(x + off, data):
            ax.text(xi, d[0] * 1.35, f"{d[0]:.4f}" if d[0] < 0.01 else f"{d[0]:.3f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.axhspan(4.5e-4, 8e-4, color=LL, alpha=0.9)
    ax.set_yscale("log"); ax.set_ylim(4.5e-4, 0.4); ax.set_xticks(x); ax.set_xticklabels(ctxs)
    ax.set_title("Does KL grow with context? Not for these two", fontsize=12.5)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.92))
    ax.text(0.02, 0.95, "Ours LOSSY KV variants: KL in R36\n(exact emulation, paper Tables 4.1-4.3).\n32K = one chunk with repeated text:\nabsolute KL not comparable with 2K/4K.", transform=ax.transAxes, va="top", fontsize=9.5, color=MUTED)
    better(ax, "down", at=(0.67, 0.60), L=0.16)
    finish(fig, "fig30_kv_kl_ladder", "How much quality do built-in KV types lose? (KL vs F16 KV)",
           "",
           note="Source: R12 (context 2048, CPU), R19 (context 4096, GPU), R29 (context 32768, GPU, one chunk). Qwen3-4B Q4_K_M weights; error bars = reported +- values.", top=0.87, bottom=0.08)


# ================================================================ quality scoreboard
def fig31_quality_scoreboard():
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.8), gridspec_kw={"width_ratios": [0.8, 1.7, 0.9]})
    ax = axs[0]
    names = ["F16", "Q8_0", "Q4_0"]; vals = [25, 25, 21]; cols = [GR["F16"], GR["Q8_0"], GR["Q4_0"]]
    ax.bar(range(3), vals, 0.65, color=cols, edgecolor=INK, lw=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, f"{v}/39", ha="center", fontsize=11.5, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(["F16 KV\nreference", "Q8_0 KV\nlost 0, gained 0\np = 1.0", "Q4_0 KV\nlost 5, gained 1\np = 0.22"], fontsize=9.5); ax.set_ylim(0, 33); ax.set_ylabel("cases passed of 39 (frozen suite)")
    ax.set_title("KV types, Qwen3-4B, ctx 16K (R20)", fontsize=11.5)
    better(ax, "up", at=(1.04, 0.45), L=0.18, lab_right=True)
    ax = axs[1]
    wn = ["BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M\np=0.021", "Q2_K\n38/39 hit\nlimit", "BF16+\nFFN->Q8", "Q8_0+\nFFN->Q6K", "Q4KM+\nemb->Q4K", "Q4KM+FFN\nup/gate\n->Q3K"]
    wv = [27, 28, 28, 25, 27, 20, 0, 28, 29, 28, 25]
    wc = [BASE] * 7 + [LY] * 4
    ax.bar(range(len(wv)), wv, 0.7, color=wc, edgecolor=INK, lw=0.8)
    for i, v in enumerate(wv): ax.text(i, v + 0.5, str(v), ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(wv))); ax.set_xticklabels(wn, fontsize=7.8); ax.set_ylim(0, 33)
    ax.axvline(6.5, color="#bbbbbb", ls=":"); ax.text(3, 31.5, "built-in weight types (base)", ha="center", fontsize=10.5, color=MUTED)
    ax.text(9, 31.5, "our recipes (lossy)", ha="center", fontsize=10.5, color=LY, fontweight="bold")
    ax.set_title("Weights, Qwen3.5-2B, ctx 8K (R20; recipes from COMPARISON_TABLE)", fontsize=11.5)
    better(ax, "up", at=(1.02, 0.45), L=0.18, lab_right=True)
    ax = axs[2]
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(-1.2, 2.7); ax.axis("off")
    ax.set_title("Needle in a haystack (R27, R32)", fontsize=11.5)
    for j, (c, t) in enumerate([("4K", "5/5"), ("32K", "5/5"), ("94K\n(YaRN)", "3/3")]):
        ax.text(j, 2.45, c, ha="center", va="center", fontsize=11, fontweight="bold")
        for i, f in enumerate(["F16", "Q8_0", "Q4_0"]):
            ax.add_patch(FancyBboxPatch((j - 0.4, 1.7 - i * 0.9 - 0.3), 0.8, 0.6, boxstyle="round,pad=0,rounding_size=0.08", fc=LL, ec="white"))
            ax.text(j, 1.7 - i * 0.9, t, ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    for i, f in enumerate(["F16", "Q8_0", "Q4_0"]):
        ax.text(-0.5, 1.7 - i * 0.9, f, ha="right", va="center", fontsize=11, color="#444")
    ax.text(1, -0.9, "all formats find every needle: this test\ndoes not separate KV types (too easy)", ha="center", fontsize=9.5, color=MUTED)
    finish(fig, "fig31_quality_scoreboard", "Quality scoreboard: tasks and needle-in-a-haystack",
           "",
           note="39 deterministic cases, greedy decoding, paired sign tests; with 39 cases only large losses are detectable. Arithmetic cases fail in all arms (scorer strictness). Our KV variants: tasks n/m.", top=0.86, bottom=0.08)


# ================================================================ lossless F16 codecs vs page size
def fig32_lossless_f16_codecs():
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.8))
    ax = axs[0]
    P = [16, 64, 256]
    codecs = [("zlib-6,\nbytes as stored", [0.930, 0.928, 0.928]), ("byte planes\n+ zlib-6", [0.883, 0.872, 0.864]),
              ("byte planes\n+ LZMA-6", [0.874, 0.855, 0.847]), ("sign+exp bytes zlib,\nmantissa raw", [0.902, 0.896, 0.893])]
    w = 0.26
    shades = ["#7fcdb5", LL, "#0f6b50"]
    for pi, p in enumerate(P):
        for ci, (n, v) in enumerate(codecs):
            ax.bar(ci + (pi - 1) * w, v[pi], w * 0.92, color=shades[pi], edgecolor="white", hatch=HAT, label=f"{p}-token pages" if ci == 0 else None)
            ax.text(ci + (pi - 1) * w, v[pi] + 0.003, f"{v[pi]:.3f}", ha="center", fontsize=8.3, rotation=90, va="bottom")
    ax.axhline(1.0, color=GR["F16"], ls="--", lw=1.8); ax.text(3.55, 1.002, "F16 = 1.00", ha="right", va="bottom", fontsize=10, color=GR["F16"])
    ax.axhline(0.865, color="#777777", ls=":", lw=1.6, label="order-0 field-entropy estimate K 0.865 / V 0.852")
    ax.set_ylim(0.78, 1.04); ax.set_xticks(range(4)); ax.set_xticklabels([c[0] for c in codecs], fontsize=10); ax.set_ylabel("size relative to F16 (K+V pooled)")
    ax.set_title("Size of bit-exact CPU codecs by page size (R10)", fontsize=12.5)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.80), fontsize=9.5)
    better(ax, "down", at=(1.02, 0.60), L=0.16, lab_right=True)
    ax = axs[1]
    dec = [("zlib-6", 235, 148), ("planes+zlib", 244, 159), ("planes+LZMA", 20, 24), ("sign+exp zlib", 120, 53)]
    x = np.arange(4)
    ax.bar(x - 0.19, [d[1] for d in dec], 0.36, color="#7fcdb5", edgecolor="white", hatch=HAT, label="16-token pages")
    ax.bar(x + 0.19, [d[2] for d in dec], 0.36, color="#0f6b50", edgecolor="white", hatch=HAT, label="256-token pages")
    for xi, d in zip(x, dec):
        ax.text(xi - 0.19, d[1] * 1.08, str(d[1]), ha="center", fontsize=10); ax.text(xi + 0.19, d[2] * 1.08, str(d[2]), ha="center", fontsize=10)
    ax.set_yscale("log"); ax.set_ylim(5, 800); ax.set_xticks(x); ax.set_xticklabels([d[0] for d in dec], fontsize=10.5); ax.set_ylabel("CPU decode speed, MB/s (log)")
    ax.set_title("...but decoding is single-thread Python+stdlib, far from GPU speed", fontsize=12.5)
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    better(ax, "up", at=(1.02, 0.40), L=0.16, lab_right=True)
    finish(fig, "fig32_lossless_f16_codecs_pages", "Lossless F16 KV on CPU: what the codecs reach",
           "",
           note="Source: R10 (artifacts/kv_lossless_proto_20261021/results.json): 339,738,624 B of real KV, every page compressed independently and compared byte for byte; 8 B index per stream per page charged. No GPU decode, VRAM size or tok/s was measured for these codecs.", top=0.87, bottom=0.07)


# ================================================================ lossless on top of three bases
def fig33_lossless_on_bases():
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.8), sharey=True)
    key = {"zlib": "split_zlib_P256", "lzma": "split_lzma_P256", "huff": "huffman_codes", "fix": "fixed_escape"}
    ser = [("planes + zlib", "zlib", 1), ("planes + LZMA", "lzma", 1), ("static Huffman on codes", "huff", 1), ("fixed-rate + escape\n(EXP3 / 7-bit / 3-bit)", "fix", 1)]
    shades = ["#7fcdb5", "#0f6b50", "#a8dccb", LL]
    for ax, comp in zip(axs, ["K", "V"]):
        groups = ["F16 base", "Q8_0 base", "Q4_0 base"]
        w = 0.2
        f16 = {"zlib": 0.864, "lzma": 0.847, "huff": None, "fix": 0.931}
        for si, (nm_, k, cpu) in enumerate(ser):
            for gi, g in enumerate(["f16", "q8_0", "q4_0"]):
                v = f16[k] if g == "f16" else KVQ[g][comp]["ratio_vs_base"][key[k]]
                xp = gi + (si - 1.5) * w
                if v is None:
                    nm_text(ax, xp, 0.62); continue
                hat = HAT if not (g == "f16" and k == "fix") else None
                ax.bar(xp, v, w * 0.92, color=shades[si], edgecolor="white", hatch=hat, label=nm_ if gi == 0 else None)
                ax.text(xp, v + 0.006, f"{v:.3f}", ha="center", fontsize=7.8, rotation=90, va="bottom")
        ax.axhline(1.0, color=GR["F16"], ls="--", lw=1.8); ax.text(-0.5, 1.003, "own base = 1.00", ha="left", va="bottom", fontsize=10, color=GR["F16"])
        ax.set_ylim(0.6, 1.2); ax.set_xticks(range(3)); ax.set_xticklabels(groups, fontsize=11.5); ax.set_xlim(-0.55, 2.6)
        ax.set_title(f"{comp} cache" + ("  (F16 bars: K+V pooled)" if comp == "K" else ""), fontsize=13)
        better(ax, "down", at=(1.02, 0.60), L=0.14, lab_right=True)
    axs[0].set_ylabel("size relative to own base"); axs[0].legend(frameon=False, loc="upper left", fontsize=9.5, ncol=2)
    legend_fig(fig, [("Ours, LOSSLESS (GPU-measured)", LL), ("hatched = CPU prototype / size only", "#bbbbbb", HAT)], y=0.03)
    finish(fig, "fig33_lossless_on_three_bases", "Lossless re-coding on top of F16, Q8_0 and Q4_0 KV",
           "",
           note="Source: R33 (artifacts/kv_quant_lossless_20261021/results.json; 256-token pages; 2304 tokens real KV), R10, R26. All sizes bit-exact. Huffman for F16: n/m. Q4_0 fixed-rate in the GPU layout measured 1.17-1.25 (larger than base).", top=0.86, bottom=0.11)


# ================================================================ k sweep + EXP3 cost
def fig34_exp3_ksweep_cost():
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.8))
    ax = axs[0]
    groups = [("K, 32 channels\nof 1 token", [1.061, 0.999, 0.920]), ("K, 1 channel\nover 32 tokens", [1.034, 0.921, 0.906]),
              ("V, 32 channels\nof 1 token", [1.054, 0.954, 0.908]), ("V, 1 channel\nover 32 tokens", [1.049, 0.943, 0.907])]
    shades = ["#a8dccb", "#5fbd9f", "#0f6b50"]
    w = 0.26
    for ki in range(3):
        for gi, (n, v) in enumerate(groups):
            ax.bar(gi + (ki - 1) * w, v[ki], w * 0.92, color=shades[ki], edgecolor="white", hatch=HAT, label=f"k = {ki + 1} offset bits" if gi == 0 else None)
            ax.text(gi + (ki - 1) * w, v[ki] + 0.004, f"{v[ki]:.3f}", ha="center", fontsize=8.5, rotation=90, va="bottom")
    ax.axhline(1.0, color=GR["F16"], ls="--", lw=1.8); ax.text(3.5, 1.003, "F16 = 1.00", ha="right", va="bottom", fontsize=10, color=GR["F16"])
    ax.axhline(0.9, color="#999", ls=":", lw=1.5); ax.text(-0.45, 0.897, "10% goal", ha="left", va="top", fontsize=10, color="#777")
    ax.set_ylim(0.8, 1.2); ax.set_xticks(range(4)); ax.set_xticklabels([g[0] for g in groups], fontsize=10); ax.set_ylabel("size relative to F16 (CPU estimate)")
    ax.set_title("More offset bits = smaller, until about k = 3 (R11)", fontsize=12.5)
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.99), fontsize=9.5)
    better(ax, "down", at=(1.02, 0.50), L=0.14, lab_right=True)
    ax = axs[1]
    T = [4096, 32768, 65536, 122880]
    size = [pocb("exp3", t, "ratio_vs_f16") for t in T]; spd = [pocb("f16", t, "median_ms") / pocb("exp3", t, "median_ms") for t in T]
    x = np.arange(4)
    ax.bar(x - 0.19, size, 0.36, color=LL, edgecolor="white", label="bytes stored (vs F16)")
    ax.bar(x + 0.19, spd, 0.36, color=LLL, edgecolor=LL, label="read speed (vs F16)")
    for xi, a, b in zip(x, size, spd):
        ax.text(xi - 0.19, a + 0.012, f"{a:.2f}", ha="center", fontsize=10.5); ax.text(xi + 0.19, b + 0.012, f"{b:.2f}", ha="center", fontsize=10.5)
    ax.axhline(1.0, color=GR["F16"], ls="--", lw=1.8)
    ax.set_xticks(x); ax.set_xticklabels(["4K", "32K", "64K", "120K"]); ax.set_ylim(0, 1.25); ax.set_xlabel("tokens read (one layer, real KV)   |   bytes: lower is better, speed: higher is better")
    ax.set_title("GPU: 4-7% fewer bytes, but reading is 27-35% SLOWER (R26)", fontsize=12.5)
    ax.legend(frameon=False, loc="lower left", fontsize=10)
    better(ax, "up", at=(1.02, 0.40), L=0.16, label="better (speed)", lab_right=True)
    finish(fig, "fig34_exp3_ksweep_and_gpu_cost", "EXP3: the size gain is real, the speed price is higher",
           "",
           note="Source: R11 (CPU, groups of 32), R26 (near-clean GPU, bit-exact at every T; extra decode 2.01 ps/element). Hatched = CPU estimate.", top=0.87, bottom=0.07)


# ================================================================ compute buffer growth
def fig35_compute_buffer():
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.8))
    ctx = ["4K", "16K", "32K", "64K", "120K"]
    cbf = [19.75, 22.75, 26.75, 34.75, 48.75]; cbq = [23.58, 74.58, 142.58, 278.58, 516.58]
    kvf = [576, 2304, 4608, 9216, 17280]; kv8 = [306, 1224, 2448, 4896, 9180]; kv4 = [162, 648, 1296, 2592, 4860]
    ax = axs[0]
    ax.plot(range(5), cbf, "o-", color=GR["F16"], lw=2.8, ms=8, mec=INK, label="F16 KV")
    ax.plot(range(5), cbq, "s-", color=LY, lw=2.8, ms=8, mec=INK, label="Q8_0 / Q4_0 KV (same value)")
    for i, (a, b) in enumerate(zip(cbf, cbq)):
        ax.text(i + 0.03, a - 38, f"{a:.0f}", ha="left", va="top", fontsize=10); ax.text(i - 0.06, b + 14, f"{b:.0f}", ha="right", fontsize=10, color=LY, fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels(ctx); ax.set_ylim(0, 600); ax.set_xlabel("context (-c)"); ax.set_ylabel("compute buffer, MiB")
    ax.set_title("The scratch buffer of quantized KV grows with context (R18, R28)", fontsize=12.5)
    ax.legend(frameon=False, loc="upper left")
    better(ax, "down", at=(0.965, 0.50), L=0.16)
    ax = axs[1]
    x = np.arange(5); w = 0.2
    d8 = [a - b for a, b in zip(kvf, kv8)]; d4 = [a - b for a, b in zip(kvf, kv4)]
    p8 = [3246 - 2984, 4978 - 3954, 7286 - 5246, 11902 - 7830, 19980 - 12352]; p4 = [3246 - 2840, 4978 - 3378, 7286 - 4094, 11902 - 5526, 19980 - 8032]
    ax.bar(x - 1.5 * w, d8, w * 0.92, color=GR["Q8_0"], edgecolor=INK, lw=0.6, label="Q8_0: KV buffer saved")
    ax.bar(x - 0.5 * w, p8, w * 0.92, color=GR["Q8_0"], edgecolor=INK, lw=0.6, hatch="\\\\\\", label="Q8_0: process saving (measured)")
    ax.bar(x + 0.5 * w, d4, w * 0.92, color=GR["Q4_0"], edgecolor=INK, lw=0.6, label="Q4_0: KV buffer saved")
    ax.bar(x + 1.5 * w, p4, w * 0.92, color=GR["Q4_0"], edgecolor=INK, lw=0.6, hatch="\\\\\\", label="Q4_0: process saving (measured)")
    ax.set_xticks(x); ax.set_xticklabels(ctx); ax.set_ylabel("MiB saved vs F16 KV"); ax.set_xlabel("context (-c)")
    ax.set_title("Saving of KV buffer vs saving of the whole process", fontsize=12.5)
    ax.text(0.02, 0.97, f"At 120K the compute buffer eats {d8[-1] - p8[-1]:.0f} MiB (Q8_0)\nand {d4[-1] - p4[-1]:.0f} MiB (Q4_0) of the KV saving (derived).", transform=ax.transAxes, va="top", fontsize=10.5, color=LY, fontweight="bold")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.84), fontsize=9.8)
    better(ax, "up", at=(1.02, 0.35), L=0.16, lab_right=True)
    finish(fig, "fig35_compute_buffer_growth", "Hidden cost of quantized KV: a growing compute buffer",
           "",
           note="Source: R18, R28 (llama.cpp declared buffers), R30 (process peaks). F16 compute buffer stays at 20-49 MiB. Differences derived by subtraction.", top=0.87, bottom=0.07)


# ================================================================ real KV structure
def fig40_real_kv_structure():
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.6), gridspec_kw={"width_ratios": [1.05, 1.2, 0.9]})
    ax = axs[0]
    rows = [("order-0 over\nthe 16-bit word", 13.81, 13.58), ("sign+exp+mantissa\nas fields", 13.85, 13.63), ("exponent given\nthe channel", 13.35, 13.51), ("time delta\nper channel", 13.70, 14.08)]
    y = np.arange(4)[::-1]
    ax.barh(y + .19, [r[1] for r in rows], 0.36, color=LL); ax.barh(y - .19, [r[2] for r in rows], 0.36, color=LL, alpha=0.5)
    for yi, r in zip(y, rows):
        ax.text(r[1] + 0.03, yi + .19, f"K {r[1]:.2f}", va="center", fontsize=10); ax.text(r[2] + 0.03, yi - .19, f"V {r[2]:.2f}", va="center", fontsize=10)
    ax.axvline(16, color=GR["F16"], ls="--", lw=1.8); ax.text(15.95, 0.0, "F16 = 16 bits", ha="right", va="center", fontsize=10, color=GR["F16"])
    ax.set_xlim(12.5, 16.2); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5); ax.set_xlabel("bits per element (dark K, light V)")
    ax.set_title("Lossless entropy estimate by model of the data (R9)", fontsize=12)
    better(ax, "left", at=(0.97, 0.56), L=0.2, label="better = fewer bits")
    ax = axs[1]
    pk = KVE["components"]["K"]["per_layer"]["f16_word"]; pv = KVE["components"]["V"]["per_layer"]["f16_word"]
    ax.plot(range(36), pk, "-", color=LL, lw=2.6, label="K"); ax.plot(range(36), pv, "-", color=LL, lw=2.6, alpha=0.5, label="V")
    ax.axhline(16, color=GR["F16"], ls="--", lw=1.8); ax.text(35.5, 15.95, "16 bits (F16)", ha="right", va="top", fontsize=10, color=GR["F16"])
    ax.set_ylim(13, 16.3); ax.set_xlabel("layer (36 layers)"); ax.set_ylabel("bits per element, order-0")
    ax.set_title("Every layer is alike: 13.5-14.0 of 16 bits", fontsize=12)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    better(ax, "down", at=(0.965, 0.55), L=0.16)
    ax = axs[2]
    lay = ["0", "9", "18", "27", "35"]
    rk = [KVE["components"]["K"]["svd_energy_rank"][l]["rank_for_99"] for l in lay]; rv = [KVE["components"]["V"]["svd_energy_rank"][l]["rank_for_99"] for l in lay]
    x = np.arange(5)
    ax.bar(x - 0.19, rk, 0.36, color=LY, edgecolor="white", hatch=HAT); ax.bar(x + 0.19, rv, 0.36, color=LY, alpha=0.5, edgecolor="white", hatch=HAT)
    for xi, a, b in zip(x, rk, rv):
        ax.text(xi - 0.19, a + 15, str(a), ha="center", fontsize=9.5); ax.text(xi + 0.19, b + 15, str(b), ha="center", fontsize=9.5)
    ax.axhline(1024, color=GR["F16"], ls="--", lw=1.8); ax.text(4.5, 1034, "full rank 1024", ha="right", fontsize=10, color=GR["F16"])
    ax.set_xticks(x); ax.set_xticklabels(lay); ax.set_ylim(0, 1150); ax.set_xlabel("layer"); ax.set_ylabel("rank for 99% of the energy")
    ax.set_title("Not low-rank: ~600 of 1024 needed", fontsize=12)
    better(ax, "down", at=(1.02, 0.50), L=0.16, lab_right=True)
    finish(fig, "fig40_real_kv_structure", "What real KV data looks like: how compressible, how low-rank",
           "",
           note="Source: R9 / kv_entropy.json. One capture (Qwen3-4B Q4_K_M, F16 KV, T = 2304, one text, 36 layers x 8 heads x 128). Rank shown for layers 0, 9, 18, 27, 35 (dark K, light V).", top=0.86, bottom=0.07)


# ================================================================ PoC-B bandwidth by T
def fig41_pocb_gbps():
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.8))
    T = [4096, 32768, 65536, 122880]
    fm = [("f16", "F16", GR["F16"], "o"), ("q8_0", "Q8_0", GR["Q8_0"], "s"), ("q4_0", "Q4_0", GR["Q4_0"], "^"), ("exp3", "EXP3 lossless (ours)", LL, "D")]
    ax = axs[0]
    for f, n, c, m in fm:
        v = [pocb(f, t, "GBps_stored") for t in T]
        ax.plot(T, v, marker=m, color=c, mec=INK, ms=8, lw=2.6, label=n)
    ax.axhline(936, color=G, ls="--", lw=2); ax.text(4300, 950, "peak 936 GB/s", fontsize=10.5, color=G, fontweight="bold", va="bottom")
    ctx_axis(ax, T, ["4K", "32K", "64K", "120K"]); ax.set_ylim(0, 1050); ax.set_ylabel("bytes really read per second, GB/s")
    ax.set_xlabel("tokens read (one layer)"); ax.set_title("Stored-bytes throughput (R26, near-clean)", fontsize=12.5)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.84), fontsize=10)
    better(ax, "up", at=(0.965, 0.72), L=0.16)
    ax = axs[1]
    for f, n, c, m in fm:
        v = [pocb(f, t, "median_ms") for t in T]
        ax.plot(T, v, marker=m, color=c, mec=INK, ms=8, lw=2.6, label=n)
    ctx_axis(ax, T, ["4K", "32K", "64K", "120K"]); ax.set_yscale("log"); ax.set_ylabel("time for one pass, ms (log)")
    ax.set_xlabel("tokens read (one layer)"); ax.set_title("Time per pass: Q8_0 and Q4_0 are not faster than F16", fontsize=12.5)
    ax.text(0.03, 0.93, "Three lines overlap: F16, Q8_0, Q4_0 take the same time\nwith 47-72% fewer bytes. EXP3 is 1.4-1.5x slower.", transform=ax.transAxes, va="top", fontsize=10, color=MUTED)
    better(ax, "down", at=(0.965, 0.40), L=0.16)
    finish(fig, "fig41_pocb_throughput_and_time", "PoC-B CUDA microbenchmark: bytes vs time",
           "",
           note="Source: R26 (artifacts/gpu_long_20261021/poc_b.jsonl; REPS 12, ITER 10; real captured KV tiled to T tokens; EXP3 bit-exact at every T). Our kernel reaches 520-530 GB/s (57% of peak) on F16.", top=0.87, bottom=0.07)


# ================================================================ KV size vs context
def fig42_kv_size_vs_context():
    fig, ax = plt.subplots(figsize=(13, 7.4))
    t = np.array([0, 131072]); per = 0.140625   # MiB per token F16, Qwen3-4B (147,456 B)
    lines = [("F16", 1.0, GR["F16"], "-", "o"), ("Q8_0", 8.5 / 16, GR["Q8_0"], "-", "s"), ("Q4_0", 4.5 / 16, GR["Q4_0"], "-", "^")]
    for n, r, c, ls, m in lines:
        ax.plot(t, t * per * r, ls, color=c, lw=3)
    pts = {"F16": [576, 2304, 4608, 9216, 17280], "Q8_0": [306, 1224, 2448, 4896, 9180], "Q4_0": [162, 648, 1296, 2592, 4860]}
    xs = [4096, 16384, 32768, 65536, 122880]
    for n, r, c, ls, m in lines:
        ax.scatter(xs, pts[n], marker=m, s=90, color=c, edgecolor=INK, zorder=5)
    ax.plot(t, t * per * 0.929, "-", color=LL, lw=2.6); ax.plot(t, t * per * 0.847, "--", color=LL, lw=2.6)
    hs = [Line2D([0], [0], color=GR["F16"], lw=3, marker="o", mec=INK, label="F16 (base)"), Line2D([0], [0], color=GR["Q8_0"], lw=3, marker="s", mec=INK, label="Q8_0 (base, x0.531)"),
          Line2D([0], [0], color=GR["Q4_0"], lw=3, marker="^", mec=INK, label="Q4_0 (base, x0.281)"),
          Line2D([0], [0], color=LL, lw=2.6, label="F16 + EXP3 lossless x0.929 (GPU format)"), Line2D([0], [0], color=LL, lw=2.6, ls="--", label="F16 + planes+LZMA x0.847 (CPU only)")]
    ax.legend(handles=hs, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=11)
    free = 24576 - 2376
    ax.axhline(free, color="#999", ls=":", lw=1.8); ax.text(2000, free + 250, "24 GiB card minus Q4_K_M weights = ~22,200 MiB left for KV (+ buffers)", fontsize=10.5, color="#666")
    ax.axvline(40960, color="#bbbbbb", ls=":", lw=1.5); ax.text(41500, 800, "Qwen3-4B trained context\n40,960 tokens", fontsize=10, color="#777")
    ax.set_xlim(0, 133000); ax.set_ylim(0, 24500); ax.set_xticks([0, 32768, 65536, 98304, 131072]); ax.set_xticklabels(["0", "32K", "64K", "96K", "128K"])
    ax.set_xlabel("tokens in the KV cache"); ax.set_ylabel("KV cache size, MiB (Qwen3-4B: 147,456 B per token in F16)")
    ax.set_title("The size formulas match the measured llama.cpp buffers exactly (dots)", fontsize=12.5)
    better(ax, "down", at=(0.80, 0.42), L=0.12)
    finish(fig, "fig42_kv_size_vs_context", "KV size vs context length",
           "",
           note="Dots: R18/R28 KV buffers (4K, 16K, 32K, 64K, 120K). Lines: tokens x 147,456 B x format ratio (Q8_0 34/64, Q4_0 18/64); ours: R26 EXP3 0.929, R10 LZMA 0.847 (CPU only).", top=0.87, bottom=0.07)
