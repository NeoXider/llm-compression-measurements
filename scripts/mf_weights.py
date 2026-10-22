# mf_weights.py -- weight-side figures 24-26 and 36-39 (+ combo).
from mf_common import *
import re, statistics as st

HAT = "////"
# R7 (Qwen3.5-2B, KL vs Q8_0 reference, CPU): config -> (file MiB, mean KL)
R7 = {"BF16": (3716.8, 0.00133), "Q8_0": (1980.5, 0.0), "Q6_K": (1531.9, 0.01063), "Q5_K_M": (1387.4, 0.02059),
      "Q4_K_M": (1251.4, 0.06016), "Q3_K_M": (1075.6, 0.24053), "Q2_K": (944.6, 1.849)}
TASKS = {"BF16": 27, "Q8_0": 28, "Q6_K": 28, "Q5_K_M": 25, "Q4_K_M": 27, "Q3_K_M": 20, "Q2_K": 0}     # R20 (GPU)
# COMPARISON_TABLE D-F: our recipes: name, base, size ratio vs base, KL, tasks, speed d0, speed 4K
REC = [("BF16 + FFN->Q8_0", "BF16", 0.770, 0.0013, 28, 1.160, 1.181),
       ("Q8_0 + FFN->Q6_K", "Q8_0", 0.889, 0.0046, 29, 1.043, 1.058),
       ("Q4_K_M + emb->Q4_K", "Q4_K_M", 0.899, 0.0767, 28, 1.024, 1.046),
       ("Q4_K_M + FFN up/gate->Q3_K", "Q4_K_M", 0.936, 0.1254, 25, 0.982, 0.950)]
# lossless (CPU): name, base, size ratio
LLW = [("BF16: exponent Huffman", "BF16", 0.666), ("BF16: planes+zlib", "BF16", 0.707), ("Q8_0: planes+zlib", "Q8_0", 0.944), ("Q4_K_M: planes+zlib", "Q4_K_M", 0.978)]


def fig24_weights_size_vs_quality():
    fig = plt.figure(figsize=(18, 7.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.55], wspace=0.22)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axk = fig.add_subplot(gs[0, 2]); axk.axis("off")
    items = []   # id, name, MiB, KL, tasks, kind, hatched
    names = list(R7)
    for i, n in enumerate(names):
        mb, kl = R7[n]
        items.append((i + 1, n + (" (base)" if n in ("BF16", "Q8_0", "Q4_K_M") else ""), mb, max(kl, 1.2e-4), TASKS[n], "base" if n in ("BF16", "Q8_0", "Q4_K_M") else "other", 0))
    k = len(items)
    for j, (n, b, r, kl, t, s0, s4) in enumerate(REC):
        items.append((k + j + 1, "ours lossy: " + n, R7[b][0] * r, kl, t, "ly", 0))
    k = len(items)
    for j, (n, b, r) in enumerate(LLW):
        items.append((k + j + 1, "ours lossless: " + n, R7[b][0] * r, 1.2e-4, None, "ll", 1))
    for ai, ax in enumerate(axs):
        for (i, n, mb, kl, t, kind, cpu) in items:
            y = kl if ai == 0 else t
            if y is None: continue
            col = {"base": BASE, "other": BASE, "ly": LY, "ll": LL}[kind]
            mk = "D" if kind == "ll" else ("s" if kind == "other" else "o")
            ax.scatter([mb], [y], s=360, marker=mk, color="white" if kind == "other" else col, edgecolor=BASE if kind == "other" else ("white" if cpu else INK),
                       linewidths=2.2 if kind == "other" else 1, hatch=HAT if cpu else None, zorder=5)
            ax.text(mb, y, str(i), ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=7, color=INK if kind in ("other", "ll") else "white")
        ax.set_xlim(700, 3950); ax.set_xlabel("file size, MiB (Qwen3.5-2B)")
    ax = axs[0]
    ax.axhspan(8e-5, 1.6e-4, color=LL, alpha=0.15); ax.text(3900, 1.2e-4, "KL = 0 (lossless, or Q8_0 = reference)", ha="right", va="center", fontsize=9.5, color=LL)
    ax.set_yscale("log"); ax.set_ylim(7e-5, 4); ax.set_ylabel("mean KL vs Q8_0 logits (log)"); ax.set_title("Size vs KL", fontsize=13)
    better(ax, "dl", at=(0.55, 0.50), L=0.12)
    ax = axs[1]
    ax.set_ylim(-2, 33); ax.set_ylabel("cases passed of 39 (GPU, R20)"); ax.set_title("Size vs tasks (lossless = same tasks as its base)", fontsize=13)
    better(ax, "ul", at=(0.55, 0.40), L=0.12)
    axk.set_xlim(0, 1); axk.set_ylim(0, len(items) + 2)
    axk.text(0, len(items) + 1.2, "KEY", fontsize=12, fontweight="bold")
    for j, (i, n, mb, kl, t, kind, cpu) in enumerate(items):
        col = {"base": "#555555", "other": "#777777", "ly": LY, "ll": LL}[kind]
        axk.text(0, len(items) - j, f"{i:>2}  {n}", fontsize=10.2, color=col, fontweight="bold" if kind != "other" else "normal")
    legend_fig(fig, [("Base / built-in weight types", BASE), ("Ours LOSSY recipe", LY), ("Ours LOSSLESS (diamonds, CPU size only)", LL, HAT)], y=0.03)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.77, bottom=0.16)
    finish(fig, "fig24_weights_size_vs_quality", "Weights: size vs quality, bases and our recipes",
           "Веса: размер против качества. Наши рецепты (FFN в Q8_0 и т.п.) сдвигаются влево почти без потери; без потерь сжимается мало.",
           note="Source: R7 (file MiB, KL), R20 (tasks), COMPARISON_TABLE D-F (recipe ratios; recipe MiB = ratio x base file MiB, derived), R34 (lossless). Qwen3.5-2B; KL vs Q8_0 logits (Q8_0 = 0 by definition).",
           top=0.82, bottom=0.12, tl=False)


def _load_w():
    """tok/s medians by recipe label and depth from artifacts/gpu_bases_20261021_rerun/w/*.jsonl"""
    out = {}
    for f in sorted(glob.glob(os.path.join(ART, "gpu_bases_20261021_rerun", "w", "*.jsonl"))):
        lab = re.sub(r"-b\d\.jsonl$", "", re.sub(r"^\d+-", "", os.path.basename(f)))
        for r in jl(f):
            if r.get("n_gen", 0) > 0:
                out.setdefault((lab, r["n_depth"]), []).extend(r["samples_ts"])
    return {k: st.median(v) for k, v in out.items()}


def fig25_weights_speed():
    W = _load_w()
    # R37 clean-window reruns (decode tok/s at depth 0 / 4096); the last recipe keeps its R34 ratio applied to the clean base
    _clean = {"BF16": (164.5, 166.7), "BF16+ffnQ8": (190.8, 196.8), "Q8_0": (230.7, 239.8), "Q8_0+ffnQ6K": (240.6, 253.8),
              "Q4_K_M": (286.3, 297.2), "Q4_K_M+embQ4K": (293.3, 310.8),
              "Q4_K_M+ffnupgateQ3K": (286.3 * 0.982, 297.2 * 0.950)}
    for _k, (_a, _b) in _clean.items():
        W[(_k, 0)] = _a; W[(_k, 4096)] = _b
    lab = {"BF16": "BF16", "BF16+ffnQ8": "BF16 + FFN->Q8_0", "Q8_0": "Q8_0", "Q8_0+ffnQ6K": "Q8_0 + FFN->Q6_K", "Q4_K_M": "Q4_K_M",
           "Q4_K_M+embQ4K": "Q4_K_M + emb->Q4_K", "Q4_K_M+ffnupgateQ3K": "Q4_K_M + FFN up/gate->Q3_K"}
    order = ["BF16", "BF16+ffnQ8", "Q8_0", "Q8_0+ffnQ6K", "Q4_K_M", "Q4_K_M+embQ4K", "Q4_K_M+ffnupgateQ3K"]
    fig, axs = plt.subplots(1, 2, figsize=(17, 7.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]
    names = [r[0] for r in REC]
    r0 = [r[5] for r in REC]; r4 = [r[6] for r in REC]
    x = np.arange(4)
    ax.bar(x - 0.19, r0, 0.36, color=LY, edgecolor="white", label="context 0")
    ax.bar(x + 0.19, r4, 0.36, color=LY, alpha=0.55, edgecolor="white", label="context 4K")
    for xi, a, b in zip(x, r0, r4):
        ax.text(xi - 0.19, a + 0.012, f"{a:.3f}", ha="center", fontsize=10, rotation=90, va="bottom"); ax.text(xi + 0.19, b + 0.012, f"{b:.3f}", ha="center", fontsize=10, rotation=90, va="bottom")
    ax.axhline(1.0, color=BASE, ls="--", lw=2); ax.text(-0.45, 1.004, "own base = 1.00", ha="left", va="bottom", fontsize=10, color="#555")
    ax.set_xticks(x); ax.set_xticklabels(["BF16\n+ FFN->Q8_0", "Q8_0\n+ FFN->Q6_K", "Q4_K_M\n+ emb->Q4_K", "Q4_K_M\n+ FFN up/gate\n->Q3_K"], fontsize=9.5)
    ax.set_ylim(0.8, 1.38); ax.set_ylabel("decode speed relative to own base")
    ax.set_title("Our lossy recipes: speed vs own base (R37 clean window; last bar R34)", fontsize=12.5)
    ax.text(0.02, 0.97, "Lossless weight re-codings: speed n/m\n(no GPU decoder exists)", transform=ax.transAxes, va="top", fontsize=10, color=LL, fontweight="bold")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.92))
    better(ax, "up", at=(1.02, 0.40), L=0.16, lab_right=True)
    ax = axs[1]
    x = np.arange(len(order)); cols = [BASE, LY, BASE, LY, BASE, LY, LY]
    v0 = [W[(k, 0)] for k in order]; v4 = [W[(k, 4096)] for k in order]
    ax.bar(x - 0.19, v0, 0.36, color=cols, edgecolor="white"); ax.bar(x + 0.19, v4, 0.36, color=cols, alpha=0.55, edgecolor="white")
    for xi, a, b in zip(x, v0, v4):
        ax.text(xi - 0.19, a + 3, f"{a:.0f}", ha="center", fontsize=9.5, rotation=90, va="bottom"); ax.text(xi + 0.19, b + 3, f"{b:.0f}", ha="center", fontsize=9.5, rotation=90, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(["BF16", "BF16\n+ FFN->Q8_0", "Q8_0", "Q8_0\n+ FFN->Q6_K", "Q4_K_M", "Q4_K_M\n+ emb->Q4_K", "Q4_K_M\n+ FFN up/gate\n->Q3_K"], fontsize=8.8)
    ax.set_ylim(0, 340); ax.set_ylabel("decode tok/s (median, 6 samples; last bar derived from R34 ratio)")
    ax.set_title("Absolute decode tok/s (dark = context 0, light = 4K)", fontsize=12.5)
    better(ax, "up", at=(1.02, 0.40), L=0.16, lab_right=True)
    legend_fig(fig, [("Base", BASE), ("Ours LOSSY recipe", LY)], y=0.03)
    # sanity print against COMPARISON_TABLE
    for k, base in (("BF16+ffnQ8", "BF16"), ("Q8_0+ffnQ6K", "Q8_0"), ("Q4_K_M+embQ4K", "Q4_K_M"), ("Q4_K_M+ffnupgateQ3K", "Q4_K_M")):
        print("  speed check", k, round(W[(k, 0)] / W[(base, 0)], 3), round(W[(k, 4096)] / W[(base, 4096)], 3))
    finish(fig, "fig25_weights_recipe_speed", "Weights: decode speed of our recipes",
           "Скорость decode наших рецептов весов: BF16 с FFN в Q8_0 быстрее на 17-20%; FFN up/gate в Q3_K - медленнее базы.",
           note="Source: R34 / COMPARISON_TABLE (ratios), artifacts/gpu_bases_20261021_rerun/w/*.jsonl (absolute tok/s re-derived as medians; llama-bench, ABBA, KV F16). Qwen3.5-2B, RTX 3090.",
           top=0.86, bottom=0.11)


def fig26_combo_model_plus_kv():
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]
    combos = ["BF16 weights\n+ F16 KV", "Q8_0 weights\n+ Q8_0 KV", "Q4_K_M weights\n+ Q4_0 KV"]
    base = {"4K": [151.1, 218.3, 256.4], "32K": [142.8, 194.4, 225.0]}
    ours = {"4K": [None, 220.0, 270.6], "32K": [None, 196.3, 231.5]}
    w = 0.2; x = np.arange(3)
    for ci, (ctx, off) in enumerate((("4K", -1.5 * w), ("32K", 0.5 * w))):
        for xi, b, o in zip(x, base[ctx], ours[ctx]):
            ax.bar(xi + off, b, w * 0.92, color=BASE, alpha=1 if ctx == "4K" else 0.6, edgecolor="white")
            ax.text(xi + off, b + 3, f"{b:.0f}", ha="center", fontsize=9.5, rotation=90, va="bottom")
            if o is None:
                nm_text(ax, xi + off + w, 4, txt="n/m")
            else:
                ax.bar(xi + off + w, o, w * 0.92, color=LY, alpha=1 if ctx == "4K" else 0.6, edgecolor="white")
                ax.text(xi + off + w, o + 3, f"{o:.0f}", ha="center", fontsize=9.5, rotation=90, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(combos, fontsize=11.5); ax.set_ylim(0, 330); ax.set_ylabel("decode tok/s (llama-bench, Qwen3.5-2B)")
    ax.set_title("Decode: base combo vs combo + our weight recipe (KV unchanged)", fontsize=12.5)
    ax.text(0.02, 0.97, "per pair: gray = base, orange = + our recipe; dark = depth 4K, light = 32K\nonly 6 of 24 layers have KV, so KV effects are small here", transform=ax.transAxes, va="top", fontsize=9.8, color=MUTED)
    better(ax, "up", at=(1.02, 0.40), L=0.16, lab_right=True)
    ax = axs[1]
    sz = [None, 0.889, 0.899]
    for xi, s in zip(x, sz):
        ax.bar(xi - 0.19, 1.0, 0.36, color=BASE, edgecolor="white")
        if s is None:
            nm_text(ax, xi + 0.19, 0.05, txt="n/m in combo\n(weights alone:\n0.770)", rot=0, fs=9)
        else:
            ax.bar(xi + 0.19, s, 0.36, color=LY, edgecolor="white"); ax.text(xi + 0.19, s + 0.015, f"{s:.3f}", ha="center", fontsize=10.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["BF16", "Q8_0", "Q4_K_M"]); ax.set_ylim(0, 1.25); ax.set_ylabel("weights size relative to base")
    ax.set_title("Weights size (VRAM of combos: n/m)", fontsize=12.5)
    better(ax, "down", at=(1.02, 0.60), L=0.16, lab_right=True)
    legend_fig(fig, [("Base combo", BASE), ("+ ours LOSSY weight recipe", LY)], y=0.03)
    finish(fig, "fig26_combo_weights_plus_kv", "Combo: model weights + KV format",
           "Комбо «модель + KV»: рецепт весов даёт +1% / +5% к скорости при KV без изменений; VRAM комбо и комбо без потерь - не измерены.",
           note="Source: R35 / COMPARISON_TABLE G (decode tok/s at depth 4096 / 32768, 6 samples/cell; combos run: Q8_0+FFN-Q6K with Q8_0 KV; Q4_K_M+emb-Q4_K with Q4_0 KV). Lossless on both weights and KV together: analytic only. Combo quality and VRAM: n/m.",
           top=0.86, bottom=0.11)


def fig36_q4k_field_entropy():
    fig, axs = plt.subplots(1, 2, figsize=(16.5, 7), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    rows = [("Q4_K  4-bit codes", 4, 3.866), ("Q4_K  6-bit scales/mins", 8, 6.98), ("Q4_K  d (f16)", 16, 9.26), ("Q4_K  dmin (f16)", 16, 10.19),
            ("Q6_K  6-bit codes", 6, 5.42), ("Q6_K  int8 scales", 8, 6.99), ("Q6_K  d (f16)", 16, 8.40)]
    y = np.arange(len(rows))[::-1]
    for yi, (n, nom, e) in zip(y, rows):
        ax.barh(yi, nom, 0.78, color="#eeeeee", edgecolor="#bbbbbb"); ax.barh(yi, e, 0.5, color=LL, edgecolor="white", hatch=HAT)
        ax.text(nom + 0.2, yi, f"{e:.2f} of {nom}  ({e / nom:.2f})", va="center", fontsize=10.5)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=11); ax.set_xlim(0, 25); ax.set_xlabel("bits per symbol (grey = stored, green = order-0 entropy)")
    ax.set_title("Where the redundancy hides in k-quant blocks (R4, Qwen3-4B Q4_K_M)", fontsize=12)
    better(ax, "left", at=(0.97, 0.36), L=0.2, label="better = fewer bits")
    ax = axs[1]
    nat = [1568931840, 147087360, 24514560, 24514560, 663306240, 55275520, 6909440]
    ide = [1516534981, 128374750, 14186343, 15609066, 599225558, 48318202, 3626672]
    sv = [(a - b) / 1e6 for a, b in zip(nat, ide)]
    ax.barh(y, sv, 0.6, color=LL, edgecolor="white", hatch=HAT)
    for yi, s in zip(y, sv): ax.text(s + 1, yi, f"{s:.1f} MB", va="center", fontsize=10.5)
    ax.set_yticks(y); ax.set_yticklabels([""] * len(rows)); ax.set_xlim(0, 92); ax.set_xlabel("bytes saved by an ideal order-0 coder, MB (derived)")
    ax.set_title("Most savings sit in the big code fields", fontsize=12)
    ax.text(0.97, 0.18, "all quantized tensors: ideal 0.9339\n= -6.61% (only an order-0 bound,\nno codec tables or random access)", transform=ax.transAxes, ha="right", fontsize=10, color=MUTED)
    better(ax, "right", at=(0.30, 0.04), L=0.2, label="better = more saved")
    finish(fig, "fig36_k_quant_field_entropy", "Q4_K / Q6_K weights: which fields can be squeezed",
           "Какие поля k-квантов сжимаются: масштабы d, dmin - сильнее всего (до 0.58), но основной выигрыш в кодах; итого около -6.6% и это потолок.",
           note="Source: R4 (artifacts/q4k_code_entropy_20261021/entropy.json): order-0 empirical entropy per tensor. Saved bytes = native - ideal (derived). Hatched: CPU analysis only.", top=0.86, bottom=0.08)


def fig37_headroom_ladder():
    fig, ax = plt.subplots(figsize=(14, 6.8))
    rows = [("Order-0 entropy ideal (bound)", 0.9339, BASE, None, "R4"),
            ("zlib-6, 256 KiB frames, whole file", 2387186155 / 2497280256, LL, HAT, "R4 repo screen"),
            ("planes + LZMA-6, 256 KiB frames", 0.9572, LL, HAT, "R34 data"),
            ("planes + zlib, 64 KiB frames", 0.9623, LL, HAT, "R34 data"),
            ("zlib on whole file at 4 KiB frames", 0.9960, LL, HAT, "R4 (-0.37...-0.42%)"),
            ("CQW2 GPU-shaped palette format (Q4_K sample)", 1.0061, LL, None, "R3: LARGER than base")]
    y = np.arange(len(rows))[::-1]
    for yi, (n, v, c, h, src) in zip(y, rows):
        ax.barh(yi, v, 0.62, color=c, edgecolor="white", hatch=h)
        ax.text(v + 0.0015, yi, f"{v:.3f}   ({src})", va="center", fontsize=10.5)
    ax.axvline(1.0, color=GR["F16"], ls="--", lw=2); ax.text(0.9995, -0.62, "native Q4_K_M = 1.00", ha="right", fontsize=10.5, color=GR["F16"])
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=11.5); ax.set_xlim(0.9, 1.04); ax.set_ylim(-0.95, len(rows) - 0.4)
    ax.set_xlabel("size relative to native Qwen3-4B Q4_K_M weights")
    ax.set_title("The more a format is shaped for the GPU (small random-access pieces), the less it saves", fontsize=12.5)
    better(ax, "left", at=(0.97, 0.90), L=0.2, label="better = smaller")
    finish(fig, "fig37_weights_headroom_vs_reality", "Lossless weights: the ladder from ideal to GPU-shaped",
           "Лестница сжатия весов без потерь: потолок -6.6%, большие кадры ~-4.4%, маленькие произвольного доступа ~-0.4%, GPU-палитра +0.6%.",
           note="Source: R3, R4, R34. Qwen3-4B Q4_K_M. Bars: bound = grey; lossless codecs = green (hatched = CPU only). Frame sizes are what a GPU decoder could use for random access.", top=0.87, bottom=0.08)


def fig38_weights_headroom_by_file():
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.6), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]
    files = [("Qwen3-4B\nQ4_K_M", 0.9339), ("2B Q8_0", 0.9274), ("2B Q6_K", 0.9534), ("2B Q5_K_M", 0.9685), ("2B Q4_K_M", 0.9509), ("2B Q3_K_M", 0.9624), ("2B Q2_K", 0.9499)]
    x = np.arange(len(files))
    ax.bar(x, [f[1] for f in files], 0.68, color=BASE, edgecolor="white")
    for xi, f in zip(x, files): ax.text(xi, f[1] + 0.002, f"{f[1]:.3f}", ha="center", fontsize=10, rotation=90, va="bottom")
    ax.axhline(1.0, color=GR["F16"], ls="--", lw=1.8)
    ax.set_ylim(0.85, 1.03); ax.set_xticks(x); ax.set_xticklabels([f[0] for f in files], fontsize=10); ax.set_ylabel("ideal order-0 size / native size")
    ax.set_title("Ideal lossless headroom of k-quant files: only 2-7% (R16)", fontsize=12.5)
    better(ax, "down", at=(1.02, 0.55), L=0.16, lab_right=True)
    ax = axs[1]
    pairs = [("Qwen3-4B\nQ4_K_M", 0.9339, 0.9623), ("2B Q8_0", 0.9274, 0.9440), ("2B Q4_K_M", 0.9509, 0.9782), ("2B Q3_K_M", 0.9624, 0.9855), ("2B BF16\n(Huffman)", 0.6655, 0.7066)]
    x = np.arange(len(pairs)); w = 0.36
    ax.bar(x - w / 2, [p[1] for p in pairs], w * 0.95, color=BASE, edgecolor="white", label="ideal order-0 / Huffman size")
    ax.bar(x + w / 2, [p[2] for p in pairs], w * 0.95, color=LL, edgecolor="white", hatch=HAT, label="real: planes + zlib, 64 KiB frames (CPU)")
    for xi, p in zip(x, pairs):
        ax.text(xi - w / 2, p[1] + 0.01, f"{p[1]:.3f}", ha="center", fontsize=9, rotation=90, va="bottom"); ax.text(xi + w / 2, p[2] + 0.01, f"{p[2]:.3f}", ha="center", fontsize=9, rotation=90, va="bottom")
    ax.axhline(1.0, color=GR["F16"], ls="--", lw=1.8)
    ax.set_ylim(0.5, 1.28); ax.set_xticks(x); ax.set_xticklabels([p[0] for p in pairs], fontsize=10); ax.set_ylabel("size relative to native")
    ax.set_title("Ideal vs achieved (R16, R34)", fontsize=12.5)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    better(ax, "down", at=(1.02, 0.55), L=0.16, lab_right=True)
    finish(fig, "fig38_weights_lossless_headroom_by_file", "How much can quantized weights shrink without loss?",
           "Сколько можно сжать веса без потерь: у квантованных всего 2-7%, у BF16 - до 33% (как в DFloat11), но только на CPU.",
           note="Source: R16 (gguf_quant_entropy.json), R34 / weights_lossless_20261021 (bf16.json, quant.json). BF16 'ideal' = exponent Huffman with sign+mantissa raw (size from code lengths).", top=0.86, bottom=0.07)


def fig39_outliers_and_sparsity():
    fig, axs = plt.subplots(1, 2, figsize=(16.5, 6.8))
    ax = axs[0]
    names = ["weights (Q4_K)", "K cache", "V cache"]
    a = [0.787, 0.595, 0.741]; b = [0.557, 0.240, 0.419]
    x = np.arange(3); w = 0.36
    ax.bar(x - w / 2, a, w * 0.95, color=LY, edgecolor="white", hatch=HAT, label="0.1% outliers moved aside (+0.032 b/elt)")
    ax.bar(x + w / 2, b, w * 0.95, color=LY, alpha=0.55, edgecolor="white", hatch=HAT, label="1% outliers moved aside (+0.320 b/elt)")
    for xi, p, q in zip(x, a, b):
        ax.text(xi - w / 2, p + 0.012, f"{p:.2f}", ha="center", fontsize=10.5); ax.text(xi + w / 2, q + 0.012, f"{q:.2f}", ha="center", fontsize=10.5)
    ax.axhline(1.0, color=BASE, ls="--", lw=2); ax.text(2.45, 1.01, "4-bit quantizer without clipping = 1.00", ha="right", fontsize=10, color="#555")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11.5); ax.set_ylim(0, 1.2); ax.set_ylabel("4-bit MSE relative to no clipping")
    ax.set_title("Setting 1% outliers aside quarters the K error (R17)", fontsize=12.5)
    ax.legend(frameon=False, loc="lower left", fontsize=9.5)
    better(ax, "down", at=(1.02, 0.55), L=0.16, lab_right=True)
    ax = axs[1]
    items = [("Q4_K code value 0\n(weights)", 4.50), ("sub-blocks with\none code (weights)", 0.022), ("|x| < 1e-3 of max\n(K cache)", 3.36), ("|x| < 1e-3 of max\n(V cache)", 1.89),
             ("exact zeros\n(K cache)", 0.000005), ("exact zeros\n(V cache)", 0.000012), ("constant 256-weight\nblocks", 0.0)]
    vals = [i[1] for i in items]
    ax.barh(range(len(items))[::-1], [max(v, 1e-6) for v in vals], 0.6, color=BASE, edgecolor="white")
    for yi, (n, v) in zip(range(len(items))[::-1], items): ax.text(max(v, 1e-6) * 1.4, yi, f"{v:g}%", va="center", fontsize=10.5)
    ax.set_xscale("log"); ax.set_xlim(1e-6, 60); ax.set_yticks(range(len(items))[::-1]); ax.set_yticklabels([i[0] for i in items], fontsize=10)
    ax.set_xlabel("share of elements, % (log; 0 drawn at the left edge)")
    ax.set_title("Almost no zeros or trivial blocks to exploit (R17)", fontsize=12.5)
    ax.text(0.97, 0.10, "Q4_K code 0 = block minimum,\nnot a zero weight", transform=ax.transAxes, ha="right", fontsize=9.8, color=MUTED)
    better(ax, "left", at=(0.97, 0.88), L=0.2, label="better = more to exploit")
    finish(fig, "fig39_outliers_and_sparsity", "Outliers and zeros: error statistics, not quality",
           "Выбросы и нули: убрав 1% выбросов, ошибка 4-битного K падает в 4 раза; нулей и тривиальных блоков почти нет. Это статистика, не KL.",
           note="Source: R17 (3,792,960 Q4_K blocks of Qwen3-4B; KV from the R9 capture). Error statistics only: no KL or task quality measured. Hatched = CPU only.", top=0.86, bottom=0.08)
