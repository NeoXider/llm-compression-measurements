# mf_kv.py -- KV-cache figures 14-21 (Pareto, heatmaps, VRAM, speed, bandwidth, lazy pool).
from mf_common import *

# ---- numbers copied from RESULTS.md (R-numbers in comments) -----------------------------------------
DEPTH_L = ["4K", "32K", "64K", "120K"]
DEC = {"F16": [162.2, 88.2, 59.1, 37.6], "Q8_0": [143.2, 64.6, 40.4, 23.6], "Q4_0": [144.2, 63.9, 39.5, 23.3]}   # R25 tok/s
DEC_P5 = [158.8, 84.4, 57.3, 37.0]; DEC_P95 = [171.7, 89.8, 60.5, 37.7]                                         # R25 F16 p5-p95
DEC_R = {"F16": [1, 1, 1, 1], "Q8_0": [0.883, 0.732, 0.684, 0.629], "Q4_0": [0.889, 0.725, 0.668, 0.622]}       # R25
PRE_F16 = [6030, 2215, 1316, 770]; PRE_R = {"Q8_0": [0.944, 0.953, 0.935, 0.920], "Q4_0": [0.917, 0.961, 0.945, 0.913]}  # R25
VR_CTX = ["4K", "16K", "32K", "64K", "120K"]                                                                     # R18 (16K), R30
VRAM = {"F16": [3246, 4978, 7286, 11902, 19980], "Q8_0": [2984, 3954, 5246, 7830, 12352], "Q4_0": [2840, 3378, 4094, 5526, 8032]}
VR_PCT = {"Q8_0": [-8.1, -20.6, -28.0, -34.2, -38.2], "Q4_0": [-12.5, -32.1, -43.8, -53.6, -59.8]}
LBL = {"F16": "F16 KV", "Q8_0": "Q8_0 KV", "Q4_0": "Q4_0 KV"}
MK = {"F16": "o", "Q8_0": "s", "Q4_0": "^"}


def _lines(ax, x, series, **kw):
    for k, v in series.items():
        ax.plot(x, v, marker=MK[k], color=GR[k], mec=INK, mew=0.8, ms=8, lw=2.6, **kw)


def _endlabel(ax, x, y, text, color=INK, dx=6, dy=0, ha="left", fs=11):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, va="center", fontsize=fs, color=color, fontweight="bold")


# =========================================================================== 4. Pareto size vs error
def fig14_pareto_size_error():
    fig = plt.figure(figsize=(18, 7.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.62], wspace=0.22)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axk = fig.add_subplot(gs[0, 2]); axk.axis("off")
    # id, name, size vs F16, K err, V err, kind, hatched
    P = [
        (1, "F16 (base)", 1.0, 0, 0, "base", 0),
        (2, "Q8_0 (base)", 8.5 / 16, .00892, .00604, "base", 0),
        (3, "Q4_0 (base)", 4.5 / 16, .1219, .0965, "base", 0),
        (4, "q5 symmetric (other type)", 5.5 / 16, .0681, .0511, "other", 0),
        (5, "q4_1 (other type)", 5.0 / 16, .0986, .0836, "other", 0),
        (6, "q3 symmetric (other type)", 3.5 / 16, .2645, .2519, "other", 0),
        (7, "F16 + EXP3 lossless (GPU)", 0.931, 0, 0, "ll", 0),
        (8, "F16 + planes+LZMA lossless", 0.847, 0, 0, "ll", 1),
        (9, "Q8_0 + LZMA lossless", None, .00892, .00604, "ll", 1),
        (10, "Q4_0 + LZMA lossless", None, .1219, .0965, "ll", 1),
        (11, "F16 drop 2 mantissa bits", 0.875, .00134, .00134, "ly", 1),
        (12, "F16 drop 4 mantissa bits (GPU)", 0.797, .0063, .0063, "ly", 0),
        (13, "Q8_0 7-bit codes (GPU)", 0.9118 * 8.5 / 16, .0229, .0160, "ly", 0),
        (14, "Q8_0 6-bit + 1% outliers", 6.82 / 16, .01855, .01985, "ly", 1),
        (15, "Q4_0 4-bit + 1% outliers", 4.82 / 16, .0722, .0780, "ly", 1),
        (16, "Q4_0 3-bit + 1% outliers", 3.82 / 16, .1856, .2052, "ly", 1),
        (17, "per-channel int2", 3.0 / 16, .1965, .4206, "ly", 1),
        (18, "KIVI-style 4-bit (K per-channel)", 5.0 / 16, .0389, .0836, "ly", 1),
        (19, "Q8_0 + per-channel K scale", 8.5 / 16, .00382, None, "ly", 1),
    ]
    LR = {"K": [(2.89 / 16, .230), (5.78 / 16, .161), (11.56 / 16, .084)], "V": [(2.89 / 16, .419), (5.78 / 16, .273), (11.56 / 16, .121)]}
    LQ = {9: (0.892 * 8.5 / 16, 0.931 * 8.5 / 16), 10: (0.790 * 4.5 / 16, 0.869 * 4.5 / 16)}
    for ai, (ax, comp) in enumerate(zip(axs, ["K", "V"])):
        for (i, name, size, ek, ev, kind, cpu) in P:
            e = ek if comp == "K" else ev
            if e is None: continue
            s = size if size is not None else LQ[i][0 if comp == "K" else 1]
            col = {"base": BASE, "other": BASE, "ll": LL, "ly": LY}[kind]
            mk = "D" if kind == "ll" else ("s" if kind == "other" else "o")
            face = "white" if kind == "other" else col
            ax.scatter([s], [e], s=330, marker=mk, color=face, edgecolor=col if kind == "other" else ("white" if cpu else INK),
                       linewidths=2.2 if kind == "other" else 1, hatch="////" if cpu and kind != "other" else None, zorder=5)
            ax.text(s, e, str(i), ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=7,
                    color=INK if kind == "other" else ("white" if not cpu else INK), bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.55) if cpu and kind != "other" else None)
        xs, ys = zip(*LR[comp]); ax.plot(xs, ys, "--", color=LY, lw=1.4, zorder=2)
        ax.scatter(xs, ys, s=110, marker="v", color=LY, edgecolor="white", hatch="////", zorder=4)
        ax.annotate("L", (xs[1], ys[1]), textcoords="offset points", xytext=(0, 12), fontsize=10, color=LY, fontweight="bold", ha="center")
        ax.set_yscale("symlog", linthresh=1e-3, linscale=0.35); ax.set_ylim(-0.0004, 0.7); ax.set_xlim(0.08, 1.12)
        ax.set_yticks([0, 1e-3, 1e-2, 1e-1, 5e-1]); ax.set_yticklabels(["0 (bit-exact)", "0.1%", "1%", "10%", "50%"])
        ax.axhline(0, color="#999999", lw=1)
        ax.set_xlabel("size relative to F16 (1.0 = no compression)"); ax.set_title(f"{comp} cache ({'keys' if comp == 'K' else 'values'})", fontsize=14)
        better(ax, "dl", at=(0.42, 0.26), L=0.12, label="better: smaller, less error")
        if ai == 0: ax.set_ylabel("reconstruction error vs original F16\n(relative Frobenius norm, symlog axis)")
        else: ax.set_ylabel("")
    # key
    axk.set_xlim(0, 1); axk.set_ylim(0, len(P) + 3)
    axk.text(0, len(P) + 2.4, "KEY", fontsize=12, fontweight="bold")
    for j, (i, name, size, ek, ev, kind, cpu) in enumerate(P):
        y = len(P) + 1.2 - j
        col = {"base": "#555555", "other": "#777777", "ll": LL, "ly": LY}[kind]
        axk.text(0, y, f"{i:>2}  {name}", fontsize=10.5, color=col, fontweight="bold" if kind in ("base", "ll") else "normal")
    axk.text(0, 0.1, " L  low-rank r=128/256/512 (worst)", fontsize=10.5, color=LY)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.15)
    legend_fig(fig, [("Base (F16, Q8_0, Q4_0)", BASE), ("other built-in types", "white"), ("Ours LOSSLESS (diamonds)", LL), ("Ours LOSSY", LY),
                     ("hatched = CPU / error-proxy only", "#bbbbbb", "////")], y=0.03, ncol=5)
    finish(fig, "fig14_pareto_size_vs_error", "Size vs error: where every KV format sits",
           "",
           note="Error is a reconstruction PROXY, not KL (KL of our KV variants: R36, paper Tables 4.1-4.3). Source: R13, R33 (kv_lossy_over_base results.json, kv_quant_lossless), R26. Lossless on Q8_0/Q4_0 keeps the base's own error.",
           top=0.86, bottom=0.12, tl=False)


# =========================================================================== 5. Pareto size vs read speed
def fig15_pareto_size_speed():
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.4), sharey=True)
    for ax, T, tl in zip(axs, (32768, 122880), ("32K tokens", "120K tokens")):
        tf = pocc("f16", T, "median_ms")
        fb = pocc("f16", T, "bytes")
        pts = []   # name, size, speed, kind
        pts.append(("F16", 1.0, 1.0, "base"))
        pts.append(("Q8_0", pocc("q8_0", T, "bytes") / fb, tf / [r for r in POC_C if r["format"] == "q8_0" and r["T"] == T][0]["median_ms"], "base"))
        pts.append(("Q4_0", pocc("q4_0", T, "bytes") / fb, tf / [r for r in POC_C if r["format"] == "q4_0" and r["T"] == T][0]["median_ms"], "base"))
        pts.append(("F16 drop-4 (12-bit)", pocc("f16+drop4", T, "bytes") / fb, tf / pocc("f16+drop4", T, "median_ms"), "ly"))
        pts.append(("Q8 7-bit codes", pocc("q8_0+7bit", T, "bytes") / fb, tf / pocc("q8_0+7bit", T, "median_ms"), "ly"))
        pts.append(("Q4 3-bit + escape", pocc("q4_0+esc3", T, "bytes") / fb, tf / pocc("q4_0+esc3", T, "median_ms"), "ll"))
        pts.append(("F16 EXP3", pocb("exp3", T, "ratio_vs_f16"), pocb("f16", T, "median_ms") / pocb("exp3", T, "median_ms"), "ll"))
        offs = {"F16": (-6, 12), "Q8_0": (10, 12), "Q4_0": (8, 8), "F16 drop-4 (12-bit)": (-12, -30), "Q8 7-bit codes": (-30, -36), "Q4 3-bit + escape": (8, -4), "F16 EXP3": (-10, -26)}
        for name, s, v, kind in pts:
            col = {"base": BASE, "ll": LL, "ly": LY}[kind]
            ax.scatter([s], [v], s=190, marker="o" if kind != "ll" else "D", color=col, edgecolor=INK, linewidths=1, zorder=5)
            ax.annotate(f"{name}\n({s:.2f}, {v:.2f}x)", (s, v), textcoords="offset points", xytext=offs[name], fontsize=9.5, color=INK if kind == "base" else col, fontweight="bold")
        # CPU lossless sizes, speed n/m
        cpu = [("F16 planes+LZMA", 0.847), ("Q8 K LZMA", 0.892 * 8.5 / 16), ("Q8 V LZMA", 0.931 * 8.5 / 16), ("Q4 K LZMA", 0.790 * 4.5 / 16), ("Q4 V LZMA", 0.869 * 4.5 / 16)]
        for i, (n_, s) in enumerate(cpu):
            ax.scatter([s], [0.30], s=120, marker="D", color=LL, edgecolor="white", hatch="////", zorder=4)
        ax.text(0.7, 0.255, "CPU lossless (planes+LZMA): F16 0.85, Q8 0.47-0.49, Q4 0.22-0.24\nread speed n/m (no GPU decoder)", ha="center", va="top", fontsize=9.5, color=LL)
        ax.axhline(1.0, color=GR["F16"], ls="--", lw=1.4); ax.axvline(1.0, color=GR["F16"], ls="--", lw=1.4)
        ax.set_xlim(0.1, 1.3); ax.set_ylim(0.12, 1.28)
        ax.set_xlabel("size relative to F16 KV"); ax.set_title(f"One-layer read microbenchmark, {tl}", fontsize=13)
        better(ax, "ul", at=(0.55, 0.45), L=0.14, label="better: smaller, faster")
    axs[0].set_ylabel("read speed relative to F16  (>1 = faster)")
    legend_fig(fig, [("Base", BASE), ("Ours LOSSLESS (GPU)", LL), ("Ours LOSSY (GPU)", LY), ("hatched = CPU only (speed n/m)", "#bbbbbb", "////")], y=0.035)
    finish(fig, "fig15_pareto_size_vs_speed", "Size vs read speed: fewer bytes did not give faster reads",
           "",
           note="Speed = F16 time / format time inside one run of the CUDA microbenchmark (R33 poc_c.jsonl; EXP3 from R26 poc_b.jsonl, ratio within its own run); not llama.cpp decode. Size = stored bytes / F16 bytes.",
           top=0.86, bottom=0.12)


# =========================================================================== 6. Heatmap decode speed
def fig16_heatmap_decode_speed():
    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    nan = np.nan
    rows = ["", "F16 KV  (base)", "Q8_0 KV  (base)", "Q4_0 KV  (base)",
            "",
            "F16 + EXP3 lossless", "F16 + drop-4 bits lossy", "Q8_0 + 7-bit codes lossy", "Q4_0 + 3-bit escape lossless"]
    M = [[nan] * 4, DEC_R["F16"], DEC_R["Q8_0"], DEC_R["Q4_0"], [nan] * 4,
         [0.73, 0.68, 0.65, 0.65], [0.854, 0.977, nan, 0.925], [1.004, 0.984, nan, 0.939], [0.803, 0.750, nan, 0.726]]
    heat(ax, M, rows, ["4K", "32K", "64K", "120K"], CMAP_GOOD_UP, TwoSlopeNorm(1.0, 0.55, 1.2), fs=14, header_rows=[0, 4],
         header_text={0: "DECODE speed in llama.cpp  (x vs F16 KV)", 4: "READ speed of one layer, our formats (x vs own base)"},
         rowcolors=[INK, "#555", "#555", "#555", INK, LL, LY, LY, LL])
    for t, c in zip(ax.get_yticklabels(), [INK, "#444", "#444", "#444", INK, LL, LY, LY, LL]):
        t.set_color(c)
    sm = plt.cm.ScalarMappable(norm=TwoSlopeNorm(1.0, 0.55, 1.2), cmap=CMAP_GOOD_UP)
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.06); cb.set_label("speed vs base  (1.0 = same)")
    cb_better(cb, "up")
    ax.set_title("filled context depth", fontsize=12, pad=30)
    fig.text(0.5, 0.075, "n/m = not measured (our formats at 64K were not run).  Q8_0 / Q4_0 rows: same speed for both - the dequantization, not the bytes, limits them.",
             ha="center", fontsize=10, color=MUTED)
    finish(fig, "fig16_heatmap_decode_speed", "Speed of each KV format vs context depth (heat map)",
           "",
           note="Rows 2-4: llama.cpp decode, RTX 3090, near-clean window, ABBA, 12 samples/cell (R25). Rows 6-9: CUDA read microbenchmark (R26 for EXP3; R33 poc_c for the rest).",
           top=0.88, bottom=0.11, right=0.97)


# =========================================================================== 7. Heatmap VRAM
def fig17_heatmap_vram():
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.8))
    rows = ["F16 KV  (base)", "Q8_0 KV  (base)", "Q4_0 KV  (base)"]
    M = [VRAM[k] for k in ("F16", "Q8_0", "Q4_0")]
    norm = Normalize(2500, 20000)
    heat(axs[0], M, rows, VR_CTX, CMAP_GOOD_DOWN.reversed().reversed(), norm, fmt="{:.0f}", fs=14, rowcolors=["#444"] * 3)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_GOOD_DOWN); cb = fig.colorbar(sm, ax=axs[0], fraction=0.035, pad=0.06); cb.set_label("MiB per process")
    cb_better(cb, "down")
    axs[0].set_title("Whole-process VRAM, MiB (peak), by filled context", fontsize=13, pad=34)
    M2 = [[0] * 5, VR_PCT["Q8_0"], VR_PCT["Q4_0"]]
    norm2 = Normalize(-60, 0)
    cm2 = LinearSegmentedColormap.from_list("sv", ["#ffffff", "#bfe6d9", LL])
    heat(axs[1], [[nan_ for nan_ in [0, 0, 0, 0, 0]], VR_PCT["Q8_0"], VR_PCT["Q4_0"]], rows, VR_CTX, cm2.reversed(), Normalize(-60, 0), fmt="{:+.1f}%", fs=14, header_rows=[0],
         header_text={0: "F16 is the reference = 0%"}, rowcolors=["#444"] * 3)
    sm = plt.cm.ScalarMappable(norm=Normalize(-60, 0), cmap=cm2.reversed()); cb = fig.colorbar(sm, ax=axs[1], fraction=0.035, pad=0.06); cb.set_label("change vs F16 KV (%)")
    cb_better(cb, "down")
    axs[1].set_title("Change vs F16 KV, by filled context", fontsize=13, pad=34)
    fig.text(0.5, 0.075, "4K, 32K: R18/R25 runs;  16K: R18 sampled peak;  64K, 120K: R30 (truly filled, YaRN).  Qwen3-4B Q4_K_M weights (2376 MiB) are in every cell.  All cells fit 24 GiB.",
             ha="center", fontsize=10, color=MUTED)
    finish(fig, "fig17_heatmap_vram", "VRAM of the whole process: KV format x context (heat map)",
           "",
           note="Source: R18, R30 (DXGI per-process peak, 2 repeats identical to the MiB). F16 KV at a filled 120K still fits the 24 GB card (19,980 MiB).",
           top=0.80, bottom=0.12, right=0.97)


# =========================================================================== 8. Stacked VRAM
def fig18_vram_stack():
    ctxs = ["4K", "32K", "120K"]
    KV = {"4K": (576, 306, 162), "32K": (4608, 2448, 1296), "120K": (17280, 9180, 4860)}                 # R18, R28
    CB = {"4K": (19.75, 23.58, 23.58), "32K": (26.75, 142.58, 142.58), "120K": (48.75, 516.58, 516.58)}   # R18, R28
    TOT = {"4K": (3246, 2984, 2840), "32K": (7286, 5246, 4094), "120K": (19980, 12352, 8032)}            # R30
    W = 2375.91                                                                                            # R1 model size, MiB
    fig, ax = plt.subplots(figsize=(14, 7.6))
    x = []; pos = 0; labels = []
    for ci, c in enumerate(ctxs):
        for fi, f in enumerate(("F16", "Q8_0", "Q4_0")):
            kv = KV[c][fi]; cb = CB[c][fi]; tot = TOT[c][fi]; other = tot - W - kv - cb
            b = 0
            for h, col, hat, lab in [(W, "#d0d0d0", None, "weights (Q4_K_M)"), (kv, GR[f], None, "KV buffer"), (cb, LY, None, "compute buffer"), (other, "#f2f2f2", "////", "other (CUDA context etc., by subtraction)")]:
                ax.bar(pos, h, 0.8, bottom=b, color=col, edgecolor="white" if hat is None else "#bbbbbb", hatch=hat, label=lab if (ci == 0 and fi == 0) else None)
                if h > 900: ax.text(pos, b + h / 2, f"{h:.0f}", ha="center", va="center", fontsize=10, color="white" if lab == "KV buffer" and f != "Q4_0" else INK)
                b += h
            ax.text(pos, tot + 250, f"{tot:.0f}", ha="center", fontsize=11, fontweight="bold")
            labels.append(f if f != "F16" else "F16"); x.append(pos); pos += 1
        ax.text(pos - 2, -2300, c + " context", ha="center", fontsize=13, fontweight="bold")
        pos += 0.8
    ax.set_xticks(x); ax.set_xticklabels([l.replace("_0", "") for l in labels], fontsize=12)
    ax.axhline(24576, color="#999999", ls=":", lw=1.8); ax.text(x[-1] + 0.4, 24900, "24 GiB card (RTX 3090)", ha="right", fontsize=10.5, color="#666")
    ax.set_ylim(0, 26500); ax.set_ylabel("process VRAM, MiB")
    ax.set_title("What fills the GPU: weights are constant, KV grows with context", fontsize=13)
    better(ax, "down", at=(0.66, 0.60), L=0.14)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.88), fontsize=10.5)
    fig.subplots_adjust(bottom=0.2, top=0.82, left=0.08, right=0.97)
    finish(fig, "fig18_vram_stacked_breakdown", "VRAM breakdown: weights + KV + compute buffer",
           "",
           note="Source: R1 (model 2375.91 MiB), R18/R28 (KV and compute buffers), R30 (totals). 'other' = total - weights - KV - compute (derived, 274-280 MiB). Compute buffer for quantized KV grows 24 -> 517 MiB.",
           top=0.88, bottom=0.14, tl=False)


# =========================================================================== 9. Absolute tok/s
def fig19_abs_toks():
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.6))
    ax = axs[0]
    _lines(ax, CTX_X, DEC)
    ax.errorbar(CTX_X, DEC["F16"], yerr=[np.array(DEC["F16"]) - DEC_P5, np.array(DEC_P95) - DEC["F16"]], fmt="none", ecolor=INK, capsize=4, lw=1.5)
    _endlabel(ax, CTX_X[-1], DEC["F16"][-1], f"F16 KV  {DEC['F16'][-1]}", color=INK)
    _endlabel(ax, CTX_X[-1], DEC["Q8_0"][-1], "Q8_0 / Q4_0 KV\n23.6 / 23.3", color=INK, dy=-4)
    ctx_axis(ax); ax.set_xlim(3000, 260000); ax.set_ylim(0, 185)
    ax.set_xlabel("filled context depth (tokens)"); ax.set_ylabel("decode tok/s (median, batch 1)")
    ax.set_title("Decode speed, absolute (R25; bars = p5-p95 of F16)", fontsize=12.5)
    better(ax, "up", at=(0.965, 0.55), L=0.2)
    ax = axs[1]
    pre = {"F16": PRE_F16, "Q8_0": [a * b for a, b in zip(PRE_F16, PRE_R["Q8_0"])], "Q4_0": [a * b for a, b in zip(PRE_F16, PRE_R["Q4_0"])]}
    _lines(ax, CTX_X, pre)
    _endlabel(ax, CTX_X[-1], pre["F16"][-1], "F16 KV", color=INK, dy=10)
    _endlabel(ax, CTX_X[-1], pre["Q8_0"][-1], "Q8_0 / Q4_0 KV", color=INK, dy=-12)
    ctx_axis(ax); ax.set_xlim(3000, 260000); ax.set_yscale("log"); ax.set_ylim(500, 8000)
    ax.set_xlabel("filled context depth (tokens)"); ax.set_ylabel("prefill tok/s (512-token chunk)")
    ax.set_title("Prefill speed, absolute (R25; Q8/Q4 = F16 x ratio, derived)", fontsize=12.5)
    better(ax, "up", at=(0.965, 0.55), L=0.2)
    fig.text(0.5, 0.075, "Decode: at 120K, Q8_0 and Q4_0 run at 0.63x of F16 although they read 47-72% fewer bytes. Prefill: quantized KV is 4-9% slower.",
             ha="center", fontsize=10.5, color=MUTED)
    finish(fig, "fig19_absolute_toks", "Tokens per second by KV format (RTX 3090, Qwen3-4B Q4_K_M)",
           "",
           note="Source: R25 (near-clean window, ABBA, 12 samples/cell; 3 at 120K for F16). Beyond 40,960 tokens quality is undefined without YaRN (speed is valid).", top=0.87, bottom=0.12)


# =========================================================================== 10. Bandwidth real vs peak
def fig20_bandwidth_vs_peak():
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axs[0]
    groups = [("32K\nshared GPU (R21)", [404, 208, None]), ("32K\nnear-clean (R25)", [646, 327, 246]), ("120K\nnear-clean (R25)", [775, 286, 177])]
    w = 0.26
    for gi, (g, vals) in enumerate(groups):
        for fi, (f, v) in enumerate(zip(("F16", "Q8_0", "Q4_0"), vals)):
            xp = gi + (fi - 1) * w
            if v is None:
                nm_text(ax, xp, 10, txt="n/m"); continue
            ax.bar(xp, v, w * 0.92, color=GR[f], edgecolor=INK, lw=0.8)
            ax.text(xp, v + 15, f"{v}\n{100 * v / 936:.0f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.axhline(936, color=G, lw=2.2, ls="--"); ax.text(2.55, 950, "peak 936 GB/s", ha="right", va="bottom", fontsize=11, color=G, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups], fontsize=11.5); ax.set_ylim(0, 1100); ax.set_xlim(-0.55, 2.6)
    ax.set_ylabel("implied DRAM traffic, GB/s\n(bytes of weights + KV per token x tok/s)")
    ax.set_title("llama.cpp decode: how much of the peak is used", fontsize=12.5)
    legend_fig_ax = [Patch(facecolor=GR[k], edgecolor=INK, label=LBL[k]) for k in ("F16", "Q8_0", "Q4_0")]
    ax.legend(handles=legend_fig_ax, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.82), fontsize=10.5)
    better(ax, "up", at=(0.50, 0.62), L=0.16)
    ax = axs[1]
    T = 32768
    fm = ["f16", "q8_0", "q4_0", "exp3"]; names = ["F16", "Q8_0", "Q4_0", "EXP3\n(ours)"]
    st_ = [pocb(f, T, "GBps_stored") for f in fm]; eq = [pocb(f, T, "GBps_f16_equiv") for f in fm]
    x = np.arange(4); cols = [GR["F16"], GR["Q8_0"], GR["Q4_0"], LL]
    ax.bar(x - 0.18, st_, 0.34, color=cols, edgecolor=INK, lw=0.8)
    ax.bar(x + 0.18, eq, 0.34, color=cols, edgecolor=INK, lw=0.8, alpha=0.45)
    for xi, a, b in zip(x, st_, eq):
        ax.text(xi - 0.18, a + 12, f"{a:.0f}", ha="center", fontsize=10); ax.text(xi + 0.18, b + 12, f"{b:.0f}", ha="center", fontsize=10)
    ax.axhline(936, color=G, lw=2.2, ls="--"); ax.text(3.45, 950, "peak 936", ha="right", va="bottom", fontsize=10.5, color=G, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1100); ax.set_ylabel("GB/s")
    ax.set_title("Our simple CUDA kernel, 32K tokens (R26)", fontsize=12.5)
    ax.text(0.02, 0.80, "dark = bytes really read per second\nlight = F16-equivalent data per second\nAll formats finish in about the same time:\nthe limit is per-row instruction work,\nnot bytes.", transform=ax.transAxes, va="top", fontsize=10, color=MUTED)
    better(ax, "up", at=(0.965, 0.52), L=0.16)
    finish(fig, "fig20_bandwidth_real_vs_peak", "Where does the memory bandwidth go?",
           "",
           note="Source: R21 (404 vs 208 GB/s), R25 (646/327/246 and 775/286/177 GB/s), R26 (GBps_stored / GBps_f16_equiv). Traffic = 2.49 GB weights + KV(d) per decoded token.", top=0.87, bottom=0.08)


# =========================================================================== 11. Lazy pool vs eager
def fig21_lazy_pool():
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.8), gridspec_kw={"width_ratios": [1.3, 1]})
    fills = [0, 1024, 4096, 16384, 32768]; eager = [4608] * 5; lazy = [0, 144, 576, 2304, 4608]
    ax = axs[0]
    fx = list(range(5))
    ax.plot(fx, eager, "o--", color=BASE, lw=3, ms=8, mec=INK, label="eager reservation (today's llama.cpp, -c 32768)")
    ax.plot(fx, lazy, "o-", color=LL, lw=3.4, ms=9, mec=INK, label="ours: lazily mapped pool (PoC-A)")
    ax.fill_between(fx, lazy, eager, color=BASE, alpha=0.18)
    ax.text(2.0, 3500, "reserved but EMPTY\n(wasted while the context is short)", fontsize=11.5, color="#666", ha="center")
    for f_, l_ in zip(fx[1:4], lazy[1:4]):
        ax.annotate(f"{l_} MiB", (f_, l_), textcoords="offset points", xytext=(-12, 14), ha="right", fontsize=10.5, color=LL, fontweight="bold")
    ax.set_xlabel("tokens actually in the context"); ax.set_ylabel("KV memory of the process, MiB (F16, Qwen3-4B)")
    ax.set_ylim(0, 5400); ax.set_xlim(-0.3, 4.3); ax.set_xticks(fx); ax.set_xticklabels(["0", "1K", "4K", "16K", "32K"])
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.98), fontsize=10.5)
    ax.set_title("Process memory follows the filled tokens exactly", fontsize=12.5)
    better(ax, "down", at=(0.965, 0.55), L=0.16)
    ax = axs[1]
    waste = [e - l for e, l in zip(eager, lazy)]
    ax.bar(range(5), waste, 0.7, color=BASE, edgecolor=INK)
    for i, v in enumerate(waste):
        ax.text(i, v + 60, f"{v}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels(["0", "1K", "4K", "16K", "32K"]); ax.set_xlabel("tokens in context"); ax.set_ylabel("MiB held but unused (eager - lazy, derived)")
    ax.set_ylim(0, 5400); ax.set_title("What eager reservation wastes", fontsize=12.5)
    better(ax, "down", at=(0.965, 0.55), L=0.16)
    finish(fig, "fig21_lazy_pool_vs_eager", "Lazy KV pool vs up-front reservation (PoC-A, CUDA VMM)",
           "",
           note="Source: R23 (artifacts/poc_20261021/lazy_pool.jsonl; DXGI per-process LOCAL; baseline 222 MiB subtracted). Standalone allocator PoC, NOT integrated into llama.cpp (FORK_PLAN F1 open). 2 MiB granularity: zero waste here only because fills are multiples of 1024 tokens.", top=0.87, bottom=0.08)
