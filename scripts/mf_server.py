# mf_server.py -- figures 22-23 (overview projections) and the SERVER-GPU block S1-S5.
# EVERYTHING here except the RTX 3090 fit and the measured 3090 points is an ANALYTIC PROJECTION (*), not a measurement.
# Model = scripts/projections_long.py (R31): s/token = a + s_k*depth fitted on RTX 3090 R25 data; memory part scales with
# bandwidth, excess decode time scales with scalar FP32 throughput. 'Ideal' = bandwidth-only upper bound (no decode cost).
# Lossy 'drop-4' row = the SAME model with the decode excess measured in poc_c.jsonl (derived HERE, not in RESULTS.md).
from mf_common import *
import re

STAR = "* analytic projection, to be validated on this hardware"
# card -> BW GB/s, FP16 tensor TFLOPS (dense), scalar FP32 TFLOPS (None = unknown), VRAM GB, confidence   (projections.py, R24)
CARDS = {"RTX 3090": (936, 71, 35.6, 24, "high"), "RTX 5090*": (1792, 210, 104.8, 32, "medium"), "L40S*": (864, 362, 91.6, 48, "low"),
         "RTX PRO 6000\nBlackwell*": (1792, 250, None, 96, "low"), "A100 80GB*": (2039, 312, 19.5, 80, "medium"),
         "H100 SXM*": (3350, 989, 67.0, 80, "medium"), "H200*": (4800, 989, 67.0, 141, "medium"), "B200*": (8000, 2250, None, 192, "low")}
CN = list(CARDS)
KV_TOK = 147456; ELEM_TOK = KV_TOK / 2; W_BYTES = 2.49e9
RATIO = {"f16": 1.0, "q8_0": 8.5 / 16, "q4_0": 4.5 / 16}
R_EXP3 = 0.9286; R_DROP4 = 0.7969
CTXS = [4096, 32768, 122880]; CTX_L = ["4K", "32K", "120K"]


def _fit():
    s = json.load(open(os.path.join(ART, "gpu_long_20261021", "summary.json")))
    fit = {}
    for kv in ("f16", "q8_0", "q4_0"):
        pts = [(r["depth"], 1.0 / r["median"]) for r in s["speed"] if r["kv"] == kv and r["test"] == "decode"]
        d, t = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
        sl, ic = np.polyfit(d, t, 1); fit[kv] = (ic, sl)
    return fit


FIT = _fit()
BW0, SC0 = 936e9, 35.6
E_KV = KV_TOK / (BW0 * FIT["f16"][1]); E_W = W_BYTES / (BW0 * FIT["f16"][0])
X = {kv: FIT[kv][1] - KV_TOK * RATIO[kv] / (BW0 * E_KV) for kv in RATIO}          # decode excess per token of depth, s
X_EXP3 = 2.01e-12 * ELEM_TOK                                                       # R26 clean
_T = 122880
X_DROP4 = (pocc("f16+drop4", _T, "median_ms") - R_DROP4 * pocc("f16", _T, "median_ms")) * 1e-3 / (pocc("f16", _T, "bytes") / 2) * ELEM_TOK   # derived from poc_c


def project(card, c):
    bw, tf, sc, vram, conf = CARDS[card]; bwr = bw * 1e9
    a = W_BYTES / (bwr * E_W)
    kvm = lambda r: KV_TOK * r / (bwr * E_KV)           # memory time per token of depth, s
    tf16 = a + c * kvm(1.0)
    out = {}
    scale = (SC0 / sc) if sc else None
    for k, r in (("q8", RATIO["q8_0"]), ("q4", RATIO["q4_0"])):
        out[k + "_ideal"] = tf16 / (a + c * kvm(r))
        out[k] = None if scale is None else tf16 / (a + c * (kvm(r) + max(X["q8_0" if k == "q8" else "q4_0"] * scale, 0)))
    for k, r, x in (("ll", R_EXP3, X_EXP3), ("ly", R_DROP4, X_DROP4)):
        out[k + "_ideal"] = tf16 / (a + c * kvm(r))
        if scale is None:
            out[k + "_nf"] = out[k + "_fu"] = None
        else:
            xs = x * scale
            out[k + "_nf"] = tf16 / (a + c * (kvm(r) + xs)); out[k + "_fu"] = tf16 / (a + c * max(kvm(r), xs))
    out["f16_toks"] = 1 / tf16
    return out


PROJ = {(cd, c): project(cd, c) for cd in CN for c in CTXS}


def _check_against_artifact():
    """sanity: reproduce artifacts/projections_long_20261021.md (R31) for the rows that exist there"""
    txt = open(os.path.join(ART, "projections_long_20261021.md"), encoding="utf-8").read()
    mp = {"RTX 3090": "RTX 3090", "RTX 5090*": "RTX 5090*", "L40S*": "L40S*", "A100 80GB*": "A100 80GB*", "H100 SXM*": "H100 SXM*", "H200*": "H200*"}
    worst = 0
    for card, art in mp.items():
        for c, cl in zip(CTXS, CTX_L):
            m = re.search(r"\| " + re.escape(art) + r" \| " + cl + r" \| (\d+) \| ([\d.]+)x \| ([\d.]+)x \| ([\d.]+)x / ([\d.]+)x", txt)
            if not m: continue
            p = PROJ[(card, c)]
            for got, want in ((p["q8"], m.group(2)), (p["q4"], m.group(3)), (p["ll_nf"], m.group(4)), (p["ll_fu"], m.group(5))):
                worst = max(worst, abs(got - float(want)))
    print("  projection model vs artifacts/projections_long (R31): max abs diff = %.3f" % worst)
    print("  derived lossy decode excess: %.3f ps/element" % (X_DROP4 / ELEM_TOK * 1e12))
    return worst


_check_against_artifact()


def _style_proj(ax):
    ax.set_xlabel(STAR, fontsize=9.5, color="#9a4a00", style="italic")


def _cols(cards): return [c.replace("\n", " ") for c in cards]


# ================================================================ 12. overview of projected speed ratios + F16 tok/s
def fig22_server_roofline_overview():
    fig, axs = plt.subplots(1, 3, figsize=(20, 7.2), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    cards = [c for c in CN if c not in ("RTX PRO 6000\nBlackwell*",)]
    x = np.arange(len(cards)); w = 0.2
    for ax, c, ttl in ((axs[0], 32768, "Decode speed vs F16 KV at 32K context"), (axs[1], 122880, "... at 120K context")):
        series = [("Q8_0 (llama.cpp-style kernel)", "q8", GR["Q8_0"]), ("Q4_0 (llama.cpp-style kernel)", "q4", GR["Q4_0"]),
                  ("ours lossless EXP3, non-fused", "ll_nf", LL), ("ours lossless EXP3, fused", "ll_fu", LLL)]
        for si, (lab, k, col) in enumerate(series):
            for xi, cd in enumerate(cards):
                v = PROJ[(cd, c)][k]; xp = xi + (si - 1.5) * w
                if v is None:
                    nm_text(ax, xp, 0.04, fs=8.5); continue
                ax.bar(xp, v, w * 0.92, color=col, edgecolor=INK, lw=0.5, alpha=1 if cd == "RTX 3090" else 0.6)
                ax.text(xp, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.8, rotation=90, va="bottom")
        ax.axhline(1.0, color=GR["F16"], ls="--", lw=1.8); ax.text(len(cards) - 0.55, 1.012, "F16 KV = 1.00", ha="right", fontsize=9.5, color=GR["F16"])
        ax.set_xticks(x); ax.set_xticklabels(_cols(cards), fontsize=9, rotation=0); ax.set_ylim(0, 1.32); ax.set_title(ttl, fontsize=12.5)
        better(ax, "up", at=(1.015, 0.40), L=0.16, lab_right=True); _style_proj(ax)
    axs[0].set_ylabel("speed relative to F16 KV (x)")
    ax = axs[2]
    for si, (cl, c, m) in enumerate((("4K", 4096, "o"), ("32K", 32768, "s"), ("120K", 122880, "^"))):
        v = [PROJ[(cd, c)]["f16_toks"] for cd in cards]
        ax.plot(range(len(cards)), v, marker=m, color=GR["F16"], mec=INK, ms=8, lw=2, alpha=0.4 + 0.3 * si, label=f"context {cl}")
    ax.set_xticks(range(len(cards))); ax.set_xticklabels(_cols(cards), fontsize=8, rotation=35, ha="right"); ax.set_yscale("log"); ax.set_ylabel("F16 KV decode tok/s (Qwen3-4B Q4_K_M, batch 1)")
    ax.set_title("Absolute F16 baseline speed", fontsize=12.5); ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    ax.text(0.97, 0.12, "B200: scalar figure unknown\n(tok/s needs it): shown only in S1", transform=ax.transAxes, ha="right", fontsize=8.8, color="#666")
    better(ax, "up", at=(1.02, 0.40), L=0.16, lab_right=True); _style_proj(ax)
    legend_fig(fig, [("Q8_0 KV", GR["Q8_0"]), ("Q4_0 KV", GR["Q4_0"]), ("Ours LOSSLESS non-fused", LL), ("Ours LOSSLESS fused", LLL), ("pale = analytic (*); solid = RTX 3090 fit", "#dddddd")], y=0.02)
    finish(fig, "fig22_server_roofline_projections", "Roofline projections for other GPUs (all *)",
           "",
           note="Source: R24, R31 (artifacts/projections_long_20261021.md; model reproduced in mf_server.py, max diff vs file printed at run time). 'Same kernel design on that GPU', NOT a claim about FP8-KV kernels in server stacks. L40S confidence low; others medium; B200 / PRO 6000: n/m.",
           top=0.86, bottom=0.12)


# ================================================================ 13. capacity by card
def _cap_tokens(card, r):
    vram = CARDS[card][3]
    return (vram * 1e9 - W_BYTES - 1.5e9) / (KV_TOK * r)


def fig23_capacity_by_card():
    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    cards = CN; x = np.arange(len(cards)); w = 0.2
    series = [("F16 (base)", 1.0, GR["F16"], None), ("Q8_0 (base)", RATIO["q8_0"], GR["Q8_0"], None), ("Q4_0 (base)", RATIO["q4_0"], GR["Q4_0"], None), ("F16 + ours lossless EXP3 (x0.929)", R_EXP3, LL, None)]
    for si, (lab, r, col, h) in enumerate(series):
        v = [_cap_tokens(c, r) / 1024 for c in cards]
        ax.bar(x + (si - 1.5) * w, v, w * 0.92, color=col, edgecolor=INK, lw=0.5, label=lab, alpha=1)
        for xi, vv in zip(x + (si - 1.5) * w, v):
            ax.text(xi, vv * 1.08, f"{vv:.0f}", ha="center", fontsize=7.6, rotation=90, va="bottom")
    ax.set_yscale("log"); ax.set_ylim(40, 15000)
    ax.axhline(40, color="#999")
    ax.axhline(40960 / 1024, color="#bbbbbb", ls=":", label="Qwen3-4B trained context 40,960"); ax.axhline(131072 / 1024, color="#999999", ls="-.", lw=1.2, label="131,072 with YaRN [verify]")
    ax.set_xticks(x); ax.set_xticklabels(_cols(cards), fontsize=10.5); ax.set_ylabel("max KV context, thousand tokens (log), batch 1")
    ax.set_title("How many tokens of KV fit on each card (Qwen3-4B Q4_K_M, 147,456 B/token in F16)", fontsize=12.5)
    ax.legend(frameon=False, loc="upper left", fontsize=10.5)
    better(ax, "up", at=(1.012, 0.40), L=0.16, lab_right=True)
    _style_proj(ax)
    finish(fig, "fig23_context_capacity_by_card", "Context capacity by GPU: base vs base + our compression (*)",
           "",
           note="Capacity = (VRAM - weights 2.49 GB - 1.5 GB) / (bytes per token x format ratio): R31 formula (artifacts/projections_long_20261021.md rows reproduced; ours lossless = F16 x 1.077 derived from ratio 0.9286, R26). Ignores the compute-buffer growth of quantized KV (R28). Analytic.",
           top=0.87, bottom=0.08)


# ================================================================ S1  speedup by card (3 contexts) with uncertainty
def figS1_server_speedup_ranges():
    fig, axs = plt.subplots(1, 3, figsize=(23, 8), sharey=True)
    series = [("Q8_0 KV", "q8", "q8_ideal", GR["Q8_0"], None), ("Q4_0 KV", "q4", "q4_ideal", GR["Q4_0"], None),
              ("ours lossless EXP3", "ll_nf", "ll_ideal", LL, "ll_fu"), ("ours lossy drop-4", "ly_nf", "ly_ideal", LY, "ly_fu")]
    x = np.arange(len(CN)); w = 0.2
    for ax, c, cl in zip(axs, CTXS, CTX_L):
        for si, (lab, k, ik, col, fk) in enumerate(series):
            for xi, cd in enumerate(CN):
                p = PROJ[(cd, c)]; xp = xi + (si - 1.5) * w
                lo, hi = p[k], p[ik]
                if lo is None:     # calibrated figure unknown: only the bandwidth-only bound
                    ax.bar(xp, hi, w * 0.92, color="white", edgecolor=col, lw=1.5, ls=":", hatch=None)
                    ax.text(xp, 0.03, "n/m", ha="center", fontsize=7.5, rotation=90, va="bottom", color="#666")
                    continue
                ax.bar(xp, lo, w * 0.92, color=col, edgecolor=INK, lw=0.5, alpha=1 if cd == "RTX 3090" else 0.62)
                ax.plot([xp, xp], [lo, hi], color="#777777", lw=1.0); ax.plot([xp - w * 0.2, xp + w * 0.2], [hi, hi], color="#777777", lw=1.0)
                if fk:
                    ax.plot([xp - w * 0.42, xp + w * 0.42], [p[fk], p[fk]], color="white", lw=2.5); ax.plot([xp - w * 0.42, xp + w * 0.42], [p[fk], p[fk]], color=INK, lw=1, ls="-")
                pass
        for xi in range(len(CN)):
            ax.scatter([], [])
        # measured RTX 3090 decode ratios (R25)
        meas = {4096: (0.883, 0.889), 32768: (0.732, 0.725), 122880: (0.629, 0.622)}[c]
        ax.scatter([-1.5 * w, -0.5 * w], list(meas), marker="X", s=70, color="white", edgecolor=INK, zorder=8, linewidths=1.2)
        ax.axhline(1.0, color=GR["F16"], ls="--", lw=1.8)
        ax.set_xticks(x); ax.set_xticklabels(_cols(CN), fontsize=8.8, rotation=30, ha="right"); ax.set_ylim(0, 2.45); ax.set_title(f"context {cl}", fontsize=14)
        better(ax, "up", at=(0.97, 0.45), L=0.14); _style_proj(ax)
    axs[0].set_ylabel("expected decode speed vs F16 KV (x)")
    legend_fig(fig, [("Q8_0 KV (llama.cpp-style)", GR["Q8_0"]), ("Q4_0 KV (llama.cpp-style)", GR["Q4_0"]), ("Ours LOSSLESS EXP3", LL), ("Ours LOSSY drop-4 (derived here)", LY)], y=0.065, ncol=4)
    fig.text(0.5, 0.032, "Exact values: see figS4 (traffic light).  Bar = calibrated projection (non-fused decoder = this kernel design on that card).  Whisker top = bandwidth-only upper bound (no decode cost; what a perfect fused kernel could approach).  "
             "White tick = fused/overlapped (ours).  X = measured RTX 3090 Q8_0 / Q4_0 (R25).  Dotted outline = only the upper bound (scalar figure unknown: B200, RTX PRO 6000).",
             ha="center", fontsize=9.3, color=MUTED)
    finish(fig, "figS1_server_speedup_by_card_context", "Expected decode speed vs base on server GPUs (*)",
           "",
           note="* analytic projection, to be validated on this hardware. Confidence: 3090 measured basis; 5090/A100/H100/H200 medium; L40S/PRO 6000/B200 low. Source: R24, R31, projections_long.py; lossy row = same model with decode excess from poc_c.jsonl (new, not in RESULTS.md).",
           top=0.88, bottom=0.12)


# ================================================================ S2 capacity and max batch
def figS2_server_capacity_batch():
    fig, axs = plt.subplots(2, 1, figsize=(16, 13))
    rows = [("F16 (base)", 1.0, "base", 0), ("Q8_0 (base)", RATIO["q8_0"], "base", 0), ("Q4_0 (base)", RATIO["q4_0"], "base", 0),
            ("F16 + EXP3 lossless (GPU format)", R_EXP3, "ll", 0), ("F16 + planes+LZMA lossless (CPU only)", 0.847, "ll", 1), ("F16 + drop-4 lossy", R_DROP4, "ly", 0),
            ("Q8_0 + planes+LZMA lossless (CPU only)", RATIO["q8_0"] * 0.9115, "ll", 1), ("Q4_0 + planes+LZMA lossless (CPU only)", RATIO["q4_0"] * 0.8295, "ll", 1)]
    for ai, (ax, kind) in enumerate(zip(axs, ["tokens", "batch"])):
        M = [[_cap_tokens(cd, r) / (1024 if kind == "tokens" else 32768) for cd in CN] for _, r, _, _ in rows]
        M = np.array(M)
        norm = Normalize(0, np.log10(M.max()))
        cm = LinearSegmentedColormap.from_list("cap", ["#ffffff", "#bfe6d9", LL])
        for i in range(len(rows)):
            for j in range(len(CN)):
                v = M[i, j]
                c = cm(np.log10(max(v, 1)) / np.log10(M.max()))
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc=c, ec="white", lw=2, hatch="////" if rows[i][3] else None))
                txt = f"{v:.0f}K" if kind == "tokens" else f"{int(v)}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=11, fontweight="bold", color=INK, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.6) if rows[i][3] else None)
        ax.set_xlim(-.5, len(CN) - .5); ax.set_ylim(len(rows) - .5, -.5)
        ax.set_xticks(range(len(CN))); ax.set_xticklabels([f"{c}\n{CARDS[c][3]} GB" for c in CN], fontsize=10); ax.xaxis.tick_top()
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
        for t, r in zip(ax.get_yticklabels(), rows): t.set_color({"base": "#555", "ll": LL, "ly": LY}[r[2]])
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title("Max KV context, thousand tokens (batch 1)" if kind == "tokens" else "Max concurrent sequences of 32K context (= tokens / 32,768, rounded down)", fontsize=13, pad=62)
        better(ax, "right", at=(0.30, -0.075), L=0.2, label="better = more", fs=10.5)
        ax.set_xlabel(STAR, fontsize=9.5, color="#9a4a00", style="italic", labelpad=34)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.82, bottom=0.07, hspace=0.42)
    finish(fig, "figS2_server_capacity_and_max_batch", "Capacity and batch on server GPUs: base vs base + our compression (*)",
           "",
           note="Capacity = (VRAM - 2.49 GB weights - 1.5 GB) / (147,456 B x format ratio), Qwen3-4B (R31 formula; ratios: EXP3 0.9286 R26, LZMA planes F16 0.847 R10, Q8/Q4 + LZMA 0.9115 / 0.8295 = K,V mean of R33, drop-4 0.797 R33). Compute-buffer growth of quantized KV (R28) ignored. Batch = KV only.",
           top=0.86, bottom=0.04, tl=False)


# ================================================================ S3 ridge point and scalar budget
def figS3_ridge_scalar_budget():
    fig, axs = plt.subplots(1, 2, figsize=(18, 7.4))
    ridge = [CARDS[c][1] * 1e12 / (CARDS[c][0] * 1e9) for c in CN]
    scal = [None if CARDS[c][2] is None else CARDS[c][2] * 1e12 / (CARDS[c][0] * 1e9) for c in CN]
    x = np.arange(len(CN))
    ax = axs[0]
    ax.bar(x, ridge, 0.65, color=[GR["F16"] if c == "RTX 3090" else BASE for c in CN], edgecolor=INK, lw=0.6, alpha=1)
    for xi, v in zip(x, ridge): ax.text(xi, v + 8, f"{v:.0f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(_cols(CN), fontsize=9, rotation=30, ha="right"); ax.set_ylabel("FP16 tensor FLOP per byte (ridge point)")
    ax.set_ylim(0, 560); ax.set_title("Ridge point: above it a kernel is memory-bound", fontsize=13)
    ax.text(0.02, 0.97, "Higher ridge = decode is even more memory-bound,\nso fewer bytes SHOULD help more.", transform=ax.transAxes, va="top", fontsize=10.5, color=MUTED)
    better(ax, "up", at=(1.015, 0.40), L=0.16, lab_right=True, label="better")
    _style_proj(ax)
    ax = axs[1]
    cols = [GR["F16"] if c == "RTX 3090" else BASE for c in CN]
    for xi, v, col in zip(x, scal, cols):
        if v is None:
            nm_text(ax, xi, 2, rot=0, fs=11, txt="n/m\n(scalar\nfigure\nunknown)"); continue
        ax.bar(xi, v, 0.65, color=col, edgecolor=INK, lw=0.6); ax.text(xi, v + 2.5, f"{v:.0f}", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(scal[0], color=LL, lw=2.2, ls="--", label="RTX 3090 = 38 FLOP/B: measured non-fused EXP3 0.65-0.73x,\nfused model 1.04-1.06x")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.70), fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(_cols(CN), fontsize=9, rotation=30, ha="right"); ax.set_ylabel("scalar FP32 FLOP per byte (budget of a bit-level decoder)")
    ax.set_ylim(0, 155); ax.set_title("Scalar budget per byte: where naive decompression loses", fontsize=13)
    ax.text(0.02, 0.97, "A100, H100, H200 have 2-4x FEWER scalar operations per byte than the 3090:\na scalar bit-level decoder is projected to lose there even when fused.\nOnly tensor-core or fixed-function decompression could change that [verify].", transform=ax.transAxes, va="top", fontsize=10, color="#9a4a00")
    better(ax, "up", at=(1.015, 0.40), L=0.16, lab_right=True, label="better")
    _style_proj(ax)
    finish(fig, "figS3_ridge_point_and_scalar_budget", "Why server GPUs differ: ridge point and scalar budget per byte (*)",
           "",
           note="Source: R24 / PAPER section 8 (approximate public specs; FLOP/B = TFLOPS / bandwidth; scalar FP32 TFLOPS from projections.py; '?' = not known with confidence: B200, RTX PRO 6000 Blackwell). L40S tensor figure may be the sparse one (low confidence).",
           top=0.86, bottom=0.12)


# ================================================================ S4 traffic-light table
def _tl(v, ideal=None):
    if v is None: return ("n/m", "#d9d9d9", INK)
    if v >= 1.03: return (f"{v:.2f}x", LL, "white")
    if v >= 0.95: return (f"{v:.2f}x", AMBER, INK)
    return (f"{v:.2f}x", LY, "white")


def figS4_server_traffic_light():
    fig = plt.figure(figsize=(21, 9.2))
    gs = fig.add_gridspec(1, 2, wspace=0.08)
    cols = [("Q8_0\nllama.cpp", "q8"), ("Q4_0\nllama.cpp", "q4"), ("ours lossless\nnon-fused", "ll_nf"), ("ours lossless\nfused", "ll_fu"), ("ours lossy\nnon-fused", "ly_nf"), ("ours lossy\nfused", "ly_fu")]
    for gi, (c, cl) in enumerate(((32768, "32K"), (122880, "120K"))):
        ax = fig.add_subplot(gs[0, gi])
        for i, cd in enumerate(CN):
            p = PROJ[(cd, c)]
            for j, (_, k) in enumerate(cols):
                t, bg, fg = _tl(p[k])
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc=bg, ec="white", lw=3, hatch="...." if p[k] is None else None))
                ax.text(j, i, t, ha="center", va="center", fontsize=12, fontweight="bold", color=fg)
            # capacity column
            ax.add_patch(Rectangle((len(cols) - .5 + 0.15, i - .5), 1.1, 1, fc=LL, ec="white", lw=3, alpha=0.85))
            ax.text(len(cols) + 0.2, i, "1.07 / 1.88\n/ 3.56x", ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")
        ax.set_xlim(-.5, len(cols) + 0.85); ax.set_ylim(len(CN) - .5, -.5)
        ax.set_xticks(list(range(len(cols))) + [len(cols) + 0.2]); ax.set_xticklabels([c_[0] for c_ in cols] + ["KV capacity\nlossless / Q8 / Q4"], fontsize=9.8); ax.xaxis.tick_top()
        if gi == 0:
            ax.set_yticks(range(len(CN))); ax.set_yticklabels([f"{c}   [{CARDS[c][4]}]" for c in CN], fontsize=10.5)
        else:
            ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(length=0); ax.set_title(f"context {cl}", fontsize=14, pad=48)
    fig.text(0.5, 0.925, "", ha="center")
    legend_fig(fig, [("gain expected (>= 1.03x)", LL), ("doubtful (0.95-1.03x)", AMBER), ("loss expected (< 0.95x)", LY), ("n/m: scalar figure unknown, not projected", "#d9d9d9", "....")], y=0.075, ncol=4)
    fig.text(0.5, 0.045, "Speed vs F16 KV on the same card.  Capacity always gains (projected ~1.07x lossless EXP3, 1.88x Q8_0, 3.56x Q4_0) whatever the kernel.   [..] = projection confidence.   Lossy = F16 drop-4 bits.",
             ha="center", fontsize=9.8, color=MUTED)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.73, bottom=0.14)
    finish(fig, "figS4_server_gain_or_loss_traffic_light", "Where we expect a gain, a doubt, or a loss (*)",
           "",
           note="* analytic projection, to be validated on this hardware. Same model as figS1 (R31 calibration; Q8_0/Q4_0 = llama.cpp-style scalar kernels, NOT FP8-KV kernels of server stacks). Thresholds are ours: >= 1.03x green, 0.95-1.03x amber, < 0.95x orange.",
           top=0.85, bottom=0.03, tl=False)


# ================================================================ S5 checklist
def figS5_hardware_checklist():
    fig, ax = plt.subplots(figsize=(19, 8.6))
    cols = ["PoC-B read\nmicrobenchmark\n(kvread_bench.cu)", "ABBA decode series\nllama.cpp F16/Q8/Q4\n4K / 32K / 120K", "check specs:\ntensor + scalar FP32\nTFLOPS", "FP8 / INT8 KV in a\nserver stack (vLLM,\nTensorRT-LLM) [verify]", "hardware decompression\nengine (Blackwell)\n[verify]"]
    # status: 'done' measured on 3090; 'todo' named in PAPER section 8 'What to check'; '' not named
    st_ = {"RTX 3090": ["done", "done", "", "", ""], "RTX 5090*": ["todo", "todo", "", "", ""], "L40S*": ["todo", "", "todo", "", ""],
           "RTX PRO 6000\nBlackwell*": ["todo", "", "todo", "", ""], "A100 80GB*": ["todo", "", "", "todo", ""], "H100 SXM*": ["todo", "", "", "todo", ""],
           "H200*": ["todo", "", "", "todo", ""], "B200*": ["todo", "", "", "", "todo"]}
    for i, cd in enumerate(CN):
        for j in range(len(cols)):
            s = st_[cd][j]
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc="#f7f7f7", ec="white", lw=3))
            if s == "done":
                ax.add_patch(Circle((j, i), 0.28, fc=LL, ec="white")); ax.text(j, i, "done", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
            elif s == "todo":
                ax.add_patch(Circle((j, i), 0.28, fc="white", ec=LY, lw=3)); ax.text(j, i, "TO DO", ha="center", va="center", color=LY, fontsize=9, fontweight="bold")
        ax.add_patch(Rectangle((len(cols) - .5, i - .5), 1.6, 1, fc="#f7f7f7", ec="white", lw=3))
        sc = CARDS[cd][2]
        info = f"confidence: {CARDS[cd][4]}" + ("" if sc else "\nscalar figure: ?")
        ax.text(len(cols) + 0.3, i, info, ha="center", va="center", fontsize=9.5, color=INK if CARDS[cd][4] != "low" else "#9a4a00")
    ax.set_xlim(-.5, len(cols) + 1.1); ax.set_ylim(len(CN) - .5, -.5)
    ax.set_xticks(list(range(len(cols))) + [len(cols) + 0.3]); ax.set_xticklabels(cols + ["projection\nconfidence"], fontsize=9.8); ax.xaxis.tick_top()
    ax.set_yticks(range(len(CN))); ax.set_yticklabels([c.replace("\n", " ") for c in CN], fontsize=11.5)
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.tick_params(length=0)
    better(ax, "up", at=(1.02, 0.35), L=0.2, lab_right=True, label="better = more verified (done)")
    legend_fig(fig, [("done on the RTX 3090 (R22-R26)", LL), ("named in PAPER section 8 'What to check' - to do on that card (orange ring)", "#ffe3d0")], y=0.045)
    finish(fig, "figS5_hardware_validation_checklist", "What must be checked on the hardware (*)",
           "",
           note="Source: PAPER_V1_EN.md section 8 table, column 'What to check' (not edited). Each projection above stays '* analytic projection, to be validated on this hardware' until its row turns green.",
           top=0.84, bottom=0.08)
