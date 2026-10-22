"""Figures for E2 (llama.cpp PRs vs their bases) and E1/E4 (lossless headroom of F16 vs BF16 KV).

Reads artifacts/window2_20261022/e2/summary.json and artifacts/e1_context_20261022/captures_{4k,30k}.json.
Writes figures/fig50_e2_llamacpp_prs.png and figures/fig51_e1_e4_lossless_kv.png. Run from the paper root.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
A = Path("artifacts")
Path("figures").mkdir(exist_ok=True)

# ---- E2: quantized / F16 decode within each build, base vs PR #30202 (and #30792) ----
rows = json.loads((A / "window2_20261022/e2/summary.json").read_text(encoding="utf-8"))
med = {}
for r in rows:
    if r["test"] == "decode":
        med[(r["pr"], r["kv"], r["depth"], "base")] = r["base"]
        med[(r["pr"], r["kv"], r["depth"], "pr")] = r["prv"]
depths = sorted({r["depth"] for r in rows})
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, pr in zip(axes, ("30202", "30792")):
    for kv, col in (("q8_0", "#2a6fb0"), ("q4_0", "#c0612b")):
        for which, ls, mk in (("base", "--", "o"), ("pr", "-", "s")):
            y = [med[(pr, kv, d, which)] / med[(pr, "f16", d, which)] for d in depths]
            ax.plot([d / 1024 for d in depths], y, ls, marker=mk, color=col,
                    label=f"{kv} KV, {'base build' if which == 'base' else 'PR #' + pr}")
            # label only the PR #30202 lines and the q8_0 base line (q4_0 base is within 0.01 of it)
            if pr == "30202" and (which == "pr" or kv == "q8_0"):
                for d, v in zip(depths, y):
                    ax.annotate(f"{v:.2f}", (d / 1024, v), textcoords="offset points",
                                xytext=(6, -12 if which == "base" else 5), fontsize=7,
                                color=col if which == "pr" else "black")
    if pr == "30792":
        ax.text(0.5, 0.15, "PR and base within ±1% in every cell", transform=ax.transAxes, ha="center", fontsize=8)
    ax.axhline(1.0, color="grey", lw=1)
    ax.text(depths[0] / 1024, 1.005, "F16 KV (same build)", fontsize=8, color="grey")
    ax.set_xscale("log", base=2)
    ax.set_xticks([d / 1024 for d in depths], [f"{d // 1024}K" for d in depths])
    ax.set_xlabel("filled context (depth)")
    ax.set_title(f"PR #{pr} vs its base commit")
    ax.annotate("better", xy=(0.92, 0.95), xytext=(0.92, 0.78), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->"), ha="center", fontsize=8)
axes[0].set_ylabel("decode speed relative to F16 KV")
axes[0].set_ylim(0.55, 1.05)
axes[0].legend(fontsize=7, loc="lower left")
fig.suptitle("E2: quantized-KV decode on RTX 3090, Qwen3-4B Q4_K_M, ABBA, clean window (R39)", fontsize=10)
fig.tight_layout()
fig.savefig("figures/fig50_e2_llamacpp_prs.png")
plt.close(fig)

# ---- E1/E4: lossless ratio vs 16-bit base for F16 and BF16 KV, three texts, two lengths ----
d4 = json.loads((A / "e1_context_20261022/captures_4k.json").read_text(encoding="utf-8"))
d30 = json.loads((A / "e1_context_20261022/captures_30k.json").read_text(encoding="utf-8"))
groups = []
for text in ("code", "en", "ru"):
    groups.append((f"{text}\nF16 ~4K", d4[f"{text}_f16_4096.bin"]["results"]))
    groups.append((f"{text}\nF16 ~30K", d30[f"{text}_f16_30720.bin"]["results"]))
    groups.append((f"{text}\nBF16 ~4K", d4[f"{text}_bf16_4096.bin"]["results"]))
fig, ax = plt.subplots(figsize=(11, 4))
x = range(len(groups))
w = 0.2
series = (("K", "best_ratio_vs_f16", "K context estimate", "#2a6fb0", 0.6),
          ("K", "trace_zstd19_G16", "K TRACE-like codec", "#2a6fb0", 1.0),
          ("V", "best_ratio_vs_f16", "V context estimate", "#c0612b", 0.6),
          ("V", "trace_zstd19_G16", "V TRACE-like codec", "#c0612b", 1.0))
for i, (c, key, lab, col, alpha) in enumerate(series):
    ax.bar([xi + (i - 1.5) * w for xi in x], [g[1][c][key] for g in groups], w, color=col, alpha=alpha, label=lab)
ax.axhline(1.0, color="grey", lw=1)
ax.text(-0.4, 1.01, "16-bit base", fontsize=8, color="grey")
ax.axhline(1 / 1.81, color="black", lw=1, ls=":")
ax.text(len(groups) - 1.6, 1 / 1.81 - 0.035, "TRACE 1.81x (BF16, Llama-3.1-8B, their claim)", fontsize=7, ha="right")
ax.set_xticks(list(x), [g[0] for g in groups], fontsize=7)
ax.set_ylabel("size relative to 16-bit KV (lower is better)")
ax.set_ylim(0.45, 1.08)
ax.legend(fontsize=7, ncol=4, loc="upper center")
ax.set_title("E1/E4: lossless headroom of Qwen3-4B KV, F16 vs BF16 (R41)", fontsize=10)
fig.tight_layout()
fig.savefig("figures/fig51_e1_e4_lossless_kv.png")
plt.close(fig)
print("ok")
