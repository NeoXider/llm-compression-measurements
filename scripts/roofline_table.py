"""Analytic cards x scenarios table (NOT measurements). Batch decode roofline with a memory-fit check.

Per decode step: weights read once, KV read once per sequence (B sequences of length ctx).
t = max(bytes / (BW * eff), 2 * params * B / (FP16 dense TFLOPS * 0.5)); tokens/s = B / t.
Memory fit: weights + KV(B, ctx) + 1.5 GB overhead <= VRAM. Card figures are approximate public
specs (BW GB/s, FP16 dense TFLOPS, VRAM GB) taken from scripts/roofline_server.py; L40S TFLOPS uncertain.
Model classes (GQA, 8 KV heads, head dim 128): 8B (32 layers), 70B (80 layers); weights at Q4_K_M
~4.85 bpw. Lossless KV ratio = measured fixed-rate format 0.907 (R11) and best variable-rate 0.847 (R10);
lossless weights ratio 0.934 (R4, order-0 ceiling, optimistic). eff = 0.5 (3090 measured 0.34-0.44, R15).
"""
from __future__ import annotations

GB = 1e9
CARDS = {  # bw, fp16 dense TFLOPS, vram
    "RTX 3090": (936, 71, 24), "RTX 5090": (1792, 210, 32), "L40S": (864, 362, 48),
    "RTX PRO 6000 Blackwell": (1792, 250, 96), "A100 80GB": (2039, 312, 80),
    "H100 SXM": (3350, 989, 80), "H200": (4800, 989, 141), "B200": (8000, 2250, 192)}
MODELS = {"8B": (8e9, 32), "70B": (70e9, 80)}
BPW = 4.85
EFF = 0.5
OVER = 1.5 * GB


def kv_tok(layers):
    return 2 * layers * 8 * 128 * 2  # bytes per token, F16


def step(card, model, B, ctx, rw=1.0, kv_scale=1.0):
    bw, tf, vram = CARDS[card]
    p, layers = MODELS[model]
    w = p * BPW / 8 * rw
    kv = kv_tok(layers) * ctx * B * kv_scale
    fits = w + kv + OVER <= vram * GB
    tm = (w + kv) / (bw * GB * EFF)
    tc = 2 * p * B / (tf * 1e12 * 0.5)
    return B / max(tm, tc), fits, tm >= tc


VARIANTS = {"KV lossless 0.907": (1, 0.907), "KV Q8_0": (1, 8.5 / 16), "KV Q4_0": (1, 4.5 / 16),
            "W 0.934 + KV 0.907": (0.934, 0.907)}
SCEN = [("B=1, 8K", 1, 8192), ("B=1, 128K", 1, 131072), ("B=16, 32K", 16, 32768), ("B=64, 8K", 64, 8192)]


def cell(card, model, B, ctx):
    base, fits, mb = step(card, model, B, ctx)
    parts = []
    for name, (rw, ks) in VARIANTS.items():
        s, f, _ = step(card, model, B, ctx, rw, ks)
        if fits:
            parts.append(f"{s / base:.2f}x")
        else:
            parts.append("fits" if f else "no")
    head = f"{base:.0f} tok/s{'' if mb else ' (C)'}" if fits else "F16 KV: does not fit"
    return head + " -> " + " / ".join(parts)


def capacity(card, model):
    bw, tf, vram = CARDS[card]
    p, layers = MODELS[model]
    free = vram * GB - p * BPW / 8 - OVER
    if free <= 0:
        return "weights do not fit"
    return " / ".join(f"{int(free / (kv_tok(layers) * s)) // 1000}K" for s in (1, 0.907, 8.5 / 16, 4.5 / 16))


def main():
    print("Cell format: baseline (F16 KV, Q4_K_M weights) total tok/s -> speedup for "
          + " / ".join(VARIANTS) + "; '(C)' = compute-bound; 'fits'/'no' = whether the variant fits when F16 does not.\n")
    for model in MODELS:
        print(f"### {model}-class model (Q4_K_M weights {MODELS[model][0] * BPW / 8 / GB:.1f} GB)\n")
        print("| Card | BW GB/s | VRAM GB | " + " | ".join(s[0] for s in SCEN)
              + " | Max KV tokens F16 / lossless 0.907 / Q8 / Q4 |")
        print("|---|---|---|" + "---|" * len(SCEN) + "---|")
        for card, (bw, tf, vram) in CARDS.items():
            print(f"| {card} | {bw} | {vram} | " + " | ".join(cell(card, model, B, c) for _, B, c in SCEN)
                  + f" | {capacity(card, model)} |")
        print()


if __name__ == "__main__":
    main()
