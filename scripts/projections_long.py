"""Calibrated projections (*) at contexts 4K / 32K / 120K for Qwen3-4B-Q4_K_M-class decode (batch 1).

Calibration (RTX 3090, measured, artifacts/gpu_long_20261021/summary.json):
  per KV type k, fit seconds-per-token = a_k + s_k * depth (least squares over measured depths).
  a (depth-independent part: weights + overhead) is taken from F16.
  The F16 slope s_f16 is treated as memory time: s_f16 = kv_bytes_f16_per_token_per_depth / (BW * e_kv).
  For a quantized type, the excess over its own memory time is decode (ALU) time:
      s_k = bytes_k / (BW * e_kv) + x_k   ->   x_k = s_k - bytes_k / (BW * e_kv)
Projection to card C:
  a_C = a * BW_3090 / BW_C                      (weights read at the same efficiency)
  s_k,C = bytes_k / (BW_C * e_kv) + x_k * scalar_3090 / scalar_C   (non-fused: decode is added; as measured)
  lossless EXP3: bytes ratio 0.9286 and decode 2.01 ps/elt (clean PoC-B, T=122880) in the same way; 'fused' = max(memory, decode).
Capacity: max context with (VRAM - weights - 1.5 GB) for each KV type; Qwen3-4B's native context is 40,960.
Rows other than the RTX 3090 are analytic projections (*), to be validated on that hardware.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CARDS = {  # BW GB/s, scalar FP32 TFLOPS (None = unknown), VRAM GB, confidence
    "RTX 3090": (936, 35.6, 24, "measured basis"), "RTX 5090": (1792, 104.8, 32, "medium"),
    "L40S": (864, 91.6, 48, "low"), "A100 80GB": (2039, 19.5, 80, "medium"),
    "H100 SXM": (3350, 67.0, 80, "medium"), "H200": (4800, 67.0, 141, "medium"), "B200": (8000, None, 192, "low")}
KV_F16_TOK = 147456  # bytes per token, Qwen3-4B
BYTES = {"f16": 1.0, "q8_0": 8.5 / 16, "q4_0": 4.5 / 16}
ELEM_PER_TOKEN = KV_F16_TOK / 2
W_BYTES = 2.49e9
CTXS = [4096, 32768, 122880]


def main():
    s = json.loads(Path(sys.argv[1]).read_text())
    fit = {}
    for kv in ("f16", "q8_0", "q4_0"):
        pts = [(r["depth"], 1.0 / r["median"]) for r in s["speed"] if r["kv"] == kv and r["test"] == "decode"]
        d, t = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
        slope, inter = np.polyfit(d, t, 1)
        fit[kv] = {"a": inter, "s": slope, "depths": d.tolist()}
    bw0, sc0 = 936e9, 35.6
    e_kv = KV_F16_TOK / (bw0 * fit["f16"]["s"])  # implied efficiency of KV reads in F16 path
    e_w = W_BYTES / (bw0 * fit["f16"]["a"])
    x = {kv: fit[kv]["s"] - KV_F16_TOK * BYTES[kv] / (bw0 * e_kv) for kv in BYTES}
    x_exp3 = 2.01e-12 * ELEM_PER_TOKEN  # clean PoC-B, T=122880
    lines = [f"Calibration (3090): e_w = {e_w:.2f}, e_kv(F16) = {e_kv:.2f}; decode excess per token of depth: "
             + ", ".join(f"{k} {v * 1e9:.2f} ns" for k, v in x.items()) + f"; EXP3 {x_exp3 * 1e9:.2f} ns (non-fused)", ""]
    hdr = "| Card | ctx | F16 tok/s | Q8_0 / F16 | Q4_0 / F16 | EXP3 lossless non-fused / fused | F16 fits? | max ctx F16 / Q8 / Q4 | confidence |"
    lines += [hdr, "|---|---|---|---|---|---|---|---|---|"]
    for card, (bw, sc, vram, conf) in CARDS.items():
        bwr = bw * 1e9
        a = W_BYTES / (bwr * e_w)
        free = vram * 1e9 - W_BYTES - 1.5e9
        cap = [int(free / (KV_F16_TOK * BYTES[k])) for k in ("f16", "q8_0", "q4_0")]
        for c in CTXS:
            def t(kv):
                xs = x[kv] * (sc0 / sc) if sc else None
                if xs is None:
                    return None
                return a + c * (KV_F16_TOK * BYTES[kv] / (bwr * e_kv) + max(xs, 0))
            tf = t("f16")
            mem_e = 0.9286 * KV_F16_TOK / (bwr * e_kv)
            dec_e = x_exp3 * (sc0 / sc) if sc else None
            if tf is None:
                cells = ("?", "?", "?", "?")
            else:
                nf = a + c * (mem_e + dec_e)
                fu = a + c * max(mem_e, dec_e)
                cells = (f"{1 / tf:.0f}", f"{tf / t('q8_0'):.2f}x", f"{tf / t('q4_0'):.2f}x", f"{tf / nf:.2f}x / {tf / fu:.2f}x")
            fits = "yes" if KV_F16_TOK * c <= free else "no"
            star = "" if card == "RTX 3090" else "*"
            lines.append(f"| {card}{star} | {c // 1024}K | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {fits} | "
                         f"{cap[0] // 1024}K / {cap[1] // 1024}K / {cap[2] // 1024}K | {conf} |")
    out = "\n".join(lines)
    print(out)
    Path(sys.argv[2]).write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
