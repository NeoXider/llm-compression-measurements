"""Calibrated analytic projections (*) for KV formats on other GPUs. NOT measurements.

Calibration from RTX 3090 measurements (2026-10-21, shared GPU):
  e_bw      = 0.43  effective bandwidth fraction of llama.cpp's F16 decode at 32K (R21: ~404 GB/s of 936)
  e_bw_q8   = 0.22  the same for its Q8_0 KV path (R21: ~208 GB/s)
  c_exp3    = 3.53 ps per element of extra (non-overlapped) decode time for our fixed-rate lossless EXP3
              format vs F16 in the same kernel (PoC-B, T=32768: 0.7218 - 0.4846 ms over 67.1 M elements)
  r_exp3    = 0.931 stored-bytes ratio of EXP3 on real KV (PoC-B, T=32768)
Scaling assumption: decoder ALU cost per element scales with the card's scalar FP32/INT32 throughput
relative to the 3090 (35.6 TFLOPS). That is a simplification: real kernels can hide decode behind memory
or use tensor cores, and fixed-function decompression engines are ignored.
Card figures (approximate public specs; '?' = not known with confidence):
  BW GB/s, dense FP16 tensor TFLOPS, scalar FP32 TFLOPS, VRAM GB
"""
from __future__ import annotations

CARDS = {
    "RTX 3090":               (936, 71, 35.6, 24, "high"),
    "RTX 5090":               (1792, 210, 104.8, 32, "medium"),
    "L40S":                   (864, 362, 91.6, 48, "low (tensor figure may be sparse)"),
    "RTX PRO 6000 Blackwell": (1792, 250, None, 96, "low (scalar figure unknown)"),
    "A100 80GB":              (2039, 312, 19.5, 80, "medium"),
    "H100 SXM":               (3350, 989, 67.0, 80, "medium"),
    "H200":                   (4800, 989, 67.0, 141, "medium"),
    "B200":                   (8000, 2250, None, 192, "low (scalar figure unknown)"),
}
E_BW, E_BW_Q8, C_EXP3_PS, R_EXP3, REF_SCALAR = 0.43, 0.22, 3.53, 0.931, 35.6


def main():
    print("| Card | BW GB/s | ridge FLOP/B (FP16 tensor) | scalar FLOP/B | KV read time F16, ps/elt | EXP3 lossless non-fused | EXP3 fused (overlapped) | KV capacity gain lossless / Q8 / Q4 | confidence |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, (bw, tf, sc, vram, conf) in CARDS.items():
        ridge = tf * 1e12 / (bw * 1e9)
        t_raw = 2 / (bw * 1e9 * E_BW) * 1e12  # ps per F16 element at calibrated efficiency
        if sc:
            c = C_EXP3_PS * REF_SCALAR / sc
            nonfused = t_raw / (R_EXP3 * t_raw + c)
            fused = t_raw / max(R_EXP3 * t_raw, c)
            s1, s2, scal = f"{nonfused:.2f}x", f"{fused:.2f}x", f"{sc * 1e12 / (bw * 1e9):.0f}"
        else:
            s1 = s2 = scal = "?"
        star = "" if name == "RTX 3090" else "*"
        print(f"| {name}{star} | {bw} | {ridge:.0f} | {scal} | {t_raw:.2f} | {s1} | {s2} | 1.07 / 1.88 / 3.56 | {conf} |")
    print("\nQ8_0 KV in llama.cpp on the 3090 runs at e_bw_q8 = 0.22 vs 0.43 for F16. Whether server stacks"
          " (e.g. FP8 KV in vLLM/TensorRT-LLM/FlashInfer [verify]) avoid this depends on the kernel; not projected.")


if __name__ == "__main__":
    main()
