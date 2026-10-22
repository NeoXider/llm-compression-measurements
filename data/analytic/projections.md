| Card | BW GB/s | ridge FLOP/B (FP16 tensor) | scalar FLOP/B | KV read time F16, ps/elt | EXP3 lossless non-fused | EXP3 fused (overlapped) | KV capacity gain lossless / Q8 / Q4 | confidence |
|---|---|---|---|---|---|---|---|---|
| RTX 3090 | 936 | 76 | 38 | 4.97 | 0.61x | 1.07x | 1.07 / 1.88 / 3.56 | high |
| RTX 5090* | 1792 | 117 | 58 | 2.60 | 0.72x | 1.07x | 1.07 / 1.88 / 3.56 | medium |
| L40S* | 864 | 419 | 106 | 5.38 | 0.84x | 1.07x | 1.07 / 1.88 / 3.56 | low (tensor figure may be sparse) |
| RTX PRO 6000 Blackwell* | 1792 | 140 | ? | 2.60 | ? | ? | 1.07 / 1.88 / 3.56 | low (scalar figure unknown) |
| A100 80GB* | 2039 | 153 | 10 | 2.28 | 0.27x | 0.35x | 1.07 / 1.88 / 3.56 | medium |
| H100 SXM* | 3350 | 295 | 20 | 1.39 | 0.44x | 0.74x | 1.07 / 1.88 / 3.56 | medium |
| H200* | 4800 | 206 | 14 | 0.97 | 0.35x | 0.52x | 1.07 / 1.88 / 3.56 | medium |
| B200* | 8000 | 281 | ? | 0.58 | ? | ? | 1.07 / 1.88 / 3.56 | low (scalar figure unknown) |

Q8_0 KV in llama.cpp on the 3090 runs at e_bw_q8 = 0.22 vs 0.43 for F16. Whether server stacks (e.g. FP8 KV in vLLM/TensorRT-LLM/FlashInfer [verify]) avoid this depends on the kernel; not projected.
