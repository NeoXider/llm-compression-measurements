# METHODS — catalog of compression methods (extensible)

Each method gets one row. New methods go in as new rows, and old results are never rewritten. The "Status" column
says **measured** when RESULTS.md has real numbers, **prototype** when code exists but is only partly measured, and
**idea** when nothing is implemented. Every measured number points to a RESULTS section.

All rows share one harness, with the scripts in `scripts/`. Size and bits per weight and entropy come from
`q4k_code_entropy.py` and `kv_entropy.py`. Lossless round trips are checked with `kv_lossless_proto.py`,
`kv_fixed_rate.py` and the CQW2 self-test, and the lossy error proxy is `kv_lossy_proxy.py`. KL and PPL come from
`llama-perplexity --kl-divergence`, parsed by `parse_kld_logs.py`. Task accuracy runs on the frozen 39-case suite
with `cpu_task_eval.py` (on CPU, or `--gpu`). Per-process VRAM is read with `gpu_kv_memory_filled.py` (DXGI observer),
and `gpu_clean_window.py` does the clean-window checks and the 1 s telemetry. The reference points are Q8_0 for
weights and F16 for KV.

## KV cache

| ID | Method | Lossless? | Bits/elt (16 = F16) | Quality vs ref | Speed | VRAM | Status | Where |
|---|---|---|---|---|---|---|---|---|
| KV-0 | F16 (baseline) | — | 16 | ref | not measured (clean) | 4966 MiB process @16K alloc (Qwen3-4B) | measured | R1 |
| KV-L1 | Order-0 entropy ceiling, F16 word | yes | 13.8 K / 13.6 V | exact | — | — | measured (bound) | R9 |
| KV-L2 | Byte-plane + zlib-6, 16/64/256-token pages | yes | 14.13 / 13.95 / 13.82 | exact (round trip verified) | CPU enc 13–37 MB/s, dec 159–251 MB/s | not measured on GPU | prototype | R10 |
| KV-L3 | Byte-plane + LZMA-6, 256-token pages | yes | 13.55 (−15.3%) | exact | CPU enc 2.9 MB/s, dec 24 MB/s (too slow for GPU) | not measured on GPU | prototype | R10 |
| KV-L4 | Fixed-rate exponent offset k=3 + escapes (GPU-friendly, random access) | yes | CPU (channel×32 tokens): 14.50 K / 14.52 V (−9.3%); GPU layout (row groups) on real KV: 0.931–0.960 | exact (device check) | GPU read 0.66–0.78× of F16 (PoC-B) | not integrated | measured (negative) | R11, R22 |
| KV-A1 | Lazily mapped KV pool (CUDA VMM, 2 MiB chunks), no compression | — | — | — | — | process VRAM = filled tokens (144 MiB at 1K vs 4608 eager, ctx 32K) | measured (PoC, standalone) | R23 |
| KV-L5 | PAL3 exponent palette, 16-token pages (earlier project codec, GPU) | yes | logical −6.3% | exact | consumer 3.6–11.6× slower than RAW (standalone) | 0%/10% owned (granularity) | measured | R5 |
| KV-L6 | Lossless over Q8_0 codes (order-0 / per-channel) | yes (over Q8) | Q8 8.5 → 7.63 K / 7.94 V (order-0); 7.04 K / 7.77 V (per-channel) | exact vs Q8_0 | — | — | measured (bound) | R9 |
| KV-Q8 | llama.cpp q8_0 | no | 8.5 | KL 0.0042 (2K), 0.0027 (4K); tasks 25/39 = F16 | decode 0.97/0.88/0.79/0.75× of F16 at 0/4K/16K/32K (shared GPU) | 5246 vs 7286 MiB at 32K filled | measured | R12, R18–R21 |
| KV-Q5 | llama.cpp q5_1 / q5_0 | no | 6.0 / 5.5 | R12 | not measured | not measured | measured (KL) | R12 |
| KV-Q4 | llama.cpp q4_1 / q4_0 / iq4_nl | no | 5.0 / 4.5 / 4.5 | KL 0.054 / 0.059 / 0.098 (2K); q4_0 tasks 21/39 | q4_0 decode 0.97/0.87/0.77/0.72× of F16 (shared GPU) | q4_0: 4094 MiB at 32K filled | measured | R12, R18–R21 |
| KV-P1 | Per-channel K / per-token V int4/int2 (KIVI-style) | no | 5.0 / 3.0 | reconstruction-error proxy only | — | — | prototype (proxy) | R13 |
| KV-P2 | Low-rank SVD per layer (r = 128/256/512) | no | depends on r | proxy only | — | — | prototype (proxy) | R13 |
| KV-I1 | Token eviction / merging (H2O-, SnapKV-style) | no | — | — | — | — | idea | NEXT |
| KV-I2 | Mixed precision by layer/head | no | — | — | — | — | idea | NEXT |
| KV-I3 | Context-modelled (adaptive ANS) exponent coding with per-channel tables | yes | expected between L4 and L3 | — | — | — | idea | NEXT |

## Weights

| ID | Method | Lossless? | Size / bpw | Quality vs Q8_0 | Status | Where |
|---|---|---|---|---|---|---|
| W-Q8 | Q8_0 (reference) | — | Qwen3.5-2B: 1980.5 MiB, 8.51 bpw | ref | measured | R7 |
| W-K | Q6_K / Q5_K_M / Q4_K_M / Q3_K_M / Q2_K | no | 6.57 / 5.95 / 5.36 / 4.60 / 4.03 bpw (llama.cpp count incl. embeddings) | KL, PPL, top-1 in R7; tasks in R8 | measured (CPU) | R7, R8 |
| W-E | Order-0 entropy of Q4_K/Q6_K fields | yes (bound) | −6.6% on Qwen3-4B Q4_K_M | exact | measured | R4 |
| W-Z | zlib byte-plane on GGUF (256 KiB frames) | yes | −4.41% | exact | measured (repo) | R4 |
| W-C2 | CQW2 block palette over Q4_K (GPU-shaped) | yes | +0.61% (expands) | exact | measured (negative) | R3 |
| W-H2D | Compressed H2D transfer for partial offload | yes | — | — | idea (needs GPU window) | NEXT |
| W-I1 | Codebook / vector quantization below 4 bits (AQLM-style), mixed bits per layer | no | — | — | idea | NEXT |

### Weights, implemented in the llama.cpp fork `cq/weights-compress` (2026-10-22, R60–R66)

| ID | Method | Lossless? | Size / bpw | Quality vs base | Speed vs base (decode, real inference) | Status | Where |
|---|---|---|---|---|---|---|---|
| W-FH | `CQ_BF16H`: BF16 exponent Huffman (DFloat11-like), sign+mantissa raw, fused decode in GEMV | yes | 0.696 of BF16 (11.1 bits) | bit-identical logits | 0.60–0.71× | measured (negative for speed) | R60–R62 |
| W-FW4 | `CQ_BF16W4`: per-64 max exponent + 4-bit offsets, escape list (ZipServ-like fixed rate) | yes | 0.759 (12.1 bits) | bit-identical logits | 1.04× (4K/32K), 0.98× (120K) | measured | R60–R62 |
| W-FW3 | `CQ_BF16W3`: 3-bit offsets, 2.6% smallest values flushed | no | 0.695 (11.1 bits) | R65 | 1.10–1.14× | measured | R62, R65 |
| W-Q8L | Huffman / context coding of Q8_0 codes and scales | yes | GPU-practical 0.966 (ideal 0.92) | exact | not built: projected ≤ 0.4× from the measured H decoder rate (*) | CPU only | R64 |
| W-Q4L | same for Q4_K_M (Q4_K + Q6_K blocks) | yes | GPU-practical 0.974 (2B), 0.948 (4B) | exact | not built (< 3% saving) | CPU only, negative | R64 |
