# RELATED_WORK — positioning (abstract-level check on 2026-10-21; updated 2026-10-22 after UNIQUENESS_ASSESSMENT.md)

A row marked "abstract checked" means we read that paper's arXiv abstract page today. We did not read any full
text for this table. The numbers in the claims column are the papers' own claims, copied from their abstracts.
[verify] marks entries we have not checked at all; they came from the coordinator's search or from memory.

| Work | What it does | Compression / speed (author claims) | Overlap with us | How we differ | Check |
|---|---|---|---|---|---|
| DFloat11, Zhang et al., NeurIPS 2025, arXiv 2504.11651 | lossless BF16 weights: variable-length codes (frequency-based) on GPU with SRAM lookup tables | ~30% size reduction; 2.3–46.2× throughput vs CPU offload | lossless weights on GPU | they compress BF16 sources; we measure re-coding of already-quantized GGUF k-quants (2–7% headroom) and focus on KV | abstract checked |
| "Approaching Shannon Bound with Lossless LLM Weight Compression", Tan, Chen, Alonso, Wong, He, ISCA 2026, arXiv 2606.15789 | ANS, tile-level on-the-fly decompression aligned to GEMM tiles; entropy study bf16…int4, AWQ/SQ8, 1.5B–405B; SGLang | "effective entropy is 2-10x lower"; bit-rates within 0.01–0.1 bits of Shannon; Qwen-14B batch 47→75, up to 1.2× (Mixtral 1.6×) | lossless weights, entropy ceiling, GPU decode | **resolved by E3 (R40):** int4 code entropy is 0.80 bits for per-tensor scaling (codes collapse) but 3.6 bits for group-32/128 formats. Their low int4 entropy refers to coarse scaling, and their 1.1–1.3× for grouped formats agrees with our 2–7% headroom on k-quants. No contradiction | abstract checked |
| NeuZip, ZipNN | lossless weight compression (training / storage) | — | lossless weights | not GPU-resident inference of k-quants | [verify] |
| SpQR, Dettmers et al., arXiv 2306.03078 | outlier weights kept sparse at higher precision, rest at 3–4 bits | <1% relative PPL loss; >4× memory compression; 33B on one 24 GB GPU, ~15% speedup | dense-and-sparse idea | we only measure outlier statistics on Q4_K weights and KV (R17) | abstract checked |
| GPTQ 2210.17323, AWQ 2306.00978, LLM.int8() 2208.07339, QLoRA/NF4 2305.14314, SmoothQuant 2211.10438, AQLM 2401.06118 | lossy weight quantization | — | lossy weight baselines | not run on our machine yet | [verify] (from memory) |
| KIVI, arXiv 2402.02750 | 2-bit KV: K per-channel, V per-token | — | lossy KV below Q8 | our proxy (R13) confirms the per-channel-K benefit on real Qwen3-4B KV (2.5× lower error at 5 bits) | [verify] (from memory) |
| KVQuant, arXiv 2401.18079 | pre-RoPE per-channel K, non-uniform, dense-and-sparse outliers | — | lossy KV, outliers | our capture is post-RoPE; no implementation | [verify] |
| GEAR, Kang et al., arXiv 2403.05527 | low-bit quant + low-rank error + sparse outliers | near-lossless 4-bit KV; up to 2.38× throughput; up to 2.29× peak memory reduction | lossy KV, low-rank | our proxy shows plain low-rank is poor on this KV (R13); GEAR uses low-rank only for the residual | abstract checked |
| KVTC, Staniszewski & Łańcucki, ICLR 2026, arXiv 2511.01815 | PCA decorrelation + adaptive quantization + entropy coding of KV | up to 20×, 40× for specific uses | entropy coding of KV | lossy (quantization); we measure the **lossless** ceiling of F16/Q8 KV and its GPU-practical fraction | abstract checked |
| D-Quant, Su et al., arXiv 2609.19880 | entropy-coded KV quantization with "drift" to fixed-size per-token streams for parallel decode | no numbers in abstract | entropy coding with fixed-size streams (same goal as our fixed-rate R11) | theirs targets quantized KV; ours is exact F16 with escapes; compare designs | abstract checked |
| JoLT, Krishnan & Schulz, arXiv 2607.12550 | Tucker decomposition + rotated low-bit residual, fused Triton decode kernel | 2–3× with <0.2% PPL degradation; RULER 64K near-lossless to 3× | fused decode over compressed KV | lossy; we are lossless and measure on llama.cpp | abstract checked |
| TailorKV, Yao et al., arXiv 2505.19586 | hybrid quantization + offloading of dominant tokens over PCIe | Llama-3.1-8B, 128K, one RTX 3090, 82 ms/token | consumer 3090, PCIe-bound KV | lossy hybrid; we measure lossless ceilings and the reserved-allocation effect | abstract checked |
| TurboAngle 2603.27467, Fractal KV Archives 2607.07144, CLLA 2410.15252, KeepKV 2504.09936 | various KV compression | — | KV | not read | [verify] |
| "The KV Cache Is the New Memory Wall", Singh, arXiv 2609.30854 | analytic arithmetic intensity vs context; crossover lengths; H100/B200/MI300X; three regimes | below crossover weights dominate and KV compression gives negligible speedup | our roofline section | we calibrate against a consumer 3090 and llama.cpp/GGUF, add measured lossless ceilings, and show that VRAM reserved for the full context cancels capacity savings unless the allocator is paged | abstract checked |
| UCCL-Zip, Ma et al., arXiv 2604.17172 | lossless compression inside NCCL/P2P | up to 47.5% faster RL weight sync; up to 10% lower vLLM latency | multi-GPU / NVLink direction | future work only | abstract checked |
| PagedAttention / vLLM 2309.06180, FlashAttention 2205.14135 | paged KV allocation; IO-aware attention | — | paged allocator argument | — | [verify] (from memory) |
| SplitZip, arXiv 2605.01708 | lossless BF16 KV: exponent coding with a fixed-rate code and escapes, GPU decode | BF16 KV exponent entropy 2.89–3.59 bits (assessment summary) | same idea as our EXP3 fixed-rate exponent code | we measure it on F16 KV (5-bit exponent) in llama.cpp with a non-fused scalar kernel: 0.93–0.96 size, 0.65–0.73× read speed (R26). Our EXP3 is a variant, not new | full text per assessment [verify] |
| ZipServ (ASPLOS'26), arXiv 2603.17435 | lossless BF16 weights, TCA-TBE fixed-rate format decoded inside tensor-core GEMM | per assessment: faster than BF16 GEMM when fused | fixed-rate exponent format; GPU decode | our non-fused BF16 GEMV decode is 0.685× of BF16 (R38). The negative result holds only for non-fused scalar decoders | [verify] |
| TRACE, arXiv 2509.03377 | lossless KV/activation compression in a memory controller: channel-major grouping, bit planes, entropy coding | 1.81–1.88× on BF16 KV (Llama-3.1-8B) per assessment | context-aware lossless KV | our TRACE-like CPU codec on F16 KV of Qwen3-4B reaches 0.84–0.86 (R40); E4 compares BF16 vs F16 KV of the same model | [verify] |
| Heilper & Singer, arXiv 2508.19263 | Huffman coding of KV exponents, mantissas raw | per assessment | exponent-only redundancy | same finding (mantissa is noise) on F16 KV in llama.cpp | [verify] |
| LEXI, arXiv 2603.15589 | exponent entropy < 3 bits in BF16 activations and caches | — | exponent entropy | consistent with our 2.4–2.9 bits for F16 exponents | abstract per assessment |
| CacheGen 2310.07240, BitDecoding 2503.18773, QServe 2405.04532 | KV streaming codec; low-bit KV on tensor cores; W4A8KV4 serving with dequantization cost analysis | — | lossy KV kernels | quantized-KV decode cost is a kernel-path issue; QServe and BitDecoding are the fused designs we did not build | [verify] |
| kvcached / Prism 2505.04021, vAttention 2405.04437, llama.cpp PR #21757 | lazy / virtual-memory KV allocation | — | our PoC-A lazy pool | PoC-A is a reproduction, not a contribution | [verify] |
| Dutta et al., arXiv 2407.09141 | "accuracy is not all you need": flips and KL beyond PPL for compressed models | — | our KL / top-1 / task metrics | we follow their metric advice | abstract per assessment |
| llama.cpp issues #30790, #29371, #24166, #27796; PRs #30792, #30202 | quantized-KV FA path: VEC kernel without GQA reuse, f16 scratch that grows the compute buffer | — | our decode slowdown with Q8_0/Q4_0 KV | E2 (R39): PR #30202 recovers ~70% of the q8_0 gap at 32K–120K on the 3090; PR #30792 has no effect on this GPU | measured (E2) |

## Revised novelty claim (2026-10-22)

We present the paper as an empirical measurement study, mostly with negative results. It does not propose a new method.

- Weights. DFloat11 and ZipServ already cover lossless GPU re-compression of BF16, and so do Tan et al. (ISCA 2026).
  What we add is small: llama.cpp k-quants leave only 2–7% of order-0 headroom. E3 shows why that fits the ISCA
  numbers, since grouped and per-tensor scaling give very different code entropy.
- KV. We estimated order-0 and context entropy for F16 KV in llama.cpp, which leaves about 16–17% headroom, and ran a
  TRACE-like CPU codec that reaches 0.84–0.86. A fixed-rate exponent code of the SplitZip/ZipServ kind turned out
  slower than F16 in a non-fused scalar kernel. We also traced most of the decode slowdown of quantized KV (R39) to llama.cpp's FA path. An earlier
  reading that put most of the Q4 KV quality loss on that path (R36) was wrong; R44 corrects it: the loss comes from
  the Q4_0 format itself.
- Fork stage (R60-R87). The fixed-rate lossless BF16 weight type is a ZipServ-style design and the Huffman type is a
  DFloat11-style one; the GQA-grouped decode kernel for Q8_0/Q4_0 KV and the 12-bit KV type are engineering on known
  ideas, measured here against stock llama.cpp kernels.
- Lazy allocation, mixed-precision weight recipes and outlier-aware KV only reproduce ideas that are already known.
