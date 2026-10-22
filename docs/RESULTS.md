# RESULTS — measured numbers only

Every number below can be traced to a file on disk; the paths are absolute. "Repo" means `<private-repo>`
(private, HEAD `a4e4cf9`, and its working tree still holds uncommitted results and tools — see
`artifacts/offline_tests_20261021/git_status.txt`). "Paper dir" means `/path/to/repo`.

All GPU rows share one setup: RTX 3090 24 GB under WDDM on Windows 11, CUDA 11.8, MSVC 14.38, a llama.cpp
fork pinned at `ed7ac35e1ee49cb70e4dfa9f0a2ce39b0a5ec4ea`, and the model `Qwen3-4B-Q4_K_M.gguf`
(2,497,280,256 B, SHA256 `7485fe6f…34fdf5`) with 36 layers, GQA 32/8, head dim 128, Flash Attention on,
all weights and KV on CUDA0.

How to read the status tags: **MEASURED** means a raw receipt exists and I recomputed the number from it;
**MEASURED (repo report)** means the receipt exists but the number comes from the repo report and was only
spot-checked; **EXCLUDED** means there is no raw receipt, so we did not use the value.

---

## R1. Native KV precision vs process VRAM, context 16384 — MEASURED

The receipts come from `<private-repo>\results\gpu_wddm_native_kv_20261011_r01\` (8 slots, all exit 0), and
I recomputed the summary with `scripts/summarize_wddm_kv.py` → `artifacts/wddm_kv_summary_20261021.json`.
The metric is DXGI `IDXGIAdapter3::QueryVideoMemoryInfo` CurrentUsage for the calling process, on the adapter
whose LUID matches the CUDA device, sampled every 5 ms. A sampled peak is only a lower bound; it is not a proven
physical peak (`physical_peak_proven: false` in every receipt).

| KV type | Slots | Process LOCAL sampled peak, MiB | Δ vs F16 | NON_LOCAL, MiB | llama.cpp KV buffer, MiB | Compute buffer, MiB |
|---|---|---|---|---|---|---|
| F16 | 4 | 4966.0 (all 4 identical) | — | 84.0 | 2304.00 | 22.75 |
| Q8_0 | 2 | 3940.0 (both identical) | −1026 MiB / −20.66% | 84.0 | 1224.00 | 74.58 |
| Q4_0 | 2 | 3364.0 (both identical) | −1602 MiB / −32.26% | 84.0 | 648.00 | 74.58 |

Arithmetic on those rows, as a check:
- The KV buffer lands exactly on the analytic size: 36 layers × 2 (K,V) × 8 heads × 128 × 16384 tokens = 1,207,959,552
  elements → F16 2 B = 2304 MiB; Q8_0 34 B/32 elts = 1224 MiB; Q4_0 18 B/32 elts = 648 MiB.
- In both cases the process saves 54 MiB less than the KV buffer does (1080→1026, 1656→1602). The declared compute
  buffer grows by 51.83 MiB when KV is quantized, which accounts for most of the gap.
- Whatever is left unattributed (process LOCAL − model 2375.91 − KV − compute): F16 263.3 MiB, Q8_0 265.5 MiB,
  Q4_0 265.5 MiB. That is the CUDA context, cuBLAS and the like; we did not break it down.

Limits that have to travel with R1 whenever it is quoted:
- We processed only 28 prompt + 64 decode tokens. The 16384-cell cache was allocated but never filled, so the peak
  during a long prefill was not measured.
- This is native llama.cpp KV quantization, not the project's own codec. Q8_0/Q4_0 are lossy.
- One model, one GPU, one driver, and WDDM accounting rather than TCC/Linux.

The 8 slots also produced decode/prefill speed, but with the observer on, a shared GPU, 64 decode tokens at roughly
a 92-token context and no quiet gate. Those numbers are not comparable between rows and are not a speed result:
F16 decode 166.52 / 142.63 / 128.85 / 146.73 tok/s; Q8_0 156.39 / 142.31; Q4_0 116.25 / 137.00.

## R2. Answer quality, native F16 vs Q4_0 KV — PARTIAL, NOT ACCEPTED

The receipt is `<private-repo>\results\interrupted_quality_20260930_report.json`, from the collection
`results\gpu_quality_f16_q4_20260930_r01\`. It drives a frozen 39-case benchmark,
`benchmarks\compression_quality_v1.json` (ru 6, en 6, code 5, ar 7, fa 5, lc 10), at context 16384 with a decode
budget of 256. The run was interrupted at 29 of 78 receipts.

| Outcome over 14 complete pairs | Count | Cases |
|---|---|---|
| both pass | 9 | en-001, en-002, en-004, en-005, en-006, ru-001…ru-004 |
| both fail | 3 | code-001, code-002, ru-006 |
| F16 pass, Q4_0 fail | 1 | ru-005 |
| F16 pass, Q4_0 error | 1 | en-003 |
| F16 fail, Q4_0 pass | 0 | — |

One more receipt, code-003, exists for Q4_0 alone and fails. The ar/fa/lc strata (22 cases, including all the
long-context ones) were never run. That leaves 2 discordant pairs, both against Q4_0 — too few to conclude
anything (exact two-sided sign test p = 0.5). We have no verdict for the full suite.

## R3. CQW2 lossless re-coding of real Q4_K blocks — MEASURED (new, 2026-10-21, CPU)

The self-test lives in `artifacts/cqw2_cpu_selftest_20261021/`: `cqw_codec_cpu.exe` ran 6220 checks with 0 failures
and exit 0. The fixture is synthetic, so its ratio means nothing. The binary was built 2026-10-11 20:37, after the
last edit to the sources at 20:35.

For real weights we used `artifacts/cqw2_real_weight_samples_20261021/receipt.json` (+ `.summary.json`), produced by
the repo's `experimental/cqw-integration/standalone/real_weight_samples.py`. It takes 16 rows from the start, middle
and end of each of the 216 rank-2 Q4_K tensors, which makes 648 samples; the model SHA was checked before and after.

| Quantity | Bytes |
|---|---|
| Native Q4_K bytes sampled | 19,906,560 |
| CQW2 payload | 19,905,678 (−882 B) |
| CQW2 metadata (header, row offsets, 2-bit mode map) | 122,688 |
| CQW2 frame total | 20,028,366 |
| **Frame / native** | **1.006119 (+0.61%, i.e. larger)** |
| Blocks by mode RAW / P1 / P2 / P3 | 138,217 / 0 / 6 / 17 |

Every sample came back byte-exact and the CPU mat-vec delta was 0; the script aborts on anything else. The earlier
repo screen `results\m1w_mmq_palette_20260928\report.md` (MEASURED (repo report)) points the same way: 216 tensors,
95,760 tiles, 19 palette / 95,741 RAW, 1,765,048,320 → 1,766,570,752 B (+0.0863%). Block-palette re-coding of Q4_K
does not save memory on this model.

## R4. How much room is there? Order-0 entropy of Q4_K_M fields — MEASURED (new, 2026-10-21, CPU)

`scripts/q4k_code_entropy.py` wrote `artifacts/q4k_code_entropy_20261021/entropy.json`. For each k-quant block field
we took the order-0 empirical entropy per tensor and set the ideal size to Σ n·H / 8. That ideal leaves out codec
tables, metadata, alignment and a random-access index, all of which make a real codec bigger, and it is not a bound
for coders that model context.

| Type (tensors) | Field | Native bytes | Ideal order-0 bytes | Ratio | Bits/symbol |
|---|---|---|---|---|---|
| Q4_K (216) | 4-bit codes | 1,568,931,840 | 1,516,534,981 | 0.9666 | 3.866 of 4 |
| | 6-bit scales/mins (12 B) | 147,087,360 | 128,374,750 | 0.8728 | 6.98 of 8 |
| | d (f16) | 24,514,560 | 14,186,343 | 0.5787 | 9.26 of 16 |
| | dmin (f16) | 24,514,560 | 15,609,066 | 0.6367 | 10.19 of 16 |
| | **all fields** | 1,765,048,320 | 1,674,705,140 | **0.9488** | |
| Q6_K (37) | 6-bit codes | 663,306,240 | 599,225,558 | 0.9034 | 5.42 of 6 |
| | int8 scales | 55,275,520 | 48,318,202 | 0.8741 | 6.99 of 8 |
| | d (f16) | 6,909,440 | 3,626,672 | 0.5249 | 8.40 of 16 |
| | **all fields** | 725,491,200 | 651,170,432 | **0.8976** | |
| **All quantized tensors** | | 2,490,539,520 | 2,325,875,573 | **0.9339 (−6.61%)** | |

We left 145 small F32 tensors (784,384 B) out.

For comparison, the MEASURED (repo report) numbers in `results\m1w_screen_20260928\report.md` and
`Qwen3-4B-Q4_K_M.json` (`referenceB` = 2,387,186,155): zlib-6 on byte planes with 256 KiB frames takes the whole
file from 2,497,280,256 to 2,387,186,155 B (−4.41%), while 4 KiB frames give −0.37…−0.42%. Read together, order-0
headroom is about 6.6%; a generic coder with large frames reaches about 4.4%; small random-access frames a GPU
decoder could use reach about 0.4%; and the GPU-shaped palette format from R3 ends 0.6% larger than the input.

## R5. Lossless F16 KV page codec — MEASURED (repo report, spot-checked)

The corpus is 4096 complete F16 KV pages of 16 tokens × head-128 (16 MiB), cut from a verified native capture
(`results\m0_native_20260928\m1k-diagnostic-f16-p2048-20260928-01\`).

The CPU screen `results\m1k_lossless_codec_screen_20260928_provenance_v2\screen.json` got every page back exact.
Best zlib-6/9 framed size was 15,799,901 / 15,799,898 B (≈ −5.83%) before alignment, and the aligned payload span is
16,610,304 B (−0.99%). LZ4 only compressed 62–73 of the 4096 pages.

The GPU side is a standalone microcodec `results\m3_ckv_gpu_modes_20260928\` (18 cases, technically accepted, its own
VMM commitments, 10 repeats per case, contended GPU, `comparative_eligibility=false`):

| Codec / layout | Owned commitment RAW → codec | Consumer median RAW / codec, ms | Slowdown |
|---|---|---|---|
| PAL3 lossless / separate | 20 → 20 MiB (0%) | 0.903 / 10.439 | 11.6× |
| PAL3 lossless / coalesced | 20 → 18 MiB (10%, allocator rounding) | 0.986 / 3.554 (re-derived from `real-pal3-coalesced-02/gpu.json`) | 3.6× |
| DROP2 lossy / both layouts | 20 → 18 MiB (10%) | 0.67 / 40.6–41.6 | 60–62× |
| DROP4 lossy / both layouts | 20 → 16 MiB (20%) | 0.63–0.90 / 35.9 | 40–57× |

PAL3 logical records total 15,720,064 B against 16,777,216 B (−6.3%). Sealing peaks at 40 MiB because the old RAW
pages and the new ones are both resident. This is a standalone 16 MiB microbenchmark, not model inference.

## R6. Offline test suites — re-run 2026-10-21 (CPU only)

`artifacts/offline_tests_20261021/`: the quality scorer passed 65/65, the paired runner 13/13, the quality runner
4/4, and the comparison validator 26/26. Repo `git status` was identical before and after.

## R7. Weight quantization curve vs Q8_0 (lossy) — MEASURED (CPU, 2026-10-21)

There is no local Q8_0/BF16 Qwen3-4B and downloading one was not approved, so the curve runs on the local BF16
checkpoint of **Qwen3.5-2B** (`/path/to/models\qwen3.5-2b-bf16`, a hybrid linear/full-attention
architecture). We converted it with the pinned `convert_hf_to_gguf.py` and quantized with `llama-quantize` (build
11552, `23b0202a`, no imatrix). Logs are in `artifacts/qwen35_2b_logs/`, the table in
`artifacts/qwen35_2b_kld_20261021.json`.

Evaluation used `llama-perplexity --kl-divergence` on CPU only (CUDA hidden, and the log confirms it), context 512,
over 44 chunks of the fixed local corpus `artifacts/corpus_v1/corpus.txt` (95,659 B, SHA256 `589a50fe…`; Russian
narrative plus English technical notes written Aug–Oct 2026). The reference logits come from Q8_0.

| Config | File MiB | bpw (llama.cpp, all tensors) | PPL | PPL / PPL(Q8_0) | Mean KLD vs Q8_0 | Median KLD | p99 KLD | Same top-1 vs Q8_0 |
|---|---|---|---|---|---|---|---|---|
| BF16 | 3716.8 | 16.00 | 64.686 ± 2.232 | 0.9987 ± 0.0007 | 0.00133 | 0.00110 | 0.00624 | 96.12 ± 0.18% |
| **Q8_0 (ref)** | 1980.5 | 8.51 | 64.843 ± 2.238 | 1 | 0 | 0 | 0 | 100% |
| Q6_K | 1531.9 | 6.57 | 65.498 ± 2.264 | 1.0112 ± 0.0016 | 0.01063 | 0.00903 | 0.04593 | 87.92 ± 0.31% |
| Q5_K_M | 1387.4 | 5.95 | 66.028 ± 2.293 | 1.0194 ± 0.0023 | 0.02059 | 0.01666 | 0.09384 | 86.23 ± 0.33% |
| Q4_K_M | 1251.4 | 5.36 | 67.890 ± 2.361 | 1.0481 ± 0.0040 | 0.06016 | 0.04776 | 0.28196 | 78.94 ± 0.39% |
| Q3_K_M | 1075.6 | 4.60 | 77.982 ± 2.723 | 1.2040 ± 0.0091 | 0.24053 | 0.18371 | 1.24019 | 67.29 ± 0.44% |
| Q2_K | 944.6 | 4.03 | 314.469 ± 11.825 | ≈ 4.86 (exp of mean ln ratio 1.580 ± 0.022) | 1.849 ± 0.013 * | n/a * | n/a * | 29.97 ± 0.43% * |

\* The Q2_K run printed its PPL block but no KL summary block (exit code 0). The values are the cumulative
per-chunk figures at chunk 44 from the same log. Q8_0's "PPL" comes from its own run; recomputed from the stored
logits it is 64.772.
bpw counts the 248K-vocab embedding and output tensors, which stay at higher precision — that is why Q2_K lands at
4.03 bpw. BF16 by itself already disagrees with Q8_0 on top-1 for about 4% of tokens, which sets the noise floor
for that column.

## R8. Task accuracy, 39-case frozen suite, Qwen3.5-2B weight configs — MEASURED (CPU, partial)

`artifacts/qwen35_2b_tasks_cpu_20261021/` holds the per-config records and `paired_vs_q8.json`. This uses the repo's
own deterministic scorer, llama-server on CPU, greedy decoding, thinking disabled, context 8192.

| Config | Valid cases | Pass | Both pass | Q8-only pass | Config-only pass | Identical outputs to Q8_0 | Sign test p |
|---|---|---|---|---|---|---|---|
| BF16 | 39 | 27 | 27 | 2 | 0 | 35 | 0.50 |
| **Q8_0 (ref)** | 39 | 29 | — | — | — | 39 | — |
| Q6_K | 39 | 27 | 25 | 4 | 2 | 30 | 0.69 |
| Q5_K_M | 39 | 27 | 25 | 4 | 2 | 30 | 0.69 |
| Q4_K_M | 32 † | 21 | 20 | 3 | 1 | 23 | 0.63 |
| Q3_K_M, Q2_K | not run † | | | | | | |

† I stopped the CPU run when the coordinator told us to switch to the GPU. That broke 7 long-context cases of
Q4_K_M (harness connection errors, excluded) and left Q3_K_M/Q2_K unrun. The GPU rerun we planned next was then
blocked (R14). With 39 cases no difference is significant; the suite only detects large losses.

## R9. Lossless ceiling of real KV (entropy) — MEASURED (CPU)

`scripts/kv_entropy.py` produced `artifacts/kv_entropy_20261021/kv_entropy.json`. The input is the verified native
capture `<private-repo>\kv-captures\native\m1k-diagnostic-f16-p2048-20260928-01\state.bin` (Qwen3-4B Q4_K_M,
F16 KV, T = 2304 tokens, 36 layers × 8 KV heads × 128). That is one text at one context length.

| Model of the data | K bits/elt | V bits/elt | K ratio | V ratio |
|---|---|---|---|---|
| F16 native | 16 | 16 | 1 | 1 |
| order-0 over 16-bit word (per layer) | 13.81 | 13.58 | 0.863 | 0.849 |
| sign + exponent + mantissa as independent fields | 13.85 | 13.63 | 0.865 | 0.852 |
| exponent and sign conditioned on channel (head, dim) | 13.35 | 13.51 | 0.834 | 0.844 |
| time delta of the 16-bit word per channel | 13.70 | 14.08 | 0.856 | 0.880 |
| field entropies (all layers pooled): sign / exponent / mantissa | 1.00 / 2.89 / 9.97 | 1.00 / 3.31 / 9.97 | | |
| offline Q8_0 codes, order-0, + f16 scale per 32 (of 8.5 b) | 7.63 | 7.94 | 0.897 | 0.934 |
| offline Q8_0 codes conditioned on channel (Miller–Madow) | 7.04 | 7.77 | 0.828 | 0.914 |
| offline Q4_0 codes order-0 + scale (of 4.5 b) | 3.88 | 4.18 | 0.861 | 0.929 |

The field split says a lot: the 10-bit mantissa is essentially incompressible, the sign costs 1 bit, and the
redundancy sits entirely in the 5-bit exponent, of which about 2.9–3.3 bits get used. For lossless F16 KV that puts
the order-0 ceiling at roughly **14–17%**, not 20–35%. Low rank is not the way out either: a centred SVD needs rank
578–909 of 1024 (K) or 591–920 (V) for 99% energy in middle and late layers (layer 0 K: rank 312), so these KV
matrices are not low-rank.

## R10. Bit-exact lossless F16 KV page codecs — PROTOTYPE, MEASURED on CPU

`scripts/kv_lossless_proto.py` wrote `artifacts/kv_lossless_proto_20261021/results.json`. A page is P tokens × one
layer's K or V row. Each page is compressed on its own, then decoded and compared byte-for-byte (**all pages
exact**). Every stream in a page pays an 8 B index, and a page can always fall back to RAW. In total the run covers
339,738,624 B of KV payload.

| Codec | P=16 | P=64 | P=256 | CPU encode / decode, MB/s (P=16 → 256) |
|---|---|---|---|---|
| zlib-6, bytes as stored | 0.930 | 0.928 | 0.928 | 38→15 / 235→148 |
| byte-plane split + zlib-6 | 0.883 | 0.872 | **0.864** | 37→13 / 244→159 |
| byte-plane split + LZMA-6 | 0.874 | 0.855 | **0.847** | 3.3→2.9 / 20→24 |
| sign+exponent bytes zlib-6, mantissa packed raw | 0.902 | 0.896 | 0.893 | 18→7 / 120→53 |

The best practical lossless result on this capture is **−15.3%** (LZMA, 256-token pages) and **−13.6%** (zlib).
Those CPU speeds come from single-thread Python plus stdlib, which is far below what GPU decode needs (see R15). We
could not measure VRAM footprint, write/read speed or decode tok/s on a GPU (R14).

## R11. Fixed-rate random-access exponent format — PROTOTYPE, MEASURED on CPU

`scripts/kv_fixed_rate.py` wrote `artifacts/kv_fixed_rate_20261021/results.json`. Each group of 32 values carries a
5-bit base exponent and a k-bit offset per value plus an escape code; escaped exponents go to a per-group overflow
list (an 8-bit count/pointer is charged), and sign plus mantissa are stored verbatim. Every group is bit-exact.

| Grouping | k=1 | k=2 | k=3 | escape rate at k=3 |
|---|---|---|---|---|
| K, 32 channels of one token | 16.97 b (1.061) | 15.98 (0.999) | 14.71 (0.920) | 6.1% |
| K, one channel over 32 tokens | 16.54 (1.034) | 14.74 (0.921) | **14.50 (0.906)** | 1.8% |
| V, 32 channels of one token | 16.87 (1.054) | 15.26 (0.954) | 14.53 (0.908) | 2.5% |
| V, one channel over 32 tokens | 16.78 (1.049) | 15.08 (0.943) | **14.52 (0.907)** | 2.3% |

So a decode-friendly lossless format with arithmetic addressing saves about 9.3%, just under the 10% bar.
Variable-rate entropy coding (R10) reaches 13.6–15.3%, but it needs sequential decode inside a page.

## R12. Lossy KV below Q8: KL vs F16 KV — MEASURED (CPU)

`artifacts/kv_lossy_kld_cpu_20261021/` holds the logs and `kv_kld_summary.json` with log hashes. The weights stay
fixed at Qwen3-4B Q4_K_M and only the KV type changes; `-fa on`, context 2048, 6 chunks of the same corpus (6144
scored tokens). Everything ran on CPU with the GPU hidden. Reference logits are F16 KV (PPL 13.1235 ± 0.5236).

| KV type | bits/elt | PPL | PPL / PPL(F16 KV) | Mean KLD | Median KLD | p99 KLD | Same top-1 |
|---|---|---|---|---|---|---|---|
| F16 (ref) | 16 | 13.124 ± 0.524 | 1 | 0 | 0 | 0 | 100% |
| q8_0 | 8.5 | 13.177 ± 0.527 | 1.0054 ± 0.0019 | 0.00419 ± 0.00083 | 0.00131 | 0.0273 | 96.29 ± 0.24% |
| q5_1 | 6.0 | 13.203 ± 0.528 | 1.0074 ± 0.0032 | 0.01354 ± 0.00053 | 0.00438 | 0.1400 | 93.87 ± 0.31% |
| q5_0 | 5.5 | 13.211 ± 0.527 | 1.0080 ± 0.0035 | 0.01679 ± 0.00087 | 0.00515 | 0.1832 | 93.21 ± 0.32% |
| q4_1 | 5.0 | 13.606 ± 0.547 | 1.0382 ± 0.0064 | 0.05431 ± 0.00195 | 0.01683 | 0.5727 | 88.71 ± 0.40% |
| q4_0 | 4.5 | 13.489 ± 0.537 | 1.0292 ± 0.0065 | 0.05945 ± 0.00189 | 0.01947 | 0.6949 | 87.78 ± 0.42% |
| iq4_nl | 4.5 | 13.854 ± 0.555 | 1.0571 ± 0.0087 | 0.09831 ± 0.00387 | 0.02921 | 1.0737 | 85.04 ± 0.46% |

Q8_0 KV costs about as much KL as Q6_K weights did against Q8_0 weights in R7 — different model and reference, so
read that as indicative only. There is a step between 5-bit KV (KL ≈ 0.014–0.017) and 4-bit KV (KL ≈ 0.054–0.098),
and iq4_nl is the worst 4-bit type here. We tested only 2K context; KV errors may grow with longer contexts.

## R13. Lossy KV formats on the real capture — reconstruction-error PROXY (not KL)

`scripts/kv_lossy_proxy.py --layer-stride 4` → `artifacts/kv_lossy_proxy_20261021/results.json`, which covers 9 of
36 layers for K and for V, with relative Frobenius error averaged over layers.

| Format | bits/elt | K rel. err | K SNR dB | V rel. err | V SNR dB |
|---|---|---|---|---|---|
| q8_0 | 8.50 | 0.0089 | 41.2 | 0.0060 | 44.4 |
| q5 symmetric (close to q5_0) | 5.50 | 0.0681 | 23.6 | 0.0511 | 25.9 |
| q4_1 | 5.00 | 0.0986 | 20.2 | 0.0836 | 21.6 |
| **per-channel K / per-token V int4 (KIVI-style)** | 5.00 | **0.0389** | 28.7 | 0.0836 | 21.6 |
| q4_0 | 4.50 | 0.1219 | 18.5 | 0.0965 | 20.3 |
| q3 symmetric | 3.50 | 0.2645 | 12.0 | 0.2519 | 12.0 |
| per-channel/per-token int2 | 3.00 | 0.1965 | 14.6 | 0.4206 | 7.5 |
| low-rank r=128 / 256 / 512 (f16 factors) | 2.89 / 5.78 / 11.56 | 0.230 / 0.161 / 0.084 | 13.7 / 16.8 / 22.5 | 0.419 / 0.273 / 0.121 | 7.6 / 11.3 / 18.6 |

Per-channel scaling of K gives 2.5× lower error than per-block q4_1 at the same 5 bits. V gains nothing from it and
matches q4_1 exactly, which is what per-token groups should do. Low-rank is the worst option at every bit budget.
All of these are proxies: llama.cpp has no per-channel KV type, so no KL was measured for them.

## R14. GPU attempt on 2026-10-21 and system facts — NOT RUN

- The preflights `artifacts/gpu_preflight_20261021_1500.json` and a second preflight JSON both came back DIRTY.
  Before 16:07 the other GPU job and a training job together put us at 99%, 313 W, 77 °C, throttle 0x4 (SW power cap).
  With the other GPU jobs stopped, training PID 35196 was still alone at 32–45% and 130–165 W.
- Claude Code's permission check refused my GPU batch (`scripts/gpu_batch_20261021.sh`) and the a restart of the other GPU job,
  calling both "Interfere With Workloads". No GPU number in this file comes from today.
- `nvidia-smi` (read-only, `artifacts/nvidia_smi_q_20261021.txt`) reports WDDM, driver 617.14, PCIe link currently
  Gen2 x16 against a max of Gen3 x16 (this platform cannot do Gen4; the Gen2 reading was taken under light load and
  may be link power management), and BAR1 total 256 MiB, which means Resizable BAR is off.

## R15. Theory vs measurement (decode, Qwen3-4B Q4_K_M, RTX 3090)

The theory side is the roofline script `scripts/roofline_model.py` (written by the coordinator; the card figures are
approximate) → `artifacts/roofline_model_output_20261021.txt`. The measurements are the R1 slots, taken in a
contended window.

| Quantity | Theory | Measured | Gap / where it goes |
|---|---|---|---|
| Decode upper bound at ~0 context, 936 GB/s, weights 2.49 GB/token | 376 tok/s at 100% bandwidth; 188 tok/s at eff 0.5 | 128.9–166.5 tok/s (R1, 4 F16 slots, contended) | implied effective bandwidth 321–415 GB/s = 34–44% of peak; kernel efficiency at batch 1 plus the foreign load |
| KV F16 at 16K allocated | 2304 MiB | 2304 MiB buffer, process 4966 MiB | exact match for KV; about 263 MiB unattributed (CUDA context, libraries) |
| Q8_0 / Q4_0 KV saving at 16K | 1080 / 1656 MiB | 1026 / 1602 MiB | compute buffer +51.8 MiB for quantized KV |
| Lossless KV ratio (PAL3) | ~0.86–0.94 bytes | logical 0.937, owned memory 1.00 or 0.90 | 2 MiB allocation granularity on a 16 MiB pool |
| PAL3 consumer speed | with r = 0.94 and decode overlapped, ≥ 1.0× | 3.6–11.6× **slower** | not explained by bandwidth math. Design reading (`docs/CKV_GPU_V2_DESIGN_RU.md`): per-page index and span validation, strict fallback and checksum, LSB-first variable-width bit extraction (88-bit packer), index indirection on 16-token pages, standalone kernel instead of fusion into attention. No profiler trace exists, so this is analysis, not a measured profile. |
| Model capacity (roofline script: F16 139K, lossless 0.86 162K, Q8 262K, Q4 494K tokens) | as listed | not measured | caveat: Qwen3-4B's trained context is 40,960, so on 24 GB memory is not the limit for this model; the script also assumes PCIe at 25 GB/s, but this machine is Gen3 (≈ 15.75 GB/s theoretical) |

## R16. Lossless headroom of weights by k-quant block type — MEASURED (CPU)

`scripts/gguf_quant_entropy.py` → `artifacts/gguf_quant_entropy_20261021.json`. It gives the order-0 per-field ideal
size over native size. For Q5_K and Q3_K we treat the high bit and the low bits as independent fields, so the ideal
size there is an upper bound.

| File | Overall ratio | By block type (tensors, ratio) |
|---|---|---|
| Qwen3-4B Q4_K_M | 0.9339 | Q4_K (216) 0.949, Q6_K (37) 0.898 |
| Qwen3.5-2B Q8_0 | 0.9274 | Q8_0 (195) 0.927 |
| Qwen3.5-2B Q6_K | 0.9534 | Q6_K (195) 0.953 |
| Qwen3.5-2B Q5_K_M | 0.9685 | Q5_K (168) 0.981, Q6_K (27) 0.953 |
| Qwen3.5-2B Q4_K_M | 0.9509 | Q4_K (168) 0.949, Q6_K (27) 0.953 |
| Qwen3.5-2B Q3_K_M | 0.9624 | Q3_K (137) 0.983, Q4_K (54) 0.949, Q5_K (3) 0.981, Q6_K (1) 0.953 |
| Qwen3.5-2B Q2_K | 0.9499 | Q2_K (137) 0.928, Q3_K (32) 0.983, Q4_K (25) 0.950, Q6_K (1) 0.953 |

At order 0, lossless headroom on llama.cpp quantized weights is **2–7%**. That is a long way from the
"2-10x lower entropy" claim for low-bit formats; RELATED_WORK.md lists it as unresolved, and the granularity and
formats in that paper still need checking.

## R17. Zeros, trivial blocks and outlier clipping — STATISTICS ONLY (CPU)

`scripts/sparsity_outlier_stats.py` → `artifacts/sparsity_outlier_stats_20261021.json`. For weights we sampled up to
20,000 random blocks per Q4_K tensor of Qwen3-4B Q4_K_M, 3,792,960 blocks in total. For KV we used the same F16
capture as R9.

| Quantity | Weights (Q4_K) | KV K | KV V |
|---|---|---|---|
| code value 0 share (Q4_K code 0 = block minimum, **not** a zero weight) | 4.50% | — | — |
| fully constant 256-weight blocks | 0 | — | — |
| 32-weight sub-blocks with one code | 0.022% | — | — |
| exact zeros | — | 0.000005% | 0.000012% |
| abs value < 10⁻³ of the layer max | — | 3.36% | 1.89% |

The clipping experiment: values above the (100 − p) percentile of |x|, with one global threshold over the pooled
sample, move to a sparse side list (16-bit index + 16-bit value), and the dense remainder is quantized with
symmetric 4-bit absmax per 32. MSE is reported relative to the same quantizer with no clipping.

| p | Side-list cost, b/elt | Weights MSE ratio | K MSE ratio | V MSE ratio | Mean group range kept (W / K / V) |
|---|---|---|---|---|---|
| 0.1% | 0.032 | 0.787 | 0.595 | 0.741 | 0.997 / 0.974 / 0.994 |
| 1% | 0.320 | 0.557 | 0.240 | 0.419 | 0.967 / 0.854 / 0.966 |

Pulling 1% of outliers out of K quarters the 4-bit error for +0.32 b/elt, which is the dense-and-sparse idea used by
SpQR/KVQuant-style work. These are error statistics only, not KL or task quality.

## R18. Per-process VRAM with a FILLED context (GPU, shared window) — MEASURED

We ran this on 2026-10-21 at 18:10, after approval in the session. The window was shared with the r04 training job
PID 35196 (21–37% utilization, about 4 GB; `artifacts/gpu_preflight_20261021_shared_batch.json`, 1 s telemetry
`artifacts/gpu_batch_20261021/telemetry_1s.csv`). Per-process memory does not depend on what the other process is
doing. Data: `artifacts/gpu_batch_20261021/memory_filled/rows.json`, with every slot's stderr validated by the repo's
`validate_receipt.py`.
Model Qwen3-4B Q4_K_M with an observer-enabled llama-cli (`wddm-cuda-luid` build); the prompt fills most of `-c`,
then 32 decode tokens, with 2 interleaved repeats per cell (F16, Q8, Q4, then Q4, Q8, F16). Every repeat matched to
the MiB.

| Context (prompt tokens) | KV | Process LOCAL after init, MiB | Sampled peak, MiB | Δ peak vs F16 | KV buffer, MiB | Compute buffer, MiB | NON_LOCAL, MiB |
|---|---|---|---|---|---|---|---|
| 4096 (2903) | F16 | 3232 | 3246 | — | 576 | 19.75 | 82 |
| 4096 | Q8_0 | 2968 | 2984 | −262 (−8.1%) | 306 | 23.58 | 82 |
| 4096 | Q4_0 | 2824 | 2840 | −406 (−12.5%) | 162 | 23.58 | 82 |
| 16384 (10551) | F16 | 4964 | 4978 | — | 2304 | 22.75 | 86 |
| 16384 | Q8_0 | 3938 | 3954 | −1024 (−20.6%) | 1224 | 74.58 | 86 |
| 16384 | Q4_0 | 3362 | 3378 | −1600 (−32.1%) | 648 | 74.58 | 86 |
| 32768 (24772) | F16 | 7272 | 7286 | — | 4608 | 26.75 | 90 |
| 32768 | Q8_0 | 5230 | 5246 | −2040 (−28.0%) | 2448 | 142.58 | 90 |
| 32768 | Q4_0 | 4078 | 4094 | −3192 (−43.8%) | 1296 | 142.58 | 90 |

What the numbers show:
- Filling the context adds only **14–16 MiB** over the after-init level, because `-c` reserves the memory at
  start-up. That confirms the R1 allocation finding on a filled cache (16K F16: 4978 MiB filled against 4966 MiB in
  R1 with 92 tokens).
- **The compute buffer for quantized KV grows with context** (23.6 → 74.6 → 142.6 MiB for 4K → 16K → 32K), while
  F16 stays at 20–27 MiB. At 32K this eats about 116 MiB of the KV saving.
- Prompt-processing time from the same slots (shared GPU, diagnostic; the two repeats agree within about 1% at
  16K/32K): at 16K F16 4.32/4.35 s against Q8 4.80/4.74 s and Q4 4.71/4.70 s; at 32K F16 13.87/13.82 s against Q8
  15.52/15.50 s and Q4 15.48/15.40 s. Quantized KV makes prefill about 9–12% slower.

## R19. KV KL at context 4096 on GPU (shared window) — MEASURED

Logs: `artifacts/gpu_batch_20261021/kv_kld_base_f16.log`, `kld_kv_q8_0.log`, `kld_kv_q4_0.log`. llama-perplexity
build 11552, `-ngl 99 -fa on`, context 4096, 6 chunks, F16 KV as the reference (PPL 27.828 ± 0.893).

| KV | PPL ratio | Mean KLD | p99 KLD | Same top-1 |
|---|---|---|---|---|
| q8_0 | 1.0052 ± 0.0017 | 0.00268 ± 0.00007 | 0.0253 | 95.86 ± 0.18% |
| q4_0 | 1.0246 ± 0.0049 | 0.06939 ± 0.00336 | 0.6470 | 84.45 ± 0.33% |

That lines up with R12 (CPU, context 2048), where q8_0 was 0.0042 and q4_0 0.059.

## R20. Task accuracy on GPU (shared window) — MEASURED

Sources: `artifacts/gpu_batch_20261021/tasks_*` and `tasks_paired_summary.json`. 39 frozen cases, greedy decoding,
thinking off, llama-server build 11552 on CUDA0. No arm hit a harness error.

Qwen3-4B Q4_K_M, KV type varied (ctx 16384), F16 KV as reference:

| KV | Pass | Lost vs F16 | Gained vs F16 | Identical outputs | Sign test p |
|---|---|---|---|---|---|
| F16 | 25/39 | — | — | 39 | — |
| Q8_0 | 25/39 | 0 | 0 | 35 | 1.0 |
| Q4_0 | 21/39 | 5 | 1 | 22 | 0.22 |

By stratum, Q4_0 lost one case each in ru, en, factual and long-context (lc 10→9). All 7 arithmetic cases fail in
every arm, because the model writes "5638 + 900 = 6538" and the frozen scorer takes only a bare number. That is
harness strictness, and it hits all arms the same way.

Qwen3.5-2B weight configs on GPU, Q8_0 as reference (ctx 8192):

| Weights | Pass | Lost vs Q8 | Gained | Identical outputs | p |
|---|---|---|---|---|---|
| BF16 | 27 | 1 | 0 | 36 | 1.0 |
| Q8_0 | 28 | — | — | 39 | — |
| Q6_K | 28 | 2 | 2 | 32 | 1.0 |
| Q5_K_M | 25 | 3 | 0 | 33 | 0.25 |
| Q4_K_M | 27 | 3 | 2 | 28 | 1.0 |
| Q3_K_M | 20 | 9 | 1 | 21 | **0.021** |
| Q2_K | 0 | 28 | 0 | 0 | <0.001 (38/39 hit the token limit) |

GPU and CPU runs give slightly different outputs — Q8_0 scores 28 on GPU against 29 on CPU — because the two
backends differ numerically. Every comparison stays inside one backend.

## R21. Decode and prefill speed vs KV type and filled depth (GPU, shared window, ABBA) — MEASURED

`scripts/gpu_speed_abba.sh` → `artifacts/gpu_speed_abba_20261021/` (48 llama-bench jsonl files, `summary.json`,
preflight, 1 s telemetry). Model Qwen3-4B Q4_K_M, llama-bench build 11552, `-ngl 99 -fa 1`, with a prefill of 512
tokens and a decode of 64 tokens on top of a filled depth d. The order was ABBA (F16, Q8, Q4 then Q4, Q8, F16) × 2
blocks with 3 repetitions per call, so there are **12 samples per cell**.
**Window: shared GPU (DIRTY).** The r04 training PID 35196 ran throughout. Telemetry: median utilization 97%, power
median 290 W / max 315 W, max temperature 77 °C, SM clock median 1845 / min 1560 MHz, throttle codes 0x4 (SW power
cap), 0x24, 0x48, 0x400, 0x404. We interpret only the ratios inside the interleaving; absolute tok/s will be higher
in a clean window.

| Test | Depth | F16 median tok/s (p5–p95) | Q8_0 | Q4_0 | Q8_0 / F16 | Q4_0 / F16 |
|---|---|---|---|---|---|---|
| decode | 0 | 108.1 (106.2–109.8) | 104.8 | 104.5 | 0.970 | 0.967 |
| decode | 4096 | 99.2 (96.9–101.1) | 87.4 | 86.6 | 0.881 | 0.874 |
| decode | 16384 | 74.8 (73.0–75.4) | 58.8 | 57.8 | 0.786 | 0.772 |
| decode | 32768 | 55.2 (54.0–55.7) | 41.2 | 39.7 | 0.746 | 0.719 |
| prefill | 0 | 4872 (4243–5004) | 4691 | 4492 | 0.963 | 0.922 |
| prefill | 4096 | 3731 (3281–3847) | 3486 | 3596 | 0.934 | 0.964 |
| prefill | 16384 | 2174 (2093–2262) | 2067 | 2064 | 0.950 | 0.949 |
| prefill | 32768 | 1427 (1365–1451) | 1327 | 1340 | 0.930 | 0.939 |

Decode CV is 0.8–2.1% per cell; prefill CV is 1.3–6.5%. At depth ≥ 4K the decode differences are 6–25× larger than
the noise.

This is a negative result, and a clean one: on this GPU and build, quantized KV is slower for decode, and the gap
widens with context (−12% at 4K, −21…−23% at 16K, −25…−28% at 32K). Prefill is 4–8% slower. The bandwidth roofline
predicts the opposite (Q8 KV about 1.3× faster at 16K and 1.45× at 32K, R15). Implied bytes moved per second at 32K:
F16 (2.49 GB weights + 4.83 GB KV per token) × 55.2 = 404 GB/s ≈ 43% of 936 GB/s; Q8 (2.49 + 2.57 GB) × 41.2 =
208 GB/s ≈ 22%. The quantized-KV attention path is limited by dequantization and kernel efficiency, not by memory
bandwidth. Fewer bytes did not mean more speed. The same mechanism explains why PAL3 and any non-fused decoder lose
speed.

## R22. PoC-B: CUDA read microbenchmark F16 vs Q8_0 vs Q4_0 vs lossless EXP3 — MEASURED (GPU, shared)

`release/poc/kvread_bench.cu` (nvcc 11.8, sm_86) → `artifacts/poc_20261021/kvread_bench.jsonl` (+ 1 s
telemetry). The kernel makes one decode-attention-shaped pass over T tokens of one layer: a K pass (scores for 32
query heads, GQA 4, one warp per 128-value row) and a V pass (weighted sum, block reduction). The data is real
captured Qwen3-4B KV, with all 36 layers tiled to T tokens. Each format is decoded on the device first and then
checked: **EXP3 bit-exact at every T**; Q8_0 relative RMS error 0.012, Q4_0 0.094–0.107. Timing used 20
back-to-back launches per sample and 12 ABBA rounds, median shown. The GPU was shared with r04, so the min–max
ranges are wide (F16 at 32K spans 0.44–1.08 ms).

| T | F16 ms (GB/s) | Q8_0 ms / vs F16 | Q4_0 ms / vs F16 | EXP3 ms / vs F16 | EXP3 bytes ratio | EXP3 escapes |
|---|---|---|---|---|---|---|
| 4096 | 0.0792 (212) | 0.0789 / 1.00× | 0.0749 / 1.06× | 0.1201 / 0.66× | 0.960 | 10.7% |
| 8192 | 0.1422 (236) | 0.1186 / 1.20× | 0.1142 / 1.25× | 0.1818 / 0.78× | 0.943 | 7.3% |
| 16384 | 0.2648 (254) | 0.2556 / 1.04× | 0.2510 / 1.05× | 0.3636 / 0.73× | 0.937 | 6.1% |
| 32768 | 0.4846 (277) | 0.4927 / 0.98× | 0.4947 / 0.98× | 0.7218 / 0.67× | 0.931 | 5.0% |

(Speed ratio = F16 time / format time; above 1 means faster.)

What we take from it:
- This simple kernel tops out at 212–277 GB/s for F16 (23–30% of peak), limited by per-row instruction work — decode
  plus 4 warp reductions per row — rather than by bytes. **That is why Q4_0, at 28% of the bytes, is not faster.**
  The llama.cpp FA kernels (R21) sit in the same regime.
- **The lossless EXP3 format costs 1.3–1.5× the time of F16** for a 4–7% byte saving: about 3.5 ps of extra decode
  per element at 32K. Variable-length bit extraction, escape prefix scans and exponent rebuild cost more than the
  bandwidth they save.
- On real KV in this GPU layout (bases per 32 values inside a row), EXP3 saves only 4–7%, below the 9.3% CPU
  estimate (R11). The CPU estimate grouped one channel over 32 tokens (escape rate 1.8–2.3%); row grouping sees
  5–11% escapes.
- **The condition for building a fused attention decoder — format decodes faster than the Q8 path — is NOT MET.**
  We built no fused kernel; see paper §7.

## R23. PoC-A: lazily mapped KV pool — MEASURED (GPU)

`release/poc/lazy_pool.cu` → `artifacts/poc_20261021/lazy_pool.jsonl`. It uses Qwen3-4B KV geometry (147,456 B per
token), context 32768, CUDA VMM at 2 MiB granularity. Every mapped byte was written. Per-process LOCAL usage came
from DXGI on the CUDA-matched adapter, measured against a 222 MiB process baseline.

| Arm | Filled 0 | 1024 | 4096 | 16384 | 32768 |
|---|---|---|---|---|---|
| eager `cudaMalloc` (today's behaviour) | 4608 MiB at any fill | | | | |
| lazy, one shared pool | 0 | 144 | 576 | 2304 | 4608 MiB |
| lazy, 72 per-tensor pools (K/V per layer) | 0 | 144 | 576 | 2304 | 4608 MiB |

Process memory tracks the filled tokens exactly, and reserving virtual address space costs nothing. Granularity
waste is zero here only because the fills are multiples of 1024 tokens (2048 B per token per tensor × 1024 = 2 MiB);
arbitrary fills waste up to 2 MiB per pool, so ≤ 144 MiB across 72 pools. This is a standalone allocator PoC and is
**not integrated into llama.cpp**: FORK_PLAN F1 is still open and we did not modify the fork.

## R24. Calibrated projections for other GPUs (*) — ANALYTIC, NOT MEASURED

`scripts/projections.py` → `artifacts/projections_20261021.md`. The model is calibrated on the 3090 measurements:
effective bandwidth 0.43 (llama.cpp F16 decode at 32K), 0.22 for its Q8 KV path, EXP3 extra decode cost
3.53 ps/element and a bytes ratio of 0.931 (PoC-B). Decode cost scales with scalar FP32 throughput relative to the
3090. **Every row marked * is an analytic projection that still needs validating on that hardware.**

| Card | BW GB/s | Ridge FLOP/B (FP16 tensor) | Scalar FLOP/B | EXP3 lossless KV, non-fused | EXP3, fused/overlapped | KV capacity lossless / Q8 / Q4 | Confidence |
|---|---|---|---|---|---|---|---|
| RTX 3090 (model; PoC-B measured 0.67×) | 936 | 76 | 38 | 0.61× | 1.07× | 1.07 / 1.88 / 3.56× | high |
| RTX 5090* | 1792 | 117 | 58 | 0.72× | 1.07× | same | medium |
| L40S* | 864 | 419 | 106 | 0.84× | 1.07× | same | low |
| RTX PRO 6000 Blackwell* | 1792 | 140 | ? | ? | ? | same | low |
| A100 80GB* | 2039 | 153 | 10 | 0.27× | 0.35× | same | medium |
| H100 SXM* | 3350 | 295 | 20 | 0.44× | 0.74× | same | medium |
| H200* | 4800 | 206 | 14 | 0.35× | 0.52× | same | medium |
| B200* | 8000 | 281 | ? | ? | ? | same | low |

Server cards are even more memory-bound at the tensor ridge point, but they run *fewer* scalar operations per byte
than the 3090. A scalar, bit-level lossless KV decoder is therefore projected to *lose* even when fused on
A100/H100/H200, and only the capacity gain carries over. A server-side speed gain would need tensor-core or
fixed-function decompression [verify: Blackwell decompression engine], or a much cheaper format. We do not project
Q8/FP8 KV in server stacks, because the slowdown we measured belongs to llama.cpp's kernel.

## R25. Long-context decode/prefill speed, near-clean window — MEASURED (supersedes R21 for absolute numbers)

`scripts/gpu_long_batch.sh` → `artifacts/gpu_long_20261021/speed/` and `summary.md/json`. Run 2026-10-21,
19:46–20:11. Other GPU jobs were stopped and r04 had finished. Preflight showed no compute jobs; the LM Studio desktop app
was open with no model loaded (GPU 29–33 W idle, 13–21% desktop utilization). During the runs the GPU drew a median
313 W (power-capped; throttle codes 0x4/0x20/0x400 seen). Model Qwen3-4B Q4_K_M, llama-bench build 11552, `-fa 1`,
prefill 512 and decode 64 on top of a filled depth d. ABBA order, 12 samples per cell at 4K/32K/64K; at 120K only
one ABBA block, so 6 samples for Q8/Q4 and **3 for F16**, because the first F16 call at 120K died silently (exit
127, no message) and the repeat worked. We used no rope scaling. llama-bench does not cap at the model's 40,960
trained context, so the speed numbers are valid, but answer quality beyond 40,960 is undefined without YaRN.

| Depth | F16 decode tok/s (p5–p95) | Q8_0 tok/s / vs F16 | Q4_0 tok/s / vs F16 | Prefill F16 tok/s | Prefill Q8_0 / Q4_0 vs F16 |
|---|---|---|---|---|---|
| 4096 | 162.2 (158.8–171.7) | 143.2 / 0.883 | 144.2 / 0.889 | 6030 | 0.944 / 0.917 |
| 32768 | 88.2 (84.4–89.8) | 64.6 / 0.732 | 63.9 / 0.725 | 2215 | 0.953 / 0.961 |
| 65536 | 59.1 (57.3–60.5) | 40.4 / 0.684 | 39.5 / 0.668 | 1316 | 0.935 / 0.945 |
| 122880 | 37.6 (37.0–37.7), n=3 | 23.6 / 0.629 | 23.3 / 0.622 | 770 | 0.920 / 0.913 |

Decode CV 0.7–2.7%. Implied DRAM traffic per decoded token = 2.49 GB of weights + KV(d):

| Depth | F16 GB/s (% of 936) | Q8_0 GB/s | Q4_0 GB/s |
|---|---|---|---|
| 32768 | 646 (69%) | 327 (35%) | 246 (26%) |
| 122880 | 775 (83%) | 286 (31%) | 177 (19%) |

**The slowdown from quantized KV grows with context: −11% at 4K, −27% at 32K, −32…−33% at 64K, −37…−38% at 120K.**
At long context the F16 path gets close to the bandwidth limit while the quantized paths use a falling fraction of
it, so dequantization and compute hold them back.

## R26. PoC-B in the near-clean window, up to 122,880 tokens — MEASURED

`artifacts/gpu_long_20261021/poc_b.jsonl` (same binary as R22, REPS=12, ITER=10). EXP3 came back bit-exact at
every T.

| T | F16 ms (GB/s) | Q8_0 vs F16 | Q4_0 vs F16 | EXP3 vs F16 | EXP3 bytes ratio |
|---|---|---|---|---|---|
| 4096 | 0.0487 (344) | 1.07× | 1.12× | 0.73× | 0.960 |
| 32768 | 0.2566 (523) | 1.01× | 1.02× | 0.68× | 0.931 |
| 65536 | 0.5184 (518) | 1.00× | 1.08× | 0.65× | 0.928 |
| 122880 | 0.9498 (530) | 0.99× | 1.01× | 0.65× | 0.929 |

Without the foreign load our simple kernel reaches 520–530 GB/s (57% of peak) for F16. Q8/Q4 are still no faster,
even with 47–72% fewer bytes. EXP3 decode costs (1.4554 − 0.9498) ms / 251.7 M elements = **2.01 ps per element** in
the clean run, against 3.53 ps in the shared run R22.

## R27. Needle-in-a-haystack at 4K and 32K — MEASURED

`artifacts/gpu_long_20261021/niah/niah.json`. llama-server build 11552, greedy, thinking off. The needle is a
sentence carrying a unique 4-digit code placed at depths 10/30/50/70/90%, and the haystack is the fixed corpus.

| KV | 4K (3,610 tokens) | 32K (29,783 tokens) |
|---|---|---|
| F16 | 5/5 | 5/5 |
| Q8_0 | 5/5 | 5/5 |
| Q4_0 | 5/5 | 5/5 |

Single-needle retrieval this simple does not separate the KV types at these lengths. KL (R29) is the more sensitive
measure.

## R28. Per-process VRAM at 64K / 120K context: ALLOCATED, not filled — MEASURED

`artifacts/gpu_long_20261021/memory_filled/rows.json`, slots at ctx 65536 and 122880. Without rope scaling,
llama-cli caps the slot at the trained context (`n_ctx_slot = 40960`) and rejected the 58K/114K prompts ("request
exceeds the available context size"). The KV buffers were still allocated for the full `-c`, which is what these
numbers show. Repeats were identical.

| ctx | F16 MiB (KV buffer) | Q8_0 MiB (KV) | Q4_0 MiB (KV) | Compute buffer F16 / quantized |
|---|---|---|---|---|
| 65536 | 11888 (9216) | 7814 (4896), −34.3% | 5510 (2592), −53.7% | 34.75 / 278.58 |
| 122880 | **19966 (17280)** | 12336 (9180), −38.2% | 8016 (4860), −59.9% | 48.75 / 516.58 |

**F16 KV at 120K fits on the 24 GB card**: 19,966 MiB per process with nothing else on the GPU. The compute buffer
for quantized KV grows roughly linearly with context (23.6 → 74.6 → 142.6 → 278.6 → 516.6 MiB from 4K to 120K) and
eats about 468 MiB of the saving at 120K. The truly filled 64K/120K runs with YaRN are in R30, and they match these
allocated values within 14–16 MiB.

## R29. KV KL at context 32768 on GPU — MEASURED

`artifacts/gpu_long_20261021/kld_*_c32768.log`. This is one 32,768-token chunk of the tripled corpus. Scoring covers
the second half of the chunk, which partly repeats text seen about 25K tokens earlier, so it also tests long-range
recall. Reference F16 KV (PPL 7.425 ± 0.194).

| KV | PPL ratio | Mean KLD | p99 KLD | Same top-1 |
|---|---|---|---|---|
| q8_0 | 1.0023 ± 0.0010 | 0.00128 ± 0.00012 | 0.0122 | 97.35 ± 0.13% |
| q4_0 | 1.0123 ± 0.0030 | 0.03487 ± 0.00171 | 0.3958 | 90.88 ± 0.23% |

We had only one chunk, and its repeated-text structure means the absolute KL is not comparable with R12/R19. The
ordering is the same: Q8_0 sits close to F16, and Q4_0 is about 27× higher in mean KL.

## R30. Per-process VRAM with a truly FILLED 64K / 120K context (YaRN) — MEASURED

`artifacts/gpu_long_20261021/memory_filled_yarn/rows.json`. Same procedure as R18, plus
`--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768`. Prompts of 57,443 and 114,444 tokens were processed,
with 2 interleaved repeats that were identical to the MiB.

| ctx (prompt tokens) | F16 MiB | Q8_0 MiB (vs F16) | Q4_0 MiB (vs F16) |
|---|---|---|---|
| 4096 (2,903) | 3246 | 2984 (−8.1%) | 2840 (−12.5%) |
| 32768 (24,772) | 7286 | 5246 (−28.0%) | 4094 (−43.8%) |
| 65536 (57,443) | 11902 | 7830 (−34.2%) | 5526 (−53.6%) |
| 122880 (114,444) | **19980** | 12352 (−38.2%) | 8032 (−59.8%) |

(4K/32K come from the R18/R25 runs, which were identical in both.) **F16 KV at a filled 120K context fits the 24 GB
RTX 3090: 19,980 MiB per process, with about 4.6 GB free.** Filling adds 14–16 MiB over the allocated level (R28).
The decode-speed numbers of R25 at 64K/120K are without YaRN: llama-bench does the same work either way, but quality
beyond 40,960 without YaRN is undefined.

## R32. Needle-in-a-haystack at ~94K tokens with YaRN — MEASURED

`artifacts/gpu_long_20261021/niah_100k_yarn/niah.json`. 93,581 prompt tokens with needles at 10/50/90%. F16 3/3,
Q8_0 3/3, Q4_0 3/3. Single-needle retrieval does not separate the KV types even at 94K, and that is too few cases
to say more than "no gross failure".

## R31. Recalibrated projections at 4K / 32K / 120K (*) — ANALYTIC

`scripts/projections_long.py` → `artifacts/projections_long_20261021.md`, fitted on the R25 decode data as
s/token = a + s_k·depth. The fit implies a weight-read efficiency of 0.48, an F16 KV-read efficiency of 0.92, and
extra decode time per token of depth in llama.cpp's kernels of 206 ns (Q8_0) and 254 ns (Q4_0); EXP3 adds 148 ns
non-fused (2.01 ps/elt, R26). The projection scales the memory part with bandwidth and the decode excess with
scalar FP32 throughput, so it describes "this llama.cpp kernel design on that GPU" and says nothing about FP8 KV
kernels in server stacks.

| GPU | ctx | F16 tok/s | Q8_0 vs F16 | Q4_0 vs F16 | EXP3 lossless non-fused / fused | Max ctx F16 / Q8 / Q4 |
|---|---|---|---|---|---|---|
| RTX 3090 (fit) | 4K / 32K / 120K | 159 / 89 / 37 | 0.92 / 0.73 / 0.63 | 0.92 / 0.72 / 0.63 | 0.92→0.62 / 1.01→1.06 | 132K / 249K / 471K |
| RTX 5090* | 4K / 32K / 120K | 304 / 171 / 72 | 0.97 / 0.87 / 0.80 | 0.97 / 0.89 / 0.84 | 0.95→0.72 / 1.01→1.06 | 185K / 349K / 659K |
| L40S* | 4K / 32K / 120K | 147 / 82 / 35 | 1.00 / 1.02 / 1.03 | 1.02 / 1.10 / 1.18 | see file | 291K / 548K / 1036K |
| A100 80GB* | 4K / 32K / 120K | see file | strongly < 1 (low scalar FLOP/B) | strongly < 1 | < 1 even fused | 503K / 947K / 1789K |
| H100 SXM* | 4K / 32K / 120K | see file | < 1 | < 1 | < 1 even fused | 503K / 947K / 1789K |
| H200* | 4K / 32K / 120K | see file | < 1 | < 1 | < 1 even fused | 907K / 1708K / 3226K |
| B200* | — | ? (scalar figure unknown) | ? | ? | ? | 1245K / 2343K / 4427K |

(* = analytic projection, to be validated on that hardware; full numbers in the artifact.) The capacity column is
arithmetic for Qwen3-4B (native context 40,960, 131,072 with YaRN [verify]).

## R33. MAIN STRUCTURE: compression ON TOP OF each KV baseline (F16, Q8_0, Q4_0)

F16 / Q8_0 / Q4_0 are the BASELINES, meaning formats that already exist. "Ours" is our compression sitting on top
of one of them, and every ratio is relative to that row's own baseline. Sources:
`artifacts/kv_quant_lossless_20261021/results.json` (CPU, bit-exact codecs above Q8_0/Q4_0, where the base is the
real captured KV requantized under llama.cpp's block rules), `artifacts/kv_lossy_over_base_20261021/` (error
proxies), and `artifacts/gpu_bases_20261021/poc_c.jsonl` (GPU read microbenchmark of base against base+ours, same
kernel shape as PoC-B, T = 4K/32K/120K, 12 ABBA rounds; near-clean window with the LM Studio UI open and one
single-threaded background CPU job). Read speed there is a microbenchmark of one layer's K and V pass, not llama.cpp
decode, because our formats have no llama.cpp kernel. **KL/top-1/tasks for our KV variants are not measured** — that
would take a custom KV type in llama.cpp — so we give error proxies instead.

### KV base F16 (2 bytes per value)

| Row | Size vs base | Read speed vs base (4K / 32K / 120K) | Quality | Status |
|---|---|---|---|---|
| F16 (baseline) | 1.000 | 1.00 | reference | measured |
| + ours lossless: byte planes + zlib, 256-token pages | 0.864 | not measured on GPU (CPU decode 159 MB/s) | bit-exact | CPU prototype |
| + ours lossless: byte planes + LZMA, 256-token pages | 0.847 | not measured on GPU (CPU 24 MB/s) | bit-exact | CPU prototype |
| + ours lossless: EXP3 fixed-rate, random access | 0.929–0.960 | 0.73 / 0.68 / 0.65 (R26) | bit-exact (GPU check) | measured |
| + ours lossy: drop 4 mantissa bits (12-bit packed) | 0.797 (0.75 without row header) | 0.85 / 0.98 / 0.93 | rel. error 0.62% of F16; KL not measured | measured (GPU read) + proxy |
| + ours lossy: drop 2 mantissa bits | 0.875 | not measured | rel. error 0.13% | proxy |

### KV base Q8_0 (8.5 bits per value)

| Row | Size vs base (K / V) | Read speed vs base | Quality | Status |
|---|---|---|---|---|
| Q8_0 (baseline) | 1.000 | 1.00 | KL vs F16 KV 0.0013 (32K)–0.0042 (2K); tasks 25/39 = F16 | measured |
| + ours lossless: split scale/code planes + zlib, 256-token pages | 0.903 / 0.930 | not measured on GPU | bit-exact | CPU prototype |
| + ours lossless: split planes + LZMA, 256-token pages | 0.892 / 0.931 | not measured on GPU | bit-exact | CPU prototype |
| + ours lossless: static Huffman on codes (per layer) | 0.917 / 0.955 | not measured | exact (size from code lengths) | CPU size |
| + ours lossless: fixed-rate 7-bit + escape | **1.063 / 1.115 (larger)** | not run | exact | measured, negative (16–22% escapes) |
| + ours lossy: codes rounded to 7 bits | 0.912 | 1.00 / 0.98 / 0.94 | error 2.3–2.6× the base error; KL not measured | measured (GPU read) + proxy |
| + ours lossy: 6-bit dense + 1% outliers exact | 0.802 | not measured | error 2.1× (K) / 3.3× (V) the base error | proxy |
| + ours lossy, quality up: per-channel K scales | 1.000 | not measured | K error 0.43× the base error | proxy |

### KV base Q4_0 (4.5 bits per value)

| Row | Size vs base (K / V) | Read speed vs base | Quality | Status |
|---|---|---|---|---|
| Q4_0 (baseline) | 1.000 | 1.00 | KL vs F16 KV 0.035 (32K)–0.069 (4K); tasks 21/39 | measured |
| + ours lossless: split planes + zlib, 256-token pages | 0.828 / 0.892 | not measured on GPU | bit-exact | CPU prototype |
| + ours lossless: split planes + LZMA, 256-token pages | **0.790 / 0.869** | not measured on GPU | bit-exact | CPU prototype |
| + ours lossless: static Huffman on 4-bit codes | 0.869 / 0.935 | not measured | exact (size) | CPU size |
| + ours lossless: fixed-rate 3-bit + escape (GPU) | **1.17–1.25 (larger)** | 0.80 / 0.75 / 0.73 | bit-exact vs base (GPU check) | measured, negative (19–23% escapes) |
| + ours lossy: 3-bit dense + 1% outliers | 0.849 | not measured | error 1.5× (K) / 2.1× (V) the base error | proxy |
| + ours lossy, quality up: 4-bit + 1% outliers | 1.071 | not measured | error 0.59× (K) / 0.81× (V) | proxy |
| + ours lossy, quality up: per-channel K / per-token V 4-bit (KIVI-style) | 1.111 | not measured | K error 0.32×, V 0.87× | proxy |

In relative terms, lossless coding does better on top of a quantized base than on F16 (Q4_0 K −21%, Q8_0 K −11%,
F16 −15%), but only with the variable-rate coders we have not implemented on GPU. Every fixed-rate, random-access
variant we tried is bigger than its base (Q8_0, Q4_0) or slower (EXP3 on F16).

## R34. MAIN STRUCTURE: compression ON TOP OF each weight baseline (BF16, Q8_0, Q4_K_M)

Model: Qwen3.5-2B (the local BF16 checkpoint plus our own GGUF quantizations). "Ours, lossy" means our
mixed-precision recipes built with `llama-quantize --tensor-type` and the like, so they run inside llama.cpp and are
measured end to end. Quality is measured against Q8_0 logits (fixed corpus, context 512) and on the 39-case suite on
GPU. Speed is llama-bench decode tok/s at depth 0 and 4096, KV F16, ABBA, 6 samples per cell
(`artifacts/gpu_bases_20261021_rerun/w/`, near-clean window, CPU quiet). The lossless rows come from
`artifacts/weights_lossless_20261021/` (CPU, frame round trip checked); no GPU decoder exists for them.

### Weights base BF16 (3706 MiB tensors)

| Row | Size vs base | Decode speed vs base (d=0 / 4K) | Quality (KL vs Q8_0, top-1, tasks/39) | Status |
|---|---|---|---|---|
| BF16 (baseline) | 1.000 | 1.00 (152.5 / 157.4 tok/s) | 0.0013, 96.1%, 27 | measured |
| + ours lossless: exponent Huffman, sign+mantissa raw (DFloat11-style) | **0.666** | not measured (no GPU decoder) | exact (size from code lengths) | CPU size |
| + ours lossless: split byte planes + zlib, 64 KiB frames | 0.707 | not measured | bit-exact | CPU prototype |
| + ours lossy: FFN tensors to Q8_0, rest BF16 | 0.770 | **1.195 / 1.167** | 0.0013, 96.2%, 28 | measured |

### Weights base Q8_0 (1970 MiB)

| Row | Size vs base | Decode speed vs base | Quality | Status |
|---|---|---|---|---|
| Q8_0 (baseline) | 1.000 | 1.00 (207.3 / 219.2 tok/s) | reference (0), 100%, 28 | measured |
| + ours lossless: split header/code planes + zlib, 64 KiB frames | 0.944 | not measured (no GPU decoder) | bit-exact | CPU prototype |
| + ours lossless: split planes + LZMA, 256 KiB frames | 0.945 | not measured | bit-exact | CPU prototype |
| + ours lossless: byte Huffman on codes | 0.961 | not measured | exact (size) | CPU size |
| + ours lossy: FFN tensors to Q6_K | 0.889 | **1.055 / 1.036** | 0.0046, 90.8%, 29 | measured |

### Weights base Q4_K_M (1241 MiB)

| Row | Size vs base | Decode speed vs base | Quality | Status |
|---|---|---|---|---|
| Q4_K_M (baseline) | 1.000 | 1.00 (248.7 / 271.4 tok/s) | 0.0602, 78.9%, 27 | measured |
| + ours lossless: split planes + zlib, 64 KiB frames | 0.978 | not measured | bit-exact | CPU prototype |
| + ours lossless: split planes + LZMA | 0.980 | not measured | bit-exact | CPU prototype |
| + ours lossless: CQW2 palette (GPU-shaped, Qwen3-4B, R3) | **1.006 (larger)** | not measured | bit-exact | measured, negative |
| (order-0 entropy ceiling, Qwen3-4B Q4_K tensors, R4) | 0.949 | — | — | bound |
| + ours lossy: token embedding + output to Q4_K | 0.899 | **1.110 / 1.048** | 0.0767 (1.27× base KL), 74.2%, 28 | measured |
| + ours lossy: FFN up/gate to Q3_K | 0.936 | **0.982 / 0.950 (slower)** | 0.1254 (2.1×), 74.7%, 25 | measured, negative |

The reference rows add no new runs, only earlier measurements: Q6_K, Q5_K_M, Q3_K_M, Q2_K weights (R7, R20); q5_0,
q5_1, q4_1, iq4_nl KV (R12). We also got one lossless measurement on top of Q3_K_M weights by accident: 0.986
(split zlib).

## R35. COMBO: weights + KV (base vs base + ours)

Measured with llama-bench (Qwen3.5-2B; only 6 of its 24 layers carry a KV cache, so KV effects stay small here;
`artifacts/gpu_bases_20261021_rerun/c/`, 6 samples per cell). Decode tok/s at depth 4096 / 32768:

| Combo | Base tok/s | + ours (lossy weight recipe; KV unchanged) | Speed vs base | Weight size vs base |
|---|---|---|---|---|
| BF16 + F16 KV | 151.1 / 142.8 | BF16+FFN-Q8 + F16 KV: not run in combo (weights alone: 1.17–1.20×) | — | 0.770 |
| Q8_0 + Q8_0 KV | 218.3 / 194.4 | Q8_0+FFN-Q6K + Q8_0 KV: 220.0 / 196.3 | 1.008 / 1.010 | 0.889 |
| Q4_K_M + Q4_0 KV | 256.4 / 225.0 | Q4_K_M+emb-Q4K + Q4_0 KV: 270.6 / 231.5 | 1.055 / 1.029 | 0.899 |

Our lossless KV and weight formats cannot run inside llama.cpp yet, so any combo that uses "our lossless compression
of both" is **analytic only**. Using R31's calibration for Qwen3-4B (KV in all layers), a lossless KV ratio of
0.85–0.93 and weights of 0.94–0.98 change memory by those factors. Decode speed would fall unless the decoder is
fused and cheaper than about 2 ps/element (R26). Combo quality (KL/tasks) is not measured. For full-KV combos on
Qwen3-4B, the measured baselines are in R25/R30 (Q4_K_M weights + F16 / Q8_0 / Q4_0 KV).

## R36. Stage 1: MEASURED quality of our lossy KV variants on top of each base (exact emulation in llama.cpp)

> **Correction (R44):** in this series the rope/rms_norm+set_rows fusion of F16 caches went around the emulation hook,
> so only V was emulated. The KL, task and needle values below describe V-only quantization. See R44/R45 for K+V.

Branch `cq/kv-compress` (worktree `D:\Git\llama.cpp-kvc`, from pin `ed7ac35e`, commit `d435179`, not pushed), patch
`release/patches/0001-cq-kv-emu-set-rows.patch`. An env-gated hook (`CQ_KV_EMU`) in CUDA `set_rows` sends every K/V
row written to the F16 cache through the variant's quantize→dequantize, so attention sees exactly the values the
packed format would decode to, plus the final F16 rounding of each dequantized value. **Quality is real; memory and
speed of a packed format are not represented.** Model Qwen3-4B Q4_K_M on GPU, KL via llama-perplexity (4K: 6 chunks;
32K: one chunk of the tripled corpus). Tasks: 39 frozen cases, ctx 16384. Needle: 5 depths at 4K and at 32K. Data in
`artifacts/stage1_emu_20261021/summary.md/json`, logs, task records.

| Variant | Base | KL vs F16 KV, 4K / 32K | top-1 vs F16, 4K / 32K | KL vs own base, 4K / 32K | Tasks /39 (F16 = 25) | Needle 4K / 32K |
|---|---|---|---|---|---|---|
| F16 drop 2 mantissa bits | F16 | 0.00249 / 0.00125 | 95.86% / 97.57% | — | 25 | 5/5 / 5/5 |
| F16 drop 4 mantissa bits | F16 | 0.00267 / 0.00256 | 96.00% / 97.66% | — | 25 | 5/5 / 5/5 |
| q8_0 emulated (base check) | Q8_0 | 0.00267 / 0.00122 | 95.95% / 97.38% | — | 25 | 5/5 / 5/5 |
| real llama.cpp q8_0 type | Q8_0 | 0.00268 / 0.00128 | 95.86% / 97.35% | — | 25 (R20) | 5/5 / 5/5 (R27) |
| q8_0 + 7-bit codes | Q8_0 | 0.00287 / 0.00281 | 95.75% / 97.39% | 0.00300 / 0.00272 | 25 | 5/5 / 5/5 |
| q8_0 6-bit + 1% outliers | Q8_0 | 0.00276 / 0.00178 | 95.61% / 97.33% | 0.00300 / 0.00160 | 25 | 5/5 / 5/5 |
| q4_0 emulated (format only) | Q4_0 | **0.00799 / 0.00515** | 92.98% / 95.38% | — | 24 | 5/5 / 5/5 |
| real llama.cpp q4_0 type | Q4_0 | **0.06939 / 0.03487** | 84.45% / 90.88% | — | 21 (R20) | 5/5 / 5/5 (R27) |
| q4_0 + 1% outliers exact | Q4_0 | 0.00627 / 0.00374 | 93.92% / 96.34% | 0.01100 / 0.00661 | 25 | 5/5 / 5/5 |
| 3-bit + 1% outliers exact | Q4_0 | 0.02321 / 0.01179 | 88.97% / 93.04% | 0.02889 / 0.01487 | 25 | 5/5 / 5/5 |

Findings:
- The emulation reproduces the real q8_0 type (KL 0.00267 against 0.00268 at 4K), and that is what validates the
  method.
- **The Q4_0 format itself costs KL 0.008, but llama.cpp's real q4_0 KV path costs 0.069 (8.7×) at 4K and 6.8× at
  32K, plus 4 tasks.** Most of the q4_0 KV quality loss therefore comes from the computation path, not from the
  stored values. An integer dot product with a quantized query in the vec kernel is a plausible cause, but that is a
  hypothesis and we did not test it.
- Dropping 4 mantissa bits from F16 (12 bits in all) is as good as Q8_0, with KL 0.0027 at 4K.
- Keeping 1% of values exact on top of 4-bit gives lower KL against F16 than plain Q4_0 (0.0063 against 0.0080) for
  +7% size; 3-bit with outliers costs about 3× the Q4_0 KL.
- No variant moves task accuracy by more than ±1 case, and needle retrieval is perfect for all of them. These short
  tasks cannot separate the variants, while KL can.
- Per-channel K scales (KIVI-style) need token groups and cannot be emulated per row, so they are not measured.

## R37. Clean-window speed reruns (LM Studio closed, other GPU jobs stopped, no CPU jobs) — MEASURED

`artifacts/clean_reruns_20261021/`. Preflight found no compute processes and the desktop sat at 10–11%
utilization; the card stayed power-capped at about 312 W (throttle 0x4/0x400). This is the cleanest window this
desktop can give us.

Qwen3-4B Q4_K_M KV (llama-bench build 11552, ABBA; 12 samples at 4K/32K, 6 at 120K):

| Depth | F16 decode tok/s | Q8_0 / F16 | Q4_0 / F16 | Prefill Q8_0 / F16 | Prefill Q4_0 / F16 |
|---|---|---|---|---|---|
| 4096 | 166.2 | 0.890 | 0.887 | 0.951 | 0.955 |
| 32768 | 91.2 | 0.727 | 0.725 | 0.954 | 0.959 |
| 122880 | 38.1 | 0.640 | 0.628 | 0.964 | 0.961 |

These confirm R25 within 1–2%.

Qwen3.5-2B weight recipes (decode tok/s at depth 0 / 4096; 6 samples):

| Base → + ours | Base tok/s | Ours tok/s | Speed vs base |
|---|---|---|---|
| BF16 → FFN Q8_0 | 164.5 / 166.7 | 190.8 / 196.8 | **1.160 / 1.181** |
| Q8_0 → FFN Q6_K | 230.7 / 239.8 | 240.6 / 253.8 | 1.043 / 1.058 |
| Q4_K_M → embedding Q4_K | 286.3 / 297.2 | 293.3 / 310.8 | 1.024 / 1.046 |

(The Q4_K_M recipe gains less here than in the earlier near-clean run, R34: 1.11 / 1.05.)

## R38. PoC-D: lossless on top of BF16 weights inside a GEMV — MEASURED (clean window)

`release/poc/bf16_gemv_bench.cu` → `artifacts/clean_reruns_20261021/poc_d_bf16_gemv.jsonl`. It uses real
Qwen3.5-2B FFN-up weights (49,152 × 2,048 BF16) and a fixed-rate format: base exponent per 32, 3-bit offset plus
escape, sign and mantissa raw, 11 bits per weight. That is close in spirit to ZipServ TCA-TBE, but with non-fused
scalar decode.

| Format | Size vs BF16 | GEMV time vs BF16 | Notes |
|---|---|---|---|
| BF16 base | 1.000 | 1.000 (0.246 ms, 818 GB/s = 87% of peak) | — |
| + ours lossless fixed-rate | **0.716** (escape rate 2.3%) | **0.685× speed** (0.360 ms) | decode bit-exact, GEMV output identical |
| + exponent Huffman (CPU size only, R34) | 0.666 | — | no GPU decoder |

28% fewer bytes did not make our scalar decoder faster than a well-optimized BF16 GEMV. ZipServ reports the
opposite with a fused, tensor-core-oriented design; our result covers non-fused scalar decoding only.

## R39. E2: llama.cpp PRs #30792 / #30202 vs their own base commits — MEASURED (clean window)

Builds (llama-bench only, CUDA 11.8, sm_86, separate worktrees): #30792 head `3c253754` against base `99db719b`;
#30202 head `b9e92c43` against base `dac30873`. Model Qwen3-4B Q4_K_M, `-fa 1`, prefill 512 / decode 64 at depth d,
order base–PR–PR–base with 3 repetitions each, so 6 samples per cell. Window: other GPU jobs stopped, LM Studio closed, no
CPU jobs. Data in `artifacts/window2_20261022/e2/` (jsonl, `summary.md/json`, preflight, telemetry).

Decode speed of each PR relative to its own base (prefill stayed within ±1% in every cell):

| KV | depth | #30792 PR / base | #30202 PR / base |
|---|---|---|---|
| f16 | 4K / 32K / 120K | 1.003 / 0.995 / 0.997 | 0.998 / 1.000 / 1.016 |
| q8_0 | 4K / 32K / 120K | 1.003 / 1.006 / 1.002 | 1.030 / **1.255 / 1.361** |
| q4_0 | 4K / 32K / 120K | 0.995 / 1.007 / 1.008 | 0.993 / **1.130 / 1.201** |

Quantized against F16 decode inside the #30202 builds:

| depth | base: q8_0 / f16 | PR: q8_0 / f16 | base: q4_0 / f16 | PR: q4_0 / f16 |
|---|---|---|---|---|
| 4096 | 0.886 | 0.915 | 0.883 | 0.879 |
| 32768 | 0.733 | **0.919** | 0.726 | 0.821 |
| 122880 | 0.642 | **0.860** | 0.636 | 0.752 |

On the RTX 3090 with CUDA, PR #30202 (on-the-fly dequantization of q8/q4 KV in the vec kernels) recovers about 70%
of the q8_0 decode gap at long context (0.73 → 0.92 at 32K, 0.64 → 0.86 at 120K) and about a third of the q4_0 gap.
PR #30792 (native q8_0 KV in the FA tile kernel, measured upstream on ROCm) changes nothing on this GPU beyond
±1%. Even with #30202, quantized KV is still slower than F16 at long context. Cell CVs are 0.1–3.6%.

## R40. E1 / E3: context-aware lossless estimate for F16 KV, and int4 granularity — MEASURED (CPU)

E1 (`scripts/kv_e1_context.py` → `artifacts/e1_context_20261022/capture0.json`; existing Qwen3-4B capture of 2304
tokens, every 6th layer). Bits per value:

| Model of the data | K | V |
|---|---|---|
| order-0 16-bit word | 13.81 | 13.58 |
| exponent order-0 / given channel / given previous token / given both | 2.88 / 2.50 / 2.59 / **2.38** | 2.68 / 2.59 / 2.64 / **2.54** |
| sign / sign given channel | 1.00 / 0.91 | 1.00 / 0.96 |
| mantissa: top 3 bits given exponent / low 7 bits | 2.95 of 3 / 7.00 of 7 | 2.94 / 7.00 |
| **best context estimate** | **13.23 (0.827)** | **13.43 (0.840)** |
| TRACE-style channel-major groups (G=16/64/256), bit planes + zstd-19, 4 KiB chunks (bit-exact) | 0.842 / 0.847 / 0.851 | 0.863 / 0.866 / 0.872 |
| same with lz4 | 0.892 / 0.885 / 0.886 | 0.907 / 0.899 / 0.900 |
| zstd-19 on channel-major bytes, 4 KiB | 0.920 | 0.871 |

For F16 KV of Qwen3-4B, context modelling only moves the estimate from about 14% to about 16–17%. The mantissa is
noise, which caps any lossless F16 KV codec at roughly 1 + 2.4 + 10 ≈ 13.2–13.4 bits. We do not reproduce TRACE's
1.8× on BF16 KV of Llama-3.1-8B for F16 KV here. R41 answers whether that comes from the format (5- vs 8-bit
exponent), the model (QK-norm in Qwen3) or the data: mostly the format, because BF16 reaches 0.64–0.66 on the same
activations. Whatever is left would need a model without QK-norm, and none is local.

E3 (`scripts/e3_int4_entropy.py` → `artifacts/e3_int4_entropy_20261022.json`; Qwen3.5-2B BF16, 40 tensors):

| int4 scheme on the same weights | code entropy, bits of 4 | ratio incl. scales |
|---|---|---|
| per-tensor symmetric | **0.80** | 0.199 |
| per-row symmetric | 2.88 | 0.720 |
| group-128 asymmetric (AWQ/GPTQ-like layout, RTN, no activation-aware step) | 3.60 | 0.906 |
| group-32 symmetric (Q4_0-like) | 3.59 | 0.908 |
| llama.cpp Q4_K (R4, Qwen3-4B) | — | 0.949 |

Low int4 entropy (0.6–1.0 bits, the kind Tan et al. report for "int4") shows up only with per-tensor scaling,
where most codes collapse. Grouped formats keep 5–10% headroom, which agrees with Tan et al.'s 1.1–1.3× for group
formats and with our 2–7% for k-quants. The two do not contradict each other.

## R41. E1 (more texts and lengths) and E4 (BF16 vs F16 KV) — MEASURED (captures on GPU, analysis on CPU)

Captures: llama-server slot save, Qwen3-4B Q4_K_M, `-fa on`, F16 or BF16 KV, three texts (code, English prose,
Russian prose), about 4K and about 30K tokens (window 2, `C:` temp, 15 GB, not in the repo). Analysis with
`scripts/kv_e1_context.py` (every 6th layer, 6 of 36), and the TRACE-like codec passed a bit-exact round trip. Data:
`artifacts/e1_context_20261022/captures_4k.json`, `captures_30k.json`.

Each column is a ratio to the 16-bit base (lower is better). "Context estimate" takes the best of: sign,
sign | channel, exponent given channel and/or previous token, top 3 mantissa bits given exponent, and low mantissa
bits order-0.

| KV dtype | text | tokens | K context est. | V context est. | K TRACE-like zstd G16 | V TRACE-like zstd G16 | K / V lz4 G16 |
|---|---|---|---|---|---|---|---|
| F16 | code | 4194 | 0.828 | 0.841 | 0.842 | 0.863 | 0.893 / 0.909 |
| F16 | en | 4300 | 0.829 | 0.842 | 0.841 | 0.863 | 0.892 / 0.908 |
| F16 | ru | 3478 | 0.828 | 0.842 | 0.842 | 0.863 | 0.893 / 0.910 |
| F16 | code | 28485 | 0.830 | 0.842 | 0.837 | 0.862 | 0.889 / 0.909 |
| F16 | en | 30184 | 0.829 | 0.843 | 0.835 | 0.862 | 0.887 / 0.908 |
| F16 | ru | 26099 | 0.828 | 0.842 | 0.836 | 0.862 | 0.888 / 0.909 |
| **BF16** | code | 4194 | **0.641** | **0.654** | **0.656** | **0.677** | 0.708 / 0.723 |
| **BF16** | en | 4300 | 0.641 | 0.655 | 0.654 | 0.676 | 0.706 / 0.723 |
| **BF16** | ru | 3478 | 0.640 | 0.654 | 0.655 | 0.677 | 0.708 / 0.724 |

Bits per value (code, 4K; the other texts stay within ±0.02):

| | F16 K | BF16 K | F16 V | BF16 V |
|---|---|---|---|---|
| exponent order-0 / best context | 2.90 / 2.40 | 2.90 / 2.40 | 2.68 / 2.56 | 2.68 / 2.56 |
| top 3 mantissa bits given exponent | 2.95 of 3 | 2.95 of 3 | 2.94 of 3 | 2.94 of 3 |
| remaining mantissa bits | 7.00 of 7 | 4.00 of 4 | 7.00 of 7 | 4.00 of 4 |

Findings:
- **E1 (texts, lengths):** over code, English and Russian and from about 4K to 30K tokens, the F16 KV estimate comes
  out at 0.828–0.830 (K) and 0.841–0.843 (V). The TRACE-like codec gains a little with length (K 0.842 → 0.836) and
  stays 1–2 points above the estimate. Lossless headroom for F16 KV of this model is **16–17%**, and inside this
  range neither text nor length moves it.
- **E4 (BF16 vs F16):** exponent and sign statistics are nearly identical (within 0.002 bits), because they describe
  the same activations. BF16 compresses much better only because it carries 3 fewer noise bits of mantissa: the
  context estimate is 0.64–0.655 (1.53–1.56×) and the TRACE-like codec reaches 0.654–0.677 (1.48–1.53×). That
  explains most of the gap to TRACE's 1.81–1.88× on BF16 KV — the dtype, not a weaker codec. What remains (1.53×
  against 1.81–1.88×) depends on model and data (TRACE: Llama-3.1-8B without QK-norm; ours: Qwen3-4B with QK-norm)
  and stays open until we can capture a model without QK-norm.
- Practical note: llama.cpp offers BF16 KV, but F16 is the default and both are 16 bits, so BF16 KV only helps if a
  lossless codec runs on top. In that case the codec would reach 0.65 of the base instead of 0.84.

## R42. Fork stage 2: real packed KV type f16d4 and a GQA-grouped decode kernel — FIRST MEASUREMENTS (interim)

Branch `cq/kv-compress` (commit `b82396db`, not pushed). New `GGML_TYPE_F16D4` (60): F16 with the 4 low mantissa bits
dropped (same rounding as the f16drop4 emulation), 12 bits per value, block of 32 = 48 bytes. The block holds a byte
plane of the high bytes and a nibble plane arranged so that 8 values decode with 2 loads, 2 ALU ops and 4 byte
permutes.
- Write: CUDA `set_rows` quantizer.
- Prefill: conversion to an f16 scratch, the path every quantized KV type takes.
- Decode: fused into the FlashAttention vector kernel.
- New kernel `flash_attn_ext_vec_gqa` (`fattn-vec-gqa.cu`): single-token decode that serves all query heads of one
  GQA group from one K/V read. It is the default for f16d4; `CQ_FA_VEC_GQA=1` also enables it for f16, q8_0 and
  q4_0, and `=0` turns it off.

Session 1 went to `artifacts/fork_s1_20261022/`. Qwen3-4B Q4_K_M on an RTX 3090, with the GPU lock held and the
other GPU jobs stopped. The preflight was still DIRTY: desktop applications (Parsec, a game client, browsers) held the GPU at
19–28% at idle and were not ours to close. The speeds are medians of 6 samples (2 rounds × r3) and are interim.

Quality, prefill path (KL against F16 KV, 4K, 6 chunks):

| KV | KL | same top-1 |
|---|---|---|
| f16d4 (real type) | 0.002647 | 95.62% |
| f16drop4 emulation (R36) | 0.0027 | — |
| q8_0 | 0.002680 | 95.86% |

The real type reproduces the emulated quality.

Decode tok/s relative to F16 KV in the same build (stock kernels for F16, q8_0, q4_0):

| KV / kernel | 4K | 32K |
|---|---|---|
| F16 (stock: MMA kernel with GQA) | 157.5 tok/s = 1.000 | 87.8 tok/s = 1.000 |
| **f16d4 + GQA vec kernel (ours)** | 0.939 | **1.029** |
| f16d4 + per-head vec kernel | 0.927 | 0.812 |
| F16 + GQA vec kernel | 0.972 | 0.978 |
| q8_0 (stock vec) | 0.883 | 0.736 |
| q8_0 + GQA vec kernel | 0.752 | 0.611 |
| q4_0 (stock vec) | 0.880 | 0.707 |
| q4_0 + GQA vec kernel | 0.759 | 0.609 |

Prefill relative to F16: f16d4 0.93–0.95, the same as q8_0 and q4_0, because all three convert to an f16 scratch.

Interim reading:
- At 32K the packed 12-bit KV decodes 3% faster than F16 while storing 25% fewer KV bytes, and 40% faster than q8_0.
- Grouping the query heads of a GQA group is what makes the difference; without it, f16d4 is as slow as the stock
  quantized path.
- On q8_0 and q4_0 our GQA kernel is slower than the stock vec kernel. Their dot product quantizes Q to q8_1 and
  needs 32-thread reductions per K row, while the f16-like layout needs 8.

## R43. Fork stage 2: decode-path correctness, tasks/needle with real f16d4, lazily mapped KV buffer — MEASURED

Session 2 (`artifacts/fork_s2_20261022/`): GPU lock held, other GPU jobs stopped. The desktop background was the same as in
R42, and we took no speed numbers this session.

Decode-path KL. `llama-perplexity -b 1 -ub 1` sends every token through the single-token FA kernel. 1 chunk of 4K,
compared against the F16 prefill base:

| KV / kernel | KL | same top-1 |
|---|---|---|
| F16, stock | 0.001678 | 96.55% |
| F16, GQA vec kernel (ours) | 0.001794 | 96.52% |
| f16d4, GQA vec kernel | 0.001926 | 96.26% |
| f16d4, per-head vec kernel | 0.001920 | 96.28% |
| q8_0, stock vec | 0.002108 | 96.54% |
| q8_0, GQA vec kernel (q8_1 path, before the half-math change) | 0.001985 | 96.43% |

The 0.0017 floor comes from batch-1 against batch-512 numerics in the weight matmuls, not from KV. Our kernels are
numerically equivalent to the stock ones. A batch-1 base for exact decode comparisons is scheduled.

Tasks and needle with the real f16d4 type (llama-server, decode path): 25/39 tasks, the same as F16 (R20/R36). Needle
retrieval was 10/10 at 4K and 32K.

Lazily mapped KV buffer. `CQ_KV_LAZY=1` creates the KV cache in a CUDA VMM buffer and maps rows [0, n_kv) per
ubatch. Process VRAM is DXGI local usage (`CQ_VRAM_REPORT=1`), measured at a fixed `-c 122880` (YaRN) with the prompt
filled to N tokens:

| KV | mode | 702 tok | 2,922 tok | 24,791 tok | 114,463 tok |
|---|---|---|---|---|---|
| F16 | eager (stock) | 20,118 | 20,120 | 20,118 | 20,118 |
| F16 | **lazy** | **2,982** | **3,272** | **6,438** | 18,966 |
| q8_0 | eager | 12,452 | 12,454 | 12,452 | 12,452 |
| q8_0 | lazy | 3,452 | 3,634 | 5,252 | 11,948 |
| q4_0 | eager | 8,132 | 8,134 | 8,132 | 8,132 |
| q4_0 | lazy | 3,416 | 3,526 | 4,388 | 7,916 |
| f16d4 | eager | 16,232 | 16,234 | 16,232 | 16,232 |
| f16d4 | lazy | 3,416 | 3,706 | 6,008 | 15,368 |

All values in MiB.

Readings:
- Inside the real llama.cpp KV cache, the lazy buffer makes process memory follow the filled tokens. F16 at 702 tokens
  of a 120K context drops from 20.1 GB to 3.0 GB. That confirms PoC-A (R23) inside llama.cpp; the technique itself
  is known (vAttention, kvcached).
- With a lazy buffer, the quantized types (q8_0, q4_0, f16d4) carry about 470 MiB more than F16 at small fills. The
  extra is the f16 scratch for prefill conversion, which the compute buffer reserves for the worst-case context
  (#24166). At 24.8K tokens it eats most of f16d4's saving (6,008 against 6,438 MiB).
- Fix in the fork: `CQ_FA_SCRATCH_POOL=1` takes the scratch from the lazily growing CUDA pool at run time. Still to
  be measured.

## R44. CORRECTION to R36: the emulation quantized V only; KL noise floor from kernel fusion — MEASURED

Sessions 3, 3b and 3c (`artifacts/fork_s3_20261022/`, `fork_s3b_*`, `fork_s3c_*`) used real packed types and the
emulation with CUDA fusions switched off (`GGML_CUDA_DISABLE_FUSION=1`).

1. **The R36 emulation never quantized K.** For F16/F32 caches, ggml-cuda fuses rope (and rms_norm+mul+rope) with
   `set_rows` (`ggml_cuda_should_fuse_rope_set_rows` accepts only F32/F16 destinations). The fused kernel writes K
   directly, so the `CQ_KV_EMU` hook in the `set_rows` kernel only ever saw V. With fusion off, emulated q4_0 costs
   **KL 0.116** instead of 0.0080 (4K, against F16 without fusion), and emulated q4 + 1% outliers costs 0.038 instead
   of 0.0063. We **withdraw** the R36 claim that "the Q4_0 format costs 0.008 and llama.cpp's q4_0 path 9× more":
   most of the q4_0 loss is K quantization. Stage 1 is being redone with fusion off (R45).
2. **A KL noise floor of about 0.0024.** Turning fusion off for F16 alone gives KL 0.00235 against the fused F16 base
   (4K, 6 chunks; same top-1 96.2%). Any pipeline that is numerically different but equivalent lands at this floor.
   That includes q8_0 (0.0027), f16d4 (0.0027–0.0028), the f16drop2/f16drop4 emulations (0.0025–0.0028) and even
   lossless f16e3 through the prefill path (0.0025–0.0026: identical KV values, but a different FA kernel
   configuration from F16, which goes through the non-quantized path). At this sample size, KL differences below about
   0.001 between such variants mean nothing.
3. **The f16e3 codec is exact.** A standalone CUDA round trip (`work/f16e3_test/`) on 8.4M real KV values (Qwen3-4B,
   4 layers × 1024 tokens, K and V) gave 0 mismatches for both decoders (element getter and fused decode8). It had
   160,502 escapes and 295 overflowed superblocks (0.45%) stored in the pool. Exactness inside the attention kernel is
   being checked with the same GQA kernel and fusion off on both sides (R45).
4. **Lazy KV + pooled scratch** (`CQ_KV_LAZY=1 CQ_FA_SCRATCH_POOL=1`, -c 122880, MiB; F16 row = lazy F16 from R43):

| tokens | F16 | q8_0 | q4_0 | f16d4 | f16e3 (incl. 512 MiB pool) |
|---|---|---|---|---|---|
| 702 | 2,982 | 3,022 | 2,986 | 2,986 | 3,592 |
| 2,922 | 3,272 | 3,210 | 3,102 | 3,282 | 3,904 |
| 24,791 | 6,438 | 4,916 | 4,052 | **5,672** | 6,908 |
| 114,463 | 18,966 | 11,956 | 7,924 | **15,376** | 19,294 |

   Pooling the scratch removes the ~430 MiB penalty of quantized KV at small fills. f16d4 then saves 12% of process
   memory at 25K tokens and 19% at 114K against lazy F16. f16e3 does not pay off here: its fixed 512 MiB overflow pool
   is far larger than the pool use we measured (0.09 MiB at 114K tokens) and larger than its 4.7% KV saving. It needs a
   right-sized pool (R45).
5. **Quality with the real types (decode path, 39 tasks / needle):** f16d4 25/39 and 10/10; f16e3 25/39 and 10/10;
   F16 25/39.
6. **q4o (4-bit + 2 exact outliers per 128), real type:** KL 0.0576 (prefill path) and 0.0394 (decode, half-math GQA
   kernel), against q4_0 at 0.0698 and 0.0458. That is about 15–18% better than Q4_0 at 1.11× the Q4_0 size, but far
   from F16-class quality.
7. **Decode KL with the half-math GQA kernel** (against the F16 decode base): q8_0 0.00088 against stock 0.00091;
   q4_0 0.0453 against stock 0.0467. Removing the q8_1 quantization of Q does **not** fix q4_0 quality. The q4_0 loss is
   in the K/V values.
8. Operational: `llama-perplexity` reserves `n_ctx × n_vocab × 4` bytes of host memory for logits when
   `n_batch < n_ctx` (20 GB at 32K). With other workloads on the machine this failed with `0xC0000409` at 16K and 32K.
   32K runs now use `-b <n_ctx> -ub 512`.

## R45. KV quality redone with K and V: real packed types and emulation, fusions off, rotation on/off — MEASURED

Sessions 4, 4b and 5: `artifacts/stage1_nofusion_20261022/`, `fork_s5_20261022/`, `fork_exact_20261022/`. Qwen3-4B
Q4_K_M, 4K context, 6 chunks. KL is measured against F16 KV with all CUDA fusions off, and every run also has fusions
off, so both K and V go through `set_rows` and the emulation hook. "rot" is llama.cpp's Hadamard attention rotation,
which is on by default for quantized KV types; `LLAMA_ATTN_ROT_DISABLE=1` turns it off.

**Lossless f16e3 is exact end to end.** With rotation off, raw logits of f16e3 KV are bit-identical to F16 KV for a
1024-token prompt plus 64 greedy tokens. That holds both as one prefill batch and token by token, the latter through
the GQA decode kernel (`llama-cq-logits --kv`, combined build). KL = 0.000000. The earlier non-zero KL came from the
rotation, which the fork now disables for f16e3.

| KV (bits/value) | how | KL vs F16 | same top-1 | 39 tasks (F16 = 24) | needle 4K+32K |
|---|---|---|---|---|---|
| f16e3 (15.25, lossless) | real | **0.000000** | 99.98% | 25 (R44) | 10/10 |
| f16d4 (12) | real, rot on / off | 0.00279 / 0.00283 | 95.7% | 24 (rot off) | 10/10 |
| f16drop2 (14) | emulated | 0.00286 | 95.8% | 25 | 10/10 |
| q8_0 (8.5) | real, rot on / off | 0.00349 / 0.00346 | 95.9% / 95.7% | 25 / 24 | 10/10 |
| q8_0 7-bit codes (7.75) | emulated | 0.0337 | 90.0% | 25 | 10/10 |
| q8_0 6-bit + 1% outliers (6.8) | emulated | 0.00456 | 95.0% | 25 | 10/10 |
| q4_0 (4.5) | real, rot on (llama.cpp default) | 0.0677 | 84.6% | 22 | 10/10 |
| q4_0 (4.5) | real, rot off (= emulated) | 0.1160 | 80.2% | 25 | 10/10 |
| **q4o (5.0)** = 4-bit + 2 exact per 128 | real, **rot off** | **0.0328** | **88.8%** | 24 | 10/10 |
| q4o (5.0) | real, rot on | 0.0596 | 85.4% | 20 | 10/10 |
| q4 + 1% outliers per row | emulated | 0.0381 | 88.5% | 25 | 10/10 |
| q3 + 1% outliers | emulated | 0.2363 | 71.8% | 18 | 9/10 |

Readings:
- The noise floor is about 0.0024: F16 with fusion against F16 without fusion already scores 0.00235 (R44). f16d4,
  f16drop2 and q8_0 all sit at the floor, so the 12-bit f16d4 is as good as Q8_0.
- **For 4-bit KV, keeping 2 of every 128 values exact (q4o) halves the KL of llama.cpp's q4_0 (0.033 against 0.068)
  for 11% more bytes.** It needs the rotation off, because the Hadamard rotation spreads the outliers that q4o keeps
  exact. On plain q4_0 the rotation is what makes it usable (0.068 against 0.116).
- The emulation matches the real types whenever both quantize K and V: emulated q4_0 without rotation gives 0.115962,
  and real q4_0 without rotation gives 0.115962.
- The R36 conclusion "the q4_0 computation path costs 9× more than the format" was an artifact — in R36 the emulation
  quantized only V (R44). The q4_0 loss belongs to the format, partly hidden by the rotation.
- 39 tasks do not separate these variants (a spread of ±3 tasks even for F16-class types). Needle retrieval passes
  everywhere except emulated 3-bit.
- The fork now disables the rotation by default for f16e3, f16d4 and q4o.
- We could not measure 8K and 32K KL in this environment. `llama-perplexity` needs `n_ctx × n_vocab × 4` bytes of host
  memory for logits and only about 11 GB of commit was free. Long-context quality rests on needle retrieval at 32K.

## R46. Fork decode/prefill speed at 4K and 32K: packed KV types, GQA kernel, upstream PR #30202 — MEASURED

`artifacts/fork_speed_a/` (`work/kv_speed.sh`, `scripts/summarize_fork_speed.py`). Qwen3-4B Q4_K_M, llama-bench
`-p 512 -n 64 -d 4096,32768 -r 3`. 14 configurations in ABBA order, 2 rounds, 6 samples per cell, cell CV
≤ 3.4%. GPU lock held, other GPU jobs stopped. Preflight DIRTY: desktop applications at 14–30% idle GPU utilisation, as in R42.
A configuration reads binary:KV:GQA kernel:rotation. "fork" is the `cq/kv-compress` build and "pr" the upstream
PR #30202 build (R39). Ratios are against fork F16 with stock kernels (decode 160.5 / 88.7 tok/s, prefill
5,899 / 2,229 tok/s).

| KV (bits/value) | kernel | decode 4K | decode 32K | prefill 4K | prefill 32K |
|---|---|---|---|---|---|
| F16 (16) | stock (MMA, GQA) | 1.000 | 1.000 | 1.000 | 1.000 |
| F16 (16) | ours, GQA vec | 0.972 | 0.985 | 1.001 | 0.985 |
| **f16d4 (12)** | **ours, GQA vec** | 0.975 | **1.065** | 0.974 | 0.961 |
| f16e3 (15.25, lossless) | ours, GQA vec (first decoder) | 0.500 | 0.233 | 0.902 | 0.876 |
| q4o (5.0) | ours, GQA vec (first decoder) | 0.712 | 0.517 | 0.921 | 0.911 |
| q8_0 (8.5) | stock vec (llama.cpp) | 0.893 | 0.726 | 0.950 | 0.940 |
| q8_0 (8.5) | **ours, GQA vec, half math** | 0.920 | **1.040** | 0.954 | 0.947 |
| q8_0 (8.5) | upstream PR #30202 | 0.894 | 0.892 | 0.953 | 0.946 |
| q4_0 (4.5) | stock vec | 0.886 | 0.716 | 0.947 | 0.947 |
| q4_0 (4.5) | **ours, GQA vec, half math** | 0.928 | **1.069** | 0.917 | 0.952 |
| q4_0 (4.5) | upstream PR #30202 | 0.867 | 0.805 | 0.940 | 0.956 |
| q8_0 / q4_0 | stock, rotation off | 0.913 / 0.911 | 0.737 / 0.722 | 0.978 / 0.983 | 0.938 / 0.951 |

Readings:
- **At 32K, every KV format whose kernel shares one K/V read across the GQA group decodes faster than F16 KV:** f16d4
  1.065×, q8_0 1.040×, q4_0 1.069×. With the stock kernels they reach 0.72–0.73×, and with upstream PR #30202
  0.81–0.89×. The decode slowdown of quantized KV in llama.cpp is therefore a kernel-organisation problem (one KV read
  per query head), not a byte or dequantization problem.
- At 4K quantized KV is still 3–8% slower than F16: fixed overheads (set_rows quantization, Q conversion) are not yet
  amortised.
- Prefill: every non-F16 type loses 3–12% to the conversion into an f16 scratch before the MMA kernel. We left the
  fused prefill path alone.
- The first f16e3 (lossless) and q4o decoders were compute-bound — 7 loads and ~40 ALU ops per 8 values for f16e3,
  plus a local-memory spill in the q4o outlier patch. The optimised versions are measured in R85 (the planned R47 was
  never written).
- The attention rotation costs 1–3% of decode for q8_0/q4_0.

## EXCLUDED

- Q8 tile-MMA attention A/B (64.90–66.84 against 74.27–76.07 µs): there is no raw receipt in the repo (adversarial
  audit B1). Not used.
- "Second-family final read-only audit PASS": no artifact (B2). Not used.
- Any whole-model speedup, HBM bytes/token, or own-codec VRAM saving in real inference: **never measured**.

## R60. Weights fork: lossless and lossy BF16 weight types inside llama.cpp — IMPLEMENTED, bit-exact (2026-10-22)

Fork worktree `D:\Git\llama.cpp-w`, branch `cq/weights-compress` (from pin `ed7ac35e`, not pushed), commits `758867f7`,
`a1053718`, `b4da56a2` (author reset to the noreply identity). Patches: `release/patches/01xx-cq-weights-*.patch` (the
final list is in R6x below). The work adds three ggml types, GGUF tensors plus a companion I8 tensor `<name>.cqx`, a
converter `llama-cq-convert`, and CUDA kernels in `ggml-cuda/cqw.cu`:

| Type | What is stored | Lossless? | Size vs BF16, Qwen3.5-2B (all 2-D BF16 tensors) |
|---|---|---|---|
| `CQ_BF16H` | sign+7-bit mantissa byte raw; 8-bit exponent with a per-tensor canonical Huffman code (length ≤ 16), u16 anchor per (row, lane, split), two-level LUT + pair LUT | yes | 0.696 (0.681 without the row split, R61) |
| `CQ_BF16W4` | sign+mantissa byte; per 64 values the max exponent E and a 4-bit offset per value; offset 15 = escape, exact exponent in a per-row escape list | yes | 0.759 (escapes 1.0e-4 of values) |
| `CQ_BF16W3` | same with 3-bit offsets; values more than 6 binades below E are flushed to ±0 | **no** | 0.695 (2.59% of values flushed) |

Earlier work: exponent Huffman coding of BF16 weights is DFloat11 (Zhang et al., 2025), and ZipNN with ZipServ (TCA-TBE
fixed-rate exponent windows) are the closest relatives of H and W4. The code here is ours, written for llama.cpp's
GEMV.

**Exactness design.** In the batch-1 GEMV, warp step m of lane l handles the value pair p = l + 32·m, which is exactly
the pair that thread (p mod 256) of llama.cpp's BF16 `mul_mat_vec_f` handles in iteration p/256 (block size 256 for
every row length that is a multiple of 512). Each lane keeps the 8 partial sums of the 8 "virtual threads" it emulates
and adds the products in the same order, and the two-level warp reduction (butterfly, then butterfly over the 8 warp
sums padded with zeros) is reproduced. The fused up/gate GLU path is covered too. For batch > 1 we decode each weight
into a BF16 scratch buffer and call the unchanged BF16 path (mmf/cuBLAS) with the same shapes.

**Bit-exactness, MEASURED** (`llama-cq-logits`, raw float32 logits compared with `cmp`; `artifacts/cqw_s1_20261022/`,
`artifacts/cqw_kbench/v3|v8la|v10|v11|v12swz/exact.txt`): a prompt of 534 tokens as one batch with logits for every
position plus 64 greedy tokens one by one, and then 150 prompt tokens one by one plus 32 greedy tokens. `CQ_BF16H` and
`CQ_BF16W4` came out **identical to BF16** in both modes after every kernel change. `CQ_BF16W3` differs, as it is
lossy. The converter also decodes every tensor on the CPU and aborts on any mismatch in the lossless modes.

## R61. Fork kernels in isolation: GEMV (batch 1) per tensor, compressed vs BF16 — MEASURED (interim)

`llama-cq-kbench`: one tensor uploaded per format, 50 GEMVs inside a single ggml graph so launch overhead does not
dominate, 6 alternating repetitions, minimum time per GEMV. All formats get the same random x, and outputs are compared
bit for bit. RTX 3090, other GPU jobs stopped, KV agent idle during the session. Data:
`artifacts/cqw_kbench/v3/kbench_b1_base.txt` (W4/W3, default split) and `artifacts/cqw_kbench/v12swz/kbench_b1_base.txt`
(H, final kernel).

| Tensor (K × N) | BF16 µs (GB/s) | H µs, speed vs BF16 | W4 µs, speed vs BF16 | W3 (lossy) µs, speed vs BF16 |
|---|---|---|---|---|
| output head = token_embd (2048 × 248320, 970 MiB) | 1183 (860) | 1633, **0.72×** | 947, **1.25×** | 843, **1.40×** |
| attn_qkv (2048 × 6144) | 30.5 (825) | 47.9, 0.64× | 28.8, 1.06× | 25.1, 1.22× |
| ffn_up (2048 × 6144) | 30.0 (839) | 47.7, 0.63× | 28.6, 1.05× | 25.5, 1.18× |
| ffn_down (6144 × 2048) | 31.2 (808) | 51.4, 0.61× | 29.0, 1.07× | 27.5, 1.13× |
| ssm_out (2048 × 2048) | 11.4 (736) | 20.1, 0.57× | 10.6, 1.08× | 9.0, 1.27× |

What we tried for H (each step measured, all bit-exact): one warp per row, then splitting rows over 2/4 warps for
short-wide matrices (fixed per tensor by the row count, +1.5% size for the extra anchors); two symbols per LUT lookup
(pair LUT, 2048 × u32); a 32-bit funnel-shift bit reader with 1–4 words of look-ahead; persistent CTAs; XOR-swizzled
LUTs against bank conflicts; and the LUT in global memory instead of shared. Diagnostics (`CQ_DBG`): when the Huffman
lookups are replaced by a fake fixed-length decode and the shared-memory LUT is removed, the same kernel runs the
output head in 941–1081 µs (about W4), so memory is not the problem — the decode chain is. With the fake decode but
12.8 KiB of (unused) shared memory per CTA it drops to 1705 µs: this kernel is very sensitive to the L1 capacity the
shared-memory carve-out takes away (W4 loses only 5–8% under the same padding). Nsight Compute could not collect
counters on this machine ("driver resource unavailable"), so the exact stall reason is not identified.

## R62. Variant B in real inference, base BF16 + KV F16 (Qwen3.5-2B): our weight formats vs BF16 — MEASURED

Every row uses the same fork binary (a `llama-bench` build of `cq/weights-compress`), `-ngl 99 -fa 1 -ctk f16 -ctv f16 -p 512 -n 64 -d DEPTH`,
3 samples per run, runs in forward then backward model order, 2 blocks → **12 samples per cell**. Per-process VRAM is
the peak of the WDDM counter "GPU Process Memory / Dedicated Usage" for the llama-bench process
(`scripts/cqw_vram_probe.py`, about 1 sample/s, a separate run with 6 repetitions per cell). Load is
`llama-cq-logits --load-only`, median of 3 warm loads with the file already in the OS cache; cold-cache load was not
controlled. Window: other GPU jobs stopped, KV agent not running GPU jobs during these series. Data in
`artifacts/cqw_speed/bf16_d4k32k/`, `bf16_d120k/`, `bf16_vram/`, summary `artifacts/cqw_speed/summary_bf16.json|md`.
Decode is tg64 tok/s and prefill pp512 tok/s, both after DEPTH filled tokens; medians, though the 120K cells hold a few
outliers from the first block (see min..max).

| Depth | Row | Tensors MiB (vs base) | Decode tok/s (vs base) [min..max] | Prefill tok/s (vs base) | VRAM MiB (Δ) | Load ms |
|---|---|---|---|---|---|---|
| 4K | BF16 (base, A) | 3706 (1.000) | 158.9 (1.000) [149..163] | 9546 (1.000) | 4442 | 2736 |
| 4K | + H lossless (B) | 2579 (0.696) | 95.7 (**0.602**) [91..99] | 7289 (0.764) | 3366 (−1076) | 2669 |
| 4K | + W4 lossless (B) | 2813 (0.759) | 165.7 (**1.043**) [150..170] | 8032 (0.841) | 3602 (−840) | 2228 |
| 4K | + W3 lossy (B) | 2578 (0.695) | 177.9 (1.120) [164..181] | 7790 (0.816) | 3334 (−1108) | 2638 |
| 4K | + FFN in Q8_0, rest BF16 (lossy recipe, R34) | 2855 (0.770) | 183.4 (1.154) [178..189] | 10868 (1.138) | 3622 (−820) | 2370 |
| 32K | BF16 (base, A) | 3706 | 144.9 (1.000) [139..150] | 7497 (1.000) | 4784 | |
| 32K | + H lossless | 2579 | 91.5 (0.631) [87..92] | 5947 (0.793) | 3702 (−1082) | |
| 32K | + W4 lossless | 2813 | 150.9 (1.041) [142..153] | 6339 (0.846) | 3938 (−846) | |
| 32K | + W3 lossy | 2578 | 160.0 (1.104) [135..166] | 6311 (0.842) | 3710 (−1074) | |
| 32K | + FFN Q8_0 | 2855 | 166.1 (1.147) [158..174] | 8327 (1.111) | 3978 (−806) | |
| 120K | BF16 (base, A) | 3706 | 104.9 (1.000) [61..120] | 4263 (1.000) | 5840 | |
| 120K | + H lossless | 2579 | 74.3 (0.709) [55..80] | 3805 (0.893) | 4766 (−1074) | |
| 120K | + W4 lossless | 2813 | 103.1 (0.983) [41..115] | 3482 (0.817) | 4986 (−854) | |
| 120K | + W3 lossy | 2578 | 119.2 (1.137) [110..130] | 3617 (0.848) | 4766 (−1074) | |
| 120K | + FFN Q8_0 | 2855 | 124.4 (1.186) [26..133] | 4424 (1.038) | 5034 (−806) | |

Findings:
- **Lossless, entropy-coded (H, DFloat11-like): 30% smaller and −1.08 GiB VRAM, but decode runs at 0.60–0.71× of
  BF16.** The fused Huffman decoder cannot keep up with the 3090's bandwidth (R61).
- **Lossless, fixed-rate (W4): 24% smaller, −0.84 GiB, decode 1.04× at 4K/32K** (within noise at 120K). It is the only
  lossless format here that is not slower than BF16 at batch 1, and the gain is small because 4% fewer bytes per token
  is all the GEMVs can give once non-GEMV work is included (the output-head GEMV alone runs 1.25×, R61).
- **Prefill is 0.76–0.85× for all our formats**: each ubatch decodes every weight into a BF16 scratch buffer first,
  which costs about 2× a GEMV per matrix (R61 batch-512 rows). The cost per token shrinks with a larger ubatch, though
  we did not measure that here.
- **The lossy fixed-rate W3 (−30% size) decodes 1.10–1.14×**, but the plain mixed-precision recipe "FFN in Q8_0" (−23%
  size) is faster in both decode (1.15–1.19×) and prefill (1.04–1.14×), because Q8_0 uses llama.cpp's tuned MMVQ/MMQ
  kernels. Quality of both is in R65.
- Load time from a warm OS cache does not differ measurably: 2.2–2.7 s for all, with noise around ±15%.

## R63. Variant B for the Q8_0 and Q4_K_M bases (Qwen3.5-2B): lossy weight recipes vs their base — MEASURED

The integration step wrote this up from the weights agent's raw series, because the agent stopped before writing it.
Same method as R62: weights-fork binary (`cq/weights-compress`), `-ngl 99 -fa 1 -p 512 -n 64 -d DEPTH`, 3 samples per
run, forward/backward order, 2 blocks → 12 samples per cell; KV is the pair's base type. VRAM is the peak WDDM
per-process dedicated usage of one llama-bench run. Load is the median of 3 warm `llama-cq-logits --load-only`. Window:
GPU lock held, desktop background (preflight DIRTY at 4–13% utilization, no other compute process). Data in
`artifacts/cqw_speed/q8/`, `artifacts/cqw_speed/q4/`, summary `work/combo_prev_summary.md` (`scripts/cqw_summarize.py --speed q8 q4`). No lossless GPU type exists for these bases (R64). Decode tok/s median (vs base) / prefill tok/s
(vs base) / VRAM MiB:

| Base pair | Row | MiB (vs base) | 4K | 32K | 120K | Load ms |
|---|---|---|---|---|---|---|
| Q8_0 + Q8_0 KV | Q8_0 (A) | 1970 | 228.7 / 11722 / 2716 | 203.4 / 8766 / 2898 | 144.0 / 4772 / 3468 | 1469 |
| | FFN in Q6_K (B, lossy) | 1752 (0.889) | 241.6 (1.056) / 10815 (0.923) / 2510 | 209.3 (1.029) / 8306 (0.948) / 2688 | 144.2 (1.001) / 4669 (0.978) / 3258 | 1420 |
| | emb+output in Q6_K (B, lossy) | 1853 (0.941) | 234.1 (1.024) / 11489 (0.980) / 2602 | 207.6 (1.021) / 8753 (0.999) / 2780 | 144.0 (1.000) / 4717 (0.988) / 3350 | 1448 |
| Q4_K_M + Q4_0 KV | Q4_K_M (A) | 1241 | 282.9 / 11333 / 2010 | 220.3 / 8348 / 2096 | 152.5 / 4841 / 2402 | 2158 |
| | emb+output in Q4_K (B, lossy) | 1116 (0.899) | 298.1 (1.054) / 11434 (1.009) / 1874 | 239.0 (1.085) / 8788 (1.053) / 1970 | 155.5 (1.020) / 4862 (1.004) / 2276 | 1924 |
| | imatrix, same types (B, lossy, quality-up) | 1241 (1.000) | 280.0 (0.990) / 11226 (0.991) / 2010 | 224.3 (1.018) / 8549 (1.024) / 2096 | 150.3 (0.986) / 4749 (0.981) / 2402 | 2652 |
| | imatrix + FFN in IQ3_S (B, lossy) | 1161 (0.936) | 274.6 (0.971) / 11268 (0.994) / 1934 | 228.7 (1.038) / 8590 (1.029) / 2020 | 150.6 (0.988) / 4790 (0.990) / 2326 | 1975 |

Notes: the Q4_K_M base cell at 4K contains one slow run (min 164 tok/s, CV 12.5%), so medians are used. Size savings of
6–11% buy 0–8% decode gain, and the gain shrinks with depth because the KV read and the recurrent layers do not change.

## R64. Lossless on top of Q8_0 and Q4_K_M: best context models, and why no GPU type was built — MEASURED (CPU) + projection (*)

(From the weights agent's draft `work/cqw/R64_draft.md`, with numbers re-read from
`artifacts/wq_context_entropy_20261022/results.json`.) `scripts/wq_context_entropy.py` counts every byte of every
quantized block (codes, scales/mins, d/dmin), charges the per-tensor static tables and their cost, and samples every
4th row. "GPU-practical" means length-limited (≤ 12 bits) Huffman on the codes with a u16 anchor every 64 or 256
values, and header fields raw.

| File | Native MiB | Order-0 | Best context model (ideal) | GPU-practical, anchor/256 | with context tables, anchor/256 | anchor/64 |
|---|---|---|---|---|---|---|
| Qwen3.5-2B Q8_0 | 1906 | 0.928 | **0.921** | 0.968 | **0.966** | 0.988 |
| Qwen3.5-2B Q4_K_M | 1203 | 0.950 | 0.938 | 0.984 | **0.974** | 1.009 |
| Qwen3-4B Q4_K_M | 2375 | 0.932 | 0.891 | 0.973 | **0.948** | 0.986 |

The rule we set was that a GPU-practical saving below 3% counts as a negative result. Qwen3.5-2B Q4_K_M at 2.6% →
negative. Q8_0 at 3.4% and Qwen3-4B Q4_K_M at 5.2% pass the size rule, but a variable-length decoder must decode one
Huffman symbol per weight. Our fused Huffman decoder (H, R61) handles 3.1·10^11 symbols/s on the output head, and a
Q8_0 GEMV of the same tensor is bandwidth-bound at about 0.64 ms. **Projection (*):** a Huffman-coded Q8_0 GEMV would
take ≥ 1.6 ms for ≤ 3.4% fewer bytes, which is ≤ 0.4× of Q8_0 decode speed. So we built no lossless GPU type on top
of Q8_0 or Q4_K_M, and their variant-B rows are lossy recipes (R63, R65).

## R65. Quality of variant B (all three weight bases, Qwen3.5-2B): KL at context 512 — MEASURED

`artifacts/cqw_quality/kl/` (weights-fork binary, `scripts/cqw_gpu_quality.sh`): llama-perplexity on GPU, fixed corpus,
context 512, all 44 chunks. KV is the pair's base type for "vs own base" and F16 KV for "vs Q8_0 ref" (Q8_0 weights +
F16 KV, as in R7). We did not run the task suite in that series.

| Row | KL vs own base | same top-1 vs own base | KL vs Q8_0 ref | same top-1 vs ref |
|---|---|---|---|---|
| BF16 (A) | — | — | 0.00134 | 95.86% |
| H lossless | 0.00000 | 99.97% | 0.00134 | 95.86% (identical to BF16) |
| W4 lossless | 0.00000 | 99.97% | 0.00134 | 95.86% (identical to BF16) |
| W3 lossy | 0.00061 | 97.78% | 0.00192 | 95.46% |
| FFN in Q8_0 (lossy recipe) | 0.00054 | 97.58% | 0.00131 | 96.12% |
| Q8_0 + FFN Q6_K | 0.00318 | 92.08% | 0.00311 | 92.28% |
| Q8_0 + emb Q6_K | 0.00222 | 93.79% | 0.00217 | 93.87% |
| Q4_K_M (A) | — | — | 0.06109 | 78.58% |
| Q4_K_M + emb Q4_K | 0.03155 | 78.40% | 0.07644 | 74.11% |
| Q4_K_M + imatrix | 0.07152 | 79.46% | **0.03293** | **82.74%** |
| Q4_K_M + imatrix + FFN IQ3_S | 0.12069 | 75.24% | 0.07132 | 77.96% |

Notes: the lossless rows show 99.97% (not 100%) "same top-1" against their own base because llama-perplexity stores
the base log-probabilities in 16 bits; Q8_0 against a base file of itself gives 99.99% and KL 0.00000 in the same way.
Bit identity of the lossless rows is shown with raw float32 logits (R60, R81). For a lossy recipe, "KL vs own base" is
not a quality measure on its own: the imatrix file sits farther from Q4_K_M (0.072) but closer to the reference (0.033
against 0.061), which makes it better.

## R80. Combined fork: one binary for rows A-D — SETUP (2026-10-22)

Branch `cq/combined` in the worktree `D:\Git\llama.cpp-combo` (not pushed), HEAD `74b2da22` = merge `ec5d9e7e` of
`cq/kv-compress` at `0499d389` (KV types f16d4 = 60, f16e3 = 61, q4o = 62, GQA decode kernel, lazy KV, VRAM report) and
`cq/weights-compress` at `b4da56a2` (weight types CQ_BF16H = 50, CQ_BF16W4 = 51, CQ_BF16W3 = 52), plus `llama-cq-logits --kv`.
The enums do not collide (`GGML_TYPE_COUNT` = 64) and the only textual conflict was the enum block in `ggml.h`
(`artifacts/fork_patches/combined/merge-ec5d9e7e-conflict-resolution.txt`). Rebuilt with `cq_build.bat` (CUDA 11.8, MSVC
14.38, sm_86, Ninja, FA instances q4_0, q8_0, f16, bf16, f16d4, f16e3; `work/combo_build.log`, `work/combo_rebuild.log`:
up to date). Patches: `artifacts/fork_patches/{kv,weights,combined}/` (`git format-patch` from the pin `ed7ac35e`; full diff
`combined/cq-combined-full-vs-ed7ac35e.diff`, 43 files, +4178/−13 lines).

**Every row A-D below runs on this one binary**, so row A follows the same code path as the stock types in this build.
Pairs and rows:

| Pair (weights + KV base) | A | B (weights ours) | C (KV ours) | D (both) |
|---|---|---|---|---|
| BF16 + F16 KV | BF16 + f16 | lossless W4, H; lossy W3, FFN-Q8_0 recipe | lossless f16e3; lossy f16d4 | W4/H + f16e3; W3/FFN-Q8_0 + f16d4 |
| Q8_0 + Q8_0 KV | Q8_0 + q8_0 | lossy FFN-Q6_K recipe (no lossless GPU type, R64) | no packed type below 8.5 bit; our GQA decode kernel on q8_0 (same bytes) | FFN-Q6_K + q8_0 with GQA kernel |
| Q4_K_M + Q4_0 KV | Q4_K_M + q4_0 | lossy emb/output-Q4_K recipe; imatrix (same size) | lossy q4o (5.0 bit, quality-up); GQA kernel on q4_0 | emb-Q4_K + q4o; emb-Q4_K + q4_0 with GQA kernel |
| Qwen3-4B Q4_K_M + KV F16 / Q8_0 / Q4_0 | each KV base, stock kernel | — (no BF16 source of this model on disk) | f16e3, f16d4; GQA kernel on q8_0/q4_0; q4o | — |

Some rows are impossible, and for concrete reasons. Lossless B/C on top of Q8_0 / Q4_K_M weights and Q8_0 / Q4_0 KV has
no GPU-practical lossless type to use: for weights the headroom for a GPU-decodable code is ≤ 3.4% and the Huffman
decoder is projected at ≤ 0.4× speed (R64), while for KV every fixed-rate variant came out larger than its base (R33).
D on a model with KV in all layers is out because the lossless weight types need a BF16 GGUF and only Qwen3-4B Q4_K_M
exists locally; Qwen3.5-2B keeps KV in 6 of its 24 layers (the other 18 are recurrent Gated-DeltaNet layers), so KV
formats barely move anything there. That is why we added the Qwen3-4B KV rows.

Method. Speed: `llama-bench -ngl 99 -fa 1 -ctk KV -ctv KV -p 512 -n 64 -d DEPTH -r 3`, configurations in forward then
backward order, 2 blocks → 12 samples per cell (Qwen3-4B: 1 block → 6 samples). Decode is tg64 after DEPTH filled
tokens, prefill pp512 after DEPTH filled tokens. Depths 4,096 / 32,768 / 122,880. Process VRAM comes from (a) the fork's
report `CQ_VRAM_REPORT=1` (DXGI local-memory use of the process after every `llama_decode`, maximum over the run;
primary) and (b) the peak of the WDDM counter sampled about once per second (`scripts/cqw_vram_probe.py`; can miss short
peaks at 4K). "Pooled scratch" means the same with `CQ_FA_SCRATCH_POOL=1`. Load: median of 3 warm
`llama-cq-logits --load-only` (file in the OS cache). Quality: llama-perplexity, fixed corpus, context 4096 (5 chunks), KL
and same top-1 against row A of the pair and against BF16 + F16 KV; this is the prefill path. Decode-path quality: the
39-case task suite through llama-server (ctx 8192), plus the number of answers identical to row A. Lossless: raw float32
logits from `llama-cq-logits`, compared with `cmp`. Scripts: `scripts/combo_session.sh` (driver), `combo_matrix.sh`,
`combo_exact.sh`, `combo_quality.sh`, `combo_vram.sh`, `combo_speed_rerun.sh`, summaries `combo_summarize.py` →
`artifacts/combo/summary.json|md`, tables `combo_results_tables.py` → `artifacts/combo/results_tables.md`, figures
`make_figures_combo.py`.

Window. We held the GPU lock for every series, and preflight came back DIRTY in all of them. No other compute process
ran on the GPU, but desktop applications (a game client, Parsec, browsers) kept 5–13% utilisation at idle and caused
occasional slow runs. At 4K and 32K the spread between runs of one configuration reached 30% in the first session (CV up
to 18%); at 120K the CV was 0.6–5.6%. Each decode cell therefore reports the median and, in brackets, the ratio of the
fastest runs, since interference only ever slows a run down. The core rows were re-measured with 3 blocks (18 samples)
and `--prio 2` (R85). Every number below is **measured** (no emulation, no projection) unless marked.

There were earlier interim fork series that never got written up (`artifacts/fork_speed_a2/`, `artifacts/fork_speed_b/`,
announced as "R47" in R46). They used the `cq/kv-compress` build before or at `0499d389` in a dirty window and are
superseded by R85, which measures the same configurations in the combined build. R46 remains as the first 4K/32K
measurement.

## R81. Combined fork: are the lossless rows bit-identical? — MEASURED

`artifacts/combo/exact/` (`scripts/combo_exact.sh`, `combo_logit_diff.py`; `cmp.txt`, `diff.txt`), plus
`artifacts/combo/exact_d/` (D against C directly). We compare raw float32 logits from `llama-cq-logits` on the prompt
`artifacts/prompts_filled/c1024.txt` (531 tokens with the Qwen3.5 tokenizer, 671 with Qwen3) plus 64 greedy tokens.
"batch" means the prompt as one batch (GEMM/MMA paths, prefill FA); "per-token" means every token on its own (GEMV,
single-token FA). "exact env" is `GGML_CUDA_DISABLE_FUSION=1 CQ_FA_VEC_GQA=1`, which removes the rope+set_rows fusion
and makes F16 use the same GQA decode kernel as the packed types; "default env" means no switches.

| Comparison | batch, exact env | per-token, exact env | batch, default env | per-token, default env |
|---|---|---|---|---|
| B: W4 + F16 vs A: BF16 + F16 (2B) | **identical** | **identical** | **identical** | **identical** |
| B: H + F16 vs A (2B) | **identical** | **identical** | **identical** | **identical** |
| C: f16e3 vs F16 KV, Qwen3-4B (head 128) | **identical** | **identical** | differs (kernel path) | differs (kernel path) |
| C: f16e3 vs F16 KV, Qwen3.5-2B (head 256) | prompt identical; 64 generated positions differ by ≤ 5.4e-4 | 467 of 595 positions differ by ≤ 1.0e-3 | differs (kernel path) | differs (kernel path) |
| D: W4 / H + f16e3 vs A (2B) | same differences as C (max 5.4e-4) | same as C (max 1.0e-3) | same as C | same as C |
| D vs C: W4 / H + f16e3 vs BF16 + f16e3 (2B) | **identical** | **identical** | **identical** | **identical** |

Where f16e3 differs from F16 on Qwen3.5-2B in the exact env, `diff.txt` gives mean KL 5.7e-12 (batch) and 2.8e-10
(per-token); top-1 agrees at every position and the greedy continuation is identical. The first 128 positions of the
per-token run match, the rest differ. The stored values are exact — the batch prompt, which reads the cache back through
the prefill path, is identical — but once more than 128 KV rows are processed, the decode kernel for head size 256 sums the
f16e3 path in a different order from the F16 path. So bit identity end to end holds at head size 128 and not yet at head
size 256, which is an open kernel issue rather than a format issue. In the default env F16 runs fused rope+set_rows and
the stock FA kernels, and f16e3 against F16 then differs by KL 4.2e-8 on Qwen3.5-2B and 7.4e-4 on Qwen3-4B (prompt
positions, per-token). That is the same as switching F16 itself from the default to the exact env (4.2e-8 and 6.8e-4),
so the difference sits in the kernel and fusion path, not the KV format.

Lossy references (exact env, per-token, prompt positions): f16d4 against F16, KL 4.6e-5 (2B) and 1.0e-3 (4B, top-1
659/671); W3 against BF16, KL 3.6e-4 (2B, greedy continuation diverges at the 5th generated token).

**f16e3 overflow pool — the lossless guarantee has a capacity limit.** A superblock with more escapes than its in-block
slots goes to a global overflow pool (`CQ_F16E3_POOL_MIB`, default 16 MiB = 65,536 superblocks). Slots come from an
atomic counter and are never returned; once the pool is full, a superblock keeps only its in-block escapes and its other
out-of-window values are **not exact** (counted as "lost", `set-rows.cu`). The fork's VRAM report shows
(`artifacts/combo/*/vram_report/*.err`):
- Qwen3.5-2B, llama-bench fill of 125K tokens: 10,947 pool blocks used, 0 lost.
- Qwen3-4B, llama-bench fill (llama-bench fills the cache with random token ids): 1.8% of the superblocks need the pool
  (1,302,374 of 71.1 M at 123,392 tokens). The pool fills after about 8K tokens: 0 lost at a 4K fill, 295,785 lost at 32K,
  1,236,838 lost at 120K (1.7% of all superblocks).
- Real text (`artifacts/combo/f16e3_pool/`, `scripts/combo_f16e3_pool.py`: llama-server, prompt sent twice without prompt
  cache): on Qwen3-4B with a 24,764-token prompt, 207,797 superblocks needed the pool (8.4 per token, 1.5%), the pool
  filled after about 8K tokens and 142,261 superblocks were lost; the second identical request took 220,601 new slots,
  since slots are never reused. Qwen3-4B, 2,891 tokens twice: 36,654 used, 0 lost. Qwen3.5-2B, 22,787 tokens twice:
  11,534 used, 0 lost.
  R44 measured 0.45% overflowing superblocks with 8 in-block escape slots; the current commit `0499d389` keeps 7, because
  one slot now holds the group mask. The higher rate fits that change, though we did not isolate it.
Consequences: (1) the Qwen3-4B f16e3 speed rows at 32K and 120K (R85) ran partly in the lossy fallback; (2) with a bump
allocator, a long-running server will exhaust any fixed pool, because every cache rewrite takes new slots. f16e3 needs a
right-sized, reclaimable overflow store before anyone can call it lossless in service use.

## R82. Pair BF16 weights + F16 KV (Qwen3.5-2B): rows A-D — MEASURED

`artifacts/combo/bf16_f16/` (speed, WDDM VRAM, load), `bf16_f16/vram_report/` (fork VRAM report), `quality/` (KL, tasks).
Row A is absolute; the other rows give speed as ×A (median, with the ratio of the fastest runs in brackets) and VRAM as
Δ MiB against A from the fork report. Weights MiB is the tensor bytes llama-bench reports. KL is at context 4096 on the
prefill path. Tasks: 39 cases on the decode path, with the number of answers identical to row A in brackets.

| Row | Config | Weights MiB | Load ms | VRAM MiB 4K / 32K / 120K | Decode 4K / 32K / 120K | Prefill 4K / 32K / 120K | KL vs A (same top-1) | KL vs BF16+F16 | Tasks /39 (outputs = A) |
|---|---|---|---|---|---|---|---|---|---|
| A | BF16 weights + F16 KV | 3706 | 2320 | 4446 / 4782 / 5838 | 156.2 / 117.0 / 124.2 tok/s | 9242 / 6031 / 4672 tok/s | — | ref | 28 (39/39) |
| B | W4 weights (lossless, fixed-rate) + F16 KV | 2813 (0.759) | 1915 | -846 / -846 / -846 Δ | 1.07 (1.07) / 1.05 (1.03) / 1.03 (1.04) ×A | 0.85 / 0.93 / 0.92 ×A | 0.00000 (100.0%) | 0.00000 (100.0%) | 28 (39/39) |
| B | H weights (lossless, Huffman) + F16 KV | 2579 (0.696) | 1807 | -1074 / -1074 / -1074 Δ | 0.63 (0.62) / 0.73 (0.63) / 0.67 (0.67) ×A | 0.84 / 0.97 / 0.89 ×A | 0.00000 (100.0%) | 0.00000 (100.0%) | n/m |
| B | W3 weights (lossy, 3-bit window) + F16 KV | 2578 (0.695) | 1772 | -1074 / -1074 / -1074 Δ | 1.19 (1.17) / 1.45 (1.16) / 1.09 (1.10) ×A | 0.85 / 1.06 / 0.92 ×A | 0.00084 (97.8%) | 0.00084 (97.8%) | 27 (37/39) |
| B | FFN in Q8_0, rest BF16 (lossy recipe) + F16 KV | 2855 (0.770) | 1805 | -806 / -806 / -806 Δ | 1.21 (1.22) / 1.46 (1.19) / 1.13 (1.13) ×A | 1.18 / 1.29 / 1.07 ×A | 0.00050 (97.8%) | 0.00050 (97.8%) | n/m |
| C | BF16 weights + f16e3 KV (lossless) | 3706 (1.000) | 2320 | +14 / -2 / -50 Δ | 0.85 (0.83) / 0.73 (0.60) / 0.38 (0.38) ×A | 1.00 / 1.18 / 0.93 ×A | 0.00000 (100.0%) | 0.00000 (100.0%) | 28 (39/39) |
| C | BF16 weights + f16d4 KV (lossy, 12 bit) | 3706 (1.000) | 2320 | -12 / -96 / -360 Δ | 1.04 (1.00) / 0.86 (0.77) / 0.76 (0.76) ×A | 1.07 / 0.99 / 0.98 ×A | 0.00017 (99.0%) | 0.00017 (99.0%) | 27 (37/39) |
| D | W4 weights + f16e3 KV (both lossless) | 2813 (0.759) | 1915 | -832 / -848 / -896 Δ | 0.89 (0.85) / 0.60 (0.54) / 0.38 (0.38) ×A | 0.91 / 0.80 / 0.86 ×A | 0.00000 (100.0%) | 0.00000 (100.0%) | 28 (39/39) |
| D | H weights + f16e3 KV (both lossless) | 2579 (0.696) | 1807 | -1060 / -1076 / -1124 Δ | 0.57 (0.55) / 0.45 (0.39) / 0.32 (0.32) ×A | 0.77 / 0.75 / 0.83 ×A | 0.00000 (100.0%) | 0.00000 (100.0%) | n/m |
| D | W3 weights + f16d4 KV (both lossy) | 2578 (0.695) | 1772 | -1086 / -1170 / -1434 Δ | 1.06 (1.10) / 0.94 (0.85) / 0.82 (0.82) ×A | 0.76 / 0.86 / 0.88 ×A | 0.00101 (97.7%) | 0.00101 (97.7%) | 28 (38/39) |
| D | FFN-Q8_0 recipe + f16d4 KV (both lossy) | 2855 (0.770) | 1805 | -818 / -902 / -1166 Δ | 1.09 (1.14) / 0.99 (0.90) / 0.82 (0.83) ×A | 1.06 / 1.12 / 1.01 ×A | 0.00061 (97.5%) | 0.00061 (97.5%) | n/m |

Readings:
- **Lossless weights work as a drop-in.** W4 (12.1 bits) is bit-identical to BF16 (R81), saves 893 MiB of tensors and
  846 MiB of process VRAM at every depth, and decodes 1.03–1.05× (R86, 18 samples in a quieter window; 1.03–1.07×
  here). Prefill is 0.84–0.93× because every weight is decoded to a BF16 scratch before the GEMM (R62). H (Huffman
  exponent, 11.1 bits) saves 1,074 MiB but decodes 0.63–0.73×. All 39 task answers of W4 match A.
- **Lossless KV (f16e3) does not pay off on this model.** Only 6 of 24 layers have a KV cache, so f16e3 saves 50 MiB at
  120K (−4.7% of 1,447 MiB of KV, minus its pool) while decode drops to 0.80 / 0.61 / 0.39× at 4K / 32K / 120K (R86).
  The decoder costs more than the 4.7% of bytes it saves. Outputs: identical prompt logits, ≤ 1e-3 logit differences in
  decode at head size 256 (R81), 39/39 identical task answers.
- **Lossy KV (f16d4) saves 25% of KV bytes (−360 MiB at 120K) at KL 0.00017 against A** (same top-1 99.0%; f16e3 and the
  W4/H rows are 0 here), but decode is 0.96 / 0.90 / 0.77× (R86). R86 shows this is the decode kernel on this head-256
  model, not the format: our GQA kernel on plain F16 data is 0.97 / 0.87 / 0.75×.
- **Row D lossless (W4 + f16e3)** is the sum of both: −896 MiB at 120K, decode 0.89 / 0.60 / 0.38× (dominated by f16e3),
  outputs as in C. **Row D lossy (W3 + f16d4)**: −1,434 MiB (−25% of process VRAM) at 120K, KL 0.00101 against A,
  28/39 tasks, decode 1.09 / 0.99 / 0.82× (R86; at long context the kernel cost wins). The plain recipe FFN-Q8_0 +
  f16d4 is similar (−1,166 MiB, KL 0.00061).
- The best speed in this pair comes from the weights alone: W3 1.11–1.13× (R86), and the FFN-Q8_0 recipe 1.13× at 120K.
  Its 4K/32K cells here are noisy — the 32K medians of 1.45–1.46× come from two slow base runs, and R62 measured
  1.15–1.19×.
- Window: at 4K and 32K the base row had two slow runs out of four (CV 15–18%), so use the ratios in brackets or R86.

## R83. Pair Q8_0 weights + Q8_0 KV (Qwen3.5-2B): rows A-D — MEASURED

`artifacts/combo/q8_q8/`, `quality/`. Same layout as R82. Window: CV 1–3% in every cell.

| Row | Config | Weights MiB | Load ms | VRAM MiB 4K / 32K / 120K | Decode 4K / 32K / 120K | Prefill 4K / 32K / 120K | KL vs A (same top-1) | KL vs BF16+F16 | Tasks /39 (outputs = A) |
|---|---|---|---|---|---|---|---|---|---|
| A | Q8_0 weights + Q8_0 KV | 1970 | 1390 | 2726 / 2904 / 3466 | 235.7 / 206.8 / 145.2 tok/s | 11877 / 9028 / 4830 tok/s | — | 0.00132 (96.8%) | 28 (39/39) |
| B | FFN in Q6_K (lossy recipe) + Q8_0 KV | 1752 (0.889) | 1275 | -210 / -210 / -210 Δ | 1.04 (1.04) / 1.04 (1.03) / 1.02 (1.01) ×A | 0.93 / 0.96 / 0.98 ×A | 0.00311 (94.8%) | 0.00297 (95.0%) | 28 (36/39) |
| C | Q8_0 weights + Q8_0 KV, our GQA decode kernel (same bytes) | 1970 (1.000) | 1390 | +0 / +0 / +0 Δ | 0.94 (0.95) / 0.94 (0.94) / 0.94 (0.94) ×A | 1.00 / 1.00 / 1.00 ×A | 0.00000 (100.0%) | 0.00132 (96.8%) | 28 (38/39) |
| D | FFN-Q6_K recipe + Q8_0 KV with our GQA kernel | 1752 (0.889) | 1275 | -210 / -210 / -210 Δ | 0.99 (1.00) / 0.97 (0.96) / 0.96 (0.96) ×A | 0.94 / 0.96 / 0.98 ×A | 0.00311 (94.8%) | 0.00297 (95.0%) | 27 (37/39) |

Readings:
- This pair has no lossless row (R64: ≤ 3.4% headroom for a GPU-decodable code; R33: fixed-rate KV variants larger than
  the base). B is a lossy recipe: FFN in Q6_K, −218 MiB tensors (−11%), −210 MiB VRAM, decode 1.02–1.04×, prefill
  0.93–0.98×, KL 0.0031 against A, tasks 28/39 with 36 answers identical to A.
- Our GQA decode kernel on the unchanged q8_0 cache is **slower** on this head-256 / 2-KV-head model: 0.94× at every
  depth, the opposite of Qwen3-4B (R85). Its prefill KL against A is 0, since the prefill path does not use it; in decode,
  38 of 39 task answers are identical to A.
- D is B plus the kernel: 0.96–0.99×. For this model the useful row is B alone.

## R84. Pair Q4_K_M weights + Q4_0 KV (Qwen3.5-2B): rows A-D — MEASURED

`artifacts/combo/q4km_q4/`, `quality/`; the core rows were re-measured in R86. Same layout as R82.

| Row | Config | Weights MiB | Load ms | VRAM MiB 4K / 32K / 120K | Decode 4K / 32K / 120K | Prefill 4K / 32K / 120K | KL vs A (same top-1) | KL vs BF16+F16 | Tasks /39 (outputs = A) |
|---|---|---|---|---|---|---|---|---|---|
| A | Q4_K_M weights + Q4_0 KV | 1241 | 1098 | 2008 / 2102 / 2400 | 276.5 / 178.3 / 151.1 tok/s | 11192 / 6949 / 4640 tok/s | — | 0.07705 (83.0%) | 27 (39/39) |
| B | embedding+output in Q4_K (lossy recipe) + Q4_0 KV | 1116 (0.899) | 1098 | -126 / -126 / -126 Δ | 1.05 (1.04) / 1.07 (1.10) / 1.01 (1.03) ×A | 1.01 / 0.93 / 1.03 ×A | 0.02903 (84.9%) | 0.08897 (79.7%) | 26 (31/39) |
| B | Q4_K_M with imatrix (same size) + Q4_0 KV | 1241 (1.000) | 1200 | +0 / +0 / +0 Δ | 0.99 (0.99) / 0.99 (1.04) / 0.95 (0.98) ×A | 0.99 / 0.95 / 0.98 ×A | 0.07992 (83.0%) | 0.04404 (86.2%) | n/m |
| C | Q4_K_M weights + q4o KV (lossy, 5.0 bit, +11% KV) | 1241 (1.000) | 1098 | +2 / +12 / +44 Δ | 0.82 (0.81) / 0.63 (0.52) / 0.52 (0.52) ×A | 0.98 / 0.89 / 0.98 ×A | 0.01954 (90.0%) | 0.07500 (82.9%) | 27 (34/39) |
| C | Q4_K_M weights + Q4_0 KV, our GQA decode kernel (same bytes) | 1241 (1.000) | 1098 | +0 / +0 / +0 Δ | 0.93 (0.92) / 0.91 (0.76) / 1.04 (1.04) ×A | 1.01 / 0.88 / 1.01 ×A | 0.00000 (100.0%) | 0.07705 (83.0%) | n/m |
| D | emb-Q4_K recipe + q4o KV | 1116 (0.899) | 1098 | -124 / -114 / -82 Δ | 0.87 (0.85) / 0.66 (0.54) / 0.52 (0.52) ×A | 1.01 / 0.91 / 0.99 ×A | 0.03456 (83.6%) | 0.08561 (79.9%) | 27 (32/39) |
| D | emb-Q4_K recipe + Q4_0 KV with our GQA kernel | 1116 (0.899) | 1098 | -126 / -126 / -126 Δ | 0.99 (1.00) / 0.98 (0.82) / 1.07 (1.08) ×A | 0.99 / 0.91 / 1.01 ×A | 0.02903 (84.9%) | 0.08897 (79.7%) | n/m |

Readings:
- B: the emb/output-Q4_K recipe saves 125 MiB (−10%) and decodes 1.01–1.07×, but it costs KL — 0.029 against A, and
  against BF16 + F16 it moves from 0.077 (A) to 0.089. The imatrix row keeps the size and lands **closer to the
  reference** (KL 0.044 against 0.077), which means better quality at the same speed. That is the one weight change in
  this pair that improves quality at no cost.
- C, q4o (Q4_0 rule plus 2 exact values per 128, 5.0 bits): +11% KV bytes (+44 MiB at 120K), KL against BF16 + F16
  0.075 against 0.077 for q4_0. On this model the KV error is small next to the Q4_K_M weight error, so q4o's KV gain
  (R45: 0.033 against 0.068 KL on Qwen3-4B) hardly shows. Its decode is slow here, 0.52× at 120K.
- C, GQA kernel on q4_0: 0.97 / 1.00× at 4K / 32K (R86) and 1.04× at 120K; D with the recipe gives 1.00 / 1.05 / 1.07×.
- B with the recipe: 1.03 / 1.08× at 4K / 32K (R86), 1.01× at 120K.
- There is no lossless row, for the same reasons as R83.

## R85. Qwen3-4B Q4_K_M, KV in all 36 layers: KV types and the GQA decode kernel in the combined build — MEASURED

`artifacts/combo/q4b_kv/` (6 samples per cell, CV ≤ 2% except one 120K cell; VRAM from the fork report). Rows A are each
KV base with the stock kernel; rows C are our KV type or our kernel. Quality of these KV types on this model was
measured in R43–R45 (KL against F16 KV at 4K: f16e3 0, f16d4 0.0028, q8_0 0.0035, q4_0 0.068 with rotation, q4o 0.033
without; 39 tasks within ±3; needle 10/10; decode-path KL of the GQA kernel equal to the stock kernel: q8_0 0.00088
against 0.00091, q4_0 0.0453 against 0.0467).

| Row | Config | Weights MiB | Load ms | VRAM MiB 4K / 32K / 120K | Decode 4K / 32K / 120K | Prefill 4K / 32K / 120K |
|---|---|---|---|---|---|---|
| A(F16) | Qwen3-4B Q4_K_M + F16 KV | 2376 | 1390 | 3586 / 7618 / 20290 | 168.5 / 89.7 / 37.6 tok/s | 6092 / 2254 / 757 tok/s |
| C | + f16e3 KV (lossless) | 2376 (1.000) | 1390 | -12 / -200 / -458 Δ | 0.50 (0.50) / 0.21 (0.20) / 0.14 (0.13) ×A | 0.91 / 0.89 / 0.88 ×A |
| C | + f16d4 KV (lossy, 12 bit) | 2376 (1.000) | 1390 | -162 / -1170 / -4002 Δ | 0.98 (0.98) / 1.08 (1.05) / 1.15 (1.14) ×A | 0.99 / 0.96 / 0.96 ×A |
| A(Q8_0) | + Q8_0 KV (stock kernel) | 2376 | 1390 | 3284 / 5426 / 12486 | 150.3 / 66.1 / 24.2 tok/s | 5840 / 2130 / 732 tok/s |
| C | + Q8_0 KV, our GQA kernel | 2376 (1.000) | 1390 | +0 / +0 / +0 Δ | 1.05 (1.05) / 1.43 (1.43) / 1.72 (1.75) ×A | 1.00 / 1.01 / 0.98 ×A |
| A(Q4_0) | + Q4_0 KV (stock kernel) | 2376 | 1390 | 3122 / 4256 / 8148 | 148.5 / 64.0 / 22.6 tok/s | 5932 / 2151 / 721 tok/s |
| C | + Q4_0 KV, our GQA kernel | 2376 (1.000) | 1390 | +0 / +0 / +0 Δ | 1.07 (1.07) / 1.53 (1.53) / 1.93 (1.93) ×A | 0.99 / 1.00 / 0.99 ×A |
| C | + q4o KV (lossy, 5.0 bit) | 2376 (1.000) | 1390 | +20 / +146 / +550 Δ | 0.80 (0.80) / 0.98 (0.98) / 1.05 (1.08) ×A | 0.95 / 0.96 / 0.98 ×A |

KV bytes per token, taken from the VRAM slope between the 32K and 120K fills: F16 144.0 KiB, f16e3 141.1, f16d4 111.8
(−22%), q8_0 80.2, q4_0 44.2, q4o 48.8.

Readings:
- **Our GQA decode kernel removes the long-context decode penalty of quantized KV.** With the stock kernels, q8_0 / q4_0
  KV decode at 0.64 / 0.60× of F16 at 120K (24.2 / 22.6 against 37.6 tok/s). With our kernel on the same bytes:
  41.6 / 43.7 tok/s = **1.11 / 1.16× of F16** and 1.72 / 1.93× of the stock path, while the process uses 12,486 / 8,148
  MiB instead of 20,290 MiB (−38% / −60%). At 32K that is 1.43 / 1.53× of stock and 1.05 / 1.09× of F16; at 4K,
  1.05 / 1.07× of stock. Prefill stays unchanged within ±2%.
- **f16d4 (12-bit KV, F16-class quality)**: −4,002 MiB (−20%) at 120K and decode 1.15× of F16 (43.4 tok/s), 1.08× at
  32K, 0.98× at 4K; prefill 0.96–0.99×.
- **f16e3 (lossless)**: −458 MiB (−2.3%) at 120K; decode 0.50 / 0.21 / 0.14× of F16. At 32K and 120K its overflow pool
  was full (R81), so those rows are not lossless. Negative result.
- **q4o**: +550 MiB against q4_0 at 120K, decode 1.05× of stock q4_0 but 0.54× of q4_0 with our kernel. Its value is
  quality (R45), not speed.
- Context capacity (projection *, linear in the measured slope against a 23.5 GB budget): F16 ≈ 146K tokens, f16d4 ≈
  189K, q8_0 ≈ 263K, q4_0 ≈ 478K. Qwen3-4B itself is trained to 32K (128K with YaRN), so for this model the gain is free
  memory at 120K (4.0 / 7.8 / 12.1 GB), which a larger model or a second sequence could use.

## R86. Re-measurement of the core rows with 18 samples and high process priority; kernel vs format on head size 256 — MEASURED

`artifacts/combo/bf16_f16_rerun/` and `artifacts/combo/q4km_q4_rerun/` (`scripts/combo_speed_rerun.sh`): same binary and
parameters as R82/R84, plus `llama-bench --prio 2`, 3 ABBA blocks → 18 samples per cell (the Q4 pair only at 4K and 32K,
since its 120K cells already had CV ≤ 12%). Preflight DIRTY as before (desktop applications, no other compute process);
cell CV is now 1.3–2.9% for the BF16 pair and 3.0–6.1% for the Q4 pair. The table adds a row "F16 KV + our GQA kernel"
so the kernel can be separated from the KV format.

| Row | Config | Decode 4K / 32K / 120K | Prefill 4K / 32K / 120K |
|---|---|---|---|
| A | BF16 weights + F16 KV | 159.3 / 149.4 / 119.8 tok/s | 9385 / 7613 / 4491 tok/s |
| C | BF16 weights + F16 KV, our GQA decode kernel (same bytes) | 0.97 (0.96) / 0.87 (0.89) / 0.75 (0.74) ×A | 1.01 / 1.01 / 1.01 ×A |
| B | W4 weights (lossless, fixed-rate) + F16 KV | 1.03 (1.02) / 1.05 (1.05) / 1.03 (1.03) ×A | 0.84 / 0.86 / 0.92 ×A |
| B | W3 weights (lossy, 3-bit window) + F16 KV | 1.13 (1.12) / 1.13 (1.12) / 1.11 (1.11) ×A | 0.83 / 0.89 / 0.92 ×A |
| C | BF16 weights + f16e3 KV (lossless) | 0.80 (0.80) / 0.61 (0.60) / 0.39 (0.38) ×A | 0.99 / 0.96 / 0.95 ×A |
| C | BF16 weights + f16d4 KV (lossy, 12 bit) | 0.96 (0.96) / 0.90 (0.90) / 0.77 (0.77) ×A | 1.01 / 1.00 / 1.00 ×A |
| D | W4 weights + f16e3 KV (both lossless) | 0.83 (0.84) / 0.63 (0.62) / 0.39 (0.38) ×A | 0.83 / 0.85 / 0.86 ×A |
| D | W3 weights + f16d4 KV (both lossy) | 1.09 (1.07) / 0.99 (1.00) / 0.82 (0.82) ×A | 0.82 / 0.84 / 0.90 ×A |

| Row | Config | Decode 4K / 32K / 120K | Prefill 4K / 32K / 120K |
|---|---|---|---|
| A | Q4_K_M weights + Q4_0 KV | 272.3 / 219.9 / – tok/s | 10900 / 8519 / – tok/s |
| B | embedding+output in Q4_K (lossy recipe) + Q4_0 KV | 1.03 (1.05) / 1.08 (1.08) / – ×A | 1.01 / 1.00 / – ×A |
| C | Q4_K_M weights + q4o KV (lossy, 5.0 bit, +11% KV) | 0.84 (0.83) / 0.70 (0.69) / – ×A | 0.99 / 0.97 / – ×A |
| C | Q4_K_M weights + Q4_0 KV, our GQA decode kernel (same bytes) | 0.97 (0.96) / 1.00 (1.01) / – ×A | 0.99 / 1.00 / – ×A |
| D | emb-Q4_K recipe + q4o KV | 0.88 (0.86) / 0.72 (0.71) / – ×A | 1.03 / 1.00 / – ×A |
| D | emb-Q4_K recipe + Q4_0 KV with our GQA kernel | 1.00 (1.03) / 1.05 (1.05) / – ×A | 1.02 / 1.00 / – ×A |

Readings:
- With a quiet window the BF16 pair is consistent across depths: W4 (lossless) 1.03–1.05× decode, W3 (lossy) 1.11–1.13×.
  The 32K medians of R82 (W3 1.45×, FFN-Q8_0 1.46×) were inflated by two slow base runs, and this table supersedes them.
- **It is the kernel, not the format, that makes f16d4 slow on Qwen3.5-2B.** Our GQA vector kernel on plain F16 data
  runs at 0.97 / 0.87 / 0.75× of the stock F16 path at 4K / 32K / 120K, while f16d4 through the same kernel is as fast
  as F16 through it at 4K and 3% faster at 32K and 120K (0.995 / 1.028 / 1.032×). The 25% byte saving helps a little;
  the kernel loses 3–25%. On Qwen3-4B (head 128, 8 KV heads) the same kernel beats every stock path (R85). A likely
  cause, not tested: with 2 KV heads and head size 256 the grid has too few blocks and too many registers per thread
  for the 3090.
- Q4 pair at 4K/32K: the emb-Q4_K recipe 1.03 / 1.08×; q4o 0.84 / 0.70×; the GQA kernel on q4_0 0.97 / 1.00×.

## R87. What the combined fork gives in practice (summary of R81–R86) — MEASURED unless marked *

One RTX 3090, one binary (`cq/combined` `74b2da22`), real llama.cpp inference.

| Goal | Best measured option in the fork | Effect vs the regular row A of the same pair | Quality cost | Where |
|---|---|---|---|---|
| Smaller model, no quality change (BF16) | W4 weights (lossless) | −893 MiB tensors (−24%), −846 MiB VRAM; decode 1.03–1.05×, prefill 0.84–0.92× | none: bit-identical logits, 39/39 identical answers | R81, R82, R86 |
| Smallest lossless model (BF16) | H weights | −1,127 MiB tensors (−30%); decode 0.63–0.73× | none | R82 |
| Faster long-context decode with quantized KV (full-KV model) | our GQA decode kernel on q8_0 / q4_0 | Qwen3-4B at 120K: 1.72 / 1.93× of the stock path = 1.11 / 1.16× of F16 KV, at 12.5 / 8.1 GB instead of 20.3 GB | equal to the stock kernel (decode-path KL, R43–R44) | R85 |
| F16-class KV in less memory (full-KV model) | f16d4 KV | Qwen3-4B at 120K: −4.0 GB (−20%), decode 1.15× of F16 | KL 0.0028 vs F16 (noise floor 0.0024; R45) | R85 |
| Most compact BF16-quality setup (2B) | W3 + f16d4 (lossy) | −1,434 MiB VRAM (−25%) at 120K; decode 1.09 / 0.99 / 0.82× | KL 0.0010 vs A; 28/39 tasks (A 28) | R82, R86 |
| Lossless KV | f16e3 | −4.7% of KV bytes (−2.3% process VRAM, Qwen3-4B at 120K), decode 0.14–0.85× | lossless only until its 16 MiB overflow pool is full (≈ 8K tokens of Qwen3-4B) | R81, R85 |
| Lossless on top of Q8_0 / Q4_K_M weights or Q8_0 / Q4_0 KV | none built | — | — | R64, R33 |

Negative and model-specific results: on Qwen3.5-2B (KV in 6 of 24 layers, head size 256, 2 KV heads) every KV-side
change is at most a few hundred MiB, and our GQA kernel is slower than the stock path (0.75–0.97× on F16, 0.94× on
q8_0). On Qwen3-4B (36 KV layers, head 128, 8 KV heads) the same kernel is the largest speed gain of the study. The
lossless KV type f16e3 is not practical in its current form: the decoder is slow and the overflow pool fills. Weight
savings are a fixed number of MiB independent of context; KV savings grow with context. Process-VRAM capacity
projection * (linear in the measured KV slope, 23.5 GB budget, Qwen3-4B Q4_K_M): F16 ≈ 146K tokens, f16d4 ≈ 189K,
q8_0 ≈ 263K, q4_0 ≈ 478K. With our kernel the last two are no longer slower than F16.

