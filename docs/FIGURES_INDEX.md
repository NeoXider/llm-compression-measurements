# FIGURES_INDEX — new figures (37), one row each

All files are in `figures/` as `.png` (150 dpi) and `.svg`. Generator: `scripts/make_more_figures.py`
(parts `mf_common.py`, `mf_formats.py`, `mf_kv.py`, `mf_kv2.py`, `mf_weights.py`, `mf_server.py`, `mf_info.py`;
run `python scripts/make_more_figures.py [fig_prefix ...]`). Older figures `fig1`-`fig10` are untouched.

Conventions on every figure: gray `#8c8c8c` = base (the three bases F16/Q8_0/Q4_0 are three grays), green `#1b9e77` =
ours lossless (light green = fused/secondary), orange `#e66101` = ours lossy, hatching = CPU-only prototype, `n/m` = not
measured, green arrow "better" on each data axis, English title. Every projection for
GPUs other than the RTX 3090 is marked `*` and carries the line "* analytic projection, to be validated on this hardware".
Amber is used only in the traffic-light figures (partly / doubtful).

"Derived" = plain arithmetic on numbers from the source files (ratio, difference, division by a count); no new measurement.

## Mapping to the requested list

| # | Requested |
| --- | --- |
| 1 | fp16 bit layout |
| 2 | Q8_0 / Q4_0 blocks + entropy |
| 3 | EXP3 format + escapes |
| 4 | Pareto size vs error |
| 5 | Pareto size vs read speed |
| 6 | heat map decode speed |
| 7 | heat map VRAM |
| 8 | stacked VRAM |
| 9 | absolute tok/s |
| 10 | bandwidth real vs peak |
| 11 | lazy pool vs reservation |
| 12 | roofline projections, other GPUs |
| 13 | context capacity by card |
| 14 | weights size vs KL / tasks |
| 15 | weights decode speed by recipe |
| 16 | combo model + KV |
| 17 | dashboard |
| 18 | pipeline infographic + verdict map |
| extra | quality, codecs, structure, costs |
| server block | (a)-(e) from the second message |

## Main figures 11-29

| File | What it shows | Numbers from | Caption EN |
| --- | --- | --- | --- |
| fig11_fp16_bit_layout | 16 boxes (sign 1 / exponent 5 / mantissa 10), entropy of each part for K and V, free bits, what can be compressed | R9 field entropies 1.00 / 2.89 (K) 3.31 (V) / 9.97; R10 (-13.6%, -15.3%); R33 (drop 2/4 bits: 0.875, 0.75-0.80; error 0.13%, 0.62%) | Inside one F16 number: the exponent has 2-3 spare bits, the mantissa has none. |
| fig12_q8_q4_blocks_entropy | Block layout of Q8_0 (34 B/32) and Q4_0 (18 B/32); entropy of codes and scale | R9 / `kv_entropy.json`: Q8 codes K 7.263 / V 7.591, by channel 6.670 / 7.422, scale 11.685 / 11.050 of 16, Q4 codes 3.376 / 3.680. Q4_0 scale entropy n/m | Quantized KV blocks leave little slack: 7.3 of 8 and 3.4 of 4 bits. |
| fig13_exp3_format_escapes | EXP3 scheme (base exponent + 3-bit offset + escape) and escape share / size for CPU groupings and GPU row layout | R11 (6.1, 1.8, 2.5, 2.3%; sizes 0.920, 0.906, 0.908, 0.907); `poc_b.jsonl` (derived: escapes / elements = 10.7%, 5.0%, 4.5%; ratio 0.960, 0.931, 0.929) | EXP3 stores 14 bits per value; 5-11% of values need the escape on the GPU layout. |
| fig14_pareto_size_vs_error | Size vs reconstruction error (symlog), K and V panels, 19 labelled formats incl. lossless on Q8/Q4 | R13, R33 (`kv_lossy_over_base results.json`, `kv_quant_lossless`), R26. Size vs F16 derived = bits/16 or ratio x base fraction | Lossless moves the bases left at the same error; lossy trades error for size. KL of our variants n/m. |
| fig15_pareto_size_vs_speed | Size vs read speed of one layer at 32K and 120K tokens; CPU lossless sizes with speed n/m | R33 `poc_c.jsonl` (speed = F16 time / format time, derived), R26 `poc_b.jsonl` for EXP3, R10/R33 CPU sizes | Fewer bytes did not give faster reads. |
| fig16_heatmap_decode_speed | Heat map: format x depth 4K/32K/64K/120K; decode in llama.cpp and read speed of our formats | R25 (0.883...0.622), R26 (EXP3 0.73/0.68/0.65/0.65), R33 (drop-4 0.854/0.977/n/m/0.925, Q8 7-bit 1.004/0.984/n/m/0.939, Q4 esc3 0.803/0.750/n/m/0.726). 64K of our formats n/m | Quantized KV loses speed with depth; our formats do not recover it. |
| fig17_heatmap_vram | Heat map: process VRAM (MiB) and change vs F16 for 4K/16K/32K/64K/120K | R18 (16K), R30 | The longer the context, the more quantized KV saves (Q4_0 -60% at 120K). |
| fig18_vram_stacked_breakdown | Stack: weights + KV buffer + compute buffer + other, F16/Q8/Q4 at 4K/32K/120K | R1 (2375.91 MiB), R18/R28 buffers, R30 totals; "other" = total - rest (derived, 274-280 MiB) | Weights are constant; KV dominates at long context. |
| fig19_absolute_toks | Decode tok/s (with p5-p95 of F16) and prefill tok/s vs depth | R25 (prefill Q8/Q4 = F16 x ratio, derived) | F16 is the fastest at every depth. |
| fig20_bandwidth_real_vs_peak | Implied GB/s vs 936 peak (404 vs 208 shared; 646/327/246; 775/286/177) and own kernel GB/s | R21, R25, R26 (`GBps_stored`, `GBps_f16_equiv`) | F16 reaches 83% of peak at 120K, quantized KV only 19-35%. |
| fig21_lazy_pool_vs_eager | Eager reservation vs lazy VMM pool by filled tokens; wasted MiB | R23 (0/144/576/2304/4608 vs 4608); waste derived | Process memory follows the filled tokens. Not integrated into llama.cpp. |
| fig22_server_roofline_projections * | Projected speed vs F16 at 32K and 120K for Q8/Q4/EXP3 (non-fused, fused) and F16 tok/s by card | R24, R31; model re-implemented in `mf_server.py`, max abs diff to `projections_long_20261021.md` = 0.005 | * Native Q8/Q4 kernels lose on server cards; fused EXP3 at best equals F16. |
| fig23_context_capacity_by_card * | Max KV context (K tokens, log) for F16/Q8/Q4/F16+EXP3 on 8 cards | R31 formula (reproduced; ours = F16 x 1/0.9286 derived) | * Capacity grows with fewer bytes per token regardless of kernel speed. |
| fig24_weights_size_vs_quality | Weights size vs KL (log) and vs tasks, bases, built-in types, our recipes, lossless | R7, R20, COMPARISON D-F; recipe MiB = ratio x base MiB (derived) | Our recipes move left almost for free; lossless shrinks little. |
| fig25_weights_recipe_speed | Decode speed of recipes vs own base, plus absolute tok/s | R34 / COMPARISON; absolute medians re-derived from `gpu_bases_20261021_rerun/w/*.jsonl` (ratios reproduce the table) | BF16 with FFN in Q8_0 decodes 17-20% faster. |
| fig26_combo_weights_plus_kv | Base combo vs combo + weight recipe, decode at 4K/32K; weights size | R35 / COMPARISON G; BF16 combo with recipe and all combo VRAM n/m | A weight recipe adds +1..+5% on top of a KV format. |
| fig27_dashboard | Six panels: VRAM, speed, KV size on bases, KL, weight recipes, weight lossless | R3, R19, R25, R26, R29, R30, R33, R34 | The whole study on one page. |
| fig28_pipeline_infographic | Data -> codecs -> GPU measurements -> verdict | RESULTS overview (R4, R9, R16, R25, R30 ...) | From data to verdict. |
| fig29_verdict_map | 20-row traffic light: worked / partly / did not work / not measured | RESULTS R-numbers on each row | What worked, what did not, what is not measured. |

## Extra figures 30-42

| File | What it shows | Numbers from | Caption EN |
| --- | --- | --- | --- |
| fig30_kv_kl_ladder | KL of six KV types at 2K; Q8/Q4 at 2K/4K/32K | R12, R19, R29 | Q8_0 KV is close to F16; 4-bit types cost KL 0.054-0.098. |
| fig31_quality_scoreboard | Tasks (KV types, weight types, recipes) and needle grid | R20, R27, R32, COMPARISON D-F | Q8_0 KV equals F16; Q3_K_M weights and below lose clearly. |
| fig32_lossless_f16_codecs_pages | Lossless F16 codec sizes by page size; CPU decode MB/s | R10 | Bit-exact CPU codecs reach -15% but decode far below GPU speed. |
| fig33_lossless_on_three_bases | Lossless sizes on F16/Q8_0/Q4_0, K and V | R33 `kv_quant_lossless` results.json, R10, R26 | Variable-rate coders help; fixed-rate ones are larger than Q8_0/Q4_0. |
| fig34_exp3_ksweep_and_gpu_cost | EXP3 size vs k = 1,2,3; GPU bytes vs read speed by T | R11, R26 | 4-7% fewer bytes, 27-35% slower reads. |
| fig35_compute_buffer_growth | Compute buffer vs context; KV-buffer saving vs process saving | R18, R28, R30 (differences derived: 472 MiB eaten at 120K) | Quantized KV has a growing compute buffer. |
| fig36_k_quant_field_entropy | Entropy per k-quant field and saved MB | R4 (saved MB derived) | Most of the ideal -6.6% sits in the code fields. |
| fig37_weights_headroom_vs_reality | Ladder: ideal 0.934, zlib big frames 0.956, small frames 0.996, CQW2 1.006 | R3, R4, R34 `quant.json` | The more GPU-shaped a format, the less it saves. |
| fig38_weights_lossless_headroom_by_file | Ideal headroom by file; ideal vs real | R16, R34 (`bf16.json`, `quant.json`) | Quantized weights: 2-7%; BF16: up to -33%. |
| fig39_outliers_and_sparsity | Outlier-clipping MSE ratios; sparsity statistics | R17 | Setting 1% outliers aside quarters the K error. |
| fig40_real_kv_structure | Entropy by data model, per-layer entropy, SVD rank | R9 / `kv_entropy.json` | Real KV is neither very compressible nor low-rank. |
| fig41_pocb_throughput_and_time | PoC-B stored GB/s and time per pass by T | R26 `poc_b.jsonl` | Q8_0/Q4_0 take as long as F16 with fewer bytes. |
| fig42_kv_size_vs_context | KV size vs tokens: formulas, measured dots, ours | R18/R28 buffers, R26, R10; 147,456 B/token | Size is linear and the formula matches llama.cpp buffers. |

## Server-GPU block (all `*`; analytic projections, not measurements)

Source for all five: R24, R31, `scripts/projections.py`, `scripts/projections_long.py`, `scripts/roofline_server.py` output,
PAPER_V1_EN.md section 8 (read only). Confidence per card: RTX 3090 high (measured basis), RTX 5090 / A100 / H100 / H200 medium,
L40S / RTX PRO 6000 Blackwell / B200 low. RTX PRO 6000 Blackwell and B200 have no scalar-throughput figure, so their calibrated
speed is n/m and only the bandwidth-only bound is drawn.

| File | What it shows | Caption EN |
| --- | --- | --- |
| figS1_server_speedup_by_card_context | Expected decode speed vs F16 per card at 4K/32K/120K for Q8_0, Q4_0, our lossless EXP3 and our lossy drop-4; bar = calibrated non-fused projection, whisker = bandwidth-only upper bound, white tick = fused (ours), X = measured 3090 | * Bar = this kernel design on that card; whisker = what a perfect fused kernel could approach. |
| figS2_server_capacity_and_max_batch | Heat maps: max KV context (K tokens) and max concurrent 32K sequences, base vs base + ours (EXP3, LZMA CPU only, drop-4, Q8/Q4 + LZMA) | * Capacity always gains; batch counts KV only. |
| figS3_ridge_point_and_scalar_budget | Ridge point (FLOP/B) and scalar FP32 FLOP/B per card | * A100/H100/H200 have 2-4x fewer scalar operations per byte than the 3090: scalar decompression loses there. |
| figS4_server_gain_or_loss_traffic_light | Cards x formats, traffic light gain / doubtful / loss at 32K and 120K, capacity column | * Speed loses on server cards; capacity gains everywhere. |
| figS5_hardware_validation_checklist | What to verify per card (PAPER section 8 "What to check"), done only for the 3090 | * No server card is validated yet. |

New derived quantities in the server block (not present in RESULTS.md): (1) the lossy drop-4 row uses the same model with a
decode excess of 1.08 ps/element derived from `gpu_bases_20261021/poc_c.jsonl` at T = 122,880 (EXP3 uses 2.01 ps/element from R26);
(2) capacity of "ours lossless" = F16 capacity x 1/0.9286; (3) the traffic-light thresholds (>= 1.03 green, 0.95-1.03 amber,
< 0.95 orange) are ours. The max-batch rows in figS2 count 32K-context sequences of KV only; they ignore weights beyond the
2.49 GB model, activations, and the compute-buffer growth of quantized KV (R28).

## n/m (drawn as "n/m", never as a number)

Huffman on F16 KV; Q4_0 scale entropy; speed of CPU lossless codecs on GPU; speed of our formats at 64K; KL/top-1/tasks of
our lossy KV variants; BF16 + recipe in a combo; VRAM of any combo; lossless weights + KV combo; scalar-based speed of
B200 and RTX PRO 6000 Blackwell; any measurement on a server GPU.

## Checks done

Every PNG was opened and checked for overlapping labels and palette. Cross-checks printed by the script: weight-recipe
speed ratios reproduce COMPARISON_TABLE (1.195/1.167, 1.055/1.036-1.037, 1.110/1.048, 0.982/0.950); the server
projection model reproduces `projections_long_20261021.md` within 0.005.

## Fork-stage figures (60-84)

Files in `figures/` (PNG). Numbers are in the R-sections of `RESULTS.md`; scripts are `fig_r62_bf16_weights.py`, `make_figures_fork.py`, `cqw_figures.py` and `make_figures_combo.py`.

| File | What it shows |
| --- | --- |
| fig60_bf16_weights_real | BF16 weights on real inference: size and decode speed of the lossless and lossy weight types against BF16 (Qwen3.5-2B) |
| fig62_cqw_bf16_decode_prefill_by_depth | BF16 weight types: decode and prefill speed by depth, ABBA series |
| fig80_combo_decode_ad | Combined compressed weights and KV: decode speed at 4K / 32K / 120K |
| fig81_combo_prefill_ad | Combined compressed weights and KV: prefill speed |
| fig82_combo_vram_120k | Combined compressed weights and KV: process VRAM at a 120K context |
| fig83_combo_quality_kl | Combined compressed weights and KV: KL against the uncompressed base |
| fig84_combo_qwen3_4b_kv | Combined fork, Qwen3-4B: decode, prefill and process VRAM at 120K for the KV types and decode kernels |
