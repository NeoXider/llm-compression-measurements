# KV-cache and weight compression for GPU-resident LLM inference: measurements on a consumer GPU

Companion material for the paper *"How Much Can Be Added on Top of Existing KV-Cache and Weight Formats? An
Empirical Study of Lossless and Lossy Compression for GPU-Resident LLM Inference on a Consumer GPU"*
(`paper/PAPER_V2_EN.md`). It is an empirical measurement study with negative results and proposes no new method.

DOI: https://doi.org/10.5281/zenodo.23578725 (Zenodo archive of v3.0.1)

Also on Hugging Face (paper PDF and records): https://huggingface.co/datasets/neoxider2/llm-compression-measurements

Status: v3.0, 2026-10-23. License: MIT for code and scripts; the paper text and the figures are also available
under CC BY 4.0 (see `LICENSE`). Author: NeoXider (GitHub: NeoXider).

## What is here

| Folder | Content |
|---|---|
| `paper/` | the paper (Markdown, canonical) and `paper.tex` (generated, not compiled locally) |
| `docs/` | `RESULTS.md` (all measurement records R1-R87 with methods and caveats), `METHODS.md`, `RELATED_WORK.md`, `FIGURES_INDEX.md` |
| `patches/` | llama.cpp patches on commit `ed7ac35e`: `0001-cq-kv-emu-set-rows.patch` (exact emulation of lossy KV formats), `kv/` (packed KV types `f16d4`, `f16e3`, `q4o`, GQA decode kernel), `weights/` (compressed BF16 weight types), `combined/` (merge of both and the `llama-cq-logits --kv` tool) |
| `poc/` | CUDA proofs of concept for KV-read formats, base-versus-base-plus-ours reads and a BF16 GEMV |
| `scripts/` | measurement, analysis and figure scripts (entropy estimates, lossless page codecs, KL/tasks parsing, GPU preflight and telemetry, llama-bench ABBA series, the `combo_*` and `cqw_*` fork series) |
| `data/` | result summaries used in the paper tables (`kv/`, `weights/`, `gpu/`, `fork_stage1/`, `e2_llamacpp_prs/`, `combo/`, `cqw/`); raw logs and model files are not included |
| `figures/` | all figures of the paper (English-only) |

## Hardware and software

RTX 3090 24 GB (Windows 11, WDDM, PCIe Gen3 x16), i7-10700K, 64 GB RAM; llama.cpp pinned at commit `ed7ac35e`
(fork branches `cq/kv-compress`, `cq/weights-compress`, `cq/combined`; the patches here reproduce them);
build 11552 `23b0202a` for stock comparisons; CUDA 11.8; Python 3.13.

Models: Qwen3-4B-Q4_K_M (official GGUF) and Qwen3.5-2B (converted from the official BF16 checkpoint and quantized
locally, with and without imatrix).

## Reproducing

1. KV and weight entropy and codec estimates run on the CPU from a captured F16 KV state or a GGUF file:
   `scripts/kv_entropy.py`, `kv_lossless_proto.py`, `kv_fixed_rate.py`, `kv_lossy_proxy.py`, `gguf_quant_entropy.py`.
2. Quality: `llama-perplexity --kl-divergence-base / --kl-divergence`, parsed with `scripts/parse_kld_logs.py`;
   tasks with `scripts/cpu_task_eval.py` (needs a task file in the same schema; ours is not included).
3. Fork builds: apply the patches under `patches/` to `ed7ac35e`, build with CUDA 11.8 for `sm_86`
   (`-DGGML_CUDA_FA_QUANTS` as listed in `patches/combined/`).
4. Speed: `scripts/gpu_clean_window.py preflight` must report a clean window, then the `combo_*` and `cqw_*` series
   (llama-bench depth series in ABBA order, 12 samples per cell). All windows in our series were marked dirty by
   the preflight (desktop applications used 4-28% of the GPU while idle); no other compute process ran.
5. Other GPUs: the CUDA proofs of concept build for other architectures; rows for server GPUs in this repository
   are analytic projections (marked with a star), not measurements.

## Main measured findings (details and caveats in the paper)

Existing formats serve as baselines (KV F16/Q8_0/Q4_0; weights BF16/Q8_0/Q4_K_M). We apply compression on top of
each baseline and compare with that baseline in the same binary.

- **Lossless weights, BF16 (Qwen3.5-2B):** a fixed-rate format (W4) is 24% smaller, decodes 1.03-1.05x as fast as BF16
  and gives bit-identical logits. An entropy-coded format (Huffman exponent, as in DFloat11) is 30% smaller but
  decodes at 0.63-0.73x.
- **Lossy weights:** mixed-precision recipes are the practical option: BF16 with the FFN in Q8_0 is 23% smaller and
  1.15-1.19x faster; Q8_0 with the FFN in Q6_K is 11% smaller (1.00-1.06x); Q4_K_M with embedding and output in Q4_K
  is 10% smaller (1.01-1.08x, slightly worse KL). With an importance matrix, Q4_K_M keeps its size and gets better KL.
- **KV, F16 base:** a 12-bit packed type (`f16d4`, 4 low mantissa bits dropped) has the quality of Q8_0 (KL 0.0028),
  is 20% smaller at a filled 120K context and decodes 1.15x as fast as F16 on Qwen3-4B. The lossless type
  (`f16e3`, 15.25 bits per value) is bit-exact but saves under 5% of KV bytes, decodes slowly (0.14-0.85x) and its
  overflow pool fills up after about 8K tokens of real text, so its long-context rows are not lossless.
- **KV, Q8_0 and Q4_0 bases:** a GQA-grouped decode kernel serves the same bytes at 1.11x (Q8_0) and 1.16x (Q4_0) of
  F16 speed at 120K on Qwen3-4B (41.6 and 43.7 tokens/s against 37.6), where llama.cpp's stock kernels reach 0.64x
  and 0.60x. Process VRAM is 12.5 and 8.1 GB instead of 20.3 GB. The comparison is against the stock kernels before
  upstream PR #30202, which closes part of the gap. On Qwen3.5-2B (KV in 6 of 24 layers) the same kernel is slower
  than the stock one.
- **Q4_0 KV quality:** the loss comes from the Q4_0 format (KL 0.068 with llama.cpp's rotation, 0.116 without), not from
  the computation path; an earlier emulation that quantized only V suggested otherwise (corrected in R44). Keeping
  2 of every 128 values exact (`q4o`, +11% bytes) halves the KL (0.033) when the rotation is off.
- **Lossless on top of Q8_0 and Q4_0:** variable-rate coders save 7-21% on the CPU; every fixed-rate format we built is
  larger than its base, so no GPU type exists for these bases.
- **Lazily mapped KV buffer:** process VRAM follows the filled tokens (20.1 GB eager against 3.0 GB at 702 tokens of a
  120K context); a known technique, reproduced in llama.cpp.
- **Combined (compressed weights and KV together)**, Qwen3.5-2B: W3 plus `f16d4` uses 25% less memory with KL 0.001 but decodes
  at 1.09x / 0.99x / 0.82x at 4K / 32K / 120K.

Server-GPU numbers in this repository are projections, not measurements.

## Limits

One consumer GPU, two models, windows marked dirty by our own preflight, KL measured on prefill at 4K context only
(long-context quality rests on needle retrieval), 39-task suite too small to separate the variants. See the
Limitations section of the paper.
