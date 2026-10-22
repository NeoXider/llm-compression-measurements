# REPRODUCE

Placeholders: `<llama.cpp-build>` = a llama.cpp build with llama-bench / llama-perplexity / llama-server
(we used build 11552, commit 23b0202a); `<state>` = a llama.cpp sequence-state file with F16 KV; `<model>` =
Qwen3-4B-Q4_K_M.gguf (official, SHA256 7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5).
Every script refuses to overwrite an existing output.

## CPU only

```bash
python scripts/kv_entropy.py          --state <state> --output out/kv_entropy.json
python scripts/kv_lossless_proto.py   --state <state> --page-tokens 16 64 256 --output out/kv_pages.json
python scripts/kv_fixed_rate.py       --state <state> --output out/kv_fixed_rate.json
python scripts/kv_lossy_proxy.py      --state <state> --layer-stride 4 --output out/kv_proxy.json
python scripts/sparsity_outlier_stats.py --model <model> --state <state> --output out/sparsity.json
python scripts/gguf_quant_entropy.py  --model <file.gguf> --output out/weights_entropy.json
# KL / PPL (CUDA hidden with CUDA_VISIBLE_DEVICES=-1, or -ngl 99 on GPU)
<llama.cpp-build>/llama-perplexity -m ref.gguf -f corpus.txt -c 512 --kl-divergence-base ref.kld
<llama.cpp-build>/llama-perplexity -m cand.gguf -c 512 --kl-divergence-base ref.kld --kl-divergence
<llama.cpp-build>/llama-perplexity -m <model> -f corpus.txt -c 2048 --chunks 6 -fa on -ctk f16 -ctv f16 --kl-divergence-base kv.kld
<llama.cpp-build>/llama-perplexity -m <model> -c 2048 --chunks 6 -fa on -ctk q8_0 -ctv q8_0 --kl-divergence-base kv.kld --kl-divergence
```

## GPU (clean window required)

```bash
python scripts/gpu_clean_window.py preflight --samples 10 --output out/preflight.json   # must be CLEAN
RUN=myrun BLOCKS=2 bash scripts/gpu_speed_abba.sh                                       # llama-bench ABBA, depths 0/4K/16K/32K
python scripts/parse_speed_abba.py --dir out/gpu_speed_abba_myrun
poc/build.bat                                     # set ARCH=sm_80 (A100), sm_90 (H100/H200), sm_86 (RTX 30xx)
REPS=12 ITER=20 poc/bin/kvread_bench.exe <state> 4096 8192 16384 32768 > out/poc_b.jsonl
poc/bin/lazy_pool.exe > out/poc_a.jsonl          # Windows (DXGI); on Linux replace local_mib() with nvidia-smi/NVML
```

Filled-context per-process VRAM (`scripts/gpu_kv_memory_filled.py`) needs a llama-cli build with the
DXGI in-process observer, which is not part of this release. `poc/lazy_pool.cu` shows the measurement
method. On server GPUs (Linux, TCC-like), NVML per-process memory can be used instead.

## Lossy KV quality with exact emulation (Stage 1, paper §4.1–4.3)

Apply `patches/0001-cq-kv-emu-set-rows.patch` to llama.cpp commit `ed7ac35e`, build with CUDA
(`cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`; targets llama-perplexity, llama-server). With
F16 KV, the env var `CQ_KV_EMU` selects the emulated format on cache write: `f16drop2`, `f16drop4`, `q8_0`, `q8_7bit`,
`q8_6bit_out1`, `q4_0`, `q4_out1`, `q3_out1` (unset = plain F16). `scripts/gpu_stage1_emu.sh` runs KL vs F16 and vs
the own base at 4K and 32K, the 39-case task suite (`cpu_task_eval.py --gpu`) and needle retrieval (`niah_eval.py`);
`scripts/summarize_stage1.py <dir>` writes the summary.

## E1/E4: context entropy and TRACE-like codec on F16 and BF16 KV (CPU after a GPU capture)

```bash
python scripts/capture_kv_slots.py --server <llama.cpp-build>/llama-server --model <model> \
  --text code=code.txt --text en=en.txt --text ru=ru.txt --plan f16:4096 --plan f16:30720 --plan bf16:4096 --out captures/
python scripts/kv_e1_context.py --state captures/code_f16_4096.bin --state captures/code_bf16_4096.bin \
  --layer-stride 6 --groups 16,64 --output out/e1.json     # accepts native state files and server slot saves
```

## E2: upstream llama.cpp PRs vs their base commits

`scripts/e2_build_prs.sh` builds llama-bench for PR #30202 (`b9e92c43`, base `dac30873`) and PR #30792 (`3c253754`,
base `99db719b`) in separate worktrees. `scripts/gpu_window2.sh` runs base–PR–PR–base for KV f16/q8_0/q4_0 at
depths 4096/32768/122880 (`-fa 1 -p 512 -n 64 -r 3`); `scripts/summarize_e2.py <dir>` writes the tables.

## E3 and PoC-C/PoC-D

```bash
python scripts/e3_int4_entropy.py --model <Qwen3.5-2B-BF16.gguf> --gguf-root <llama.cpp>/gguf-py --output out/e3.json
poc/bin/kvread_base_plus.exe <state> 4096 32768 122880 > out/poc_c.jsonl          # base vs base + our KV formats
python scripts/export_bf16_rows.py --model <Qwen3.5-2B-BF16.gguf> --gguf-root <llama.cpp>/gguf-py --output rows.bin
poc/bin/bf16_gemv_bench.exe rows.bin <rows> <cols> > out/poc_d.jsonl               # BF16 GEMV vs fixed-rate lossless BF16
```

## Server-GPU validation of the projections (*)

Run, on the target GPU: (1) `gpu_clean_window.py preflight`; (2) `kvread_bench` (PoC-B) with
`ARCH` set for that GPU; (3) `gpu_speed_abba.sh`; (4) compare the ratios with
`data/analytic/projections.md`. A few hours of rented time per GPU type are enough.
