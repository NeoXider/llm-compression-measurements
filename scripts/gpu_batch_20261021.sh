#!/usr/bin/env bash
# One GPU batch (2026-10-21). Load-independent measurements only: per-process VRAM, KL/PPL, task accuracy.
# Timing values captured on the way are diagnostic (window was DIRTY: foreign training PID 35196).
set -u
P=<release-root>
A=$P/artifacts/gpu_batch_20261021
R=/d/Git/<project-repo>
LC=<llama.cpp-build-11552>
M4=$R/models/Qwen3-4B-Q4_K_M.gguf
W2=$P/work/qwen35_2b
KB=<user-temp>
mkdir -p $A $KB
w() { cygpath -w "$1"; }
echo "start $(date -Is)" > $A/batch.log
python -B $P/scripts/gpu_clean_window.py log --seconds 10800 --stop-file $A/STOP --output $A/telemetry_1s.csv &
LOGPID=$!

# 1. Filled-context per-process memory, Qwen3-4B Q4_K_M, KV f16/q8_0/q4_0, ctx 4K/16K/32K, 2 interleaved repeats
python -B $P/scripts/gpu_kv_memory_filled.py --cli "$(w $R/experimental/wddm-cuda-luid/cli-only/bin/llama-cli.exe)" \
  --validator "$(w $R/experimental/wddm-cuda-luid/validate_receipt.py)" --model "$(w $M4)" \
  --case 4096=$(w $P/artifacts/prompts_filled/c4096.txt) --case 16384=$(w $P/artifacts/prompts_filled/c16384.txt) \
  --case 32768=$(w $P/artifacts/prompts_filled/c32768.txt) --repeats 2 --decode 32 \
  --out-dir "$(w $A/memory_filled)" >> $A/batch.log 2>&1
echo "memory exit=$? $(date -Is)" >> $A/batch.log

# 2. KL of native KV quantization vs KV F16, Qwen3-4B Q4_K_M, ctx 4096, same corpus
C=$(w $P/artifacts/corpus_v1/corpus.txt)
$LC/llama-perplexity.exe -m "$(w $M4)" -f "$C" -c 4096 -b 512 -ngl 99 -fa on -ctk f16 -ctv f16 \
  --kl-divergence-base "$(w $KB/q3_4b_kvf16_c4096.kld)" > $A/kv_kld_base_f16.log 2>&1
echo "kv base exit=$? $(date -Is)" >> $A/batch.log
for kv in q8_0 q4_0; do
  $LC/llama-perplexity.exe -m "$(w $M4)" -c 4096 -b 512 -ngl 99 -fa on -ctk $kv -ctv $kv \
    --kl-divergence-base "$(w $KB/q3_4b_kvf16_c4096.kld)" --kl-divergence > $A/kld_kv_$kv.log 2>&1
  echo "kv $kv exit=$? $(date -Is)" >> $A/batch.log
done

# 3. 39-case suite, Qwen3-4B Q4_K_M, KV f16/q8_0/q4_0, ctx 16384, GPU
for kv in f16 q8_0 q4_0; do
  python -B $P/scripts/cpu_task_eval.py --gpu --kv-type $kv --server "$(w $LC/llama-server.exe)" \
    --model qwen3_4b_q4km_kv_$kv="$(w $M4)" --ctx-size 16384 --port 18941 \
    --out-dir "$(w $A/tasks_qwen3_4b_kv_$kv)" >> $A/batch.log 2>&1
  echo "tasks kv $kv exit=$? $(date -Is)" >> $A/batch.log
done

# 4. 39-case suite, Qwen3.5-2B weight-quant curve, GPU, default KV
ARGS=""
for t in BF16 Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K; do ARGS="$ARGS --model $t=$(w $W2/qwen35-2b-$t.gguf)"; done
python -B $P/scripts/cpu_task_eval.py --gpu --server "$(w $LC/llama-server.exe)" $ARGS --ctx-size 8192 --port 18942 \
  --out-dir "$(w $A/tasks_qwen35_2b_gpu)" >> $A/batch.log 2>&1
echo "tasks 2b exit=$? $(date -Is)" >> $A/batch.log

touch $A/STOP; wait $LOGPID
echo "end $(date -Is)" >> $A/batch.log
