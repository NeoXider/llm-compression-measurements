#!/usr/bin/env bash
# Session 1 (weights fork): bit-exactness of lossless types + first speed look. Runs under the GPU lock.
set -u
P=/path/to/repo
B=/d/Git/llama.cpp-w/build-w/bin
W=$P/work/cqw
A=$P/artifacts/cqw_s1_20261022
mkdir -p $A/logits
w() { cygpath -w "$1"; }
source $P/scripts/cqw_gpu_lock.sh
cq_lock_acquire
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem --format=csv > $A/preflight.csv
nvidia-smi --query-compute-apps=pid,process_name --format=csv >> $A/preflight.csv
declare -A M=( [bf16]=$P/work/qwen35_2b/qwen35-2b-BF16.gguf [h]=$W/qwen35-2b-BF16-cqH.gguf [w4]=$W/qwen35-2b-BF16-cqw4.gguf [w3]=$W/qwen35-2b-BF16-cqw3.gguf )
for k in bf16 h w4 w3; do
  timeout 600 $B/llama-cq-logits.exe -m "$(w ${M[$k]})" -f "$(w $W/prompt1.txt)" -n 64 -o "$(w $A/logits/$k.batch.bin)" > $A/logits/$k.batch.log 2>&1
  echo "$k batch exit=$?" >> $A/run.log
  timeout 600 $B/llama-cq-logits.exe -m "$(w ${M[$k]})" -f "$(w $W/prompt_short.txt)" -n 32 --per-token -o "$(w $A/logits/$k.tok.bin)" > $A/logits/$k.tok.log 2>&1
  echo "$k tok exit=$?" >> $A/run.log
done
for k in h w4 w3; do
  for mode in batch tok; do
    if cmp -s $A/logits/bf16.$mode.bin $A/logits/$k.$mode.bin; then r=IDENTICAL; else r=DIFFERENT; fi
    echo "$k $mode vs bf16: $r" >> $A/exact.txt
  done
done
for k in bf16 h w4 w3; do
  timeout 900 $B/llama-bench.exe -m "$(w ${M[$k]})" -ngl 99 -fa 1 -p 512 -n 128 -r 3 -o jsonl > $A/bench_$k.jsonl 2> $A/bench_$k.err
  echo "$k bench exit=$?" >> $A/run.log
done
cq_lock_release
echo done >> $A/run.log
