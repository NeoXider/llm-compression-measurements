#!/usr/bin/env bash
# Exactness check (full model logits, batch + per-token) and kernel micro-benchmark, under the GPU lock.
# Usage: cqw_gpu_check.sh TAG   (env CONFIGS for kbench variants, MODELS for the logits check)
set -u
TAG=$1
P=/path/to/repo
B=/d/Git/llama.cpp-w/build-w/bin
W=$P/work/cqw
A=$P/artifacts/cqw_kbench/$TAG
mkdir -p $A/logits
w() { cygpath -w "$1"; }
source $P/scripts/cqw_gpu_lock.sh
cq_lock_acquire
declare -A M=( [bf16]=$P/work/qwen35_2b/qwen35-2b-BF16.gguf [h]=$W/qwen35-2b-BF16-cqH.gguf [w4]=$W/qwen35-2b-BF16-cqw4.gguf [w3]=$W/qwen35-2b-BF16-cqw3.gguf )
for k in ${MODELS:-bf16 h w4}; do
  timeout 600 $B/llama-cq-logits.exe -m "$(w ${M[$k]})" -f "$(w $W/prompt1.txt)" -n 64 -o "$(w $A/logits/$k.batch.bin)" > $A/logits/$k.batch.log 2>&1
  echo "$k batch exit=$?" >> $A/run.log
  timeout 600 $B/llama-cq-logits.exe -m "$(w ${M[$k]})" -f "$(w $W/prompt_short.txt)" -n 32 --per-token -o "$(w $A/logits/$k.tok.bin)" > $A/logits/$k.tok.log 2>&1
  echo "$k tok exit=$?" >> $A/run.log
done
for k in ${MODELS:-bf16 h w4}; do
  [ $k = bf16 ] && continue
  for mode in batch tok; do
    if cmp -s $A/logits/bf16.$mode.bin $A/logits/$k.$mode.bin; then r=IDENTICAL; else r=DIFFERENT; fi
    echo "$k $mode vs bf16: $r" >> $A/exact.txt
  done
done
rm -f $A/logits/*.bin
REF="$(w $P/work/qwen35_2b/qwen35-2b-BF16.gguf)"
CQ="--cq $(w $W/qwen35-2b-BF16-cqH.gguf) --cq $(w $W/qwen35-2b-BF16-cqw4.gguf) --cq $(w $W/qwen35-2b-BF16-cqw3.gguf)"
for cfg in ${CONFIGS:-base}; do
  envs=""; [ "$cfg" != base ] && envs=$(echo $cfg | tr ',' ' ')
  env $envs timeout 900 $B/llama-cq-kbench.exe --ref "$REF" $CQ --iters ${ITERS:-50} --reps 6 > $A/kbench_b1_$cfg.txt 2> $A/kbench_b1_$cfg.err
  echo "b1 $cfg exit=$?" >> $A/run.log
done
cq_lock_release
echo done >> $A/run.log
