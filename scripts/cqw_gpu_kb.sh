#!/usr/bin/env bash
# Kernel micro-benchmark session (weights fork) under the GPU lock. Usage: cqw_gpu_kb.sh TAG
# Runs kbench for each env configuration in $CONFIGS (space-separated, items like "CQ_ROWS_H=2,CQ_ROWS_W=2" or "base").
set -u
TAG=$1
P=/path/to/repo
B=/d/Git/llama.cpp-w/build-w/bin
W=$P/work/cqw
A=$P/artifacts/cqw_kbench/$TAG
mkdir -p $A
w() { cygpath -w "$1"; }
source $P/scripts/cqw_gpu_lock.sh
cq_lock_acquire
REF="$(w $P/work/qwen35_2b/qwen35-2b-BF16.gguf)"
CQ="--cq $(w $W/qwen35-2b-BF16-cqH.gguf) --cq $(w $W/qwen35-2b-BF16-cqw4.gguf) --cq $(w $W/qwen35-2b-BF16-cqw3.gguf)"
for cfg in ${CONFIGS:-base}; do
  envs=""; [ "$cfg" != base ] && envs=$(echo $cfg | tr ',' ' ')
  env $envs timeout 900 $B/llama-cq-kbench.exe --ref "$REF" $CQ --iters ${ITERS:-50} --reps 6 > $A/kbench_b1_$cfg.txt 2> $A/kbench_b1_$cfg.err
  echo "b1 $cfg exit=$?" >> $A/run.log
done
if [ -n "${B512:-}" ]; then
  timeout 900 $B/llama-cq-kbench.exe --ref "$REF" $CQ --iters 10 --reps 4 --batch 512 --tensor '^blk\.5\.(ffn_up|ffn_down)\.weight$' > $A/kbench_b512.txt 2> $A/kbench_b512.err
  echo "b512 exit=$?" >> $A/run.log
fi
cq_lock_release
echo done >> $A/run.log
