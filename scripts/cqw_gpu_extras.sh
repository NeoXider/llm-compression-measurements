#!/usr/bin/env bash
# Extras under the GPU lock: final kernel micro-benchmark (all formats in one session), batch-512 GEMM with
# decode-to-BF16, and prefill with a larger ubatch (decode cost per token shrinks with the ubatch size).
set -u
P=/path/to/repo
B=/d/Git/llama.cpp-w/build-w/bin
W=$P/work/qwen35_2b
C=$P/work/cqw
A=$P/artifacts/cqw_kbench/final
mkdir -p $A/ub
w() { cygpath -w "$1"; }
source $P/scripts/cqw_gpu_lock.sh
cq_lock_acquire
log() { echo "$(date -Iseconds) $*" >> $A/run.log; }
REF="$(w $W/qwen35-2b-BF16.gguf)"
CQ="--cq $(w $C/qwen35-2b-BF16-cqH.gguf) --cq $(w $C/qwen35-2b-BF16-cqw4.gguf) --cq $(w $C/qwen35-2b-BF16-cqw3.gguf)"
timeout 1200 $B/llama-cq-kbench.exe --ref "$REF" $CQ --iters 50 --reps 10 > $A/kbench_b1_base.txt 2> $A/kbench_b1_base.err
log "b1 exit=$?"
timeout 1200 $B/llama-cq-kbench.exe --ref "$REF" $CQ --iters 10 --reps 6 --batch 512 > $A/kbench_b512.txt 2> $A/kbench_b512.err
log "b512 exit=$?"
j=0
for blk in 1 2; do
  for m in BF16 H W4 W3; do
    case $m in BF16) f=$W/qwen35-2b-BF16.gguf;; H) f=$C/qwen35-2b-BF16-cqH.gguf;; W4) f=$C/qwen35-2b-BF16-cqw4.gguf;; W3) f=$C/qwen35-2b-BF16-cqw3.gguf;; esac
    for ub in 512 2048; do
      timeout 900 $B/llama-bench.exe -m "$(w $f)" -ngl 99 -fa 1 -p 2048 -n 0 -b 2048 -ub $ub -r 3 -o jsonl > $A/ub/$(printf %03d $j)-$m-ub$ub-b$blk.jsonl 2>/dev/null
      log "ub $m $ub exit=$?"; j=$((j+1))
    done
  done
done
cq_lock_release
log end
