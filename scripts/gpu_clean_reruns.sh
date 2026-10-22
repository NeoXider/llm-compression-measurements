#!/usr/bin/env bash
# Clean-window reruns (LM Studio closed, other GPU jobs stopped, no CPU jobs): PoC-D BF16 GEMV,
# Qwen3-4B KV F16/Q8/Q4 decode+prefill ABBA at 4K/32K/120K, Qwen3.5-2B weight recipes ABBA at d=0/4096.
set -u
P=<release-root>
A=$P/artifacts/clean_reruns_20261021
LB=<llama.cpp-build-11552>/llama-bench.exe
M4=/d/Git/<project-repo>/models/Qwen3-4B-Q4_K_M.gguf
W=$P/work/qwen35_2b
w() { cygpath -w "$1"; }
mkdir -p $A/kv $A/w
log() { echo "$1 $(date -Is)" >> $A/batch.log; }
log start
python -B $P/scripts/gpu_clean_window.py preflight --samples 10 --output $A/preflight.json > $A/preflight.out 2>&1
cp $A/preflight.json $A/kv/preflight.json
python -B $P/scripts/gpu_clean_window.py log --seconds 10800 --stop-file $A/STOP --output $A/telemetry_1s.csv &
LOGPID=$!
REPS=12 ITER=20 timeout 900 $P/work/poc_bin/bf16_gemv_bench.exe "$(w $P/work/bf16/ffn_up_rows.bin)" 49152 2048 > $A/poc_d_bf16_gemv.jsonl 2> $A/poc_d.err
log "pocd exit=$?"
i=0
kv_depth() { local d=$1 blocks=$2
  for blk in $(seq 1 $blocks); do
    for order in "f16 q8_0 q4_0" "q4_0 q8_0 f16"; do
      for kv in $order; do
        f=$(printf "%03d" $i)-d$d-b$blk-$kv
        timeout 3600 $LB -m "$(w $M4)" -ngl 99 -fa 1 -ctk $kv -ctv $kv -p 512 -n 64 -d $d -r 3 -o jsonl > $A/kv/$f.jsonl 2> $A/kv/$f.err
        echo "$f exit=$?" >> $A/kv/progress.txt; i=$((i+1))
      done
    done
  done
}
kv_depth 4096 2; kv_depth 32768 2; kv_depth 122880 1
log "kv done"
MODELS="BF16 BF16+ffnQ8 Q8_0 Q8_0+ffnQ6K Q4_K_M Q4_K_M+embQ4K"
j=0
for blk in 1 2; do
  ORDER=$MODELS; [ $blk = 2 ] && ORDER=$(echo $MODELS | tr ' ' '\n' | tac | tr '\n' ' ')
  for m in $ORDER; do
    f=$(printf "%03d" $j)-$m-b$blk
    timeout 900 $LB -m "$(w $W/qwen35-2b-$m.gguf)" -ngl 99 -fa 1 -p 512 -n 64 -d 0,4096 -r 3 -o jsonl > $A/w/$f.jsonl 2> $A/w/$f.err
    echo "$f exit=$?" >> $A/w/progress.txt; j=$((j+1))
  done
done
log "weights done"
touch $A/STOP; wait $LOGPID
log end
