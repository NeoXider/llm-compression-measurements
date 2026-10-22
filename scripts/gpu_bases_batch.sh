#!/usr/bin/env bash
# GPU batch for the base-vs-base+ours tables. Other GPU workloads must be stopped before start.
# 1) PoC-C: KV read F16 / F16+drop4 / Q8_0 / Q8_0+7bit / Q4_0 / Q4_0+esc3 at T = 4096, 32768, 122880.
# 2) Weights (Qwen3.5-2B): BF16, Q8_0, Q4_K_M and our lossy recipes; llama-bench decode/prefill at depth 0 and 4096,
#    KV F16, ABBA over models, 3 reps x 2 blocks = 6 samples per cell.
# 3) Combos (Qwen3.5-2B): BF16+F16 KV, Q8_0+Q8_0 KV, Q4_K_M+Q4_0 KV, and recipe+KV at depth 4096 and 32768.
set -u
P=<release-root>
A=$P/artifacts/gpu_bases_20261021
W=$P/work/qwen35_2b
LB=<llama.cpp-build-11552>/llama-bench.exe
S=/d/Git/<project-repo>/kv-captures/native/m1k-diagnostic-f16-p2048-20260928-01/state.bin
w() { cygpath -w "$1"; }
mkdir -p $A/w $A/c
log() { echo "$1 $(date -Is)" >> $A/batch.log; }
log start
python -B $P/scripts/gpu_clean_window.py preflight --samples 10 --output $A/preflight.json > $A/preflight.out 2>&1
python -B $P/scripts/gpu_clean_window.py log --seconds 7200 --stop-file $A/STOP --output $A/telemetry_1s.csv &
LOGPID=$!
REPS=12 ITER=10 timeout 2400 $P/work/poc_bin/kvread_base_plus.exe "$(w $S)" 4096 32768 122880 > $A/poc_c.jsonl 2> $A/poc_c.err
log "pocc exit=$?"
MODELS="BF16 Q8_0 Q4_K_M BF16+ffnQ8 Q8_0+ffnQ6K Q4_K_M+ffnupgateQ3K Q4_K_M+embQ4K"
i=0
for blk in 1 2; do
  ORDER=$MODELS; [ $blk = 2 ] && ORDER=$(echo $MODELS | tr ' ' '\n' | tac | tr '\n' ' ')
  for m in $ORDER; do
    f=$(printf "%03d" $i)-$m-b$blk
    timeout 900 $LB -m "$(w $W/qwen35-2b-$m.gguf)" -ngl 99 -fa 1 -p 512 -n 64 -d 0,4096 -r 3 -o jsonl > $A/w/$f.jsonl 2> $A/w/$f.err
    echo "$f exit=$?" >> $A/w/progress.txt; i=$((i+1))
  done
done
log "weights done"
COMBOS="BF16:f16 Q8_0:q8_0 Q4_K_M:q4_0 Q8_0+ffnQ6K:q8_0 Q4_K_M+embQ4K:q4_0"
i=0
for blk in 1 2; do
  ORDER=$COMBOS; [ $blk = 2 ] && ORDER=$(echo $COMBOS | tr ' ' '\n' | tac | tr '\n' ' ')
  for c in $ORDER; do
    m=${c%%:*}; kv=${c##*:}
    f=$(printf "%03d" $i)-$m-$kv-b$blk
    timeout 1200 $LB -m "$(w $W/qwen35-2b-$m.gguf)" -ngl 99 -fa 1 -ctk $kv -ctv $kv -p 512 -n 64 -d 4096,32768 -r 3 -o jsonl > $A/c/$f.jsonl 2> $A/c/$f.err
    echo "$f exit=$?" >> $A/c/progress.txt; i=$((i+1))
  done
done
log "combos done"
touch $A/STOP; wait $LOGPID
log end
