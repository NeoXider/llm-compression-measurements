#!/usr/bin/env bash
# Speed series: Qwen3-4B Q4_K_M, KV f16/q8_0/q4_0, depth 0/4096/16384/32768 (filled context via -d),
# prefill 512 tokens and decode 64 tokens at that depth, -r 3 per call, blocks in ABBA order
# (f16,q8_0,q4_0 then q4_0,q8_0,f16), BLOCKS rounds. 1 s telemetry runs alongside.
# Speed is compared only within the interleaving; window labelled shared-GPU unless preflight is CLEAN.
set -u
P=<release-root>
A=$P/artifacts/gpu_speed_abba_${RUN:-20261021}
LB=<llama.cpp-build-11552>/llama-bench.exe
M=$(cygpath -w /d/Git/<project-repo>/models/Qwen3-4B-Q4_K_M.gguf)
BLOCKS=${BLOCKS:-2}
mkdir -p $A
python -B $P/scripts/gpu_clean_window.py preflight --samples 5 --output $A/preflight.json > $A/preflight.out 2>&1
python -B $P/scripts/gpu_clean_window.py log --seconds 10800 --stop-file $A/STOP --output $A/telemetry_1s.csv &
LOGPID=$!
i=0
for blk in $(seq 1 $BLOCKS); do
  for order in "f16 q8_0 q4_0" "q4_0 q8_0 f16"; do
    for kv in $order; do
      for d in 0 4096 16384 32768; do
        f=$(printf "%03d" $i)-b$blk-$kv-d$d
        $LB -m "$M" -ngl 99 -fa 1 -ctk $kv -ctv $kv -p 512 -n 64 -d $d -r 3 -o jsonl > $A/$f.jsonl 2> $A/$f.err
        echo "$f exit=$? $(date -Is)" >> $A/progress.txt
        i=$((i+1))
      done
    done
  done
done
touch $A/STOP; wait $LOGPID
echo done >> $A/progress.txt
