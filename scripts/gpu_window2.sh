#!/usr/bin/env bash
# GPU window 2: E1/E4 KV captures + E2 (llama.cpp PRs vs bases) ABBA. Other GPU workloads are stopped first and always restarted on exit (trap).
P=<release-root>
A=$P/artifacts/window2_20261022
M=/d/Git/<project-repo>/models/Qwen3-4B-Q4_K_M.gguf
LC=<llama.cpp-build-11552>
CAP=<user-temp>; mkdir -p $A/e2 $CAP
L=$A/window.log
w() { cygpath -w "$1"; }
# operator: stop every other GPU workload before this point and restart it afterwards
sleep 5
python -B $P/scripts/gpu_clean_window.py preflight --samples 10 --output $A/preflight.json > $A/preflight.out 2>&1
cp $A/preflight.json $A/e2/preflight.json
python -B $P/scripts/gpu_clean_window.py log --seconds 14400 --stop-file $A/STOP --output $A/telemetry_1s.csv &
LOGPID=$!
T=$P/work/texts
timeout 3600 python -B $P/scripts/capture_kv_slots.py --server "$(w $LC/llama-server.exe)" --model "$(w $M)" \
  --text code=$(w $T/code.txt) --text en=$(w $T/en.txt) --text ru=$(w $T/ru.txt) \
  --plan f16:4096 --plan f16:30720 --plan bf16:4096 --out "$(w $CAP)" >> $L 2>&1
echo "captures exit=$? $(date -Is)" >> $L
# E2 ABBA: each PR vs its base, same KV/depth, order base,pr,pr,base
for pair in 30792 30202; do
  B0=<llama.cpp-checkout>-e2-base$pair/build-e2/bin/llama-bench.exe
  B1=<llama.cpp-checkout>-e2-pr$pair/build-e2/bin/llama-bench.exe
  [ -x $B0 ] && [ -x $B1 ] || { echo "missing binaries for $pair" >> $L; continue; }
  for kv in f16 q8_0 q4_0; do
    for d in 4096 32768 122880; do
      i=0
      for which in base pr pr base; do
        BIN=$B0; [ $which = pr ] && BIN=$B1
        f=$A/e2/$pair-$kv-d$d-$i-$which
        timeout 1800 $BIN -m "$(w $M)" -ngl 99 -fa 1 -ctk $kv -ctv $kv -p 512 -n 64 -d $d -r 3 -o jsonl > $f.jsonl 2> $f.err
        echo "$pair $kv $d $i $which exit=$? $(date -Is)" >> $A/e2/progress.txt
        i=$((i+1))
      done
    done
  done
done
touch $A/STOP; wait $LOGPID
echo "window2 done $(date -Is)" >> $L
