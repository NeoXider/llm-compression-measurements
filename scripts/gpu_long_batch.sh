#!/usr/bin/env bash
# Long-context GPU batch (clean window expected: r04 finished, other GPU workloads stopped before start).
# Order: preflight, telemetry, PoC-B to 122880, filled-context VRAM 4K/32K/64K/120K, KV KL at 32K,
# needle-in-a-haystack 4K/32K, llama-bench speed ABBA at depths 4096/32768/65536/122880.
set -u
P=<release-root>
A=$P/artifacts/gpu_long_20261021
R=/d/Git/<project-repo>
LC=<llama.cpp-build-11552>
M=$R/models/Qwen3-4B-Q4_K_M.gguf
S=$R/kv-captures/native/m1k-diagnostic-f16-p2048-20260928-01/state.bin
KB=<user-temp>
w() { cygpath -w "$1"; }
mkdir -p $A $KB
log() { echo "$1 $(date -Is)" >> $A/batch.log; }
log start
python -B $P/scripts/gpu_clean_window.py preflight --samples 10 --output $A/preflight.json > $A/preflight.out 2>&1
python -B $P/scripts/gpu_clean_window.py log --seconds 14400 --stop-file $A/STOP --output $A/telemetry_1s.csv &
LOGPID=$!

# PoC-B
REPS=12 ITER=10 timeout 2400 $P/work/poc_bin/kvread_bench.exe "$(w $S)" 4096 32768 65536 122880 > $A/poc_b.jsonl 2> $A/poc_b.err
log "pocb exit=$?"

# Filled-context VRAM
timeout 5400 python -B $P/scripts/gpu_kv_memory_filled.py --cli "$(w $R/experimental/wddm-cuda-luid/cli-only/bin/llama-cli.exe)" \
  --validator "$(w $R/experimental/wddm-cuda-luid/validate_receipt.py)" --model "$(w $M)" \
  --case 4096=$(w $P/artifacts/prompts_filled/c4096.txt) --case 32768=$(w $P/artifacts/prompts_filled/c32768.txt) \
  --case 65536=$(w $P/artifacts/prompts_filled/c65536.txt) --case 122880=$(w $P/artifacts/prompts_filled/c122880.txt) \
  --repeats 2 --decode 16 --out-dir "$(w $A/memory_filled)" >> $A/batch.log 2>&1
log "memory exit=$?"

# KV KL at 32K (one chunk of a doubled corpus: the second half repeats the first, so this tests long-range recall)
C2=$(w $P/artifacts/corpus_v1/corpus_x2.txt)
timeout 1800 $LC/llama-perplexity.exe -m "$(w $M)" -f "$C2" -c 32768 -b 512 -ngl 99 -fa on -ctk f16 -ctv f16 --chunks 1 \
  --kl-divergence-base "$(w $KB/q34b_kvf16_c32768.kld)" > $A/kld_base_f16_c32768.log 2>&1
log "kl base exit=$?"
for kv in q8_0 q4_0; do
  timeout 1800 $LC/llama-perplexity.exe -m "$(w $M)" -c 32768 -b 512 -ngl 99 -fa on -ctk $kv -ctv $kv --chunks 1 \
    --kl-divergence-base "$(w $KB/q34b_kvf16_c32768.kld)" --kl-divergence > $A/kld_${kv}_c32768.log 2>&1
  log "kl $kv exit=$?"
done

# Needle in a haystack
timeout 3600 python -B $P/scripts/niah_eval.py --server "$(w $LC/llama-server.exe)" --model "$(w $M)" \
  --corpus "$(w $P/artifacts/corpus_v1/corpus.txt)" --lengths 4096,32768 --out "$(w $A/niah)" >> $A/batch.log 2>&1
log "niah exit=$?"

# Speed ABBA: depths 4096/32768/65536 with 2 ABBA blocks x 3 reps (12 samples); 122880 with 1 block x 3 reps (6 samples)
LB=$LC/llama-bench.exe
i=0
run_depth() { local d=$1 blocks=$2
  for blk in $(seq 1 $blocks); do
    for order in "f16 q8_0 q4_0" "q4_0 q8_0 f16"; do
      for kv in $order; do
        f=$(printf "%03d" $i)-d$d-b$blk-$kv
        timeout 3600 $LB -m "$(w $M)" -ngl 99 -fa 1 -ctk $kv -ctv $kv -p 512 -n 64 -d $d -r 3 -o jsonl > $A/speed/$f.jsonl 2> $A/speed/$f.err
        echo "$f exit=$? $(date -Is)" >> $A/speed/progress.txt
        i=$((i+1))
      done
    done
  done
}
mkdir -p $A/speed
cp $A/preflight.json $A/speed/preflight.json
run_depth 4096 2; run_depth 32768 2; run_depth 65536 2; run_depth 122880 1
log "speed done"
touch $A/STOP; wait $LOGPID
log end
