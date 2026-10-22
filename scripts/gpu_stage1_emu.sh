#!/usr/bin/env bash
# Stage 1: quality of OUR lossy KV variants via the CQ_KV_EMU hook (branch cq/kv-compress build).
# Qwen3-4B Q4_K_M, F16 KV cache with emulated codec on write. Other GPU workloads must be stopped before start.
set -u
P=<release-root>
A=$P/artifacts/stage1_emu_20261021
B=<llama.cpp-checkout>-kvc/build-kvc/bin
M=/d/Git/<project-repo>/models/Qwen3-4B-Q4_K_M.gguf
KB=<user-temp>
w() { cygpath -w "$1"; }
mkdir -p $A $KB
log() { echo "$1 $(date -Is)" >> $A/batch.log; }
log start
python -B $P/scripts/gpu_clean_window.py preflight --samples 10 --output $A/preflight.json > $A/preflight.out 2>&1
python -B $P/scripts/gpu_clean_window.py log --seconds 10800 --stop-file $A/STOP --output $A/telemetry_1s.csv &
LOGPID=$!
PPL=$B/llama-perplexity.exe
run_ppl() {  # name ctx corpus chunks emu kvtype extra...
  local name=$1 ctx=$2 corpus=$3 chunks=$4 emu=$5 kv=$6; shift 6
  if [ "$emu" = none ]; then unset CQ_KV_EMU; else export CQ_KV_EMU=$emu; fi
  timeout 2400 $PPL -m "$(w $M)" -f "$(w $corpus)" -c $ctx -b 512 -ngl 99 -fa on -ctk $kv -ctv $kv --chunks $chunks "$@" > $A/$name.log 2>&1
  log "$name exit=$?"
  unset CQ_KV_EMU
}
for ctx in 4096 32768; do
  if [ $ctx = 4096 ]; then C=$P/artifacts/corpus_v1/corpus.txt; N=6; else C=$P/artifacts/corpus_v1/corpus_x3.txt; N=1; fi
  # bases (store logits)
  run_ppl base_f16_c$ctx $ctx $C $N none f16 --kl-divergence-base "$(w $KB/f16_c$ctx.kld)"
  run_ppl base_q8emu_c$ctx $ctx $C $N q8_0 f16 --kl-divergence-base "$(w $KB/q8emu_c$ctx.kld)"
  run_ppl base_q4emu_c$ctx $ctx $C $N q4_0 f16 --kl-divergence-base "$(w $KB/q4emu_c$ctx.kld)"
  # every variant vs F16 KV
  for e in f16drop2 f16drop4 q8_0 q8_7bit q8_6bit_out1 q4_0 q4_out1 q3_out1; do
    run_ppl kl_vsF16_${e}_c$ctx $ctx $C $N $e f16 --kl-divergence-base "$(w $KB/f16_c$ctx.kld)" --kl-divergence
  done
  # real llama.cpp types vs F16 (emulation sanity)
  for kv in q8_0 q4_0; do
    run_ppl kl_vsF16_real_${kv}_c$ctx $ctx $C $N none $kv --kl-divergence-base "$(w $KB/f16_c$ctx.kld)" --kl-divergence
  done
  # variants vs their own emulated base
  for e in q8_7bit q8_6bit_out1; do
    run_ppl kl_vsQ8_${e}_c$ctx $ctx $C $N $e f16 --kl-divergence-base "$(w $KB/q8emu_c$ctx.kld)" --kl-divergence
  done
  for e in q4_out1 q3_out1; do
    run_ppl kl_vsQ4_${e}_c$ctx $ctx $C $N $e f16 --kl-divergence-base "$(w $KB/q4emu_c$ctx.kld)" --kl-divergence
  done
done
# tasks (39) and needle per mode (KV F16 storage + emulation); "none" = F16 baseline
for e in none f16drop2 f16drop4 q8_0 q8_7bit q8_6bit_out1 q4_0 q4_out1 q3_out1; do
  if [ "$e" = none ]; then unset CQ_KV_EMU; else export CQ_KV_EMU=$e; fi
  timeout 1800 python -B $P/scripts/cpu_task_eval.py --gpu --kv-type f16 --server "$(w $B/llama-server.exe)" \
    --model emu_$e="$(w $M)" --ctx-size 16384 --port 18971 --out-dir "$(w $A/tasks_$e)" >> $A/batch.log 2>&1
  log "tasks $e exit=$?"
  timeout 1800 python -B $P/scripts/niah_eval.py --server "$(w $B/llama-server.exe)" --model "$(w $M)" \
    --corpus "$(w $P/artifacts/corpus_v1/corpus.txt)" --lengths 4096,32768 --kv f16 --port 18972 --out "$(w $A/niah_$e)" >> $A/batch.log 2>&1
  log "niah $e exit=$?"
  unset CQ_KV_EMU
done
touch $A/STOP; wait $LOGPID
log end
