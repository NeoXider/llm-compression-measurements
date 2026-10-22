#!/usr/bin/env bash
# Fix-up for the long batch: (1) KV KL at 32K with a corpus long enough (>= 2 x ctx tokens);
# (2) truly FILLED 64K / 120K contexts using YaRN (factor 4 over 32768, the model's recommended
# long-context setting) so llama-cli does not cap the slot at n_ctx_train = 40960;
# (3) needle-in-a-haystack at ~100K with the same YaRN setting.
set -u
P=<release-root>
A=$P/artifacts/gpu_long_20261021
R=/d/Git/<project-repo>
LC=<llama.cpp-build-11552>
M=$R/models/Qwen3-4B-Q4_K_M.gguf
KB=<user-temp>
YARN="--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768"
w() { cygpath -w "$1"; }
log() { echo "$1 $(date -Is)" >> $A/batch_fix.log; }
log start
rm -f $A/STOP2
python -B $P/scripts/gpu_clean_window.py log --seconds 7200 --stop-file $A/STOP2 --output $A/telemetry_fix_1s.csv &
LOGPID=$!
C3=$(w $P/artifacts/corpus_v1/corpus_x3.txt)
timeout 1800 $LC/llama-perplexity.exe -m "$(w $M)" -f "$C3" -c 32768 -b 512 -ngl 99 -fa on -ctk f16 -ctv f16 --chunks 1 \
  --kl-divergence-base "$(w $KB/q34b_kvf16_c32768.kld)" > $A/kld_base_f16_c32768.log 2>&1
log "kl base exit=$?"
for kv in q8_0 q4_0; do
  timeout 1800 $LC/llama-perplexity.exe -m "$(w $M)" -c 32768 -b 512 -ngl 99 -fa on -ctk $kv -ctv $kv --chunks 1 \
    --kl-divergence-base "$(w $KB/q34b_kvf16_c32768.kld)" --kl-divergence > $A/kld_${kv}_c32768.log 2>&1
  log "kl $kv exit=$?"
done
timeout 5400 python -B $P/scripts/gpu_kv_memory_filled.py --cli "$(w $R/experimental/wddm-cuda-luid/cli-only/bin/llama-cli.exe)" \
  --validator "$(w $R/experimental/wddm-cuda-luid/validate_receipt.py)" --model "$(w $M)" \
  --case 65536=$(w $P/artifacts/prompts_filled/c65536.txt) --case 122880=$(w $P/artifacts/prompts_filled/c122880.txt) \
  --repeats 2 --decode 16 --extra "$YARN" --out-dir "$(w $A/memory_filled_yarn)" >> $A/batch_fix.log 2>&1
log "memory yarn exit=$?"
timeout 3600 python -B $P/scripts/niah_eval.py --server "$(w $LC/llama-server.exe)" --model "$(w $M)" \
  --corpus "$(w $P/artifacts/corpus_v1/corpus.txt)" --lengths 102400 --depths 0.1,0.5,0.9 --extra "$YARN" \
  --port 18952 --out "$(w $A/niah_100k_yarn)" >> $A/batch_fix.log 2>&1
log "niah100k exit=$?"
touch $A/STOP2; wait $LOGPID
log end
