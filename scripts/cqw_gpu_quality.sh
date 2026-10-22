#!/usr/bin/env bash
# Variant B quality in the weights fork: KL/top-1 vs the group base (same KV type) and vs Q8_0 (KV f16, as in R7),
# llama-perplexity on GPU, fixed corpus, ctx 512, all 44 chunks. Then the 39-case task suite (GPU llama-server).
# Usage: cqw_gpu_quality.sh TAG   (runs under the GPU lock)
set -u
TAG=$1
P=/path/to/repo
B=/d/Git/llama.cpp-w/build-w/bin
W=$P/work/qwen35_2b
C=$P/work/cqw
A=$P/artifacts/cqw_quality/$TAG
KB=/path/to/home/AppData/Local/Temp/vramc_kl/cqw
mkdir -p $A $KB
w() { cygpath -w "$1"; }
CORPUS="$(w $P/artifacts/corpus_v1/corpus.txt)"
source $P/scripts/cqw_gpu_lock.sh
cq_lock_acquire
log() { echo "$(date -Iseconds) $*" >> $A/run.log; }
ppl() { # label model kv extra...
  local label=$1 model=$2 kv=$3; shift 3
  timeout 1800 $B/llama-perplexity.exe -m "$(w $model)" -f "$CORPUS" -c 512 -b 512 -ngl 99 -fa on -ctk $kv -ctv $kv "$@" > $A/$label.log 2>&1
  log "$label exit=$?"
}
# group definitions: base model, KV type, other models
declare -A M=(
  [BF16]=$W/qwen35-2b-BF16.gguf [H]=$C/qwen35-2b-BF16-cqH.gguf [W4]=$C/qwen35-2b-BF16-cqw4.gguf [W3]=$C/qwen35-2b-BF16-cqw3.gguf
  [BF16ffnQ8]=$W/qwen35-2b-BF16+ffnQ8.gguf
  [Q8_0]=$W/qwen35-2b-Q8_0.gguf [Q8ffnQ6K]=$W/qwen35-2b-Q8_0+ffnQ6K.gguf [Q8embQ6K]=$C/qwen35-2b-Q8_0+embQ6K.gguf
  [Q4_K_M]=$W/qwen35-2b-Q4_K_M.gguf [Q4embQ4K]=$W/qwen35-2b-Q4_K_M+embQ4K.gguf
  [Q4imat]=$C/qwen35-2b-Q4_K_M+imat.gguf [Q4imatFfnIQ3S]=$C/qwen35-2b-Q4_K_M+imat+ffnIQ3S.gguf
)
GROUPS_DEF=${GROUPS_DEF:-"BF16:f16:H,W4,W3,BF16ffnQ8 Q8_0:q8_0:Q8ffnQ6K,Q8embQ6K Q4_K_M:q4_0:Q4embQ4K,Q4imat,Q4imatFfnIQ3S"}
# reference logits: Q8_0 weights, KV f16 (R7 reference, here on GPU)
[ -f $KB/ref_q8_f16.kld ] || ppl base_ref_Q8_0_f16 ${M[Q8_0]} f16 --kl-divergence-base "$(w $KB/ref_q8_f16.kld)"
for gdef in $GROUPS_DEF; do
  base=${gdef%%:*}; rest=${gdef#*:}; kv=${rest%%:*}; others=${rest#*:}
  [ -f $KB/base_${base}_$kv.kld ] || ppl base_${base}_$kv ${M[$base]} $kv --kl-divergence-base "$(w $KB/base_${base}_$kv.kld)"
  for m in $base $(echo $others | tr ',' ' '); do
    [ -f "${M[$m]}" ] || { log "missing $m"; continue; }
    [ $m != $base ] && ppl kl_${m}_vs_${base}_$kv ${M[$m]} $kv --kl-divergence-base "$(w $KB/base_${base}_$kv.kld)" --kl-divergence
    ppl kl_${m}_vs_Q8ref_f16 ${M[$m]} f16 --kl-divergence-base "$(w $KB/ref_q8_f16.kld)" --kl-divergence
  done
done
log "kl done"
if [ -z "${NO_TASKS:-}" ]; then
  for gdef in $GROUPS_DEF; do
    base=${gdef%%:*}; rest=${gdef#*:}; kv=${rest%%:*}; others=${rest#*:}
    ARGS=""
    for m in $base $(echo $others | tr ',' ' '); do
      [ -f "${M[$m]}" ] && ARGS="$ARGS --model $m=$(w ${M[$m]})"
    done
    timeout 7200 python -B $P/scripts/cpu_task_eval.py --gpu --kv-type $kv --server "$(w $B/llama-server.exe)" $ARGS --ctx-size 8192 \
      --port 18961 --out-dir "$(w $A/tasks_$base)" >> $A/tasks.log 2>&1
    log "tasks $base exit=$?"
  done
fi
cq_lock_release
log end
