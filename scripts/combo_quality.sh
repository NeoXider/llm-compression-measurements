#!/usr/bin/env bash
# Quality of the A-D rows in the combined fork (call from combo_session.sh under the GPU lock).
# KL / same top-1 with llama-perplexity: fixed corpus (artifacts/corpus_v1), context 4096 (5 chunks, the second half
# of each chunk is scored), -b 4096 -ub 512. Every row is compared with the row A of its own pair, and every row of the
# Q8_0 and Q4_K_M pairs also with the BF16 + F16-KV row (reference = best available).
# Then the 39-case task suite (llama-server, ctx 8192) for selected rows.
set -u
source /path/to/repo/scripts/combo_common.sh
A=$P/artifacts/combo/quality
KB=/path/to/home/AppData/Local/Temp/vramc_kl/combo
mkdir -p $A $KB
CORPUS="$(w $P/artifacts/corpus_v1/corpus.txt)"
log() { echo "$(date -Iseconds) $*" >> $A/run.log; }
ppl() { # outlabel cfg extra...
  local out=$1 c=$2; shift 2
  ( cfg_env $c
    timeout 2400 $CB/llama-perplexity.exe -m "$(w $(cfg_model $c))" -f "$CORPUS" -c 4096 -b 4096 -ub 512 -ngl 99 -fa on \
      -ctk $(cfg_kv $c) -ctv $(cfg_kv $c) "$@" > $A/$out.log 2>&1 )
  log "$out exit=$?"
}
G1="BF16:f16 W4:f16 H:f16 W3:f16 BF16ffnQ8:f16 BF16:f16e3 BF16:f16d4 W4:f16e3 H:f16e3 W3:f16d4 BF16ffnQ8:f16d4"
G2="Q8_0:q8_0 Q8ffnQ6K:q8_0 Q8_0:q8_0:gqa Q8ffnQ6K:q8_0:gqa"
G3="Q4_K_M:q4_0 Q4embQ4K:q4_0 Q4imat:q4_0 Q4_K_M:q4o Q4_K_M:q4_0:gqa Q4embQ4K:q4o Q4embQ4K:q4_0:gqa"
if [ -z "${SKIP_KL:-}" ]; then
  for G in "$G1" "$G2" "$G3"; do
    set -- $G; base=$1; shift
    bl=$(cfg_label $base)
    [ -f $KB/base_$bl.kld ] || ppl base_$bl $base --kl-divergence-base "$(w $KB/base_$bl.kld)"
    for c in "$@"; do ppl kl_$(cfg_label $c)_vs_$bl $c --kl-divergence-base "$(w $KB/base_$bl.kld)" --kl-divergence; done
    if [ $bl != BF16_f16 ]; then
      for c in $base "$@"; do ppl kl_$(cfg_label $c)_vs_BF16_f16 $c --kl-divergence-base "$(w $KB/base_BF16_f16.kld)" --kl-divergence; done
    fi
    progress "quality: KL группы $bl готово"
  done
fi
if [ -z "${SKIP_TASKS:-}" ]; then
  # one llama-server run per (KV type, env); identical outputs vs the pair base are counted in the summary
  tasks() { # outdir kv gqa(0/1) models...
    local od=$1 kv=$2 g=$3; shift 3; local args=""
    for m in "$@"; do args="$args --model $m=$(w ${MODEL[$m]})"; done
    ( unset CQ_FA_VEC_GQA; [ $g = 1 ] && export CQ_FA_VEC_GQA=1
      timeout 7200 python -B $P/scripts/cpu_task_eval.py --gpu --kv-type $kv --server "$(w $CB/llama-server.exe)" $args \
        --ctx-size 8192 --port 18971 --out-dir "$(w $A/$od)" >> $A/tasks.log 2>&1 )
    log "tasks $od exit=$?"
  }
  tasks tasks_f16 f16 0 BF16 W4 W3
  tasks tasks_f16e3 f16e3 0 BF16 W4
  tasks tasks_f16d4 f16d4 0 BF16 W3
  tasks tasks_q8_0 q8_0 0 Q8_0 Q8ffnQ6K
  tasks tasks_q8_0_gqa q8_0 1 Q8_0 Q8ffnQ6K
  tasks tasks_q4_0 q4_0 0 Q4_K_M Q4embQ4K
  tasks tasks_q4o q4o 0 Q4_K_M Q4embQ4K
  progress "quality: задачи (39) готовы"
fi
log end
