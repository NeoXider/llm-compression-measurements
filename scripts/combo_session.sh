#!/usr/bin/env bash
# Driver for the combined-fork A-D matrix (cq/combined): one GPU-lock session, preflight + 1 s telemetry,
# then smoke -> bit-exactness -> speed/VRAM/load matrices -> quality -> Qwen3-4B extension.
# Usage: combo_session.sh [PHASES]   (default: "smoke exact m_bf16 m_q8 m_q4 quality m_4b")
set -u
source /path/to/repo/scripts/combo_common.sh
PHASES=${1:-"smoke exact m_bf16 m_q8 m_q4 quality m_4b"}
A=$P/artifacts/combo
mkdir -p $A
source $P/work/gpu_lock_tools.sh
progress "сессия: жду GPU-лок (фазы: $PHASES)"
gpu_lock_acquire combo
STAMP=$(date +%Y%m%d_%H%M)
cleanup() { touch $A/STOP_$STAMP; sleep 2; gpu_lock_release combo; progress "сессия $STAMP: лок освобождён"; }
trap cleanup EXIT
progress "сессия $STAMP: лок взят"
python -B $P/scripts/gpu_clean_window.py preflight --samples 10 --output $A/preflight_$STAMP.json > $A/preflight_$STAMP.out 2>&1
python -B $P/scripts/gpu_clean_window.py log --seconds 28800 --stop-file $A/STOP_$STAMP --output $A/telemetry_$STAMP.csv &
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > $A/compute_apps_$STAMP.txt 2>&1
tasklist //FO CSV > $A/tasklist_$STAMP.csv 2>&1
progress "preflight: $(grep -o '"verdict": "[A-Z]*"' $A/preflight_$STAMP.json)"
for ph in $PHASES; do
  progress "фаза $ph: старт"
  case $ph in
    smoke)
      mkdir -p $A/smoke
      for c in BF16:f16 H:f16 W4:f16 W3:f16 BF16ffnQ8:f16 BF16:f16e3 BF16:f16d4 W4:f16e3 H:f16e3 W3:f16d4 BF16ffnQ8:f16d4 \
               Q8_0:q8_0 Q8ffnQ6K:q8_0 Q8_0:q8_0:gqa Q4_K_M:q4_0 Q4embQ4K:q4_0 Q4imat:q4_0 Q4_K_M:q4o Q4_K_M:q4_0:gqa Q4embQ4K:q4o \
               4B:f16 4B:f16e3 4B:f16d4 4B:q8_0 4B:q8_0:gqa 4B:q4_0 4B:q4_0:gqa 4B:q4o; do
        L=$(cfg_label $c); cfg_env $c
        timeout 600 $CB/llama-bench.exe -m "$(w $(cfg_model $c))" -ngl 99 -fa 1 -ctk $(cfg_kv $c) -ctv $(cfg_kv $c) -p 128 -n 16 -d 1024 -r 1 -o jsonl \
          > $A/smoke/$L.jsonl 2> $A/smoke/$L.err
        echo "$L exit=$? lines=$(grep -c avg_ts $A/smoke/$L.jsonl)" >> $A/smoke/smoke.txt
        unset CQ_FA_VEC_GQA
      done ;;
    exact) bash $P/scripts/combo_exact.sh ;;
    m_bf16) bash $P/scripts/combo_matrix.sh bf16_f16 "4096 32768 122880" 3 2 \
              BF16:f16 W4:f16 H:f16 W3:f16 BF16ffnQ8:f16 BF16:f16e3 BF16:f16d4 W4:f16e3 H:f16e3 W3:f16d4 BF16ffnQ8:f16d4 ;;
    m_q8) bash $P/scripts/combo_matrix.sh q8_q8 "4096 32768 122880" 3 2 \
              Q8_0:q8_0 Q8ffnQ6K:q8_0 Q8_0:q8_0:gqa Q8ffnQ6K:q8_0:gqa ;;
    m_q4) bash $P/scripts/combo_matrix.sh q4km_q4 "4096 32768 122880" 3 2 \
              Q4_K_M:q4_0 Q4embQ4K:q4_0 Q4imat:q4_0 Q4_K_M:q4o Q4_K_M:q4_0:gqa Q4embQ4K:q4o Q4embQ4K:q4_0:gqa ;;
    quality) bash $P/scripts/combo_quality.sh ;;
    m_4b) bash $P/scripts/combo_matrix.sh q4b_kv "4096 32768 122880" 3 1 \
              4B:f16 4B:f16e3 4B:f16d4 4B:q8_0 4B:q8_0:gqa 4B:q4_0 4B:q4_0:gqa 4B:q4o ;;
  esac
  progress "фаза $ph: конец"
done
python -B $P/scripts/gpu_clean_window.py preflight --samples 5 --output $A/postflight_$STAMP.json > /dev/null 2>&1
