#!/usr/bin/env bash
# Second GPU-lock session of the combined-fork matrix: VRAM with the fork's own DXGI report for every Qwen3.5-2B config
# (4K, 32K, 120K) and the Qwen3-4B configs (4K, 32K, 120K). Uses the current work/gpu_lock_tools.sh (closes
# leftover windows on release).
set -u
source /path/to/repo/scripts/combo_common.sh
source /path/to/repo/scripts/combo_vram.sh
source $P/work/gpu_lock_tools.sh
A=$P/artifacts/combo
progress "сессия 2 (VRAM по CQ_VRAM_REPORT): жду GPU-лок"
gpu_lock_acquire combo2
rm -f $P/work/gpu_lock/waiters/combo_next
STAMP=$(date +%Y%m%d_%H%M)
cleanup() { touch $A/STOP_$STAMP; sleep 2; gpu_lock_release combo2; progress "сессия 2 $STAMP: лок освобождён"; }
trap cleanup EXIT
progress "сессия 2 $STAMP: лок взят"
python -B $P/scripts/gpu_clean_window.py preflight --samples 10 --output $A/preflight_$STAMP.json > $A/preflight_$STAMP.out 2>&1
python -B $P/scripts/gpu_clean_window.py log --seconds 28800 --stop-file $A/STOP_$STAMP --output $A/telemetry_$STAMP.csv &
progress "preflight 2: $(grep -o '"verdict": "[A-Z]*"' $A/preflight_$STAMP.json)"
PH=${1:-"v_bf16 v_q8 v_q4 v_4b"}
for ph in $PH; do
  case $ph in
    v_bf16) vram_group bf16_f16 "4096 32768 122880" BF16:f16 W4:f16 H:f16 W3:f16 BF16ffnQ8:f16 BF16:f16e3 BF16:f16d4 W4:f16e3 H:f16e3 W3:f16d4 BF16ffnQ8:f16d4 ;;
    v_q8) vram_group q8_q8 "4096 32768 122880" Q8_0:q8_0 Q8ffnQ6K:q8_0 Q8_0:q8_0:gqa Q8ffnQ6K:q8_0:gqa ;;
    v_q4) vram_group q4km_q4 "4096 32768 122880" Q4_K_M:q4_0 Q4embQ4K:q4_0 Q4imat:q4_0 Q4_K_M:q4o Q4_K_M:q4_0:gqa Q4embQ4K:q4o Q4embQ4K:q4_0:gqa ;;
    v_4b) VRAM_R=1 vram_group q4b_kv "4096 32768 122880" 4B:f16 4B:f16e3 4B:f16d4 4B:q8_0 4B:q8_0:gqa 4B:q4_0 4B:q4_0:gqa 4B:q4o ;;
  esac
  progress "сессия 2: фаза $ph готова"
done
