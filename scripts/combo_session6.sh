#!/usr/bin/env bash
# Short GPU-lock session (lock name combo6): f16e3 overflow-pool use on real text (combo_f16e3_pool.py).
set -u
source /path/to/repo/scripts/combo_common.sh
source $P/work/gpu_lock_tools.sh
A=$P/artifacts/combo/f16e3_pool
mkdir -p $A
gpu_lock_acquire combo6
cleanup() { gpu_lock_release combo6; progress "f16e3_pool: лок освобождён"; }
trap cleanup EXIT
progress "f16e3_pool: лок взят"
PF=$P/artifacts/prompts_filled
python -B $P/scripts/combo_f16e3_pool.py "$(w ${MODEL[4B]})" "$(w $PF/c32768.txt)" 36864 "$(w $A/4B_c32768)" > $A/4B_c32768.out 2>&1
python -B $P/scripts/combo_f16e3_pool.py "$(w ${MODEL[4B]})" "$(w $PF/c4096.txt)" 8192 "$(w $A/4B_c4096)" > $A/4B_c4096.out 2>&1
python -B $P/scripts/combo_f16e3_pool.py "$(w ${MODEL[BF16]})" "$(w $PF/c32768.txt)" 36864 "$(w $A/2B_c32768)" > $A/2B_c32768.out 2>&1
progress "f16e3_pool: готово"
