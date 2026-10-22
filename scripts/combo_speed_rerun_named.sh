#!/usr/bin/env bash
# Copy of combo_speed_rerun.sh with the lock name from env LOCK_NAME (default combo4), for a second queued rerun.
# Speed-only rerun of core A-D configs in the combined fork with more ABBA blocks and llama-bench --prio 2 (high
# process priority of the benchmark itself) to reduce the run-to-run spread caused by other desktop GPU/CPU users.
# Own GPU-lock session (work/gpu_lock_tools.sh). Usage: combo_speed_rerun.sh TAG "DEPTHS" REPS BLOCKS CFG [CFG ...]
set -u
TAG=$1; DEPTHS=$2; REPS=$3; BLOCKS=$4; shift 4
source /path/to/repo/scripts/combo_common.sh
source $P/work/gpu_lock_tools.sh
A=$P/artifacts/combo/$TAG
mkdir -p $A/bench
CFGS=("$@"); N=${#CFGS[@]}
log() { echo "$(date -Iseconds) $*" >> $A/run.log; }
progress "rerun $TAG: жду GPU-лок"
gpu_lock_acquire ${LOCK_NAME:-combo4}
STAMP=$(date +%Y%m%d_%H%M)
cleanup() { touch $A/STOP; sleep 2; gpu_lock_release ${LOCK_NAME:-combo4}; progress "rerun $TAG: лок освобождён"; }
trap cleanup EXIT
python -B $P/scripts/gpu_clean_window.py preflight --samples 10 --output $A/preflight.json > $A/preflight.out 2>&1
python -B $P/scripts/gpu_clean_window.py log --seconds 28800 --stop-file $A/STOP --output $A/telemetry_1s.csv &
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > $A/compute_apps.txt 2>&1
progress "rerun $TAG: лок взят, preflight $(grep -o '"verdict": "[A-Z]*"' $A/preflight.json)"
log "start depths=$DEPTHS reps=$REPS blocks=$BLOCKS prio=2 cfgs=${CFGS[*]} commit=$(git -C /d/Git/llama.cpp-combo rev-parse --short HEAD)"
for d in $DEPTHS; do
  j=0
  for blk in $(seq 1 $BLOCKS); do
    for dir in fwd bwd; do
      if [ $dir = fwd ]; then idx=$(seq 0 $((N-1))); else idx=$(seq $((N-1)) -1 0); fi
      for i in $idx; do
        c=${CFGS[$i]}; L=$(cfg_label $c); kv=$(cfg_kv $c); cfg_env $c
        f=$(printf "%03d" $j)-d$d-b$blk-$dir-$L
        timeout 7200 $CB/llama-bench.exe -m "$(w $(cfg_model $c))" -ngl 99 -fa 1 -ctk $kv -ctv $kv -p 512 -n 64 -d $d -r $REPS --prio 2 -o jsonl \
          > $A/bench/$f.jsonl 2> $A/bench/$f.err
        log "bench $f exit=$?"
        unset CQ_FA_VEC_GQA
        j=$((j+1))
      done
    done
  done
  progress "rerun $TAG: глубина $d готова"
done
log end
