#!/usr/bin/env bash
# Variant B speed/VRAM/load series in the weights fork (one group = one weights base + its KV base).
# Usage: cqw_gpu_speed.sh GROUP_TAG KV "DEPTHS" LABEL=PATH [LABEL=PATH ...]
#   first model = base (variant A), the rest = ours (variant B). Same binary for all.
# Speed: llama-bench -p 512 -n 64 -d DEPTH -r 3, order forward then backward, 2 blocks -> 12 samples/cell.
# VRAM: one llama-bench run per (model, depth) under cqw_vram_probe.py (per-process dedicated memory peak).
# Load: 3 x llama-cq-logits --load-only per model (file already in the OS cache after the first run).
# Runs under the GPU lock (cqw_gpu_lock.sh).
set -u
TAG=$1; KV=$2; DEPTHS=$3; shift 3
P=/path/to/repo
B=/d/Git/llama.cpp-w/build-w/bin
A=$P/artifacts/cqw_speed/$TAG
mkdir -p $A/bench $A/vram $A/load
w() { cygpath -w "$1"; }
declare -a LABELS PATHS
for spec in "$@"; do LABELS+=("${spec%%=*}"); PATHS+=("${spec#*=}"); done
N=${#LABELS[@]}
source $P/scripts/cqw_gpu_lock.sh
cq_lock_acquire
log() { echo "$(date -Iseconds) $*" >> $A/run.log; }
python -B $P/scripts/gpu_clean_window.py preflight --samples 10 --output $A/preflight.json > $A/preflight.out 2>&1
python -B $P/scripts/gpu_clean_window.py log --seconds 14400 --stop-file $A/STOP --output $A/telemetry_1s.csv &
LOGPID=$!
log "start kv=$KV depths=$DEPTHS models=${LABELS[*]}"
# load time
[ -z "${VRAM_ONLY:-}" ] && for i in $(seq 0 $((N-1))); do
  for r in 1 2 3 4; do
    $B/llama-cq-logits.exe -m "$(w ${PATHS[$i]})" --load-only > $A/load/${LABELS[$i]}_r$r.txt 2> $A/load/${LABELS[$i]}_r$r.err
  done
done
log "load done"
for d in $DEPTHS; do
  # VRAM peak per model at this depth
  for i in $(seq 0 $((N-1))); do
    timeout 3600 python -B $P/scripts/cqw_vram_probe.py --out $A/vram/${LABELS[$i]}_d$d.json -- $B/llama-bench.exe -m "$(w ${PATHS[$i]})" \
      -ngl 99 -fa 1 -ctk $KV -ctv $KV -p 512 -n 64 -d $d -r ${VRAM_REPS:-6} -o jsonl > /dev/null 2>&1
    log "vram ${LABELS[$i]} d=$d exit=$?"
  done
  [ -n "${VRAM_ONLY:-}" ] && continue
  j=${J0:-0}
  for blk in 1 2; do
    for dir in fwd bwd; do
      if [ $dir = fwd ]; then idx=$(seq 0 $((N-1))); else idx=$(seq $((N-1)) -1 0); fi
      for i in $idx; do
        f=$(printf "%03d" $j)-d$d-b$blk-$dir-${LABELS[$i]}
        timeout 7200 $B/llama-bench.exe -m "$(w ${PATHS[$i]})" -ngl 99 -fa 1 -ctk $KV -ctv $KV -p 512 -n 64 -d $d -r 3 -o jsonl \
          > $A/bench/$f.jsonl 2> $A/bench/$f.err
        log "bench $f exit=$?"
        j=$((j+1))
      done
    done
  done
done
touch $A/STOP; wait $LOGPID
log end
cq_lock_release
