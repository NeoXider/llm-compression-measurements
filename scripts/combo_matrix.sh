#!/usr/bin/env bash
# A-D speed / VRAM / load series in the combined fork for one group of configs (call from combo_session.sh,
# which holds the GPU lock). Usage: combo_matrix.sh TAG "DEPTHS" REPS BLOCKS CFG [CFG ...]
#   CFG = MODEL:KV[:gqa]; the first CFG is the group base (row A).
# Speed: llama-bench -p 512 -n 64 -d DEPTH -r REPS, configs in forward then backward order, BLOCKS blocks
#        -> REPS*2*BLOCKS samples per cell (ABBA).
# VRAM:  one llama-bench run per (config, depth) under cqw_vram_probe.py (peak per-process WDDM dedicated usage);
#        for quantized KV a second probe with CQ_FA_SCRATCH_POOL=1 (fork option: f16 prefill scratch from the pool).
# Load:  4 x llama-cq-logits --load-only per distinct model (first = OS-cache warm-up, median of runs 2..4).
set -u
TAG=$1; DEPTHS=$2; REPS=$3; BLOCKS=$4; shift 4
source /path/to/repo/scripts/combo_common.sh
A=$P/artifacts/combo/$TAG
mkdir -p $A/bench $A/vram $A/load
CFGS=("$@"); N=${#CFGS[@]}
log() { echo "$(date -Iseconds) $*" >> $A/run.log; }
log "start depths=$DEPTHS reps=$REPS blocks=$BLOCKS cfgs=${CFGS[*]} commit=$(git -C /d/Git/llama.cpp-combo rev-parse --short HEAD)"
# load time per distinct model
if [ -z "${SKIP_LOAD:-}" ]; then
  declare -A SEEN=()
  for c in "${CFGS[@]}"; do
    m=$(echo $c | cut -d: -f1); [ -n "${SEEN[$m]:-}" ] && continue; SEEN[$m]=1
    for r in 1 2 3 4; do
      $CB/llama-cq-logits.exe -m "$(w ${MODEL[$m]})" --load-only > $A/load/${m}_r$r.txt 2> $A/load/${m}_r$r.err
    done
    log "load $m done"
  done
fi
for d in $DEPTHS; do
  for c in "${CFGS[@]}"; do
    L=$(cfg_label $c); kv=$(cfg_kv $c); cfg_env $c
    timeout 3600 python -B $P/scripts/cqw_vram_probe.py --out $A/vram/${L}_d$d.json -- $CB/llama-bench.exe -m "$(w $(cfg_model $c))" \
      -ngl 99 -fa 1 -ctk $kv -ctv $kv -p 512 -n 64 -d $d -r 1 -o jsonl > /dev/null 2>&1
    log "vram $L d=$d exit=$?"
    if [ "$kv" != f16 ] && [ -n "${SCRATCH_PROBE-1}" ]; then
      CQ_FA_SCRATCH_POOL=1 timeout 3600 python -B $P/scripts/cqw_vram_probe.py --out $A/vram/${L}_d${d}_pool.json -- $CB/llama-bench.exe \
        -m "$(w $(cfg_model $c))" -ngl 99 -fa 1 -ctk $kv -ctv $kv -p 512 -n 64 -d $d -r 1 -o jsonl > /dev/null 2>&1
      log "vram-pool $L d=$d exit=$?"
    fi
    unset CQ_FA_VEC_GQA
  done
  j=0
  for blk in $(seq 1 $BLOCKS); do
    for dir in fwd bwd; do
      if [ $dir = fwd ]; then idx=$(seq 0 $((N-1))); else idx=$(seq $((N-1)) -1 0); fi
      for i in $idx; do
        c=${CFGS[$i]}; L=$(cfg_label $c); kv=$(cfg_kv $c); cfg_env $c
        f=$(printf "%03d" $j)-d$d-b$blk-$dir-$L
        timeout 7200 $CB/llama-bench.exe -m "$(w $(cfg_model $c))" -ngl 99 -fa 1 -ctk $kv -ctv $kv -p 512 -n 64 -d $d -r $REPS -o jsonl \
          > $A/bench/$f.jsonl 2> $A/bench/$f.err
        log "bench $f exit=$?"
        unset CQ_FA_VEC_GQA
        j=$((j+1))
      done
    done
  done
  progress "matrix $TAG: глубина $d готова"
done
log end
