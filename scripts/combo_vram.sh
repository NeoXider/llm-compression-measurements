#!/usr/bin/env bash
# Process-VRAM pass for the combined-fork matrix with the fork's own report (CQ_VRAM_REPORT=1: DXGI local-memory usage of
# the process after every llama_decode, plus f16e3 overflow-pool use). Complements the ~1 s WDDM sampling of
# combo_matrix.sh, which can miss the peak of short (4K) runs. One llama-bench run (-r 3) per (config, depth); for
# quantized KV a second run with CQ_FA_SCRATCH_POOL=1. Own GPU-lock session (work/gpu_lock_tools.sh).
# Usage: combo_vram.sh TAG "DEPTHS" CFG [CFG ...]   (several groups: call repeatedly inside one session via VRAM_GROUPS)
set -u
source /path/to/repo/scripts/combo_common.sh
vram_group() {
  local TAG=$1 DEPTHS=$2; shift 2
  local A=$P/artifacts/combo/$TAG/vram_report
  mkdir -p $A
  for d in $DEPTHS; do
    for c in "$@"; do
      local L kv; L=$(cfg_label $c); kv=$(cfg_kv $c)
      for pool in 0 1; do
        [ $pool = 1 ] && [ $kv = f16 ] && continue
        local sfx=""; [ $pool = 1 ] && sfx=_pool
        ( cfg_env $c; export CQ_VRAM_REPORT=1; [ $pool = 1 ] && export CQ_FA_SCRATCH_POOL=1
          timeout 3600 $CB/llama-bench.exe -m "$(w $(cfg_model $c))" -ngl 99 -fa 1 -ctk $kv -ctv $kv -p 512 -n 64 -d $d -r ${VRAM_R:-3} \
            -o jsonl > $A/${L}_d$d$sfx.jsonl 2> $A/${L}_d$d$sfx.err )
        echo "$(date -Iseconds) $L d=$d pool=$pool exit=$? max_local_mib=$(grep -o 'local_mib=[0-9.]*' $A/${L}_d$d$sfx.err | cut -d= -f2 | sort -g | tail -1)" >> $A/run.log
      done
    done
  done
}
