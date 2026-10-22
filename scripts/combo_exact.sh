#!/usr/bin/env bash
# Bit-exactness of the lossless rows in the combined fork (call from combo_session.sh under the GPU lock).
# llama-cq-logits writes raw float32 logits for every prompt position + 64 greedy tokens; files are compared
# with cmp against the base row of the same model (BF16_f16 for Qwen3.5-2B, 4B_f16 for Qwen3-4B).
# Two prompt modes: one prefill batch (GEMM/MMA paths) and token by token (GEMV + single-token FA paths).
# Two environments:
#   exact   : GGML_CUDA_DISABLE_FUSION=1 CQ_FA_VEC_GQA=1 (same FA kernel and no rope+set_rows fusion for every KV type)
#   default : no env switches (stock kernels for f16; the packed KV types always use the GQA vector kernel)
# Logit files (~1 GB each) live in a temp dir on C: and are deleted after the comparison; cmp results, logit
# statistics (combo_logit_diff.py), logs and generated token ids stay in artifacts/combo/exact.
set -u
source /path/to/repo/scripts/combo_common.sh
A=$P/artifacts/combo/exact
T=/path/to/home/AppData/Local/Temp/vramc_kl/combo_exact
mkdir -p $A $T
PR=$P/artifacts/prompts_filled/c1024.txt
log() { echo "$(date -Iseconds) $*" >> $A/run.log; }
run() { # envname cfg mode
  local envn=$1 c=$2 mode=$3 L extra=""
  L=$(cfg_label $c); [ $mode = pertoken ] && extra="--per-token"
  ( if [ $envn = exact ]; then export GGML_CUDA_DISABLE_FUSION=1 CQ_FA_VEC_GQA=1; fi
    $CB/llama-cq-logits.exe -m "$(w $(cfg_model $c))" -f "$(w $PR)" -n 64 --kv $(cfg_kv $c) --ctx 4096 $extra \
      -o "$(w $T/${envn}_${L}_$mode.bin)" > $A/${envn}_${L}_$mode.log 2>&1 )
  log "$envn $L $mode exit=$?"
  cp $T/${envn}_${L}_$mode.bin.tokens.txt $A/ 2>/dev/null
}
for mode in batch pertoken; do
  for c in BF16:f16 BF16:f16e3 W4:f16 H:f16 W4:f16e3 H:f16e3 W3:f16 BF16:f16d4 W3:f16d4 4B:f16 4B:f16e3 4B:f16d4; do run exact $c $mode; done
  for c in BF16:f16 W4:f16 H:f16 BF16:f16e3 W4:f16e3 H:f16e3 4B:f16 4B:f16e3; do run default $c $mode; done
  for envn in exact default; do
    for base in BF16_f16 4B_f16; do
      bf=$T/${envn}_${base}_$mode.bin
      [ -f $bf ] || continue
      for f in $T/${envn}_*_$mode.bin; do
        L=$(basename $f .bin); L=${L#${envn}_}; L=${L%_$mode}
        case $base in 4B_*) [[ $L == 4B_* ]] || continue;; *) [[ $L == 4B_* ]] && continue;; esac
        [ "$L" = "$base" ] && continue
        if cmp -s $bf $f; then r=IDENTICAL; else r=DIFFERENT; fi
        echo "$envn $mode $L vs $base: $r ($(stat -c %s $f) bytes)" >> $A/cmp.txt
        [ $r = DIFFERENT ] && python -B $P/scripts/combo_logit_diff.py "$(w $bf)" "$(w $f)" "$envn $mode $L vs $base" >> $A/diff.txt 2>&1
      done
    done
  done
  # base vs default-env base (does the env switch alone change logits?)
  for base in BF16_f16 4B_f16; do
    if cmp -s $T/exact_${base}_$mode.bin $T/default_${base}_$mode.bin; then r=IDENTICAL; else r=DIFFERENT; fi
    echo "env $mode exact_$base vs default_$base: $r" >> $A/cmp.txt
    [ $r = DIFFERENT ] && python -B $P/scripts/combo_logit_diff.py "$(w $T/exact_${base}_$mode.bin)" "$(w $T/default_${base}_$mode.bin)" "env $mode exact vs default $base" >> $A/diff.txt 2>&1
  done
  rm -f $T/*_$mode.bin
  progress "exact: режим $mode сравнен ($(grep -c IDENTICAL $A/cmp.txt) IDENTICAL всего)"
done
log end
