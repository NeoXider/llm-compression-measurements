#!/usr/bin/env bash
# Direct check of row D lossless against row C lossless: W4/H weights + f16e3 KV vs BF16 weights + f16e3 KV (Qwen3.5-2B),
# raw float32 logits compared with cmp (prompt as one batch and token by token; default env and exact env).
# Own short GPU-lock session (lock name combo5).
set -u
source /path/to/repo/scripts/combo_common.sh
source $P/work/gpu_lock_tools.sh
A=$P/artifacts/combo/exact_d
T=/path/to/home/AppData/Local/Temp/vramc_kl/combo_exact_d
mkdir -p $A $T
PR=$P/artifacts/prompts_filled/c1024.txt
gpu_lock_acquire combo5
cleanup() { rm -f $T/*.bin; gpu_lock_release combo5; progress "exact_d: лок освобождён"; }
trap cleanup EXIT
progress "exact_d: лок взят"
for envn in default exact; do
  for mode in batch pertoken; do
    extra=""; [ $mode = pertoken ] && extra="--per-token"
    for c in BF16:f16e3 W4:f16e3 H:f16e3; do
      L=$(cfg_label $c)
      ( if [ $envn = exact ]; then export GGML_CUDA_DISABLE_FUSION=1 CQ_FA_VEC_GQA=1; fi
        $CB/llama-cq-logits.exe -m "$(w $(cfg_model $c))" -f "$(w $PR)" -n 64 --kv f16e3 --ctx 4096 $extra -o "$(w $T/${envn}_${L}_$mode.bin)" > $A/${envn}_${L}_$mode.log 2>&1 )
    done
    for L in W4_f16e3 H_f16e3; do
      if cmp -s $T/${envn}_BF16_f16e3_$mode.bin $T/${envn}_${L}_$mode.bin; then r=IDENTICAL; else r=DIFFERENT; fi
      echo "$envn $mode $L vs BF16_f16e3: $r ($(stat -c %s $T/${envn}_${L}_$mode.bin) bytes)" >> $A/cmp.txt
    done
    rm -f $T/*.bin
  done
done
