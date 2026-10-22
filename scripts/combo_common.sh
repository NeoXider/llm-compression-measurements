#!/usr/bin/env bash
# Shared definitions for the combined-fork (cq/combined) measurement series (source this file).
# One binary for every row: D:\Git\llama.cpp-combo\build-combo\bin (weights types 50..52 + KV types 60..62).
P=/path/to/repo
CB=/d/Git/llama.cpp-combo/build-combo/bin
W2=$P/work/qwen35_2b
CQ=$P/work/cqw
declare -A MODEL=(
  [BF16]=$W2/qwen35-2b-BF16.gguf
  [H]=$CQ/qwen35-2b-BF16-cqH.gguf
  [W4]=$CQ/qwen35-2b-BF16-cqw4.gguf
  [W3]=$CQ/qwen35-2b-BF16-cqw3.gguf
  [BF16ffnQ8]=$W2/qwen35-2b-BF16+ffnQ8.gguf
  [Q8_0]=$W2/qwen35-2b-Q8_0.gguf
  [Q8ffnQ6K]=$W2/qwen35-2b-Q8_0+ffnQ6K.gguf
  [Q4_K_M]=$W2/qwen35-2b-Q4_K_M.gguf
  [Q4embQ4K]=$W2/qwen35-2b-Q4_K_M+embQ4K.gguf
  [Q4imat]=$CQ/qwen35-2b-Q4_K_M+imat.gguf
  [4B]=/path/to/workdir/models/Qwen3-4B-Q4_K_M.gguf
)
w() { cygpath -w "$1"; }
# config string MODEL:KV[:gqa]  ->  label MODEL_KV[_gqa]; cfg_env exports the env switches of that config
cfg_label() { echo "$1" | tr ':' '_'; }
cfg_env() {
  local g; g=$(echo "$1" | cut -d: -f3)
  unset CQ_FA_VEC_GQA
  [ "$g" = gqa ] && export CQ_FA_VEC_GQA=1
  return 0
}
cfg_model() { echo "${MODEL[$(echo "$1" | cut -d: -f1)]}"; }
cfg_kv() { echo "$1" | cut -d: -f2; }
progress() { echo "$(date '+%F %H:%M') $*" >> $P/work/COMBO_PROGRESS.md; }
