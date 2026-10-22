#!/usr/bin/env bash
# Lossy "ours" recipes for the Q8_0 and Q4_K_M bases, built with the fork's llama-quantize from the BF16 GGUF.
# imatrix on GPU from a calibration text that is disjoint from the evaluation corpus (code + English, 300 KB,
# 0 of 2000 random 50-char corpus snippets found in it). Runs under the GPU lock (imatrix uses the GPU, and the
# CPU-heavy quantization should not disturb other measurements).
set -u
P=/path/to/repo
B=/d/Git/llama.cpp-w/build-w/bin
W=$P/work/qwen35_2b
C=$P/work/cqw
A=$P/artifacts/cqw_recipes_20261022
mkdir -p $A
w() { cygpath -w "$1"; }
source $P/scripts/cqw_gpu_lock.sh
cq_lock_acquire
log() { echo "$(date -Iseconds) $*" >> $A/run.log; }
SRC="$(w $W/qwen35-2b-BF16.gguf)"
if [ ! -f $C/imatrix_code_en.gguf ]; then
  $B/llama-imatrix.exe -m "$SRC" -f "$(w $C/calib_code_en.txt)" -o "$(w $C/imatrix_code_en.gguf)" -ngl 99 -c 512 -b 512 > $A/imatrix.log 2>&1
  log "imatrix exit=$?"
fi
IM="$(w $C/imatrix_code_en.gguf)"
# Q8_0 base: token embedding (= tied output head) in Q6_K
$B/llama-quantize.exe --token-embedding-type q6_k "$SRC" "$(w $C/qwen35-2b-Q8_0+embQ6K.gguf)" Q8_0 > $A/quantize_Q8_0+embQ6K.log 2>&1
log "Q8_0+embQ6K exit=$?"
# Q4_K_M base: reference with imatrix, and FFN up/gate as IQ3_S codebook quantization (3.44 bpw) with imatrix
$B/llama-quantize.exe --imatrix "$IM" "$SRC" "$(w $C/qwen35-2b-Q4_K_M+imat.gguf)" Q4_K_M > $A/quantize_Q4_K_M+imat.log 2>&1
log "Q4_K_M+imat exit=$?"
$B/llama-quantize.exe --imatrix "$IM" --tensor-type ffn_up=iq3_s --tensor-type ffn_gate=iq3_s "$SRC" \
  "$(w $C/qwen35-2b-Q4_K_M+imat+ffnIQ3S.gguf)" Q4_K_M > $A/quantize_Q4_K_M+imat+ffnIQ3S.log 2>&1
log "Q4_K_M+imat+ffnIQ3S exit=$?"
cq_lock_release
log end
