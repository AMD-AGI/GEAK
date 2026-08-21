#!/usr/bin/env bash
# Arm B: all three patches applied.
set -eu
D=/home/ethany/tuning_workspace/experiment_standalone/qwen3_14b_fp8_tuning/analysis/ab
cp "$D/tuned_gemm_qwen3_14b.csv" /sgl-workspace/aiter/aiter/configs/model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen3_14b.csv
cp "$D/pa_ragged.B.py" /sgl-workspace/aiter/csrc/cpp_itfs/pa/pa_ragged.py
cp "$D/fp8_utils.B.py" /sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp8_utils.py
rm -rf /tmp/aiter_configs
