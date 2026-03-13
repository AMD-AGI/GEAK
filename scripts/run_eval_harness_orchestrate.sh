#!/usr/bin/env bash
# Run geak-orchestrate for each eval_harness preprocess directory.
# Intended to run inside the geak-agent-sdubagun container.
#
# Usage:
#   bash scripts/run_eval_harness_orchestrate.sh          # all kernels sequentially
#   bash scripts/run_eval_harness_orchestrate.sh topk     # single kernel

set -euo pipefail

BASE_DIR="/workspace/outputs/eval_harness"
GPU_IDS="0,1,2,3,4,5,6,7"
MAX_ROUNDS="${GEAK_MAX_ROUNDS:-2}"
ALLOWED_AGENTS="${GEAK_ALLOWED_AGENTS:-swe_agent}"

KERNELS=(
  topk
  fused_qkv_rope
  fused_rms_fp8
  fast_rms_layernorm
  lean_atten_paged
  llama_ff_triton
  moe_routing_sigmoid_top1
)

if [[ $# -gt 0 ]]; then
  KERNELS=("$@")
fi

for kernel in "${KERNELS[@]}"; do
  pp_dir="$BASE_DIR/$kernel/preprocess"
  if [[ ! -f "$pp_dir/resolved.json" ]]; then
    echo "SKIP $kernel — no preprocess dir at $pp_dir"
    continue
  fi

  echo "============================================================"
  echo "  ORCHESTRATE: $kernel"
  echo "  preprocess-dir: $pp_dir"
  echo "  gpu-ids: $GPU_IDS  max-rounds: $MAX_ROUNDS  allowed-agents: $ALLOWED_AGENTS"
  echo "============================================================"

  geak-orchestrate \
    --preprocess-dir "$pp_dir" \
    --gpu-ids "$GPU_IDS" \
    --max-rounds "$MAX_ROUNDS" \
    --allowed-agents "$ALLOWED_AGENTS" \
    2>&1 | tee "$BASE_DIR/$kernel/orchestrate.log"

  echo ""
  echo "Done: $kernel (log: $BASE_DIR/$kernel/orchestrate.log)"
  echo ""
done

echo "All kernels complete."
