#!/usr/bin/env bash
set -uo pipefail

# Run all non-HIP-C++ kernels from kernel_manifest.json through the
# GEAK benchmark pipeline (preprocess + homo + hetero).
# Each kernel gets its own subdirectory under OUTPUT_BASE.

GPU_IDS="${1:-0,1,2,3,4,5,6,7}"
MAX_ROUNDS="${2:-2}"
OUTPUT_BASE="${3:-outputs/eval_benchmark}"

declare -A KERNELS=(
  [fused_rms_fp8]="https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/_triton_kernels/quant/fused_fp8_quant.py#L43"
  [rope]="https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/_triton_kernels/rope/rope.py#L91"
  [gemm]="https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py#L54"
  [topk]="https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/_triton_kernels/topk.py#L45"
  [ff_backward]="https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py#L34"
  [nsa_forward]="https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L472"
  [nsa_backward]="https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py#L617"
  [mla_decode]="https://github.com/ROCm/aiter/blob/main/aiter/mla.py#L19"
  [mla_prefill_reduce]="https://github.com/ROCm/aiter/blob/main/aiter/mla.py#L490"
  [moe_stage1]="https://github.com/ROCm/aiter/blob/main/aiter/fused_moe.py#L1409"
  [moe_stage2]="https://github.com/ROCm/aiter/blob/main/aiter/fused_moe.py#L1514"
  [rmsnorm_2d_fwd]="https://github.com/ROCm/aiter/blob/main/aiter/ops/rmsnorm.py#L62"
  [batched_gemm_a16wfp4]="https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_a16wfp4.py#L51"
  [fused_qk_rope_cache_mla]="https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/_triton_kernels/fusions/fused_kv_cache.py#L99"
  [gemm_a16w16_atomic]="https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16_atomic.py#L37"
  [fused_append_shared_experts]="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py#L1081"
  [moe_sorting]="https://github.com/ROCm/aiter/blob/main/aiter/ops/moe_sorting.py#L12"
  [fused_mxfp4_quant_moe_sort]="https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/_triton_kernels/quant/fused_mxfp4_quant.py#L820"
  [refk_identity]="https://github.com/gpu-mode/reference-kernels/blob/main/problems/amd/identity/submission.py"
  [refk_fp8_blockwise_mm]="https://github.com/gpu-mode/reference-kernels/blob/main/problems/amd/fp8-mm/submission.py"
  [refk_mla_decode]="https://github.com/gpu-mode/reference-kernels/blob/main/problems/amd/mla-decode/submission.py"
  [refk_moe]="https://github.com/gpu-mode/reference-kernels/blob/main/problems/amd/moe/submission.py"
)

# Deterministic ordering
KERNEL_ORDER=(
  fused_rms_fp8 rope gemm topk
  ff_backward nsa_forward nsa_backward
  mla_decode mla_prefill_reduce
  moe_stage1 moe_stage2 rmsnorm_2d_fwd
  batched_gemm_a16wfp4 fused_qk_rope_cache_mla gemm_a16w16_atomic
  fused_append_shared_experts moe_sorting fused_mxfp4_quant_moe_sort
  refk_identity refk_fp8_blockwise_mm refk_mla_decode refk_moe
)

mkdir -p "$OUTPUT_BASE"
OUTPUT_BASE="$(cd "$OUTPUT_BASE" && pwd)"
MASTER_LOG="${OUTPUT_BASE}/master.log"

total=${#KERNEL_ORDER[@]}
passed=0
failed=0
skipped=0

echo "============================================================" | tee "$MASTER_LOG"
echo " GEAK Eval Benchmark — ${total} kernels"                      | tee -a "$MASTER_LOG"
echo " GPU IDs:    ${GPU_IDS}"                                       | tee -a "$MASTER_LOG"
echo " Max rounds: ${MAX_ROUNDS}"                                    | tee -a "$MASTER_LOG"
echo " Output:     ${OUTPUT_BASE}"                                   | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo ""                                                              | tee -a "$MASTER_LOG"

for i in "${!KERNEL_ORDER[@]}"; do
  name="${KERNEL_ORDER[$i]}"
  url="${KERNELS[$name]}"
  idx=$((i + 1))
  out_dir="${OUTPUT_BASE}/${name}"

  echo "[${idx}/${total}] ${name}" | tee -a "$MASTER_LOG"
  echo "  URL: ${url}" | tee -a "$MASTER_LOG"

  if [[ -f "${out_dir}/summary.log" ]]; then
    echo "  SKIP — already completed (${out_dir}/summary.log exists)" | tee -a "$MASTER_LOG"
    skipped=$((skipped + 1))
    echo "" | tee -a "$MASTER_LOG"
    continue
  fi

  start_ts=$(date +%s)
  scripts/run_benchmark.sh "$url" \
    --gpu-ids "$GPU_IDS" \
    --output-dir "$out_dir" \
    --max-rounds "$MAX_ROUNDS" \
    2>&1 | tee "${out_dir}.log" || true
  exit_code=${PIPESTATUS[0]}
  elapsed=$(( $(date +%s) - start_ts ))

  if [[ $exit_code -eq 0 ]]; then
    echo "  DONE  (${elapsed}s)" | tee -a "$MASTER_LOG"
    passed=$((passed + 1))
  else
    echo "  FAIL  exit=${exit_code} (${elapsed}s)" | tee -a "$MASTER_LOG"
    failed=$((failed + 1))
  fi
  echo "" | tee -a "$MASTER_LOG"
done

echo "============================================================" | tee -a "$MASTER_LOG"
echo " Final: ${passed} passed, ${failed} failed, ${skipped} skipped out of ${total}" | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
