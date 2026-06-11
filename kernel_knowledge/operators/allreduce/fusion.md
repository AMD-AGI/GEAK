---
title: allreduce — fusion
kind: technique
operator: allreduce
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://docs.vllm.ai/en/latest/design/fusions/
  - https://github.com/vllm-project/vllm/issues/26768
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/custom_all_reduce.cu
---

# allreduce — fusion

## Fusions that exist on AMD today
| fusion | what it merges | where | payoff |
|---|---|---|---|
| **all-reduce + RMSNorm** | AR followed by the next layer's RMSNorm in one pattern | vLLM `allreduce_rms_fusion.py`; aiter fused kernel | one pass over the activation → [[fused_allreduce_rmsnorm]] |
| **all-reduce + add + RMSNorm (+ fp8 quant)** | residual add + norm + static fp8 quant after AR | vLLM `AllReduceFusedAddRMSNormStaticQuantFP8Pattern` | folds 3–4 ops into the AR epilogue |
| **AR → reduce-scatter + RMSNorm + all-gather** (sequence parallelism) | replace AR with RS + local norm + AG, then AsyncTP fuses RS/AG with the surrounding GEMMs | vLLM SP + AsyncTP pass | overlaps comm with the GEMMs |
| **qknorm + all-reduce** | QK-norm fused with the AR (MiniMax-M2.x) | aiter `qknorm_allreduce_fusion_kernel_2stage` | ~10–15% TPS on prefill TP=2/4 (AMD-reported) |

## The vLLM all-reduce fusion patterns (`enable_fi_allreduce_fusion`)
The DeepSeek-V3 ROCm perf plan enumerates: `AllReduceRMSNormPattern`, `AllReduceFusedAddRMSNormPattern`,
`AllReduceFusedRMSNormStaticQuantFP8Pattern`, `AllReduceFusedAddRMSNormStaticQuantFP8Pattern`. These are
torch.compile fusion passes that match the AR→norm(→quant) chain and emit a fused kernel — see
[[fused_allreduce_rmsnorm]] for the operator detail.

## Sequence-parallelism rewrite (the other direction)
Instead of fusing *into* the AR, SP **replaces** it: `all_reduce` → `reduce_scatter` + local RMSNorm +
`all_gather`, splitting the sequence dim across TP ranks. The AsyncTP pass then fuses the RS/AG with the
adjacent GEMMs ([[reduce_scatter]], [[allgather]]) so comm overlaps compute. This wins when the norm can run
on the sharded (smaller) tensor and the RS/AG overlap the GEMMs.

## Overlap (the cheapest "fusion")
Even without a fused kernel, overlap the AR with the next GEMM: `TORCH_NCCL_HIGH_PRIORITY=1`,
`GPU_MAX_HW_QUEUES=2`, `SGLANG_ROCM_USE_MULTI_STREAM=1`, `-G 1` graph capture. Verify overlap in a trace.

## Cross-links
[[fused_allreduce_rmsnorm]] · [[reduce_scatter]] · [[allgather]] ·
[`backends/mori_rccl/rccl_tuning.md`](../../backends/mori_rccl/rccl_tuning.md) ·
[`backends/aiter/integration.md`](../../backends/aiter/integration.md).

## Sources
- vLLM fusion passes / patterns: https://docs.vllm.ai/en/latest/design/fusions/ ; https://github.com/vllm-project/vllm/issues/26768
- aiter custom AR + qknorm+AR fusion: `ROCm/aiter@a6bb49937:csrc/kernels/custom_all_reduce.cu`; https://github.com/ROCm/aiter/releases
