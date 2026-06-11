---
title: fused_allreduce_rmsnorm — fusion
kind: technique
operator: fused_allreduce_rmsnorm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://docs.vllm.ai/en/latest/design/fusions/
  - https://github.com/vllm-project/vllm/issues/26768
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py
---

# fused_allreduce_rmsnorm — fusion

This operator **is** the fusion. The dimensions are: how much of the chain you fold, and AR-epilogue vs
SP-rewrite.

## The fusion patterns (vLLM, `enable_fi_allreduce_fusion`)
| pattern | chain folded |
|---|---|
| `AllReduceRMSNormPattern` | AR → RMSNorm |
| `AllReduceFusedAddRMSNormPattern` | AR → (+residual) → RMSNorm |
| `AllReduceFusedRMSNormStaticQuantFP8Pattern` | AR → RMSNorm → fp8 static quant |
| `AllReduceFusedAddRMSNormStaticQuantFP8Pattern` | AR → (+residual) → RMSNorm → fp8 static quant |

These are torch.compile passes (`vllm/compilation/passes/fusion/allreduce_rms_fusion.py`) that match the
chain and emit one kernel — folding 2–4 ops (collective + add + norm + quant) into the AR epilogue.

## The two realizations
1. **AR-epilogue fusion**: keep the all-reduce, fuse add/norm/quant into its epilogue (the vLLM patterns).
2. **SP rewrite, one kernel**: aiter `reduce_scatter_rmsnorm_quant_all_gather` (`fused_pipeline_kernel`)
   does RS → RMSNorm (sharded) → fp8 quant → AG in a single Iris-based Triton kernel — the fully-fused SP
   collective. Pick this when norm-on-shard + RS/AG overlap wins.

## Sibling: qknorm + all-reduce
aiter's `qknorm_allreduce_fusion_kernel_2stage` (MiniMax-M2.x) fuses QK-norm with the AR (grid-strided, no
80-token cap) — **~10–15% prefill TPS at TP=2/4** (AMD-reported). Same family: fold a norm onto the
collective.

## Fusion neighbors
- collective half: [[allreduce]] / [[reduce_scatter]] + [[allgather]].
- norm half: [[rmsnorm]] / [[fused_add_rmsnorm]].
- quant epilogue: [[fused_norm_quant]] → feeds an fp8 [[scaled_quant_gemm]] / [[fused_moe_grouped_gemm]].

## What is NOT yet fused
- A single kernel that fuses the AR+norm with the **next GEMM** (so the normed+quanted activation never hits
  global memory before the GEMM) — AsyncTP overlaps but doesn't fuse into the GEMM kernel.

## Cross-links
[[allreduce]] · [[reduce_scatter]] · [[allgather]] · [[rmsnorm]] · [[fused_norm_quant]] ·
[`backends/aiter/integration.md`](../../backends/aiter/integration.md) ·
[`backends/mori_rccl/rccl_tuning.md`](../../backends/mori_rccl/rccl_tuning.md).

## Sources
- vLLM fusion patterns / passes: https://docs.vllm.ai/en/latest/design/fusions/ ; https://github.com/vllm-project/vllm/issues/26768
- aiter fused SP kernel + qknorm+AR: `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py`; https://github.com/ROCm/aiter/releases
