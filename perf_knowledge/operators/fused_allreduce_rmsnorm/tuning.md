---
title: fused_allreduce_rmsnorm — tuning
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

# fused_allreduce_rmsnorm — tuning

## What you actually tune
Whether to fuse (compile flag), which fusion pattern matches your layer (with/without residual add, with/
without fp8 quant), and whether to take the **SP rewrite** (RS+norm+AG) vs the **AR-epilogue** fusion.

## Two strategies
1. **AR-epilogue fusion** (vLLM `enable_fi_allreduce_fusion`): the torch.compile pass matches
   `AllReduceRMSNormPattern` / `AllReduceFusedAddRMSNormPattern` /
   `AllReduceFusedRMSNormStaticQuantFP8Pattern` / `AllReduceFusedAddRMSNormStaticQuantFP8Pattern` and emits
   one fused kernel. **Knob**: enable the flag; ensure your layer's exact chain (residual? fp8 quant?)
   matches a pattern, or the pass skips it.
2. **SP rewrite** (one kernel): aiter `reduce_scatter_rmsnorm_quant_all_gather` does RS → norm (on the 1/P
   shard) → optional fp8 quant → AG. **Knob**: choose SP when the norm-on-shard saving + RS/AG overlap
   beats the extra AG vs a plain AR.

## Levers
- **`enable_fi_allreduce_fusion`** (vLLM) to turn on the AR+RMSNorm fusion pass.
- **fp8 static quant epilogue**: fuse the post-norm fp8 quant when the next GEMM is fp8 ([[fused_norm_quant]]).
- **Overlap**: `GPU_MAX_HW_QUEUES=2`, `TORCH_NCCL_HIGH_PRIORITY=1`, `SGLANG_ROCM_USE_MULTI_STREAM=1`, graph
  capture — even unfused, overlap the AR with the next GEMM.
- **qknorm+AR sibling**: aiter's `qknorm_allreduce_fusion_kernel_2stage` (grid-strided, no 80-token cap) for
  MiniMax-M2.x — ~10–15% prefill TPS TP=2/4; a template for AR+norm fusion shape handling.

## RMSNorm-side knobs (memory-bound)
- One block per row, hidden in LDS; fp32 reduction; wave64 reduce.
- Vectorize the hidden-dim load (`global_load_dwordx4`); `__restrict__`.
- Fuse the residual add and the fp8 quant into the same pass (no extra read/write).

## Pitfalls in tuning
- Pattern mismatch → the fusion pass silently skips (no fusion, no error). Confirm the fused kernel in the
  compiled graph.
- SP rewrite adds an all-gather — only wins if norm-on-shard + overlap pays for it.
- The fused SP kernel uses CUs (it's norming+quanting) — it won't also SDMA-offload the copy.

## How to verify
Inspect the compiled graph (vLLM `-O`/`--compilation-config`) to confirm the fused AR+RMSNorm op replaced
the pair; rocprof for the fused kernel + overlap; isolated fused-vs-separate timing; e2e tok/s.

## Sources
- vLLM fusion passes / patterns / `enable_fi_allreduce_fusion`: https://docs.vllm.ai/en/latest/design/fusions/ ; https://github.com/vllm-project/vllm/issues/26768
- aiter fused SP kernel + qknorm+AR: `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py`; https://github.com/ROCm/aiter/releases
