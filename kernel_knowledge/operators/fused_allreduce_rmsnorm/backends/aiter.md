---
title: fused_allreduce_rmsnorm on aiter — SOTA card
kind: sota_card
operator: fused_allreduce_rmsnorm
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py
  - https://github.com/ROCm/aiter/releases
  - https://docs.vllm.ai/en/latest/design/fusions/
---

# fused_allreduce_rmsnorm × aiter

## TL;DR
> aiter ships the fused comm+norm kernels on AMD: the **SP collective in one kernel**
> (`reduce_scatter_rmsnorm_quant_all_gather`) and the **qknorm+AR** fusion. It also supplies the fused
> add+rmsnorm(+quant) ops that vLLM's AR-epilogue fusion pass emits. Use it for TP layers, especially
> sub-island TP and fp8 models.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| fused RS+RMSNorm+quant+AG (SP) | `aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py` | gfx942/950; bf16/fp16/fp8 | — (one-kernel SP collective) | SP rewrite of AR+norm |
| qknorm + all-reduce | `qknorm_allreduce_fusion_kernel_2stage` (aiter releases) | gfx942/950 | **~10–15% prefill TPS TP=2/4** (AMD-reported) | MiniMax-M2.x; AR+norm fusion template |
| fused add+rmsnorm(+fp8 quant) ops | aiter RMSNorm kernels (`csrc/kernels/rmsnorm*.cu`) | gfx942/950 | — | the ops vLLM's AR-epilogue pass fuses |

## Config space / knobs
- `enable_fi_allreduce_fusion` (vLLM) to engage the AR-epilogue fusion that calls aiter's fused norm/quant.
- SP kernel: norm ε/axis, optional `_quantize_fp8_stage`, grid = CU count (Iris).
- `VLLM_ROCM_USE_AITER=1` + `_RMSNORM=1` for the aiter RMSNorm path.

## Numerics / parity
fp32 norm reduction + fp32 AR accumulate → parity-safe (bf16); fp8 static-quant epilogue is a gate (fnuz on
gfx942); residual add before norm. See [numerics.md](../numerics.md).

## Integration (rebind seam)
aiter fused ops registered as custom ops (survive torch.compile); vLLM's `rocm_aiter_fusion.py` fuses the
rms+quant chains. The SP kernel is a drop-in for AR+norm in an SP layout.

## Pitfalls & anti-patterns
- Pattern mismatch → the vLLM pass skips the fusion silently; confirm in the compiled graph.
- fp8 epilogue without an eval = accuracy regression.
- SP kernel uses CUs (norm+quant) — won't SDMA-offload the copy.
- `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` is a different (MoE) flag — don't confuse with AR fusion.

## How to verify
Inspect compiled graph for the fused op; rocprof for the fused kernel + overlap; isolated fused-vs-separate;
e2e tok/s + parity/eval.

## Alternatives / cross-links
[hip.md](hip.md) · [mori_rccl.md](rccl.md) ·
[`backends/aiter/integration.md`](../../../backends/aiter/integration.md) · [overview.md](../overview.md).

## Sources
- on-box fused SP kernel: `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py`, `csrc/kernels/rmsnorm*.cu`.
- qknorm+AR ~10–15% TPS: https://github.com/ROCm/aiter/releases
- vLLM AR fusion passes: https://docs.vllm.ai/en/latest/design/fusions/
