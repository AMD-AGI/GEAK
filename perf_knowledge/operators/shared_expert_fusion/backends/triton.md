---
title: shared_expert_fusion on Triton — SOTA card
kind: sota_card
operator: shared_expert_fusion
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
  - https://github.com/vllm-project/vllm/pull/17955
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# shared_expert_fusion × Triton

## TL;DR
> In Triton the shared expert is usually a **separate dense MLP** ([[dense_gemm]] + SwiGLU) added to the
> routed Triton fused-MoE output — fusion here is **overlap** (separate stream) + folding the residual add,
> not a single kernel. It's the editable/portable path and the correctness fallback when aiter's fused
> shared path is unavailable. vLLM's AITER fused-MoE can also compute the routing softmax + shared-expert
> sigmoid in one launch when the kernel supports it.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Triton routed fused-MoE + separate shared MLP | vLLM/sglang `fused_moe.py` + a dense Linear | gfx942/950; bf16, fp8 | — (overlap-dependent) | `VLLM_ROCM_USE_AITER_MOE=0` fallback; new variants |
| routing softmax + shared sigmoid single launch | vLLM AITER fused-MoE V1 (PR #17955) | gfx942/950 | — | when the topk kernel supports sigmoid fusion |

Recommend: aiter fused shared path for production; Triton when prototyping or when aiter's shared fusion is
missing.

## Config space / knobs
- Shared dense GEMM Triton knobs: `BLOCK_M/N/K`, `GROUP_SIZE_M` (XCD=8 multiple), `num_warps=4` (wave64,
  not 8), `num_stages=2`, `matrix_instr_nonkdim=16`, `kpack=2` (gfx942).
- Overlap: launch the shared GEMM on a separate stream / `GPU_MAX_HW_QUEUES=2`; add into the routed output.
- If injecting shared as a synthetic routed expert in the Triton fused-MoE, give it weight 1 and a reserved
  expert slot.

## Numerics / parity
Sequential add (shared after routed) is deterministic — Triton parity vs the unfused stack is tighter than
aiter's atomic-add path. fp8 shared is a quant gate; honor the requested scoring func. See
[numerics.md](../numerics.md).

## Integration (rebind seam)
`VLLM_ROCM_USE_AITER_MOE=0` → Triton fused-MoE + the model's shared Linear runs separately. Editable Python
— the Tier-C rewrite seam for a custom shared fusion.

## Pitfalls & anti-patterns
- `num_warps=8` from NVIDIA → spill on the shared dense GEMM; use 4.
- Without an explicit separate stream, the shared and routed GEMMs serialize — no overlap win.
- Double-counting or wrong-order residual add.

## How to verify
rocprof: shared and routed kernels overlap (separate queues); `TRITON_PRINT_AUTOTUNING=1` for the shared
GEMM; greedy parity vs the unfused stack.

## Alternatives / cross-links
[aiter.md](aiter.md) · [hip.md](hip.md) · [`languages/triton_amd/knobs.md`](../../../languages/triton_amd/knobs.md) ·
[overview.md](../overview.md).

## Sources
- Triton fused-MoE reference: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
- routing + shared-expert single launch: https://github.com/vllm-project/vllm/pull/17955
- AMD Triton knobs: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
