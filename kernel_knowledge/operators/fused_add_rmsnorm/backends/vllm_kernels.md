---
title: fused_add_rmsnorm on vllm_kernels — SOTA card
kind: sota_card
operator: fused_add_rmsnorm
backend: vllm_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu
  - https://github.com/vllm-project/vllm/pull/14959
  - https://github.com/vllm-project/vllm/pull/29181
---

# fused_add_rmsnorm × vllm_kernels

## TL;DR
Two worlds: native HIP `fused_add_rms_norm_kernel` (`csrc/layernorm_kernels.cu`) and AITER
`rmsnorm2d_fwd_with_add` (PR #14959), gated by `VLLM_ROCM_USE_AITER_RMSNORM`. On MI300X with AITER on,
AITER wins; native HIP is the default for `USE_AITER=0`. Note: the all-reduce+rmsnorm **fusion pass** can
emit a separate Triton norm op (#29181) — confirm which kernel actually runs.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| AITER `rmsnorm2d_fwd_with_add` (via layer) | `model_executor/layers/layernorm.py`, PR #14959 | gfx942/950 | live when `_RMSNORM=1` | MI300X serving default — [aiter.md](aiter.md) |
| native HIP `fused_add_rms_norm_kernel` | `csrc/layernorm_kernels.cu` (PR #22602) | gfx942/950 | vectorized, bandwidth-bound | `USE_AITER=0` / Tier-C edit |

## Config space / knobs
- `VLLM_ROCM_USE_AITER=1` + `VLLM_ROCM_USE_AITER_RMSNORM=1` (default on) → AITER; `=0` → native HIP.
- Native: `width` template, shared-mem row cache, int64 strides.
- Compilation: the AR+rmsnorm fusion pass (`rocm_aiter_fusion.py`) can stitch or, conversely, emit an
  extra Triton norm (#29181) — inspect the compiled graph.

## Numerics / parity
add in IO dtype, store residual_out; fp32 Σ; γ fp32-promote (#42325). AITER vs native order → greedy
re-gate (residual compounds). See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- Python: `model_executor/layers/layernorm.py` `RMSNorm.forward_hip`.
- Native HIP: `csrc/layernorm_kernels.cu` + `torch_bindings.cpp`; rebuild.
- torch.compile: AITER custom-op preserved; ROCm fusion pass for AR+rms / rms+quant.

## Pitfalls & anti-patterns
- ⚠ Extra/duplicate Triton norm kernel from the AR+rmsnorm fusion (#29181) — verify in the graph/trace.
- ⚠ Image mismatch (`USE_AITER=1`, no aiter).
- ⚠ γ-dtype regression #42325.

## How to verify
rocprofv3 norm kernel (AITER `*ck_*` vs native vs Triton); confirm no duplicate norm; `residual_out`
parity; greedy e2e.

## Alternatives / cross-links
[aiter.md](aiter.md) · [hip.md](hip.md) · [triton.md](triton.md) · [[fused_allreduce_rmsnorm]] ·
[[backends/vllm_kernels/aiter_integration]].

## Sources
- vLLM HIP fused_add_rms_norm: https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu.
- AITER with_add integration: https://github.com/vllm-project/vllm/pull/14959.
- AR+rmsnorm fusion extra-kernel issue: https://github.com/vllm-project/vllm/issues/29181.
