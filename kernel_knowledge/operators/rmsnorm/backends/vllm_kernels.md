---
title: rmsnorm on vllm_kernels — SOTA card
kind: sota_card
operator: rmsnorm
backend: vllm_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu
  - https://github.com/vllm-project/vllm/pull/14959
  - https://github.com/vllm-project/vllm/pull/22602
---

# rmsnorm × vllm_kernels

## TL;DR
vLLM has **two RMSNorm worlds** on ROCm: (1) its own hand-written HIP kernel in
`csrc/layernorm_kernels.cu` (`rms_norm_kernel`, `fused_add_rms_norm_kernel`), and (2) the **AITER** fused
kernels wired in via `model_executor/layers/layernorm.py` and gated by `VLLM_ROCM_USE_AITER_RMSNORM`. On
MI300X with AITER on (recommended), AITER wins; the native HIP kernel is the editable fallback and the
default when `VLLM_ROCM_USE_AITER=0`.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| AITER `rmsnorm2d_fwd_with_add` (via vLLM layer) | `model_executor/layers/layernorm.py`, PR #14959 | gfx942/950, bf16/fp16/fp8 | live path when `VLLM_ROCM_USE_AITER_RMSNORM=1` | **MI300X serving (default)** — see [aiter.md](aiter.md) |
| native HIP `rms_norm_kernel` / `fused_add_rms_norm_kernel` | `csrc/layernorm_kernels.cu` (vectorized PR #22602) | gfx942/950, bf16/fp16/fp8 | `[16384,1024]` fp16 **105.9→42.6 µs** (~2.5×, NVIDIA-measured) | `USE_AITER=0`, or editable Tier-C base |
| `LAUNCH_FUSED_ADD_RMS_NORM(width)` (width-templated) | same | gfx942/950 | width = vector lanes; aligned vector I/O | tuning the vectorization width |

## Config space / knobs
- **Master/sub gates**: `VLLM_ROCM_USE_AITER=1` (default 0) then `VLLM_ROCM_USE_AITER_RMSNORM=1` (default
  on) → AITER. Set `USE_AITER=0` to force the native HIP kernel.
- **Native kernel**: `width` template (vector lanes), shared-mem row cache (PR #22602), same launch
  config as the unfused path to preserve numerics. `int64_t` strides (overflow fix).
- **fp8 fused quant**: `rms_norm_kernel` has an FP8-quant variant (PR #40860) — note it fp32-multiplies γ
  (correct for quant; the bug was mirroring that into the plain kernel, #42325).

## Numerics / parity
fp32 accumulate; γ fp32-promote (**#42325 regression** is here — the plain kernel must not copy the
quant kernel's behavior in the input dtype). fnuz fp8 on gfx942 + `VLLM_ROCM_FP8_PADDING=1` for the fast
FP8 linear path. AITER vs native reduction order differs → greedy re-gate. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- Python layer: `vllm/model_executor/layers/layernorm.py` `RMSNorm.forward_hip` → AITER or native.
- Native HIP: `csrc/layernorm_kernels.cu` + `csrc/torch_bindings.cpp` (`_C::rms_norm`,
  `_C::fused_add_rms_norm`); rebuild vLLM to edit.
- torch.compile: AITER ops registered via `direct_register_custom_op` survive Inductor; the ROCm fusion
  pass stitches rms+quant chains ([[backends/vllm_kernels/aiter_integration]]).
- Verify: rocprofv3 → AITER `*ck_*`/asm norm vs native `rms_norm_kernel` vs a Triton name.

## Pitfalls & anti-patterns
- ⚠ Image mismatch: `USE_AITER=1` but no aiter in image → import error; use matched `vllm/vllm-openai-rocm`.
- ⚠ γ-dtype regression #42325 (v0.20.0+) — verify on your version.
- ⚠ CUDA13/CCCL3 `cub::Sum` build break (#24464) — ROCm side unaffected but pin CCCL if cross-building.
- `num_tokens==0` grid-launch crash pattern (guard).

## How to verify
`grep` the layer for the chosen path; rocprofv3 Top-N kernel name; isolated bench AITER vs native;
greedy/temp=0 parity after toggling `VLLM_ROCM_USE_AITER_RMSNORM`.

## Alternatives / cross-links
[aiter.md](aiter.md) · [hip.md](hip.md) · [triton.md](triton.md) ·
[[backends/vllm_kernels/rocm_kernels]] · [[fused_add_rmsnorm]] · [[fused_norm_quant]].

## Sources
- Native HIP RMSNorm kernels + vectorization: https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu, https://github.com/vllm-project/vllm/pull/22602.
- AITER RMSNorm integration + gate: https://github.com/vllm-project/vllm/pull/14959.
- γ-dtype regression / FP8 PR: https://github.com/vllm-project/vllm/issues/42325 (PR #40860).
