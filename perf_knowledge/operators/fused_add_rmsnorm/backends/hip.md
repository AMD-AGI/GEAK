---
title: fused_add_rmsnorm on hip — SOTA card
kind: sota_card
operator: fused_add_rmsnorm
backend: hip
gens: [gfx908, gfx90a, gfx942, gfx950, gfx1151]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu
  - https://github.com/vllm-project/vllm/pull/22602
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
---

# fused_add_rmsnorm × hip

## TL;DR
vLLM's `fused_add_rms_norm_kernel` (`csrc/layernorm_kernels.cu`, width-templated via
`LAUNCH_FUSED_ADD_RMS_NORM(width)`) is the canonical HIP impl: load x+residual vectorized, add, store
`residual_out`, wave64 reduce Σ(r')² in fp32, write y. The vectorization PR #22602 (aligned vector I/O +
shared-mem row cache) is the perf reference.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| vLLM `fused_add_rms_norm_kernel` | `csrc/layernorm_kernels.cu` (PR #22602) | gfx942/950, bf16/fp16/fp8 | vectorized; `[16384,1024]` fp16 norm 105.9→42.6 µs class (traffic) | editable HIP / `USE_AITER=0` |
| aiter asm `add_rmsnorm` | [aiter.md](aiter.md) | gfx942/950 | floor | aiter path |

## Config space / knobs
- `width` template = vector lanes (`float4`/`__half2`); shared-mem cache of the bf16 row (PR #22602).
- Block ×64; one block/row (prefill) or persistent grid-stride (decode).
- Same launch config as the unfused path (PR #22602 preserves numerics this way).
- `int64_t` strides (overflow fix). `__launch_bounds__(block, 4)`.

## Numerics / parity
add in IO dtype, store residual_out; fp32 Σ(r')²; γ fp32-promote (#42325 class). FP8-quant variant
fp32-multiplies γ (correct there). See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- vLLM: `csrc/layernorm_kernels.cu` + `torch_bindings.cpp` (`_C::fused_add_rms_norm`); rebuild to edit.
- Standalone: `torch.utils.cpp_extension`.

## Pitfalls & anti-patterns
- ⚠ `int` indexing overflow at M·N>2³¹ (PR #22602 review caught it) — `int64_t`.
- ⚠ `residual_out` dtype/precision must match the unfused add (compounds).
- ⚠ CUDA13/CCCL3 `cub::Sum` build break (#24464) — pin CCCL if cross-building.

## How to verify
`-Rpass-analysis=kernel-resource-usage`; `--save-temps` grep `dwordx4`; `residual_out` parity vs `x+r`;
greedy e2e.

## Alternatives / cross-links
[aiter.md](aiter.md) · [triton.md](triton.md) · [vllm_kernels.md](vllm_kernels.md) ·
[[rmsnorm/backends/hip]] · [[languages/hip_cpp/patterns]] §1.

## Sources
- vLLM HIP fused_add_rms_norm + vectorization: https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu, https://github.com/vllm-project/vllm/pull/22602.
- wave64 reduce: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html.

## On gfx1151 (RDNA3.5, Strix Halo)
`gfx1151` appears in `gens:` because this backend's **source is portable** to RDNA — it
compiles/JITs there with no vendor asset table. It is **not** a claim that anything on this
card was measured on RDNA: every perf number, ranking and tuning recipe above is CDNA
(gfx90a/942/950) evidence. Three differences bite before any of it transfers — WMMA not MFMA,
wave32 not wave64, and 40 CU behind a 32 MB MALL on ~229 GB/s shared LPDDR5X rather than HBM —
so tile shapes, occupancy targets and the roofline ceiling all move. Any `fp8_*` entry in
`dtypes:` above is CDNA-only: gfx1151 has **no fp8 matrix instruction and no block-scaled
FP4/FP6**, so an fp8 candidate there runs EMULATED — it passes correctness and loses
performance silently. See [`../../../hardware/rdna35_gfx1151/`](../../../hardware/rdna35_gfx1151/)
and MEASURE on the box.
