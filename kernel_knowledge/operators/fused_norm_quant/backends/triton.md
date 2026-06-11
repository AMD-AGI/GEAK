---
title: fused_norm_quant on triton — SOTA card
kind: sota_card
operator: fused_norm_quant
backend: triton
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, int8, mxfp4]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py
  - /sgl-workspace/aiter/aiter/ops/triton/normalization/norm.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# fused_norm_quant × triton

## TL;DR
Triton is the authorable SOTA, and aiter ships the Triton norm+quant kernels directly:
`_quant_rms_norm_kernel`, `_quant_fused_add_rmsnorm_kernel` (rmsnorm.py) and
`layernorm2d_fwd_with_dynamicquant` / `_with_smoothquant` / `_with_add_dynamicquant` (norm.py). The abs-max
for the scale is computed in the same sweep that produces `y`.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `_quant_rms_norm_kernel` | `aiter/ops/triton/normalization/rmsnorm.py` | gfx942/950, fp8 | persistent grid; abs-max in normalize sweep | rmsnorm + dyn fp8 |
| `_quant_fused_add_rmsnorm_kernel` | same | gfx942/950, fp8 | residual+norm+quant triple | block input + quant |
| `layernorm2d_fwd_with_dynamicquant` / `_with_smoothquant` / `_with_add_dynamicquant` | `.../norm.py` | gfx942/950, fp8/int8 | LayerNorm + quant (the live LN-quant path; C++ is commented out) | LayerNorm + quant |

## Config space / knobs
- `BLOCK_SIZE = min(65536//elt, next_pow2(N))`; group/block-aligned for group-128 / mxfp4 block-32.
- Grid `min(M, num_sms)` persistent; `num_warps=2–4`; fp32 norm + fp32 scale + RNE.
- Scale granularity (per-token / group / block-32) as a constexpr — match the consumer GEMM.

## Numerics / parity
fp32 norm/scale; fnuz fp8 gfx942; quantize the fp32 `y` (no double-rounding); task gate. See
[../numerics.md](../numerics.md).

## Integration (rebind seam)
- Direct: `from aiter.ops.triton.normalization.rmsnorm import rmsnorm2d_fwd_with_dynamicquant`.
- Notably the **LayerNorm dynamic-quant** path is Triton (the aiter C++ entrypoint is commented out) — for
  LN+dynamicquant, Triton is *the* impl, not just a fallback.
- torch.compile: Inductor can emit the norm but not the fp8 group-quant — prefer the aiter Triton kernel.

## Pitfalls & anti-patterns
- `num_warps=8` → spill.
- Quantizing a bf16-rounded `y` (double rounding) instead of the fp32 `y`.
- Scale granularity ≠ consumer GEMM.
- mxfp4 on gfx942 (no HW).

## How to verify
`TRITON_PRINT_AUTOTUNING=1`; isolated dequant vs fp64 norm; gsm8k delta; ISA `dwordx4`; fnuz on gfx942.

## Alternatives / cross-links
[aiter.md](aiter.md) · [hip.md](hip.md) · [[rmsnorm/backends/triton]] · [[layernorm/backends/triton]] ·
[[quant_dequant_fp8]] · [[languages/triton_amd/patterns]] §5.

## Sources
- aiter Triton norm+quant kernels: `/sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py`, `norm.py`.
- memory-bound knobs: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html.
