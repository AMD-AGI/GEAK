---
title: quant_int8 on vllm_kernels — SOTA card
kind: sota_card
operator: quant_int8
backend: vllm_kernels
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [int8]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - vllm-project/vllm@HEAD:csrc/quantization/compressed_tensors/int8_quant_kernels.cu
  - https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
---

# quant_int8 × vllm_kernels

## TL;DR
vLLM's `csrc/quantization/compressed_tensors/int8_quant_kernels.cu` is the reference INT8 quant for
compressed-tensors / AWQ-int8 / SmoothQuant-int8 checkpoints, with **both symmetric and asymmetric (azp)**
paths, static and dynamic, per-token. It is gen-agnostic (INT8 has no FNUZ/OCP split) so it's the natural
INT8 path on **CDNA1/2** (no FP8 HW) as well as CDNA3/4. Use it when the model is INT8-quantized; on
CDNA3/4 FP8 usually wins on accuracy.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `static_scaled_int8_quant_kernel` | `int8_quant_kernels.cu:95` | all gens, int8 | per-token cast w/ precomputed scale | calibrated/static |
| `dynamic_scaled_int8_quant_kernel` | `:126` | all gens | per-token amax + cast | dynamic activations |
| `dynamic_scaled_int8_azp_quant_kernel` | `:158` | all gens | per-token amax+min, azp | asymmetric activations |

## Config space / knobs
- symmetric vs azp (`azp` optional tensor) — azp adds a min-reduce + zero-point.
- per-token (default) row layout; vectorized cast over `hidden_size`.
- `float_to_int8_rn` = `nearbyint` (FE_TONEAREST) — fixed rounding.

## Numerics / parity
INT32 accumulate in the GEMM; `int32_to_int8` saturates. Symmetric `s=amax/127`; asymmetric
`zp=round(-128-min/s)` with epilogue correction. Gate on task accuracy → [[numerics.md]].

## Integration (rebind seam)
`torch.ops._C.static_scaled_int8_quant` / `dynamic_scaled_int8_quant` (compressed-tensors linear method).
Selected by the model's quant config. Tier-C edit = rebuild vLLM.

## Pitfalls & anti-patterns
- azp path without the GEMM epilogue azp correction → wrong result.
- Per-tensor on outlier activations — use per-token + SmoothQuant.
- Picking INT8 over FP8 on CDNA3/4 without a reason (FP8 more accurate at same throughput).

## How to verify
rocprofv3 → confirm the int8 quant kernel ran; round-trip error + gsm8k parity.

## Alternatives / cross-links
[aiter.md](aiter.md) · [hip.md](hip.md) · [triton.md](triton.md) · [overview.md](../overview.md) ·
[[operators/scaled_quant_gemm]].

## Sources
- vLLM int8 quant (symmetric/azp, static/dynamic, nearbyint): `vllm-project/vllm@HEAD:csrc/quantization/compressed_tensors/int8_quant_kernels.cu`.
- env gates: https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
