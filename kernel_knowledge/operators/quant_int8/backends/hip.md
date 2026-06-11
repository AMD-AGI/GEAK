---
title: quant_int8 on hip — SOTA card
kind: sota_card
operator: quant_int8
backend: hip
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [int8]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - vllm-project/vllm@HEAD:csrc/quantization/compressed_tensors/int8_quant_kernels.cu
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/quant_kernels.cu
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
---

# quant_int8 × hip

## TL;DR
HIP/C++ is the editable source under both INT8 backends — vLLM's `int8_quant_kernels.cu` and aiter's
`quant_kernels.cu` (`smooth_per_token_scaled_quant_kernel`). Reach for it for a custom INT8 fusion or to
control the exact rounding/saturation. Bandwidth-bound; the craft is vectorized loads/stores, a clean
per-token amax (and min, for azp), and `nearbyint` rounding. Gen-agnostic — the right INT8 layer on
CDNA1/2 where FP8 MFMA doesn't exist.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| vLLM `dynamic_scaled_int8[_azp]_quant_kernel` | `int8_quant_kernels.cu:126,158` | all gens | per-token amax(+min), `nearbyint` | symmetric / asymmetric |
| aiter `smooth_per_token_scaled_quant_kernel` | `csrc/kernels/quant_kernels.cu:451` | gfx942/950 | smooth + per-token, transpose opt | SmoothQuant W8A8 |
| `float_to_int8_rn` / `int32_to_int8` helpers | `int8_quant_kernels.cu:15,74` | all | RNE + saturate | the cast primitives |

## Config space / knobs
- `thread_data_size` / vector width; block size mult of 64; grid ≥ 1024.
- symmetric vs azp (extra min-reduce); `transpose_out_dim01` (aiter, for layout).
- `-munsafe-fp-atomics` if using atomic-max for a per-tensor amax.
- rounding fixed to FE_TONEAREST (don't change).

## Numerics / parity
INT32 accumulate; `int32_to_int8` saturates; symmetric `amax/127`; azp `round(-128-min/s)`. Gate on task
accuracy → [[numerics.md]], [[languages/hip_cpp]].

## Integration (rebind seam)
Edit the `.cu`, rebuild aiter JIT or vLLM; register via `compile_ops`/`torch_bindings`. Tier-C seam.

## Pitfalls & anti-patterns
- VGPR spill from oversized `thread_data_size`.
- `warpSize==64` in the row-reduce.
- Early INT32→INT8 down-convert (overflow).

## How to verify
`-Rpass-analysis=kernel-resource-usage`; rocprofv3 bandwidth vs HBM peak; round-trip error.

## Alternatives / cross-links
[vllm_kernels.md](vllm_kernels.md) · [aiter.md](aiter.md) · [triton.md](triton.md) ·
[[languages/hip_cpp]] · [overview.md](../overview.md).

## Sources
- vLLM int8 HIP kernels + helpers: `vllm-project/vllm@HEAD:csrc/quantization/compressed_tensors/int8_quant_kernels.cu`.
- aiter smooth per-token int8 kernel: `ROCm/aiter@a6bb49937:csrc/kernels/quant_kernels.cu`.
- HIP wave64/launch_bounds: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
