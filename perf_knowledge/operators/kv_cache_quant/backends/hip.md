---
title: kv_cache_quant on hip — SOTA card
kind: sota_card
operator: kv_cache_quant
backend: hip
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3, int8]
regimes: [decode, prefill, both]
status: sota
updated: 2026-06-08
sources:
  - vllm-project/vllm@HEAD:csrc/cache_kernels.cu
  - vllm-project/vllm@HEAD:csrc/quantization/fp8/amd/quant_utils.cuh
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
---

# kv_cache_quant × hip

## TL;DR
HIP is the editable source under the KV store-quant: vLLM's `reshape_and_cache[_flash]` and aiter's fused
QK-norm+RoPE+cache-quant kernels are HIP/C++. Reach for it for a custom KV layout, a new scale granularity,
or to control the FP8 convert. The kernel is a scatter-with-cast: one thread block per token/slot,
vectorized `scaled_convert` over head_dim, writing into `slot_mapping`. Use the arch FP8 helpers (fnuz
gfx942 / ocp gfx950).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| vLLM `reshape_and_cache[_flash]_kernel` | `csrc/cache_kernels.cu:153,207` | gfx942/950, e4m3fnuz | scatter + inline scaled_convert | the editable KV-write |
| `fp8::scaled_convert<cache_t,scalar_t,kv_dt>` | `csrc/quantization/fp8/amd/quant_utils.cuh` | gfx942/950 | the cast primitive | inside the KV kernel |
| aiter fused KV-quant kernels (HIP) | aiter `csrc/kernels/fused_qk_norm_rope_cache_quant.cu` | gfx942/950 | fused norm+RoPE+write+quant | production decode |

## Config space / knobs
- `block_size` (paged), `x` (K-layout split), `kv_cache_dtype` template select.
- vector width over head_dim; one block per slot; grid = num_tokens.
- `__restrict__`, `-munsafe-fp-atomics` (if any atomic scale path).

## Numerics / parity
fnuz gfx942 / ocp gfx950; online softmax fp32 in the read kernel; per-tensor or block scale. Gate on
gsm8k → [[numerics.md]], [[languages/hip_cpp]].

## Integration (rebind seam)
Edit the `.cu`, rebuild vLLM / aiter JIT; bound via `torch_bindings` / `compile_ops`. Tier-C seam.

## Pitfalls & anti-patterns
- FNUZ↔OCP convert-helper mismatch with the attention read.
- Wrong paged layout (`block_size`/`x`) vs the cache allocation.
- `warpSize==64`; VGPR spill from oversized tiles.

## How to verify
`-Rpass-analysis=kernel-resource-usage`; rocprofv3 confirm the KV-write kernel; round-trip KV error; gsm8k.

## Alternatives / cross-links
[vllm_kernels.md](vllm_kernels.md) · [aiter.md](aiter.md) · [triton.md](triton.md) ·
[[languages/hip_cpp]] · [overview.md](../overview.md) · [[operators/paged_kv_copy]].

## Sources
- vLLM KV-write HIP + scaled_convert: `vllm-project/vllm@HEAD:csrc/cache_kernels.cu`, `csrc/quantization/fp8/amd/quant_utils.cuh`.
- HIP wave64/launch_bounds: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
