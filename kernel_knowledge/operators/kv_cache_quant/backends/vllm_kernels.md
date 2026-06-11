---
title: kv_cache_quant on vllm_kernels — SOTA card
kind: sota_card
operator: kv_cache_quant
backend: vllm_kernels
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3, int8]
regimes: [decode, prefill, both]
status: sota
updated: 2026-06-08
sources:
  - vllm-project/vllm@HEAD:csrc/cache_kernels.cu
  - vllm-project/vllm@HEAD:csrc/quantization/fp8/amd/quant_utils.cuh
  - https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
---

# kv_cache_quant × vllm_kernels

## TL;DR
vLLM's `reshape_and_cache` / `reshape_and_cache_flash` (`csrc/cache_kernels.cu`) do the KV store-quant
**inline** as they scatter K/V into the paged cache: `fp8::scaled_convert<cache_t, scalar_t,
kv_dt>(tgt, k_scale)`. This is the canonical KV-quant write on vLLM; the `kv_cache_dtype` string
(`DISPATCH_BY_KV_CACHE_DTYPE`) selects the convert template (auto/fp8/fp8_e4m3/fp8_e5m2). The matching
read-side dequant is in the paged-attention / FA kernels.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `reshape_and_cache_flash_kernel` | `csrc/cache_kernels.cu:207` | gfx942/950, e4m3fnuz | inline scaled_convert per slot | FA paged cache (default) |
| `reshape_and_cache_kernel` | `:153` | gfx942/950 | inline scaled_convert | classic paged layout |
| `fp8::scaled_convert` / `quant_utils.cuh` | `csrc/quantization/fp8/amd/` | gfx942/950 | the cast primitive | inside the cache kernels |

## Config space / knobs
- `kv_cache_dtype` ("auto"/"fp8"/"fp8_e4m3"/"fp8_e5m2"/"int8"); `k_scale`/`v_scale` (scalar `double`).
- `--calculate-kv-scales` to derive scales; otherwise from the quant config.
- `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` for ROCM_AITER_FA at concurrency ≥32.

## Numerics / parity
`Float8_e4m3fnuz` on ROCm (fnuz); online softmax fp32; per-tensor k_scale/v_scale. Gate on gsm8k; FNUZ↔OCP
must match the read → [[numerics.md]].

## Integration (rebind seam)
`torch.ops._C_cache_ops.reshape_and_cache[_flash]`; engaged by `--kv-cache-dtype fp8`. Tier-C edit =
rebuild vLLM.

## Pitfalls & anti-patterns
- FNUZ↔OCP mismatch with the attention read.
- Layout-shuffle mismatch between write and FA read.
- Forgetting scales (auto vs calculated) → saturation/underflow.

## How to verify
rocprofv3 → confirm `reshape_and_cache_flash_kernel` ran; e2e gsm8k with/without FP8 KV; max batch gain.

## Alternatives / cross-links
[aiter.md](aiter.md) (fully fused chain) · [hip.md](hip.md) · [triton.md](triton.md) ·
[overview.md](../overview.md) · [[operators/paged_kv_copy]] · [[operators/attention_decode_paged]].

## Sources
- vLLM reshape_and_cache + scaled_convert + dispatch: `vllm-project/vllm@HEAD:csrc/cache_kernels.cu`, `csrc/quantization/fp8/amd/quant_utils.cuh`.
- kv-cache-dtype / KV shuffle env: https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
