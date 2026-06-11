---
title: paged_kv_copy on vLLM kernels — SOTA card
kind: sota_card
operator: paged_kv_copy
backend: vllm_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode, prefill, both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/tree/main/csrc/rocm
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
  - https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
  - https://docs.vllm.ai/en/latest/design/paged_attention/
---

# paged_kv_copy × vLLM kernels

## TL;DR
vLLM ships its **own HIP** `reshape_and_cache_flash` (`csrc/cache_kernels.cu`, vendor-guarded ROCm path) and
copy/swap block ops, plus the AMD-specific **shuffled KV-cache layout** op: a custom `reshape_and_cache`
that writes the cache in the layout AITER's `pa_fwd_asm` reads with **zero conversion**. This is the SOTA
write path on vLLM-V1/ROCm, gated by `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` for high-concurrency MHA.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `reshape_and_cache_flash` (HIP, `CALL_RESHAPE_AND_CACHE`) | `vllm/csrc/cache_kernels.cu` (ROCm-guarded) | gfx942/950, bf16/fp16/FP8 | memory-bound; no isolated GB/s published — measure | vLLM-V1 KV write |
| shuffled-layout `reshape_and_cache` | vLLM ROCm attention path | gfx942 | removes per-step KV conversion for `ROCM_AITER_FA` at **≥32 concurrency** (AMD/vLLM-reported) | high-concurrency MHA |
| `copy_blocks` / `swap_blocks` | `csrc/cache_kernels.cu` | both | bulk block copy (prefix cache, CPU offload) | prefix reuse / offload |

## Config space / knobs
- `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` (default **0**; set **1** for ≥32 conc MHA + `ROCM_AITER_FA`).
- FP8 KV: `e4m3` on AMD (vs e5m2 NVIDIA); per-head scale array (vllm#30141).
- Cache layout `k:[num_blocks, num_kv_heads, head_size/x, block_size, x]` (`x` = vectorization pack).
- HIP-graph capture of the decode step; `HIP_FORCE_DEV_KERNARG=1`.

## Numerics / parity
non-quant exact; FP8 KV (e4m3 fnuz on gfx942) → accuracy-gate, dialect must match attention read; shuffled
layout is loss-free given a correct reader. See [[operators/paged_kv_copy/numerics.md]].

## Integration (rebind seam)
The HIP source (`csrc/cache_kernels.cu`) is the Tier-C edit seam (rebuild vLLM). Engaged automatically by the
KV-cache manager; shuffled layout by the env flag + `ROCM_AITER_FA` backend. Registered in
`csrc/rocm/torch_bindings.cpp` for ROCm-specific ops.

## Pitfalls & anti-patterns
- ⚠ Shuffled layout below ~32 concurrency can cost more than the conversion it saves — A/B it.
- ⚠ V0-era KV env vars are silently ignored on V1.
- ⚠ Editing `csrc/*.cu` needs a vLLM rebuild (not Python-only).
- ⚠ FP8 KV dialect (fnuz gfx942) mismatch with the reader → 2× error.

## How to verify
rocprofv3 decode trace → `reshape_and_cache_flash` present, inside the graph, not a hotspot; with shuffle on,
no separate conversion kernel before `pa_fwd_asm`; greedy/temp=0 parity.

## Alternatives / cross-links
[backends/aiter.md](aiter.md) (fused QK + asm layout) · [backends/hip.md](hip.md) · [backends/triton.md](triton.md) ·
[[backends/vllm_kernels/rocm_kernels.md]] · [[operators/attention_decode_paged/overview.md]].

## Sources
- vLLM ROCm cache kernels (reshape_and_cache_flash, copy/swap, vendor guards): https://github.com/vllm-project/vllm/tree/main/csrc/rocm · https://docs.vllm.ai/en/latest/design/paged_attention/
- Shuffled KV layout / pa_fwd_asm zero-conversion / ≥32 conc: https://vllm.ai/blog/2026-02-27-rocm-attention-backend · https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
