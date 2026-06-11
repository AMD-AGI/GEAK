---
title: paged_kv_copy on aiter — SOTA card
kind: sota_card
operator: paged_kv_copy
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [decode, prefill, both]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/cache.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/fused_qk_norm_rope_cache_quant.py
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
---

# paged_kv_copy × aiter

## TL;DR
aiter ships the full paged-KV write family as JIT-compiled HIP (`module_cache`): `reshape_and_cache(_flash)`,
`copy_blocks`, `swap_blocks`, `concat_and_cache_mla`, plus **quant** (`_with_pertoken_quant`,
`_with_block_quant`, `_for_asm_pa`) and **fused** (`fused_qk_rope_concat_and_cache_mla`,
`fused_qk_norm_rope_cache_quant`) variants. It's the SOTA write path when the attention is AITER FA/MLA,
because the cache is written in the layout `pa_fwd_asm` consumes. Verified on-box at
`ROCm/aiter@a6bb49937`.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `reshape_and_cache_flash` (+ asm-PA layout) | `aiter/ops/cache.py` | gfx942/950, bf16/fp16 | memory-bound; fused into the AITER decode step (not separately timed) — **honest gap**, measure | AITER FA decode |
| `reshape_and_cache_with_pertoken/block_quant` | `aiter/ops/cache.py` | FP8/int8 KV | halves KV HBM traffic for the attention read | FP8 KV serving |
| `fused_qk_norm_rope_cache_quant` / `fused_qk_rope_concat_and_cache_mla` | `aiter/ops/fused_qk_norm_rope_cache_quant.py` | bf16→FP8 | 4 passes → 1 | MLA / fused QK path |

## Config space / knobs
- `kv_cache_dtype` ("auto"/"fp8"/...); quant granularity (per-token vs per-block).
- MLA: `kv_lora_rank + pe_dim` cache shape (`concat_and_cache_mla`).
- JIT module `module_cache` compiled on first use (cached `.so`); `AITER_LOG_MORE=1` to see it.

## Numerics / parity
non-quant exact; FP8/int8 KV → accuracy-gate; FNUZ(gfx942)/OCP(gfx950) must match the attention read. See
[[operators/paged_kv_copy/numerics.md]].

## Integration (rebind seam)
Called inside the AITER attention path (`VLLM_ROCM_USE_AITER_MHA/MLA=1` / SGLang AITER attention). The fused
QK kernels are wired by the attention backend, not a standalone env. Pairs with the shuffled-layout write
([backends/vllm_kernels.md](vllm_kernels.md)).

## Pitfalls & anti-patterns
- ⚠ KV dtype/dialect mismatch with the attention read → 2× FP8 error.
- ⚠ Using the plain (non-asm) layout under `pa_fwd_asm` → a conversion the shuffled path avoids.
- ⚠ JIT first-call compile latency — warm before timing.

## How to verify
`AITER_LOG_MORE=1` → the cache kernel + (if used) fused QK kernel fire; rocprofv3 decode trace → write not a
latency hotspot, inside the graph; oracle `allclose`.

## Alternatives / cross-links
[backends/vllm_kernels.md](vllm_kernels.md) (shuffled-layout op) · [backends/hip.md](hip.md) ·
[backends/triton.md](triton.md) · [[operators/kv_cache_quant/overview.md]] ·
[[operators/attention_decode_paged/overview.md]].

## Sources
- On-box: `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0`: `aiter/ops/cache.py` (reshape_and_cache family, MLA, quant), `aiter/ops/fused_qk_norm_rope_cache_quant.py`.
- aiter as central engine: https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
