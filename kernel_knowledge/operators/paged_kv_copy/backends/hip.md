---
title: paged_kv_copy on HIP — SOTA card
kind: sota_card
operator: paged_kv_copy
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [decode, prefill, both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/tree/main/csrc/rocm
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/cache.py
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# paged_kv_copy × HIP

## TL;DR
The actual `reshape_and_cache`/`copy_blocks` kernels in both aiter (`module_cache`) and vLLM
(`csrc/cache_kernels.cu`) **are** HIP — so HIP is the editable seam for the KV write: vectorize the
head_size payload, fuse quant/RoPE, and capture into a graph. Author HIP to add a layout/quant variant the
libraries don't have; otherwise rebind to aiter/vllm_kernels.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| vectorized block write (one wg/token, `dwordx4`) | this card / vLLM & aiter HIP | gfx942/950, all dtypes | memory-bound (measure) | the editable KV write |
| fused write + quant/RoPE | aiter `cache.py` / `fused_qk_*` (HIP) | bf16→FP8 | one HBM pass | custom fused variant |

## Config space / knobs
- Coalesce `head_size` into `float4` (`global_store_dwordx4`); `slot_mapping` = the one scattered index.
- `__launch_bounds__`, block multiple of 64; **HIP-graph** the decode step (the dominant decode lever).
- `-munsafe-fp-atomics` only if a variant accumulates (rare here — writes overwrite).
- Match the cache layout to the attention reader (shuffled / asm-PA) — the layout *is* the perf lever.

## Numerics / parity
non-quant exact; FP8/int8 → accuracy-gate; FNUZ/OCP must match the reader. See
[[operators/paged_kv_copy/numerics.md]].

## Integration (rebind seam)
`.hip` compiled `--offload-arch=gfx942[ gfx950]`, bound via torch custom op; in vLLM the seam is
`csrc/cache_kernels.cu` (rebuild). Usually subsumed by aiter/vllm_kernels — author for a missing variant.

## Pitfalls & anti-patterns
- ⚠ Per-step launch overhead at decode → graph-capture (bigger win than copy micro-opt).
- ⚠ Un-vectorized head_size write → bandwidth-starved.
- ⚠ Writing a layout the attention kernel must convert → wasted per-step conversion.

## How to verify
rocprofv3 decode trace → write inside the graph, not a hotspot; ISA `global_store_dwordx4`; oracle `allclose`.

## Alternatives / cross-links
[backends/aiter.md](aiter.md) · [backends/vllm_kernels.md](vllm_kernels.md) · [backends/triton.md](triton.md) ·
[[languages/hip_cpp/patterns.md]] · [[operators/paged_kv_copy/tuning.md]].

## Sources
- vLLM/aiter HIP cache kernels: https://github.com/vllm-project/vllm/tree/main/csrc/rocm · ROCm/aiter@a6bb49937:aiter/ops/cache.py.
- Graph capture / coalescing: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html · [[languages/hip_cpp/patterns.md]].
