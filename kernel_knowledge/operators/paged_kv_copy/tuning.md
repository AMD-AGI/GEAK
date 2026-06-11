---
title: paged_kv_copy — tuning (coalesced block write, shuffled layout, graph capture, fused quant)
kind: technique
operator: paged_kv_copy
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [decode, prefill, both]
updated: 2026-06-08
sources:
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
  - https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/cache.py
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# paged_kv_copy — tuning

Memory-bound and on the decode critical path. Four levers: **coalesce the payload**, **match the read layout**,
**kill launch overhead**, **fuse the quant**.

## 1. Coalesce the head_size payload (128-bit)
Map one token's `[num_heads, head_size]` write so the 64 lanes of a wave cover contiguous `head_size` bytes →
`global_store_dwordx4`. The `slot_mapping`-driven **block index is the (single) scattered address**; the
payload must stay coalesced (same principle as [[operators/gather_scatter/tuning.md]] §1). The vLLM cache
layout `[num_blocks, num_kv_heads, head_size/x, block_size, x]` packs `x` contiguous elements precisely so the
write/read is vectorized.

## 2. Match the attention read layout (the AMD shuffled-KV trick)
The biggest AMD-specific lever: **write the cache already in the layout the attention kernel wants** so the
read needs **zero conversion**. vLLM's decode path uses a **shuffled KV-cache layout** via a custom
`reshape_and_cache` op so AITER's `pa_fwd_asm` reads it directly. Gate:
- `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` — for **ROCM_AITER_FA** at **high concurrency (≥32 reqs)** MHA; default 0.
- Below ~32 concurrency the conversion-free read doesn't pay for the shuffled write — A/B it.

This is the KV-cache analog of weight pre-shuffle ([[operators/layout_shuffle/overview.md]]): pay the layout
cost on **write** so the **read** (hotter, every attention step) is free.

## 3. Kill launch overhead (HIP graphs)
At decode the write is 1 token → the kernel is tiny and **launch-overhead-bound**. Capture the decode step
(attention + reshape_and_cache + ...) into a **HIP graph** so the per-step CPU launch cost vanishes
(`hipStreamBeginCapture` → `hipGraphInstantiate` → `hipGraphLaunch`). `HIP_FORCE_DEV_KERNARG=1`,
`GPU_MAX_HW_QUEUES=2`. This matters more than micro-optimizing the copy itself.

## 4. Fuse the quant (and RoPE/norm)
Don't write bf16 then re-read+quantize — write FP8/int8 directly:
- aiter `reshape_and_cache_with_pertoken_quant` / `_with_block_quant` write quantized KV + scale in one pass.
- `fused_qk_rope_concat_and_cache_mla` / `fused_qk_norm_rope_cache_quant` fold QK-norm + RoPE + cache write +
  quant into one kernel — one HBM pass instead of four. See [[operators/paged_kv_copy/fusion.md]].
FP8 KV **halves the KV HBM traffic** the attention read pays — the dominant decode bandwidth term.

## 5. Block copy / swap (prefix cache, offload)
`copy_blocks`/`swap_blocks` are bulk bandwidth — vectorize the per-block copy (`dwordx4`), use a dedicated
stream so GPU↔CPU offload (pinned host memory → true async DMA) overlaps compute. Prefix-cache hits make these
free vs recompute; the copy is the cost of reuse.

## Verify
rocprofv3 decode trace: reshape_and_cache is **not** a top latency contributor and is **inside the graph**
(no per-step launch gap); with shuffled layout, no separate KV-conversion kernel before attention. ISA:
`global_store_dwordx4`. Oracle: `allclose` (exact non-quant; FP8 within tol).

## Sources
- Shuffled KV layout, `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` (≥32 conc, pa_fwd_asm zero-conversion): https://vllm.ai/blog/2026-02-27-rocm-attention-backend · https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
- aiter quant/fused cache ops: ROCm/aiter@a6bb49937:aiter/ops/cache.py, aiter/ops/fused_qk_norm_rope_cache_quant.py.
- Graph capture / coalescing / pinned-DMA: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html · [[languages/hip_cpp/patterns.md]].
