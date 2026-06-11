---
title: paged_kv_copy — overview
kind: operator_overview
operator: paged_kv_copy
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, int8]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/cache.py
  - https://github.com/vllm-project/vllm/tree/main/csrc/rocm
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
  - https://docs.vllm.ai/en/latest/design/paged_attention/
---

# paged_kv_copy  (reshape_and_cache / block copy / append)

## TL;DR
The memory-movement glue of paged attention: take freshly computed K/V (and on MLA, the latent) and
**reshape + write** them into the **paged KV-cache blocks** (`reshape_and_cache`), plus **copy/swap blocks**
for prefix-cache reuse and CPU offload. Pure bandwidth, but on the **decode critical path** every step, so
it's latency-sensitive. The AMD-specific lever is the **shuffled KV-cache layout**
(`VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT`): writing the cache already in the AITER `pa_fwd_asm`-friendly layout so
the attention kernel reads it with **zero conversion**. Tightly coupled to
[[operators/attention_decode_paged/overview.md]] and [[operators/kv_cache_quant/overview.md]].

## Math contract
- **reshape_and_cache(_flash)**: write `key/value[num_tokens, num_heads, head_size]` into
  `kv_cache` blocks indexed by `slot_mapping[token]` (block_idx, block_offset). vLLM cache layout:
  `k_cache:[num_blocks, num_kv_heads, head_size/x, block_size, x]`, `v_cache:[num_blocks, num_kv_heads,
  head_size, block_size]` (the `x` packing is the "reshape").
- **copy_blocks / swap_blocks**: copy whole blocks (GPU↔GPU prefix-cache, GPU↔CPU offload) by a
  `block_mapping`.
- **concat_and_cache_mla**: write the MLA latent + pe into the cache (`[num_blocks, block_size,
  kv_lora_rank+pe_dim]`).
- **quant variants**: `reshape_and_cache_with_pertoken_quant` / `_with_block_quant` write FP8/int8 KV with a
  scale in one pass → [[operators/kv_cache_quant/overview.md]].
- Element move (+ optional quant); dtype in = bf16/fp16, out = same or FP8/int8 KV.

## Shape regimes
- **decode**: 1 token/seq/step → tiny per-call write, but **every step** → latency-bound, launch-overhead
  sensitive (HIP-graph it).
- **prefill / chunked**: many tokens written at once → bandwidth-bound; coalesce the head_size payload.
- **prefix cache**: block copies of reused prompt prefixes (bandwidth-bound bulk copy).
- Memory-bound throughout; ideal ≈ `2·KV_bytes / 5.3 TB/s` (or 1× for a one-directional write).

## Where it matters (Amdahl)
Not a top-N compute kernel, but it's on the **decode hot path at every token**, so its **launch overhead**
and any **layout-conversion** cost compound across thousands of steps. The shuffled-layout trick removes a
per-step conversion for high-concurrency MHA (≥32 reqs); FP8 KV quant (fused into the write) is the bigger
win, halving KV HBM traffic for the attention read. Spend on (1) graph capture, (2) shuffled layout, (3)
fused FP8 quant.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (`reshape_and_cache` family + quant + MLA; asm `pa_fwd_asm` layout) | [backends/aiter.md](backends/aiter.md) |
| vllm_kernels | 🟢 sota (own HIP `reshape_and_cache_flash`, shuffled-layout op, copy/swap) | [backends/vllm_kernels.md](backends/vllm_kernels.md) |
| hip | 🟢 (the editable HIP source; vectorized block write) | [backends/hip.md](backends/hip.md) |
| triton | 🟡 competitive (Triton reshape_and_cache; portable/fallback) | [backends/triton.md](backends/triton.md) |

## Fusion neighbors
**+RoPE +norm +quant**: aiter `fused_qk_rope_concat_and_cache_mla` / `fused_qk_norm_rope_cache_quant` fold
QK-norm, RoPE, the cache write, and FP8 quant into one kernel; **+FP8/int8 quant** on write
([[operators/kv_cache_quant/overview.md]]); **shuffled layout** so attention reads with no conversion. See
[fusion.md](fusion.md).

## Numerics
Plain copy = byte-exact. FP8/int8 KV quant introduces error (scale choice) → accuracy-gate; FNUZ (gfx942) vs
OCP (gfx950) dialect must match the attention read. See [numerics.md](numerics.md).

## How to bench
Isolated: time `reshape_and_cache_flash` at decode (1 tok) and prefill (chunk) shapes; oracle `allclose`
(exact for non-quant). e2e: decode tok/s with/without graph capture and shuffled layout. rocprofv3 to confirm
the write isn't a launch-overhead-dominated micro-kernel.

## Sources
- aiter cache ops (reshape_and_cache family, MLA, quant variants): ROCm/aiter@a6bb49937:aiter/ops/cache.py.
- vLLM ROCm reshape_and_cache_flash + shuffled layout (`VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT`, pa_fwd_asm zero-conversion): https://github.com/vllm-project/vllm/tree/main/csrc/rocm · https://vllm.ai/blog/2026-02-27-rocm-attention-backend
- Paged KV-cache layout: https://docs.vllm.ai/en/latest/design/paged_attention/
