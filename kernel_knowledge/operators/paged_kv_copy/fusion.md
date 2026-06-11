---
title: paged_kv_copy — fusion (RoPE+norm+cache+quant; shuffled layout)
kind: technique
operator: paged_kv_copy
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode, prefill, both]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/fused_qk_norm_rope_cache_quant.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/cache.py
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---

# paged_kv_copy — fusion

The cache write touches every K/V element once; fuse the neighbors into that single HBM pass.

## Fusion targets
| pattern | fused kernel (aiter) | effect | link |
|---|---|---|---|
| **QK-norm + RoPE + cache write + FP8 quant** | `fused_qk_norm_rope_cache_quant` | 4 passes → 1; the K/V is normed, rotated, quantized, and written in one kernel | [[operators/rope/overview.md]], [[operators/kv_cache_quant/overview.md]] |
| **RoPE + concat + cache (MLA)** | `fused_qk_rope_concat_and_cache_mla` | folds the MLA latent concat + RoPE + write | [[operators/mla_attention/overview.md]] |
| **cache write + FP8/int8 quant** | `reshape_and_cache_with_pertoken_quant` / `_with_block_quant` | write quantized KV + scale in one pass (no bf16-then-requant) | [[operators/kv_cache_quant/overview.md]] |
| **write-layout = read-layout** (shuffled KV) | vLLM custom `reshape_and_cache` (`VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1`) | attention (`pa_fwd_asm`) reads with **zero conversion** | [[operators/attention_decode_paged/overview.md]] |

## The two fusion ideas
1. **Producer-side fusion** (norm/RoPE/quant → write): the K/V values are already in registers after attention
   projection; norm, rotate, and quantize them *there* and write once. aiter's fused QK kernels do exactly
   this. The alternative (separate norm, RoPE, quant, write kernels) is 4× the HBM traffic.
2. **Layout fusion** (shuffle on write): a one-time layout cost on the **write** removes a per-step conversion
   on the **read**. Same trade as weight pre-shuffle ([[operators/layout_shuffle/overview.md]]) — pay where it's
   cold (write, once per token) to save where it's hot (read, every attention step). Only wins at high
   concurrency (≥32) for MHA — A/B it.

## Don't over-fuse
At low concurrency the shuffled write can cost more than the conversion it saves. The FP8 quant fusion almost
always wins (halves KV traffic). Graph-capture the whole decode step regardless ([[operators/paged_kv_copy/tuning.md]] §3).

## Cross-links
[[operators/paged_kv_copy/tuning.md]] · [[operators/kv_cache_quant/overview.md]] ·
[[operators/attention_decode_paged/overview.md]] · [[operators/layout_shuffle/overview.md]] (the analog).

## Sources
- aiter fused QK-norm/RoPE/cache/quant kernels: ROCm/aiter@a6bb49937:aiter/ops/fused_qk_norm_rope_cache_quant.py, aiter/ops/cache.py (`fused_qk_rope_concat_and_cache_mla`, `reshape_and_cache_with_*_quant`).
- Shuffled-layout zero-conversion read: https://vllm.ai/blog/2026-02-27-rocm-attention-backend
