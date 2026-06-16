---
title: chunked_prefill — fusion
kind: operator_overview
operator: chunked_prefill
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/unified_attention.py
  - https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---

# chunked_prefill — fusion

The headline fusion *is* chunked prefill's reason to exist: **fuse prefill and decode into one kernel
launch per step** (unified attention), instead of one prefill launch + one decode launch.

## Unified attention = prefill+decode launch fusion
```
step batch = {seq with prefill chunk (query_len>1), seq with decode token (query_len=1)}
   split path:    [context_attention_fwd]  +  [paged decode]      ← two launches
   unified path:  [unified_attention]                              ← one launch (query_len ≥ 1)
```
`unified_attention` handles `query_len ≥ 1` uniformly with a segmented KV (cached context + current),
so both the chunk-prefill queries and the decode queries go through one kernel. At small per-step batch
(launch-bound), removing the second launch is the win — this is `SGLANG_USE_AITER_UNIFIED_ATTN=1` /
vLLM `TRITON_ATTN`.

## In-kernel fusions (folded into the one launch)
- **GQA broadcast**: the query-head group packs into `BLOCK_M` (one KV tile serves R query heads).
- **softcap / sliding-window / ALiBi / attention-sinks**: applied inside the kernel (no separate bias
  passes).
- **fp8 descale**: `q/k/v_descale` and `output_scale` folded in (no separate quant/dequant pass).

## Pre-step fusion
Before attention, the new tokens' K/V (for both prefill chunk and decode) are rotated + written to the
paged cache + optionally fp8-quantized — the fused **RoPE + KV-write + quant** kernel (aiter). Sized to
`num_kv_heads` (GQA). One pass for the whole step's KV write.

## What does NOT fuse
- The QKV / O projection GEMMs stay separate (different shape; tuned independently).
- Non-causal attention has no unified path (the kernel asserts causal).

## Backend support
| fusion | unified (Triton/aiter) | split (chunked_prefill_paged_decode) |
|---|---|---|
| prefill+decode one launch | **yes** | no (two kernels) |
| GQA tile packing | yes | yes |
| softcap/SWA/alibi/sinks in-kernel | yes | partial |
| fp8 descale in-kernel | yes | yes |

## Where fusion moves e2e
At small/medium per-step batch chunked-prefill serving is launch-bound, so the prefill+decode launch
fusion (unified attention) is where throughput/TPOT improve — part of the AITER FA mixed-serving TPS win.
At large batch the split path is fine (launches amortize).

## Sources
- `unified_attention` one-kernel prefill+decode, GQA packing, softcap/SWA/alibi/sinks/fp8 descale: on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/unified_attention.py`.
- `SGLANG_USE_AITER_UNIFIED_ATTN`: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py
- AITER FA mixed-serving TPS: https://vllm.ai/blog/2026-02-27-rocm-attention-backend
