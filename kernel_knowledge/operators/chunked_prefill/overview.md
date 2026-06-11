---
title: chunked_prefill — overview
kind: operator_overview
operator: chunked_prefill
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/unified_attention.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/chunked_pa_prefill.py
  - https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---

# chunked_prefill  (mixed prefill+decode / unified attention)

## TL;DR
Chunked prefill splits a long prompt's prefill into **chunks** so each scheduler step mixes a slice of
prefill tokens with the running batch's decode tokens — keeping the GPU busy and bounding TTFT. The kernel
problem is **one attention call that serves variable-length queries** (chunk-prefill queries with
`query_len > 1`, decode queries with `query_len = 1`) against a **paged KV-cache** with prior context.
The SOTA on AMD is the **unified attention** kernel (`unified_attention`, also vLLM's `TRITON_ATTN` /
sglang `SGLANG_USE_AITER_UNIFIED_ATTN=1`): one kernel for chunked-prefill + decode, which avoids running
separate prefill and decode launches per step.

## Math contract
`O = softmax(QKᵀ·scale + causal_mask + window + softcap)·V` over a **paged** KV-cache, with per-sequence
variable query length:
- `cu_seqlens_q` (cumulative query lengths — the ragged batch), `seqused_k` (per-seq KV length incl.
  cached context), `block_table` (paged KV), `causal=True` (the kernel asserts causal), `window_size`
  (sliding window), `softcap`, fp8 `q/k/v_descale`, optional `alibi_slopes`, `sinks` (attention sinks).
- GQA built in: `num_queries_per_kv = num_query_heads // num_kv_heads` drives `BLOCK_M`/`BLOCK_Q` (the
  query-head group packs into the M tile). fp32 online-softmax accumulate.

## The two designs
- **Unified attention** (`unified_attention`): a single kernel that segments each sequence's KV
  (`NUM_SEGMENTS_PER_SEQ`, flash-decoding-style) and handles `query_len ≥ 1` uniformly — so chunk-prefill
  and decode tokens go through the **same** kernel. Best when small per-step batches make separate
  prefill/decode launches the bottleneck (launch-bound).
- **Separate prefill+decode** (`chunked_prefill_paged_decode`): calls `context_attention_fwd` for the
  prefill (`query_len>1`) tokens and the paged-decode kernel for the `query_len=1` tokens — the classic
  two-kernel path. Simpler, fine when batches are large.

## Shape regimes
Mixed: some sequences contribute a prefill chunk (`query_len = chunk_size`, e.g. 512/2048), others a
single decode token (`query_len = 1`), all against their cached context. The ragged `cu_seqlens_q` is the
defining shape. Bridges [[../attention_prefill_fmha/overview.md]] and
[[../attention_decode_paged/overview.md]].

## Where it matters (Amdahl)
Chunked prefill is the **scheduler-level** lever for throughput-vs-latency: it keeps decode latency bounded
while a long prompt is ingested. The unified kernel removes per-step launch overhead — material at small
batch (launch-bound), and part of the AITER FA 1.2–4.4× TPS story for mixed serving.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| triton (unified attention) | 🟢 sota (the unified kernel) | [backends/triton.md](backends/triton.md) |
| aiter | 🟢 (unified + separate, dispatch) | [backends/aiter.md](backends/aiter.md) |
| sglang_kernels | 🟢 (selects unified vs split) | [backends/sglang_kernels.md](backends/sglang_kernels.md) |
| vllm_kernels | 🟢 (TRITON_ATTN unified, scheduler) | [backends/vllm_kernels.md](backends/vllm_kernels.md) |

## Fusion neighbors
Unified kernel fuses prefill+decode into one launch; pre-step RoPE+KV-write+quant; softcap/alibi/sinks
in-kernel. See [fusion.md](fusion.md).

## Numerics
fp32 online-softmax over segmented KV; causal-only; fp8 descales; same equivalence-class tie-flips. See
[numerics.md](numerics.md).

## How to bench
Mixed batch: N sequences with a prefill chunk + M sequences with 1 decode token, shared paged cache;
compare unified vs split at small/large batch; e2e throughput + TPOT + TTFT.

## Sources
- `unified_attention` (cu_seqlens_q, seqused_k, block_table, causal, window, softcap, fp8 descales, alibi, sinks, GQA BLOCK_M packing, NUM_SEGMENTS_PER_SEQ): on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/unified_attention.py`.
- `chunked_prefill_paged_decode` (context_attention_fwd + paged decode): same repo `:aiter/ops/triton/attention/chunked_pa_prefill.py`.
- sglang unified attn flag: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py
- AITER FA mixed-serving TPS: https://vllm.ai/blog/2026-02-27-rocm-attention-backend
