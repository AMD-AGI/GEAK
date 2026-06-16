---
title: chunked_prefill — tuning
kind: operator_overview
operator: chunked_prefill
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/unified_attention.py
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# chunked_prefill — tuning

Two levers, at two layers: the **scheduler** lever (chunk size → throughput vs TTFT) and the **kernel**
lever (unified vs split, segment count, GQA tile packing).

## Scheduler lever — chunk size
- **Chunk size** (`--chunked-prefill-size` / `--max-num-batched-tokens`) sets how many prefill tokens
  enter per step. Larger chunk → fewer steps, higher prefill throughput, but longer decode-step latency
  (TPOT spikes while a big chunk runs). Smaller chunk → smoother decode latency, more scheduling
  overhead. Typical 512–2048; tune to the latency SLO.
- The goal is to keep each step's total tokens near a fixed budget so decode latency is bounded — a
  serving-policy tune, not a kernel tune.

## Kernel lever 1 — unified vs split
- **Unified attention** (`SGLANG_USE_AITER_UNIFIED_ATTN=1` / vLLM `TRITON_ATTN`): one kernel for
  chunk-prefill + decode tokens. Wins when small per-step batches make two separate launches the
  bottleneck (**launch-bound** small batch).
- **Split** (`chunked_prefill_paged_decode`): `context_attention_fwd` for prefill + paged decode for the
  single-token queries. Fine when batches are large enough that the two launches amortize.
- Bake off both at the served step shape.

## Kernel lever 2 — segmentation (flash-decoding for the cached context)
The unified kernel segments each sequence's KV history: `NUM_SEGMENTS_PER_SEQ` (≈ `min(128,
ceil(max_seqlen_k / TILE_SIZE))`, floor `MIN_SEGMENTS = 16 if TILE_SIZE≤16 else 8`). More segments fill
CUs for long-context decode tokens in the mix; the kernel auto-derives it from `max_seqlen_k`. This is
the decode-side splitKV inside the unified kernel.

## Kernel lever 3 — GQA tile packing
`BLOCK_M = 16 if num_queries_per_kv ≤ 16 else next_pow2(num_queries_per_kv)`; `BLOCK_Q = BLOCK_M /
num_queries_per_kv`. The query-head group packs into M so one KV tile serves R query heads (the GQA win,
see [[../gqa_mqa_attention/tuning.md]]). The kernel sets these from the head ratio — you mainly ensure the
ratio is correct.

## Kernel lever 4 — standard FA/decode knobs
`num_warps=4` (wave64), `num_stages=1`, `matrix_instr_nonkdim=16`, `waves_per_eu∈{2,3,4}`. fp8 KV via
`q/k/v_descale` (fnuz on gfx942). `--page-size` for the paged cache. The kernel asserts **causal** — no
non-causal unified path.

## CDNA3 vs CDNA4
- LDS 64 KB (gfx942) / 160 KB (gfx950): the unified kernel co-resides prefill Q tiles and decode segments
  in LDS; gfx950's larger LDS allows bigger chunks / more segments.
- fp8: FNUZ (gfx942) vs OCP (gfx950), wrong dialect off by 2×.

## How to verify a tune helped
Mixed-batch bench (N prefill chunks + M decode tokens, shared paged cache) unified vs split at
small/large batch; e2e throughput + TPOT + TTFT at the chosen chunk size; rocprofv3 to confirm one
unified kernel vs two; greedy temp=0 parity.

## Sources
- `unified_attention` BLOCK_M/BLOCK_Q from num_queries_per_kv, NUM_SEGMENTS_PER_SEQ derivation, causal assert, fp8 descales: on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/unified_attention.py`.
- Triton AMD knobs (num_stages=1, wave64): https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- ≥1024 grid / MFMA: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
