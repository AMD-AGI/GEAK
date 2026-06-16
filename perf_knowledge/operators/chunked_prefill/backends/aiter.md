---
title: chunked_prefill on aiter — SOTA card
kind: sota_card
operator: chunked_prefill
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/unified_attention.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/chunked_pa_prefill.py
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---

# chunked_prefill × aiter

## TL;DR
aiter provides both chunked-prefill kernels: the **unified** `unified_attention` (one launch for
prefill+decode) and the **split** `chunked_prefill_paged_decode` (`context_attention_fwd` + paged decode).
These are the kernels sglang/vLLM call for mixed serving; aiter owns the catalog and the dispatch. Use
aiter; choose unified for launch-bound small batch, split for large batch (bake off).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `unified_attention` | `ROCm/aiter@a6bb49937:aiter/ops/triton/attention/unified_attention.py` | gfx942/950; bf16/fp16/fp8; GQA/SWA/softcap/alibi/sinks/paged | one-launch; part of AITER FA 1.2–4.4× TPS for mixed serving (vendor) | launch-bound mixed batches |
| `chunked_prefill_paged_decode` (split) | same repo `:aiter/ops/triton/attention/chunked_pa_prefill.py` | as above | two-launch (prefill + paged decode) | large per-step batches |

## Config space / knobs
Unified: see [triton.md](triton.md) (BLOCK_M/BLOCK_Q from GQA ratio, NUM_SEGMENTS_PER_SEQ, causal,
window, softcap, fp8 descales). Split: `context_attention_fwd` (prefill) + the paged-decode path, with
`block_table`/`b_loc`. Framework: `VLLM_ROCM_USE_AITER=1`; sglang `SGLANG_USE_AITER=1` +
`SGLANG_USE_AITER_UNIFIED_ATTN=1` for unified. `--chunked-prefill-size` (scheduler chunk).

## Numerics / parity
fp32 online-softmax over segmented KV; chunk-invariance gate; unified-vs-split tie-flips benign; fp8
descales (fnuz on gfx942). See [../numerics.md](../numerics.md).

## Integration (rebind seam)
sglang `--attention-backend aiter` + `SGLANG_USE_AITER_UNIFIED_ATTN=1`; vLLM `TRITON_ATTN` / AITER FA. The
unified kernel import (`from aiter.ops.triton.attention.unified_attention import unified_attention`) is the
literal dispatch surface in sglang's `aiter_backend.py`.

## Pitfalls & anti-patterns
- Unified kernel is **causal only**.
- `seqused_k`/`block_table` must track cached context (chunk-boundary correctness).
- Master switch `VLLM_ROCM_USE_AITER=1` / `SGLANG_USE_AITER=1` required.
- Bake off unified vs split per batch size.

## How to verify
Chunk-invariance test; `AITER_LOG_MORE=1`; mixed-batch bench unified vs split; rocprofv3 (one kernel vs
two); greedy temp=0 parity; e2e throughput + TPOT + TTFT.

## Alternatives / cross-links
[triton.md](triton.md) · [sglang_kernels.md](sglang_kernels.md) · [vllm_kernels.md](vllm_kernels.md) ·
`backends/aiter/attn_mla.md` · [[../overview.md]].

## Sources
- `unified_attention` + `chunked_prefill_paged_decode` (on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/`).
- AITER FA mixed-serving TPS: https://vllm.ai/blog/2026-02-27-rocm-attention-backend
