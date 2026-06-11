---
title: chunked_prefill on Triton — SOTA card
kind: sota_card
operator: chunked_prefill
backend: triton
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/unified_attention.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
---

# chunked_prefill × Triton

## TL;DR
The **unified attention** Triton kernel (`unified_attention`) is the SOTA chunked-prefill primitive on
AMD: one kernel for chunk-prefill + decode tokens over a paged KV-cache, with GQA, sliding-window,
softcap, ALiBi, attention-sinks, and fp8 descales built in. It is vLLM's `TRITON_ATTN` and sglang's
`SGLANG_USE_AITER_UNIFIED_ATTN=1`. Use it as the default for mixed serving; it is also fully editable
(Tier-C seam).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `unified_attention` (Triton) | `ROCm/aiter@a6bb49937:aiter/ops/triton/attention/unified_attention.py` | gfx90a/942/950; bf16/fp16/fp8; GQA; SWA/softcap/alibi/sinks; paged | one-launch prefill+decode; part of AITER FA 1.2–4.4× TPS for mixed serving (vendor) | launch-bound mixed batches |
| split: `context_attention_fwd` + paged decode | same repo `:aiter/ops/triton/attention/chunked_pa_prefill.py` | as above | two-launch; fine at large batch | large per-step batches |

## Config space / knobs
`BLOCK_M = 16 if num_queries_per_kv ≤16 else next_pow2(ratio)`, `BLOCK_Q = BLOCK_M/ratio`,
`NUM_SEGMENTS_PER_SEQ` (auto from `max_seqlen_k`, floor 8/16), `num_warps=4`, `num_stages=1`,
`matrix_instr_nonkdim=16`, `waves_per_eu∈{2,3,4}`. Inputs: `cu_seqlens_q`, `seqused_k`, `block_table`,
`causal=True` (asserted), `window_size`, `softcap`, `q/k/v_descale`, `alibi_slopes`, `sinks`,
`output_scale`. See [../tuning.md](../tuning.md).

## Numerics / parity
fp32 online-softmax over segmented KV; **chunk-invariance is the primary gate** (chunked == un-chunked).
fp8 descales (fnuz on gfx942). See [../numerics.md](../numerics.md).

## Integration (rebind seam)
sglang: `SGLANG_USE_AITER_UNIFIED_ATTN=1` (+ `--attention-backend aiter`/`triton`). vLLM:
`--attention-backend TRITON_ATTN`. The `@triton.jit` kernel is the Tier-C edit seam.

## Pitfalls & anti-patterns
- **Causal only** — the unified kernel asserts `causal`; no non-causal path.
- `seqused_k`/`block_table` must track cached context exactly or attention is wrong (not a tie-flip).
- `num_stages>1` hurts; keep at 1.
- At large batch the split path may match/beat unified (launches amortize) — bake off.

## How to verify
Chunk-invariance test (chunked == un-chunked output); mixed-batch bench unified vs split at small/large
batch; rocprofv3 to confirm one kernel vs two; `AMDGCN_ENABLE_DUMP=1` ISA check; greedy temp=0 parity.

## Alternatives / cross-links
[aiter.md](aiter.md) · [sglang_kernels.md](sglang_kernels.md) · [vllm_kernels.md](vllm_kernels.md) ·
`languages/triton_amd/` · [[../../attention_decode_paged/backends/triton.md]] · [[../overview.md]].

## Sources
- `unified_attention` (on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/unified_attention.py`).
- vLLM `TRITON_ATTN` unified attention: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py
- Triton AMD knobs: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
