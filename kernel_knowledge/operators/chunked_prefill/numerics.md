---
title: chunked_prefill — numerics
kind: operator_overview
operator: chunked_prefill
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/unified_attention.py
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://arxiv.org/abs/2205.14135
---

# chunked_prefill — numerics

## Chunking must be exact across chunk boundaries
The load-bearing correctness property: splitting a prompt's prefill into chunks must produce the **same**
attention output as a single un-chunked prefill. This works because a chunk's queries attend over **all
prior context** (the already-cached KV from earlier chunks) plus the chunk's own keys up to the causal
position — the `seqused_k` per sequence includes the cached context length. The online softmax over the
segmented KV (cached + current) is mathematically the full softmax; the chunk boundary is not a
mathematical boundary, only a scheduling one. If `seqused_k` / `block_table` mis-tracks the cached
context, you get **wrong attention** (not tie-flips) — this is the thing to gate hard.

## Accumulation
fp32 online-softmax over the segmented KV (`NUM_SEGMENTS_PER_SEQ`), per-segment `(m, ℓ)` combined with
log-sum-exp — same as flash-decoding. bf16/fp16 storage of P/O. The unified kernel asserts **causal**
(no non-causal unified path).

## Masking inside the unified kernel
- **Causal** within the chunk + against cached context (a chunk query at global position `p` attends KV
  positions `≤ p`).
- **Sliding window** (`window_size`): `SLIDING_WINDOW = 1 + window_size[0]` lower-bounds attended
  positions.
- **Softcap** (`softcap`): `softcap·tanh(S/softcap)` before mask/softmax.
- **ALiBi** (`alibi_slopes`) and **attention sinks** (`sinks`, one per query head) are supported in the
  unified kernel. Apply all of these before the row-max.

## Unified vs split must agree
The unified kernel and the split path (`context_attention_fwd` + paged decode) reduce KV differently →
bf16 argmax tie-flips on long greedy decode. Benign equivalence-class differences — gate with a
≥10-prompt greedy temp=0 parity probe, accept post-near-tie. But **un-chunked vs chunked** should agree to
reduction-order noise (a chunk-boundary bug shows as a real divergence, not a near-tie flip).

## fp8 (q/k/v_descale)
fp8 chunked attention uses `q_descale`/`k_descale`/`v_descale` applied before the fp32 softmax (FNUZ on
gfx942, OCP on gfx950 — wrong dialect off by 2×). fp8 KV is a task-accuracy gate. `output_scale` for fp8
output.

## Verify
- **Chunk-invariance**: chunked prefill output == un-chunked prefill output to reduction-order noise (the
  primary gate — catches `seqused_k`/`block_table` bugs).
- Unified vs split: greedy temp=0 parity ≥10 prompts, accept post-near-tie.
- fp8: gsm8k/eval accuracy gate; confirm dialect matches gen.

## Sources
- `unified_attention` causal assert, sliding window, softcap, alibi, sinks, fp8 descales, NUM_SEGMENTS: on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/unified_attention.py`.
- flash-decoding segmented softmax: https://arxiv.org/abs/2205.14135
- FNUZ vs OCP fp8 2× trap: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html ; `hardware/shared/dtype_numerics.md`
