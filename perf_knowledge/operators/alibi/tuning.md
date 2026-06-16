---
title: alibi — tuning
kind: technique
operator: alibi
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
updated: 2026-06-08
sources:
  - https://arxiv.org/abs/2108.12409
  - https://github.com/vllm-project/vllm/blob/main/csrc/rocm/attention.cu
---

# alibi — tuning

ALiBi has essentially **nothing to tune as a standalone op** — it's a fp32 multiply-add inside the
attention score loop. The "tuning" is at the attention-kernel level.

## 1. Compute the bias, don't materialize it
Inside the FMHA tile, `bias = −m_h·(i − j)` is a cheap per-element expression from the tile's (i, j)
indices and the head slope `m_h`. **Never materialize** a `[seq, seq]` bias tensor — compute it inline as
the score is formed. A pre-materialized bias is an HBM read of `O(seq²)` and defeats the point.

## 2. Slopes in registers/constant
The per-head slopes `m_h` (H values) fit in registers/constant memory — load once per head. They're a
fixed geometric sequence; precompute on the host or in a tiny prologue.

## 3. It rides the attention kernel's tuning
The bias add costs one FMA per score; the attention kernel's tuning (tile sizes, `num_stages=1`,
`schedule_hint="attention"`, wave64 softmax reduce) dominates. See [[attention_prefill_fmha]] /
[[languages/triton_amd/patterns]] §4. ALiBi adds no new knob.

## 4. Decode
For decode, the bias is `−m_h·(q_pos − j)` over the KV positions — a vector add to the score row before the
online softmax. Same near-zero cost.

## The only real decision: does the backend support it?
If the attention backend has an ALiBi path (Triton FMHA bias arg, CK/HIP bias), enable it. If not, you're
forced to a slower attention path or a materialized bias — that's the cost, and it's a *support* question,
not a tuning one.

## Sources
- inline bias in FMHA (no materialization): https://github.com/vllm-project/vllm/blob/main/csrc/rocm/attention.cu.
- ALiBi slopes: https://arxiv.org/abs/2108.12409.
- attention kernel tuning (where the real knobs are): perf_knowledge [[languages/triton_amd/patterns]] §4.
