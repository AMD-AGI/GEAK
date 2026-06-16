---
title: embedding — numerics
kind: technique
operator: embedding
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py
---

# embedding — numerics

## TL;DR
The lookup itself is **bit-exact** (a copy, no FP math, no accumulation) — there is no tolerance band.
All correctness risk lives in the **vocab-parallel mask + all-reduce** plumbing and in **id range**
handling, not in the kernel. Parity after a backend swap is byte-exact for the gather; the only thing
that can perturb hidden states is the all-reduce reduction order at TP>1 (and even that is usually
order-stable for a one-hot-per-token sum).

## Where bits can move (and where they can't)
| concern | risk | note |
|---|---|---|
| Gather `W[ids]` | **none** — exact copy | dtype out == dtype weight; no convert |
| Vocab-parallel sum | low | each token has exactly **one** non-zero contributor → the all-reduce sums (row, 0, 0, …); FP add of zeros is exact, so order doesn't matter here |
| Mask correctness | **high if buggy** | a missed `masked_fill_(0)` on an out-of-range id lets a *wrong* local row leak through and get summed → silently corrupt hidden state, not a crash |
| OOV / id ≥ V | crash/garbage | ids must be in `[0, V)`; vocab-parallel remap assumes valid global ids — pad/unk handled upstream |
| Tied weight dtype | none | shared tensor read identically by [[lm_head_logits]] GEMM; the embedding read and the logits read see the same bytes |

## The one real gate
At TP>1, verify the masked-gather + all-reduce reproduces the single-GPU embedding **bit-for-bit** for a
random id batch spanning all shards (including ids at shard boundaries and the last partial shard). A
boundary-off-by-one in `get_masked_input_and_mask` is the classic bug: it manifests as a few tokens with
the wrong embedding, which downstream looks like a quality regression, not a crash.

## bf16 vs fp32 weight
Storing the embedding in bf16 (the serving default) is **lossless for the lookup** — the bytes you stored
are the bytes you read. There is no accuracy reason to upcast on load; doing so only doubles HBM traffic
([tuning.md](tuning.md)). fp32 master weights matter only in **training** (gradient accumulation into the
sparse rows), out of scope for inference serving.

## Cross-links
[overview.md](overview.md) · [tuning.md](tuning.md) · [[allreduce]] (reduction-order parity) ·
[[lm_head_logits]] (shares the weight) · backend [vllm_kernels.md](backends/vllm_kernels.md).

## Sources
- Masked input/mask + `masked_fill_(0)` + all-reduce (the correctness-bearing code):
  https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py
