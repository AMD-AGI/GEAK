---
title: embedding — fusion
kind: technique
operator: embedding
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py
---

# embedding — fusion

## TL;DR
The only fusion that actually happens (and the only one worth it) is the **vocab-parallel mask path**
collapsing into the gather under `torch.compile`. The gather is too cheap to justify fusing into the
downstream norm, and the "fusion" with [[lm_head_logits]] is a **weight-sharing** (tied weights), not a
compute fusion.

## Fusion neighbors
| neighbor | type | done? | why |
|---|---|---|---|
| remap + mask + zero-fill | elementwise → gather | ✅ via `@torch.compile` | vLLM fuses `get_masked_input_and_mask` into one kernel (avoids 4–5 launches) |
| all-reduce | collective after gather | ✅ (TP>1, always) | required by the Megatron vocab split; rides RCCL/custom-AR → [[allreduce]] |
| first [[rmsnorm]] / [[rope]] | gather → norm | ❌ not worth it | gather ≪1% of GPU time; fusing complicates the graph for ~0 gain |
| [[lm_head_logits]] | weight share | ✅ memory only | `tie_word_embeddings` → one `[V,d]` tensor, two access patterns |

## The mask fusion in detail
At TP>1, per-rank embedding needs: (1) subtract the shard's `valid_offset` to map global→local id,
(2) build `vocab_mask` for ids outside this shard's range, (3) `F.embedding` on the clamped/local ids,
(4) `masked_fill_(input_mask, 0)` so foreign rows contribute zero, (5) `all_reduce(SUM)`. Steps 1–2 and 4
are pointwise; vLLM's `@torch.compile` on `get_masked_input_and_mask` fuses them so the only HBM-heavy
events are the gather (step 3) and the all-reduce (step 5). This is the entire optimization surface.

## Tied weights (memory fusion)
With `tie_word_embeddings`, `ParallelLMHead.weight` *is* `embed_tokens.weight` — a single `[V,d]`
allocation (up to ~4 GB at V=256k, d=8k, bf16) read row-wise here and as a GEMM operand in
[[lm_head_logits]]. This halves weight memory and improves cache residency across the two ops, but they
remain separate kernels (a gather vs a GEMM). GGUF-quantized embeddings are the exception (returns the
`embed_tokens` module rather than sharing the raw weight).

## Anti-fusion (don't)
- Don't fuse gather→rmsnorm: see [tuning.md](tuning.md).
- Don't try to fuse the all-reduce into the gather; it's a collective with its own optimized path.

## Cross-links
[overview.md](overview.md) · [tuning.md](tuning.md) · [numerics.md](numerics.md) · [[allreduce]] ·
[[rmsnorm]] · [[rope]] · [[lm_head_logits]].

## Sources
- `@torch.compile` mask fusion + tie_weights + GGUF exception:
  https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py
