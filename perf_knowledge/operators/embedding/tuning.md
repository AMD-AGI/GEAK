---
title: embedding — tuning
kind: technique
operator: embedding
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py
---

# embedding — tuning

## TL;DR
There is almost nothing to tune: token embedding is a bandwidth-bound row gather that is **already**
optimal as `torch.embedding` (rocPRIM/library gather of `d·dtype`-byte rows). The realistic levers are
(1) make the vocab-parallel mask/zero-fill **fuse** under `torch.compile`, (2) avoid the all-reduce when
TP=1, and (3) keep the weight in the right dtype so the gather is HBM-bandwidth-limited, not
conversion-limited. Do **not** write a bespoke gather kernel hoping for a win — there is no Amdahl room.

## The (small) lever set
1. **Let Inductor fuse the mask path.** vLLM decorates `get_masked_input_and_mask` with `@torch.compile`
   precisely so the remap + `valid_offset` add + `vocab_mask` compare + `masked_fill_(0)` collapse into a
   single elementwise kernel instead of 4–5 separate launches. Verify with `TORCH_LOGS=+inductor` /
   rocprofv3: one fused kernel, not a chain of `masked_fill`/`add` rows.
2. **Skip the all-reduce at TP=1.** The `tensor_model_parallel_all_reduce` only runs when `tp_size>1`;
   at TP=1 the masked path is bypassed entirely (`forward` returns the raw gather). Confirm you are not
   paying a 1-rank "all-reduce" no-op.
3. **Coalesced rows.** `d` is a multiple of the 128-byte cache line for hidden ∈ {4096, 5120, 8192}, so
   each token row is naturally coalesced; nothing to do. Tiny `d` (rare) would underfill the line.
4. **Grid fill (only if hand-authoring).** A custom HIP/Triton gather must launch ≥1024 workgroups
   across 304 CUs — but with `T`≤16k rows and one row per program you rarely hit that, so the library
   gather (grid-strided) is the safe choice. See [[gather_scatter]].

## What NOT to do
- Don't tune `BLOCK`/`num_warps` for a gather you didn't write — the library kernel already saturates HBM
  for this access pattern.
- Don't fuse the gather into the first [[rmsnorm]]: the gather is ≪1% and the fusion saves nothing
  measurable while complicating the graph.
- Don't fp32-upcast the embedding weight on load just for "accuracy" — it doubles the 1–4 GB read with no
  task-quality benefit (the lookup is a copy).

## Cross-links
[overview.md](overview.md) · [numerics.md](numerics.md) · [fusion.md](fusion.md) ·
[[gather_scatter]] · [[lm_head_logits]] (the weight read that *does* cost time) · [[allreduce]].

## Sources
- vLLM masked-gather `@torch.compile` fusion + TP>1 gate:
  https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py
- Coalesced reads / ≥1024 grid for memory-bound ops:
  https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
