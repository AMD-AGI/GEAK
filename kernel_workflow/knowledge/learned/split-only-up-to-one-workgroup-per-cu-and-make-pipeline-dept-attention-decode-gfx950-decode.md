---
key: bf16 paged MLA-style split-KV decode attention, Triton on gfx950/MI355X, where small batch underfills the CUs and large batch already saturates DRAM
type: lever
confidence: ★★
effect: 1.63x isolated geomean, director-verified over two independent runs agreeing within 0.03%; per case batch=2 1.65x, batch=32 1.69x, batch=64 1.54x
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: split-only-up-to-one-workgroup-per-cu-and-make-pipeline-dept-attention-decode-gfx950-decode
description: Cap the parallelism split at one workgroup per CU and make pipeline depth a function of launched WGs: 1.63x geomean on paged split-KV decode attention.
keywords: ['paged-decode', 'attention-decode', 'split-kv', 'num-stages', 'waves-per-eu', 'launch-shape', 'occupancy', 'gfx950', 'triton']
kernels: ['_fwd_grouped_kernel_stage1']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-08
roofline: per-program compute-efficiency bound -> memory-bound at ~104% (batch=64) and ~93% (batch=32) of measured achievable DRAM read rate; small batch stays parallelism-starved at ~26%
---
# Split only up to one workgroup per CU, and make pipeline depth a function of launched WGs
- lever: Cap the extra-parallelism split (bit-exact partial-output column split) at one workgroup per CU instead of a fixed multiple of CU count, and select pipeline depth from the launched WG count rather than pinning it as a constant.
- apply: Move split factor and pipeline depth into a thin host launcher that computes them per shape from (grid rows x split) versus CU count; below the one-WG-per-CU line pick the shallower pipeline, at or above it the deeper one.
- stack: total 1.63x isolated (director-verified) = three directions compounded
  - 1. per-program knobs (deeper pipeline + waves_per_eu=2, formula tail-peel, head-mask specialization) - 1.48x standalone (round 1, verified) - the bulk of the win
  - 2. bit-exact partial-output column split - +7-8% on top of (1) (round 1 integrate, verified), landing entirely on batch=2 where the grid underfills the CUs
  - 3. split cap corrected to one WG per CU + shape-dependent pipeline depth - +2.2% on top of (1,2) (round 2, verified), essentially all on batch=32
  - note: attribution is incremental in landing order, not independent; the launcher rewrite alone measured 1.003x.
- verify: Re-time every shape against the frozen baseline and sweep the split per shape: the peak sat exactly at the CU line (batch=32 split 1/2/4 = 1.63/1.69/1.59, batch=64 split 1/2 = 1.54/1.39, batch=2 split 4/8/16 = 1.60/1.66/1.62).
- pitfall: The assigned hypothesis said the largest batch wanted more split -> it over-split a bus that was already saturated and cost ~10% -> derive the cap from CU count, and note attainable bandwidth is itself a function of WG count (a gather probe held its full rate at 256 WGs but only ~55% of it at 128).
Retiling the KV block or swapping in exp2 re-rounded the online softmax -> parity gate blew up golden-vs-golden (max_rel 46.9 for one retile, 2.1e-2 for exp2) -> only math-preserving knobs (warps, pipeline depth, waves_per_eu, split) clear this gate.
- caution: Inherited backend guidance for this decode family claimed deeper pipelining hurts and single-warp launches win; here the opposite measured best and single-warp was the worst point at 0.660x - also verify pipeline depth and warp count on your own box before carrying that guidance over.
- source: run _fwd_grouped_kernel_stage1-own16h, 2026-08-08, director-validated geomean 1.6337
