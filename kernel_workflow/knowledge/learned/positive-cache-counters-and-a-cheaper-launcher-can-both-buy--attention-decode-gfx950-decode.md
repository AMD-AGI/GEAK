---
key: paged split-KV decode attention on gfx950 whose small-batch case is parallelism-starved rather than bandwidth-bound, with a launcher already far under the dispatch knee
type: anti-pattern
confidence: ★★
effect: ~1.00x across four probes: locality swizzle 1.5838x against a 1.5919x round entry despite batch=2 L2 hit 16.9%->66.7% and DRAM amplification 2.45x->0.98x; full-op graph capture 1.5777x with eager-vs-replay 0.740/0.789/0.801x per case; thin launcher alone 1.003x
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: positive-cache-counters-and-a-cheaper-launcher-can-both-buy--attention-decode-gfx950-decode
description: Cache-locality swizzle and host/graph-capture levers both returned ~1.00x on a decode attention op that was neither bandwidth- nor dispatch-bound.
keywords: ['attention-decode', 'paged-decode', 'launch-overhead', 'l2-locality', 'xcd-swizzle', 'cuda-graph', 'anti-pattern', 'gfx950']
kernels: ['_fwd_grouped_kernel_stage1']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-08
---
# Positive cache counters and a cheaper launcher can both buy zero clock
- lever: Classify each shape first - its fraction of achievable DRAM read rate, and its per-call host cost against the measured dispatch knee - and treat cache placement and host/dispatch as candidates only for shapes that classification says are actually held there.
- apply: Both probes are cheap: a program-id swizzle that lands sibling programs on one L2 slice, and full-op graph capture with an allocation-free fast path. Run them as probes and gate promotion on the frozen-baseline clock rather than on the counter they move.
- verify: Pair every counter improvement with an isolated A/B on the same shape; a hit-rate or amplification improvement that leaves the clock inside the noise band is a closed axis, not a partial win.
- pitfall: Counters moved hugely (L2 hit ~4x, amplification to ~1.0) while the clock stayed flat -> the small-batch case was parallelism-starved at a quarter of CU fill, not bandwidth-bound -> the round was better spent on grid shape.
Graph capture measured slower than eager -> the launcher already sat ~4x under the dispatch knee and the residual gap was the benchmark's own barrier packets, paid identically by the baseline -> compare per-call host cost to the knee before staffing this lane.
- caution: Both dead-end directions were still merged at zero measured cost for robustness (safe stream handling, kwargs guard) - also verify whether a clock-neutral direction is worth keeping for safety before discarding it.
- source: run _fwd_grouped_kernel_stage1-own16h, 2026-08-08, round-2 directions d1/d2 verified against a frozen baseline
