---
key: skipping dead work in a coarsened int4 W4A16 fused-MoE grouped GEMM on gfx950/CDNA4, Triton, small- to large-batch — the FORM the predicate takes
type: lever
confidence: ★★
effect: 2.2023x isolated geomean vs the frozen baseline, director-verified over 3 repeats with <0.6% spread, non-overlapping; per case ~1.57x on the small-batch case (2 tokens/expert) and 2.58x / 2.64x on the two large-batch cases. Matrix-op count per launched wave per K-iteration reached exactly 1.00x of ideal on every case (was 2.60x / 1.11x / 1.06x).
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: triton 3.6.0 / torch 2.11.0 / gfx950
last_seen: 2026-08-12
name: hoist-a-work-skipping-predicate-out-of-the-loop-into-constex-moe-grouped-gemm-gfx950-both
description: Skip the dead row-groups by emitting one branch-free loop body per live count behind a scalar switch, not an in-loop branch: +15% on a coarsened int4 MoE GEMM
keywords: ['compile-time-specialization', 'dequant', 'moe', 'grouped-gemm', 'int4-dequant', 'm-coarsening', 'mfma', 'software-pipelining', 'isa-check', 'lds-bank-conflict', 'gfx950']
kernels: ['fused_moe_int4_w4a16']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-12
roofline: dequant-latency-bound -> balanced; VALU ~55% of peak issue, LDS pipe ~49% busy with bank conflicts at ~19% of ALL cycles, matrix pipe ~9% but at exactly 1.00x the ideal op count; occupancy FELL to ~0.8x its previous waves/SIMD while the kernel got faster and issue-wait dropped ~6 points
levers: ['compute.dead-work-elimination']
---
# Hoist a work-skipping predicate out of the loop into constexpr-specialised bodies
- lever: Once a grouped/MoE GEMM coarsens several row-blocks onto one dequantised weight tile, the partially-filled groups keep issuing dead matrix ops and dead operand loads; delete them by emitting one branch-free copy of the shared jit body per live-group count G=1..G_max behind a single scalar switch at entry, so the dead passes are never emitted and the G_max copy stays instruction-identical to the un-predicated kernel (big shapes keep their schedule).
- apply: One @triton.jit inner body parameterised by a constexpr live count, instantiated G_max times; size the partition FIRST with a CPU replay of the run's own routing metadata — it costs minutes, and here it both found a further 22% of dequant passes (a per-segment sub-worker absorbing its own maximal same-group run, which made the partition provably optimal) and proved the level after that worthless.
- stack: total 2.20x isolated geomean (director-verified) = three directions compounded
  - 1. row-block coarsening: G separate small-M dots over ONE dequantised weight tile, contiguous-run absorption, survivor lane rotating across XCDs — 1.78x standalone (round 1, verified) — the bulk of the win
  - 2. hoisted constexpr specialisation + the sub-worker absorption above — +15% on top of (1) (round 2, verified); this is what took the small-batch case from ~1.09x to ~1.58x
  - 3. splitting the reduction into even/odd-K partial dots to delete the K-interleave cross-lane permute — +4.3% standalone vs a same-session control, +4.7% on top of (1,2) (round 2, verified), uniform across all three shapes
  - note: attribution is incremental in landing order, not independent.
- verify: count matrix ops per launched wave per K-iteration against the ideal for the shape (exactly 1.00x is both the confirmation and the sign the lever is spent), read register/spill/occupancy from the compiled object, and take the verdict from a same-session control that re-times the incumbent in the same batch.
- pitfall: the same idea written as a runtime `if` inside the K loop did delete the dead matrix ops yet measured 15-30% slower in both placements tried -> the scf.if regions defeat the software pipeliner and the operand prefetch -> hoist the predicate out of the loop and specialise the body instead.
- caution: Also re-profile what REPLACED the traffic you deleted: the cross-lane permute went to zero but a staging round-trip took its place, absolute bank-conflict cycles doubled and barriers rose ~8x — a net win here, and now the largest remaining cost. Also verify the number of specialised copies before collapsing them: 4 ways beat 2 ways by ~2% because the middle counts are common on the small-batch case.
- source: run kernel_20_geak_0811_2h_kb_new, 2 rounds / 6 directions, correctness 8/8 vs golden, 2026-08-12
