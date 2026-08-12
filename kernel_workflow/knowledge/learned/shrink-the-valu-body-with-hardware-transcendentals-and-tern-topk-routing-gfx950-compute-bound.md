---
name: shrink-the-valu-body-with-hardware-transcendentals-and-tern-topk-routing-gfx950-compute-bound
description: Once a small-k routing body turns VALU-issue-bound, hand-roll softmax on hardware exp/rcp and coax med3/min3/max3 into the selection network: +9.7% geomean
keywords: ['isa-check', 'topk', 'valu-bound', 'compute-bound', 'inline-asm', 'convert-layout', 'cross-lane', 'bit-exact', 'measurement-method', 'interleaved-ab']
kernels: ['_topk_forward']
platforms: ['gfx950']
kernel_class: topk_routing
regime: compute-bound
key: the arithmetic body of a small-k top-k / routing op on gfx950/CDNA4 after the host launch path has already been collapsed, i.e. once the largest grid is VALU-issue-bound and the smaller grids sit on the harness dispatch floor
layer: learned
type: lever
levers: [compute.valu-shrink, compute.transcendental, mem.convert-layout]
cost: L3
lifecycle: active
confidence: ★★
effect: +9.7% geomean over an already 1.87x state, to 2.06x director-verified vs the frozen baseline (per-case 2.23x / 2.25x / 1.69x from smallest to largest grid). VALU per wave -18.7%, total issued instructions -20%, VMEM -30%; device time -27% / -12% / -9%. Output bit-identical to the oracle (indices and packed words exact, values max-rel 0).
roofline: still compute/VALU-issue-bound but now on ONE case only - the largest grid is at ~85% of the VALU issue roof on median device time (~97% on its minimum), while VMEM is ~2% and LDS ~0.1% of their roofs; the two smaller grids crossed under the dispatch floor and their wall no longer tracks device time.
verified_on: 2026-08-12
last_seen: 2026-08-12
confirms_cited: 0
confirms_blind: 0
losses: 0
attempts: 0
---
# Shrink the VALU body with hardware transcendentals and ternary min/max
- lever: When a routing/select op is issue-bound on VALU, the epilogue math is worth a round even though it is a small share of the source. Two things pay: (a) hand-roll the softmax over the k winners - one shared reciprocal via inline asm instead of k divides, and exp2 with log2(e) folded into the operand scale (cut that region to under half its ops); (b) rewrite the selection compare-exchanges so the compiler emits the ternary v_med3/v_min3/v_max3 forms.
- apply: Canonical ternary pattern is umax(umin(umax(x,y),z), umin(x,y)); rotate the operand pairing per triple so no sub-min/sub-max is CSE-shared with a neighbour. Price edits without compiling with a weighted op model - selection VALU = 4*sort4_ops + 6*merge_ops when the in-lane sort runs at 4 elements/lane and the cross-lane merge costs a butterfly per level. Also join a per-word epilogue into ONE [BLOCK_M, W] tile so the word axis becomes a store axis rather than a cross-lane axis.
- stack: total 2.06x director-verified vs the frozen baseline = four directions compounded over two rounds
  - 1. host launch-path collapse (armed launcher) - 1.49x (round 1, verified) - pays only on the launch-bound grids
  - 2. small-k selection restated on a chunk axis - 1.01x standalone (round 1, verified) but supplies the whole 1.49x -> 1.87x jump once (1) removed the host floor
  - 3. hand-rolled softmax + fused word-axis epilogue store - +7.7% on top of (1,2) (round 2, verified)
  - 4. ternary med3/min3/max3 + V-shaped half-cleaner in the selection network - +1.9% on top of (1,2) (round 2, verified); integrating (3)+(4) landed +9.5%
  - note: attribution is incremental in landing order and (3)+(4) were super-additive when kept in disjoint file lanes; (2)'s standalone wall contribution was never isolatable above the then-current host floor.
- verify: Static ISA census as a pre-filter (static VALU matched the per-wave VALU counter exactly here, and a -3.3% census delta tracked a -3.2% device delta), but count ds_*/s_barrier/store ops alongside VALU - the epilogue win censused as +1 VALU and an instruction-count-only gate would have discarded it. Confirm the dispatch count per timed iteration is unchanged in a kernel trace, then adjudicate on the one case still above its floor with order-flipped ABBA paired medians plus a control arm re-measuring the frozen baseline.
- pitfall: a census-selected key transform scored -2 VALU and produced wrong results -> the all-32-bit sign-flip form let the low half pick up the sign bits before the index was OR-ed in -> run the oracle on every censused variant, not only the one you intend to ship.
- pitfall: a rejoin of a split tile compiled and returned silently wrong indices -> tl.split/tl.join permutation is not linear order (reshaping to (M,2,2) then splitting yields elements 0,2 and 1,3) -> only the exact inverse rejoin is safe.
- pitfall: the same algorithm censused 6 VALU heavier than expected -> a sub-max shared between two neighbouring triples was CSE-ed and silently degraded the ternary lowering to four plain ops -> rotate the operand pairing per triple and re-census.
- pitfall: engineer-authored patches applied to nothing -> the working tree lived under a gitignored path so a repo diff came out empty -> produce patches with a plain unified diff using a/ b/ labels and gate them with an apply --check from the canonical tree.
- caution: Also verify the per-call floor IN SITU every round (same source with an early return at the top of the body, same launcher and harness) rather than inheriting an earlier probe - a shipped win here pushed a mid-size grid under the floor, after which its wall stopped tracking device time entirely and any candidate scored on it measures the harness. Also verify that a region declared closed was closed by MECHANISM and not by area: the epilogue returned exactly 0 to a loop-folding attempt and then paid via a layout rewrite of the same code.
- caution: Also verify a hardware-transcendental rewrite against the parity gate rather than assuming it risks it - the ~1-ulp fp32 exp and reciprocal rounded to the same bf16 as the reference over the whole output here, so the numerics budget may be larger than assumed.
- source: run kernel_20_geak_0811_2h_kb_new 2026-08-12
