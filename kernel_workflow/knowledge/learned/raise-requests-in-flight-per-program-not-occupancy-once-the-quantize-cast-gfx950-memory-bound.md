---
key: bf16 -> fp8 per-token-group quantize/cast on gfx950 Triton, streaming (large-footprint) cases, once occupancy is already pinned at the hardware waves-per-SIMD cap
type: lever
confidence: ★★
effect: +5.2% on the largest streaming case from the half-tile split alone, at identical tile size and identical bytes moved, bit-exact; it is the last memory-side lever inside a 4.01x director-verified geomean stack over the frozen baseline.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: raise-requests-in-flight-per-program-not-occupancy-once-the-quantize-cast-gfx950-memory-bound
description: At the wave cap, split a tile into two half-tiles in one program to raise requests-in-flight: +5.2% on the streamed case, inside a 4.01x fp8 quant-cast stack.
keywords: ['memory-bound', 'quantize-cast', 'fp8', 'tile-shape', 'ilp', 'memory-latency', 'non-temporal', 'xcd-swizzle', 'isa-check', 'occupancy']
kernels: ['_per_token_group_quant_fp8', '_per_token_group_quant_fp8_flat']
platforms: ['gfx950']
kernel_class: quantize_cast
regime: memory-bound
layer: learned
lifecycle: active
levers: [mem.requests-in-flight, mem.non-temporal, algo.xcd-swizzle]
cost: L2
verified_on: 2026-08-12
roofline: memory-bound ~0.18 of nameplate HBM before the stack -> ~0.92 / ~0.87 of nameplate on the two streaming cases after (about 1.49x an on-box bf16 device-to-device copy of the same bytes); the tiny case sits at the timing-bracket floor, not at a kernel limit
---
# requests-in-flight per program is a memory axis of its own, separate from occupancy and tile size
- lever: When a streaming kernel is memory-bound but occupancy is already clamped by the waves-per-CU
  hardware maximum, the remaining knob is how many loads ONE program has outstanding. Keep the tile size,
  the bytes, and the grid exactly as they are, and issue the tile as two independent half-tiles whose
  loads are both in flight before either is consumed.
- apply: In the jit body, replace the single `[TILE_G, W]` load/convert/store chain with two
  `[TILE_G/2, W]` loads issued back-to-back, then the two converts, then the two stores; recompute the
  row base per half so it composes with any pid remap. Gate it behind the same footprint predicate that
  already selects the streaming path, and expose it as an env knob so it can be A/B'd without a rebuild.
- stack: total 4.01x geomean, director-verified against the frozen baseline (3 cases) = six directions
  compounded; attribution is INCREMENTAL in landing order, not independent.
  - 1. host.launch-relaunch — 1.47x standalone (round 1, verified) — re-tile the frozen one-program-per-row
    grid through an exported launcher object; see reinterpret-frozen-launch-via-wrapper-object card.
  - 2. compute.narrow-dtype-emit — 1.14x standalone (round 1, verified) — native fp8 emit replacing the
    emulated convert. (1)+(2) integrated to 3.33x, i.e. SUPERLINEAR versus the 1.67x product, because each
    direction was the other's wall: alone, each is capped by the resource the other removes.
  - 3. mem.non-temporal-per-footprint — +12% on top of (1,2) (round 2, verified 3.74x cumulative).
  - 4. algo.xcd-window-swizzle — +2.5% incremental after being gated against (3) (round 2 integrate 3.75x).
  - 5. mem.requests-in-flight half-tile split — +5.2% on the largest case (round 3, verified 3.84x cumulative)
    — THIS card's lever.
  - 6. host.nullary-replay of the pre-bound launch — +2.4% on the smallest case (round 3, verified 3.88x).
  - round-3 integrate 3.96x; Director's independent re-measurement 4.01x. Two later rounds added nothing.
- verify: Dump the ISA and confirm the loads are still the same width and still carry the non-temporal
  modifier, and that BOTH appear before the first consumer — if the compiler sank one load next to its use,
  the mechanism never engaged and the measurement is meaningless. The gain must show up on the streaming
  cases specifically; a gain that appears only on the smallest case is the timing bracket, not the kernel.
- pitfall: two halves paid, four quarters were 12% worse, and splitting a doubled tile lost -> the effect is
  non-monotone in the split count, so "split more" is not the mechanism -> sweep at least THREE points; the
  middle point is what separates a small real effect from no effect.
- pitfall: the same split on the small-footprint branch was byte-exact, legal, and measured EXACTLY zero ->
  that grid fills the machine once, has no steady state, and is latency/measurement-floor bound rather than
  bandwidth bound -> gate the split on the footprint predicate instead of enabling it globally.
- pitfall: pushing the same in-flight idea to the STORE side regressed -> stores are fire-and-forget so
  there is no stall to overlap, and deferring them lengthens register live-range at the wave cap -> keep the
  transformation on the load side only.
- pitfall: enabling the non-temporal READ hint and the non-temporal WRITE hint together was worse than
  either alone, on both streaming sizes -> the two hints interact through the same cache policy -> sweep each
  hint SEPARATELY per footprint; an earlier single-point sweep had scored this whole axis at zero.
- pitfall: the XCD co-residency swizzle and the non-temporal read policy are anti-additive on the read
  stream -> both are trying to own read locality -> disable the swizzle exactly where the read is
  non-temporal (one predicate, reused) and each keeps its own case's win.
- pitfall: an occupancy sweep looked like the same axis and cost a whole round for no patch -> 4 warps were
  already at the 8-waves/SIMD hardware cap, so 8 and 16 warps have identical occupancy and identical loads in
  flight with coarser tails (-11% / -43%) -> check waves-per-CU against the cap BEFORE sweeping num_warps.
- pitfall: a plain diff stack of two independently-authored lane patches applied cleanly and would have
  scored zero -> the operator ships several shape-specialised inner kernels and the diff landed on the dead
  fallback -> hand-transplant into every live inner body and confirm the patched path actually executes.
- pitfall: a single-order read on the smallest case manufactured a phantom ~6% win that a control then
  reproduced twice -> that case's noise band is ~6% while the streaming cases sit near ~0.3% -> measure
  control-candidate-control with >=3 full runs per position before believing any small-case delta.
- caution: Also verify bit-exactness rather than tolerance after splitting — it is safe here only because
  the per-group reduction axis lies entirely inside a half and never crosses the split boundary; a layout
  where the group spans the split changes the reduction order.
- caution: Also verify the achieved fraction of nameplate before budgeting this: on the two cases already
  near the ceiling it bought the last few percent, and it has no headroom to convert on a case that is
  dispatch- or bracket-floored.
- source: /shared_nfs/zihao/kernel_agent/2608/GEAK/kernel_workflow/exp/kernel_20_geak_0811_2h_kb_new/_per_token_group_quant_fp8 (tech_lead_report.md, round_3_shift_analysis.md, director_validation.json), 2026-08-12
