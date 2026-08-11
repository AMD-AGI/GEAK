---
name: price-the-cross-block-fence-before-fusing-a-split-reduction--attention-gfx950-decode
description: Price the cross-block fence before fusing a split reduction into its producer: the fence cost ~20-30x the decode dispatch it removes, a 7x regression
keywords: [fence, dispatch-floor, launch-overhead, decode, attention, control-experiment]
kernels: [_fwd_grouped_kernel_stage1]
platforms: [gfx950]
kernel_class: attention
regime: decode
key: fusing a split-KV reduce dispatch back into its attention decode producer on gfx950, where ~1024 co-resident blocks need cross-block partial visibility
lifecycle: active
type: anti-pattern
confidence: ★★
effect: Fusing the split-KV reduction into the producer's epilogue to delete one dispatch was budgeted at 1.3x and returned actual 0, verdict dead_end. The correct form needs a device-scope threadfence for cross-block partial visibility across ~1024 co-resident blocks: measured at ~20-30x the dispatch it removes, a 7x regression, because the agent-scope fence's L2 writeback serialises the grid. The fence-free variant (atomics plus inline reduce) still cost ~1.8x that dispatch. Independently, on the largest decode case the reduce dispatch was already ~95% launch overhead and only ~22% of the producer's device time, so nothing inside it was worth attacking either, and narrowing its partial-output dtype measured exactly 1.0; the two smaller decode cases were already host/second-dispatch overhead-bound.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 2
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.73h / 17 passes, 2026-08-11
last_seen: 2026-08-11
---
# Price the cross-block fence before fusing a split reduction back into its producer
- lever: When a two-dispatch split/reduce structure looks like an obvious fusion target, size the synchronisation the fusion would require before writing it: with a grid of ~1000 co-resident blocks a device-scope fence forces an L2 writeback that serialises the grid and can cost an order of magnitude more than the dispatch launch it saves. Compare, on paper, the fence-free lower bound (atomic accumulate plus inline reduce) against the measured cost of the dispatch being removed — if that lower bound already exceeds it, the two-dispatch shape is welded and the axis can be closed for the rest of the campaign. A second dispatch that profiles as ~95% launch overhead is a launch-count problem, not a body problem.
- apply: Measure the standalone cost of the reduce dispatch first, then the cheapest legal synchronisation for the fused form.
- verify: Confirm the block count really is co-resident (grid vs occupancy) — the fence cost scales with that, not with the nominal grid.
- pitfall: a fusion budgeted at 1.3x came back at actual 0 -> the cheapest correct cross-block fence cost more than an order of magnitude over the dispatch it deleted -> price the fence, not the dispatch, before funding the round.
- caution: Also verify whether a host-side launch collapse addresses the same overhead more cheaply than fusion.
- source: chuschen 16h time-budget campaign run, 15.73h / 17 passes, 2026-08-11
