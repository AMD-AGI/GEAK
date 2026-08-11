---
name: count-the-block-waves-before-spending-a-round-on-split-k-moe-grouped-gemm-gfx950-mixed
description: Count block-waves before funding split-K on a token-sorted MoE grouped GEMM: the grid was already ~20 waves deep and split-K measured 0.85x, not 1.62x
keywords: [split-k, grid-geometry, occupancy, moe, control-experiment, measurement-method]
kernels: [fused_moe_kernel]
platforms: [gfx950]
kernel_class: moe_grouped_gemm
regime: mixed
key: split-K / grid-fill levers on a token-sorted fused-MoE grouped GEMM (Triton) on gfx950, smallest token-count case
lifecycle: active
type: anti-pattern
confidence: ★★
effect: Falsified: expected 1.62x, measured 0.851x on the smallest token-count case. Split-K was plumbed and numerically exact (err_ratio 0.0000, cos_diff 8.4e-08) yet that case regressed monotonically with the split factor — ~5.1x / ~7.4x / ~9.6x SLOWER than unsplit at SplitK 2 / 3 / 4. The premise 'the small case is grid-underfilled at ~1 block per CU' was false at block level: sorted_ids ~32760 / block_m 64 = ~512 M-blocks x 12 N-tiles = ~6144 blocks over 304 CUs = ~20 block-waves, already well filled. The ~56% dep_wait was per-block dependency-stall latency at occupancy 2, an axis split-K cannot touch, while the split added a ~32760x1536 fp32 partial-buffer memset per call plus a cross-partial reduction and a deferred epilogue pass — fixed overhead worth roughly 4x the fused GEMM's own time. The axis was recorded CLOSED; the incumbent it failed to beat was the 1.4655x geomean carried in state (per-case 1.2955 / 1.5469 / 1.5754 at the small / mid / large token counts), and the best single pass of the 24-pass campaign reached 1.4865x.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 1
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.70h / 24 passes, 2026-08-11
last_seen: 2026-08-11
---
# Count the block-waves before spending a round on split-K
- lever: When a grouped GEMM shows a high dependency-wait share on its smallest case, derive the actual block count before attributing it to an underfilled grid: for a token-sorted grouped launch that is (padded sorted-token count / block_m) x N-tiles, compared against the CU count. Several block-waves of occupancy means grid-fill levers (split-K, KBatch, stream-K) have no headroom to recover and only add partial-buffer zeroing, a reduction pass and a deferred epilogue on top of a fused one.
- verify: Re-read a surviving dependency-wait share as a per-block chain-latency number rather than a supply number, and note that bit-exactness of the split proves the plumbing, not the premise.
- pitfall: split-K landed bit-exact yet the small case got monotonically slower with the split factor -> the grid was already ~20 block-waves deep, so the stall was per-block chain latency at occupancy 2 and the split only added fixed partial-buffer, reduction and epilogue cost -> count blocks from (sorted tokens / block_m) x N-tiles against CU count before opening the axis.
- caution: The split was probed on the smallest token count only, which is the geomean-limiting case; the larger cases were not re-measured under split-K, so the closure is argued from the block-count arithmetic rather than measured on them. Also verify where the residual latency lives before re-opening the axis on a larger case.
- source: chuschen 16h time-budget campaign run, 15.70h / 24 passes, 2026-08-11
