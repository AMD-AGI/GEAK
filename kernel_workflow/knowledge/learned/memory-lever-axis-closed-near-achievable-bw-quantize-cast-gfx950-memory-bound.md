---
key: elementwise fp8 quantize/cast already near achievable HBM on gfx950 — where further memory-side tuning stops paying
type: anti-pattern
confidence: ★★
effect: Zero gain from the whole memory axis once the two large cases sat at ~63% of nameplate bandwidth: num_warps=8 regresses, tile width up or down regresses, store cache-modifier variants and load hints tie, stage count inert, flat 1D cross-row tiling bit-exact but slower at every tile width, manual store repack ties-or-regresses; two directions returned 0 (no patch) against expectations of 2.55x and 2.4x on top of 2.29x.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-11
name: memory-lever-axis-closed-near-achievable-bw-quantize-cast-gfx950-memory-bound
description: Past ~63% of nameplate HBM on a 3x-traffic elementwise cast, every bandwidth knob measured zero: warps, tile width, cache modifiers, store vectorization.
keywords: ['hbm-ceiling', 'memory-bound', 'closed-axis', 'store-vectorization', 'cache-modifier', 'num-warps', 'quantize-cast', 'assembly-inspection']
kernels: ['_per_token_group_quant_fp8']
platforms: ['gfx950']
kernel_class: quantize_cast
regime: memory-bound
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-11
roofline: memory-bound at ~0.58 of empirical roof; residual gap is read/write turnaround plus fixed 3x traffic, not store efficiency
---
# memory-lever-axis-closed-near-achievable-bw
- lever: Treat this axis as spent rather than as unexplored: on a cast whose traffic is fixed at read-wide + write-narrow + a scalar scale per group, the distance left to nameplate is turnaround, so bandwidth knobs have nothing to convert.
- apply: Before budgeting a round on store vectorization, dump the assembly and read the store width: a contiguous output tile already lowers to a single max-width buffer store, so a manual pack can only add shift/or ops. Register and occupancy stats (no spill, max occupancy) close the resource story at the same time.
- verify: Compute achieved fraction of nameplate analytically from the traffic model and compare against a known-good prior measurement; if it is already at the practical ceiling, spend the round on the other case class instead.
- pitfall: A store-vectorization direction looked obviously open from source -> the assumption was byte-granular scatter -> the disassembly showed one max-width store already, so the direction closed with no patch and one wasted round.
- caution: Also verify the ceiling on your own shapes before reusing this: it was measured where the output tile is contiguous, and a strided or ragged group layout can reopen the store path.
- source: GEAK 16h per-kernel time-budget campaign, quantize/cast lane, waves 2 and 4, 2026-08-11
