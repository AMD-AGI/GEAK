---
key: a tiny-shape case inside a weighted per-case score on gfx950, where per-call host timing dominates the measured window and caps the achievable geomean
type: anti-pattern
confidence: ★★
effect: the smallest case stayed at 1.19-1.25x while the two large cases reached 3.10x and 3.23x; graph capture/replay bought ~12% pure throughput on that case but +0.3% scored (noise) and regressed the large cases 4-7%; persistent grid-stride was faster-or-equal never faster on it
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-12
name: a-stuck-tiny-case-may-be-floored-by-the-timing-bracket-not-t-method-gfx950-launch-bound
description: A tiny case that will not move can be floored by the measurement bracket itself, not by the GPU; graph capture and persistent grids both scored ~1.00x there
keywords: ['launch-bound', 'launch-overhead', 'closed-axis', 'small-batch', 'hip-graph', 'persistent-grid', 'quantize-cast', 'measurement']
kernels: ['_per_token_group_quant_fp8']
platforms: ['gfx950']
kernel_class: method
regime: launch-bound
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-07-30
---
# A stuck tiny case may be floored by the timing bracket, not the GPU
- lever: before spending rounds on dispatch levers for a small shape, measure it twice - once inside the scored bracket and once as pure back-to-back throughput; if the two disagree, the floor is host-side per-call overhead in the bracket and it bounds the best geomean the weighting can ever reach, which is worth knowing at plan time.
- apply: a graph capture/replay wrapper is still worth one try, but size-gate it to the small shape and treat the scored delta, not the throughput delta, as the result.
- verify: compare the same candidate under identical warmup with a few repetitions per config, and check the large cases separately - a small-shape win here came with a regression on both large ones.
- pitfall: a persistent grid-stride rewrite was bit-exact yet never beat the plain one-program-per-row launch → the small case was not ramp-bound at all, so cutting live workgroups only removed parallelism that was hiding tiny compute while the grid-stride divmod added work → reverted, no patch.
- caution: also verify that any apparent win is not a warmup artifact of the replay path before banking it.
- source: run _per_token_group_quant_fp8-ch16h, 16h per-kernel time-budget campaign, 2026-07-30
