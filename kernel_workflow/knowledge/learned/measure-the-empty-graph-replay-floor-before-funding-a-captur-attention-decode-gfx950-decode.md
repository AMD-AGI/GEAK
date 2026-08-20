---
key: small-batch paged decode attention on gfx950 whose per-call cost is already at a two-dispatch hardware floor, considered for host-side graph capture
type: anti-pattern
confidence: ★★
effect: closed axis over two independent arms: launcher-level capture 0.80x on all 8 small-batch cases, wrapper-level capture 0.61-0.70x; the measured empty-graph replay floor on this box is longer than the entire baseline of every one of those cases, so no capture placement can win
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-12
name: measure-the-empty-graph-replay-floor-before-funding-a-captur-attention-decode-gfx950-decode
description: Closed axis: graph capture at launcher and wrapper level both lose on a launch-floored decode kernel - the replay floor exceeds the whole op.
keywords: ['decode', 'paged-attention', 'launch-overhead', 'graph-capture', 'small-batch', 'dispatch-collapse', 'anti-pattern']
kernels: ['paged_attention_ll4mi_QKV_mfma16_kernel']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
origin_kernels: ['paged_attention_large']
---
# Measure the empty-graph replay floor before funding a capture round
- lever: before spending a direction on graph capture, time an empty capture-and-replay on the same box and compare it to the op's own per-call baseline as a ratio; if replay costs more than the op, the axis is decided in minutes instead of two rounds
- apply: one throwaway harness that captures nothing and replays it; the same number then settles launcher-level and wrapper-level placement together, since wrapper level adds strictly more per-call work than launcher level
- verify: compare the replay floor against the per-call baseline of the smallest shape in the suite; on this run wrapper level lost about twice as hard as launcher level, which is the signature of the floor rather than of a bad implementation
- pitfall: the first capture arm was read as an implementation problem and re-issued at a different level seven rounds later -> the cost is in the replay mechanism, not the placement -> measure the floor once and reuse the number for every placement
- caution: also verify this on kernels whose per-call cost is well above the replay floor - the conclusion here is a ratio between the two, not a property of capture; on a launch-floored decode kernel the dispatch collapse that did pay lived inside the kernel
- source: run paged_attention_large-own16h, 2026-08-12, rounds 2 and 9, director-validated run (accepted, correctness PASS)
