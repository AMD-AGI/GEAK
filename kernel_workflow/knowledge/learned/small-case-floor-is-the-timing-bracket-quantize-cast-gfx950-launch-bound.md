---
key: tiny-shape case of an elementwise fp8 quantize/cast on gfx950 whose per-call floor comes from the harness timing bracket rather than from GPU dispatch
type: anti-pattern
confidence: ★★
effect: The smallest case stayed at 1.19-1.25x while the two larger cases reached 3.10x and 3.23x, capping the geomean near 2.3x. Graph capture/replay: ~12% pure-throughput on that case but +0.3% (noise) on the scored geomean, and the larger cases regress 4-7% so it only survives behind a size gate. Persistent grid-stride: 0 gain, faster-or-equal never beating the incumbent launch on any case.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-11
name: small-case-floor-is-the-timing-bracket-quantize-cast-gfx950-launch-bound
description: A tiny-shape case can be floored by the measurement bracket itself; graph replay and persistent grids buy pure throughput but ~0 scored, and cost the big cases.
keywords: ['launch-overhead', 'closed-axis', 'measurement-floor', 'hip-graph', 'persistent-kernel', 'small-batch', 'quantize-cast']
kernels: ['_per_token_group_quant_fp8']
platforms: ['gfx950']
kernel_class: quantize_cast
regime: launch-bound
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-11
---
# small-case-floor-is-the-timing-bracket
- lever: Separate the two floors before spending a round: host-side per-call cost inside the timing bracket is not removable by anything on the device, so a dispatch-side fix reads as a lever and scores as noise.
- apply: Bound the floor empirically first — hold the device work constant and vary only the launch path; if the per-call time does not move, the bracket owns it, and the weighted geomean ceiling it implies can be computed before any patch is written.
- verify: Score every candidate through the same bracket the gate uses, not a warm inner loop, and A/B each shape class separately; a size-gated candidate has to be re-timed on the shapes it is gated off.
- pitfall: A ~12% pure-throughput improvement on the small case read as a real lever -> the scored path re-enters the host bracket per call so replay never removes it -> net movement was within noise, and the same change regressed the large cases.
- caution: Also verify whether the same floor exists in the deployment path: a harness-side per-call cost may be absent in production, in which case the pure-throughput result is worth revisiting there rather than here.
- source: GEAK 16h per-kernel time-budget campaign, quantize/cast lane, waves 2 and 3, 2026-08-11
