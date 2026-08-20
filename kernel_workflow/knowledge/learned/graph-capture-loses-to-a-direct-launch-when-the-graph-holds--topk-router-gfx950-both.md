---
key: graph capture around a single tiny router-select launch, dispatch-floored · gfx950 · Triton
type: anti-pattern
confidence: ★★
effect: 1.00x banked (reverted bit-identical): replay ~2x SLOWER than direct launch on every case (2 / 32 / 64 rows); both capture layers, launcher and wrapper, measured dead
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-12
name: graph-capture-loses-to-a-direct-launch-when-the-graph-holds--topk-router-gfx950-both
description: Graph capture around one tiny launch replays ~2x slower than a direct launch at both host layers — a closed axis for dispatch-bound single-kernel ops.
keywords: ['launch-overhead', 'host-runtime', 'dispatch-bound', 'graph-capture', 'triton', 'small-batch', 'moe-router', 'anti-pattern']
kernels: ['_topk_forward']
platforms: ['gfx950']
kernel_class: topk_router
regime: both
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
origin_kernels: ['_topk_forward']
---
# Graph capture loses to a direct launch when the graph holds one tiny kernel
- lever: Graph capture pays only when it amortises many launches; with a single tiny kernel the command-processor dispatch of a replay already exceeds a whole direct launch, so treat this axis as closed and spend the round elsewhere.
- apply: If tried anyway, capture at the launcher layer and at the wrapper layer and time both — they are separate paths and both were measured here.
- verify: Time replay against direct launch under the same event timer; the timer does capture the replay dispatch, so a graph that looks free is a mis-scoped measurement.
- pitfall: Graph replay expected to remove the dispatch floor → replay dispatch itself is the larger cost at this size → reverted bit-identical, no patch banked.
- caution: This was measured at one-kernel granularity; also verify the ratio again if you can batch several launches into one capture, where the amortisation argument changes sign.
- source: run _topk_forward-ch16h, 2026-08-12 (host_runtime direction, measured directly at both layers)
