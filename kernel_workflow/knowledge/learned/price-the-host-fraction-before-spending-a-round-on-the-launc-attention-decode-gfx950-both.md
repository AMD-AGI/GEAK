---
key: launch and host-side overhead on a paged attention kernel on gfx950 whose per-call GPU work is several times the host plus two-dispatch cost
type: anti-pattern
confidence: ★★
effect: closed axis over two directions: wrapper graph capture 0.986x weighted geomean (net regression - helps one short decode-shaped case about +3% and costs about 2% on the seven others), host memoization 1.009x, inside the harness noise band
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-12
name: price-the-host-fraction-before-spending-a-round-on-the-launc-attention-decode-gfx950-both
description: GPU-bound paged attention: wrapper-level HIP-graph capture and host marshalling memoization both land at or below 1.00x - the launch/host axis is closed.
keywords: ['anti-pattern', 'closed-axis', 'launch-overhead', 'hip-graph', 'graph-replay', 'host-runtime', 'attention-decode', 'paged-kv', 'gfx950']
kernels: ['paged_attention_ragged']
platforms: ['gfx950']
kernel_class: attention_decode
regime: both
layer: learned
lifecycle: archived
cost: L2
verified_on: 2026-08-12
levers: ['host.launch-overhead', 'host.graph-capture']
---
# Price the host fraction before spending a round on the launch floor
- lever: Before planning a launch-overhead round, express per-call GPU work as a multiple of host plus dispatch cost; here GPU work was several times larger, and both host-side directions returned about 1.00x even though each was implemented correctly.
- apply: Capture both launches of the split-KV pair into one graph at the wrapper level (not the launcher) and replay it; separately, memoize the per-call host marshalling.
- verify: Require graph-off to reproduce the baseline exactly so the capture is not the variable, then ask whether the metric window (per-call GPU events) can see host time at all before believing either sign.
- pitfall: Capture was real and correct and collapsed both dispatches, yet geomean fell below 1 -> the async queue already hides host and dispatch cost behind GPU work, and a GPU-event window measures none of it -> measure the host fraction first.
A per-shape never-regress gate was added to rescue it -> the true replay-vs-eager delta of about 1-2% sits under the harness noise band of about 1.2% -> when a lever's true delta is below the noise floor, no per-shape selection gate converts it into a reliable win.
- caution: Also verify whether the wrapper allocates per call: a sibling paged decode kernel whose wrapper did gained a large multi-x win from caching those allocations, so 'the host is invisible' is conditioned on there being no wrapper-side allocation to remove.
- source: run paged_attention_ragged-ch16h, 16h time-scaling campaign, 2026-08-12, directions r1_d1 (host memoization) and r2_d0 (wrapper graph capture)
