---
key: single-dispatch fused MoE GEMM wrapper on gfx950 where host enqueue is a small async-hidden fraction of the timed region
type: anti-pattern
confidence: ★★
effect: replay/eager 1.0292 on the smallest token case, 1.0058 and 1.0061 on the two larger ones — a loss on all three, worst where a launch win should be largest; incumbent held bit-exact at 1.47x
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: graph-replay-only-pays-if-there-is-a-launch-floor-to-collaps-moe-grouped-gemm-gfx950-small-batch
description: Closed axis: HIP-graph capture/replay at the wrapper layer lost on every case (1.006-1.029x slower) when host enqueue is already async-hidden
keywords: ['launch-overhead', 'hip-graph', 'host-runtime', 'anti-pattern', 'moe', 'grouped-gemm']
kernels: ['moe_stage1']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: small-batch
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-07-29
origin_kernels: ['moe_stage1']
---
# Graph replay only pays if there is a launch floor to collapse
- lever: Measure the host-enqueue share of the timed region first; if it is a few percent and already overlapped with GPU work, graph capture has nothing to collapse and replay enqueue costs more than eager.
- apply: Capture worked mechanically at the wrapper layer on all cases, so 'captured=True' is not evidence of a win — the A/B is.
- verify: Compare replay against eager per case rather than against the frozen baseline only, and keep the graph path default-off so the incumbent stays byte-exact.
- pitfall: Capture succeeds and the numbers still get worse -> one dispatch per call, host already hidden behind compute -> the axis closes at the wrapper layer too, not only at the launcher layer.
Allocation caching looked adjacent but was a non-lever here: the taken path writes into the output buffer, so there is no per-call allocation.
- caution: Also verify the dispatch count: a wrapper that enqueues many small kernels per call, or a decode step with a real launch floor, is the situation where this can still pay — re-measure rather than inherit this result.
- source: run moe_stage1-ch16h (16h per-kernel time-budget campaign), 2026-07-29
