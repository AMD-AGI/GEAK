---
name: submit-a-single-dispatch-copy-through-the-raw-driver-module--unmatched-gfx950-launch-bound
description: Submit a tiny single-dispatch copy through the raw driver module-launch instead of graph capture: 2.613x cumulative against 1.322x for capture/replay
keywords: [launch-overhead, dispatch-floor, host-launch, launch-bound, graph-capture, kernel-cache, memory-movement, measurement-method]
kernels: [write_req_to_token_pool_triton]
platforms: [gfx950]
kernel_class: memory_movement
regime: launch-bound
key: host submission path for a tiny paged-KV copy dispatch on gfx950, graded per call by an event-pair harness across batch 2/32/64 with no body-bound case
lifecycle: active
type: lever
confidence: ★★
effect: Raw single-kernel ctypes hipModuleLaunchKernel with pre-packed kernelParams measured 2.613x cumulative and was the whole win; in-file HIP-graph capture/replay on the same kernel measured only 1.322x, and flattening the replay bracket to minimal Python added <1% (1.2814x). Best accepted pass 2.8051x; per-case end state 2.921x / 2.603x / 2.345x at batch 2 / 32 / 64. Bottleneck stayed 'overhead' throughout; the residual is driver dispatch plus the 2-event timer, whose empty bracket alone caps the axis at ~4.7x, so this reached ~55-63% of that ceiling.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 6
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: run chuschen16h 2026-08-11
last_seen: 2026-08-11
---
# Submit a single-dispatch copy through the raw driver module-launch, not through graph capture
- lever: When per-call time on a tiny dispatch is dominated by submission rather than the body, treat graph capture as a middle rung rather than the destination: a single kernel has nothing to batch, so replay still pays a graph-launch entry, while packing the argument buffer once and calling the driver's module-launch entry point directly removes signature binding and option packing too.
- apply: Reach for it when the graded case count is small, the argument set is stable across calls, and the device body is already coalesced.
- verify: Price the ceiling first from an empty timed bracket, so you know what fraction of it a submission lever can even address - here the timer bracket alone bounded the axis at ~4.7x.
- caution: Also verify how much of that ceiling is left before funding another round on the same axis - this lever reached ~55-63% of it, and the next rung down is the driver itself.
- source: run chuschen16h 2026-08-11
