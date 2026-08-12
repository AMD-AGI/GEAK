---
key: tiny int32 scatter/fill on gfx950 once the Python launch path is already bypassed - four further host-side axes were measured to the floor and none of them paid
type: anti-pattern
confidence: ★★
effect: four axes closed by measurement: wrapper graph capture/replay lost every case (0.94x at B=64, ~2.2-2.7x costlier per launch than the direct C launcher); side streams 2.1x (back-wait) to 4.9x (both-waits) worse than same-stream; a device-resident argument block cut enqueue yet moved reported time +0.3%; the fixed harness constant is one event record, invariant to iteration count, queue state, arrival rate and stream placement
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: four-host-side-axes-that-a-dispatch-bound-tiny-op-has-alread-memory-movement-gfx950-both
description: On a dispatch-bound tiny op, graph replay, side streams, launch-arg slimming and harness-constant attacks all returned <=1.00x on gfx950
keywords: ['anti-pattern', 'closed-axis', 'hip-graph', 'graph-replay', 'dispatch-bound', 'host-runtime', 'launch-overhead', 'memory-movement', 'measurement-floor', 'gfx950']
kernels: ['write_req_to_token_pool_triton']
platforms: ['gfx950']
kernel_class: memory_movement
regime: both
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
levers: ['host.launch-overhead']
---
# Four host-side axes that a dispatch-bound tiny op has already paid for
- lever: spend the round elsewhere unless a probe shows a mechanism different from these four; each was root-caused here rather than merely measured slow.
- apply: if one is re-opened, price the intercept itself rather than enqueue: graph enqueue is cheap and the launch-side latency is what loses; and price a host win against the harness knee (below a few units of host time the reported number stops moving) before investing a direction in it.
- verify: an A-vs-A null in the same comparator, plus a dispatch-count gate; a win that appears only in enqueue counters and not in the reported number has landed under the knee.
- pitfall: a launch-value reduction worked mechanically (enqueue down) yet reported +0.3% -> the whole host win sat below the harness knee -> check where the knee is before optimising further host time.
- pitfall: the same patch regressed the guard-miss path ~3.5x under fresh-metadata calls -> the benchmark reuses objects, so the regression was structurally invisible -> add a fresh-metadata micro-probe.
- caution: also verify the residual constant is candidate-controlled at all before attacking it; and also verify how much device work is left worth chasing - here the remaining device excess was worth ~+0.5% reported, under the acceptance gate.
- source: run kernel_20_geak_0808_16h, memory-movement scatter lane, rounds 1-4, 2026-08-12
