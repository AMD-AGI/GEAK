---
key: tiny index-scatter / pool-write Triton op on gfx950 whose per-call host submit costs more than the kernel body
type: lever
confidence: ★★
effect: 2.613x isolated vs frozen baseline, reproduced over dozens of resumed passes (2.49x-2.81x band); per-case 2.92x at batch=2, 2.60x at batch=32, 2.35x at batch=64 - the gain is pure submit cost, so it grows as the case shrinks. Same op: graph capture/replay only 1.32x.
confirms_cited: 3
confirms_blind: 0
losses: 0
attempts: 6
toolchain: unknown
last_seen: 2026-08-12
name: raw-driver-module-launch-dispatch-bound-copy-memory-movement-gfx950-launch-bound
description: Dispatch-bound tiny memory-movement Triton kernels: replacing the Python launch wrapper with a raw ctypes driver module-launch gives ~2.6x.
keywords: ['launch-overhead', 'host-dispatch', 'latency-bound', 'memory-movement', 'scatter', 'triton', 'ctypes', 'small-batch']
kernels: ['write_req_to_token_pool_triton']
platforms: ['gfx950']
kernel_class: memory_movement
regime: launch-bound
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-11
---
# raw-driver-module-launch-dispatch-bound-copy
- lever: When the profile calls the op latency-bound with no roofline coverage, treat the host submit path itself as the optimization target: launch the compiled module directly through a ctypes driver module-launch instead of the framework's Python launch wrapper.
- apply: Warm up once: cache the module/function handle and a pre-packed kernelParams buffer; per call only rewrite the pointer/scalar slots that changed, then call the driver launch entry. Keep the original launcher as a fallback path.
- verify: Frozen-baseline isolated A/B on the smallest case first (it has the largest submit share), plus a bit-exact int64 parity check against the golden tensor rather than a tolerance compare; time an empty launch bracket to learn the reachable ceiling (here the driver-dispatch + 2-event-timer floor caps the op near 4.7x, so 2.6x is ~55-63% of what is attainable).
- pitfall: Planned gain 1.9x but measured 2.613x -> the roofline reported n/a and the bottleneck as latency-bound, which hides how much of the time is per-launch host cost -> size the empty-bracket floor before allocating rounds to kernel-body work.
- caution: Also verify the harness's own synchronization style survives the new launcher, and re-check parity on the largest case: the submit share falls with batch, so the ratio you measure on one case is not the ratio you get on another.
- source: 16h per-kernel time-budget campaign, 62 resumed passes, gfx950, 2026-08-11
