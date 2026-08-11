---
key: tiny skinny-GEMV / thin dense linear on gfx950 whose per-call wall time is dominated by the torch->hipLaunchKernel + event dispatch path rather than by the device kernel
type: anti-pattern
confidence: ★★
effect: GPU-side rewrite verified 1.61x device time and effective bandwidth ~11%->~17% of peak, yet same-session wall A/B 1.018x; per-case wall unchanged at batch 2/32/64 (all three cases inside +-3-4% clock noise). Three further device axes (split-K CU-fill, grid-trim, occupancy-2 via LDS shrink) each measured 1.00x.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-11
name: device-side-wins-are-invisible-under-a-host-dispatch-floor-dense-gemm-gfx950-decode
description: On tiny skinny-GEMV decode ops the host dispatch floor sits above GPU time: a correct 1.61x GPU-side roofline rewrite moved the scored wall 1.02x.
keywords: ['launch-overhead', 'dispatch-floor', 'skinny-m', 'gemv', 'decode', 'roofline', 'memory-latency']
kernels: ['wvSplitK']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: decode
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-11
---
# Device-side wins are invisible under a host dispatch floor
- lever: before spending rounds on device rewrites for a sub-dispatch-scale op, measure device kernel time against the scored wall; when the wall floor is above device time the profitable axes are host-side, and device axes are worth a cheap probe at most.
- apply: read the device duration from the profiler dispatch record and compare it with the harness wall per call; the gap is the floor. If device < wall, target the host path (allocation reuse, lookup caching) instead of occupancy/CU-fill.
- verify: same-session interleaved A/B of the device patch against the seed on the scored wall metric, not on profiled device time; a device-only improvement with a flat wall confirms the floor.
- pitfall: occupancy work looked mandatory from the profile -> the grid launches one block per CU but only ~16 blocks do work, so a freed occupancy slot is filled by an exit-immediately block -> LDS shrink and forced occupancy-2 both engaged (confirmed in the ELF group segment size) and both measured 1.00x or worse from register spill.
- caution: also verify whether the shape regime is genuinely below the floor: the same class at larger batch can climb above it, at which point the device axes reopen.
- source: chuschen 16h per-kernel time-budget campaign, 18 passes, 2026-08-11
