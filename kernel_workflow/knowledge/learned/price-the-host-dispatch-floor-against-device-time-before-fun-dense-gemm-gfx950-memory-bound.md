---
name: price-the-host-dispatch-floor-against-device-time-before-fun-dense-gemm-gfx950-memory-bound
description: Price the host dispatch floor against device time before funding device work: a 1.61x device win on a tiny dense GEMM moved the scored wall only 1.018x
keywords: [dispatch-floor, launch-overhead, host-launch, launch-bound, occupancy, vgpr, lds, measurement-method, control-experiment, dense-gemm]
kernels: [_gemm_a16_w16_kernel]
platforms: [gfx950]
kernel_class: dense_gemm
regime: memory-bound
key: tiny-dispatch fp16 dense GEMM on gfx950 whose scored wall is a torch/hipLaunchKernel + event-pair floor sitting above device time, N=2 and N=4 cases
lifecycle: active
type: anti-pattern
confidence: ★★
effect: A column-per-block rewrite fixed the real device root cause -- profiled effective bandwidth ~11% -> ~17% of peak HBM BW, device kernel time 1.61x, correctness PASS -- and moved the scored wall 0: same-session A/B vs seed measured 1.018x. The scored wall floors on torch -> hipLaunchKernel + event dispatch at roughly 2x the post-rewrite device time. Three further device axes returned exactly 1.0x: split-K to fill 256 CUs, grid-trim to the 16 blocks that do work, and an LDS shrink (group_segment 163840 -> 65536 at N=4, 32768 at N=2) that was perf-neutral because VGPR (110 at N=4, 68 at N=2), not LDS, is the occupancy limiter.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 5
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: time-budget-16h campaign run, 2026-07-30
last_seen: 2026-08-11
---
# Price the host dispatch floor against device time before funding any device-side round
- lever: On a dispatch small enough that the launch path is a comparable cost, get the device kernel time (rocprofv3) and the scored wall on the same shapes in the same session before planning device work. If device time already sits below the wall, the wall is measuring the launch path and every correct device improvement will read as zero -- a fact worth establishing in one cheap round rather than discovering after a rewrite. When the grid is mostly blocks that exit immediately, occupancy is a similarly empty axis: check how many blocks actually do work and what the occupancy limiter really is (VGPR, not LDS, here) before spending a round widening it.
- apply: One profiling round: rocprofv3 for device time and dispatch count, harness wall for the scored number, and count working vs exiting blocks from the early-exit predicate. Report both numbers so the reviewer sees the gap.
- verify: A same-session interleaved A/B of the device change against the seed on the scored metric, alongside the profiled device delta -- a large profiled win with a wall delta inside noise is the floor asserting itself, not a broken patch.
- pitfall: forcing occupancy 2 at N=2 via launch_bounds did reach occ 2 yet measured device time ~31% WORSE -> register spill, and with only 16 of 304 launched blocks doing work each already owns its own CU, so the freed slot is filled by an exit-immediately block -> count the working blocks before buying occupancy.
- caution: Also verify the gap in the other direction: where device time sits well above the wall floor, the same measurement funds the device lane instead, so this is a routing measurement rather than a reason to skip device work.
- source: time-budget-16h campaign run, 2026-07-30
