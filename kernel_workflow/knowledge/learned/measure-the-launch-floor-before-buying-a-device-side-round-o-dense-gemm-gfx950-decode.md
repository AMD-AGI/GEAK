---
key: bf16 skinny GEMV / decode linear on gfx950 MI355X, tiny per-call op whose scored wall sits on the host launch floor
type: anti-pattern
confidence: ★★
effect: device-side rewrite won 1.61x profiled kernel time and ~11%->~17% of peak HBM BW, yet the same-session wall A/B read 1.018x on all three decode cases (tokens=2 and tokens=4) - inside run-to-run noise
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-12
name: measure-the-launch-floor-before-buying-a-device-side-round-o-dense-gemm-gfx950-decode
description: When the harness wall floors above device time, a correct roofline rewrite of a tiny decode GEMV scores ~1.00x; measure the launch floor first.
keywords: ['dispatch-floor', 'launch-overhead', 'decode', 'skinny-m', 'gemv', 'roofline', 'cu-underfill', 'anti-pattern']
kernels: ['wvSplitK_hf_sml_']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: decode
layer: learned
lifecycle: active
origin_kernels: ['wvSplitK']
---
# Measure the launch floor before buying a device-side round on a tiny decode op
- lever: compare profiled device time against the scored wall before spending rounds on device-side levers; when the wall floor sits above device time, redirect the round to host/launch-path work
- apply: rocprofv3 with dispatch_count=1 gives device time; the harness wall geomean minus that is the launch/enqueue floor - here the host share was ~45% of the per-call wall before any change, and 100% of the remaining headroom after
- verify: same-session interleaved A/B per case against the frozen baseline; a device-time win that does not move the wall is real physics but zero score
- pitfall: a rewrite that spread streaming from ~16 active CUs across all 256 fixed the true underfill and still moved the wall 0% -> the scored metric floors on the launch path -> re-scope to host levers rather than re-exploring the device side
- caution: also verify that an occupancy lever adds ACTIVE waves: shrinking LDS per WG freed a slot but VGPR was the real limiter and the freed slot was taken by an exit-immediately block, so forced occupancy-2 measured neutral-to-worse from spill
- source: run wvSplitK-ch16h, 16h per-kernel time-budget campaign, 2026-08-12
