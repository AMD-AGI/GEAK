---
key: occupancy tuning on a register-tight fp16 dense GEMM at gfx950 where the backend pins waves-per-eu
type: anti-pattern
confidence: ★★
effect: 0 of 5 occupancy directions beat the incumbent on any case: occupancy-1 register double-buffer ~0.5x (VGPR 304), 8-wave workgroups cost ~80% (2 workgroups/CU collapse to 1), coarsening to 3 tiles 1.00x, tile-shrink and 32x32 MFMA both <1.0x, BM=512 ~0.1x from accumulator spill. All three shapes agreed.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-11
name: occupancy-axis-closed-under-frozen-waves-per-eu-dense-gemm-gfx950-prefill
description: On compute-bound fp16 GEMM at gfx950 the occupancy axis is closed: every occ-raising/lowering variant lost, some catastrophically.
keywords: ['dense-gemm', 'occupancy', 'waves-per-eu', 'num-warps', 'ping-pong', 'register-double-buffer', 'anti-pattern', 'gfx950', 'fp16']
kernels: ['_gemm_a16_w16_kernel']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: prefill
layer: learned
lifecycle: archived
---
# occupancy-axis-closed-under-frozen-waves-per-eu
- lever: Spend at most one round on occupancy here: the AMD backend emits a fixed waves-per-eu at the end of make_llir, so no in-body primitive changes it, and the shape wants the occupancy it already has.
- apply: If you do probe it: num_warps is overridable from a launcher wrapper even when the harness passes a default, so that is the seam to test 8 waves with; the waves-per-eu pin itself is only reachable harness-side.
- verify: Read back the emitted waves-per-eu and the achieved workgroups/CU before attributing any delta to the occupancy change, and re-time the largest-M case, which is the one that self-warms.
- pitfall: Intra-workgroup ping-pong at 8 waves produced accumulator corruption as well as a slowdown -> two waves interleaving into one accumulator -> the direction was closed rather than debugged, since even a correct version starts ~80% behind.
- caution: Also verify the arithmetic intensity first: this shape is low-intensity enough that inter-tile overlap across 2 workgroups/CU is what hides load latency, so a higher-intensity GEMM may well score the opposite way.
- source: 16h per-kernel time-budget campaign chuschen16h, 44 passes, 2026-08-11
