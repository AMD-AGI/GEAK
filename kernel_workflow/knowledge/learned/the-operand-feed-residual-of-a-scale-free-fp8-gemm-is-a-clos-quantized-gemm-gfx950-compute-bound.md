---
key: operand-feed residual of a scale-free fp8 GEMM on gfx950, where occupancy and latency-hiding rounds stop paying
type: anti-pattern
confidence: ★★
effect: ~1.00x or worse across five directions on all three per-case shapes: num_stages=3 does not build, a 256x64 tile measures 0.64x, nonkdim 16->32 measures 0.916x, VGPR shave and a register-resident LDS-bypass rewrite both land at ~1.00x; the geomean sat unchanged over six re-measurements
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 6
toolchain: unknown
last_seen: 2026-08-12
name: the-operand-feed-residual-of-a-scale-free-fp8-gemm-is-a-clos-quantized-gemm-gfx950-compute-bound
description: Once the fp8 GEMM K-loop is scale-free, the latency-hiding axes (num_stages, bigger tiles, nonkdim 32, VGPR shave, LDS bypass) all return <=1.0x on gfx950
keywords: ['fp8', 'quantized-gemm', 'mfma', 'occupancy', 'lds-tiling', 'num-stages', 'closed-axis', 'gfx950']
kernels: ['gemm_a8w8_blockscale']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: compute-bound
layer: learned
lifecycle: active
origin_kernels: ['gemm_a8w8_blockscale']
---
# The operand-feed residual of a scale-free fp8 GEMM is a closed axis
- lever: Price the occupancy and pipelining walls arithmetically before spending a round on them: both are computable from the tile before any candidate is authored.
- apply: Compute accumulator VGPR/lane for the tile (an fp32 128x128 accumulator over eight waves is ~32 VGPR/lane on its own; the kernel sat at 88, giving ~5 waves/SIMD and ~2 workgroups/CU) and check unroll*num_stages<=5 against the 160KB/CU LDS budget. If both bind, the round is better spent on a different mechanism.
- verify: Re-time every candidate against the frozen baseline rather than against the previous champion, and treat an LDS-overflow build failure as the wall itself, not as a config typo to work around.
- pitfall: MFMAs issue back-to-back yet stall repeatedly -> the current-iteration LDS read feeding each MFMA is gated by lgkmcnt and is inherently serial with its consumer, so it cannot be prefetched -> even a hand-scheduled register-resident rewrite that bypasses LDS entirely failed to beat the ceiling.
- caution: These closures were measured with the K-loop already scale-free and occupancy pinned at ~2 workgroups/CU; on a different tile, dtype or accumulator width the walls may not bind, so also verify the VGPR and LDS arithmetic for your own shape before reusing the conclusion.
- source: run gemm_a8w8_blockscale-ch16h, 2026-08-12, 16h per-kernel time-budget campaign on gfx950
