---
key: irreducible per-K-iteration dequant in a block-scaled fp8 GEMM on gfx950, medium/large shapes on the non-split-K path
type: lever
confidence: ★★
effect: 1.5263x incremental on the medium/large non-split-K path over an already 15.8x seed, bit-exact; carries cumulative to 24.08x; the large case then sits at ~91% of its per-1x128 dequant-plus-MFMA floor and ~59% of native-fp8 MFMA partition peak
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-11
name: rank-1-scale-collapse-plus-2-deep-overlap-unroll-hides-dequa-quantized-gemm-gfx950-compute-bound
description: Per-1x128 dequant that is uniform across the N tile collapses to a rank-1 per-row FMA; unrolling K by 2 overlaps the second MFMA with the dequant VALU work
keywords: ['dequant', 'block-scale', 'mfma', 'ilp', 'unroll', 'block-pingpong', 'fp8', 'quantized-gemm']
kernels: ['_gemm_a8w8_blockscale_kernel']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: compute-bound
layer: learned
lifecycle: active
verified_on: 2026-08-11
---
# Rank-1 scale collapse plus 2-deep overlap unroll hides dequant behind MFMA
- lever: Two co-levers on the dequant path: (a) when the weight scale is uniform across the N tile, load it as a scalar and apply dequant as a rank-1 per-row FMA instead of a full [M,N] outer product every K iteration; (b) unroll K by 2 so two independent dots are in flight and the ping-pong scheduler issues the second MFMA while the first block's dequant occupies the VALU.
- apply: Hoist the scale pointer to a scalar load per tile; restructure the K-loop into exactly two independent dot accumulations with num_stages=2 and an EVEN_K guard, leaving the compiler's natural register allocation alone.
- verify: Confirm both dots survive into the ISA (two MFMA groups interleaved with the scale FMA), then re-time against the frozen baseline per case; on the tiny grid-starved case expect no gain, the win is on the medium/large shapes.
- pitfall: 4-deep unroll never compiled -> LDS request exceeded the hardware limit by ~60% -> the depth that pays is exactly 2; the rank-1 scale index is also only valid while the N tile equals the scale group width, so it silently mis-indexes under a sub-group-width N tile.
- caution: The overlap depends on two dots and stage count staying paired, so also verify the win survives a config re-tune rather than assuming it composes with an independent tile change.
- source: 16h per-kernel time-budget campaign (chuschen16h lane), 2026-08-11, two deep_explore directions, confirmed twice
