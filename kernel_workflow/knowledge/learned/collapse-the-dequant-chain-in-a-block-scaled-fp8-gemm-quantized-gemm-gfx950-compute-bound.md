---
key: per-1x128 block-scaled fp8 (a8w8) GEMM in Triton on gfx950/MI355X, dequant-chain bound, arbitrary fp32 scales (not E8M0)
type: lever
confidence: ★★
effect: ~1.53x incremental over an already-tuned seed, non-overlapping on the frozen-baseline A/B and reproduced by a confirmatory re-verify (~1.53x again); per-case cumulative 24.3x at the smallest M, 23.8x and 23.9x at the two larger M shapes; bit-exact on every config
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-12
name: collapse-the-dequant-chain-in-a-block-scaled-fp8-gemm-quantized-gemm-gfx950-compute-bound
description: Block-scaled fp8 GEMM, gfx950: hw-cvt upcast + rank-1 scale collapse + 2-deep dot overlap lift a dequant-bound inner loop ~1.53x over a tuned seed
keywords: ['fp8', 'block-scale', 'dequant', 'quantized-gemm', 'mfma', 'ilp', 'unroll', 'triton']
kernels: ['_gemm_a8w8_blockscale_kernel']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: compute-bound
layer: learned
lifecycle: active
---
# Collapse the dequant chain in a block-scaled fp8 GEMM
- lever: Three co-levers on the dequant path, not on occupancy: (a) hardware-cvt fp8 upcast (fnuz->OCP) instead of VALU-emulated conversion, which is the real few-percent-of-peak wall; (b) when the per-1x128 B scale is uniform across the N tile, collapse it to a scalar and dequantize as a rank-1 per-row FMA instead of a full [M,N] outer product every K iteration; (c) unroll K by 2 into two independent dots so the second MFMA issues while the first tile's dequant occupies the VALU.
- apply: Triton body rewrite; fold the fnuz->OCP 0.25 factor into the accumulator (exact for finite positive-normal fp8). The overlap holds only with exactly 2 dots, num_stages=2 and EVEN_K.
- stack: total ~1.53x incremental over the tuned seed = the rank-1 scale collapse and the 2-deep overlap unroll landed together in ONE direction and were never isolated from each other; the hw-cvt upcast is an earlier, separately measured win already inside the seed the 1.53x is taken over.
- verify: Grep the compiled ISA for the hardware cvt ops and for two interleaved MFMA blocks, then re-time every case against the frozen baseline and check outputs stay bit-exact after the accumulator fold.
- pitfall: A direction reported no improvement while the workspace source md5 equalled the seed -> the patch had not applied at all -> only trust an improved/not-improved verdict after grep-confirming the patch marker and config delta inside the verify workspace.
- caution: Also verify the rank-1 index is valid for your tile — the collapse assumes the N tile equals the scale group, so a smaller-than-group N tile indexes the wrong scale silently; and also verify LDS headroom before deepening the unroll past 2, which overflowed and never compiled at a 128x128 tile.
- source: run _gemm_a8w8_blockscale_kernel-ch16h, 2026-08-12 — 16h time-scaling campaign, deep_explore directions r1_d0 / r2_d0
