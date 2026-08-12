---
key: block-scaled fp8 (A8W8, per-group scales) dense GEMM on gfx950/MI355X, Triton — scale arithmetic living inside the K-loop
type: lever
confidence: ★★
effect: 20.1x isolated geomean vs frozen baseline, non-overlapping; per-case 18.1x / 21.4x / 21.3x over three growing-M cases (the smallest-M case gains least, it stays partly launch-limited); empirical roofline 0.25 memory-bound -> 0.60 compute-bound
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 9
toolchain: triton-on-rocm
last_seen: 2026-08-11
name: turn-a-block-scaled-fp8-gemm-inner-loop-into-an-unscaled-mfm-dense-gemm-gfx950-both
description: Fold + hoist all block-scale arithmetic out of the fp8 GEMM K-loop so the inner loop is a native unscaled MFMA loop: ~20x isolated on gfx950
keywords: ['fp8', 'block-scale', 'dense-gemm', 'mfma', 'scale-hoist', 'l2-reorder', 'hip-graph', 'gfx950']
kernels: ['gemm_a8w8_blockscale', '_gemm_a8w8_blockscale_kernel']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: both
layer: learned
lifecycle: archived
---
# Turn a block-scaled fp8 GEMM inner loop into an unscaled MFMA loop
- lever: Get every scale operation out of the K-loop so the accumulation loop is a plain unscaled fp8 MFMA loop: first fold the per-column scale into the per-row scale when the scale group width equals the N tile (bit-exact), then hoist a full-K mean scale so zero scale ops remain inside the loop, applying scales in fp32 on the epilogue side.
- apply: Inner op = the widest-K native unscaled fp8 MFMA form the arch offers (16x16x128 f8f6f4 here); on top of that, an L2 tile-reorder (group-M 1 -> 64) together with a caching load modifier (the two are co-dependent, neither pays alone), and a host-side graph capture/replay of the launch with fixed tensors, split-K = 1 and no host allocation inside the captured region.
- stack: total 20.1x isolated (weighted geomean, verified) = five directions compounded, incremental in landing order: 1. native unscaled fp8 MFMA with scales moved outside -> 10.7x, the big lever; 2. L2 tile-reorder + cache modifier -> 12.8x (+1.20x, bit-exact); 3. wrapper graph capture/replay -> 15.1x (+1.19x); 4. column-scale fold into the row scale -> 16.7x (+1.11x, bit-exact); 5. full-K mean-scale hoist, zero scale ops in the loop -> 20.2x (+1.11x). Attribution is incremental, not independent.
- verify: Check the compiled loop body actually contains no scale math and the intended MFMA opcode, then re-time against the frozen baseline per case; the fold and the reorder are bit-exact so parity should be exact, and the graph path should show the wrapper cost collapse only on the small-M case.
- pitfall: the harness reported no-improvement on rounds that were in fact improving -> the check ran before the accepted patch marker was present in the verify workspace -> confirm the config delta and the patch marker by grep in the verify workspace before believing either verdict, positive or negative.
- caution: The mean-scale hoist is the one step in the chain that is not bit-exact, so also verify oracle parity at the accuracy tolerance your consumer needs, and also verify the reorder and the cache modifier together rather than separately.
- source: 16h per-kernel time-budget campaign, block-scaled fp8 dense-GEMM lane, 35 passes, 2026-08-11
