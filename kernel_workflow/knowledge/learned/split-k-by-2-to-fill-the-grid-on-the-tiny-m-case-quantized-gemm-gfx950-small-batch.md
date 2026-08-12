---
key: smallest-M shape of a block-scaled fp8 (a8w8) Triton GEMM on gfx950/MI355X, where the grid underfills the device rather than the body being slow
type: lever
confidence: ★★
effect: the tiny-M case ends at 24.3x, slightly ahead of the 23.8x/23.9x of the two larger per-case shapes it used to trail; split-K depth 3 and 4 measured worse than 2 on that same case, and a narrowed N tile came out ~11% slower there
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-12
name: split-k-by-2-to-fill-the-grid-on-the-tiny-m-case-quantized-gemm-gfx950-small-batch
description: Tiny-M block-scaled fp8 GEMM, gfx950: split-K=2 doubles grid fill with a fused reduce; deeper split-K and a narrower N tile both lose
keywords: ['split-k', 'grid-fill', 'skinny-m', 'quantized-gemm', 'fp8', 'block-scale', 'tile-size', 'triton']
kernels: ['_gemm_a8w8_blockscale_kernel']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: small-batch
layer: learned
lifecycle: active
---
# Split K by 2 to fill the grid on the tiny-M case
- lever: When the small-M shape launches roughly one CTA per CU, split K by 2: CTA count doubles, the per-CTA K loop halves (48->24 iterations here), and the best body config inverts toward fewer warps and more stages.
- apply: Add a split-K axis plus a fused low-precision reduction over the partials; a torch.sum-style multi-kernel reduce costs more than the split gains.
- verify: Re-time the small-M case alone against the frozen baseline (the larger shapes are unaffected), and re-tune warps/stages after splitting rather than carrying the pre-split config over.
- pitfall: Deeper splits regressed — depth 3 and 4 lost to 2 on the tiny case, because the extra reduction work stops amortizing once the per-CTA K loop is already short.
- caution: Also verify the launch grid is recomputed when tile sizes are overridden: one config's non-split grid formula was not, so a tile-size sweep measures the wrong geometry.
- source: run _gemm_a8w8_blockscale_kernel-ch16h, 2026-08-12 — 16h time-scaling campaign, deep_explore + grid/tail directions
