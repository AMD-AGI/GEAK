---
name: empty-the-k-loop-of-scale-arithmetic-entirely-fold-the-co-al-quantized-gemm-gfx950-compute-bound
description: Empty the K loop of scale arithmetic on a block-scaled fp8 GEMM: three folds/hoists compound +10%, +9.35%, +10.6% to a 20.20x banked geomean
keywords: [quantized-gemm, fp8, quantization-group, dequant, mfma, operand-reuse, roofline, compute-bound]
kernels: [_w8a8_triton_block_scaled_mm]
platforms: [gfx950]
kernel_class: quantized_gemm
regime: compute-bound
key: block/group-scaled fp8 GEMM in Triton on gfx950 whose K loop still carries per-group scale arithmetic after the native non-scaled fp8 MFMA dialect step
lifecycle: active
type: lever
confidence: ★★
effect: Three stacked scale-removal steps on a block-scaled fp8 GEMM whose inner loop already used the native non-scaled fp8 MFMA (that dialect step alone banked 10.686x cumulative). End state 20.20x banked geomean (best single director-validated pass reported 20.13x), per-case 18.07x / 21.41x / 21.31x; empirical roofline 0.250 -> 0.600 and the bottleneck moved memory-bound -> compute-bound.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 3
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign, 2026-08-11
last_seen: 2026-08-11
---
# Empty the K loop of scale arithmetic entirely: fold the co-aligned operand scale, then hoist a full-K mean scale
- lever: On a block/group-scaled quantized GEMM, treat the two scale tensors as a removable cost rather than a fixed one - check the alignments first, since a scale group equal to the N tile is constant over the tile and folds into the other operand's per-row scale bit-exactly, and what is left is often a per-k-group gather that a mean scale can lift out of the loop completely.
- apply: Let the residual profile decide how far to push - here the leftover was a gather rather than an FMA, so a mean scale over the k-group and then over the whole K extent removed it.
- stack: total 20.20x banked geomean = three scale-removal steps on top of the native fp8 MFMA dialect step
  - 1. fold the second operand's block scale into the per-row first-operand scale where the scale group equals the N tile - bit-exact, no epilogue change, +10% to 16.70x
  - 2. per-group mean scale at k-group depth 2 - +9.35% to 18.26x
  - 3. full-K mean-scale hoist, leaving ZERO scale ops in the K loop - +10.6% to 20.20x
  - note: attribution is incremental in landing order, and (3) superseded (2) rather than compounding with it.
- verify: Confirm bit-exactness on the fold step, and re-check that the k-group structure still earns its keep for MFMA scheduling once the scales it was built for have gone - it did here.
- caution: Also verify what a scale-free loop wants downstream: a native scaled-MFMA variant re-introduces a scale feed into a loop that no longer has one, and measured as no path here.
- source: chuschen 16h time-budget campaign, 2026-08-11
