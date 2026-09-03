---
key: fp8 A8W8 block-scaled linear GEMM, Triton on a gfx950/MI355X partition, compute-bound after de-scaling
type: lever
confidence: ★★
effect: total 20.20x isolated geomean vs frozen baseline, non-overlapping; per-case 18.1x on the smallest-M case and 21.4x / 21.3x on the two larger-M cases
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-12
name: de-scale-the-fp8-gemm-k-loop-then-feed-the-native-non-scaled-quantized-gemm-gfx950-compute-bound
description: Fold and hoist block scales out of the fp8 GEMM K-loop until the inner loop is a plain non-scaled MFMA: 20.2x per-case stacked on gfx950
keywords: ['fp8', 'block-scale', 'quantized-gemm', 'mfma', 'dequant-hoist', 'k-loop', 'l2-swizzle', 'hip-graph', 'gfx950']
kernels: ['gemm_a8w8_blockscale']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: compute-bound
layer: learned
lifecycle: active
origin_kernels: ['gemm_a8w8_blockscale']
---
# De-scale the fp8 GEMM K-loop, then feed the native non-scaled MFMA
- lever: Move every block-scale operation out of the accumulate loop so the K-loop reduces to the widest native non-scaled fp8 MFMA, and apply the scales once in fp32 on the epilogue accumulator.
- apply: Two folds compose: when GROUP_N equals BLOCK_SIZE_N the per-column scale collapses into the per-row scale bit-exactly; the remainder becomes a full-K mean scale hoisted above the loop. Emit v_mfma_f32_16x16x128_f8f6f4 (widest-K, nonkdim 16), keep the K-group structure only as MFMA scheduling, and capture the wrapper in a HIP graph.
- stack: total 20.20x isolated geomean = five directions compounded
  - 1. native non-scaled fp8 MFMA, scales applied in fp32 outside the loop -> 10.69x (the big lever)
  - 2. L2-reorder GROUP_SIZE_M 1->64 plus a .ca cache modifier, co-dependent and bit-exact -> 12.77x
  - 3. wrapper HIP-graph capture/replay, fixed tensors, no host allocation in the captured region -> 15.1x
  - 4. per-column scale folded into the per-row scale, bit-exact -> 16.70x
  - 5. full-K mean-scale hoist, zero scale ops left in the K-loop -> 20.20x
  - note: attribution is incremental in landing order, not independent
- verify: Read the compiled ISA and confirm the non-scaled MFMA opcode is what the loop issues and that no scale arithmetic survives inside it, then re-time against the frozen baseline and check oracle parity on every case.
- pitfall: Scale cost read as an in-loop FMA -> the real residual was the per-row scale GATHER, not the multiply -> hoisting the gather (not just the multiply) is what banked the last +10.6%.
- caution: The mean-scale hoist is only bit-exact under the fold condition; on shapes where GROUP_N differs from BLOCK_SIZE_N, also verify oracle parity per case before crediting the win, and also verify the scaled-MFMA variant separately since it re-introduces a scale feed into a loop you just made scale-free.
- source: run gemm_a8w8_blockscale-ch16h, 2026-08-12, 16h per-kernel time-budget campaign on gfx950
