---
key: row-block merging in a packed-weight (GPTQ/AWQ) fused-MoE grouped GEMM on gfx950, where the group-id table caps tile height, large-batch shapes
type: lever
confidence: ★
effect: Director-verified per-case 1.39x on the smallest shape and 2.03-2.16x on the two large shapes (geomean 1.83x) for the patch it anchors. Within that patch the merge is what broke the plateau on the large shapes (~1.23x) after BLOCK_N/BLOCK_K growth had already flattened; on the smallest shape every merged configuration measured 6-30% SLOWER than un-merged.
confirms_cited: 0
confirms_blind: 1
losses: 3
attempts: 9
toolchain: triton 3.6.0 / torch 2.11.0 / gfx950
last_seen: 2026-08-11
name: merge-adjacent-same-group-row-blocks-to-amortise-weight-dequ-moe-grouped-gemm-gfx950-large-batch
description: Merge adjacent same-group row-blocks into a double-height tile to amortise weight dequant: ~1.23x on large-batch MoE shapes, slower on the smallest
keywords: ['dequant', 'tile-shape', 'moe', 'operand-reuse', 'vgpr', 'large-batch', 'interleaved-ab']
kernels: ['fused_moe_kernel_gptq_awq']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: large-batch
lifecycle: active
---
# Merge adjacent same-group row-blocks to amortise weight dequant
- lever: In a grouped/MoE GEMM the per-row-block group-id table caps the natural tile height at that block size; fusing two ADJACENT row-blocks carrying the SAME id into one double-height tile halves weight unpack/dequant work per output element - which BLOCK_N and BLOCK_K cannot do, since the mfma-per-dequantised-element ratio scales with tile height alone.
- apply: Read both row-block ids, reduce 'same id' in-kernel, take the merged tile when they match and fall back to the single-block tile when they do not (a few percent of pairs mismatch at a few hundred groups); gate the merged path on the grid still filling the device.
- verify: Cross-process full-benchmark repeats per shape, plus the compiled artifact for spill and waves/SIMD, since doubling tile height doubles the accumulator VGPR.
- pitfall: the small-shape delta came out the wrong size and one sign flipped when promoted to a real edit -> in-process interleaved medians biased it by about the size of the effect being hunted -> score it from cross-process full-benchmark repeats instead.
- caution: Also verify the packed-weight read is already K-major and coalesced - the two shipped in one patch here, so the merge alone may not carry it. Also verify at the small end and beyond merge factor 2: factor 4 lost at zero spill in every configuration tried, factor 3 did not compile, and forcing the merged path on the small shape lost at every (merge factor, BLOCK_N) point.
- source: run kernel_20_geak_0808_4h 2026-08-08
- caution: cited 9 time(s) with 3 non-improving outcome(s) as of 2026-08-11 - also verify it engages on your shapes before spending a round on it.
