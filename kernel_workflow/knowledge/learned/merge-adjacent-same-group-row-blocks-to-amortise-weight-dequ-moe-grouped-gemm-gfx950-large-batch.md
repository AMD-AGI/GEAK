---
key: moe grouped gemm · gfx950 · large-batch
type: lever
confidence: ★★
effect: Director-verified per-case 1.39x on the smallest shape and 2.03-2.16x on the two large shapes (geomean 1.83x) for the patch it anchors. Within that patch the merge is what broke the plateau on the large shapes (~1.23x) after BLOCK_N/BLOCK_K growth had already flattened; on the smallest shape every merged configuration measured 6-30% SLOWER than un-merged.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 6
toolchain: triton 3.6.0 / torch 2.11.0 / gfx950
last_seen: 2026-08-08
---
# Merge adjacent same-group row-blocks to amortise weight dequant
- lever: In a grouped/MoE GEMM the per-row-block group-id table caps the natural tile height at that block size; fusing two ADJACENT row-blocks carrying the SAME id into one double-height tile halves weight unpack/dequant work per output element - which BLOCK_N and BLOCK_K cannot do, since the mfma-per-dequantised-element ratio scales with tile height alone.
- apply: Read both row-block ids, reduce 'same id' in-kernel, take the merged tile when they match and fall back to the single-block tile when they do not (a few percent of pairs mismatch at a few hundred groups); gate the merged path on the grid still filling the device.
- verify: Cross-process full-benchmark repeats per shape (in-process interleaved medians biased the small-shape delta by about the size of the effect being hunted, and one sign flipped when promoted to a real edit), plus the compiled artifact for spill and waves/SIMD, since doubling tile height doubles the accumulator VGPR.
- caution: Also verify the packed-weight read is already K-major and coalesced - the two shipped in one patch here, so the merge alone may not carry it. Also verify at the small end and beyond merge factor 2: factor 4 lost at zero spill in every configuration tried, factor 3 did not compile, and forcing the merged path on the small shape lost at every (merge factor, BLOCK_N) point.
- source: run kernel_20_geak_0808_4h 2026-08-08
