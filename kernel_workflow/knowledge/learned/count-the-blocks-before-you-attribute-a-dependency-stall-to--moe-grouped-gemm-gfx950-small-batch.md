---
key: small-token grouped MoE GEMM on gfx950 that profiles as dependency-stalled at occupancy 2, where the stall is mistaken for grid underfill
type: anti-pattern
confidence: ★★
effect: 0.85x vs frozen baseline; on the smallest token case the cost grew monotonically with the split factor (~5x, ~7x, ~10x the unsplit case at K=2,3,4) while staying numerically exact; the unsplit incumbent held at 1.47x
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: count-the-blocks-before-you-attribute-a-dependency-stall-to--moe-grouped-gemm-gfx950-small-batch
description: Closed axis: split-K/KBatch on a dep-stall-bound grouped MoE GEMM whose grid already spans ~20 block-waves regressed monotonically (0.85x)
keywords: ['moe', 'grouped-gemm', 'split-k', 'grid-occupancy', 'dep-stall', 'anti-pattern', 'composable-kernel']
kernels: ['moe_stage1']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: small-batch
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-07-29
origin_kernels: ['moe_stage1']
---
# Count the blocks before you attribute a dependency stall to grid underfill
- lever: Before spending a round on split-K/KBatch, compute the real block count — (sorted token-id length / block_m) x N-tiles — and compare it to the CU count.
- apply: Here that arithmetic gave roughly 20 block-waves over the CUs even on the smallest case, so the premise 'about one block per CU' was false and the ~56% dependency-wait was per-block latency at occupancy 2.
- verify: If you try it anyway, check parity (it stays exact) and then look for the added fixed cost: a padded fp32 partial buffer zeroed per call, a cross-partial reduction that grows with the split factor, and a deferred activation pass.
- pitfall: Split-K plumbed correctly and bit-exact yet slower at every split factor -> the fixed reduction/partial-buffer cost dwarfs a fused GEMM that is already latency-bound, not throughput-bound -> the axis closes on measurement, not on a bug.
- caution: Also verify which regime you are in: this closure was measured where the grid was already full and occupancy was capped by accumulator VGPRs — a genuinely tall-K, block-starved shape is a different situation and worth re-measuring.
- source: run moe_stage1-ch16h (16h per-kernel time-budget campaign), 2026-07-29
