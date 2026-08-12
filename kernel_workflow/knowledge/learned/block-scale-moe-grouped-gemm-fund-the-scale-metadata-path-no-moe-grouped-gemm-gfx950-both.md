---
key: fp8 e4m3 block-scale MoE stage-1 grouped GEMM with fused activation, Composable-Kernel template instances, gfx950/MI355 - where the win lives when the MFMA loop is latency-limited
type: lever
confidence: ★★
effect: 1.84x isolated geomean vs frozen baseline, director-verified, bit-identical outputs; per-case 1.56x at the smallest-concurrency shape and 2.01x / 1.99x at the two large ones; repeat spread under 0.6%
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: block-scale-moe-grouped-gemm-fund-the-scale-metadata-path-no-moe-grouped-gemm-gfx950-both
description: fp8 block-scale MoE grouped GEMM: pipeline-version row + scale-metadata staging (gather packing, LDS slabs, batched prologue fill) compounded to 1.84x
keywords: ['moe', 'grouped-gemm', 'block-scale', 'fp8', 'lds-staging', 'pipeline-version', 'xcd-swizzle', 'epilogue', 'prologue', 'composable-kernel']
kernels: ['kernel_moe_gemm', 'ck_moe_stage1_gemm']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: archived
cost: L2
verified_on: 2026-08-12
---
# Block-scale MoE grouped GEMM: fund the scale-metadata path, not the MFMA loop
- lever: Try the pipeline-version row of the template instance first, then attack the per-block scale metadata path - gather packing, LDS staging of the scale slabs, batching the prologue fill - before tuning the MFMA loop.
- apply: Every mechanism is a small edit in the gridwise/common headers plus one token on the instance row; keep each behind its own default-on macro so an ablation is a rebuild rather than a revert.
- stack: total 1.84x isolated geomean (director-verified) = five mechanism groups compounded
  - 1. pipeline version V3 -> V1 on the same tile row - 1.33x standalone - halves LDS, removes the register spill, -27% per-wave time at unchanged occupancy; the cheapest lever in the run
  - 2. XCD-contiguous block-to-tile remap - +8.7% on top of (1) - dispatch-order locality leaked ~9% on every per-case shape, not only the small one
  - 3. scale-gather KPack=2, then A- and B-scale LDS slabs - +7.2% then +7.7% - in-loop scale traffic goes to exactly zero
  - 4. epilogue store width widened to the template's own vector constant + transcendental-free activation - +6.8% in the once-per-workgroup tail
  - 5. batched prologue scale fill, 9 serialised round trips down to 3 - +2.75%, uniform per-case
  - note: attribution is incremental in landing order, not independent
- verify: Interleave control/candidate/control/candidate against the frozen baseline in one sitting, require the correctness cos-diff to sit at the baseline's own value, and re-time a fresh pristine copy the same day to show no box drift.
- pitfall: hand-merging three individually verified directions lost the round -> the directions interact -> re-measure the best single direction alone as a same-sitting control and merge only single-hunk diffs
  - pitfall: five verified results were booked as zero while a full perf report already sat in the build dir -> a lane reported a lock timeout as a measurement -> read the build report json before believing a zero; the cumulative understated the run by ~10% for five rounds
- caution: Also verify the scale-block constants before planning a tile sweep: when the scale block equals the K (or N) tile, that tile dimension is welded shut and only M can move.
- source: run moe_stage1-own16h, 2026-08-12 - 13-round campaign, director-validated at geomean 1.8431x, correctness pass
