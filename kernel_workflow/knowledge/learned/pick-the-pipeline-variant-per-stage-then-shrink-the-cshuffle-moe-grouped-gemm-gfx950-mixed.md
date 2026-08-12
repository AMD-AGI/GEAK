---
key: fp8 1x128 block-scaled fused-MoE grouped GEMM inside a frozen CK/aiter C++ stack on gfx950, compute-bound, tuned per stage rather than globally
type: lever
confidence: ★★
effect: 1.29x weighted isolated vs the frozen baseline, non-overlapping; per-case 1.23x at batch 2, 1.30x at batch 32, 1.31x at batch 64 — the win grows with batch
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: pick-the-pipeline-variant-per-stage-then-shrink-the-cshuffle-moe-grouped-gemm-gfx950-mixed
description: Re-route the down-proj stage to the narrow-N V1 pipeline (32x32 MFMA) and shrink the CShuffle M-cluster to 1 XDL/wave: ~1.29x on fp8 block-scaled MoE GEMM
keywords: ['moe', 'grouped-gemm', 'fp8-blockscale', 'composable-kernel', 'mfma', 'cshuffle', 'pipeline-variant', 'tile-shape', 'compute-bound']
kernels: ['moe_gemm_fp8_blockscale']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
layer: learned
lifecycle: archived
cost: L1
verified_on: 2026-07-30
roofline: compute-bound 0.37 -> 0.52 of the empirical roof
---
# Pick the pipeline variant per stage, then shrink the CShuffle cluster
- lever: The two grouped-GEMM stages of an MoE layer want different instance shapes: routing the narrow-N down-proj stage from the wide V3 pipeline to the V1 256x64 instance buys a 32x32 MFMA there, and a second, cheap pass shrinking the CShuffle per-shuffle M-cluster to one XDL per wave pays on top of it.
- apply: Emit the alternate instance from the generator script the vendor stack already ships (the hand-written instance lists are frozen and not editable); set the stage-2 route to the V1 bm64 instance for all expert buckets, then set MXDLPerWave to 1 in the CShuffle epilogue of both stages.
- stack: total ~1.29x weighted isolated (director-validated) = four directions compounded
  - 1. stage-2 pipeline variant V3->V1, 32x32 MFMA — 1.27x standalone, carries essentially the whole win
  - 2. CShuffle store vector width 2->8 on stage 1 — +0.35% on top of (1)
  - 3. host-side scale-transpose + routing-metadata cache — +0.27% on top of (1,2), and only on the smallest batch case
  - 4. CShuffle M-cluster MXDLPerWave->1, bit-exact — +1.01% on top of (1,2,3)
  - note: attribution is incremental in landing order, not independent.
- verify: Re-time against the pinned frozen baseline on every batch case, not just the largest — (3) moved only the smallest one; confirm the new instance actually got selected by disassembling and checking the MFMA shape changed, since a mis-routed bucket silently falls back to the old instance and still passes parity.
- pitfall: A stage-level tile/route change looked like a global win but only paid on one stage → the two stages have different N extents → route per stage and re-measure each; the per-stage M-tile split that this suggests is structurally impossible in this pipeline and returned 1.00x.
- caution: Also verify the epilogue change is bit-exact against the oracle before crediting it: the cluster shrink was, but the wider store width in (2) changes the write pattern and deserves its own parity check.
- source: run moe_gemm_fp8_blockscale-ch16h, 2026-07-30 — 16h per-kernel time-budget campaign, 24 passes, director-validated geomean
