---
key: two-stage fp8 block-scaled MoE grouped GEMM on gfx950 where up-proj and down-proj are separately routable to generated pipeline instances
type: lever
confidence: ★★
effect: total 1.29x weighted vs frozen baseline, per-case 1.23x / 1.30x / 1.31x from small to large batch — the win is carried almost entirely by one direction (1.27x standalone) and grows with batch
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 24
toolchain: unknown
last_seen: 2026-08-11
name: moe-blockscale-stage2-pipeline-variant-route-moe-grouped-gemm-gfx950-both
description: Routing the down-projection stage to the 32x32-MFMA pipeline variant carries a two-stage block-scaled MoE GEMM win; three later knobs add ~2%.
keywords: ['moe', 'grouped-gemm', 'fp8', 'blockscale', 'mfma', 'tile-config', 'cshuffle', 'instance-generation', 'gfx950']
kernels: ['moe_gemm_fp8_blockscale']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: archived
---
# moe-blockscale-stage2-pipeline-variant-route
- lever: route the two stages of a fused MoE GEMM to DIFFERENT generated pipeline instances — the down-projection stage preferred a lower-version pipeline with 32x32 MFMA while the up-projection kept its tile, and the converged config was one shared tile/route (256x64, block-M 64) rather than per-bucket tuning
- apply: instance-generation is the reachable seam: add the variant to the generator, then re-dispatch each stage independently; per-bucket routing tables were measured and did not beat the single converged route
- stack: total 1.29x weighted (director-verified) = four directions compounded; 1. stage-2 route to the 32x32-MFMA pipeline variant, all buckets — 1.27x standalone, the whole win; 2. widen stage-1 epilogue store 2->8 — +0.35% on top; 3. host-side cache of the activation-scale transpose and routing metadata — +0.27%, small-batch case only; 4. epilogue XDL M-cluster per-wave -> 1, both stages — +1.0%, bit-exact; attribution is incremental in landing order, not independent
- verify: confirm the new instance actually dispatched (the generated symbol changes) before trusting the timing, then re-time all cases against the frozen baseline — the epilogue knobs are sub-1% and drown in noise if only the geomean is read
- pitfall: an LDS bank-conflict pad flag that won on a sibling stage-only kernel of the same family transferred as exactly 0 here -> the pad only pays when the LDS access pattern is the conflicting one -> re-measure family-sibling wins instead of porting them
- caution: the epilogue and host-cache directions were each under 1% here and one was one-case-only; also verify each stacked knob separately on your shapes rather than landing the merged patch on the strength of the total
- source: run moe-blockscale-16h campaign, 2026-07-29..2026-08-11, ledger directions w1..w3
