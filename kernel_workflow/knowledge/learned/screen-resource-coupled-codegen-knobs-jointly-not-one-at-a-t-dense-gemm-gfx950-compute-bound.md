---
key: one-at-a-time codegen knob sweeps on a Triton fp16 dense GEMM on gfx950 where async-copy staging depth and the K-tile spend the same LDS budget, large-M cases
type: method
confidence: ★★
effect: A pair of knobs each already filed as a dead end from single-knob screens (async-copy staging alone +0.7%/-1.6%; halving the K-tile alone -12.7%/-13.0%) measured +8.9%/+5.8% TOGETHER, carrying the round from 2.98x to 3.19x geomean (+7.2%, the largest in-language gain of the run) on the two large-M cases with the small-M case flat.
confirms_cited: 3
confirms_blind: 1
losses: 0
attempts: 14
toolchain: rocm 7.2 / triton 3.6.0 / torch 2.11.0
last_seen: 2026-08-12
name: screen-resource-coupled-codegen-knobs-jointly-not-one-at-a-t-dense-gemm-gfx950-compute-bound
description: Screen codegen knobs that share a hardware budget jointly: two each filed dead alone measured +8.9%/+5.8% together, carrying the round 2.98x -> 3.19x geomean
keywords: ['config-sweep', 'lds', 'vgpr', 'pipeline-stages', 'async-copy', 'tile-shape', 'occupancy', 'measurement-method', 'dense-gemm', 'compute-bound']
kernels: ['_gemm_a16_w16_kernel']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: compute-bound
lifecycle: active
---
# Screen resource-coupled codegen knobs jointly, not one at a time
- lever: Knobs that spend the SAME hardware budget (LDS bytes, VGPRs, pipeline stages) are not separable: one alone can only overspend or underuse the budget, so a one-at-a-time sweep reports both as losses. Re-screen the pairs that share a resource before declaring a knob dead.
- apply: Group the knob list by the resource each one consumes, then sweep the small cross-product within a group (e.g. staging depth x K-tile x unroll at fixed LDS) rather than one axis at a time; write the resource arithmetic down first so the affordable combinations are enumerable instead of guessed.
- verify: Screen the cross-product on compiler-reported spills/LDS before timing anything (initialise the compiled-kernel handles first, or the spill counters read as zero and screen nothing), then A/B only the survivors and check the pair beats both singletons.
- pitfall: two knobs both recorded as dead ends -> screened one at a time, each alone either overspends or underuses the shared LDS/VGPR budget -> re-screen the resource-sharing pair before the axis is closed; the pair was the round's largest gain.
- caution: Also verify a ratio you are optimising toward is actually monotone with time before using it as the objective - here a config that halved the LDS-read-per-MFMA ratio ran at 0.51x, so the ratio was a proxy that inverted.
- source: run kb_on_0810 2026-08-10
