---
key: deciding whether a near-null direction is closed, after a fix that changed register pressure on a block-scaled fp8 dense GEMM on gfx950
type: method
confidence: ★★
effect: hoisting the rank-1 per-column scale out of the K loop measured ~1.01x standalone before the compute fix and 1.84x incremental after it, holding on all three cases (M=2048/32768/65536)
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-12
name: re-measure-shelved-partials-after-the-bound-class-moves-method-gfx950-n-a
description: A lever shelved at ~1.01x can pay 1.84x once a bigger fix relieves register pressure: re-measure shelved partials on top of each new incumbent
keywords: ['method', 'register-pressure', 'bottleneck-shift', 'dense-gemm', 'fp8', 'block-scale', 'triton', 'gfx950']
kernels: ['_gemm_a8w8_blockscale_kernel']
platforms: ['gfx950']
kernel_class: method
regime: n/a
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-12
origin_kernels: ['_gemm_a8w8_blockscale_kernel']
---
# Re-measure shelved partials after the bound class moves
- lever: after a change that moves the bound class or frees registers, re-run the directions that previously measured near 1.00x instead of treating them as closed
- apply: keep each shelved partial as a re-appliable patch against the source file, and re-measure it on top of the current incumbent rather than standalone against the old baseline
- verify: interleaved A/B/A/B/A in one window against the new incumbent; require the gain to exceed the in-window repeatability of the largest case, and attribute incrementally in landing order
- pitfall: the scale hoist first scored ~1.01x -> AGPR moves cancelled the VALU saving while the emulation path owned the registers -> the same patch was worth 1.84x once the compute fix freed the budget
- caution: also verify the pair does not anti-stack: a host-dispatch direction that was real standalone lost to the new incumbent on every configuration it selected, and the naive stack of two winners regressed until the second was gated by shape
- source: run _gemm_a8w8_blockscale_kernel-own16h, 2026-08-12, rounds 1-3
