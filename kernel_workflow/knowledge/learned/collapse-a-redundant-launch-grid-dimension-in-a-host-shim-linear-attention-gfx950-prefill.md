---
key: chunked linear-attention forward (K-Kt chunk product) on gfx950, launched by a Triton harness whose grid carries a batch dim each program ignores
type: lever
confidence: ★★
effect: cumulative 5.7x -> 12.1x on the frozen-baseline isolated A/B; per-case: largest-batch case ~2.8x, mid case ~1.5x, tiny case unchanged (already few workgroups); bit-identical output
confirms_cited: 2
confirms_blind: 0
losses: 0
attempts: 3
toolchain: unknown
last_seen: 2026-08-12
name: collapse-a-redundant-launch-grid-dimension-in-a-host-shim-linear-attention-gfx950-prefill
description: Host-shim collapse of a redundant batch dim in the launch grid kills ~98% empty workgroups on chunked linear attention: ~2.8x on the largest case
keywords: ['launch-overhead', 'grid-collapse', 'host-shim', 'linear-attention', 'varlen', 'empty-workgroups']
kernels: ['chunk_scaled_dot_kkt_fwd_kernel']
platforms: ['gfx950']
kernel_class: linear_attention
regime: prefill
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-11
roofline: dispatch/overhead-bound -> memory-bound; empirical roofline fraction 0.13 -> 0.63 over the campaign
---
# collapse a redundant launch-grid dimension in a host shim
- lever: when the caller launches grid dim = B*H but each program's result depends on only one of the B slots, rewrite the grid in a host-side launcher shim so that dim becomes H and remap program ids
- apply: wrap the entry point in a small launcher object that recomputes the grid; an in-kernel early-return guard on the redundant index is the cheap first version and banks part of the win before the shim supersedes it
- verify: check the launched workgroup product actually fell, confirm bit-identical output against the frozen baseline, then re-time the isolated A/B per case rather than on the geomean
- pitfall: the guard-only version still dispatches the redundant workgroups -> the cost is the dispatch flood, not the arithmetic -> move the collapse into the launch grid itself
- caution: the bound class flips once the flood is gone, so also verify a fresh profile before choosing the next direction; the pre-collapse roofline no longer describes the kernel
- source: 16h single-kernel time-budget campaign (48 passes), 2026-08-11; directions r1_d0 and r2_d0, both Director-verified
