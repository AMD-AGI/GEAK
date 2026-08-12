---
key: VALU-issue and MFMA-dependency-bound Triton GEMM on gfx950 where occupancy, hand pipelining and host-side replay are all spent lanes
type: anti-pattern
confidence: ★★
effect: 5 lanes, 0 wins against the seed: hand-pipelined K-block prefetch ~10% slower (loads only) and ~20% slower (loads+cvt) on every case via an occupancy 3->2 collapse at VGPR 154->202 with zero spill; no-transpose B re-layout ~25% slower at occupancy 2; host graph replay regressed all three cases; occupancy 4->5 unreachable
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 6
toolchain: unknown
last_seen: 2026-08-11
name: occupancy-and-manual-pipelining-dead-on-issue-bound-gemm-quantized-gemm-gfx950-compute-bound
description: Issue/dep-chain-bound Triton GEMM on gfx950: manual K-block pipelining, occupancy raising, B re-layout and host graph replay all lost - 5 lanes, 0 wins
keywords: ['occupancy', 'software-pipelining', 'num-stages', 'vgpr-pressure', 'launch-overhead', 'hip-graph', 'lds-tiling', 'quantized-gemm', 'gfx950', 'anti-pattern']
kernels: ['_w8a8_triton_block_scaled_mm']
platforms: ['gfx950']
kernel_class: quantized_gemm
regime: compute-bound
layer: learned
lifecycle: active
---
# occupancy-and-manual-pipelining-dead-on-issue-bound-gemm
- lever: Before spending a round on occupancy or hand-written software pipelining, classify the bound: when the profile says VALU-issue then MFMA-dep-chain, more waves cannot raise per-SIMD issue rate, and the compiler's num_stages pipeliner is already prefetching the raw loads across the outer loop.
- apply: Price a manual prefetch in registers first: compare the seed's VGPR count against the architecture's headroom for its current occupancy tier (here roughly 16 registers of slack) - if the loop-carried prefetch state does not fit, the restructure buys scheduling and pays an occupancy tier.
- verify: Read the occupancy and VGPR count out of the static AMDGCN of the candidate, and check for spill separately: the whole loss here was a tier drop at zero spill, which a spill-only check reports as clean.
- pitfall: pitfall: a scheduler-opaque inline-asm sequence read as a free win -> it breaks the auto-pipeliner it sits inside -> it measured ~10% slower than the plain form.
pitfall: host-side graph replay was expected to remove a launch floor -> the async queue was already saturated and the enqueue cost is negligible against the device time -> no case improved.
- caution: All of this is conditioned on an issue/dep-chain-bound profile at occupancy 3 with no spill; also verify your own bound class and register headroom, since the same lanes are live on a genuinely latency- or occupancy-limited shape.
- source: 16h single-kernel time-budget campaign, chuschen16h wave, rounds 1-4 dead-end lanes, 2026-08-11
