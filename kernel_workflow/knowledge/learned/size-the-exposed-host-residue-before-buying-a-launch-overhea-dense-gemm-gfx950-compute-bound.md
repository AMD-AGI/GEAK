---
key: host dispatch / graph-replay lane on a bf16 dense GEMM whose smallest case still runs several times longer than one host call, gfx950
type: anti-pattern
confidence: ★★
effect: 0 gain: per-case graph replay 0.92x/0.99x/1.00x and the whole lane ceils at 1.005x; a real shape-independent 25.6% cut in per-call host cost moved the geomean by 1.0009x across all three cases
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: size-the-exposed-host-residue-before-buying-a-launch-overhea-dense-gemm-gfx950-compute-bound
description: Size exposed host residue first: with kernel time far above per-call host cost, a 25.6% host cut and graph replay each bought 0 on a bf16 dense GEMM.
keywords: ['dense-gemm', 'gfx950', 'launch-overhead', 'hip-graph', 'host-dispatch', 'anti-pattern', 'dispatch-shim', 'bf16']
kernels: ['_gemm_a16_w16_kernel']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: compute-bound
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
origin_kernels: ['_gemm_a16_w16_kernel']
---
# Size the exposed host residue before buying a launch-overhead round
- lever: Estimate residue first as (per-call host cost) / (kernel duration) on the smallest case plus the measured inter-kernel gap; when host cost is a small fraction of the kernel and the queue never starves, the exposed residue is 0 and the lane cannot pay whatever the profile suggests.
- apply: Measure per-call host cost and inter-kernel gap directly, then decompose the host cost into non-editable harness floor / framework dispatch / your own code - here only a small tail was ours, so even eliminating it entirely was bounded near 1.00x.
- verify: Re-time the full paired A/B after the host cut rather than trusting the host-side delta; the host number improved by a quarter while the geomean stayed inside the noise floor.
- pitfall: Graph capture and replay was assumed a free win and measured negative on the smallest case -> capture/replay overhead and its allocation behaviour outweigh a gap that was already under a fraction of one dispatch -> the lane was closed twice, once from each side.
Two earlier rounds each estimated a couple of dispatches' worth of exposed residue from the profile; direct measurement put it at exactly 0.
- caution: This holds where the kernel dominates the call; also verify the opposite regime (many tiny dispatches, or a queue that visibly starves) separately, where the same lane can be the whole win.
- source: run _gemm_a16_w16_kernel-own16h, 2026-08-12, kernel_workflow 16h campaign, rounds 4 and 8 host_runtime lanes
