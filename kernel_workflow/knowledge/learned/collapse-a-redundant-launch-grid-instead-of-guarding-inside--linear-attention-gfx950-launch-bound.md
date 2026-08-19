---
key: varlen chunked linear-attention forward whose caller grid spans batch x heads while the kernel guards to the diagonal, gfx950 / Triton harness launch path
type: lever
confidence: ★★
effect: 2.76x on the largest case from the grid collapse alone; cumulative geomean 5.73x -> 12.12x over the two steps, bit-identical, and the bottleneck class flips from dispatch-overhead to memory
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-12
name: collapse-a-redundant-launch-grid-instead-of-guarding-inside--linear-attention-gfx950-launch-bound
description: A caller grid whose kernel guards to the diagonal still dispatches ~98% empty workgroups; a host shim collapsing that dim was the largest single win
keywords: ['launch-overhead', 'grid-collapse', 'host-shim', 'empty-workgroups', 'varlen', 'linear-attention', 'gfx950', 'triton']
kernels: ['chunk_scaled_dot_kkt_fwd_kernel']
platforms: ['gfx950']
kernel_class: linear_attention
regime: launch-bound
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
roofline: dispatch/overhead-bound -> memory-bound once the empty workgroups stop being launched
levers: ['host.launch-overhead', 'host.grid-geometry']
origin_kernels: ['chunk_scaled_dot_kkt_fwd_kernel']
---
# Collapse a redundant launch grid instead of guarding inside the kernel
- lever: If a varlen op's caller builds a quadratic grid and the kernel's first statement is an index guard that returns, the guarded dispatches are still launched: wrap the launch in a host-side shim that rewrites that grid dimension to the set of indices that survive the guard.
- apply: A callable shim around the entry point that recomputes the grid tuple and remaps the surviving index inside the kernel; the kernel body is unchanged, so parity is structural.
- stack: total 12.12x isolated (accepted) = two directions compounded
  - 1. in-kernel index guard turning the quadratic recompute into linear work - 5.73x standalone (round 1, verified)
  - 2. host shim collapsing the same dimension in the launch grid - 2.11x on top of (1) (round 2, verified) - it pays only because (1) proved the extra dispatches compute nothing
- verify: Compare the launched workgroup count against the count that clears the guard, then A/B against the frozen baseline; parity should be bit-identical since no arithmetic changed.
- pitfall: The in-kernel guard alone looked like the whole win -> the guarded workgroups still cost dispatch -> the remaining gain only appeared once the grid itself shrank on the host side.
- caution: Also verify the collapsed grid still oversubscribes the CUs: this run separately measured that pushing the workgroup count further down costs the store pipe its latency hiding.
- source: run chunk_scaled_dot_kkt_fwd_kernel-ch16h, 2026-08-12 (16h per-kernel time-budget campaign, rounds 1-2 algorithm + host_runtime directions)
