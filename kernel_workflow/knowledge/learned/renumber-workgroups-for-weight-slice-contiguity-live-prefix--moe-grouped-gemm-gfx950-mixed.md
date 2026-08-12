---
key: workgroup renumbering for weight-slice L2 reuse in a CK/HIP templated fp8 block-scaled MoE stage-1 grouped GEMM on gfx950
type: lever
confidence: ★★
effect: director-verified 1.49x geomean end state, per-case 1.38x / 1.57x / 1.54x from the smallest to the largest token count; this axis contributed ~+11% cumulative in two steps - the live-prefix renumber paid +7.7% at the small token count, where the weight stream IS the cost, and +4.9% / +4.3% on the large ones; then halving the chunk count paid a further +2.9% on every case with an essentially byte-identical code object.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: rocm7.2 / torch2.11.0 / hip (CK C++ templates, JIT-compiled)
last_seen: 2026-08-08
name: renumber-workgroups-for-weight-slice-contiguity-live-prefix--moe-grouped-gemm-gfx950-mixed
description: Renumber workgroups into long contiguous runs over one weight slice, live prefix only: ~+11% cumulative on an MoE grouped GEMM whose cost is the weight stream
keywords: ['pid-remap', 'l2-locality', 'operand-reuse', 'moe', 'correctness-gate', 'isa-check', 'bijection', 'grid-geometry']
kernels: ['moe_stage1']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
layer: learned
lifecycle: archived
---
# Renumber workgroups for weight-slice contiguity, live prefix only
- lever: When consecutive tiles of one group/expert all stream the same large weight slice, the natural round-robin of the flat workgroup id scatters those tiles across every L2, so each cache holds a fraction of the available reuse; renumber the id so consecutive workgroups form long contiguous runs over one weight slice.
- apply: Write it as a pure index bijection in the prologue with compile-time-foldable divisors, leaving the loop body, the accumulation order and the output bytes identical so only placement changes; restrict the bijection to the LIVE prefix of the grid, derived from the already-loaded token or tile count, so ids beyond it keep their natural value and fall into the existing early exit.
- stack: total ~+11% on this axis = two directions compounded
  - 1. live-prefix renumber - +7.7% at the small token count, +4.9% / +4.3% on the large ones (verified) - the bulk of the win
  - 2. halving the chunk count (fewer, longer runs) - a further +2.9% on every case, code object essentially byte-identical (verified) - the currency is run LENGTH, not affinity to any particular cache
- verify: Gate on correctness before believing any timing, then diff the inner-loop ISA, where instruction, memory-class and register counts should all be unchanged so the entire delta is placement; a real locality win also tends to be the lowest-variance build, since the jitter source is cache contention.
- pitfall: a candidate read as +3% -> the mapping was not a bijection and silently skipped tiles (error ratio 0.22) - fast-and-wrong is the signature -> gate every remap on the golden before timing it.
- caution: Also verify chunk counts at powers of two before non-power-of-2 ones (one non-power-of-2 count measured 46% slower than its neighbour), and price ordering ideas with a bit-exact permutation of the live tiles first: that cheap screen showed a random scatter costs +90-94% and a de-interleave +21-33%, i.e. the default ordering already banks most of the prize.
- source: run kernel_20_geak_0808_4h 2026-08-08
