---
key: the scheduling family - prefetch, scalar hoisting, predication removal, per-tensor cache policy - on an already-tuned Triton paged attention on gfx950
type: anti-pattern
confidence: ★★
effect: five scheduling arms at or below 1.00x across fourteen late rounds: reordering the scalar prologue yields byte-identical device assembly on all five case configs; issuing the page gather early costs 3.7% on the narrow-head case and 1.8-2.5% on the mid case; hoisting the descale scalars costs 2.0%; removing an always-true tail predicate costs 6.8-7.5% on the 1-trip narrow-head case even though its assembly shrank 893->734 instructions; a mixed per-tensor cache policy lands at 0.78-0.87 of the diagonal arm
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 8
toolchain: unknown
last_seen: 2026-08-17
name: only-adding-or-removing-a-dependency-moves-a-tuned-paged-att-attention-decode-gfx950-decode
description: Closed axis: on a tuned paged attention, prefetch/hoist/reorder arms return ~1.00x or worse - the ISA is byte-identical and longer KV-address liveness loses.
keywords: ['anti-pattern', 'closed-axis', 'software-prefetch', 'isa-diff', 'instruction-schedule', 'paged-attention', 'decode', 'cache-modifier', 'dependency-chain', 'triton']
kernels: ['kernel_unified_attention_3d']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
verified_on: 2026-08-17
---
# Only adding or removing a dependency moves a tuned paged attention
- lever: Once the heavy case sits near its achievable bandwidth, price any direction phrased as re-scheduling with a static assembly diff before booking a measurement session, and spend the round on arms that add or remove work instead - cutting the arity of a per-lane division in the split-K reduce dispatch was the one late arm that paid.
- apply: Build each arm into its own compiler cache directory, strip comments and location lines, and diff the device assembly per case config; identical streams mean the arm is a no-op by construction and costs no GPU time to reject.
- verify: Paired median ratios, arms interleaved inside one lock session against the frozen baseline; for a claim under about 0.3% take two blocks with the arm order swapped - one three-session block read as a wash on an effect that pooled to +0.17% at n=118.
- pitfall: a prefetch or double-buffer arm regressed with no register spill -> there is no latency to hide, and the longer-lived KV address chain costs more than the overlap buys -> arms that shorten address liveness pay, arms that lengthen it lose
a per-tensor cache policy looked like two independent levers -> both streams contend for the same L1 sets, so mixed arms land at the worse uniform arm rather than between them -> treat it as one lever plus a way to reach the worse arm
removing an always-true predicate looked free -> the scalar branch replacing it doubles the branch count and acts as a mid-chain scheduling barrier -> the predicated load is the cheaper form on this backend
- caution: Also verify per-arm compiler cache isolation when arms are selected by module-level flags rather than kernel arguments: a shared JIT cache can serve one arm's binary to another and the mix-up is silent.
- source: GEAK 12h per-kernel time-budget campaign, run mi355x_vllm_triton_unified_attention-bmk7-12h, 2026-08-17, rounds 6-19 dead-end ledger, director-validated accepted, correctness PASS
