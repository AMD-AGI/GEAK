---
name: freeze-one-operand-s-pointer-at-a-time-to-put-an-upper-bound-moe-grouped-gemm-gfx950-mixed
description: Freeze one operand's pointer at a time to upper-bound every memory-side direction: capped the whole memory system at ~10% on the largest MoE GEMM case
keywords: [control-experiment, measurement-method, l2-locality, operand-reuse, moe, counters]
kernels: [fused_moe_kernel]
platforms: [gfx950]
kernel_class: moe_grouped_gemm
regime: mixed
key: per-operand memory-ceiling probes on a fused-MoE grouped GEMM (Triton) on gfx950, largest and smallest token-count cases
lifecycle: active
type: method
confidence: ★★
effect: Bounded the whole memory system at ~10% of the largest case's time with per-operand ceilings of -4.8% / -4.1% / -1.4% / 0.0%, which retired four separately-funded memory directions before any of them was built, and on the smallest case the same probe read -37% and correctly declared it compulsory-bound and finished. It also showed request count is invariant to ADDRESS: collapsing an operand's footprint to a single tile moved L2 misses 57% and L2 requests only 2.7%.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 3
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: run kb_on_0810 2026-08-11
last_seen: 2026-08-11
---
# Freeze one operand's pointer at a time to put an upper bound on every memory-side direction
- lever: Before funding work on any operand's traffic, build one arm per operand in which only that operand's pointer is frozen to a single tile (output deliberately wrong, correctness gate off) and measure wall clock. Each delta is a hard ceiling on everything reachable by improving that operand — locality, layout, prefetch, staging — and the deltas usually sum to far less than the profiler's data-wait bucket suggests.
- apply: One arm per operand plus an all-frozen arm, all from one build behind compile-time switches so nothing else moves; collect memory counters on every arm so request-vs-miss behaviour is attributed at the same time as the clock.
- verify: Include a gated arm that is bit-identical to the control and require it to measure 1.000 before believing sub-1% readings; interleave arms in one warm loop with at least six pairs, and re-measure your own control rather than quoting the reported cumulative.
- caution: Also verify what the freeze actually deleted: if requests barely move while misses collapse, the operand's cost is locality, not request count, and only removing a load SITE will change the latter — so also check whether a candidate direction changes the instruction stream at all before pricing it.
- source: run kb_on_0810 2026-08-11
