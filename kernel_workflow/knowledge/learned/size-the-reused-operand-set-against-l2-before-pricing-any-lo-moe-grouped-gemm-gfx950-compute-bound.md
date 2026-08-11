---
name: size-the-reused-operand-set-against-l2-before-pricing-any-lo-moe-grouped-gemm-gfx950-compute-bound
description: Size the reused operand set against L2 first: on a weight-streaming MoE GEMM whose reuse set is ~120x L2, the whole locality family ceilings at ~1%
keywords: [l2-locality, xcd, pid-remap, operand-reuse, moe, dequant, control-experiment, compute-bound]
kernels: [fused_moe_kernel_gptq_awq]
platforms: [gfx950]
kernel_class: moe_grouped_gemm
regime: compute-bound
key: L2-locality and eviction-policy levers on a weight-streaming MoE grouped GEMM on gfx950 whose per-CU reuse set is ~120x L2 capacity, batch 2/32/64
lifecycle: active
type: anti-pattern
confidence: ★★
effect: An XCD de-interleave that gives each XCD single-expert L2 residency - the right mechanism for the round-robin dispatch on this part - measured a ~1% CEILING: batch 32 +1.2%, batch 64 neutral, batch 2 unchanged, moving the cumulative 41.9525 -> 42.2393x. It was recorded as dead_end, then overturned to partial, then CLOSED as a ceiling. Load cache_modifier / eviction_policy tuning for L2 residency was a flat dead_end at the same incumbent. The reason is scale: the streamed weight per CU is ~120x L2 capacity, so the reuse set never fits and the traffic is not pollution to be managed - and the second-wave experiment that doubled that per-CU streamed traffic cost -55%.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 3
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.57h / 35 passes, 2026-08-11
last_seen: 2026-08-11
---
# Size the reused operand set against L2 before pricing any locality or eviction-policy reorder
- lever: Locality levers - XCD/pid swizzles, cache_modifier, eviction policy, NT hints - pay when the reused operand set is near L2 capacity and cache lines are being thrown away early. Compute reuse-set bytes per CU against L2 size first: when reuse bytes vastly exceed L2 the kernel is traffic-bound rather than pollution-bound, the whole family recovers only the small thrash fraction, and a ~1% result is the ceiling of the axis, not a tuning failure to be pushed on. The same swizzle mechanism can pay on kernels whose reuse set does fit, so treat the reuse-vs-L2 ratio as the discriminator when borrowing that lever across kernels.
- apply: Estimate per-CU reused bytes for the dominant operand, compare with L2, and rank locality directions below dtype/traffic directions when the ratio is large.
- verify: L2 hit rate and L2 miss bytes with the reorder toggled in the same binary - if miss traffic barely moves, the ceiling is the reuse set.
- pitfall: the same reorder was filed dead_end, overturned to partial, then closed -> a ~1% result was read as a tuning failure rather than as the ceiling of the axis -> price reuse bytes per CU against L2 up front so the expected size of the win is known before the round.
- caution: Also verify the reorder engaged at all before recording a neutral; a self-disabling swizzle reads exactly like a measured negative.
- source: chuschen 16h time-budget campaign run, 15.57h / 35 passes, 2026-08-11
