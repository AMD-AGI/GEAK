---
key: already-optimized narrow-float grouped GEMM sitting at occupancy 1 on gfx950, where the residual stall is a dequant->MFMA dependency chain rather than exposed load latency
type: anti-pattern
confidence: ★★
effect: closed axis: ~1.00x cumulative across four pipeline/occupancy directions; a genuine weight-tile double-buffer -14% (both large cases), num_stages>1 -43% when the existing buffer is disabled, sequential sub-tile wave -55%, occupancy 2 -76%, XCD re-order a ~1% ceiling
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 6
toolchain: unknown
last_seen: 2026-08-12
name: diagnose-dependency-chain-vs-load-latency-before-spending-a--moe-grouped-gemm-gfx950-mixed
description: At occupancy 1 with a dequant->MFMA dependency chain, pipeline-depth and occupancy levers return ~1.00x or regress; a regressing double-buffer is the tell
keywords: ['moe', 'grouped-gemm', 'fp4', 'software-prefetch', 'num-stages', 'occupancy', 'l2-residency', 'dep-chain', 'anti-pattern']
kernels: ['fused_moe_kernel']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-07-30
origin_kernels: ['fused_moe_kernel']
---
# Diagnose dependency chain vs load latency before spending a round on pipelining
- lever: Cheap discriminator: software double-buffer the operand you believe is on the critical path. If that REGRESSES, the residual is a dependency chain (dequant/scale -> dot_scaled), not exposed load latency, and the whole pipeline-depth / occupancy / cache-hint axis is closed for that body.
- apply: Spend one round on the discriminator, then reallocate to shape or algorithm work instead of sweeping stage counts, prefetch depths, sub-tile splits and eviction policies one at a time.
- verify: Frozen-baseline isolated A/B per case, and read the flag's source before sweeping it: a value named like a depth knob was a single 1-deep double-buffer whose out-of-set values merely turned it off, so the sweep measured on-vs-off, not depth.
- pitfall: Raising occupancy was reachable and numerically valid yet cost ~76% -> the second wave doubles per-CU streamed weight traffic to ~100x the L2 capacity and thrashes -> occupancy is a lever only when a single wave is latency-stalled.
Extra buffering also raises register pressure on a body already at occupancy 1, so the pipelining attempt makes the dependency chain worse rather than neutral.
- caution: Also verify the reused operand set actually fits L2 before borrowing a non-temporal / eviction-policy cache lever: when the reuse working set greatly exceeds L2 the op is traffic-bound, not pollution-bound, and the hint returns ~1.00x.
- source: run fused_moe_kernel-ch16h, 16h per-kernel time-budget campaign, 2026-07-30
