---
key: fp8 blockscale grouped-MoE GEMM with preshuffled B on gfx950/CDNA4: where the nameplate resource axes were priced and found already solved
type: anti-pattern
confidence: ★★
effect: ~1.00x across 12 directions on five nameplate axes: 2.5x more LDS moves occupancy not at all; the arm that removed every spill was 6.7-7.5% slower; deleting 68% of the code object moved <=0.10% against a measured 0.42% run-to-run floor; scatter traffic is invariant to the megabyte under the most extreme grid reorder and a per-case atomicity oracle caps that whole axis at 7.4% on the large-batch cases / 2.1% on the small-batch case; host-side capture of a single-node graph is a regression
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 12
toolchain: unknown
last_seen: 2026-08-12
name: nameplate-resources-are-already-solved-on-preshuffled-b-bloc-moe-grouped-gemm-gfx950-mixed
description: Occupancy, spill, LDS capacity, atomic write amplification and code size all return ~1.00x here; wins live in issue, dependency structure and LDS bank phase
keywords: ['anti-pattern', 'occupancy', 'register-spill', 'lds-capacity', 'atomics', 'code-size', 'counter-guided', 'moe-grouped-gemm', 'fp8-blockscale', 'gfx950']
kernels: ['moe_stage2']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-11
roofline: at stop: ~18-21% active, 53-61% dependency wait, 21-27% issue wait; 31-44% of the achievable HBM roof with occupancy unchanged from baseline
origin_kernels: ['moe_stage2']
---
# Nameplate resources are already solved on preshuffled-B blockscale MoE GEMM
- lever: Spend early rounds on instruction issue, dependency structure and LDS bank phase; treat occupancy, spill count, LDS capacity, atomic write amplification and code size as axes to PRICE cheaply rather than to patch.
- apply: Price each axis with a one-shot oracle instead of a candidate: inflate the LDS allocation and watch occupancy, reorder the grid and diff the atomic traffic counters, delete most of the code object, and delay-inject the host chain to measure how much of it is even editable.
- verify: Accept a closure only when a counter refuses to move or an address derivation forbids the effect; a direction that merely failed to help is not a closed axis, and re-price after any change that alters the tile shape or moves an operand into LDS.
- pitfall: 'Remove the register spill' looked self-evident -> the spill-free arm was slower because the allocator traded spill for a worse schedule -> read spill count as a description, gate on time.
A wave-grid retune did reach the register target at a doubled block size -> occupancy there quantises in two-wave steps so the intended 3 waves/SIMD is not expressible -> those arms lost 41-64%.
- caution: This is a pricing result for this class and arch, not a verdict on the levers: also verify the oracles on your own shapes, since the atomic and LDS-capacity closures depend on B staying out of LDS and on the scatter being a device-scope atomic.
- source: run moe_stage2-own16h, rounds 1-27, 2026-08-12; each closure counter- or derivation-backed
