---
key: profiler-counter-guided tuning of a dequant-bound packed-int4 GEMM on gfx950, where the loop is latency- and dependency-bound rather than instruction-supply bound
type: anti-pattern
confidence: ★★
effect: closed axis, ~1.00x or worse across six counter families: removing 85% of LDS bank conflicts measured 18.6% SLOWER; deleting 17% of K-loop VALU moved the clock 0%; deleting 75% of the empty CTAs was under +0.15%; breaking the in-flight-load cap cost 28%; DRAM traffic priced at ~0.17% of wall per 1% removed; and the final winner took M=65536 down 3.5% while K-loop VALU went UP 1.2%
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 12
toolchain: unknown
last_seen: 2026-08-12
name: price-a-counter-with-a-deletion-control-before-funding-a-rou-moe-grouped-gemm-gfx950-prefill
description: Counter-guided directions (bank conflicts, VALU, barriers, occupancy, empty CTAs, traffic) returned ~1.00x or worse; time a deletion control first.
keywords: ['moe-grouped-gemm', 'w4a16', 'int4-dequant', 'occupancy', 'lds-tiling', 'counter-falsification', 'anti-pattern', 'launch-config', 'gfx950']
kernels: ['fused_moe_kernel_gptq_awq']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: prefill
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-12
origin_kernels: ['fused_moe_int4_w4a16']
---
# Price a counter with a deletion control before funding a round on it
- lever: before funding a direction aimed at a large counter, build the cheapest control that DELETES that counter and time it - across 24 rounds only dependency-chain depth, SLP packing width and K-loop schedule perturbation ever moved the clock on this class.
- apply: keep the deletion control byte-comparable (same shapes, same launch config, only the counter-bearing transform toggled) and read the clock, not the counter; for the levers that did pay, target chain depth (widen the packed integer domain before nibble extraction) and SLP operand shape (keep non-value operands scalar-broadcast so f32 elementwise stays packed).
- verify: interleave the A/B against byte-identical controls: the official harness here carried ~2% run-to-run spread over 9 runs, enough that a headline can be a ~90th-percentile draw, so treat the official run as a gate and the interleaved A/B as the meter.
- pitfall: occupancy swept as waves/SIMD found nothing for five rounds -> the currency is CTAs/CU = floor(512/VGPR)*4/num_warps, and waves/SIMD times peak in-flight loads is conserved at ~20 with HEAD already at the per-SIMD maximum -> sweep the launch-kwargs pair jointly instead.
compile-only probes were used to SELECT a variant and picked one with identical registers, waves and CTAs that measured 32.3% slower -> a compile probe is a kill gate, not a select gate -> time every survivor.
- caution: also verify on your own shapes before treating a family as closed here - the traffic lever was causal, merely cheap, and on a bandwidth-bound shape the same 1% of traffic can be worth far more than it was here.
- source: run kernel_20_geak_0808_16h, proposal fused_moe_int4_w4a16-own16h, 2026-08-12; 21 static counters individually falsified as win predictors across 24 rounds
