---
key: an int4 w4a16 fused-MoE grouped GEMM already sitting at a 240-register allocation with zero spills, 2 waves/SIMD and no pipe above ~44% busy, Triton on gfx950/MI355
type: anti-pattern
confidence: ★★
effect: disconfirming, all isolated vs the frozen baseline: deleting 100% of the dequant buys only +2.4% on the merged path and +6.5% on the M=2048 case; 7 bit-exact VALU cuts measured 0.5-2.9% slower; split-K 0.962x and 0.934x; grid compaction (4x fewer workgroups, bit-exact) -1.7%; 12 alternative tile geometries -7% to -90%; a hand-written non-Triton MFMA port -17.9% at M=32768 and -14.4% at M=65536; backend knobs inside a +-0.5% control spread
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: axes-that-returned-about-1-00x-on-a-register-bound-quantized-moe-grouped-gemm-gfx950-both
description: On a register-bound int4 MoE grouped GEMM, VALU cuts, occupancy, tile geometry, split-K, grid compaction and a hand-written port all return ~1.00x or worse
keywords: ['int4', 'w4a16', 'moe-grouped-gemm', 'dequant', 'occupancy', 'tile-geometry', 'split-k', 'anti-pattern', 'gfx950', 'triton']
kernels: ['fused_moe_kernel_gptq_awq']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: active
cost: L1
verified_on: 2026-08-12
levers: ['compute.valu-reduction', 'compute.occupancy', 'compute.split-k', 'host.grid-compaction']
---
# Axes that returned about 1.00x on a register-bound quantized MoE GEMM
- lever: Price the ceiling before funding a round: run the free upper-bound oracle first (delete the candidate work while holding operand traffic constant) and do the register arithmetic (accumulator registers vs the budget the next occupancy step needs) before writing any variant.
- apply: Dump the register census and the pipe-busy counters as the round's first GPU job; if the busiest pipe is well under half busy at 2 waves/SIMD the body is dependency/issue-latency bound, and both cures (more waves, more ILP per wave) compete for the same register file.
- verify: Re-derive the headroom from counters rather than from a percent-of-roof figure: a low MFMA busy fraction that is co-limited by LDS is not spare capacity, and here the honest remaining headroom in this language measured ~2%.
- pitfall: '43% of the MFMA roof' was read as roughly 2x of headroom -> MFMA idles because LDS is the co-limiter and neither moves without occupancy the register file forbids -> retire the roofline framing and quote the measured pipe-busy set instead.
- caution: Also verify the opposite end of a two-variable trade before treating it as closed: occupancy looked dead by a 72-register gap on the large tile, yet a smaller tile did reach 3 waves/SIMD (+5.5% within its own family) and still lost 2.0% overall.
- source: run fused_moe_kernel_gptq_awq-own16h, 19-round campaign 2026-08-08/09, ~15 closed directions, director-validated 2026-08-12
