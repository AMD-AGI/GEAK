---
name: cutting-dequant-valu-op-count-buys-nothing-when-the-wall-is--moe-grouped-gemm-gfx950-compute-bound
description: Separate issue-bound from latency-bound before a bit-trick dequant rewrite: on an int4 MoE GEMM a -6% VALU issue count ran ~7% slower
keywords: [dequant, moe, vgpr, occupancy, mfma, counters, compute-bound, operand-reuse, dtype-dialect]
kernels: [fused_moe_kernel_gptq_awq]
platforms: [gfx950]
kernel_class: moe_grouped_gemm
regime: compute-bound
key: int4 weight dequant in the k loop of a bf16 fused-MoE grouped GEMM on gfx950 whose profile shows a large VALU share
lifecycle: active
type: anti-pattern
confidence: ★★
effect: Three disconfirmations, all leaving the cumulative at exactly 3.3326x. (1) A lop3/magic-bitcast int4->bf16 dequant cut the bottleneck's issue count (v_ALU 1168 -> 1093, v_cvt 226 -> 98) and still ran the two large buckets ~7% SLOWER. (2) A bf16 dequant intermediate was bit-correct 8/8 and regressed the geomean 3.33x -> 2.95x. (3) Forcing occupancy 3 at BLOCK_N=128 made the mid case ~4% worse, so it is not occupancy/register-file bound either. The fixed [128,256] fp32 accumulator pins ~254 VGPR at 2 waves, and 2 waves already saturate the per-wave dequant -> downcast -> join/trans chain.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 3
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.72h / 49 passes, 2026-08-11
last_seen: 2026-08-11
---
# Cutting dequant VALU op count buys nothing when the wall is dependency latency plus accumulator VGPRs
- lever: On an int4/low-bit weight path, a profile showing a large VALU share invites the assumption that fewer instructions means less time; separate 'issue-throughput bound' from 'dependency-latency bound' before funding a bit-trick rewrite. The cheap discriminator is to land one op-count reduction and read BOTH the counter and the latency.
- verify: if the issue counter falls and the time rises, the chain length and the accumulator's VGPR footprint are the wall, and both narrowing the dequant intermediate dtype and buying occupancy will also lose; at that point the remaining honest moves are outside the source-level Triton axes (a native block-scaled MFMA body, or a tile change owned by the caller), which is worth saying out loud rather than spending more rounds.
- pitfall: a dequant bit-trick cut v_ALU and v_cvt counts yet the two large buckets got ~7% slower -> the per-wave dequant -> downcast -> join chain is latency-bound and the fixed fp32 accumulator already pins 2 waves -> stop pricing this family by op count and re-price it by chain depth and register footprint.
- source: chuschen 16h time-budget campaign run, 15.72h / 49 passes, 2026-08-11
