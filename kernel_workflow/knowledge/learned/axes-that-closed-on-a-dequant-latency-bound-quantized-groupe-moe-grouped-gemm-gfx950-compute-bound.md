---
key: ceiling diagnosis for a dequant-bound int4 weight-only MoE grouped GEMM on gfx950 sitting at 2 waves behind a mandatory fp32 accumulator tile
type: anti-pattern
confidence: ★★
effect: 10 consecutive directions returned exactly 1.000x per-case (cumulative frozen at 3.33x across all three cases); regressions where they moved: num_warps=4 up to 6.5x worse, MFMA nonkdim 32 +18.4% on the small-M case and +1.2%/+1.7% on the larger ones, bf16 dequant intermediate 3.33x->2.95x, host graph replay 0.985x, hand-scheduled MFMA rewrite 7-13x slower
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 10
toolchain: unknown
last_seen: 2026-08-12
name: axes-that-closed-on-a-dequant-latency-bound-quantized-groupe-moe-grouped-gemm-gfx950-compute-bound
description: Dequant-VALU/latency-bound int4 MoE grouped GEMM on gfx950: eight knob and rewrite axes all measured flat or negative (~1.00x) - low-prior directions
keywords: ['moe', 'grouped-gemm', 'int4', 'weight-only-quant', 'w4a16', 'anti-pattern', 'closed-axis', 'split-k', 'num-warps', 'mfma-nonkdim', 'cuda-graph', 'vgpr-pressure', 'compute-bound']
kernels: ['fused_moe_int4_w4a16']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: compute-bound
layer: learned
lifecycle: active
verified_on: 2026-07-29
---
# Axes that closed on a dequant-latency-bound quantized grouped GEMM
- lever: Give these a low prior once the body is dequant-VALU/dependency-latency bound: MFMA nonkdim x kpack, num_warps, BLOCK_K x num_stages, split-K, a narrower dequant intermediate dtype, lop3/bitcast dequant tricks, host-side graph capture, and full kernel rewrites - each was swept here and none beat the tuned host config.
- apply: Cheap discriminators first: compare host enqueue against per-call GPU time (dispatch was ~1/14 of it, so graph capture cannot pay), and check whether the accumulator tile already pins occupancy before treating occupancy as the lever.
- verify: Every combination stayed bit-correct against the oracle, so these are pure perf outcomes; re-measure per M bucket rather than on the geomean, since one bucket can hide another's regression.
- pitfall: Bitcast dequant cut VALU and convert op counts substantially yet the two large cases got ~7% slower -> the wall is dependency latency plus VGPR pressure (fp32 accumulator at the register limit, 2 waves), not issue throughput -> reducing op count alone does not move it.
- caution: Forcing a third wave by shrinking tile N made the mid case worse, and a 16x16->32x32 MFMA win imported from a sibling MoE kernel did not transfer - also verify the wall class on your own tile and accumulator dtype before reusing this list, which was measured on one large fp32-accumulator tile.
- source: run fused_moe_int4_w4a16-ch16h (16h per-kernel time-budget campaign, 2026-07-28/29), 14-entry direction ledger
