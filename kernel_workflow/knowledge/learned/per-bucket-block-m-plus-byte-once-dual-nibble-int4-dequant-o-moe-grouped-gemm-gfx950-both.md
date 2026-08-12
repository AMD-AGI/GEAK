---
key: int4 weight-only (GPTQ/AWQ-style) fused-MoE grouped GEMM on gfx950/CDNA4, Triton, batch-swept from single-token to large-batch
type: lever
confidence: ★★
effect: 4.58x isolated geomean vs the frozen baseline, non-overlapping; per case batch-2 3.43x, batch-32 4.88x, batch-64 5.74x (best single pass 5.11x). Empirical roofline 0.10 -> 0.60, compute-bound throughout.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 6
toolchain: unknown
last_seen: 2026-08-11
name: per-bucket-block-m-plus-byte-once-dual-nibble-int4-dequant-o-moe-grouped-gemm-gfx950-both
description: Per-bucket BLOCK_M/num_warps plus byte-once dual-nibble int4 dequant lifts a weight-quantized MoE grouped GEMM ~4.6x, empirical roofline 0.10 to 0.60.
keywords: ['moe', 'int4', 'weight-only-quant', 'grouped-gemm', 'block-m', 'num-warps', 'mfma-nonkdim', 'dequant', 'software-pipeline', 'gfx950']
kernels: ['fused_moe_kernel_gptq_awq']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: archived
verified_on: 2026-08-11
---
# Per-bucket BLOCK_M plus byte-once dual-nibble int4 dequant on weight-quantized MoE GEMM
- lever: Tune the tile/launch config PER BATCH BUCKET rather than once, and load each packed 4-bit weight byte once, extracting both nibbles with constant shifts held in registers.
- apply: Bucket the autotune space by M: small-M gets a small BLOCK_M with a no-prefetch JIT K-loop, large-M gets BLOCK_M 256/512 with num_warps 8 and matrix_instr_nonkdim=16; replace the two-load nibble path with one byte load + dual-nibble constant-shift extract interleaved over 32 rows; keep a hand-written register double-buffer as the pipeline.
- stack: total 4.58x isolated geomean (frozen baseline) = four directions compounded
  - 1. byte-once dual-nibble int4 dequant + register interleave — 1.82x standalone (verified) — the largest single step
  - 2. num_warps 4->8 + matrix_instr_nonkdim=16 — 2.23x cumulative on top of (1) (verified); num_warps 16 spills
  - 3. per-bucket BLOCK_M enlargement (64 / 256 / 512) — 2.23x -> 4.00x cumulative (verified) — amortizes cross-tile weight re-dequant
  - 4. small-M no-prefetch gate — 1.13x on top of (1-3), 1.46x on the batch-2 case only
  - note: attribution is incremental in landing order, not independent
- verify: Re-time every bucket against the frozen baseline separately, not just the geomean: the small-batch bucket is dispatch/latency-bound at high occupancy and moves for different reasons than the register-tight large-M buckets, and a config that wins one can regress another.
- pitfall: The harness IMPROVED=false verdict was a false negative on rounds that actually won -> the marker/delta grep and the verified geomean disagreed -> confirm the edit landed in the source and judge the geomean against a ~1-2% run-to-run noise floor before discarding a round.
Widening BLOCK_N instead of BLOCK_M regressed -> the accumulator register floor is set by the M extent -> sweep BLOCK_M per bucket and leave BLOCK_N alone unless a re-measure says otherwise.
- caution: Also verify that deferring the per-group weight scale out of the inner loop is numerically legal for your gate: with signed operands, deep K and a per-element relative-error bound, one cancellation element blew the tolerance by ~5 orders of magnitude even though the same deferral is fine on 8-bit siblings.
- source: 16h single-kernel time-budget campaign, run id chuschen16h, 32 passes, 2026-08-11
