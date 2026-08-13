---
key: int4 group-quantized-weight fused-MoE grouped GEMM on gfx950/CDNA4, per-expert token buckets spanning a tiny-M dispatch-bound case and large-M MFMA-bound cases
type: lever
confidence: ★★
effect: ~4.58x isolated geomean vs frozen baseline, non-overlapping per-case: 3.43x tiny-M, 4.88x mid-M, 5.74x large-M; roofline fraction 0.10 -> 0.60 of its own roof
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 5
toolchain: unknown
last_seen: 2026-08-12
name: per-bucket-tile-shape-carries-an-int4-weight-moe-grouped-gem-moe-grouped-gemm-gfx950-mixed
description: Per-bucket BLOCK_M/num_warps/nonkdim plus byte-once dual-nibble int4 dequant stacks to ~4.6x on int4-weight MoE grouped GEMM, gfx950
keywords: ['moe', 'grouped-gemm', 'int4', 'dequant', 'block-m', 'num-warps', 'mfma-nonkdim', 'per-bucket-tuning', 'occupancy', 'gfx950']
kernels: ['fused_moe_kernel_gptq_awq']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
levers: ['mem.tile-shape', 'compute.mfma-nonkdim16', 'mem.dequant-reuse']
---
# Per-bucket tile shape carries an int4-weight MoE grouped GEMM
- lever: Treat each token-count bucket as its own tuning problem, and pay the int4 unpack once per byte instead of once per nibble.
- apply: Load each packed int4 byte once, extract both nibbles by constant shift over the row block and interleave in registers; then per bucket set BLOCK_M (small/medium/large, e.g. 64/256/512), num_warps (4/4/8), matrix_instr_nonkdim=16 on the largest, and gate the software prefetch off for the small-M bucket.
- stack: total ~4.58x isolated = four directions compounded
  - 1. byte-once dual-nibble int4 dequant — 1.82x standalone (verified) — the bulk of the win
  - 2. num_warps 4->8 + matrix_instr_nonkdim=16 — 2.23x on top of (1) (verified)
  - 3. per-bucket BLOCK_M enlargement — 1.80x on top of (1,2), 2.23x -> 4.00x (verified)
  - 4. small-M no-prefetch gate + num_warps re-sweep — 1.13x, tiny-M case only (verified)
  - note: attribution is incremental in landing order, not independent.
- verify: Re-time every bucket against the frozen baseline separately: the large-M gain and the tiny-M gain come from opposite levers, and a geomean hides either one flipping.
- pitfall: num_warps=16 spilled and BLOCK_N=32 regressed → the fp32 [M,64] accumulator VGPR floor is set by the M-side tile, not the N-side → sweep BLOCK_M per bucket and leave BLOCK_N at its default.
  - pitfall: the harness reported IMPROVED=false on rounds that had actually won → report-script false negative → grep the applied-marker and delta yourself and judge the verified geomean against the ~1-2% run-to-run noise floor.
- caution: Also verify the per-element relative-error gate before deferring group scales out of the operand: with signed operands and deep K, one cancelling element blew the gate by ~5 orders of magnitude even in full fp32.
- source: 16h per-kernel time-budget campaign, run fused_moe_kernel_gptq_awq-ch16h, 2026-08-12
