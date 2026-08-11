---
key: fp4-weight (block-scaled) MoE grouped GEMM on gfx950, Triton dot_scaled, weight-streaming across batch sizes
type: lever
confidence: ★★
effect: 42.2x cumulative isolated (weighted geomean, director-validated) vs frozen baseline; per-case ~29.9x at the smallest batch, ~50.0x and ~50.2x at the two larger batches; empirical roofline 0.02 -> 0.51 of achievable peak, bound class latency-bound -> compute-bound.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-11
name: fp4-weight-storage-moe-grouped-gemm-moe-grouped-gemm-gfx950-both
description: Store the streamed MoE expert-weight operand as fp4 consumed natively by scaled MFMA: 42x isolated on a weight-streaming grouped GEMM, gfx950.
keywords: ['fp4', 'dot-scaled', 'mfma', 'moe', 'grouped-gemm', 'weight-streaming', 'hbm-bound', 'nonkdim', 'xcd-swizzle']
kernels: ['fused_moe_kernel']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: active
verified_on: 2026-08-11
---
# fp4-weight-storage-moe-grouped-gemm
- lever: When the grouped-GEMM weight operand is streamed once per block and dominates traffic, re-store it in fp4 and let the scaled-MFMA path consume it natively; then stack a wider-K MFMA shape (matrix_instr_nonkdim=16), a per-case num_stages bump on the smallest batch, and XCD de-interleave on the large grid.
- apply: Env/flag-gated variants over one source: fp4 storage for the streamed operand with a per-block fp32 scale folded into the epilogue scale, activation operand kept at fp8, SUPER_M=8 / BM=128; nonkdim override only on the large grid; num_stages 1->2 only on the small-batch case.
- stack: total 42.2x isolated (weighted, director-verified) = four directions compounded, incremental in landing order
  - 1. fp4 storage of the streamed weight operand — 39.95x cumulative (verified) — carries essentially the whole win by breaking the memory roof
  - 2. wider-K scaled MFMA via nonkdim=16 on the large grid — 41.57x cumulative, ~+4% on top of (1)
  - 3. num_stages 1->2, small-batch case only — 41.95x cumulative, ~+1% on top of (1,2)
  - 4. XCD de-interleave for per-XCD single-expert L2 residency — 42.24x cumulative, ~+0.7% on top; a ceiling, not a lever
  - note: attribution is incremental in landing order, not independent.
- verify: Re-time every variant against the same frozen baseline and check the parity gate (cosine) per case; confirm the low-precision storage flag actually engaged by watching the small-batch case move independently of the large ones, and confirm the wider-K shape only helps where the grid is well filled.
- pitfall: Cumulative best stalled while individual variants verified faster -> the pass script reported no improvement because the work tree was ignored and its diff came back empty -> rebuild the patch as a textual diff against the canonical source and refresh the recorded best by hand.
- caution: Also verify the scale operand vectorizes on your MFMA output layout before counting on the epilogue: with a 16x16x128 instruction the per-row scale multiply stays scalar, and reassociating that epilogue measured -2% to -23% here.
- source: 16h per-kernel time-budget campaign, lane chuschen16h, 35 resumed passes, 2026-08-11
