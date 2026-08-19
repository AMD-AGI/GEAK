---
key: MoE grouped GEMM whose weight operand dominates HBM traffic, narrowed to e2m1 fp4 with per-block scales, on gfx950 / Triton
type: lever
confidence: ★★
effect: ~42x isolated vs frozen baseline, non-overlapping; per-case ~30x on the tiny-M case and ~50x on both large-M cases; roofline-emp 0.02 latency-bound -> 0.51 compute-bound
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: narrow-the-streamed-weight-operand-first-then-chase-the-mfma-moe-grouped-gemm-gfx950-mixed
description: Store the MoE grouped-GEMM weight operand as e2m1 fp4 consumed natively by MFMA, then nonkdim=16 + XCD de-interleave: ~42x isolated on gfx950
keywords: ['moe', 'grouped-gemm', 'fp4', 'dot-scaled', 'mfma-nonkdim16', 'xcd-partitioning', 'l2-residency', 'weight-quantization']
kernels: ['fused_moe_kernel']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: mixed
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-07-30
roofline: latency-bound at ~0.02 of achievable peak -> compute-bound at ~0.51
origin_kernels: ['fused_moe_kernel']
---
# Narrow the streamed weight operand first, then chase the MFMA shape
- lever: When one operand of a grouped GEMM is streamed and the other is minority traffic, store the streamed one as e2m1 fp4 with per-block scales folded into the epilogue so the dot-scaled MFMA consumes it natively; this breaks the HBM roofline before any tiling work.
- apply: Keep the activation operand at fp8; fp4 weights + dot_scaled; matrix_instr_nonkdim=16 on the large-grid config to reach the 2x-K f8f6f4 MFMA; num_stages 1->2 only on the tiny-M case; de-interleave the round-robin XCD dispatch so one expert's weights stay L2-resident per XCD.
- stack: total ~42x isolated (weighted, director-verified) = four directions compounded
  - 1. fp4 weight storage - ~40x standalone (verified) - carries essentially all of the win
  - 2. nonkdim=16 large-grid MFMA - +4% on top of (1) (verified)
  - 3. num_stages 1->2, tiny-M case only - +0.9% on top of (1,2) (verified); the same knob is net-negative on the large cases
  - 4. XCD de-interleave - +0.6% on top of (1,2,3) (partial): +1.2% on one large case, neutral on the other
  - note: incremental in landing order, not independent contributions.
- verify: Per-case frozen-baseline isolated A/B plus the cosine parity gate; confirm the narrow-float path actually engaged by inspecting the emitted MFMA variant (cbsz/blgp operand-format field), not just the config flag.
- pitfall: Ported the winning MFMA shape to the tiny-M case and it regressed -> that case is grid-starved, not MFMA-shape limited -> keep shape and stage knobs per-case rather than global.
Script reported IMPROVED=false while the A/B showed verified gains -> the working tree sits under a gitignored path so git diff came back empty -> build the patch with diff -u against the canonical source.
- caution: Also verify that the per-row/per-column scale multiply in the epilogue vectorizes under the chosen MFMA output register layout - a 16x16x128 layout interleaves M-rows across a lane's register group and can leave the scale as a long chain of scalar multiplies.
- source: run fused_moe_kernel-ch16h, 16h per-kernel time-budget campaign, 2026-07-30
