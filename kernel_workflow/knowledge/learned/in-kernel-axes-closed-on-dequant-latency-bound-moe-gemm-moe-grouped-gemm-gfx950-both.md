---
key: int4 W4A16 fused-MoE grouped GEMM on gfx950, Triton, at its host-config optimum — which in-kernel knob axes were already swept to exhaustion
type: anti-pattern
confidence: ★★
effect: seven axes disconfirmed against a 3.33x incumbent: MFMA nonkdim 32 regressed +18.4% / +1.2% / +1.7% across the three M-bucket cases and kpack 0/1/2 was inert; num_warps=4 up to 6.5x worse (fp32 accumulator spill); block-K 64 +12-58% every case; block-K>=256 or num_stages=3 aborted on LDS overflow for the two large cases; a bf16 dequant intermediate took the geomean 3.33x -> 2.95x; genuine split-K lost in every case; wrapper graph replay 0.985x
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 11
toolchain: unknown
last_seen: 2026-08-12
name: in-kernel-axes-closed-on-dequant-latency-bound-moe-gemm-moe-grouped-gemm-gfx950-both
description: On a dequant-latency-bound int4 MoE grouped GEMM at the host-config optimum, seven in-kernel axes each re-measured as no-gain or a regression.
keywords: ['moe-grouped-gemm', 'int4-dequant', 'anti-pattern', 'closed-axis', 'split-k', 'num-warps', 'matrix-instr-nonkdim', 'num-stages', 'cuda-graph']
kernels: ['fused_moe_int4_w4a16']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: active
---
# in-kernel-axes-closed-on-dequant-latency-bound-moe-gemm
- lever: Once the host-config optimum is in hand for this class, prefer spending the round on the operand/instruction-format axis over re-sweeping in-kernel knobs: nonkdim x kpack, block-K x num_stages, num_warps, split-K, a narrowed dequant intermediate dtype, lop3-style bit-trick dequant and graph capture were each re-measured here and none moved the incumbent.
- apply: Treat those axes as already-swept priors and re-open one only when something invalidates the measurement - a different tile, a non-fp32 accumulator, or a grid too small to fill the CUs.
- verify: The expected signature of a closed axis is an empty patch: cumulative geomean unchanged to four decimals across the whole direction, which reads as nothing-to-gain rather than measurement failure; every combination stayed bit-correct, so the loss is pure performance.
- pitfall: A 16x16 -> 32x32 MFMA tile win carried over from a different MoE-stage kernel class regressed here -> that op was issue-count bound and this one is dependency-latency bound -> check the bound class before transferring a tile-shape or wave-count result across kernel classes.
- caution: Also verify the grid is genuinely CU-saturating before treating split-K as closed - the negative here held at roughly 3072 tiles over 256 CUs, and a smaller grid would reopen it.
- source: 16h per-kernel time-budget campaign, 49 resumed passes, 9 dead-end directions, 2026-08-11
