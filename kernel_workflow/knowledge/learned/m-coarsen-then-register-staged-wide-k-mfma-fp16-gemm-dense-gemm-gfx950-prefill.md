---
key: fp16 (a16w16) dense GEMM with small-M / K=2880 / N=5120 shapes on gfx950, Triton 3.6 with a Gluon rewrite available
type: lever
confidence: ★★
effect: 2.66x weighted vs frozen baseline; per-case 1.89x on the smallest-M case (M=2048, tail/occupancy limited) and 3.15x / 3.18x on the two larger-M cases. Intermediate stages on the same cases: 2.11x from LDS-per-workgroup reduction, 2.60x after in-body M-coarsening.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-11
name: m-coarsen-then-register-staged-wide-k-mfma-fp16-gemm-dense-gemm-gfx950-prefill
description: Frozen BLOCK_M is not a wall: in-body M-coarsening then a register-staged wide-K MFMA rewrite gives ~2.66x weighted on fp16 dense GEMM (gfx950).
keywords: ['dense-gemm', 'fp16', 'm-coarsening', 'mfma', 'register-staging', 'wide-k', 'lds-tiling', 'num-stages', 'gfx950', 'skinny-m']
kernels: ['_gemm_a16_w16_kernel']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: prefill
layer: learned
lifecycle: archived
---
# m-coarsen-then-register-staged-wide-k-mfma-fp16-gemm
- lever: Treat a frozen BLOCK_M as a knob you can still move from inside the kernel body (coarsen each program over 2 output tiles), then go further with a register-staged wide-K MFMA rewrite at a large M tile.
- apply: Stage 1: num_stages 3 -> 1 to cut LDS per workgroup ~4.5x. Stage 2: in-body COARSEN=2 over M (VGPR 170->216, occupancy 2, zero spills). Stage 3: rewrite with the Gluon/CDNA4 buffer_load + mfma primitives at BM=256, BN=128, KT=64, instr_shape=[16,16,32], warps_per_cta=[4,1], k_width=16, accumulating in registers over a wide K step.
- stack: total 2.66x weighted = three directions compounded, incremental in landing order: 1. LDS/num_stages reduction 2.11x standalone; 2. in-body M-coarsening 2.60x cumulative (+23% on top of 1); 3. register-staged wide-K rewrite 2.66x cumulative (+2% on top of 1,2, and it is what lifts the two large-M cases past 3x).
- verify: Re-time each case against the frozen baseline separately: the small-M case is warmup/DVFS sensitive while the largest case self-warms, so a weighted geomean can move without any case moving. Confirm VGPR stays under the spill cliff and occupancy stays at 2.
- pitfall: Harness reported IMPROVED=false while the candidate was in fact several times faster -> false negative in the improvement detector -> grep the source for the patch's own markers before believing a negative. Also: KT has to be a power of two AND divide K, and one compiled kernel serves all shapes, so in-body variants have to be runtime-gated rather than constexpr-gated.
- caution: Also verify the tile does not overshoot: at BM=512 the fp32 accumulator spills and the case regresses ~10x, so scale BM up one step at a time and check the spill count each step.
- source: 16h per-kernel time-budget campaign chuschen16h, 44 passes, 2026-08-11
