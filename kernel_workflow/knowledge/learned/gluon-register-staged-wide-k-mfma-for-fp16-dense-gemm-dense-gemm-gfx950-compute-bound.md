---
key: fp16 A16W16 dense GEMM with small M and a non-power-of-2 K, Triton/Gluon on gfx950 CDNA4, one compiled kernel serving several shapes
type: lever
confidence: ★★
effect: 2.66x geomean vs the frozen baseline, non-overlapping; per-case 1.89x on the small-M case and 3.15x / 3.18x on the two large-M cases; roofline-emp 0.51 -> 1.47, bound class stays compute
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: gluon-register-staged-wide-k-mfma-for-fp16-dense-gemm-dense-gemm-gfx950-compute-bound
description: fp16 dense GEMM on gfx950: cut num_stages to 1, coarsen M in-body, then rewrite in Gluon with a big-BM register-staged wide-K MFMA loop — 2.66x
keywords: ['dense-gemm', 'gluon', 'mfma', 'register-staging', 'lds-tiling', 'num-stages', 'm-coarsening', 'gfx950', 'fp16', 'compute-bound']
kernels: ['_gemm_a16_w16_kernel']
platforms: ['gfx950']
kernel_class: dense_gemm
regime: compute-bound
layer: learned
lifecycle: active
---
# Gluon register-staged wide-K MFMA for fp16 dense GEMM
- lever: Three stacked moves on an MFMA-bound fp16 GEMM: drop num_stages to 1 (frees most of the per-workgroup LDS), coarsen M inside the body to overturn a frozen BLOCK_M, then rewrite the loop in Gluon (@gluon.jit, arch buffer_load + mfma) as a register-staged wide-K MFMA with BM=256, BN=128, instr_shape 16x16x32, warps_per_cta [4,1], k_width=16.
- apply: One compiled kernel serves every shape here (M/N/K arrive as runtime args), so in-body variants have to be runtime-gated rather than constexpr; the tile-K is picked as the only power of two that divides K exactly.
- stack: total 2.66x weighted vs the frozen baseline = three directions, cumulative in landing order
  - 1. num_stages 3 -> 1 — 2.11x cumulative (round 1, verified) — the bulk of the win
  - 2. in-body M-coarsening COARSEN=2 — 2.60x cumulative (round 2, verified); VGPR 170 -> 216, occupancy 2, no spills
  - 3. full Gluon register-staged rewrite — 2.66x cumulative (verified, reproduced twice later at 2.667x)
  - note: attribution is cumulative in landing order, not independent
- verify: Re-time every case against the frozen baseline and grep the built source for the patch markers before believing a negative verdict — this harness reported IMPROVED=false on candidates that were in fact 6x or better.
- pitfall: bigger BM kept paying, so BM=512 looked next -> the fp32 accumulator tile spills -> roughly 10x regression; keep the register budget inside the occupancy-2 envelope (about 214 of 256 VGPR here) and skip explicit register prefetch, which also spills.
- caution: also verify the weighted number and not just the geomean: the smallest-M case gains only about 60% of what the large-M cases gain, and it is warmup/DVFS sensitive while the largest case self-warms.
- source: run _gemm_a16_w16_kernel-ch16h (16h single-kernel budget, 44 passes), 2026-08-12
