---
key: fp8 block-scale fused MoE grouped GEMM (gate+up stage) on gfx950, Composable-Kernel gufusion-style pipeline with host-side block_m grouping
type: lever
confidence: ★★
effect: 1.4655x geomean isolated vs frozen baseline, non-overlapping; per-case 1.30x (smallest token case) / 1.55x / 1.58x — monotone in token count, no case regressed; reproduced across two passes (1.44x, 1.47x)
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 3
toolchain: unknown
last_seen: 2026-08-12
name: 32x32-mfma-remap-carries-a-block-scale-moe-grouped-gemm-epil-moe-grouped-gemm-gfx950-both
description: Remap a CK block-scale MoE grouped GEMM pipeline to 32x32 MFMA (+CShuffle epilogue, +A-LDS pad): 1.47x isolated, all three token cases
keywords: ['moe', 'grouped-gemm', 'mfma', 'block-scale', 'composable-kernel', 'cshuffle-epilogue', 'lds-padding', 'fp8']
kernels: ['moe_stage1']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: both
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-07-29
roofline: memory-bound at ~0.40 of its roof -> compute-bound at ~0.53 of its roof
origin_kernels: ['moe_stage1']
---
# 32x32 MFMA remap carries a block-scale MoE grouped GEMM; epilogue and LDS pad are thin
- lever: Switch the grouped-GEMM pipeline variant from 16x16 to 32x32 MFMA with a matching host-side wave shuffle, then add a CShuffle write-out epilogue and one row of A-LDS padding.
- apply: The pipeline-version remap lives in the modifiable header the build recompiles (an -I path include), so no new kernel is needed; host shuffle shape must be changed in the same patch or the variant never engages.
- stack: total 1.4655x isolated (verified, incremental in landing order)
  - 1. MFMA 16x16->32x32 + host shuffle — 1.444x standalone (round 1, verified) — carries the win
  - 2. CShuffle write-out epilogue — +1.25% geomean on top of (1) (verified, bit-exact)
  - 3. A-LDS extra-M pad 0->1 — +0.33% geomean on top of (1,2) (verified, bit-exact but thin)
- verify: Confirm the remapped variant actually runs (the small-token case is the one that silently falls back), then re-time all cases against the frozen baseline; all three landings were bit-exact so parity drift means the wrong variant compiled.
- pitfall: Perf/correctness runs read a stale artifact -> the build step deletes the shared object and the JIT dir -> compile first inside every measurement invocation.
Widening the M tile to 128 corrupted results -> MPerBlock is coupled to the host block_m and 128 straddles two expert groups -> keep MPerBlock == block_m.
- caution: Also verify the accumulator width before chasing occupancy: here the fp32 gate+up accumulator IS the output and pins occupancy at 2, and forcing 3 spilled. Also verify a low-precision B storage format cuts op count at all — the f8f6f4 MFMA path consumes the same K per instruction, so fp4-B returned ~1.00x.
- source: run moe_stage1-ch16h (16h per-kernel time-budget campaign), 2026-07-29
