---
type: Kernel Case Study
title: ck_moe_stage2_gemm (MoE stage-2 down-proj CK block-scale GEMM)
description: An occupancy-bound MoE down-proj CK block-scale GEMM sped up 1.31x geomean by host-side block_m routing away from the high-LDS V3 pipeline toward V1 (2 blocks/CU) instances, with no C++ rebuild and bit-exact output.
tags: [domain-moe, bottleneck-occupancy, lever-host-side, gfx942]
speedup: "1.31x geomean (per-shape 1.25-1.33x)"
correctness: PASS, bit-exact grade (err_ratio 0.0000, cos_diff ~1.5e-7)
kept: kept-deployed
timestamp: 2026-06-22T00:00:00Z
---

# Baseline
- Kernel: `ck_moe_stage2_gemm`, CK `DeviceMoeGemmBlockScale` down-proj GEMM, MI300X gfx942.
- Regime: occupancy/latency-bound, NOT HBM-bound (PROFILE: ~36% HBM, ~19% MFMA). Short K=768 loop; reads the small `inter` operand + AtomicAdd scatter write.
- Stock dispatch: `block_m = 64 if token>32 else 16`, routing all prefill to the **V3 instance (256x64x128x128)** — double-buffered LDS prefetch (~48 KB LDS) => only 1 block/CU on gfx942's 64 KB LDS.
- Baseline geomean latency: 0.6256 ms (harness run) / 2.264 ms (KernelForge run, tokens {2048,32768,65536}). Both runs measure the same stock V3 path.

# What changed (the win)
Two complementary levers, both reaching V1-pipeline instances (single LDS buffer ~24 KB => 2 blocks/CU):

- **Host-side block_m routing (deployed winner, `aiter/fused_moe.py`)** — ~15-line, NO C++ rebuild. Stage-2's stock bm16/bm32 instances are **already V1**, so routing away from bm64 captures the occupancy win with zero codegen change:
  - `block_m=16` for **sparse** routing (tokens-per-expert <= ~8, e.g. token=256): ~1.53x, cuts ~87% routing-pad waste.
  - `block_m=32` (V1) for **all dense** prefill (token>256): ~1.24-1.42x, 2 blocks/CU vs V3's 1.
  - Gated to `q_type==per_1x128` 2-stage MoE; other configs unaffected.
- **Force PipelineVersion v1 in stage-2 `.cuh` (KernelForge variant)** — hard-code `DeviceMoeGemmBlockScale` PipelineVer to `ck::BlockGemmPipelineVersion::v1` on the Python-locked block_m=64 instance: ~1.28x geomean. Confirms the same occupancy mechanism from the C++ side when block_m cannot be changed.

Root cause: 1 block/CU cannot hide launch/fill + AtomicAdd-scatter latency on the short K loop. Doubling co-resident blocks/CU hides it. 2 blocks/CU is the hard LDS ceiling for this tile (A+B tile = 24 KB).

# Result
| metric | value |
|---|---|
| headline speedup | **1.31x geomean** (ako winner; per-shape 1.25-1.33x) |
| baseline -> opt (harness) | 0.6256 ms -> 0.476 ms |
| per-shape (harness) | t256 1.53x, t2048 1.41x, t4096 1.40x, t8192 1.24x, t11264 1.32x, t16384 1.34x, t17920 1.34x |
| correctness | PASS, bit-exact grade (err_ratio 0.0000, cos_diff ~1.5e-7) |
| rebuild | none — host-only routing edit |

Cross-stage finding: the V1-pipeline lever FLIPS SIGN between stages — moe_stage1 (weight-stream/L2-BW bound) prefers V3 double-buffering, moe_stage2 (occupancy-bound) prefers V1 + smaller block_m. Always re-A/B the pipeline version per kernel.

# What was tried and reverted
- **bm64-V1 for dense** (expert/kda/geak added a C++ codegen instance): uniformly slower than bm32-V1 by direct A/B (8192: 0.843 vs 0.789 ms; 16384: 1.565 vs 1.487; 17920: 1.723 vs 1.624). The added C++ instance is never worth using; bm32-V1 is the uniform dense optimum. The apparent expert>ako crossover was thermal noise.
- **bm128-V1 for large prefill** (geak): regressed (4096 0.47->0.63) — larger MPerBlock starves the occupancy-bound kernel.
- **bm32 with KPerBlock=256** (kda): regressed — raises reg/LDS pressure, costs occupancy. Keep KPerBlock=128.
- **bm16 for ALL prefill** (expert): worse than bm32 at dense; bm16 only wins the sparse regime.
- **Interwave scheduler** (KernelForge): compile FAIL — `Interwave` + v1 unsupported for this `DeviceMoeGemmBlockScale` instance. Reverted.
- **NSwizzle=true** (KernelForge): GPU memory-access fault — N-tile remap not honored by the AtomicAdd-scatter N offset => OOB global writes. Reverted.

# Patterns
- [CK PipelineVersion v1 for occupancy](/patterns/ck-pipeline-v1-occupancy.md)
- [block_m routing by sparsity](/patterns/block-m-routing-sparsity.md)

# Citations
1. spare_kernels/arena_tasks/hip2hip/moe_stage2/RESULTS.md
2. KernelForge/results/moe_stage2/tasks/cli/9a9c10e5-c1c5-4ca4-a394-f98293380c4b/workspace/optimization_report.md
