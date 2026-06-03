# HIP Kernel Optimization: Reproducibility Comparison

## Overview

Unified reproducibility benchmark across 3 optimization runs on 12 kernels (excluding mla_decode).
All results measured with the **same baseline** per kernel (3 runs, median), same GPU (MI300X gfx942),
same benchmark harness (`task_runner.py performance`). Speedup = geomean across test shapes.

|                     | Team cc4.6                        | Team cc4.8                        | Team workflow                     |
| ------------------- | --------------------------------- | --------------------------------- | --------------------------------- |
| **Skill**           | team (Director→TechLead→Engineer) | team (Director→TechLead→Engineer) | team (Director→TechLead→Engineer) |
| **LLM**             | Claude Opus 4.6                   | Claude Opus 4.8                   | Claude Opus 4.7 (via Workflow)    |
| **Budget**          | 6 per kernel                      | 6 per kernel                      | 6 per kernel                      |
| **Date (optimize)** | 2026-06-01                        | 2026-05-31                        | 2026-05-31                        |
| **Date (repro)**    | 2026-06-02                        | 2026-06-02                        | 2026-06-02                        |
| **GPU**             | AMD MI300X (gfx942)               | AMD MI300X (gfx942)               | AMD MI300X (gfx942)               |

## Per-Kernel Results (Geomean Speedup, Unified Baseline)

| # | Kernel                | cc4.6    | cc4.8    | workflow | Winner   |
|---|----------------------|----------|----------|----------|----------|
| 1 | knn                  | 14.33x   | 12.94x   | **19.94x** | workflow |
| 2 | roipoint_pool3d      | 14.57x   | 17.84x   | **19.37x** | workflow |
| 3 | roiaware_pool3d      | 9.77x    | 11.53x   | **14.55x** | workflow |
| 4 | ball_query           | **9.80x** | 8.14x   | 8.79x    | cc4.6    |
| 5 | three_nn             | **6.23x** | 5.47x   | 3.53x    | cc4.6    |
| 6 | points_in_boxes      | **3.03x** | 2.24x   | 2.14x    | cc4.6    |
| 7 | gather_points        | **2.45x** | 1.66x   | 2.15x    | cc4.6    |
| 8 | assign_score_withk   | **2.11x** | 1.78x   | 2.11x    | cc4.6    |
| 9 | three_interpolate    | 1.38x    | 1.40x   | **1.90x** | workflow |
| 10| furthest_point_sample| 1.29x    | 1.28x   | **1.46x** | workflow |
| 11| silu                 | 1.16x    | 1.24x   | **1.27x** | workflow |
| 12| matrix_multiplication| 1.02x    | 1.03x   | **1.04x** | workflow |

## Aggregate (12 kernels, excluding mla_decode)

| Metric                        | cc4.6    | cc4.8    | workflow |
| ----------------------------- | -------- | -------- | -------- |
| **Geo-of-Geo (overall)**      | 3.56x    | 3.32x    | **3.68x** |
| **Win count (best of 3)**     | 5        | 0        | **7**    |
| Failures                      | 0        | 0        | 0        |

## Analysis

### Overall Ranking: workflow > cc4.6 > cc4.8

The three runs are close in aggregate (3.32x–3.68x geomean), but show distinct strengths:

1. **workflow (3.68x, 7 wins)**: Best on the large compute-bound kernels (knn 19.94x, roipoint_pool3d 19.37x, roiaware_pool3d 14.55x). Also leads on overhead-bound kernels (three_interpolate 1.90x, furthest_point_sample 1.46x). The Workflow orchestration with Claude Opus 4.7 appears to produce deeper algorithmic rewrites on complex kernels.

2. **cc4.6 (3.58x, 5 wins)**: Competitive overall and leads on medium-complexity kernels. Strongest on ball_query (9.80x — warp-cooperative ballot/popcll), three_nn (6.23x — warp-cooperative + sqrt fusion), points_in_boxes (3.03x — unconditional writes + fused C++ API), gather_points (2.45x — multi-channel kernel + fast C++ path), and assign_score_withk (2.11x — atomicAdd removal + backward fusion). Claude Opus 4.6 running as direct subagents.

3. **cc4.8 (3.32x, 0 wins)**: Never the best on any kernel, but competitive on large kernels (roipoint_pool3d 17.84x, knn 12.94x, roiaware_pool3d 11.53x). Claude Opus 4.8 running as direct subagents — underperforms relative to the other two across the board.

### Pattern: Where each system excels

- **Large 3D point-cloud kernels** (knn, roipoint_pool3d, roiaware_pool3d): workflow > cc4.8 > cc4.6. These kernels have the most room for deep algorithmic restructuring (warp-cooperative search, kernel fusion, parallel collection).
  
- **Medium-complexity kernels** (ball_query, three_nn, points_in_boxes, gather_points): cc4.6 > workflow > cc4.8. These benefit from targeted optimizations (specific warp intrinsics, template dispatch, wrapper bypass).

- **Overhead-bound kernels** (three_interpolate, furthest_point_sample, silu, matrix_multiplication): workflow > cc4.8 ≈ cc4.6. Limited headroom — gains come from wrapper streamlining and compiler hints. All three systems achieve similar results (1.0x–1.9x).

### Reproducibility Notes

- Baselines verified to match original optimization measurements (±5% for all kernels).
- Each benchmark: 3 runs, median selected, with `gpu_lock.sh` for exclusive GPU access.
- PyTorch JIT extension cache cleared per-kernel before each compile to prevent stale optimized `.so`.
- Hipified `*_hip.*` files scrubbed from workspace for kernels that use torch JIT (regenerated during compile); kept for kernels with Makefile-based builds.
- Result data: `/wekafs/zihao/2026/geak_cc/PerfSkills/exp/team_cc4.6_time/repro_results/*.json`
