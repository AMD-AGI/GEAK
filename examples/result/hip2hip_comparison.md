# HIP Kernel Optimization: GEAK_v3 vs GEAK Skill vs Team Skill

## Overview

| | GEAK_v3 | GEAK Skill | Team Skill |
|---|---|---|---|
| **Architecture** | Standalone agent + centralized evaluator | Single-agent orchestrator + workers (Claude Code) | Multi-agent hierarchy: Director → Tech Lead → Engineers (Claude Code) |
| **Iteration** | Single round | 1 round, workers start from baseline | Multi-round, engineers build on current best |
| **Re-profiling** | No | No | Yes (Tech Lead re-profiles after each round) |
| **Parallelism** | Sequential | 2 parallel workers, dedicated GPU per worker | 3 parallel engineers, flock-based GPU locking |
| **GPU requirement** | 1 GPU | 2 GPUs per kernel (26 total for batch) | 1 shared GPU per kernel |
| **Date** | 2026-05-13 | 2026-05-13 | 2026-05-14 |
| **GPU** | AMD MI300X (gfx942) | AMD MI300X (gfx942) | AMD MI300X (gfx942) |

## Per-Kernel Results (Arithmetic Mean Speedup)

| # | Kernel | GEAK_v3 | GEAK Skill | Team Skill | Best |
|---|--------|---------|------------|------------|------|
| 1 | roipoint_pool3d | 16.82x | 14.60x | **30.32x** | Team |
| 2 | knn | FAIL | 6.56x | **23.00x** | Team |
| 3 | ball_query | 11.62x | **12.64x** | 10.79x | GEAK Skill |
| 4 | roiaware_pool3d | **10.24x** | 9.91x | 10.45x | Team |
| 5 | three_nn | 1.43x | **8.82x** | 3.57x | GEAK Skill |
| 6 | assign_score_withk | 3.76x | 4.00x | **4.13x** | Team |
| 7 | points_in_boxes | 1.03x | 1.03x | **2.37x** | Team |
| 8 | gather_points | **1.32x** | 0.96x | 1.34x | Team |
| 9 | silu | 1.21x | 1.26x | **1.28x** | Team |
| 10 | matrix_multiplication | 1.14x | **1.19x** | 1.09x | GEAK Skill |
| 11 | furthest_point_sample | FAIL | 1.05x | **1.11x** | Team |
| 12 | three_interpolate | 1.01x | **1.15x** | 1.04x | GEAK Skill |
| 13 | mla_decode | N/A | **589.20x** | — | GEAK Skill |

## Aggregate (12 Common Kernels, FAIL = 1.0x)

| Metric | GEAK_v3 | GEAK Skill | Team Skill |
|--------|---------|------------|------------|
| **Arith Mean Speedup** | 4.30x | 5.26x | **7.54x** |
| Wins (best of 3) | 1 | 4 | **8** |
| Failures | 2 | 0 | 0 |

> Note: GEAK_v3 FAIL counted as 1.0x for arithmetic mean. mla_decode excluded from aggregate since only GEAK Skill ran it.

## Win Distribution

| Kernel | GEAK_v3 | GEAK Skill | Team Skill |
|--------|---------|------------|------------|
| roipoint_pool3d | | | +108% vs GEAK Skill |
| knn | FAIL | | +251% vs GEAK Skill |
| ball_query | | +9% vs GEAK_v3 | |
| roiaware_pool3d | | | +2% vs GEAK_v3 |
| three_nn | | +517% vs GEAK_v3 | |
| assign_score_withk | | | +3% vs GEAK Skill |
| points_in_boxes | | | +130% vs GEAK Skill |
| gather_points | | | +2% vs GEAK_v3 |
| silu | | | +2% vs GEAK Skill |
| matrix_multiplication | | +4% vs GEAK_v3 | |
| furthest_point_sample | FAIL | | +6% vs GEAK Skill |
| three_interpolate | | +14% vs GEAK_v3 | |

## Analysis

### Progression: GEAK_v3 → GEAK Skill → Team Skill

1. **GEAK_v3 → GEAK Skill (+22%)**: Running GEAK as a Claude Code skill improved reliability (0 failures vs 2) and overall speedup (4.30x → 5.26x). Key gains came from three_nn (1.43x → 8.82x) and knn (FAIL → 6.56x), where the skill's better error recovery and knowledge base enabled deeper algorithmic rewrites.

2. **GEAK Skill → Team Skill (+43%)**: The Team skill's multi-round iteration with re-profiling pushed overall speedup from 5.26x to 7.54x. The largest gains came from kernels where the bottleneck shifted after initial optimization:
   - **knn**: 6.56x → 23.00x — Round 1 matched GEAK's result (~6x), then re-profiling revealed the bottleneck shifted from compute to memory, leading to LDS tiling + multi-query workgroups in Round 2
   - **roipoint_pool3d**: 14.60x → 30.32x — More aggressive fusion eliminating all per-call hipMalloc/hipFree
   - **points_in_boxes**: 1.03x → 2.37x — Team applied wrapper-level `torch.empty` + fast math that GEAK missed

### Where GEAK Skill still leads

- **three_nn** (8.82x vs 3.57x): GEAK Skill's warp-cooperative kernel with `__shfl_xor` tree reduction outperformed Team's template + sorted insertion approach
- **ball_query** (12.64x vs 10.79x): GEAK Skill's LDS tiling with SOA layout was more effective
- **matrix_multiplication** (1.19x vs 1.09x): GEAK Skill used larger 128x128 tiles
- **three_interpolate** (1.15x vs 1.04x): Better vectorized loads + unrolling

### Key takeaway

The three systems form a clear progression in optimization capability. Each layer adds a distinct advantage:
- **GEAK_v3 → GEAK Skill**: Better reliability and error recovery through Claude Code integration
- **GEAK Skill → Team Skill**: Iterative re-profiling enables compounding optimizations across rounds

Team Skill's primary advantage is its ability to discover *new* bottlenecks after initial optimization. On kernels where a single round captures most gains (three_nn, matrix_multiplication), GEAK Skill performs comparably or better. The ideal system would combine Team's iterative loop with GEAK Skill's stronger single-round algorithmic exploration.
