# HIP Kernel Optimization: GEAK_v3 vs GEAK Skill vs Team Skill

## Overview


|                     | GEAK_v3                                  | GEAK Skill                                        | Team Skill                                                                      |
| ------------------- | ---------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Architecture**    | Standalone agent + centralized evaluator | Single-agent orchestrator + workers (Claude Code) | Director → TechLead → Engineers + Merge Engineer (Claude Code)                  |
| **Iteration**       | Single round                             | 1 round, workers start from baseline              | Multi-round, budget-controlled, with wrapper overhead detection                 |
| **Re-profiling**    | No                                       | No                                                | Yes + bottleneck shift analysis + wrapper overhead detection                    |
| **Parallelism**     | Sequential                               | 2 parallel workers, dedicated GPU per worker      | Up to 3 parallel engineers, flock-based GPU locking, merge engineer per round   |
| **GPU requirement** | 1 GPU                                    | 2 GPUs per kernel (26 total for batch)            | 1 shared GPU per kernel                                                         |
| **Knowledge base**  | Built-in                                 | Built-in                                          | 7 knowledge files (MI300X, HIP, Triton, strategies, profiling, wrapper, self-monitoring) |
| **Date**            | 2026-05-13                               | 2026-05-13                                        | 2026-05-18                                                                      |
| **GPU**             | AMD MI300X (gfx942)                      | AMD MI300X (gfx942)                               | AMD MI300X (gfx942)                                                             |


## Per-Kernel Results (Arithmetic Mean Speedup)


| #   | Kernel                | GEAK_v3    | GEAK Skill  | Team Skill    | Best       |
| --- | --------------------- | ---------- | ----------- | ------------- | ---------- |
| 1   | knn                   | FAIL       | 6.56x       | **25.50x**    | Team       |
| 2   | roipoint_pool3d       | 16.82x     | 14.60x      | **24.76x**    | Team       |
| 3   | roiaware_pool3d       | 10.24x     | 9.91x       | **23.30x**    | Team       |
| 4   | three_nn              | 1.43x      | 8.82x       | **11.50x**    | Team       |
| 5   | ball_query            | 11.62x     | **12.64x**  | 10.82x        | GEAK Skill |
| 6   | assign_score_withk    | 3.76x      | 4.00x       | **4.02x**     | Team       |
| 7   | points_in_boxes       | 1.03x      | 1.03x       | **2.69x**     | Team       |
| 8   | three_interpolate     | 1.01x      | 1.15x       | **1.40x**     | Team       |
| 9   | gather_points         | 1.32x      | 0.96x       | **2.68x**     | Team       |
| 10  | furthest_point_sample | FAIL       | 1.05x       | **1.34x**     | Team       |
| 11  | silu                  | 1.21x      | **1.26x**   | 1.13x         | GEAK Skill |
| 12  | matrix_multiplication | 1.14x      | **1.19x**   | 1.10x         | GEAK Skill |


## Aggregate (12 Kernels, FAIL = 1.0x)


| Metric                 | GEAK_v3 | GEAK Skill | Team Skill |
| ---------------------- | ------- | ---------- | ---------- |
| **Arith Mean Speedup** | 4.30x   | 5.26x      | **9.19x**  |
| Wins (best of 3)       | 0       | 3          | **9**      |
| Failures               | 2       | 0          | 0          |


> Note: GEAK_v3 FAIL counted as 1.0x for arithmetic mean.

## Analysis

### Progression: GEAK_v3 → GEAK Skill → Team Skill

1. **GEAK_v3 → GEAK Skill (+22%)**: Running GEAK as a Claude Code skill improved reliability (0 failures vs 2) and overall speedup (4.30x → 5.26x). Key gains came from three_nn (1.43x → 8.82x) and knn (FAIL → 6.56x), where the skill's better error recovery and knowledge base enabled deeper algorithmic rewrites.

2. **GEAK Skill → Team Skill (+75%)**: The Team skill's multi-round iteration with structured knowledge base and wrapper overhead detection pushed overall speedup from 5.26x to 9.19x. Key improvements:
   - **knn**: 6.56x → 25.50x — Warp-cooperative search with shared-memory merge, template parameterization, wrapper optimization
   - **roipoint_pool3d**: 14.60x → 24.76x — Multi-round compounding with re-profiling after each round
   - **roiaware_pool3d**: 9.91x → 23.30x — Kernel re-parallelization + wrapper overhead detection
   - **three_nn**: 8.82x → 11.50x — Warp-cooperative search pattern + sqrt fusion
   - **points_in_boxes**: 1.03x → 2.69x — C++-side tensor allocation + wrapper overhead reduction
   - **gather_points**: 0.96x → 2.68x — Wrapper overhead detection redirected optimization to host code

### Where GEAK Skill still leads

- **ball_query** (12.64x vs 10.82x): GEAK Skill's LDS tiling with SOA layout was more effective than Team's approach.
- **matrix_multiplication** (1.19x vs 1.10x): Both bottlenecked by hipMemcpy/runtime init (~99% of measured time). Marginal difference.
- **silu** (1.26x vs 1.13x): Both near peak memory bandwidth. Marginal difference.

### Team Skill key design advantages

1. **Wrapper overhead detection**: Automatic detection of overhead-bound scenarios triggers wrapper optimization tasks. This drove breakthroughs on roiaware_pool3d, points_in_boxes, and gather_points.
2. **Multi-round re-profiling**: After each round, re-profile to detect bottleneck shifts and adapt strategy accordingly. This enabled compounding gains on roipoint_pool3d and roiaware_pool3d.
3. **Budget-controlled early exit**: Stops spending budget when diminishing returns detected, saving time on already-efficient kernels.
4. **Director validation**: Independent re-measurement catches measurement errors (all 12 cases validated within 10%).
