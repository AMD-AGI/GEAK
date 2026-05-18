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
| **Date**            | 2026-05-13                               | 2026-05-13                                        | 2026-05-16                                                                      |
| **GPU**             | AMD MI300X (gfx942)                      | AMD MI300X (gfx942)                               | AMD MI300X (gfx942)                                                             |


## Per-Kernel Results (Arithmetic Mean Speedup)


| #   | Kernel                | GEAK_v3    | GEAK Skill  | Team Skill    | Best       |
| --- | --------------------- | ---------- | ----------- | ------------- | ---------- |
| 1   | knn                   | FAIL       | 6.56x       | **25.53x**    | Team       |
| 2   | roipoint_pool3d       | 16.82x     | 14.60x      | **21.27x**    | Team       |
| 3   | roiaware_pool3d       | 10.24x     | 9.91x       | **20.97x**    | Team       |
| 4   | three_nn              | 1.43x      | 8.82x       | **12.09x**    | Team       |
| 5   | ball_query            | 11.62x     | **12.64x**  | 6.01x         | GEAK Skill |
| 6   | assign_score_withk    | 3.76x      | 4.00x       | **4.04x**     | Team       |
| 7   | points_in_boxes       | 1.03x      | 1.03x       | **2.82x**     | Team       |
| 8   | three_interpolate     | 1.01x      | 1.15x       | **2.06x**     | Team       |
| 9   | gather_points         | 1.32x      | 0.96x       | **1.70x**     | Team       |
| 10  | furthest_point_sample | FAIL       | 1.05x       | **1.43x**     | Team       |
| 11  | silu                  | 1.21x      | **1.26x**   | 1.22x         | GEAK Skill |
| 12  | matrix_multiplication | 1.14x      | **1.19x**   | 1.10x         | GEAK Skill |


## Aggregate (12 Kernels, FAIL = 1.0x)


| Metric                 | GEAK_v3 | GEAK Skill | Team Skill |
| ---------------------- | ------- | ---------- | ---------- |
| **Arith Mean Speedup** | 4.30x   | 5.26x      | **8.35x**  |
| Wins (best of 3)       | 0       | 3          | **9**      |
| Failures               | 2       | 0          | 0          |


> Note: GEAK_v3 FAIL counted as 1.0x for arithmetic mean.

## Win Distribution


| Kernel                | GEAK_v3 | GEAK Skill         | Team Skill               |
| --------------------- | ------- | ------------------ | ------------------------ |
| knn                   | FAIL    |                    | **+289% vs GEAK Skill**  |
| roipoint_pool3d       |         |                    | **+46% vs GEAK_v3**      |
| roiaware_pool3d       |         |                    | **+105% vs GEAK_v3**     |
| three_nn              |         |                    | **+37% vs GEAK Skill**   |
| ball_query            |         | **+9% vs GEAK_v3** |                          |
| assign_score_withk    |         |                    | **+1% vs GEAK Skill**    |
| points_in_boxes       |         |                    | **+174% vs GEAK Skill**  |
| three_interpolate     |         |                    | **+79% vs GEAK Skill**   |
| gather_points         |         |                    | **+29% vs GEAK_v3**      |
| furthest_point_sample | FAIL    |                    | **+36% vs GEAK Skill**   |
| silu                  |         | **+3% vs GEAK_v3** |                          |
| matrix_multiplication |         | **+4% vs GEAK_v3** |                          |


## Analysis

### Progression: GEAK_v3 → GEAK Skill → Team Skill

1. **GEAK_v3 → GEAK Skill (+22%)**: Running GEAK as a Claude Code skill improved reliability (0 failures vs 2) and overall speedup (4.30x → 5.26x). Key gains came from three_nn (1.43x → 8.82x) and knn (FAIL → 6.56x), where the skill's better error recovery and knowledge base enabled deeper algorithmic rewrites.

2. **GEAK Skill → Team Skill (+59%)**: The Team skill's multi-round iteration with structured knowledge base and wrapper overhead detection pushed overall speedup from 5.26x to 8.35x. Key improvements:
   - **knn**: 6.56x → 25.53x (+289%) — Warp-cooperative 64-thread search with shared-memory merge tree, template K, wrapper optimization (autograd bypass, torch.empty, direct output format)
   - **roiaware_pool3d**: 9.91x → 20.97x (+112%) — Kernel re-parallelization (1 thread/box → 1 thread per point×box) + buffer caching + autograd bypass, triggered by wrapper overhead detection
   - **three_nn**: 8.82x → 12.09x (+37%) — Knowledge base guided engineers to warp-cooperative search pattern + sqrt fusion into HIP kernel
   - **three_interpolate**: 1.15x → 2.06x (+79%) — C++ fast path with single Python-to-C++ boundary crossing
   - **points_in_boxes**: 1.03x → 2.82x (+174%) — C++-side tensor allocation + remove overhead checks

### Where GEAK Skill still leads

- **ball_query** (12.64x vs 6.01x): GEAK Skill's LDS tiling with SOA layout was more effective than Team's warp-cooperative + early exit approach.
- **matrix_multiplication** (1.19x vs 1.10x): Both are bottlenecked by hipMemcpy/runtime init (~99% of measured time). Marginal difference.
- **silu** (1.26x vs 1.22x): Both near peak memory bandwidth (63.7% of 5.3 TB/s). Marginal difference.

### Team Skill key design advantages

1. **Wrapper overhead detection**: Automatic detection of overhead-bound scenarios (all test cases at similar latency) triggers PW-category engineer tasks targeting Python/C++ wrapper optimization. This drove breakthroughs on roiaware_pool3d, points_in_boxes, and gather_points.
2. **Structured knowledge base**: 7 knowledge files guide engineers to high-impact patterns (warp-cooperative for search kernels, template parameterization for oversized arrays, hipify-safe code patterns).
3. **Budget-controlled early exit**: Stops spending budget when diminishing returns detected (<5% improvement or 2 consecutive no-improvement rounds), saving time on already-efficient kernels.
4. **Director validation**: Independent re-measurement by Director catches measurement errors (all 13 cases validated within 10%).

### Key takeaway

The three systems show consistent progression in optimization capability:

- **GEAK_v3 → GEAK Skill**: Better reliability through Claude Code integration
- **GEAK Skill → Team Skill**: Structured knowledge base + wrapper overhead detection + iterative re-profiling + budget control

Team Skill wins 9 of 12 kernels, demonstrating broad improvement. Its largest gains come from kernels where wrapper overhead dominates (roiaware_pool3d, three_interpolate, points_in_boxes) or where the knowledge base guides engineers to warp-cooperative algorithms (three_nn, knn). GEAK Skill retains advantages on kernels where single-round deep algorithmic exploration suffices (ball_query).
