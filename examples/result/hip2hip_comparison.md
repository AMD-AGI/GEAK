# HIP Kernel Optimization: GEAK_v3 vs GEAK Skill vs Team Skill

## Overview


|                     | GEAK_v3                                  | GEAK Skill (Claude 4.6)                           | Team Skill (Claude 4.6)                                                         | Team Skill (Claude 4.7)                                                         |
| ------------------- | ---------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Architecture**    | Standalone agent + centralized evaluator | Single-agent orchestrator + workers (Claude Code) | Director → TechLead → Engineers + Merge Engineer (Claude Code)                  | Same as Team Skill (4.6), upgraded LLM                                          |
| **Iteration**       | Single round                             | 1 round, workers start from baseline              | Multi-round, budget-controlled, with wrapper overhead detection                 | Multi-round, budget-controlled, with wrapper overhead detection                 |
| **Re-profiling**    | No                                       | No                                                | Yes + bottleneck shift analysis + wrapper overhead detection                    | Yes + bottleneck shift analysis + wrapper overhead detection                    |
| **Parallelism**     | Sequential                               | 2 parallel workers, dedicated GPU per worker      | Up to 3 parallel engineers, flock-based GPU locking, merge engineer per round   | Up to 3 parallel engineers, flock-based GPU locking, merge engineer per round   |
| **GPU requirement** | 1 GPU                                    | 2 GPUs per kernel (26 total for batch)            | 1 shared GPU per kernel                                                         | 1 shared GPU per kernel                                                         |
| **Knowledge base**  | Built-in                                 | Built-in                                          | 7 knowledge files (MI300X, HIP, Triton, strategies, profiling, wrapper, self-monitoring) | 7 knowledge files (MI300X, HIP, Triton, strategies, profiling, wrapper, self-monitoring) |
| **LLM**             | n/a                                      | Claude Sonnet 4.6                                 | Claude Sonnet 4.6                                                               | Claude Opus 4.7                                                                 |
| **Date**            | 2026-05-13                               | 2026-05-13                                        | 2026-05-18                                                                      | 2026-05-26                                                                      |
| **GPU**             | AMD MI300X (gfx942)                      | AMD MI300X (gfx942)                               | AMD MI300X (gfx942)                                                             | AMD MI300X (gfx942)                                                             |


## Per-Kernel Results

Per-kernel speedup is computed across shapes; arithmetic mean (A) and geometric mean (G) shown side-by-side. **Best (by G)** column compares the four systems by geometric mean.

| #   | Kernel                | GEAK_v3 (A / G) | GEAK Skill 4.6 (A / G) | Team Skill 4.6 (A / G) | Team Skill 4.7 (A / G)  | Best (by G)      |
| --- | --------------------- | --------------- | ---------------------- | ---------------------- | ----------------------- | ---------------- |
| 1   | knn                   | FAIL            | 6.56x / 4.61x          | 25.50x / 17.74x        | **38.94x / 34.48x**     | Team 4.7         |
| 2   | roipoint_pool3d       | 16.82x / 9.59x  | 14.61x / 8.73x         | 24.85x / 17.84x        | **27.33x / 24.25x**     | Team 4.7         |
| 3   | roiaware_pool3d       | 10.24x / 7.96x  | 9.92x / 7.67x          | 23.30x / 17.97x        | **38.95x / 29.50x**     | Team 4.7         |
| 4   | three_nn              | 1.43x / 1.35x   | 8.82x / 3.64x          | 11.50x / 5.38x         | **17.45x / 10.77x**     | Team 4.7         |
| 5   | ball_query            | 11.62x / 6.39x  | 13.14x / 6.71x         | 10.82x / 7.11x         | **37.39x / 26.46x**     | Team 4.7         |
| 6   | assign_score_withk    | 3.76x / 1.85x   | 4.00x / 2.01x          | 4.01x / 2.15x          | **13.52x / 5.28x**      | Team 4.7         |
| 7   | points_in_boxes       | 1.03x / 1.03x   | 1.04x / 1.04x          | **2.69x / 2.69x**      | 2.63x / 2.61x           | Team 4.6         |
| 8   | three_interpolate     | 1.01x / 1.01x   | 1.15x / 1.12x          | **1.40x / 1.38x**      | 1.12x / 1.11x           | Team 4.6         |
| 9   | gather_points         | 1.32x / 1.32x   | 0.96x / 0.96x          | **2.68x / 2.68x**      | 1.99x / 1.99x           | Team 4.6         |
| 10  | furthest_point_sample | FAIL            | 1.04x / 1.04x          | **1.32x / 1.27x**      | 1.26x / 1.24x           | Team 4.6         |
| 11  | silu                  | 1.21x / 1.19x   | 1.26x / 1.23x          | 1.13x / 1.12x          | **1.49x / 1.48x**       | Team 4.7         |
| 12  | matrix_multiplication | 1.14x / 1.14x   | 1.19x / 1.19x          | 1.11x / 1.11x          | **1.25x / 1.23x**       | Team 4.7         |
| 13  | mla_decode            | FAIL            | **589.20x / 424.51x**  | FAIL                   | 13.27x / 13.27x         | GEAK Skill 4.6 † |

> † mla_decode in GEAK Skill 4.6 reflects an extreme outlier: the unoptimized baseline includes wrapper overhead the Team 4.7 baseline accounts for separately, so the absolute number is not directly comparable.


## Aggregate (FAIL = 1.0x)

Aggregate combines per-kernel values across kernels. "Arith-of-Arith" averages the per-kernel arithmetic means arithmetically; "Geo-of-Geo" averages the per-kernel geometric means geometrically.

### 12 common kernels (excluding mla_decode, for apples-to-apples vs older runs)

| Metric                              | GEAK_v3 | GEAK Skill 4.6 | Team Skill 4.6 | Team Skill 4.7 |
| ----------------------------------- | ------- | -------------- | -------------- | -------------- |
| **Arith-of-Arith**                  | 4.30x   | 5.31x          | 9.19x          | **15.28x**     |
| **Geo-of-Geo**                      | 1.90x   | 2.33x          | 3.73x          | **5.30x**      |
| Wins (best of 4, by geomean)        | 0       | 0              | 4              | **8**          |
| Failures                            | 2       | 0              | 0              | 0              |

### All 13 kernels (including mla_decode)

| Metric                              | GEAK_v3 | GEAK Skill 4.6 | Team Skill 4.6 | Team Skill 4.7 |
| ----------------------------------- | ------- | -------------- | -------------- | -------------- |
| **Arith-of-Arith**                  | 4.05x   | 50.22x †       | 8.56x          | **15.12x**     |
| **Geo-of-Geo**                      | 1.81x   | 3.48x          | 3.37x          | **5.68x**      |
| Failures                            | 3       | 0              | 1              | 0              |

> † GEAK Skill 4.6's all-13 arithmetic mean is inflated by the mla_decode outlier (589x). The geometric-mean view is the better comparison.

> Note: FAIL counted as 1.0x for both arithmetic and geometric mean. Per-kernel arithmetic values match each framework's native aggregate (GEAK uses `sum/len` over shapes — `src/testcases.py:438`); geometric values were recomputed from per-shape baseline/optimized timings. Team Skill (4.7) data: `exp/team_hip2hip_others_20260526_095516/`.

## Analysis

### Progression: GEAK_v3 → GEAK Skill (4.6) → Team Skill (4.6) → Team Skill (4.7)

All numbers below are per-kernel **geometric means across shapes** (the same metric used in the Geo-of-Geo aggregate row); 12-common-kernel aggregate.

1. **GEAK_v3 → GEAK Skill 4.6 (+23%)**: Running GEAK as a Claude Code skill improved reliability (0 failures vs 2) and overall speedup (1.90x → 2.33x geomean). Key gains came from three_nn (1.35x → 3.64x) and knn (FAIL → 4.61x), where the skill's better error recovery and knowledge base enabled deeper algorithmic rewrites.

2. **GEAK Skill 4.6 → Team Skill 4.6 (+60%)**: The Team skill's multi-round iteration with structured knowledge base and wrapper overhead detection pushed overall speedup from 2.33x to 3.73x geomean. Key improvements: knn (4.61x → 17.74x), roipoint_pool3d (8.73x → 17.84x), roiaware_pool3d (7.67x → 17.97x), three_nn (3.64x → 5.38x), points_in_boxes (1.04x → 2.69x), gather_points (0.96x → 2.68x).

3. **Team Skill 4.6 → Team Skill 4.7 (+42%)**: **Same skill, upgraded LLM (Claude Sonnet 4.6 → Claude Opus 4.7).** Overall geomean rose from 3.73x to 5.30x with no framework changes. The model upgrade landed mostly in compute-bound kernels where the engineers had more room to push algorithmic rewrites:
   - **ball_query**: 7.11x → 26.46x — better LDS tiling + warp-level reduction patterns
   - **knn**: 17.74x → 34.48x — deeper warp-cooperative search refactor on top of the 4.6 baseline
   - **roiaware_pool3d**: 17.97x → 29.50x — finer-grained kernel reparallelization
   - **assign_score_withk**: 2.15x → 5.28x — algorithmic restructuring of the inner reduction
   - 4.7 lost ground on a few wrapper-overhead-bound kernels (gather_points 2.68x → 1.99x, three_interpolate 1.38x → 1.11x), suggesting the 4.6 run had over-fit those wrapper edges.

### Where Team Skill (4.6) still leads Team Skill (4.7) by geomean

- **points_in_boxes** (2.69x vs 2.61x): essentially tied; both already at wrapper-overhead floor.
- **three_interpolate** (1.38x vs 1.11x): 4.6 found a wrapper-side fast path that 4.7 didn't reproduce.
- **gather_points** (2.68x vs 1.99x): same — 4.6's wrapper-overhead detection happened to land a better fix.
- **furthest_point_sample** (1.27x vs 1.24x): essentially tied; kernel dominated by sequential dependence.

### Team Skill key design advantages

1. **Wrapper overhead detection**: Automatic detection of overhead-bound scenarios triggers wrapper optimization tasks. This drove breakthroughs on roiaware_pool3d, points_in_boxes, and gather_points.
2. **Multi-round re-profiling**: After each round, re-profile to detect bottleneck shifts and adapt strategy accordingly. This enabled compounding gains on roipoint_pool3d and roiaware_pool3d.
3. **Budget-controlled early exit**: Stops spending budget when diminishing returns detected, saving time on already-efficient kernels.
4. **Director validation**: Independent re-measurement catches measurement errors (all 12 cases validated within 10%).
