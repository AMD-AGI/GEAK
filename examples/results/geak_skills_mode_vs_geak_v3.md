# GEAK Kernel Optimization Comparison

## Overview

Comparison of GEAK_v3 and GEAK Skills Mode optimization results on AMD MI300X (gfx942) across 13 HIP kernels. All speedups are arithmetic mean across test cases.

## Setup

- **Hardware**: AMD MI300X (gfx942), 304 CUs
- **GEAK_v3**: Standalone GEAK agent with centralized evaluator
- **GEAK Skills Mode**: GEAK skills running on Claude Code (Opus), 2 parallel workers per kernel, 3 optimization rounds

## Per-Kernel Results

| Kernel | GEAK_v3 | Skills Mode | Winner |
|--------|---------|-------------|--------|
| roipoint_pool3d | 16.82x | 14.61x | GEAK_v3 |
| ball_query | 11.62x | 13.14x | Skills Mode |
| roiaware_pool3d | 10.24x | 9.92x | GEAK_v3 |
| three_nn | 1.43x | 8.82x | Skills Mode |
| knn | FAIL | 6.56x | Skills Mode |
| assign_score_withk | 3.76x | 4.00x | Skills Mode |
| silu | 1.21x | 1.26x | Skills Mode |
| matrix_multiplication | 1.14x | 1.19x | Skills Mode |
| three_interpolate | 1.01x | 1.15x | Skills Mode |
| furthest_point_sample | FAIL | 1.04x | Skills Mode |
| points_in_boxes | 1.03x | 1.04x | Tie |
| gather_points | 1.32x | 0.96x | GEAK_v3 |
| mla_decode | N/A | 586.00x | Skills Mode |

## Summary

| Metric | GEAK_v3 | Skills Mode |
|--------|---------|-------------|
| Wins | 3 | 9 |
| Failures | 2 | 0 |
| Arith Mean Speedup (12 common, fail=1.0x) | 4.30x | 5.31x |
| Arith Mean Speedup (10 common, excl failures) | 4.96x | 5.61x |

## Analysis

- **Skills Mode large wins**: three_nn (1.43x vs 8.82x, 6x gap) and knn (FAIL vs 6.56x). Both kernels benefited from deeper algorithmic restructuring including warp-cooperative algorithms and K-split parallelization.
- **GEAK_v3 wins**: roipoint_pool3d (16.82x vs 14.61x), roiaware_pool3d (10.24x vs 9.92x), gather_points (1.32x vs 0.96x). GEAK_v3 achieved better memory access optimization on these kernels.
- **Reliability**: Skills Mode produced valid results on all 13 kernels. GEAK_v3 failed on 2 kernels (knn, furthest_point_sample).
- **Overall**: On the 12 common kernels, Skills Mode achieved 5.31x arithmetic mean speedup vs GEAK_v3's 4.30x, a 23% lead.
