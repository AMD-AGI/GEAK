# GEAK Kernel Optimization Comparison

## Overview

Comparison of GEAK_v3 and GEAK Skills Mode optimization results on AMD MI300X (gfx942) across 13 HIP kernels. All speedups are arithmetic mean across test cases.

## Setup

- **Hardware**: AMD MI300X (gfx942), 304 CUs
- **GEAK_v3**: Standalone GEAK agent with centralized evaluator
- **GEAK Skills Mode**: GEAK skills running on Claude Code (Opus), 2 parallel workers per kernel (each worker on a dedicated GPU), 3 optimization rounds, 13 kernels launched in parallel across 26 GPUs

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

## Reproduce

### Prerequisites

1. AMD MI300X GPU with ROCm 6.x installed
2. [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI
3. GEAK skill installed (see [README](../../README.md#geak-skills-mode-claude-code))
4. A kernel task directory with source file and Makefile

### Single Kernel

Copy the following prompt into Claude Code to optimize a single kernel (2 workers, 2 GPUs):

```
/geak --kernel_path /absolute/path/to/kernel.hip --repo_path /absolute/path/to/kernel_dir --num_parallel 2 --gpu_ids 0,1 --max_rounds 3
```

Example with a SiLU kernel:

```
/geak --kernel_path /home/user/kernel_tasks/silu/silu.hip --repo_path /home/user/kernel_tasks/silu --num_parallel 2 --gpu_ids 0,1 --max_rounds 3
```

If you only have 1 GPU available, use `--num_parallel 1 --gpu_ids 0`.

### Batch Run (Multiple Kernels)

To reproduce the full 13-kernel benchmark, paste this prompt into Claude Code. It will launch one GEAK optimization per kernel in parallel, each allocated a dedicated GPU pair:

```
Below is a list of kernel directories, each containing a .hip kernel file and a Makefile.
For each kernel, run the GEAK skill to optimize it. Launch all kernels in parallel using
separate background agents, each on its own dedicated GPU pair.

Kernel directories (under /path/to/kernel_tasks/):
- assign_score_withk
- ball_query
- furthest_point_sample
- gather_points
- knn
- matrix_multiplication
- mla_decode
- points_in_boxes
- roiaware_pool3d
- roipoint_pool3d
- silu
- three_interpolate
- three_nn

For each kernel directory DIR:
1. Find the .hip file: KERNEL=$(find /path/to/kernel_tasks/$DIR -name "*.hip" | head -1)
2. Run: /geak --kernel_path $KERNEL --repo_path /path/to/kernel_tasks/$DIR --num_parallel 2 --gpu_ids <assigned_gpu_pair>

Each kernel needs 2 dedicated GPUs (num_parallel=2). Assign GPU pairs sequentially
so no GPU is shared: kernel 0 gets GPUs 0,1; kernel 1 gets GPUs 2,3; etc.
This requires 26 GPUs total for 13 kernels.

After all optimizations complete, collect results from each kernel_eval/*/report/final_report.json
and generate a summary table with arithmetic mean speedup per kernel.
```

Replace `/path/to/kernel_tasks/` with the actual absolute path to your kernel task directory.

For fewer GPUs, reduce `num_parallel` to 1 (1 GPU per kernel = 13 GPUs total).
