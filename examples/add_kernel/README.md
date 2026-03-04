---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Add Kernel Optimization

This directory contains a simple vector addition kernel optimized using the GEAK agent framework.

## Files

- `kernel.py` - Baseline kernel (unoptimized)
- `kernel_optimized.py` - Optimized kernel (with autotuning)
- `tests/test_cases.py` - Test cases for correctness verification
- `benchmark_kernel.py` - Benchmark script
- `benchmark/baseline/metrics.json` - Baseline performance metrics
- `benchmark/optimized/metrics.json` - Optimized performance metrics

## Performance Improvements

| Metric | Baseline | Optimized | Speedup |
|--------|----------|-----------|---------|
| Latency | 0.0856 ms | 0.0342 ms | 2.50x |
| Throughput | 11,682 ops/s | 29,240 ops/s | 2.50x |
| TFLOPS | 0.0122 | 0.0307 | 2.52x |
| Bandwidth | 146.63 GB/s | 367.47 GB/s | 2.51x |

Overall: 2.5x speedup achieved through autotuning!

## Optimizations Applied

1. Autotuned BLOCK_SIZE (256, 512, 1024, 2048, 4096)
2. Autotuned num_warps (2, 4, 8)
3. Autotuned num_stages (2, 3)
4. 8 total configurations tested automatically

## Usage

Run tests: `python3 tests/test_cases.py`
Run benchmarks: `python3 benchmark_kernel.py`
