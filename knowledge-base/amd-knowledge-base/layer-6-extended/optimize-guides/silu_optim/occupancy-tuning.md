Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "occupancy", "hip", "silu", "performance-tuning"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html"
rocm_version: "6.0+"
last_updated: 2026-01-15
---

# Occupancy Tuning for Element-wise Kernels

## Overview

Occupancy is the ratio of active wavefronts to the maximum supported wavefronts per Compute Unit (CU), and it directly impacts a GPU's ability to hide memory latency through parallel execution. For element-wise operations like SiLU, high occupancy is critical because these kernels are memory-bound: while waiting for memory transactions to complete (200-400 cycles), the GPU can context-switch to other wavefronts and maintain productive execution. Optimal occupancy for memory-bound kernels typically ranges from 75-100%, ensuring sufficient wavefronts are available to saturate memory bandwidth.

On AMD CDNA2 architecture (MI200 series), each CU can support up to 32-40 wavefronts depending on resource availability. Occupancy is limited by three primary resources: (1) available registers per CU (divided among all active wavefronts), (2) available Local Data Share (LDS/shared memory) per CU, and (3) maximum wavefronts per CU (architectural limit). For simple element-wise operations, register usage is typically the limiting factor, while LDS usage is minimal or zero.

Understanding and optimizing occupancy involves balancing resource usage per thread with parallelism. Higher register usage per thread allows more complex computations but reduces the number of concurrent wavefronts. For SiLU, the optimal configuration typically uses 20-32 registers per thread with 256 threads per block (4 wavefronts per block), achieving 75-100% occupancy and maximizing memory throughput.

## Technical Details

Occupancy is calculated as:
```
Occupancy = (Active Wavefronts per CU) / (Max Wavefronts per CU)
```

Resource limitations on AMD CDNA2 (MI200 series):
- **Max Wavefronts per CU**: 32-40 (architecture dependent)
- **Registers per CU**: 65536 vector registers (VGPRs)
- **LDS per CU**: 64 KB
- **Wavefront Size**: 64 threads

Register-limited occupancy calculation:
```
Registers per Wavefront = Registers per Thread × 64
Max Concurrent Wavefronts = 65536 / Registers per Wavefront
```

Example: If a kernel uses 32 registers per thread:
- Registers per wavefront = 32 × 64 = 2048
- Max wavefronts = 65536 / 2048 = 32 wavefronts
- Occupancy = 32 / 40 = 80% (good)

Launch bounds influence occupancy through:
```cpp
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_CU)
```
- **MAX_THREADS_PER_BLOCK**: Hint for expected block size, guides register allocation
- **MIN_BLOCKS_PER_CU**: Minimum blocks per CU to maintain, forces lower register usage per thread

For SiLU kernels:
- Target: 256 threads/block (4 wavefronts)
- Min blocks per CU: 4 (ensures 16 wavefronts per CU minimum)
- Expected register usage: 20-32 registers/thread
- Expected occupancy: 75-100%

Factors affecting occupancy:
1. **Register Spilling**: If kernel uses too many registers, compiler spills to local memory (very slow), reducing effective performance despite high occupancy
2. **Thread Block Size**: Smaller blocks (64-128 threads) allow more blocks per CU but may reduce per-block efficiency; larger blocks (512+ threads) may limit concurrent blocks
3. **LDS Usage**: Even small shared memory usage can reduce occupancy significantly if it pushes total LDS over 64 KB per CU
4. **Compiler Optimization**: `-O3` and architecture-specific flags impact register allocation

## Code Examples

### Example 1: Basic Occupancy Optimization with Launch Bounds

```cpp
#include <hip/hip_runtime.h>

// Well-tuned SiLU kernel with launch bounds
__launch_bounds__(256, 4)  // 256 threads/block, min 4 blocks per CU
__global__ void silu_optimized_occupancy(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop allows flexibility in grid size
    for (int i = idx; i < num_elements; i += stride) {
        float x = input[i];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float result = x * sigmoid_x;
        output[i] = result;
    }
}

// Query occupancy before launch
void check_occupancy() {
    int blockSize = 256;
    int minGridSize, gridSize;

    hipOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &gridSize,
        silu_optimized_occupancy,
        0,  // No dynamic shared memory
        0   // blockSizeLimit (0 = no limit)
    );

    std::cout << "Suggested block size: " << gridSize << "\n";
    std::cout << "Minimum grid size for max occupancy: " << minGridSize << "\n";

    // Get achieved occupancy
    int numBlocks = 1024;
    float occupancy;
    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        silu_optimized_occupancy,
        blockSize,
        0  // Dynamic shared memory
    );

    occupancy = (float)numBlocks / 40.0f;  // Assuming max 40 blocks per CU
    std::cout << "Achieved occupancy: " << (occupancy * 100.0f) << "%\n";
}
```

### Example 2: Trading Registers for Occupancy

```cpp
#include <hip/hip_runtime.h>

// High-occupancy variant: simpler computation, fewer registers
__launch_bounds__(128, 8)  // 128 threads/block, min 8 blocks per CU
__global__ void silu_high_occupancy(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        // Simplified sigmoid to reduce register pressure
        float x = input[idx];

        // Fast approximation using fewer registers
        float sigmoid_x = 0.5f + 0.25f * x - 0.02083f * x * x * x;
        sigmoid_x = fmaxf(0.0f, fminf(1.0f, sigmoid_x));  // Clamp

        output[idx] = x * sigmoid_x;
    }
}

// Medium-occupancy variant: better accuracy, more registers
__launch_bounds__(256, 4)  // 256 threads/block, min 4 blocks per CU
__global__ void silu_medium_occupancy(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        float x = input[idx];

        // More accurate sigmoid with more registers
        float exp_neg_x = expf(-x);
        float sigmoid_x = 1.0f / (1.0f + exp_neg_x);

        output[idx] = x * sigmoid_x;
    }
}

// Low-occupancy variant: maximum accuracy, many registers
__launch_bounds__(512, 2)  // 512 threads/block, min 2 blocks per CU
__global__ void silu_low_occupancy(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        float x = input[idx];

        // High-precision polynomial sigmoid (uses many registers)
        float x2 = x * x;
        float x3 = x2 * x;
        float x4 = x2 * x2;
        float x5 = x3 * x2;

        float sigmoid_x = 0.5f + 0.25f * x - 0.020833f * x3 + 0.003125f * x5;
        sigmoid_x = fmaxf(0.0f, fminf(1.0f, sigmoid_x));

        output[idx] = x * sigmoid_x;
    }
}
```

### Example 3: Dynamic Occupancy Selection

```cpp
#include <hip/hip_runtime.h>
#include <iostream>

// Benchmark different occupancy configurations
struct OccupancyConfig {
    int block_size;
    int min_blocks_per_cu;
    float elapsed_ms;
    float occupancy;
    float bandwidth_gb_s;
};

template<int BLOCK_SIZE, int MIN_BLOCKS>
__launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
__global__ void silu_template(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < num_elements; i += stride) {
        float x = input[i];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output[i] = x * sigmoid_x;
    }
}

OccupancyConfig benchmark_config(
    const float* d_input,
    float* d_output,
    int num_elements,
    int block_size,
    void (*kernel)(const float*, float*, int)
) {
    int blocks = (num_elements + block_size - 1) / block_size;

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Warmup
    hipLaunchKernelGGL(kernel, dim3(blocks), dim3(block_size),
                      0, 0, d_input, d_output, num_elements);
    hipDeviceSynchronize();

    // Measure
    hipEventRecord(start);
    for (int i = 0; i < 100; i++) {
        hipLaunchKernelGGL(kernel, dim3(blocks), dim3(block_size),
                          0, 0, d_input, d_output, num_elements);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float elapsed_ms;
    hipEventElapsedTime(&elapsed_ms, start, stop);
    elapsed_ms /= 100.0f;

    size_t bytes = num_elements * sizeof(float) * 2;
    float bandwidth = (bytes / 1e9) / (elapsed_ms / 1000.0f);

    hipEventDestroy(start);
    hipEventDestroy(stop);

    return {block_size, 0, elapsed_ms, 0.0f, bandwidth};
}

void find_optimal_occupancy() {
    int num_elements = 16 * 1024 * 1024;
    float *d_input, *d_output;
    hipMalloc(&d_input, num_elements * sizeof(float));
    hipMalloc(&d_output, num_elements * sizeof(float));

    std::cout << "Block Size\tElapsed(ms)\tBandwidth(GB/s)\n";
    std::cout << "==============================================\n";

    // Test different configurations
    auto cfg64 = benchmark_config(d_input, d_output, num_elements, 64,
                                  silu_template<64, 8>);
    auto cfg128 = benchmark_config(d_input, d_output, num_elements, 128,
                                   silu_template<128, 6>);
    auto cfg256 = benchmark_config(d_input, d_output, num_elements, 256,
                                   silu_template<256, 4>);
    auto cfg512 = benchmark_config(d_input, d_output, num_elements, 512,
                                   silu_template<512, 2>);

    std::cout << cfg64.block_size << "\t\t" << cfg64.elapsed_ms
              << "\t\t" << cfg64.bandwidth_gb_s << "\n";
    std::cout << cfg128.block_size << "\t\t" << cfg128.elapsed_ms
              << "\t\t" << cfg128.bandwidth_gb_s << "\n";
    std::cout << cfg256.block_size << "\t\t" << cfg256.elapsed_ms
              << "\t\t" << cfg256.bandwidth_gb_s << "\n";
    std::cout << cfg512.block_size << "\t\t" << cfg512.elapsed_ms
              << "\t\t" << cfg512.bandwidth_gb_s << "\n";

    hipFree(d_input);
    hipFree(d_output);
}
```

### Example 4: Profiling Register Usage

```cpp
#include <hip/hip_runtime.h>

// Kernel with explicit register management
__launch_bounds__(256, 4)
__global__ void silu_register_managed(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        float x = input[idx];

        // Minimize register pressure by reusing variables
        float temp = -x;           // Reuse temp variable
        temp = expf(temp);         // exp(-x)
        temp = 1.0f + temp;        // 1 + exp(-x)
        temp = 1.0f / temp;        // sigmoid(x)
        temp = x * temp;           // silu(x)

        output[idx] = temp;
    }
}

// Compile with: hipcc -O3 --save-temps -Xptxas=-v kernel.cpp
// Check output for: "used X registers, Y bytes smem"
```

## Best Practices

**Target 75-100% Occupancy for Memory-Bound Kernels**: For SiLU and similar element-wise operations, aim for high occupancy to hide memory latency. Use `__launch_bounds__(256, 4)` as a starting point, which typically achieves 80-100% occupancy.

**Use Occupancy Calculator**: Call `hipOccupancyMaxPotentialBlockSize()` to get compiler-recommended block size based on your kernel's resource usage. This accounts for actual register and shared memory consumption.

**Monitor Register Usage**: Compile with `-Xptxas=-v` or `--ptxas-options=-v` to see register count. Target 20-32 registers per thread for good occupancy. If >40 registers, simplify computations or use `__launch_bounds__` to force lower usage.

**Avoid Over-Optimization**: 100% occupancy isn't always fastest. Sometimes 75% occupancy with better per-thread efficiency (more registers for complex computation) yields higher throughput. Always profile to confirm.

**Balance Block Size and Grid Size**: For variable-sized workloads, use grid-stride loops with moderate block sizes (256-512) rather than massive blocks (1024+). This provides flexibility and avoids under-utilizing the GPU on small inputs.

**Don't Ignore Shared Memory**: Even if your kernel doesn't explicitly use shared memory, be aware that compiler might use it for register spilling or other purposes. Check LDS usage with profiling tools.

**Profile with rocprof**: Use `rocprof --stats` to measure actual occupancy during execution. Compare against theoretical maximum to identify gaps. Low occupancy (<50%) suggests resource limitations need addressing.

**Common Pitfalls**:
- Assuming maximum occupancy always gives best performance
- Not checking register usage before deploying kernels
- Using large shared memory unnecessarily
- Ignoring the MIN_BLOCKS_PER_CU parameter in `__launch_bounds__`
- Over-complicating kernels to the point where register pressure kills occupancy

## Performance Impact

Occupancy impact on SiLU performance (MI250X, 16M elements):

**Low Occupancy (25-40%, 512 threads/block, high register use)**:
- Elapsed: 80-100 μs
- Bandwidth: 800-1000 GB/s
- Issue: Memory latency not fully hidden

**Medium Occupancy (50-75%, 256 threads/block)**:
- Elapsed: 55-70 μs
- Bandwidth: 1100-1300 GB/s
- Good balance for most kernels

**High Occupancy (75-100%, 128-256 threads/block, low register use)**:
- Elapsed: 45-60 μs
- Bandwidth: 1200-1500 GB/s
- Optimal for memory-bound operations

**Over-Optimized (100% occupancy, 64 threads/block, minimal registers)**:
- Elapsed: 50-65 μs
- Bandwidth: 1150-1350 GB/s
- Not always faster due to reduced per-thread efficiency

**Key Insight**: For memory-bound SiLU, 75-90% occupancy is the sweet spot, providing enough parallelism to saturate memory bandwidth without over-constraining the kernel.

## References

- AMD Official Documentation: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html
- HIP Programming Model: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
- Hardware Implementation: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- Related APIs: `__launch_bounds__`, `hipOccupancyMaxPotentialBlockSize`, `hipOccupancyMaxActiveBlocksPerMultiprocessor`, register optimization
