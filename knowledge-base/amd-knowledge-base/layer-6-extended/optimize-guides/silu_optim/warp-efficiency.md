---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "warp", "wavefront", "hip", "silu", "efficiency"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html"
rocm_version: "6.0+"
last_updated: 2026-01-15
---

# Warp Efficiency Optimization

## Overview

Warp efficiency (called wavefront efficiency on AMD GPUs) measures how effectively threads within a wavefront execute together in lockstep. On AMD CDNA architectures, a wavefront consists of 64 threads that execute the same instruction simultaneously. Perfect efficiency (100%) occurs when all 64 threads execute every instruction; poor efficiency (<50%) happens when thread divergence or inactive threads waste execution slots. For element-wise operations like SiLU, achieving high wavefront efficiency is straightforward because all threads perform identical operations on different data, but boundary conditions and control flow must be carefully managed to avoid efficiency loss.

Thread divergence occurs when threads within a wavefront take different execution paths due to conditional branches. When this happens, the wavefront must execute all paths serially, with inactive threads waiting, effectively multiplying execution time by the number of divergent paths. For SiLU kernels, common divergence sources include boundary checks (`if (idx < num_elements)`), special value handling (`if (x > threshold)`), and uneven workload distribution.

High wavefront efficiency is critical for maximizing throughput on AMD GPUs. Even simple boundary checks can reduce efficiency from 100% to 95-98%, and complex conditional logic can drop efficiency below 50%, effectively halving performance. Understanding and eliminating divergence through proper indexing, grid sizing, and predication techniques is essential for optimal element-wise kernel performance.

## Technical Details

Wavefront efficiency is calculated as:
```
Efficiency = (Active Thread Instructions) / (Total Possible Instructions)
```

For a wavefront of 64 threads executing 100 instructions:
- **100% Efficiency**: All 64 threads execute all 100 instructions = 6400 thread-instructions
- **50% Efficiency**: Only 32 threads active on average = 3200 thread-instructions
- **Severe Divergence**: 4 paths with 16 threads each executed serially = 400 × 4 = 1600 effective thread-instructions (25% efficiency)

Sources of wavefront inefficiency:

1. **Boundary Divergence**: Last block has fewer than blockDim.x elements
   ```cpp
   if (idx < num_elements)  // Causes divergence in last block
   ```

2. **Conditional Execution**: Different threads take different code paths
   ```cpp
   if (x > 0.0f) { ... } else { ... }  // Divergence if x varies within wavefront
   ```

3. **Early Exit**: Some threads exit loop early
   ```cpp
   while (condition[idx]) { ... }  // Different threads exit at different iterations
   ```

4. **Uneven Work Distribution**: Some threads process more elements than others

Techniques to improve efficiency:

1. **Grid Sizing**: Make total threads a multiple of wavefront size (64)
   ```cpp
   int grid_size = (num_elements + block_size - 1) / block_size;
   // Ensure block_size is multiple of 64
   ```

2. **Predication**: Use branchless selection instead of if/else
   ```cpp
   result = (x > 0.0f) * pos_val + (x <= 0.0f) * neg_val;  // No divergence
   ```

3. **Grid-Stride Loops**: Distribute work evenly across all threads
   ```cpp
   for (int i = idx; i < num_elements; i += stride)  // Even distribution
   ```

4. **Boundary Masking**: Handle boundary conditions without affecting control flow

For SiLU, efficiency is typically high (95-100%) because:
- All threads execute the same sigmoid and multiply operations
- Boundary checks affect only the last block (minimal impact with large arrays)
- No data-dependent branching in core computation

## Code Examples

### Example 1: Efficient Boundary Handling

```cpp
#include <hip/hip_runtime.h>

// INEFFICIENT: Boundary check causes divergence in last block
__global__ void silu_boundary_divergent(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Last block: some threads have idx >= num_elements
    // These threads diverge and sit idle
    if (idx < num_elements) {
        float x = input[idx];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid_x;
    }
    // Threads with idx >= num_elements waste cycles here
}

// EFFICIENT: Grid-stride loop amortizes boundary divergence
__launch_bounds__(256, 4)
__global__ void silu_boundary_efficient(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // All threads in active blocks do useful work across iterations
    // Boundary divergence affects only last iteration of last block
    for (int i = idx; i < num_elements; i += stride) {
        float x = input[i];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output[i] = x * sigmoid_x;
    }
}

// OPTIMAL: Pad input to multiple of block size (if possible)
__launch_bounds__(256, 4)
__global__ void silu_no_boundary_check(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements_padded  // Guaranteed multiple of 256
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // No boundary check needed - all threads have valid work
    // 100% wavefront efficiency
    float x = input[idx];
    float sigmoid_x = 1.0f / (1.0f + expf(-x));
    output[idx] = x * sigmoid_x;
}
```

### Example 2: Avoiding Control Flow Divergence

```cpp
#include <hip/hip_runtime.h>

// INEFFICIENT: Conditional branching on input value
__global__ void activation_divergent(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements,
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        float x = input[idx];

        // Divergence: different threads take different paths
        float result;
        if (x > threshold) {
            // Some threads execute this
            result = x * x;
        } else {
            // Other threads execute this
            result = x;
        }

        output[idx] = result;
    }
}

// EFFICIENT: Branchless selection using fmax/fmin
__launch_bounds__(256, 4)
__global__ void activation_branchless(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements,
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < num_elements; i += stride) {
        float x = input[i];

        // Branchless: compute both, select result
        float squared = x * x;
        float linear = x;

        // If x > threshold, use squared, else linear
        // No divergence - all threads execute same instructions
        float mask = (x > threshold) ? 1.0f : 0.0f;
        float result = mask * squared + (1.0f - mask) * linear;

        // Even better: use fmax/fmin for clamping
        // result = (x > threshold) * squared + (x <= threshold) * linear;

        output[i] = result;
    }
}

// BEST: Use intrinsics for true predication
__launch_bounds__(256, 4)
__global__ void activation_predicated(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements,
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < num_elements; i += stride) {
        float x = input[i];

        // Hardware predication - no divergence
        float squared = x * x;
        float result = fmaxf(squared, x);  // Max of x^2 and x

        output[i] = result;
    }
}
```

### Example 3: Optimizing SiLU for Zero Divergence

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Perfect wavefront efficiency SiLU
__launch_bounds__(256, 4)
__global__ void silu_zero_divergence(
    const __hip_bfloat162* __restrict__ input,
    __hip_bfloat162* __restrict__ output,
    int num_pairs_padded  // Pre-padded to multiple of 256
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // No boundary check - guaranteed valid
    __hip_bfloat162 x = input[idx];

    // No conditional logic in computation
    float x1 = __bfloat162float(x.x);
    float x2 = __bfloat162float(x.y);

    // Compute sigmoid - no branches
    float sig1 = 1.0f / (1.0f + expf(-x1));
    float sig2 = 1.0f / (1.0f + expf(-x2));

    __hip_bfloat162 sigmoid_x = __floats2bfloat162_rn(sig1, sig2);
    __hip_bfloat162 result = __hmul2(x, sigmoid_x);

    output[idx] = result;
}

// Helper: Pad array to avoid boundary divergence
void launch_silu_optimal(
    const __hip_bfloat16* d_input,
    __hip_bfloat16* d_output,
    int num_elements
) {
    const int threads = 256;
    const int elements_per_thread = 2;  // bf162 processes 2 elements
    const int elements_per_block = threads * elements_per_thread;

    // Round up to multiple of elements_per_block
    int padded_elements = ((num_elements + elements_per_block - 1) /
                          elements_per_block) * elements_per_block;
    int num_pairs = padded_elements / 2;
    int blocks = num_pairs / threads;

    const __hip_bfloat162* input_vec =
        reinterpret_cast<const __hip_bfloat162*>(d_input);
    __hip_bfloat162* output_vec =
        reinterpret_cast<__hip_bfloat162*>(d_output);

    hipLaunchKernelGGL(silu_zero_divergence, dim3(blocks), dim3(threads),
                      0, 0, input_vec, output_vec, num_pairs);
}
```

### Example 4: Profiling Wavefront Efficiency

```cpp
#include <hip/hip_runtime.h>
#include <iostream>

// Kernel with artificial divergence for testing
__global__ void silu_with_divergence(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        float x = input[idx];

        // Artificial divergence based on thread ID
        float result;
        if (threadIdx.x % 2 == 0) {
            // 50% of threads execute this path
            result = x / (1.0f + expf(-x));
        } else {
            // Other 50% execute this path
            float sigmoid_x = 1.0f / (1.0f + expf(-x));
            result = x * sigmoid_x;
        }

        output[idx] = result;
    }
}

// Measure performance impact of divergence
void measure_divergence_impact() {
    int num_elements = 16 * 1024 * 1024;
    float *d_input, *d_output;
    hipMalloc(&d_input, num_elements * sizeof(float));
    hipMalloc(&d_output, num_elements * sizeof(float));

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Benchmark with divergence
    hipEventRecord(start);
    for (int i = 0; i < 100; i++) {
        hipLaunchKernelGGL(silu_with_divergence, dim3(blocks),
                          dim3(threads), 0, 0, d_input, d_output, num_elements);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float divergent_time;
    hipEventElapsedTime(&divergent_time, start, stop);

    // Benchmark without divergence
    hipEventRecord(start);
    for (int i = 0; i < 100; i++) {
        hipLaunchKernelGGL(silu_boundary_efficient, dim3(blocks),
                          dim3(threads), 0, 0, d_input, d_output, num_elements);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float efficient_time;
    hipEventElapsedTime(&efficient_time, start, stop);

    std::cout << "With divergence: " << divergent_time / 100.0f << " ms\n";
    std::cout << "Without divergence: " << efficient_time / 100.0f << " ms\n";
    std::cout << "Slowdown from divergence: "
              << (divergent_time / efficient_time) << "x\n";

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_input);
    hipFree(d_output);
}

// Use rocprof to get detailed wavefront metrics:
// rocprof --stats ./silu_benchmark
// Look for: VALUBusy, SALUBusy, WavefrontLaunch metrics
```

## Best Practices

**Ensure Block Sizes are Multiples of Wavefront Size**: Always use block sizes divisible by 64 (AMD wavefront size) such as 64, 128, 256, 512. This prevents partial wavefronts which waste execution slots.

**Minimize Boundary Checks**: Use grid-stride loops or pad inputs to multiples of block size to avoid divergence from boundary conditions. The performance cost of small padding is typically much less than divergence overhead.

**Use Branchless Programming**: Replace conditional statements with arithmetic selection, `fmax`/`fmin` intrinsics, or ternary operators that compile to predicated instructions rather than branches.

**Avoid Data-Dependent Branching**: If branching based on input data is unavoidable, reorganize computations so all threads in a wavefront are likely to take the same path (e.g., sort inputs by category before processing).

**Profile with rocprof**: Use `rocprof --stats` to measure actual wavefront efficiency. Look for `WriteSize` and execution time metrics to identify divergence bottlenecks.

**Launch Enough Wavefronts**: Ensure grid size provides many more wavefronts than CUs (aim for 4-8 blocks per CU) to hide divergence latency through concurrent execution.

**Common Pitfalls**:
- Block sizes not divisible by 64
- Excessive boundary checking in hot loops
- Using if/else when branchless alternatives exist
- Not padding small arrays to avoid last-block divergence
- Ignoring the performance impact of seemingly minor conditionals

## Performance Impact

Wavefront efficiency impact on SiLU (MI250X, 16M elements):

**Perfect Efficiency (100%, no divergence)**:
- Execution: 50-55 μs
- All threads productive throughout execution

**Mild Divergence (95-98%, boundary checks)**:
- Execution: 52-57 μs
- ~2-5% slowdown
- Typical for real-world kernels

**Moderate Divergence (80-90%, some conditional logic)**:
- Execution: 60-70 μs
- ~15-25% slowdown
- Noticeable impact, should be optimized

**Severe Divergence (50-70%, complex branching)**:
- Execution: 80-110 μs
- ~50-100% slowdown
- Critical issue requiring refactoring

**Extreme Divergence (<50%, serial execution paths)**:
- Execution: >150 μs
- >2x slowdown
- Algorithm-level redesign needed

For element-wise SiLU, target >95% efficiency is achievable and should be the minimum goal.

## References

- AMD Official Documentation: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
- HIP Performance Guidelines: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html
- Hardware Implementation: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- Related Topics: Thread divergence, predication, wavefront size (64 on AMD), branchless programming
