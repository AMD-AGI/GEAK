---
tags: ["optimization", "bf16", "hip", "silu", "vectorization"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/doxygen/html/hip__bfloat16_8h_source.html"
rocm_version: "6.0+"
last_updated: 2026-01-15
---

# BF16 Vector Load and Store Operations

## Overview

Vectorized load and store operations are fundamental to achieving high memory bandwidth utilization in HIP kernels, especially for element-wise operations like SiLU activation. The bfloat16 (bf16) data type, introduced in ROCm 5.x and standardized in ROCm 6.0+, provides a 16-bit floating-point format that maintains the range of fp32 while using half the memory bandwidth. For SiLU optimization, vectorized bf16 loads/stores can double throughput compared to scalar operations and achieve 2x bandwidth savings compared to fp32, making them critical for memory-bound kernels.

Vector operations in HIP allow processing multiple bf16 values in a single instruction, typically using `__hip_bfloat162` (2 elements), `float4` reinterpreted as 8 bf16 values, or custom vector types. These operations are essential when the computational intensity is low (as in element-wise operations) and memory bandwidth becomes the bottleneck.

## Technical Details

HIP provides several approaches for vectorized bf16 memory access. The most common patterns involve:

1. **Packed Vector Types**: Using `__hip_bfloat162` to load/store two bf16 values simultaneously. This type is defined in `hip_bfloat16.h` and provides direct hardware support on AMD GPUs.

2. **Reinterpreted Vector Types**: Using larger vector types (float2, float4) and reinterpreting them as multiple bf16 values. This approach can load 4-8 bf16 values per instruction.

3. **Memory Alignment**: Vector loads require proper memory alignment. For `__hip_bfloat162`, addresses must be 4-byte aligned. For float4 (8 bf16 values), 16-byte alignment is required.

Performance impact factors include:
- **Coalescing**: Consecutive threads accessing consecutive memory locations achieve peak bandwidth (70-90% of theoretical maximum)
- **Transaction Efficiency**: Vector loads reduce the number of memory transactions by up to 4-8x compared to scalar loads
- **Register Pressure**: Vectorization increases register usage, which may reduce occupancy if not carefully managed

For SiLU kernels, where each element requires computing `x * sigmoid(x)`, the memory bandwidth savings from bf16 vectorization directly translate to performance gains since the operation is memory-bound rather than compute-bound.

## Code Examples

### Example 1: Basic Vector Load/Store with __hip_bfloat162

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

__global__ void silu_bf16_vectorized(
    const __hip_bfloat162* __restrict__ input,
    __hip_bfloat162* __restrict__ output,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes 2 bf16 values
    if (idx < num_elements / 2) {
        // Vectorized load: reads 4 bytes (2 x bf16)
        __hip_bfloat162 val = input[idx];

        // Process the packed values (computation shown separately)
        // For demonstration, simple passthrough
        __hip_bfloat162 result = val;

        // Vectorized store: writes 4 bytes (2 x bf16)
        output[idx] = result;
    }
}

// Launch configuration for 1M elements
void launch_silu_bf16() {
    int num_elements = 1024 * 1024;
    int threads_per_block = 256;
    // Process 2 elements per thread
    int num_blocks = (num_elements / 2 + threads_per_block - 1) / threads_per_block;

    hipLaunchKernelGGL(silu_bf16_vectorized,
                       dim3(num_blocks),
                       dim3(threads_per_block),
                       0, 0,
                       d_input, d_output, num_elements);
}
```

### Example 2: Higher Vectorization with float4 (8 bf16 values)

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

__global__ void silu_bf16_vec8(
    const __hip_bfloat16* __restrict__ input,
    __hip_bfloat16* __restrict__ output,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base_idx = idx * 8;

    if (base_idx + 8 <= num_elements) {
        // Load 16 bytes (8 bf16 values) using float4
        float4 vec_data = reinterpret_cast<const float4*>(input)[idx];

        // Reinterpret as 8 bf16 values
        __hip_bfloat16* bf16_array = reinterpret_cast<__hip_bfloat16*>(&vec_data);

        // Process each bf16 value
        __hip_bfloat16 results[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            // Placeholder for actual computation
            results[i] = bf16_array[i];
        }

        // Store back as float4
        float4 output_vec;
        __hip_bfloat16* output_bf16 = reinterpret_cast<__hip_bfloat16*>(&output_vec);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            output_bf16[i] = results[i];
        }

        reinterpret_cast<float4*>(output)[idx] = output_vec;
    }

    // Handle remainder elements
    else if (base_idx < num_elements) {
        for (int i = 0; i < num_elements - base_idx; i++) {
            output[base_idx + i] = input[base_idx + i];
        }
    }
}
```

### Example 3: Memory-Aligned Vector Access Pattern

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Ensure proper alignment for vectorized access
__global__ void silu_bf16_aligned(
    const __hip_bfloat162* __restrict__ input,  // 4-byte aligned
    __hip_bfloat162* __restrict__ output,
    int num_pairs  // num_elements / 2
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop for better occupancy
    for (int idx = tid; idx < num_pairs; idx += stride) {
        // Coalesced vector load
        __hip_bfloat162 val = input[idx];

        // Process (actual SiLU computation would go here)
        __hip_bfloat162 result = val;

        // Coalesced vector store
        output[idx] = result;
    }
}

// Host code to ensure alignment
void allocate_aligned_bf16(void** ptr, size_t num_elements) {
    size_t bytes = num_elements * sizeof(__hip_bfloat16);
    // Align to 128 bytes for optimal coalescing
    size_t aligned_bytes = (bytes + 127) & ~127;
    hipMalloc(ptr, aligned_bytes);
}
```

## Best Practices

**Memory Alignment**: Always ensure bf16 arrays are properly aligned for vectorized access. Use `hipMalloc` which provides 256-byte alignment by default, sufficient for all vector types. For `__hip_bfloat162`, minimum 4-byte alignment is required; for float4 (8 bf16 values), 16-byte alignment is required.

**Vectorization Width Selection**: Choose vectorization width based on your workload characteristics. Use `__hip_bfloat162` (2 elements) as the baseline, which provides 2x bandwidth improvement with minimal complexity. Use float4 (8 elements) for very high-bandwidth applications, but be aware of increased register pressure and alignment requirements. Profile to find the optimal balance between vectorization and occupancy.

**Coalescing Optimization**: Structure your thread indexing so consecutive threads access consecutive memory locations. For a 1D array with vectorization factor V, thread T should access elements [T*V, (T+1)*V-1]. Avoid strided access patterns that can reduce effective bandwidth to 5-15% of peak.

**Boundary Handling**: Always implement proper boundary checks when the total number of elements is not a multiple of the vectorization width. Use a cleanup loop or conditional logic to handle remainder elements safely without buffer overruns.

**Register Pressure Management**: Vectorization increases register usage per thread. Monitor occupancy using `rocprof` or `--ptxas-options=-v` compilation flag. If occupancy drops below 50%, consider reducing vectorization width or using `__launch_bounds__` to guide the compiler.

**Common Pitfalls**:
- Unaligned memory access causes undefined behavior or performance degradation
- Over-vectorization can reduce occupancy and overall performance
- Ignoring remainder elements leads to incorrect results or memory violations
- Mixing scalar and vector operations on the same data can cause coherency issues

## Performance Considerations

For a SiLU kernel processing 1M bf16 elements:
- Scalar access: ~1M memory transactions
- `__hip_bfloat162` vectorized: ~500K transactions (2x reduction)
- float4 vectorized (8 bf16): ~125K transactions (8x reduction)

With proper coalescing, bf16 vectorization can achieve:
- 2x bandwidth savings vs fp32
- 70-90% of theoretical peak bandwidth
- Significant speedup for memory-bound element-wise operations

## References

- AMD Official Documentation: https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/doxygen/html/hip__bfloat16_8h_source.html
- ROCm Precision Support: https://rocm.docs.amd.com/en/latest/reference/precision-support.html
- HIP Performance Guidelines: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html
- Related APIs: `__hip_bfloat16`, `__hip_bfloat162`, `hip_bfloat16.h`, vector load/store intrinsics
