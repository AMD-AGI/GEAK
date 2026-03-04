---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "best-practices"
category: "performance"
tags: ["kernel", "optimization", "cuda"]
cuda_version: "13.0+"
last_updated: 2025-11-17
---

# Kernel Optimization for Nvidia GPUs

*Best practices for optimizing CUDA kernels*

## Occupancy Optimization

### Choose Optimal Block Size

```cuda
// Start with 256 threads per block
dim3 blockSize(256);
dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
kernel<<<gridSize, blockSize>>>(data, N);

// Or use occupancy calculator
int blockSize;
int minGridSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);
```

### Minimize Register Usage

```cuda
// Compile with register limit
nvcc --maxrregcount=64 kernel.cu

// Check register usage
nvcc --ptxas-options=-v kernel.cu
```

## Memory Access Patterns

### Coalescing

```cuda
// Good: Sequential access
__global__ void coalesced(float *out, float *in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx] * 2.0f;  // Coalesced
    }
}

// Bad: Strided access
__global__ void strided(float *out, float *in, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx * stride] * 2.0f;  // Not coalesced
    }
}
```

### Use Shared Memory

```cuda
__global__ void useShared(float *out, float *in) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load to shared memory
    shared[tid] = in[idx];
    __syncthreads();
    
    // Reuse data from shared memory
    float sum = 0.0f;
    for (int i = 0; i < blockDim.x; i++) {
        sum += shared[i];
    }
    
    out[idx] = sum;
}
```

## Loop Optimization

### Loop Unrolling

```cuda
// Manual unrolling
for (int i = 0; i < N; i += 4) {
    sum += data[i];
    sum += data[i+1];
    sum += data[i+2];
    sum += data[i+3];
}

// Pragma unrolling
#pragma unroll 4
for (int i = 0; i < N; i++) {
    sum += data[i];
}
```

## Best Practices

1. **Profile first**: Use Nsight Compute
2. **Optimize hot paths**: Focus on slow kernels
3. **Check occupancy**: Aim for >50%
4. **Minimize divergence**: Avoid if/else in warps
5. **Use intrinsics**: __fmaf, __expf, etc.

## External Resources

- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute](https://developer.nvidia.com/nsight-compute)

## Related Guides

- [GPU Optimization](gpu-optimization.md)
- [Memory Optimization](memory-optimization.md)
- [CUDA Profiling](../../layer-2-compute-stack/cuda/cuda-profiling.md)

