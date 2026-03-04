---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "best-practices"
category: "performance"
subcategory: "memory"
tags: ["memory", "optimization", "performance", "bandwidth"]
rocm_version: "7.0+"
last_updated: 2025-11-01
---

# Memory Optimization for AMD GPUs

## Memory Hierarchy

AMD GPUs (CDNA/RDNA) have:
1. **Registers**: ~256KB per CU, fastest
2. **LDS (Local Data Share)**: 64-128KB per CU, shared memory
3. **L1 Cache**: 16-32KB per CU
4. **L2 Cache**: 4-8MB, shared across GPU
5. **HBM (High Bandwidth Memory)**: 32-128GB, 1.6-3.2 TB/s

## Coalesced Memory Access

```cpp
// BAD: Strided access
__global__ void uncoalesced(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Each thread accesses data 64 elements apart
    data[idx * 64] = idx;  // BAD!
}

// GOOD: Sequential access
__global__ void coalesced(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Sequential access pattern
    data[idx] = idx;  // GOOD!
}
```

## Use Shared Memory (LDS)

```cpp
#define TILE_SIZE 16

__global__ void tiled_matmul(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < N && t * TILE_SIZE + tx < N)
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (col < N && t * TILE_SIZE + ty < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute from shared memory (fast!)
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

## Avoid Bank Conflicts

```cpp
// BAD: Bank conflicts
__shared__ float shared[256];
shared[threadIdx.x] = data[threadIdx.x];
float val = shared[threadIdx.x + 1];  // May cause conflicts

// GOOD: Padding to avoid conflicts
__shared__ float shared[256 + 1];  // Extra padding
shared[threadIdx.x] = data[threadIdx.x];
float val = shared[threadIdx.x + 1];  // No conflicts
```

## Use Async Memory Copy

```cpp
// Overlap data transfer with computation
hipStream_t streams[2];
hipStreamCreate(&streams[0]);
hipStreamCreate(&streams[1]);

for (int i = 0; i < num_batches; i++) {
    int s = i % 2;
    
    // Async copy
    hipMemcpyAsync(d_data[s], h_data[i], size, 
                   hipMemcpyHostToDevice, streams[s]);
    
    // Launch kernel (overlaps with copy)
    kernel<<<grid, block, 0, streams[s]>>>(d_data[s]);
    
    // Async copy back
    hipMemcpyAsync(h_result[i], d_result[s], size,
                   hipMemcpyDeviceToHost, streams[s]);
}
```

## Reduce Register Pressure

```bash
# Check register usage
hipcc --resource-usage kernel.cpp

# Limit registers
hipcc --maxrregcount=64 kernel.cpp

# May reduce occupancy but improve memory bandwidth
```

## Memory Pool for Allocations

```cpp
// BAD: Frequent allocations
for (int i = 0; i < 1000; i++) {
    float* temp;
    hipMalloc(&temp, size);
    // Use temp
    hipFree(temp);  // Expensive!
}

// GOOD: Reuse memory
float* temp;
hipMalloc(&temp, size);
for (int i = 0; i < 1000; i++) {
    // Reuse temp
}
hipFree(temp);
```

## PyTorch Memory Management

```python
import torch

# Enable memory pooling
torch.cuda.empty_cache()  # Clear cache

# Monitor memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Reduce peak memory with checkpointing
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(model, x):
    # Trade computation for memory
    return checkpoint(model.layer1, x)
```

## References

- [AMD GPU Memory Optimization](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides.html)
