# HIP Shared Memory (LDS) Optimization Best Practices

## Overview

Shared memory on AMD GPUs is called Local Data Share (LDS). It provides low-latency, high-bandwidth on-chip memory that can be used as a staging area to transform strided global memory access into coalesced patterns.

## AMD GPU Memory Hierarchy

AMD GPUs (CDNA/RDNA) have the following memory hierarchy:
1. Registers: ~256KB per CU, fastest
2. LDS (Local Data Share): 64-128KB per CU, shared memory
3. L1 Cache: 16-32KB per CU
4. L2 Cache: 4-8MB, shared across GPU
5. HBM (High Bandwidth Memory): 32-128GB, 1.6-3.2 TB/s

## LDS Performance Characteristics

- Internal bandwidth: ~20-40 TB/s (50-100x faster than global memory)
- Latency: 4-8 cycles vs 200-400 cycles for global memory
- Capacity: 64 KB per CU on MI300/MI250

## Key Optimization Techniques

### 1. Tiled Matrix Multiplication with Shared Memory

Use __shared__ arrays to cache tiles of input matrices:
- Load tiles cooperatively into shared memory
- Synchronize with __syncthreads()
- Compute from fast shared memory
- Repeat for all tiles

### 2. Bank Conflict Avoidance

AMD LDS bank organization (CDNA2 architecture):
- Number of Banks: 32 banks
- Bank Width: 4 bytes (32 bits)
- Wavefront Size: 64 threads

Address-to-bank mapping: Bank Number = (Address / 4) % 32

Common conflict patterns to avoid:
- Power-of-2 stride conflicts
- Column-major access of row-major arrays
- Constant column index access

### 3. Shared Memory Buffering for Strided Access

When strided access cannot be avoided:
- Buffer data through shared memory
- Enable coalesced reads from global memory into LDS
- Perform strided access within fast shared memory

Effective when:
- Stride is moderate to large (>4)
- Data is reused multiple times
- Access pattern is irregular but predictable

### 4. SoA Layout in LDS

Use Structure of Arrays (SoA) layout for simpler addressing:
- Separate arrays for each component (s_x[], s_y[], s_z[])
- Better than interleaved AoS layout

### 5. Loop Unrolling

Use #pragma unroll to reduce loop overhead and increase ILP

## Best Practices Summary

1. Use shared memory for data reuse - Load once, use multiple times
2. Ensure coalesced global memory access when loading into shared memory
3. Avoid bank conflicts - Use padding or swizzling techniques
4. Use __syncthreads() - Synchronize after loading and before using
5. Consider LDS capacity - 64 KB per CU limit
6. Use vectorized access - float4/int4 for coalesced memory operations
7. Precompute pointers - Use __restrict__ for base pointers
8. Tile appropriately - Balance between occupancy and data reuse

## When to Use Shared Memory

Effective for:
- Matrix multiplication (tiling)
- Convolution operations
- Reduction operations
- Data transpose/reorganization
- Cooperative operations between threads
- Caching frequently accessed data

Unnecessary for:
- Simple element-wise operations (like SiLU)
- Operations where each thread processes independent elements

## Key Optimization Patterns from vLLM

- PagedAttention: Shared memory caching, MQA, V2 partitioning
- Fused MoE: Block alignment, L2 cache, FP8
- ROCm Skinny GEMM: MFMA, Split-K, LDS optimization
- Cache Kernels: Vectorization, FP8, MLA support

