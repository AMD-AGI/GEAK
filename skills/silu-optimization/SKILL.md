---
name: silu-optimization
description: This skill should be used when optimizing silu kernel on AMD GPUs.
---

# Silu Kernel Optimization

## Variant Context
- Input semantic type: SiLU activation with element-wise multiplication (Gated Linear Unit)
- Datatype(s): bf16 (bfloat16)
- Data representation: Dense tensor [B, 2H] input, [B, H] output
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel computes the SiLU (Sigmoid Linear Unit) gated activation commonly used in transformer FFN layers. Given input tensor of shape [B, 2H], it splits into x and y components of shape [B, H] each, then computes output = silu(x) * y, where silu(x) = x / (1 + exp(-x)). This is a memory-bound kernel that benefits from maximizing memory bandwidth utilization.

## Optimization 1: Vectorized Memory Access with float4 (8 bf16 elements)
- Commit ID: N/A (directory comparison)
- Optimization type: memory
- Summary: Use float4 (16 bytes) to load/store 8 bf16 elements per memory transaction
- Detailed explanation: The optimized kernel uses float4 vectorized loads and stores, which pack 8 bf16 values (16 bytes) into a single memory transaction. This maximizes memory bandwidth utilization on AMD GPUs which support 128-bit (16-byte) memory transactions. The baseline processes elements individually, while the optimized version processes 8 at a time.
- Code excerpt:
    ```cpp
    // Process 8 elements at a time using float4 (16 bytes = 8 bf16)
    const int64_t H8 = H >> 3;
    
    // Cast to float4 for 16-byte vectorized access
    const float4* __restrict__ in_x_vec = reinterpret_cast<const float4*>(in + in_base);
    const float4* __restrict__ in_y_vec = reinterpret_cast<const float4*>(in + in_base + H);
    float4* __restrict__ out_vec = reinterpret_cast<float4*>(out + out_base);
    
    for (int64_t i = threadIdx.x; i < H8; i += blockDim.x) {
      // Load 8 bf16 values at once
      float4 x_packed = in_x_vec[i];
      float4 y_packed = in_y_vec[i];
    ```
- Evidence mapping:
  - "float4 vectorized access" → `reinterpret_cast<const float4*>` for 16-byte loads
  - "8 bf16 elements" → `H8 = H >> 3` (H/8 iterations)
  - "Single memory transaction" → `float4 x_packed = in_x_vec[i]` loads 8 bf16 values

## Optimization 2: Fast Math Intrinsics for SiLU Computation
- Commit ID: N/A (directory comparison)
- Optimization type: compute
- Summary: Use __fdividef and __expf intrinsics for faster SiLU computation
- Detailed explanation: The optimized kernel uses hardware intrinsics `__fdividef` (fast divide) and `__expf` (fast exponential) instead of standard division and `expf()`. These intrinsics map directly to GPU special function units and provide faster execution with slightly reduced precision, which is acceptable for bf16 computations.
- Code excerpt:
    ```cpp
    // Baseline:
    // __device__ __forceinline__ float silu_f(float x){
    //   return x / (1.0f + expf(-x));
    // }
    
    // Optimized:
    // Fast SiLU using intrinsics
    __device__ __forceinline__ float silu_f(float x){
      return __fdividef(x, 1.0f + __expf(-x));
    }
    ```
- Evidence mapping:
  - "Fast divide" → `__fdividef()` instead of `/` operator
  - "Fast exponential" → `__expf()` instead of `expf()`
  - "Hardware intrinsics" → Both functions map to GPU SFU

## Optimization 3: Reduced Block Size with Better Occupancy
- Commit ID: N/A (directory comparison)
- Optimization type: launch
- Summary: Changed block size from 1024 to 512 threads with launch bounds annotation
- Detailed explanation: The optimized kernel uses 512 threads per block instead of 1024, combined with `__launch_bounds__(512)` annotation. With vectorized processing (8 elements per thread), fewer threads are needed to cover the same data. The smaller block size can improve occupancy by allowing more blocks to run concurrently, and the launch bounds help the compiler optimize register usage.
- Code excerpt:
    ```cpp
    // Baseline:
    // dim3 grid(B), block(1024);
    
    // Optimized:
    __global__ __launch_bounds__(512)
    void silu_mul_kernel(...)
    
    // In main():
    dim3 grid(B), block(512);
    ```
- Evidence mapping:
  - "Reduced block size" → `block(512)` instead of `block(1024)`
  - "Launch bounds" → `__launch_bounds__(512)` attribute
  - "Better occupancy" → Smaller blocks allow more concurrent blocks

## Optimization 4: Loop Unrolling for Element Processing
- Commit ID: N/A (directory comparison)
- Optimization type: compute
- Summary: Added pragma unroll for the inner loop processing 8 bf16 elements
- Detailed explanation: The optimized kernel adds `#pragma unroll` to the inner loop that processes the 8 bf16 elements within each float4. This eliminates loop overhead and allows the compiler to schedule all 8 SiLU computations optimally, potentially executing them in parallel on different ALUs.
- Code excerpt:
    ```cpp
    // Reinterpret as bf16 arrays
    const bf16* x_vals = reinterpret_cast<const bf16*>(&x_packed);
    const bf16* y_vals = reinterpret_cast<const bf16*>(&y_packed);
    
    float4 out_packed;
    bf16* out_vals = reinterpret_cast<bf16*>(&out_packed);
    
    // Compute silu(x) * y for each element
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      out_vals[j] = __float2bfloat16(silu_f(__bfloat162float(x_vals[j])) * __bfloat162float(y_vals[j]));
    }
    
    out_vec[i] = out_packed;
    ```
- Evidence mapping:
  - "Loop unrolling" → `#pragma unroll` directive
  - "8 elements" → Loop bound `j < 8`
  - "Parallel execution" → Unrolled operations can be scheduled on multiple ALUs

## Optimization 5: Restrict Pointers for Compiler Optimization
- Commit ID: N/A (directory comparison)
- Optimization type: memory
- Summary: Added __restrict__ qualifiers to vectorized pointers for better aliasing analysis
- Detailed explanation: The optimized kernel adds `__restrict__` qualifiers to the vectorized pointer declarations. This tells the compiler that these pointers don't alias each other, enabling more aggressive optimizations like reordering loads and stores, and potentially combining memory operations.
- Code excerpt:
    ```cpp
    // Cast to float4 for 16-byte vectorized access
    const float4* __restrict__ in_x_vec = reinterpret_cast<const float4*>(in + in_base);
    const float4* __restrict__ in_y_vec = reinterpret_cast<const float4*>(in + in_base + H);
    float4* __restrict__ out_vec = reinterpret_cast<float4*>(out + out_base);
    ```
- Evidence mapping:
  - "Restrict qualifiers" → `__restrict__` on all three pointer declarations
  - "No aliasing" → Compiler can assume pointers don't overlap
  - "Better optimization" → Enables load/store reordering and combining
