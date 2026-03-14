---
name: fused_bucketized-optimization
description: This skill should be used when optimizing fused_bucketized kernel on AMD GPUs.
---

# Fused Bucketized Kernel Optimization
## Variant Context
- Input semantic type: Element-wise bucketization (assigning float values to bucket indices)
- Datatype(s): fp32 input, int64 output
- Data representation: Dense arrays with boundary lookup
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel performs fused element-wise bucketization across multiple tensors. For each input float value, it determines which bucket the value falls into based on a sorted boundary array. The baseline uses binary search, while the optimized version uses multiple techniques to improve performance.

## Optimization 1: Shared Memory Caching for Boundary Values
- Commit ID: N/A (directory comparison)
- Optimization type: memory
- Summary: Cache boundary values in shared memory to reduce global memory access latency
- Detailed explanation: The optimized kernel loads boundary values into shared memory once per block, allowing all threads in the block to access these frequently-used values from fast shared memory instead of repeatedly fetching from global memory. This is particularly effective since all threads in a block process values against the same boundary array.
- Code excerpt:
    ```cpp
    // Cache boundaries in shared memory
    __shared__ float shared_boundaries[8];
    __shared__ int shared_len;
    
    int64_t vec_id = blockIdx.y;
    
    if (threadIdx.x == 0) {
      shared_len = b[vec_id].len;
    }
    __syncthreads();
    
    int len = shared_len;
    if (threadIdx.x < len && threadIdx.x < 8) {
      shared_boundaries[threadIdx.x] = b[vec_id].boundaries[threadIdx.x];
    }
    __syncthreads();
    ```
- Evidence mapping:
  - "Cache boundary values in shared memory" → `__shared__ float shared_boundaries[8]` declaration and cooperative loading with `shared_boundaries[threadIdx.x] = b[vec_id].boundaries[threadIdx.x]`
  - "Reduce global memory access" → Boundaries loaded once per block, then accessed from shared memory

## Optimization 2: Register Caching for Boundary Values
- Commit ID: N/A (directory comparison)
- Optimization type: memory
- Summary: Load boundary values from shared memory into registers for fastest possible access during computation
- Detailed explanation: After loading boundaries into shared memory, each thread further caches these values into registers (b0-b4). Register access is the fastest memory tier on GPUs, and since the boundary values are used repeatedly in the inner loop, this provides significant speedup.
- Code excerpt:
    ```cpp
    // Load boundaries to registers for fastest access
    float b0 = (len > 0) ? shared_boundaries[0] : 1e30f;
    float b1 = (len > 1) ? shared_boundaries[1] : 1e30f;
    float b2 = (len > 2) ? shared_boundaries[2] : 1e30f;
    float b3 = (len > 3) ? shared_boundaries[3] : 1e30f;
    float b4 = (len > 4) ? shared_boundaries[4] : 1e30f;
    ```
- Evidence mapping:
  - "Register caching" → Local float variables b0-b4 holding boundary values
  - "Fastest access" → Register variables accessed directly in bucketize_value function

## Optimization 3: Linear Search Instead of Binary Search
- Commit ID: N/A (directory comparison)
- Optimization type: compute
- Summary: Replace binary search with linear search for small boundary arrays to reduce branch divergence
- Detailed explanation: For small boundary arrays (5 elements in this case), linear search with sequential comparisons is faster than binary search. Binary search has unpredictable branches that cause warp divergence on GPUs. Linear search with sequential if-statements allows the compiler to generate predicated instructions, avoiding divergence entirely.
- Code excerpt:
    ```cpp
    // Inline bucketize function using linear search for small boundary arrays
    // This is faster than binary search for 5 elements due to reduced branch divergence
    __device__ __forceinline__ int64_t bucketize_value(float value, float b0, float b1, float b2, float b3, float b4) {
      int64_t bucket = 0;
      if (value >= b0) bucket = 1;
      if (value >= b1) bucket = 2;
      if (value >= b2) bucket = 3;
      if (value >= b3) bucket = 4;
      if (value >= b4) bucket = 5;
      return bucket;
    }
    ```
- Evidence mapping:
  - "Linear search" → Sequential if-statements checking each boundary
  - "Reduced branch divergence" → Independent if-statements (not if-else) allow predication
  - "Inline function" → `__forceinline__` attribute ensures no function call overhead

## Optimization 4: Vectorized Memory Access (float2/longlong2)
- Commit ID: N/A (directory comparison)
- Optimization type: memory
- Summary: Use vectorized loads (float2) and stores (longlong2) to process 2 elements per memory transaction
- Detailed explanation: The optimized kernel processes 2 elements at a time using float2 for input loads and longlong2 for output stores. This doubles the memory bandwidth utilization per transaction, as GPUs can load/store 8 or 16 bytes more efficiently than individual 4-byte accesses.
- Code excerpt:
    ```cpp
    // Process 2 elements per thread using float2 load and longlong2 store
    // This improves memory bandwidth utilization
    int64_t vec2_count = size_local / 2;
    const float2* input2 = reinterpret_cast<const float2*>(input);
    longlong2* output2 = reinterpret_cast<longlong2*>(output);
    
    for (int64_t i = tid; i < vec2_count; i += threads_num) {
      float2 vals = input2[i];
      
      longlong2 result;
      result.x = bucketize_value(vals.x, b0, b1, b2, b3, b4);
      result.y = bucketize_value(vals.y, b0, b1, b2, b3, b4);
      
      output2[i] = result;
    }
    ```
- Evidence mapping:
  - "Vectorized loads" → `const float2* input2` and `float2 vals = input2[i]`
  - "Vectorized stores" → `longlong2* output2` and `output2[i] = result`
  - "2 elements per transaction" → Processing vals.x and vals.y together

## Optimization 5: Increased Grid Size for Better Occupancy
- Commit ID: N/A (directory comparison)
- Optimization type: launch
- Summary: Increase grid size multiplier from 8 to 16 SMs to improve GPU occupancy
- Detailed explanation: The optimized launcher increases the grid size calculation from `sm_count * 8` to `sm_count * 16`, allowing more thread blocks to be scheduled. Combined with vectorized processing (2 elements per thread), this maintains high occupancy while reducing the total number of iterations per thread.
- Code excerpt:
    ```cpp
    // Optimize grid size for vectorized processing (2 elements per thread)
    int64_t vec2_count = max_size / 2;
    int64_t block_num = min(sm_count * 16, (vec2_count + KBLOCK_SIZE - 1) / KBLOCK_SIZE);
    block_num = max(block_num, (int64_t)1);
    ```
- Evidence mapping:
  - "Increased grid size" → `sm_count * 16` vs baseline's `sm_count * 8`
  - "Vectorized processing adjustment" → `vec2_count = max_size / 2` accounts for 2 elements per thread
