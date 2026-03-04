---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: fused_bucketized

## Variant Context
- Input semantic type: Element-wise bucketization (assigning float values to discrete buckets)
- Datatype(s): fp32 (input values), int64 (output bucket indices)
- Data representation: Dense arrays with boundary thresholds
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel performs fused element-wise bucketization across multiple tensors. For each input float value, it determines which bucket the value falls into based on a set of boundary thresholds. The baseline uses a generic binary search approach via a functor, while the optimized version specializes for a fixed boundary length of 5.

## Optimization 1: Branchless Bucket Computation
- Optimization type: Compute
- Summary: Replaced binary search loop with branchless comparison-based bucket computation for fixed boundary length 5
- Detailed explanation: The baseline kernel uses a generic `BucketizeFactory` functor that performs binary search with a while loop and conditional branches to find the correct bucket. This approach has variable iteration count and branch divergence. The optimized version introduces a specialized `bucketize_5` function that uses a fully branchless approach - it simply adds up boolean comparison results (0 or 1) for each boundary. This eliminates branch divergence and provides predictable execution time regardless of input values.
- Code excerpt (baseline):
    ```cpp
    struct BucketizeFactory {
      __device__ int operator()(const float value, const BucketizeData& data) {
        int bucket = 0;
        int count = data.len;
        auto boundaries = data.boundaries;
        while (count > 0) {
          int left = bucket;
          int step = count / 2;
          left += step;
          if (!(value < boundaries[left])) {
            bucket = ++left;
            count -= step + 1;
          } else {
            count = step;
          }
        }
        return bucket;
      }
    };
    ```
- Code excerpt (optimized):
    ```cpp
    __device__ __forceinline__ int64_t bucketize_5(const float value, const float b0, const float b1, const float b2, const float b3, const float b4) {
      int64_t bucket = 0;
      bucket += (value >= b0);
      bucket += (value >= b1);
      bucket += (value >= b2);
      bucket += (value >= b3);
      bucket += (value >= b4);
      return bucket;
    }
    ```
- Evidence mapping:
  - "Branchless computation" → Each `bucket += (value >= bN)` is a predicated add without branches
  - "Eliminates loop overhead" → No while loop, fixed 5 comparisons
  - "Reduces branch divergence" → All threads execute the same instruction sequence

## Optimization 2: Shared Memory Caching for Boundaries and Pointers
- Optimization type: Memory
- Summary: Cache frequently accessed boundaries and pointers in shared memory to reduce global memory traffic
- Detailed explanation: The optimized kernel loads the 5 boundary values and input/output pointers into shared memory once per block (by thread 0), then all threads read from shared memory. This reduces redundant global memory accesses since all threads in a block process the same vector and need the same boundary values. The baseline accesses boundaries through a pointer in the BucketizeData struct on every element access.
- Code excerpt (optimized):
    ```cpp
    // Shared memory for caching boundaries and pointers
    __shared__ float s_b0, s_b1, s_b2, s_b3, s_b4;
    __shared__ const A* s_a_ptr;
    __shared__ C* s_c_ptr;
    __shared__ int64_t s_size;
    
    const int64_t vec_id = blockIdx.y;
    const int tid = threadIdx.x;
    
    // Load boundaries and pointers into shared memory
    if (tid == 0) {
      s_a_ptr = a[vec_id];
      s_c_ptr = c[vec_id];
      s_size = sizes[vec_id];
      const float* b = boundaries[vec_id];
      s_b0 = b[0];
      s_b1 = b[1];
      s_b2 = b[2];
      s_b3 = b[3];
      s_b4 = b[4];
    }
    
    __syncthreads();
    
    // Cache in registers
    const A* __restrict__ local_a = s_a_ptr;
    C* __restrict__ local_c = s_c_ptr;
    const int64_t size_local = s_size;
    const float b0 = s_b0, b1 = s_b1, b2 = s_b2, b3 = s_b3, b4 = s_b4;
    ```
- Evidence mapping:
  - "Shared memory caching" → `__shared__` declarations for boundaries and pointers
  - "Single load per block" → `if (tid == 0)` loads data once
  - "Register caching" → Local variables `b0, b1, b2, b3, b4` hold values in registers after shared memory load

## Optimization 3: Use of __restrict__ Qualifiers
- Optimization type: Memory
- Summary: Added __restrict__ qualifiers to pointer parameters to enable compiler optimizations
- Detailed explanation: The optimized kernel adds `__restrict__` qualifiers to all pointer parameters, informing the compiler that these pointers do not alias. This allows the compiler to perform more aggressive optimizations such as better instruction scheduling and avoiding redundant loads.
- Code excerpt (optimized):
    ```cpp
    template<typename A, typename C>
    __global__ void fused_element_wise_kernel(
        const A* __restrict__ * __restrict__ a,
        const float* __restrict__ * __restrict__ boundaries,
        C* __restrict__ * __restrict__ c,
        int64_t N,
        int64_t* __restrict__ sizes) {
    ```
- Evidence mapping:
  - "Pointer aliasing hints" → `__restrict__` on all pointer parameters
  - "Enables compiler optimization" → Compiler can assume no pointer aliasing

## Optimization 4: Simplified Kernel Interface
- Optimization type: Compute
- Summary: Removed generic Factory template parameter and specialized for the specific use case
- Detailed explanation: The baseline kernel uses a generic Factory template that adds indirection and prevents certain compiler optimizations. The optimized version removes this abstraction and directly implements the bucketization logic, reducing function call overhead and enabling better inlining.
- Code excerpt (baseline):
    ```cpp
    template <typename A, typename B, typename C, typename Factory>
    __global__ void fused_element_wise_kernel(const A** a, const B* b, C** c,
                                              int64_t N, int64_t* sizes,
                                              Factory factory) {
      // ...
      c[vec_id][index] = factory(a[vec_id][index], b[vec_id]);
    }
    ```
- Code excerpt (optimized):
    ```cpp
    template<typename A, typename C>
    __global__ void fused_element_wise_kernel(
        const A* __restrict__ * __restrict__ a,
        const float* __restrict__ * __restrict__ boundaries,
        C* __restrict__ * __restrict__ c,
        int64_t N,
        int64_t* __restrict__ sizes) {
      // ...
      local_c[index] = bucketize_5(val, b0, b1, b2, b3, b4);
    }
    ```
- Evidence mapping:
  - "Removed Factory template" → No Factory parameter in optimized kernel signature
  - "Direct function call" → `bucketize_5()` called directly instead of through functor
