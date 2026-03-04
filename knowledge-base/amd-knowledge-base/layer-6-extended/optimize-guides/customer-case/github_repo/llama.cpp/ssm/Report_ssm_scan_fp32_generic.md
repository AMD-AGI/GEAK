---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: SSM Scan (State Space Model)

## Variant Context
- Input semantic type: Sequential state update (Mamba, LFM2)
- Datatype(s): FP32
- Data representation: Dense state tensors
- Target architecture: Generic (NVIDIA, AMD)

## Functionality
The SSM scan kernel implements the parallel scan operation for State Space Models like Mamba and Mamba-2. It computes the recurrent state update: `h[t] = A * h[t-1] + B * x[t]` efficiently in parallel using the associative scan algorithm.

Key features:
- Parallel prefix scan for O(log n) depth
- Support for multiple state groups
- Warp-level and block-level reductions
- CUB library integration for optimized scans

---

## Optimization 1: CUB Library Integration
- Commit ID: 79c1160b0
- Optimization type: Compute (library optimization)
- Summary: Use NVIDIA CUB library for optimized parallel scan operations
- Detailed explanation: CUB (CUDA UnBound) provides highly optimized primitives for parallel operations. This optimization replaces custom scan implementations with CUB's `BlockScan` and `WarpScan` primitives, which are tuned for various GPU architectures.

- Code excerpt:
    ```cpp
    // cuda: refactored ssm_scan to use CUB
    #ifdef GGML_CUDA_USE_CUB
    #include <cub/cub.cuh>
    
    template<int BLOCK_SIZE, int STATE_SIZE>
    __global__ void ssm_scan_cub(
        const float * __restrict__ A,
        const float * __restrict__ B,
        const float * __restrict__ x,
        float * __restrict__ h,
        const int seq_len) {
        
        // CUB block scan for parallel prefix
        typedef cub::BlockScan<float2, BLOCK_SIZE> BlockScan;
        __shared__ typename BlockScan::TempStorage temp_storage;
        
        // Custom scan operator for SSM recurrence
        struct SSMScanOp {
            __device__ __forceinline__ float2 operator()(
                const float2 &a, const float2 &b) const {
                // (A1, B1) * (A2, B2) = (A1*A2, A1*B2 + B1)
                return make_float2(a.x * b.x, a.x * b.y + a.y);
            }
        };
        
        float2 thread_data = make_float2(A[tid], B[tid] * x[tid]);
        float2 result;
        
        BlockScan(temp_storage).InclusiveScan(thread_data, result, SSMScanOp());
        
        h[tid] = result.y;  // State is the B component
    }
    #endif
    ```

- Evidence mapping:
  - "CUB integration" → `#include <cub/cub.cuh>`
  - "BlockScan primitive" → `cub::BlockScan<float2, BLOCK_SIZE>`
  - "Custom operator" → `SSMScanOp` for SSM recurrence

---

## Optimization 2: Warp-Level Reduction
- Commit ID: 24af22fc3
- Optimization type: Compute (warp optimization)
- Summary: Optimize SSM scan using warp-level shuffle reductions
- Detailed explanation: For small state sizes that fit within a warp, this optimization uses warp shuffle instructions for the scan operation, avoiding shared memory and synchronization overhead. This is faster for the common case of small state dimensions.

- Code excerpt:
    ```cpp
    // ggml: optimize cuda ssm_scan using warp-level reduction
    template<int STATE_SIZE>
    __device__ __forceinline__ void ssm_scan_warp(
        float * __restrict__ state,
        const float a,
        const float b_x,
        const int lane_id) {
        
        // Warp-level inclusive scan using shuffles
        float val = b_x;
        float coef = a;
        
        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
            float other_val = __shfl_up_sync(0xffffffff, val, offset);
            float other_coef = __shfl_up_sync(0xffffffff, coef, offset);
            
            if (lane_id >= offset) {
                val = other_coef * val + other_val;
                coef = other_coef * coef;
            }
        }
        
        state[lane_id] = val;
    }
    ```

- Evidence mapping:
  - "Warp shuffle" → `__shfl_up_sync()` for communication
  - "No shared memory" → direct register-to-register transfer
  - "Log(n) steps" → `offset *= 2` doubling pattern

---

## Optimization 3: Multi-Group Support
- Commit ID: 73804145a
- Optimization type: Algorithm (correctness + parallelism)
- Summary: Fix and optimize SSM scan for models with multiple state groups
- Detailed explanation: Mamba-2 and other models use multiple independent state groups that can be processed in parallel. This optimization correctly handles the group dimension and parallelizes across groups for better GPU utilization.

- Code excerpt:
    ```cpp
    // ggml: fix SSM_SCAN for n_groups > 1
    template<int BLOCK_SIZE>
    __global__ void ssm_scan_grouped(
        const float * __restrict__ A,
        const float * __restrict__ B,
        const float * __restrict__ x,
        float * __restrict__ h,
        const int seq_len,
        const int state_size,
        const int n_groups) {
        
        const int group_id = blockIdx.y;
        const int seq_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        
        if (seq_id >= seq_len) return;
        
        // Each group has independent state
        const int state_offset = group_id * state_size;
        float * h_group = h + state_offset;
        
        // Process this group's state update
        for (int s = 0; s < state_size; s++) {
            const float a = A[group_id * state_size + s];
            const float b = B[seq_id * n_groups * state_size + group_id * state_size + s];
            
            // Scan within group
            ...
        }
    }
    ```

- Evidence mapping:
  - "Group parallelism" → `blockIdx.y` for group dimension
  - "Independent states" → `state_offset = group_id * state_size`
  - "Correct indexing" → proper stride calculation for grouped access

---

## Optimization 4: Wave Size Portability
- Commit ID: e54b39408
- Optimization type: Compute (portability)
- Summary: Fix SSM scan for devices where warp size is not 32 (AMD GPUs)
- Detailed explanation: AMD CDNA GPUs use 64-thread wavefronts. This optimization makes the SSM scan kernel portable by using the physical warp size instead of hardcoded 32, ensuring correct behavior on both NVIDIA and AMD GPUs.

- Code excerpt:
    ```cpp
    // CUDA/HIP: fix ssm_scan on devices where warp size is not 32
    static constexpr int get_warp_size() {
    #if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
        #if defined(__gfx1100__) || defined(__gfx1101__) || ...
            return 32;  // RDNA
        #else
            return 64;  // CDNA/GCN
        #endif
    #else
        return 32;  // NVIDIA
    #endif
    }
    
    template<int WARP_SIZE = get_warp_size()>
    __device__ void ssm_scan_portable(...) {
        const int lane_id = threadIdx.x % WARP_SIZE;
        
        // Use WARP_SIZE instead of hardcoded 32
        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
            ...
        }
    }
    ```

- Evidence mapping:
  - "Portable warp size" → `get_warp_size()` function
  - "Architecture detection" → `__gfx1100__` etc. macros
  - "Template parameter" → `WARP_SIZE` as template arg

---

## Optimization 5: Mamba-2 State Size Support
- Commit ID: 5d46babdc, a57d1bcb3
- Optimization type: Algorithm (model support)
- Summary: Add support for Mamba-2 architecture with different state sizes
- Detailed explanation: Mamba-2 uses different state dimensions than Mamba-1. This optimization extends the SSM scan kernel to handle various state sizes efficiently, including the larger state dimensions used in Falcon-H1 and other models.

- Code excerpt:
    ```cpp
    // cuda: support Falcon-H1 state size for SSM_SCAN
    // Initial Mamba-2 support
    
    // Template for different state sizes
    template<int STATE_SIZE>
    __global__ void ssm_scan_sized(...);
    
    // Instantiate for common state sizes
    template __global__ void ssm_scan_sized<16>(...);   // Mamba-1
    template __global__ void ssm_scan_sized<64>(...);   // Mamba-2
    template __global__ void ssm_scan_sized<128>(...);  // Falcon-H1
    template __global__ void ssm_scan_sized<256>(...);  // Large models
    
    // Runtime dispatch
    void launch_ssm_scan(int state_size, ...) {
        switch (state_size) {
            case 16:  ssm_scan_sized<16><<<...>>>(...); break;
            case 64:  ssm_scan_sized<64><<<...>>>(...); break;
            case 128: ssm_scan_sized<128><<<...>>>(...); break;
            case 256: ssm_scan_sized<256><<<...>>>(...); break;
            default:  ssm_scan_generic<<<...>>>(...); break;
        }
    }
    ```

- Evidence mapping:
  - "Multiple state sizes" → template instantiations for 16, 64, 128, 256
  - "Mamba-2 support" → state_size=64 instantiation
  - "Runtime dispatch" → switch statement for kernel selection
