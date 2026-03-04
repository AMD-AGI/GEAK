---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: Flash Attention Tile (fattn-tile)

## Variant Context
- Input semantic type: Attention (Query-Key-Value dot product with softmax)
- Datatype(s): FP16, FP32
- Data representation: Dense tensors, supports quantized KV cache
- Target architecture: Generic (NVIDIA Pascal+, AMD GCN+)

## Functionality
The tile-based Flash Attention kernel is designed for GPUs without Tensor Core support or as a fallback for unsupported configurations. It uses a tiled algorithm with shared memory to compute attention efficiently without materializing the full attention matrix.

Key features:
- Works on older GPUs (Pascal, GCN)
- Supports FP16 and FP32 computation
- Handles various head dimensions
- Quantized KV cache support

---

## Optimization 1: Larger SRAM Reads for Better Bandwidth
- Commit ID: 0e6ff0046
- Optimization type: Memory (bandwidth)
- Summary: Use larger memory transactions for loading K/V data into shared memory
- Detailed explanation: By using vectorized loads (float4, half4), the kernel achieves better memory bandwidth utilization. This is especially important for the tile kernel which relies heavily on shared memory throughput.

- Code excerpt:
    ```cpp
    // CUDA: larger SRAM reads for tile FA, AMD FP16 dot
    template<typename T, int vec_size>
    __device__ __forceinline__ void load_tile_vectorized(
        T * __restrict__ tile,
        const T * __restrict__ src,
        const int stride,
        const int tile_size) {
        
        using vec_t = typename std::conditional<
            vec_size == 4, float4,
            typename std::conditional<vec_size == 2, float2, float>::type
        >::type;
        
        const int tid = threadIdx.x;
        const int n_vecs = tile_size / vec_size;
        
        for (int i = tid; i < n_vecs; i += blockDim.x) {
            // Vectorized load
            vec_t val = *reinterpret_cast<const vec_t*>(src + i * vec_size);
            *reinterpret_cast<vec_t*>(tile + i * vec_size) = val;
        }
    }
    ```

- Evidence mapping:
  - "Vectorized loads" → `float4`, `float2` types
  - "Better bandwidth" → fewer memory transactions
  - "Shared memory" → loading to `tile` in SRAM

---

## Optimization 2: Pascal and AMD Optimization
- Commit ID: 79bc42926
- Optimization type: Compute (architecture support)
- Summary: Optimize tile FA for Pascal GPUs and AMD, add head size 256 support
- Detailed explanation: Pascal GPUs lack Tensor Cores, so the tile kernel is the primary FA implementation. This optimization tunes the tile sizes and thread configurations for Pascal's architecture and adds support for larger head dimensions.

- Code excerpt:
    ```cpp
    // CUDA: faster tile FA (Pascal/AMD), headsize 256
    template<int D, int TILE_KV>
    __global__ void flash_attn_tile_f16(
        const half * __restrict__ Q,
        const half * __restrict__ K,
        const half * __restrict__ V,
        half * __restrict__ dst,
        const int seq_len,
        const float scale) {
        
        // Tile sizes tuned for Pascal/AMD
        constexpr int TILE_Q = 16;  // Q tile size
        // TILE_KV passed as template param (32, 64, or 128)
        
        __shared__ half tile_Q[TILE_Q][D + 1];  // +1 for bank conflict avoidance
        __shared__ half tile_K[TILE_KV][D + 1];
        __shared__ half tile_V[TILE_KV][D + 1];
        __shared__ float tile_S[TILE_Q][TILE_KV + 1];
        
        // Process tiles...
    }
    
    // Head size 256 support
    template __global__ void flash_attn_tile_f16<256, 32>(...);
    template __global__ void flash_attn_tile_f16<256, 64>(...);
    ```

- Evidence mapping:
  - "Pascal optimization" → tile sizes tuned for SM 6.x
  - "Head size 256" → `D=256` template instantiation
  - "Bank conflict avoidance" → `D + 1` padding

---

## Optimization 3: Occupancy and Tile Size Optimization
- Commit ID: c959b676b
- Optimization type: Launch configuration
- Summary: Fix FA occupancy issues and optimize tile kernel parameters
- Detailed explanation: The tile kernel's performance depends heavily on achieving good occupancy. This optimization adjusts shared memory usage and thread counts to maximize the number of concurrent thread blocks.

- Code excerpt:
    ```cpp
    // CUDA: fix FA occupancy, optimize tile kernel
    static int get_tile_fa_block_size(int D, int cc) {
        // Adjust block size based on head dimension and compute capability
        if (D <= 64) return 256;
        if (D <= 128) return 128;
        return 64;  // Larger heads need fewer threads for occupancy
    }
    
    static int get_tile_fa_shmem(int D, int tile_kv, int block_size) {
        // Calculate shared memory to maximize occupancy
        const int shmem_q = 16 * (D + 1) * sizeof(half);
        const int shmem_kv = 2 * tile_kv * (D + 1) * sizeof(half);
        const int shmem_s = 16 * (tile_kv + 1) * sizeof(float);
        return shmem_q + shmem_kv + shmem_s;
    }
    ```

- Evidence mapping:
  - "Occupancy optimization" → block size selection based on D
  - "Shared memory calculation" → `get_tile_fa_shmem()` function
  - "Head dimension aware" → different configs for D=64, 128, etc.

---

## Optimization 4: Out-of-Bounds Checks and More Head Sizes
- Commit ID: 11f0af550
- Optimization type: Algorithm (correctness + coverage)
- Summary: Add out-of-bounds checks and support for additional head sizes
- Detailed explanation: This optimization adds proper bounds checking for edge cases (last tiles, variable sequence lengths) and extends support to more head dimensions used in various models.

- Code excerpt:
    ```cpp
    // CUDA: faster tile FA, add oob checks, more HSs
    template<int D, int TILE_KV, bool CHECK_OOB>
    __global__ void flash_attn_tile_checked(
        const half * __restrict__ Q,
        const half * __restrict__ K,
        const half * __restrict__ V,
        half * __restrict__ dst,
        const int seq_len,
        const int kv_len,
        const float scale) {
        
        const int q_idx = blockIdx.x * TILE_Q + threadIdx.y;
        const int kv_start = blockIdx.y * TILE_KV;
        
        // Load Q tile with bounds check
        if constexpr (CHECK_OOB) {
            if (q_idx < seq_len) {
                load_q_row(tile_Q[threadIdx.y], Q + q_idx * D, D);
            } else {
                zero_row(tile_Q[threadIdx.y], D);
            }
        } else {
            load_q_row(tile_Q[threadIdx.y], Q + q_idx * D, D);
        }
        
        // Similar checks for K, V tiles...
    }
    
    // More head sizes
    template __global__ void flash_attn_tile_checked<40, 64, true>(...);
    template __global__ void flash_attn_tile_checked<72, 64, true>(...);
    template __global__ void flash_attn_tile_checked<112, 64, true>(...);
    ```

- Evidence mapping:
  - "OOB checks" → `CHECK_OOB` template parameter
  - "Bounds checking" → `if (q_idx < seq_len)` condition
  - "More head sizes" → D=40, 72, 112 instantiations
