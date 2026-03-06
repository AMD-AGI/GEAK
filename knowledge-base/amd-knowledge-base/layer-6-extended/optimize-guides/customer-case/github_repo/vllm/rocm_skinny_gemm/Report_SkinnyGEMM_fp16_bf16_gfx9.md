Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: ROCm Skinny GEMM

## Variant Context
- Input semantic type: Matrix multiplication with skinny matrices (small M or N dimension)
- Datatype(s): fp16, bf16, fp8
- Data representation: Dense matrices with one small dimension
- Target architecture: AMD HIP gfx90a (MI200), gfx942 (MI300X), gfx950 (MI350)

## Functionality
The ROCm Skinny GEMM kernel provides optimized matrix multiplication for cases where one dimension is small (typically batch size during decode phase). Standard GEMM kernels are optimized for large square matrices, but LLM inference often involves skinny matrices (e.g., batch_size x hidden_dim @ hidden_dim x vocab_size). This kernel uses specialized tiling and memory access patterns for these cases.

## Optimization 1: Initial Skinny GEMM Implementation
- Commit ID: 188b7f9b8
- Optimization type: Compute / Memory
- Summary: Add optimized skinny GEMM kernels for unquantized linear layers on ROCm
- Detailed explanation:
  This optimization introduces specialized GEMM kernels for skinny matrices on AMD GPUs. The kernels use different tiling strategies optimized for small M dimensions, with wavefront-level parallelism across the N dimension.
- Code excerpt:
    ```cpp
    // Skinny GEMM kernel for small M (batch size)
    template <typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K>
    __global__ void skinny_gemm_kernel(
        T* __restrict__ out,
        const T* __restrict__ A,  // [M, K]
        const T* __restrict__ B,  // [K, N]
        int M, int N, int K) {
      
      // For skinny matrices, parallelize across N dimension
      // Each workgroup handles BLOCK_M rows and BLOCK_N columns
      const int wg_id_n = blockIdx.x;
      const int wg_id_m = blockIdx.y;
      
      // Use LDS (Local Data Share) for A matrix reuse
      __shared__ T A_shared[BLOCK_M][BLOCK_K];
      
      // Each thread computes multiple output elements
      T accum[BLOCK_M][BLOCK_N / WARP_SIZE] = {0};
      
      for (int k = 0; k < K; k += BLOCK_K) {
        // Cooperative load of A tile to LDS
        load_tile_to_lds(A, A_shared, wg_id_m * BLOCK_M, k);
        __syncthreads();
        
        // Compute partial products
        for (int kk = 0; kk < BLOCK_K; kk++) {
          T a_val = A_shared[threadIdx.y][kk];
          T b_val = B[(k + kk) * N + wg_id_n * BLOCK_N + threadIdx.x];
          accum[threadIdx.y][0] += a_val * b_val;
        }
        __syncthreads();
      }
      
      // Write results
      store_results(out, accum, wg_id_m, wg_id_n);
    }
    ```
- Evidence mapping:
  - "Skinny optimization" → Parallelization across N dimension for small M
  - "LDS usage" → `__shared__ T A_shared` for A matrix reuse
  - "Wavefront parallelism" → Thread indexing optimized for AMD wavefronts

## Optimization 2: BF16 MFMA Optimization
- Commit ID: 5a499e70d
- Optimization type: Compute
- Summary: Add BF16 MFMA (Matrix Fused Multiply-Add) optimization for ROCm skinny GEMMs
- Detailed explanation:
  AMD's MFMA instructions provide high-throughput matrix operations. This optimization uses BF16 MFMA instructions for skinny GEMM, significantly improving throughput on MI200/MI300 GPUs.
- Code excerpt:
    ```cpp
    #if defined(__gfx90a__) || defined(__gfx942__)
    // Use MFMA for BF16 skinny GEMM
    template <>
    __device__ void mfma_gemm<__hip_bfloat16, 16, 16, 4>(
        float* accum,
        const __hip_bfloat16* a,
        const __hip_bfloat16* b) {
      
      // Pack BF16 values for MFMA
      using bf16x4 = __attribute__((ext_vector_type(4))) __hip_bfloat16;
      bf16x4 a_packed = *reinterpret_cast<const bf16x4*>(a);
      bf16x4 b_packed = *reinterpret_cast<const bf16x4*>(b);
      
      // MFMA instruction: 16x16x4 BF16
      // Computes C += A @ B where A is 16x4, B is 4x16
      *reinterpret_cast<float4*>(accum) = __builtin_amdgcn_mfma_f32_16x16x4bf16(
          a_packed, b_packed, 
          *reinterpret_cast<float4*>(accum),
          0, 0, 0);  // MFMA modifiers
    }
    #endif
    ```
- Evidence mapping:
  - "MFMA usage" → `__builtin_amdgcn_mfma_f32_16x16x4bf16` intrinsic
  - "Architecture guard" → `#if defined(__gfx90a__) || defined(__gfx942__)`
  - "Packed BF16" → `bf16x4` vector type for efficient MFMA input

## Optimization 3: GFX950 Support with Larger LDS
- Commit ID: 306d60401
- Optimization type: Memory
- Summary: Add gfx950 (MI350) support with larger LDS utilization
- Detailed explanation:
  MI350 (gfx950) has 160KB LDS compared to 64KB on MI300. This optimization leverages the larger LDS for bigger tiles and better data reuse.
- Code excerpt:
    ```cpp
    // LDS size based on architecture
    #if defined(__gfx950__)
      #define LDS_SIZE 160 * 1024
    #else
      #define LDS_SIZE 64 * 1024
    #endif
    
    int get_lds_size() {
      static bool is_cached = false;
      static int result;
      if (is_cached == false) {
        auto dprops = at::cuda::getCurrentDeviceProperties();
        std::string device_arch = dprops->gcnArchName;
        size_t substring = device_arch.find("gfx95");
        result = (substring == std::string::npos ? 64 * 1024 : 160 * 1024);
        is_cached = true;
      }
      return result;
    }
    
    // Use larger tiles on gfx950
    template <int LDS_BYTES>
    constexpr int get_block_k() {
      if constexpr (LDS_BYTES >= 160 * 1024) {
        return 128;  // Larger K tile for more reuse
      } else {
        return 64;
      }
    }
    ```
- Evidence mapping:
  - "Architecture detection" → `device_arch.find("gfx95")`
  - "Larger LDS" → `160 * 1024` bytes for gfx950
  - "Bigger tiles" → `BLOCK_K = 128` when LDS allows

## Optimization 4: Split-K with Atomic Reduction
- Commit ID: 7a1030431
- Optimization type: Compute / Scheduling
- Summary: Optimize atomic reduction counting for Split-K skinny GEMMs
- Detailed explanation:
  Split-K parallelizes the K dimension reduction across multiple workgroups. This optimization improves the atomic reduction phase by using counting-based synchronization instead of barriers.
- Code excerpt:
    ```cpp
    // Split-K with optimized atomic counting
    template <typename T, int SPLIT_K>
    __global__ void skinny_gemm_splitk_kernel(
        T* __restrict__ out,
        T* __restrict__ partial_sums,  // [SPLIT_K, M, N]
        int* __restrict__ counters,     // [M, N] atomic counters
        const T* __restrict__ A,
        const T* __restrict__ B,
        int M, int N, int K) {
      
      const int split_idx = blockIdx.z;
      const int k_start = split_idx * (K / SPLIT_K);
      const int k_end = (split_idx + 1) * (K / SPLIT_K);
      
      // Compute partial sum for this K-split
      T partial = compute_partial(A, B, k_start, k_end);
      
      // Store partial sum
      partial_sums[split_idx * M * N + row * N + col] = partial;
      
      // Atomic increment counter
      int count = atomicAdd(&counters[row * N + col], 1);
      
      // Last workgroup to finish does the reduction
      if (count == SPLIT_K - 1) {
        T sum = 0;
        for (int s = 0; s < SPLIT_K; s++) {
          sum += partial_sums[s * M * N + row * N + col];
        }
        out[row * N + col] = sum;
      }
    }
    ```
- Evidence mapping:
  - "Split-K parallelism" → `blockIdx.z` for K-dimension splitting
  - "Atomic counting" → `atomicAdd(&counters[...], 1)` for synchronization
  - "Last-writer reduction" → `if (count == SPLIT_K - 1)` triggers final sum

## Optimization 5: Improved Tile and Balance Heuristics
- Commit ID: 2e7054da0
- Optimization type: Launch Configuration
- Summary: Improve tile size and workload balance heuristics for Split-K
- Detailed explanation:
  Choosing optimal tile sizes and Split-K factors depends on problem dimensions and GPU occupancy. This optimization improves the heuristics for selecting these parameters.
- Code excerpt:
    ```cpp
    // Heuristics for Split-K configuration
    struct SplitKConfig {
      int split_k;
      int block_m;
      int block_n;
      int block_k;
    };
    
    SplitKConfig get_splitk_config(int M, int N, int K, int num_cus) {
      SplitKConfig config;
      
      // For very skinny M, use more splits
      if (M <= 4) {
        config.split_k = min(K / 256, num_cus / (N / 128));
        config.block_m = M;
        config.block_n = 128;
        config.block_k = 256;
      } else if (M <= 16) {
        config.split_k = min(K / 128, num_cus / ((M / 4) * (N / 128)));
        config.block_m = 4;
        config.block_n = 128;
        config.block_k = 128;
      } else {
        // Standard tiling for larger M
        config.split_k = 1;
        config.block_m = 32;
        config.block_n = 128;
        config.block_k = 64;
      }
      
      // Ensure good CU utilization
      int num_blocks = (M / config.block_m) * (N / config.block_n) * config.split_k;
      if (num_blocks < num_cus) {
        // Increase split_k for better parallelism
        config.split_k = min(config.split_k * 2, K / config.block_k);
      }
      
      return config;
    }
    ```
- Evidence mapping:
  - "M-dependent tiling" → Different configs for `M <= 4`, `M <= 16`, etc.
  - "CU utilization" → `num_blocks < num_cus` check for occupancy
  - "Adaptive split_k" → Split factor adjusted based on problem size

## Optimization 6: Bias Support for Multiple Datatypes
- Commit ID: a3a782801
- Optimization type: Fusion
- Summary: Add bias support for FP16, BF16, and FP8 skinny GEMMs
- Detailed explanation:
  This optimization adds fused bias addition to the skinny GEMM kernels, supporting multiple datatypes. The bias is added in the epilogue to avoid extra memory traffic.
- Code excerpt:
    ```cpp
    // Skinny GEMM with fused bias
    template <typename T, typename BiasT>
    __global__ void skinny_gemm_bias_kernel(
        T* __restrict__ out,
        const T* __restrict__ A,
        const T* __restrict__ B,
        const BiasT* __restrict__ bias,  // [N]
        int M, int N, int K) {
      
      // ... GEMM computation ...
      T result = compute_gemm(A, B, row, col);
      
      // Add bias with type conversion if needed
      if (bias != nullptr) {
        if constexpr (std::is_same_v<T, BiasT>) {
          result += bias[col];
        } else {
          result += static_cast<T>(bias[col]);
        }
      }
      
      out[row * N + col] = result;
    }
    
    // Dispatch based on bias presence and type
    void skinny_gemm_dispatch(out, A, B, bias) {
      if (bias.defined()) {
        if (bias.scalar_type() == at::kHalf) {
          launch_kernel<half, half>(out, A, B, bias);
        } else if (bias.scalar_type() == at::kBFloat16) {
          launch_kernel<half, __hip_bfloat16>(out, A, B, bias);
        }
      } else {
        launch_kernel<half, half>(out, A, B, nullptr);
      }
    }
    ```
- Evidence mapping:
  - "Fused bias" → Bias added in same kernel as GEMM
  - "Multi-dtype support" → Template parameters for `T` and `BiasT`
  - "Type conversion" → `static_cast<T>(bias[col])` when types differ
