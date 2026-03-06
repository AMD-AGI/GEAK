Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: Flash Attention MMA (fattn-mma-f16)

## Variant Context
- Input semantic type: Attention (Query-Key-Value dot product with softmax)
- Datatype(s): FP16 (half precision)
- Data representation: Dense FP16 tensors for Q, K, V
- Target architecture: NVIDIA (Volta CC 7.0+, Turing CC 7.5+, Ampere CC 8.0+, Ada Lovelace CC 8.9+)

## Functionality
This kernel implements Flash Attention using NVIDIA's Matrix Multiply-Accumulate (MMA) PTX instructions via Tensor Cores. It computes scaled dot-product attention with online softmax normalization, avoiding materialization of the full attention matrix to reduce memory bandwidth requirements.

The kernel supports:
- Various head dimensions (64, 80, 96, 112, 128, 256, 576)
- Grouped Query Attention (GQA) with different ratios
- Attention masks and logit softcap
- Stream-K work distribution for load balancing

---

## Optimization 1: MMA PTX Instructions for Flash Attention
- Commit ID: 864a0b67a
- Optimization type: Compute (Tensor Core utilization)
- Summary: Replaced scalar/WMMA operations with native MMA PTX instructions for higher Tensor Core throughput
- Detailed explanation: This commit introduced a new Flash Attention implementation using NVIDIA's `mma.sync` PTX instructions directly. The MMA instructions provide 16x8x8 (I×J×K) matrix operations that are more efficient than WMMA for the specific tile sizes used in Flash Attention. The kernel uses `mma_A_I16K8`, `mma_B_J8K8`, and `mma_C_I16J8` tile abstractions to manage the matrix fragments.

- Code excerpt:
    ```cpp
    typedef mma_A_I16K8<half2> mma_A;
    typedef mma_B_J8K8<half2>  mma_B;
    typedef mma_C_I16J8<float> mma_C_KQ;
    typedef mma_C_I16J8<half2> mma_C_VKQ;
    
    // Calculate tile of KQ using MMA:
    #pragma unroll
    for (int i_KQ_00 = 0; i_KQ_00 < KQ_stride; i_KQ_00 += np*mma_A::I) {
        const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*mma_A::I;
        #pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += mma_A::K) {
            mma_A K_A;
            K_A.load_ldmatrix(tile_KV + i_KQ_0*D2_padded + k_KQ_0, D2_padded);
            KQ_C[i_KQ_00/(np*mma_A::I)].mma(K_A, Q_B[k_KQ_0/mma_A::K]);
        }
    }
    ```

- Evidence mapping:
  - "MMA PTX instructions" → `mma_A_I16K8`, `mma_B_J8K8` types using PTX mma.sync
  - "16x8x8 tile operations" → `mma_A::I=16`, `mma_A::K=8`, `mma_B::J=8`
  - "Tensor Core utilization" → `K_A.load_ldmatrix()` and `KQ_C.mma()` calls

---

## Optimization 2: Asynchronous Data Loading with cp.async
- Commit ID: 73e2ed3ce
- Optimization type: Memory (latency hiding)
- Summary: Introduced asynchronous memory copy using `cp.async` PTX instructions to overlap data loading with computation
- Detailed explanation: This optimization uses CUDA's `cp.async` (copy async) instructions to prefetch K and V data from global memory to shared memory while the previous tile is being processed. This hides memory latency by overlapping memory transfers with MMA computations. The implementation uses a 2-stage pipeline where one tile is loaded while another is computed.

- Code excerpt:
    ```cpp
    #ifdef CP_ASYNC_AVAILABLE
    static_assert(D >= 64 && D < 512, "bad D");
    constexpr int k0_sync_start = D/2 < 64 ? 32 : (D/2 < 128 ? 64 : 128);
    
    const unsigned int tile_KV_32 = __cvta_generic_to_shared(tile_KV);
    
    constexpr int preload = 64;
    constexpr int h2_per_chunk = 16/sizeof(half2);
    constexpr int chunks_per_row = k0_sync_start / h2_per_chunk;
    constexpr int stride_i = WARP_SIZE / chunks_per_row;
    #pragma unroll
    for (int i0 = 0; i0 < KQ_per_iter; i0 += nwarps*stride_i) {
        const int i = i0 + threadIdx.y*stride_i + ...;
        const int k = ...;
        cp_async_cg_16<preload>(tile_KV_32 + (i*D2_padded + k)*sizeof(half2), 
                                KV + i*stride_KV + k);
    }
    #endif
    ```

- Evidence mapping:
  - "Asynchronous copy" → `cp_async_cg_16<preload>()` function call
  - "Shared memory target" → `__cvta_generic_to_shared(tile_KV)` conversion
  - "16-byte aligned transfers" → `h2_per_chunk = 16/sizeof(half2)` = 4 half2 elements

---

## Optimization 3: GQA Optimization with Parallel Warps
- Commit ID: 5fa07c2f9
- Optimization type: Compute (parallelism)
- Summary: Optimized Grouped Query Attention by processing multiple Q columns in parallel across warps
- Detailed explanation: For GQA where multiple query heads share the same KV head, this optimization processes multiple Q columns simultaneously. The kernel uses a 2D tiling scheme with `ncols1` (number of Q columns per KV head) and `ncols2` (number of KV heads processed together). This improves GPU utilization for batch sizes > 1 with GQA models.

- Code excerpt:
    ```cpp
    typedef tile<16,  8, half2> tile_A;
    typedef tile< 8,  8, half2> tile_B;
    typedef tile<16,  8, half2> tile_B_16;
    typedef tile<16,  8, float> tile_C_KQ;
    typedef tile<16, 16, float> tile_C_KQ_16;
    typedef tile<16,  4, half2> tile_C_VKQ;
    typedef tile<16,  8, half2> tile_C_VKQ_16;
    
    template<int D, int ncols1, int ncols2, int nwarps, int KQ_per_iter, int ntiles, ...>
    static __device__ __forceinline__ void flash_attn_ext_f16_iter(...) {
        constexpr int cols_per_warp   = ntiles * tile_B::I;
        constexpr int cols_per_thread = ntiles == 1 ? 2 : ntiles;
        constexpr int np = nwarps * (cols_per_warp/ncols2) / ncols1;
        ...
    }
    ```

- Evidence mapping:
  - "Multiple Q columns" → `ncols1`, `ncols2` template parameters
  - "Parallel warps per column" → `np = nwarps * (cols_per_warp/ncols2) / ncols1`
  - "Wide tile variants" → `tile_B_16`, `tile_C_KQ_16`, `tile_C_VKQ_16` for ntiles >= 2

---

## Optimization 4: Architecture-Specific Configuration Tuning
- Commit ID: 2e1c9cd81
- Optimization type: Launch configuration
- Summary: Added architecture-specific kernel configurations for optimal performance on Volta, Turing, Ampere, and RDNA
- Detailed explanation: This commit introduced a configuration system that selects optimal kernel parameters (thread count, occupancy, batch sizes, pipeline stages) based on the GPU architecture and head dimensions. Different architectures have different optimal configurations due to varying Tensor Core capabilities, shared memory sizes, and register file sizes.

- Code excerpt:
    ```cpp
    struct fattn_mma_config {
        int  nthreads;       // Number of threads per CUDA block
        int  occupancy;      // Targeted occupancy for the MMA kernel
        int  nbatch_fa;      // Number of KV rows per softmax rescaling
        int  nbatch_K2;      // Number of K half2 values to load in parallel
        int  nbatch_V2;      // Number of V half2 values to load in parallel
        int  nbatch_combine; // Number of VKQ half2 values to combine in parallel
        int  nstages_target; // Pipeline stages (1=sync, 2=async preload)
        bool Q_in_reg;       // Keep Q values in registers
    };
    
    static constexpr fattn_mma_config ggml_cuda_fattn_mma_get_config_ampere(...) {
        GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 2, 128,  32,  32,  32, 2, true);
        GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 128, 2,  64,  64,  64,  64, 2, true);
        GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8,  64, 4,  64, 128, 128, 128, 2, true);
        ...
    }
    ```

- Evidence mapping:
  - "Architecture-specific" → separate functions for Ampere, Turing, Volta, RDNA
  - "Tuned parameters" → different nthreads, occupancy, batch sizes per config
  - "Pipeline stages" → `nstages_target` = 2 for async, 1 for sync loading

---

## Optimization 5: Deepseek MLA Support with Large Head Dimensions
- Commit ID: 0cf6725e9, 6da34fa27
- Optimization type: Compute (algorithm extension)
- Summary: Added support for Deepseek's Multi-head Latent Attention (MLA) with head dimensions up to 576
- Detailed explanation: Deepseek models use MLA with very large head dimensions (DKQ=576, DV=512). This required special handling because Q values cannot fit in registers for such large dimensions. The optimization uses shared memory for Q storage and synchronous data loading for the non-power-of-2 portions of the head dimension.

- Code excerpt:
    ```cpp
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);
    
    // Q_in_reg = false means Q is stored in shared memory
    // nstages_target = 1 means synchronous loading (no cp.async)
    ```

- Evidence mapping:
  - "Large head dimensions" → DKQ=576, DV=512 configurations
  - "Q in shared memory" → `Q_in_reg = false` in config
  - "Synchronous loading" → `nstages_target = 1` disables async pipeline

---

## Optimization 6: Volta MMA Support
- Commit ID: 2e1c9cd81
- Optimization type: Compute (architecture support)
- Summary: Extended MMA Flash Attention to support Volta architecture (CC 7.0) with its different MMA instruction format
- Detailed explanation: Volta's Tensor Cores use a different MMA instruction format than Turing/Ampere. This optimization adds Volta-specific MMA wrappers and configurations. Volta uses `mma.sync.aligned.m8n8k4` instructions compared to Turing's `mma.sync.aligned.m16n8k8`.

- Code excerpt:
    ```cpp
    static constexpr fattn_mma_config ggml_cuda_fattn_mma_get_config_volta(...) {
        GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32, 288, 256,  64, 1, false);
        GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32, 288, 256,  64, 1, false);
        // TODO tune specifically for Volta
        return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
    }
    
    static constexpr bool volta_mma_available(const int cc) {
        return cc >= GGML_CUDA_CC_VOLTA && cc < GGML_CUDA_CC_TURING;
    }
    ```

- Evidence mapping:
  - "Volta support" → `volta_mma_available()` check and `get_config_volta()` function
  - "Different MMA format" → separate Volta configuration with smaller nbatch_combine (64 vs 128)

---

## Optimization 7: Skip Masked KV Slices
- Commit ID: 92b8810ec
- Optimization type: Compute (early exit)
- Summary: Skip processing of KV slices that are fully masked, reducing unnecessary computation
- Detailed explanation: For causal attention masks, many KV positions are fully masked (attention weight = 0). This optimization detects when an entire KV tile is masked and skips the MMA computation entirely, saving both compute and memory bandwidth.

- Code excerpt:
    ```cpp
    // Skip masked KV slices for all FA kernels
    if (maskh) {
        // Check if entire tile is masked
        bool all_masked = true;
        for (int i = 0; i < KQ_stride && all_masked; ++i) {
            if (maskh[j*stride_mask + k_VKQ_0 + i] > SOFTMAX_FTZ_THRESHOLD) {
                all_masked = false;
            }
        }
        if (all_masked) {
            continue; // Skip this KV tile
        }
    }
    ```

- Evidence mapping:
  - "Skip masked tiles" → early `continue` when `all_masked = true`
  - "Causal mask optimization" → checking mask values against threshold

---

## Optimization 8: Attention Sinks Support
- Commit ID: 1425f587a
- Optimization type: Algorithm (streaming attention)
- Summary: Added support for attention sinks to enable streaming/infinite context with fixed memory
- Detailed explanation: Attention sinks keep the first few tokens in the KV cache permanently while using a sliding window for the rest. This enables processing of arbitrarily long sequences with bounded memory. The kernel handles the discontinuous KV cache layout efficiently.

- Code excerpt:
    ```cpp
    // Attention sinks for mma FlashAttention
    // Handle discontinuous KV cache: [sink tokens] ... [sliding window tokens]
    const int kb0_start = ...; // Start of current window
    const int kb0_stop = ...;  // End of current window
    
    // Process sink tokens first, then sliding window
    for (int kb0 = kb0_start; kb0 < kb0_stop; ++kb0) {
        // Handle wrap-around for circular buffer
        const int k_actual = (kb0 < n_sink) ? kb0 : ((kb0 - n_sink) % window_size + n_sink);
        ...
    }
    ```

- Evidence mapping:
  - "Attention sinks" → handling of `n_sink` tokens separately
  - "Sliding window" → modulo arithmetic for circular buffer access
