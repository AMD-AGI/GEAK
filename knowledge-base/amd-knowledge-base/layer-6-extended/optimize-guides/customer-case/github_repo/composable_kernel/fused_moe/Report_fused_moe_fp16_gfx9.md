Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: fused_moe (Fused Mixture of Experts)

## Variant Context
- Input semantic type: Mixture of Experts computation for LLM inference
- Datatype(s): FP16/BF16/FP8
- Data representation: Token-expert routing with sparse activation
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
The Fused MoE kernel implements the Mixture of Experts layer commonly used in large language models like Mixtral and GPT-4. It handles token routing to experts, expert computation (typically GEMM), and result aggregation. The kernel supports TopK expert selection and various quantization schemes.

## Optimization 1: Optimized Token Sorting with Local Tokens
- Commit ID: a4e1248db
- Optimization type: compute / scheduling
- Summary: Added "local_tokens" feature for expert-parallel (EP) cases, optimizing token sorting for distributed MoE.

- Detailed explanation:
  In expert-parallel deployment, each GPU handles a subset of experts. The local_tokens optimization tracks which tokens are routed to local experts, avoiding unnecessary computation and communication for tokens routed to remote experts.

- Code excerpt:
    ```cpp
    // MoE sorting with local token tracking
    template <typename Problem>
    struct MoESortingKernel
    {
        // Track tokens routed to local experts
        struct LocalTokenInfo
        {
            index_t token_id;
            index_t expert_id;
            float routing_weight;
        };
        
        CK_TILE_DEVICE void sort_tokens_local(
            const float* routing_scores,
            const index_t* expert_ids,
            LocalTokenInfo* local_tokens,
            index_t* local_token_count,
            index_t local_expert_start,
            index_t local_expert_end)
        {
            // Only process tokens routed to local experts
            for(index_t i = tid; i < num_tokens; i += blockDim.x)
            {
                index_t expert = expert_ids[i];
                if(expert >= local_expert_start && expert < local_expert_end)
                {
                    index_t local_idx = atomicAdd(local_token_count, 1);
                    local_tokens[local_idx] = {i, expert, routing_scores[i]};
                }
            }
        }
    };
    ```

- Evidence mapping:
  - "Local tokens" → `LocalTokenInfo` struct and filtering
  - "Expert-parallel" → `local_expert_start/end` range check
  - "Avoid remote tokens" → Only processing tokens for local experts

## Optimization 2: Subtoken Logic Refactoring
- Commit ID: 8aff45a8a
- Optimization type: compute
- Summary: Refactored subtoken logic to enable more kernels to use the memory-efficient (MP) kernel variant.

- Detailed explanation:
  The subtoken optimization splits large tokens into smaller subtokens that can be processed more efficiently. The refactoring allows more problem configurations to use the optimized MP (memory-packed) kernel instead of falling back to slower variants.

- Code excerpt:
    ```cpp
    // Subtoken configuration for MP kernel eligibility
    template <typename Problem>
    struct SubtokenConfig
    {
        // Determine if MP kernel can be used
        static constexpr bool kCanUseMPKernel = 
            (Problem::kHiddenDim % kSubtokenSize == 0) &&
            (kSubtokenSize >= kMinSubtokenSize) &&
            (Problem::kNumExperts <= kMaxExpertsForMP);
        
        // Subtoken size selection
        static constexpr index_t kSubtokenSize = []() {
            if constexpr(Problem::kHiddenDim >= 4096)
                return 256;
            else if constexpr(Problem::kHiddenDim >= 2048)
                return 128;
            else
                return 64;
        }();
    };
    ```

- Evidence mapping:
  - "Subtoken logic" → `kSubtokenSize` calculation
  - "MP kernel eligibility" → `kCanUseMPKernel` check
  - "More configurations" → Relaxed constraints for MP kernel

## Optimization 3: SGPR Buffer Resource Optimization
- Commit ID: ef4307878
- Optimization type: compute
- Summary: Used `__builtin_amdgcn_readfirstlane` for buffer resource in fused_moe, reducing VGPR pressure.

- Detailed explanation:
  Buffer resource descriptors are uniform across all lanes in a wave. By using `readfirstlane` to broadcast the resource from lane 0, we can store it in SGPRs instead of VGPRs, freeing up vector registers for computation.

- Code excerpt:
    ```cpp
    // Use SGPR for buffer resource descriptor
    CK_TILE_DEVICE auto make_buffer_resource_sgpr(const void* ptr, index_t size)
    {
        // Read pointer from first lane to SGPR
        uint64_t ptr_val = reinterpret_cast<uint64_t>(ptr);
        uint32_t ptr_lo = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(ptr_val));
        uint32_t ptr_hi = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(ptr_val >> 32));
        
        // Construct buffer resource in SGPR
        int32x4_t rsrc;
        rsrc[0] = ptr_lo;
        rsrc[1] = ptr_hi;
        rsrc[2] = size;
        rsrc[3] = 0x00027000;  // Buffer resource flags
        
        return rsrc;
    }
    ```

- Evidence mapping:
  - "readfirstlane" → `__builtin_amdgcn_readfirstlane` intrinsic
  - "SGPR storage" → Buffer resource in scalar registers
  - "VGPR pressure reduction" → Freeing vector registers

## Optimization 4: GFX950 Shuffle Fix
- Commit ID: 00c46785a
- Optimization type: correctness / compute
- Summary: Fixed shuffle operations for GFX950 architecture compatibility.

- Detailed explanation:
  GFX950 has different shuffle instruction behavior compared to earlier architectures. This fix ensures correct cross-lane data movement for MoE token routing and aggregation.

- Code excerpt:
    ```cpp
    // Architecture-specific shuffle implementation
    template <typename T>
    CK_TILE_DEVICE T wave_shuffle(T val, index_t src_lane)
    {
    #if defined(__gfx950__)
        // GFX950-specific shuffle
        return __builtin_amdgcn_ds_bpermute(src_lane << 2, val);
    #else
        // Standard shuffle for other architectures
        return __builtin_amdgcn_ds_swizzle(val, src_lane);
    #endif
    }
    ```

- Evidence mapping:
  - "GFX950 specific" → `#if defined(__gfx950__)` guard
  - "Shuffle fix" → Different intrinsic for GFX950
  - "Cross-lane movement" → `ds_bpermute` vs `ds_swizzle`

## Optimization 5: Sorting Optimization for Local Tokens
- Commit ID: cfe211cc6
- Optimization type: compute
- Summary: Optimized local_token sorting to reduce synchronization overhead.

- Detailed explanation:
  The sorting optimization uses warp-level primitives to reduce the number of atomic operations and synchronization points when counting and sorting tokens for each expert.

- Code excerpt:
    ```cpp
    // Warp-level token counting optimization
    CK_TILE_DEVICE void count_tokens_per_expert_optimized(
        const index_t* expert_ids,
        index_t* expert_counts,
        index_t num_tokens)
    {
        // Warp-level histogram
        __shared__ index_t warp_histograms[kNumWarps][kNumExperts];
        
        // Each warp builds local histogram
        for(index_t i = warp_start + lane_id; i < warp_end; i += kWarpSize)
        {
            index_t expert = expert_ids[i];
            atomicAdd(&warp_histograms[warp_id][expert], 1);
        }
        
        __syncthreads();
        
        // Reduce warp histograms (one thread per expert)
        if(tid < kNumExperts)
        {
            index_t count = 0;
            for(index_t w = 0; w < kNumWarps; ++w)
                count += warp_histograms[w][tid];
            expert_counts[tid] = count;
        }
    }
    ```

- Evidence mapping:
  - "Warp-level histogram" → Per-warp counting in shared memory
  - "Reduced atomics" → Atomics only within warp, then reduction
  - "Less synchronization" → Single syncthreads instead of global atomics
