---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: Batched and Grouped GEMM

## Variant Context
- Input semantic type: Batched/Grouped matrix multiplication
- Datatype(s): FP16/BF16/FP32/FP8
- Data representation: Multiple matrices with uniform or variable sizes
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
- **Batched GEMM**: Multiple GEMMs with identical M, N, K dimensions but different data
- **Grouped GEMM**: Multiple GEMMs with potentially different M, N, K per group

These kernels are essential for transformer models (batched attention), mixture of experts (grouped expert computation), and multi-head operations.

## Optimization 1: Persistent Kernel for Grouped GEMM with MultiD
- Commit ID: bebf0e9d1
- Optimization type: scheduling / compute
- Summary: Extended Grouped GEMM with Multi-D feature to use persistent kernel option for better GPU utilization.

- Detailed explanation:
  Persistent kernels keep workgroups alive to process multiple tiles, reducing kernel launch overhead. For grouped GEMM, this is particularly beneficial when groups have varying sizes, as workgroups can dynamically pick up work from any group.

- Code excerpt:
    ```cpp
    // Persistent Grouped GEMM with MultiD
    template <typename Problem>
    struct GroupedGemmPersistentKernel
    {
        // Multi-D: support multiple D tensors (bias, residual, etc.)
        static constexpr index_t kNumDTensors = Problem::kNumDTensors;
        
        CK_TILE_DEVICE void operator()(
            const GroupedGemmArgs& args,
            void* workspace)
        {
            // Persistent loop: each workgroup processes multiple tiles
            index_t tile_idx = blockIdx.x;
            const index_t total_tiles = args.total_tiles;
            const index_t grid_size = gridDim.x;
            
            while(tile_idx < total_tiles)
            {
                // Find which group this tile belongs to
                auto [group_idx, local_tile_idx] = 
                    find_group_and_tile(args.group_tile_offsets, tile_idx);
                
                // Get group-specific dimensions
                index_t M = args.Ms[group_idx];
                index_t N = args.Ns[group_idx];
                index_t K = args.Ks[group_idx];
                
                // Compute tile indices within group
                auto [m_tile, n_tile] = decompose_tile_idx(local_tile_idx, M, N);
                
                // Execute GEMM for this tile
                gemm_tile(args.As[group_idx], args.Bs[group_idx], 
                          args.Cs[group_idx], args.Ds[group_idx],
                          m_tile, n_tile, M, N, K);
                
                // Move to next tile (strided by grid size)
                tile_idx += grid_size;
            }
        }
    };
    ```

- Evidence mapping:
  - "Persistent kernel" → `while(tile_idx < total_tiles)` loop
  - "Multi-D" → `kNumDTensors` and `args.Ds[group_idx]`
  - "Dynamic work" → `find_group_and_tile` for variable group sizes

## Optimization 2: Multi-D GEMM Integration
- Commit ID: a44bea45b
- Optimization type: fusion
- Summary: Integrated Multi-D GEMMs into Grouped GEMMs for fused bias/residual operations.

- Detailed explanation:
  Multi-D GEMM computes C = A × B + D1 + D2 + ... where Di are additional tensors (bias, residual). Integrating this with grouped GEMM enables efficient fused operations for each group.

- Code excerpt:
    ```cpp
    // Multi-D GEMM epilogue
    template <typename Problem, index_t NumD>
    struct MultiDEpilogue
    {
        CK_TILE_DEVICE void operator()(
            AccType* c_acc,
            const std::array<const DType*, NumD>& d_ptrs,
            OutputType* output,
            index_t m, index_t n)
        {
            // Load accumulator
            AccType val = c_acc[m * N + n];
            
            // Add each D tensor
            for(index_t i = 0; i < NumD; ++i)
            {
                if(d_ptrs[i] != nullptr)
                {
                    val += static_cast<AccType>(d_ptrs[i][m * N + n]);
                }
            }
            
            // Store output
            output[m * N + n] = static_cast<OutputType>(val);
        }
    };
    ```

- Evidence mapping:
  - "Multi-D" → `std::array<const DType*, NumD>` for multiple D tensors
  - "Fused operations" → All additions in single epilogue
  - "Optional tensors" → `nullptr` check for unused D tensors

## Optimization 3: Pipeline Selection Fix for Variable Tail
- Commit ID: db79fad16
- Optimization type: correctness
- Summary: Fixed pipeline selection when tail_num varies per group, preventing numerical errors.

- Detailed explanation:
  Different groups may have different K dimensions, leading to different "tail" sizes (remainder after full tiles). The fix ensures correct pipeline selection for each group's tail handling.

- Code excerpt:
    ```cpp
    // Per-group pipeline selection
    template <typename Problem>
    struct GroupedGemmPipelineSelector
    {
        CK_TILE_DEVICE auto select_pipeline(index_t K, index_t group_idx)
        {
            // Compute tail for this group
            index_t num_k_tiles = K / kKPerBlock;
            index_t tail_k = K - num_k_tiles * kKPerBlock;
            
            // Select pipeline based on tail size
            if(tail_k == 0)
            {
                return FullTilePipeline{};
            }
            else if(tail_k >= kKPerBlock / 2)
            {
                return LargeTailPipeline{};
            }
            else
            {
                return SmallTailPipeline{};
            }
        }
    };
    ```

- Evidence mapping:
  - "Variable tail" → `tail_k` computed per group
  - "Pipeline selection" → Different pipelines for different tail sizes
  - "Per-group" → `group_idx` parameter for group-specific handling

## Optimization 4: Batched GEMM IsSupported Checks
- Commit ID: 302160421
- Optimization type: correctness / robustness
- Summary: Added comprehensive IsSupported function checks for batched GEMM kernel.

- Detailed explanation:
  The IsSupported function validates kernel arguments before launch, catching invalid configurations early. This includes alignment checks, size limits, and stride validation.

- Code excerpt:
    ```cpp
    // Batched GEMM argument validation
    template <typename Problem>
    struct BatchedGemmKernel
    {
        CK_TILE_HOST static bool IsSupported(const BatchedGemmArgs& args)
        {
            // Check alignment requirements
            bool a_aligned = (reinterpret_cast<uintptr_t>(args.a_ptr) % kAlignmentA) == 0;
            bool b_aligned = (reinterpret_cast<uintptr_t>(args.b_ptr) % kAlignmentB) == 0;
            bool c_aligned = (reinterpret_cast<uintptr_t>(args.c_ptr) % kAlignmentC) == 0;
            
            if(!a_aligned || !b_aligned || !c_aligned)
            {
                return false;
            }
            
            // Check stride alignment
            bool stride_a_valid = (args.stride_a % kAlignmentA) == 0;
            bool stride_b_valid = (args.stride_b % kAlignmentB) == 0;
            bool stride_c_valid = (args.stride_c % kAlignmentC) == 0;
            
            if(!stride_a_valid || !stride_b_valid || !stride_c_valid)
            {
                return false;
            }
            
            // Check size limits
            if(args.M > kMaxM || args.N > kMaxN || args.K > kMaxK)
            {
                return false;
            }
            
            // Check batch count
            if(args.batch_count <= 0)
            {
                return false;
            }
            
            return true;
        }
    };
    ```

- Evidence mapping:
  - "IsSupported checks" → Comprehensive validation function
  - "Alignment" → Pointer and stride alignment checks
  - "Size limits" → `kMaxM`, `kMaxN`, `kMaxK` bounds

## Optimization 5: CShuffle Epilogue Wave Per Shuffle Fix
- Commit ID: 3b773109e
- Optimization type: correctness
- Summary: Fixed wave per shuffle calculation in CShuffle epilogue for correct data movement.

- Detailed explanation:
  CShuffle (C-shuffle) is a technique to rearrange output data for efficient memory writes. The fix ensures correct wave assignment for the shuffle operation.

- Code excerpt:
    ```cpp
    // CShuffle epilogue with correct wave assignment
    template <typename Problem>
    struct CShuffleEpilogue
    {
        // Fixed wave per shuffle calculation
        static constexpr index_t kWavesPerShuffle = 
            (kMPerBlock * kNPerBlock) / (kWaveSize * kElementsPerThread);
        
        CK_TILE_DEVICE void shuffle_and_store(
            const AccType* c_acc,
            OutputType* output)
        {
            const index_t wave_id = get_wave_id();
            const index_t lane_id = get_lane_id();
            
            // Determine shuffle group
            index_t shuffle_group = wave_id / kWavesPerShuffle;
            index_t wave_in_group = wave_id % kWavesPerShuffle;
            
            // Compute output indices with shuffle
            index_t out_idx = compute_shuffled_index(shuffle_group, wave_in_group, lane_id);
            
            // Store with coalesced access pattern
            output[out_idx] = static_cast<OutputType>(c_acc[lane_id]);
        }
    };
    ```

- Evidence mapping:
  - "Wave per shuffle" → `kWavesPerShuffle` calculation
  - "Shuffle groups" → `shuffle_group` and `wave_in_group`
  - "Coalesced access" → Shuffled index for memory efficiency
