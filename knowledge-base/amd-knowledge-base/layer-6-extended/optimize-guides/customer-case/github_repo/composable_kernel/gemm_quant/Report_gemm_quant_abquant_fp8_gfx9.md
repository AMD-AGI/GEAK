---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: gemm_quant (Quantized GEMM with Block Scale)

## Variant Context
- Input semantic type: Matrix multiplication with quantization
- Datatype(s): FP8/INT8/FP4 with block-scale quantization
- Data representation: Block-scale quantized with per-group scale factors
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
The quantized GEMM kernel computes C = dequant(A) × dequant(B) where A and B are stored in low-precision formats (FP8, INT8, FP4) with per-block or per-group scale factors. The kernel supports various quantization schemes including A-only quantization (AQuant), B-only quantization (BQuant), and both A and B quantization (ABQuant).

## Optimization 1: GEMM Blockscale ABQuant Hot Loop Scheduler
- Commit ID: 31a35ecab
- Optimization type: scheduling / compute
- Summary: Optimized the hot loop scheduler for ABQuant GEMM by normalizing scheduling costs based on loop iteration count and improving VALU instruction interleaving.

- Detailed explanation:
  The optimization improves the instruction scheduling in the hot loop by:
  1. Normalizing DS read and MFMA instruction counts by the number of loop iterations (`nloop`)
  2. Increasing VALU scheduling slots from 2 to 4 to better overlap blockscale calculations with MFMA
  3. Using `__builtin_amdgcn_sched_group_barrier` for fine-grained instruction scheduling

  This ensures better overlap between memory operations, compute operations, and the auxiliary VALU instructions needed for blockscale calculations.

- Code excerpt:
    ```cpp
    /**
     * @tparam nloop The number of iterations in the hot loop,
     * used to normalize scheduling costs.
     */
    template <index_t nloop>
    CK_TILE_HOST_DEVICE static constexpr auto HotLoopScheduler()
    {
        static_assert(nloop > 0, "nloop must be greater than 0");
        // Estimated number of VMEM vector loads for A per block
        constexpr index_t Aload_inst =
            kMPerBlock * kKPerBlock / (kBlockSize * sizeof(ADataType) * 4);
        // Estimated number of VMEM vector loads for B per block
        constexpr index_t Bload_inst =
            kNPerBlock * kKPerBlock / (kBlockSize * sizeof(BDataType) * 4);
        // Quantization scale loads
        constexpr index_t BQload_inst = kNPerBlock * kKPerBlock / 
            (kBlockSize * BQuantGroupSize::kN * BQuantGroupSize::kK);
        
        // Total VMEM load instructions
        constexpr index_t buffer_load_inst = Aload_inst + Bload_inst + BQload_inst;
        // Normalize LDS reads by loop count
        constexpr index_t ds_read_inst = kMPerBlock / kLdsInstCycle / nloop;
        constexpr index_t ds_write_inst = Aload_inst;
        // Normalize MFMA instructions by loop count squared
        constexpr index_t mfma_inst =
            ((kMPerBlock / WG::kM) / nloop) * ((kNPerBlock / WG::kN) / nloop);
        
        // Schedule interleaving ratios
        constexpr index_t ds_rep = mfma_inst / (ds_read_inst + ds_write_inst);
        constexpr index_t vmem_rep = mfma_inst / buffer_load_inst;
        
        // Apply scheduling with increased VALU slots for blockscale
        static_for<0, mfma_inst, 1>{}([&](auto) {
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0);
            if constexpr(i % ds_rep == 0) {
                __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::DS, 1, 0);
            }
            if constexpr(i % vmem_rep == 0) {
                __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VMEM, 1, 0);
            }
            // Increased VALU slots for blockscale calculation overlap
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 4, 0);
        });
    }
    ```

- Evidence mapping:
  - "Normalize by loop count" → `/ nloop` in `ds_read_inst` and `mfma_inst` calculations
  - "Increased VALU slots" → `VALU, 4, 0` (changed from 2 to 4)
  - "Fine-grained scheduling" → `__builtin_amdgcn_sched_group_barrier` calls

## Optimization 2: Improved Pipeline Comments and Structure
- Commit ID: 31a35ecab
- Optimization type: code maintainability / scheduling
- Summary: Added clear comments documenting the pipeline stages (ds_write, ds_read, prefetch) for better code understanding and maintenance.

- Detailed explanation:
  The optimization adds explicit comments marking each pipeline stage in the hot loop, making it easier to understand the data flow and identify optimization opportunities.

- Code excerpt:
    ```cpp
    while(iCounter > 0)
    {
        __builtin_amdgcn_sched_barrier(0);
        // Prefill A(2i+1) ds_write
        a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
        store_tile(a_copy_lds_window_pong, a_block_tile_tmp);

        // ... GEMM computation ...
        
        // prefetch Q(2i+1)
        aq_block_tile_2 = load_tile(aq_copy_dram_window);
        move_tile_window(aq_copy_dram_window, {0, KPerBlockAQ});
        bq_block_tile_2 = load_tile(bq_copy_dram_window);
        move_tile_window(bq_copy_dram_window, {0, KPerBlockBQ});

        // Preload A(2i+1) ds_read
        static_for<0, m_preload, 1>{}([&](auto loadIter) {
            // ... preload logic ...
        });
        
        // Preload A(2i+2) ds_read
        static_for<0, m_preload, 1>{}([&](auto loadIter) {
            // ... preload logic ...
        });
    }
    ```

- Evidence mapping:
  - "Pipeline stage comments" → `// Prefill A(2i+1) ds_write`, `// prefetch Q(2i+1)`, `// Preload A(2i+1) ds_read`
  - "Clear data flow" → Sequential comments showing A tile, Q tile, and preload stages

## Optimization 3: Interwave Scheduler for AQuant Memory Pipeline
- Commit ID: b8751e505
- Optimization type: scheduling / compute
- Summary: Added interwave scheduling for the AQuant memory pipeline, enabling better utilization of multiple MAC clusters through wave-level parallelism.

- Detailed explanation:
  The interwave scheduler divides the K dimension across multiple MAC clusters (waves), allowing different waves to work on different K slices simultaneously. This improves hardware utilization by overlapping memory access and compute across waves.

  Key features:
  - K dimension is divided into `KRepeat` chunks, each processed by different waves
  - Local prefetch is performed per K chunk to hide memory latency
  - Scheduling barriers ensure proper synchronization between waves

- Code excerpt:
    ```cpp
    template <typename GemmTraits>
    struct BlockGemmImpl<GemmPipelineScheduler::Interwave, GemmTraits>
    {
        static constexpr index_t KPerThread     = GemmTraits::KPerThread;
        static constexpr index_t NumMacClusters = GemmTraits::InterWaveSchedulingMacClusters;

        static constexpr index_t KPerInnerLoop =
            ck_tile::max(KPerThread / NumMacClusters, WarpGemm::kKPerThread);
        static constexpr index_t KRepeat        = KPerThread / KPerInnerLoop;
        static constexpr index_t KInnerLoopIter = KPerInnerLoop / WarpGemm::kKPerThread;

        template <index_t KIdx, typename ASmemBlockWindow, typename BSmemBlockWindow>
        CK_TILE_DEVICE void LocalPrefetch(const ASmemBlockWindow& a_block_window,
                                          const BSmemBlockWindow& b_block_window)
        {
            constexpr auto k_idx_offset = KIdx * KPerInnerLoop;
            // Load A and B tiles for this K chunk
            auto a_lds_gemm_window = make_tile_window(
                a_block_window.get_bottom_tensor_view(), a_lds_shape, 
                {0, k_idx_offset}, a_lds_load_distr);
            auto b_lds_gemm_window = make_tile_window(
                b_block_window.get_bottom_tensor_view(), b_lds_shape,
                {0, k_idx_offset}, b_lds_load_distr);

            load_int4_tile<BDataType, ComputeDataType>(a_warp_tile_, a_lds_gemm_window);
            load_int4_tile<BDataType, ComputeDataType>(b_warp_tile_, b_lds_gemm_window);
        }
        
        // Main GEMM loop with interwave scheduling
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor, ...)
        {
            index_t current_k_repeat_loaded = -1;

            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    static_for<0, Traits::QScalesPerBlockRow, 1>{}([&](auto kQScale) {
                        static_for<0, Traits::KIterPerQScale, 1>{}([&](auto kIterInQScale) {
                            constexpr auto kRepeatIdx = kIterGlobal / KInnerLoopIter;
                            constexpr auto kInnerIdx  = kIterGlobal % KInnerLoopIter;

                            // Prefetch new K chunk if needed
                            if constexpr(kInnerIdx == 0)
                            {
                                if(current_k_repeat_loaded != kRepeatIdx)
                                {
                                    LocalPrefetch<kRepeatIdx>(a_block_window, b_block_window);
                                    __builtin_amdgcn_sched_barrier(0);
                                    
                                    if constexpr(kRepeatIdx != 0 || KRepeat == 1)
                                    {
                                        __builtin_amdgcn_s_barrier();
                                    }
                                    current_k_repeat_loaded = kRepeatIdx;
                                }
                            }
                            // ... GEMM accumulation ...
                        });
                    });
                });
                
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_setprio(0);
            });
        }
    };
    ```

- Evidence mapping:
  - "Multiple MAC clusters" → `NumMacClusters` and `KRepeat = KPerThread / KPerInnerLoop`
  - "K dimension division" → `KPerInnerLoop` and `KInnerLoopIter` calculations
  - "Wave-level parallelism" → `LocalPrefetch<kRepeatIdx>` template parameter
  - "Scheduling barriers" → `__builtin_amdgcn_sched_barrier(0)` and `__builtin_amdgcn_s_barrier()`
  - "Priority control" → `__builtin_amdgcn_s_setprio(1)` and `__builtin_amdgcn_s_setprio(0)`
