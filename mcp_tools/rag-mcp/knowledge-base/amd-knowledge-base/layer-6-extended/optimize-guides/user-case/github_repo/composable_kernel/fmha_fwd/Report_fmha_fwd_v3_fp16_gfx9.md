# Kernel: fmha_fwd_v3 (Flash Attention V3 Forward Pipeline)

## Variant Context
- Input semantic type: Attention (Query, Key, Value matrices for transformer models)
- Datatype(s): FP16/BF16
- Data representation: Dense tensors with optional causal masking
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
The FAv3 (Flash Attention Version 3) forward pipeline is an optimized implementation of scaled dot-product attention that uses advanced scheduling techniques to maximize hardware utilization. It features wave-group based scheduling, packed FP32 operations, and fine-grained instruction interleaving for MFMA, VALU, SALU, and memory operations.

## Optimization 1: Wave-Group Based Core Loop Scheduler
- Commit ID: d876e87fe
- Optimization type: scheduling / compute
- Summary: Introduced a wave-group based scheduler that interleaves MFMA, TRANS (transpose), and VALU instructions across different phases for optimal instruction-level parallelism.

- Detailed explanation:
  The FAv3 pipeline uses a `CoreLoopScheduler` template that schedules instructions differently based on:
  1. Wave group (0 or 1) - different waves handle different phases
  2. Phase (0-3) - different computation stages within the attention loop
  3. Masking mode - different scheduling for masked vs non-masked attention

  The scheduler uses `__builtin_amdgcn_sched_group_barrier` to control instruction ordering:
  - 0x008: MFMA instructions
  - 0x200: TRANS (transpose) instructions  
  - 0x002: VALU instructions
  - 0x004: SALU instructions

- Code excerpt:
    ```cpp
    template <typename PipelineProblem>
    struct CoreLoopScheduler<PipelineProblem, /*kIsMasking=*/true>
    {
        template <ck_tile::index_t WaveGroup, ck_tile::index_t Phase>
        CK_TILE_DEVICE static constexpr void schedule(ck_tile::number<WaveGroup>,
                                                      ck_tile::number<Phase>)
        {
            if constexpr(WaveGroup == 0)
            {
                if constexpr(Phase == 0)
                {
                    static_for<0, 8, 1>{}([&](auto) {
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x200, 2, 0); // TRANS
                        __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
                    });
                }
                else if constexpr(Phase == 2)
                {
    #if !CK_TILE_DISABLE_PACKED_FP32
                    __builtin_amdgcn_sched_group_barrier(0x002, 4, 0); // VALU
    #endif
                    static_for<0, 8, 1>{}([&](auto) {
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x002, 4, 0); // VALU
                    });
                }
            }
        }
    };
    ```

- Evidence mapping:
  - "Wave-group based scheduling" → `WaveGroup == 0` and `WaveGroup == 1` branches
  - "Phase-based scheduling" → `Phase == 0`, `Phase == 1`, etc. branches
  - "MFMA/TRANS/VALU interleaving" → `sched_group_barrier` calls with different masks

## Optimization 2: Packed FP32 Operations with Inline Assembly
- Commit ID: d876e87fe
- Optimization type: compute
- Summary: Uses packed FP32 operations via inline assembly to maximize ALU throughput and hide data movement latency.

- Detailed explanation:
  The pipeline uses inline assembly for critical FP32 operations to ensure optimal instruction encoding and avoid compiler-introduced inefficiencies:
  - `v_fma_f32`: Fused multiply-add with scalar operand
  - `v_pk_mul_f32`: Packed FP32 multiply (2 operations per instruction)
  - `v_cvt_pk_f16_f32` / `v_cvt_pk_bf16_f32`: Packed conversion for output

- Code excerpt:
    ```cpp
    namespace detail {
    CK_TILE_DEVICE float fma_impl_vsv(float a, float b, float c)
    {
    #if CK_TILE_DISABLE_PACKED_FP32
        return a * b + c;
    #else
        float result;
        asm volatile("v_fma_f32 %[result], %[a], %[b], %[c]"
                     : [result] "=v"(result)
                     : [a] "v"(a), [b] "s"(b), [c] "v"(c));
        return result;
    #endif
    }

    CK_TILE_DEVICE fp32x2_t pk_mul_f32(fp32x2_t lhs, fp32x2_t rhs)
    {
        fp32x2_t result;
        asm volatile("v_pk_mul_f32 %[result], %[lhs], %[rhs]"
                     : [result] "=v"(result)
                     : [lhs] "v"(lhs), [rhs] "v"(rhs));
        return result;
    }

    CK_TILE_DEVICE fp16x2_t cvt_pk_fp16_f32(float a, float b)
    {
        fp16x2_t result;
        asm volatile("v_cvt_pk_f16_f32 %[result], %[a], %[b]"
                     : [result] "=v"(result)
                     : [a] "v"(a), [b] "v"(b));
        return result;
    }
    } // namespace detail
    ```

- Evidence mapping:
  - "Packed FP32 operations" → `v_pk_mul_f32` instruction
  - "Fused multiply-add" → `v_fma_f32` with scalar operand `[b] "s"(b)`
  - "Packed conversion" → `v_cvt_pk_f16_f32` and `v_cvt_pk_bf16_f32`

## Optimization 3: Fine-tuned Scheduling with SALU Interleaving
- Commit ID: 7fbc9d6c9
- Optimization type: scheduling
- Summary: Added SALU (scalar ALU) instruction scheduling to previously empty phases, preventing wave stalls and improving overall throughput.

- Detailed explanation:
  The optimization fills in previously empty scheduling phases with VALU and SALU barriers. This ensures that scalar operations (address calculations, loop counters) are properly interleaved with vector operations, preventing the scalar unit from becoming a bottleneck.

- Code excerpt:
    ```cpp
    // Before optimization: empty phases
    else if constexpr(Phase == 1) {}
    else if constexpr(Phase == 3) {}
    
    // After optimization: VALU + SALU scheduling
    else if constexpr(Phase == 1)
    {
        __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
        __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
    }
    else if constexpr(Phase == 3)
    {
        __builtin_amdgcn_sched_group_barrier(0x002, 2, 0); // VALU
        __builtin_amdgcn_sched_group_barrier(0x004, 4, 0); // SALU
    }
    ```

- Evidence mapping:
  - "SALU scheduling" → `0x004` mask for SALU instructions
  - "Previously empty phases" → Phases 1 and 3 that were empty `{}`
  - "Interleaving" → Both VALU (0x002) and SALU (0x004) in same phase

## Optimization 4: Dynamic Memory Access Count
- Commit ID: 7fbc9d6c9
- Optimization type: memory
- Summary: Changed from hardcoded memory access counts to dynamic calculation based on tile window properties.

- Detailed explanation:
  Instead of hardcoding the number of memory load instructions (which was incorrect for some tile sizes), the optimization now queries the tile window for the actual number of accesses needed. This ensures correct scheduling regardless of tile configuration.

- Code excerpt:
    ```cpp
    // Before: hardcoded values
    static constexpr int K_mem_su_ld_insts = 1;
    static constexpr int V_mem_su_ld_insts = 1;
    
    // After: dynamic calculation
    constexpr int K_mem_su_ld_insts = k_dram_window.get_num_of_access();
    constexpr int V_mem_su_ld_insts = v_dram_window.get_num_of_access();
    ```

- Evidence mapping:
  - "Dynamic calculation" → `get_num_of_access()` method call
  - "Tile window properties" → `k_dram_window` and `v_dram_window` objects

## Optimization 5: O_acc Rescaling Distribution
- Commit ID: 7fbc9d6c9
- Optimization type: compute
- Summary: Increased the number of output accumulator rescaling instructions moved to earlier phases to avoid SIMD idle cycles.

- Detailed explanation:
  The optimization increases `fmha_alu_D_reg_cnt` from 0 to 6, which controls how many output accumulator rescaling instructions are moved to the `fmha_alu1()` phase. This better distributes the rescaling work across phases, avoiding SIMD idle cycles at the end of the loop.

- Code excerpt:
    ```cpp
    // Before: no rescaling moved
    constexpr index_t fmha_alu_D_reg_cnt = 0;
    
    // After: move 6 rescaling instructions
    constexpr index_t fmha_alu_D_reg_cnt = 6; // threshold to decide how many fmha_alu_D_upd()
                                              // instructions should we move to fmha_alu1()
    static_assert(fmha_alu_D_reg_cnt <= o_acc.thread_buf_.size());
    ```

- Evidence mapping:
  - "Rescaling distribution" → `fmha_alu_D_reg_cnt = 6` (changed from 0)
  - "Avoid SIMD idle" → Comment explaining purpose of the threshold
