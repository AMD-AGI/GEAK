# Kernel: Buffer Addressing and Async Loading

## Variant Context
- Input semantic type: Memory operations for all kernel types
- Datatype(s): All (FP16/BF16/FP32/FP8/INT8)
- Data representation: Buffer-addressed global memory with LDS staging
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
The buffer addressing infrastructure provides optimized memory access primitives for AMD GPUs. It includes buffer load/store operations with various vector widths, async global-to-LDS transfers, and cache coherency controls. These primitives are fundamental to all high-performance kernels in CK.

## Optimization 1: Async Buffer Load with dwordx4 Support
- Commit ID: 4a7ecce09
- Optimization type: memory
- Summary: Extended async buffer load to support dwordx3 and dwordx4 operations on gfx950, enabling 3x-4x wider memory transfers per instruction.

- Detailed explanation:
  The original async buffer load only supported single dword (4 bytes) transfers. This optimization adds support for:
  - `buffer_load_dwordx3`: 12 bytes per instruction
  - `buffer_load_dwordx4`: 16 bytes per instruction (gfx950 only)

  Wider loads reduce instruction count and improve memory bandwidth utilization, particularly important for FMHA kernels with large tile sizes.

- Code excerpt:
    ```cpp
    template <unsigned num_dwords, bool pre_nop = false>
    CK_TILE_DEVICE void async_buffer_load_dwordxn_v(void* smem,
                                                    int32x4_t rsrc,
                                                    index_t voffset,
                                                    index_t /*soffset*/,
                                                    index_t ioffset /*max 0xFFF*/,
                                                    index_t /*flag*/       = 0,
                                                    bool_constant<pre_nop> = {})
    {
    #define CK_TILE_ASYNC_LOAD_WITH_INSTR(instr)                            \
        if constexpr(pre_nop)                                               \
            asm volatile("s_nop 4\n" instr " %1, %2, 0 offen offset:%3 lds" \
                         : "=r"(smem) /*dummy dependency for smem*/         \
                         : "v"(voffset), "s"(rsrc), "n"(ioffset)            \
                         : "memory");                                       \
        else                                                                \
            asm volatile(instr " %1, %2, 0 offen offset:%3 lds"             \
                         : "=r"(smem) /*dummy dependency for smem*/         \
                         : "v"(voffset), "s"(rsrc), "n"(ioffset)            \
                         : "memory");

        if constexpr(num_dwords == 1)
        {
            CK_TILE_ASYNC_LOAD_WITH_INSTR("buffer_load_dword");
        }
    #if defined(__gfx950__)
        else if constexpr(num_dwords == 3)
        {
            CK_TILE_ASYNC_LOAD_WITH_INSTR("buffer_load_dwordx3");
        }
        else if constexpr(num_dwords == 4)
        {
            CK_TILE_ASYNC_LOAD_WITH_INSTR("buffer_load_dwordx4");
        }
    #endif
    #undef CK_TILE_ASYNC_LOAD_WITH_INSTR
    }
    ```

- Evidence mapping:
  - "dwordx4 support" → `buffer_load_dwordx4` instruction
  - "gfx950 specific" → `#if defined(__gfx950__)` guard
  - "Wider transfers" → Template parameter `num_dwords` supporting 1, 3, 4

## Optimization 2: Flexible Vector Width Selection
- Commit ID: 4a7ecce09
- Optimization type: memory
- Summary: Added compile-time vector width selection based on data size, automatically choosing the widest available load instruction.

- Detailed explanation:
  The `amd_async_buffer_load_impl` function now automatically selects the appropriate load width based on the total data size:
  - 4 bytes → dword
  - 12 bytes → dwordx3
  - 16 bytes → dwordx4

  This enables optimal memory bandwidth utilization without manual instruction selection.

- Code excerpt:
    ```cpp
    template <typename T, index_t N, bool pre_nop = false>
    CK_TILE_DEVICE void amd_async_buffer_load_impl(CK_TILE_LDS_ADDR T* smem,
                                                   int32x4_t src_wave_buffer_resource,
                                                   index_t src_thread_addr_offset,
                                                   index_t src_wave_addr_offset,
                                                   index_t src_immediate_addr_offset = 0,
                                                   bool_constant<pre_nop>            = {})
    {
        constexpr index_t num_bytes = sizeof(T) * N;
        constexpr index_t num_words = num_bytes / 4;
        static_assert(num_bytes % 4 == 0 && (num_words == 1 || num_words == 3 || num_words == 4),
                      "wrong! only support in dword, dwordx3, dwordx4");

        async_buffer_load_dwordxn_v<num_words>(smem,
                                               src_wave_buffer_resource,
                                               src_thread_addr_offset,
                                               src_wave_addr_offset,
                                               src_immediate_addr_offset,
                                               0,
                                               bool_constant<pre_nop>{});
    }
    ```

- Evidence mapping:
  - "Compile-time selection" → `constexpr index_t num_words = num_bytes / 4`
  - "Automatic width" → Template deduction from `sizeof(T) * N`
  - "Validation" → `static_assert` ensuring valid configurations

## Optimization 3: Pre-NOP Insertion for Hazard Avoidance
- Commit ID: 4a7ecce09
- Optimization type: scheduling
- Summary: Added optional s_nop insertion before async loads to avoid hardware hazards on certain architectures.

- Detailed explanation:
  Some GPU architectures require NOPs before certain memory operations to avoid hardware hazards. The `pre_nop` template parameter enables inserting `s_nop 4` before the buffer load instruction when needed, ensuring correct execution without manual hazard management.

- Code excerpt:
    ```cpp
    if constexpr(pre_nop)
        asm volatile("s_nop 4\n" instr " %1, %2, 0 offen offset:%3 lds"
                     : "=r"(smem) /*dummy dependency for smem*/
                     : "v"(voffset), "s"(rsrc), "n"(ioffset)
                     : "memory");
    else
        asm volatile(instr " %1, %2, 0 offen offset:%3 lds"
                     : "=r"(smem) /*dummy dependency for smem*/
                     : "v"(voffset), "s"(rsrc), "n"(ioffset)
                     : "memory");
    ```

- Evidence mapping:
  - "Pre-NOP insertion" → `s_nop 4\n` prefix in assembly
  - "Optional" → `if constexpr(pre_nop)` compile-time branch
  - "Hazard avoidance" → Purpose of NOP before memory operation

## Optimization 4: LDS Direct Store with Dummy Dependency
- Commit ID: 4a7ecce09
- Optimization type: memory / correctness
- Summary: Uses dummy output dependency on smem pointer to ensure correct instruction ordering without explicit barriers.

- Detailed explanation:
  The async buffer load uses a dummy output dependency (`"=r"(smem)`) to inform the compiler that the LDS memory is modified. This ensures proper instruction ordering without requiring explicit memory barriers, which would be more expensive.

- Code excerpt:
    ```cpp
    asm volatile(instr " %1, %2, 0 offen offset:%3 lds"
                 : "=r"(smem) /*dummy dependency for smem*/
                 : "v"(voffset), "s"(rsrc), "n"(ioffset)
                 : "memory");
    ```

- Evidence mapping:
  - "Dummy dependency" → `"=r"(smem)` output constraint with comment
  - "Memory clobber" → `"memory"` clobber for ordering
  - "LDS target" → `lds` suffix in instruction
