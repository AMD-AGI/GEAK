# Kernel: gemm_quant (Quantized GEMM with FP4 Support)

## Variant Context
- Input semantic type: Matrix multiplication with ultra-low precision quantization
- Datatype(s): FP4 (E2M1 format) with block-scale quantization
- Data representation: Packed FP4 (2 values per byte) with per-block scale factors
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
The FP4 quantized GEMM kernel computes C = dequant(A) × dequant(B) where A and B are stored in 4-bit floating point format (E2M1: 2-bit exponent, 1-bit mantissa). This provides 4x memory bandwidth reduction compared to FP16 while maintaining reasonable accuracy through block-scale quantization. The kernel supports both OCP FP8 and FNUZ FP8 intermediate representations.

## Optimization 1: FP4 to FP8 Look-up Table Conversion
- Commit ID: 6a6177a24
- Optimization type: compute / memory
- Summary: Implemented efficient FP4 to FP8 conversion using precomputed look-up tables, avoiding expensive runtime conversion calculations.

- Detailed explanation:
  FP4 (E2M1) has only 16 possible values, making look-up table conversion highly efficient. The optimization provides separate tables for OCP FP8 (E4M3) and FNUZ FP8 formats, enabling direct conversion without runtime computation.

  The E2M1 format represents values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6

- Code excerpt:
    ```cpp
    struct pk_float4_e2m1_t
    {
        // FP4 to FP8 conversion functions
        CK_TILE_HOST_DEVICE constexpr fp8_t to_fp8(float scale = 1.f) const;
        CK_TILE_HOST_DEVICE constexpr fp8x2_t to_fp8x2(float scale = 1.f) const;

    #if CK_TILE_USE_OCP_FP8
        // FP8 E4M3 (OCP) representation look-up table
        static constexpr fp8_t e2m1_to_fp8_table[16] = {
            fp8_t(static_cast<uint8_t>(0x00)), //  0
            fp8_t(static_cast<uint8_t>(0x30)), //  0.5
            fp8_t(static_cast<uint8_t>(0x38)), //  1
            fp8_t(static_cast<uint8_t>(0x3C)), //  1.5
            fp8_t(static_cast<uint8_t>(0x40)), //  2
            fp8_t(static_cast<uint8_t>(0x44)), //  3
            fp8_t(static_cast<uint8_t>(0x48)), //  4
            fp8_t(static_cast<uint8_t>(0x4C)), //  6
            fp8_t(static_cast<uint8_t>(0x00)), // -0
            fp8_t(static_cast<uint8_t>(0xB0)), // -0.5
            fp8_t(static_cast<uint8_t>(0xB8)), // -1
            fp8_t(static_cast<uint8_t>(0xBC)), // -1.5
            fp8_t(static_cast<uint8_t>(0xC0)), // -2
            fp8_t(static_cast<uint8_t>(0xC4)), // -3
            fp8_t(static_cast<uint8_t>(0xC8)), // -4
            fp8_t(static_cast<uint8_t>(0xCC))  // -6
        };
    #else // CK_TILE_USE_FNUZ_FP8
        // FP8 E4M3 FNUZ look-up table
        static constexpr fp8_t e2m1_to_fp8_table[16] = {
            fp8_t(static_cast<uint8_t>(0x00)), //  0
            fp8_t(static_cast<uint8_t>(0x38)), //  0.5
            // ... (different encoding for FNUZ)
        };
    #endif
    };
    ```

- Evidence mapping:
  - "Look-up table conversion" → `e2m1_to_fp8_table[16]` static array
  - "16 possible values" → Array size of 16 covering all E2M1 values
  - "OCP vs FNUZ" → `#if CK_TILE_USE_OCP_FP8` conditional compilation

## Optimization 2: Packed FP4 Decoding During Load
- Commit ID: 6a6177a24
- Optimization type: memory / compute
- Summary: Implemented decode-while-loading for packed FP4 data, unpacking two FP4 values per byte during the memory load operation.

- Detailed explanation:
  FP4 values are packed 2 per byte. The optimization decodes these packed values during the load operation, converting them to the compute datatype (FP8 or FP16) as they are loaded from memory. This overlaps memory access with compute and avoids separate unpacking passes.

- Code excerpt:
    ```cpp
    // Packed FP4 type: 2 values per byte
    struct pk_float4_e2m1_t
    {
        uint8_t data;  // Contains 2 FP4 values
        
        // Unpack to get individual values
        template <index_t I>
        CK_TILE_HOST_DEVICE constexpr pk_float4_e2m1_t unpack(number<I>) const
        {
            static_assert(I < 2, "FP4 pack contains only 2 values");
            pk_float4_e2m1_t result;
            if constexpr(I == 0)
                result.data = data & 0x0F;  // Lower nibble
            else
                result.data = (data >> 4) & 0x0F;  // Upper nibble
            return result;
        }
        
        // Convert packed FP4 to FP8x2 (both values at once)
        CK_TILE_HOST_DEVICE constexpr fp8x2_t to_fp8x2(float scale = 1.f) const
        {
            fp8_t lo = e2m1_to_fp8_table[data & 0x0F];
            fp8_t hi = e2m1_to_fp8_table[(data >> 4) & 0x0F];
            return fp8x2_t{lo, hi};
        }
    };
    
    // In load_interleaved_pk_type.hpp
    template <typename DstType, typename SrcType>
    CK_TILE_DEVICE auto load_and_decode_fp4(const SrcType* ptr)
    {
        // Load packed FP4 and decode to compute type in one operation
        pk_float4_e2m1_t packed = *reinterpret_cast<const pk_float4_e2m1_t*>(ptr);
        return packed.to_fp8x2();  // Returns 2 FP8 values
    }
    ```

- Evidence mapping:
  - "2 values per byte" → `uint8_t data` containing 2 FP4 values
  - "Decode during load" → `load_and_decode_fp4` function
  - "Nibble extraction" → `data & 0x0F` and `(data >> 4) & 0x0F`

## Optimization 3: Mixed Precision Compute Type Utilities
- Commit ID: 6a6177a24
- Optimization type: compute
- Summary: Added type utilities for determining appropriate MFMA compute types based on input datatypes, enabling optimal instruction selection.

- Detailed explanation:
  Different input datatypes require different MFMA instruction variants. The optimization adds compile-time utilities to determine the correct compute type (FP32, FP16, etc.) based on the input types, ensuring the most efficient MFMA instruction is selected.

- Code excerpt:
    ```cpp
    // mixed_prec_compute_type.hpp
    
    // Determine MFMA compute type based on input types
    template <typename AType, typename BType>
    struct MfmaComputeType
    {
        // Default: use FP32 accumulator
        using type = float;
    };
    
    // Specialization for FP4 inputs: use FP32 accumulator
    template <>
    struct MfmaComputeType<pk_float4_e2m1_t, pk_float4_e2m1_t>
    {
        using type = float;
    };
    
    // Specialization for FP8 inputs
    template <>
    struct MfmaComputeType<fp8_t, fp8_t>
    {
        using type = float;
    };
    
    // Helper alias
    template <typename AType, typename BType>
    using mfma_compute_type_t = typename MfmaComputeType<AType, BType>::type;
    ```

- Evidence mapping:
  - "Type utilities" → `MfmaComputeType` template struct
  - "Compile-time determination" → Template specializations
  - "MFMA instruction selection" → Different compute types for different input types

## Optimization 4: FP4 Weight Preshuffle Support
- Commit ID: 6a6177a24
- Optimization type: memory
- Summary: Added preshuffle support for FP4 weights, reorganizing weight data layout for optimal memory access patterns during GEMM.

- Detailed explanation:
  Weight preshuffling reorganizes the weight tensor layout to match the access pattern of the GEMM kernel, improving memory coalescing and reducing bank conflicts. For FP4, this is particularly important due to the packed format.

- Code excerpt:
    ```cpp
    // In gemm_wp_abquant_pipeline_ag_bg_cr_v2.hpp
    
    // Preshuffle mode for packed FP4 weights
    template <typename Problem>
    struct WeightPreshufflePolicy
    {
        // For FP4: preshuffle to match MFMA access pattern
        static constexpr bool kEnablePreshuffle = 
            is_packed_fp4_v<typename Problem::BDataType>;
        
        // Preshuffle layout transformation
        static constexpr auto GetPreshuffleLayout()
        {
            // Reorganize packed FP4 for coalesced access
            // Original: [N, K/2] (K/2 because 2 FP4 per byte)
            // Preshuffled: [N/VecN, K/2/VecK, VecN, VecK]
            return make_tuple(
                number<Problem::NPerBlock / VectorN>{},
                number<Problem::KPerBlock / 2 / VectorK>{},
                number<VectorN>{},
                number<VectorK>{}
            );
        }
    };
    ```

- Evidence mapping:
  - "Preshuffle support" → `kEnablePreshuffle` flag for FP4
  - "Layout reorganization" → `GetPreshuffleLayout()` transformation
  - "Packed format handling" → `K/2` accounting for 2 FP4 per byte
