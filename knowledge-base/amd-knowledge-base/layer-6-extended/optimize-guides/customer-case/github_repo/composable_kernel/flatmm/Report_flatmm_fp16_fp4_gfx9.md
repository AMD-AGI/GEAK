Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: FlatMM (Flat Matrix Multiply)

## Variant Context
- Input semantic type: Matrix multiplication with flattened/packed formats
- Datatype(s): FP16/BF16/FP8/FP4 (MXFP4)
- Data representation: Microscaling (MX) formats, packed low-precision
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
FlatMM kernels implement matrix multiplication with specialized support for microscaling (MX) formats like MXFP4 and MXFP8. These formats use block-wise scaling with very low precision data, enabling extreme memory bandwidth reduction while maintaining acceptable accuracy for inference.

## Optimization 1: FP8 × FP4 Mixed Precision FlatMM
- Commit ID: 57e1e4a84
- Optimization type: compute / memory
- Summary: Added FP8×FP4 mixed precision FlatMM for efficient inference with different precision for activations and weights.

- Detailed explanation:
  This optimization enables using FP8 for activations (higher precision for dynamic values) and FP4 for weights (lower precision acceptable for static values). This provides:
  - 2x memory reduction for weights vs FP8
  - Better accuracy than pure FP4
  - Efficient MFMA utilization

- Code excerpt:
    ```cpp
    // FP8 × FP4 FlatMM configuration
    template <typename Problem>
    struct FlatmmFP8xFP4Pipeline
    {
        using ADataType = fp8_t;   // Activations in FP8
        using BDataType = fp4_t;   // Weights in FP4
        using AccType = float;     // FP32 accumulator
        
        // Decode FP4 weights to FP8 for MFMA
        CK_TILE_DEVICE auto decode_weights(const fp4_t* weights, index_t count)
        {
            // Use LUT for FP4 to FP8 conversion
            fp8_t decoded[count];
            for(index_t i = 0; i < count; ++i)
            {
                decoded[i] = fp4_to_fp8_lut[weights[i].data];
            }
            return decoded;
        }
        
        // Mixed precision GEMM
        CK_TILE_DEVICE void gemm_fp8xfp4(
            const fp8_t* a_tile,
            const fp4_t* b_tile,
            AccType* c_tile)
        {
            // Decode B to FP8
            auto b_decoded = decode_weights(b_tile, kKPerBlock);
            
            // Use FP8 MFMA
            mfma_fp8(a_tile, b_decoded, c_tile);
        }
    };
    ```

- Evidence mapping:
  - "FP8 activations" → `ADataType = fp8_t`
  - "FP4 weights" → `BDataType = fp4_t`
  - "LUT conversion" → `fp4_to_fp8_lut` lookup

## Optimization 2: Eliminate Runtime Division for MXFP4
- Commit ID: 878b4e7f4
- Optimization type: compute
- Summary: Optimized MXFP4 FlatMM by eliminating runtime division by 2 operations.

- Detailed explanation:
  MXFP4 packs 2 values per byte, requiring division by 2 for index calculations. The optimization replaces runtime divisions with compile-time shifts and precomputed offsets.

- Code excerpt:
    ```cpp
    // Before: runtime division
    index_t byte_offset = element_idx / 2;
    index_t nibble_idx = element_idx % 2;
    
    // After: compile-time optimization
    template <index_t ElementsPerByte = 2>
    struct MXFP4IndexOptimization
    {
        static_assert(ElementsPerByte == 2, "MXFP4 packs 2 elements per byte");
        
        // Use bit shift instead of division
        CK_TILE_DEVICE static index_t get_byte_offset(index_t element_idx)
        {
            return element_idx >> 1;  // Equivalent to / 2
        }
        
        CK_TILE_DEVICE static index_t get_nibble_idx(index_t element_idx)
        {
            return element_idx & 1;   // Equivalent to % 2
        }
    };
    ```

- Evidence mapping:
  - "Eliminate division" → `>> 1` instead of `/ 2`
  - "Compile-time" → Template-based optimization
  - "Bit operations" → `& 1` instead of `% 2`

## Optimization 3: Byte Pointer Arithmetic for A Tensor
- Commit ID: 2220cbaba
- Optimization type: memory
- Summary: Use byte pointer arithmetic for A tensor access to handle packed formats correctly.

- Detailed explanation:
  For packed formats like MXFP4, pointer arithmetic must account for sub-byte element sizes. Using byte pointers with explicit offset calculations ensures correct memory access.

- Code excerpt:
    ```cpp
    // Byte pointer arithmetic for packed formats
    template <typename PackedType>
    struct BytePointerAccess
    {
        static constexpr index_t kBitsPerElement = sizeof_bits<PackedType>::value;
        static constexpr index_t kElementsPerByte = 8 / kBitsPerElement;
        
        CK_TILE_DEVICE auto load_elements(
            const uint8_t* base_ptr,
            index_t element_offset,
            index_t count)
        {
            // Convert element offset to byte offset
            index_t byte_offset = element_offset / kElementsPerByte;
            index_t sub_byte_offset = element_offset % kElementsPerByte;
            
            // Load bytes and unpack
            const uint8_t* ptr = base_ptr + byte_offset;
            return unpack_elements<PackedType>(ptr, sub_byte_offset, count);
        }
    };
    ```

- Evidence mapping:
  - "Byte pointer" → `const uint8_t* base_ptr`
  - "Sub-byte offset" → `sub_byte_offset` calculation
  - "Packed format" → `kElementsPerByte` for MXFP4

## Optimization 4: Split-K for A16W4 GEMM
- Commit ID: dae85ead6
- Optimization type: compute / scheduling
- Summary: Added Split-K support for A16W4 (FP16 activations, INT4 weights) GEMM.

- Detailed explanation:
  Split-K parallelizes the K dimension reduction across multiple workgroups. For A16W4, this is particularly beneficial when the K dimension is large relative to M and N.

- Code excerpt:
    ```cpp
    // Split-K for A16W4 GEMM
    template <typename Problem>
    struct A16W4SplitKPipeline
    {
        using ADataType = fp16_t;  // 16-bit activations
        using BDataType = int4_t;  // 4-bit weights
        
        static constexpr index_t kSplitK = Problem::kSplitK;
        
        CK_TILE_DEVICE void operator()(
            const ADataType* a,
            const BDataType* b,
            AccType* partial_c,
            index_t split_idx)
        {
            // Each split handles a portion of K
            index_t k_start = split_idx * kKPerSplit;
            index_t k_end = min(k_start + kKPerSplit, K);
            
            // Compute partial GEMM
            for(index_t k = k_start; k < k_end; k += kKPerBlock)
            {
                auto a_tile = load_a(a, k);
                auto b_tile = load_and_dequant_b(b, k);
                gemm_accumulate(a_tile, b_tile, c_acc);
            }
            
            // Store partial result
            store_partial(partial_c, split_idx, c_acc);
        }
    };
    ```

- Evidence mapping:
  - "Split-K" → `kSplitK` and `split_idx` parameters
  - "A16W4" → `fp16_t` activations, `int4_t` weights
  - "Partial results" → `store_partial` for later reduction

## Optimization 5: M Padding Fix for MX FlatMM
- Commit ID: b0ea67e37
- Optimization type: correctness
- Summary: Fixed M dimension padding for MX FlatMM to handle non-aligned problem sizes.

- Detailed explanation:
  When M dimension is not aligned to tile size, padding is needed. The fix ensures correct handling of partial tiles at the M boundary.

- Code excerpt:
    ```cpp
    // M padding for MX FlatMM
    template <typename Problem>
    struct MXFlatmmMPadding
    {
        CK_TILE_DEVICE void load_a_with_padding(
            const ADataType* a,
            ATile& a_tile,
            index_t m_offset,
            index_t M)
        {
            // Check if this tile extends beyond M
            bool needs_padding = (m_offset + kMPerBlock) > M;
            
            if(needs_padding)
            {
                // Load valid elements, zero-pad rest
                index_t valid_m = M - m_offset;
                load_partial_tile(a, a_tile, valid_m);
                zero_pad_tile(a_tile, valid_m, kMPerBlock);
            }
            else
            {
                // Full tile load
                load_tile(a, a_tile);
            }
        }
    };
    ```

- Evidence mapping:
  - "M padding" → `needs_padding` check
  - "Partial tile" → `load_partial_tile` for boundary
  - "Zero padding" → `zero_pad_tile` for invalid elements
