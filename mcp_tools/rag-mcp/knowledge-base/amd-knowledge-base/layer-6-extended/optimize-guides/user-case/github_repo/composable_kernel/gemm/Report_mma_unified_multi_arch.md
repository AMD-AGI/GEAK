# Kernel: MMA Unified (MFMA/WMMA Unification)

## Variant Context
- Input semantic type: Matrix multiply-accumulate operations
- Datatype(s): FP16/BF16/FP32/FP8/INT8
- Data representation: Wave-level matrix fragments
- Target architecture: Multi-architecture (gfx9 MFMA, gfx11/gfx12 WMMA)

## Functionality
The MMA (Matrix Multiply-Accumulate) unified abstraction provides a common interface for wave-level matrix operations across different AMD GPU architectures. It abstracts the differences between MFMA (Matrix Fused Multiply-Add) on CDNA/gfx9 architectures and WMMA (Wave Matrix Multiply-Accumulate) on RDNA3/gfx11 and RDNA4/gfx12 architectures, enabling portable high-performance code.

## Optimization 1: Architecture-Agnostic MMA Selector
- Commit ID: b9c6cb145
- Optimization type: compute / code portability
- Summary: Implemented a unified MMA selector that automatically chooses the optimal matrix instruction based on target architecture, datatypes, and fragment dimensions.

- Detailed explanation:
  The `MmaDefaultSelector` template automatically selects the best available MMA instruction for the given configuration:
  - For gfx9 (CDNA): Selects appropriate MFMA instruction (16x16x16, 32x32x8, etc.)
  - For gfx11 (RDNA3): Selects WMMA instruction with wave32
  - For gfx12 (RDNA4): Selects WMMA instruction with updated encoding

  The selector prioritizes instructions with larger K dimensions for better compute efficiency.

- Code excerpt:
    ```cpp
    /**
     * @class MmaDefaultSelector
     * @brief Implements a default mma selector strategy for the current target architecture.
     * Given the particular datatypes and Fragment dimensions, the selector will attempt to
     * select the instruction with the largest K dimension that is supported on the current target
     * architecture.
     */
    template <typename ADataType,
              typename BDataType,
              typename CDataType,
              uint32_t FragM,
              uint32_t FragN,
              uint32_t FragK,
              typename CompilerTarget,
              typename Enable = void>
    struct MmaDefaultSelector
    {
        // By default, no selection is made, and we fall back to a pass-through unsupported
        // implementation.
        using SelectedOp =
            amdgcn_mma<ADataType, BDataType, CDataType, FragM, FragN, FragK, void, amdgcn_target<>>;
    };

    #if defined(__cpp_concepts) && __cpp_concepts >= 201907L
    /**
     *  @concept MmaSelectorI
     *  @brief  Expresses the required members for each MmaSelector class.
     */
    template <typename MmaSelector>
    concept MmaSelectorI = requires(MmaSelector op) {
        typename MmaSelector::SelectedOp;
    };
    #endif
    ```

- Evidence mapping:
  - "Architecture-agnostic" → Template with `CompilerTarget` parameter
  - "Automatic selection" → `MmaDefaultSelector` with SFINAE-based specializations
  - "Largest K dimension" → Comment explaining selection strategy

## Optimization 2: Unified amdgcn_mma Interface
- Commit ID: b9c6cb145
- Optimization type: compute / code portability
- Summary: Created a unified `amdgcn_mma` template that provides a common interface for both MFMA and WMMA instructions.

- Detailed explanation:
  The `amdgcn_mma` template provides a unified interface with:
  - Common type aliases (AVecType, BVecType, CVecType)
  - Common layout constants (kAMBlock, kBNBlock, kAMLane, etc.)
  - Common `exec()` method for executing the MMA operation

  Architecture-specific implementations are provided through template specializations.

- Code excerpt:
    ```cpp
    // amdgcn_mma.hpp
    
    /**
     * @struct amdgcn_mma
     * @brief Base template for AMD GCN matrix multiply-accumulate operations.
     * Specializations provide architecture-specific implementations.
     */
    template <typename ADataType,
              typename BDataType,
              typename CDataType,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename CtrlFlags,
              typename CompilerTarget,
              typename Enable = void>
    struct amdgcn_mma
    {
        // Default: unsupported configuration
        using OpType = void;
        
        // Register types (to be specialized)
        using AVecType = void;
        using BVecType = void;
        using CVecType = void;
        
        // Layout constants (to be specialized)
        static constexpr index_t kAMBlock = 0;
        static constexpr index_t kBNBlock = 0;
        // ...
        
        // Execution method (to be specialized)
        CK_TILE_DEVICE static auto
        exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType;
    };
    ```

- Evidence mapping:
  - "Unified interface" → Common template structure for all architectures
  - "Common type aliases" → `AVecType`, `BVecType`, `CVecType`
  - "Common layout constants" → `kAMBlock`, `kBNBlock`, etc.

## Optimization 3: MFMA Specialization for GFX9
- Commit ID: b9c6cb145
- Optimization type: compute
- Summary: Provided optimized MFMA specializations for gfx9 architectures with support for various tile sizes and datatypes.

- Detailed explanation:
  The MFMA specializations for gfx9 (gfx908, gfx90a, gfx942, gfx950) provide:
  - FP16 16x16x16 and 32x32x8 instructions
  - BF16 16x16x16 and 32x32x8 instructions
  - FP8 support on gfx950
  - Control flags for broadcasting and rotation (Cbsz, Abid, Blgp)

- Code excerpt:
    ```cpp
    // mfma_gfx9.hpp
    
    /**
     * @struct amdgcn_mma
     * @brief Specialization for MFMA on GFX9 targets (fp16 16x16x16)
     */
    template <typename CtrlFlags, typename CompilerTarget>
    struct amdgcn_mma<fp16_t,
                      fp16_t,
                      fp32_t,
                      16u,
                      16u,
                      16u,
                      CtrlFlags,
                      CompilerTarget,
                      enable_if_target_family_gfx9_t<CompilerTarget>>
    {
        using OpType = MfmaOp;

        // Register types for 16x16x16 MFMA
        using AVecType = ext_vector_t<fp16_t, 4>;
        using BVecType = ext_vector_t<fp16_t, 4>;
        using CVecType = ext_vector_t<fp32_t, 4>;

        // Layout constants
        static constexpr index_t kAMLane     = 16;
        static constexpr index_t kBNLane     = 16;
        static constexpr index_t kABKLane    = 4;
        static constexpr index_t kABKPerLane = 4;

        CK_TILE_DEVICE static auto
        exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
        {
            return {__builtin_amdgcn_mfma_f32_16x16x16f16(aVec,
                                                          bVec,
                                                          cVec,
                                                          static_cast<int>(CtrlFlags::Cbsz),
                                                          static_cast<int>(CtrlFlags::Abid),
                                                          static_cast<int>(CtrlFlags::Blgp))};
        }
    };
    ```

- Evidence mapping:
  - "GFX9 specialization" → `enable_if_target_family_gfx9_t<CompilerTarget>`
  - "16x16x16 tile" → Template parameters `16u, 16u, 16u`
  - "MFMA intrinsic" → `__builtin_amdgcn_mfma_f32_16x16x16f16`

## Optimization 4: WMMA Specialization for GFX11/GFX12
- Commit ID: b9c6cb145
- Optimization type: compute
- Summary: Provided WMMA specializations for RDNA3 (gfx11) and RDNA4 (gfx12) architectures with wave32 support.

- Detailed explanation:
  The WMMA specializations provide:
  - Wave32 execution model (vs wave64 for MFMA)
  - 16x16x16 tile sizes for FP16/BF16
  - Different register layouts optimized for RDNA architecture
  - GFX12-specific instruction encodings

- Code excerpt:
    ```cpp
    // wmma_gfx11.hpp
    
    /**
     * @struct amdgcn_mma
     * @brief Specialization for WMMA on GFX11 targets (fp16 16x16x16)
     */
    template <typename CtrlFlags, typename CompilerTarget>
    struct amdgcn_mma<fp16_t,
                      fp16_t,
                      fp32_t,
                      16u,
                      16u,
                      16u,
                      CtrlFlags,
                      CompilerTarget,
                      enable_if_target_family_gfx11_t<CompilerTarget>>
    {
        using OpType = WmmaOp;

        // Register types for WMMA (wave32)
        using AVecType = ext_vector_t<fp16_t, 16>;
        using BVecType = ext_vector_t<fp16_t, 16>;
        using CVecType = ext_vector_t<fp32_t, 8>;

        // Layout constants for wave32
        static constexpr index_t kAMLane     = 16;
        static constexpr index_t kBNLane     = 16;
        static constexpr index_t kABKLane    = 16;
        static constexpr index_t kABKPerLane = 1;

        CK_TILE_DEVICE static auto
        exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
        {
            return {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(aVec, bVec, cVec)};
        }
    };
    ```

- Evidence mapping:
  - "GFX11 specialization" → `enable_if_target_family_gfx11_t<CompilerTarget>`
  - "Wave32" → `_w32` suffix in intrinsic and different vector sizes
  - "WMMA intrinsic" → `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`
