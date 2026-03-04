---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: Permute and Batched Transpose

## Variant Context
- Input semantic type: Tensor dimension reordering and transposition
- Datatype(s): FP16/BF16/FP32/FP8
- Data representation: Multi-dimensional tensors
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
Permute and Transpose kernels reorder tensor dimensions for layout transformations required by various neural network operations. These are essential for:
- Converting between NHWC and NCHW layouts
- Transposing attention matrices (Q, K, V)
- Reshaping tensors for batched operations

## Optimization 1: Vectorized Transpose for Batched Operations
- Commit ID: 9d1e44e56
- Optimization type: memory
- Summary: Implemented vectorized transpose for batched transpose operations, improving memory throughput.

- Detailed explanation:
  The vectorized transpose uses vector load/store instructions to move multiple elements at once, maximizing memory bandwidth utilization. For batched transpose, each batch is processed independently, enabling parallelism across batches.

- Code excerpt:
    ```cpp
    // Vectorized batched transpose
    template <typename DataType, index_t VectorSize>
    struct BatchedTransposeVectorized
    {
        static constexpr index_t kVectorSize = VectorSize;
        using VectorType = vector_type_t<DataType, kVectorSize>;
        
        CK_TILE_DEVICE void operator()(
            const DataType* input,
            DataType* output,
            index_t batch_idx,
            index_t M, index_t N,
            index_t input_batch_stride,
            index_t output_batch_stride)
        {
            // Offset to current batch
            const DataType* batch_input = input + batch_idx * input_batch_stride;
            DataType* batch_output = output + batch_idx * output_batch_stride;
            
            // Vectorized load from input (row-major)
            const index_t row = blockIdx.y * blockDim.y + threadIdx.y;
            const index_t col_base = (blockIdx.x * blockDim.x + threadIdx.x) * kVectorSize;
            
            if(row < M && col_base + kVectorSize <= N)
            {
                // Vector load
                VectorType vec = *reinterpret_cast<const VectorType*>(
                    &batch_input[row * N + col_base]);
                
                // Transpose and store (col-major output)
                for(index_t i = 0; i < kVectorSize; ++i)
                {
                    output[(col_base + i) * M + row] = vec[i];
                }
            }
        }
    };
    ```

- Evidence mapping:
  - "Vectorized" → `VectorType` for multi-element load
  - "Batched" → `batch_idx` and batch stride handling
  - "Memory throughput" → Vector load with scalar transpose stores

## Optimization 2: Rectangular Block Tile Sizes
- Commit ID: ab2602683
- Optimization type: compute / memory
- Summary: Added support for rectangular block tile sizes in batched transpose for better performance on non-square matrices.

- Detailed explanation:
  Different matrix shapes benefit from different tile configurations. Rectangular tiles (e.g., 32x64 instead of 32x32) can better match the aspect ratio of the input matrix, improving cache utilization.

- Code excerpt:
    ```cpp
    // Rectangular tile configuration
    template <index_t TileM, index_t TileN>
    struct TransposeTileConfig
    {
        static constexpr index_t kTileM = TileM;
        static constexpr index_t kTileN = TileN;
        
        // Shared memory for tile transpose
        static constexpr index_t kSmemSize = kTileM * kTileN * sizeof(DataType);
        
        CK_TILE_DEVICE void transpose_tile(
            const DataType* input,
            DataType* output,
            DataType* smem,
            index_t m_offset, index_t n_offset,
            index_t M, index_t N)
        {
            // Load tile to shared memory (coalesced)
            for(index_t i = threadIdx.y; i < kTileM; i += blockDim.y)
            {
                for(index_t j = threadIdx.x; j < kTileN; j += blockDim.x)
                {
                    index_t m = m_offset + i;
                    index_t n = n_offset + j;
                    if(m < M && n < N)
                    {
                        smem[i * kTileN + j] = input[m * N + n];
                    }
                }
            }
            
            __syncthreads();
            
            // Store transposed (coalesced output)
            for(index_t j = threadIdx.y; j < kTileN; j += blockDim.y)
            {
                for(index_t i = threadIdx.x; i < kTileM; i += blockDim.x)
                {
                    index_t m = m_offset + i;
                    index_t n = n_offset + j;
                    if(m < M && n < N)
                    {
                        output[n * M + m] = smem[i * kTileN + j];
                    }
                }
            }
        }
    };
    ```

- Evidence mapping:
  - "Rectangular tiles" → Separate `kTileM` and `kTileN` parameters
  - "Shared memory" → `smem` for tile staging
  - "Coalesced access" → Loop ordering for memory efficiency

## Optimization 3: Wave32 Support
- Commit ID: 9fcc1ee9f
- Optimization type: compute
- Summary: Added Wave32 support for RDNA architectures in transpose operations.

- Detailed explanation:
  RDNA GPUs (gfx11, gfx12) use Wave32 execution model instead of Wave64. The optimization adapts the transpose kernel to work efficiently with both wave sizes.

- Code excerpt:
    ```cpp
    // Wave-size agnostic transpose
    template <typename Problem>
    struct TransposeWaveAgnostic
    {
        static constexpr index_t kWaveSize = get_warp_size();  // 32 or 64
        
        CK_TILE_DEVICE void warp_transpose(
            DataType* data,
            index_t lane_id)
        {
            // Adjust shuffle pattern based on wave size
            if constexpr(kWaveSize == 32)
            {
                // Wave32: use 32-lane shuffle
                for(index_t i = 1; i < 32; i *= 2)
                {
                    DataType other = __shfl_xor(data[0], i);
                    // ... transpose logic
                }
            }
            else
            {
                // Wave64: use 64-lane shuffle
                for(index_t i = 1; i < 64; i *= 2)
                {
                    DataType other = __shfl_xor(data[0], i);
                    // ... transpose logic
                }
            }
        }
    };
    ```

- Evidence mapping:
  - "Wave32 support" → `kWaveSize == 32` branch
  - "Wave-size agnostic" → `get_warp_size()` runtime detection
  - "Shuffle patterns" → Different loop bounds for 32 vs 64

## Optimization 4: Permute with Arbitrary Dimension Order
- Commit ID: (permute_kernel.hpp)
- Optimization type: compute
- Summary: General permute kernel supporting arbitrary dimension reordering.

- Detailed explanation:
  The permute kernel supports any permutation of tensor dimensions, not just simple transpose. It computes output indices from input indices using the permutation mapping.

- Code excerpt:
    ```cpp
    // General permute kernel
    template <index_t NumDims, typename PermutationType>
    struct PermuteKernel
    {
        CK_TILE_DEVICE void operator()(
            const DataType* input,
            DataType* output,
            const index_t* input_dims,
            const index_t* input_strides,
            const index_t* output_strides,
            const PermutationType& perm)
        {
            const index_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            // Convert linear index to multi-dimensional input index
            index_t input_indices[NumDims];
            index_t remaining = linear_idx;
            for(index_t d = 0; d < NumDims; ++d)
            {
                input_indices[d] = remaining / input_strides[d];
                remaining = remaining % input_strides[d];
            }
            
            // Apply permutation to get output indices
            index_t output_indices[NumDims];
            for(index_t d = 0; d < NumDims; ++d)
            {
                output_indices[d] = input_indices[perm[d]];
            }
            
            // Compute output linear index
            index_t output_linear_idx = 0;
            for(index_t d = 0; d < NumDims; ++d)
            {
                output_linear_idx += output_indices[d] * output_strides[d];
            }
            
            // Copy element
            output[output_linear_idx] = input[linear_idx];
        }
    };
    ```

- Evidence mapping:
  - "Arbitrary permutation" → `perm[d]` mapping
  - "Multi-dimensional" → `NumDims` template parameter
  - "Index computation" → Stride-based linear to multi-dim conversion
