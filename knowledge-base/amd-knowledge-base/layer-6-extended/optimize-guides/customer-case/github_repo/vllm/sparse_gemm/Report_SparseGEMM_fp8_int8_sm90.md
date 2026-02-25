# Kernel: Sparse GEMM (2:4 Sparsity)

## Variant Context
- Input semantic type: Matrix multiplication with structured sparsity
- Datatype(s): fp8, int8, fp16, bf16
- Data representation: 2:4 structured sparse weights
- Target architecture: CUDA sm90+ (Hopper and later)

## Functionality
The Sparse GEMM kernel leverages NVIDIA's 2:4 structured sparsity support in Tensor Cores. In 2:4 sparsity, exactly 2 out of every 4 consecutive elements are zero, enabling 2x speedup with minimal accuracy loss.

## Optimization 1: CUTLASS 2:4 Sparse GEMM with FP8/INT8
- Commit ID: 60508ffda
- Optimization type: Compute
- Summary: Integrate CUTLASS sparse GEMM with FP8 and INT8 quantization support
- Detailed explanation:
  This optimization combines structured sparsity with quantization for maximum efficiency. The sparse weights are stored in compressed format, and the Tensor Cores handle both decompression and computation.
- Code excerpt:
    ```cpp
    // Sparse scaled MM with FP8
    template <typename ElementA, typename ElementB, typename ElementD>
    void sparse_scaled_mm_sm90(
        torch::Tensor& out,
        torch::Tensor const& a,           // Dense activations
        torch::Tensor const& b_compressed, // 2:4 sparse weights (compressed)
        torch::Tensor const& b_meta,       // Sparsity metadata
        torch::Tensor const& a_scales,
        torch::Tensor const& b_scales) {
      
      using Gemm = cutlass::gemm::device::SparseGemm<
        ElementA, cutlass::layout::RowMajor,
        ElementB, cutlass::layout::ColumnMajor,
        ElementD, cutlass::layout::RowMajor,
        float,  // Accumulator
        cutlass::arch::OpClassSparseTensorOp,  // Sparse Tensor Core
        cutlass::arch::Sm90
      >;
      
      // Launch sparse GEMM
      Gemm gemm_op;
      gemm_op({M, N, K}, 
              {a.data_ptr(), lda},
              {b_compressed.data_ptr(), ldb},
              {b_meta.data_ptr()},  // Sparsity pattern
              ...);
    }
    ```
- Evidence mapping:
  - "2:4 sparsity" → `OpClassSparseTensorOp` operation class
  - "Compressed storage" → `b_compressed` with `b_meta` for pattern
  - "FP8/INT8 support" → Template parameters for element types

## Optimization 2: Sparse Weight Compression
- Commit ID: (sparse_compressor_c3x.cuh)
- Optimization type: Memory
- Summary: Efficient compression of dense weights to 2:4 sparse format
- Detailed explanation:
  Converting dense weights to 2:4 sparse format requires selecting which 2 elements to keep in each group of 4. This kernel implements efficient compression with magnitude-based selection.
- Code excerpt:
    ```cpp
    // Compress dense weights to 2:4 sparse format
    template <typename Element>
    __global__ void compress_to_sparse_kernel(
        Element* __restrict__ compressed,
        uint16_t* __restrict__ metadata,
        const Element* __restrict__ dense,
        int rows, int cols) {
      
      // Each thread handles one group of 4 elements
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      int group_idx = idx;
      int row = group_idx / (cols / 4);
      int col_group = group_idx % (cols / 4);
      
      // Load 4 elements
      Element vals[4];
      for (int i = 0; i < 4; i++) {
        vals[i] = dense[row * cols + col_group * 4 + i];
      }
      
      // Find indices of 2 largest magnitude elements
      int keep[2];
      find_top2_magnitude(vals, keep);
      
      // Store compressed values and metadata
      compressed[row * (cols/2) + col_group * 2 + 0] = vals[keep[0]];
      compressed[row * (cols/2) + col_group * 2 + 1] = vals[keep[1]];
      
      // Encode sparsity pattern in metadata
      uint16_t meta = encode_pattern(keep[0], keep[1]);
      metadata[row * (cols/4) + col_group] = meta;
    }
    ```
- Evidence mapping:
  - "Magnitude selection" → `find_top2_magnitude` keeps largest values
  - "Compressed storage" → Output is half the size of input
  - "Metadata encoding" → Pattern stored for Tensor Core decompression
