Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: Marlin GEMM Kernels

## Variant Context
- Input semantic type: Quantized matrix multiplication for INT4/INT8 weights
- Datatype(s): INT4, INT8 weights with FP16/BF16 activations
- Data representation: Marlin-format packed weights with group scales
- Target architecture: SM80+ (Ampere and later)

## Functionality
Marlin kernels implement highly optimized GEMM for quantized weights, particularly for GPTQ and AWQ quantization formats. The Marlin format repacks weights for optimal memory access patterns on modern GPUs.

## Optimization 1: GPTQ Marlin Optimized Layout
- Commit ID: (gptq_marlin.cu)
- Optimization type: Memory Layout
- Summary: Uses Marlin's optimized weight layout for GPTQ quantized models, achieving near-FP16 performance with INT4 weights.

- Detailed explanation:
  Marlin repacks GPTQ weights into a format that:
  1. Maximizes memory coalescing for INT4 loads
  2. Enables efficient dequantization in registers
  3. Overlaps dequantization with tensor core computation

- Code excerpt:
    ```cpp
    // Marlin weight layout for optimal access
    // Weights are packed as: [N/8, K/16, 8, 16] for INT4
    // This enables 128-bit loads that map directly to tensor core inputs
    
    __global__ void gptq_marlin_gemm_kernel(
        const uint32_t* __restrict__ B,  // Packed INT4 weights
        const half* __restrict__ A,       // FP16 activations
        half* __restrict__ C,             // FP16 output
        const half* __restrict__ scales,  // Per-group scales
        const int* __restrict__ g_idx,    // Group indices
        int M, int N, int K,
        int group_size
    ) {
        // Load 8 INT4 values (32 bits) at once
        uint32_t packed_weights = B[weight_offset];
        
        // Dequantize in registers
        half2 weights[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int4 w = (packed_weights >> (i * 8)) & 0xF;
            weights[i] = __half2{scale * (w.x - 8), scale * (w.y - 8)};
        }
        
        // Tensor core GEMM with dequantized weights
        // ...
    }
    ```

- Evidence mapping:
  - "Packed layout" → `[N/8, K/16, 8, 16]` format
  - "Efficient loads" → 32-bit load for 8 INT4 values
  - "Register dequant" → dequantization before tensor core ops

## Optimization 2: AWQ Marlin Repack
- Commit ID: (awq_marlin_repack.cu)
- Optimization type: Data Transformation
- Summary: Repacks AWQ quantized weights into Marlin format for optimized inference.

- Detailed explanation:
  AWQ uses a different weight layout than Marlin expects. This kernel efficiently repacks AWQ weights into Marlin format, enabling the use of Marlin's optimized GEMM kernels with AWQ-quantized models.

- Code excerpt:
    ```cpp
    __global__ void awq_marlin_repack_kernel(
        const uint32_t* __restrict__ awq_weights,
        uint32_t* __restrict__ marlin_weights,
        int N, int K
    ) {
        // Repack from AWQ layout to Marlin layout
        // AWQ: [K/8, N] with 8 INT4 per uint32
        // Marlin: [N/8, K/16, 8, 16] for optimal tensor core access
        
        int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int k_idx = blockIdx.y * blockDim.y + threadIdx.y;
        
        // Read AWQ format
        uint32_t awq_packed = awq_weights[k_idx * N / 8 + n_idx / 8];
        
        // Reorder for Marlin format
        // ...
    }
    ```

- Evidence mapping:
  - "Layout transformation" → AWQ to Marlin format conversion
  - "Offline repack" → done once at model load time
