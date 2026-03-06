Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: Copy and Convert Operations

## Variant Context
- Input semantic type: Data movement and type conversion
- Datatype(s): FP32, FP16, BF16, Quantized types
- Data representation: Dense and strided tensors
- Target architecture: Generic (NVIDIA, AMD, Moore Threads)

## Functionality
The copy and convert kernels handle data movement between tensors with optional type conversion. These are fundamental operations used throughout the inference pipeline for:
- Copying data between devices or memory regions
- Converting between data types (FP32 ↔ FP16 ↔ BF16)
- Handling non-contiguous (strided) tensors
- Quantization and dequantization

---

## Optimization 1: Large Tensor Copy Fix
- Commit ID: e86f3c222
- Optimization type: Correctness (large tensors)
- Summary: Fix copy of large tensors where ggml_nbytes exceeds INT_MAX
- Detailed explanation: For very large tensors (>2GB), the byte count can exceed INT_MAX. This fix ensures correct handling of large tensor copies by using 64-bit arithmetic for size calculations.

- Code excerpt:
    ```cpp
    // cuda: fix copy of large tensors (ggml_nbytes <= INT_MAX assertion)
    template<typename T_SRC, typename T_DST>
    __global__ void cpy_large(
        const T_SRC * __restrict__ src,
        T_DST * __restrict__ dst,
        const int64_t n) {  // Use int64_t for large sizes
        
        const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        
        dst[idx] = (T_DST)src[idx];
    }
    
    void ggml_cuda_cpy(
        const ggml_tensor * src,
        ggml_tensor * dst,
        cudaStream_t stream) {
        
        const int64_t n = ggml_nelements(src);  // 64-bit element count
        
        // Use 64-bit grid calculation
        const int64_t blocks = (n + 255) / 256;
        
        // Handle very large grids
        if (blocks > INT_MAX) {
            // Split into multiple launches
            ...
        }
        
        cpy_large<<<blocks, 256, 0, stream>>>(src_data, dst_data, n);
    }
    ```

- Evidence mapping:
  - "64-bit sizes" → `int64_t n` parameter
  - "Large tensor support" → handles >2GB tensors
  - "Grid calculation" → 64-bit block count

---

## Optimization 2: Vectorized Copy Utilities
- Commit ID: Various (cpy-utils.cuh)
- Optimization type: Memory (bandwidth)
- Summary: Provide vectorized copy utilities for efficient memory transfers
- Detailed explanation: Memory bandwidth is maximized when using larger transaction sizes. These utilities provide templated vectorized copy functions that automatically select the optimal vector width based on alignment and size.

- Code excerpt:
    ```cpp
    // cpy-utils.cuh: Vectorized copy utilities
    template<int BYTES>
    __device__ __forceinline__ void memcpy_aligned(
        void * __restrict__ dst,
        const void * __restrict__ src) {
        
        if constexpr (BYTES == 16) {
            *reinterpret_cast<float4*>(dst) = 
                *reinterpret_cast<const float4*>(src);
        } else if constexpr (BYTES == 8) {
            *reinterpret_cast<float2*>(dst) = 
                *reinterpret_cast<const float2*>(src);
        } else if constexpr (BYTES == 4) {
            *reinterpret_cast<float*>(dst) = 
                *reinterpret_cast<const float*>(src);
        } else {
            // Byte-by-byte for unaligned
            for (int i = 0; i < BYTES; i++) {
                reinterpret_cast<char*>(dst)[i] = 
                    reinterpret_cast<const char*>(src)[i];
            }
        }
    }
    
    // Strided copy with vectorization
    template<typename T, int VEC_SIZE>
    __global__ void cpy_strided_vec(
        const T * __restrict__ src,
        T * __restrict__ dst,
        const int64_t src_stride,
        const int64_t dst_stride,
        const int64_t n_rows,
        const int64_t n_cols) {
        
        using vec_t = typename std::conditional<
            VEC_SIZE == 4, float4,
            typename std::conditional<VEC_SIZE == 2, float2, float>::type
        >::type;
        
        const int row = blockIdx.y;
        const int col = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;
        
        if (row < n_rows && col < n_cols) {
            vec_t val = *reinterpret_cast<const vec_t*>(
                src + row * src_stride + col);
            *reinterpret_cast<vec_t*>(
                dst + row * dst_stride + col) = val;
        }
    }
    ```

- Evidence mapping:
  - "Vectorized access" → `float4`, `float2` types
  - "Alignment handling" → template specializations
  - "Strided support" → separate strides for src/dst

---

## Optimization 3: Type Conversion Kernels
- Commit ID: Various (convert.cu)
- Optimization type: Compute (type conversion)
- Summary: Efficient kernels for converting between FP32, FP16, BF16, and quantized types
- Detailed explanation: Type conversion is a common operation when moving data between different precision levels. These kernels provide optimized conversion with proper rounding and handling of special values (NaN, Inf).

- Code excerpt:
    ```cpp
    // convert.cu: Type conversion kernels
    
    // FP32 to FP16
    __global__ void convert_f32_to_f16(
        const float * __restrict__ src,
        half * __restrict__ dst,
        const int n) {
        
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        
        dst[idx] = __float2half(src[idx]);
    }
    
    // FP32 to BF16
    __global__ void convert_f32_to_bf16(
        const float * __restrict__ src,
        __nv_bfloat16 * __restrict__ dst,
        const int n) {
        
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        
        dst[idx] = __float2bfloat16(src[idx]);
    }
    
    // Vectorized FP16 to FP32
    __global__ void convert_f16_to_f32_vec(
        const half2 * __restrict__ src,
        float2 * __restrict__ dst,
        const int n) {
        
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n/2) return;
        
        dst[idx] = __half22float2(src[idx]);
    }
    
    // Quantized to FP32 (dequantization)
    template<typename Q_TYPE>
    __global__ void dequantize_kernel(
        const Q_TYPE * __restrict__ src,
        float * __restrict__ dst,
        const int n) {
        
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        
        dst[idx] = dequantize<Q_TYPE>(src, idx);
    }
    ```

- Evidence mapping:
  - "Type conversion" → `__float2half`, `__float2bfloat16`
  - "Vectorized" → `half2`, `float2` for 2x throughput
  - "Dequantization" → template for all quantized types

---

## Optimization 4: Non-Contiguous Tensor Support
- Commit ID: Various
- Optimization type: Memory (strided access)
- Summary: Support copying non-contiguous tensors with arbitrary strides
- Detailed explanation: Tensors from slicing or transposition may have non-unit strides. These kernels handle arbitrary stride patterns while maintaining good memory access patterns where possible.

- Code excerpt:
    ```cpp
    // Non-contiguous copy with 4D stride support
    template<typename T>
    __global__ void cpy_4d_strided(
        const T * __restrict__ src,
        T * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, 
        const int64_t ne2, const int64_t ne3,
        const int64_t s0_src, const int64_t s1_src,
        const int64_t s2_src, const int64_t s3_src,
        const int64_t s0_dst, const int64_t s1_dst,
        const int64_t s2_dst, const int64_t s3_dst) {
        
        const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int i1 = blockIdx.y;
        const int i2 = blockIdx.z % ne2;
        const int i3 = blockIdx.z / ne2;
        
        if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) return;
        
        const int64_t src_idx = i0*s0_src + i1*s1_src + i2*s2_src + i3*s3_src;
        const int64_t dst_idx = i0*s0_dst + i1*s1_dst + i2*s2_dst + i3*s3_dst;
        
        dst[dst_idx] = src[src_idx];
    }
    ```

- Evidence mapping:
  - "4D strides" → separate stride for each dimension
  - "Arbitrary layout" → handles transposed, sliced tensors
  - "Index calculation" → `i*stride` for each dimension
