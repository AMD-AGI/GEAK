Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: Matrix-Vector Multiplication Quantized (MMVQ)

## Variant Context
- Input semantic type: Matrix-vector multiplication (GEMV)
- Datatype(s): Quantized weights (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, K-quants, IQ types)
- Data representation: Block-wise quantized weights, FP16/FP32 activations
- Target architecture: Generic (NVIDIA, AMD, Moore Threads)

## Functionality
The MMVQ kernel performs matrix-vector multiplication where the matrix (weights) is stored in a quantized format. This is the primary kernel used during single-token generation (batch size = 1), where the activation is a single vector rather than a matrix.

Key features:
- Optimized for batch size 1 inference
- Support for all quantization formats
- Fused dequantization during computation
- Warp-level parallelism for dot products

---

## Optimization 1: General GEMV Fusion
- Commit ID: f77c13b91
- Optimization type: Fusion (kernel fusion)
- Summary: Fuse bias addition and gate operations with GEMV for reduced memory traffic
- Detailed explanation: In transformer FFN layers, the GEMV output is often followed by bias addition and gating (e.g., SiLU activation with gate). This optimization fuses these operations into the GEMV kernel, avoiding intermediate memory writes.

- Code excerpt:
    ```cpp
    // CUDA: General GEMV fusion
    template<typename T, int qk, int qr, typename dequant_func>
    __global__ void mmvq_fused(
        const void * __restrict__ x,      // Quantized weights
        const T * __restrict__ y,         // Input vector
        const T * __restrict__ bias,      // Optional bias
        const T * __restrict__ gate,      // Optional gate values
        T * __restrict__ dst,
        const int ncols,
        const int nrows) {
        
        const int row = blockIdx.x * blockDim.y + threadIdx.y;
        if (row >= nrows) return;
        
        float sum = 0.0f;
        
        // Compute dot product with dequantization
        for (int col = threadIdx.x; col < ncols; col += WARP_SIZE) {
            const float weight = dequant_func(x, row, col);
            sum += weight * (float)y[col];
        }
        
        // Warp reduction
        sum = warp_reduce_sum(sum);
        
        if (threadIdx.x == 0) {
            // Fused bias and gate
            if (bias) sum += (float)bias[row];
            if (gate) sum *= silu((float)gate[row]);
            dst[row] = (T)sum;
        }
    }
    ```

- Evidence mapping:
  - "Fused operations" → bias and gate applied in same kernel
  - "Reduced memory traffic" → no intermediate buffer for GEMV output
  - "SiLU gating" → `silu((float)gate[row])` fused with output

---

## Optimization 2: Fast Division for Index Computation
- Commit ID: 5143fa895
- Optimization type: Compute (integer arithmetic)
- Summary: Use fast integer division for computing block indices in quantized data
- Detailed explanation: Quantized formats use block structures (e.g., 32 elements per block). Computing which block a given index belongs to requires division, which is expensive. This optimization uses the fast division technique with precomputed magic numbers.

- Code excerpt:
    ```cpp
    // CUDA: fastdiv, launch bounds for mmvq + q8_1 quant
    __device__ __forceinline__ int get_block_idx(int col, uint3 fastdiv_consts) {
        // Fast division: col / block_size
        return fastdiv(col, fastdiv_consts);
    }
    
    template<typename T, int block_size>
    __global__ void mmvq_q4_0(
        const void * __restrict__ x,
        const T * __restrict__ y,
        T * __restrict__ dst,
        const int ncols,
        const int nrows,
        const uint3 block_div_consts) {  // Precomputed for fast division
        
        const int row = blockIdx.x * blockDim.y + threadIdx.y;
        
        float sum = 0.0f;
        for (int col = threadIdx.x; col < ncols; col += WARP_SIZE) {
            // Fast block index computation
            const int block_idx = get_block_idx(col, block_div_consts);
            const int elem_idx = col - block_idx * block_size;
            
            // Dequantize and accumulate
            const block_q4_0 * block = ((const block_q4_0 *)x) + row * (ncols / block_size) + block_idx;
            const float d = __half2float(block->d);
            const int qs = block->qs[elem_idx / 2];
            const float weight = d * ((elem_idx % 2 == 0 ? qs & 0xF : qs >> 4) - 8);
            
            sum += weight * (float)y[col];
        }
        ...
    }
    ```

- Evidence mapping:
  - "Fast division" → `fastdiv()` with precomputed constants
  - "Block index" → `get_block_idx()` replaces expensive division
  - "Launch parameter" → `block_div_consts` passed to kernel

---

## Optimization 3: IQ Format Optimization
- Commit ID: cb5fad4c6
- Optimization type: Compute (lookup table)
- Summary: Refactor and optimize IQ (integer quantization) formats for MMVQ
- Detailed explanation: IQ formats use non-linear quantization with lookup tables. This optimization loads the lookup tables into shared memory and uses vectorized loads for efficient dequantization during the dot product computation.

- Code excerpt:
    ```cpp
    // CUDA: refactor and optimize IQ MMVQ
    template<typename T>
    __global__ void mmvq_iq4_nl(
        const void * __restrict__ x,
        const T * __restrict__ y,
        T * __restrict__ dst,
        const int ncols,
        const int nrows) {
        
        // Load IQ4_NL lookup table to shared memory
        __shared__ int8_t iq4nl_lut[16];
        if (threadIdx.x < 16 && threadIdx.y == 0) {
            iq4nl_lut[threadIdx.x] = kvalues_iq4nl[threadIdx.x];
        }
        __syncthreads();
        
        const int row = blockIdx.x * blockDim.y + threadIdx.y;
        float sum = 0.0f;
        
        for (int col = threadIdx.x; col < ncols; col += WARP_SIZE) {
            const block_iq4_nl * block = ...;
            const float d = __half2float(block->d);
            const uint8_t qs = block->qs[col % QK4_NL];
            
            // Lookup table dequantization
            const float w0 = d * iq4nl_lut[qs & 0xF];
            const float w1 = d * iq4nl_lut[qs >> 4];
            
            sum += w0 * (float)y[col] + w1 * (float)y[col + 1];
        }
        ...
    }
    ```

- Evidence mapping:
  - "Shared memory LUT" → `__shared__ int8_t iq4nl_lut[16]`
  - "Non-linear dequant" → `iq4nl_lut[qs & 0xF]` lookup
  - "Vectorized processing" → process two elements per iteration

---

## Optimization 4: Warp-Level Parallelism Refactoring
- Commit ID: 10f2e8180
- Optimization type: Compute (parallelism)
- Summary: Refactor MMVQ to unify warp and row calculations between host and device code
- Detailed explanation: The MMVQ kernel uses multiple warps to process different rows in parallel. This optimization ensures consistent calculation of the number of warps and rows per block between host (launch configuration) and device (kernel) code, avoiding mismatches that could cause incorrect results.

- Code excerpt:
    ```cpp
    // CUDA/HIP: refactor mmvq to unify calculation of nwarps and rows per block
    static int get_mmvq_nwarps(const int nrows, const int ncols) {
        // Consistent calculation for host and device
        const int max_warps = 8;
        const int rows_per_warp = 1;
        const int nwarps = min(max_warps, (nrows + rows_per_warp - 1) / rows_per_warp);
        return nwarps;
    }
    
    static int get_mmvq_rows_per_block(const int nwarps) {
        return nwarps;  // One row per warp
    }
    
    // Device code uses same logic
    template<typename T, int nwarps>
    __global__ void mmvq_kernel(...) {
        static_assert(nwarps == get_mmvq_nwarps(...), "Mismatch!");
        const int row = blockIdx.x * nwarps + threadIdx.y;
        ...
    }
    ```

- Evidence mapping:
  - "Unified calculation" → `get_mmvq_nwarps()` used by host and device
  - "Consistent parallelism" → same nwarps in launch config and kernel
  - "One row per warp" → `rows_per_warp = 1` for simplicity

---

## Optimization 5: Non-Contiguous Input Support
- Commit ID: 658987cfc
- Optimization type: Memory (strided access)
- Summary: Add support for non-contiguous inputs and batched MUL_MAT_ID for MoE
- Detailed explanation: For Mixture of Experts models, different tokens are routed to different experts, requiring batched GEMV with non-contiguous inputs. This optimization adds stride parameters and proper index calculations for handling such cases.

- Code excerpt:
    ```cpp
    // CUDA: noncont MMVQ + batched bs1 MUL_MAT_ID
    template<typename T>
    __global__ void mmvq_batched(
        const void * __restrict__ x,
        const T * __restrict__ y,
        T * __restrict__ dst,
        const int32_t * __restrict__ ids,  // Expert routing
        const int ncols,
        const int nrows,
        const int64_t stride_y,   // Input stride (may be non-contiguous)
        const int64_t stride_dst, // Output stride
        const int n_experts) {
        
        const int batch = blockIdx.z;
        const int expert_id = ids[batch];
        const int row = blockIdx.x * blockDim.y + threadIdx.y;
        
        // Offset to correct expert weights
        const void * x_expert = (const char *)x + expert_id * expert_stride;
        
        // Use strides for non-contiguous access
        const T * y_batch = y + batch * stride_y;
        T * dst_batch = dst + batch * stride_dst;
        
        float sum = 0.0f;
        for (int col = threadIdx.x; col < ncols; col += WARP_SIZE) {
            sum += dequant(x_expert, row, col) * (float)y_batch[col];
        }
        ...
    }
    ```

- Evidence mapping:
  - "Batched operation" → `blockIdx.z` for batch dimension
  - "Expert routing" → `ids[batch]` for MoE
  - "Non-contiguous" → `stride_y`, `stride_dst` parameters
