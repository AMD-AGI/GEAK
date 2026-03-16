# Kernel: Normalization (LayerNorm, RMSNorm, GroupNorm)

## Variant Context
- Input semantic type: Normalization operations
- Datatype(s): FP32 (primary), FP16 (input/output conversion)
- Data representation: Dense tensors
- Target architecture: Generic (NVIDIA, AMD, Moore Threads)

## Functionality
The normalization kernels implement various normalization operations used in transformer models:
- **RMSNorm**: Root Mean Square normalization, used in LLaMA, Mistral, etc.
- **LayerNorm**: Layer normalization with mean subtraction
- **GroupNorm**: Group normalization for vision models

These kernels compute statistics (mean, variance, RMS) across the feature dimension and normalize the input accordingly.

---

## Optimization 1: Fused RMSNorm with Element-wise Operations
- Commit ID: 8c988fa41, 009b709d6
- Optimization type: Fusion (kernel fusion)
- Summary: Fuse RMSNorm with subsequent multiply and add operations to reduce memory traffic
- Detailed explanation: In transformer models, RMSNorm is typically followed by element-wise multiply (with learned scale) and sometimes add operations. By fusing these operations into a single kernel, we avoid writing the normalized values to global memory and reading them back, significantly reducing memory bandwidth requirements.

- Code excerpt:
    ```cpp
    // CUDA: fused rms norm with multiply
    template<int block_size>
    __global__ void rms_norm_mul_f32(
        const float * __restrict__ x,
        const float * __restrict__ weight,
        float * __restrict__ dst,
        const int ncols,
        const float eps) {
        
        const int row = blockIdx.x;
        const float * x_row = x + row * ncols;
        float * dst_row = dst + row * ncols;
        
        // Compute sum of squares
        float sum_sq = 0.0f;
        for (int col = threadIdx.x; col < ncols; col += block_size) {
            float val = x_row[col];
            sum_sq += val * val;
        }
        sum_sq = block_reduce<block_reduce_method::SUM>(sum_sq, smem);
        
        // Compute RMS and normalize with fused multiply
        const float rms = rsqrtf(sum_sq / ncols + eps);
        for (int col = threadIdx.x; col < ncols; col += block_size) {
            // Fused: normalize * weight in single pass
            dst_row[col] = x_row[col] * rms * weight[col];
        }
    }
    ```

- Evidence mapping:
  - "Fused operations" → normalize and multiply in same loop
  - "Reduced memory traffic" → single read of x, single write to dst
  - "Weight application" → `* weight[col]` fused with normalization

---

## Optimization 2: Fused Add with RMSNorm
- Commit ID: 009b709d6
- Optimization type: Fusion (residual connection)
- Summary: Fuse residual add operation with RMSNorm to eliminate intermediate memory access
- Detailed explanation: Transformer blocks typically have residual connections where the input is added to the output before normalization. This optimization fuses the add operation with RMSNorm, computing `RMSNorm(x + residual)` in a single kernel pass.

- Code excerpt:
    ```cpp
    // CUDA: fuse add with rms norm
    template<int block_size>
    __global__ void add_rms_norm_f32(
        const float * __restrict__ x,
        const float * __restrict__ residual,
        const float * __restrict__ weight,
        float * __restrict__ dst,
        const int ncols,
        const float eps) {
        
        const int row = blockIdx.x;
        
        // Fused add and sum of squares computation
        float sum_sq = 0.0f;
        for (int col = threadIdx.x; col < ncols; col += block_size) {
            float val = x[row * ncols + col] + residual[row * ncols + col];
            sum_sq += val * val;
            // Store intermediate for second pass
            smem_vals[col] = val;
        }
        
        sum_sq = block_reduce<block_reduce_method::SUM>(sum_sq, smem);
        const float rms = rsqrtf(sum_sq / ncols + eps);
        
        // Normalize with weight
        for (int col = threadIdx.x; col < ncols; col += block_size) {
            dst[row * ncols + col] = smem_vals[col] * rms * weight[col];
        }
    }
    ```

- Evidence mapping:
  - "Fused add" → `x[...] + residual[...]` in first loop
  - "Single kernel" → add, compute RMS, normalize all in one launch
  - "Residual connection" → common transformer pattern optimized

---

## Optimization 3: Fast Integer Division for Row Indexing
- Commit ID: 661ae31c9
- Optimization type: Compute (integer arithmetic)
- Summary: Use fast integer division/modulo for computing row indices in multi-dimensional tensors
- Detailed explanation: Integer division and modulo operations are expensive on GPUs. This optimization precomputes magic numbers for fast division using the "magic number" technique, replacing expensive division with multiplication and shifts.

- Code excerpt:
    ```cpp
    // Fast division using magic number technique
    struct fastdiv_consts {
        uint32_t magic;
        uint32_t shift;
    };
    
    __host__ fastdiv_consts compute_fastdiv(uint32_t divisor) {
        // Compute magic number for fast division
        uint32_t shift = 32 - __clz(divisor - 1);
        uint64_t magic = ((1ULL << (32 + shift)) + divisor - 1) / divisor;
        return {(uint32_t)magic, shift};
    }
    
    __device__ __forceinline__ uint32_t fastdiv(uint32_t n, fastdiv_consts c) {
        return __umulhi(n, c.magic) >> c.shift;
    }
    
    __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, uint32_t d, fastdiv_consts c) {
        return n - fastdiv(n, c) * d;
    }
    
    // Usage in rms_norm kernel
    const uint32_t row = fastdiv(blockIdx.x, row_div_consts);
    const uint32_t col_block = fastmodulo(blockIdx.x, ncols_blocks, row_div_consts);
    ```

- Evidence mapping:
  - "Fast division" → `fastdiv()` using `__umulhi` and shift
  - "Precomputed constants" → `fastdiv_consts` struct
  - "Replaces expensive ops" → no actual division in kernel

---

## Optimization 4: Flexible Block Size Selection
- Commit ID: 661ae31c9
- Optimization type: Launch configuration
- Summary: Support multiple block sizes to optimize for different row lengths
- Detailed explanation: Different row lengths benefit from different block sizes. Small rows need fewer threads to avoid underutilization, while large rows benefit from more threads for parallelism. This optimization adds support for multiple block sizes and selects the optimal one based on row length.

- Code excerpt:
    ```cpp
    // Support more block_size values in rms_norm_f32
    template<int block_size>
    __global__ void rms_norm_f32(...);
    
    // Instantiate for multiple block sizes
    template __global__ void rms_norm_f32<64>(...);
    template __global__ void rms_norm_f32<128>(...);
    template __global__ void rms_norm_f32<256>(...);
    template __global__ void rms_norm_f32<512>(...);
    template __global__ void rms_norm_f32<1024>(...);
    
    // Select optimal block size at runtime
    static int get_optimal_block_size(int ncols) {
        if (ncols <= 64) return 64;
        if (ncols <= 128) return 128;
        if (ncols <= 256) return 256;
        if (ncols <= 512) return 512;
        return 1024;
    }
    ```

- Evidence mapping:
  - "Multiple block sizes" → template instantiations for 64-1024
  - "Runtime selection" → `get_optimal_block_size()` function
  - "Adaptive parallelism" → match threads to problem size

---

## Optimization 5: Non-Contiguous Tensor Support
- Commit ID: fd08255d0
- Optimization type: Memory (strided access)
- Summary: Add support for non-contiguous tensors in normalization kernels
- Detailed explanation: When tensors have non-unit strides (e.g., from slicing or transposition), the kernel must handle strided memory access. This optimization adds stride parameters and computes correct memory offsets for non-contiguous inputs.

- Code excerpt:
    ```cpp
    // CUDA: non-contiguous (RMS) norm support
    template<int block_size>
    __global__ void rms_norm_f32_noncontig(
        const float * __restrict__ x,
        float * __restrict__ dst,
        const int ncols,
        const int64_t stride_x,   // Input stride
        const int64_t stride_dst, // Output stride
        const float eps) {
        
        const int row = blockIdx.x;
        
        // Use strides for non-contiguous access
        const float * x_row = x + row * stride_x;
        float * dst_row = dst + row * stride_dst;
        
        float sum_sq = 0.0f;
        for (int col = threadIdx.x; col < ncols; col += block_size) {
            float val = x_row[col];  // Strided access
            sum_sq += val * val;
        }
        ...
    }
    ```

- Evidence mapping:
  - "Stride parameters" → `stride_x`, `stride_dst` arguments
  - "Strided access" → `x + row * stride_x` offset calculation
  - "Non-contiguous support" → handles sliced/transposed tensors

---

## Optimization 6: Block Reduce Reuse
- Commit ID: 36f013246
- Optimization type: Compute (code reuse)
- Summary: Use shared block_reduce function for consistent and optimized reductions
- Detailed explanation: The block_reduce function provides an optimized two-stage reduction (warp shuffle + shared memory) that is reused across RMSNorm, LayerNorm, GroupNorm, and L2Norm kernels. This ensures consistent performance and reduces code duplication.

- Code excerpt:
    ```cpp
    // Use block_reduce in all norm kernels
    __global__ void rms_norm_f32(...) {
        float sum_sq = 0.0f;
        for (int col = threadIdx.x; col < ncols; col += block_size) {
            float val = x_row[col];
            sum_sq += val * val;
        }
        
        // Shared block_reduce function
        sum_sq = block_reduce<block_reduce_method::SUM, float, block_size>(sum_sq, smem);
        
        const float rms = rsqrtf(sum_sq / ncols + eps);
        ...
    }
    
    __global__ void layer_norm_f32(...) {
        // Same block_reduce for mean
        float mean = block_reduce<block_reduce_method::SUM, float, block_size>(sum, smem) / ncols;
        // Same block_reduce for variance
        float var = block_reduce<block_reduce_method::SUM, float, block_size>(sum_sq, smem) / ncols;
        ...
    }
    ```

- Evidence mapping:
  - "Shared function" → `block_reduce<...>()` template
  - "Consistent optimization" → same efficient reduction everywhere
  - "Code reuse" → single implementation for all norm kernels
