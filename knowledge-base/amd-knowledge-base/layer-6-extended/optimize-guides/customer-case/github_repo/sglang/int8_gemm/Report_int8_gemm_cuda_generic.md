# Kernel: INT8 GEMM Kernels

## Variant Context
- Input semantic type: Matrix multiplication with INT8 quantized weights
- Datatype(s): INT8 weights, FP16/BF16 activations, FP32 accumulation
- Data representation: Per-channel or per-tensor quantized weights
- Target architecture: SM80+ (Ampere and later with INT8 tensor cores)

## Functionality
These kernels implement INT8 quantized GEMM for efficient inference with W8A8 (8-bit weights, 8-bit activations) or W8A16 (8-bit weights, 16-bit activations) configurations.

## Optimization 1: Per-Channel INT8 GEMM
- Commit ID: (int8_gemm_kernel.cu)
- Optimization type: Precision / Compute
- Summary: Implements per-channel quantized INT8 GEMM with efficient scale handling.

- Detailed explanation:
  Per-channel quantization uses different scales for each output channel, providing better accuracy than per-tensor quantization. This kernel efficiently handles per-channel scales during the GEMM computation.

- Code excerpt:
    ```cpp
    __global__ void int8_gemm_per_channel_kernel(
        const int8_t* __restrict__ A,    // [M, K] INT8 activations
        const int8_t* __restrict__ B,    // [K, N] INT8 weights
        half* __restrict__ C,            // [M, N] FP16 output
        const float* __restrict__ a_scale,  // [M] per-token scales
        const float* __restrict__ b_scale,  // [N] per-channel scales
        int M, int N, int K
    ) {
        // Use INT8 tensor cores for GEMM
        // Accumulate in INT32
        int32_t acc[TILE_M][TILE_N] = {0};
        
        for (int k = 0; k < K; k += TILE_K) {
            // Load INT8 tiles
            int8_t a_tile[TILE_M][TILE_K];
            int8_t b_tile[TILE_K][TILE_N];
            // ... load logic
            
            // INT8 tensor core GEMM
            acc = mma_int8(a_tile, b_tile, acc);
        }
        
        // Dequantize with per-channel scales
        for (int m = 0; m < TILE_M; m++) {
            float a_s = a_scale[row + m];
            for (int n = 0; n < TILE_N; n++) {
                float b_s = b_scale[col + n];
                C[...] = __float2half(acc[m][n] * a_s * b_s);
            }
        }
    }
    ```

- Evidence mapping:
  - "Per-channel scales" → `b_scale[col + n]` indexed by output channel
  - "INT8 tensor cores" → `mma_int8` for hardware acceleration
  - "INT32 accumulation" → `int32_t acc` for precision

## Optimization 2: Per-Token Group Quantization
- Commit ID: (per_token_group_quant_8bit.cu)
- Optimization type: Precision
- Summary: Implements per-token group quantization for better accuracy with INT8.

- Detailed explanation:
  Per-token group quantization divides each token's features into groups, with separate scales per group. This provides better accuracy than per-token quantization while maintaining efficiency.

- Code excerpt:
    ```cpp
    __global__ void per_token_group_quant_8bit_kernel(
        const half* __restrict__ input,
        int8_t* __restrict__ output,
        float* __restrict__ scales,
        int num_tokens,
        int hidden_size,
        int group_size
    ) {
        int token_idx = blockIdx.x;
        int group_idx = blockIdx.y;
        
        // Find max in group
        float max_val = 0.0f;
        for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
            int idx = token_idx * hidden_size + group_idx * group_size + i;
            max_val = fmaxf(max_val, fabsf(__half2float(input[idx])));
        }
        max_val = blockReduceMax(max_val);
        
        // Compute scale
        float scale = max_val / 127.0f;
        if (threadIdx.x == 0) {
            scales[token_idx * (hidden_size / group_size) + group_idx] = scale;
        }
        
        // Quantize
        for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
            int idx = token_idx * hidden_size + group_idx * group_size + i;
            float val = __half2float(input[idx]);
            output[idx] = __float2int_rn(val / scale);
        }
    }
    ```

- Evidence mapping:
  - "Group-wise scales" → `scales[token_idx * (hidden_size / group_size) + group_idx]`
  - "Per-token processing" → `token_idx = blockIdx.x`
  - "Block reduction" → `blockReduceMax` for finding group maximum
