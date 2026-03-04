---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: Fused Add RMSNorm Kernel

## Variant Context
- Input semantic type: Residual addition followed by RMS normalization
- Datatype(s): FP16, BF16, FP32
- Data representation: Token embeddings/hidden states
- Target architecture: Generic CUDA and ROCm

## Functionality
This kernel fuses the residual addition and RMS normalization operations commonly found in transformer architectures:
1. Residual addition: x = x + residual
2. RMS normalization: y = x * weight / sqrt(mean(x^2) + eps)

Fusing these operations reduces memory bandwidth by avoiding intermediate tensor materialization.

## Optimization 1: Single-Pass Variance Computation
- Commit ID: (fused_add_rms_norm_kernel.cu)
- Optimization type: Memory / Compute
- Summary: Computes the RMS (root mean square) in a single pass while performing the residual addition.

- Detailed explanation:
  Instead of:
  1. Add residual and store
  2. Load, compute mean of squares
  3. Load again, normalize
  
  The fused kernel:
  1. Loads both inputs once
  2. Computes sum and sum of squares simultaneously
  3. Normalizes and stores in one pass

- Code excerpt:
    ```cpp
    __global__ void fused_add_rmsnorm_kernel(
        float* output,
        float* input,
        const float* residual,
        const float* weight,
        float eps,
        int hidden_size
    ) {
        int row = blockIdx.x;
        float* row_input = input + row * hidden_size;
        const float* row_residual = residual + row * hidden_size;
        float* row_output = output + row * hidden_size;
        
        // Single-pass: add residual and compute sum of squares
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            float val = row_input[i] + row_residual[i];
            row_input[i] = val;  // Store fused result
            sum_sq += val * val;
        }
        
        // Reduce sum of squares across threads
        sum_sq = blockReduceSum(sum_sq);
        
        // Compute RMS
        __shared__ float s_rms;
        if (threadIdx.x == 0) {
            s_rms = rsqrtf(sum_sq / hidden_size + eps);
        }
        __syncthreads();
        
        // Normalize with weight
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            row_output[i] = row_input[i] * s_rms * weight[i];
        }
    }
    ```

- Evidence mapping:
  - "Single pass" → residual add and sum_sq computed in same loop
  - "Fused operations" → `val = row_input[i] + row_residual[i]` followed by `sum_sq += val * val`
  - "Block reduction" → `blockReduceSum(sum_sq)` for efficient parallel reduction

## Optimization 2: Vectorized Memory Access
- Commit ID: (fused_add_rms_norm_kernel.cu)
- Optimization type: Memory
- Summary: Uses vectorized loads/stores (float4) to maximize memory bandwidth utilization.

- Detailed explanation:
  By loading and storing 4 float values at a time, the kernel achieves better memory coalescing and higher effective bandwidth. This is particularly important for the hidden dimension which is typically large (4096+).

- Code excerpt:
    ```cpp
    // Vectorized version
    __global__ void fused_add_rmsnorm_kernel_vec4(
        float4* output,
        float4* input,
        const float4* residual,
        const float4* weight,
        float eps,
        int hidden_size_vec4
    ) {
        int row = blockIdx.x;
        float4* row_input = input + row * hidden_size_vec4;
        
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < hidden_size_vec4; i += blockDim.x) {
            float4 in = row_input[i];
            float4 res = residual[row * hidden_size_vec4 + i];
            
            // Vectorized add
            float4 val;
            val.x = in.x + res.x;
            val.y = in.y + res.y;
            val.z = in.z + res.z;
            val.w = in.w + res.w;
            
            row_input[i] = val;
            sum_sq += val.x * val.x + val.y * val.y + 
                      val.z * val.z + val.w * val.w;
        }
        // ... rest of kernel
    }
    ```

- Evidence mapping:
  - "Vectorized types" → `float4*` pointers
  - "4 elements per access" → `val.x, val.y, val.z, val.w`
  - "Coalesced access" → sequential `i` values map to consecutive float4s

## Optimization 3: Warp-Level Reduction
- Commit ID: (fused_add_rms_norm_kernel.cu)
- Optimization type: Compute
- Summary: Uses warp shuffle instructions for efficient parallel reduction of the sum of squares.

- Detailed explanation:
  Instead of using shared memory for the reduction, warp shuffle instructions provide lower latency communication between threads in the same warp. The reduction is done in two stages:
  1. Warp-level reduction using `__shfl_xor_sync`
  2. Cross-warp reduction using shared memory (only for the warp leaders)

- Code excerpt:
    ```cpp
    __device__ float warpReduceSum(float val) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
        return val;
    }

    __device__ float blockReduceSum(float val) {
        __shared__ float shared[32];  // One slot per warp
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;
        
        // Warp-level reduction
        val = warpReduceSum(val);
        
        // Write warp result to shared memory
        if (lane == 0) {
            shared[wid] = val;
        }
        __syncthreads();
        
        // Final reduction by first warp
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
        if (wid == 0) {
            val = warpReduceSum(val);
        }
        
        return val;
    }
    ```

- Evidence mapping:
  - "Warp shuffle" → `__shfl_xor_sync(0xffffffff, val, offset)`
  - "Two-stage reduction" → warp-level then cross-warp
  - "Minimal shared memory" → only 32 floats for warp leaders
