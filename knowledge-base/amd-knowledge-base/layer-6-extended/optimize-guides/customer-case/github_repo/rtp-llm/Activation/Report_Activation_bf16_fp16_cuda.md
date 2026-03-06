Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: Activation Kernels (GELU, SiLU, etc.)

## Variant Context
- Input semantic type: Activation functions (non-linear transformations in FFN)
- Datatype(s): bf16, fp16, fp32
- Data representation: Token-wise activation with optional gating
- Target architecture: CUDA (SM70+), ROCm (gfx942)

## Functionality
These kernels implement activation functions used in transformer FFN layers:
- **GELU**: Gaussian Error Linear Unit (used in BERT, GPT)
- **SiLU/Swish**: Sigmoid Linear Unit (used in LLaMA, Qwen)
- **GeGLU/SwiGLU**: Gated variants with element-wise multiplication

The kernels support:
- Fused gating operations
- Vectorized computation
- Optional bias addition

## Optimization 1: Fused SiLU and Mul (SwiGLU)
- Commit ID: (core implementation)
- Optimization type: Fusion
- Summary: Fuse SiLU activation with gating multiplication
- Detailed explanation:
  SwiGLU combines SiLU activation with element-wise gating:
  - `output = SiLU(gate) * up`
  - Fusing reduces memory traffic by 2x
  - Single kernel for both operations

- Code excerpt:
    ```cpp
    // From activation_kernels.cu
    template<typename T>
    __global__ void siluAndMulKernel(
        T* output,
        const T* gate,
        const T* up,
        int batch_size, int hidden_size) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= batch_size * hidden_size) return;
        
        float gate_val = static_cast<float>(gate[idx]);
        float up_val = static_cast<float>(up[idx]);
        
        // SiLU: x * sigmoid(x)
        float silu_val = gate_val / (1.0f + expf(-gate_val));
        
        // Multiply with up projection
        output[idx] = static_cast<T>(silu_val * up_val);
    }
    
    // Vectorized version
    template<typename T, int VEC_SIZE>
    __global__ void siluAndMulVecKernel(
        T* output,
        const T* gate,
        const T* up,
        int batch_size, int hidden_size) {
        
        using VecT = typename VecType<T, VEC_SIZE>::Type;
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int vec_idx = idx * VEC_SIZE;
        if (vec_idx >= batch_size * hidden_size) return;
        
        VecT gate_vec = *reinterpret_cast<const VecT*>(&gate[vec_idx]);
        VecT up_vec = *reinterpret_cast<const VecT*>(&up[vec_idx]);
        VecT out_vec;
        
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            float g = static_cast<float>(gate_vec.data[i]);
            float u = static_cast<float>(up_vec.data[i]);
            float silu = g / (1.0f + expf(-g));
            out_vec.data[i] = static_cast<T>(silu * u);
        }
        
        *reinterpret_cast<VecT*>(&output[vec_idx]) = out_vec;
    }
    ```
- Evidence mapping:
  - Fused operations → SiLU and multiply in single kernel
  - Vectorized version → `VecType<T, VEC_SIZE>` for wider loads
  - Unrolled computation → `#pragma unroll` for vector elements

## Optimization 2: Fast GELU Approximation
- Commit ID: (core implementation)
- Optimization type: Compute
- Summary: Use fast approximation for GELU activation
- Detailed explanation:
  GELU can be approximated using tanh:
  - `GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
  - Faster than exact erf-based computation
  - Sufficient accuracy for inference

- Code excerpt:
    ```cpp
    // Fast GELU approximation
    __device__ __forceinline__ float fastGelu(float x) {
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coef = 0.044715f;
        
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coef * x3);
        return 0.5f * x * (1.0f + tanhf(inner));
    }
    
    template<typename T>
    __global__ void geluKernel(T* output, const T* input, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size) return;
        
        float val = static_cast<float>(input[idx]);
        output[idx] = static_cast<T>(fastGelu(val));
    }
    ```
- Evidence mapping:
  - Tanh approximation → `tanhf(inner)` instead of `erff`
  - Polynomial approximation → `x + coef * x³`
  - Inline function → `__forceinline__` for no call overhead

## Optimization 3: Fused Bias and Activation
- Commit ID: (core implementation)
- Optimization type: Fusion
- Summary: Fuse bias addition with activation function
- Detailed explanation:
  Many layers have bias followed by activation:
  - Fusing reduces memory round-trip
  - Single kernel for bias + activation
  - Supports all activation types

- Code excerpt:
    ```cpp
    // Fused bias and activation
    template<typename T, ActivationType ACT>
    __global__ void biasActivationKernel(
        T* output,
        const T* input,
        const T* bias,
        int batch_size, int hidden_size) {
        
        int batch_idx = blockIdx.x;
        int tid = threadIdx.x;
        
        for (int i = tid; i < hidden_size; i += blockDim.x) {
            float val = static_cast<float>(input[batch_idx * hidden_size + i]);
            val += static_cast<float>(bias[i]);
            
            // Apply activation based on template parameter
            if constexpr (ACT == ActivationType::GELU) {
                val = fastGelu(val);
            } else if constexpr (ACT == ActivationType::SILU) {
                val = val / (1.0f + expf(-val));
            } else if constexpr (ACT == ActivationType::RELU) {
                val = fmaxf(val, 0.0f);
            }
            
            output[batch_idx * hidden_size + i] = static_cast<T>(val);
        }
    }
    ```
- Evidence mapping:
  - Fused bias → `val += bias[i]` before activation
  - Template activation → `if constexpr` for compile-time dispatch
  - Multiple types → GELU, SILU, RELU supported

## Optimization 4: ROCm Masked SiLU and Mul
- Commit ID: 80eeede7f
- Optimization type: Compute / Architecture-specific
- Summary: Optimized SiLU with masking for ROCm
- Detailed explanation:
  ROCm-specific implementation with:
  - Masking for variable sequence lengths
  - Optimized for AMD wavefront size (64)
  - Vectorized for AMD memory system

- Code excerpt:
    ```cpp
    // From rocm/masked_silu_and_mul/mask_kernel.cu
    template<typename T>
    __global__ void maskedSiluAndMulKernel(
        T* output,
        const T* gate,
        const T* up,
        const int* seq_lens,
        int batch_size, int max_seq_len, int hidden_size) {
        
        int batch_idx = blockIdx.x;
        int seq_idx = blockIdx.y;
        int tid = threadIdx.x;
        
        // Check if within valid sequence length
        if (seq_idx >= seq_lens[batch_idx]) return;
        
        int offset = (batch_idx * max_seq_len + seq_idx) * hidden_size + tid;
        
        // Vectorized SiLU and mul
        using VecT = float4;  // 128-bit for AMD
        if (tid * 4 < hidden_size) {
            VecT g = *reinterpret_cast<const VecT*>(&gate[offset]);
            VecT u = *reinterpret_cast<const VecT*>(&up[offset]);
            VecT out;
            
            out.x = (g.x / (1.0f + expf(-g.x))) * u.x;
            out.y = (g.y / (1.0f + expf(-g.y))) * u.y;
            out.z = (g.z / (1.0f + expf(-g.z))) * u.z;
            out.w = (g.w / (1.0f + expf(-g.w))) * u.w;
            
            *reinterpret_cast<VecT*>(&output[offset]) = out;
        }
    }
    ```
- Evidence mapping:
  - Masking → `if (seq_idx >= seq_lens[batch_idx]) return`
  - AMD vectorization → `float4` for 128-bit access
  - Per-element SiLU → Explicit computation for each vector element
