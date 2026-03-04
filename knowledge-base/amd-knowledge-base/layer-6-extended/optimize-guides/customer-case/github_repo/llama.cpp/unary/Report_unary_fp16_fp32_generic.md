---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: Unary Operations

## Variant Context
- Input semantic type: Element-wise unary operations
- Datatype(s): FP16, FP32
- Data representation: Dense tensors
- Target architecture: Generic (NVIDIA, AMD, Moore Threads)

## Functionality
The unary kernel implements various element-wise activation functions and mathematical operations used in neural networks. These include GELU, SiLU (Swish), ReLU, Tanh, Sigmoid, and many others. The kernel is optimized for memory bandwidth as these operations are typically memory-bound.

Key features:
- Support for 20+ activation functions
- FP16 and FP32 computation
- Fused GLU variants (GEGLU, SWIGLU, REGLU)
- Vectorized memory access

---

## Optimization 1: GLU Fusion (GEGLU, SWIGLU, REGLU)
- Commit ID: a0535ffa0, 28657a822
- Optimization type: Fusion (kernel fusion)
- Summary: Implement fused Gated Linear Unit variants to reduce memory traffic
- Detailed explanation: GLU operations combine an activation function with element-wise gating. By fusing these operations, we avoid writing intermediate results to memory. GEGLU uses GELU activation, SWIGLU uses SiLU, and REGLU uses ReLU.

- Code excerpt:
    ```cpp
    // ggml: implement REGLU/GEGLU/SWIGLU ops
    template<typename T, typename ACT_FUNC>
    __global__ void glu_fused(
        const T * __restrict__ x,      // Input [batch, 2*dim]
        T * __restrict__ dst,          // Output [batch, dim]
        const int dim) {
        
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= dim) return;
        
        const int batch = blockIdx.y;
        const T * x_batch = x + batch * 2 * dim;
        
        // First half: gate, Second half: value
        const float gate = ACT_FUNC::apply((float)x_batch[idx]);
        const float value = (float)x_batch[idx + dim];
        
        dst[batch * dim + idx] = (T)(gate * value);
    }
    
    // Activation functions
    struct GELU_ACT {
        static __device__ float apply(float x) {
            return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        }
    };
    
    struct SILU_ACT {
        static __device__ float apply(float x) {
            return x / (1.0f + expf(-x));
        }
    };
    
    struct RELU_ACT {
        static __device__ float apply(float x) {
            return fmaxf(0.0f, x);
        }
    };
    ```

- Evidence mapping:
  - "Fused GLU" → gate and value processed in single kernel
  - "Multiple variants" → template with ACT_FUNC parameter
  - "Reduced memory" → no intermediate tensor for gate output

---

## Optimization 2: GELU ERF Variant
- Commit ID: 4c32832c5
- Optimization type: Precision (accuracy)
- Summary: Add GELU implementation using error function for higher accuracy
- Detailed explanation: The standard GELU approximation using tanh can have small numerical differences from the exact definition using erf. This optimization adds the exact erf-based GELU for models that require higher precision.

- Code excerpt:
    ```cpp
    // ggml: add ggml_gelu_erf() CUDA kernel
    __device__ __forceinline__ float gelu_erf(float x) {
        // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
    }
    
    template<typename T>
    __global__ void gelu_erf_kernel(
        const T * __restrict__ x,
        T * __restrict__ dst,
        const int n) {
        
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        
        dst[idx] = (T)gelu_erf((float)x[idx]);
    }
    ```

- Evidence mapping:
  - "ERF-based" → `erff()` function call
  - "Exact definition" → matches PyTorch's GELU exactly
  - "Higher precision" → no tanh approximation error

---

## Optimization 3: FP16 Unary Operations
- Commit ID: 87abb7e90
- Optimization type: Precision (FP16 support)
- Summary: Increase support for FP16 unary operations with proper handling
- Detailed explanation: FP16 operations need careful handling to avoid precision loss. This optimization adds proper FP16 support for unary operations, computing in FP32 internally when needed but keeping data in FP16 for memory efficiency.

- Code excerpt:
    ```cpp
    // cuda/cpu: Increase support for fp16 unary operations
    template<typename T, typename FUNC>
    __global__ void unary_fp16_safe(
        const T * __restrict__ x,
        T * __restrict__ dst,
        const int n) {
        
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        
        // Convert to FP32 for computation
        float val = (float)x[idx];
        
        // Apply function in FP32
        val = FUNC::apply(val);
        
        // Convert back to FP16
        dst[idx] = (T)val;
    }
    
    // Vectorized version for better bandwidth
    template<typename FUNC>
    __global__ void unary_fp16_vec2(
        const half2 * __restrict__ x,
        half2 * __restrict__ dst,
        const int n) {
        
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n/2) return;
        
        half2 val = x[idx];
        float2 val_f = __half22float2(val);
        
        val_f.x = FUNC::apply(val_f.x);
        val_f.y = FUNC::apply(val_f.y);
        
        dst[idx] = __float22half2_rn(val_f);
    }
    ```

- Evidence mapping:
  - "FP32 computation" → convert, compute, convert back
  - "Vectorized" → `half2` for 2x memory bandwidth
  - "Safe conversion" → `__half22float2`, `__float22half2_rn`

---

## Optimization 4: Float Computation with Deduplication
- Commit ID: b64d7cc27
- Optimization type: Code quality (deduplication)
- Summary: Refactor unary ops to use float computation and deduplicate code
- Detailed explanation: Many unary operations share similar patterns. This optimization refactors the code to use a common template structure, reducing code duplication while ensuring all operations use FP32 computation for accuracy.

- Code excerpt:
    ```cpp
    // cuda: unary ops as float + de-duplicate
    
    // Common unary operation template
    template<typename T, float (*FUNC)(float)>
    __global__ void unary_op(
        const T * __restrict__ x,
        T * __restrict__ dst,
        const int n) {
        
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        
        dst[idx] = (T)FUNC((float)x[idx]);
    }
    
    // Function definitions
    __device__ float op_gelu(float x) { return gelu_impl(x); }
    __device__ float op_silu(float x) { return x / (1.0f + expf(-x)); }
    __device__ float op_relu(float x) { return fmaxf(0.0f, x); }
    __device__ float op_tanh(float x) { return tanhf(x); }
    __device__ float op_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
    
    // Instantiate for each type
    template __global__ void unary_op<float, op_gelu>(...);
    template __global__ void unary_op<half, op_gelu>(...);
    ```

- Evidence mapping:
  - "Common template" → single `unary_op` template
  - "Function pointer" → `float (*FUNC)(float)` parameter
  - "Deduplication" → one implementation for all ops

---

## Optimization 5: Additional Unary Operations
- Commit ID: 7db35a795, 389ac78b2
- Optimization type: Algorithm (new operations)
- Summary: Add FLOOR, CEIL, ROUND, TRUNC, SOFTPLUS, EXPM1 operations
- Detailed explanation: Various models require additional mathematical operations. This optimization adds support for rounding operations and other mathematical functions needed by newer model architectures.

- Code excerpt:
    ```cpp
    // CUDA: add FLOOR, CEIL, ROUND, TRUNC unary ops
    __device__ float op_floor(float x) { return floorf(x); }
    __device__ float op_ceil(float x) { return ceilf(x); }
    __device__ float op_round(float x) { return roundf(x); }
    __device__ float op_trunc(float x) { return truncf(x); }
    
    // ggml: add ops SOFTPLUS, EXPM1
    __device__ float op_softplus(float x) {
        // softplus(x) = log(1 + exp(x))
        // Numerically stable version
        if (x > 20.0f) return x;
        return logf(1.0f + expf(x));
    }
    
    __device__ float op_expm1(float x) {
        // expm1(x) = exp(x) - 1
        // More accurate than exp(x) - 1 for small x
        return expm1f(x);
    }
    ```

- Evidence mapping:
  - "Rounding ops" → `floorf`, `ceilf`, `roundf`, `truncf`
  - "Numerical stability" → softplus with overflow check
  - "Accuracy" → `expm1f` for small x values

---

## Optimization 6: ELU Activation Support
- Commit ID: e743cddb6
- Optimization type: Algorithm (new activation)
- Summary: Add ELU (Exponential Linear Unit) activation support
- Detailed explanation: ELU is an activation function that can produce negative outputs, helping with the vanishing gradient problem. This optimization adds CUDA support for ELU with configurable alpha parameter.

- Code excerpt:
    ```cpp
    // cuda: add ELU support
    __device__ float op_elu(float x, float alpha) {
        // ELU(x) = x if x > 0, else alpha * (exp(x) - 1)
        return x > 0.0f ? x : alpha * (expf(x) - 1.0f);
    }
    
    template<typename T>
    __global__ void elu_kernel(
        const T * __restrict__ x,
        T * __restrict__ dst,
        const float alpha,
        const int n) {
        
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        
        dst[idx] = (T)op_elu((float)x[idx], alpha);
    }
    ```

- Evidence mapping:
  - "ELU function" → conditional with exponential
  - "Configurable alpha" → parameter for negative slope
  - "Negative outputs" → `alpha * (exp(x) - 1)` for x < 0
