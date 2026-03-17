# Kernel: LayerNorm and Activation Kernels

## Variant Context
- Input semantic type: Normalization and activation functions
- Datatype(s): fp16, bf16, fp32
- Data representation: Dense tensors
- Target architecture: Generic CUDA/HIP

## Functionality
These kernels implement:
1. RMSNorm (Root Mean Square Normalization) - used in LLaMA, Mistral, etc.
2. LayerNorm - used in GPT-2, BERT, etc.
3. Fused activation functions (SiLU, GELU, etc.)
4. Fused QK normalization with RoPE for attention

## Optimization 1: Vectorized RMSNorm
- Commit ID: (part of initial implementation)
- Optimization type: Memory / Compute
- Summary: Implement RMSNorm with vectorized memory access and warp-level reduction
- Detailed explanation:
  RMSNorm requires computing the root mean square of input values and then normalizing. This implementation uses vectorized loads, warp-level parallel reduction, and fused multiply-add operations for efficiency.
- Code excerpt:
    ```cpp
    template <typename scalar_t, int VEC_SIZE>
    __global__ void rms_norm_kernel(
        scalar_t* __restrict__ out,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ weight,
        float epsilon,
        int hidden_size) {
      
      using VecType = typename Vec<scalar_t, VEC_SIZE>::Type;
      
      const int token_idx = blockIdx.x;
      const scalar_t* token_input = input + token_idx * hidden_size;
      scalar_t* token_output = out + token_idx * hidden_size;
      
      // Compute sum of squares using vectorized loads
      float sum_sq = 0.0f;
      for (int i = threadIdx.x * VEC_SIZE; i < hidden_size; i += blockDim.x * VEC_SIZE) {
        VecType vec = *reinterpret_cast<const VecType*>(token_input + i);
        #pragma unroll
        for (int v = 0; v < VEC_SIZE; v++) {
          float val = static_cast<float>(vec[v]);
          sum_sq += val * val;
        }
      }
      
      // Warp-level reduction
      sum_sq = warp_reduce_sum(sum_sq);
      
      // Block-level reduction via shared memory
      __shared__ float shared_sum[32];
      if (threadIdx.x % 32 == 0) {
        shared_sum[threadIdx.x / 32] = sum_sq;
      }
      __syncthreads();
      
      if (threadIdx.x < 32) {
        sum_sq = shared_sum[threadIdx.x];
        sum_sq = warp_reduce_sum(sum_sq);
      }
      
      // Compute RMS and normalize
      __shared__ float rms_inv;
      if (threadIdx.x == 0) {
        rms_inv = rsqrtf(sum_sq / hidden_size + epsilon);
      }
      __syncthreads();
      
      // Vectorized output with weight multiplication
      for (int i = threadIdx.x * VEC_SIZE; i < hidden_size; i += blockDim.x * VEC_SIZE) {
        VecType in_vec = *reinterpret_cast<const VecType*>(token_input + i);
        VecType w_vec = *reinterpret_cast<const VecType*>(weight + i);
        VecType out_vec;
        #pragma unroll
        for (int v = 0; v < VEC_SIZE; v++) {
          out_vec[v] = static_cast<scalar_t>(
              static_cast<float>(in_vec[v]) * rms_inv * static_cast<float>(w_vec[v]));
        }
        *reinterpret_cast<VecType*>(token_output + i) = out_vec;
      }
    }
    ```
- Evidence mapping:
  - "Vectorized access" → `VecType` for multi-element loads/stores
  - "Warp reduction" → `warp_reduce_sum` for parallel sum
  - "Fused weight multiply" → Weight applied in same pass as normalization

## Optimization 2: Fused SiLU Activation
- Commit ID: (part of activation_kernels.cu)
- Optimization type: Fusion / Memory
- Summary: Fuse SiLU (Swish) activation with element-wise multiplication for MLP
- Detailed explanation:
  The SwiGLU activation used in LLaMA requires computing `silu(gate) * up`. This optimization fuses both operations into a single kernel, halving memory traffic.
- Code excerpt:
    ```cpp
    // Fused SiLU and multiply for SwiGLU
    template <typename scalar_t>
    __global__ void silu_and_mul_kernel(
        scalar_t* __restrict__ out,
        const scalar_t* __restrict__ input,  // [num_tokens, 2 * hidden_size]
        int hidden_size) {
      
      const int token_idx = blockIdx.x;
      const int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;
      
      if (dim_idx < hidden_size) {
        // Load gate and up projections
        float gate = static_cast<float>(input[token_idx * 2 * hidden_size + dim_idx]);
        float up = static_cast<float>(input[token_idx * 2 * hidden_size + hidden_size + dim_idx]);
        
        // SiLU: x * sigmoid(x)
        float silu_gate = gate / (1.0f + expf(-gate));
        
        // Fused multiply
        out[token_idx * hidden_size + dim_idx] = static_cast<scalar_t>(silu_gate * up);
      }
    }
    ```
- Evidence mapping:
  - "Fused operations" → SiLU and multiply in single kernel
  - "Reduced memory traffic" → Input read once, output written once
  - "SwiGLU pattern" → `silu(gate) * up` computation

## Optimization 3: Fused LayerNorm with Quantization
- Commit ID: (layernorm_quant_kernels.cu)
- Optimization type: Fusion
- Summary: Fuse LayerNorm with FP8 quantization for reduced memory traffic
- Detailed explanation:
  When using FP8 quantization, the output of LayerNorm needs to be quantized. This optimization fuses both operations, avoiding an intermediate FP16/BF16 tensor.
- Code excerpt:
    ```cpp
    // Fused LayerNorm + FP8 quantization
    template <typename scalar_t, typename out_t>
    __global__ void layernorm_quant_kernel(
        out_t* __restrict__ out,
        float* __restrict__ scale,  // Output scale for dequantization
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ weight,
        const scalar_t* __restrict__ bias,
        float epsilon,
        int hidden_size) {
      
      const int token_idx = blockIdx.x;
      
      // Compute mean and variance (standard LayerNorm)
      float mean = compute_mean(input + token_idx * hidden_size, hidden_size);
      float var = compute_variance(input + token_idx * hidden_size, mean, hidden_size);
      float inv_std = rsqrtf(var + epsilon);
      
      // Find max for quantization scale
      float max_val = 0.0f;
      for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (input[token_idx * hidden_size + i] - mean) * inv_std;
        float weighted = normalized * weight[i] + bias[i];
        max_val = fmaxf(max_val, fabsf(weighted));
      }
      max_val = block_reduce_max(max_val);
      
      // Compute and store scale
      float token_scale = max_val / FP8_MAX;
      if (threadIdx.x == 0) {
        scale[token_idx] = token_scale;
      }
      __syncthreads();
      
      // Normalize, weight, and quantize in one pass
      for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (input[token_idx * hidden_size + i] - mean) * inv_std;
        float weighted = normalized * weight[i] + bias[i];
        out[token_idx * hidden_size + i] = float_to_fp8(weighted / token_scale);
      }
    }
    ```
- Evidence mapping:
  - "Fused quantization" → FP8 conversion in same kernel as LayerNorm
  - "Dynamic scaling" → Per-token scale computed from max value
  - "Single pass output" → No intermediate FP16 tensor

## Optimization 4: Fused QK Normalization with RoPE
- Commit ID: (fused_qknorm_rope_kernel.cu)
- Optimization type: Fusion
- Summary: Fuse query/key normalization with rotary position embedding
- Detailed explanation:
  Some models (like Gemma) apply normalization to Q and K before RoPE. This optimization fuses both operations to reduce memory traffic and kernel launch overhead.
- Code excerpt:
    ```cpp
    // Fused QK norm + RoPE
    template <typename scalar_t>
    __global__ void fused_qknorm_rope_kernel(
        scalar_t* __restrict__ query,   // [num_tokens, num_heads, head_size]
        scalar_t* __restrict__ key,     // [num_tokens, num_kv_heads, head_size]
        const scalar_t* __restrict__ q_norm_weight,
        const scalar_t* __restrict__ k_norm_weight,
        const int64_t* __restrict__ positions,
        float rope_theta,
        int head_size,
        int rotary_dim) {
      
      const int token_idx = blockIdx.x;
      const int head_idx = blockIdx.y;
      const int64_t position = positions[token_idx];
      
      // Load Q/K for this head
      scalar_t* q_head = query + token_idx * num_heads * head_size + head_idx * head_size;
      scalar_t* k_head = key + token_idx * num_kv_heads * head_size + head_idx * head_size;
      
      // Compute RMS for normalization
      float q_rms = compute_rms(q_head, head_size);
      float k_rms = compute_rms(k_head, head_size);
      
      // Apply normalization and RoPE together
      for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
        // Normalize
        float q_val = static_cast<float>(q_head[d]) / q_rms * q_norm_weight[d];
        float k_val = static_cast<float>(k_head[d]) / k_rms * k_norm_weight[d];
        
        // Apply RoPE to rotary dimensions
        if (d < rotary_dim) {
          int half_dim = rotary_dim / 2;
          float freq = 1.0f / powf(rope_theta, (d % half_dim) * 2.0f / rotary_dim);
          float angle = position * freq;
          float cos_val = cosf(angle);
          float sin_val = sinf(angle);
          
          if (d < half_dim) {
            // First half: x * cos - y * sin
            float q_other = static_cast<float>(q_head[d + half_dim]) / q_rms * q_norm_weight[d + half_dim];
            q_val = q_val * cos_val - q_other * sin_val;
            // Similar for k
          } else {
            // Second half: x * sin + y * cos
            // ...
          }
        }
        
        q_head[d] = static_cast<scalar_t>(q_val);
        k_head[d] = static_cast<scalar_t>(k_val);
      }
    }
    ```
- Evidence mapping:
  - "Fused operations" → Norm and RoPE in single kernel
  - "In-place update" → Q and K modified directly
  - "Rotary embedding" → Position-dependent rotation applied

## Optimization 5: Dynamic Per-Token Quantization
- Commit ID: (fused_kernels/fused_layernorm_dynamic_per_token_quant.cu)
- Optimization type: Fusion / Precision
- Summary: Fuse LayerNorm with dynamic per-token FP8 quantization
- Detailed explanation:
  Dynamic quantization computes the scale factor at runtime based on actual values. This optimization fuses the scale computation with LayerNorm for maximum efficiency.
- Code excerpt:
    ```cpp
    // Dynamic per-token quantization with LayerNorm
    template <typename scalar_t>
    __global__ void fused_layernorm_dynamic_quant_kernel(
        __nv_fp8_e4m3* __restrict__ out,
        float* __restrict__ scales,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ weight,
        const scalar_t* __restrict__ bias,
        float epsilon,
        int hidden_size) {
      
      extern __shared__ float shared_mem[];
      float* normalized = shared_mem;  // Temporary storage
      
      const int token_idx = blockIdx.x;
      
      // Pass 1: Compute LayerNorm and find max
      float mean = 0.0f, var = 0.0f, max_abs = 0.0f;
      
      // Welford's online algorithm for mean/variance
      for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(input[token_idx * hidden_size + i]);
        // Update mean and variance...
      }
      
      // Reduce mean and variance across threads
      mean = block_reduce_sum(mean) / hidden_size;
      var = block_reduce_sum(var) / hidden_size;
      float inv_std = rsqrtf(var + epsilon);
      
      // Normalize and find max for quantization
      for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(input[token_idx * hidden_size + i]);
        float norm_val = (val - mean) * inv_std;
        float weighted = norm_val * weight[i] + bias[i];
        normalized[i] = weighted;
        max_abs = fmaxf(max_abs, fabsf(weighted));
      }
      max_abs = block_reduce_max(max_abs);
      
      // Compute scale
      float scale = max_abs / 448.0f;  // FP8 E4M3 max
      if (threadIdx.x == 0) {
        scales[token_idx] = scale;
      }
      __syncthreads();
      
      // Pass 2: Quantize
      for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out[token_idx * hidden_size + i] = __nv_cvt_float_to_fp8(
            normalized[i] / scale, __NV_SATFINITE, __NV_E4M3);
      }
    }
    ```
- Evidence mapping:
  - "Dynamic scaling" → Scale computed from actual max value
  - "Two-pass algorithm" → First pass normalizes, second quantizes
  - "Shared memory" → Temporary storage avoids extra global memory
