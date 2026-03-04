---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: DeepSeek V3 Specialized Kernels

## Variant Context
- Input semantic type: DeepSeek V3 model-specific operations (router GEMM, fused activation GEMM)
- Datatype(s): BF16, FP32, FP8
- Data representation: MoE routing logits, expert activations
- Target architecture: SM90 (Hopper) and later

## Functionality
These kernels implement DeepSeek V3 model-specific optimizations:
1. Router GEMM: Efficient computation of routing logits for MoE expert selection
2. Fused A GEMM: Fused activation and GEMM for expert computation
3. Optimized for DeepSeek V3's specific MoE architecture with 256 experts

## Optimization 1: DeepSeek V3 Router GEMM with BF16/Float Output
- Commit ID: 04b35190e (dsv3_router_gemm)
- Optimization type: Compute / Precision
- Summary: Specialized GEMM kernel for computing MoE routing logits with configurable output precision.

- Detailed explanation:
  DeepSeek V3 uses a large number of experts (256), making the router computation a significant portion of the forward pass. This kernel:
  1. Computes routing logits: logits = hidden_states @ router_weight.T
  2. Supports both BF16 and FP32 output for flexibility in routing precision
  3. Optimized tile sizes for the specific dimensions of DeepSeek V3's router

- Code excerpt:
    ```cpp
    // Router GEMM with BF16 output
    // sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu
    template <typename Config>
    __global__ void dsv3_router_gemm_bf16_kernel(
        const __nv_bfloat16* hidden_states,  // [batch, hidden_dim]
        const __nv_bfloat16* router_weight,  // [num_experts, hidden_dim]
        __nv_bfloat16* output,               // [batch, num_experts]
        int batch_size,
        int hidden_dim,
        int num_experts
    ) {
        // Optimized for num_experts = 256
        // Uses tensor cores for BF16 GEMM
        // ...
    }

    // Router GEMM with FP32 output for higher precision routing
    // sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu
    template <typename Config>
    __global__ void dsv3_router_gemm_float_kernel(
        const __nv_bfloat16* hidden_states,
        const __nv_bfloat16* router_weight,
        float* output,  // FP32 output for precise softmax
        int batch_size,
        int hidden_dim,
        int num_experts
    ) {
        // ...
    }
    ```

- Evidence mapping:
  - "Dual precision support" → separate `bf16_out` and `float_out` kernels
  - "256 experts optimization" → tile sizes tuned for num_experts=256
  - "Router-specific" → input/output shapes match router computation

## Optimization 2: Fused Activation GEMM for Expert Computation
- Commit ID: 04b35190e
- Optimization type: Fusion
- Summary: Fuses the activation function (SiLU) with the GEMM operation for expert gate-up projection.

- Detailed explanation:
  In DeepSeek V3's MoE layer, each expert computes:
  1. gate = hidden @ W_gate
  2. up = hidden @ W_up
  3. output = SiLU(gate) * up
  
  This kernel fuses the gate GEMM with SiLU activation, reducing memory traffic by avoiding materialization of the gate output.

- Code excerpt:
    ```cpp
    // sgl-kernel/csrc/gemm/dsv3_fused_a_gemm.cu
    template <typename Config>
    __global__ void dsv3_fused_a_gemm_kernel(
        const __nv_bfloat16* input,
        const __nv_bfloat16* weight_gate,
        const __nv_bfloat16* weight_up,
        __nv_bfloat16* output,
        int M, int N, int K
    ) {
        // Compute gate and up projections
        // Fuse SiLU activation with gate
        // Multiply and store
        
        // Tile computation
        float gate_acc[TILE_M][TILE_N] = {0};
        float up_acc[TILE_M][TILE_N] = {0};
        
        // GEMM loop
        for (int k = 0; k < K; k += TILE_K) {
            // Load input tile
            // Compute gate_acc += input @ weight_gate
            // Compute up_acc += input @ weight_up
        }
        
        // Fused activation and multiply
        for (int i = 0; i < TILE_M; i++) {
            for (int j = 0; j < TILE_N; j++) {
                float gate = gate_acc[i][j];
                float up = up_acc[i][j];
                // SiLU: x * sigmoid(x)
                float silu_gate = gate * (1.0f / (1.0f + expf(-gate)));
                output[...] = __float2bfloat16(silu_gate * up);
            }
        }
    }
    ```

- Evidence mapping:
  - "Fused computation" → gate and up computed in same kernel
  - "SiLU fusion" → `silu_gate = gate * (1.0f / (1.0f + expf(-gate)))` applied inline
  - "No intermediate storage" → gate_acc used directly, not stored to global memory

## Optimization 3: Optimized Tile Sizes for DeepSeek V3 Dimensions
- Commit ID: 04b35190e
- Optimization type: Launch Configuration
- Summary: Uses tile sizes specifically tuned for DeepSeek V3's hidden dimension (7168) and intermediate size.

- Detailed explanation:
  DeepSeek V3 has specific dimensions:
  - Hidden dimension: 7168
  - Intermediate size: 18432 (per expert)
  - Number of experts: 256
  
  The kernel uses tile sizes that divide evenly into these dimensions for optimal performance.

- Code excerpt:
    ```cpp
    // Tile configuration for DeepSeek V3
    struct DSV3Config {
        static constexpr int TILE_M = 128;
        static constexpr int TILE_N = 128;
        static constexpr int TILE_K = 32;
        
        // 7168 / 32 = 224 (even division)
        // 18432 / 128 = 144 (even division)
    };
    ```

- Evidence mapping:
  - "Dimension-specific tiles" → TILE_K=32 divides 7168 evenly
  - "Intermediate size alignment" → TILE_N=128 divides 18432 evenly

## Optimization 4: Shared Expert Append and Flatten Quant for AMD
- Commit ID: eff7df6d0
- Optimization type: Fusion / AMD-specific
- Summary: Enables fused shared expert append and flatten quantization for FP8 DeepSeek R1 on AMD GPUs.

- Detailed explanation:
  DeepSeek models use shared experts that are always activated alongside the routed experts. This optimization fuses:
  1. Appending shared expert outputs to routed expert outputs
  2. Flattening the expert dimension
  3. Quantizing to FP8 for the next layer
  
  This is particularly optimized for AMD MI300X GPUs running DeepSeek R1.

- Code excerpt:
    ```python
    # AMD-specific optimization for DeepSeek
    if is_hip() and use_fp8:
        # Fused shared expert append + flatten + quant
        output = fused_shared_expert_append_flatten_quant(
            routed_output,      # [batch, num_routed_experts, hidden]
            shared_output,      # [batch, num_shared_experts, hidden]
            expert_weights,     # [batch, num_experts]
        )
    ```

- Evidence mapping:
  - "AMD-specific" → `if is_hip()` condition
  - "Fused operations" → single function call for append+flatten+quant
  - "FP8 output" → quantization integrated into the fusion
