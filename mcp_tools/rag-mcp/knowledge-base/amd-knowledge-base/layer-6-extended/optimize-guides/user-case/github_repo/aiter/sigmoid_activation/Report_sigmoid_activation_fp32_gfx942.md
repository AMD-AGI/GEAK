# Kernel: sigmoid_activation

## Variant Context
- Input semantic type: Element-wise activation function
- Datatype(s): FP32 (internal computation), FP16/BF16 (input/output)
- Data representation: Dense tensor
- Target architecture: gfx942 (MI300X), gfx950 (MI350)

## Functionality
This kernel computes the sigmoid activation function: sigmoid(x) = 1 / (1 + exp(-x)). It is used in various neural network layers including attention mechanisms and gating functions.

## Optimization 1: AMD Fast Math Intrinsics
- Commit ID: daa23b1f9
- Optimization type: Compute
- Summary: Replaced standard math functions with AMD-specific fast math intrinsics, achieving 20% average speedup on MI300X.

- Detailed explanation:
  The optimization replaces the standard `expf()` and division operations with AMD GPU-specific intrinsics:
  1. `__builtin_amdgcn_exp2f()` - Fast exp2 computation using native hardware
  2. `__builtin_amdgcn_rcpf()` - Fast reciprocal using native hardware
  
  The mathematical transformation used:
  - exp(x) = exp2(x × log2(e)) where log2(e) ≈ 1.442695
  - sigmoid(x) = 1 / (1 + exp(-x)) = rcp(1 + exp2(-x × log2(e)))
  
  These intrinsics map directly to single AMD GPU instructions (v_exp_f32, v_rcp_f32), avoiding the overhead of software implementations.

- Code excerpt:
    ```cpp
    // BEFORE: Standard math functions
    template <typename T>
    inline __device__ static T apply(T x)
    {
        return static_cast<T>(1.0f / (1.0f + expf(static_cast<float>(-x))));
    }
    
    // AFTER: AMD fast math intrinsics
    template <typename T>
    inline __device__ static T apply(T x)
    {
        // Use AMD fast math intrinsics for better performance
        // sigmoid(x) = 1 / (1 + exp(-x))
        // exp(x) = exp2(x * log2(e)) where log2(e) ≈ 1.442695
        float neg_x = static_cast<float>(-x);
        constexpr float LOG2E = 1.442695040888963407359924681001892137426645954152985934135449406931f;

        // Use __builtin_amdgcn_exp2f for fast exp2 computation
        float exp_val = __builtin_amdgcn_exp2f(neg_x * LOG2E);
        float denom = 1.0f + exp_val;

        // Use __builtin_amdgcn_rcpf for fast reciprocal
        float result = __builtin_amdgcn_rcpf(denom);

        return static_cast<T>(result);
    }
    ```

- Evidence mapping:
  - Fast exp2 → `__builtin_amdgcn_exp2f(neg_x * LOG2E)` maps to v_exp_f32 instruction
  - Fast reciprocal → `__builtin_amdgcn_rcpf(denom)` maps to v_rcp_f32 instruction
  - Mathematical equivalence → `exp(x) = exp2(x * log2(e))` transformation
  - 20% speedup → Commit message states "20% average speedup on MI300X"
