# Kernel: Fused MoE (FP8 Quantized)

## Variant Context
- Input semantic type: MoE GEMM (Token routing and expert computation with FP8 quantization)
- Datatype(s): fp8_e4m3 (weights and/or activations)
- Data representation: FP8 quantized expert weights with per-tensor or block-wise scaling
- Target architecture: CUDA sm89+ (Ada/Hopper), HIP gfx90a+ (MI200/MI300)

## Functionality
The FP8 Fused MoE kernel extends the base MoE implementation to support FP8 quantization for reduced memory bandwidth and increased throughput. It supports:
1. FP8 weights with FP16/BF16 activations (W8A16)
2. FP8 weights and activations (W8A8)
3. Per-tensor and block-wise scaling for accuracy

## Optimization 1: Initial FP8 MoE Support with Dynamic Scaling
- Commit ID: eace8bf0b
- Optimization type: Memory / Compute
- Summary: Add FP8 support for MoE kernel with dynamic per-tensor scaling
- Detailed explanation:
  This optimization enables FP8 quantization for Mixtral and similar MoE models. Dynamic scaling computes the scale factor at runtime based on the actual tensor values, eliminating the need for calibration datasets while maintaining accuracy.
- Code excerpt:
    ```python
    @triton.jit
    def fused_moe_kernel_fp8(
        a_ptr, b_ptr, c_ptr,
        a_scale_ptr, b_scale_ptr,  # FP8 scaling factors
        topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
        ...,
        use_fp8: tl.constexpr,
    ):
        # Load FP8 weights and dequantize
        if use_fp8:
            b_scale = tl.load(b_scale_ptr + off_experts)
            b = tl.load(b_ptrs, ...).to(tl.float32) * b_scale
        else:
            b = tl.load(b_ptrs, ...)
        
        # Accumulate in FP32 for accuracy
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, ...)
            if use_fp8:
                a_scale = tl.load(a_scale_ptr)
                a = a.to(tl.float32) * a_scale
            accumulator += tl.dot(a, b)
    ```
- Evidence mapping:
  - "FP8 support" → `use_fp8` flag and scale pointer parameters
  - "Dynamic scaling" → Scales loaded from tensors, not hardcoded
  - "FP32 accumulation" → `dtype=tl.float32` for numerical accuracy

## Optimization 2: Static FP8 Scales for Performance
- Commit ID: 12628d3c7
- Optimization type: Compute
- Summary: Optimize FP8 MoE with static (pre-computed) scaling factors
- Detailed explanation:
  While dynamic scaling is flexible, static scales computed during model loading can be faster as they avoid runtime scale computation. This optimization supports pre-computed scales from quantized model checkpoints.
- Code excerpt:
    ```python
    def fused_moe_fp8(hidden_states, w1, w2, gating_output, topk,
                      w1_scale, w2_scale, a_scale=None):
        """
        FP8 fused MoE with static scales.
        
        Args:
            w1_scale: Pre-computed scale for gate/up projection weights
            w2_scale: Pre-computed scale for down projection weights
            a_scale: Optional activation scale (None for dynamic)
        """
        if a_scale is None:
            # Dynamic activation scaling
            a_scale = hidden_states.abs().max() / FP8_MAX
            hidden_states_fp8 = (hidden_states / a_scale).to(torch.float8_e4m3fn)
        else:
            hidden_states_fp8 = (hidden_states / a_scale).to(torch.float8_e4m3fn)
        
        # Launch kernel with static weight scales
        invoke_fused_moe_kernel(
            hidden_states_fp8, w1, w2, 
            a_scale=a_scale, w1_scale=w1_scale, w2_scale=w2_scale,
            ...)
    ```
- Evidence mapping:
  - "Static scales" → `w1_scale`, `w2_scale` passed as arguments
  - "Optional dynamic" → `a_scale=None` triggers runtime computation
  - "Pre-quantized weights" → Weights already in FP8 format

## Optimization 3: Block-wise FP8 Quantization for DeepSeek
- Commit ID: 2072924d1, 9b0c4bab3
- Optimization type: Precision
- Summary: Add block-wise FP8 quantization support for DeepSeek-V3 models
- Detailed explanation:
  DeepSeek-V3 uses block-wise quantization where different blocks of the weight matrix have different scales. This provides better accuracy than per-tensor scaling by adapting to local value distributions.
- Code excerpt:
    ```python
    @triton.jit
    def fused_moe_kernel_fp8_blockwise(
        a_ptr, b_ptr, c_ptr,
        a_scale_ptr, b_scale_ptr,
        ...,
        BLOCK_K: tl.constexpr,
        scale_block_k: tl.constexpr,  # Block size for scaling
    ):
        # Each K-block has its own scale
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            # Load block-specific scale
            scale_idx = k // (scale_block_k // BLOCK_K)
            b_scale = tl.load(b_scale_ptr + off_experts * num_scale_blocks + scale_idx)
            
            # Load and dequantize with block scale
            b = tl.load(b_ptrs, ...).to(tl.float32) * b_scale
            a = tl.load(a_ptrs, ...)
            
            accumulator += tl.dot(a, b)
            b_ptrs += BLOCK_K * stride_bk
    ```
- Evidence mapping:
  - "Block-wise scaling" → `scale_block_k` parameter defines scale granularity
  - "Per-block scale load" → Scale indexed by `k // (scale_block_k // BLOCK_K)`
  - "DeepSeek support" → Matches DeepSeek-V3 quantization format

## Optimization 4: Triton Configs for FP8 Block Quantization
- Commit ID: 9b0c4bab3
- Optimization type: Launch Configuration
- Summary: Add optimized Triton configurations for FP8 block-quantized MoE
- Detailed explanation:
  FP8 block quantization requires different tile sizes than per-tensor quantization due to the additional scale loads. This optimization provides tuned configurations that balance compute efficiency with scale loading overhead.
- Code excerpt:
    ```python
    # Tuned config for FP8 block quantization on H100
    FP8_BLOCK_QUANT_CONFIGS = {
        "E=256,N=7168": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,  # Larger K to amortize scale loads
            "GROUP_SIZE_M": 8,
            "num_warps": 8,
            "num_stages": 3,
        },
        # ... more configs
    }
    
    def get_fp8_block_config(E, N, block_k):
        """Select config based on problem size and block quantization params."""
        key = f"E={E},N={N}"
        if key in FP8_BLOCK_QUANT_CONFIGS:
            config = FP8_BLOCK_QUANT_CONFIGS[key].copy()
            # Adjust for block size
            config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], block_k)
            return config
        return None
    ```
- Evidence mapping:
  - "FP8-specific configs" → Separate `FP8_BLOCK_QUANT_CONFIGS` dictionary
  - "Larger K blocks" → `BLOCK_SIZE_K: 128` to amortize scale overhead
  - "Block size alignment" → Config adjusted based on `block_k` parameter

## Optimization 5: GPTQ/AWQ FP8 MoE Support
- Commit ID: 27b78c73c
- Optimization type: Compute
- Summary: Add Triton fused MoE kernel support for GPTQ and AWQ quantized models
- Detailed explanation:
  This optimization extends the fused MoE kernel to support GPTQ and AWQ quantization formats, which use different packing and scaling schemes than standard FP8. The kernel handles unpacking and dequantization inline.
- Code excerpt:
    ```python
    @triton.jit
    def fused_moe_kernel_gptq_awq(
        a_ptr, b_ptr, c_ptr,
        b_scale_ptr, b_zp_ptr,  # Scales and zero points
        ...,
        group_size: tl.constexpr,
        has_zp: tl.constexpr,
        use_int4_w4a16: tl.constexpr,
        use_int8_w8a16: tl.constexpr,
    ):
        # Load quantized weights
        b_packed = tl.load(b_ptrs, ...)
        
        # Unpack based on quantization type
        if use_int4_w4a16:
            # Unpack INT4 from packed format
            b = unpack_int4(b_packed)
        elif use_int8_w8a16:
            b = b_packed.to(tl.float32)
        
        # Apply scale and zero point
        scale_idx = k // group_size
        b_scale = tl.load(b_scale_ptr + scale_idx)
        if has_zp:
            b_zp = tl.load(b_zp_ptr + scale_idx)
            b = (b - b_zp) * b_scale
        else:
            b = b * b_scale
    ```
- Evidence mapping:
  - "GPTQ/AWQ support" → `b_zp_ptr` for zero points, `group_size` for group quantization
  - "INT4 unpacking" → `use_int4_w4a16` flag and `unpack_int4` function
  - "Group-wise scaling" → Scale indexed by `k // group_size`

## Optimization 6: Float Cast and Renormalize Fusion
- Commit ID: 75c7ad991
- Optimization type: Fusion
- Summary: Fuse float cast and renormalization into topk softmax kernel
- Detailed explanation:
  When using FP8, the gating scores need to be cast to float and renormalized. This optimization fuses these operations into the topk softmax kernel, reducing memory traffic.
- Code excerpt:
    ```python
    @triton.jit
    def topk_softmax_kernel_fused(
        topk_weights_ptr, topk_ids_ptr,
        gating_output_ptr,  # May be FP8 or FP16
        ...,
        input_dtype: tl.constexpr,
        renormalize: tl.constexpr,
    ):
        # Load gating scores with type conversion
        if input_dtype == tl.float8e4nv:
            scores = tl.load(gating_output_ptr + ...).to(tl.float32)
        else:
            scores = tl.load(gating_output_ptr + ...)
        
        # Compute softmax over all experts
        max_score = tl.max(scores, axis=0)
        exp_scores = tl.exp(scores - max_score)
        sum_exp = tl.sum(exp_scores, axis=0)
        
        # Select top-k and optionally renormalize
        # ... top-k selection ...
        
        if renormalize:
            # Renormalize selected weights to sum to 1
            topk_sum = tl.sum(topk_weights, axis=0)
            topk_weights = topk_weights / topk_sum
        
        tl.store(topk_weights_ptr + ..., topk_weights)
    ```
- Evidence mapping:
  - "Fused cast" → `input_dtype` parameter handles FP8 input
  - "Fused renormalize" → `renormalize` flag enables in-kernel normalization
  - "Single kernel" → All operations in one kernel launch
