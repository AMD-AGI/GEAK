# Kernel: moe_gemm_a8w8_blockscale

## Variant Context
- Input semantic type: Mixture of Experts (MoE) GEMM
- Datatype(s): FP8 (E4M3) activation and weight with block-wise scaling
- Data representation: Block-wise quantized with per-block scales
- Target architecture: gfx942 (MI300), gfx950 (MI350)

## Functionality
This kernel implements FP8 GEMM for Mixture of Experts layers with block-wise scaling. It handles:
1. Expert-parallel matrix multiplication
2. Token routing and gathering
3. Block-wise dequantization during computation
4. Optional SwiGLU activation fusion
5. Optional output quantization

## Optimization 1: XCD Swizzle for Load Balancing
- Commit ID: f600a109b
- Optimization type: Scheduling
- Summary: Added XCD (Accelerator Complex Die) swizzle to reorder block assignments for better memory access patterns.

- Detailed explanation:
  The XCD swizzle reorders how thread blocks are assigned to hardware units. Instead of sequential assignment (blocks 0,1,2,3... to units 0,1,2,3...), it groups consecutive blocks to the same unit. This improves:
  1. L2 cache locality within each XCD
  2. Reduced cross-XCD memory traffic
  3. Better workload distribution

- Code excerpt:
    ```python
    @triton.jit
    def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: tl.constexpr):
        """
        Swizzle the program id based on integer XCD_SWIZZLE.
        This is useful for reording how blocks are ordered.
        """
        # Number of pids per group in the new arrangement
        pids_per_group = domain_size // XCD_SWIZZLE
        extra_pid_groups = domain_size % XCD_SWIZZLE

        # Compute current current and local pid within the group
        group = pid % XCD_SWIZZLE
        local_pid = pid // XCD_SWIZZLE

        # Calculate new pid based on the new grouping
        new_pid = group * pids_per_group + min(group, extra_pid_groups) + local_pid
        return new_pid
    ```

- Evidence mapping:
  - XCD swizzle function → `xcd_swizzle()` with configurable `XCD_SWIZZLE` parameter
  - Block reordering → `new_pid` calculation groups consecutive blocks
  - Commit message → "Add XCD swizzle to a8w8 blockscale"

## Optimization 2: Fused SwiGLU Activation
- Commit ID: f600a109b
- Optimization type: Fusion
- Summary: Fused SwiGLU activation into the GEMM kernel to avoid separate kernel launch and memory traffic.

- Detailed explanation:
  SwiGLU is a gated activation function used in modern LLMs. The kernel fuses it directly into the GEMM output:
  1. Splits output into gelu and linear components
  2. Applies sigmoid-weighted gating
  3. Combines with linear component
  
  This avoids writing intermediate results to global memory.

- Code excerpt:
    ```python
    @triton.jit
    def _swiglu(input, alpha, limit):
        gelu, linear = tl.split(tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
        gelu = gelu.to(tl.float32)
        if limit is not None:
            gelu = clip(gelu, limit, clip_lower=False)
        linear = linear.to(tl.float32)
        if limit is not None:
            linear = clip(linear, limit, clip_lower=True)
        s = gelu / (1 + tl.exp2(-1.44269504089 * alpha * gelu))
        return tl.fma(s, linear, s)  # (s * (linear + 1))
    ```

- Evidence mapping:
  - SwiGLU fusion → `_swiglu()` function called within GEMM kernel
  - Gating computation → `s = gelu / (1 + tl.exp2(-1.44269504089 * alpha * gelu))`
  - FMA optimization → `tl.fma(s, linear, s)` for fused multiply-add

## Optimization 3: Layer-Specific Kernel Naming for Profiling
- Commit ID: f600a109b
- Optimization type: Debugging / Profiling
- Summary: Added layer1/layer2 suffixes to kernel names for easier performance profiling.

- Detailed explanation:
  MoE layers typically have two GEMM operations (up-projection and down-projection). The optimization adds suffixes to distinguish them in profiling tools.

- Code excerpt:
    ```python
    gindx = args.get("GatherIndx", None)
    if gindx is not None:
        ret["name"] += "_layer1"
    else:
        ret["name"] += "_layer2"
    if args["B"] is not None:
        ret["name"] += "_bias"
    if args["APPLY_SWIGLU"]:
        ret["name"] += "_swiglu"
    ```

- Evidence mapping:
  - Layer naming → `_layer1` and `_layer2` suffixes
  - Feature flags → `_bias`, `_swiglu`, `_quant` suffixes
