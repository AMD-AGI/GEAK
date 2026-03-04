---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: gemm_afp4wfp4

## Variant Context
- Input semantic type: Matrix multiplication (GEMM)
- Datatype(s): MXFP4 (E2M1) activation and weight with E8M0 scales
- Data representation: Microscaling FP4 with 32-element group scaling
- Target architecture: gfx950 (MI350) with CDNA4 architecture

## Functionality
This kernel performs FP4 GEMM using AMD's MXFP4 (Microscaling FP4) format. Both activations and weights are in E2M1 format (4-bit floating point with 2 exponent bits and 1 mantissa bit), with E8M0 per-group scales (32 elements per scale). The kernel uses Triton's experimental Gluon DSL for fine-grained control over AMD CDNA4 hardware features.

## Optimization 1: Gluon DSL with Custom Memory Layouts
- Commit ID: ef827c75e
- Optimization type: Memory / Compute
- Summary: Implemented custom blocked and linear layouts using Gluon DSL for optimal data movement and compute mapping.

- Detailed explanation:
  The kernel uses Triton's Gluon DSL to define custom memory layouts that match AMD CDNA4 hardware:
  1. `BlockedLayout` for A and B matrix tiles with specific thread/warp mappings
  2. `DistributedLinearLayout` for scale tensors with register/lane/warp bases
  3. `SwizzledSharedLayout` for shared memory with bank conflict avoidance
  4. `AMDMFMALayout` for MFMA instruction output mapping

- Code excerpt:
    ```python
    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )

    linear_as: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 2], [0, 4], [64, 0], [128, 0]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]],
        warp_bases=[[0, 0], [0, 0], [32, 0]],
        block_bases=[],
        shape=[BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE],
    )

    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0]
    )

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32],
        transposed=True,
        warps_per_cta=[2, num_warps // 2],
    )
    ```

- Evidence mapping:
  - Custom blocked layout → `BlockedLayout` with specific thread/warp configuration
  - Scale tensor layout → `DistributedLinearLayout` with register/lane/warp bases
  - Bank conflict avoidance → `SwizzledSharedLayout` with vec=16, per_phase=2
  - MFMA mapping → `AMDMFMALayout` version=4 for CDNA4

## Optimization 2: MFMA Scaled Instructions for FP4
- Commit ID: ef827c75e
- Optimization type: Compute
- Summary: Used AMD CDNA4's native MFMA scaled instructions for FP4 matrix multiplication with integrated scaling.

- Detailed explanation:
  The kernel uses `gl.amd.cdna4.mfma_scaled()` which is a native CDNA4 instruction that:
  1. Takes FP4 (E2M1) inputs for both A and B matrices
  2. Applies E8M0 scales during the multiply-accumulate
  3. Accumulates in FP32 for numerical stability
  This avoids separate dequantization and scaling operations.

- Code excerpt:
    ```python
    accumulator = gl.amd.cdna4.mfma_scaled(
        a=curr_a,
        a_scale=curr_a_scales,
        a_format="e2m1",
        b=curr_b,
        b_scale=curr_b_scales,
        b_format="e2m1",
        acc=accumulator,
    )
    ```

- Evidence mapping:
  - Native FP4 MFMA → `gl.amd.cdna4.mfma_scaled()` function
  - E2M1 format → `a_format="e2m1"` and `b_format="e2m1"`
  - Integrated scaling → `a_scale` and `b_scale` parameters

## Optimization 3: XCD Remapping for Load Balancing
- Commit ID: ef827c75e
- Optimization type: Scheduling
- Summary: Implemented XCD (Accelerator Complex Die) remapping for better workload distribution across chiplets.

- Detailed explanation:
  MI350 GPUs have multiple XCDs (chiplets). The kernel remaps program IDs to ensure continuous chunks of work are assigned to each XCD, improving cache locality and reducing cross-chiplet communication.

- Code excerpt:
    ```python
    GRID_MN = gl.cdiv(M, BLOCK_SIZE_M) * gl.cdiv(N, BLOCK_SIZE_N)

    pid_unified = gl.program_id(axis=0)
    # remap so that XCDs get continuous chunks of pids (of CHUNK_SIZE).
    pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)

    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    ```

- Evidence mapping:
  - XCD remapping → `remap_xcd()` function call
  - 8 XCDs → `NUM_XCDS=8` parameter
  - Continuous chunks → Remapping ensures locality per XCD

## Optimization 4: Split-K for Large K Dimensions
- Commit ID: ef827c75e
- Optimization type: Compute / Scheduling
- Summary: Implemented split-K parallelization with a separate reduce kernel for large K dimensions.

- Detailed explanation:
  For matrices with large K dimensions, the kernel splits the K dimension across multiple thread blocks:
  1. Main kernel computes partial results for each K split
  2. Reduce kernel sums the partial results
  This increases parallelism and GPU utilization for K-bound problems.

- Code excerpt:
    ```python
    if config["NUM_KSPLIT"] > 1:
        SPLITK_BLOCK_SIZE = (
            triton.cdiv(
                (2 * triton.cdiv(K, config["NUM_KSPLIT"])), config["BLOCK_SIZE_K"]
            )
            * config["BLOCK_SIZE_K"]
        )
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=x.device
        )
    
    # ... main kernel execution ...
    
    if config["NUM_KSPLIT"] > 1:
        _gemm_afp4wfp4_reduce_kernel[grid_reduce](
            y_pp, y, M, N,
            y_pp.stride(0), y_pp.stride(1), y_pp.stride(2),
            y.stride(0), y.stride(1),
            REDUCE_BLOCK_SIZE_M, REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT, triton.next_power_of_2(config["NUM_KSPLIT"]),
        )
    ```

- Evidence mapping:
  - Split-K parallelization → `NUM_KSPLIT` configuration parameter
  - Partial results → `y_pp` tensor with shape `(NUM_KSPLIT, M, N)`
  - Reduce kernel → `_gemm_afp4wfp4_reduce_kernel` for final summation

## Optimization 5: Buffer Load with Cache Hints
- Commit ID: ef827c75e
- Optimization type: Memory
- Summary: Used AMD-specific buffer load instructions with cache modifiers for optimized memory access.

- Detailed explanation:
  The kernel uses `gl.amd.cdna4.buffer_load()` with configurable cache modifiers to optimize memory access patterns. This allows fine-grained control over L1/L2 cache behavior.

- Code excerpt:
    ```python
    a = gl.amd.cdna4.buffer_load(
        ptr=a_ptr,
        offsets=offs_a,
        mask=offs_ak[:, None] < K - k * (BLOCK_SIZE_K // 2),
        cache=cache_modifier,
    )
    a_scales = gl.amd.cdna4.buffer_load(
        ptr=a_scales_ptr,
        offsets=offs_as,
        mask=offs_ks[:, None] < (2 * K // SCALE_GROUP_SIZE) - k * (BLOCK_SIZE_K // SCALE_GROUP_SIZE),
        cache=cache_modifier,
    )
    ```

- Evidence mapping:
  - Buffer load → `gl.amd.cdna4.buffer_load()` function
  - Cache control → `cache=cache_modifier` parameter
  - Masked loads → `mask` parameter for boundary handling
