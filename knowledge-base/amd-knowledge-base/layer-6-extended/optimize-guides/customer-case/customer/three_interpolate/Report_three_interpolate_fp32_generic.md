---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: three_interpolate

## Variant Context
- Input semantic type: Three-point weighted interpolation for point cloud upsampling
- Datatype(s): fp32
- Data representation: Dense tensors for points (B, C, M), indices (B, N, 3), weights (B, N, 3)
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel performs weighted interpolation using three nearest neighbors for point cloud feature upsampling. For each output point, it computes a weighted sum of features from three source points, where the weights and indices are precomputed (typically from three_nn kernel).

## Optimization 1: Precomputed Base Pointers with Size_t for Large Tensors
- Commit ID: baseline → optimized
- Optimization type: Compute / Memory
- Summary: Precompute per-(B,C) base pointers using size_t to prevent overflow and reduce per-iteration arithmetic
- Detailed explanation: The baseline modifies pointers with `+=` operations using int arithmetic. The optimized version precomputes base pointers for each (batch, channel) combination using size_t casts to prevent integer overflow for large tensors. This also moves the base address computation outside the main loop.

- Code excerpt (baseline):
    ```cpp
    weight += bs_idx * n * 3 + pt_idx * 3;
    points += bs_idx * c * m + c_idx * m;
    idx += bs_idx * n * 3 + pt_idx * 3;
    out += bs_idx * c * n + c_idx * n;

    out[pt_idx] = weight[0] * points[idx[0]] + weight[1] * points[idx[1]] +
                  weight[2] * points[idx[2]];
    ```

- Code excerpt (optimized):
    ```cpp
    // Precompute per-(B,C) base pointers to reduce per-iteration arithmetic
    const float* __restrict__ points_bc =
        points + static_cast<size_t>(bs_idx) * static_cast<size_t>(c_stride) * static_cast<size_t>(m_stride) +
                 static_cast<size_t>(c_idx) * static_cast<size_t>(m_stride);
    const float* __restrict__ weight_bc =
        weight + static_cast<size_t>(bs_idx) * static_cast<size_t>(n_stride) * 3;
    const int*   __restrict__ idx_bc    =
        idx + static_cast<size_t>(bs_idx) * static_cast<size_t>(n_stride) * 3;
    float*       __restrict__ out_bc    =
        out + static_cast<size_t>(bs_idx) * static_cast<size_t>(c_stride) * static_cast<size_t>(n_stride) +
                 static_cast<size_t>(c_idx) * static_cast<size_t>(n_stride);
    ```

- Evidence mapping:
  - "size_t for overflow prevention" → `static_cast<size_t>(bs_idx) * static_cast<size_t>(c_stride)`
  - "Precomputed base pointers" → `points_bc`, `weight_bc`, `idx_bc`, `out_bc`
  - "__restrict__ qualifiers" → Hints no aliasing for better optimization

## Optimization 2: Grid-Stride Loop Pattern
- Commit ID: baseline → optimized
- Optimization type: Compute / Scheduling
- Summary: Use grid-stride loop to improve CU utilization for large N
- Detailed explanation: The baseline processes one point per thread. The optimized version uses a grid-stride loop pattern where each thread can process multiple points. This improves compute unit utilization when N is large and allows better load balancing across the GPU.

- Code excerpt (baseline):
    ```cpp
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || c_idx >= c || pt_idx >= n) return;
    // ... process single point
    ```

- Code excerpt (optimized):
    ```cpp
    // Grid-stride loop over points to improve CU utilization when n is large
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int pt_idx = tid; pt_idx < n; pt_idx += stride, w_off += w_step, i_off += i_step, out_off += o_step) {
        // ... process point
    }
    ```

- Evidence mapping:
  - "Grid-stride loop" → `for (int pt_idx = tid; pt_idx < n; pt_idx += stride, ...)`
  - "Stride computation" → `const int stride = blockDim.x * gridDim.x;`

## Optimization 3: Rolling Offset Updates to Avoid Per-Iteration Multiplications
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Maintain rolling offsets with additive updates instead of multiplicative index computation
- Detailed explanation: Instead of computing `pt_idx * 3` for each iteration, the optimized version maintains rolling offset variables that are incremented by fixed steps. This replaces multiplications with additions in the hot loop.

- Code excerpt (optimized):
    ```cpp
    // Maintain rolling offsets to avoid per-iteration multiplications
    int w_off = tid * 3;
    int i_off = tid * 3;
    int out_off = tid;
    const int w_step = stride * 3;
    const int i_step = stride * 3;
    const int o_step = stride;

    for (int pt_idx = tid; pt_idx < n; pt_idx += stride, w_off += w_step, i_off += i_step, out_off += o_step) {
        const float w0 = weight_bc[w_off + 0];
        const float w1 = weight_bc[w_off + 1];
        const float w2 = weight_bc[w_off + 2];
        // ...
    }
    ```

- Evidence mapping:
  - "Rolling offsets" → `w_off += w_step` instead of `pt_idx * 3`
  - "Precomputed steps" → `const int w_step = stride * 3;`
  - "Additive updates in loop" → `w_off += w_step, i_off += i_step, out_off += o_step`

## Optimization 4: Explicit Load-Compute-Store Pattern
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Separate loads, computation, and stores for better instruction scheduling
- Detailed explanation: The optimized version explicitly loads all weights and indices into registers first, then performs the gather operations, then computes the weighted sum step by step, and finally stores the result. This separation can help the compiler schedule instructions better.

- Code excerpt (baseline):
    ```cpp
    out[pt_idx] = weight[0] * points[idx[0]] + weight[1] * points[idx[1]] +
                  weight[2] * points[idx[2]];
    ```

- Code excerpt (optimized):
    ```cpp
    // Load weights and indices into registers
    const float w0 = weight_bc[w_off + 0];
    const float w1 = weight_bc[w_off + 1];
    const float w2 = weight_bc[w_off + 2];

    const int i0 = idx_bc[i_off + 0];
    const int i1 = idx_bc[i_off + 1];
    const int i2 = idx_bc[i_off + 2];

    // Gather from points using computed indices
    const float v0 = points_bc[i0];
    const float v1 = points_bc[i1];
    const float v2 = points_bc[i2];

    // Accumulate in the same arithmetic order as the original
    float result = w0 * v0;
    result += w1 * v1;
    result += w2 * v2;

    // Store the result
    out_bc[out_off] = result;
    ```

- Evidence mapping:
  - "Separate load phase" → Weights and indices loaded into named variables
  - "Separate gather phase" → `v0`, `v1`, `v2` loaded from points
  - "Step-by-step accumulation" → `result = w0 * v0; result += w1 * v1; result += w2 * v2;`
