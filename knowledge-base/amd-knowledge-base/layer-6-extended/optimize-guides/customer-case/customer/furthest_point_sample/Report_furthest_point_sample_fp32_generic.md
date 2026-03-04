---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: furthest_point_sampling

## Variant Context
- Input semantic type: Furthest point sampling for point cloud downsampling
- Datatype(s): fp32
- Data representation: Dense point cloud tensors (B, N, 3)
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel performs furthest point sampling (FPS) on point clouds. Starting from an initial point, it iteratively selects the point that is furthest from all previously selected points. This is commonly used for point cloud downsampling in 3D deep learning.

## Optimization 1: Loop Unrolling for Distance Computation
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Added #pragma unroll 4 directive for the main distance computation loop
- Detailed explanation: The optimized version adds a `#pragma unroll 4` directive to the loop that computes distances from each point to the previously selected point. This reduces loop overhead and enables better instruction-level parallelism by allowing the compiler to schedule multiple iterations' instructions together.

- Code excerpt (baseline):
    ```cpp
    for (int k = tid; k < n; k += stride) {
      float x2, y2, z2;
      x2 = dataset[k * 3 + 0];
      y2 = dataset[k * 3 + 1];
      z2 = dataset[k * 3 + 2];
      float d =
          (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
      // ...
    }
    ```

- Code excerpt (optimized):
    ```cpp
    // Each thread computes candidate distances and updates local best
    #pragma unroll 4
    for (int k = tid; k < n; k += stride) {
      const float x2 = dataset[k * 3 + 0];
      const float y2 = dataset[k * 3 + 1];
      const float z2 = dataset[k * 3 + 2];

      // distance squared between (x1,y1,z1) and (x2,y2,z2)
      const float dx = x2 - x1;
      const float dy = y2 - y1;
      const float dz = z2 - z1;
      float d = dx * dx + dy * dy + dz * dz;
      // ...
    }
    ```

- Evidence mapping:
  - "Loop unrolling" → `#pragma unroll 4`
  - "Applied to distance loop" → Loop over `k` from `tid` to `n`

## Optimization 2: Explicit Difference Variables for Distance Computation
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Store coordinate differences in named variables before squaring
- Detailed explanation: The baseline computes `(x2 - x1) * (x2 - x1)` inline, which may cause the subtraction to be computed twice. The optimized version stores the differences in `dx`, `dy`, `dz` variables first, then squares them. This ensures the subtraction is computed only once.

- Code excerpt (baseline):
    ```cpp
    float d =
        (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
    ```

- Code excerpt (optimized):
    ```cpp
    const float dx = x2 - x1;
    const float dy = y2 - y1;
    const float dz = z2 - z1;
    float d = dx * dx + dy * dy + dz * dz;
    ```

- Evidence mapping:
  - "Named difference variables" → `const float dx = x2 - x1;`
  - "Single subtraction per dimension" → Each difference computed once and reused

## Optimization 3: Use of fminf/fmaxf Intrinsics
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Use fminf/fmaxf intrinsics instead of min/max or ternary operators
- Detailed explanation: The optimized version uses `fminf` and `fmaxf` intrinsics for floating-point min/max operations. These intrinsics map directly to hardware instructions and can be more efficient than generic min/max functions or ternary operators.

- Code excerpt (baseline):
    ```cpp
    float d2 = min(d, temp[k]);
    temp[k] = d2;
    besti = d2 > best ? k : besti;
    best = d2 > best ? d2 : best;
    ```

- Code excerpt (optimized):
    ```cpp
    // If temporary slot contains a previous minimum, take the minimum
    float d2 = fminf(d, temp[k]);
    temp[k] = d2;

    // Track the farthest distance so far
    besti = (d2 > best) ? k : besti;
    best = fmaxf(best, d2);
    ```

- Evidence mapping:
  - "fminf intrinsic" → `float d2 = fminf(d, temp[k]);`
  - "fmaxf intrinsic" → `best = fmaxf(best, d2);`

## Optimization 4: Const Qualifiers for Loaded Values
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Added const qualifiers to loaded coordinate values
- Detailed explanation: The optimized version marks loaded coordinate values (x1, y1, z1, x2, y2, z2) as const, helping the compiler understand these values won't change and potentially enabling better register allocation and optimization.

- Code excerpt (baseline):
    ```cpp
    float x1 = dataset[old * 3 + 0];
    float y1 = dataset[old * 3 + 1];
    float z1 = dataset[old * 3 + 2];
    // ...
    float x2, y2, z2;
    x2 = dataset[k * 3 + 0];
    ```

- Code excerpt (optimized):
    ```cpp
    // Cache coordinates of the previously selected point to avoid repeated global reads
    const float x1 = dataset[old * 3 + 0];
    const float y1 = dataset[old * 3 + 1];
    const float z1 = dataset[old * 3 + 2];
    // ...
    const float x2 = dataset[k * 3 + 0];
    const float y2 = dataset[k * 3 + 1];
    const float z2 = dataset[k * 3 + 2];
    ```

- Evidence mapping:
  - "Const for reference point" → `const float x1 = dataset[old * 3 + 0];`
  - "Const for candidate point" → `const float x2 = dataset[k * 3 + 0];`
