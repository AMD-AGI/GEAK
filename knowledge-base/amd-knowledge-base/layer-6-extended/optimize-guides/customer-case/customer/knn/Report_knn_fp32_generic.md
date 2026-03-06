Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: knn (K-Nearest Neighbors)

## Variant Context
- Input semantic type: K-nearest neighbors search in point cloud
- Datatype(s): fp32
- Data representation: Dense point cloud tensors (xyz coordinates)
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel performs K-nearest neighbors search for point cloud processing. For each query point in `new_xyz`, it finds the K nearest points from `xyz` using a max-heap data structure. The output includes both the indices of the K nearest neighbors and their squared distances.

## Optimization 1: Precomputed Base Pointers with Size_t for Large Tensors
- Commit ID: baseline → optimized
- Optimization type: Compute / Memory
- Summary: Precompute base pointers once and use size_t to prevent integer overflow for large tensors
- Detailed explanation: The baseline uses pointer arithmetic with implicit int multiplication which can overflow for large tensors. The optimized version precomputes base pointers at the start using size_t casts to prevent overflow, and stores them in dedicated pointer variables. This reduces repeated address calculations and ensures correctness for large batch sizes.

- Code excerpt (baseline):
    ```cpp
    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;
    dist2 += bs_idx * m * nsample + pt_idx * nsample;
    ```

- Code excerpt (optimized):
    ```cpp
    // Base pointers per batch and point (use size_t to avoid overflow for big tensors)
    const float* __restrict__ xyz_base = xyz + (size_t)bs_idx * (size_t)n * 3;
    const float* __restrict__ new_xyz_ptr = new_xyz + (size_t)bs_idx * (size_t)m * 3 + (size_t)pt_idx * 3;
    int* __restrict__ idx_base = idx + (size_t)bs_idx * (size_t)m * (size_t)nsample + (size_t)pt_idx * (size_t)nsample;
    float* __restrict__ dist2_base = dist2 + (size_t)bs_idx * (size_t)m * (size_t)nsample + (size_t)pt_idx * (size_t)nsample;
    ```

- Evidence mapping:
  - "size_t for overflow prevention" → `(size_t)bs_idx * (size_t)n * 3`
  - "Dedicated base pointers" → `xyz_base`, `new_xyz_ptr`, `idx_base`, `dist2_base`
  - "__restrict__ qualifier" → Hints to compiler that pointers don't alias

## Optimization 2: Explicit Difference Variables for Distance Computation
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Store coordinate differences in named variables before squaring
- Detailed explanation: The baseline computes `(new_x - x) * (new_x - x)` inline, which may cause the subtraction to be computed twice. The optimized version stores the differences in `dx`, `dy`, `dz` variables first, then squares them. This ensures the subtraction is computed only once and may help the compiler generate better code.

- Code excerpt (baseline):
    ```cpp
    float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
    ```

- Code excerpt (optimized):
    ```cpp
    const float dx = new_x - x;
    const float dy = new_y - y;
    const float dz = new_z - z;
    const float d2 = dx * dx + dy * dy + dz * dz;
    ```

- Evidence mapping:
  - "Named difference variables" → `const float dx = new_x - x;`
  - "Single subtraction per dimension" → Each difference computed once and reused

## Optimization 3: Loop Unrolling with Pragma for Initialization and Output
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Added #pragma unroll for initialization and output loops
- Detailed explanation: The optimized version adds `#pragma unroll` directives to the initialization loop and the output writing loop. This hints to the compiler to unroll these loops, reducing loop overhead and enabling better instruction scheduling for these fixed-iteration loops.

- Code excerpt (baseline):
    ```cpp
    for(int i = 0; i < nsample; i++){
        best_dist[i] = 1e10;
        best_idx[i] = 0;
    }
    // ...
    for(int i = 0; i < nsample; i++){
        idx[i] = best_idx[i];
        dist2[i] = best_dist[i];
    }
    ```

- Code excerpt (optimized):
    ```cpp
    // Initialize only the first nsample entries to +inf and 0
    #pragma unroll
    for (int i = 0; i < nsample; ++i) {
        best_dist[i] = 1e10f;
        best_idx[i] = 0;
    }
    // ...
    // Write results for nsample entries
    #pragma unroll
    for (int i = 0; i < nsample; ++i) {
        idx_base[i] = best_idx[i];
        dist2_base[i] = best_dist[i];
    }
    ```

- Evidence mapping:
  - "Unroll directive for init" → `#pragma unroll` before initialization loop
  - "Unroll directive for output" → `#pragma unroll` before output writing loop

## Optimization 4: Const Qualifiers for Intermediate Values
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Added const qualifiers to intermediate computed values
- Detailed explanation: The optimized version marks intermediate values like `x`, `y`, `z`, `dx`, `dy`, `dz`, and `d2` as const. This helps the compiler understand these values won't change and may enable better register allocation and optimization.

- Code excerpt (optimized):
    ```cpp
    const float x = xyz_base[(size_t)i * 3 + 0];
    const float y = xyz_base[(size_t)i * 3 + 1];
    const float z = xyz_base[(size_t)i * 3 + 2];
    const float dx = new_x - x;
    const float dy = new_y - y;
    const float dz = new_z - z;
    const float d2 = dx * dx + dy * dy + dz * dz;
    ```

- Evidence mapping:
  - "Const for loaded values" → `const float x = ...`
  - "Const for computed values" → `const float dx = ...`, `const float d2 = ...`
