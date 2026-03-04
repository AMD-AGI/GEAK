---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: assign_score_withk

## Variant Context
- Input semantic type: Point cloud score assignment with K-nearest neighbors
- Datatype(s): fp32
- Data representation: Dense tensors (points, centers, scores, knn_idx)
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel performs score-weighted point feature aggregation for point cloud processing. It computes weighted differences between points and their center neighbors based on KNN indices and scores. The forward pass computes: `output[b,o,n,k] = sum_m(scores[b,n,k,m] * (points[b,kn,m,o] - centers[b,cn,m,o]))` where kn and cn are neighbor indices from knn_idx.

## Optimization 1: Elimination of atomicAdd with Local Accumulation
- Commit ID: baseline → optimized
- Optimization type: Memory / Compute
- Summary: Replaced atomicAdd operations with local register accumulation and a single direct store
- Detailed explanation: The baseline kernel used atomicAdd for each iteration of the M loop, causing significant contention and serialization. The optimized version accumulates results in a local register variable `acc` and writes once to global memory at the end. This eliminates atomic operation overhead and allows the compiler to keep the accumulator in registers.

- Code excerpt (baseline):
    ```cpp
    // Baseline: atomicAdd inside M loop
    for (int m = 0; m < M; m++) {
        // ...
        atomicAdd(output + b*N1*O*K + o*N1*K + n*K + k,
            points[b*N0*M*O + kn*M*O + m*O + o] * scores[b*N1*K*M + n*K*M + k*M + m]
                - centers[b*N0*M*O + cn*M*O + m*O + o] * scores[b*N1*K*M + n*K*M + k*M + m]);
    }
    ```

- Code excerpt (optimized):
    ```cpp
    // Optimized: Local accumulation with single store
    float acc = 0.0f;
    int m = 0;
    #pragma unroll 8
    for (; m + 7 < M; m += 8) {
        acc += points_ptr[0] * scores_ptr[0] - centers_ptr[0] * scores_ptr[0];
        // ... (8 iterations unrolled)
    }
    for (; m < M; ++m) {
        acc += points_ptr[0] * scores_ptr[0] - centers_ptr[0] * scores_ptr[0];
        // ...
    }
    // Single store to output (no atomics)
    output[out_base] = acc;
    ```

- Evidence mapping:
  - "Replaced atomicAdd with local accumulation" → `float acc = 0.0f;` and `acc += ...` pattern
  - "Single direct store" → `output[out_base] = acc;` at the end

## Optimization 2: Loop Unrolling with Pragma Unroll
- Commit ID: baseline → optimized
- Optimization type: Compute / Scheduling
- Summary: Manual loop unrolling by factor of 8 with #pragma unroll directive
- Detailed explanation: The M-dimension loop is manually unrolled by a factor of 8, processing 8 elements per iteration. This reduces loop overhead (branch instructions, loop counter updates) and enables instruction-level parallelism. The compiler can better schedule independent operations when they are explicitly written out.

- Code excerpt (optimized):
    ```cpp
    #pragma unroll 8
    for (; m + 7 < M; m += 8) {
        // m + 0
        acc += points_ptr[0] * scores_ptr[0] - centers_ptr[0] * scores_ptr[0];
        // m + 1
        acc += points_ptr[stride_PO] * scores_ptr[1] - centers_ptr[stride_PO] * scores_ptr[1];
        // m + 2
        acc += points_ptr[stride_PO * 2] * scores_ptr[2] - centers_ptr[stride_PO * 2] * scores_ptr[2];
        // m + 3
        acc += points_ptr[stride_PO * 3] * scores_ptr[3] - centers_ptr[stride_PO * 3] * scores_ptr[3];
        // m + 4
        acc += points_ptr[stride_PO * 4] * scores_ptr[4] - centers_ptr[stride_PO * 4] * scores_ptr[4];
        // m + 5
        acc += points_ptr[stride_PO * 5] * scores_ptr[5] - centers_ptr[stride_PO * 5] * scores_ptr[5];
        // m + 6
        acc += points_ptr[stride_PO * 6] * scores_ptr[6] - centers_ptr[stride_PO * 6] * scores_ptr[6];
        // m + 7
        acc += points_ptr[stride_PO * 7] * scores_ptr[7] - centers_ptr[stride_PO * 7] * scores_ptr[7];

        points_ptr  += stride_PO * 8;
        centers_ptr += stride_PO * 8;
        scores_ptr  += 8;
    }
    // handle remaining iterations
    for (; m < M; ++m) {
        acc += points_ptr[0] * scores_ptr[0] - centers_ptr[0] * scores_ptr[0];
        points_ptr  += stride_PO;
        centers_ptr += stride_PO;
        scores_ptr  += 1;
    }
    ```

- Evidence mapping:
  - "Loop unrolling by factor of 8" → `for (; m + 7 < M; m += 8)` with 8 explicit accumulation statements
  - "Remainder handling" → `for (; m < M; ++m)` loop after unrolled section

## Optimization 3: Pointer Arithmetic Optimization with Precomputed Strides
- Commit ID: baseline → optimized
- Optimization type: Memory / Compute
- Summary: Replaced repeated index calculations with pointer increments and precomputed base addresses
- Detailed explanation: The baseline recalculates full array indices on every iteration using expensive multiplications. The optimized version precomputes base addresses and strides once, then uses simple pointer increments. This reduces integer arithmetic overhead and improves memory access patterns by using `__restrict__` pointers to hint no aliasing.

- Code excerpt (baseline):
    ```cpp
    // Baseline: Full index recalculation each iteration
    for (int m = 0; m < M; m++) {
        // ...
        atomicAdd(output + b*N1*O*K + o*N1*K + n*K + k,
            points[b*N0*M*O + kn*M*O + m*O + o] * scores[b*N1*K*M + n*K*M + k*M + m]
                - centers[b*N0*M*O + cn*M*O + m*O + o] * scores[b*N1*K*M + n*K*M + k*M + m]);
    }
    ```

- Code excerpt (optimized):
    ```cpp
    // Precompute base offsets and strides
    const int stride_points_B = N0 * M * O;      // points/centers stride per batch
    const int stride_scores_B = N1 * K * M;      // scores stride per batch
    const int stride_scores_N1 = K * M;          // scores stride per N1
    const int stride_scores_K = M;               // scores stride per K

    // Base addresses for current thread
    const int base_points_b = b * stride_points_B;
    const int base_centers_b = b * stride_points_B;
    const int base_scores_bnk = b * stride_scores_B + n * stride_scores_N1 + k * stride_scores_K;

    // Precompute per-neighbor base offsets for points/centers
    const int base_points_kn = base_points_b + kn * M * O;
    const int base_centers_cn = base_centers_b + cn * M * O;

    // Set up pointers with incremental strides
    const float* __restrict__ points_ptr  = points  + base_points_kn  + (long)o;
    const float* __restrict__ centers_ptr = centers + base_centers_cn + (long)o;
    const float* __restrict__ scores_ptr  = scores  + base_scores_bnk;

    const int stride_PO = O;   // points/centers advance by O for each m-step
    ```

- Evidence mapping:
  - "Precomputed strides" → `const int stride_points_B = N0 * M * O;` and similar declarations
  - "Pointer-based access" → `const float* __restrict__ points_ptr = points + base_points_kn + (long)o;`
  - "Simple pointer increments" → `points_ptr += stride_PO * 8;` instead of full index recalculation
  - "__restrict__ hint" → `const float* __restrict__ points_ptr` for compiler optimization

## Optimization 4: Early Exit Optimization
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Moved bounds check before computation to avoid unnecessary work
- Detailed explanation: The optimized version checks if the neighbor index is out of bounds immediately after loading it, returning early before any computation. This avoids wasted cycles on invalid data paths.

- Code excerpt (optimized):
    ```cpp
    const int kn = (int)knn_idx[bK_N1 + (long)n * (long)K + (long)k];

    // Bounds check; skip if out of range to avoid useless work
    if (kn >= N0 || kn < 0) {
        return;
    }

    // Precompute base offsets and strides (only executed for valid indices)
    const int stride_points_B = N0 * M * O;
    // ...
    ```

- Evidence mapping:
  - "Early bounds check" → `if (kn >= N0 || kn < 0) { return; }` placed before stride computations
  - "Avoids unnecessary computation" → All precomputation and loop code comes after the check
