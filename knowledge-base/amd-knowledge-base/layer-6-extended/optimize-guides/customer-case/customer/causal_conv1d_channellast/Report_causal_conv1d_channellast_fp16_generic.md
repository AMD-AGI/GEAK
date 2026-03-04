---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: causal_conv1d_channellast

## Variant Context
- Input semantic type: 1D causal convolution with channel-last memory layout
- Datatype(s): fp16 (half precision)
- Data representation: Channel-last tensor layout (B, L, C)
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel performs causal 1D convolution with channel-last memory layout. It processes input sequences in chunks, loading data into shared memory for efficient access. The convolution uses a sliding window of width `kWidth` and optionally applies SiLU activation. The channel-last layout is optimized for memory coalescing when accessing channel dimensions.

## Optimization 1: Precomputed Sequence Indices
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Moved sequence index computation outside the main computation loop
- Detailed explanation: The baseline computes sequence indices inside the output computation loop. The optimized version precomputes all sequence indices at the beginning of the kernel and stores them in a local array. This reduces redundant computation in the hot loop and improves instruction-level parallelism.

- Code excerpt (baseline):
    ```cpp
    int seq_idx_thread[kWidth - 1 + kLPerThread];
    if constexpr (kHasSeqIdx) {
        #pragma unroll
        for (int i = 0; i < kWidth - 1 + kLPerThread; ++i) {
            seq_idx_thread[i] = chunk_l_id * kChunkSizeL + col_idx * kLPerThread + i - (kWidth - 1) >= 0 ? seq_idx[col_idx * kLPerThread + i - (kWidth - 1)] : -1;
        }
    }
    // ... later in the loop:
    const int seq_idx_cur = !kHasSeqIdx ? 0 : seq_idx_thread[i + kWidth - 1];
    ```

- Code excerpt (optimized):
    ```cpp
    // Precompute sequence indices for this thread's span
    int seq_idx_thread[kWidth - 1 + kLPerThread];
    if constexpr (kHasSeqIdx) {
        #pragma unroll
        for (int i = 0; i < kWidth - 1 + kLPerThread; ++i) {
            const int sidx = chunk_l_id * kChunkSizeL + c_idx * kLPerThread + i - (kWidth - 1);
            seq_idx_thread[i] = (sidx >= 0) ? seq_idx[sidx] : -1;
        }
    }
    // ... precompute seq_idx_cur before the loop:
    int seq_idx_cur;
    if constexpr (kHasSeqIdx) {
        seq_idx_cur = seq_idx_thread[kWidth - 1];
    }
    ```

- Evidence mapping:
  - "Precomputed sequence indices" → `// Precompute sequence indices for this thread's span` comment and early computation
  - "Moved outside loop" → `seq_idx_cur` computed before the output computation loop

## Optimization 2: Precomputed Validity Flags
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Precomputed bounds check flags to avoid repeated condition evaluation in hot loop
- Detailed explanation: The optimized version computes validity flags (`valid_x`, `valid_out`) once before the main computation loop. These flags are then used in the inner loop instead of recomputing the bounds checks each iteration, reducing branch overhead and enabling better compiler optimization.

- Code excerpt (optimized):
    ```cpp
    // Precompute valid flags to avoid repeated bounds checks in the hot loop
    const bool valid_x = (chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim);
    const bool valid_out = valid_x && (chunk_l_id * kChunkSizeL + l_idx < params.seqlen);

    // ... in the computation loop:
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) {
        out_vals[i] = bias_val;
        const bool valid_pos = valid_out && ((chunk_l_id * kChunkSizeL + l_idx + i) < params.seqlen);
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            if constexpr (!kHasSeqIdx) {
                if (valid_pos) {
                    out_vals[i] += weight_vals[w] * x_vals[i + w];
                }
            }
            // ...
        }
    }
    ```

- Evidence mapping:
  - "Precomputed validity flags" → `const bool valid_x = ...` and `const bool valid_out = ...`
  - "Used in hot loop" → `if (valid_pos)` check using precomputed flags

## Optimization 3: Early Index Computation
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Moved row_idx and col_idx computation to the beginning of the kernel
- Detailed explanation: The baseline computes `row_idx` and `col_idx` later in the kernel after some operations. The optimized version computes these indices immediately after computing `l_idx` and `c_idx`, allowing the compiler to better schedule instructions and potentially overlap computation with memory operations.

- Code excerpt (baseline):
    ```cpp
    const int l_idx = tid / kNThreadsPerC;
    const int c_idx = tid % kNThreadsPerC;
    // ... many lines of code ...
    constexpr int kLPerThread = constexpr_min(kChunkSizeL * kChunkSizeC / kNThreads, kChunkSizeL);
    // ... more code ...
    const int row_idx = tid / kNThreadsPerRow;
    const int col_idx = tid % kNThreadsPerRow;
    ```

- Code excerpt (optimized):
    ```cpp
    const int l_idx = tid / kNThreadsPerC;
    const int c_idx = tid % kNThreadsPerC;
    const int row_idx = tid / kNThreadsPerRow;
    const int col_idx = tid % kNThreadsPerRow;
    ```

- Evidence mapping:
  - "Early computation" → `row_idx` and `col_idx` computed immediately after `l_idx` and `c_idx`
  - "Grouped index computations" → All four index variables computed together at kernel start

## Optimization 4: Const Pointer Qualifiers for Memory Loads
- Commit ID: baseline → optimized
- Optimization type: Memory
- Summary: Added const qualifiers to pointer casts for read-only memory accesses
- Detailed explanation: The optimized version uses `const vec_t*` casts when reading from memory, which helps the compiler understand that these are read-only accesses. This can enable better memory access optimization and potentially allow the compiler to use read-only cache paths.

- Code excerpt (baseline):
    ```cpp
    reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(x + l * kLPerLoad * params.x_l_stride);
    ```

- Code excerpt (optimized):
    ```cpp
    reinterpret_cast<vec_t*>(x_vals_load)[0] = *reinterpret_cast<const vec_t*>(x + l * kLPerLoad * params.x_l_stride);
    ```

- Evidence mapping:
  - "Const pointer for reads" → `*reinterpret_cast<const vec_t*>(...)` instead of `*reinterpret_cast<vec_t*>(...)`
  - "Applied to all read operations" → Similar pattern used for `initial_states`, `bias_ptr`, and shared memory reads
