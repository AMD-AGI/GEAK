---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: histogram256_block

## Variant Context
- Input semantic type: 256-bin histogram computation
- Datatype(s): uint8 input, uint32 output bins
- Data representation: Dense byte array
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel computes a 256-bin histogram for blocks of unsigned char data. Each thread processes multiple items, accumulating counts in shared memory with a bank-conflict-avoiding layout. The per-thread histograms are then reduced to produce per-block histogram bins.

## Optimization 1: Vectorized 32-bit Initialization
- Commit ID: baseline → optimized
- Optimization type: Memory
- Summary: Use 32-bit stores to initialize shared memory instead of byte-by-byte initialization
- Detailed explanation: The baseline initializes shared memory one byte at a time in a loop of 256 iterations. The optimized version casts the shared memory to uint32 pointers and initializes 4 bytes at a time, reducing the number of store operations by 4x and improving memory throughput.

- Code excerpt (baseline):
    ```cpp
    // Initialize 'thread_bins' to 0
    for(int i = 0; i < bin_size; ++i)
    {
        thread_bins[i + bin_size * sh_thread_id] = 0;
    }
    ```

- Code excerpt (optimized):
    ```cpp
    // Phase 1: initialize this thread's column to zero using 32-bit stores where possible
    unsigned char* __restrict__ my_col = thread_bins + (size_t)bin_size * (size_t)sh_thread_id;
    unsigned int* __restrict__ my_col_u32 = reinterpret_cast<unsigned int*>(my_col);
    const int col_elems4 = bin_size >> 2; // 256/4 = 64
    #pragma unroll
    for (int w = 0; w < col_elems4; ++w) {
        my_col_u32[w] = 0u;
    }
    ```

- Evidence mapping:
  - "32-bit stores" → `unsigned int* __restrict__ my_col_u32 = reinterpret_cast<unsigned int*>(my_col);`
  - "4x fewer iterations" → `col_elems4 = bin_size >> 2; // 256/4 = 64`
  - "Unrolled loop" → `#pragma unroll`

## Optimization 2: Vectorized Data Loading with uchar4
- Commit ID: baseline → optimized
- Optimization type: Memory
- Summary: Load 4 bytes at a time using uchar4 vector type for coalesced memory access
- Detailed explanation: The baseline loads one byte at a time from global memory. The optimized version uses uchar4 vector loads to read 4 bytes in a single memory transaction, improving memory bandwidth utilization and reducing loop overhead.

- Code excerpt (baseline):
    ```cpp
    for(int i = 0; i < items_per_thread; i++)
    {
        const unsigned int value = data[(block_id * block_size + thread_id) * items_per_thread + i];
        thread_bins[value * block_size + sh_thread_id]++;
    }
    ```

- Code excerpt (optimized):
    ```cpp
    // Vectorized path: process 4 items at a time if divisible
    int k = 0;
    const int vec = 4;
    int limit = items_per_thread & ~(vec - 1);

    #pragma unroll 4
    for (; k < limit; k += vec) {
        // Use uchar4 loads for coalescing and reduced loop overhead
        uchar4 v = reinterpret_cast<const uchar4*>(data)[(base_idx + k) >> 2];
        thread_bins[(int)v.x * block_size + sh_thread_id]++;
        thread_bins[(int)v.y * block_size + sh_thread_id]++;
        thread_bins[(int)v.z * block_size + sh_thread_id]++;
        thread_bins[(int)v.w * block_size + sh_thread_id]++;
    }

    // Tail elements
    for (; k < items_per_thread; ++k) {
        unsigned int value = data[base_idx + k];
        thread_bins[(int)(value & 0xFF) * block_size + sh_thread_id]++;
    }
    ```

- Evidence mapping:
  - "Vector load" → `uchar4 v = reinterpret_cast<const uchar4*>(data)[(base_idx + k) >> 2];`
  - "Process 4 elements" → `v.x`, `v.y`, `v.z`, `v.w` accessed separately
  - "Tail handling" → `for (; k < items_per_thread; ++k)` for remaining elements

## Optimization 3: Vectorized Reduction with 32-bit Loads
- Commit ID: baseline → optimized
- Optimization type: Memory / Compute
- Summary: Use 32-bit loads and bit extraction for faster histogram reduction
- Detailed explanation: The baseline reduction reads one byte at a time from shared memory. The optimized version reads 4 bytes at a time using uint32 loads, then extracts individual bytes using bit shifts and masks. This reduces the number of memory operations and can be more efficient on GPUs.

- Code excerpt (baseline):
    ```cpp
    // Accumulate bins.
    unsigned int bin_acc = 0;
    for(int j = 0; j < block_size; ++j)
    {
        // Sum the result from the j-th thread from the 'block_size'-sized 'bin_id'th bin.
        bin_acc += thread_bins[bin_sh_id * block_size + j];
    }
    ```

- Code excerpt (optimized):
    ```cpp
    // Vectorized reduction: sum 4 bytes at a time via uint32 loads
    const int step4 = block_size >> 2; // number of uint32 entries in the row
    const uint32_t* __restrict__ row_u32 = reinterpret_cast<const uint32_t*>(row);
    #pragma unroll
    for (int w = 0; w < step4; ++w) {
        uint32_t x = row_u32[w];
        bin_acc += (unsigned int)( x        & 0xFFu);
        bin_acc += (unsigned int)((x >>  8) & 0xFFu);
        bin_acc += (unsigned int)((x >> 16) & 0xFFu);
        bin_acc += (unsigned int)((x >> 24) & 0xFFu);
    }

    // Handle any remaining elements if block_size is not divisible by 4
    for (int j = (step4 << 2); j < block_size; ++j) {
        bin_acc += (unsigned int)row[j];
    }
    ```

- Evidence mapping:
  - "32-bit loads" → `uint32_t x = row_u32[w];`
  - "Bit extraction" → `(x >> 8) & 0xFFu`, `(x >> 16) & 0xFFu`, `(x >> 24) & 0xFFu`
  - "4x fewer memory ops" → `step4 = block_size >> 2`

## Optimization 4: Precomputed Base Index
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Precompute base index once instead of recalculating in loop
- Detailed explanation: The optimized version computes the base index for data access once before the loop, avoiding repeated multiplication in each iteration.

- Code excerpt (optimized):
    ```cpp
    const int base_idx = (block_id * block_size + thread_id) * items_per_thread;

    // ... in loop:
    uchar4 v = reinterpret_cast<const uchar4*>(data)[(base_idx + k) >> 2];
    ```

- Evidence mapping:
  - "Precomputed base" → `const int base_idx = (block_id * block_size + thread_id) * items_per_thread;`
  - "Simple offset in loop" → `base_idx + k` instead of full recalculation
