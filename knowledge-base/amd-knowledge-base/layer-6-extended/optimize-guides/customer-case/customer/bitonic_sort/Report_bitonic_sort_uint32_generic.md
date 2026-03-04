---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: bitonic_sort

## Variant Context
- Input semantic type: Sorting (bitonic sort algorithm)
- Datatype(s): uint32 (unsigned int)
- Data representation: Dense array
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel implements the bitonic sort algorithm on GPU. Each kernel invocation handles one stage within one step of the bitonic sort. Threads compare and potentially swap pairs of elements at specific distances determined by the current step and stage parameters. The algorithm builds increasingly larger sorted sequences until the entire array is sorted.

## Optimization 1: Bitwise Operations Instead of Division/Modulo
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Replaced expensive division and modulo operations with faster bitwise operations
- Detailed explanation: The baseline uses division and modulo operations to compute indices, which are expensive on GPUs. The optimized version uses bitwise AND for modulo (when divisor is power of 2) and bit shifts for division/multiplication by powers of 2. This significantly reduces ALU cycles for index computation.

- Code excerpt (baseline):
    ```cpp
    const unsigned int same_order_block_width = 1 << step;
    const unsigned int pair_distance = 1 << (step - stage);
    const unsigned int sorted_block_width = 2 * pair_distance;

    const unsigned int left_id
        = (thread_id % pair_distance) + (thread_id / pair_distance) * sorted_block_width;
    const unsigned int right_id = left_id + pair_distance;

    if((thread_id / same_order_block_width) % 2 == 1)
        sort_increasing = !sort_increasing;
    ```

- Code excerpt (optimized):
    ```cpp
    const unsigned int shift             = step - stage;        // log2(pair_distance)
    const unsigned int pair_distance     = 1u << shift;         // 2^(step - stage)
    const unsigned int mask              = pair_distance - 1u;  // for fast modulo
    const unsigned int block_shift       = shift + 1;           // sorted_block_width = 2 * pair_distance

    // Compute indices using bitwise operations to avoid division/modulo
    const unsigned int left_id  = (thread_id & mask) + ((thread_id >> shift) << block_shift);
    const unsigned int right_id = left_id + pair_distance;

    // Determine sorting direction branchlessly
    const unsigned int order_toggle = (thread_id >> step) & 1u;
    ```

- Evidence mapping:
  - "Bitwise AND for modulo" → `(thread_id & mask)` instead of `(thread_id % pair_distance)`
  - "Bit shift for division" → `(thread_id >> shift)` instead of `(thread_id / pair_distance)`
  - "Bit shift for multiplication" → `<< block_shift` instead of `* sorted_block_width`
  - "Precomputed mask" → `const unsigned int mask = pair_distance - 1u;`

## Optimization 2: Branchless Sort Direction Computation
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Replaced conditional branch with branchless XOR operation for determining sort direction
- Detailed explanation: The baseline uses an if-statement to toggle the sort direction, which causes thread divergence. The optimized version uses XOR operation to compute the final sort direction without branching, ensuring all threads in a warp execute the same instructions.

- Code excerpt (baseline):
    ```cpp
    if((thread_id / same_order_block_width) % 2 == 1)
        sort_increasing = !sort_increasing;
    ```

- Code excerpt (optimized):
    ```cpp
    // Determine sorting direction branchlessly; toggle when (thread_id / (1<<step)) is odd
    const unsigned int order_toggle = (thread_id >> step) & 1u;
    const bool increasing = static_cast<bool>(sort_increasing ^ static_cast<bool>(order_toggle));
    ```

- Evidence mapping:
  - "Branchless toggle" → `sort_increasing ^ static_cast<bool>(order_toggle)` instead of if-statement
  - "Bitwise extraction of toggle bit" → `(thread_id >> step) & 1u`

## Optimization 3: Conditional Store to Reduce Memory Bandwidth
- Commit ID: baseline → optimized
- Optimization type: Memory
- Summary: Skip global memory stores when elements are already in correct order
- Detailed explanation: The baseline always writes both elements back to global memory regardless of whether a swap occurred. The optimized version checks if a swap is needed and only performs stores when necessary. This reduces global memory write traffic when elements are already correctly ordered.

- Code excerpt (baseline):
    ```cpp
    const unsigned int greater = (left_element > right_element) ? left_element : right_element;
    const unsigned int lesser  = (left_element > right_element) ? right_element : left_element;
    array[left_id]             = (sort_increasing) ? lesser : greater;
    array[right_id]            = (sort_increasing) ? greater : lesser;
    ```

- Code excerpt (optimized):
    ```cpp
    // Compare once and decide if a swap is needed (to avoid unnecessary global stores)
    const bool           left_gt = (left_element > right_element);
    const unsigned int   greater = left_gt ? left_element : right_element;
    const unsigned int   lesser  = left_gt ? right_element : left_element;
    const bool           need_swap = increasing ? left_gt : !left_gt;

    // Store results according to sorting order
    if (need_swap) {
        array[left_id]  = increasing ? lesser : greater;
        array[right_id] = increasing ? greater : lesser;
    }
    // else: elements are already in the correct order; skip stores to save memory bandwidth
    ```

- Evidence mapping:
  - "Conditional store" → `if (need_swap) { array[left_id] = ...; array[right_id] = ...; }`
  - "Skip unnecessary writes" → Comment: "else: elements are already in the correct order; skip stores to save memory bandwidth"
  - "Single comparison for swap decision" → `const bool need_swap = increasing ? left_gt : !left_gt;`

## Optimization 4: Reduced Redundant Comparisons
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Compute comparison result once and reuse it
- Detailed explanation: The baseline computes `(left_element > right_element)` twice - once for greater and once for lesser. The optimized version computes this comparison once and stores it in a boolean variable, then uses it for both assignments.

- Code excerpt (baseline):
    ```cpp
    const unsigned int greater = (left_element > right_element) ? left_element : right_element;
    const unsigned int lesser  = (left_element > right_element) ? right_element : left_element;
    ```

- Code excerpt (optimized):
    ```cpp
    const bool           left_gt = (left_element > right_element);
    const unsigned int   greater = left_gt ? left_element : right_element;
    const unsigned int   lesser  = left_gt ? right_element : left_element;
    ```

- Evidence mapping:
  - "Single comparison" → `const bool left_gt = (left_element > right_element);`
  - "Reuse comparison result" → `left_gt ? left_element : right_element` and `left_gt ? right_element : left_element`
