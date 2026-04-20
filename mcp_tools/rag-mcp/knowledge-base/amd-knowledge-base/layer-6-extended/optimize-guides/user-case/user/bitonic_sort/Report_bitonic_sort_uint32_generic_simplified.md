# Kernel: bitonic_sort (Simplified)

## Variant Context
- Input: Array of unsigned integers
- Datatype: uint32
- Architecture: Generic HIP/AMD GPU

## Key Optimizations

1. **Bitwise Index Computation**: Replaced division/modulo with bitwise ops. Uses `thread_id & mask` for modulo and `thread_id >> shift` for division since pair_distance is power-of-2. Eliminates expensive integer division.

2. **Branchless Sort Direction**: Replaced if-statement with `(thread_id >> step) & 1u` and XOR operation to compute sort direction. Eliminates warp divergence from conditional branch.

3. **Conditional Memory Stores**: Added `need_swap` check to skip global memory writes when elements already in correct order. Reduces memory bandwidth for partially sorted data.

4. **Single Comparison Reuse**: Computed `left_gt = (left_element > right_element)` once, reused for greater/lesser selection and swap decision. Avoids redundant comparisons.

## Performance Impact
- Bitwise ops ~10x faster than integer division
- Branchless code improves warp efficiency
- Conditional stores reduce memory traffic by up to 50%
