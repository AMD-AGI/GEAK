# Kernel: gather_points (Simplified)

## Variant Context
- Input: Point features (B, C, N) with indices (B, M)
- Datatype: fp32/fp16 (template)
- Architecture: Generic HIP/AMD GPU

## Key Optimizations

1. **64-bit Offset Computation**: Uses `static_cast<size_t>()` for all offset calculations to prevent integer overflow with large tensors (large B, C, N, M).

2. **Precomputed Base Pointers**: Computes `grad_out_base`, `idx_base`, `grad_points_base` once with `__restrict__` qualifiers. Simple indexing `ptr[pt_idx]` in main computation.

3. **Const Index Variables**: Added `const` to `bs_idx`, `c_idx`, `pt_idx` for compiler optimization hints.

## Performance Impact
- Prevents overflow bugs for large point clouds
- Better compiler optimization with restrict/const
- Cleaner code structure
