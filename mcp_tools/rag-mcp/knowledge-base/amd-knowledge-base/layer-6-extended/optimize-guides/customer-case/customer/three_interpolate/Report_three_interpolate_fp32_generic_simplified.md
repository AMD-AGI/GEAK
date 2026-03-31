# Kernel: three_interpolate (Simplified)

## Variant Context
- Input: Point features (B, C, M), weights/indices (B, N, 3)
- Datatype: fp32
- Architecture: Generic HIP/AMD GPU

## Key Optimizations

1. **Grid-Stride Loop**: Each thread processes multiple points via `for (pt_idx = tid; pt_idx < n; pt_idx += stride)`. Better CU utilization for large N.

2. **Rolling Offset Updates**: Maintains `w_off`, `i_off`, `out_off` incremented by fixed steps each iteration. Avoids per-iteration `pt_idx * 3` multiplication.

3. **64-bit Base Pointers**: Uses `static_cast<size_t>()` for base pointer computation to prevent overflow with large tensors.

4. **Restrict Base Pointers**: Precomputes `points_bc`, `weight_bc`, `idx_bc`, `out_bc` per (B,C) with `__restrict__`.

## Performance Impact
- Grid-stride improves load balancing and occupancy
- Rolling offsets reduce integer arithmetic
- Prevents overflow for large point clouds
