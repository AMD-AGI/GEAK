# Kernel: floyd_warshall (Simplified)

## Variant Context
- Input: Adjacency matrix (N x N)
- Datatype: int32/uint32
- Architecture: Generic HIP/AMD GPU

## Key Optimizations

1. **__ldg() Texture Cache Loads**: Uses `__ldg(&matrix[idx])` for all adjacency matrix reads. Routes through texture cache for better read-only data throughput.

2. **Const Qualifiers**: Added `const` to x, y indices and loaded values (d_x_y, d_x_k_y).

## Performance Impact
- Texture cache can improve memory throughput for read-heavy workloads
- Better compiler optimization with const hints
