# Kernel: emb_segment_reduce_forward (Simplified)

## Variant Context
- Input: Segmented embeddings with offsets
- Datatype: fp32
- Architecture: Generic HIP/AMD GPU

## Key Optimizations

1. **Precomputed MEAN Scaling**: Computes `w_scale = 1.0 / length` once per segment. Multiplies instead of dividing per element.

2. **Early Exit for Empty Segments**: Checks `if (length <= 0) { continue; }` to skip empty segments.

3. **Dimension-First Tiling**: Outer loop over dimension D, inner loop over indices. Improves locality for weight/reverse_indices access.

4. **Const Qualifiers**: Added `const` to start, end, length for compiler optimization.

## Performance Impact
- Reduced division operations in MEAN mode
- Skips unnecessary work for empty segments
- Better memory access patterns with dimension-first iteration
