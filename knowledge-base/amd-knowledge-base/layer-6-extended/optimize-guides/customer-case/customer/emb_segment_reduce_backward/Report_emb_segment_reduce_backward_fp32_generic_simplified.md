# Kernel: emb_segment_reduce_backward (Simplified)

## Variant Context
- Input: Segmented embedding gradients
- Datatype: fp32
- Architecture: Generic HIP/AMD GPU

## Key Optimizations

1. **LDS Caching for Segment Gradients**: In SUM/MEAN modes, cooperatively loads `grad_output[s*D:s*D+D]` into shared memory once per segment. All threads then read from LDS instead of global memory. Effective when D≤4096.

2. **Precomputed Stride Decomposition**: Computes `step_rows = step/D` and `step_rem = step%D` once before loop. Uses incremental updates `dp += step_rem; idx += step_rows` with conditional wrap-around, avoiding per-iteration division/modulo.

3. **Precomputed Reciprocal (MEAN)**: Computes `inv_len = 1.0/length` once per segment. Multiplies `w_base * inv_len` instead of dividing per element.

4. **Double Pack Processing**: Processes two vector packs per loop iteration when possible, increasing ILP and reducing loop overhead.

5. **Restrict Pointer**: Added `__restrict__` to output pointer enabling compiler optimizations.

## Performance Impact
- LDS caching reduces global memory bandwidth significantly for SUM/MEAN modes
- Eliminates expensive integer division/modulo from hot loop
- Better instruction-level parallelism with double packing
