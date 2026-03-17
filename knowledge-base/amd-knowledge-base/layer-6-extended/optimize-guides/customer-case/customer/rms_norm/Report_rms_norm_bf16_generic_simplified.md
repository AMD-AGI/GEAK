# Kernel: rms_norm, fusedQkRmsNorm RMS normalization

## Variant: bf16 (bfloat16), QK RMS Normalization, Generic HIP/AMD GPU (wave64)

## Key Optimizations

1. **Vectorized bf16x2 Memory Access**: Use uint32_t to load/store 2 bf16 values per transaction via union type, doubling memory bandwidth utilization.
  
2. **Fully Unrolled Warp Reduction**: Replace templated loop-based reduction with 6 explicit `__shfl_xor` operations for wave64, eliminating loop overhead.
  
3. **Eliminated Shared Memory for Scale**: Remove shared memory and sync barrier by computing scale directly after warp reduction (all threads have same value).
  
4. **Fast Reciprocal Square Root**: Use `__frsqrt_rn()` hardware intrinsic instead of `rsqrtf()` for faster SFU execution.
  
5. **Launch Bounds Annotation**: Added `__launch_bounds__(64)` to enable compiler register allocation optimization.
  
6. **Loop Elimination**: Removed iteration loops by assuming fixed norm_size=128 with vectorized 2-element processing per thread.
  

## Performance Impact

* 2x memory bandwidth via bf16 vectorization
* Reduced instruction count through unrolling and loop elimination
* Eliminated shared memory latency and sync overhead
* Hardware-accelerated math operations