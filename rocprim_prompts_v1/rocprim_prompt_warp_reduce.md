I use `WORK_REPO/benchmark/benchmark_warp_reduce.cpp` to test `WORK_REPO/rocprim/include/rocprim/warp/warp_reduce.hpp` performance. But the performance is too bad (low bandwidth).

1. You must find the all files related to `WORK_REPO/rocprim/include/rocprim/warp/warp_reduce.hpp`. And review and edit it.
2. You MUST edit all files related to `WORK_REPO/rocprim/include/rocprim/warp/warp_reduce.hpp`.
3. The files in `benchmark` is NOT allowed to be edited.
4. The files in `test` is NOT allowed to be edited.
5. The file `test_scripts/test_correctness_benchmark.py` is forbidden to be edited.
6. All CMAKEList and CMAKE files are forbidden to be edited.
7. You can modify multiple files at once.
8. Before Action `submit`, You MUST run the Performance test.
9. When the performance does not reach 1.2 times the baseline, further optimization must be carried out and Do not take Action `submit`.
10. Your edit should not effect the compile of other kernels


**Test Performance**
1. Baseline: Before changing any code, you should run baseline numbers.
2. Optimized test: After changing, you should test the code. If fail to pass the correctness test, MUST debug the kernel carefully. If run correctness and performance test successfully, you can get the bandwidth (bytes_per_second) of the kernel under different input key type.
 
**Optimization Guidelines**
Before modifying any code, you must perform the following steps:

1. **Automatically read the current GPU hardware information** (such as architecture model, number of SMs, maximum threads per SM, shared memory size, register count, memory bandwidth, etc.).
   - Recommended to use `rocminfo`, `rocm-smi`, or directly query the CUDA/HIP runtime API.
2. **Based on hardware characteristics, and considering the nature of the block_histogram algorithm, propose multiple possible optimization directions and provide a brief analysis for each.** These optimization directions should include, but are not limited to:
   - Increasing GPU occupancy
   - Maximizing memory bandwidth utilization
   - Optimizing shared memory access patterns to reduce bank conflicts
   - Adaptive tuning of warp/block parameters for different architectures (e.g., GFX9, GFX10, etc.)
   - Efficient allocation of registers/shared memory/local memory
   - Using pipelining or prefetching techniques to hide memory latency
   - Any other innovative, architecture-specific or parallel-level optimizations targeting the hardware
3. **Assign a priority to each optimization direction** (ranking them according to the potential performance gain), and **experiment with them one by one in order of priority**.

Final workflow:
- First, output all key hardware information (in a table or list, clearly readable)
- Output a list of hardware-targeted optimization directions, with explanations and expected benefits, and an assigned priority
- Only after analyzing hardware and formulating optimization strategies, begin specific code optimizations. Try one at a time, test performance, and then proceed to the next, until the target performance improvement is achieved

This aims to ensure you perform systematic, hardware-adaptive optimization rather than simply applying generic practices.