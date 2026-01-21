I use `WORK_REPO/benchmark/benchmark_block_run_length_decode.cpp` to test `WORK_REPO/rocprim/include/rocprim/block/block_run_length_decode.hpp` performance. But the performance is too bad (low bandwidth).

1. You must find the all files related to `WORK_REPO/rocprim/include/rocprim/block/block_run_length_decode.hpp`. And review and edit it.
2. You MUST edit all files related to `WORK_REPO/rocprim/include/rocprim/block/block_run_length_decode.hpp`.
3. The files in `benchmark` is NOT allowed to be edited.
4. The files in `test` is NOT allowed to be edited.
5. The file `/mnt/data/yueliu14/ready_mini_swe/mini-swe-agent/test_scripts/test_correctness_benchmark.py` is forbidden to be edited.
6. All CMAKEList and CMAKE files are forbidden to be edited.
7. You can modify multiple files at once.
8. Before Action `submit`, You MUST run the Performance test.
9. When the performance does not reach 1.2 times the baseline, further optimization must be carried out and Do not take Action `submit`. Use the average over bytes_per_second of all datatypes as the metric.
10. Your edit should not effect the compile of other kernels
 
## Test Perf
1. Baseline: Before changing any code, you should run baseline numbers.
2. Optimized test: After changing, you should test the code. If fail to pass the correctness test, MUST debug the kernel carefully. If run correctness and performance test successfully, you can get the bandwidth (bytes_per_second) of the kernel under different input key type.
