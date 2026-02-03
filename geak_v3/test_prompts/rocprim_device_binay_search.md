I use `rocPRIM_device_binary_search/benchmark/benchmark_device_binary_search.cpp` to test `rocPRIM_device_binary_search/rocprim/include/rocprim/device/device_binary_search.hpp` performance. But the performance is too bad (low bandwidth).

1. You must find the all files related to `rocPRIM_device_binary_search/rocprim/include/rocprim/device/device_binary_search.hpp`. And review and edit it.
2. You MUST edit all files related to `rocPRIM_device_binary_search/rocprim/include/rocprim/device/device_binary_search.hpp`.
3. The files in `benchmark` is NOT allowed to be edited.
4. The files in `test` is NOT allowed to be edited.
5. The file `test_benchmark.py` is forbidden to be edited.
6. All CMAKEList and CMAKE files are forbidden to be edited.
7. You can modify multiple files at once.
8. Before Action `submit`, You MUST run the Performance test.
9. When the performance does not reach 1.2 times the baseline, further optimization must be carried out and Do not take Action `submit`. Use the average over bytes_per_second of all datatypes as the metric.
10. Your edit should not effect the compile of other kernels

## Test Perf
1. Baseline: Before changing any code, you should run baseline numbers.
2. Test performance: run `python /lvm-data/data/yueliu14/clean_miniswe/test_scripts/test_benchmark.py benchmark_device_binary_search /lvm-data/data/yueliu14/clean_miniswe/rocPRIM_device_binary_search`