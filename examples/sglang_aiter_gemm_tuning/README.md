run_sglang_test.sh is an example input script for a GEMM tuning workload.

You can start the optimization with:
```bash
geak-gemm-tuning -t "Optimize the E2E performance of the workload via GEMM tuning. The benchmark script is run_sglang_test.sh"
```

After the optimization is completed, the reproduction instructions can be found in:
```text
optimization_logs/gemm_tuning_20260515_082539_514549/final_report.json
```
