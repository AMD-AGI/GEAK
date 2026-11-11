---
kind: landscape
title: Autotuning tools and records
updated: 2026-10-16
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, fp4_e2m1]
regimes: [prefill, decode, both]
---

# Autotuning tools and records

Select a tuner that reaches the actual callable. Library algorithm selection,
parameter search and code generation solve different problems; a tuned CSV does
not help if the runtime never reads it.

| Tool | What it tunes | Details to preserve |
|---|---|---|
| Triton `@triton.autotune` | Configurations for an authored kernel | Key covers behavior-affecting inputs; `reset_to_zero`/restore hooks matter for mutating kernels |
| PyTorch TunableOp | Supported library operations | Enable/tuning controls, CSV validator headers and exact library/device versions |
| hipBLASLt offline tuning | Available matmul solutions | Shape, transpose, dtype, workspace, output database and runtime selection |
| TensileLite | Generated library solutions | Build/search cost and solution compatibility, distinct from selecting installed kernels |
| CK profiler | Registered CK instances | `IsSupportedArgument`, layout/type specialization and compiled instance coverage |
| aiter gradlib | Per-shape dispatch records | The installed CSV key and `AITER_CONFIG_*` consumer; do not invent a universal key or error gate |
| TileLang autotune | Generated kernel configurations | Target/backend, correctness checks and cache identity |
| TVM MetaSchedule | Schedule search and tuning records | Workload/target matching and database selection |
| Kernel Tuner | Parameter search strategies | Validity constraints, output verification and supported execution backend |

Retain source/build/device identity, the complete lookup key, tested configurations,
correctness result and timing method. Refresh records when relevant source or
compiler behavior changes. Reusing a record and confirming it applies can be cheaper
than retuning. A persistent cache, heuristic or learned cost model proposes choices;
measure finalists and preserve fallback on unsupported cases.

For the repository's GEMM procedure, see [GEMM tuning](../workflows/gemm_tuning_workflow.md).
Keep search cost separate from the accepted kernel timing. Benchmark stateful
candidates with valid resets; repeated invocation can otherwise reward stale state.

## Sources

- [Triton autotune API](https://triton-lang.org/main/python-api/generated/triton.autotune.html)
- [PyTorch TunableOp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md)
- [hipBLASLt offline tuning](https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/how-to/how-to-use-hipblaslt-offline-tuning.html)
- [CK profiler](https://github.com/ROCm/composable_kernel/blob/develop/profiler/README.md)
- [aiter](https://github.com/ROCm/aiter)
- [TileLang tuning](https://tilelang.com/tutorials/auto_tuning.html)
- [TVM MetaSchedule](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)
- [Kernel Tuner](https://kerneltuner.github.io/kernel_tuner/stable/optimization.html)
