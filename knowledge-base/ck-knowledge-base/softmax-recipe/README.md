# CK Harness Recipe: Softmax

## Overview

This folder represents an example of the files needed to create a build system and test harness
for optimizing Composable Kernel (CK) kernels.
This example targets the specific kernel `ck/tensor_operation/gpu/device/impl/device_softmax_impl.hpp`. It must be adapted to work for other target kernels

## Kernel Family

- Device type: `DeviceSoftmaxImpl`
- Header: `ck/tensor_operation/gpu/device/impl/device_softmax_impl.hpp`
- CK examples: `23_softmax/softmax_blockwise.cpp`
- API style: `MakeArgumentPointer` (pointer-based)

## Architecture

The harness consists of 5 files + local copy of the target kernel. The C++ kernels are compiled to shared libraries
(`.so`) with `hipcc` and loaded into a Python test script via `ctypes`. The baseline
kernel output is ground truth; the optimized kernel is verified against it.

| File | Role | Mutable? |
|------|------|----------|
| `device_softmax_impl.hpp`| Target kernel to optimize -- LLM edits | Each iteration |
| `baseline.cpp` | Compiled to `libbaseline.so` -- frozen ground truth | Never |
| `optimized.cpp` | Compiled to `liboptimized.so` -- LLM edits TUNING PARAMETERS | Each iteration |
| `test_harness.py` | Loads both `.so` via ctypes, verifies, benchmarks | Never |
| `compile.py` | Auto-detects GPU arch from PyTorch, calls `make` | Never |
| `Makefile` | `hipcc -shared -fPIC -O3` build rules | Never |

## Tunable Template Parameters

| Parameter | Position | Type | Constraints |
|-----------|----------|------|-------------|
| BlockSize | 7 | int | Power of 2: 64, 128, 256, 512, 1024 |
| ClusterM | 8 | int | ClusterM * ClusterK must equal BlockSize |
| ClusterK | 9 | int | ClusterM * ClusterK must equal BlockSize |
| SliceM | 10 | int | >= 1 |
| SliceK | 11 | int | Must align with vector size |
| SrcVecDim | 12 | int | 0 = vectorize along M, 1 = vectorize along K |
| SrcScalarPerVector | 13 | int | Power of 2: 1, 2, 4, 8; must divide tensor dim |
| OutScalarPerVector | 14 | int | Power of 2: 1, 2, 4, 8; must divide tensor dim |

Not all combinations are valid. `IsSupportedArgument` returns false for invalid
configs; the harness reports these as UNSUPPORTED (return value -1.0f).

## Adapting for Other Kernels

When adapting this recipe for a different CK kernel family, change three things:

1. **C ABI signature** in both `.cpp` files -- match the kernel's input/output arguments
2. **`load_kernel()` and `call_kernel()`** in `test_harness.py` -- mirror the new C ABI with ctypes
3. **Shape lists** -- replace with shapes appropriate for the kernel family

Everything else (`compile.py`, `Makefile`, 4-mode structure, `_gpu_warmup()` function and its call at 
the start of `mode_benchmark()`, baseline-vs-optimized comparison, benchmark reporting) stays identical.

## Directory structure

The files must be on the same level as the target kernel. The generated test harness will be used by
optimizer agents which are given the original target kernel location. The optimizer agents will make a
worktree copy (GEAK_WORK_DIR) of the target kernel folder and make edits, compile and test in the worktree folder.

For the example target kernel the original directory is with `rocm-libraries/projects/composablekernel/include/ck/tensor_operation/gpu/device/impl/device_softmax_impl.hpp`.

Therefore, your generated files (excluding test harness) must be in:

```
rocm-libraries/
└── projects/
    └── composablekernel/
        └─── include/
            └── ck/
                └── tensor_operation/
                    └── gpu/
                        └── device/
                            └── impl/
                                │   device_softmax_impl.hpp (Target kernel already exist, it's not generated.)
                                │   baseline.cpp
                                │   optimized.cpp
                                │   test_harness.py
                                │   compile.py
                                │   Makefile
                                │   ...
```

The optimizer agent's output directory, including the worktree will look like:

```
example_run
├── worktrees
│   └── agent_0
│       │   device_softmax_impl.hpp
│       │   baseline.cpp
│       │   optimized.cpp
│       │   compile.py
│       │   Makefile
│       │   ...
├── test_harness.py
│   ...
```