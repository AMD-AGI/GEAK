# CK Harness Recipe: GEMM + Bias

## Overview

This recipe demonstrates the shared-library test harness for CK GEMM kernels
using the **Scalar API** pattern. The kernel computes `E = A * B + D` where D
is a per-row bias vector broadcast along M.

## Kernel Family

- Device type: `DeviceGemmMultipleD_Xdl_CShuffle`
- Header: `ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp`
- CK examples: `03_gemm_bias_relu/gemm_bias_relu_xdl_fp16.cpp`
- API style: `MakeArgument` (value-based, scalar parameters)

## Architecture

| File | Role | Mutable? |
|------|------|----------|
| `baseline.cpp` | Compiled to `libbaseline.so` — frozen ground truth | Never |
| `optimized.cpp` | Compiled to `liboptimized.so` — LLM edits TUNING PARAMETERS | Each iteration |
| `test_harness.py` | Loads both `.so` via ctypes, verifies, benchmarks | Never |
| `compile.py` | Auto-detects GPU arch from PyTorch, calls `make` | Never |
| `Makefile` | `hipcc -shared -fPIC -O3` build rules | Never |

## C ABI Signature

```cpp
extern "C" float run_kernel(
    const void* p_a, const void* p_b, const void* p_d0, void* p_e,
    int64_t M, int64_t N, int64_t K,
    int64_t StrideA, int64_t StrideB, int64_t StrideD0, int64_t StrideE,
    bool time_kernel, int warmup, int nrepeat)
```

- A: (M, K) row-major, StrideA = K
- B: (K, N) column-major, StrideB = K
- D: (1, N) bias, StrideD0 = 0 (broadcast along M)
- E: (M, N) row-major output, StrideE = N

## MakeArgument Pattern

```cpp
auto argument = device_op.MakeArgument(
    p_a, p_b,
    std::array<const void*, 1>{p_d0},
    p_e,
    M, N, K,
    StrideA, StrideB,
    std::array<ck::index_t, 1>{StrideD0},
    StrideE,
    AElementOp{}, BElementOp{}, CDEElementOp{});
```

## Adapting for Variants

### Pure GEMM (0 D tensors)
Use `DeviceGemm_Xdl_CShuffle` or `DeviceGemmMultipleD_Xdl_CShuffle` with empty D:
- Replace `std::array<const void*, 1>{p_d0}` with `std::array<const void*, 0>{}`
- Replace `std::array<ck::index_t, 1>{StrideD0}` with `std::array<ck::index_t, 0>{}`
- Use `ck::Tuple<>` for DsLayout and DsDataType
- Remove p_d0 and StrideD0 from the C ABI

### GEMM + 2 D tensors (bias + residual)
- Use `std::array<const void*, 2>{p_d0, p_d1}`
- Use `ck::Tuple<D0Layout, D1Layout>` and `ck::Tuple<D0DataType, D1DataType>`

### GEMM + Reduce (DeviceGemmMultipleDMultipleR_Xdl_CShuffle)
- Add R output pointers: `std::array<void*, NumR>{p_r0, ...}`
- Add QsElementOp, RsElementOp, RsThreadReduceOp, RsGlobalReduceOp
- Initialize R buffers before launch (numeric lowest for Max, zero for Sum)
