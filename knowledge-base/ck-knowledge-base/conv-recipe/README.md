# CK Harness Recipe: 2D Grouped Convolution Forward

## Overview

This recipe demonstrates the shared-library test harness for CK convolution
kernels using the **Array API** pattern. The kernel computes grouped 2D
convolution forward (GNHWC layout).

This is the pattern that GEAK failed to produce on output51 — the wrapper
must use `ConvParam` and CK's `make_*_packed` helper functions to build
correctly-typed `std::array` arguments, not hand-roll strides from scalars.

## Kernel Family

- Device type: `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle`
- Header: `ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp`
- CK examples: `09_convnd_fwd/convnd_fwd_xdl_fp16.cpp`
- API style: `MakeArgument` (value-based, typed array parameters)

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
    const void* p_in, const void* p_wei, void* p_out,
    int64_t G, int64_t N, int64_t K, int64_t C,
    int64_t Hi, int64_t Wi,
    int64_t Y, int64_t X,
    int64_t stride_h, int64_t stride_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t pad_h, int64_t pad_w,
    bool time_kernel, int warmup, int nrepeat)
```

Flat scalar args for convolution geometry. The wrapper internally reconstructs
`ConvParam` and uses CK helper functions to build typed `std::array` arguments.

## Critical Wrapper Pattern

The key to making convolution wrappers compile is correctly building `std::array`
descriptor arguments in CK's canonical dimension order. The `ConvParam` +
`make_*_packed` helper approach requires linking against CK's library source, so
this recipe computes the arrays inline from flat scalar arguments instead.

CK's canonical dimension order for NDimSpatial=2:
- Input:  `G, N, C, H, W`  (a_lengths, a_strides)
- Weight: `G, K, C, Y, X`  (b_lengths, b_strides)
- Output: `G, N, K, Ho, Wo` (e_lengths, e_strides)

Strides must reflect the physical packed layout (GNHWC, GKYXC, GNHWK):

```cpp
// Input GNHWC packed -> canonical G,N,C,H,W
std::array<idx, 5> a_lengths = {G, N, C, Hi, Wi};
std::array<idx, 5> a_strides = {
    N*Hi*Wi*C, Hi*Wi*C, 1, Wi*C, C};  // C is innermost

// Weight GKYXC packed -> canonical G,K,C,Y,X
std::array<idx, 5> b_lengths = {G, K, C, Y, X};
std::array<idx, 5> b_strides = {
    K*Y*X*C, Y*X*C, 1, X*C, C};      // C is innermost

// Output GNHWK packed -> canonical G,N,K,Ho,Wo
std::array<idx, 5> e_lengths = {G, N, K, Ho, Wo};
std::array<idx, 5> e_strides = {
    N*Ho*Wo*K, Ho*Wo*K, 1, Wo*K, K};  // K is innermost
```

For the spatial parameters (conv strides, dilations, pads), use `std::array<idx, 2>`.

Do NOT try to hand-roll alternative orderings — the canonical order above must match
what `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle::MakeArgument` expects.

## Adapting for Variants

### Conv + Reduce (example 10, DeviceGroupedConvFwdMultipleDMultipleR)
- Add R output pointers: `std::array<void*, NumR>{p_r0}`
- R lengths/strides use `NDimSpatial+2` arrays (G, N, Ho, Wo — no K dimension)
- Initialize R buffers before launch: `SetValue(ck::NumericLimits<R0DataType>::Lowest())`
- Add QsElementOp, RsElementOp, RsThreadReduceOp, RsGlobalReduceOp

### Conv + Bias (example 30, same DeviceOp with non-empty D)
- Use `std::array<const void*, NumD>{p_d0, ...}` for D pointers
- D lengths/strides arrays: `std::array<std::array<index_t, NDimSpatial+3>, NumD>`
- Use `ck::Tuple<DLayout>` and `ck::Tuple<DDataType>` in the DeviceInstance typedef

### Backward Data (example 38, DeviceGroupedConvBwdDataMultipleD)
- Different device op header
- Input/output tensor roles swapped

### Backward Weight (example 20, DeviceGroupedConvBwdWeight)
- Different device op header
- May require `split_k` parameter

## Directory structure

The files must be on the same level as the target kernel. The generated test harness will be used by
optimizer agents which are given the original target kernel location. The optimizer agents will make a
worktree copy (GEAK_WORK_DIR) of the target kernel folder and make edits, compile and test in the worktree folder.

For the example target kernel the original directory is `rocm-libraries/projects/composablekernel/example/09_convnd_fwd/convnd_fwd_xdl_fp16.cpp`.

Therefore, your generated files (excluding test harness) must be in:

```
rocm-libraries/
└── projects/
    └── composablekernel/
        └─── example/
            └── 09_convnd_fwd/
                │   convnd_fwd_xdl_fp16.cpp (Target kernel already exist, it's not generated.)
                │   baseline.cpp
                │   optimized.cpp
                │   compile.py
                │   Makefile
                │   ...
```

The optimizer agent's output directory, including the worktree will look like:

```
example_run
├── worktrees
│   └── agent_0
│       │   convnd_fwd_xdl_fp16.cpp
│       │   baseline.cpp
│       │   optimized.cpp
│       │   compile.py
│       │   Makefile
│       │   ...
├── test_harness.py
│   ...
```