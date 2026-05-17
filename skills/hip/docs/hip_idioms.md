# HIP — Idioms Reference

Short reference of "what HIP code looks like here" — used to identify
the kernel entry point, the launcher, and (for pybind11 wrappers) the
Python-callable function the harness imports.

Lifted from `src/minisweagent/kernel_languages/hip/idioms.md` and
`src/minisweagent/kernel_languages/hip/builder_hints.md`.

---

## Universal harness contract (same on every language)

The generated harness must expose argparse with these flags:

```
--correctness       run correctness against a reference
--benchmark         time the kernel; print `GEAK_RESULT_LATENCY_MS=<float>`
--full-benchmark    time + verify; print `GEAK_RESULT_SPEEDUP=<float>`
--profile           run under the profiler with device_launch capture
```

---

## Kernel shape

```cpp
#include <hip/hip_runtime.h>

__global__ void my_kernel(const float* __restrict__ x,
                          const float* __restrict__ y,
                          float* __restrict__ out,
                          int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + y[idx];
    }
}
```

## Launch shape

```cpp
dim3 block(256);
dim3 grid((n + block.x - 1) / block.x);
hipLaunchKernelGGL(my_kernel, grid, block, 0, 0, x, y, out, n);
```

Identify the kernel by the `__global__ void` decoration; the launcher
is whatever C++ wrapper calls `hipLaunchKernelGGL` with the kernel
function as its first argument.

---

## pybind11 / torch-extension wrapper

This is the most common shape in `aiter`-style repos. The harness
imports the compiled module and calls the Python-level function.

```cpp
#include <torch/extension.h>

torch::Tensor my_op(torch::Tensor x, torch::Tensor y) {
    auto out = torch::empty_like(x);
    const int n = x.numel();
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    hipLaunchKernelGGL(my_kernel, grid, block, 0, 0,
                       x.data_ptr<float>(),
                       y.data_ptr<float>(),
                       out.data_ptr<float>(), n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_op", &my_op);
}
```

The harness imports the compiled `TORCH_EXTENSION_NAME` module and
calls `mod.my_op(*_inputs)`.

---

## MFMA intrinsic (context only — the harness does not edit kernel code)

```cpp
__device__ void dot16x16x16_fp16(
    const half16& a, const half16& b, float4& acc
) {
    acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);
}
```

---

## Reference selection (correctness oracle)

- For **pybind11 wrappers**, the same wrapper usually exposes a
  reference implementation (a PyTorch fallback). Use it directly.
- For **make + ./bench** or **raw hipcc** shapes, the user test file
  contains either a CPU reference or a separate validation run.
  Preserve that path — don't reinvent.

---

## Timing loop

- Warm up 5 iterations before the timed run; measure 100 and take the
  **median** (not mean).
- `hipDeviceSynchronize()` before AND after each measurement.
- For pybind11 shapes, `torch.cuda.Event(enable_timing=True)` is fine
  and integrates cleanly with the harness's `torch.cuda.synchronize()`.
