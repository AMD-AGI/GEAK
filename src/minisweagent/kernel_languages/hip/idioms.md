# HIP — idioms reference

Short reference appended near the top of the task body so the agent
has "what HIP code looks like here" as context.

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

## pybind11 / torch-extension wrapper

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

## MFMA block

```cpp
__device__ void dot16x16x16_fp16(
    const half16& a, const half16& b, float4& acc
) {
    acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);
}
```
