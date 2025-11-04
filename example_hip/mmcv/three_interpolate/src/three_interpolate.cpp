// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/interpolate.cpp

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>



void three_interpolate_wrapper(int b, int c, int m, int n,
                               at::Tensor points_tensor, at::Tensor idx_tensor,
                               at::Tensor weight_tensor, at::Tensor out_tensor);

void three_interpolate_kernel_launcher(int b, int c, int m, int n,
                                       const float *points, const int *idx,
                                       const float *weight, float *out,
                                       cudaStream_t stream);

void three_interpolate_grad_wrapper(int b, int c, int n, int m,
                                    at::Tensor grad_out_tensor,
                                    at::Tensor idx_tensor,
                                    at::Tensor weight_tensor,
                                    at::Tensor grad_points_tensor);

void three_interpolate_grad_kernel_launcher(int b, int c, int n, int m,
                                            const float *grad_out,
                                            const int *idx, const float *weight,
                                            float *grad_points,
                                            cudaStream_t stream);

void three_interpolate_wrapper(int b, int c, int m, int n,
                               at::Tensor points_tensor, at::Tensor idx_tensor,
                               at::Tensor weight_tensor,
                               at::Tensor out_tensor) {
  const float *points = points_tensor.data_ptr<float>();
  const float *weight = weight_tensor.data_ptr<float>();
  float *out = out_tensor.data_ptr<float>();
  const int *idx = idx_tensor.data_ptr<int>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  three_interpolate_kernel_launcher(b, c, m, n, points, idx, weight, out,
                                    stream);
}

void three_interpolate_grad_wrapper(int b, int c, int n, int m,
                                    at::Tensor grad_out_tensor,
                                    at::Tensor idx_tensor,
                                    at::Tensor weight_tensor,
                                    at::Tensor grad_points_tensor) {
  const float *grad_out = grad_out_tensor.data_ptr<float>();
  const float *weight = weight_tensor.data_ptr<float>();
  float *grad_points = grad_points_tensor.data_ptr<float>();
  const int *idx = idx_tensor.data_ptr<int>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  three_interpolate_grad_kernel_launcher(b, c, n, m, grad_out, idx, weight,
                                         grad_points, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("three_interpolate_wrapper", &three_interpolate_wrapper,
        "three_interpolate_wrapper");
  m.def("three_interpolate_grad_wrapper", &three_interpolate_grad_wrapper,
        "three_interpolate_grad_wrapper");
}
