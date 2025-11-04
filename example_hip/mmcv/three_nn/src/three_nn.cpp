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


void three_nn_wrapper(int b, int n, int m, at::Tensor unknown_tensor,
                      at::Tensor known_tensor, at::Tensor dist2_tensor,
                      at::Tensor idx_tensor);

void three_nn_kernel_launcher(int b, int n, int m, const float *unknown,
                              const float *known, float *dist2, int *idx,
                              cudaStream_t stream);


void three_nn_wrapper(int b, int n, int m, at::Tensor unknown_tensor,
                      at::Tensor known_tensor, at::Tensor dist2_tensor,
                      at::Tensor idx_tensor) {
  const float *unknown = unknown_tensor.data_ptr<float>();
  const float *known = known_tensor.data_ptr<float>();
  float *dist2 = dist2_tensor.data_ptr<float>();
  int *idx = idx_tensor.data_ptr<int>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  three_nn_kernel_launcher(b, n, m, unknown, known, dist2, idx, stream);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("three_nn_wrapper", &three_nn_wrapper, "three_nn_wrapper");
}
