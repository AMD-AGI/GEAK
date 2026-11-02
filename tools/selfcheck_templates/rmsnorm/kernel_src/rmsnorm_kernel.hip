// EDITABLE optimization surface: the RMSNorm GPU kernel + launch wrapper.
// GEAK may rewrite the kernel body, tiling, reduction, block size, vectorization
// — anything — as long as `rmsnorm_launch` (declared in include/rmsnorm_op.h)
// keeps its name + signature and the result stays numerically correct.
#include <hip/hip_runtime.h>
#include "rmsnorm_op.h"

// One block per row; block-stride reduction in shared memory. RMSNorm needs a
// single reduction over x[i]^2 (no mean subtraction, no bias term).
__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               const float* __restrict__ gamma,
                               float* __restrict__ y, int N, float eps) {
  extern __shared__ float smem[];  // [blockDim.x] scratch
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const float* xr = x + (size_t)row * N;
  float* yr = y + (size_t)row * N;

  float lsum = 0.f;
  for (int i = tid; i < N; i += blockDim.x) {
    float v = xr[i];
    lsum += v * v;
  }
  smem[tid] = lsum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid + s];
    __syncthreads();
  }
  const float inv = rsqrtf(smem[0] / N + eps);

  for (int i = tid; i < N; i += blockDim.x)
    yr[i] = xr[i] * inv * gamma[i];
}

extern "C" void rmsnorm_launch(const float* d_x, const float* d_gamma,
                               float* d_y, int M, int N, float eps) {
  const int threads = 256;
  const size_t shmem = threads * sizeof(float);
  hipLaunchKernelGGL(rmsnorm_kernel, dim3(M), dim3(threads), shmem, 0, d_x,
                     d_gamma, d_y, N, eps);
}
