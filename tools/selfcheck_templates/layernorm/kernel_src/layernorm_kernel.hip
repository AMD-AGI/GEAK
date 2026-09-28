// EDITABLE optimization surface: the LayerNorm GPU kernel + launch wrapper.
// GEAK may rewrite the kernel body, tiling, reduction, block size, vectorization
// — anything — as long as `layernorm_launch` (declared in include/layernorm_op.h)
// keeps its name + signature and the result stays numerically correct.
#include <hip/hip_runtime.h>
#include "layernorm_op.h"

// One block per row; block-stride reduction in shared memory.
__global__ void layernorm_kernel(const float* __restrict__ x,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  float* __restrict__ y, int N, float eps) {
  extern __shared__ float smem[];  // [blockDim.x] scratch
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const float* xr = x + (size_t)row * N;
  float* yr = y + (size_t)row * N;

  float local = 0.f;
  for (int i = tid; i < N; i += blockDim.x) local += xr[i];
  smem[tid] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid + s];
    __syncthreads();
  }
  const float mean = smem[0] / N;
  __syncthreads();

  float lvar = 0.f;
  for (int i = tid; i < N; i += blockDim.x) {
    float d = xr[i] - mean;
    lvar += d * d;
  }
  smem[tid] = lvar;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid + s];
    __syncthreads();
  }
  const float inv = rsqrtf(smem[0] / N + eps);

  for (int i = tid; i < N; i += blockDim.x)
    yr[i] = (xr[i] - mean) * inv * gamma[i] + beta[i];
}

extern "C" void layernorm_launch(const float* d_x, const float* d_gamma,
                                 const float* d_beta, float* d_y, int M, int N,
                                 float eps) {
  const int threads = 256;
  const size_t shmem = threads * sizeof(float);
  hipLaunchKernelGGL(layernorm_kernel, dim3(M), dim3(threads), shmem, 0, d_x,
                     d_gamma, d_beta, d_y, N, eps);
}
