// MVP: simple LayerNorm, GPU (HIP) vs CPU, timed + verified.
// Op: y = (x - mean) / sqrt(var + eps) * gamma + beta, per row of length N.
// Build: hipcc -O3 -std=c++17 --offload-arch=gfx1151 layernorm_mvp.hip -o layernorm_mvp
// Run:   ./layernorm_mvp [rows] [cols] [iters]
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

#define HIP_CHECK(e)                                                        \
  do {                                                                      \
    hipError_t _e = (e);                                                    \
    if (_e != hipSuccess) {                                                 \
      std::fprintf(stderr, "HIP error %s at %s:%d\n",                       \
                   hipGetErrorString(_e), __FILE__, __LINE__);             \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

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

static void layernorm_cpu(const float* x, const float* gamma, const float* beta,
                          float* y, int M, int N, float eps) {
  for (int r = 0; r < M; ++r) {
    const float* xr = x + (size_t)r * N;
    float* yr = y + (size_t)r * N;
    float mean = 0.f;
    for (int i = 0; i < N; ++i) mean += xr[i];
    mean /= N;
    float var = 0.f;
    for (int i = 0; i < N; ++i) { float d = xr[i] - mean; var += d * d; }
    var /= N;
    float inv = 1.f / std::sqrt(var + eps);
    for (int i = 0; i < N; ++i) yr[i] = (xr[i] - mean) * inv * gamma[i] + beta[i];
  }
}

int main(int argc, char** argv) {
  int M = argc > 1 ? std::atoi(argv[1]) : 4096;
  int N = argc > 2 ? std::atoi(argv[2]) : 2880;
  int iters = argc > 3 ? std::atoi(argv[3]) : 50;
  const float eps = 1e-5f;

  std::printf("LayerNorm MVP  rows=%d cols=%d iters=%d\n", M, N, iters);

  std::vector<float> x((size_t)M * N), g(N), b(N), y_gpu((size_t)M * N),
      y_cpu((size_t)M * N);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = ((i * 1103515245u + 12345u) & 0xffff) / 32768.f - 1.f;
  for (int i = 0; i < N; ++i) { g[i] = 1.f + (i % 7) * 0.01f; b[i] = (i % 5) * 0.02f; }

  float *dx, *dg, *db, *dy;
  HIP_CHECK(hipMalloc(&dx, x.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&dg, N * sizeof(float)));
  HIP_CHECK(hipMalloc(&db, N * sizeof(float)));
  HIP_CHECK(hipMalloc(&dy, x.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dg, g.data(), N * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(db, b.data(), N * sizeof(float), hipMemcpyHostToDevice));

  int threads = 256;
  size_t shmem = threads * sizeof(float);

  // warmup + correctness
  hipLaunchKernelGGL(layernorm_kernel, dim3(M), dim3(threads), shmem, 0, dx, dg,
                     db, dy, N, eps);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(y_gpu.data(), dy, x.size() * sizeof(float), hipMemcpyDeviceToHost));

  auto c0 = std::chrono::high_resolution_clock::now();
  layernorm_cpu(x.data(), g.data(), b.data(), y_cpu.data(), M, N, eps);
  auto c1 = std::chrono::high_resolution_clock::now();
  double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();

  double max_abs = 0.0;
  for (size_t i = 0; i < y_gpu.size(); ++i)
    max_abs = std::fmax(max_abs, std::fabs((double)y_gpu[i] - y_cpu[i]));
  bool ok = max_abs < 2e-3;

  // GPU timing
  hipEvent_t s, e;
  HIP_CHECK(hipEventCreate(&s));
  HIP_CHECK(hipEventCreate(&e));
  HIP_CHECK(hipEventRecord(s));
  for (int it = 0; it < iters; ++it)
    hipLaunchKernelGGL(layernorm_kernel, dim3(M), dim3(threads), shmem, 0, dx,
                       dg, db, dy, N, eps);
  HIP_CHECK(hipEventRecord(e));
  HIP_CHECK(hipEventSynchronize(e));
  float gpu_total = 0.f;
  HIP_CHECK(hipEventElapsedTime(&gpu_total, s, e));
  double gpu_ms = gpu_total / iters;

  std::printf("Verify: max_abs_err=%.3e  %s\n", max_abs, ok ? "PASS" : "FAIL");
  std::printf("CPU:  %.4f ms  (single pass)\n", cpu_ms);
  std::printf("GPU:  %.4f ms  (median-of-%d avg)\n", gpu_ms, iters);
  std::printf("Speedup (CPU/GPU): %.1fx\n", cpu_ms / gpu_ms);

  hipFree(dx); hipFree(dg); hipFree(db); hipFree(dy);
  return ok ? 0 : 1;
}
