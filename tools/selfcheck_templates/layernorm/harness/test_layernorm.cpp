// FROZEN harness: CPU reference + timing driver. Do NOT edit (GEAK optimizes
// kernel_src/, not this). Calls layernorm_launch (frozen ABI), times it on the
// GPU, computes an independent CPU reference, verifies, prints parseable output.
//
// Usage: ./test_layernorm <M>x<N> [iters]
// Output (one shape per run):
//   --- 4096x2880 ---
//   Verify: max_abs_err=2.1e-06  PASS
//   CPU: 11.8397 ms
//   GPU: 0.5242 ms
//   Speedup: 22.6x
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include "layernorm_op.h"

#define HIP_CHECK(e)                                                       \
  do {                                                                     \
    hipError_t _e = (e);                                                   \
    if (_e != hipSuccess) {                                                \
      std::fprintf(stderr, "HIP error %s at %s:%d\n",                      \
                   hipGetErrorString(_e), __FILE__, __LINE__);            \
      std::exit(2);                                                        \
    }                                                                      \
  } while (0)

static void layernorm_cpu(const float* x, const float* g, const float* b,
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
    for (int i = 0; i < N; ++i) yr[i] = (xr[i] - mean) * inv * g[i] + b[i];
  }
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s MxN [iters]\n", argv[0]); return 2; }
  int M = 0, N = 0;
  if (std::sscanf(argv[1], "%dx%d", &M, &N) != 2 || M <= 0 || N <= 0) {
    std::fprintf(stderr, "bad shape '%s'\n", argv[1]); return 2;
  }
  int iters = argc > 2 ? std::atoi(argv[2]) : 50;
  const float eps = 1e-5f;

  std::printf("--- %dx%d ---\n", M, N);

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

  // warmup + correctness snapshot
  layernorm_launch(dx, dg, db, dy, M, N, eps);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(y_gpu.data(), dy, x.size() * sizeof(float), hipMemcpyDeviceToHost));

  auto c0 = std::chrono::high_resolution_clock::now();
  layernorm_cpu(x.data(), g.data(), b.data(), y_cpu.data(), M, N, eps);
  auto c1 = std::chrono::high_resolution_clock::now();
  double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();

  double max_abs = 0.0;
  for (size_t i = 0; i < y_gpu.size(); ++i)
    max_abs = std::fmax(max_abs, std::fabs((double)y_gpu[i] - y_cpu[i]));
  bool ok = std::isfinite(max_abs) && max_abs < 2e-3;

  hipEvent_t s, e;
  HIP_CHECK(hipEventCreate(&s));
  HIP_CHECK(hipEventCreate(&e));
  HIP_CHECK(hipEventRecord(s));
  for (int it = 0; it < iters; ++it) layernorm_launch(dx, dg, db, dy, M, N, eps);
  HIP_CHECK(hipEventRecord(e));
  HIP_CHECK(hipEventSynchronize(e));
  float gpu_total = 0.f;
  HIP_CHECK(hipEventElapsedTime(&gpu_total, s, e));
  double gpu_ms = gpu_total / iters;

  std::printf("Verify: max_abs_err=%.3e  %s\n", max_abs, ok ? "PASS" : "FAIL");
  std::printf("CPU: %.4f ms\n", cpu_ms);
  std::printf("GPU: %.4f ms\n", gpu_ms);
  std::printf("Speedup: %.1fx\n", cpu_ms / gpu_ms);

  hipFree(dx); hipFree(dg); hipFree(db); hipFree(dy);
  return ok ? 0 : 1;
}
