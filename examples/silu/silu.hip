// Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <functional>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
  fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); \
  std::exit(1);} } while(0)

using bf16 = __hip_bfloat16;

// ---- device helpers ----
__device__ __forceinline__ float silu_f(float x){
  return x / (1.0f + expf(-x));
}

__global__ void silu_mul_kernel(
    bf16* __restrict__ out,          // [B, H]
    const bf16* __restrict__ in,     // [B, 2H]
    int64_t B, int64_t H)
{
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < H; idx += blockDim.x) {
    const float x = __bfloat162float(in[token_idx * 2 * H + idx]);
    const float y = __bfloat162float(in[token_idx * 2 * H + H + idx]);
    out[token_idx * H + idx] = __float2bfloat16(silu_f(x) * y);
  }
}

static void fill_random(std::vector<bf16>& buf,
                        float lo=-3.f,float hi=3.f,uint32_t seed=123){
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo,hi);
  for (auto& v: buf) v = __float2bfloat16(dist(rng));
}

static void host_ref(std::vector<bf16>& out,
                     const std::vector<bf16>& in,
                     int64_t B, int64_t H){
  auto silu_h = [](double x){ return x/(1.0+std::exp(-x)); };
  for (int64_t b=0;b<B;++b){
    int64_t in_row=b*(2*H), out_row=b*H;
    for (int64_t i=0;i<H;++i){
      float x = __bfloat162float(in[in_row+i]);
      float y = __bfloat162float(in[in_row+H+i]);
      out[out_row+i] = __float2bfloat16((float)(silu_h(x)*y));
    }
  }
}

static void max_diff(const std::vector<bf16>& a,
                     const std::vector<bf16>& b,
                     double& max_abs, double& max_rel){
  max_abs=0; max_rel=0;
  for (size_t i=0;i<a.size();++i){
    double va = (double)__bfloat162float(a[i]);
    double vb = (double)__bfloat162float(b[i]);
    double ad = std::abs(va-vb);
    double rd = ad/(std::abs(vb)+1e-8);
    max_abs = std::max(max_abs, ad);
    max_rel = std::max(max_rel, rd);
  }
}

static float time_kernel_ms(std::function<void()> launch,
                            int warmup=5,int iters=100){
  hipEvent_t s,t; HIP_CHECK(hipEventCreate(&s)); HIP_CHECK(hipEventCreate(&t));
  for(int i=0;i<warmup;++i) launch();
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipEventRecord(s)); for(int i=0;i<iters;++i) launch();
  HIP_CHECK(hipEventRecord(t)); HIP_CHECK(hipEventSynchronize(t));
  float ms=0.f; HIP_CHECK(hipEventElapsedTime(&ms,s,t));
  HIP_CHECK(hipEventDestroy(s)); HIP_CHECK(hipEventDestroy(t)); return ms/iters;
}

int main(int argc, char** argv){
  int64_t B=4096, H=6400;
  for (int i=1;i<argc;++i){
    if (std::string(argv[i])=="--B" && i+1<argc) B=std::atoll(argv[++i]);
    else if (std::string(argv[i])=="--H" && i+1<argc) H=std::atoll(argv[++i]);
    else {
      printf("Usage: %s [--B <batch>] [--H <hidden>]\n", argv[0]);
      return 0;
    }
  }

  size_t in_e  = (size_t)B*(size_t)(2*H);
  size_t out_e = (size_t)B*(size_t)H;

  std::vector<bf16> h_in(in_e), h_out(out_e), h_ref(out_e);
  fill_random(h_in);

  bf16 *d_in=nullptr, *d_out=nullptr;
  HIP_CHECK(hipMalloc(&d_in,  in_e*sizeof(bf16)));
  HIP_CHECK(hipMalloc(&d_out, out_e*sizeof(bf16)));
  HIP_CHECK(hipMemcpy(d_in, h_in.data(), in_e*sizeof(bf16), hipMemcpyHostToDevice));

  dim3 grid(B), block(1024);
  auto launch = [&](){
    hipLaunchKernelGGL(silu_mul_kernel, grid, block, 0, 0, d_out, d_in, B, H);
  };

  //lauch and verify
  launch(); HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(h_out.data(), d_out, out_e*sizeof(bf16), hipMemcpyDeviceToHost));
  host_ref(h_ref, h_in, B, H);

  double max_abs=0, max_rel=0; max_diff(h_out, h_ref, max_abs, max_rel);
  const double atol=2e-2, rtol=6e-2;  // bf16 合理阈值
  bool ok = (max_abs <= atol) || (max_rel <= rtol);
  printf("Check: max_abs=%.4g  max_rel=%.4g  -> %s\n",
         max_abs, max_rel, ok ? "PASS":"FAIL");

  // get latency and gbs
  float us = time_kernel_ms(launch, 5, 100)*1000.f;
  double bytes = (double)(in_e + out_e) * sizeof(bf16);
  double gbs = (bytes / (us*1e-6)) / 1e9;
  printf("Perf: %.3f us/launch | ~BW: %.1f GB/s\n", us, gbs);

  HIP_CHECK(hipFree(d_in)); HIP_CHECK(hipFree(d_out));
}