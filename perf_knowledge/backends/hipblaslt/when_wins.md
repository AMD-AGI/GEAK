---
title: hipBLASLt vs rocBLAS — when each wins on Instinct
kind: backend
backend: hipblaslt
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.docs.amd.com/projects/rocBLAS/en/latest/how-to/what-is-rocblas.html
---

# hipBLASLt vs rocBLAS — when each wins

## TL;DR
**hipBLASLt is the more MI300X-optimized GEMM library and the preferred default for modern LLM workloads;
rocBLAS (Tensile backend) wins mainly as a fallback** — when hipBLASLt has no solution for a shape, errors,
on unsupported GPUs, or occasionally on small/odd shapes the hipBLASLt heuristic mis-selects. The two are
nested in recent ROCm: rocBLAS can itself call hipBLASLt (`ROCBLAS_USE_HIPBLASLT=1`) and fall back to
Tensile. Practical rule: **don't trust either default heuristic for production shapes — tune** (and on
sglang/vLLM, tune via aiter, which bypasses both).

## The relationship
- **rocBLAS**: the classic BLAS; uses **Tensile** (installed with rocBLAS) for high-perf GEMM. Can also use
  hipBLASLt as a backend.
- **hipBLASLt**: separate package; "lt" API with plans/epilogues/FP8 scaling; the default for gfx12 and the
  preferred path on gfx942/gfx950.
- Backend choice is automatic by arch + problem; `ROCBLAS_USE_HIPBLASLT=1` makes rocBLAS prefer hipBLASLt
  and **fall back to Tensile** when hipBLASLt lacks a solution or errors. `ROCBLAS_USE_HIPBLASLT=0` forces
  Tensile; `ROCBLAS_USE_HIPBLASLT_BATCHED` is the batched-only variant.

## Decision table

| Situation | Winner | Why |
|---|---|---|
| Dense bf16/fp16/fp8 LLM GEMM, mid/large shapes | **hipBLASLt** | most MI300X-optimized; the default LLM path |
| gfx12 architecture | **hipBLASLt** | the default backend there |
| hipBLASLt has no solution / errors for a shape | **rocBLAS (Tensile)** | automatic fallback |
| Unsupported / consumer GPU, hipBLASLt errors | **rocBLAS** | set `TORCH_BLAS_PREFER_HIPBLASLT=0` |
| Small-batch / odd / non-aligned shapes | **tune both** | neither default heuristic reliably picks best |
| Inductor decides rocBLAS/MIOpen is faster for an op | **rocBLAS** | Inductor compares and won't use Triton if rocBLAS wins |
| sglang/vLLM serving | **hipBLASLt under aiter** | aiter calls a tuned solidx via `hipb_mm`; tune via aiter |

## PyTorch routing
- `TORCH_BLAS_PREFER_HIPBLASLT=1` prefers hipBLASLt over hipBLAS for GEMM (already set in vLLM ROCm
  images; set manually on bare metal). `=0` falls back (use on unsupported GPUs / to dodge hipBLASLt
  errors).
- Historical bug (fixed): `torch.matmul` used hipBLASLt but `F.linear` defaulted to rocBLAS — once a reason
  Linear layers underperformed on AMD.

## MI300X shape gotchas (apply to both)
- **Small GEMMs underperform**: ~200 GFLOPs is needed to approach peak; large K needs larger batch for
  equivalent efficiency.
- **L2 efficiency**: prefer M/N multiples of the XCD count (8 on MI300X → 24/32/40…).
- **512-byte stride hotspot**: a GEMM matrix stride that is a multiple of 512 B causes Tagram channel
  hotspotting and a large perf drop (worst on TN); pad the stride off 512-B multiples.

## Why you tune anyway
The hipBLASLt/rocBLAS default heuristic frequently mis-selects the algorithm for non-standard shapes
(unlike cuBLAS on H100), so per-shape tuning is the dependable path. Options:
- **Raw torch**: hipBLASLt offline tuning ([offline_tuning.md](offline_tuning.md)) or PyTorch TunableOp
  (searches both rocBLAS + hipBLASLt; watch the HBM workspace leak — set a workspace cap).
- **Serving (sglang/vLLM)**: tune via aiter's DB — it bypasses both libraries' heuristics and the
  hipBLASLt override file ([../aiter/tuned_gemm.md](../aiter/tuned_gemm.md)).
- As math libs improve, the marginal tuning win shrinks and may be `Default`/no-win — **always measure
  before/after**.

## Cross-links
[api.md](api.md) · [offline_tuning.md](offline_tuning.md) · [tensilelite.md](tensilelite.md) ·
[env.md](env.md) · [`backends/rocblas_tunableop/`](../rocblas_tunableop/) ·
[`operators/dense_gemm/backends/hipblaslt.md`](../../operators/dense_gemm/backends/hipblaslt.md).

## Sources
- MI300X workload optimization (shape/stride/XCD guidance, TORCH_BLAS_PREFER_HIPBLASLT): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- rocBLAS design (Tensile backend, `ROCBLAS_USE_HIPBLASLT` fallback): https://rocm.docs.amd.com/projects/rocBLAS/en/latest/how-to/what-is-rocblas.html
- ~45–50% of peak / shape sensitivity reality check: https://arxiv.org/pdf/2510.27583
- Per-shape tuning gains (up to 7.2×): https://www.nscale.com/blog/nscale-benchmarks-amd-mi300x-gpus-with-gemm-tuning-improves-throughput-and-latency-by-up-to-7-2x
