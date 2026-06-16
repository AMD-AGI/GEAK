---
title: ckProfiler — gemm_universal sweep, args, OPTIMIZE_EPILOGUE
kind: profiling
backend: ck_lib
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode]
status: sota
updated: 2026-06-05
sources:
  - https://github.com/ROCm/composable_kernel/blob/develop/profiler/README.md
  - https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html
---

# ckProfiler

## TL;DR
`ckProfiler <op> <args...>` sweeps all matching compiled CK instances, optionally verifies, times them, and
prints the fastest with its tile/pipeline config + TFLOPS/GB/s. It is how you (a) confirm an instance
**covers** your shape and (b) pick the best one. The FP8 LLM path is `gemm_universal`. Build it filtered
(`GPU_TARGETS`, `DTYPES`, `CK_PROFILER_OP_FILTER`) to keep compile time sane.

> CK is moving header-only — `ckProfiler` will no longer ship by default; build it standalone when needed.

## Build (gfx942 + FP8, fast)
```bash
git clone https://github.com/ROCm/composable_kernel && cd composable_kernel && mkdir build && cd build
cmake -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
      -D CMAKE_BUILD_TYPE=Release -D GPU_TARGETS="gfx942" -D DTYPES="fp8;bf16;fp16" \
      -D CK_PROFILER_OP_FILTER="gemm_universal" ..
make -j ckProfiler
```
Compile-time knobs: `GPU_TARGETS` (`gfx942;gfx950`), `DTYPES` (subset of `fp64;fp32;tf32;fp16;fp8;bf16;int8`),
`CK_PROFILER_OP_FILTER` (regex of ops, e.g. `"^grouped_gemm$"`), `CK_PROFILER_INSTANCE_FILTER`. Ops are
registered via `ProfilerOperationRegistry` (`profiler/src/profiler.cpp`) and filtered at compile time.

## gemm_universal (the FP8 path)
```
ckProfiler gemm_universal <dtype> <layout> <verify> <init> <print> <time> \
           M N K StrideA StrideB StrideC <splitK> [warmup] [iters] [rotbuf_MB]
```
`<dtype>` (arg2): 0 fp32 · 1 fp16 · 2 bf16 · 3 int8 · 4 f8@f16 · 5 f16@f8 · 6 f16→f8 ·
**7 f8→bf16, comp f8 (FP8 in/compute, bf16 out — the common LLM path)** · 8 f16@i4 · 9 bf16@i4.
`<layout>` (arg3): 0 NN · **1 NT (typical weight layout)** · 2 TN · 3 TT.
```bash
# bf16 4096^3 NT, splitK=1, 1 warmup, 10 iters, no rotbuf
ckProfiler gemm_universal 2 1 1 1 0 1 4096 4096 4096 4096 4096 4096 1 1 10 0
# FP8 (f8 in/compute, bf16 out) 4096^3 NT, 256 MB rotating buffer to defeat L2 reuse
ckProfiler gemm_universal 7 1 1 1 0 1 4096 4096 4096 4096 4096 4096 1 5 50 256
```
`<splitK>` (arg14) splits K across workgroups (tall-skinny / decode). `rotbuf_MB` (arg17) rotates inputs so
cache reuse doesn't inflate numbers — use ≥ L2 (MI300X L2 is 32 MB) for honest FP8 numbers.

Standard `gemm` (legacy): `ckProfiler gemm <dtype> <layout> <verify> <init> <print> <repeat> M N K sA sB sC`
(dtype 0=fp32 1=fp16). Batched: `ckProfiler batched_gemm_multi_d ... BatchCount`.

## OPTIMIZE_EPILOGUE
```bash
export OPTIMIZE_EPILOGUE=1   # store MFMA accumulator directly in MFMA layout (skip the reblock)
```
`=1` avoids converting the MFMA accumulator to a blocked layout before the global store — saves a reblock
at the cost of lower `global_store` vector width; net usually faster for **fused epilogues**. Default `0`
maximizes store-vector length. (Build/runtime env consumed by CK GEMM examples; pair with the chosen
instance.)

## grouped_gemm / FMHA / MoE
- **grouped_gemm**: no fixed-arg profiler line — consume via the instance-factory loop (see
  [api.md](api.md)) or aiter's MoE path. Build base op with `CK_PROFILER_OP_FILTER="^grouped_gemm$"`.
- **FMHA**: CK-Tile FMHA is **codegen**, not a fixed-arg op — the generator
  `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py` emits thousands of instances (tile × warp × mask/bias).
  Bench via the example binary / `script/smoke_test_fwd.sh`.
- **MoE**: CK fused MoE (sorting + grouped GEMM + activation) is normally consumed through
  `aiter.fused_moe` (`moe_ck2stages_*`).

## Reading output
Per best instance: instance name (tile/pipeline), elapsed ms, **TFLOPS**, **GB/s**. CK warms ~50 launches
then averages ~50. For deeper analysis profile the chosen instance with **rocprofv3 / rocprof-compute**.

## Pitfalls
- No supported instance → "does not support this GEMM problem" (coverage gap — see [api.md](api.md)).
- Without a rotating buffer FP8 numbers are inflated (small footprint sits in cache).
- Don't combine CK's timer + verification on accumulating kernels (grouped_conv_bwd_weight, col2img).

## Cross-links
[api.md](api.md) · [instances.md](instances.md) · [`profiling/`](../../profiling/).

## Sources
- ckProfiler README (op arg tables): https://github.com/ROCm/composable_kernel/blob/develop/profiler/README.md
- CK-Tile GEMM hands-on (build/epilogue): https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html
- Header-only / rocm-libraries move + ProfilerOperationRegistry / CK_USE_FP8_ON_UNSUPPORTED_ARCH: https://github.com/ROCm/composable_kernel · https://rocm.docs.amd.com/en/docs-7.2.0/release/changelog.html
