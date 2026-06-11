---
title: CK instances — traits, coverage gotcha, selection & tuning
kind: backend
backend: ck_lib
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1, int8]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-05
sources:
  - https://github.com/ROCm/composable_kernel
  - https://github.com/sgl-project/sglang/issues/16025
---

# CK instances — traits, coverage, selection

## TL;DR
A CK **instance** is one fully-specialized device op: a fixed combination of **tile, MFMA instruction,
pipeline depth, scheduler, and vector widths**. CK ships thousands of them; the consumer picks the fastest
that **covers** the shape. The single most common production failure is **missing instance coverage**
(`device_gemm ... does not support this GEMM problem`) — no compiled instance satisfies
`IsSupportedArgument` for your (M,N,K, strides, dtype, layout).

## Instance traits (what varies between instances)

| Trait | Meaning | MI300X/CDNA guidance |
|---|---|---|
| **Tile (M/N/K block)** | output macro-tile + K-step per instance | larger tiles for big GEMM; small tiles for skinny/decode |
| **MFMA instruction** | the matrix-core op (16×16 vs 32×32) | prefer `mfma_16x16` on CDNA3/4 |
| **Pipeline (num stages)** | global→LDS prefetch depth / scheduler | per-instance; async-input/persistent schedulers on gfx950 |
| **Vector widths** | load/store/convert vectorization | drives alignment requirements (the coverage gate) |
| **Split-K / GSU** | K parallelism | needed for tall-skinny / decode |
| **Layout** | NN/NT/TN/TT | NT is the typical weight layout |
| **dtype combo** | A/B/compute/out dtypes | gfx942 FP8=FNUZ; gfx950 adds OCP FP8 + FP4/MXFP4 |

CK-Tile `gemm_universal` recently added mixed-precision fp8×bf8, **weight-preshuffle GEMM**, and a
compute-async pipeline + persistent async input scheduler on gfx950.

## The coverage gotcha
```
device_gemm with the specified compilation parameters does not support this GEMM problem
```
Means **no compiled instance** passes `IsSupportedArgument` for your problem. Causes & fixes:

| Cause | Fix |
|---|---|
| Built with restricted `DTYPES`/`OP_FILTER`; your dtype/op filtered out | rebuild including it |
| Odd strides / alignment no instance covers | pad M/N/K to a covered multiple; change layout |
| FP8 model GEMM with no instance | `ckProfiler gemm_universal 7 ...` on the exact shape; if none, generate/build one |
| Wrong arch built | ensure `GPU_TARGETS` includes `gfx942`/`gfx950` |
| MoE expert/inter shape uncovered (real SGLang report) | pad expert/inter dims; tune to a covered shape |

This is a live serving issue under aiter/vLLM too — e.g. `VLLM_ROCM_USE_AITER=1` hitting the CK MoE
coverage error on large MoE models (Qwen3-235B-A22B, DeepSeek) when the stage-2 `moe_ck2stages_*` instance
doesn't cover the shape.

## Selection / tuning workflow (library consumer)
1. Extract the real (M,N,K, layout, dtype) shapes from your model.
2. For each, `ckProfiler gemm_universal <fp8_mode> <layout> 1 1 0 1 M N K ... <splitK> <warmup> <iters> <rotbuf>`
   (see [ckprofiler.md](ckprofiler.md)).
3. Record the winning instance + TFLOPS; if none is supported, that shape needs a new instance (build/
   codegen) or a hipBLASLt path instead.
4. Wire the winner: select that instance index in your factory loop, or rely on the aiter/vLLM wrapper
   whose internal CK selection should now find it.
5. Re-verify after ROCm/CK upgrades — instance lists and best configs change.

## Epilogue knob
`OPTIMIZE_EPILOGUE=1` keeps the MFMA-layout result instead of reblocking before the store — usually faster
for fused epilogues (see [ckprofiler.md](ckprofiler.md)). ROCm 7.2 CK added GEMM+GEMM fusion and a fused
clamp GEMM epilogue.

## Pitfalls
- Instance indices/best configs are **build- and version-specific** — never treat a chosen index as
  portable; re-profile after upgrades (sourcing rule #2).
- Restricting `DTYPES`/`OP_FILTER` at build time to cut compile time can later cause coverage crashes —
  keep the dtypes/ops you actually serve.

## Cross-links
[api.md](api.md) · [ckprofiler.md](ckprofiler.md) · [../aiter/fmoe.md](../aiter/fmoe.md) (CK stage-2 MoE) ·
[../aiter/flydsl_path.md](../aiter/flydsl_path.md) (A4W4 → CK fallback).

## Sources
- CK repo (instances, factory, `IsSupportedArgument`): https://github.com/ROCm/composable_kernel
- DeviceGemm reference: https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/structck_1_1tensor__operation_1_1device_1_1_device_gemm.html
- CK MoE coverage error (SGLang #16025): https://github.com/sgl-project/sglang/issues/16025
- aiter+CK MoE coverage error (vLLM #22245): https://github.com/vllm-project/vllm/issues/22245
