---
title: Composable Kernel as a library — DeviceGemm API & how aiter/vLLM consume it
kind: backend
backend: ck_lib
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp4_e2m1, int8]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-05
sources:
  - https://github.com/ROCm/composable_kernel
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
---

# Composable Kernel (consumed as a library)

## TL;DR
This card treats **Composable Kernel (CK) as a library you consume** — pick and wire prebuilt **instances**
— not as the CK/CK-Tile authoring DSL (see `languages/composable_kernel/`). CK is the broad, robust
instance library under much of the AMD stack: **aiter, vLLM, and SGLang call CK** for GEMM, grouped-GEMM,
FMHA, and fused MoE (it is aiter's default fallback when no asm path matches, and the stage-2 MoE kernel —
`moe_ck2stages_*`). The consumer's job: (1) confirm an instance covers your shape/dtype, (2) pick the
fastest (`ckProfiler`, see [ckprofiler.md](ckprofiler.md)).

> Repo note: the standalone `ROCm/composable_kernel` is **deprecated and moved to
> `ROCm/rocm-libraries`** (develop is a read-only mirror). In an upcoming major ROCm release CK becomes
> **header-only** — neither `ckProfiler` nor the static libs are packaged by default (ckProfiler can still
> be built standalone). The instance/factory concepts below are unchanged.

## Mental model: instances & the instance factory

| Concept | Meaning |
|---|---|
| **Device op** | a templated kernel family, e.g. `ck::tensor_operation::device::DeviceGemm<ALayout,BLayout,CLayout,ADataType,BDataType,CDataType,AElementwiseOp,BElementwiseOp,CElementwiseOp>` |
| **Instance** | one fully-specialized device op: fixed tile (M/N/K block), MFMA instr, pipeline (stages), scheduler, vector widths |
| **Instance factory** | prebuilt per-op instance lists queried at runtime: `add_device_gemm_xdl_..._instances(...)` |
| **`MakeArgument`/`MakeInvoker`/`GetWorkSpaceSize`** | per-instance: build args, get a runnable invoker, size scratch |
| **`IsSupportedArgument`** | per-instance predicate — **false** if the instance doesn't cover this shape/stride/dtype. The gate behind the "device_gemm does not support this GEMM problem" crash. |

Consumer pattern (what `ckProfiler` and aiter's CK wrapper automate):
```cpp
auto ops = DeviceOperationInstanceFactory<DeviceGemm<...>>::GetInstances();
for (auto& op : ops) {
  auto arg = op->MakeArgumentPointer(A,B,C,M,N,K,sA,sB,sC,aop,bop,cop);
  if (!op->IsSupportedArgument(arg.get())) continue;     // skip uncovered instances
  float ms = op->MakeInvokerPointer()->Run(arg.get(), {nullptr, /*time=*/true});
  // keep the fastest
}
```

## How aiter / vLLM / SGLang call CK

| Consumer | CK usage |
|---|---|
| **aiter** | default fallback for GEMM / grouped-GEMM / FMHA / MoE when no asm path matches; FlyDSL A4W4 MoE falls back to CK; fmoe stage-2 = `moe_ck2stages_*`. `CK_BLOCK_GEMM=1`, `SGLANG_ROCM_AITER_BLOCK_MOE=1` route to CK block-scale paths. |
| **vLLM** | CK Flash Attention 2 alongside Triton FA (`VLLM_USE_FLASH_ATTN_TRITON=False` → CK); FP8 linear reaches CK or hipBLASLt by shape. |
| **SGLang** | CK FMHA / block GEMM; AITER MoE wraps CK grouped GEMM (`CK_BLOCK_GEMM=1`). |

So you rarely call CK's C++ API by hand for serving — you (a) ensure the packaged CK build has instances
for your shapes and (b) use `ckProfiler` to confirm/select when chasing a regression.

## FP8 / FP4 specifics
- gfx942: FP8 = **FNUZ** (E4M3FNUZ/E5M2FNUZ); native MFMA + hardware FP8 conversion (`fmed3f` clipping,
  packed 2-at-a-time convert). The standard quantized-linear path is `gemm_universal` mode 7 (f8 in, f8
  compute, bf16 out). Do **not** set `CK_USE_FP8_ON_UNSUPPORTED_ARCH` on gfx942 (it has native FP8) — that
  flag is only for functional FP8 on gfx908/gfx90a.
- gfx950: adds OCP FP8 + FP4/MXFP4 + block scaling.

## Numerics / parity
Same-math GEMM, different tiling → parity-safe; FP8/FP4 introduce quant error gated by the consumer's
recipe.

## Pitfalls
- ⚠ **Missing instance coverage** is the #1 production failure: "device_gemm ... does not support this GEMM
  problem" = no compiled instance satisfies `IsSupportedArgument` for your (M,N,K, strides, dtype, layout).
  Fixes: rebuild including the dtype/op, pad to a covered multiple, or generate an instance.
- CK header-only transition: don't assume `ckProfiler`/static libs are present; build standalone if needed.

## Cross-links
[ckprofiler.md](ckprofiler.md) · [instances.md](instances.md) · [`languages/composable_kernel/`](../../languages/composable_kernel/)
· [../aiter/fmoe.md](../aiter/fmoe.md) (CK stage-2 MoE).

## Sources
- CK repo (deprecated → rocm-libraries; instances, examples): https://github.com/ROCm/composable_kernel
- DeviceGemm reference (`IsSupportedArgument`): https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/structck_1_1tensor__operation_1_1device_1_1_device_gemm.html
- Optimizing with Composable Kernel: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
- aiter CK consumption (on-box `moe_ck2stages_*`): `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/configs/tuned_fmoe.csv`
