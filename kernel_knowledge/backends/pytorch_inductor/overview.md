---
title: PyTorch Inductor on ROCm — overview (torch.compile codegen backend)
kind: backend
backend: pytorch_inductor
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both, training]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
  - https://rocm.blogs.amd.com/artificial-intelligence/pytorch-amd-gpus/README.html
  - https://rocm.docs.amd.com/en/docs-6.3.1/how-to/rocm-for-ai/inference-optimization/workload.html
---

# PyTorch Inductor on ROCm

## TL;DR
Inductor is `torch.compile`'s default codegen backend: it lowers an FX graph to **Triton** kernels (plus
calls into ATen/rocBLAS/MIOpen), fuses pointwise/reduction ops, and — under **max-autotune** — benchmarks
Triton templates against library kernels per shape. On ROCm it emits **ROCm-Triton** that MFMA-targets
CDNA, and PyTorch ships **device-specific autotune heuristics** (`ROCmConfigHeuristic`) that tune MFMA
`matrix_instr_nonkdim`, `waves_per_eu`, `kpack`. It is the **fusion + portability** layer; for the last
bit of dense-GEMM perf, hipBLASLt/TunableOp still often win — measure. The headline knob is
[max_autotune.md](max_autotune.md).

## Concepts
- **Lowering + fusion**: Inductor fuses elementwise/reduction chains into single Triton kernels (the main
  free win — fewer launches, less HBM traffic). Matmul/conv default to ATen (rocBLAS/MIOpen) unless
  max-autotune is on.
- **max-autotune**: benchmark a static list of Triton GEMM/conv configs (and optionally library kernels)
  per shape; pick the fastest. Longer compile, faster runtime.
- **Backends per op**: `max_autotune_gemm_backends` ∈ {`TRITON`, `ATEN`, `CK`} (default `TRITON,ATEN`).
- **Device heuristics**: `torch/_inductor/template_heuristics.py` has `ROCmConfigHeuristic` (vs CUDA/CPU/XPU)
  supplying ROCm-tuned Triton configs (PRs #143286, #144985).
- **Custom-op preservation**: opaque torch custom ops (e.g. AITER ops registered via
  `direct_register_custom_op`) are **not** decomposed by Inductor — that's how vLLM keeps hand-tuned AITER
  kernels through `torch.compile` ([../vllm_kernels/aiter_integration.md](../vllm_kernels/aiter_integration.md)).

## The levers
| lever | env / config | effect |
|---|---|---|
| enable autotune (mm/conv) | `TORCHINDUCTOR_MAX_AUTOTUNE=1` / `torch._inductor.config.max_autotune=True` | benchmark+lower mm/conv to Triton |
| pointwise autotune | `config.max_autotune.pointwise=True` | tune pointwise/reduction tiling |
| GEMM backends | `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON` | limit to Triton → more fused mm (vs going to rocBLAS) |
| add CK backend | append `CK` to the backend list | autotune can pick Composable Kernel instances |
| inspect codegen | `TORCH_COMPILE_DEBUG=1` | dumps `output_code.py` Triton kernels to `torch_compile_debug/` |
| autotune logging | `TORCHINDUCTOR_BENCHMARK_KERNEL=1` (per-kernel timing) | see chosen config |

## Where it sits in the stack
PyTorch op → (torch.compile) → Inductor → Triton kernel **or** ATen→hipBLASLt/rocBLAS/MIOpen. Inductor
does **not** replace AITER/CK hand-tuned ops that are registered as opaque custom ops; it owns the *fusable
glue* and any mm/conv it can beat the library on under max-autotune. Triton is **not** used if MIOpen/
rocBLAS is faster for that op.

## Measured (version-tagged)
- ROCm-specific GEMM autotune configs (PR #143286), Dynamo HuggingFace **inference**, bf16,
  `TORCHINDUCTOR_MAX_AUTOTUNE=1` + `GEMM_BACKENDS=TRITON`: geomean **1.36× → 1.42×** speedup (before→after
  the ROCm configs); **~9%** on an internal addmm bench (PyTorch-reported, 2025).
- MI300X MFMA rule of thumb: **`mfma_16x16` typically beats `mfma_32x32`** even at large tiles (ROCm Triton
  optimization guide).

## Pitfalls
- max-autotune **compile time is large** (benchmarks many configs) — warm/cache before serving; not for
  fast iteration loops.
- Limiting to `TRITON` can win via fusion **or** lose vs a tuned hipBLASLt solution — A/B per model.
- A backend swap (ATen→Triton) can change reduction order → re-check greedy/temp=0 parity.
- Inductor's Triton must MFMA-target CDNA; a generic config can be far off — rely on `ROCmConfigHeuristic`
  (recent PyTorch) rather than CUDA-default configs.

## Verify
- `TORCH_COMPILE_DEBUG=1` → read `output_code.py`; confirm fused Triton kernels and MFMA tiles
  (`matrix_instr_nonkdim`).
- Compare max-autotune e2e vs eager + TunableOp ([../rocblas_tunableop/tunableop.md])
  — Inductor wins on fusion-heavy graphs; library wins on isolated big GEMMs.

## Sources
- Inductor config (max_autotune, gemm backends): https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
- ROCm GEMM autotune configs PR (1.36→1.42×): https://github.com/pytorch/pytorch/pull/143286 ; heuristic refactor: https://github.com/pytorch/pytorch/pull/144985
- PyTorch on AMD ROCm (Inductor/Triton on MI300X): https://rocm.blogs.amd.com/artificial-intelligence/pytorch-amd-gpus/README.html
- MI300X workload optimization (max-autotune, Triton-only, CK backend, mfma_16x16): https://rocm.docs.amd.com/en/docs-6.3.1/how-to/rocm-for-ai/inference-optimization/workload.html
