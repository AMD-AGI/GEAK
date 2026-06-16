---
title: Inductor max-autotune on ROCm — GEMM/conv config search, Triton-only, CK backend
kind: backend
backend: pytorch_inductor
operator: dense_gemm
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both, training]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/pytorch/pytorch/pull/143286
  - https://github.com/pytorch/pytorch/pull/144985
  - https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
  - https://rocm.docs.amd.com/en/docs-6.3.1/how-to/rocm-for-ai/inference-optimization/workload.html
---

# Inductor max-autotune on ROCm

## TL;DR
max-autotune is the Inductor mode that **benchmarks a static list of Triton (and optionally CK/ATen)
configs per GEMM/conv shape** at compile time and bakes in the winner. On ROCm it tunes MFMA-aware Triton
params (`matrix_instr_nonkdim`, `waves_per_eu`, `kpack`, `GROUP_M`) via device-specific
`ROCmConfigHeuristic`. Turn it on with `TORCHINDUCTOR_MAX_AUTOTUNE=1`; consider `GEMM_BACKENDS=TRITON` to
unlock more fused mm; add `CK` to also search Composable Kernel. Pays for itself on fusion-heavy graphs;
A/B against eager + hipBLASLt/TunableOp for isolated big GEMMs.

## When to use it
- **Use** when a model has many fusable mm + pointwise chains, or when Triton fusion can absorb a GEMM
  epilogue (bias/act) that the library can't.
- **Skip** when GPU time is one or two huge isolated GEMMs already served well by hipBLASLt's tuned DB — the
  long autotune compile won't pay back.
- Decode (skinny-M) GEMMs: Triton split-K configs in the search can beat a generic library kernel; A/B.

## Knobs / config space
| knob | value | effect |
|---|---|---|
| `TORCHINDUCTOR_MAX_AUTOTUNE` | `1` | enable mm/conv autotune+lowering (`config.max_autotune=True`) |
| `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` | `TRITON` / `TRITON,ATEN` (default) / `TRITON,ATEN,CK` | candidate backends for mm |
| `config.max_autotune.pointwise` | `True` | also tune pointwise/reduction tiling |
| (CK backend) | append `CK` | autotune searches Composable Kernel instances |
| `TORCH_COMPILE_DEBUG` | `1` | dump generated Triton (`output_code.py`) |

ROCm-tuned Triton search dims (from PR #143286 / `ROCmConfigHeuristic`): tile `BLOCK_M/N/K`, `num_stages`,
`num_warps`, MFMA `matrix_instr_nonkdim` (16 vs 32), `waves_per_eu` (VGPR/occupancy hint), `kpack`, and
`GROUP_M` for the Triton GEMM. **`mfma_16x16` typically beats `mfma_32x32` on MI300X** even at large tiles.

## Measured
- PR #143286 (ROCm-specific GEMM configs), Dynamo HuggingFace **inference**, bf16,
  `TORCHINDUCTOR_MAX_AUTOTUNE=1 TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON`: geomean **1.36× → 1.42×**
  (before→after ROCm configs); **~9%** on an internal addmm bench (PyTorch-reported, 2025).
- PR #144985 refactored ROCm configs into `template_heuristics.py` (`ROCmConfigHeuristic` alongside
  CUDA/CPU/XPU) — cleaner per-device config surface.

## Numerics / parity
Switching ATen→Triton (or picking a different tile) changes accumulation order → a same-dtype result can
flip a borderline bf16 argmax. Re-run greedy/temp=0 parity after enabling max-autotune; gate on a task
eval if it diverges.

## Integration (rebind seam)
- `torch.compile(model, mode="max-autotune")` or set the env and compile normally.
- vLLM/sglang call `torch.compile`; their AITER ops registered via `direct_register_custom_op` stay opaque
  so Inductor fuses around them rather than replacing them
  ([../vllm_kernels/aiter_integration.md](../vllm_kernels/aiter_integration.md)).
- Generated `output_code.py` Triton kernels are themselves an editable seam for a Tier-C rewrite.

## Pitfalls
- **Compile-time blow-up**: max-autotune benchmarks many configs per shape — cache (`TORCHINDUCTOR_CACHE_DIR`)
  and warm before serving; dynamic shapes re-tune.
- `GEMM_BACKENDS=TRITON` removes ATen/rocBLAS from the search — wins via fusion but can lose a tuned
  hipBLASLt solution; verify it's actually faster e2e.
- CK backend needs CK available in the build; absent → it silently isn't searched.
- Don't assume CUDA-default Triton configs port — use the ROCm heuristic (recent PyTorch) or you get a
  far-from-optimal kernel.

## How to verify
`TORCH_COMPILE_DEBUG=1` → inspect `output_code.py` for the chosen tile/MFMA; time the compiled vs eager
path (median of ≥3 warm runs) and against eager + TunableOp; confirm parity.

## Alternatives / cross-links
[overview.md](overview.md) · TunableOp / hipBLASLt:
[../rocblas_tunableop/tunableop.md] · operators `dense_gemm`, `conv2d`.

## Sources
- ROCm GEMM tuning params PR (waves_per_eu/kpack/matrix_instr_nonkdim/GROUP_M; 1.36→1.42×): https://github.com/pytorch/pytorch/pull/143286
- Template-heuristic refactor (`ROCmConfigHeuristic`): https://github.com/pytorch/pytorch/pull/144985
- Inductor config (max_autotune, gemm backends): https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
- MI300X workload optimization (enable max-autotune, Triton-only, CK backend, mfma_16x16, TORCH_COMPILE_DEBUG): https://rocm.docs.amd.com/en/docs-6.3.1/how-to/rocm-for-ai/inference-optimization/workload.html
