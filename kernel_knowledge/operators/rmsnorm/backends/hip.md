---
title: rmsnorm on hip — SOTA card
kind: sota_card
operator: rmsnorm
backend: hip
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# rmsnorm × hip

## TL;DR
Hand-written HIP/C++ is the **reference and ceiling** for RMSNorm — vLLM's `csrc/layernorm_kernels.cu`
(`rms_norm_kernel`) is the canonical editable kernel, and aiter's asm/CU tier is the same idea hand-tuned.
Reach for HIP when you need a fusion the library can't express or to own the exact ISA. The kernel is
trivial; the win is **128-bit vectorized I/O + wave64 block-reduce + fp32 accumulate**.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| vLLM `rms_norm_kernel` (vectorized) | `vllm/csrc/layernorm_kernels.cu` (PR #22602) | gfx942/950, bf16/fp16/fp8 | `[16384,1024]` fp16 **105.9→42.6 µs** (~2.5×, NVIDIA-measured, traffic reduction) via aligned vector I/O + shared-mem row cache | the editable HIP kernel / non-AITER vLLM |
| aiter asm/CU `module_rmsnorm` | `/sgl-workspace/aiter/aiter/ops/rmsnorm.py` | gfx942/950 | bandwidth-bound floor | aiter serving path (see [aiter.md](aiter.md)) |
| AMD lab-notes block-reduce template | https://gpuopen.com/learn/amd-lab-notes/ | all | reference wave64 reduce | from-scratch authoring |

## Config space / knobs
- **Block**: `blockDim = min(next_pow2(N), 1024)`, multiple of 64; one block per row (prefill) or persistent
  grid-stride over rows (decode, `gridDim = min(M, 304·occ)`).
- **Vector I/O**: read x as `float4`/`__half2` (`reinterpret_cast`, require 16-B alignment / N%8==0) →
  `global_load_dwordx4`. `__restrict__` on all pointers.
- **Reduce**: intra-wave `__shfl_down` (64-lane, 6 steps), then `__shared__ float partial[blockDim/64]`,
  final wave-reduce. `cub::BlockReduce` works but pin the CCCL version (vLLM #24464: CUDA13/CCCL3 broke
  `cub::Sum`).
- **Occupancy**: `__launch_bounds__(blockDim, wavesPerEU)` — norm is VGPR-light → `wavesPerEU=4`.
- Compile: `hipcc --offload-arch=gfx942 -O3 -munsafe-fp-atomics` (atomics only if cross-block reduce).

## Numerics / parity
fp32 `float acc` for Σx²; **promote γ to fp32 before multiply** (vLLM #42325 regression — the canonical
bug); ε inside mean. Output convert last. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- vLLM: edit `csrc/layernorm_kernels.cu`, rebuild vLLM (Tier-C); op registered in `torch_bindings.cpp`
  (`_C::rms_norm`, `fused_add_rms_norm`). Not a Python-only change.
- Standalone: compile to a `.so`, bind via `torch.utils.cpp_extension` / `TORCH_LIBRARY`.

## Pitfalls & anti-patterns
- ⚠ γ-in-input-dtype multiply (vLLM #42325) — promote to fp32.
- ⚠ `dim3 grid(num_tokens)` with `num_tokens==0` → `hipErrorInvalidConfiguration`, sticky (sglang #23609
  for activation; same trap) — early-return guard.
- Unaligned N (not %8) → scalar loads, no vectorization. Pad or handle the tail.
- 32-bit `int` indexing overflows at `M·N > 2³¹` (vLLM #22602 review) — use `int64_t` strides.

## How to verify
`-Rpass-analysis=kernel-resource-usage` (VGPR/LDS); `--save-temps` → grep `global_load_dwordx4` in `.s`;
isolated bench vs aiter; fp64 oracle within band; greedy e2e parity.

## Alternatives / cross-links
[aiter.md](aiter.md) · [triton.md](triton.md) · [vllm_kernels.md](vllm_kernels.md) ·
[[languages/hip_cpp/patterns]] §1 (wave64 block reduction) · [../tuning.md](../tuning.md).

## Sources
- vLLM HIP `rms_norm_kernel` + vectorization PR: https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu, https://github.com/vllm-project/vllm/pull/22602.
- γ-dtype regression: https://github.com/vllm-project/vllm/issues/42325.
- wave64 / __launch_bounds__ / 128-bit loads: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html, https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html.
