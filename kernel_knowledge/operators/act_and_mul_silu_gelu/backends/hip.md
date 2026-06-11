---
title: act_and_mul_silu_gelu on hip — SOTA card
kind: sota_card
operator: act_and_mul_silu_gelu
backend: hip
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/csrc/activation.cu
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
---

# act_and_mul_silu_gelu × hip

## TL;DR
Hand-written HIP is the reference: a trivial **elementwise** kernel (no reduction) — load gate+up halves
vectorized, `act(gate)·up` in fp32, write. vLLM's `csrc/activation.cu` is the canonical editable kernel
(`silu_and_mul`, `mul_and_silu`, `gelu_and_mul`, `gelu_tanh_and_mul`, `swigluoai_and_mul`, `fatrelu_and_mul`).
Reach for HIP to fuse into a custom GEMM epilogue or add a quant variant.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| vLLM HIP `silu_and_mul` / `gelu_and_mul` / `gelu_tanh_and_mul` | `vllm/csrc/activation.cu` | gfx942/950, bf16/fp16 | bandwidth-bound, vectorized | editable HIP / non-AITER |
| `silu_and_mul_quant` (HIP fused fp8) | same | gfx942/950, fp8 out | act + fp8 quant | quant fusion |
| aiter asm `module_activation` | [aiter.md](aiter.md) | gfx942/950 | floor | aiter path |

## Config space / knobs
- Elementwise grid-stride: `gridDim = min(M·d/blk, 304·occ)`, block ×64. One thread per output element (or
  vectorized: `float4` → 4 elements/thread).
- Load `gate = x[row·2d + col]`, `up = x[row·2d + d + col]` — same vectorized pattern offset by d.
- fp32 act (`__expf`/`erff`/`tanhf`), `__restrict__`, 16-B alignment.
- `__launch_bounds__(block, 4)` (VGPR-light); `hipcc --offload-arch=gfx942 -O3`.

## Numerics / parity
fp32 act; match GeLU variant + gated half; fp8 fnuz on gfx942. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- vLLM: edit `csrc/activation.cu`, rebuild; op in `torch_bindings.cpp` (`_C::silu_and_mul`).
- Standalone: `torch.utils.cpp_extension`.

## Pitfalls & anti-patterns
- ⚠ `dim3 grid(num_tokens)` with `num_tokens==0` → `hipErrorInvalidConfiguration`, **sticky** (sglang
  #23609, exactly this op) — early-return guard.
- ⚠ Wrong gated half (silu_and_mul vs mul_and_silu).
- Unaligned d → scalar loads.

## How to verify
`--save-temps` grep `dwordx4`; fp64 oracle; correct variant; greedy parity; test `num_tokens==0`.

## Alternatives / cross-links
[aiter.md](aiter.md) · [triton.md](triton.md) · [vllm_kernels.md](vllm_kernels.md) ·
[[languages/hip_cpp/patterns]] §2 (grid-stride elementwise).

## Sources
- vLLM HIP activation kernels: https://github.com/vllm-project/vllm/blob/main/csrc/activation.cu.
- num_tokens==0 crash (this op): https://github.com/sgl-project/sglang/issues/23609.
- vectorized elementwise / wave64: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html.
