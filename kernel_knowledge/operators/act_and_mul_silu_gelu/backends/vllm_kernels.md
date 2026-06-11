---
title: act_and_mul_silu_gelu on vllm_kernels — SOTA card
kind: sota_card
operator: act_and_mul_silu_gelu
backend: vllm_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/csrc/activation.cu
  - https://docs.vllm.ai/en/v0.10.2/api/vllm/model_executor/layers/activation.html
---

# act_and_mul_silu_gelu × vllm_kernels

## TL;DR
vLLM has its own HIP gated-activation kernels (`csrc/activation.cu`) plus an AITER path. The Python layer
`model_executor/layers/activation.py` registers `SiluAndMul`, `MulAndSilu`, `GeluAndMul`, `SwigluOAIAndMul`
as custom ops with `forward_hip`. On MI300X with AITER on, AITER wins for the quant-fused / MoE forms;
native HIP is the default for plain bf16 act_and_mul.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| native HIP `silu_and_mul`/`gelu_and_mul`/... | `csrc/activation.cu` | gfx942/950, bf16/fp16 | bandwidth-bound | plain act_and_mul |
| `silu_and_mul_quant` (HIP fp8) | same | gfx942/950, fp8 out | act + fp8 quant | quant fusion |
| AITER act / fused-MoE stage-1 | via `VLLM_ROCM_USE_AITER` | gfx942/950 | inside fused-MoE | MoE expert activation — [aiter.md](aiter.md) |

## Config space / knobs
- Registry maps `"silu"→SiluAndMul`, `"gelu"/"geglu"→GeluAndMul`, `"swigluoai"→SwigluOAIAndMul(alpha=1.702,
  limit=7.0)`.
- `VLLM_ROCM_USE_AITER=1` (+ MoE gate) → AITER activation/MoE; `=0` → native HIP.
- Native: vector width, fp32 act.

## Numerics / parity
fp32 act; **match GeLU variant** (`gelu` exact vs `gelu_tanh` — #43326 was a MoE GELU_TANH gap); gated half
correct; fnuz fp8 gfx942. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- Python: `model_executor/layers/activation.py` (`_ACTIVATION_AND_MUL_REGISTRY`, `forward_hip`).
- Native HIP: `csrc/activation.cu` + `torch_bindings.cpp`; rebuild to edit.
- torch.compile: custom-op registered → Inductor fuses around; or decomposes to generated Triton when
  custom ops disabled (`backend==inductor` + mode≠NONE appends `"none"`).

## Pitfalls & anti-patterns
- ⚠ `num_tokens==0` crash (#23609 pattern, sticky CUDA/HIP error) — guard.
- ⚠ GELU variant mismatch (#43326).
- ⚠ Image mismatch with `USE_AITER=1`.

## How to verify
rocprofv3 kernel name (native vs AITER vs Triton); isolated bench; correct variant; greedy parity.

## Alternatives / cross-links
[aiter.md](aiter.md) · [hip.md](hip.md) · [triton.md](triton.md) ·
[[backends/vllm_kernels/rocm_kernels]] · [[fused_moe_grouped_gemm]].

## Sources
- vLLM HIP activation kernels: https://github.com/vllm-project/vllm/blob/main/csrc/activation.cu.
- activation registry / variants: https://docs.vllm.ai/en/v0.10.2/api/vllm/model_executor/layers/activation.html.
- GELU_TANH MoE gap: https://github.com/vllm-project/vllm/issues/43326.
