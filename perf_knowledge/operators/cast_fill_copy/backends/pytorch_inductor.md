---
title: cast_fill_copy on PyTorch Inductor — SOTA card
kind: sota_card
operator: cast_fill_copy
backend: pytorch_inductor
gens: [gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz]
regimes: [both, training]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
  - https://rocm.blogs.amd.com/artificial-intelligence/pytorch-amd-gpus/README.html
  - https://rocm.docs.amd.com/en/docs-6.3.1/how-to/rocm-for-ai/inference-optimization/workload.html
---

# cast_fill_copy × pytorch_inductor

## TL;DR
Inductor is the **automatic** SOTA: under `torch.compile` it **elides redundant casts and `.contiguous()`
copies** and **fuses surviving casts into adjacent pointwise/reduction kernels** — the cast becomes the
store of a kernel that was already running. You don't write a kernel; you let Inductor remove the
data-movement traffic. This is the realization of [../fusion.md](../fusion.md) for free.

## SOTA implementation(s)
| impl | source | gens/dtypes | mechanism | when best |
|---|---|---|---|---|
| cast fused into pointwise/reduction store | `torch._inductor` | gfx942/950 | dtype-convert at the kernel store | any `torch.compile` graph |
| redundant cast / `.contiguous()` elision | `torch._inductor` | gfx942/950 | stride tracking, dead-cast removal | graphs with layout churn |

## Config space / knobs
| knob | value | effect |
|---|---|---|
| (default) fusion + cast elision | on | casts fold into neighbors; redundant copies dropped |
| `TORCH_COMPILE_DEBUG` | `1` | `output_code.py` shows where casts landed (or vanished) |
| `config.max_autotune.pointwise` | `True` | tune the tiling of the fused (cast-carrying) kernel |
| `TORCHINDUCTOR_BENCHMARK_KERNEL` | `1` | per-kernel timing |

## Numerics / parity
copy/fill bit-exact; fused float→float cast is RNE (matches torch). A cast fused *before* a downstream
round can be slightly more accurate than the unfused double-round (a gain). fp8 casts on the serving path
usually go through aiter/CK quant ops (opaque), not Inductor — Inductor fuses *around* them. See
[../numerics.md](../numerics.md).

## Integration (rebind seam)
- `torch.compile(model)` — elision/fusion on by default.
- Opaque custom ops (aiter fp8 quant) are not decomposed → Inductor fuses surrounding casts into the
  pointwise glue, not across the op.
- `output_code.py` is the editable seam to confirm/tweak.

## Pitfalls & anti-patterns
- A user `.contiguous()`/`.to()` inside an opaque custom op is **hidden** from Inductor → can't be elided;
  keep layout/cast glue in eager torch so Inductor sees it.
- Dynamic shapes re-trigger codegen.
- A cast at a graph boundary (input/output) can't always be fused — sometimes a standalone kernel remains.
- Don't assume Inductor removes *every* copy; verify with `TORCH_COMPILE_DEBUG`.

## How to verify
`TORCH_COMPILE_DEBUG=1` → `output_code.py`: redundant casts/`.contiguous()` gone, surviving casts inlined
into a fused kernel's store; rocprof kernel count + `WRITE_SIZE` drop vs eager.

## Alternatives / cross-links
[triton.md](triton.md) (manual fused authoring) · [hip.md](hip.md) (runtime copy/fill) ·
[../fusion.md](../fusion.md) · backend overview
[`../../../backends/pytorch_inductor/overview.md`](../../../backends/pytorch_inductor/overview.md).

## Sources
- Inductor cast elision / stride tracking / pointwise fusion: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
- ROCm Inductor/Triton on MI300X: https://rocm.blogs.amd.com/artificial-intelligence/pytorch-amd-gpus/README.html
- max-autotune / TORCH_COMPILE_DEBUG: https://rocm.docs.amd.com/en/docs-6.3.1/how-to/rocm-for-ai/inference-optimization/workload.html
