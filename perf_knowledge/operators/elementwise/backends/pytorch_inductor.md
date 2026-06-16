---
title: elementwise on PyTorch Inductor — SOTA card
kind: sota_card
operator: elementwise
backend: pytorch_inductor
gens: [gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz]
regimes: [both, training]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
  - https://rocm.blogs.amd.com/artificial-intelligence/pytorch-amd-gpus/README.html
  - https://github.com/pytorch/pytorch/pull/143286
---

# elementwise × pytorch_inductor

## TL;DR
Inductor is the **default, automatic** SOTA for elementwise on ROCm: under `torch.compile` it **fuses
pointwise/reduction chains into single ROCm-Triton kernels** with zero user code — exactly the win
[../fusion.md](../fusion.md) describes. You don't write a kernel; you let Inductor merge N pointwise ops
(plus a trailing reduction) into one HBM pass. For an isolated pointwise op it generates a Triton kernel
equivalent to [triton.md](triton.md); the value is the **graph-level fusion**.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Inductor pointwise/reduction fusion → ROCm-Triton | `torch._inductor` | gfx942/950, bf16/fp16/fp32 | fewer kernels + less HBM traffic (fusion removes intermediate round-trips); GEMM-fusion bench geomean **1.36→1.42×** w/ ROCm configs (PyTorch-reported, 2025) | any `torch.compile` graph with pointwise chains |

The headline isn't a single-op speedup — it's **kernel-count and byte-traffic reduction** across the
graph. Inductor merges adjacent pointwise nodes (and a following reduction) so the data is read once,
computed through the whole expression, written once.

## Config space / knobs
| knob | value | effect |
|---|---|---|
| (default) fusion | on | pointwise/reduction chains fused automatically — no flag needed |
| `config.max_autotune.pointwise` | `True` | also **tune pointwise/reduction tiling** (BLOCK, warps) |
| `TORCH_COMPILE_DEBUG` | `1` | dump `output_code.py` to see the fused Triton kernel |
| `TORCHINDUCTOR_BENCHMARK_KERNEL` | `1` | per-kernel timing of the fused result |
| `ROCmConfigHeuristic` | (auto) | ROCm-tuned Triton configs (`matrix_instr_nonkdim`/`waves_per_eu`/`kpack`) — relevant when the chain ends a GEMM |

## Numerics / parity
Fusing reorders nothing in a pure pointwise chain (per-element), so parity holds; but a fused
GEMM-epilogue add happens in fp32 *before* the bf16 round (slight accuracy *gain* vs unfused). If you A/B
against an unfused reference, expect tiny bf16 LSB diffs at the fusion boundary — not a bug. See
[../numerics.md](../numerics.md).

## Integration (rebind seam)
- `torch.compile(model)` — fusion is on by default; `mode="max-autotune"` adds tiling autotune.
- Opaque torch custom ops (e.g. AITER, or your own HIP/Triton kernel registered via
  `direct_register_custom_op`) are **not** decomposed — Inductor fuses the *surrounding* pointwise into
  them, not across them.
- Generated `output_code.py` is itself an editable seam for a Tier-C rewrite.

## Pitfalls & anti-patterns
- A custom op registered as opaque **blocks fusion across it** — keep small glue pointwise in eager torch
  (so Inductor sees it) rather than hiding it in a custom op, unless the op is the hand-tuned hot path.
- Dynamic shapes re-trigger codegen; warm the cache (`TORCHINDUCTOR_CACHE_DIR`).
- A non-contiguous reshape/transpose between pointwise nodes can block fusion or force a strided
  (non-128-bit) load in the fused kernel.

## How to verify
`TORCH_COMPILE_DEBUG=1` → `output_code.py` shows one fused Triton kernel for the chain (not N kernels);
rocprof kernel count + `FETCH_SIZE`/`WRITE_SIZE` drop vs eager.

## Alternatives / cross-links
[triton.md](triton.md) (manual authoring of the same kernel) · [hip.md](hip.md) ·
[../fusion.md](../fusion.md) · backend overview
[`../../../backends/pytorch_inductor/overview.md`](../../../backends/pytorch_inductor/overview.md),
[`../../../backends/pytorch_inductor/max_autotune.md`](../../../backends/pytorch_inductor/max_autotune.md).

## Sources
- Inductor pointwise/reduction fusion (config surface): https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
- ROCm Inductor/Triton on MI300X: https://rocm.blogs.amd.com/artificial-intelligence/pytorch-amd-gpus/README.html
- ROCm GEMM/pointwise autotune configs (1.36→1.42×): https://github.com/pytorch/pytorch/pull/143286
