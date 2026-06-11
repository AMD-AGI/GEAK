---
title: TileLang autotuning on MI300X
kind: language
gens: [gfx90a, gfx942]
dtypes: [fp16, bf16]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
  - https://github.com/tile-ai/tilelang
---

# TileLang autotuning

## TL;DR
TileLang couples `@tilelang.autotune` with `@tilelang.jit`: you define a candidate config space with
`itertools.product`, and the tuner JIT-compiles and times each, caching inputs. For the MI300X
FlashAttention kernel this is **108 candidate configs** searched in **~1 second**, landing on
`block_M=128, block_N=32, threads=512, enable_rasterization=True`. This is the main reason TileLang beats
Triton ~1.53× on FA without hand tuning. See [primitives.md](primitives.md) for the knobs being swept.

## Core concepts — the decorators
```python
@tilelang.autotune(configs=get_configs(), cache_input_tensors=True, supply_prog=supply_tensors_gpu)
@tilelang.jit(out_idx=[3])
def flashattn(...): ...
```
- `configs=get_configs()` — a list of candidate dicts (the search space).
- `cache_input_tensors=True` — reuse the GPU input tensors across trials (cheap re-timing).
- `supply_prog=...` — supplies the input tensors for timing.
- `@tilelang.jit(out_idx=[3])` — JIT-compiles each variant; `out_idx` marks the output arg.

`get_configs()` enumerates the full Cartesian product:
```python
configs = list(itertools.product(block_M, block_N, num_split_q, threads,
                                 num_stages, qk_coalesced_width, v_coalesced_width))
# 108 candidates for the FA kernel
```

## The levers (the swept space)
| param | role |
|---|---|
| `block_M`, `block_N` | tile sizes (optimal FA: 128, 32) |
| `num_split_q` | split Q across blocks (occupancy on MI300X's 304 CUs) |
| `threads` | block size (optimal: 512) |
| `num_stages` | `T.Pipelined` prefetch depth (LDS-bounded at 64 KB) |
| `qk_coalesced_width`, `v_coalesced_width` | vectorized global-load widths (≥128 bit) |
| `enable_rasterization` | block-schedule swizzle for cache reuse (optimal: True) |

Keep the candidate lists small and physically valid (respect the 64 KB LDS and 512-VGPR budgets) so the
product stays in the ~100s, not 1000s — that is why the search is ~1 s.

## Pitfalls
- An uncapped `itertools.product` explodes; bound each list to plausible values for the shape.
- A config that overruns LDS/VGPR will fail to compile or spill — the tuner should skip it, but a manual
  pin can break.
- The autotuned config is **shape-specific** (b/h/s/d) and **build-specific** (ROCm/TileLang version) —
  re-tune per serving shape and after upgrades; don't ship a frozen config as portable.
- Tiny default search spaces can miss the optimum on unusual shapes — widen deliberately.

## Verify
- Print the winning config and confirm it is sane (e.g. FA: `128/32/512`, rasterization on).
- Re-measure latency vs Triton/PyTorch at the exact shape (≥3 warm repeats, median) — the blog reports
  FA latency 0.36 ms (TileLang) vs 0.55 ms (Triton) vs 0.97 ms (PyTorch) at b=1,h=8,s=4096,d=128 on
  ROCm 7.0.1 / TileLang 0.1.7 (vendor-reported).

## Sources
- TileLang FlashAttention on MI300X (108 configs, itertools.product, autotune+jit decorators, ~1 s search, optimal 128/32/512 + rasterization, latency table): https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-tilelang-kernel/README.html
- tile-ai/tilelang (autotune API): https://github.com/tile-ai/tilelang
