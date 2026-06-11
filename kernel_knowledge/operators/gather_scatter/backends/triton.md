---
title: gather_scatter on Triton — SOTA card
kind: sota_card
operator: gather_scatter
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang
  - https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# gather_scatter × Triton

## TL;DR
Triton is the SOTA *authoring* path for gather/scatter on AMD: a **one-program-per-row, BLOCK_D-tiled** load
gives coalesced row payloads even with a scattered index, and `tl.atomic_add` covers scatter-reduce. It's
also what Inductor emits for `index_select`/`index_add`. The one hard limit: **no 2-D scalar indexing into
an accumulator**, so a fully-fused down-proj+scatter isn't expressible — the scatter stays its own kernel.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| BLOCK_D-tiled gather (`out[pid]=in[idx[pid]]`) | [[operators/gather_scatter/tuning.md]] §1 | gfx942/950, bf16/fp16 | row payload near HBM roofline; MoE permute ~54% peak BW (community) | MoE permute, embedding |
| `tl.atomic_add` scatter-reduce | same | both | ~30% coalescing on AMD MoE sort path (community) — improving | MoE unpermute |
| Inductor `index_*` codegen | `torch.compile` | both | portable, unfused | the `torch.compile` path |

No first-party AMD GB/s published for a standalone op — measure on your `[N,H]` + index distribution.

## Config space / knobs
- `BLOCK_D` over hidden (128–512) for 128-bit coalesced rows; `num_warps` 2–4; `num_stages=1` (memory-bound).
- Grid = `N` rows (≥1024); for small `N`, split `H` across programs to fill 304 CUs.
- AMD knobs in `triton.Config({...})`; `OPTIMIZE_EPILOGUE=1` if a cast/dequant is fused.

## Numerics / parity
gather exact; scatter-reduce non-deterministic (atomics) → `allclose` / task-accuracy gate. See
[[operators/gather_scatter/numerics.md]].

## Integration (rebind seam)
`@triton.jit` from a torch custom op; or auto-emitted by Inductor. In the MoE path it's the prologue gather
inside the Triton fused-MoE; aiter's catalog uses asm/CK for the production sort+grouped-GEMM
([backends/aiter.md](aiter.md)) — Triton is the editable/fallback path.

## Pitfalls & anti-patterns
- ⚠ **No 2-D scalar indexing into `acc`** → can't fully fuse down+scatter; keep scatter separate.
- ⚠ Un-tiled (per-element) gather → uncoalesced, bandwidth-starved; always BLOCK_D-tile the row.
- ⚠ Sort the index first or eat the ~30% coalescing penalty.
- ⚠ `num_warps=8` carried from NVIDIA → spill; memory-bound wants 2–4.

## How to verify
`TRITON_PRINT_AUTOTUNING=1`; rocprof memory chart (coalescing/L2) + HBM-bound row payload; ISA
`global_*_dwordx4`. Oracle: gather `torch.equal`, scatter `allclose`.

## Alternatives / cross-links
[backends/hip.md](hip.md) (full fuse + HW atomics) · [backends/aiter.md](aiter.md) (production MoE sort) ·
[[operators/gather_scatter/tuning.md]] · [[operators/gather_scatter/fusion.md]].

## Sources
- BLOCK_D tiling, 54% BW, 30% coalescing, 2-D scalar-index limit: https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang · https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/
- Triton AMD coalescing/tuning: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
