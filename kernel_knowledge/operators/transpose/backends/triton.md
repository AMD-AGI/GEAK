---
title: transpose on Triton — SOTA card
kind: sota_card
operator: transpose
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
---

# transpose × Triton

## TL;DR
A 2-D block load + `tl.trans` + transposed store is the fast way to **author** a transpose, and it's the
form Inductor emits for an unfused `permute`+`contiguous`. Triton handles the LDS swizzle for you via its
layout assignment, but you have **less direct control** of padding/swizzle than HIP — the honest play is
still to **fuse the transpose into the producer/consumer** ([[operators/transpose/fusion.md]]) and only
keep a standalone Triton transpose for prototyping or genuinely-unfusable layouts.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| 2-D block transpose (`tl.load` block → `tl.trans` → store) | [[languages/triton_amd/patterns.md]] | gfx942/950, bf16/fp16 | HBM-bound on coalesced tiles; no public AMD GB/s — measure | prototyping / Inductor path |
| fold into the consumer kernel | [[operators/transpose/fusion.md]] | both | removes the pass entirely | **preferred** |

## Config space / knobs
- `BLOCK_M`/`BLOCK_N` tile (32–64 each); `num_warps` 2–4 (memory-bound), `num_stages=1`.
- Coalescing: ensure the **read** is coalesced (consecutive lanes → consecutive HBM) so the compiler emits
  `global_load_dwordx4`; the transpose then makes the *store* the strided side — pick the block so the
  strided side goes through LDS, not HBM.
- AMD knobs go in `triton.Config({...})` (e.g. `waves_per_eu`), not Python vars.
- `OPTIMIZE_EPILOGUE=1` to drop redundant converts on a fused cast.

## Numerics / parity
Byte-exact; oracle `torch.equal`. Test ragged dims (Triton masks handle partial tiles — verify). See
[[operators/transpose/numerics.md]].

## Integration (rebind seam)
`@triton.jit` called from a torch custom op, or emitted automatically by `torch.compile` /
Inductor `max-autotune` for `x.permute(...).contiguous()`. No aiter seam (aiter has no standalone transpose).

## Pitfalls & anti-patterns
- ⚠ A standalone `permute().contiguous()` showing in a profile = fusion opportunity, not "tune the transpose".
- ⚠ Triton gives indirect LDS control — if rocprof still shows bank conflicts, the layout/tile choice is the
  lever (or drop to HIP for explicit swizzle, [backends/hip.md](hip.md)).
- ⚠ `num_warps=8` carried from NVIDIA → VGPR spill; memory-bound transpose wants 2–4.

## How to verify
`TRITON_PRINT_AUTOTUNING=1` + rocprof bank-conflict counter ≈ 0 and HBM-bound; `AMDGCN_ENABLE_DUMP=1` →
want `global_*_dwordx4`, `ds_*_b128`. Oracle `torch.equal`.

## Alternatives / cross-links
[backends/hip.md](hip.md) (explicit swizzle / gfx950 ds_read_tr) · [[operators/transpose/tuning.md]] ·
[[operators/transpose/fusion.md]] · [[languages/triton_amd/patterns.md]].

## Sources
- Triton AMD tuning (coalescing, OPTIMIZE_EPILOGUE, LDS): https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- LDS bank-conflict background: https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html
- HIPOptions / layout passes: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
