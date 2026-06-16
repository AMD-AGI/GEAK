---
title: transpose on HIP — SOTA card
kind: sota_card
operator: transpose
backend: hip
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
---

# transpose × HIP

## TL;DR
HIP is the right backend for a standalone transpose: full control of the LDS tile, its **padding/XOR
swizzle**, 128-bit `ds_*_b128` access, and (gfx950) the **`ds_read_*_tr_b16`** hardware transpose. But the
first question is always *"can I delete this transpose?"* ([[operators/transpose/fusion.md]]) — author a
HIP transpose only when the move is unavoidable.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| LDS-tiled transpose, XOR-swizzled, `ds_*_b128` | this card / [[languages/hip_cpp/lds_async.md]] | gfx942, all dtypes | HBM-bound (≈`2·bytes/5.3 TB/s` target) with ~0 bank conflicts; no public single-kernel number — **honest gap**, measure on your shape | gfx942, any standalone transpose |
| `+1`-padded LDS tile | AMD lds-bank-conflict blog | gfx942 | conflict-free at +12.5–25% LDS | quick/no-swizzle, LDS budget permits |
| `ds_read_b64_tr_b16` transpose-on-read | CDNA4 ISA (DS) / `amdgpu.ds_read_tr` | **gfx950 only**, 16-bit elts | crossbar transpose, no swizzle/2nd barrier | gfx950 bf16/fp16 |

No vendor publishes a standalone-transpose GB/s for CDNA — the operator is normally fused. Treat the
target as the HBM roofline and verify with rocprof.

## Config space / knobs
- **Tile**: 32×32 or 64×64 (LDS = `tile²·elt_size`; keep ≥2 wg/CU vs 64 KB CDNA3 / 160 KB CDNA4).
- **Conflict fix**: `[T][T+1]` padding **or** `col XOR (row & (banks-1))` swizzle (prefer swizzle).
- **Vectorize**: `float4` HBM (`global_*_dwordx4`) + `ds_*_b128`; `__restrict__`, 16-B alignment.
- **Block** = multiple of 64 (256 typical); **grid ≥1024** wg.
- **gfx950**: replace the swizzled column read with `__builtin_amdgcn_ds_read_tr*` (no padding needed).

## Numerics / parity
Byte-exact (`torch.equal`), no tolerance. Test ragged dims (odd M/N) for partial-tile masking. See
[[operators/transpose/numerics.md]].

## Integration (rebind seam)
Standalone `.hip` compiled with `hipcc --offload-arch=gfx942[ --offload-arch=gfx950]`, bound via a torch
custom op. In practice you rarely register it — you fold it into the GEMM operand layout or
[[operators/layout_shuffle/overview.md]]. There is **no aiter standalone-transpose seam** (aiter fuses).

## Pitfalls & anti-patterns
- ⚠ **Naive `tile[T][T]` column read = 4–32-way conflict, −75% LDS BW.** Always pad or swizzle.
- ⚠ Direct-to-LDS **without** the swizzle = the classic regression (201M conflicts / −28%).
- ⚠ `ds_read_tr` is **gfx950-only** — guard the gfx942 build or it fails to lower.
- ⚠ Shipping a standalone transpose at all when it could be fused = a needless HBM round-trip.

## How to verify
rocprof-compute / rocprofv3 → LDS bank-conflict counter ≈ 0 **and** kernel HBM-bound; ISA shows
`ds_*_b128`, `global_*_dwordx4` (and `ds_read_*_tr_*` on gfx950); oracle `torch.equal`.

## Alternatives / cross-links
[backends/triton.md](triton.md) · [[operators/transpose/tuning.md]] · [[operators/transpose/fusion.md]] ·
[[languages/hip_cpp/lds_async.md]].

## Sources
- XOR swizzle vs padding, −75% BW, rocprof verify: https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html
- LDS banks / access phases / 64 KB-CU: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- `ds_read_*_tr_b16` (gfx950): AMD CDNA4 ISA reference (DS) + https://mlir.llvm.org/docs/Dialects/AMDGPU/.
