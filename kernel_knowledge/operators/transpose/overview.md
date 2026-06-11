---
title: transpose — overview
kind: operator_overview
operator: transpose
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html
  - https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
---

# transpose  (`B[j,i] = A[i,j]`)

## TL;DR
A pure-bandwidth memory shuffle whose **only** performance lever on CDNA is staging through LDS without
**bank conflicts**: a naive `__shared__ tile[T][T]` makes the column read hit one bank → up-to-4-way
conflict → **−75% effective LDS bandwidth**. The fix is a **+1 pad** or (preferred on AMD) an **XOR
swizzle** of the column index — zero extra LDS, restores full BW. On CDNA4 (gfx950) the new
**`ds_read_b*_tr_b16` read-with-transpose** LDS instructions do the transpose in the LDS crossbar and
remove the staging entirely.

## Math contract
`B[N,M] = A[M,N]ᵀ` — element move only, no arithmetic, dtype preserved. Variants: 2-D matrix transpose,
batched/permute (`torch.permute` of a 3D/4D tensor → a generalized stride remap), and the **implicit**
transpose inside a GEMM (`transpose_b` is *not* this op — it is folded into the MFMA operand layout; see
[[operators/dense_gemm/overview.md]]). A standalone transpose op only appears when a layout cannot be
fused into the consumer (e.g. an attention `K`/`V` reshape, an activation `[B,S,H]→[B,H,S]`).

## Shape regimes
- **Square/large 2-D** (weights, activations): bandwidth-bound; tile 32×32 or 64×64 staged in LDS.
- **Skinny / batched permute** (attention reshapes): often **fusible** into the producer/consumer — a
  standalone transpose here is usually an anti-pattern (extra HBM round-trip). Prefer fusing.
- The op is **HBM-bandwidth bound**: ideal time ≈ `2·bytes / 5.3 TB/s` (read once, write once) on MI300X.

## Where it matters (Amdahl)
Standalone transpose is rarely a top-N kernel on a tuned LLM serving path — most transposes are folded
into GEMM operand layout or attention reshapes. It matters as a **building block** (the LDS-transpose
pattern is reused by GEMM operand staging, [[operators/layout_shuffle/overview.md]] weight pre-shuffle,
and gather/scatter), and when a framework emits an *unfused* `permute`+`contiguous` that shows up in a
profile — the fix is fusion, not a faster transpose.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| hip | 🟢 sota (full LDS/swizzle control; ds_read_tr on gfx950) | [backends/hip.md](backends/hip.md) |
| triton | 🟡 competitive (`tl.trans`/2-D load+store; padding via layout) | [backends/triton.md](backends/triton.md) |
| ck | 🟢 (CK-Tile `TileWindow` applies XOR swizzle automatically) | backends/ck.md (P2) |
| aiter | ⚪ na (no standalone transpose op; transposes are fused) | — |

## Fusion neighbors
The whole point: **don't ship a standalone transpose.** Fold it into (a) the GEMM operand layout
(`transpose_b`, [[operators/dense_gemm/overview.md]]); (b) the consumer's load (transpose-on-read into
LDS); (c) [[operators/layout_shuffle/overview.md]] when the target is an MFMA-friendly weight layout. See
[fusion.md](fusion.md).

## Numerics
Byte-exact element move — no accumulation, no tolerance. dtype unchanged (an fp8/int8 transpose just
moves bytes). See [numerics.md](numerics.md).

## How to bench
Isolated: time `B = A.t().contiguous()` vs an authored kernel on `[M,N]` for M,N ∈ {1k,4k,8k,16k};
oracle = `torch.allclose(out, ref)` (exact). Profile with `rocprof-compute`/`rocprofv3` and read the
**LDS bank-conflict** counter — a good kernel shows ~0 conflicts and HBM-bound time.

## Sources
- LDS bank conflict on transpose, XOR swizzle vs padding, 4-way = 75% BW loss: https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html
- CK-Tile LDS bank model (32 banks×4B CDNA3 / 64 banks CDNA4, access phases): https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html
- CDNA4 `ds_read_*_tr_b16` read-with-transpose: AMD CDNA4 ISA reference guide (DS instructions) + https://mlir.llvm.org/docs/Dialects/AMDGPU/ (`amdgpu.ds_read_tr`/ROCDL gfx950+).
- LDS bank model cross-ref: [[hardware/shared/memory_model_lds_bank.md]], [[languages/hip_cpp/lds_async.md]].
