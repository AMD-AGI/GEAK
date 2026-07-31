# Gluon API (gfx950 / gfx942) — mechanism index

Owned, self-contained Gluon API reference for tile-programming-triton-gluon. **Start at
`../hardware/atlas.md`** (phase read order + backbone layer map). Organized by the
**tile-programming mechanism / backbone layer** you are working on, not as an
alphabetical API dump. gfx950 (CDNA4) is the default; gfx942 (CDNA3) is the
downgrade. RDNA4 client WMMA (gfx1201, R9700 / RX9070 XT): `rdna-wmma-reference.md`.
gfx1250 (CDNA5 / MI450) WMMA+TDM: structured sub-target, separate data-center fork.

Planning constants: `../hardware/planning-constants.md`. ISA shapes: `../hardware/isa-mechanisms.md`.
Support level and evidence: `../hardware/capability-matrix.md` (not restated here).
Compiler env knobs: `../tile-programming/compiler-contract.md`.

## Mechanism -> API route (by backbone layer)

| Backbone layer / need | Read | capability-matrix cells to check |
| --- | --- | --- |
| imports / `@gluon.jit` / launcher / host layout factories | `imports-and-launching.md` | Gluon capability level |
| memory path (`gl.load/store`, buffer ops, async copy) | `memory-reference.md` | memory & scheduling matrix |
| LDS layout (padding / swizzle for conflict-free `ds_read`) | `layout-reference.md` + `../tile-programming/layout-recipes.md` | `Swizzled`/`PaddedSharedLayout` rows |
| matrix (MFMA result/operand/convert/acc, scaled MFMA) | `matrix-reference.md` | matrix-op matrix + instr_shape rules |
| pipeline / slicing / scheduling | `../tile-programming/{pipeline,slicing,compiler-contract}.md` | version-sensitive knobs |
| low-precision side path (FP8/FP4 scaled) | `matrix-reference.md` + `../tile-programming/low-precision.md` | FP8/FP4/scaled rows |
| shared memory / async / AOT | `shared-aot-reference.md` | async-copy + AOT rows |
| gfx950 smoke / minimal probe | `gfx950-minimal-examples.md` | Gluon capability level |
| compile / lowering / correctness failure | `../failure-triage.md` | the failing target/dtype cell |

## Lowering-chain order (matrix path)

result layout -> operand layouts -> `convert_layout` -> accumulator -> target op
-> epilogue/store layout. Full chain in `matrix-reference.md`.

## gfx950 <-> gfx942 at a glance

| Aspect | gfx950 (CDNA4, default) | gfx942 (CDNA3, downgrade) |
| --- | --- | --- |
| MFMA op | `gl.amd.cdna4.mfma` | `gl.amd.cdna3.mfma` |
| MFMA layout | `AMDMFMALayout(version=4)` | `AMDMFMALayout(version=3)` |
| scaled MFMA | `cdna4.mfma_scaled` (mandatory for FP8/FP4; regular `mfma` has no fp8/fp4 intrinsic) | target-specific-blocker -> plain `tl.dot` |
| FP8 / FP4 | native OCP `e4m3fn` / `e2m1`; Gluon path is `mfma_scaled` (None scale -> unit e8m0) | regular `cdna4.mfma` has **no fp8/fp4 intrinsic**; `cdna3.mfma` FP8 version/API-blocker; `e4m3fnuz` may upcast on gfx950; plain `tl.dot` (scale-less `tt.dot_scaled`) ok as comparator |
| buffer ops | `gl.amd.cdna4.buffer_load/store` | `gl.amd.cdna3.buffer_load/store` |

Namespace is not architecture support; full status + evidence live in
`../hardware/capability-matrix.md`.

## The TTGIR -> Gluon bridge

When transcribing a plain-Triton kernel (`../phases/transcribe.md`), map the
plain `.ttgir` layouts to these Gluon objects 1:1 via the recovery table in
`../tile-programming/layout-recipes.md`. That file is the bridge from budget /
transcription to this API.

## Wave-size note

gfx950 and gfx942 are both **wave64** (`product(threads_per_warp) == 64`). All
layout recipes here assume wave64.
