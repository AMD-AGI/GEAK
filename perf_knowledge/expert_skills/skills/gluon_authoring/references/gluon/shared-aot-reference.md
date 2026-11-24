# Gluon AOT, Prebuilt, And Wrapper Contracts (gfx950 / gfx942)

This page owns the packaging boundary only. Shared memory: `smem-lds-reference.md`; shared-layout
constructors: `layout-reference.md ## Shared Layouts: The Three Constructors`; async copy:
`memory-reference.md`; fill-vs-read ordering: `pipeline/async-ordering.md ## Ordering the fill
against the read — class S` (no `fence_async_shared` on AMD — `appendix-api.md ## Spellings that do
not exist on the CDNA Gluon path`).

## AOT Signature And Prebuilt Notes

- Target triple and warp size: `hip:gfx950:64` / `hip:gfx942:64`.
- Launch attributes `num_warps`, `num_ctas`, `waves_per_eu`; `num_stages` is inert on the Gluon path
  and survives only as a signature field.
- Pointer hints / divisibility in the signature; constexprs that disappear between source and AOT
  signature; wrappers that reject nonzero scratch; prebuilt vs fallback selection. A local JIT
  compile proves none of these, so a JIT compile is not evidence that a fallback gate is removable.

**Three version surfaces.** (1) A 3.8.0-only *symbol* (`compute_efficient_padded_shared_layout`,
defaulted `._reinterpret`, `atomic_scatter_*`) makes the source unbuildable on older toolchains.
(2) A changed triple / launch attribute makes the artifact wrong while the source builds. (3)
**`loop_unroll_factor` is accepted on every version but only read by a 3.8.0 pass: an older
toolchain silently produces a correct artifact without the transform** — record the toolchain minor
of the artifact; do not infer it (`pipeline/loop-knobs-and-targets.md ## Roll the loop (cut i-cache pressure)`).
