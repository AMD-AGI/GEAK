# Gluon Shared Memory, Async, And AOT (gfx950 / gfx942)

Required companions: `imports-and-launching.md` and the relevant layout, memory,
or matrix reference. Use this file only after a simpler layout, memory, or matrix
candidate is correct. gfx1250 descriptor / TDM / tensor-memory paths are out of
scope.

## Shared Memory And Async

Shared memory, barriers, and async copy on CDNA:

- `allocate_shared_memory`;
- `barrier` / `fence_async_shared`;
- swizzled or padded shared layouts (`SwizzledSharedLayout`,
  `PaddedSharedLayout`) — see `../tile-programming/layout-recipes.md` for the
  padding-vs-swizzle bank-conflict recipe;
- async transfer phases: issue, commit, wait, consume (the basis of the pipeline
  layer, `../tile-programming/pipeline.md`).

API names and layout contracts are target-specific; confirm against local source
before use (`../missing-doc-protocol.md`).

## AOT Signature And Prebuilt Notes

When AOT, prebuilt, or wrapper-sensitive paths are involved, check:

- target triple and warp-size assumptions (`hip:gfx950:64` / `hip:gfx942:64`);
- launch attributes such as `num_warps`, `num_ctas`, `waves_per_eu`, and
  `num_stages`;
- pointer-type hints or divisibility assumptions encoded in the signature;
- disappearing constexpr values between source and AOT signature;
- scratch behavior and whether wrappers reject nonzero scratch sizes;
- prebuilt and fallback selection logic.

Compile success alone does not prove package/runtime contract validity. Do not
delete fallback gates or prebuilt selection just because a local JIT path
compiles. Triton minor-version sensitivity for AOT metadata lives in
`../hardware/capability-matrix.md` / `../hardware/planning-constants.md`.

## Acceptance Checklist

```text
simple anchor path already correct:
target family (gfx950 / gfx942):
source-proven API names:
shared layout constraints + bank-conflict plan:
launch attributes:
wrapper or artifact selection:
correctness oracle / benchmark boundary / fallback policy:
```
