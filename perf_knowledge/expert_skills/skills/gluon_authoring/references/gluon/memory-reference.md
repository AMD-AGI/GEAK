# Gluon Memory Reference (gfx950 / gfx942)

Required companions: `imports-and-launching.md` for launcher rules and
`layout-reference.md` for distributed offset/mask layouts. Per-target memory-path
support lives in `../hardware/capability-matrix.md`.

Use this file when deciding between generic `gl.load` / `gl.store` and AMD buffer
ops, or when debugging memory-path dtype, fallback, offset, or store issues. This
is the **memory-path layer** of the backbone.

For non-matrix Gluon body work, keep the same discipline: first prove the generic
memory anchor, then add buffer ops, shared staging, or async copy only when they
remove a named body cost (branchy masks, address-register pressure, traffic,
repeated layout conversion, or a consumer-layout mismatch).

## Start With Generic Memory Ops

Start with generic `gl.load` / `gl.store` unless there is a clear AMD memory
mechanism.

Move to AMD buffer ops only when:

- the path is streaming or memory-bound;
- an existing AMD Gluon path uses buffer operations;
- dtype, fallback value, value layout, and target family are known;
- the memory path is hot enough to pay for extra specificity;
- a same-shape correctness probe has passed against the generic anchor.

`buffer_load` / `buffer_store` are namespaced by target: `gl.amd.cdna4.*` on
gfx950, `gl.amd.cdna3.*` on gfx942. Namespace swap alone is not an optimization
direction (equivalent operand types can generate the same ISA). Do not add buffer
ops only to avoid a `where`; measure the memory path first. Buffer paths have
been observed to compile and run but read wrong values — require a correctness
gate, not a blanket rejection.

## Buffer Pointer And Offset Rules

- `buffer_load(base_ptr, offsets, ...)` / `buffer_store(base_ptr, offsets, ...)`
  use a scalar base pointer plus distributed offsets.
- They are not textual replacements for `gl.load(ptr + offsets)`.
- offsets must be int32 or uint32 on supported AMD buffer paths.
- int64 offsets from page/block ids multiplied by large strides are a hard
  constraint, not a performance preference.
- treat offsets as byte offsets unless operator-local source proves otherwise.
- for non-1-byte element types, re-check source expressions written as element
  offsets and multiply by bytes per element when byte offsets are required.
- the base pointer must be scalar pointer-like; restructure `ptr + offsets`
  tensor pointers before buffer ops.

## Buffer Dtype Rules

- `buffer_load(..., other=...)` casts the fallback to the loaded pointer element
  type. For generated code prefer a typed Gluon value:

```python
other = gl.full(shape, 0.0, ptr.dtype.element_ty, layout=layout)
```

- `buffer_store(..., stored_value=...)` must store a value whose element dtype
  matches the destination pointer element type. Cast widened accumulators before
  storing.

## Async Copy To Shared

`buffer_load_to_shared` (CDNA4 async copy) requires 32-bit offsets and an
offset tensor whose distributed layout matches rank, dtype, unit, and the
shared-memory consumer. A `BlockedLayout` lowering failure is **layout-contract
evidence** before it is a build ceiling — retry with a source-proven
`DistributedLinearLayout` matched to the consumer before recording a toolchain
ceiling. Do not interleave ordinary loads/stores with async paths without a
traffic/scheduling hypothesis. This is the main mechanism behind the pipeline
layer (`../tile-programming/pipeline.md`).

### Minimum per-thread granularity (applicability by dtype)

`buffer_load_to_shared` also requires the **per-thread chunk to clear the HW
async-copy granularity floor (~16 B/thread)**. A sub-floor load (e.g. a
small-element dtype with a narrow contiguous run) is **rejected at lowering**
(an `unrealized_conversion_cast` / lowering failure), which is **not** a build
ceiling and **not** fixable by switching the `BlockedLayout` family — it is the
per-thread *byte count*. The fix is to raise `size_per_thread` on the contiguous
dim until per-thread bytes clear the floor:

| Operand dtype | bytes/elem | contiguous elems/thread to clear ~16 B |
| --- | --- | --- |
| bf16 / fp16 | 2 | >= 8 |
| fp8 / int8 | 1 | >= 16 |
| fp4 (e2m1, packed 2/byte) | 0.5 | >= 32 |

Tensors that are inherently sub-floor (a few elems/thread — typically block
scales) cannot use direct-to-LDS at all; route them through the register-staging
side path GR -> LW -> LR (`../tile-programming/low-precision.md`). This is a
layout/granularity constraint, not an async-path build ceiling — do not record it
as a `buffer_load_to_shared` capability failure.

> **Arch note (the floor is per arch, not universal).** The ~16 B/thread figure is
> the **CDNA4** 128-bit direct-to-LDS chunk. On **gfx942 / CDNA3 the widest
> direct-to-LDS load is 32-bit (4 B, 2×bf16) per thread** — 128-bit is CDNA4-only
> (`supportsDirectToLdsLoadBitWidth`: CDNA3 = {32}, CDNA4 = {128, 32}, GFX1250 =
> {128, 64, 32}). So on gfx942 the async path moves fewer bytes per instruction and
> tends to become `s_waitcnt`-bound; measure it against sync staging rather than
> assuming a win (`../hardware/capability-matrix.md ## Direct-to-LDS granularity
> (per arch)`).

### Shared-layout family + transpose-on-read (layout dependency)

Not every shared layout lowers for the async direct-to-LDS path:

- `SwizzledSharedLayout` lowers for `buffer_load_to_shared`; the padded-identity
  variant (`PaddedSharedLayout.with_identity_for`) has been seen to **fail to
  lower** for the async path — record as a known layout delta, not a build ceiling,
  and prefer `SwizzledSharedLayout` (or a source-proven `DistributedLinearLayout`
  matched to the consumer) for async.
- **Transpose-on-read**: to consume one LDS tile in two operand orientations
  (instead of storing it twice), store it once in natural order and read the
  transposed operand via `smem.permute((1, 0)).load(dot_layout)`. The conflict
  behaviour of reading a tile both ways and its mitigation live in
  `../tile-programming/layout-recipes.md ## Padding vs swizzle (LDS bank conflicts)`.

## gfx950 <-> gfx942 delta

- buffer ops: `gl.amd.cdna4.buffer_load/store` (gfx950) vs
  `gl.amd.cdna3.buffer_load/store` (gfx942);
- **async direct-to-LDS IS available on gfx942 — a namespace gap is not a silicon
  gap.** The `gl.amd.cdna3` namespace has no `async_copy` submodule (no
  `commit_group` / `wait_group` under `cdna3`), but
  `gl.amd.cdna4.async_copy.buffer_load_to_shared` + `commit_group` / `wait_group` +
  `load_shared_relaxed` **compile and run on gfx942** (verified correct). Do **not**
  conclude "gfx942 has no async copy" from the empty `cdna3.async_copy` — probe the
  `cdna4.async_copy` entry first, and only fall back to synchronous register staging
  once that path is shown non-viable for the specific layout. Backend truth:
  direct-to-LDS width is arch-gated (`supportsDirectToLdsLoadBitWidth`: CDNA3 = {32},
  CDNA4 = {128, 32}, GFX1250 = {128, 64, 32}) — see
  `../hardware/capability-matrix.md ## Direct-to-LDS granularity (per arch)`.
- **Real gfx942 async limits (verify each — these are layout/width facts, not the
  namespace myth):**
  - gfx942 direct-to-LDS granularity = **32-bit / 2×bf16 per thread** (the 128-bit
    chunk is CDNA4-only). A non-GFX1250 target also requires the fast dim to be
    **exactly covered** by `threads_per_warp * size_per_thread` (no replication).
  - transpose-B / complex dot-operand offset layouts under `AMDMFMALayout(v3)` +
    `SwizzledSharedLayout` can **fail LLVM translation**
    (`builtin.unrealized_conversion_cast`); the fix is `PaddedSharedLayout` + a
    source-proven `DistributedLinearLayout` matched to the consumer (a block-scale
    GEMM that stages a transposed operand is one upstream example of this pattern).
  - the **generic** `ttg.async_copy_global_to_local` is explicitly illegal on the
    gfx942 backend — use the `cdna4.async_copy` entry, not the generic op.
  - `compute_efficient_padded_shared_layout` asserts **v4-only** (unavailable on
    gfx942 / v3).
- Synchronous register staging (`buffer_load` -> shared `store` -> `LR` -> `DOT`,
  ordered by LDS read-after-write) is the fallback, not the default.

Full status/evidence: `../hardware/capability-matrix.md` (memory & scheduling
matrix; `## Direct-to-LDS granularity (per arch)`) and
`../hardware/cdna3-gfx942.md`.

## Reduce divergence (predicate, not branch)

When VALU active-threads/wave is low (Branch Util high), replace boundary `if`-branches
around loads with predicated `buffer_load(..., mask=, other=)` so the wave stays
converged; pad/pow2-align ragged axes so the mask is uniform per wave.

```python
# predicated load instead of `if in_bounds: load` — uniform control flow
mask = offs_m[:, None] < M
a = gl.amd.cdna3.buffer_load(a_ptr, offs, mask=mask, other=0.0)   # no wavefront branch
```

Ref: `cdna3/cdna4.buffer_load(mask=, other=)` predication (upstream
`triton/experimental/gluon/language/amd/cdna3`). Note CDNA uses `mask=` on buffer ops
where gfx1250 uses a `pred=` on `tdm.async_load`. Verify: active-threads/wave up toward
wave_size, Branch Util down.

## Minimal Acceptance Checklist

```text
generic gl.load anchor is correct:
base pointer is scalar:
offset dtype is int32 or uint32:
offset unit is byte or source-proven:
mask layout matches offsets:
fallback dtype matches pointer element type:
stored value dtype matches destination:
benchmark boundary:
```

If any item is unknown, keep the generic path or shrink the probe. Symptom -> fix
routing: `../failure-triage.md`.
