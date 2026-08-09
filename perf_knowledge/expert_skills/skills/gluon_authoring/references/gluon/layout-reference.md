# Gluon Layout Reference (gfx950 / gfx942, wave64)

Required companion: `imports-and-launching.md` for imports, launcher contract,
and host-created layout rules. For matrix operand/result layouts continue with
`matrix-reference.md`. The tile recipe + TTGIR -> Gluon recovery map live in
`../tile-programming/layout-recipes.md`.

Use layouts to remove a measured cost, not as cosmetic spelling. The usual
body-level layout mechanisms are: coalesced offset ownership, cheaper
mask/broadcast construction, avoiding hot-loop `convert_layout`, matching a
shared-memory consumer, or providing matrix operand layouts for a material matrix
path.

## Layout Derivation

Derive the first layout from source facts before tuning it:

1. Recover the logical tile from source `tl.arange` bounds, tile shape, masks,
   reductions, matrix dimensions, and launcher constants.
2. Identify the physical contiguous dimension from pointer arithmetic and strides.
3. Assume **wave64** (gfx950/gfx942).
4. Check `size_per_thread * threads_per_warp * warps_per_cta` against the logical
   tile.
5. For each logical 2D or 3D expression, define a parent layout before deriving
   `SliceLayout`, `DotOperandLayout`, or shared layouts.
6. Keep shape-, target-, `num_warps`-, and instruction-dependent layouts in host
   layout-factory code unless operator-local source proves a
   `gl.constexpr = Layout(...)` JIT-body pattern.

Do not fix verifier failures by inserting arbitrary powers of two. If the layout
does not cover the logical tile, recompute it from the launch contract.

## BlockedLayout Constraints (wave64)

- `product(threads_per_warp) == 64` (both gfx950 and gfx942 are wave64);
- `warps_per_cta` is consistent with launch `num_warps`;
- each `size_per_thread` entry is a positive power of two;
- the coverage product must not exceed the logical tile unless source-local masks
  and ownership prove the extra lanes are valid;
- for vector memory paths, choose `size_per_thread` from intended vector width,
  not only from the smallest compiling value.

Construction recipe for a 2D tile:

```text
1. Choose warps_per_cta = [Wm, Wn] so Wm * Wn == num_warps.
2. Choose threads_per_warp = [Tm, Tn] so Tm * Tn == 64.
3. Compute size_per_thread = [M / (Wm * Tm), N / (Wn * Tn)].
4. Verify both entries are positive integers and cover the logical tile.
```

Known-good wave64 starting points (still verify locally):

| Tile | num_warps | size_per_thread | threads_per_warp | warps_per_cta | order |
| --- | --- | --- | --- | --- | --- |
| `[64, 64]` | 4 | `[4, 4]` | `[8, 8]` | `[2, 2]` | `[1, 0]` |
| `[64, 32]` | 4 | `[2, 4]` | `[8, 8]` | `[4, 1]` | `[1, 0]` |
| `[32, 64]` | 4 | `[4, 2]` | `[8, 8]` | `[1, 4]` | `[1, 0]` |

When a layout uses explicit `warps_per_cta`, launch `num_warps` is layout-coupled
and cannot be swept independently without redesigning all related layouts.

## DistributedLinearLayout Basis Design

`DistributedLinearLayout` basis vectors encode how register, lane, and warp bits
map to tensor-coordinate offsets. They can silently freeze tile sizes.

Rules:

- each basis offset for dimension `d` must be valid for requested `shape[d]`;
- a basis like `[128, 0]` requires the row dimension to include that offset, so it
  is incompatible with a `BLOCK_SIZE_M=128` tile;
- shrinking a tile often requires removing or redesigning the highest basis bit
  for the shrunken dimension, not only changing launcher config;
- changing a basis changes elements per thread and register pressure, so treat it
  as body/layout work rather than launch-only config search.
- async-copy offset tensors for `buffer_load_to_shared` often need a
  `DistributedLinearLayout`-style thread mapping that matches rank, units, and
  shared-memory consumer. A `BlockedLayout` lowering failure is layout-contract
  evidence before it is a toolchain ceiling.

Audit record:

```text
layout_name / shape:
reg_bases / lane_bases / warp_bases:
largest_offset_per_dim / tile_dim_coverage:
knobs_frozen_by_layout:
```

If a tile-size change causes an LLVM or layout-surjectivity error, inspect the
basis vectors before broadening config search.

## Shape APIs And Layout Propagation

- `gl.arange` creates a 1D distributed tensor and requires an explicit layout in
  generated performance code.
- A 2D `DotOperandLayout` is not an arange layout; derive a 1D `SliceLayout` from
  the parent or construct a 2D offset tensor in the parent layout.
- `reshape(..., can_reorder=True)` is not generally supported; preserve source
  transformation semantics.
- `permute`, `split`, and `join` preserve or infer layout through the semantic
  layer.
- `reshape` and `split` may infer `DistributedLinearLayout` even when the store
  path expects `BlockedLayout`; plan the post-transform store layout.

## `convert_layout` Decision Table

| Situation | Use `convert_layout`? | Reason |
| --- | --- | --- |
| Moving matrix operands into `DotOperandLayout` | yes | Matrix instructions require operand layouts. |
| Re-parenting a 1D slice before broadcast | no | Regenerate the index from the correct parent `SliceLayout`. |
| Fixing measured non-coalesced memory access | maybe | Benchmark against safe anchor and change one memory path at a time. |
| Repeated conversion inside the innermost loop | avoid | Layout movement can dominate the optimized work (`../gluon-negative-patterns.md`). |
| Equivalent-layout conversion | maybe with `assert_trivial=True` | Fail early if not a trivial reinterpretation. |
| Cosmetic conversion | no | It adds cost without a hypothesis. |

### Shared Memory Versus Register Conversion

When shared memory is used only to change layout, benchmark it against direct
register layout conversion. On targets with large register files such as
gfx950/CDNA4, direct `gl.convert_layout` can beat store-to-shared plus reload for
small hot-loop conversions. The winner is target-, tile-, and layout-dependent;
test both before assuming LDS staging is required.

## Slice And Broadcast Recipe

Broadcasting is parent-layout sensitive. Treat `[:, None]`, `[None, :]`,
`expand_dims`, masks, and offsets as layout operations.

`SliceLayout(dim, parent)` means dimension `dim` was removed from `parent`.
`gl.expand_dims(x, axis=dim)` requires `x.layout == SliceLayout(dim, parent)`.

For a 2D parent `[M, N]`:

| Goal | Index spelling | Required layout | Meaning |
| --- | --- | --- | --- |
| `[1, N]` | `x[None, :]` / `expand_dims(x, axis=0)` | `SliceLayout(0, parent_mn)` | M removed; x is an N-vector |
| `[M, 1]` | `x[:, None]` / `expand_dims(x, axis=1)` | `SliceLayout(1, parent_mn)` | N removed; x is an M-vector |

Rules:

1. Pick the logical parent layout for each 2D/3D expression.
2. Derive every broadcasted 1D index from `SliceLayout(axis, parent)` of that
   exact parent.
3. Create `SliceLayout` objects on the host and pass them as `gl.constexpr`.
4. Expand 1D tensors before combining them into offsets, masks, or strides.
5. Use separate index tensors for separate parent contexts (`idx_m_mn` vs
   `idx_m_mk`).
6. Do not use `convert_layout` to re-parent arbitrary 1D tensors before
   broadcasting.

The parent layout is part of a tensor's meaning; do not reuse `idx_x_xy` as
`idx_x_xz` even when both share symbolic dimension `X`.

## Pre-Run Scan

Before benchmarking generated Gluon code, scan for:

- leftover tensor-dataflow `tl.arange`, `tl.load`, `tl.store`, `tl.zeros`,
  `tl.full`, or `tl.dot`;
- runtime layout objects in generated `@gluon.jit` code;
- helper definitions that are never launched;
- launcher changes that silently alter path selection;
- index-tensor layout conversions added only for cleanup.

Symptom -> fix routing for layout/broadcast failures: `../failure-triage.md`.
