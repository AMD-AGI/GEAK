# Gluon Memory Reference (gfx950 / gfx942)

Generic `gl.load` / `gl.store` is the baseline; AMD buffer ops, shared staging and async copy each pay
when they remove a named cost. Per-target support: `../hardware/capability-matrix.md`. Distributed offset /
mask layouts: `layout-reference.md`.

## Buffer Pointer And Offset Rules

`buffer_load` / `buffer_store` live at `gl.amd.cdna4.*` (gfx950) and `gl.amd.cdna3.*` (gfx942). A
namespace swap alone is not an optimization. Buffer paths can compile, run and read wrong values —
gate each one on correctness against the generic anchor.

- **`buffer_store(stored_value, base_ptr, offsets, ...)` — the value comes FIRST**;
  `buffer_load(base_ptr, offsets, ...)`. Scalar base pointer + distributed offsets; restructure
  `ptr + offsets` tensor pointers first.
- **Offsets are ELEMENT offsets, not bytes** — pre-multiplying scales twice and silently reads wrong values.
- Offsets must be int32 / uint32; int64 offsets from page/block id x large stride are a hard
  constraint.
- `other=` is cast to the pointer element type — pass a typed value
  (`gl.full(shape, 0.0, ptr.dtype.element_ty, layout=layout)`). The stored value's dtype must equal
  the destination element type (cast widened accumulators first).
- `mask=` / `other=` predicates a load without branching, which keeps the wave converged.

## Async Copy To Shared

**Three preconditions, from the docstrings of `global_load_to_shared` and `buffer_load_to_shared`
(v3.8.0: "the following conditions must be met or lowering to LLVM will fail"):** per-thread
`size per thread * bits per element` must be 128 or 32 (of `offsets` / `ptr`); writes to `dest` must
be coalesced; a swizzled `dest` may only swizzle within a warp boundary. Root cause: the instruction
takes a per-lane global address but **one warp-uniform LDS address**.

- **"Coalesced" is the predicate `threads_per_warp[contig] * size_per_thread[contig] == tile extent
  along contig`** — `==`, not `>=` (over- or under-coverage makes the per-warp LDS address
  non-uniform). The failure is a translation error naming the enclosing function
  (`../pitfalls/negative-patterns.md ## A translation failure names the allocation, not the murderer`)
  and is indistinguishable from the width gate below; the fixes move different fields.
- `buffer_load_to_shared` needs 32-bit offsets and an offset layout matched to the shared consumer.
  A `BlockedLayout` lowering failure is layout-contract evidence before a toolchain ceiling; for a
  padded consumer the match is constructed: `## Deriving the producer layout for a padded async
  destination`. Ordering (`commit_group` / `wait_group` / barriers): `../tile-programming/pipeline.md`.
- **Merging two copies into one is a dependency edit as much as a memory one.** Measured,
  bit-exact: two `[32, 256]` copies vs one `[64, 256]` gave identical memory-side ISA, fewer total
  instructions (1279 -> 1252) and `s_waitcnt` (110 -> 76), yet **slower at most shapes**: VGPR
  174 -> 190 and one `wait_group` now gates both operands. Price a merge by A/B.

### Minimum per-thread granularity (applicability by dtype)

Legal per-thread widths are a set, not a floor: gfx950 128 or 32 bit
(`triton/experimental/gluon/language/amd/cdna4/async_copy.py` docstring), gfx942 32 bit only; 64 bit is
illegal on both (compile-verified 3.8.0). The failure strings:
`failed to translate module to LLVM IR` (`LLVM Translation failed for operation:
builtin.unrealized_conversion_cast`) from `buffer_load_to_shared`, `PassManager::run failed`
(`failed to legalize operation 'ttg.async_copy_global_to_local'`) from `global_load_to_shared`.
This is **not** a build ceiling and **not** fixable by switching the `BlockedLayout` family — move
`size_per_thread` on the contiguous dim so the byte count lands **exactly on** a legal width:

| Operand dtype | bytes/elem | contiguous elems/thread at 128 bit | at 32 bit |
| --- | --- | --- | --- |
| bf16 / fp16 | 2 | exactly 8 | exactly 2 |
| fp8 / int8 | 1 | exactly 16 | exactly 4 |
| fp4 (e2m1, packed 2/byte) | 0.5 | exactly 32 | exactly 8 |
| fp32 | 4 | exactly 4 | exactly 1 |

fp32 block scales therefore copy at 1 or 4 elems/thread; 2 (64 bit) does not lower (production:
`COPY_WIDTH = 1 if SPLITS > 1 else 4`). Register staging
GR -> LW -> LR (`../tile-programming/low-precision.md`) is the fallback for a width that misses the
set, not a route block scales belong on by kind.

**A legal width is necessary, not sufficient: the lane's elements must be contiguous in the
source.** Measured: `size_per_thread = 4` on `uint8` (32 bit, legal) fails LLVM translation once the
offsets carry a vendor-permuted order that puts a lane's 4 bytes 64 B apart; plain `gl.load` on the
same offsets gives four `global_load_ubyte`. **Same error text as an illegal width** — check the
offsets before concluding. A permutation that moves whole 16 B blocks keeps 16 B runs intact and
lowers with an identical ISA histogram; the same permutation at element granularity does not lower.
Copy-back check per `(destination layout, producer layout, dtype, width, source access pattern)`.

gfx942 width cost (4 B/lane max) and its `order` gate (destination order = source fast dim):
`pipeline/marker-and-version-gates.md ## The gfx942 async-copy width gate — available, and narrow`,
`../hardware/capability-matrix.md ## Direct-to-LDS granularity (per arch)`.

### Shared-layout family + transpose-on-read (layout dependency)

Which shared layouts an async direct-to-LDS copy (`buffer_load_to_shared` / `global_load_to_shared`)
can target. Same rule on gfx950 and gfx942 (Triton 3.8.0; compile-verified gfx950 + gfx942; every
padded destination that lowers copies back exact on gfx950 (GPU), at both entry points):

- **`SwizzledSharedLayout` lowers** at any legal width. It is the usual destination.
- **A padded destination lowers iff two conditions hold** (recipe: `## Deriving the producer layout
  for a padded async destination`):
  1. **interval ≥ one wave's copy:** `interval_bytes >= 64 lanes x per-lane width` — 1 KiB at
     128 bit, 256 B at 32 bit. fp16 64x64 `with_identity_for([[512, 8]], ...)` lowers,
     `[[256, 8]]` fails; at 32 bit/lane a 256 B interval lowers and 128 B fails.
  2. **the producer's reg/lane/warp bases are the destination's `offset_bases`** — derive it.
     For `with_identity_for` an exact-cover row-major `BlockedLayout` already *is* that layout and
     lowers; for `compute_efficient_padded_shared_layout` (permuted row bases) a row-major
     `BlockedLayout` fails and the derived producer lowers (either operand index).
  Miss either and the error is the generic one: `failed to translate module to LLVM IR`
  (`LLVM Translation failed for operation: builtin.unrealized_conversion_cast`) from
  `buffer_load_to_shared`, `PassManager::run failed` (`failed to legalize operation
  'ttg.async_copy_global_to_local'`) from `global_load_to_shared`. A per-lane run that is not
  contiguous in global memory (operand B from the factory with the default `is_k_contig=True` over a
  row-major source) fails the same way — pass `is_k_contig` to match the source.
- **Lowering is not correctness — a swizzled destination can be silently wrong.** With a plain
  `BlockedLayout` producer, `max_phase > 1` **and** `vec < size_per_thread` compiled, ran, and
  returned wrong data at both entry points (GPU copy-back, 3.8.0 gfx950; the sync control was
  exact). `max_phase = 1` and `vec >= size_per_thread` stay outside that case; a larger `max_phase`
  is a correctness change, so copy the LDS tile back and compare once per
  `(destination, producer, dtype, width)`. A padded async arm forced through by splicing a
  coalescing pass was the fastest arm measured **while returning NaN** — check numerics before
  believing a clock.
- **Same message, different cause.** The per-lane width gate (`### Minimum per-thread granularity
  (applicability by dtype)`) and "warps must cover contiguous rows of the destination exactly" fail
  with the same strings. Check the width first, then the warp-to-row mapping, then the family.
- Whether `compute_efficient_padded_shared_layout` returns a layout at all moves with
  `instr_shape` and `operand_index` (small non-K tiles return `None`); its refusal conditions are
  owned by `layout-reference.md ## Let 3.8 Compute The Padded Layout For You (CDNA4)`.
- **Transpose-on-read**: to consume one LDS tile in two operand orientations, store it once and read
  the transposed operand via `smem.permute((1, 0)).load(dot_layout)` (conflict behaviour:
  `../tile-programming/layout-recipes.md ## Padding vs swizzle (LDS bank conflicts)`). **A saving
  only when the transpose is not already free — check the ISA for `ds_read_*_tr_*` first.** On a
  gfx950 MLA forward (1 wave/SIMD), moving V's transpose from registers into the LDS read removed
  instructions, left `mfma` / `ds_read` / `shared` identical, was bit-equal — and measured slower (small,
  beyond noise in an interleaved A/B). The register path already used
  `ds_read_b64_tr_b16`, and the replaced `v_mov`s had co-issued under MFMA for free
  (`../pitfalls/negative-patterns.md ## Instruction count is not the objective function`).

## Deriving the producer layout for a padded async destination

A padded shared layout describes the **consumer** side — how the matrix instruction reads the tile
back out of LDS. When that tile arrives by async direct-to-LDS copy you also owe the compiler the
**producer** side: the distributed layout of the offset tensor `buffer_load_to_shared` writes
through. The two are not independently choosable. A padded destination under a producer layout that
describes a dense tile is a contradiction, and the lowering has to reject it somehow.

So derive the producer from the padded layout instead of authoring it alongside.
`PaddedSharedLayout` exposes `.offset_bases` (and `.interval_padding_pairs`, `.shape`) as readable
attributes. `.offset_bases` is a flat list of basis vectors ordered **register, then lane, then
warp**, and splitting it at the right two points gives the three basis lists
`gl.DistributedLinearLayout` wants:

```python
@triton.constexpr_function
def async_producer_layout(shared_layout, elems_per_lane, num_warps):
    """Register-side offset layout for the async copy that fills `shared_layout`."""
    bases = shared_layout.offset_bases            # register -> lane -> warp order
    r = elems_per_lane.bit_length() - 1           # low bits one lane owns contiguously
    lane_end = r + 6                              # 6 == log2(64), wave64
    warp_end = lane_end + (num_warps.bit_length() - 1)
    return gl.DistributedLinearLayout(
        reg_bases=bases[:r] + bases[warp_end:],
        lane_bases=bases[r:lane_end],
        warp_bases=bases[lane_end:warp_end],
        block_bases=[],
        shape=shared_layout.shape,
    )

# shared_a / shared_b came from compute_efficient_padded_shared_layout
load_a: gl.constexpr = async_producer_layout(shared_a, 16, NUM_WARPS)   # 16 fp8 = 16 B/lane
load_b: gl.constexpr = async_producer_layout(shared_b, 16, NUM_WARPS)
```

The three split points are not free parameters:

- `6` is `log2(64)` — wave64 on gfx950/gfx942.
- the warp count is the launch's `num_warps`, so this layout is `num_warps`-coupled like every
  other layout carrying an explicit warp dimension.
- `r` is set by how many contiguous elements one lane copies, i.e. the per-thread byte width of
  the copy divided by the element size. That is the same quantity that has to land in the
  direct-to-LDS width set (`### Minimum per-thread granularity (applicability by dtype)`): 16 fp8
  elements and 8 bf16 elements are both 16 B/lane, giving `r` of 4 and 3.

Verified on Triton 3.8.0: compile-verified gfx950 + gfx942 (32 bit); copy-back exact on gfx950 (GPU;
uint8 / bf16 / fp8 / fp16; factory op0, and op1 with `is_k_contig=False`). The other two conditions — the padding
interval covers one wave's copy and the global offsets' contiguous dim is `offset_bases[0]`'s dim —
are in `### Shared-layout family + transpose-on-read (layout dependency)`; the factory's intervals
(512 elements and up) satisfy the first.

Get `r` wrong and the bases are partitioned between registers and lanes incorrectly — the copy
still has a layout, it just does not describe the tile the matrix instruction will read. A plain
`gl.BlockedLayout` of the same tile describes a dense tile, so every offset past the first padding
interval names the wrong byte.

## gfx950 <-> gfx942 delta

- Buffer ops: `gl.amd.cdna3.buffer_load/store` on gfx942.
- **Async direct-to-LDS IS available on gfx942 through the `cdna4` namespace** (`gl.amd.cdna3` has
  no `async_copy` submodule): `gl.amd.cdna4.async_copy.buffer_load_to_shared` + `commit_group` /
  `wait_group` + `load_shared_relaxed` compile and run correctly on gfx942 at 32 bit/lane, so
  "gfx942 has no async copy" is wrong. Width / `order` gate and the measured loss vs sync staging on one kernel:
  `pipeline/marker-and-version-gates.md ## The gfx942 async-copy width gate — available, and narrow`.
- **`ttg.async_copy_global_to_local` (what `global_load_to_shared` emits) fails to legalize on gfx942
  above 32 bit/lane; at 32 bit it lowers to `global_load_lds_dword`** (3.8.0 compile-checked). Both
  `cdna4.async_copy` entry points obey the same width gate there.
- The exact-cover predicate (`## Async Copy To Shared`) bites first at gfx942's narrow width.
- gfx942 transpose-B / dot-operand reads from a `SwizzledSharedLayout` under `AMDMFMALayout(v3)` lowered
  in every shape tested (compile-only, Triton 3.6-3.8); the `builtin.unrealized_conversion_cast` failure
  reproduced on gfx942 only for an **async copy whose producer does not exactly cover the destination**.
  If it fires, check the exact-cover predicate first (derive the producer from the destination's
  `offset_bases`); `PaddedSharedLayout` + a derived producer is one fix, a matching exact-cover producer
  another. Copy back and compare once.
- `compute_efficient_padded_shared_layout` asserts v4 (unavailable on gfx942).
- LDS 64 KiB / 32 banks vs 160 KiB / 64 banks
  (`smem-lds-reference.md ## LDS is a second, independent occupancy limiter`).
- **Synchronous register staging (`buffer_load` -> shared store -> LR -> DOT) is the usual gfx942 form**;
  the 32-bit async path is an A/B option. On gfx950 the async ring is the form AMD's tutorial GEMMs are
  built on; register prefetch remains an A/B alternative on loops with few MFMAs per iteration.

Evidence: `../hardware/capability-matrix.md`, `../hardware/cdna3-gfx942.md`. Symptom routing:
`../method/triage.md`.
