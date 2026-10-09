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
- offsets are **element** offsets, not byte offsets. `BufferOpsEmitter.cpp`
  converts them itself (`vOffsetBytes = elementByteWidth * vOffsetElems`), on
  every version checked and for both the load and the store emitter.
- so do **not** pre-multiply by bytes per element. Doing it anyway scales the
  offset twice: 2x past the intended address on a 2-byte dtype, 4x on a 4-byte
  one. It compiles, it runs, and it reads the wrong values — which is the shape
  the caveat above this section describes.
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

**Three preconditions, stated by the op itself.** On `v3.8.0` both
`global_load_to_shared` and `buffer_load_to_shared` carry the same three
conditions in their docstrings, prefixed by *"the following conditions must be met
or lowering to LLVM will fail"*:

- *"For the `offsets` layout, size per thread * bits per element must be 128 or
  32"* (`ptr` layout for `global_load_to_shared`) — this is the width set below,
  stated by the API rather than inferred from the backend;
- *"Writes to `dest` must be coalesced"*;
- *"If `dest` is swizzled, it only can be swizzled within warp boundary."*

The reason all three exist is in the sentence above them: the hardware instruction
uses **a per-thread register for the global address but one register for the LDS
address for the whole warp**. That is the single fact to carry — every one of the
three conditions is a consequence of the destination address being warp-uniform,
which is also why the destination's layout family is a gate in its own right
(`### Shared-layout family + transpose-on-read (layout dependency)`).

**"Coalesced" has a testable form, and it is checkable before you write the call.** The second
condition is the one that reads like a quality guideline and is actually a predicate. For a warp
writing a tile, it holds when the producer layout **exactly covers** the destination's fastest
dimension:

```
threads_per_warp[contig] * size_per_thread[contig] == tile extent along contig
```

Not `>=`. Over-covering wraps a warp's lanes onto a second row, which makes the per-warp LDS
address non-uniform and violates the same warp-uniform-destination fact above; under-covering leaves
the row to a second warp and does the same. Evaluate that product against the tile shape when you
author the `BlockedLayout`, because the failure arrives as a translation error that names the
enclosing function rather than the layout
(`../pitfalls/negative-patterns.md ## A translation failure names the allocation, not the murderer`),
and because after the fact it is indistinguishable from the width gate below — both are "it did not
lower", and the two fixes move different fields.

`buffer_load_to_shared` (CDNA4 async copy) requires 32-bit offsets and an
offset tensor whose distributed layout matches rank, dtype, unit, and the
shared-memory consumer. A `BlockedLayout` lowering failure is **layout-contract
evidence** before it is a build ceiling — retry with a source-proven
`DistributedLinearLayout` matched to the consumer before recording a toolchain
ceiling. When the consumer is a padded layout, "matched" has a concrete
construction rather than a search: `## Deriving the producer layout for a padded
async destination`. Do not interleave ordinary loads/stores with async paths without a
traffic/scheduling hypothesis. This is the main mechanism behind the pipeline
layer (`../tile-programming/pipeline.md`).

**The copy costs per byte per lane, not per call, so merging two of them is a dependency edit
rather than a memory optimization.** "One big load instead of two" is a natural instinct and its
default sign is negative. Measured, bit-exact, ISA-diffed: two copies of a `[32, 256]` tile versus
one copy of `[64, 256]` over the same bytes lower to **identical** machine code on the memory side
— same `buffer_load_dword` count, same `buffer_load` count, same `ds_read`/`ds_write`, same LDS —
and the merged form still measured **2.6 % slower**, negative in 6 of 7 shapes, even though its
total instruction count *fell* (1279 → 1252) and its `s_waitcnt` count fell harder (110 → 76). What
moved against it was VGPR (174 → 190) and the dependency structure: one `wait_group` now gates
twice as much data before *either* operand is usable, so the first consumer waits on bytes it does
not need. Price a merge by differencing the two arms, not by counting the calls it removes.

### Minimum per-thread granularity (applicability by dtype)

`buffer_load_to_shared` requires the per-thread load width to be one of a **small
set of legal bit widths — it is not a floor, and treating it as one produces an
illegal width.** Upstream gates it per ISA family, with the intermediate widths
disabled because the hardware would widen them and overwrite neighbouring data:

| family | legal per-thread widths | explicitly rejected |
| --- | --- | --- |
| CDNA4 (gfx950) | **128 bit (16 B)** and **32 bit (4 B)** | 8, 16, 96 |
| CDNA3 (gfx942, downgrade) | **32 bit (4 B)** only | 8, 16 |

So **4 B per thread is legal on both**, 16 B is legal only on CDNA4, and **8 B
(64 bit) is illegal on both** — the width a "raise it until it clears ~16 B"
reading lands on first. A load at an unsupported width is **rejected at
lowering** (an `unrealized_conversion_cast` / lowering failure), which is **not**
a build ceiling and **not** fixable by switching the `BlockedLayout` family — it
is the per-thread *width*. The fix is to move `size_per_thread` on the contiguous
dim to a value whose byte count lands **on** a legal width, not merely above one:

| Operand dtype | bytes/elem | contiguous elems/thread for the 128-bit width |
| --- | --- | --- |
| bf16 / fp16 | 2 | >= 8 |
| fp8 / int8 | 1 | >= 16 |
| fp4 (e2m1, packed 2/byte) | 0.5 | >= 32 |

**The test is per-thread *bit width against the set the architecture encodes*, not
"is this tensor a block scale".** `supportsDirectToLdsLoadBitWidth`
(`v3.8.0 TargetFeatures.cpp:197-204`) admits CDNA3 = {32}, CDNA4 = {128, 32},
GFX1250 = {128, 64, 32}, and nothing else — the narrower widths are excluded
because they get extended and would overwrite. So on CDNA4 a few elems/thread is
**not** disqualifying: four contiguous fp32 (16 B = 128 bit) is in the set and so
is a single fp32 (4 B = 32 bit), while **two** fp32 (8 B) is not, and that is the
case that is rejected. Production reaches direct-to-LDS for fp32 scales exactly
this way, behind an explicit constexpr gate
(`async_scales = M > 1024 and XSK == 1 and XSM % 4 == 0`) with
`COPY_WIDTH = 1 if SPLITS > 1 else 4` — fp32 × {1, 4} = {4 B, 16 B}, never the
2 that would land on 8 B. The register-staging side path GR -> LW -> LR
(`../tile-programming/low-precision.md`) is the fallback for **a width that misses
the set**, not a route block scales belong on by kind. Either way this is a
layout/granularity constraint, not an async-path build ceiling — do not record it
as a `buffer_load_to_shared` capability failure.

**A legal width is necessary and not sufficient: the thread's elements have to be contiguous *in the
source* at that width.** Measured on one scale plane, everything held fixed except the offset
tensor: `size_per_thread = 4` on `uint8` — a 32-bit width, in the set, the same width that copies
bit-exactly one row above — **fails LLVM translation** once the offsets carry a vendor-permuted
element order, because under that permutation a thread's four logical bytes sit 64 B apart and there
is no width at which they are one load. The plain `gl.load` control succeeds on the same offsets and
shows what the hardware does when it is allowed to: four `global_load_ubyte` instead of one
`global_load_dword`. Two consequences worth carrying:

- **The error text is the same as an illegal width**, so a failure here is routinely mis-recorded as
  the width gate and "fixed" by changing `size_per_thread` — which cannot work, since no legal width
  exists for a scattered run. Check the offsets as well as the width before concluding the path is
  unavailable.
- **Separable is not contiguous.** A permutation can be perfectly separable in its indices and still
  scatter a per-thread run. The property that decides whether you can read a permuted layout in
  place rather than inverting it on the host is whether the permutation moves blocks at least as
  large as the per-thread run: a permutation that relocates whole 16 B blocks leaves a 16 B run
  intact and lowers with an *identical* ISA histogram, while the same permutation applied at
  single-element granularity on a narrower plane does not lower at all. Two structurally identical
  edits, opposite outcomes, and the block size of the permutation is what tells them apart.

So the copy-back correctness check is per `(destination layout, producer layout, dtype, per-thread
width, source access pattern)` — the last axis is easy to leave out and it is the one that moves
when you start reading someone else's layout directly.

> **What the width set costs you per arch.** The set above **is** the rule, so read
> the element table as the CDNA4 128-bit case rather than as a universal floor. On the
> gfx942 downgrade the widest direct-to-LDS load is 32-bit (4 B, 2×bf16) per thread, a
> quarter of CDNA4's; what that costs, the extra `order=[1, 0]` gate and the measured
> loss against sync staging are owned by
> `pipeline/marker-and-version-gates.md ## The gfx942 async-copy width gate — available, and narrow`
> (and `../hardware/capability-matrix.md ## Direct-to-LDS granularity (per arch)`).
>
> **Where this predicate lives.** The width set is owned by
> `../tile-programming/low-precision.md`, which is where it was first stated
> correctly; this page restates it for the API surface. A correction to the rule
> belongs there and then here — fixing one page of a layered pack does not fix the
> other copies of the same rule, and this one already cost two revisions.

### Shared-layout family + transpose-on-read (layout dependency)

Not every shared layout lowers for the async direct-to-LDS path:

- **`SwizzledSharedLayout` lowers for `buffer_load_to_shared`; a padded destination
  has not, on either constructor.** This is the measured correction to an earlier
  revision of this page, which said the delta belonged to
  `PaddedSharedLayout.with_identity_for` and not to
  `gl.amd.cdna4.compute_efficient_padded_shared_layout`'s output. It belongs to
  both. Everything else held fixed and every success verified by copying the LDS
  tile back and comparing with `torch.equal`, on gfx950 / `v3.8.0` with a `uint8`
  operand:

  | destination layout | copy entry | BM=16 | BM=32 | BM=128 |
  | --- | --- | --- | --- | --- |
  | `compute_efficient_padded_shared_layout` | `buffer_load_to_shared` | layout is `None` | **fails LLVM translation** | **fails LLVM translation** |
  | `SwizzledSharedLayout(1, 1, 1)` | `buffer_load_to_shared` | **OK** | **OK** | **OK** |
  | `SwizzledSharedLayout(1, 1, 1)` | `global_load_to_shared` | **OK** | **OK** | **OK** |
  | `compute_efficient_padded_shared_layout` | `global_load_to_shared` | layout is `None` | `PassManager::run failed` | `PassManager::run failed` |
  | `compute_efficient_padded_shared_layout` | `gl.load` -> `smem.store` (control) | layout is `None` | OK | OK |

  > **Read every `OK` in this table as `max_phase = 1` only.** All of the swizzled cells were run
  > at `SwizzledSharedLayout(1, 1, 1)`, and that is the one setting in the family which **cannot
  > exhibit the failure described immediately below**. The table establishes that the swizzled
  > *family* lowers where the padded one does not. It does **not** establish that an arbitrary
  > swizzled destination is safe.

  Three things to read off it. **The gate is the destination layout, not
  `BLOCK_M`** — the swizzled row clears every tile size the padded row fails at.
  **`BM=16` is an independent problem wearing the same coat**: there the padded
  layout does not exist at all (`layout-reference.md ### Contract points`, cause 3),
  so the async copy never gets the chance to fail, and reading that cell as an async
  limit is a misattribution. And the **control row is what proves the padding
  itself is sound** — the same layout is a perfectly good synchronous destination at
  BM=32 and BM=128, so a failure on the async row is about the copy's warp-uniform
  destination address, not about the padding pattern. Record it as a known
  layout x entry-point delta, not a build ceiling.

  **Every failure in that table is a loud one, and the most dangerous failure on this path is
  not.** The table's whole shape — `None`, translation failure, pass-manager failure — trains the
  reading "if it compiled, the layout is legal." **That is false.** Measured on gfx950 / `v3.8.0`
  with a plain `gl.BlockedLayout` producer and a **swizzled** `uint8` destination: when
  `max_phase > 1` **and** `vec < size_per_thread`, the copy **returns wrong data** — no exception,
  no warning, no diagnostic, a clean compile and a kernel that runs. Eight cells were swept; the
  synchronous control was correct in all of them, and the behaviour reproduced at **both** entry
  points. The `SwizzledSharedLayout(1, 1, 1)` used throughout this page has `max_phase = 1`, which
  is precisely the setting where the condition cannot be met, so none of the rows above could have
  caught it.

  Three consequences, and the first one is the reason this block exists:

  1. **A compile is not a layout check on the async path.** Any async-copy result has to be
     verified numerically — copy the LDS tile back and compare — at least once per
     `(destination layout, producer layout, dtype, per-thread width)` combination. This is the same
     copy-back that produced the `OK`s above; it is cheap, and it is the only thing standing
     between you and a silently wrong kernel.
  2. **`vec` and `size_per_thread` are a pair, not two independent knobs.** A producer whose
     per-thread run is shorter than the swizzle's vector unit lets the swizzle permute inside data
     the copy treats as contiguous. Raising `size_per_thread` for the width gate
     (`### Minimum per-thread granularity (applicability by dtype)`) is exactly the edit that can
     walk a working kernel into this condition, so re-verify numerically after any width change.
  3. **Keep `max_phase = 1` unless something measured asks for more.** It costs nothing here, and
     it is outside the failing region by construction. A swizzle with `max_phase > 1` is a bank
     conflict remedy — adopt it against a measured conflict, with a numerical check, not as a
     default.

  The per-thread width interacts with this and is a separate gate: for `uint8`,
  `size_per_thread` must be 16 or 4 (the 128-bit and 32-bit widths of the table
  below). A `size_per_thread` of 8 fails to translate with **the same opaque
  message** as the padded rows above, which is why the two get confused — check the
  width first, because it is the cheaper of the two to check.

  So: if you are staging a matrix operand by async copy, **get the copy working
  against a `SwizzledSharedLayout` destination first**, and treat moving it onto a
  padded destination as a separate step with its own evidence. For a buffer that is
  **not** read back as a matrix operand, `SwizzledSharedLayout` (or a source-proven
  `DistributedLinearLayout` matched to the consumer) remains the default anyway,
  because swizzle costs no LDS capacity.

  **And when a padded destination is made to lower, it can be a silent miscompile.** A padded
  shared layout on an async arm whose lowering was forced through (a coalescing bounce spliced
  in) removed the residual bounce and was the **fastest arm measured while returning NaN across
  nearly every element**, with the identical sync kernel exact. Check numerics on any padded-shared
  async arm before believing its clock — the same copy-back rule as above.

  *Scope:* one dtype, one operand shape family, one `instr_shape`, one
  `operand_index`, and every swizzled cell at `max_phase = 1`. What is established
  is that "padded destinations are ordinary async destinations" is false as a
  general statement, not that no padded destination can ever lower.

  **`operand_index` in that scope line is doing real work — do not read the rows as a
  function of the tile's non-K extent alone.** Whether the padded-layout factory returns a layout
  at all moves with `instr_shape` *and* with the operand index, and both flip cells. At
  `[16, 16, 128]` the small-tile row is `None` for both operands; at `[32, 32, 64]` operand 0 is
  still `None` one tile size up, while operand 1 **does** get a layout at the smallest tile —
  i.e. exactly where the other rows say non-existence is the whole story. `warps_per_cta` did not
  matter across the two values tried. So re-derive the threshold on your own
  `(instr_shape, operand_index)` rather than carrying a tile number across; the factory's refusal
  conditions are owned by
  `layout-reference.md ## Let 3.8 Compute The Padded Layout For You (CDNA4)`.

  *And one confound this table cannot rule out.* `PassManager::run failed` has at
  least **three** known causes on this path, and the two the table is written
  around — a padded destination, and the arch (`global_load_to_shared` fails the
  pass manager outright on gfx942) — are not the only ones. The third is that the
  copy's **warps must cover contiguous rows of the destination exactly**; it is the
  lowering's reading of the docstring's *"writes to `dest` must be coalesced"*, it
  is currently written down only as a gfx942 signature, and it has been hit on
  **gfx950**. The cells above do not record their per-cell `threadsPerWarp` or
  `size_per_thread`, so **whether the padded rows carry that confound cannot be
  determined from this table** — stated as undetermined, not as a defect in the
  rows. If you are reproducing them, log those two fields per cell; if you hit
  `PassManager::run failed` yourself, check the warp-to-row mapping before
  concluding anything about the destination layout family.
- **Transpose-on-read**: to consume one LDS tile in two operand orientations
  (instead of storing it twice), store it once in natural order and read the
  transposed operand via `smem.permute((1, 0)).load(dot_layout)`. The conflict
  behaviour of reading a tile both ways and its mitigation live in
  `../tile-programming/layout-recipes.md ## Padding vs swizzle (LDS bank conflicts)`.

  **It is a saving only when the transpose is not already free. Check the ISA for
  an existing `ds_read_*_tr_*` before reaching for it.** The competing form —
  load in a linear layout and transpose in registers (`gl.permute` +
  `convert_layout`) — looks strictly worse in a census and can be strictly
  faster: on a gfx950 MLA forward pinned at 1 wave/SIMD, moving V's transpose
  from registers into the LDS read removed 51 instructions, left `mfma` and
  `ds_read` counts and `shared` bytes/WG **identical**, was bit-for-bit
  numerically equal — and lost **1.4 %** (1.0433x vs 1.0585x, interleaved A/B in
  one clean window; record the window's spread alongside, per `ab_bench.py`).
  The register path was already landing 384 `ds_read_b64_tr_b16`, i.e. the
  hardware was already doing the cheap part, so the transpose did not disappear:
  it moved into that read's bank pattern, while the `v_mov`s it replaced had been
  co-issuing under MFMA for free. On the same kernel a padding change to the same
  buffer cost 10–11 pp in both directions, so the LDS access pattern was the
  sensitive axis and the registers were not.

  The general rule this is an instance of is in
  [`../pitfalls/negative-patterns.md ## Instruction count is not the objective function`](../pitfalls/negative-patterns.md).

## Deriving the producer layout for a padded async destination

A padded shared layout describes the **consumer** side — how the matrix instruction reads the tile
back out of LDS. When that tile arrives by async direct-to-LDS copy you also owe the compiler the
**producer** side: the distributed layout of the offset tensor `buffer_load_to_shared` writes
through. The two are not independently choosable. A padded destination under a producer layout that
describes a dense tile is a contradiction, and the lowering has to reject it somehow.

So derive the producer from the padded layout instead of authoring it alongside.
`PaddedSharedLayout` exposes `.offset_bases` (and `.interval_padding_pairs`, `.shape`) as readable
attributes — the object the layout factory returns is part of the usable API surface, not an opaque
handle. `.offset_bases` is a flat list of basis vectors ordered **register, then lane, then warp**,
and splitting it at the right two points gives the three basis lists
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

Get `r` wrong and the bases are partitioned between registers and lanes incorrectly — the copy
still has a layout, it just does not describe the tile the matrix instruction will read.

**The failure this prevents** is substituting a plain `gl.BlockedLayout` of the same tile shape.
That layout describes a dense tile, so every offset past the first padding interval names the wrong
byte, and the inserted padding is exactly what makes the two disagree. The `## Async Copy To
Shared` rule — a `BlockedLayout` lowering failure is layout-contract evidence before it is a
toolchain ceiling — is this case.

### What is and is not established here

An earlier revision of this section offered two competing explanations for the observed async
lowering failure — the `with_identity_for` constructor specifically, or any padded destination
whose producer layout was not derived from its `.offset_bases` — and asked for a 3x2 compile
comparison to settle it. **One column of that grid has since been run**
(`### Shared-layout family + transpose-on-read (layout dependency)`), and it rules the second
explanation out as a *complete* one:

| | producer derived from `.offset_bases` | plain `BlockedLayout` producer |
| --- | --- | --- |
| layout from `compute_efficient_padded_shared_layout` | **fails LLVM translation** | not run |
| layout from `PaddedSharedLayout.with_identity_for` | not run | fails to lower (earlier observation) |
| `SwizzledSharedLayout` | **lowers, verified by copy-back** | **lowers — and is silently wrong when `max_phase > 1` and `vec < size_per_thread`** |

So a correctly-derived producer does **not** rescue a padded destination, and the delta is not
confined to `with_identity_for`. The bottom-right cell has since been run too, and it is the
reason this grid is not a pass/fail grid: a plain `BlockedLayout` producer against a swizzled
destination **compiles and produces incorrect results** under the condition named in it — so
"lowers" and "correct" are different columns, and only the copy-back distinguishes them
(`### Shared-layout family + transpose-on-read (layout dependency)`). What is still open is the
`with_identity_for` row's left cell — i.e. whether the producer derivation matters *at all* on
this path, or only the destination family does. Deriving the producer layout from `.offset_bases`
is still the right thing to do when the destination is padded (it is the only construction that
describes the tile), but it is **no longer a candidate fix for an async lowering failure** — swap
the destination to `SwizzledSharedLayout` for that, and measure the LDS-capacity and
bank-conflict cost of the swap rather than assuming it.

## gfx950 <-> gfx942 delta

Everything above is written for gfx950. This is the gfx942 downgrade for the memory path:

- buffer ops: `gl.amd.cdna4.buffer_load/store` (gfx950) vs
  `gl.amd.cdna3.buffer_load/store` (gfx942);
- **async direct-to-LDS IS available on gfx942 — a namespace gap is not a silicon
  gap — but it is not the gfx942 default.** The `gl.amd.cdna3` namespace has no
  `async_copy` submodule, while `gl.amd.cdna4.async_copy.buffer_load_to_shared` +
  `commit_group` / `wait_group` + `load_shared_relaxed` **compile and run on gfx942**
  (verified correct) at 32 bits per thread. Do **not** record "gfx942 has no async copy"
  from the empty `cdna3.async_copy`. The width gate, the `order=[1, 0]` destination rule and
  the measured loss against sync staging are owned by
  `pipeline/marker-and-version-gates.md ## The gfx942 async-copy width gate — available, and narrow`;
  backend truth is `supportsDirectToLdsLoadBitWidth` (`### Minimum per-thread granularity
  (applicability by dtype)` above;
  `../hardware/capability-matrix.md ## Direct-to-LDS granularity (per arch)`).
- **Real gfx942 async limits (verify each — these are layout/width facts, not the
  namespace myth):**
  - the fast dim must be **exactly covered** by `threads_per_warp * size_per_thread`
    (no replication) — the coalescing predicate in `## Async Copy To Shared`, which bites
    first at gfx942's narrow width.
  - transpose-B / complex dot-operand offset layouts under `AMDMFMALayout(v3)` +
    `SwizzledSharedLayout` can **fail LLVM translation**
    (`builtin.unrealized_conversion_cast`); the recorded fix is `PaddedSharedLayout` + a
    source-proven `DistributedLinearLayout` matched to the consumer (a block-scale
    GEMM that stages a transposed operand is one upstream example of this pattern).
    Read that as a **restricted option for this gfx942 case**, not as the general
    async recipe: the measured gfx950 result above is that padded async destinations
    fail translation (or, forced through, return wrong data), so on either arch build
    the async destination swizzled first, bisect the failure by deleting ops
    (`../pitfalls/negative-patterns.md ## A translation failure names the allocation, not the murderer`),
    and move to padding only with a copy-back check.
  - the **generic** `ttg.async_copy_global_to_local` is explicitly illegal on the
    gfx942 backend — use the `cdna4.async_copy` entry, not the generic op.
  - `compute_efficient_padded_shared_layout` asserts **v4-only** (unavailable on
    gfx942 / v3).
  - LDS capacity and banking also drop (64 KiB / 32 banks against 160 KiB / 64 banks),
    which moves both the occupancy divisor and any conflict-free swizzle
    (`smem-lds-reference.md ## LDS is a second, independent occupancy limiter`).
- **Synchronous register staging** (`buffer_load` -> shared `store` -> `LR` -> `DOT`,
  ordered by LDS read-after-write) **is the gfx942 default**; the 32-bit async path is a
  correctness-preserving option to A/B against it, not an upgrade. (On gfx950 the
  ordering is the reverse: the async ring is the main line and sync staging is its
  control.)

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
a = gl.amd.cdna4.buffer_load(a_ptr, offs, mask=mask, other=0.0)   # no wavefront branch
# gfx942 downgrade: identical call under gl.amd.cdna3
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
offsets are element counts, not pre-multiplied to bytes:
mask layout matches offsets:
fallback dtype matches pointer element type:
stored value dtype matches destination:
benchmark boundary:
```

If any item is unknown, keep the generic path or shrink the probe. Symptom -> fix
routing: `../method/triage.md`.
