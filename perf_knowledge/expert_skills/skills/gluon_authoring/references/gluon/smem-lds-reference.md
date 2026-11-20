# Gluon — shared memory / LDS layouts

## CDNA (gfx942/gfx950)

- LDS declared via TTGIR shared layout. **gfx950: 160 KiB per CU, 64 banks** (4-byte stride; a
  full-conflict `ds_read_b128` stride of 256 B and a CDNA4-only 2-way stride of 128 B). **gfx942
  downgrade: 64 KiB per CU, 32 banks** (full-conflict `ds_read_b128` stride 128 B) — so a swizzle that
  was conflict-free on one is not known to be on the other, and a depth that does not fit on gfx942
  may simply fit on gfx950. Values are the per-arch keys in
  `perf_knowledge/hardware/data/hw_constants.json` (`lds_per_cu_kib`, `lds_banks`,
  `ds_read_b128_*_stride_bytes`); take them from there with `--arch` rather than carrying a figure
  across generations. The **allocation granule is not established** on gfx950: `hw_constants.json` carries
  `lds_align_bytes: 1280`, which is an alignment and **not** the allocator's round-up quantum —
  `../hardware/capability-matrix.md` records `lds_min_alloc_bytes` as a declared unknown for this
  arch precisely because a measured allocation need not be a multiple of 1280.
- **Three shared-layout constructors, and on 3.8.0 / gfx950 you usually do not write the padded one
  by hand.** `SwizzledSharedLayout` costs no capacity and is the default for any buffer that is not
  read back as a matrix operand; `PaddedSharedLayout` is for the ones that are, and on 3.8.0
  `gl.amd.cdna4.compute_efficient_padded_shared_layout(dot_operand_layout, shape, dtype)` derives it
  from the operand layout for you (returning **`None`**, not raising, outside its covered set).
  **That factory decides the padding, it does not decide the family** — it prices the operand
  *read* and is blind to the LDS *write* that fills the buffer, and a padded fill has been measured
  running fully scalar (128 `ds_write_b8`-class writes per K step) where the swizzled fill emitted
  none, losing 2–7 % on that alone. A/B the two on matrix-operand buffers
  (`layout-reference.md ### The selection rule is a starting point, not a result — the write side
  can reverse it`).
  `SharedLinearLayout` is the explicit-bases escape hatch. Hand-computed padding is the **downgrade**
  path for 3.7.1 and earlier, not the starting point. Layouts recovered from TTGIR transcription
  arrive as whichever of these the plain kernel used.
  `layout-reference.md ## Shared Layouts: The Three Constructors` and
  `layout-reference.md ## Let 3.8 Compute The Padded Layout For You (CDNA4)` own the contracts.
- Pipeline stages = separate LDS buffers per stage, hand-sized. `num_stages` does not size them
  and never has on this path — it is accepted and inert on all four versions, dead in 3.8.0
  (`pipeline/loop-knobs-and-targets.md ## Roll the loop (cut i-cache pressure)`).

## LDS is a second, independent occupancy limiter

Registers are the limiter people predict; LDS is the one that surprises them. They are separate
resources with separate divisors, and **a kernel can be capped by both at once** — relieving one
then moves nothing (`../method/budget.md ## 1. Your hardware budget`). Read both limiters the moment anything compiles, not
after the first disappointing measurement (`../method/entry.md`).

The arithmetic is a floor division, which is why LDS savings are step functions and not slopes:

```text
workgroups_per_CU = lds_capacity // lds_per_wg_allocated    # 160 KiB gfx950, 64 KiB gfx942
```

So the only question a shared-memory change has to answer is **which divisor did it cross**. Worked
on gfx950: a workgroup allocating 40 KiB gets `163840 // 40960 = 4` workgroups/CU, and reaching 5
needs the allocated size at or under `163840 // 5 = 32768 B`. A 6 KiB trim from 40 KiB to 34 KiB
crosses nothing and buys nothing; the last 1.5 KiB is what buys the tier.

**Read `lds_per_wg_allocated`, do not derive it.** The allocator may round the request up, and the
quantum it rounds to is not established on gfx950 (see the arch bullet above), so a threshold
computed from the *request* is an upper bound on workgroups/CU and the real number can be one tier
worse. The measurable value is the Triton cache metadata's `shared` field, which is the same source
dial A3 uses: `kernel_workflow/scripts/kernel_tools/asm_loop_audit.py --meta <ir-dir>` (the pack's
`scripts/asm_loop_audit.py` is a shim to it). Two readings that are **not** evidence
here — the kernel descriptor's `group_segment_fixed_size` and rocprof-compute `7.1.8` — are
structurally 0 for every Triton kernel, so a 0 from either is not a small allocation.

If you need the quantum itself, it is a probe, not a constant: sweep the requested bytes in small
steps across a suspected boundary and read `shared` back at each step. Record the result rather than
assuming 1280 B; that assumption is what this page previously built a threshold on.

Two consequences for the layout choices on this page and in
`layout-reference.md ## Shared Layouts: The Three Constructors`:

- **Padding is an occupancy decision, and on the async path it is a lowering decision too.**
  `PaddedSharedLayout` costs capacity by construction, so compute the new workgroups/CU before
  adopting it. A swizzle that resolves the same conflict costs none, which is the main reason to
  prefer it when both work — and there is now a second reason: a padded destination under
  `buffer_load_to_shared` has been measured **failing LLVM translation** where a
  `SwizzledSharedLayout` destination lowers and verifies on the same kernel
  (`memory-reference.md ### Shared-layout family + transpose-on-read (layout dependency)`). So if
  the buffer is filled by async copy, swizzle is the first thing to try and padding is the thing
  you justify.
- **Carve, do not allocate twice.** `.slice(...)` / `.index(...)` / `.permute(...)` views over one
  allocation (below) are free; two allocations are two contributions to the divisor. This is the
  mechanism behind the LDS-dedup lever in `../tile-programming/slicing.md`.

If the SPI occupancy limiter comes back **empty**, neither resource is capping you and none of this
is your problem — go back to the bound classification rather than shrinking buffers
(`../method/profile.md`).

## Reshaping an allocation without reallocating it

`gl.allocate_shared_memory` returns a `shared_memory_descriptor`, and four of its methods produce
a **new descriptor over the same bytes** rather than new storage: `.index(i)` (subview along dim
0), `.slice(start, length, dim)`, `.permute(order)`, and `._reinterpret(...)`. Staging one buffer
and consuming it under two different shapes costs nothing at runtime; it is a type-level move.

`._reinterpret(dtype, shape, layout)` is the general one — a byte-identical reinterpretation, so
it is how you stage as one dtype and read as another, or attach a second layout to a buffer whose
writer needs a different one (the gfx942 `order=[1, 0]` async-destination conflict in
`pipeline/marker-and-version-gates.md ## The gfx942 async-copy width gate — available, and narrow`
is exactly this shape).

> **Its signature changed in 3.8.0, and the convenient spelling is the one that breaks.** Checked
> 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0: before 3.8.0 all three of `dtype`, `shape`, `layout` are
> **required**; from 3.8.0 each defaults to the descriptor's current value, so `d._reinterpret(dtype=gl.int8)`
> — change one facet, inherit the rest — is 3.8.0-only and raises a `TypeError` on 3.7.1.
> **Passing all three explicitly works on every version**, so write it that way unless you have
> pinned 3.8.0. Note also the leading underscore: this is not a stable public name.

## `_keep_alive()` — the name is the effect, the mechanism is its opposite

`shared_memory_descriptor._keep_alive()` reads like an annotation that pins a buffer. Its whole
body is `_semantic.shared_dealloc(self)`, which emits a `ttg.local_dealloc` — an op whose own
description is *"deallocates a buffer explicitly … using the buffer after this operation is
undefined"* (checked on `v3.8.0`; the docstring is one line, "Dummy use to keep the shared memory
descriptor alive"). **It marks the END of a live range, not the beginning.**

That is not a contradiction, and the reason is the sentence immediately after in the op's
description: *"If you don't explicitly dealloc a buffer, the compiler assumes it's deallocated at
the first point that post-dominates all uses of the alloc."* The default live range is therefore
as short as the uses allow, and the allocator is free to hand those bytes to something else from
that point on. Placing a `_keep_alive()` **later than the last real use** moves the end of the
range out to where you put it. You are not adding a use; you are moving a boundary.

Two consequences, and only the first is free:

- **It can remove a hazard rather than fence it.** If the reason a barrier appears is that the
  allocator gave the same bytes to a later buffer, extending the first range past the conflict
  makes the hazard not exist, and the analysis has nothing to guard. Production uses it this way
  across several kernel families — enough sites to call it a family of practice, though within any
  single kernel family the sample is small enough that it may be one author's discipline rather
  than a rule. Treat it as a **third escape hatch** alongside the two in
  `../tile-programming/warp-pipeline.md ## Membar is a standing risk, not a one-time bug`, and gate
  it the same way: the determinism race-test, because you are now making the guarantee yourself.
- **It is not a free annotation.** A longer live range is a larger overlap in the allocator's
  interference graph, so it can raise peak LDS and therefore cost a workgroup per CU
  (`## LDS is a second, independent occupancy limiter`). The exact cost is **not established
  here** — read it off the Triton cache metadata's `shared` before and after, the same source A3
  uses, rather than predicting it.

## Indexed access to LDS — .gather / .scatter

Available from **3.7.0** (absent on 3.6.0; checked 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0). These read and
write LDS at positions chosen by a runtime index tensor, which is what a lookup table, a
per-token routing map, or a paged index chain needs:

- `smem.gather(indices, axis)` → tensor. For each output position `I`, the coordinate at `axis`
is replaced by `indices[I]`: `result[I] = src[I[0], ..., indices[I], ..., I[n]]`. The result has
the **shape of `indices`**, not of the descriptor.
- `smem.scatter(values, indices, axis)` → writes `dst[..., indices[I], ...] = values[I]`.
`values` and `indices` are broadcast to a common shape first, so `values` of `[N, 1]` against
`indices` of `[1, M]` at `axis=1` behaves as if both were `[N, M]`.

> **`smem.gather` and `gl.gather` are different mechanisms with the same verb.** This one is an
> LDS access by construction and costs `ds_read`; `gl.gather` indexes a tensor already held in
> registers, and whether it touches memory at all is decided by its layouts
> (`layout-reference.md ## gl.gather — a lane exchange only when the layouts allow it`). Reaching
> for the wrong one is not a performance mistake so much as a structural one — you will have
> staged data through LDS unconditionally where a layout fix could have kept it in registers.

**Neither op documents any cross-wave ordering or collision behaviour**, so do not assume one:
treat `.scatter` as an ordinary LDS write for synchronisation purposes — barrier between the
writes and any read, exactly as with `.store()` — and treat colliding indices within a scatter as
unspecified rather than last-writer-wins. If your algorithm needs a defined winner, resolve it
before the scatter.

## Atomic RMW in LDS — the atomic_scatter_* family

Available from **3.8.0** only (absent on 3.6.0 / 3.7.0 / 3.7.1; the version-gate row lives in
`../workloads/moe.md ## Version gates`). These are the read-modify-write siblings of `.scatter`:
each lane applies an operation to the LDS word its index names, and the call returns **the value
that was there before the operation**, per lane.

> **Read the name before concluding the mechanism is absent.** There is **no pointer-form
> `smem.atomic_add`** on the shared-memory descriptor — the only spelling is the *scatter* form,
> `atomic_scatter_<op>`. Searching a body, a diff, or a pack for `atomic_add` on a descriptor
> therefore returns nothing whether or not the mechanism is in use, and that clean-looking nothing
> is the single most common way this surface gets recorded as unused. It is not unused: uptake is
> small and concentrated in one kind of kernel, which is a different statement. **A negative about
> an API is only worth what the enumeration behind it was** — enumerate the descriptor's method
> surface, then conclude.

The family is `atomic_scatter_add`, `_max`, `_min`, `_and`, `_or`, `_xor`, `_xchg`, all with the
same shape:

```python
prev = smem.atomic_scatter_add(values, indices, axis, mask=active)
```

- `values` and `indices` broadcast to a common shape, exactly as for `.scatter`.
- `axis` is the descriptor axis the index replaces; a 1-D counter table is `axis=0`.
  **The parameter is named `axis`, not `dim`** — passing `dim=` is a `TypeError`, and
  the surrounding vocabulary in this pack says "dim" often enough that it is an easy slip.
- `mask=` selects which lanes participate; masked-off lanes perform no RMW.
- the result carries the **pre-operation** value per lane, which is what makes the `add` form a
  slot reservation and not just an accumulate.

**dtypes are not uniform across the family.** `add` and `xchg` accept integer *and* floating
types; `max`, `min`, `and`, `or` and `xor` are **integer-only** and raise at compile time
otherwise. So a floating-point maximum has to go through an order-preserving integer key. `max` and
`min` are signedness-aware — an unsigned tile dispatches to the unsigned comparison, which is
exactly what an integer-key scheme wants.

### When an LDS atomic is the right instrument, and when a barrier plus a reduction is

The discriminator is **whether each lane's destination is data-dependent**, not how expensive
anything is.

- **A reduction needs an axis.** `gl.sum` / `gl.max` / `gl.associative_scan` combine along a
  *known* axis of a tile, so `barrier` + reduction + one write is the right shape whenever the
  destination of a lane's contribution is decided by its position. It is deterministic, it needs no
  atomic, and it is available on every version. **This is the default** — reach for it first.
- **An atomic is for a scatter with collisions.** When the destination is a value the lane
  *computed* — a bucket id, an expert id, a bin — several lanes can land on the same word and
  there is no axis to reduce along. A histogram in LDS, a CTA-local counting sort, or a
  reservation (`prev` is the caller's unique slot in its bucket) are the shapes that have no
  reduction form, and they are where this family earns its place.
- **A conflict-tolerant publish** is the `_xchg` case: several lanes may write the same word, the
  return value is discarded, and the atomic is there only so the colliding writes are *defined*
  rather than to combine anything.

> **The ordering is nondeterministic, and that is a correctness question, not a tuning one.**
> Which colliding lane receives which `prev` is not defined and can differ run to run, so slot
> assignment is not reproducible. That is harmless when a later **ordered** step folds the results
> by a stable key, and it is a reproducibility bug the moment anything downstream combines
> partials in slot order. Adopting this family means owning that argument explicitly — the same
> argument `../workloads/moe.md ## Bucketing: three shapes, and what selects one` makes for the
> global-atomic reservation shape.

### Authoring rules

- **The atomic is not a rendezvous.** It orders accesses to the word it touches and nothing else,
  so the surrounding discipline is the `.scatter` discipline unchanged: the table has to be
  initialised and made visible before the first atomic, and a `gl.barrier()` has to separate the
  atomics from any read-back of the table. `gl.allocate_shared_memory` accepts an initial value,
  which is the clean way to seed a counter table without a separate store phase — but it still
  needs the barrier after it, because the seeding lanes are not the reading lanes.
- **Keep the table trivially laid out.** A counter table is single-word indexed access, so a
  swizzle buys nothing here and complicates the index-to-address mapping; a plain
  `SwizzledSharedLayout(1, 1, 1, [0])` over a 1-D `int32` allocation is the shape to start from.
  Whether the resulting bank behaviour matters for your index distribution is a measurement, not
  something the layout choice settles.
- **Fall back to `.gather` the moment you stop needing the RMW.** Reading the finished table is an
  ordinary indexed read, and mixing an atomic in where a `.gather` would do adds the ordering
  caveat above for nothing.

```python
# CTA-local bucket counting: reserve a slot per active route, then read the finished counts back
NBUCKET: gl.constexpr = 256
COUNT_SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])      # 1-D, unswizzled
LIN: gl.constexpr = gl.SliceLayout(1, BLK)                                # layout of the seed tile

counts = gl.allocate_shared_memory(gl.int32, [NBUCKET], COUNT_SH,
                                   gl.zeros([NBUCKET], gl.int32, layout=LIN))
gl.barrier()                                    # seeding lanes are not the reading lanes

slot = counts.atomic_scatter_add(                # prev value == this lane's slot in its bucket
    gl.full(bucket.shape, 1, gl.int32, bucket.type.layout), bucket, 0, mask=active)

gl.barrier()                                    # atomics are not a rendezvous
total = counts.gather(bucket, 0)                 # plain indexed read; no RMW needed here
```

### Downgrade below 3.8.0

The failure mode here is the **opposite** of the one `loop_unroll_factor` has
(`pipeline/loop-knobs-and-targets.md ## Roll the loop (cut i-cache pressure)` — a keyword that is
accepted everywhere and read only on 3.8.0), and that is the useful thing to
know: these are *methods*, so on 3.6.0 / 3.7.0 / 3.7.1 the call does not exist and the kernel fails
at trace time with an attribute error. It is loud, it fails on import-and-trace rather than at a
shape you did not run, and a `hasattr` probe settles it before you write anything.

Two downgrades, and they are not equivalent:

- **Global atomics on a scratch buffer.** `gl.atomic_add` and friends are the plain builtins,
  present on every version, and they express the same reservation at global scope. The table
  leaves LDS, the reservation becomes visible to every CTA rather than one, and the ordering
  caveat above gets *wider* rather than going away. Use it when one CTA was never the right
  parallelism anyway.
- **Scan plus gather.** Where the collisions can be resolved by sorting or by a scan over a
  known axis, `gl.associative_scan` + `.gather` (3.7.0+ for the shared-descriptor half) removes
  the atomic entirely and restores determinism. It costs a second pass over the data and the
  scratch to hold it. This is the downgrade to prefer when reproducibility was load-bearing.

## gfx1250

- `PartitionedSharedLayout` for WMMA operand staging (wave32).
- 320 KiB LDS partition model — probe occupancy per kernel.

## vs CuTeDSL smem/TMEM

| CuTeDSL | Gluon |
| --- | --- |
| TMEM accumulator (sm_100) | **scoped ceiling** — AGPR/VGPR on AMD |
| TMA swizzle modes | LDS swizzle + buffer resource |
| `make_swizzle` XOR | padded/blocked shared layouts |

## Footguns

- gfx950 **1280 B** LDS alignment can waste bytes on small tiles — read the waste from `shared`,
  do not compute it as a round-up (see the arch bullet at the top of this page).
- gfx942 kernel on gfx950 without relayout → wrong results or spill.

## Anchors

- [triton-lang/triton `gluon/language/amd/`](https://github.com/triton-lang/triton/tree/main/python/triton/experimental/gluon/language/amd)
