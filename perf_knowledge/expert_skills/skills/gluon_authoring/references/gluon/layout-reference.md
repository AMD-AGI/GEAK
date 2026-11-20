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
  and ownership prove the extra lanes are valid — **that clause is a legality
  escape hatch, not an efficiency one.** An over-covering layout that masks
  correctly still re-fetches the same bytes *within a lane*, which no amount of
  cross-lane coalescing can merge; one measured instance was paying 2x on that
  tensor and cutting the coverage to equal the tile was worth 5.3–5.8 % on the
  kernel (`../pitfalls/negative-patterns.md ## Instruction count is not the objective
  function`). Treat a coverage product above the tile extent as a cost to price,
  not as a box already ticked;
- for vector memory paths, choose `size_per_thread` from intended vector width,
  not only from the smallest compiling value.

The last rule is arithmetic, not taste. The width of the load the backend can emit for a tile is
set by how many **contiguous bytes one lane owns**:

```text
bytes_per_lane = size_per_thread[contiguous_dim] * dtype_bytes
```

so a `size_per_thread` of 8 on a bf16 tile is 16 B/lane and a `dwordx4`-class access is reachable;
the same 8 on an fp8 tile is 8 B and it is not. Two consequences that catch people out:

- **A narrower dtype tightens this, it does not relax it.** Halving `dtype_bytes` halves
  `bytes_per_lane` for the same `size_per_thread`, so moving a kernel to fp8 moves it *away* from
  the wide-access thresholds unless the per-thread extent grows to compensate.
- **The per-lane width and the wave's coalescing are different questions.** 64 lanes each owning
  2 contiguous bytes is still a contiguous 128 B footprint for the wave. Judge coalescing on the
  wave's footprint and instruction width on the lane's. This is also why an inventory of narrow
  load instructions is not a measurement of the access — what that inventory bounds, and what to
  read instead, is in `../tile-programming/memory-path.md ## A load's width is issue-side; a
  transaction's width is access-side`.

`bytes_per_lane` is also the quantity the async direct-to-LDS floor is stated in — per-dtype element
minimums and the per-arch chunk widths are in
`memory-reference.md ### Minimum per-thread granularity (applicability by dtype)`, which is the
owner of that threshold. And `size_per_thread` is coupled: raising it changes the register budget,
the emitted width, and (on a tile carrying a `gl.gather`) the shuffle count, so it is never a
one-line edit.

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
and cannot be swept independently without redesigning all related layouts. **A mismatch is a
compile-time failure, not a slower number** — `product(warps_per_cta) != num_warps` does not lower,
so a `num_warps` sweep over a hand-authored Gluon kernel returns build errors rather than a curve,
and there is no arm of it to time. Plan the sweep as "one layout redesign per point", and see
`../pitfalls/negative-patterns.md ## warps_per_cta is not an independently tunable knob` for what
the diagnostic looks like when the layouts are *internally* inconsistent instead (an unattributed
assert, not a message about `num_warps`).

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

## Shared Layouts: The Three Constructors

Everything above is the distributed (register) side. The LDS side is where a wrong choice shows up
as bank conflicts rather than as a verifier error, and the three shared layouts are named all over
this pack without anyone saying what they take. Build them on the host and pass them as
`gl.constexpr`, like every other layout here.

| Constructor (3.8.0 spelling) | Reach for it when |
| --- | --- |
| `SwizzledSharedLayout(vec, per_phase, max_phase, order, cga_layout=[])` | Default. An XOR swizzle costs no capacity, and it is the variant confirmed to lower on the async direct-to-LDS path (`memory-reference.md ### Shared-layout family + transpose-on-read (layout dependency)`). |
| `PaddedSharedLayout(interval_padding_pairs, offset_bases, cga_layout, shape)` | A matrix-operand buffer whose conflict pattern is not expressible as a swizzle — on gfx950 + 3.8 derive it with `## Let 3.8 Compute The Padded Layout For You (CDNA4)` rather than by hand. gfx942 downgrade: the v3 transpose-B path failed LLVM translation and padding is the recorded fix there (`../hardware/cdna3-gfx942.md`). Costs LDS capacity, and as an **async** destination it has not lowered (`memory-reference.md`). |
| `SharedLinearLayout(offset_bases, block_bases=[], alignment=16)` | You are recovering an explicit linear layout out of TTGIR and want it back verbatim (`../method/transcribe.md`). Most general, least self-documenting. |

What the constructors assert — so a wrong value fails at trace time rather than mis-laying the
tile silently:

- **`PaddedSharedLayout`**: at least one `[interval, padding]` pair; every interval and every
  padding a power of two; the intervals **distinct** — a repeated interval raises rather than
  summing; and every `offset_bases` / `cga_layout` basis the same rank as `shape`. Where a position
  falls inside several intervals the paddings **add**.
- **`SharedLinearLayout`**: `offset_bases` non-empty, all bases one rank, `alignment` a positive
  power of two (default 16).
- **`PaddedSharedLayout.with_identity_for(interval_padding_pairs, shape, order, cga_layout=[])`**
  derives `offset_bases` for you as the identity mapping implied by `order`, which is what you want
  whenever you are padding a tile you did not permute. It requires `len(shape) == len(order)` and
  every `shape` entry a power of two, and it is a `@constexpr_function` — host side. **It is not the
  one to reach for on the async path**: the padded-identity variant has been seen to fail to lower
  there — and so, measured since, has the 3.8 factory's padded output, so the delta belongs to padded
  async destinations generally, not to this constructor alone
  (`memory-reference.md ### Shared-layout family + transpose-on-read (layout dependency)`;
  `### You are only half done when the layout comes back` below).

Two capacity facts belong in this choice and are easy to leave out of it: the LDS allocation
granularity can round a small padded tile up to more memory than the padding itself explains, and
LDS caps resident waves independently of VGPRs (`smem-lds-reference.md`). A padding decision is an
occupancy decision.

### Checking the choice at compile time

`gl.bank_conflicts` answers "does this distributed layout read that shared layout without excess
accesses" from the two **types** alone — no GPU and no profile. `appendix-api.md ## The compile-time
oracle worth knowing about` is its owner and states what the number means. Three facts belong here
instead, because they are about how you author the call:

- **It is a `@builtin`, not a host-side `@constexpr_function`.** It is evaluated during tracing, so
  the call goes *inside* the `@gluon.jit` body even though everything it reads is static.
- **Its result is a `gl.constexpr`, so pair it with `gl.static_assert`** and a layout requirement
  becomes a build failure instead of a comment:

  ```python
  @gluon.jit
  def kernel(reg_ty: gl.constexpr, smem_ty: gl.constexpr, ...):
      conflicts: gl.constexpr = gl.bank_conflicts(reg_ty, smem_ty)
      gl.static_assert(conflicts == 0)
  ```

  The two types are built on the host — `gl.distributed_type(dtype, shape, layout)` and
  `gl.shared_memory_descriptor_type(dtype, shape, layout, alloc_shape)` — and passed in as
  `gl.constexpr`, which is the same layout-factory discipline as everywhere else on this page.
- **It refuses more than it answers, and the refusals are informative.** The two shapes must be
  equal and the two dtypes must be equal — a mismatch raises rather than returning a large number.
  And **subslices are not implemented**: if the descriptor's shape is not the tail of its
  `alloc_shape`, it raises. So you cannot ask it about a `smem.slice(...)` view, which is exactly
  the shape the LDS-dedup recipe produces. Query the parent allocation's type instead, and treat
  the sliced view as unchecked.

Use it to eliminate a hypothesis before spending a run on it. Zero is not evidence the kernel is
fast; non-zero is a defect you can fix while authoring.

## Let 3.8 Compute The Padded Layout For You (CDNA4)

`gl.amd.cdna4.compute_efficient_padded_shared_layout(dot_operand_layout, shape, dtype, is_k_contig=True)`
takes the dot-operand layout the tile is headed for and returns a `PaddedSharedLayout` chosen to
avoid bank conflicts. It is a `@constexpr_function`, so it belongs in the host-side layout factory.
On gfx950 + 3.8 this is how you get a padded layout — hand-computed padding is the downgrade path,
not the starting point. It is **not** automatically the right layout for the buffer: whether padded
or swizzled wins on a matrix-operand buffer is measured, not derived
(`### The selection rule is a starting point, not a result — the write side can reverse it`).

### Which buffers get it — the selection rule, decided per buffer

> **Reach for this factory for a shared buffer if and only if that buffer's contents are read back
> as an MFMA dot operand. Every other shared buffer takes a `SwizzledSharedLayout`.**

"Every other" is, concretely: fp32 block-scale vectors and lookup tables, cross-wave reduction and
exchange scratch, and the epilogue output tile. A kernel that stages A and B for a matrix
instruction *and* keeps an fp32 scale table in LDS makes **both** choices, in the same function, on
adjacent `allocate_shared_memory` calls. Treating the layout family as a per-kernel style — "this
is a padded kernel" — is the mistake; the unit of the decision is the buffer.

The rule is not a convention, it falls out of the signature: the first argument is a
`DotOperandLayout`. A buffer no matrix instruction reads has no operand layout to derive padding
from, so there is nothing to hand this function, and the capacity-free default applies
(`## Shared Layouts: The Three Constructors`). Padding spends LDS capacity, and LDS caps resident
waves independently of VGPRs (`smem-lds-reference.md`), so every buffer you move onto this factory
is an occupancy decision as well as a layout one — which is the second reason to make the call per
buffer rather than per kernel.

### The selection rule is a starting point, not a result — the write side can reverse it

The selection rule is derived from the signature, so it tells you which buffers the factory *can*
serve. It does not tell you that the padded layout is faster for them, and there is now a measured
case where it is not.

The factory optimizes the **read**: it picks padding that avoids bank conflicts when the matrix
instruction pulls the operand back out of LDS. Nothing in it prices the **write** that fills the
buffer, and a padding pattern that is ideal on the read side can be unvectorizable on the write
side. Measured on gfx950 / `v3.8.0`, one operand shape, comparing only the shared layout: the
padded destination emitted **96 `ds_write_b8` plus 32 `ds_write_b8_d16_hi` per K step** — 128
single-byte LDS writes, because the padding intervals break the store into scalars — while the
swizzled destination emitted **zero** of them. Swapping that buffer to `SwizzledSharedLayout` was
worth **2–7 %** on its own, *before* any async copy entered the picture.

> **So A/B the two shared layouts on a matrix-operand buffer; do not inherit the answer from the
> selection rule.** Read `ds_write_b8` / `ds_write_b8_d16_hi` counts out of the ISA to *locate* the
> candidate — a non-zero count on a staged operand means the fill is running scalar — **and then
> time it.** The count ranks candidates; it does not price them.

That last half-sentence is not a hedge. The same count was read on a second byte-write population
in the same archetype and priced at nothing: cutting `ds_write_b8` from 16 to 4 on a *scale* tile's
global→LDS staging path measured **0.0 % at the small shape and 0.45 % slower at the large one**,
and the 2.6–2.8 % that had been credited to it turned out to be the **global** side of the same edit
(the load width went from 1 to 4 bytes per lane, `global_load_ubyte` 148 → 132). Both sites are
scalar LDS writes and the counts move the same way; what differs is whether the fill is on the
critical path and how many bytes it is. 128 scalar writes per K step on the main matrix operand is
worth 2–7 %; 16 of them on a 16-byte scale tile is worth zero. Nothing in the count distinguishes
the two, so a count that improves is a reason to run the clock, not a result.

Three reasons to check rather than assume: padding spends LDS capacity and swizzle does not
(`smem-lds-reference.md ## LDS is a second, independent occupancy limiter`); the swizzled
destination is also the one that lowers on the async path
(`memory-reference.md ### Shared-layout family + transpose-on-read (layout dependency)`); and the
conflict the padding was bought to remove may not be the buffer's actual limiter. **Scope:** one
operand shape, one dtype, one kernel — this establishes that "padded is the efficient default for
matrix operands" is false as a general statement, not that swizzle always wins.

### Contract points — places this goes wrong quietly

1. **The parent must be an `AMDMFMALayout` with `version == 4`.** Both are asserts. There is no
   gfx942 form of this function — on v3 you are hand-computing
   (`../hardware/cdna3-gfx942.md`).
2. **`shape` is the shared-memory tile, not the problem tile**: `[BM, BK]` for operand A, `[BK, BN]`
   for operand B. Passing the logical GEMM shape produces a layout for a tile you never allocate.
3. **The layout is rank-2; multi-buffered allocations are not.** A ring allocation prepends a stage
   dimension — `gl.allocate_shared_memory(dtype, [STAGES, BM, BK], shared_a)` under a layout
   computed for `[BM, BK]`. The stage dimension belongs at the allocation, never at this call: pass
   the **single-stage** tile, and let the rank-3 allocation be built on the rank-2 layout. Passing
   the ringed shape asks for padding for a tile you do not have, and the rank mismatch between a
   rank-3 `shape` and the rank-2 tile the dot-operand layout describes is not something the
   function can diagnose for you.
4. **Element bit-width is the contract, and it is {4, 8, 16}.** That is what the signature admits —
   anything else is outside the covered set (point 5). Within it, the documented mechanism for
   packed fp4 is to pass `ttgl.uint8`: two values share a byte and, at the LDS level, the 4-bit
   case reuses the 8-bit padding pattern. Treat that as the signature's statement rather than as
   settled practice — we have no compile-verified packed-fp4 instance of it, so if you are the
   first, verify the returned layout against your allocation before building on it. Note also that
   `uint8` has a second, unrelated use here: byte-transparent staging of 8-bit data whose addresses
   you compute in bytes. Seeing `uint8` passed to this function is therefore **not** evidence of an
   fp4 path.
5. **`None` is a return value, not an error — and whether you must test for it is conditional.**
   The function returns `None` when the inputs fall outside the algorithm's covered set: `k_width`
   not in {4, 8, 16}, element bit-width not in {4, 8, 16}, or an MFMA instruction-shape /
   `k_width` combination it does not handle.

   **The third cause is the one that fires in practice, and the variable it turns on is the tile,
   not the `k_width`.** Measured on gfx950 / `v3.8.0` at `AMDMFMALayout(version=4,
   instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 4])` with `dtype=ttgl.uint8`,
   sweeping operand index x `k_width` x shape: `k_width=32` is `None` everywhere (cause 1, as
   written), but inside the covered `{4, 8, 16}` the answer is decided by the **non-K tile
   length** — at operand 0, `k_width=16` needs a non-K extent of at least 32 and `k_width` of 4 or
   8 needs at least 64; at operand 1, `k_width` of 4 or 8 is non-`None` across 16..256 while
   `k_width=16` again needs at least 32. So **a short non-K tile takes the helper away**, which
   makes this the ordinary outcome in the decode band and a non-event at prefill tiles.

   **The threshold is not a function of the tile alone — it is a function of
   `(instr_shape, operand_index, k_width, tile)`, and all four move it.** Reading the numbers above
   as "the padded layout appears at `BLOCK_M >= 32`" is the mistake this paragraph exists to
   prevent. A second `instr_shape` moves the answer in both directions at once: at
   `[32, 32, 64]`, operand 0 is still `None` at a non-K extent of 32 — *later* than the
   `[16, 16, 128]` sweep would predict — while operand 1 already exists at 16, *earlier*. So the
   operand index is a first-class axis here and not a detail, and a kernel that gates only on
   `BLOCK_M` will be wrong for one of its two operands. The three causes are the docstring's and
   are general; **every threshold on this page is configuration-local.** Probe the call for your
   own `(instr_shape, operand_index, k_width, dtype, shape)` — it is a `@constexpr_function`, so
   this costs a host-side loop and no compile.

   What follows from the `None` depends on your arguments:

   - **If every argument is statically pinned** — one `instr_shape`, one `k_width`, one `dtype`,
     with nothing switching between them — the call constant-folds to a single combination that
     either works or fails the first time you compile. `None` cannot appear later, and a fallback
     branch is dead code you have no way to exercise.
   - **If any argument varies at compile time**, you must handle it. The shape that produces this
     is a flag that switches instruction shape, `k_width` and `dtype` *together* (a
     "native-fp8 or not" constexpr is the usual one), or a `k_width` you are sweeping: different
     configurations then land on different points of the covered set. Gate the whole LDS path:

     ```python
     A_SHARED: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
         dot_a, [BM, BK], DOT_TYPE)
     B_SHARED: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
         dot_b, [BK, BN], DOT_TYPE)
     USE_LDS: gl.constexpr = A_SHARED is not None and B_SHARED is not None
     ```

     with the `not USE_LDS` arm running the register-staged variant. Gating the *path*, not just
     the allocation, is the point — a fallback layout under a pipeline that assumed the padded one
     buys nothing.
   - **Where `k_width` is the cause, clamp instead of branching.** `k_width` is the argument most
     likely to be tunable; write `gl.DotOperandLayout(0, mma, min(K_WIDTH, 16))` and keep the call
     inside the covered set, rather than testing the result afterwards. **The clamp does not cover
     cause (3)**, though — a short non-K tile returns `None` at every legal `k_width`, so a kernel
     that sweeps `BLOCK_M` or `BLOCK_N` still needs the `USE_LDS` gate above and cannot clamp its
     way out.

   A factory that does neither and passes the result straight into `allocate_shared_memory` fails
   somewhere downstream with a message about `None`, not about coverage.

### `is_k_contig` — leave it alone unless you are staging an attention second operand

`shape` is passed in the operand's **logical** order (`[M, K]` for operand 0, `[K, N]` for operand
1). `is_k_contig` answers a different question about the same tile: in the allocation you will
actually make, is the dot's K dimension the fast-moving one? For a GEMM staging both operands the
two orders coincide, K is `shape[-1]` for A and `shape[0]` for B in a tile laid out to match, and
the default is correct for both. Leave it at the default there.

The two come apart in one situation worth naming: the **second dot operand of an attention or
KV-cache kernel**, where the cache tile is stored key-index-major with the head dimension strided.
There, the tile you allocate is not in the operand's logical order, so `is_k_contig` and the axis
order of `shape` disagree — and the pair `(shape order, is_k_contig)` is how you say so. Neither
operand 0 of anything, nor either operand of a GEMM, needs this.

How to decide without guessing: write down the tile **as you will allocate it**, then ask which
axis the dot consumes as K. If that axis is the fast one, `is_k_contig=True`; if it is the outer
one, `False`. If the allocated axis order then differs from the logical order you passed as
`shape`, rebuild the returned layout with its bases swapped — the returned object's
`.interval_padding_pairs` and `.offset_bases` are readable attributes, not an opaque handle:

```python
@triton.constexpr_function
def cache_tile_layout(block, head_dim, dtype, is_key):
    mma = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32],
                               transposed=True, warps_per_cta=[1, 4])
    operand = gl.DotOperandLayout(1, mma, 8)
    # Logical [K, N] for this operand: K is head_dim for the key tile, block for the value tile.
    shape = [head_dim, block] if is_key else [block, head_dim]
    shared = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        operand, shape, dtype, is_k_contig=is_key)
    if not is_key:
        return shared                      # allocated [block, head_dim]; K is the outer axis
    # Key tile is allocated [block, head_dim] too, so express the same padding with axes swapped.
    return gl.PaddedSharedLayout(
        shared.interval_padding_pairs,
        [[b[1], b[0]] for b in shared.offset_bases],
        [], [block, head_dim])
```

Get the flag wrong and nothing raises: you get a padding pattern computed for the transpose of the
tile you allocated, which is a bank-conflict pattern rather than an error. `gl.bank_conflicts`
(`### Checking the choice at compile time`) is how you catch it without a GPU.

### You are only half done when the layout comes back

This factory gives you the **consumer** side — how the matrix instruction reads the tile back out
of LDS. If the tile arrives by async direct-to-LDS copy, the **producer** side is a second layout
you owe the compiler, and it is derived from this one by reading the returned object's
`.offset_bases`, not authored beside it.
`memory-reference.md ## Deriving the producer layout for a padded async destination` owns that
derivation and the worked helper. Read it before you write the `buffer_load_to_shared`, not after
it fails to lower.

**And read `memory-reference.md ### Shared-layout family + transpose-on-read (layout dependency)`
before you assume the pair lowers at all.** An earlier revision of this page said the async-path
caveat belonged to `PaddedSharedLayout.with_identity_for` and **not** to this function's output.
That is now measured false: with everything else held fixed, a destination from this factory failed
LLVM translation under `buffer_load_to_shared` at the tile sizes tried, while a plain
`SwizzledSharedLayout` destination succeeded at the same sizes and verified by copy-back. So the
**producer layout is not the only variable** — the destination's layout family is one too, and if
you are staging by async copy, get the copy working against a swizzled destination before you spend
a round on the padding. The full 2x3 result is in that section.

On 3.7.1 and earlier this function does not exist. The downgrade is arithmetic you do yourself, and
`../tile-programming/layout-recipes.md ## Padding vs swizzle (LDS bank conflicts)` is the recipe;
the four-version table in `pipeline-reference.md` remains the authority on which build has what.

## The Scale Operand's Layout For mfma_scaled

fp4 on CDNA4 has no regular matrix intrinsic at all and must go through
`gl.amd.cdna4.mfma_scaled`. **fp8 is a shape question, not a dtype one**: the registry is keyed per
`(version, M, N, K)`, and `v3.8.0`'s MFMA v4 table registers regular `mfma_f32_16x16x32_fp8_fp8`
and `mfma_f32_32x32x16_fp8_fp8` — so fp8 at `[16, 16, 32]` / `[32, 32, 16]` has a regular intrinsic,
while `[16, 16, 128]` / `[32, 32, 64]` exist only as the scaled `mfma_scale_f32_*_f8f6f4` entries and
are what forces this path (`matrix-reference.md ## Matrix-Family Details`, which owns the
instruction's signature). That instruction's **scale** operand has its own layout, and it is
derivable rather than guessable:

```python
scale_layout = gl.amd.cdna4.get_mfma_scale_layout(dot_operand_layout, shape, scale_factor=32)
```

- It is a `@constexpr_function` (host side) and returns a `DistributedLinearLayout` — not something
  you would have written by hand, which is the point.
- **`scale_factor` is asserted to be 32.** The assertion message says so outright: only 32 is
  supported for CDNA4 scaled MFMA. This is the hardware group size expressed as an API contract, and
  it is the same fact the quantization producers are held to — a group of 128 is a software
  convention the instruction has no notion of, and NVFP4's 16 is a different hardware's number.
  `../workloads/prologue.md ## The quant epilogue's scale layout is a contract` is the producer side
  of the same contract. **On the producer side the same number pins a tile extent, and getting it
  wrong is silent.** A kernel that quantizes its own output to this format must tile the quantized
  dimension at exactly the group size, because each tile computes one scale for the elements it
  holds: measured, a block extent of 16 and a block extent of 64 both compiled, launched, threw
  nothing, produced no NaN, and failed an exact comparison against the reference at **all 7**
  shapes, while 32 was exact. This is the same defect class as an `mfma_scaled` `k_width` chosen
  without validation, and it has the same remedy — a `gl.static_assert` tying the block extent to
  the group size, written once, rather than a fact to remember.
- The parent of `dot_operand_layout` must be an `AMDMFMALayout` (asserted). The returned layout is
  derived from the parent's `instr_shape`, `tiles_per_warp` and `warps_per_cta` together with the
  operand index — so **it moves whenever the matrix tile moves**. Recompute it in the same place you
  recompute `tiles_per_warp`, not once at the top of the file.
- `shape` is the shape of the scale tensor, not of the operand.

The consequence for a fused producer/consumer pair: the scale layout is a *contract between two
kernels*, and only the consumer can derive it. Generate it on the consumer's dot-operand layout and
pass it to the producer, rather than writing a layout on the producer side and hoping.

**The *producer* layout that stages that tile is a 2–3 candidate per-kernel sweep, not a formula.**
The constraints are hard and they are few — `size_per_thread * element_bits ∈ {32, 128}` if the
tile is staged by async copy, `threads_per_warp[contig] * size_per_thread[contig] == extent along
contig`, and `threads_per_warp[other] <= the tile's other extent` — but inside them the ranking is
kernel-specific and no invariant proposed so far survives contact with two kernels. The same
nominal scale-tile change measured **+2.6 %** on one kernel and **0.9997x (nothing)** on another
whose baseline layout was byte-identical; and "take the largest legal width" is itself a trap,
because the widest width can force a `threads_per_warp` that over-covers the tile's minor extent
and idles half the warp — one such point regressed to **0.9877x**. At least three quantities move
independently here (per-lane width, cross-lane contiguity along the group axis, and row coverage
against the minor extent) and none predicts alone, so enumerate the two or three legal candidates,
time them, and allow the sweep to return "the one you already had".

### The scale operand is register-resident, so staging it through LDS is a choice

`mfma_scaled(a, a_scale, ...)` takes `a_scale` as a **distributed (register) tensor**. Nothing in
the API asks for shared memory. Worked examples stage both the payload and the scales through LDS
and quote a single LDS bill for the pair, which reads like a requirement and is not one — the index
tensors can be built directly in the MMA-derived layout, e.g.
`gl.arange(0, BM, gl.SliceLayout(1, get_mfma_scale_layout(dA, [BM, NG])))`, so the scales load from
global straight into the layout the instruction wants. Three spellings, payload path byte-identical
across all of them, outputs bit-exact:

| spelling | compiled LDS | VGPR | spills |
| --- | --- | --- | --- |
| stage: global → regs → LDS → scale layout | 43520 B | 124 | 0 |
| convert: global → regs → `convert_layout` to scale layout | 41984 B | 118 | 0 |
| **direct: indices built in the scale layout** | **40960 B** | 124 | 0 |

**Know which way this points before you spend a round on it: on a CDNA4 MoE gemm1 the direct form
was 3.9–7.4 % SLOWER end to end**, at three shapes, above the detection floor at each, with null
controls clean — while doing exactly what it advertises (scale buffers not allocated, `ds_read_u8`
72 → 48, `buffer_load_dword` 18 → 6, zero spills, bit-exact). Two candidate mechanisms were not
separated: the direct form emits byte-granular global loads into a kernel that previously had none,
and the scale bytes lose a prefetch — staged, they were fetched a ring stage ahead by the async
copy; direct, they are fetched at use. So state the lever as two-sided: **LDS staging costs capacity
and `ds_read` instructions and buys back a prefetch stage and a wide layout-changing read.** Reach
for the direct form when LDS capacity is the binding constraint and you have checked that it is —
if the register file binds first, freeing LDS buys no occupancy and you have paid the transport
cost for nothing.

Three traps sit on any attempt to measure this:

- **The `convert_layout` spelling returns only part of the LDS**, because cross-lane layout
  conversion lowers through shared scratch (`### Shared Memory Versus Register Conversion`). "Move
  it out of LDS with a convert" silently pays a large fraction of the cost back.
- **`metadata.shared` is `max(explicit allocations, conversion scratch)`**, not the sum of your
  allocations. An epilogue `convert_layout` of a `[128, 256]` fp32 accumulator to a blocked layout
  costs 65536 B of scratch by itself — enough to sit on top of a 2560 B effect and make the whole
  experiment read flat. A probe that measures an allocation must store in the MMA layout or it is
  measuring its own epilogue.
- **The second operand must be read as `[K, N]`.** `deduce_scale_factor` reads K off dim −2 for
  `operand_index=1`, so a `[N, K]`-shaped shared buffer loaded without `.permute([1, 0])` makes it
  compute the wrong K and derive a group size of 128, failing with *"scale factor must be 16 or 32.
  Got 128"* — a message that names neither the operand nor the permute. The `.permute([1, 0])` in
  working source is a requirement, not a stylistic preference.

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

### The two layouts that carry no shape

`AutoLayout` and `CoalescedLayout` are distributed layouts that describe *nothing* about placement,
and both raise `ValueError` on `.rank` rather than answering — which is the clearest statement of
what they are. They show up in generated and transcribed code, so recognize them:

- **`AutoLayout`** is the placeholder a value carries when no layout was pinned. `gl.set_auto_layout(value, layout)`
  is the builtin that resolves one to a concrete layout. Leaving it unresolved on the index math is
  a documented performance failure, not a compile failure: it compiles, it is bit-exact, it passes
  the oracle, and it is several times slower
  (`../method/transcribe.md`). The `## Pre-Run Scan` item about runtime layout objects is
  the same check from the other side.
- **`CoalescedLayout`** asks the compiler for a coalesced arrangement without saying which one.
  Treat it the same way: acceptable while bringing a kernel up, a hole in the transcription once
  you are measuring, because the thing this pack asks you to author is exactly the part it elides.

## `convert_layout` Decision Table

| Situation | Use `convert_layout`? | Reason |
| --- | --- | --- |
| Moving matrix operands into `DotOperandLayout` | yes | Matrix instructions require operand layouts. |
| Re-parenting a 1D slice before broadcast | no | Regenerate the index from the correct parent `SliceLayout`. |
| Fixing measured non-coalesced memory access | maybe | Benchmark against safe anchor and change one memory path at a time. |
| Repeated conversion inside the innermost loop | avoid | Layout movement can dominate the optimized work (`../pitfalls/negative-patterns.md`). |
| Equivalent-layout conversion | maybe with `assert_trivial=True` | Fail early if not a trivial reinterpretation. |
| Cosmetic conversion | no | It adds cost without a hypothesis. |

### Shared Memory Versus Register Conversion

When shared memory is used only to change layout, benchmark it against direct
register layout conversion. On targets with large register files such as
gfx950/CDNA4, direct `gl.convert_layout` can beat store-to-shared plus reload for
small hot-loop conversions. The winner is target-, tile-, and layout-dependent;
test both before assuming LDS staging is required.

## gl.gather — a lane exchange only when the layouts allow it

`gl.gather(src, index, axis)` reads a tensor at positions chosen at runtime:
`result[I] = src[I[0], ..., index[I], ..., I[n]]`, with the shape and layout of `index`. It is a
free function, not a tensor method, and it exists on all four versions (checked 3.6.0 / 3.7.0 /
3.7.1 / 3.8.0). `src` is already a distributed tensor, so there is no pointer and no address
arithmetic here — this belongs to layout work, not to the memory path.

Whether it *costs* memory is decided by the layouts, not by the call site:

| Layouts | Lowering | Cost |
| --- | --- | --- |
| gather is warp-local | warp shuffles only | zero scratch; a shuffle count that grows quadratically (below) |
| anything else | store `src` to LDS, then index out of it | scratch sized for the **whole `src` tensor**, plus the round trip |

Warp-local requires all three conditions, checked together:

1. moving along the gather axis never changes the warp — for `src` **and** for `index`;
2. the `(block, warp)` mapping to every non-gather dimension is identical between the two layouts;
3. the `lane` mapping to every non-gather dimension is identical between the two layouts.

Conditions 1 and 2 are what make the exchange containable inside one warp: the warp holding an
index element also holds every source element that index could reach. Condition 3 only simplifies
codegen, but it is checked the same way, so failing it loses the fast path just as completely.

> **The fallback is silent, and it is not merely a round trip.** Miss any condition and the
> scratch requirement jumps from zero to the full size of `src`, which is charged against the
> kernel's shared-memory budget and therefore against occupancy — a cost that lands on the whole
> kernel, not just on the gather. Derive the index layout from the source layout (`## Slice And
> Broadcast Recipe`) rather than building it independently and hoping the two agree.

Warp-local is not automatically cheap either. The lowering emits an index shuffle for every
(index element owned, candidate source register) pair, and upstream flags this in the source as
quadratic. Widening `size_per_thread` along the gather axis grows the shuffle count faster than it
grows the work, so a gather-carrying tile wants its per-thread extent kept on the gather axis and
spent elsewhere.

`smem.gather` on a shared-memory descriptor is a different mechanism with the same verb — that one
is an LDS access by construction (`smem-lds-reference.md ## Indexed access to LDS — .gather / .scatter`).

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

Symptom -> fix routing for layout/broadcast failures: `../method/triage.md`.
