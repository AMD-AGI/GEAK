# Gluon pipeline — per-target paths (gfx950 first) and the loop knobs

## gfx950 CDNA4 path

`async_copy` is a **module**, not a callable, and the group ops live inside it — `cdna4.commit_group`
and `cdna4.wait_group` do not exist. The destination is also the first argument, not the last:

```python
# producer
gl.amd.cdna4.async_copy.buffer_load_to_shared(lds_ptr, gmem_base, elem_offsets)
gl.amd.cdna4.async_copy.commit_group()
# consumer
gl.amd.cdna4.async_copy.wait_group(n)
# mfma
acc = gl.amd.cdna4.mfma(a, b, acc)
```

The five entry points under `cdna4.async_copy` are exactly `global_load_to_shared`,
`buffer_load_to_shared`, `commit_group`, `wait_group`, `load_shared_relaxed`.

Plain Triton on gfx950 gets `tritonamdgpu-pipeline` automatically and Gluon does not, so the
staging and the sync are yours to write when escalating from plain. The ring itself — the drain
depth, the barrier, the falsifiable `ds_write == 0` signature, and the runnable C1–C4 cases — is
`async-ordering.md ## The async forms — gfx950 only, and the shape differs from A5`. Plain's passes
are present in `libtriton` and can be re-injected, but only as a diagnostic below the parity gate or
a last resort, never as the way to build this loop (`reinjection.md`).

Two gfx950 facts that decide the shape of the loop before any of the above, both owned elsewhere:
the per-thread copy width set is **{128, 32} bits** (`../memory-reference.md ### Minimum per-thread
granularity (applicability by dtype)`), and the LDS capacity that bounds the ring depth is
**160 KiB/CU** (`../smem-lds-reference.md ## LDS is a second, independent occupancy limiter`).

## gfx942 downgrade

What changes when the gfx950 loop above moves to CDNA3, and where each fact is owned:

| gfx950 form | gfx942 downgrade | owner |
| --- | --- | --- |
| async ring at 128-bit per thread | **sync staging (A5) is the default**; async exists only at **32-bit**, with a clean tiling and a destination `order=[1, 0]`, and measured slower than sync staging on every minor | `marker-and-version-gates.md ## The gfx942 async-copy width gate — available, and narrow`; the worked A5 set is `authored-overlap.md ### Worked examples — every one compiled and run on gfx942` |
| 160 KiB LDS/CU, 64 banks | **64 KiB LDS/CU, 32 banks** — deep rings of real tiles run out of LDS before latency; re-derive any swizzle | `../smem-lds-reference.md ## LDS is a second, independent occupancy limiter`; `../../hardware/cdna3-gfx942.md` |
| `ds_read_*_tr_*` transpose reads | **absent** — no transpose-on-read instruction | `../atoms-reference.md ## CDNA3 gfx942 downgrade` |
| `cdna4.mfma_scaled` | **no scaled MFMA** (cannot-select) | same |
| OCP fp8 (`e4m3fn`) | **FNUZ** fp8 | same |
| `cdna4.buffer_load` / `mfma`, `AMDMFMALayout(version=4)` | `cdna3.*`, `version=3` | `../index.md ## gfx950 <-> gfx942 at a glance` |

Writing the downgrade is a structural edit (a sync two-barrier ring instead of a wait-plus-barrier
ring), not a namespace swap.

## gfx1250 path

TDM descriptor async copy replaces buffer_load_lds for matrix operands
(`gl.amd.gfx1250.tdm`). Wave **32** — separate minimal anchor from gfx950.

## Roll the loop (cut i-cache pressure)

**Both loop knobs are writable from a Gluon body on every version, and what differs is whether
any pass reads them.** `tl.range` is usable inside a `gluon_jit` body
(`reinjection.md ## Re-injecting plain's pipeliner — the measured recipe`), and the code generator
that lowers it is shared with plain, so
`num_stages=` sets `tt.num_stages` and `loop_unroll_factor=` sets `tt.loop_unroll_factor` on the
`scf.for` no matter which minor you are on. Neither is ever rejected. The consumer is the gate:

- **`num_stages` is dead on the Gluon path on all four versions** (checked 3.6.0 / 3.7.0 /
3.7.1 / 3.8.0). Its consumers are `add_schedule_loops` / `add_pipeline`, which live in
`make_ttgir` — and `add_stages()` routes a GLUON kernel straight to `gluon_to_ttgir`, so a Gluon
kernel never enters `make_ttgir` at all. The attribute lands in the IR and nothing reads it.
So on this path `num_stages` is a **budget parameter** for buffer counting and a **champion-record
field** — never a tuning knob, and a plain winner's value is not carried over as a Gluon depth
(`../../tile-programming/pipeline.md ## Gluon: the pipeliner does not run, and that is the whole difference`).
The depth of an authored ring is the buffer count you allocate. The only thing that hands
`num_stages` to a pass is the re-injection route, which is a diagnostic / last resort
(`reinjection.md`).
- **`loop_unroll_factor` is live from 3.8.0, and inert before it.** `gluon_to_ttgir` gained
`passes.ttir.add_loop_unroll` in 3.8.0 (the ledger in `marker-and-version-gates.md`); on
3.6.0 / 3.7.0 / 3.7.1 the pass is simply not in the Gluon pass list. So on 3.8.0 the annotation is a real knob here, carrying the
same meaning it has in plain. On the three older minors the identical line compiles, sets the
attribute, and changes nothing — the downgrade is `### The downgrade is a structural edit, not a
different keyword` below.

> **Both failures are silent, which is the only reason this is worth a section.** A knob that is
> accepted, recorded in the IR, and never read produces no error and no effect — the same
> signature as the vendor-fork `TRITON_GLUON_*` names in `reinjection.md`. Do not read a null
> result from either as "unrolling does not help this loop". Read `tt.loop_unroll_factor` back
> off the TTGIR if you want to know the attribute landed, and count the body's instructions in
> the asm if you want to know the pass ran; those are two different questions.

### Where the factor goes: on the loop, never on the launch

**`loop_unroll_factor` is not a launch option, and looking for one costs a whole round.** It is a
keyword argument to the *loop construct inside the kernel body*, alongside the bounds. No launch
accepts it; `kernel[grid](..., loop_unroll_factor=8)` is not the spelling of anything, and an agent
that goes hunting for a launch-time knob and fails to find one is liable to conclude the mechanism
is unavailable on this build. It is available — it is one level down from where it was looked for.

```python
# the annotation lives on the loop, with the bounds
for k in tl.range(0, K_TILES, loop_unroll_factor=1):
    ...
```

This is the same `tl.range` the `num_stages` bullet above is about, and the difference between the
two knobs is exactly that: `num_stages` can be written *either* on the launch or on the loop, and
`loop_unroll_factor` only on the loop. Two consequences worth holding on to:

- **A host-side decision reaches it indirectly, through a `gl.constexpr`.** When the factor has to
  vary with shape, the host does not pass the factor — it passes a constexpr kernel parameter and
  the body names it in the annotation. That keeps the decision table on the host without inventing
  a launch keyword that does not exist:

  ```python
  # host: one decision function, next to the other shape-keyed knobs
  def _unroll_for(m: int) -> int:
      return 8 if m <= 256 else 1          # a table, not a formula; see the entanglement note below
  kernel[grid](..., UNROLL=_unroll_for(m), num_warps=4)

  # body
  @gluon.jit
  def kernel(..., UNROLL: gl.constexpr):
      for k in tl.range(0, K_TILES, loop_unroll_factor=UNROLL):
          ...
  ```

- **The annotation attaches to the loop wherever the loop ends up.** A `@gluon.jit` helper that
  contains an annotated loop carries the annotation into every call site it is inlined at, so a
  factor chosen for one caller silently applies to the others. If two callers want different
  factors, the factor has to be a parameter of the helper, not a literal inside it.

### `=1` is documentation, not a muzzle — and the amplify case is narrower than it looks

**Read this before you write `loop_unroll_factor=1`.** The pass bails out for any factor `<= 1`,
and **the default when the attribute is absent is already 1**. So `=1`, `=0` and *writing nothing*
are the same program: no loop metadata is emitted and no transform is skipped, because none would
have run.

> `loop_unroll_factor=1` is an **in-source assertion of intent, enforced by nothing.** It is worth
> writing for the next reader — it says "this schedule is authored, do not let a future pass
> duplicate the body" — but it buys no guarantee today, and a reviewer who believes it is a muzzle
> will not go looking for the guarantee elsewhere.

It is also the commonest spelling in the wild by a wide margin, which is easy to misread as
evidence that suppression is a technique. It is evidence that authors want to say something the
language gives them no way to enforce.

**When the concern behind `=1` is real — and what actually addresses it.** Reach for the comment
on any
loop whose body is an authored pipeline: explicit `commit_group` / `wait_group`, multi-buffer
rotation through `smem.index(...)`, authored `gl.barrier()` placement — everything
`authored-overlap.md ## Authored overlap (no compiler patch)` teaches. What goes wrong if you
leave it off is not subtle once you look for it, but it is invisible in the source: the unroller duplicates the body,
so the buffer index, the drain depth and the barrier placement you derived **for one iteration**
now appear `N` times in a body whose rotation period may not be `N`. The ring and the unroll factor
are then two independent periods over the same buffers. The rule stated under
`async-ordering.md` ### Ordering the fill against the read — class S — *a depth change is a
correctness change* — therefore applies to a depth change you did not make and cannot see in the
source. **But pinning the factor to 1 does not keep that schedule** — it only records that you wanted it.
If a future pass would duplicate this body, the thing that protects you is the authored ordering
itself: the explicit waits, the barrier placement, and `disable_licm=True` where hoisting is the
hazard. Write the `=1` as a marker if you like; do not treat it as the defence.

```python
# a hand-authored ring: the factor is defensive, not a performance dial
for k in tl.range(0, K_TILES, loop_unroll_factor=1):
    if k + 1 < K_TILES:
        _acp.buffer_load_to_shared(smem.index((k + 1) % 2), a_base, offs + (k + 1) * TILE)
    _acp.commit_group()
    _acp.wait_group(1)              # derived for ONE iteration of this body
    _barrier()
    acc = gl.amd.cdna4.mfma(smem.index(k % 2).load(A_OP), b, acc)
```

Suppression also composes with the other muzzle on the same call: `disable_licm=True` stops the
compiler hoisting loop-invariant work out of a body whose placement you chose. A loop carrying both
is a loop whose author is saying *the schedule in the source is the schedule I want*, and that is a
coherent position rather than a pair of workarounds.

**Amplify (`>1`) — and the axis that is easy to miss is partiality, not runtime bounds.** The
obvious case is a trip count that is not a compile-time value: a bound loaded from memory (a
sequence-offset / cumulative-length tensor, a per-expert token count, a page count) has no static
form, so `gl.static_range` cannot be written at all and only a pass running on the `scf.for` can
duplicate the body and emit a remainder loop.

**The less obvious case is the more common one.** `gl.static_range` is all-or-nothing — it takes no
factor. So on a loop whose trip count *is* a compile-time constant, the two tools are not
substitutes:

| you want | the tool |
| --- | --- |
| full unroll of a compile-time loop | `gl.static_range` |
| **partial** unroll of a compile-time loop — factor 2 over 10 trips, factor 4 over 16 | **`loop_unroll_factor` is the only way to say it** |
| any unroll of a runtime-bounded loop | `loop_unroll_factor` |

A majority of real amplifying uses are in the middle row: the trip count was a constant,
`gl.static_range` was writable, and it was **declined** in favour of a factor. Read that as the
answer to "why not just use `static_range`" — full unrolling of a long body is a different trade
from doubling it, and the language has no other way to ask for the second.

```python
begin  = gl.load(starts + seq)                       # runtime — no static_range form exists
end    = gl.load(starts + seq + 1)
chunks = gl.cdiv(end - begin, CHUNK)

for c in tl.range(0, chunks, loop_unroll_factor=UNROLL):
    state = _chunk_step(state, c)                    # compiler emits body x UNROLL + remainder
```

Three properties of the amplifying direction that are easy to get wrong:

- **The factor need not divide the trip count.** The pass emits a remainder; nothing requires
  `trip_count % factor == 0`, and writing the annotation as if it does will make you reject
  factors that are fine.
- **A mixed idiom is often the clean one**: annotate the steady-state loop and handle the peeled
  tail with `gl.static_range`, so the compile-time part stays compile-time and only the part that
  has to be dynamic is.
- **It is not an independent dial.** Raising the factor duplicates live values, which moves
  register pressure, which moves occupancy and the spill threshold — so it lands in the same
  budget as tile shape, the `waves_per_eu` pin and the `llvm_fn_attrs` scheduler strategy
  (`../../tile-programming/instruction-scheduling.md`). Sweep it *with* those, not against a frozen
  configuration, and re-derive it when any of them changes. Whether any particular factor pays is
  a measurement on your body; this reference states where the knob goes and what it does, not what it
  buys.

**One value to treat as unresolved: `0`.** A decision table that returns `0` for some shapes is not
obviously either "no directive" or "no unroll", and this pack has not established which the pass
takes it as. If your host table has a `0` arm, make it explicit — write `1` when you mean *do not
unroll* — rather than inheriting the ambiguity.

### Auditing a tree for this knob — do not grep for `tl.range`

If you are counting where this annotation is used, in your own source or anyone else's, **parse the
call sites; do not match the callee's spelling.** `range` is routinely imported under a local alias
— `from triton.language.core import range as loop_range` and several other names — so a search for
`tl.range(` can miss most of the sites in a tree while looking exhaustive.

The robust form is an AST pass selecting every `Call` that carries a `loop_unroll_factor` keyword,
whatever the callee is named:

```python
for n in ast.walk(ast.parse(src)):
    if isinstance(n, ast.Call):
        for kw in n.keywords:
            if kw.arg == "loop_unroll_factor":
                ...   # ast.unparse(kw.value) keeps computed forms readable
```

This is the same failure shape as searching for `gl.inline_asm` instead of `inline_asm`, or for an
atomic's plain name instead of its scatter-family spelling. **The name is not the mechanism**, and
in this codebase a mechanism usually has more than one name.

### The downgrade is a structural edit, not a different keyword

Below 3.8.0 there is no keyword to switch to: the annotation is accepted and inert, so the
downgrade has to change the *structure* instead of the argument. Which edit depends on which
direction you were asking for:

| you wanted | downgrade below 3.8.0 |
| --- | --- |
| suppress (`=1`) | nothing to do — with the pass absent, an `scf.for` written as `tl.range` / plain `range` is already not unrolled. The risk here is the reverse: a `gl.static_range` you wrote elsewhere still unrolls at trace time on every version, because that is the frontend and not the pass |
| amplify, compile-time trip count | `gl.static_range` — this never needed the knob on any version |
| amplify, **runtime** trip count | there is no source-level equivalent. Chunk the loop by hand: an outer dynamic loop over `count // F` with an inner `gl.static_range(F)` body, plus an explicit remainder loop for `count % F`. That is what the pass would have emitted, written out — and it is a real rewrite of the loop nest, so budget it as one rather than as a flag flip |

And whenever you want the structure rather than the knob on *any* version, the same structural
reading applies in reverse: `gl.static_range` fully unrolls and plain `range` does not, so use
plain `range` for large trip counts and factor a fat hot body into a shared `@gluon.jit` helper to
avoid duplicated code.

```python
for k in range(0, K, BLOCK_K):        # plain range -> rolled loop (static_range would unroll all)
    acc = _dot_step(a, b, acc)        # fat body factored into one @gluon.jit subroutine
```

Verify: the attribute in the TTGIR says the annotation landed; the body's instruction count in the
asm says the pass ran; L1I hit rate and the fetch-latency bubble say whether rolling it addressed
the thing you rolled it for. Those are three questions, and only the third needs a profile.

## Footguns

- **`wait_group` depth mismatch → a silent wrong answer that benchmarks *faster*** — not a hang and
  not a fault, because `wait_group(n)` is a no-op whenever fewer than `n` groups are outstanding, and
  intermittent because whether the data landed depends on timing. A depth sweep scored on time alone
  selects the broken arm. The semantics, the rolling-drain fix and the measured case are owned by
  `async-ordering.md ## Getting wait_group right`; the two defences are to derive `n` by counting the
  `commit_group`s **one iteration of your own body** issues (the worked ring in
  `` ### `=1` is documentation, not a muzzle — and the amplify case is narrower than it looks ``
  annotates its `wait_group(1)` that way), and to **gate every depth change on a numerical check**
  (`../../method/triage.md`).
- Deeper stages without occupancy headroom → regression (`../../tile-programming/slicing.md`).
  Start a ring at 2 buffers and deepen only against a profile: software prefetch regresses once
  waves are already VGPR- or LDS-capped, and aggressive unrolling can raise the `s_nop` count rather
  than lower it.



## vs CuTeDSL (concept filter)


| CuTeDSL                                | Gluon equivalent                                                                                                                            |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `PipelineTmaAsync` + mbarrier tx_count | `commit_group`/`wait_group` + `s_waitcnt`                                                                                                   |
| TMA multicast                          | gfx1250 `cluster` + multi-CTA (CDNA: XCD remap, not multicast) — note TMA is NVIDIA-proprietary; gfx1250 uses `cluster`/TDM, not NVIDIA TMA |
| `cutlass.range(prefetch_stages=)`      | manual prologue/steady-state in source                                                                                                      |
| warp-spec warpgroups                   | `warp_specialize` + `warp_pipeline_stage` (gfx950)                                                                                          |
| CLC                                    | **none** on CDNA; gfx1250 cluster scheduling only                                                                                           |

## Anchors

- [triton-lang/triton](https://github.com/triton-lang/triton/tree/main/python/triton/experimental/gluon/language/amd/cdna4) `cdna4/`
- [triton-lang/triton `gluon/language/amd/`](https://github.com/triton-lang/triton/tree/main/python/triton/experimental/gluon/language/amd) — the `gfx1250/` subtree moved upstream; browse the `amd/` parent rather than a pinned leaf path
