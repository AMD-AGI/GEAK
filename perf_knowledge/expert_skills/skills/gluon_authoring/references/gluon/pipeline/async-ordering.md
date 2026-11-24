# Gluon pipeline — the async forms, the drain, and the ordering class

## The async forms — gfx950 only, and the shape differs from A5

Primary worked examples for the LDS ring on gfx950. gfx942 has async copy only at 32 bits/thread
(`marker-and-version-gates.md ## The gfx942 async-copy width gate — available, and narrow`), so its
worked set (`authored-overlap.md`) stages through registers (A5).

- **A ring hides latency behind reuse:** if a staged block feeds only one or two MMA tiles there is
  little compute to overlap the copy with; reuse is the quantity that sets what a ring can hide
  (`../../workloads/moe.md ### Two reuse axes, and this page used to carry only one`).
- **Different loop shape from A5, not a different call:** the `wait_group` replaces one of A5's two
  barriers, and the wait depth must match the buffer count.
- **Falsifiable signature of a real async copy: the staging `ds_write` disappears.** A run where every
  case shows `ds_write > 0` measured a sync fallback. Runnable as `scripts/pipeline_examples_cdna4.py`
  (A5 control + C1–C5; `--compile-only` AOT-compiles for gfx950 with no GPU).

Shared preamble — `BLK` covers the `[32, 32]` tile **exactly** (one native-width access per lane). On
Triton 3.8.0 (gfx950 / gfx942, compile-checked) the contiguous dim must be covered exactly
(`threads_per_warp * size_per_thread == extent`; over or under fails lowering); over-coverage of the
outer (row) dim lowers (replicated copies) and returned correct data in a GPU check on gfx950 (two- and four-fold
outer over-coverage, buffer and global forms: replicated lanes write identical bytes) — still GPU-check a new
layout, compile is not proof of correctness:

```python
from triton.experimental.gluon.language.amd.cdna4 import async_copy as _acp

BLK: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [4, 1], [1, 0], [])   # [1*8*4, 4*8*1] = [32, 32]
SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])          # [0,1] fails to lower for this row-major tile (3.8.0)
_barrier = getattr(gl, "thread_barrier", None) or gl.barrier                 # 3.6 spelling shim
```

**`SwizzledSharedLayout(1, 1, 1)` is the destination with the fewest ways to fail on a first ring.** The
async destination must be warp-uniform-addressed (on 3.8.0: width 128 or 32 bit on gfx950, coalesced,
swizzle within a warp — `../memory-reference.md ## Async Copy To Shared`); a padded destination lowers
only under the interval/producer conditions in
`../memory-reference.md ### Shared-layout family + transpose-on-read (layout dependency)`, and
`max_phase = 1` avoids the one silent-wrong-data case found (`max_phase > 1` with `vec < size_per_thread`).
A copy-back check after each change isolates which change broke a ring —
**a compile is not a check on this path.**

**C1 — async double buffer via `global_load_to_shared`** (128-bit/thread here; on gfx942 this width fails
with `PassManager::run failed` / `failed to legalize operation 'ttg.async_copy_global_to_local'`):

```python
@gluon.jit
def c1_async_double_buffer(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    s = gl.allocate_shared_memory(gl.float32, [2, M, N], SH)
    o = (gl.arange(0, M, layout=gl.SliceLayout(1, BLK))[:, None] * N
         + gl.arange(0, N, layout=gl.SliceLayout(0, BLK))[None, :])
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)
    _acp.global_load_to_shared(s.index(0), inp + o)                  # prologue: tile 0
    _acp.commit_group()
    for i in range(ITERS):
        cur = i % 2
        nxt = (i + 1) % 2
        if i + 1 < ITERS:
            _acp.global_load_to_shared(s.index(nxt), inp + (i + 1) * M * N + o)
        _acp.commit_group()           # UNCONDITIONAL: an empty group on the last step keeps the count right
        _acp.wait_group(1)            # let the i+1 copy stay in flight
        _barrier()
        acc += s.index(cur).load(BLK)
    gl.store(out + o, acc)
```

vs A5: one barrier instead of two, `commit_group` even when no copy was issued, no `.store()` (the copy
writes LDS directly).

**C2 — `buffer_load_to_shared`** (scalar base + int32 offsets; separate lowering — on gfx942 at this width
it fails `failed to translate module to LLVM IR` / `builtin.unrealized_conversion_cast`):

```python
    _acp.buffer_load_to_shared(s.index(0), inp, o)                   # prologue
    ...
        _acp.buffer_load_to_shared(s.index(nxt), inp, o + (i + 1) * M * N)
```

**C3 — C1 + `load_shared_relaxed`** (sound only because `wait_group(1)` already retired the group that
filled `s.index(cur)`):

```python
        acc += _acp.load_shared_relaxed(s.index(cur), BLK)           # not a method on the descriptor
```

**C4 — three buffers, two copies in flight** (depth > 2 is where gfx950's 160 KiB LDS matters —
`../smem-lds-reference.md ## LDS is a second, independent occupancy limiter`). Two prologue groups, depth 2:

```python
    s = gl.allocate_shared_memory(gl.float32, [3, M, N], SH)
    _acp.global_load_to_shared(s.index(0), inp + o)                  # tile 0
    _acp.commit_group()
    _acp.global_load_to_shared(s.index(1), inp + M * N + o)          # tile 1
    _acp.commit_group()
    for i in range(ITERS):
        if i + 2 < ITERS:
            _acp.global_load_to_shared(s.index((i + 2) % 3), inp + (i + 2) * M * N + o)
        _acp.commit_group()           # unconditional, as in C1
        _acp.wait_group(2)            # two groups may stay outstanding
        _barrier()
        acc += s.index(i % 3).load(BLK)
```

> **Each iteration must read a DIFFERENT tile** (input `[ITERS, M, N]`, reference = sum over tiles) or
> reading the wrong stage / overwriting early still sums correctly and the test cannot falsify the ring.

## Getting `wait_group` right

- **`wait_group(n)` blocks until ≤ `n` commit groups are outstanding.** Lowering:
  `llvm.amdgcn.wait.asyncmark(n)` → LLVM derives the `s_waitcnt vmcnt`. **A missing `commit_group` is a
  silent race on 3.8.0, not a blocking copy** (GPU-verified on gfx950: the LDS read returned stale data): the uncommitted copy has no asyncmark, so no `wait_group`
  (not even `wait_group(0)`) covers it, and because `wait_group(N)` counts marks the preceding group can
  be under-waited too (`wait_group(1)` then emitted no `s_waitcnt` at all). The `wait_group` docstring's
  "uncommitted ops are waited upon" is not what the asyncmark lowering does — check the `.amdgcn`.
- **Tail race:** a literal depth needs an unconditional `commit_group` (C1/C4); guarding the commit with
  the copy makes the tail wait a no-op. Such a ring benchmarked faster and was wrong at only some shapes,
  so **gate every async-ring change on bit-exactness vs the pre-change kernel at every shape you run.**
- **The argument must be a trace-time Python int.** `wait_group` lowers to `ttg.async_wait` with a
  static count; in a dynamic loop (bare `range` / `tl.range` → `scf.for`, the normal case and the loop
  form `warp_pipeline_stage` builds its phase offset around) `wait_group(min(DEPTH - 1, STEPS - i - 1))`
  **fails at trace time** (compile-checked 3.8.0). In a
  dynamic loop: keep `commit_group` unconditional with a literal depth, or branch between literals
  (`if i + 1 < STEPS: wait_group(D - 1) else: wait_group(0)`). The rolling closed form is legal only
  inside `gl.static_range` / constexpr-bounded unrolled bodies, where each copy gets its own literal
  (`../../tile-programming/pipeline.md ### Hand-built buffering rules (correctness + scheduling footguns)`).
  Name depths against structure (`STAGES - 2`, per output panel, `PREFETCH - 1`) and count **groups**,
  never copies.
- The `triton/experimental/gluon/language/amd/cdna4/async_copy.py` docstrings recommend
  `buffer_load_to_shared` over `global_load_to_shared` (the latter carries a 64-bit pointer per thread:
  more registers, no hardware out-of-bounds masking); its uncommitted-op sentence does not match the
  3.8.0 lowering — see the missing-`commit_group` bullet above.
- **`load_shared_relaxed` is not a faster load:** it is `shared_load` with `ttg.amdg.syncedViaAsyncWait`.
  The compiler emits an `s_barrier` right after every `wait_group` and before each refill (no
  `gl.barrier()` needed for either ring hazard); after a real `wait_group` `load_shared_relaxed` removes
  no barrier (alias-scope metadata only), and it drops the read's barrier only when readiness comes from a
  hand-issued `s_waitcnt` (`../../tile-programming/pipeline.md ### What the compiler inserts after wait_group: barriers and load_shared_relaxed`).
- **`num_outstanding` counts groups, not copies.** Draining after N individual loads (`vmcnt(8)` →
  `vmcnt(4)` → `vmcnt(0)` within a stage) needs inline-asm `s_waitcnt` — a pacing lever that forfeits the
  compiler's dependency checking (`../../tile-programming/pipeline.md ### Draining below the group: bare s_waitcnt instead of wait_group`;
  asm rules `../inline-asm-reference.md ## Class 4 — synchronization`). Per-M host depth tables and
  asymmetric A/B rings: `../../tile-programming/pipeline.md ### Three shapes production actually builds, and the barrier placement that differs between them`.

> Async copies complete in order with ordinary loads/stores on CDNA4, so interleaving them in one loop
> makes a wait on one also wait on the other; the `async_copy.py` docstring states it costs performance,
> and an ordinary load issued after the copy is not settled by `wait_group` (`## Getting wait_group right`).

## Ordering the fill against the read — class S

Placement rules: `../../tile-programming/pipeline.md ### Three shapes production actually builds, and the barrier placement that differs between them`.

- **`wait_group`** retires this wave's groups; the op promises no CTA sync, but on 3.8.0 the compiler
  emits the `s_barrier` after it and before each refill (compile-verified).
- **`gl.barrier()`** lowers to local-scope `ttg.barrier` (rendezvous + LDS-write visibility), unlike a
  bare `s_barrier` (`../../tile-programming/warp-pipeline.md ## Barrier taxonomy`); at the head of a
  compute cluster it can pull a wait in with it.
- **A depth change is a correctness change:** re-derive the wait argument and barrier placement against
  both hazards (read-before-fill, overwrite-before-read).
- **S2 (`load_shared_relaxed`) asserts S1 ran** — on an A5 ring (no async wait) it has no referent, so
  the assertion it makes is unbacked there (unprobed).
- Count `s_waitcnt` (S1) and `s_barrier` (S4) in the asm against the ring you meant
  (`../../method/triage.md ## gfx950 / Gluon fill`).

gfx1250 (TDM / `mbarrier` model, `wmma`) and RDNA (no async surface) are out of scope; only
`commit_group` / `wait_group` carry over to gfx1250. Re-injection is measured on gfx942 only.
