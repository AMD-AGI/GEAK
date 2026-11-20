# Gluon pipeline — authored overlap: register prefetch, what is on offer, and the gfx942 downgrade set

## Authored overlap (no compiler patch)

The default way to get overlap on the Gluon path, on an upstream build with no compiler patch:
the climb builds the pipeline by hand. The order in which to reach for the hand-written forms —
and the place of re-injection beneath all of them — is defined once in
`../../tile-programming/pipeline.md`; this page is what each form is made of and which architecture
has it. It does **not** compose with re-injection in the same loop — hand-written staging is exactly
what starves that pass (`../../tile-programming/pipeline.md ## Authoring the overlap yourself (the climb default on the Gluon path)`),
and on an incumbent Gluon kernel the hand-written forms are the only ones on offer. Whether the loop
wants a pipeline at all is the question to settle first: a large share of surveyed production
kernels use none, and for a loop that is not latency-exposed that is the right answer
(`../../tile-programming/pipeline.md ### First: does this loop want a pipeline at all?`).

### Register-level prefetch — no LDS, no async surface, no marker

Keep a small number of operand bundles in flight in **registers** and rotate them with a
statically-unrolled loop: issue the loads for iteration `i + depth`, issue the matrix ops for `i`,
consume, rotate. No LDS traffic, no async surface, no marker, and no version gate beyond the
ordinary ones — so it is the same on gfx950 and gfx942. Pin the depth with a `gl.static_assert`
that the trip count divides by it, so the prologue / steady / epilogue split leaves no remainder
loop, and make the depth a launch parameter: it is the term that trades against the register
budget. Surveyed production reaches for this first on a dot loop, and it is the form an LDS ring
has to beat. Read VGPR and spill from the compiled kernel descriptor before deepening it: a
register double-buffer that fits without spilling has measured ahead of a deeper staged pipeline
(`../../pitfalls/negative-patterns.md ## gfx942 / CDNA3 negative-result signatures (mechanism + error text)`, item 2).

### What is actually on offer, and which architecture has it

Three categories: mechanisms that **move data**, mechanisms that only **influence scheduling**, and
mechanisms that decide **when a consumer may read what a producer wrote**. Conflating the first two
is how a hint gets budgeted as a prefetch. Leaving the third unnamed costs something different: an
A or B mechanism that is wrong costs you a measurement, while an S mechanism that is wrong can cost
you the answer — **some** of its failure modes are silent, and those are the ones that surface on a
wave count or a tile shape you did not run. Which failures are loud and which are silent is
per-mechanism, and is the point of the bullets under the S table.

**A — data movement**


| #   | mechanism                                                                 | CDNA4 gfx950                      | CDNA3 gfx942 (downgrade)                                                                                      | CDNA5 gfx1250            | RDNA3/4                 |
| --- | ------------------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------ | ----------------------- |
| A1  | async global→LDS multi-buffer (`commit_group` / `wait_group`)             | `cdna4.async_copy`, 128- or 32-bit | **✓ at 32-bit/thread, clean tiling, shared** `order=[1,0]` — all three, and measured slower than sync staging | different API, see below | **absent**              |
| A2  | per-tensor pipeline depth                                                 | over A1 or A5                     | **✓ over A5 or A1**                                                                                           | ✓                        | over A5                 |
| A3  | several independent chains at staggered depths                            | over A1                           | over A5 or A1                                                                                                 | ✓                        | over A5                 |
| A4  | sub-buffer splitting (`.index(i)` / `.slice(...)`)                        | ✓                                 | **✓ over A5**                                                                                                 | ✓                        | over A5                 |
| A5  | sync staging: `allocate_shared_memory` + `.store()` / `.load()` + barrier | ✓ (the control in the C cases)    | **✓ — the default there**                                                                                     | ✓                        | **✓ — the only option** |


**B — scheduling hints (move no data; must be measured, never assumed)**


| #   | mechanism                                                   | on gfx950                                                         | on gfx942 (downgrade)           | availability                                                     |
| --- | ----------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------- | ---------------------------------------------------------------- |
| B1  | `warp_pipeline_stage` cluster markers                       | **✓** — runnable as C5 in `scripts/pipeline_examples_cdna4.py` (`num_warps=8`, Gate 0) | **✓** — emits `s_setprio`       | 3.7.0+ (absent on 3.6.0)                                         |
| B2  | `sched_barrier` / `sched_group_barrier` / `set_prio` (iglp) | **✗ symbol absent**                                               | **✗ symbol absent**             | absent from `gl.amd.cdna3` and `.cdna4` on **all four** versions |
| B3  | `warp_specialize` producer/consumer partitioning            | not probed in this pack — compile-and-run it before relying on it | **✗** `PassManager::run failed` | symbol present in core `gl` on all four                          |


Every gfx942 cell above is from a compile-and-run probe, not from whether the symbol imports; the
gfx950 B1 cell is the C5 case. Three are worth stating plainly:

- **A1 is available on CDNA3 but only 32 bits wide** (the width table in `marker-and-version-gates.md`). An
earlier revision of this file recorded it as "does not lower" from a probe whose layout was at
fault; the correction and the controlled comparison are in that section. Read a lowering
failure here as a layout-contract question first.
- **B2 does not exist here at all.** A vendor attention-decode Gluon kernel imports `sched_barrier` /
`sched_group_barrier` / `set_prio` from `gl.amd.cdna3` inside a `try/except` and defines
**no-op stubs** on `ImportError`. Those symbols are absent on 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0,
so on these builds that production kernel is running the stubs and its iglp hints are dead
code. Do not copy the pattern expecting scheduling control.
**But read the absence correctly: "no source-level op" does not mean "no such control".** The
kernel does not have to emit these itself. B1's cluster markers already lower to `s_setprio` +
`sched_barrier` around each cluster in `make_llir` (`marker-and-version-gates.md` ### `gl.amd.warp_pipeline_stage` — the official marker path), so the
author's stage structure **survives into LLVM IR as scheduler-barrier boundaries** — and a pass
at that level can take those as region boundaries and emit the group declarations on the
kernel's behalf. So the reachable form of B2 is one layer down, not in `gl.*`: see
`../../tile-programming/llir-codesign.md ## Declaring the interleave` for the mechanism and
`## The plugin tier` for what it costs to get there. What is genuinely unavailable is writing
the hint *inline in Gluon*.
- **B3 imports but does not lower on CDNA3.** `gl.warp_specialize` is in core `gl` on every
version; a minimal two-partition kernel still fails in the pass manager on gfx942.
- **B1 works on both gfx950 and gfx942**, which a version-only table does not tell you — but it is
still a hint, not a data-movement mechanism. `../../pitfalls/platform-known-issues.md` records it as
source-available yet performance-negative on some kernels, and `../../hardware/capability-matrix.md`
adds that scheduling hints are not inherently profitable. It compiles and it reorders; whether it
pays is a measurement — and below `num_warps=8` it cannot create the inter-wave offset at all
(`marker-and-version-gates.md`, Gate 0).

**The B2 stub has a mirror image, and a probe built to catch one will pass the other.** The stub
case is *import fails → the symbol is silently a no-op*, which is why the first line of this
section insists the cells come from compile-and-run rather than from whether the symbol imports.
The mirror is *import succeeds → the symbol is present, callable, and semantically different*. The
observed instance is a serving-stack compatibility shim: a framework had replaced
`triton.next_power_of_2` process-wide with an ordinary Python function, which is fine everywhere
except inside a traced body, where what the kernel needs is a **`constexpr_function`** — same name,
same arity, same answers, and a different compile-time behaviour. The shim's fix is instructive
about the size of the problem: it builds a private copy of the `triton` module namespace for that
one kernel module, swaps the single symbol in the copy, asserts the replacement agrees with the
original across a list of boundary values, and then asserts the global is still untouched.

So the ladder this pack already carries for counting — **a grep hit is not code that exists, code
that exists is not a bin that ships, a bin that ships is not a symbol that imported** — takes one
more rung: **and a symbol that imported is not a symbol that means the same thing.** A capability
probe that stops at `hasattr` or at a bare `import` returns clean on this failure. Where it
matters for authoring is `../../tile-programming/layout-recipes.md ## Compute the layout instead of
tabulating it (constexpr_function)`, because a layout factory is exactly a host-callable function
whose whole value depends on being constexpr on the traced side. Recorded as one observed shape in
one compat file, not as a characterized frequency.

**So on gfx950 the authored-overlap surface is A1 async copy at 128 or 32 bits per thread (the C
cases), A5 as its control, A2 / A3 / A4 built out of A1, and B1 as a hint on top, ordered by S1 /
S2 / S4.** **The gfx942 downgrade is A5 sync staging as the default, A1 at 32 bits per thread only,
and B1 as a hint**; A2 / A3 / A4 are things you build *out of* A5 or A1 there, not separate
features. What CDNA4 adds is async *width* (128-bit), not the async path itself.

**S — ordering (move no data, hint nothing; they decide when a read is allowed)**

| #   | mechanism                                                                    | CDNA4 gfx950 | CDNA3 gfx942 (downgrade)        | CDNA5 gfx1250            | RDNA3/4               |
| --- | ---------------------------------------------------------------------------- | ------------ | ------------------------------- | ------------------------ | --------------------- |
| S1  | `commit_group` / `wait_group` — how many groups may still be in flight at the read | ✓            | with A1, at 32-bit              | only these two carry over | n/a — A1 absent       |
| S2  | `load_shared_relaxed` — assert the fill already retired, so the backend drops the wait it would otherwise insert ahead of the LDS read | ✓            | with A1                         | **absent** — no analogue in the renamed API | n/a — A1 absent       |
| S3  | hand-issued `s_waitcnt vmcnt(N)` through inline asm — a drain finer than one commit group | symbol present | symbol present on all four (see caveat) | symbol present | symbol present |
| S4  | `gl.barrier()` — CTA rendezvous, and with it the visibility of LDS writes to waves that did not issue them | ✓            | **✓**                           | ✓                        | **✓ — with A5 the only pairing** |

**Provenance, which differs from Table A's.** The gfx942 cells in Table A are compile-and-run probe
results. The S rows are **not**, and S3's row says so in its own cell: "the symbol is present" is an
import argument, which this reference rejects everywhere else. What the S rows have instead is a
**worked case that compiles and runs** — gfx942 for S4, gfx950 for S1/S2 via the C cases — which
establishes that the mechanism is reachable and not that any particular count was observed. If you
want the count, the C-case script prints a per-case `vmcnt` column: C1 against C3 is the one-command
way to see S2 actually remove a wait on your own build, and that number is not recorded here. Treat
every CDNA5 and RDNA cell in this table as unprobed — the same caveat the file states for re-injection
(`async-ordering.md` ### RDNA3 / RDNA4 have no async surface at all, and the CDNA5 section beside it).

**Two of the four are core, not AMD-namespaced.** S4 is defined once in 3.8.0 as
`gl.barrier(*, cluster=False)` — and **3.6.0 spelled it `thread_barrier`**, which is why the worked
examples below open with a `getattr` shim rather than calling it directly. S3's only door is
likewise core: `gl.inline_asm_elementwise`
(`../inline-asm-reference.md ## The only door: gl.inline_asm_elementwise`); it is the *mnemonic inside
the string* that is AMD, not the surface.
The conditional variant `amdg.cond_barrier` that `../../tile-programming/warp-pipeline.md` describes is
**not a fifth spelling you can call**: it is an MLIR op the compiler emits, and on the Gluon
pipeline the only thing that reaches it is a stage marker — the other emitter, the ping-pong pass,
is not in **stock** `gluon_to_ttgir` at all. "Stock" is load-bearing: the re-injection recipe in `reinjection.md` adds `add_block_pingpong` to the pipeline itself
(`scripts/gluon_swp.py`, the `plain_pp` / `plain_itt` recipes), so on *that* route there is a second
emitter and the un-written staging is precisely what makes it fire.

S1's ops are named inside A1's own row and S4's inside A5's, which is how this class stayed
invisible: each looked like part of the mechanism it usually accompanies. It is not, and the tell is
how each one fails. **Read these as four different failure modes, not one** — the class is defined by
what the mechanisms decide, not by a shared symptom:

- **S1** — a missing `commit_group` silently converts an async copy into a blocking one; a drain
count that no longer matches the ring either loses the overlap or releases the read before the
fill retired. One of those two outcomes is a performance result and the other is wrong data, and
they are the same edit — which is the reason to treat a depth change as a correctness change.
- **S2** — **silent in both directions, asymmetrically.** A missing assertion costs a performance
remark; a wrongly asserted one removes a barrier you did need and still compiles. The pairing rule
and the fallback are owned by
`../../tile-programming/pipeline.md ### Vetted double-buffer skeleton (copy, then specialize)`.
- **S3** — the trade in one line: "a `wait_group` that is wrong is usually a compile error; an
`s_waitcnt` that is wrong is a wrong answer at a shape you have not run yet"
(`../inline-asm-reference.md ## Class 4 — synchronization`). It also costs S2 its soundness
(`../../tile-programming/pipeline.md ### Draining below the group: bare s_waitcnt instead of wait_group`).
- **S4** — the odd one out, because **neither of its known misuse modes returns a wrong answer**,
but only one of them is loud: a barrier *inside* a `warp_pipeline_stage` fails conversion outright,
while one left in a loop you wanted re-injected makes that pass "skip it entirely **and in
silence**" (`../../tile-programming/pipeline.md`) — you get a correct kernel with no pipeline and no
diagnostic. So the thing to check for S4 is not the numerics, it is whether the transform you asked
for actually happened. What *is* a correctness failure is S4's **absence**, not its misuse.

**S4's absence has a decidable criterion, and it is the question to ask at every `wait_group`.**
The ring skeletons pair the two calls without saying when the pairing is optional, and surveyed
production source disagrees with itself here: two sibling kernels with the same double-buffered LDS
weight panel put `gl.barrier()` after **every** `wait_group` in one file and after **none** of them
in the other. Both ship, which settles nothing — a missing barrier is a race, and a race is not
obliged to fail every run. What settles it is the ownership question:

> `wait_group` retires **the issuing wave's own** outstanding copies. It is not a workgroup
> rendezvous and it carries no LDS-write visibility on anyone else's behalf. So the barrier after
> the wait is **mandatory whenever a wave reads an LDS region a different wave filled** — the
> normal case for a shared operand panel at `num_warps > 1` — and omissible only when each wave
> reads exactly what it itself issued, which at `num_warps == 1` is automatic.

Two riders, because this is the criterion for **one** of the two hazards. It answers
read-before-fill; the mirror-image barrier before the refill answers overwrite-before-read and is a
separate decision with a separate answer, and a single-slot ring needs both. If you cannot name
which of the two a barrier of yours is guarding, resolve that before tuning the depth — the three
concrete placements are owned by `../../tile-programming/pipeline.md ### Three shapes production
actually builds, and the barrier placement that differs between them`.

**There is no fence on this path.** `fence_async_shared` is NVIDIA-only
(`../appendix-api.md ## Spellings that do not exist on the CDNA Gluon path`), so a reader who arrives
searching for a fence finds nothing and can conclude the ordering is automatic. It is not: it is
written here as S1 **and** S4, two calls answering two different questions. Which question each one
answers is `async-ordering.md` ### Ordering the fill against the read — class S; where the barrier then goes is owned
by `../../tile-programming/pipeline.md ### Three shapes production actually builds, and the barrier
placement that differs between them`.

### Worked examples — every one compiled and run on gfx942

**These are the gfx942 downgrade examples.** The primary, gfx950 examples are the async ring cases
C1–C4 in `async-ordering.md ## The async forms — gfx950 only, and the shape differs from A5`, runnable
as `scripts/pipeline_examples_cdna4.py` (with A5 as its control and C5 as the marker case). Use the
set below when the target is CDNA3, or when you need the sync-staging (A5) control on either arch.
They are the CDNA3 set because that is the target whose surface is smallest and most often
mis-documented. All four are runnable as
`scripts/pipeline_examples_cdna3.py`, which prints each one's `ds_read` / `ds_write` /
`s_setprio` census next to a numerics check — **run it and read your own counts.** This page
quotes a census only where the count is the point being made; everywhere else the number you want
is the one your own build prints, not one carried over from someone else's box.

> **Read that script's `correct` column as a smoke test, not as a ring check.** Its own docstring
> says why: every iteration reads the **same** tile, so reading the wrong stage or overwriting a
> buffer early still sums to the right answer. The arithmetic is checked; the buffer index is not.
> `pipeline_examples_cdna4.py` uses one distinct tile per iteration, which is what makes a ring
> falsifiable — and that difference is the whole of the caveat under
> `async-ordering.md` ### The async forms — gfx950 only, and the shape differs from A5, applied here in reverse.

**On gfx950 run** `scripts/pipeline_examples_cdna4.py` **instead** (it reads the arch off the
device and says so when you are not on gfx950), which carries the async forms this set cannot reach and keeps A5 as the
control. Its case list is **A5, C1, C2, C3, C4, C5**.
C1–C4 are written out under `async-ordering.md` ### The async forms — gfx950 only, and the shape
differs from A5; C5 is not, because it belongs to a different mechanism — it is the marker path
(`marker-and-version-gates.md` ### `gl.amd.warp_pipeline_stage` — the official marker path), and it is the one case that needs
`num_warps=8`. Neither file substitutes for the other: the pair is split by generation, not by
preference.

> **C1–C5 here are that script's case labels and nothing else.** They are not the calibration chain
> C0–C4 (`../../method/profile.md ## 3.1 Required evidence`), which is a different namespace that happens to share the letter;
> and they are not mechanism IDs. Every case is built on **A1**, but they are not interchangeable
> instances of it: C1 and C2 differ only in entry point, C4 only in depth, while **C3 is A1 + S2**
> and **C5 is A1 + B1**. When you want the mechanism, cite the mechanism ID; when you want a
> runnable case, cite the script label — and do not read a script label as naming one mechanism.

> `BLK` **below covers** `[64, 16]` **while the tile these run at is** `[32, 32]`**, and that is safe
> HERE but must not be copied onto an async path.** These are sync cases, so the repetition only
> changes the op counts you will read; the numerics are unaffected. On an async copy it is fatal:
> direct-to-LDS requires each lane to make exactly **one** access of a native width, and a layout
> covering more than the tile makes two — failing with the same `failed to translate module to LLVM IR` text that a genuinely absent op produces, which is how the failure once got recorded as
> "the architecture refuses async copy". `pipeline_examples_cdna4.py` uses `[1,4],[8,8],[4,1]`,
> which covers `[32, 32]` exactly, and that difference between the two files is deliberate.
> Background: `../../pitfalls/platform-known-issues.md`.

**A5 — the base. Two LDS buffers, alternate, barrier between phases.** On CDNA3 this is the
default staging path (async exists there only at 32 bits per thread and measured slower); on gfx950
it is the control the async C cases are measured against.

```python
BLK: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [4, 1], [1, 0], [])
SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

# 3.6.0 spells it thread_barrier; 3.7.0 renamed it to barrier. Written this way throughout
# so a version failure on a 3.6 box cannot be misread as "CDNA3 cannot do any of this".
_barrier = getattr(gl, "thread_barrier", None) or gl.barrier

@gluon.jit
def a5_sync_double_buffer(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    s = gl.allocate_shared_memory(gl.float32, [2, M, N], SH)
    o = (gl.arange(0, M, layout=gl.SliceLayout(1, BLK))[:, None] * N
         + gl.arange(0, N, layout=gl.SliceLayout(0, BLK))[None, :])
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)
    s.index(0).store(gl.load(inp + o))                 # prologue fills buffer 0
    for i in range(ITERS):
        cur = i % 2
        nxt = (i + 1) % 2
        _barrier()
        if i + 1 < ITERS:
            s.index(nxt).store(gl.load(inp + o))       # stage i+1 while consuming i
        acc += s.index(cur).load(BLK)
        _barrier()
    gl.store(out + o, acc)
```

Note `i % 2` — a runtime buffer index, which one rule elsewhere calls an anti-pattern; whether it
costs anything is target-dependent and measurable
(`../../tile-programming/pipeline.md ### Hand-built buffering rules (correctness + scheduling footguns)`
has the scope of that rule). The rule is that `smem.index(k % nBuffers)` keeps the scheduler from
proving overwrite-safety, so buffer indices should be compile-time constants and `wait_group(N)`
recomputed whenever the prologue, region or unroll factor changes. It is **narrower than it reads**:
over *sync* staging the two index forms can emit identical ISA and identical register counts, and
shipped production Gluon indexes an **async** main loop with a runtime modulo, unrolling only its
wind-down and citing register allocation rather than scheduling for that. So unroll when the ISA
says it bought something, not by rule.

**A2 — two tensors at different lead distances in one loop.** This is the one the
auto-pipeliner structurally cannot do: it assigns a single stage schedule to the whole loop,
so a small tensor rides in the big one's stage whether that helps or not.

```python
@gluon.jit
def a2_per_tensor_depth(out, big, small, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    sb = gl.allocate_shared_memory(gl.float32, [2, M, N], SH)   # double-buffered
    ss = gl.allocate_shared_memory(gl.float32, [1, M, N], SH)   # single, no lead
    o = (gl.arange(0, M, layout=gl.SliceLayout(1, BLK))[:, None] * N
         + gl.arange(0, N, layout=gl.SliceLayout(0, BLK))[None, :])
    acc = gl.full([M, N], 0.0, gl.float32, layout=BLK)
    sb.index(0).store(gl.load(big + o))                # big leads by one iteration
    for i in range(ITERS):
        cur = i % 2
        nxt = (i + 1) % 2
        _barrier()
        if i + 1 < ITERS:
            sb.index(nxt).store(gl.load(big + o))
        ss.index(0).store(gl.load(small + o))          # small has no lead at all
        _barrier()
        acc += sb.index(cur).load(BLK) * ss.index(0).load(BLK)
    gl.store(out + o, acc)
```

A vendor library's block-scaled GEMM is the production form of this shape,
leading its operands by two K-iterations and its scales by one, and its docstring names that
split as the main win over the Triton version: keeping the scales in the operands' stage both
puts scale-load latency on the critical path and spends `ds_read` bandwidth on a tiny tensor.
A3 (several chains at staggered depths) is the same freedom applied more than twice —
`mla_gluon` runs a page-index chain alongside a KV chain and drains them separately.

**A4 — one buffer consumed as independent half-slices.** Each slice is a descriptor in its own
right, so the halves can be filled or drained on different schedules.

```python
HALF: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [4, 1], [1, 0], [])

@gluon.jit
def a4_subbuffer_slice(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    s = gl.allocate_shared_memory(gl.float32, [M, N], SH)
    ...
    for _i in range(ITERS):
        _barrier()
        s.store(gl.load(inp + o))
        _barrier()
        top = s.slice(0, M // 2, 0).load(HALF)          # rows [0, M/2)
        bot = s.slice(M // 2, M // 2, 0).load(HALF)     # rows [M/2, M)
```

The slice takes `(offset, length, dim)`; the consuming layout must match the sliced shape, not
the parent's — that mismatch is the usual first failure.

**B1 — cluster markers. A hint: it reorders, it stages nothing.**

```python
@gluon.jit
def b1_warp_pipeline_stage(out, inp, M: gl.constexpr, N: gl.constexpr, ITERS: gl.constexpr):
    ...
    for _i in range(ITERS):
        # NOTE: these two priority values are BACKWARDS from the production rule --
        # a real warp-pipeline kernel gives the MEMORY stage the higher value. Quoted
        # verbatim as measured on a loop with no LDS traffic; do not copy the numbers.
        with gl.amd.warp_pipeline_stage("load", priority=1):
            v = gl.load(inp + o)
        with gl.amd.warp_pipeline_stage("compute", priority=3):
            acc += v * 2.0
    gl.store(out + o, acc)
```

The census on this one shows `s_setprio` appearing while `ds_read` / `ds_write` stay at **zero** —
which is the point: no LDS traffic appears because no staging happened. It reorders; it moves
nothing. Pair it with A5 when you want both.

Two things this probe does not show, because it was built without LDS traffic and is quoted
here verbatim as measured. First, **its priority assignment is backwards from the production
rule** — the probe gives compute the higher value; a real warp-pipeline kernel gives the
**memory** stage the higher one, for the reason below. Do not copy the numbers out of this
snippet. Second, "it moves nothing" is a statement about *this* loop, not about the mechanism:
in a kernel that does stage through LDS, the same marker is what establishes the two-group
phase offset (`../../tile-programming/warp-pipeline.md`).
