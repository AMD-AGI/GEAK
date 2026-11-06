# Pipeline Layer (gfx950 baseline, gfx942 downgrade)

Read this for the pipeline layer (after LDS layout, before slicing). Goal: hide
HBM and LDS latency by staggering AC / LR / DOT so the MFMA unit never waits.

**Architecture and version basis.** Every mechanism here is written for **gfx950 (CDNA4,
MI350X/MI355X)** first, in its **upstream Triton 3.8.0** spelling. **gfx942 (CDNA3, MI300X/MI325X)**
follows as a *gfx942 downgrade* note on the mechanism it changes (sync staging instead of a wide
async ring; direct-to-LDS async only at 32 bits per lane with an `order=[1,0]` destination; no
`ds_read_b64_tr`; 64 KiB LDS / 32 banks instead of 160 KiB / 64 banks). 3.6 / 3.7 differences are
downgrade notes on the mechanism, not separate recipes. Worked examples: `scripts/pipeline_examples_cdna4.py`
is the main set, `scripts/pipeline_examples_cdna3.py` the downgrade set.

**This page is the single definition of the overlap order on the Gluon path**
(`### The order to reach for these in` below). `../gluon/pipeline-reference.md` is the API router
for the mechanisms; `scheduling-model.md`, `warp-pipeline.md` and `instruction-scheduling.md` own
the scheduling mechanisms; `../method/climb.md` and `../method/recover.md` apply the order per round.
They point here rather than restating it.

## Where the overlap comes from, and it is not the same question per tier

**The two tiers do not share an answer here, and the difference is structural rather than a
matter of degree.** On plain Triton the compiler builds the overlap and your job is to afford it.
On the Gluon path the compiler builds **none** of it — the pipeliner passes are simply not in
that lowering — so the overlap is something you author. That holds below the parity gate too: a
transcription that lost plain's pipeline repays the debt **by hand first**; re-injecting the
compiler's version is the lowest rung, kept as a diagnostic and a last resort.

| tier | who builds the overlap | your first move |
| --- | --- | --- |
| **plain Triton** | the compiler, in `make_ttgir` | afford it and verify it fired: `## Plain Triton: the pipeliner runs for you` |
| **Gluon, climbing above plain** | **you do** | walk `### The order to reach for these in`: register prefetch, then the authored ring (`## Authoring the overlap yourself (the climb default on the Gluon path)`), then the scheduling model — instruction pacing (`## Intra-wave: pacing the stream the staging created`) and, where the stage rules allow, the wave-level phase offset (`## Inter-wave: the two-group phase offset`) |
| **Gluon, still below the parity gate** | **you do**, re-authoring what plain built | repay the named `lost_pipeline` debt with the same hand-written rungs, starting from the recovered scaffold (`## Recovering the structure, then improving on it`); only if they cannot reach parity, or to *measure* the debt, re-inject plain's pipeliner (`## Reproduce plain's software pipeline on the Gluon path (the parity-recovery route)`) |

**Read re-injection as a measuring instrument and a last resort, not as a lever.** It reaches
*plain's* overlap and stops there — the measured outcome on this platform is at best parity with
plain, and on a measured kernel the net stayed negative on most toolchain versions. Its numbers are
labelled **`injected`** wherever they are recorded and are never reported as a win; it is never
applied to an **incumbent** kernel (one that was already Gluon at entry — there is no plain
pipeline to recover, so only the authored ring is available there). It is the wrong tool when the
job is to beat plain, because it has no way to express what plain could not.

**What production does, on this exact target.** On gfx950 Gluon kernels the authored ring is the
dominant idiom by a wide margin, instruction scheduling is a clear second, wave-level phase offset
is rare, and the compiler auto-pipeline **does not appear at all** — some kernels pass
`num_stages=1` to switch it off explicitly. The kernels that do run the auto-pipeline are **plain
Triton** ones, and they share no file with the authored ring. That disjointness is the useful
part: production treats the two as alternatives and picks
the tier accordingly, per M regime within one kernel family. So when the auto-pipeline is the
answer you want, the move is **to stay in plain Triton**, not to escalate and re-inject
(`../method/entry.md`, the bound-class decision of the escalation gate).

### The order to reach for these in

**This is the one place the order is defined.** GEAK's `kernel_workflow` round loop injects this
skill and owns acceptance (`verify_engineer`, Director); that decides who accepts a round, not which
rung the deep_engineer tries first. The order an agent infers from mechanism-page word counts is close
to the reverse of what production writes; nothing here changes a mechanism's gate — it changes which
gate you go and check first.

0. **First, does this loop want a pipeline at all?** A large share of surveyed production kernels
   use none of the mechanisms below, and for a loop that is not latency-exposed that is the right
   answer — recorded as a finding with the read that fired
   (`### First: does this loop want a pipeline at all?`).
1. **Register-level prefetch — the default, and the one with no page-count to match its use.**
   Keep a small number of operand bundles in flight in registers and rotate them with a
   statically-unrolled loop: issue the loads for iteration `i + depth`, issue the matrix ops for
   `i`, consume, rotate. No LDS traffic, no async surface, no marker, and no version or arch gate
   beyond the ordinary ones — it is identical on gfx950 and gfx942. Pin the depth with a
   `gl.static_assert` that the trip count divides by it, so the prologue/steady/epilogue split leaves
   no remainder loop, and make the depth a launch parameter — it is the term that trades against the
   register budget. This is what surveyed production reaches for first on a dot loop, and it is the
   form to beat before building anything below (`### Register-level prefetch (step 1)`).
2. **An authored LDS ring — when the tile is large enough and the staged block is reused enough to
   hide the transfer.** The precondition is arithmetic, not stylistic: if a staged block feeds only
   one or two tiles of matrix work, there is nothing to hide the copy behind and the ring is cost
   plus a correctness surface. Raising the reuse of the staged block is the enabling edit; the ring
   is downstream of it (`../workloads/moe.md ### Two reuse axes, and this page used to carry only
   one` is the worked instance).
   - **gfx950 (baseline): the async ring.** `gl.amd.cdna4.async_copy.buffer_load_to_shared` /
     `global_load_to_shared` (destination first) closed by `async_copy.commit_group()` and retired by
     `async_copy.wait_group(N)`, with `async_copy.load_shared_relaxed` for a read whose fill a
     `wait_group` already retired. Direct-to-LDS at **128 or 32 bits per lane**; the falsifiable
     signature is that the staging `ds_write` disappears. **Preconditions on the copy itself,
     before any scheduling question.** On `v3.8.0` the two async entry points state three
     conditions in their own docstrings, and *"lowering to LLVM will fail"* is what happens when one
     is missed — not a slower kernel, not a fallback: the offset (or pointer) layout's
     `size_per_thread * bits_per_element` must be **128 or 32**; writes to the destination must be
     **coalesced**; and a swizzled destination may only be **swizzled within a warp boundary**. All
     three follow from the destination LDS address being one register for the whole warp while the
     global address is per-thread. **A fourth condition is measured rather than documented: the
     destination's layout family is itself a gate.** A `SwizzledSharedLayout` destination lowers; a
     destination from `gl.amd.cdna4.compute_efficient_padded_shared_layout` has been seen to fail
     LLVM translation on the same kernel at the same tile sizes, with a correctly derived producer
     layout and with the synchronous control path on the same padded layout succeeding. So **build
     the ring against a swizzled destination first**, and treat "make the destination padded" as a
     later step with its own evidence (`../gluon/memory-reference.md ### Shared-layout family +
     transpose-on-read (layout dependency)`). Drain depth and barrier placement:
     `### Three shapes production actually builds, and the barrier placement that differs between them`
     and `../gluon/pipeline/async-ordering.md`.
   - **gfx942 downgrade: sync staging is the base.** `gl.allocate_shared_memory` + `.store()` /
     `.load()` + `gl.barrier()` (spelled `gl.thread_barrier` on 3.6.0) — the register round trip
     *is* the staging, so `ds_write > 0` is expected. The same `cdna4.async_copy` module does lower
     on gfx942, but **only at 32 bits per lane, with a layout whose threads tile the contiguous
     dimension exactly and a shared destination with `order=[1,0]`** — and in the controlled
     comparison it measured slower than sync staging
     (`../gluon/pipeline/authored-overlap.md`). Ring depth is also paid out of 64 KiB of LDS rather
     than 160 KiB, so a depth that fits on gfx950 can cost occupancy here. The skeleton for both is
     `### Vetted double-buffer skeleton (copy, then specialize)`.
3. **Pick the scheduling model, then realize it — `gl.amd.warp_pipeline_stage` is one of the
   answers, not the default** (layer 1.5: `scheduling-model.md ## How to choose`). The model
   schedules what rungs 1–2 built; it does not create overlap.
   - **`inter_wave` → `gl.amd.warp_pipeline_stage`.** Gate 0 is `num_warps >= 8`; below that the
     warp index is identically zero and the inter-wave phase shift silently does not exist, while
     the border marker's `sched_barrier` still lands as an intra-wave scheduling wall. Both effects
     are real; they are not the same effect, and a timing change from the second reported as the
     first is a misattribution (`warp-pipeline.md ## Gate 0: does the launch have two wave groups at all?`).
     Stage bodies must be free of waits, so the ring's `wait_group` has to fall on a stage
     boundary. 3.7.0+; Gluon-only.
   - **`compiler_interleave` / intra-wave pacing** — the empty-asm scheduling fence, a placed
     `s_nop`, and the per-compile scheduler strategy through `llvm_fn_attrs` (3.8.0)
     (`## Intra-wave: pacing the stream the staging created`, then `instruction-scheduling.md`).
     Upstream 3.8.0's stock `coexec` machine-scheduler strategy belongs here: the backend sets it
     automatically **only on gfx1250 and only at `num_warps <= 4`**; on gfx950 / gfx942 it is
     opt-in, process-wide with `TRITON_HIP_USE_COEXEC_SCHEDULER=1` or per compile with
     `llvm_fn_attrs=[["amdgpu-sched-strategy", "coexec"]]`, and whether the pinned LLVM honours it
     there is an assembly-diff question (`llvm-fn-attrs.md`). Note the two gates are disjoint:
     the stock `coexec` hook fires only at `num_warps <= 4`, the phase offset only at `num_warps >= 8`.
   - **Tail of rung 3: an authored LLIR scheduling pass** — only for a matrix+VALU region, only once
     rungs 1–3 are in place, and only under sanction
     (`## Instruction-level co-design: schedule WITHIN the structure (last)`).
4. **Lowest — re-inject plain's auto-pipeliner** (`scripts/gluon_swp.py`, `scripts/patch_reinject.py`,
   `scripts/patch_async_reinject.py`;
   `## Reproduce plain's software pipeline on the Gluon path (the parity-recovery route)`). Two
   uses, and only these:
   - **a diagnostic below the parity gate** — arm it on a scratch copy to measure how much of the
     gap is `lost_pipeline` debt, alongside the plain `num_stages=1` control that attributes the
     same debt from the other side (`../method/recover.md`);
   - **a last resort** when the hand-written rungs cannot reach the run-declared parity threshold
     (default 0.95, `scripts/parity_gate.py --threshold`). The full recipe is
     `../method/recover.md` (the last-resort re-injection section).

   Its numbers carry the label **`injected`**, are never recorded as a climb win, and do not make
   a gap closed: if the hand-written line is still below the gate, `parity_unreached` is recorded
   and carried on every later number, and a residual whose mechanism Gluon cannot express is handed
   back (`../method/recover.md`). **Never on an incumbent kernel.** It is mutually exclusive with
   rung 2 in the same loop (`### Re-injection and authored staging do not compose in one loop`).

**`num_stages` is not on this list.** It is dead on the Gluon path on all four versions (3.6.0 /
3.7.0 / 3.7.1 / 3.8.0) — no pass consumes it — so writing it, omitting it, or finding it absent in
someone else's kernel is not evidence about pipelining in either direction. It survives only as a
**budget parameter** (the buffer count in the bandwidth model) and as a **champion record** field;
carrying plain's tuned value across as a Gluon tuning knob is the mistake this sentence exists to
stop (`## Gluon: the pipeliner does not run, and that is the whole difference`,
`../gluon/pipeline/loop-knobs-and-targets.md ## Roll the loop (cut i-cache pressure)`).

### The four layers, ordered by what production reaches for

The tier table says *who builds* the overlap. This says *which layer to spend a round on*, and the
order is a **static site count over production source**, read as a priority. Say what that is,
because the column below is the kind of number that gets quoted as something stronger: a grep hit
is not code that exists, code that exists is not a bin that ships, and a bin that ships is not a
symbol that imported. Every one of those four rungs has produced a wrong count in practice, and
the rung that fails is not the one the author was watching. So `dominant` / `clear second` /
`rare` / `absent` rank **where the authoring effort
went**; none of them is a claim that the mechanism executed, and none is a measurement.

**The third of those four steps — "a bin that ships" — is the one that is mechanically checkable,
so check it instead of asserting it.** Where a host-side dispatcher selects a kernel with
`if <shape> in (...)` or `== <const>`, that predicate declares a finite constant set, and whatever
registers the compiled signatures (a manifest, a pack index, a tuning table) declares another.
**Intersect the two. An empty intersection is a dead tier**: source that exists, that every grep
counts, and that no call can reach. The check is purely static and scriptable, it costs no GPU
time, and a single dead tier is enough to reverse the sign of a headline conclusion about a whole
kernel family. Run it before
quoting a per-tier count, and before spending a round optimizing a tier you have not shown is
reachable. The same shape appears one layer down as a guarded branch whose guard constant is never
produced; the discriminator is identical, and so is the failure to notice.

With that caveat carried, the ranking is below. The **layer** numbers are the production census;
the **order step** column maps each layer onto `### The order to reach for these in`, which is
what decides the next round.

| layer | order step | mechanism | production reach | availability | where |
| --- | --- | --- | --- | --- | --- |
| **1 · static staging** | 1, then 2 | register-prefetch rotation, then the authored ring — `commit_group` / `wait_group(N)` over multi-buffered LDS (gfx942: sync staging) | **dominant**, by a wide margin | upstream 3.8, stock LLVM; async width is per arch | `### Register-level prefetch (step 1)`, `## Authoring the overlap yourself (the climb default on the Gluon path)` |
| **2 · intra-wave** | 3 (`compiler_interleave` / authored pacing) | empty-asm scheduling fence, placed `s_nop`, per-compile scheduler strategy (`llvm_fn_attrs`, opt-in `coexec`) | **clear second** | upstream 3.8, stock LLVM | `## Intra-wave: pacing the stream the staging created`, then `instruction-scheduling.md` |
| **3 · inter-wave** | 3 (`inter_wave`, Gate 0 `num_warps >= 8`) | two-group phase offset — the marker API, or hand-rolled `s_setprio` | **rare**, and mostly in the hand-rolled form | upstream 3.7+, stock LLVM | `## Inter-wave: the two-group phase offset`, then `warp-pipeline.md` |
| **4 · LLIR co-design** | tail of 3 | a scheduling pass the language cannot express | **absent** — and no environment knob is set at all | needs a plugin `.so` or a non-upstream build | `## Instruction-level co-design: schedule WITHIN the structure (last)` |
| (last) · re-inject plain's pipeliner | 4 — lowest; diagnostic / last resort, numbers `injected` | run plain's pipeliner passes over Gluon TTGIR | **absent** | upstream, via a stage hook | `## Reproduce plain's software pipeline on the Gluon path (the parity-recovery route)` |

**Row 2's name is a homonym — it is a layer here and a scheduling *model* one page over.**
`scheduling-model.md` uses `intra_wave` as the earlier name of its `compiler_interleave` model and
says so; this table uses *intra-wave* for the **layer** those mechanisms live in, which is where
`compiler_interleave` and the hand-authored fence/`s_nop`/`llvm_fn_attrs` forms are **both**
realized. Picking `compiler_interleave` at that node does not decide anything in this row, and
spending a round on this row does not commit you to that model
(`../tile-programming/scheduling-model.md ## Four paradigms, and why this target has three`).

Three things to read off it, because each one prevents a wasted round:

- **Layers 1 and 2 are the whole of what stock upstream gives you, and they are also what
  production uses almost to the exclusion of everything else.** Neither needs a plugin, a rebuild,
  an env var or a monkeypatch. Layer 2 is not even Gluon-only: its mechanisms are applied in
  `make_llir`, which the plain and Gluon lowerings share — which is why `llvm_fn_attrs` is
  documented as a DSL-neutral mechanism (`../tile-programming/llvm-fn-attrs.md`) and carries no
  Gluon precondition.
- **Layer 4 is last on evidence, not on taste, and the reason is granularity rather than
  capability.** Surveyed production source sets **zero** environment variables and reaches
  every equivalent effect in-source — inline asm, layout choice, `waves_per_eu`, per-launch
  `llvm_fn_attrs`. An env var is process-wide, and one serving process runs hundreds of
  specialized kernels whose decode and prefill variants of the same family want opposite
  occupancy targets. So it is not that both forms work and one is tidier; the global form cannot
  express the requirement at all.
- **The last row is not layer 0 of a ladder, and it is not the parity route by default.** It can
  measure, and as a last resort repair, a transcription that lost the pipeline plain had; it is
  mutually exclusive with the authored ring in the same loop, and its ceiling is plain's own
  overlap. It is in this table so its rank — lowest — is stated rather than inferred from where it
  sits on the page.

**One counter-weight, and it is measured on gfx950.** On this hardware every overlap mechanism
tried — async direct-to-LDS, warp-pipeline stages, a multi-region LLVM co-execution schedule at
zero AGPR — **lost to a plain synchronous restructuring** of the LDS tiling and MFMA layouts, and
the champion that came out of it uses neither async copy nor warp-pipeline stages. The async arm
did exactly what it was supposed to at the codegen level (the staging `ds_write`s disappeared and
VGPR extent fell) and still regressed, because the `s_waitcnt` count rose instead. Read
a pipeline regression as a layout question before reading it as a scheduling one
(`llir-codesign.md ## Route the loop first: who built the overlap`). Frequency in production tells you which mechanism
is worth learning; it does not tell you that it wins on your shape.

## Plain Triton: the pipeliner runs for you

`make_ttgir` runs the automatic software pipeliner (`add_schedule_loops(num_stages)` +
`add_pipeline`) and, arch-gated, the automatic ping-pong (`add_block_pingpong(num_stages)`).
Plain's expression of latency hiding is therefore a **budget** decision, not an authoring one:
`num_stages` buys buffers, and `BLOCK_K` trades against the same LDS, so sweep them together
(`../method/front-end.md`). Two things to check rather than assume:

- **Confirm the pipeliner fired.** It declines silently when LDS or registers do not fit, and a
  declined pipeliner looks exactly like a pipeliner that did not help. The tell is in the IR:
  multi-buffer `local_alloc` plus `ttg.memdesc_index`.
- **The ping-pong is arch-gated and the gate is not intuitive.** On `gfx950` it is enabled **only
  when async copy is also on** (async copy itself defaults on for gfx950 / gfx1250, so on a stock
  gfx950 build both are on); **gfx942 downgrade:** it is enabled unconditionally, over the
  synchronous pipeliner (async copy defaults off there). So the same source gets a different
  schedule across those two chips with no source change. Both gates have env overrides, which
  makes this probeable rather than a guess.

**What plain cannot pin, and this is the handoff criterion:** *where* the prefetch lands and how
many buffers exist. If the residual gap is about placement rather than depth, that is the
layout-shaped residual the escalation gate hands off — not a `num_stages` value you have not
tried yet.

### The hand-built schedules this generalizes (plain lineage)

Worth knowing because each primitive the automatic ping-pong now selects was introduced to solve
a concrete problem in one of these, so a symptom you hit maps onto the variant that first hit it.

| Schedule | Mechanism | Best for | Main risk |
| --- | --- | --- | --- |
| Pingpong x2 | dot sliced x2, memory interleaved | medium tiles | a fence/wait spilling into compute |
| Pingpong x4 | dot sliced x4, more alternation points | large tiles; cuts dot live range | barrier overhead, backend-motion sensitivity |
| Pingpong async | async copy isolated in its own cluster | async-copy-heavy paths | fragile to lowering and wait-placement drift |
| Pingpong chained-dot | memory prioritized, explicit wait discipline | VALU-heavy address math | overlap collapses if memory starves or compute is polluted |

Three details from those variants transfer directly to any hand-authored schedule, including the
Gluon ones below:

- **Two-cluster** deliberately uses a bare `s_barrier` at one boundary rather than a
  local-fencing barrier, precisely to keep the local-load wait out of the compute slice.
- **Four-cluster** puts three guards at every boundary — scheduler wall, barrier, and a priority
  step-down — to stop an incoming wave from overtaking resources the other is still using.
- **Chained-dot** pins `s_waitcnt lgkmcnt(0)` at the **end** of the memory cluster (so the
  backend does not infer its own waits inside compute) and puts the `s_barrier` at the
  **beginning** of the loop rather than the end — because at the end, the backend can schedule the
  loop-induction scalar ops after it and effectively move them into the compute cluster.

## Gluon: the pipeliner does not run, and that is the whole difference

Source-verified on 3.8.0 (`third_party/amd/backend/compiler.py`): `gluon_to_ttgir` runs **nine
passes**, and the pipeliner trio is not among them. It runs `add_warp_pipeline` instead. Three
consequences, and the middle one is the one most often stated wrongly:

- `num_stages` at the launch is **not** a pipelining trigger on the Gluon path — it is a dead
  knob there on 3.8.0 (and on 3.6.0 / 3.7.x). The only pass in that lowering which touches
  `tt.num_stages` is the loop unroller, and it only *writes* the attribute — onto an unroll
  epilogue, so that a pipeliner which never runs would skip it. The value does survive as a
  **budget parameter** (buffer count) for the bandwidth model
  (`in_flight = min(..., num_stages - 1)`) and as a field of the champion record, which is why it
  is not simply an error to pass it — and why it is never a tuning knob on this path.
- **"Gluon has no software pipeliner" does not follow.** The passes are absent from the
  lowering, not from `libtriton`, and they are re-injectable without a rebuild
  (`## Reproduce plain's software pipeline on the Gluon path (the parity-recovery route)`). What
  does **not** follow from re-injectability is priority: it is rung 4, the lowest
  (`### The order to reach for these in`).
- `loop_unroll_factor` **is** live on this path (the unroller is one of the nine), so
  `tl.range(..., loop_unroll_factor=N)` is a real knob here even though `num_stages` is not.
  This is version-gated: the unroller entered the Gluon pass list at 3.8.0.
  **Note where it goes — it is not the launch-level twin of `num_stages`.** `num_stages` is a
  launch keyword; `loop_unroll_factor` is a keyword on the loop construct *inside* the body, and
  no launch accepts it. And on a hand-staged ring like the ones this page teaches, the value you
  usually want is the one that **suppresses** unrolling, because duplicating a body you scheduled
  yourself moves your buffer rotation and your barriers. Both directions, and the reasoning for
  each, are in `../gluon/pipeline-reference.md ## Roll the loop (cut i-cache pressure)`.

### First: does this loop want a pipeline at all?

Ask before the layer table, not after it, because the table ranks *which* overlap to build and
never asks *whether*. For a substantial share of shipped kernels the answer is none, and that is
the right answer rather than an unfinished one: **a large fraction of surveyed production source
files carry no pipelining construct whatsoever** — no `commit_group`, no `wait_group`, no
`buffer_load_to_shared`, no `load_shared_relaxed`, no stage marker. Read that as a static
*file* count, in source where files and kernels are not one-to-one
(`../method/benchmark-hygiene.md`); it
ranks how often the shape appears in authored source, and it is not a claim about dispatch volume.
The same shape recurs at a comparable proportion in independently authored AMD source,
which is what makes it a paradigm rather than a set of unfinished kernels.

Three reads say the answer is none, and each one makes the cost of a ring **positive** while the
gain is zero:

- **The tile is register-resident.** Nothing is staged through LDS, so there is nothing for a
  second buffer to hold. Allocating one squeezes occupancy, which is the resource the rest of the
  climb is spending.
- **There is no loop, or the loop is fully unrollable at trace time.** A ring's whole value is
  crossing an iteration boundary; with the boundary gone the prologue and drain are the entire
  structure. (This is also the `static_range` case that silently produces no `scf.for` for the
  marker path — `warp-pipeline.md ## Authoring rules`.)
- **The body issues only a handful of matrix ops per iteration.** There is not enough compute in
  one trip to cover a copy, so the barrier is pure added latency and the buffer is pure added
  pressure.

**The lever that replaces it is source-level reordering into the dependency shadow**: move
independent work up so it issues while a long-latency result is still outstanding, without adding a
buffer, a barrier or a wait. It is the cheapest thing on this page and it is invisible in a
mechanism census, so a kernel that took it reads as a kernel that did nothing.

**Two cautions, because this section is easy to over-read.** It is not permission to skip the
climb — it applies to *this loop*, on the three reads above, and a kernel with a staged operand and
a real trip count is not in it. And "no pipeline" is a **finding to record**, with which of the
three reads fired, not a default to fall back to when the ring is hard to write: the two look
identical in the artifact and only one of them is a result.

## Authoring the overlap yourself (the climb default on the Gluon path)

**This is where a Gluon climb starts — rungs 1 and 2 of `### The order to reach for these in` —
and it is also how a below-parity transcription repays `lost_pipeline` first.** Above the parity
gate the job is to buy something plain never had — per-tensor pipeline depth, several independent
async chains at staggered depths, sub-buffer splitting — and none of those are expressible by
re-injecting plain's pipeliner, which by construction reaches only what plain reached. It is also
the only route for a loop with **no `tt.dot`**, where injection has nothing to anchor on, and the
only route on an incumbent kernel.

The `authored_stage` scheduling model in `scheduling-model.md` is this route seen from layer 1.5;
the Gluon-native API surface, the worked examples and the per-architecture availability are in
`../gluon/pipeline-reference.md ## Authored overlap (no compiler patch)`. This section is the
scheduling model and the correctness rules behind it. One exclusion to carry from the start: a loop
with hand-written staging cannot also be re-injected
(`### Re-injection and authored staging do not compose in one loop`).

### Register-level prefetch (step 1)

The rung to try, and to beat, before any LDS ring. Hold the next `depth` operand bundles in
registers and rotate them in a statically unrolled loop:

```text
static_assert(K_ITERS % DEPTH == 0)        # no remainder loop; DEPTH is a launch constexpr
prologue:  r[0..DEPTH-1] = load(k = 0..DEPTH-1)            # DEPTH bundles in flight
steady:    for k0 in range(0, K_ITERS, DEPTH):             # dynamic loop
               for j in static_range(DEPTH):               # unrolled: slot j is a literal
                   cur  = r[j]                             # bundle issued DEPTH iterations ago
                   r[j] = load(k0 + j + DEPTH)             # refill BEFORE the matrix op (guarded at the tail)
                   acc  = mfma(cur, ..., acc)              # gfx950: gl.amd.cdna4.mfma; gfx942: gl.amd.cdna3.mfma
```

(Pseudo-structure: in Gluon write the `DEPTH` bundles as separately named tensors carried through
the loop, since slot `j` must be a trace-time literal; the point is the issue order — refill
`i + depth`, matrix op on `i`.) It needs no LDS, no async surface, no
marker and no arch gate, so it is the same on gfx950 and gfx942. Its cost is registers:
`DEPTH × bundle` VGPRs live across the loop, which is why the depth is a launch parameter and
why it is checked against the occupancy budget (`## Budget before deepening`). The three
placement axes and the two rules that make a hoisted load actually land are
`### Prefetch has three orthogonal degrees of freedom, and an unpinned one does not land`; the
one that matters here is that **register prefetch and a multi-stage LDS ring are mutually
exclusive in one loop** — both answer "who holds the next tile".

### Stages

- **2-stage global prefetch + double buffer**: prefetch tile `k+1`'s `AC` while
  computing `DOT(k)`; `nBuffers=2`, `wait_group(1)`. Hides HBM latency
  (~400 cycles).
- **3-stage local prefetch**: also prefetch `LR(k+1)` (LDS->reg) while `DOT(k)`
  runs, so the MFMA does not wait on the same-iteration `ds_read`.

```text
2-stage:  AC(k+1) , LR(k)   , DOT(k)
3-stage:  AC(k+2) , LR(k+1) , DOT(k)
```

### Independence rule (correctness of pipelining)

`DOT(k)` must not depend on the same-slot `LR(k+1)` or `AC(k+2)`. Each buffer
must be retired (`wait_group`) before it is overwritten. Violations show as
wrong results or as the scheduler refusing to interleave.

### Hand-built buffering rules (correctness + scheduling footguns)

These rules apply when you are authoring the staging yourself. They do **not**
apply to a re-injected body, which has no hand-written staging at all. Every one of them is easy
to violate and hard to debug, and the three `wait_group` rules are the ones that produce
wrong results rather than slow ones:

- **Buffer index may need to be compile-time — scope this one before paying for it.** The
  concern is real only where the *async* hazard analysis has to prove overwrite-safety:
  with `smem.index(k % nBuffers)` the buffer lifetime is not statically visible, so the
  scheduler can decline to interleave, and the fix is to unroll by the buffer period and
  give each static sub-iteration a literal index (`smem.index(0)`, `smem.index(1)`, ...).
  **Two pieces of evidence say it is not a blanket rule.** A vendor library's shipped
  block-scaled (gfx950) GEMM indexes its async main loop with exactly `k_iter % NUM_STAGES`,
  unrolling only its wind-down and citing **register allocation** for that (removing a PHI that
  would push the dot operands out of AGPRs), not scheduling; and *gfx942 downgrade evidence*:
  over *sync* staging on gfx942 the two forms compiled to identical ISA — same `ds_read` /
  `ds_write` / `s_barrier` counts and the same VGPR count — so there was nothing to buy. (On
  gfx942 the async form is reachable but narrow — 32 bits per lane, a `BlockedLayout` whose
  threads tile the contiguous dimension, an `order=[1,0]` destination;
  `../gluon/pipeline-reference.md ## Authored overlap (no compiler patch)` carries the
  controlled experiment that corrected an earlier claim that it did not lower at all.)
  So: measure both forms on your target before unrolling for
  this reason alone —
  the unroll has its own cost (next bullet but one, and the WAR-barrier corollary below).
- **`wait_group(N)` must match the groups intentionally left in flight.**
  Recompute `N` whenever the prologue issue count, region count, or unroll
  factor changes -- never copy a value from another shape or stage count. A
  stale `N` either drains too early (kills overlap) or lets `smem.load` read a
  buffer before its async group retired (wrong results).
- **`N` counts COMMIT GROUPS, not buffers and not copies** -- this is the single
  easiest place to compute the wrong number. `commit_group()` closes one group
  containing every async copy issued since the previous `commit_group()`, and
  `wait_group(N)` blocks until at most `N` groups remain outstanding. So the
  arithmetic is `N = groups_per_stage * stages_to_keep_in_flight`, and
  `groups_per_stage` is a thing you chose, not a property of the pipeline. One
  copy per stage with one commit after it gives `groups_per_stage == 1`, which is
  the skeleton below and the case where `N` happens to equal the stage count --
  which is exactly why the rule gets mis-generalized. **Stage several tensors and
  the two numbers come apart:** three copies under a single `commit_group()` are
  one group (`N` unchanged, and the three retire together, so no tensor can be
  drained on its own schedule), while three copies each followed by their own
  `commit_group()` are three groups (`N` triples, and per-tensor lead distances
  become expressible). Per-tensor depth is bought with the commit placement, so
  decide `groups_per_stage` first and derive `N` from it.
- **`N` is a compile-time attribute, not a value** -- the reason the skeleton
  below branches instead of computing a depth. `wait_group` lowers to
  `ttg.async_wait` with the count as a **static attribute** on the op, and the
  builder takes a C++ `int`, so the argument must reduce to a Python integer
  while the kernel is being traced. A rolling drain written the natural way,
  `wait_group(min(DEPTH - 1, NUM - i - 1))`, does **not** compile inside a
  dynamic loop: Gluon's bare `range` produces an `scf.for` whose `i` is a runtime
  SSA value, so the expression yields a value the binding cannot accept, and it
  fails at trace time rather than lowering to a variable wait. Arbitrary
  expressions are fine **when every operand is trace-time constant** -- a
  `static_range` or a Python-level loop over constexpr bounds unrolls first, so
  each unrolled copy gets its own literal `N` and the rolling form is then both
  legal and the clean way to write the wind-down. **In a dynamic loop the branch
  is the mechanism, not a workaround for one**: the tail needs a different
  constant, and a branch between two literals is how you spell that.
- **`nBuffers` must equal the stage depth.** A 3-stage pipeline needs 3 buffers
  (one in flight, one consumed-current, one in transit); dropping to 2 forces
  `DOT` to wait on `AC` and silently collapses back to 2-stage.
- **Stage through shared memory the pipeline pass can see (phase-visibility).** A
  transform's correctness depends on what is visible to it **at the phase it runs**.
  Implicit/compiler-managed scratch created in a *later* phase — e.g. the shared
  scratch a `convert_layout` allocates after the warp-pipeline pass has already run
  — is **not protected by that pass's hazard analysis**, so staggered groups can
  race on it (silent wrong results / NaNs). Stage cross-stage data through
  **explicitly allocated** shared memory (`gl.allocate_shared_memory` +
  `buffer_load_to_shared`) that the pass can see, not through scratch a later pass
  introduces.

### Prefetch has three orthogonal degrees of freedom, and an unpinned one does not land

"Prefetch the operand earlier" is not one boolean, and sweeping it as one is how the lever gets
written off. It is three independent choices, and production carries them as three separate
switches rather than folding them into a depth:

1. **Which piece** — the first slice, the second slice, or the next iteration's.
2. **Which side of which synchronization boundary** — before or after a `wait_group`, before or
   after a prefix computation, before or after a specific matrix op. This is the axis most often
   collapsed by accident and the one with the largest effect, because it is a *phase* decision
   rather than a quantity decision.
3. **How deep** — the ring's lead distance, which is the `groups_per_stage` question above.

Two rules go with it, and both have bitten:

- **A hoisted load that is not pinned sinks back, and the change then measures as zero.** Nothing
  warns: the source says the load moved, the scheduler moved it back next to its consumer, and the
  only place the difference is visible is the disassembly. Pin the prefetched value with an
  empty-asm fence (`## Intra-wave: pacing the stream the staging created`) and **confirm in the
  `.s` that the `ds_read` actually moved before timing anything.** A prefetch round with no static
  read has not measured prefetching.
- **Hand-written prefetch and multi-stage double buffering are mutually exclusive in one loop**
  (rung 1 versus rung 2 of the order: pick one per loop).
  Both answer "who holds the next tile", and together they hold it twice. Pin the exclusion with
  `gl.static_assert` rather than a comment, because the failure is a register-pressure regression
  rather than an error.

Axis 2 is not monotone, so **sweep it rather than bisecting it** — a bisection assumes a shape a
phase effect does not have.

### Vetted double-buffer skeleton (copy, then specialize)

A correct 2-stage double buffer (`nBuffers` == stage depth == 2), **gfx950 baseline (async
ring)**. The point is the **retire ordering**: the prologue issues `k=0`; each iteration
prefetches the *other* buffer, keeps exactly one async group in flight (`wait_group(1)`), and DOTs
the current buffer; the last iteration drains (`wait_group(0)`). Copy this instead
of re-deriving the staging per kernel. The runnable, numerics-checked versions (one distinct tile
per iteration, so a wrong buffer index fails) are `scripts/pipeline_examples_cdna4.py` C1–C4.

```python
_acp = gl.amd.cdna4.async_copy                                     # a module, not a callable
s = gl.allocate_shared_memory(dtype, [2, *tile], shared_layout)  # nBuffers == 2; swizzled destination first

_acp.buffer_load_to_shared(s.index(0), base_ptr, off(0))         # prologue: issue k=0
_acp.commit_group()

for i in range(NUM):
    cur = i % 2
    nxt = (i + 1) % 2
    if i + 1 < NUM:
        _acp.buffer_load_to_shared(s.index(nxt), base_ptr, off(i + 1))  # prefetch k+1
        _acp.commit_group()
        _acp.wait_group(1)        # retire all but the one in-flight prefetch
    else:
        _acp.wait_group(0)        # drain on the last iter
    a_k = _acp.load_shared_relaxed(s.index(cur), dot_operand_layout)  # LDS -> reg, THIS iter only
    acc = gl.amd.cdna4.mfma(a_op, a_k, acc)       # DOT(k): independent of s.index(nxt)
```

Specialize: for 3 stages use `nBuffers == 3` + `wait_group(2)`, and recompute
`wait_group(N)` whenever the prologue / region / unroll changes
(`### Hand-built buffering rules`). The `i % 2` form here is the readable correctness
template; whether the literal-index unroll also schedules better is target-dependent and
measurable — see that section before assuming it does. `off(k)` is the per-iteration global
offset; `_acp` is the cdna4 async-copy module.

**gfx942 downgrade — the sync-staged double buffer.** No wide direct-to-LDS path, so the ring is
built from the register round trip plus barriers (`scripts/pipeline_examples_cdna3.py` A5 is the
runnable form):

```python
s = gl.allocate_shared_memory(dtype, [2, *tile], shared_layout)  # e.g. SwizzledSharedLayout(..., order=[1, 0])
s.index(0).store(gl.load(ptrs(0)))                               # prologue fills buffer 0
for i in range(NUM):
    cur = i % 2
    nxt = (i + 1) % 2
    gl.barrier()                                  # buffer `cur` written by every wave (3.6.0: gl.thread_barrier)
    if i + 1 < NUM:
        s.index(nxt).store(gl.load(ptrs(i + 1)))  # stage k+1 while consuming k
    a_k = s.index(cur).load(dot_operand_layout)
    acc = gl.amd.cdna3.mfma(a_op, a_k, acc)
    gl.barrier()                                  # nobody refills `cur` while a wave still reads it
```

There is no group to wait on, so no `wait_group(N)` arithmetic and no `load_shared_relaxed`;
the two barriers are the whole ordering contract, and they answer the two hazards of
`### Three shapes production actually builds, and the barrier placement that differs between them`
(read-before-fill, overwrite-before-read). The 32-bit async variant exists on gfx942 but measured
slower than this form; keep it as an A/B arm, not the default.

**`load_shared_relaxed` is a module function, not a descriptor method**, and it is written here
rather than `s.index(cur).load(...)` for one reason: every path into this read has already
executed a `wait_group` that retired the group which filled `s.index(cur)`, so the wait the
backend would otherwise emit ahead of the LDS read is redundant. The call asserts exactly that:
it sets one attribute, `ttg.amdg.syncedViaAsyncWait`, on the load and changes nothing else.
**The mechanism is an override, which is why it can be wrong.** The pass that would normally
annotate this checks for the attribute first and leaves it alone if present — so you are
substituting your own claim for its token-chasing. Two things the attribute buys: the membar
analysis skips the barrier it would otherwise insert between the async copy and this load, and
the load joins an alias scope that lets LLVM reorder more freely. And the failure mode is
silent in both directions: a **missing** attribute only costs a remark about performance, while
a **wrongly asserted** one removes a barrier you did need and still compiles. **The assertion is
yours to keep true when you specialize.** If you restructure the waits — move the drain, merge
the branches, deepen the ring — and the retire no longer precedes this read, the relaxed form
becomes a race that still compiles and usually still passes a small shape. Fall back to
`s.index(cur).load(dot_operand_layout)` whenever the pairing is not obvious by inspection; it is
the safe form, not the slow one (`../gluon/pipeline-reference.md` ## Getting `wait_group` right).

### Three shapes production actually builds, and the barrier placement that differs between them

The skeleton above is the two-buffer case where every number coincides. These are the shapes that
appear when they stop coinciding, taken from gfx950 kernels on Triton 3.8. Read them for the
*structure*: they are structural observations with no timing attached, so none of this says
one shape is faster.

**1. Rolling FIFO with a per-slot wait depth.** Named slots rather than modular indexing
(`ab0`/`ab1`/`ab2`, plus `ab3` when the slice count is 4), one commit per slot in the prologue, and
then the part worth copying: the steady state waits at a **single** depth while the drain tightens
**per slot**. The idiom is `wait_group(SLICES - 2)` in steady state and `wait_group(SLICES - 1 - k)`
for slot `k` as the tail winds down — which is what a constexpr wait table looks like in practice,
and it is legal because every operand is trace-time constant. The steady-state refill for slot 0 is
also the one place a second tensor is folded into the same commit, so `groups_per_stage` is not
uniform across slots.

**2. Asymmetric A/B depth, with the depth chosen from a host-side table.** Two independent ring
depths (`A_STAGES`, `B_STAGES`) over modular indexing, so the operand that needs more lead gets it
without deepening the other. The prologue branches on whether the depths are equal: equal depths
issue one commit per slot, unequal depths issue **two** commits to cover the difference. The
epilogue then drains with a wait that tightens against the loop counter rather than a fixed literal.
The depths come from a per-M lookup table on the host, not from autotuning — the shape is selected,
not searched.

**3. Single slot, drained fully, with scheduling layered on top.** One buffer per operand,
`wait_group(0)` every iteration, and the overlap bought back by the layers stacked on top of the ring rather than by
depth: wave priority around the MFMA block and a scheduling fence pinning the accumulator. Worth
knowing because it shows the ring is not always where the overlap comes from
(`instruction-scheduling.md`).

**The barrier goes in two different places in shapes 1 and 2, and they are not interchangeable.**
`wait_group` retires async groups and provides **no CTA synchronization** — that is explicit in the
op's own contract — so a `gl.barrier()` is a separate decision answering a separate question:

- **Shape 1 puts the barrier after the wait and before the LDS read.** The hazard is
  read-before-fill: the wait says the copy retired, the barrier says every wave in the CTA has
  reached that point, and only then is the buffer safe to read.
- **Shape 2 puts the barrier after the LDS read and before the refill write.** The hazard is
  overwrite-before-read: the slot is about to be re-filled, and the barrier is what guarantees no
  wave is still reading the old contents.

**"Not interchangeable" is not "mutually exclusive", and shape 3 is where that bites.** With one
slot per operand, the same iteration both reads the buffer and re-fills it, so *both* hazards are
live in one pass of the loop and a body can need **both** barriers — one after the wait and before
the read, one after the read and before the refill. A surveyed production kernel is built that way.
So do not read the two bullets as a choice: they are two questions, each answered independently,
and a ring deep enough to separate the read from the refill is what makes the second one
unnecessary rather than wrong. Count how many iterations separate a slot's fill from its read in
your own body; when the answer is zero, expect to pay for both.

Both are correct for their own hazard and wrong for the other one, so copy the placement together
with the ring rather than separately. If you cannot say which hazard your barrier is guarding, that
is the thing to resolve before tuning the depth.

### Draining below the group: bare s_waitcnt instead of wait_group

`wait_group(N)` retires whole groups, so the finest drain it can express is one
`commit_group()`'s worth of copies. When what you want is to release after a specific number of
*individual* loads — `vmcnt(8)` then `vmcnt(4)` then `vmcnt(0)` across one stage — there is no
group boundary to name and the abstraction has nothing finer. The only way below that floor is to
hand-issue `s_waitcnt` through inline asm
(`../gluon/inline-asm-reference.md ## Class 4 — synchronization`).

**On 3.8.0 / CDNA3 / CDNA4 the reason is stronger than granularity, and it changes what this
section is for: `wait_group(N)` does not emit an `s_waitcnt` at all.** `useAsyncMarks()` is true
for both generations, so `UpdateAsyncWaitCount` skips its instruction-counting block entirely and
`ttg.async_wait` lowers straight to `s_wait_asyncmark N` — your `N` passed through unclamped as a
*commit-group* count. The actual `vmcnt` is then LLVM's to derive from the asyncmark annotations,
**and it may derive one that is too loose, with no diagnostic anywhere**. The upstream gfx950
attention tutorial documents exactly that outcome on its own kernel: a `ds_read` issued ahead of
the async copy it depends on, which is a **correctness** failure, not a slow one. The negative
evidence is direct — `git grep "SWaitcntOp::create" v3.8.0 -- third_party/amd` returns nothing,
while 3.7.0 emitted `s_waitcnt vmcnt(min(63, numInst))` from that same path.

So hand-issued `s_waitcnt` is not merely a finer instrument than `wait_group` here — **it is the
only construct on the Gluon surface that lets you name that constant yourself.** Two corollaries
worth carrying before you spend a round: a drain depth that reads wrong is not evidence your
`groups_per_stage` arithmetic is off, because on this target that arithmetic is not what picks the
number; and `UpdateAsyncWaitCount`'s own
`numInstructions = max(1, numRegistersPerThread / ptrContig)` formula belongs to the pass that is
being skipped — usable as a bookkeeping estimate, wrong as a claim about what the compiler will do.
(Check this against your own build before relying on it: it is a per-generation branch, and a
3.7-era pin behaves the other way.)

It can be **placed**, which is the part most people assume is impossible. An inline-asm block
with `is_pure=False` declares generic read and write memory effects **to the compiler's effect
system**, so the Triton-level passes will not reorder it across the async copies or the LDS reads
around it. **That is a statement about passes, not about the machine**: no wait is issued and no
visibility is established, so the block does not replace the `wait_group` or the `gl.barrier()` it
sits next to (`../gluon/inline-asm-reference.md ## Class 2 — scheduling control`). With
`is_pure=True` it has no effects and no operand anyone reads, and it is deleted before it reaches
the backend.

**What you give up is not "compiler checking" in the abstract — it is one specific piece of
evidence, and losing it usually costs more than the finer drain buys.** `ttg.async_wait` is what
the Triton-level passes read to prove a buffer's fill has retired. Hand-issued asm is opaque to
them, so two things follow:

- `load_shared_relaxed` stops being backed by anything. It is not a check you can fail — it is
  `shared_load` with one attribute written onto the result, `ttg.amdg.syncedViaAsyncWait`, set to
  `True` unconditionally by the builtin itself
  (`../gluon/pipeline-reference.md` ## Getting `wait_group` right). The pass that would
  otherwise *derive* that attribute skips any load that already carries it, and the membar filter
  then drops LDS barriers on the strength of it. So the failure mode is not "you broke a
  precondition the compiler was relying on" — it is **the compiler believes you, and it never
  checks**. That is silent in both directions: omit the relaxed read and you lose only a
  performance remark; assert it wrongly — here, on a fill whose retirement only a hand-issued
  `s_waitcnt` establishes, which no pass can see inside — and a barrier you actually needed is
  removed, and it still compiles and still passes a small shape.
- Fall back to the plain read and the backend supplies the wait itself — it tracks the
  direct-to-LDS fill as a dependence of the `ds_read` and inserts a conservative `vmcnt` ahead of
  it. You then pay for your staged drain **and** the drain you were trying to avoid, at the
  granularity you were trying to avoid.

So the precondition is narrow enough to state as a default: **bare `s_waitcnt` is a pacing lever
before it is a readiness lever.** Use it to cap how much VMEM is in flight — a memory-pipe shaping
decision, at a point where no consumer is waiting on the data yet — and reach for `wait_group` at
every point where a consumer actually reads the buffer. For readiness it is not that `wait_group`
is merely the safer spelling; it is the only one the rest of the stack can see.

**The named exception, because production takes it and you will meet the shape.** The staged drain
this section exists for — `vmcnt(8)`, then `(4)`, then `(0)`, each tier followed immediately by the
consumption it released — *is* a readiness use, and roughly fifteen surveyed files are built that
way. Taking it means owning three things the default hands to the compiler:

1. **Every CTA-level synchronization is now yours to place.** Hand-issued asm does not carry
   `MemWaitOpTrait`, so the membar pass will not follow it with an `s_barrier` the way it would
   after a `wait_group` (`warp-pipeline.md ### Before reading a "no" as a defect`). This is a
   second semantic difference from `wait_group`, independent of the wait value, and it is the one
   that gets missed.
2. **The first tier should sit in front of an ordinary `.load()`, not a relaxed one.** The
   surveyed files do this without exception: the leading drain is placed where a plain read will
   attract the barrier the analysis still inserts, and only the later tiers run relaxed. That buys
   back one checked synchronization at the point where the schedule is least settled.
3. **The premise is not established, so the verification is not optional.** The claim underneath a
   per-wave `vmcnt` is that it proves *this wave's* copies landed — and on the surveyed kernels the
   copy-side and read-side warp partitions were hand-checked and **do not coincide**, with no way
   to rule out the cases that would make it unsound without compiling. Treat this as a form
   production ships at a risk it has not written down, not as a recommendation. Verify by
   subtraction below, then race-test; a passing smoke test says nothing here.

**Verify by subtraction, in the disassembly**: your `s_waitcnt` must appear *and* no
compiler-inserted `vmcnt` wait may sit between it and the consuming access. A second wait means
the analysis did not believe you and the technique bought nothing — which is the common outcome,
and it is visible before you ever time it. Then race-test
(`../method/benchmark-hygiene.md ## Determinism race-test (async / barrier / pipeline / layout changes)`),
because the failure mode here is a read that is correct on every launch you happened to run.

### Epilogue: interleave stores, do not burst

The prologue/epilogue are part of the schedule, not free drain code:

- Convert the accumulator into an **explicit blocked store layout**
  (`convert_layout(acc.to(out_dtype), gStoreLayoutC)`) before `buffer_store` /
  `gl.store`; an implicit per-thread store layout tends to emit uncoalesced or
  oddly-strided writes.
- **Interleave the stores with the final `DOT` regions** so each store rides an
  MFMA-cycle gap. Bursting all stores after `wait_group(0)` clusters the write
  traffic into a tail the hardware cannot hide and can undo hot-loop wins.

## Budget before deepening

```text
stall_hbm = max(0, latency_hbm_to_LDS - work_after_AC)
stall_lds = max(0, latency_LDS_to_reg - work_after_LR)
effective_pipeline_depth = min(num_stages - 1,
                               floor(32KiB / (active_waves * data_per_request)))
```

Gate before adding a stage (all must hold): (a) the async copy passed its **smoke
test** in isolation (`memory-path.md ## Async copy: smoke-test before wiring (mandatory)`);
(b) `DOT(k)` is independent of the same-slot `LR(k+1)` / `AC(k+2)`
(`### Independence rule`); (c) the **occupancy budget** shows pipeline depth is the
bound -- not LDS/occupancy (`slicing.md ## Occupancy budget (P8)`): a deeper
pipeline costs LDS buffers + prefetch registers, which can drop waves/CU and
regress. **If waves are already VGPR/LDS-capped, software prefetch (register OR LDS
double-buffer) regresses** because it adds the wave-capping resource — raise
occupancy first. This is the prior for occupancy-latency-hidden access patterns
(e.g. gather / decode where the gathered data is L2-resident and its latency is
already hidden by occupancy): check `R_total` / waves-per-CU before prefetching.
This is one face of the **overlap / occupancy / ILP tri-lemma** (overlap costs
VGPR, occupancy needs few VGPR, deep-unroll ILP costs LDS — pick two); predict the
post-change waves/CU before deepening (`slicing.md ## Occupancy budget (P8)`).

Then deepen only if `stall_reduction > extra_cost`. Extra cost = extra LDS stages
(capacity), prefetch registers (`R_prefetch` in the budget), prologue/epilogue
drain, and `wait_group` complexity. A 3rd stage that overflows LDS or pushes
`R_total` past 512 will regress (this is the over-unroll spill lesson — see
`slicing.md`).

**Unroll has a shallow sweet spot for hiding a fixed-latency hazard.** A 2-block
unroll lets block B's MFMA fill block A's MFMA-write -> VALU-read gap (`s_nop`,
`../hardware/planning-constants.md ## Extended planning (attention / fused kernels)`), so it is usually the
win. But **unrolling further does not keep helping**: the compiler does not reliably
exploit the extra independent blocks to hide that hazard, and the larger loop body
adds scheduling stalls (measured: a 4-block unroll *raised* `s_nop`/iter and
regressed vs 2-block). Treat ">2× unroll" as a tri-lemma cost (more LDS buffers, more
VGPR) that must beat the no-extra-unroll baseline, not a free ILP knob.

Corollary: **double-buffering purely to DROP a WAR barrier is not free either.** *If* your
target turns out to need a compile-time buffer index (`### Hand-built buffering rules` —
measure, do not assume), alternating buffers to remove the write-after-read barrier forces
unrolling by the buffer period,
which doubles the live compute state of the unrolled body and can SPILL even with **no
explicit prefetch**. Budget the post-unroll `R_total` before assuming the saved barrier
is a net win; on a VGPR-capped (1-wave) kernel it usually is not.

## Intra-wave: pacing the stream the staging created

**Layer 2, and the second-most-reached-for mechanism in production.** It
decides *where in the instruction stream* the overlap the ring created actually lands. Read it
after a structure exists: a lever from this layer applied to a loop with nothing to interleave
measures as noise, and that is the most common way it gets written off.

All of it is reachable on upstream 3.8 with stock LLVM. The mechanisms are applied in `make_llir`,
which the plain and Gluon lowerings share, so none of it needs a re-injection shim, a plugin or a
rebuild. **The mechanisms themselves — the empty-asm fence, placed `s_nop`, the `llvm_fn_attrs`
strategy (and the opt-in `coexec` strategy), hand-rolled `s_setprio`, and the declarative hints
that are *not* reachable — are stated once, in `instruction-scheduling.md ## What production
actually reaches for`**; the inline-asm API surface is `../gluon/inline-asm-reference.md`. This
section is the routing decision, plus the one criterion that decides whether layer 4 can be
skipped outright.

One routing fact belongs here because it couples this layer to the next: hand-rolled `s_setprio`
belongs to two layers at once, and it exists beside a checked marker API because the marker
refuses to convert a stage containing an async wait — a body with a ring cannot always be cut into
legal stages, while inline asm is subject to none of those rules. Production splits exactly along
that line
(`warp-pipeline.md ### The no-wait-inside-a-stage rule is what makes layering this over an authored ring hard`).

### What the hand-authored path can and cannot express

This is the criterion for whether layer 4 is needed at all, and it partitions by **region type,
not by kernel**. Ask one question of each region in the loop: *does it need two instruction
classes issued from the same wave?*

- **A matrix + memory region does not.** Its two classes come from two different waves, so the
  overlap is a structural property that the staging and the phase offset already deliver, and the
  hand-authored path expresses it completely. A two-stage GEMM on this target reaches near-full
  per-SIMD matrix efficiency with no plugin, no env var and no compiler co-design of any kind.
  **There is nothing for layer 4 to add here**, and a round spent there is a round not spent on
  the layout.
- **A matrix + VALU region does.** Attention's compute clusters are the case: the softmax's
  vector math has to issue from the same wave as the matrix op, in the shadow of a *particular*
  matrix instruction, in a *particular* form (packed or scalar). The hand-authored path can say
  which stages exist and what each one issues; it cannot say which matrix instruction's shadow a
  given vector op lands in, or forbid the packed form that cannot co-issue at all. That last
  placement is what a region-classifying scheduler declares, and it is the only thing on this
  page for which layer 4 has a real argument.

So the rule is short: **escalate to layer 4 only for a matrix+VALU region, and only once layers
1–3 are in place.** Two cheaper exits first — a region mixing all three instruction categories is
skipped silently by those tools anyway
(`llir-codesign.md ## Applicability gate: shapes the tool does not model`), and a loop that lands
at one resident wave per SIMD has already failed the co-execution premise, so the budget is not
its binding axis (`llir-codesign.md ## Attention: the co-execution budget`).

## Inter-wave: the two-group phase offset

**Layer 3 (rung 3, `inter_wave`).** Split the workgroup's waves into two groups and hold them
permanently a stage out of phase, so one group's compute covers the other's memory. It is the only
layer on this page plain Triton cannot express at all, and the mechanism — Gate 0
(`num_warps >= 8`), the candidacy gates, the priority rule, what the conversion emits at each
boundary, the `S-1 -> S+1` hazard window, the authoring rules — is `warp-pipeline.md`. Two facts
belong here instead, where the *ordering* is decided rather than where the mechanism is explained:

- **It ranks after the ring because it has a measured opportunity cost against it** — on gfx950
  the four-wave deep-ring build led the eight-wave two-group build at every K tested, a verdict
  that reversed across toolchain pins. The phase offset spends the register headroom and residency
  the ring was already using, so it has to beat that use rather than merely work; the measurement
  and how to re-run it are in
  `warp-pipeline.md ### The fifth question: is this better than spending the same resources on the ring?`.
- **Its two entry points are not interchangeable, and layer 1's shape picks between them.** The
  marker API is checked but refuses an async wait inside a stage; the hand-rolled `s_setprio` form
  is unchecked and works inside any body. A kernel whose ring cannot be cut so that every
  `wait_group` falls on a stage boundary has only the second route — with the corresponding loss
  of everything the pass would have verified.

## Instruction-level co-design: schedule WITHIN the structure (last)

**Layer 4 — the tail of rung 3 — and the gates below are why it is last.** Zero adoption across
surveyed production source, and the criterion for reaching it is region type rather than kernel
family (`## What the hand-authored path can and cannot express`). The co-design tier itself — what
upstream 3.8.0 reaches, sanction, toolchain identity, IR verification — is canonical in
`compiler-contract.md` and `llvm-codesign-handbook.md`; the schedule models below the DSL are
`llir-codesign.md`. This section is only the gate that decides whether you go there.

**Applicability gate -- all three must hold, or skip this section entirely:**

1. **the structure the tool in front of you models.** Two generations exist and they are not the
   same capability: an **in-tree, GEMM-only** scheduler assumes a pure MFMA -> MFMA accumulator
   chain and **asserts** (invalid IR) on VALU-between-matmul kernels, so default-skip it for
   attention-shaped ops; a **region-classifying** scheduler routes each region by content and rolls
   back what it cannot model, so for it attention is a *target*. "No portable knob for
   VALU-between-matmul" is only true of the first. How to tell them apart:
   `llir-codesign.md ## Region routing: the model is a property of the region`.
2. you are on a build that **has** the scheduling you need, or can load a pass as a plugin.
   Check the stock answer first: upstream 3.8.0 ships a `coexec` machine-scheduler strategy for the
   matrix-plus-VALU region class. The backend sets it automatically only on gfx1250 at
   `num_warps <= 4`; on gfx950 / gfx942 opt in with `TRITON_HIP_USE_COEXEC_SCHEDULER=1`
   (process-wide) or `llvm_fn_attrs=[["amdgpu-sched-strategy", "coexec"]]` (per compile), and
   accept it only on an assembly diff (`llvm-fn-attrs.md ## Verifying it took effect — the
   assembly diff is the only signal`). Env-variable names for an *in-tree* LLIR scheduler toggle
   are **fork-only — absent in upstream 3.8.0 and inert if set** (`non-upstream-reserve.md ## 6.
   Environment-variable index`). A plugin-loaded pass you author has its own build states
   (`llir-codesign.md ## The plugin tier`);
3. the overlap itself already exists — authored (rungs 1–2), or, as a labelled last resort,
   re-injected — so the residual really is instruction interleave rather than missing overlap.
   This tier schedules WITHIN a structure; it does not create one.

Two more gates that are easy to miss and both read as "the tool does nothing":

- **Who built the overlap.** A re-injected loop carries no stage markers, so it can only reach the
  throughput model -- and it arrives in *plain's* loop shape, which is not the shape that model's
  inferred regions want. Confirm the regions formed by reading the IR before budgeting on it
  (`llir-codesign.md ## Route the loop first: who built the overlap`).
- **Whether your matrix shape is in the tool's cost model.** A shape it cannot price yields a zero
  cost and the region is **skipped silently**; and adding a shape row does not finish a port,
  because the window model needs recalibrating from that target's own co-issue counts
  (`llir-codesign.md ## Applicability gate: shapes the tool does not model`).

The hand-authored AC/LR/DOT interleave above is the portable answer when any of these fails.

### The toggle and its target interleave

> For the **cadence arithmetic** as a derivation rule (MFMA per global load, the shared LDS
> read/write balance), the region-formation invariant that makes it dependency-safe, and the
> co-execution budget for VALU-between-matmul regions, see `llir-codesign.md`. For **authoring a
> pass from scratch** — the load path, extension point and transaction skeleton — see
> `llvm-codesign-handbook.md ## Out-of-tree pass plugin skeleton`. This section is the
> toggle-and-target summary.

The author writes independent AC/LR/DOT; the compiler interleaves at throughput
rates only once a scheduler is actually doing that work — which upstream does not do for you
automatically in the matrix-plus-memory case, so this is a structural target to author against
rather than a knob to set (`compiler-contract.md ## What upstream 3.8.0 actually gives you`).
Target interleave (gfx950 planning values; on gfx942 re-derive the cadence from that arch's MFMA
cycle counts in `../hardware/isa-mechanisms.md` rather than porting these):

- FP16: ~4x 16-cyc MFMA per `buffer_load` (64-cyc issue spacing), ~4x MFMA per
  `ds_read_b128`.
- BF8: ~2x 32-cyc MFMA per mem op (unless `BLOCK_K` is doubled).

A throughput-pairing GEMM scheduler is GEMM-only (pure MFMA -> MFMA accumulator chain); on
attention / VALU-between-matmul kernels such a pass **asserts** (invalid IR) rather than merely
regressing, so it is not the tool for those. That is a property of that kind of pass, not of the
problem: a region-classifying scheduler handles the same kernels through a second, co-execution
model, and the two are told apart in `llir-codesign.md`. So the ladder for VALU-between-matmul is:
check the stock `coexec` strategy — opt-in on gfx950 / gfx942, see gate 2 above
(`llir-codesign.md ## Attention: the co-execution budget`) ->
interleave AC/LR/DOT manually (above) -> the portable per-compile scheduler attribute
(`instruction-scheduling.md`) -> a plugin-loaded region-classifying pass you author
(`llir-codesign.md ## The plugin tier`) -> authoring the
policy yourself under sanction (`compiler-contract.md ## Scenario B: sanctioned compiler co-design`). IR
verification lives in `compiler-contract.md`.

## Recovering the structure, then improving on it

**This is the default parity-recovery path — the hand-written repayment of `lost_pipeline` — and
what it produces is an AUTHORED body.** The scaffold below emits explicit `wait_group(N)` staging
(gfx950) or barrier-ordered sync staging (gfx942 downgrade), which is the shape the re-injected
pipeliner refuses to work with (`### Re-injection and authored staging do not compose in one loop`).
So the two are not steps 1 and 2 of one procedure: per loop you repay the debt **by hand first**,
walking rungs 1–3 of `### The order to reach for these in`, and fall back to injection (rung 4)
only to measure the debt or when the hand-written body cannot reach the parity threshold. The
faithful layouts-only anchor stays the **attribution baseline** (built first, in transcribe); the
pipeline layer then reproduces plain's structure and improves on it:

1. **Reproduce.** The post-pipeliner plain `.ttgir` physically contains the double
   buffer (`ttg.local_alloc` multi-buffer + `async_copy`/`commit`/`wait` on gfx950; multi-buffer
   `local_alloc` with sync `local_store`/`local_load` on gfx942), and Gluon
   can express all of it, so `scripts/recover_gluon.py --with-pipeline` (or `dump_ir.sh
   --emit-gluon pipeline`) — gluon pack — emits a prologue/loop/epilogue scaffold with the recovered
   `nBuffers` / `wait_group(N)` / mask shape and the recovered layouts wired in.
   Kernel-specific addressing (base ptrs / offsets / mask guard) is left as a `TODO`
   skeleton placeholder (intentional — you fill it from the algorithm skeleton, not the
   pipeline structure). This recovers
   the dominant lost-pipeline gap (`../method/transcribe.md`, re-profile + recalibrate after the
   anchor) and gets the Gluon line back near plain quickly. Record the recovered `num_stages` as a
   budget / champion-record field only — it drives nothing on this path.
2. **Improve.** Then go beyond what plain did: deeper buffering (3-stage local
   prefetch), operand/`LR` prefetch, manual `AC`/`LR`/`DOT` interleave, and
   ping-pong **only where the data dependency allows** (an online-softmax recurrence
   blocks symmetric ping-pong, `../workloads/attention.md`). Attribute each gain
   separately and guard every register-buffered step with the **tri-lemma /
   occupancy-cliff** check (`slicing.md ## Occupancy budget (P8)`): predict the
   post-change waves/CU and verify against the no-overlap baseline before keeping it.

Attribution note: keep the faithful layouts-only anchor as the baseline so a
recovered+improved pipeline's gain is attributable; do not fold the pipeline into
the transcription step. Whether the explicit stages actually *interleave* (vs MFMA /
`ds_read` clumping) still depends on the build-specific LLIR scheduler (a post-TTGIR
concern, not recovered) — `compiler-contract.md`.

## Reproduce plain's software pipeline on the Gluon path (the parity-recovery route)

**Rung 4 — the lowest — of `### The order to reach for these in`.** The heading keeps its old name
because other pages cite it; read "parity-recovery route" as *the last-resort* parity route. The
default way to repay a `lost_pipeline` debt is by hand
(`## Recovering the structure, then improving on it`). This section is for exactly two uses:

- **Diagnostic, below the parity gate.** Arm it on a scratch copy of the transcription to measure
  how much of the remaining gap is `lost_pipeline` debt — the complement of the plain
  `num_stages=1` control, which attributes the same debt from the plain side. The armed number is an
  attribution reading, labelled **`injected`**, and is never the anchor or the champion line.
- **Last resort**, when the hand-written rungs cannot reach the run-declared parity threshold
  (default 0.95). Even then the result is recorded as **`injected`**, never as a climb win; if the
  line is still short, `parity_unreached` is recorded and carried, and an inexpressible residual is
  handed back rather than chased (`../method/recover.md`).

**Never on an incumbent kernel** — a kernel that was already Gluon at entry has no plain pipeline
to recover, so the authored ring is the only overlap available to it. And read the tier table
before spending a round here: the ceiling is plain's own overlap, so above the parity gate this has
nothing to offer. The full operational recipe (the three conditions, tail pairing, the cache trap,
the IR landing criteria) is `../method/recover.md` (the last-resort re-injection section); the
measured per-kernel, per-version numbers live in one versioned table in the Gluon pack's pipeline
chapters (`../gluon/pipeline/reinjection.md ## Re-injecting plain's pipeliner — the measured recipe`).
This section is the gate and the mechanism.

### Re-injection and authored staging do not compose in one loop

Hand-written staging is exactly what starves the pipeliner (it collects `local_load`s whose source
is a loop-carried `BlockArgument`; hand staging reaches LDS through `memdesc_index` instead), so a
kernel moving to re-injection must have its staging **removed** first, and partial removal does not
count — one `ttg.barrier` left in the loop makes the pass skip it entirely and in silence. The
refusal you get for keeping it names the op, not the cause
(`'ttg.local_alloc' op pipeliner doesn't know how to predicate this op`); read it as "the hand
staging is still there" rather than as a language limit, which one transcription concluded and had
to retract.

**The two halves go together, and even together they do not always win.** De-staging alone is a
large regression, the injection has to earn that back before it earns anything, and on a measured
kernel the net stayed NEGATIVE on most toolchain versions while the injection itself was worth a
consistent speedup on all of them. So:

- **Pre-check before rewriting the body.** Compare the *plain* kernel's own `num_stages=1`
  against its `num_stages=2`. Where plain itself gains nothing from pipelining on a version,
  recovering the pipeline for a transcription did not pay off there either — that correlation
  held on every version tested.
- **Judge the net on the same-window per-rep ratio** of armed-and-de-staged over the original
  anchor, each arm in its own process. Differencing two percentages against a shipped baseline
  whose own spread is a few percent cannot resolve an effect this size.

Both halves are per-kernel, so measure them on yours; this page carries the shape, not a number.

Plain's overlap is `add_schedule_loops(num_stages)` + `add_pipeline(...)` in `make_ttgir`. No
upstream `gluon_to_ttgir` calls them, while the passes themselves are present in `libtriton`, so
**no `libtriton.so` rebuild is needed** — what is needed is a way to run them over the module
`gluon_to_ttgir` returns.

### Two seams reach them, and neither edits an installed file

The **supported** one is upstream's own stage-inspection hook. `add_stages` passes the
`language` through to it, so a hook can test for the Gluon path and wrap `stages["ttgir"]`:

```python
import triton
from triton.knobs import runtime
from triton.compiler.compiler import Language   # the enum add_stages branches on

def _reinject(backend, stages, options, language, _):
    if language is not Language.GLUON:
        return None
    stock = stages["ttgir"]
    def ttgir(src, metadata):
        mod = stock(src, metadata)
        # second pass manager over the module the stock stage returned
        ...                                     # see the recipe page for the pass list
        return mod
    stages["ttgir"] = ttgir
    return ("reinject", 1)                      # (key, hash): keeps the compile cache honest

runtime.add_stages_inspection_hook = _reinject
```

Upstream documents this pattern — including regenerating a stage's source to splice a pass at an
arbitrary point — and ships tests for it. **Returning the `(key, hash)` pair is not optional**:
Triton's cache key cannot see your wrapper, so without it an armed and an unarmed run collide in
the cache and you measure one binary twice.

The **pack-local** one is `scripts/gluon_swp.py` (tile-programming-gluon), which swaps the
`gluon_to_ttgir` descriptor in place and runs the passes as a second pass manager. It predates the
hook, carries `capabilities()` probing and refuses to install on a fork that already splices the
passes in, and it manages its own `TRITON_CACHE_DIR` per arm for the reason above. Prefer the
hook for new work; `gluon_swp.py` remains the reference implementation of what to run.

> **`gluon_to_ttgir` builds its pass manager inline, but that does not mean there is no seam.**
> Both mechanisms wrap the *function* and run a second pass manager over its result, rather than
> inserting into the pass manager it built. An older note in this pack concluded from the inline
> construction that editing the installed file was the only option; two working mechanisms say
> otherwise.

> **There is no environment variable that arms the re-injection.** Names of the form
> `TRITON_GLUON_*` circulate for it; a three-way search of upstream, the `v3.8.0` tag and the
> whole fork lineage finds **no referent for any of them** — they are not a fork feature you are
> missing, they describe nothing. The re-injection is armed **from Python**; the on-disk patch
> form is armed by `TRITON_GLUON_SWP=N`, a variable defined and read by
> `scripts/patch_reinject.py` in the gluon pack (and `TRITON_GLUON_ASYNC=1` by
> `scripts/patch_async_reinject.py`, the GEAK addition that splices `add_coalesce_async_copy` for
> the gfx950 async path). Because those are not Triton knobs they do not enter the compile
> cache key, so pair them with `TRITON_ALWAYS_COMPILE=1` or you will A/B a cached binary against
> itself — and run each armed / unarmed variant **in its own process**; an in-process interleave
> of patched and unpatched compiles is not a supported A/B protocol here.

The pass sequence either mechanism installs, which is plain's own order:

```python
add_optimize_dot_operands; add_schedule_loops(ns); add_pipeline(use_async_copy, use_block_pingpong)
# then, to get buffer ops back (plain runs convert_to_buffer_ops twelve passes LATER, at #28):
canonicalizer; canonicalize_pointers; canonicalizer; convert_to_buffer_ops
```

> **The byte-identical-TTGIR equivalence between the two seams was verified across Triton
> MINORS, not across architectures — and gfx950 is the arch where it is least safe to assume.**
> `add_pipeline`'s `use_async_copy` argument defaults from `is_async_copy_enabled(arch)`, which is
> False on gfx942 and **True on gfx950**, so on gfx950 the pipeliner takes the async path and one
> further pass (`add_coalesce_async_copy`) joins the sequence. Treat equivalence on gfx950 as a
> thing to check on your build, not a property already established.

> **Whether the pipeliner produces correct IR from Gluon-generated TTGIR is not settled
> upstream.** It was written against `make_ttgir` output; Gluon TTGIR carries user-pinned layouts
> and explicitly allocated `ttg.memdesc` buffers, and no upstream test covers that combination.
> Re-injection is mechanically available, which is not the same as validated — read the numerics
> gate as load-bearing here rather than as a formality.

### The two conditions the kernel must meet

Measured on real gap kernels, every cell bit-exact. **The load form is a hard requirement; the
loop form is only one of two ways the depth can arrive:**

| loads written as | loop | where the depth comes from | pipelined |
| --- | --- | --- | --- |
| `gl.amd.cdna<N>.buffer_load` | either | either | ✗ |
| `gl.load` | bare `range(...)` | the loop / the launch | ✗ |
| **`gl.load`** | **`tl.range(..., num_stages=2)`** | the loop | **✓** |
| **`gl.load`** | **bare `range(...)`** | **the `add_schedule_loops(pm, ns)` argument** | **✓** |

1. **The depth has to reach `add_schedule_loops`, and there are two different ways it can — which
   is why you will find two apparently contradictory rules about a bare `range`.** The pass reads
   `tt.num_stages` off the loop, and a Gluon `for` over the builtin lowers to an `scf.for`
   carrying none (Gluon has no `range` of its own, only `static_range`, which unrolls). So:
   - On the **loop-annotation** path the loop must be a `tl.range`, and a bare `range` gets
     nothing. This is the only path available for a **dot-free** loop, where the annotation is the
     sole way to request the transform.
   - On the **injection** path the loop is not read at all — the depth is the
     `add_schedule_loops(pm, ns)` argument — so a bare `range` pipelines normally **provided the
     loop has a `tt.dot`**. Do not rewrite a loop to `tl.range` just to use the shim.

   Both statements are true within their path; a 2x2 table that omits which path it describes will
   read as a contradiction. The per-path table with the measured cells is on the recipe page.
2. **The loads must still be `tt.load` when the pipeliner runs.** It anchors on global
   `tt.load`s whose forward slice reaches a `tt.dot`; an anchor written with explicit
   `gl.amd.cdna<N>.buffer_load` — which the transcription runbook asks for, because
   `gluon_to_ttgir` runs no buffer conversion — hands it ops it cannot see. Those two pieces
   of guidance genuinely pull against each other; restoring plain's buffer-conversion order
   resolves it rather than making you choose. **It is not free, and on one real kernel it was
   negative**, and it conflicts with buffer *stores* badly enough to abort compilation. Arm it for
   the anchor conflict, not as a default.
3. **`add_pipeline` is not the end of plain's pipeline — splice plain's whole tail.** Two later
   passes decide whether the arm even runs: without `remove_layout_conversions` every operand
   takes a second trip through LDS and the arm can fail to **launch** on shared memory; without
   `in_thread_transpose` one wide LDS read becomes many narrow ones, and it is also what lets the
   pipeliner emit the rotating shared layout Gluon has no constructor for. **This tail was found
   missing twice, and neither symptom looked like "a pass is missing"** — once a launch-time
   out-of-resource, once a collapsed LDS access width.
   Two practical consequences: `in_thread_transpose` is **arch-gated** upstream
   (`is_in_thread_transpose_enabled`), so probe the gate on the build rather than reading the
   namespace — a static read cannot see it; and because that pass restores the rotating shared
   layout, an operand staging that is UNRECOVERABLE by hand comes back for free and `verify` can
   reach PASS instead of RECONCILED. `gluon_swp`'s default recipe splices the full tail and
   reports what it actually applied — read that report, since a pass absent on one minor is
   reported rather than silently skipped.

**Kernel side (give the pipeliner room), and it is one change rather than two:** load the streamed
operands **in-body** with no hand register-prefetch, and **do not hand-write the LDS staging at
all**. Building that path is what the pass exists to do, and the faithful-anchor shape starves it.
De-staging with the injection OFF regresses against the hand-staged body — you removed the staging
and nothing rebuilt it — so the two must be measured together. Also split a causal mask into two
loops: a single loop with a loop-variant `scf.if` blocks both this pass and the automatic
ping-pong.

**Proof it landed (all must move; TFLOPS alone is not proof):** `ttg.memdesc_index` appears in the
TTGIR and the `local_alloc` / `local_store` / `local_load` counts rise toward plain — that first
one is the multi-buffer tell and the cheapest single signal. Then `asm_loop_audit.py` shows
full-drain `lgkmcnt(0)` down and relaxed `lgkmcnt(N>0)` appearing, `mfma_efficiency.py` cadence
down, `MfmaUtil` up at unchanged occupancy, and the numerics gate passes.
`probe_levers.py --all` answers a **different** question — whether the symbols exist in this
`.so` — and symbols existing is not the pass biting; those two hypotheses come apart here.

## IR / asm acceptance signals

| Change | Confirm in IR/asm |
| --- | --- |
| 2/3-stage prefetch | `wait_group(N)` present; loads for `k+1`/`k+2` issue before `DOT(k)` |
| + LLIR scheduling (opt-in stock `coexec`, or an authored pass) | MFMA interleaved with `buffer_load` / `ds_read` (not all MFMA clustered); compare the iter-end `v_accvgpr_mov` block |
| pipeline correct | `wait_group` retires each LDS buffer before the next `AC` overwrites it |

## Reprofile signal

After a pipeline change, MFMA efficiency should rise toward the budget target;
if it does not, check (1) the scheduler knob is actually on (IR), (2) the LDS
layout is conflict-free (`ds_read` 16-cyc), (3) no new spills. Then reclassify —
the next bound class may now be register or memory.
