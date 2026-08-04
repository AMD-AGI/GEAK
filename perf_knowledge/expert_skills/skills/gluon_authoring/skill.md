---
id: gluon_authoring
title: "Author Gluon on CDNA: the language surface, the do-not-write list, and a TTGIR→Gluon port that goes past parity"
kind: expert_skill
authors: [qiongz]
scope: kernel
# ---- selector: the workflow matches these against the live bottleneck ----
match:
  operator: dense_gemm
  arch_class: ['*']
  gens: [gfx942, gfx950]
  dtypes: [bf16, fp16, fp8_e4m3_fnuz]
  regimes: [prefill, training]
  # triton = port a tuned plain kernel; gluon = the source is already Gluon, optimize it in place
  from_backend: [triton, gluon]
  to_backend: gluon
# ---- expected effect: the validation gate's pass criteria ----
# Measure this over the WHOLE track the skill defines — the port AND the continuation past it
# (`## Procedure` step 4) — because the port on its own is built to land near parity, not above it.
# A validation run that stops when step 3's checkpoints close has measured the transcription rather than
# the skill, and will under-report it against the floor below. See `## Sources`.
expects:
  isolated_speedup_min: 1.10
  parity: required
# ---- validation: AUTO-FILLED by validate_skill.py — do NOT hand-edit ----
validation:
  status: draft
  last_verified: ""
  gpu: ""
  model: ""
  measured: {isolated: "", e2e_pct: "", parity: ""}
  artifact: ""
role: advisory_prior
supersedes: []
---

## When to use

The workflow is working a **Gluon** kernel on gfx942/gfx950 and needs two things this file supplies: the
**Gluon API surface**, and the **do-not-write list** — the constructs and instruction choices that compile
and then silently cost you.

**Two entry states, and they do not open the same way:**

- **Porting from plain Triton** (`from_backend: triton`). A tuned plain kernel exists, so the opening move
  is the fixed transcribe-then-re-inject pair in `## Procedure`, closing against that kernel at **≥95%**.
  That bar is a checkpoint the track passes *through*; step 4 is what the port was for.
- **Already Gluon** (`from_backend: gluon`). Nothing to transcribe and no plain anchor to hold parity
  with, so skip step 1 and step 3's checkpoints; keep step 2 only if the loop is not already pipelined, and go
  straight to step 4 with the API pages and the do-not-write list as reference.

**Run steps 1–4 as one continuous track held by one agent, not as a fan-out.** Two reasons, and they
are different. The porting moves are *deterministic*: any two agents transcribing the same pinned
`.ttgir` land on the same layouts, so parallel arms do not explore alternatives — they all measure the
same transcription, and a shared bug in it identically. And the continuation in step 4 is *stateful*:
what to try next is chosen from the IR and profile of the anchor you just built, which the agent that
built it is holding and a fresh agent would have to rebuild. Splitting the track at the gates throws
that away at the moment it becomes useful.

In `kernel_workflow` terms this is the **`deep_explore` track**, not a set of specialist directions: it
runs alone in its own round, carries its own long measure→self-profile→rewrite loop, and has authority
over kernel plus wrapper — which is the shape steps 1–4 need. One mismatch to steer around: that track
is documented as a minimally-steered ground-up rewrite, and here the first half is the opposite. **Steps
1–3 are tightly specified and must be followed as written; step 4 is where the open-ended half begins.**
Do not let a ground-up rewrite replace the transcription — the transcription *is* the anchor every later
number is attributed against.

Two directions not to spend while the port is landing: a second arm on the transcription (above), and
**anything that re-tunes the plain comparator** — that moves the denominator underneath a port whose
whole definition is a ratio to it. Re-tune plain before the track starts or after it ends, never during.

**Process stays with GEAK, with three exceptions this skill does own.** It still carries no round loop,
no budget model, no bound-class model and no escalation gate — run your normal `kernel_workflow` /
`e2e_workflow` loop, and note that step 4's list is not a lever catalogue either: it is scoped to what
*this port* newly made expressible, and it ranks rather than decides. What this file does own, because
getting them wrong silently invalidates the port: the **track shape** just above; **what must and must
not be measured before step 1** (`## Procedure` preamble); and the **duty to re-verify a positive**
(`## Do-no-harm notes`), which matters most in exactly the single-track setup above, because that is
where the workflow's own independent re-benchmark may not be there to do it for you.

Do not use it to decide *whether* Gluon is the right direction. Short version: Gluon pays only when the
residual is layout-shaped, because LDS swizzle/padding choice and LDS dedup are the two things plain
Triton has no syntax for.

**Lazy-loading contract.** Read this file plus [`reference.md`](reference.md). Everything under
[`references/`](references/) is lazy — 14 files, ~2 k lines, load one only when you reach for the
construct it documents.

## Mechanism

**What Gluon is, and what becomes yours to author, is not repeated here** — that onboarding already lives
in GEAK's language layer: [`languages/gluon/overview.md`](../../../languages/gluon/overview.md) (what
Gluon is versus Triton, when to reach for it) and
[`programming_model.md`](../../../languages/gluon/programming_model.md) (explicit layouts, pipeline
stages, register budgeting, MFMA intrinsics, the two CDNA wave patterns). Read those first if you have not
written Gluon before. This section covers only what that layer does not: why the opening move on a
*migration* is mechanical, and why it takes two steps rather than one.

**Why the opening move is deterministic.** The layouts plain Triton's compiler inferred are recorded in
the tuned kernel's `.ttgir`, so the first Gluon version is not a design exercise — it is a mechanical
re-expression of layouts that already exist, and `scripts/ttgir_to_gluon.py` performs it. There is nothing
to search over: any two agents transcribing the same pinned `.ttgir` should land on the same layouts. That
is why it is one fixed action instead of a fan-out. Deterministic does not mean single-shot: `--verify`
is a diff you converge on over several passes, it just converges to one answer.

**Why the port also has to re-inject the pipeline to land.** `gluon_to_ttgir` does not run plain's
`add_schedule_loops` + `add_pipeline`, so a transcription alone loses plain's cross-iteration overlap and
reads as a regression for a reason that has nothing to do with your layouts. Those two passes are
**re-injectable**: at the TTGIR level an explicit Gluon loop is a non-pipelined `scf.for`, the kind of
loop the AMD pipeliner is written to consume, and nothing about the Gluon path makes them inapplicable.
Splicing them back is a Python-only `compiler.py` edit — **no `libtriton.so` rebuild**, the passes are
already in the `.so`.

**Re-injectable is not the same as certain to bite, and the two steps pull against each other.** The
pipeliner's job is to *create* the LDS staging and the overlap around it, so what it needs is a loop that
still has that work left in it. A transcription that hand-authors `allocate_shared_memory` +
`local_store` + `local_load` — which is how you most directly express a recovered `#shared` / `#dot_op`
pair — has already done that work by hand, and the pass will run over it and change nothing. So step 1
can walk you straight into the shape step 2 cannot act on. Step 2 says how to keep room for it and how to
tell the difference between a pass that is absent and a pass that fired and did nothing. Near-parity is
what the two steps together are *aiming* at — it is what the ≥95% gate encodes, not a property the port
arrives with.

## Procedure

The port **from plain Triton**, from the GEAK repo root with
`SKILL=perf_knowledge/expert_skills/skills/gluon_authoring`. Steps 1–3 are a sequence, not a schedule:
each one's exit condition is the next one's precondition, and step 3 names the two you have to hold.
Coming in with a kernel that is already Gluon, start at step 2 and treat step 3 as not applicable.

**Before step 1, settle the comparator, and then stop moving it.** The kernel you pin must be the plain
champion at *its own best config*, and the `.ttgir` you dump must come from that config: every number
below is relative to it, and the layouts you recover are the ones that config produced. Two consequences
worth knowing up front. A port measured against a shipped default is not a port that landed, however good
the ratio looks. And the anchor is **bound to the config it was dumped at** — the recovered layouts carry
literal `warps_per_cta` and tile extents, so if the champion's best config differs per shape bucket you
dump and transcribe per bucket rather than expecting one anchor to follow it.

**Those two things gate step 1, and nothing else does** — reproduce that comparator's number on your
harness, dump its `.ttgir`, begin. That is the whole entry cost, and in particular **do not put a full
profiling pass in front of step 1.** A profile taken before the port describes the *comparator*, and the
question it would answer — is a layout-shaped direction the right one — is decided upstream of this file,
not by it (see `## When to use`). Step 2 makes you read the Gluon IR back anyway, so the state step 4
selects its next move from arrives as a by-product of landing the port. Profile once the port has landed,
when the profile is about the kernel you are optimizing. On a short budget this ordering is the
difference between reaching step 4 and not: the port is cheap and front-loaded analysis is not.

**1. Transcribe, and drive it to layout equivalence.** Pin the tuned plain kernel, dump its IR, recover
the layouts, and iterate until `--verify` passes:

```bash
python3 "$SKILL/scripts/ttgir_to_gluon.py" --selftest    # 10 s, offline: is the converter itself sane?
bash    "$SKILL/scripts/dump_ir.sh" <compile cmd> --variant plain --out ir/ --emit-gluon layouts
python3 "$SKILL/scripts/recover_gluon.py" ...            # calls ttgir_to_gluon.py underneath
python3 "$SKILL/scripts/recover_gluon.py" ... --verify   # layout equivalence — do not skip
```

**Budget several passes here, not one.** `--verify` is a diff, and the expected shape of this step is
verify → read the missing/extra layout attributes it names → fix that one layout → recompile the anchor →
verify again, until it reports PASS. Run the converter's `--selftest` first so that a failure is
attributable to your kernel rather than to the tool. Do not proceed to step 2 on a FAIL and do not
hand-wave a near-match: `--verify` is the only check that catches a transcription which **passes the
numeric oracle while having recovered the layout wrong** (wrong `order`, wrong `kWidth`), and a
numerically-correct wrong-layout anchor poisons every delta you measure afterwards.

**The recovered layouts arrive as a preamble; applying them is yours.** `recover_gluon.py` emits a
`gl.constexpr` block holding the layouts plain's compiler inferred, and — under `--with-skeleton` — a
kernel body beside it that still carries the translator's own default MMA layout and `AutoLayout`.
Nothing connects the two. Wiring each recovered layout onto the operand it belongs to is the manual step
that turns the file into an anchor, and it is not optional: a layout the body never mentions never
reaches the TTGIR, so `--verify` will report exactly those layouts missing. Read a `missing` list that
matches your preamble as "declared but not applied" rather than as "recovered wrong".

The converter covers `#blocked`, `#amd_mfma`, `#swizzled_shared`, `#padded_shared`, `#linear`,
`ttg.dot_op` and `ttg.slice`. Two gaps that are **not** tool bugs, so do not spend verify cycles on them:
`ttg.convert_layout` placement is manual — take it from the recovery map in
`references/tile-programming/layout-recipes.md`; and `amd_rotating_shared` has no `gluon.language`
constructor at all.

**2. Re-inject the pipeline.** Start it as soon as step 1 reports PASS. **If the plain kernel is
auto-pipelined, budget this step several adjust-and-recheck cycles of its own** rather than treating one
recompile as a verdict (see the gate in step 3). Check the build first, then splice into `gluon_to_ttgir`
after
`add_combine_tensor_select_and_if`, gated on `options.num_stages > 1` so it stays cache-key-safe and
default-OFF:

```bash
# read the reinject_ttgir_pipeliner entry: are the passes in THIS libtriton.so?
python3 "$SKILL/scripts/probe_levers.py" --all --arch gfx950
```
```python
add_optimize_dot_operands; add_schedule_loops(ns); add_pipeline(use_async_copy, use_block_pingpong)
; add_convert_to_tensor_ops; canonicalizer; remove_layout_conversions; reduce_data_duplication
; (in_thread_transpose if enabled); move_up_prologue_loads; canonicalizer; cse
```

**The splice point moves with the version** — read the installed
`third_party/amd/backend/compiler.py` before editing, and expect a vendor fork to possibly carry its own
variant of this patch already:

| installed version | splice after | adjust |
| --- | --- | --- |
| 3.6 | `add_combine_tensor_select_and_if`, which ends the function | gate on `knobs.amd.use_async_copy`; drop `add_convert_to_tensor_ops`; `add_reorder_instructions` in place of `move_up_prologue_loads` |
| 3.7, 3.8 | same, but **before** `add_warp_pipeline` | the list above as written |

Give the pass room or it silently does nothing. Three ways to starve it, the first of which step 1 walks
you straight into:

- **Hand-authored LDS staging.** `allocate_shared_memory` + `local_store` + `local_load` in the loop body
  *is* the pass's output, written by you — it wants to create that staging and finds it already there, so
  it rewrites nothing. On a plain GEMM this is the likeliest reason the counts do not move — and the
  other two below do not apply there, so satisfying both of them still leaves you here. The fix is to
  un-write part of the transcription: let the operands arrive as in-body loads and reach their
  dot-operand layout through `convert_layout`, and the pass has something to stage. Keep the
  explicit-smem version — it is where you start the hand-built double buffer if the pass still will not
  bite.
- **A hand register-prefetch.** The pipeliner *is* the prefetcher; a manual one consumes the slot.
- **A loop-variant `scf.if`.** Split a causal mask into two loops; it blocks both this pass and
  `BlockPingpong`.

Launch `num_stages=2`. The recipe, and the hand-built cross-iteration double buffer that is the remaining
route once the pass is established as inert on hand-authored smem, are in
`references/tile-programming/pipeline.md` (it calls that one "Route 2"). Its vetted skeleton indexes the
buffer as `s.index(i % 2)` — see the first entry in `## Knobs & pitfalls` before you copy that.

**Confirm it fired before you time anything**, and expect to go round this loop more than once too: dump
the Gluon TTGIR again and compare its `local_alloc` / `local_store` / `local_load` counts against plain's,
and check that the full-drain `s_waitcnt lgkmcnt(0)` has become a relaxed `lgkmcnt(N>0)`. Those counts
moving toward plain's is the signal the pass landed; a throughput number is not, because a pass that
silently did nothing and a pass that fired but did not help look identical on the clock.

**Dump once with the splice gated off as well.** Counts identical on and off means the pass ran and
transformed nothing — a different failure from the passes being absent, and the one `probe_levers.py`
cannot see: `available: true` reports that the symbols are in this `libtriton.so`, not that they will
bite on your IR. Separating those two is the whole point of reading the IR back.

**3. Pass through two checkpoints — neither of them is where the track stops.**

| checkpoint | what it is | when |
| --- | --- | --- |
| Layout equivalence + bit-parity | `--verify` PASS, and no numeric delta vs the plain anchor (transcription is layout-only, so any delta is a bug, not a Gluon property) | **exit condition of step 1 — always, never deferred** |
| **≥95% of the tuned plain kernel's throughput** | the port has actually landed | exit condition of step 2, once the pass is confirmed fired |

They are ordered, not scheduled: the first is what makes the anchor trustworthy, so nothing downstream
means anything until it passes, and it is never the one allowed to slip. The second is only readable
after step 2's on/off dump says the pass fired — a throughput number taken before that is uninterpretable
either way.

The two fail differently, which is why the second one gets cycles rather than a single attempt. Step 1
converges on a diff you can read. Step 2 is an edit to `compiler.py` whose effect you can only confirm by
recompiling and reading the IR back, so on an auto-pipelined kernel expect several adjust-and-recheck
cycles — **spend them rather than reading the first recompile as a failure.** If plain was not
auto-pipelined there is nothing to reproduce and both checkpoints close together.

Below 95%, do not start optimizing layouts. The cause is almost always one of these — check them in this
order, cheapest first. Note that the first two share a symptom (`local_alloc` / `local_store` /
`local_load` counts in the Gluon TTGIR did not move toward plain's, and the full-drain
`s_waitcnt lgkmcnt(0)` never relaxed), which is why the on/off dump is what separates them:

1. **the pass never ran** — `num_stages` left at 1 so the gate stayed off, or the splice went in at the
   wrong point for this version (re-read the table above). Counts differ between splice-on and
   splice-off only if it ran at all,
2. **it ran and rewrote nothing**, because the body left it no work: counts identical on and off.
   Hand-authored LDS staging is the usual reason on a GEMM, a hand prefetch or a loop-variant mask on
   everything else,
3. **a layout recovered wrong, or recovered and never wired onto an operand** — re-run `--verify` rather
   than eyeballing it, and check its `missing` list against your own preamble before concluding the
   recovery itself was wrong.

**Bound the port by convergence, not by ambition.** Once that list is exhausted, re-injection is
confirmed to fire, and you are *still* short of 95%, stop treating it as a transcription problem: it is
now an ordinary optimization target, so carry the anchor into step 4 and say plainly that the port closed
below parity. Continuing to re-transcribe past that point is the one way to spend a whole budget and land
nothing.

**Passing is not arriving.** Both checkpoints can close on a transcription that reproduces the comparator
and stops there — that is a port which landed, and it is the *floor* of this track, not its result. Read
a closed checkpoint as permission to start step 4, and note the asymmetry it hides: the two ways of
writing the loop that step 2 discusses can both clear 95% while being different schedules, so clearing
the bar does not tell you the body you shipped is the better of the two. Step 4 settles that first.

**4. Continue past the port — this is what the port was for.** The anchor now reproduces the comparator
in a language where the allocation and the schedule are yours, so the levers below became expressible the
moment step 3 closed. Ranking and stopping stay GEAK's; what this file owes you is the list of what is
*newly available* and the order that wastes fewest cycles, because a generic loop cannot know that.

Profile the anchor first — now the profile is about your kernel — then work down:

1. **Which of step 2's two loop bodies you actually want.** You have both: the transcription that stages
   operands in LDS by hand, and the version that leaves that to the re-injected pass. They are not the
   same schedule and either can be ahead. Cheapest possible experiment, already built, so settle it
   before authoring anything new.
2. **Per-operand buffering.** `num_stages` multi-buffers every operand *uniformly*, which is why a tile
   whose uniform footprint exceeds the LDS budget cannot be double-buffered in plain at all. Explicit
   allocation lets operands differ — buffer one and not the other. This is the lever with no plain
   equivalent whenever uniform depth does not fit.
3. **The `#shared` footprint itself** — swizzled versus padded, and LDS dedup. The two things plain has
   no syntax for, so they are the reason `## When to use` sends layout-shaped residuals here.
4. **Declining to stage an operand at all**, i.e. global straight into the dot-operand layout. Note the
   coupling: the LDS round trip is often what was making the global load wide, so re-check the resulting
   load width instead of assuming it survived.
5. **Pipeline depth and the re-injected pass's own switches**, now that the loop is yours — after 1–4,
   not before, because against a body the pass cannot act on they measure nothing.
6. **Tile shape**, which is GEAK's ordinary autotuning, with `num_warps` still pinned for the reason in
   `## Knobs & pitfalls`.

Two disciplines carry over from step 2 and are what keep this half honest. **Fix a structural success
signal before you read a clock** — buffer count, barrier count, the `lgkmcnt` shape — so that an
unchanged IR is diagnosed as such instead of being mistaken for a lever that did not pay. And **be
willing to be wrong about the cap**: a residual can be correctly diagnosed as layout-shaped and still
not move when you fix it, because a second, non-layout resource binds at the same point. Occupancy is
the usual place this happens, since LDS footprint and register pressure can each pin it independently —
freeing the one you diagnosed then buys nothing on its own. Read that as the ranking being wrong, not
the diagnosis.

## Knobs & pitfalls

Do-not-write list, i.e. what compiles and then costs you:

- **Runtime buffer indices.** `smem.index(k % nBuffers)` is an anti-pattern: the scheduler cannot prove
  overwrite-safety and refuses to interleave. Buffer indices must be compile-time constant, and
  `wait_group(N)` must be recomputed whenever the prologue, region or unroll factor changes. **The vetted
  double-buffer skeleton in `references/tile-programming/pipeline.md` writes exactly this** (`cur = i % 2`)
  — it is upstream text and this list overrides it. Unroll the loop by 2 so each index is a literal.
- **A hand register-prefetch next to a re-injected pipeliner.** Not additive — it consumes the slot the
  pass wanted.
- **The LLIR scheduler toggle on anything with VALU between the matmuls** (softmax, scale, dequant). It
  assumes a pure MFMA→MFMA accumulator chain and emits **invalid IR**, not a slowdown. Default-skip on
  attention shapes. **Fork-only**: `TRITON_ENABLE_LLIR_SCHED`, `TRITON_ENABLE_AMDGCN_AS` and
  `TRITON_ENABLE_AMDGPU_RA_HINTS` (`dump_ir.sh --knobs LLIR_SCHED|AMDGCN_AS|RA_HINTS`, and the
  `gemm_compiler_stack` probe's `llir`/`ra` rungs) exist in no upstream Triton — 3.6, 3.7 and 3.8 alike.
  On a stock build they export an env var nobody reads, so the knob is a silent no-op rather than an
  error. Confirm with `probe_levers.py --all` before attributing any delta to them.
- **`disableSched`** — costs occupancy outright. Never on this path.
- **Deep pipelines by reflex.** Software prefetch regresses once waves are already VGPR- or LDS-capped,
  and aggressive unrolling can raise `s_nop` count rather than lower it. `num_stages=2` is the start;
  deepen only against a profile.
- **`convert_layout` scratch after the pipeliner pass.** Scratch allocated post-pass is outside hazard
  analysis and fails silently.
- **`amd_rotating_shared`** — has no `gluon.language` constructor; do not hand-roll a basis for it.
- **RDNA WMMA formulas.** Not vendored and not applicable; `match.gens` is CDNA.
- **Sweeping `num_warps` on a transcribed kernel.** `ttgir_to_gluon.py` emits the IR's *literal*
  `warps_per_cta`, so any other warp count disagrees with the layouts — a correctness bug, not a slow
  config. `triton.autotune` itself works on `gluon.jit` and upstream's own Gluon examples use it, but they
  sweep tile shapes and stage counts with `num_warps` pinned. Tile-shape autotuning is GEAK's to do after
  the port lands; only the warp count is off the table.

`num_stages` itself is not a Gluon dead knob — it is a buffer-count budget on the default lowering and the
pipelining trigger again after re-injection, so carry the plain winner's value over as a starting depth.
Full lists: `references/gluon-negative-patterns.md`, `references/platform-known-issues.md`.

## Do-no-harm notes

- **Advisory only.** This file supplies API surface and mechanics, never a verdict. The workflow's
  isolated A/B against the immutable oracle decides, and a result below the measured baseline is a
  negative whatever this says.
- **It does not narrow the workflow's search.** Only steps 1–3 are fixed, and only because those moves
  are mechanical; step 4 ranks but does not decide, and every matched skill enters the candidate set
  rather than pre-empting it.
- **A Gluon result slower than its comparator is a revert.** A Gluon wall does not prove the comparator
  was at its ceiling.
- **Disbelieve a fast number as hard as a slow one, and own that check yourself.** The bullet above tells
  you what to do with a regression; nothing about a *win* is self-evident either, and the single-track
  shape in `## When to use` is precisely the setup where the workflow's independent re-benchmark may not
  be running alongside you to catch it. Before a Gluon number is reported: re-measure it in a clean
  workspace, interleaved with the comparator on the same device rather than in separate batches; treat
  anything inside the run-to-run spread as no result; and **confirm the Gluon kernel actually executed.**
  That last one is not paranoia — a dispatcher that falls back to the plain kernel when some capability
  or shape condition is unmet yields a "Gluon" measurement that is the plain kernel's, and because the
  fallback is numerically perfect by construction, correctness checks endorse it. Check the launched
  kernel name, or make the fallback path fail loudly instead of silently.
- **On a port, the comparator must be the plain kernel tuned to its own best config.** A win over a
  default strawman is invalid, and so is a 95% measured against one. Starting from an existing Gluon
  kernel there is no plain comparator at all — the workflow's own frozen baseline is the floor, and the
  ≥95% bar does not apply.
- **Correctness gates before timing**, and step 1 additionally gates on layout equivalence — a
  numerically-correct wrong-layout anchor poisons every later delta, which is why that checkpoint is
  never the one allowed to slip.
- **Do not read a missing pipeline as a ceiling.** It is the auto-pipeliner that `gluon_to_ttgir` skips by
  default. An older note that re-injection "regresses" was a specific misfire — a hand-async loop with no
  in-body `tt.load` to anchor on, plus the `disableSched` cliff — not the passes.
- **Keep it off decode / skinny-M shapes.** Memory-latency-bound with no tile structure worth
  re-laying-out; authoring cost cannot be repaid there.

## Sources

- Vendored from `AMD-AGI/TileProgrammingAgentSkills@c14a583`
  (`.cursor/skills/tile-programming-triton-gluon/`), **pruned to the API surface, the do-not-write lists
  and the two mechanics above**: 14 of 72 reference files (~2 k of ~11 k lines) and 6 of 41 scripts.
  Upstream remains the SSOT; this is a one-way snapshot. `references/` is unmodified. `scripts/` carries
  **one correction**, which should go back upstream: `ttgir_to_gluon.py` dropped `tilesPerWarp` and
  `elementBitWidth` when emitting `gl.amd.AMDMFMALayout`, so a chained-dot (gfx950, 16×16 mfma) or
  scaled-MFMA kernel — both of which make `AccelerateAMDMatmul` choose non-default values — transcribed
  to a silently different layout. `--verify` caught it as a text mismatch but named no cause. Both are
  optional `AMDMFMALayout` fields in 3.6 through `main`; they are now emitted when, and only when, the
  TTGIR prints them, with a self-test on both directions.
- **Deliberately not vendored — the whole process layer**: round loop, hardware budget / roofline models,
  profiling (rocprof / ATT / PMC), the lever-card catalogue, bound-class signals, the escalation gate,
  orchestration, experiment records, benchmark hygiene, the transcription *protocol* page, and the
  workload strategy pages. GEAK's workflows own process; a second regime inside an advisory prior would
  compete with them. Retained files still cite the dropped paths in 64 places — **dead by design**, do
  not hunt for them and do not read a citation as an instruction to rebuild the regime. For the hardware
  facts among them (instruction availability per gen, LDS banking, VGPR/AGPR budget, occupancy), use
  GEAK's own [`perf_knowledge/hardware/`](../../../hardware/) (`cdna3_mi300/`, `cdna4_mi350/`, `shared/`)
  and [`perf_knowledge/languages/gluon/`](../../../languages/gluon/).
- Onboarding and concepts are **not duplicated here**; they live in
  [`languages/gluon/`](../../../languages/gluon/) (`overview.md`, `programming_model.md`,
  `gemm_cookbook.md`), which is also where the AMD-measured Gluon GEMM ceilings belong. This skill adds
  the API-page detail, the do-not-write list, and the two migration mechanics.
- Gluon language surface upstream: ROCm Gluon GEMM tutorial
  (https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html), gfx950 Gluon
  tutorials (https://github.com/ROCm/gfx950-gluon-tutorials).
- Pipeline re-injection — pass list, the `num_stages>1` gate, kernel-side conditions, proof-it-landed
  signals: `references/tile-programming/pipeline.md`
  `## Reproduce plain's software pipeline on the Gluon path`.
- **No measurement is baked into this file, by design.** It carries no timings, speedups, TFLOPS,
  occupancy percentages or bucket splits — those belong in the operator SOTA cards (`operators/<op>/`) and
  the language docs, which are versioned against a specific SKU, container and date and go stale on their
  own schedule. A method file that hard-codes yesterday's numbers ages badly and invites reasoning from
  them instead of from measurement.
- The **≥95%** bar is a **target and a floor**, not a measured result and not a finish line:
  transcription is layout-only and re-injection is what restores plain's own overlap, so near-parity is
  what those two steps are aiming at — reachable when the re-injected pass actually bites, which is why
  step 2 makes you confirm that from the IR instead of assuming it. The bar exists to trigger step 3's
  diagnostic list, and to mark where step 4 may begin. It is deliberately *not* an expectation about the
  ceiling: a port can close above it, and a body that merely reaches it is the weakest outcome step 3
  still calls landed — so the bar must never be read as the value of doing this at all.
- **The two numbers in the frontmatter measure different things**, and conflating them under-reports the
  skill. `≥95%` scopes steps 1–3, where near-parity *is* success. `expects.isolated_speedup_min` scopes
  the whole track including step 4, and it is the template floor for a win rather than anything derived
  from this file's own results — so an isolated A/B that stops when step 3 closes is not a valid
  datapoint against it, whatever it reads. `validation.status` stays `draft`, with no auto-application,
  until a `--record` run stamps it.
- **Vendored files under `references/` are unmodified and do contain upstream's own measured claims.**
  Treat those as upstream's evidence, dated to upstream, and not as a GEAK measurement.
