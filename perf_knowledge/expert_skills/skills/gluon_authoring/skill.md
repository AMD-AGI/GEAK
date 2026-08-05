---
id: gluon_authoring
title: "Author Gluon on CDNA: the plain-Triton → Gluon port with faithful layout and pipeline recovery (and the Gluon → Gluon entry), plus the language surface and the do-not-write list"
kind: expert_skill
authors: [qiongz]
scope: kernel
# ---- selector: the workflow matches these against the live bottleneck ----
match:
  # Not operator-gated. What decides whether this skill pays is the STATE OF THE SOURCE
  # (see `requires` below), not what the kernel computes: the same kernel and the same
  # starting point can yield a clear win or a clear loss depending on how the transcription
  # is done, so an operator filter selects the wrong thing in both directions.
  operator: '*'
  arch_class: ['*']
  gens: [gfx942, gfx950]
  dtypes: [bf16, fp16, fp8_e4m3_fnuz]
  regimes: ['*']   # decode included -- see the entry-gate note in Do-no-harm
  # triton = port a tuned plain kernel; gluon = the source is already Gluon, optimize it in place
  from_backend: [triton, gluon]
  to_backend: gluon
# ---- entry precondition: the one gate that IS predictive ----
# A port measured against an unoptimized plain kernel measures the config sweep, not Gluon.
requires:
  from_backend_triton:
    # all three must hold before step 1; if any fails, tune plain FIRST and re-enter
    - "the plain source is at its own best config (a config sweep has run and its winner is pinned)"
    - "`plain@ns=1` has been measured alongside it, so a mistuned `num_stages` cannot be
       mistaken for a Gluon win"
    - "the comparator recorded for every later number is that tuned kernel, never the shipped default"
  from_backend_gluon:
    - "the workflow's own frozen baseline is the floor; the >=95%-of-tuned-plain bar does not apply"
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
- **Already Gluon** (`from_backend: gluon`). Nothing to transcribe and no plain comparator to hold parity
  with, so **skip step 1 and step 3's checkpoints**, and read step 4 as the whole procedure rather than
  as a continuation (its preamble says which of its items are port-only). Step 2 is still available, but
  as an **ordinary lever rather than a debt to repay**, and it is entered differently:
  - **Step 2a does not apply** — there is no plain side to turn off, so there is no `plain@ns=1`. Ask
    the IR instead whether this loop already overlaps, using **the tell that matches its shape** (step
    2d): `ttg.memdesc_index` only where the loop has a dot; `iter_args` 0→1 plus load-count scaling
    where it does not, because `memdesc_index` is 0 on a dot-free loop by construction.
  - **If it does not overlap and the loop has a dot**, steps 2b–2d apply unchanged: the re-injected
    pipeliner is the cheapest way to add it, and it will build the LDS staging itself.
  - **If it does not overlap and the loop is dot-free**, the pipeliner has nothing to anchor on. That is
    the **authored-overlap** case, and on CDNA3 the surface is sync staging
    (`allocate_shared_memory` + `.store()`/`.load()` + barrier) plus the `warp_pipeline_stage` hint —
    *not* async copy, which does not lower on gfx942 at all. See
    [`references/gluon/pipeline-reference.md`](references/gluon/pipeline-reference.md)
    `## Authored overlap`, and `## Knobs & pitfalls` here for what is unavailable despite importing.
  - **If it already overlaps**, the pipeline layer is closed; go to step 4.

**Run steps 1–4 as one continuous track held by one agent, not as a fan-out.** Two reasons, and they
are different. The porting moves are *deterministic*: any two agents transcribing the same pinned
`.ttgir` land on the same layouts, so parallel arms do not explore alternatives — they all measure the
same transcription, and a shared bug in it identically. And the continuation in step 4 is *stateful*:
what to try next is chosen from the IR and profile of the anchor you just built, which the agent that
built it is holding and a fresh agent would have to rebuild. Splitting the track at the gates throws
that away at the moment it becomes useful.

**What the track needs from the caller.** This skill owns no budget model, but the port has a shape the
round loop has to be told about: a transcription lands **below** the comparator and climbs back, and at
the optimize-tuned defaults that phase is not representable at all — the transcription round produces
no candidate, so no patch is saved, no verify runs, and the loop stops two rounds in. `kernel_workflow`
therefore carries port-shape defaults (a port-sized budget, a candidate floor below 1.0, and a
**negative** progress band so a round that gives ground while exploring does not end the run).

**Run this port at `mode=optimize`, and say so.** Author mode writes a fresh seed that *replaces* the
source — which would overwrite the very kernel being transcribed, since the port needs that kernel's
own `.ttgir`. So the run that needs these defaults is an *optimize* run, and it declares itself by
passing **`target_language=gluon`** (inert on the optimize branch otherwise, so it means exactly "this
run ends in a different language than it started in") or an explicit **`port=true`**. A plain optimize
run with neither is unchanged. The commit gate is untouched either way — nothing sub-baseline is banked.

In `kernel_workflow` terms this is the **`deep_explore` track**, not a set of specialist directions: it
runs alone in its own round, carries its own long measure→self-profile→rewrite loop, and has authority
over kernel plus wrapper — which is the shape steps 1–4 need. One mismatch to steer around: that track
is documented as a minimally-steered ground-up rewrite, and here the opening is the opposite. **Do not
let a ground-up rewrite replace the transcription** — the transcription *is* the anchor every later
number is attributed against.

**What is actually invariant, and what is only the default route.** Three things are not negotiable,
because each one makes the run's numbers unfalsifiable rather than merely worse: the **layout
equivalence gate** at the end of step 1 (a numerically-correct wrong-layout anchor poisons every later
delta); **not mixing transcription with optimization** in the same edit (it destroys the ability to
attribute anything); and **the comparator staying frozen** for the duration. Everything else — the
order the residual's owners are worked in, whether the pipeline layer is entered at all, which of two
loop bodies ships — is a **route chosen from measurement**, and step 2a exists precisely to choose it.
Read the numbered steps as the order that wastes fewest cycles on a typical port, not as a checklist to
satisfy: plenty of kernels owe no pipeline debt at all, so for those the correct action at step 2 is to
establish that and move on.

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

**Why a faithful transcription can land below the comparator, and why that is a debt rather than a
ceiling.** `gluon_to_ttgir` does not run plain's `add_schedule_loops` + `add_pipeline` on **any**
upstream version (3.6.0 / 3.7.0 / 3.7.1 / 3.8.0 all checked), so a transcription alone loses whatever
cross-iteration overlap plain's `num_stages` bought — a regression for a reason that has nothing to do
with your layouts. Those two passes are **re-injectable**: at the TTGIR level an explicit Gluon loop is a
non-pipelined `scf.for`, exactly the object the AMD pipeliner is written to consume, and both passes are
already in `libtriton` on all four versions. Only the Python pass list omits them, so reaching them needs
**no `libtriton.so` rebuild** and — via `scripts/gluon_swp.py`, which wraps `gluon_to_ttgir` in-process —
**no edit to any installed file** either.

**But the debt is often zero, and paying a zero debt is how a round gets spent for nothing.** Two ways it
comes out zero, and both are common enough that assuming a debt is the wrong default: a champion that
compiled at `num_stages=1` has no overlap to lose at all, and a champion whose shipped `num_stages` is
itself a *pessimisation* would have the loss recovered along with the pipeline. That is why step 2 opens
with a measurement — `plain@ns=1` at the champion's own config — rather than with an injection. Size the
debt, then decide.

**Where there is a debt, injection alone still changes nothing: the kernel has to be a candidate.** Two
conditions, both required, and the faithful-transcription rules happen to violate the second. The
pipeliner anchors on global `tt.load`s whose forward slice reaches the dot, so an anchor written with
explicit `gl.amd.cdna3.buffer_load` — which is what step 1 asks for, because `gluon_to_ttgir` runs no
buffer conversion — hands it ops it cannot recognise. **The two pieces of guidance genuinely conflict**,
and the resolution is not to make the author choose: it is to restore plain's own *order*, which runs the
buffer conversion twelve passes *after* the pipeliner. Step 2 has both conditions and the switch that
restores the order.

**And on a dot kernel, step 1 walks you into a shape step 2 cannot act on.** The pipeliner's job is to
*create* the LDS staging and the overlap around it, so it needs a loop with that work left in it. A
transcription that hand-authors `allocate_shared_memory` + `local_store` + `local_load` — the most direct
way to express a recovered `#shared` / `#dot_op` pair — has already done it, and the pass runs over the
loop and changes nothing. Handing the staging back is what makes it act, with one trap worth stating
before you get there: **un-writing the staging without arming the injection is a regression, not a
neutral intermediate** — you have removed a multi-buffer that was doing real work and put nothing in its
place. The two halves go together or not at all.

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
the layouts, and iterate until `--verify` passes.

> **Follow [`references/phases/transcribe-runbook.md`](references/phases/transcribe-runbook.md) for this
> step.** It is the executable form — six numbered stages, one command and one decision each — and it
> carries two things this summary cannot: the **Apply** checklist (declaring a layout is not applying it;
> a body left on `AutoLayout` compiles, passes the oracle, and is several times slower), and the rule for
> **classifying each `ttg.local_alloc` before transcribing it**. That second one decides `shared` bytes
> per workgroup and therefore waves/SIMD, and neither `--verify` nor the numeric oracle can see it.
> Check the result with `scripts/probe.py measure` — compile-only, seconds, no GPU — as soon as the
> anchor builds, not after the first timing.

```bash
python3 "$SKILL/scripts/ttgir_bridge.py" --selftest      # 10 s, offline: is the recovery itself sane?
bash    "$SKILL/scripts/dump_ir.sh" <compile cmd> --variant plain --out ir/ \
        [--kernel-name <substr>]                         # PIN the body on a multi-kernel op
python3 "$SKILL/scripts/ttgir_bridge.py" recover --ttgir ir/plain/plain.ttgir --arch gfx942 \
        --out anchor_layouts.py                          # layouts, via the compiler's layoutToGluon
python3 "$SKILL/scripts/ttgir_bridge.py" verify --plain ir/plain/plain.ttgir \
        --anchor ir/anchor/anchor.ttgir --arch gfx942    # layout equivalence — do not skip
```

`ttgir_bridge.py` gets the layouts from upstream's own `layoutToGluon()` rather than from a mapping
table in this package, so an unsupported kind surfaces as a named `UNRECOVERABLE` row instead of a
plausible wrong constructor, and every layout carries a round-trip proof. Use `recover_gluon.py`
alongside it for anchor assembly and `--with-skeleton`, and fall back to it entirely only where
`import triton` is unavailable — `scripts/USAGE.md` has the division of labour and the reasons.

Two `verify` states pass: `PASS`, and `RECONCILED` (also exit 0) where every difference has a named
structural cause — a disclosed substitution, a pipelined-plain shape the anchor never produces, or a
layout produced by an op Gluon has no builtin for. Read the causes; each is a real fact about your
anchor. `FAIL` is reserved for a difference with none.

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
constructor on the builds checked so far. Three things about that one, in the order they save time:

- **Probe your own build before concluding it is a language gap.** An `UNRECOVERABLE` row is a prompt to
  check, not proof — the Gluon surface moves, and `amd_wmma` sat behind identical wording while being
  constructible as `AMDWMMALayout` all along.
- It is **not** necessarily a `num_stages` artefact — a body can still carry it at `num_stages=1` — so
  re-dump at `ns=1` and re-recover before concluding a body is untranscribable.
- If it really is absent, **a Python-side workaround does not exist**, and the cost is worse than a
  missing constructor. `builder.to_linear_layout(attr, shape)` wants an `ir.attribute`, and on 3.7.1 /
  3.8.0 no binding obtains an encoding attribute from a Value or a Type — so the layout's normal form is
  unreachable from Python and **a substitution for it cannot be verified against the original even in
  principle**. Closing this needs a C++ binding; an earlier claim that one Python binding
  (`to_linear_layout_from_memdesc`) would do it was tried and is **retracted**.

**Recovery audits layouts, not ops**, so a report reading 100% recovered does not mean the body is
transcribable: `amdg.in_thread_transpose` has no Gluon builtin and appeared as a *successful* row until
`ttgir_bridge.py` began naming it. And layout equivalence is blind to two things that decide real ports
— read them off `recover` rather than discovering them at the clock:

- **which operands each `buffer_load` actually carries.** `recover` buckets every site as bare /
  mask-only / mask+`other`, and you transcribe the bucket rather than the source: `tl.load(...,
  other=0.0)` frequently compiles to a load with **neither** — buffer OOB returns zero on CDNA — and
  passing `other=` in Gluon emits it and costs a `v_cndmask` per register. Adding a *mask* the compiled
  form never issued costs the same, which is why the buckets are detected rather than inferred. Two
  operands are **not reachable from `gl.amd.cdna3.buffer_load` at all**, `contiguity` and `stride`; the
  latter shows up only on the pipeliner's peeled prologue loads, which a non-pipelined anchor does not
  have and the injection puts back itself.
- **LDS allocation size**, which `verify` cannot see by construction. A shared total that crosses the
  LDS/CU divisor (**64 KiB on CDNA3/gfx942, 160 KiB on CDNA4/gfx950**) halves workgroups per CU while
  every layout still verifies. Compare `recover`'s `LDS:` line against plain's — it names the divisor for
  the `--arch` you passed, and **declines to name one** for an arch this skill has no figure for rather
  than silently applying gfx942's.
- **the `constants-digest`**, when more than one person or version is transcribing the same body. The
  emitted header carries the dump path and the recovering Triton's version, and role names legitimately
  drift with the dump (the blocked global-load layout is `A_LOAD`/`B_LOAD` on a `num_stages=2` dump and
  `FROM_SMEM` on the `ns=1` dump of the same body, with identical values), so two correct recoveries do
  not compare byte-for-byte. The digest is taken over the sorted constructor expressions with role names
  dropped, and is what to compare instead.

**2. Attribute the residual, then close the suspect that owns it.** Start as soon as step 1 reports PASS.
A faithful anchor is normally *below* the comparator, and **the pipeline is only one of the reasons** —
a pipeline debt is common but far from universal. Naming the owner before acting is what stops a round
being spent injecting into a kernel whose gap was somewhere else entirely:

| suspect | how it shows | owner |
| --- | --- | --- |
| **lost pipeline** | the anchor lands on `plain@ns=1` (2a below) | the rest of this step — re-injection first, hand-authored overlap only where the loop has no dot |
| **lost vectorization** | anchor ≈ 0.5× or worse, and the asm load-width histogram shifted (`dwordx4` → `ushort`) | a `convert_layout` folded backwards into the load; pin the staging with an explicit `allocate_shared_memory` |
| **lost layout** | step 1 is `FAIL`, or `verify` reports `MISSING` with no structural cause | back to step 1 — re-recover, never hand-derive a basis |
| **lost schedule** | the instruction *multiset* matches plain but the waits do not | reorder toward plain's program order; not a layout, pipeline or selection problem |
| **LDS budget** | every layout verifies and it is still slower; the shared total crosses the LDS/CU divisor | `recover`'s `LDS:` line against plain's — `verify` is blind to allocation size |

The rest of step 2 is the **lost-pipeline** owner. If the residual is one of the other rows, close it
there and carry the anchor into step 4 — nothing below applies. Full residual table with the measured
signatures: [`references/gluon/pipeline-reference.md`](references/gluon/pipeline-reference.md).

**2a. Measure `plain@ns=1` — this is the control that decides the whole step.** Re-run the *plain*
champion with its pipeline turned off, at its own config, and compare three numbers:

| reading | what it means | do |
| --- | --- | --- |
| `plain@ns=1` ≈ `plain` | the champion was never pipelined; there is no overlap to lose | **skip to step 3.** A faithful anchor should land ≈1.00 here, and anything well below that is a transcription defect, not a debt |
| `plain@ns=1` **faster** than `plain` | the shipped `num_stages` is a *pessimisation* | **do not recover it** — recovering it recovers a negative. Not a rare case: a library kernel whose wrapper passes no `num_stages` inherits the AMD default of 2, which nobody chose for that body. Report it to the kernel's owner |
| your anchor ≈ `plain@ns=1` **<** `plain` | the entire gap is the missing pipeline, and no layout work will move it | this is the real debt — continue to 2b |

**Read plain's `num_stages` off the LOOP, not the launch.** A launch-level `num_stages=` does nothing at
all to a bare-`range` dot-free loop — measured, plain's TTGIR was byte-identical at launch 1/2/3 — while
a `tl.range(..., num_stages=2)` annotation on the same loop went 2 → 4 → 6 loads. So a champion whose
launch passes nothing may still be fully pipelined, and the `plain@ns=1` control has to flip whichever
knob that kernel actually uses. `scripts/pipeline_survey.py <tree>` classifies a source tree by which
pipeline form each kernel can exercise; treat it as a **screen for what to measure**, not a verdict —
only a dump settles whether a given dispatch compiled pipelined.

**2b. Three conditions, all required — none of them alone does anything.**

| # | condition | why |
| --- | --- | --- |
| 1 | **the loop needs an anchor, and the anchor is a `tt.dot` — not the loop syntax** | `add_schedule_loops(pm, ns)` takes the depth as a **pass argument** and reads no attribute off the loop. So a loop **containing a dot pipelines on a bare `range`**; a **dot-free** loop has no anchor and is the *only* case that needs `tl.range(..., num_stages=N)`, where `None` inherits the launch value |
| 2 | **the loads must still be `tt.load` when the pipeliner runs** | plain orders the pipeliner at #15/#16 and `add_convert_to_buffer_ops` at **#28**, so plain's own pipeliner only ever sees `tt.load`. Write `gl.load` and arm `buffer_ops=True` to restore that order |
| 3 | **the staging must be the pipeliner's own** | a hand-written `allocate_shared_memory` + `gl.barrier()` body starves it, and it must be removed **entirely** — one `ttg.barrier` left in the loop makes the pass skip the loop wholesale |

> **Do not generalise the 2×2 in `references/gluon/pipeline-reference.md`.** It reports
> `gl.load` + bare `range` as not pipelining, which is true **on the dot-free kernel it was measured on**
> and false with a dot in the loop. Two authors rewrote working bare-`range` dot loops into `tl.range`
> for no effect before that was caught. Condition 1 above is the rule; the table is one row of it.

On condition 3, the failure has a misleading error: `'ttg.local_alloc' op pipeliner doesn't know how to
predicate this op` is the **symptom of staging not removed**, not a language wall — the first
investigation to hit it concluded that Gluon's `allocate_shared_memory` was fundamentally incompatible
with the pipeliner, and that was wrong.
`buffer_ops=True` is **opt-in** because it fails two ways: on an anchor whose **loads** are already
`buffer_load` it aborts the pass manager loudly, and on one whose **stores** are buffer ops it does not
raise at all — `LLVM ERROR: Fatal pipeliner error` kills the interpreter. Arm it only on a body written
throughout with `gl.load` / `gl.store`.

**Un-staging and injecting are one step, and the pair is only *conditionally* worth it.** Un-staging
alone removes a multi-buffer that was doing real work and puts nothing back, so it is a regression, not a
neutral intermediate. But the injection does not always cover that cost: measured across four versions on
one kernel it lost three times and won once. **Judge it on the same-window per-rep ratio `L+P ÷ L`** —
subtracting two percentages against a drifting plain arm cannot resolve a 3% verdict. Keep the
explicit-smem version either way; it is where the hand-built double buffer starts. Two other ways to
starve the pass: a hand register-prefetch (the pipeliner *is* the prefetcher, and a manual one consumes
the slot) and a loop-variant `scf.if` (split a causal mask into two loops; it blocks `BlockPingpong` too).

**The cheap pre-check that predicts the verdict:** compare **plain's own `ns=1` against its `ns=2`**. On
a version where plain itself gets no pipeline benefit, recovering the pipeline for it is not worth it
either — that held on all four versions of the kernel above, with the one version that showed plain a
real gain being the one version where the recovery paid.

**2c. Inject, without editing anything.** `scripts/gluon_swp.py` wraps `HIPBackend.gluon_to_ttgir`
in-process and runs the two passes as a second pass manager over the module the stock function returns:

```bash
python3 "$SKILL/scripts/gluon_swp.py"            # capabilities of THIS build, probed not inferred
python3 "$SKILL/scripts/gluon_swp.py" --selftest # offline; skips cleanly with no AMD backend
```
```python
import gluon_swp
with gluon_swp.pipelined(2, buffer_ops=True):    # compile INSIDE the block -- Triton caches
    out = my_anchor[grid](...)
```

It produces **byte-identical TTGIR to the on-disk splice on all four versions**, armed and unarmed, so
nothing is given up by not touching site-packages — while a read-only or shared install, a later
`pip install --force-reinstall`, and a crash mid-experiment all stop being hazards. It refuses to install
on a fork that already splices the passes in, because running them twice is a different experiment.

> **Known limitation: it stops at `add_pipeline` and omits plain's post-pipeline tail**, and the
> consequence is not "less gain". The pipeliner's `local_load` lands in a blocked layout with a separate
> `convert_layout` to the dot operand, so **each operand takes an extra LDS round trip** on top of the
> multi-buffered staging. On a tile whose staging already fills the budget that surfaces as
> `OutOfResources: Required <n>, Hardware limit 65536` — the injection succeeded and looks broken.
> `add_remove_layout_conversions` is the pass that folds the double trip. If you hit it, splice plain's
> own order after `add_pipeline`: `add_convert_to_tensor_ops` → canonicalizer →
> **`add_remove_layout_conversions`** → `add_reduce_data_duplication` → `add_move_up_prologue_loads` →
> `add_block_pingpong(ns)`. Record a pass absent on this build as `-name` rather than skipping it
> silently, or a cross-version regression becomes invisible; 3.6.0 lacks the first and the fifth and
> still reaches the same op census with the other four.
`scripts/patch_reinject.py apply|revert|status` is the on-disk form, kept for when you want the pass list
visible in `compiler.py` while reading; it is env-armed (`TRITON_GLUON_SWP=N`) so armed and unarmed are
the same binary, and its splice point is version-dependent (before `add_warp_pipeline` on 3.7+; after the
last `add_*` call on 3.6, which has no warp pipeline at all). `--selftest` pins both.

> **`TRITON_GLUON_SWP_PIPELINE` is not the knob**, and neither are `TRITON_GLUON_COOP_LDS` or
> `TRITON_GLUON_PINGPONG`. All three belong to a vendor fork's `GetEnv.h`; **no upstream version reads
> any of them.** Measured on clean 3.7.1 and 3.8.0 they are *tolerated and inert* — as is a knob invented
> on the spot — which is the worst of the three available outcomes: nothing errors, nothing changes, and
> the null result reads as "this technique does not work here".

**2d. Confirm it fired from the IR, before you time anything — and read the right tell for your shape.**
Dump the Gluon TTGIR armed and unarmed and compare. **The tell differs by whether the loop has a dot**,
and using the wrong one produces a confident false negative on a loop that did pipeline:

| loop | landing tell |
| --- | --- |
| **contains a dot** | `ttg.memdesc_index` appears — the multi-buffered-LDS signature, and the cheapest single signal |
| **dot-free** | **not** `memdesc_index`, which stays 0 because a dot-free loop prefetches into *registers* and never touches LDS. Read `iter_args` 0→1 and the load count scaling with depth (2→4→6), plus `tt.num_stages` on the `scf.for` and a visible peeled prologue |

Either way the full-drain `s_waitcnt lgkmcnt(0)` should relax to `lgkmcnt(N>0)`. IR **identical** armed
and unarmed means the pass ran and rewrote nothing — a different failure from the passes being absent,
and one no availability probe can see: `probe_levers.py --all` reports that the symbols are in this
`libtriton.so`, not that they will bite on your IR.

**A missing signal is not counter-evidence until you have seen it on the reference arm.** `s_setprio`
staying 0 after injection was once read as "the mechanism never started" — but plain's own `ns=2` build
of the same kernel also had `s_setprio == 0`, because `add_block_pingpong` does not accept that loop
shape at all. Two other signals said the injection had fired. Confirm a tell would have appeared on the
comparator before treating its absence as a verdict.

> **The cache trap has two halves, and each on its own gives a false negative. It is stated here because
> the injection is where it was first hit, but it is NOT specific to step 2 — it applies to every A/B in
> this file, including the step-4 layout sweep, where variants differ by a `constexpr` the cache key does
> not distinguish and a real win reads as zero.** *In process*, Triton's
> JIT cache is keyed on `(function, signature, constexprs)` and **knows nothing about the injection**, so
> two arms differing only by the wrapper hit the same compiled artifact and the second silently reuses
> the first's code. `TRITON_ALWAYS_COMPILE=1` does **not** fix it. *On disk*, a per-arm
> `TRITON_CACHE_DIR` does not encode the **depth**, so an `ns=3` probe pointed at the `ns=2` directory is
> served the `ns=2` binary and reads as "depth does nothing". Give each arm its own kernel object, key
> the cache dir with `gluon_swp.cache_tag()`, and confirm each arm's own `.ttgir` carries the tell above.

**Depth is a knob, not a monotone, and the mechanism is legible before you measure.** Deepening is not
free and not uniform: a depth can build a *single*-buffered rotating stage that leaves LDS at the
champion's footprint while peeling the prologue, and the next depth up genuinely double-buffers — which
doubles the shared footprint and can cross the LDS/CU divisor (**64 KiB on CDNA3/gfx942, 160 KiB on
CDNA4/gfx950**), halving workgroups per CU. `recover`'s `LDS:` line predicts that before a clock is read.
A depth plain itself cannot compile is not available to you either. Start at 2 and sweep. Full mechanism,
the per-shape behaviour, the ping-pong window and why async copy is not reachable from plain on gfx942:
`references/gluon/pipeline-reference.md`.

**3. Pass through two checkpoints — neither of them is where the track stops.**

| checkpoint | what it is | when |
| --- | --- | --- |
| Layout equivalence + bit-parity | `--verify` PASS, and no numeric delta vs the plain anchor (transcription is layout-only, so any delta is a bug, not a Gluon property) | **exit condition of step 1 — always, never deferred** |
| **≥95% of the tuned plain kernel's throughput** | the port has actually landed | exit condition of step 2 — immediately, when 2a found no debt; once the injection is confirmed fired, when it did |

They are ordered, not scheduled: the first is what makes the anchor trustworthy, so nothing downstream
means anything until it passes, and it is never the one allowed to slip. The second is only readable
once step 2 has settled which case you are in — a throughput number taken while it is still open is
uninterpretable either way.

The two fail differently, which is why the second one gets cycles rather than a single attempt. Step 1
converges on a diff you can read. Step 2's effect can only be confirmed by recompiling and reading the IR
back, so on a kernel that genuinely owes a pipeline expect several adjust-and-recheck cycles — **spend
them rather than reading the first recompile as a failure.** Where 2a found no debt there is nothing to
reproduce and both checkpoints close together.

Below 95%, do not start optimizing layouts. The cause is almost always one of these — check them in this
order, cheapest first. Note that the first two share a symptom (no `ttg.memdesc_index` in the Gluon
TTGIR, `local_alloc` / `local_store` / `local_load` counts that did not move toward plain's, and a
full-drain `s_waitcnt lgkmcnt(0)` that never relaxed), which is why the armed/unarmed dump is what
separates them:

1. **the pass never ran, or ran on a loop it does not consider a candidate** — the loop is a bare `range`
   with no dot in it, `num_stages` resolved below 2, or (on the on-disk form) the splice went in at the
   wrong point for this version. IR differs armed vs unarmed only if it ran at all,
2. **it ran and rewrote nothing**, because the body left it no work: IR identical armed and unarmed.
   Hand-authored LDS staging is the usual reason on a GEMM — and remember un-writing it is only half the
   move; `gl.amd.cdna3.buffer_load` in the loop is the other usual reason, and a hand prefetch or a
   loop-variant mask covers the rest,
3. **a layout recovered wrong, or recovered and never wired onto an operand** — re-run `--verify` rather
   than eyeballing it, and check its `missing` list against your own preamble before concluding the
   recovery itself was wrong.

**And one non-cause worth naming, because it looks like all three.** If 2a said the champion compiled at
`num_stages=1`, a residual below 95% is *not* a pipeline problem and no amount of injection will move it.
Go to the other rows — lost vectorization, lost schedule, LDS budget — in
`references/gluon/pipeline-reference.md` and the residual table it points at.

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

**4. Climb — in a language where the allocation and the schedule are yours.**

> Before the first A/B in this step, re-read the **cache trap** stated under step 2. It is not specific
> to the injection: a layout sweep produces variants that differ only by a `constexpr`, which is exactly
> what the in-process cache key cannot tell apart, so a real win here is reported as **zero** unless each
> arm gets its own kernel object and a cache dir keyed on the arming. A sweep that returns "no effect"
> across a range of layouts has usually hit this rather than found a flat space.

For a port this is what
the port was for: the levers below became expressible the moment step 3 closed. **Coming in already
Gluon, this step is the whole procedure** — the list holds, minus the two items marked *port-only*
below, which need the two loop bodies a port produces. Ranking and stopping stay GEAK's; what this file
owes you is what is *available in this language and not in plain*, and the order that wastes fewest
cycles, because a generic loop cannot know that.

Profile first — the profile is about your kernel now, not about a comparator — then work down:

1. ***(port-only)* Which of step 2's two loop bodies you actually want.** A port leaves you holding
   both: the transcription that stages operands in LDS by hand, and the version that leaves that to the
   re-injected pass. They are not the same schedule and either can be ahead. Cheapest possible
   experiment, already built, so settle it before authoring anything new.
2. **Per-operand buffering.** `num_stages` multi-buffers every operand *uniformly*, which is why a tile
   whose uniform footprint exceeds the LDS budget cannot be double-buffered in plain at all. Explicit
   allocation lets operands differ — buffer one and not the other. This is the lever with no plain
   equivalent whenever uniform depth does not fit.
3. **The `#shared` footprint itself** — swizzled versus padded, and LDS dedup. The two things plain has
   no syntax for, so they are the reason `## When to use` sends layout-shaped residuals here.
4. **Declining to stage an operand at all**, i.e. global straight into the dot-operand layout. Note the
   coupling: the LDS round trip is often what was making the global load wide, so re-check the resulting
   load width instead of assuming it survived.
5. **Pipeline depth and, *(port-only)* the re-injected pass's own switches**, now that the loop is
   yours — after 1–4, not before, because against a body the pass cannot act on they measure nothing.
   Coming in already Gluon this is depth on whatever overlap the body has, authored or injected.
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

- **Runtime buffer indices — but measure before paying for the unroll.** The rule is that
  `smem.index(k % nBuffers)` prevents the scheduler from proving overwrite-safety, so buffer indices
  should be compile-time constant and `wait_group(N)` recomputed whenever the prologue, region or unroll
  factor changes. It is **narrower than it reads**: over *sync* staging the two index forms can emit
  identical ISA and identical register counts, and shipped production Gluon exists that indexes an
  async main loop with a runtime modulo, unrolling only its wind-down and citing register allocation
  rather than scheduling for that. So unroll when the ISA says it bought something, not by rule.
- **A hand register-prefetch next to a re-injected pipeliner.** Not additive — it consumes the slot the
  pass wanted.
- **Async copy on CDNA3 — available, but narrow, and easy to misdiagnose.**
  `gl.amd.cdna4.async_copy` does lower on gfx942, at the **32-bit** per-thread direct-to-LDS width
  that generation supports; the wider chunks are CDNA4-only
  (`supportsDirectToLdsLoadBitWidth`). The trap is that a request violating the layout contract —
  threads that do not tile the contiguous dimension, or a per-thread chunk under the granularity
  floor — fails with the same `unrealized_conversion_cast` / LLVM-translation wording as a missing
  op, so a width sweep alone cannot tell "unsupported here" from "asked for it wrongly". Hold the
  op, arch and width fixed and vary only the tiling before recording an arch ceiling
  (`references/gluon/memory-reference.md ## Async Copy To Shared` states the contract). At the
  narrow width it moves a fraction of what CDNA4 does per instruction and tends to become
  `s_waitcnt`-bound, so measure it against sync staging rather than assuming either way.
- **`sched_barrier` / `sched_group_barrier` / `set_prio`.** Absent from `gl.amd.cdna3` *and* `.cdna4` on
  all four versions. Production Gluon exists that imports them inside a `try/except` and defines
  **no-op stubs** on `ImportError`, so on these builds those iglp hints are dead code while still
  reading like scheduling control. Do not copy the pattern expecting it to do anything.
- **`gl.warp_specialize`.** Present in core `gl` on every version and still fails the pass manager on
  CDNA3. `gl.amd.warp_pipeline_stage` *does* work on gfx942 and emits `s_setprio` — but it is a
  scheduling **hint**, not a data movement mechanism, so whether it pays is a measurement.
- **`TRITON_GLUON_SWP_PIPELINE`, `TRITON_GLUON_COOP_LDS`, `TRITON_GLUON_PINGPONG`.** Vendor-fork
  additions to `GetEnv.h`; no upstream version reads any of them, and on a clean build they are
  *tolerated and inert* rather than an error. Use `scripts/gluon_swp.py` for the first and read
  `s_setprio` out of the `.amdgcn` for the third. Ping-pong does fire on gfx942 but in a narrow window
  (measured: 256×256×64 at `num_warps=8`, `ns=2` → 8 `s_setprio`; the same tile at `nw=4`, and
  128×128×64 at `nw=8`, both → 0), and it never fires on hand-authored staging at all, because it
  collects `local_load`s sourced from a loop-carried `BlockArgument` and a hand-written one comes from
  `memdesc_index`. **Judge it from the ISA, never from a source config.**
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
- **Do not read a missing pipeline as a ceiling, and do not read it as a debt either.** It is the
  auto-pipeliner that `gluon_to_ttgir` skips by default, and it is recoverable — but only where the
  champion had one to lose. Step 2a is what tells the two apart, and skipping it costs a round in both
  directions: injecting where the debt is zero measures nothing, and calling a gap layout-shaped where
  the debt is the whole gap sends the next round after the wrong thing.
- **A number measured under injection is not an upstream number.** No upstream version calls these passes
  from `gluon_to_ttgir`, so every measurement taken with `gluon_swp` armed or `patch_reinject` applied has
  to say so, and a reverted tree has to be confirmed reverted before anything else is measured in it.
- **The entry gate is the state of the plain source, not the operator or the regime.** A shape-class
  exclusion here — decode, skinny-M, anything "memory-latency-bound with no tile structure worth
  re-laying-out" — is the wrong filter: the same kernel from the same starting point can clear the
  bar or fall well below its own anchor depending only on how the transcription is done. Check
  these instead, before spending the authoring cost:
  - **Is the plain side actually finished?** If a config sweep has not run and its winner is not
    pinned, the first "Gluon win" is the sweep's. Where the shipped `num_stages` is itself a
    pessimisation that can be most of the headline, so measure `plain@ns=1` too and quote both.
  - **Is there a layout-shaped residual left?** Step 2a answers it from the TTGIR: if the champion
    stages nothing through LDS, the two Gluon-only levers — swizzle/padding choice and LDS dedup —
    have no operand to apply to, and the port has to pay for itself some other way.
  - **What does the transcription do with the operand staging?** Two faithful transcriptions of one
    kernel can differ here and the gap is not small. Materialising each conversion as a user
    `allocate_shared_memory` keeps it live for the whole function and the allocator charges every
    buffer separately; leaving the same conversions as `ttg.convert_layout` lets the backend
    decompose them against one shared conversion scratch. The second form can cost less LDS and
    fewer registers for identical arithmetic, sometimes enough to change waves/SIMD at no extra
    instructions. Neither is always right — read the resulting `shared` bytes/WG and register count
    off the artifacts rather than assuming.
- **Writing a batch of variants without timing them is not a search.** Variants that differ only in
  ways the backend folds away measure the same, and without an A/B between them that is invisible,
  so a run can spend its whole budget producing them. Time each variant against the comparator as
  it lands and dispose of it; rejecting fast is where a round budget's value comes from.

## Sources

- Vendored from `AMD-AGI/TileProgrammingAgentSkills@541a180`
  (`.cursor/skills/tile-programming-gluon/`), **pruned to the API surface, the do-not-write lists and the
  two mechanics above**, plus the transcription runbook and the compile-only occupancy probe that step 1
  and step 4 depend on: 15 of 70 reference files and 11 of 51 scripts. Upstream
  remains the SSOT; this is a one-way snapshot. `references/` is unmodified. `scripts/` carries **six
  corrections**, all of which should go back upstream:
  - `ttgir_to_gluon.py` dropped `tilesPerWarp` and `elementBitWidth` when emitting
    `gl.amd.AMDMFMALayout`, so a chained-dot (gfx950, 16×16 mfma) or scaled-MFMA kernel — both of which
    make `AccelerateAMDMatmul` choose non-default values — transcribed to a silently different layout.
    `--verify` caught it as a text mismatch but named no cause. Both are optional `AMDMFMALayout` fields
    in 3.6 through `main`; they are now emitted when, and only when, the TTGIR prints them, with a
    self-test on both directions.
  - `patch_reinject.py` raised a bare `ImportError` traceback where `triton` is absent, which is the
    normal state of the offline check every other script in this package survives. It now says what it
    needs and exits 2, and it gained a `--selftest` that pins the **version-dependent splice point** on
    synthetic 3.6-shaped and 3.7-shaped function bodies — the one thing here that can be wrong without
    failing loudly.
  - `pipeline_survey.py` defaulted its search root to the author's own `aiter` checkout, so running it
    with no arguments silently surveyed whatever happened to be at that path, or nothing. It now requires
    an explicit root, and its `--selftest` pins the dot-candidacy rule in both directions (a dot loop on a
    bare `range` is a candidate; the same loop dot-free is not until annotated) — the rule upstream got
    wrong twice before it was measured.

  - `ttgir_bridge.py`'s LDS/CU divisor delegates to upstream's `amd_occupancy.lds_per_cu()`, a module in
    the occupancy/roofline regime this package deliberately does not vendor — so in this tree the lookup
    returned `None` for every arch and the `LDS:` line silently lost the divisor that decided a whole
    kernel's residual. Upstream had left the intended fallback in place but **unreachable, after the
    `return`**, and referencing a `_LDS_PER_CU` table that no longer exists (a `NameError` waiting on
    anyone who made it reachable). The fallback is now live and sourced from GEAK's own
    `perf_knowledge/hardware/` — `cdna3_mi300/arch.md` 64 KiB/CU, `cdna4_mi350/memory.md` 160 KiB/CU —
    covering exactly the two `match.gens` this skill claims and declining for anything else, since a
    divisor right for one generation is a confidently wrong occupancy verdict on another. The same
    function annotated `arch` with `Optional` without importing it — invisible at runtime only because
    `from __future__ import annotations` defers evaluation — now spelled `str | None`.

  - `gluon_swp.py`'s docstring stated the pipelining condition as "the loop must be a `tl.range`". The
    anchor is the **`tt.dot` in the loop**, not the loop syntax — `add_schedule_loops(pm, ns)` takes the
    depth as a pass argument and reads no loop attribute, and plain's own `scf.for` carries none either.
    The original was measured on a dot-free kernel, where `tl.range` genuinely is the only route, and
    generalised. Two authors rewrote working bare-`range` dot loops for no effect before it was caught.
    Corrected to three conditions, with the vendored 2×2 in `references/gluon/pipeline-reference.md`
    flagged as the dot-free row of the rule rather than the rule.
  - `ttgir_bridge.py` hard-coded `gl.amd.cdna3.*` in three pieces of advice text, which is wrong guidance
    on `gfx950` — the other generation in this skill's own `match.gens`. Now derived from `rec.arch`,
    with a placeholder rather than a guess for an unknown one.

  `ttgir_bridge.py` and the three added scripts are `ruff --no-cache` clean; the 20 pre-existing findings
  in the four older scripts are untouched. All three offline `--selftest` entry points also run from
  `scripts/smoke_test_recover.sh`, so the two rules above are guarded without a GPU.
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
- Pipeline re-injection — the two kernel-side conditions, the dot-candidacy rule, the per-shape and
  per-version numbers, the ping-pong window, and why async copy is not reachable from plain on gfx942:
  `references/gluon/pipeline-reference.md`. The pass list, the hand-built double buffer that is the
  remaining route where the pass will not bite, and the proof-it-landed signals:
  `references/tile-programming/pipeline.md`
  `## Reproduce plain's software pipeline on the Gluon path (Route 1 — default, no rebuild)`.
- **No measurement is baked into this file, by design.** It carries no timings, speedups, TFLOPS,
  occupancy percentages or bucket splits — those belong in the operator SOTA cards (`operators/<op>/`) and
  the language docs, which are versioned against a specific SKU, container and date and go stale on their
  own schedule. A method file that hard-codes yesterday's numbers ages badly and invites reasoning from
  them instead of from measurement.
- The **≥95%** bar is a **target and a floor**, not a measured result and not a finish line:
  transcription is layout-only, so near-parity is what steps 1–2 together are aiming at. How they get
  there differs by kernel — where the champion carried no pipeline the transcription should already be
  there, and where it did, re-injection is what puts it back, which is why step 2 makes you confirm the
  pass bit from the IR instead of assuming it. The bar exists to trigger step 3's
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
