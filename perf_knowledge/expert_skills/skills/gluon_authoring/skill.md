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
  # fp8 is spelled per generation, not once: CDNA3 is FNUZ and CDNA4 is OCP, and
  # `index/capability_index.yaml` uses both names — so claiming only the FNUZ spelling while
  # claiming gfx950 makes this skill silently unselectable on a CDNA4 fp8 bottleneck.
  dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
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
[`references/`](references/) is lazy — 17 files, ~3.8 k lines, load one only when you reach for the
construct it documents.

## Arch dispatch: gfx942 (CDNA3) vs gfx950 (CDNA4)

`match.gens` claims both, and the *mechanics* below are shared: the passes, the three re-injection
conditions, the landing tells, the recover→apply→verify order. What is **not** shared is this table, and
every row of it can change a verdict rather than a magnitude. Read it once, here, before trusting a figure
or asserting a digest across generations — the rest of this file assumes you have.

**Numbers live in [`references/hardware/hw_constants.json`](references/hardware/hw_constants.json), not
in prose.** It is the per-arch SSOT the occupancy probe already reads, so take the value from the named key
and pass `--arch` everywhere rather than carrying a figure across a generation. A divisor that is right for
one gen is a *confidently wrong* verdict on the other, not a rounding error.

| what differs | key / how to read it | why it changes a conclusion |
| --- | --- | --- |
| **LDS per CU** — the occupancy divisor | `lds_per_cu_kib` (CDNA4 is 2.5× CDNA3) | a depth or tile CDNA3 reasoning treats as unavailable may simply fit. `scripts/probe.py measure` applies it per artifact and prints the basis |
| **LDS banking** — conflict stride | `lds_banks`, `ds_read_b128_full_conflict_stride_bytes`, and a CDNA4-only `ds_read_b128_2way_stride_bytes` | a recovered swizzle that was conflict-free is **not known to still be**, and the sign can invert: a `per_phase` choice that costs a conflict on one gen can be free on the other. Re-derive from the keys; do not carry the rule of thumb |
| **MFMA family and shape set** | `matrix_layout_family` (`version=3` vs `4`), `mfma_cadence_cyc` | layout digests **should** differ across gens. Assert four-version consistency *within* one arch plus a passing round-trip — never cross-gen equality |
| **Block-scaled MFMA** | `scaled_mfma` | CDNA4-only. A layout family with no CDNA3 counterpart, so it is a new-capability question rather than a regression check |
| **fp8 spelling** | `fp8_dtype` | FNUZ on CDNA3, OCP on CDNA4, and the selector must claim the right name or it never matches. **The CDNA4 key is absent upstream** — do not fall back to the sibling arch's value: the CDNA3 spelling is the one dtype CDNA4 lacks, and Triton silently upcasts it to fp16 with only a warning |
| **Direct-to-LDS width** — whether FORM C async copy lowers at all | `direct_to_lds_bit_widths` (CDNA4 adds the 128-bit form) | each lane must make **one** access of a listed width. Both async entry points then lower on stock `gluon_to_ttgir`; a width off the list, or a native width **split across layout repetitions**, does not — and fails with wording that reads like a missing op. See `## Knobs & pitfalls` |
| **Read-with-transpose LDS** | `ds_read_tr` | where it exists the compiler emits it instead of `amdg.in_thread_transpose`, so the "Gluon cannot express `in_thread_transpose`" gap **has no subject** there and a faithful anchor inherits the instruction for free. No Gluon source API either way |
| **Ping-pong reachability** | probe the ISA for `s_setprio` | the enabling predicate is not arch-symmetric *and* is read only inside `make_ttgir`, so on the Gluon path satisfying it buys nothing on either gen. Judge from the ISA, never from a source config |
| **The omitted post-pipeline tail** | — | the same defect announces itself as `OutOfResources` on CDNA3 and, because the larger ceiling absorbs it, becomes a **silent slowdown** on CDNA4. Check the pass list, not the exception (`## Procedure` step 2c) |

**Three gfx950 keys are missing where gfx942 has them, and the first two are traps rather than gaps** —
`references/hardware/hw_constants.json` is vendored, so these are to report upstream rather than
patch here.
`fp8_dtype` is covered above. `lds_min_alloc_bytes` is what tells a reported `lds/WG` apart from an
allocator round-up, and gfx950's `lds_align_bytes` is **not** that granularity despite looking like it — a
measured allocation need not be a multiple of it, so do not substitute one for the other.
`cvt_off_f32_i4` is a binding-availability record rather than a silicon fact (see its note in the JSON) and
does not bear on transcription. The key that *is* read on the hot path, `lds_per_cu_kib`, is present and
correct on both.

**Two gaps this table cannot close, because they are Gluon-side rather than arch-side** and survive the
generation change unchanged: `amd_rotating_shared` has no `gluon.language` constructor on any arch; and a
user `allocate_shared_memory` is totalled **by scope** where plain's allocator peaks **by liveness**, so a
faithful transcription can report a larger shared figure than the kernel it is copying and land on the wrong
side of the divisor above while every layout still verifies. Read `shared` bytes off the artifact rather
than reasoning about them.

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

### The gates are executable, and a gate is passed by TOOL OUTPUT, not by assertion

Everything in this section was already prescribed in prose below, and prose did not hold: a run has
gone to step 4 from a **0.715×** anchor, reported 1.19× against that anchor, and closed at **0.85×**
against the champion — with every step's *text* obeyed. So each exit condition now has a command, and
**the round log must carry the command's output.** "The precondition holds" is not a gate; it is the
claim the gate exists to test.

**Two of the five are entry-state-dependent, and getting that wrong loses a run in either
direction.** `## When to use` already splits the two entry states; the gates split the same way, and
`references/entry-modes.md` is the one-page table if you want it side by side.

| # | gate | command | applies on `from_backend: triton` (PORT) | applies on `from_backend: gluon` (IN-PLACE) |
| --- | --- | --- | --- | --- |
| G0 | the harness loop shape matches the entry | inspect the launch args (see below) | **PORT shape** — `port: "true"` must be present | **ORDINARY optimize** — `port: "true"` here is the mirror-image mistake |
| G1 | the comparator can support a claim at all | `scripts/champion_gate.py --champion <bundle>` | YES, on `plain_champion.json` | YES, on an **incumbent** bundle — see `entry-modes.md` for which fields change meaning |
| G2 | the transcription debt is paid, and if not, WHO owes it | `scripts/parity_gate.py …` | YES, before the FIRST climb | **NO — nothing was transcribed.** The tool is still useful as a *diagnostic* on a mid-run regression; say which you are doing |
| G3 | the occupancy is not already lost | `scripts/probe.py measure --dir ir/<tag>/` | YES | YES |
| G4 | the number is a number | `scripts/ab_bench.py --module <adapter>.py --permute` | YES | YES |

On failure: **G0** — stop before dispatching; the loop will terminate two rounds into a port that is
working exactly as designed. **G1** — stop, edit nothing, report `blocked`; a gate failure is the
front end's bug, not yours to route around. **G2** — **DO NOT CLIMB**; exit 2 means this round's
outcome is `recovery` against the suspect it closed, never a win. **G3** — read BOTH limiters;
registers and LDS cap WGs/CU independently. **G4** — a delta under the spread is `NOT RESOLVED`.

**Why G2 is not merely skipped on an in-place entry but must not be RUN as a gate.** There is no
anchor, so there is no `anchor_ms`. Passing the incumbent as both sides returns ratio 1.00 →
CLEARED, which is *true and vacuous*: it records a gate as satisfied that was never applicable, which
is worse than not running it. And note what the in-place entry loses along with the debt — the
two-comparator discipline collapses to one, so nothing structurally reminds you that the denominator
has to be honest. The incumbent must therefore be a **measured, asserted** number, not "the file I
started from" and not a figure inherited from another GPU or another container.

**G0 is checked before anything is dispatched, and it has silently cost three runs — all three of
them ports.** Everything in the rest of this G0 discussion is about the PORT entry. On an in-place
entry the harness defaults are already right and `port: "true"` must **not** be set: a candidate floor
below 1.0 and a negative progress delta keep a genuinely stalled search alive, burning exactly the
budget the port shape exists to protect. A transcription
port must run at `mode: optimize` — it transcribes the existing source's own TTGIR, so `author` mode
would overwrite the very thing being ported — and `mode` therefore cannot be what tells the loop this
is a port. `kernel_workflow.js` provides `port: "true"` for exactly this, and its own comment
(lines 40–47) describes the consequence of omitting it better than any summary:

> *"Applied unchanged to a port they delete its recovery phase: the transcription round produces no
> candidate at all, so no patch is saved, no verify runs, `winner` is null, and the loop stops two
> rounds into a port that is working exactly as designed."*

The four defaults that flip with `PORT_SHAPE`, and what each one does to a port left on the
`optimize` values:

| knob | optimize | port | what the optimize value does to a port |
| --- | --- | --- | --- |
| `candidate_floor` | 1.0 | 0.5 | a faithful anchor is below the comparator **by construction**, so it never enters the candidate list: no patch saved, no verify, `winner = null` |
| `max_no_improve` | 2 | 4 | ends the run two rounds in, which is before the recovery round has finished |
| `progress_delta` | +`min_improve` | −0.05 | a layout experiment that costs ground is information on a port; here it reads as a stall |
| `budget` | 6 | 20 | `budget` counts **directions**, and a `deep_explore` direction costs 2 — so `BUDGET/2` is the achievable round count. 20 gives 10 rounds; 10 gives 5 |

**Two things to set beyond the port defaults, both from measurement rather than taste.**
`max_no_improve: 4` is still short of what a real climb needs — on `pa_decode` the winning levers
landed at rounds 1, 8, 8, 9 and 10 with **five consecutive** non-improving rounds in between, so 4
ends the run one round before the payoff and **6** is the value that survives the measured trajectory.
And `candidate_floor: 0.5` is uncomfortably tight: an observed naive anchor on a MoE INT4 kernel
measured **0.51×**, one bad window from falling out of its own candidate list. Set the floor from the
debt you actually took — `parity_gate.py` reports it — rather than accepting a default that has to
guess. *(The deeper point, worth sending upstream rather than working around: on a port, "is this a
candidate" is the question `parity_gate.py` answers properly, as a recovery verdict with an
attribution. A fixed ratio floor is a proxy for it, and the right proxy value is not knowable before
the anchor is measured.)*

**G2 is the one that was missing, and it is the one that decides a port.** A faithful anchor is a
regression you knowingly created — not a baseline to quietly climb from. `parity_gate.py` splits the
anchor→champion gap across the three suspects **from the compiled artifacts**, so the attribution is
measured rather than argued:

- `lost_pipeline` — the champion's TTGIR carries `ttg.memdesc_index` / `ttg.local_store` /
  `num_stages > 1` and the anchor's does not. *Note the inverse trap the tool refuses to fall into:
  `max iter_args >= 2` is NOT evidence of pipelining — every accumulator loop, including any
  online-softmax kernel, satisfies it.*
- `lost_layout` — a load-width or LDS-op histogram narrowed (`dwordx4` → `ushort`, `ds_read_b128` →
  `ds_read_u16`), or `shared` bytes/WG grew. Pass `--*-lds` from the Triton cache metadata's `shared`
  field: the `.amdgcn`'s own `LDSByteSize` is a **structural 0** on Triton kernels, and the tool will
  tell you the LDS half of this suspect went untested rather than silently clearing it.
- `lost_RA` — the instruction multiset is unchanged and the allocator serialized it anyway. The
  signal is address **rematerialization**: an address recomputed into a register immediately above
  the `ds_read` that consumes it, which puts every read behind a WAR hazard on that one register.
  This is the row a layout-equivalence checker structurally cannot see — *equivalent layouts, equal
  counters, unequal address-register pressure.* Read the `ds_read` **operands**, not just the count.

Two more rules that are about the instrument rather than the kernel, both learned from results that
passed every other check:

- **A flat result set is a cache-collision suspect, not a finding.** Two variants differing only by a
  `gl.constexpr` layout constant share a Triton cache entry, and the second silently runs the first's
  binary. It is numerically perfect — every arm computes the right answer, just not with its own code
  — and it yields the most seductive possible artifact: arms that all tie, reading as a clean
  "the layout levers do not move the clock". Give each arm its own kernel object and its own
  `TRITON_CACHE_DIR`, expose `fingerprint()` so `ab_bench.py` can prove the binaries differ, and
  before recording any flat verdict run `--permute` and check whether the numbers follow the **code**
  or the **position**.
- **A tolerance comparison cannot fail on NaN.** `NaN > tol` is False, so an all-NaN output scores
  zero out-of-tolerance elements and prints ALL PASS. `ab_bench.py` now fails any non-finite metric
  and, given an `outputs()` hook, scans the tensors itself — one level below the adapter, because
  that is where the trap lives.

**Budget: the round count is the denominator, not a ceiling.** A 20-round budget spent as ~1 round of
wall clock is not a 20-round search; it is a 1-round search that reports a 20-round budget. Size the
wall clock to the rounds, and checkpoint every kept win (diff + metrics + that variant's private IR
dir) the moment it lands, so a hard stop is a pause rather than a discard. For calibration: on
`pa_decode` the winning levers landed at rounds 1, 8, 8, 9 and 10 with **five consecutive negatives
at rounds 2–6** in between. A loop that ends at round 1 cannot reach any of them.

**Two comparators, both carried in every result.** Correctness and layout equivalence are versus the
**anchor**; performance is versus the **champion**. `vs_anchor` alone hides the whole question — the
anchor is a regression you created, so beating it proves nothing. The same climb scores 1.19× or
0.85× depending only on which denominator is read, and the honest report carries both with the
champion one deciding.

**1. Transcribe, and drive it to layout equivalence.** Pin the tuned plain kernel, dump its IR, recover
the layouts, and iterate until `--verify` passes.

> **Follow [`references/phases/transcribe-runbook.md`](references/phases/transcribe-runbook.md) for this
> step.** It is the executable form — six numbered stages, one command and one decision each — and it
> carries two things this summary cannot: the **Apply** checklist (declaring a layout is not applying it;
> a body left on `AutoLayout` compiles, passes the oracle, and is several times slower), and the rule for
> **classifying each `ttg.local_alloc` before transcribing it** — with three corrections the vendored copy
> predates (all three in [`reference.md`](reference.md); the third, that the performance sign is
> kernel-dependent, is in `## Do-no-harm notes` below): classify against a **`ns=1` dump**, because at the
> shipped depth the staging is usually the
> pipeliner's rather than the author's; and `--verify` is blind to the choice **only where the layout is
> `UNRECOVERABLE`** (which is the case that advice was written for — an excluded layout's buffer can be
> dropped invisibly). On a recoverable shared layout it FAILs and names the missing `swizzled_shared`, so
> what it cannot see is allocation **size**, not the classification.
> Check the size with `scripts/probe.py measure` — compile-only, seconds, no GPU — as soon as the
> anchor builds, not after the first timing. Read **both** limiters it prints: registers and LDS each cap
> workgroups per CU, and quoting the LDS side alone will hand you generous headroom on a kernel that is
> register-bound, which is how an arm with the most apparent LDS slack ends up the slowest of a set.

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

**A fourth structural cause exists and `verify` reports it as `FAIL`, correctly**, because it is a genuine
expressibility wall rather than a reconcilable difference: a `tt.dot` on fp8 can be lowered by plain to an
**unscaled `tt.dot_scaled`**, and Gluon has no spelling for that — the plain mfma builtin rejects the
element type, while the scaled one always materialises a default scale operand, so the anchor is forced onto
a *different instruction* than plain's. This is instruction-level, not encoding-level: every layout
multiplicity can match exactly and the port still cannot be made faithful. If you see it, the finding is the
wall itself; report it upstream rather than reconciling it away.

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
`ttgir_bridge.py` began naming it — though whether it appears at all is **arch-dependent** (`ds_read_tr` in
`## Arch dispatch`), so re-dump before treating it as your blocker. And layout equivalence is blind to two
things that decide real ports — read them off `recover` rather than discovering them at the clock:

- **which operands each `buffer_load` actually carries.** `recover` buckets every site as bare /
  mask-only / mask+`other`, and you transcribe the bucket rather than the source: `tl.load(...,
  other=0.0)` frequently compiles to a load with **neither** — buffer OOB returns zero on CDNA — and
  passing `other=` in Gluon emits it and costs a `v_cndmask` per register. Adding a *mask* the compiled
  form never issued costs the same, which is why the buckets are detected rather than inferred. Two
  operands are **not reachable from `gl.amd.cdna3.buffer_load` at all**, `contiguity` and `stride`; the
  latter shows up only on the pipeliner's peeled prologue loads, which a non-pipelined anchor does not
  have and the injection puts back itself.
- **LDS allocation size**, which `verify` cannot see by construction. A shared total that crosses the
  LDS/CU divisor (per-arch — `## Arch dispatch`) halves workgroups per CU while every layout still
  verifies. Compare `recover`'s `LDS:` line against plain's — it names the divisor for
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

**On the middle row, read `spill=` before you write the report, because a pessimising depth is usually a
register wall and then the bug is the tile rather than the depth.** Measured on one card with only the tile
varying: the wide tile pessimised at `ns=3` (1.45× worse than its own `ns=1`) while the narrow tile gained,
and the discriminator was the spill column — the losing arm pinned at the 256-VGPR cap and spilled
hundreds of bytes per wave *inside* the loop, the winners spilled nothing. `WGs/CU by LDS` was 1 for both
and could not separate them. This also means **the sign of a debt does not transfer across generations for
arch reasons alone**: a narrower MFMA shape needs more instructions and keeps twice the dot-operand
registers live per K-tile, so the same source config can sit on either side of the wall on two gens.

**Find out WHICH `num_stages` knob your champion actually uses before you flip one — there are two, they
are not equivalent, and turning the wrong one manufactures a "no debt" verdict.** The rule has three cases
and only the middle one is obvious:

| the loop | what sets the depth | flipping the launch arg |
| --- | --- | --- |
| carries `tl.range(..., num_stages=N)` | **the annotation, outright** | **inert** — every launch depth compiles byte-identically |
| bare `range`, **with a dot** | the launch argument | works; it is the only knob |
| bare `range`, **dot-free** | nothing — no anchor to pipeline | inert, and so is the launch arg |

So the trap is a kernel that hard-codes `tl.range(num_stages=1)`: the three arms above come back
**byte-identical**, read as "no pipeline to lose", and the real depth is one token away in the kernel
source. Measured, that recovered a large win on a kernel this screen had written off. Conversely a
champion whose launch passes nothing may still be fully pipelined, and one that passes `num_stages=2` may
never reach the pipeliner at all. Read the annotation, flip *that*, and re-dump to confirm the depth moved
(load count and `memdesc_index` both scale). A depth frozen in source where the tuner cannot reach it is
itself a library bug worth reporting. `scripts/pipeline_survey.py <tree>` classifies a source tree by which
pipeline form each kernel can exercise; treat it as a **screen for what to measure**, not a verdict —
only a dump settles whether a given dispatch compiled pipelined.

> **`recover`'s pipeline verdict reads `tt.num_stages`; do not let a carry count overrule it.** The
> attribute is decisive. The `iter_args` count is not — an online-softmax or reduction loop carries
> accumulators for algorithmic reasons, so it reads ≥ 2 while un-pipelined, and one dump cannot separate
> the pipeliner's carries from the algorithm's. The tool asserted a positive off the carry count alone and
> mislabelled an attention champion that its own output showed at `tt.num_stages=1`; it now lets the
> attribute decide and says INCONCLUSIVE when the attribute is absent.

**2b. Three conditions, all required — none of them alone does anything.**

| # | condition | why |
| --- | --- | --- |
| 1 | **the loop needs an anchor, and the anchor is a `tt.dot` — not the loop syntax** | `add_schedule_loops(pm, ns)` takes the depth as a **pass argument** and reads no attribute off the loop. So a loop **containing a dot pipelines on a bare `range`**; a **dot-free** loop has no anchor and is the *only* case that needs `tl.range(..., num_stages=N)`, where `None` inherits the launch value |
| 2 | **the loads must still be `tt.load` when the pipeliner runs** | plain orders the pipeliner at #15/#16 and `add_convert_to_buffer_ops` at **#28**, so plain's own pipeliner only ever sees `tt.load`. Write `gl.load` and arm `buffer_ops=True` to restore that order |
| 3 | **the staging the pipeliner is asked to build must be its own** | a hand-written `allocate_shared_memory` + `gl.barrier()` body starves it, and to give the pass a loop it will take, that staging has to come out **entirely** — one `ttg.barrier` left in the loop makes the pass skip that loop wholesale. **Scope: this is per-operand, not per-loop** (see below) |

> **Do not generalise the 2×2 in `references/gluon/pipeline-reference.md`.** It reports
> `gl.load` + bare `range` as not pipelining, which is true **on the dot-free kernel it was measured on**
> and false with a dot in the loop. Two authors rewrote working bare-`range` dot loops into `tl.range`
> for no effect before that was caught. Condition 1 above is the rule; the table is one row of it.

On condition 3, the failure has a misleading error: `'ttg.local_alloc' op pipeliner doesn't know how to
predicate this op` is the **symptom of staging not removed**, not a language wall — the first
investigation to hit it concluded that Gluon's `allocate_shared_memory` was fundamentally incompatible
with the pipeliner, and that was wrong.

**Condition 3 is also narrower than "hand-written staging blocks the pass", and reading it as the broad
claim will make you skip a loop that would have paid.** What the pass needs is *one* dot whose operands it
can trace back to a `tt.load`; a **mixed** loop qualifies. Measured on an attention body with three dots:
the SSA walk back from two of them dies at a `local_store`, but the third is a pure register path
(`gl.load` → `convert_layout` → permute → `convert_layout` → dot), so the pipeliner fires on that one and
multi-buffers the rest of the loop along with it — visibly, the body went 3 dots / 104 `v_mfma` to 6 / 208.
That arm was the kernel's best result. So confirm the pass did nothing from the **IR**, not from the shape
of your source.
`buffer_ops=True` is **opt-in** because it fails three ways: on an anchor whose **loads** are already
`buffer_load` it aborts the pass manager loudly; on one whose **stores** are buffer ops it does not
raise at all — `LLVM ERROR: Fatal pipeliner error` kills the interpreter; and a single buffer op left
**outside** the loop is enough, because the rejecting pass (`TritonAMDGPUCanonicalizePointers`) runs over
the whole function rather than the pipelined region. Arm it only on a body written throughout — the whole
function, not the loop — with `gl.load` / `gl.store`.

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
> `OutOfResources: Required <n>, Hardware limit <the arch divisor>` — the injection succeeded and looks
> broken.
>
> **Where the ceiling is roomy enough to absorb it, the exception disappears and the cost does not** —
> injecting without the tail can be *worse than not injecting at all*, by a multiple rather than a
> margin, while compiling cleanly. Do not read the LDS growth as the cause: the tell is **register
> spill**, because pipelining without the buffer conversion leaves 64-bit pointer *tensors* live across
> the peeled stages. Occupancy can look untouched — only the `spill=` field moves — so **check the pass
> list, not the exception**, and check `spill=` rather than `shared`.
>
> **The pipeliner and the tail are a pair, and each half alone is a trap in a different direction.**
> Pipeliner without the tail is the case above. Tail without the pipeliner is worse than slow: it returns
> **NaN**, because `add_block_pingpong` assumes the pipeliner has run. Neither half is a valid arm.
>
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

**Depth is a knob, not a monotone, and the LDS cost is arithmetic you can do before you measure.** The
pipeliner builds a rotating stage of **`num_stages − 1`** buffers, readable straight off the TTGIR as the
leading dimension of the staged `memdesc` (`memdesc<Nx…>` at depth `N+1`, absent at depth 1). So each depth
past the first costs one more copy of the staged tiles, `ns=2` is single-buffered with a peeled prologue
rather than double-buffered, and any tile's whole depth series is predictable from one dump rather than
swept blind. Two ceilings then apply and they are **different limits with different symptoms**: the
per-workgroup **allocation** ceiling refuses the launch outright, while the **LDS/CU divisor** silently
halves workgroups per CU — both per-arch, `## Arch dispatch`. A depth can therefore compile and still be
the wrong depth, and the curve commonly turns at the divisor rather than at the refusal. `recover`'s `LDS:`
line predicts this before a clock is read, as an **upper bound** — it sums declared allocations without
modelling liveness reuse, so quote `probe.py measure` off the artifact once one exists. A depth plain itself
cannot compile is not available to you either. Start at 2 and sweep. Full mechanism, the per-shape
behaviour, the ping-pong window and why async copy is not reachable from plain on gfx942:
`references/gluon/pipeline-reference.md`.

**3. Pass through two checkpoints — neither of them is where the track stops.**

| checkpoint | what it is | when |
| --- | --- | --- |
| Layout equivalence + bit-parity | `--verify` PASS, and no numeric delta vs the plain anchor (transcription is layout-only, so any delta is a bug, not a Gluon property) | **exit condition of step 1 — always, never deferred** |
| **≥95% of the tuned plain kernel's throughput** — decided by `scripts/parity_gate.py` (exit 0), whose output goes in the round log | the port has actually landed | exit condition of step 2 — immediately, when 2a found no debt; once the injection is confirmed fired, when it did. **Exit 2 forbids step 4**: the round's outcome is `recovery`, and the next round closes the suspect the tool named. Reaching parity is not a win either — it is getting back to a number the front end already measured |

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
- **Async copy — available on both gens, and easy to misdiagnose as unavailable.** `gl.amd.cdna4.async_copy`
  lowers from **stock** `gluon_to_ttgir` on both entry points (`global_load_to_shared`,
  `buffer_load_to_shared`) at the direct-to-LDS widths the generation supports — `direct_to_lds_bit_widths`
  in `## Arch dispatch`. Three traps, in the order they bite:
  - **Each lane must make exactly one access of a listed width.** A layout whose per-lane contribution is
    the right size but **split across repetitions** — a blocked layout that covers more than the tile, so it
    repeats and every lane accesses twice — fails, and fails with the same `PassManager::run failed` /
    LLVM-translation wording as a missing op. So a width sweep alone cannot separate "unsupported here" from
    "asked for it wrongly", and a sweep that pins one `(blocked, shared)` pair will conclude the arch lacks
    the feature. Hold op, arch and width fixed and vary only the **tiling** before recording an arch
    ceiling; check the layout covers the tile exactly. The shared layout's `vec` is not the gate.
  - **`add_coalesce_async_copy` rescues non-native patterns; it is not what enables async copy.** Splicing it
    makes an off-width or repeated access legal by adding a bounce — which is a cost, and one easily
    mistaken for an intrinsic cost of the async path.
  - **A padded shared layout on an async arm can be a silent miscompile.** It removes the residual bounce
    and can be the fastest arm measured while returning NaN across nearly every element, with the identical
    sync kernel exact. Check numerics on any padded-shared async arm before believing its clock.

  The falsifiable signature that async actually replaced staging is `ds_write == 0` with a matching count of
  direct-to-LDS loads; sync staging pays a non-zero `ds_write`. Availability is not the same as value —
  the bypass can still lose to sync staging — so A/B it (`references/gluon/memory-reference.md ## Async Copy
  To Shared` states the layout contract; `scripts/pipeline_examples_cdna4.py` is the runnable check).
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
  `memdesc_index`. **Judge it from the ISA, never from a source config** — and do not try to earn it by
  satisfying `is_pingpong_schedule_enabled`. That predicate is not arch-symmetric, but the decisive point is
  that it is consulted **only inside `make_ttgir`**, which the Gluon entry point does not go through: meeting
  its condition from Gluon source buys nothing on either generation. Ping-pong on this path needs the
  schedule decision spliced, not the predicate satisfied.
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
- **Copying the champion's `waves_per_eu` across with the rest of its config.** It is not a hint: it
  reaches LLVM as `amdgpu-waves-per-eu` and **caps** occupancy outright. Measured, an anchor left without
  it ran at 3 waves/SIMD and was the faster arm; adding the champion's `waves_per_eu=2` left VGPR count
  unchanged and dropped it to 2 waves/SIMD and a slower clock. The throttle was tuned for plain's
  register-heavy *pipelined* body, which is not what an un-injected anchor is. Carry the tile shape and
  the depth over; leave this one off until it earns its way back in.
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
    decompose them against one shared conversion scratch. The second form can cost far less LDS for
    identical arithmetic — but **cheaper LDS is not the same as faster, and the sign is kernel-dependent,
    so this is a measurement and not a rule.** Three kernels authored both ways and bit-exact: two were
    **slower** as pass-through despite cutting `shared` by 3× and 8×, one was **faster**. The two
    mechanisms that decide it are both visible in the artifacts:
    - **barriers.** Letting the backend reuse one small scratch several times per iteration buys a
      full-drain `s_barrier` on each reuse. On the two that lost, the loop's instruction *multiset* was
      identical and only `s_barrier` moved (2 → 16 and 2 → 10). Compare `s_barrier` and `lgkmcnt(0)`,
      not just `shared` bytes.
    - **whether the saving is on the binding limiter at all.** On one of them registers pinned occupancy
      at 1 WG/CU either way, so a 57 KiB saving bought exactly nothing while the barriers cost real time.
    A third cost is arch-specific: on one kernel the pass-through form lost the hardware transpose
    entirely (`ds_read_b64_tr` 64 → 0). Read all three off the artifacts rather than assuming.
- **Writing a batch of variants without timing them is not a search.** Variants that differ only in
  ways the backend folds away measure the same, and without an A/B between them that is invisible,
  so a run can spend its whole budget producing them. Time each variant against the comparator as
  it lands and dispose of it; rejecting fast is where a round budget's value comes from.
- **Re-check the generation before you trust a figure or a digest.** `match.gens` claims both, the
  mechanics transfer, and several verdicts do not — including one where the *diagnostic itself* changes
  character, so the failure you were taught to wait for never arrives. Do not carry a divisor, a
  conflict-free swizzle or a layout digest across generations; the rows that can flip, and the
  `hw_constants.json` key for each, are in `## Arch dispatch` above.

## Sources

- Vendored from `AMD-AGI/TileProgrammingAgentSkills@541a180`
  (`.cursor/skills/tile-programming-gluon/`), **pruned to the API surface, the do-not-write lists and the
  two mechanics above**, plus the transcription runbook and the compile-only occupancy probe that step 1
  and step 4 depend on: 16 of 70 reference files and 11 of 51 scripts, plus `references/entry-modes.md`
  authored here. Upstream remains the SSOT and this is a one-way snapshot, so **re-syncing by overwrite
  would silently drop the additions below** — take them with you.

  `references/` **is no longer byte-identical to upstream.** Three vendored files carry GEAK additions,
  each of them a measured finding that belongs topically where it sits rather than in a parallel file,
  and each owed upstream:
  - `platform-known-issues.md ## warps_per_cta` — the mapping is restated in every layout in a
    hand-authored file, so editing `warps_per_cta` alone crashes the pass manager with an
    `iota_range` assert and **no attribution**, which reads like an arch verdict and is not one.
  - `gluon-negative-patterns.md ## Instruction count is not the objective function` — at 1 wave/SIMD
    whether VALU costs anything depends on where it sits, not how much of it there is, so a census-
    improving edit can be slower. Four measured rounds.
  - `gluon/memory-reference.md` — `ds_read_*_tr_*` is a saving only when the transpose is not already
    free; check the ISA first. Measured: −51 instructions, identical counts, **−1.4 %**.

  `scripts/` carries **ten corrections**, all of which should go back upstream:
  - `ttgir_to_gluon.py` dropped `tilesPerWarp` and `elementBitWidth` when emitting
    `gl.amd.AMDMFMALayout`, so a chained-dot (gfx950, 16×16 mfma) or scaled-MFMA kernel — both of which
    make `AccelerateAMDMatmul` choose non-default values — transcribed to a silently different layout.
    `--verify` caught it as a text mismatch but named no cause. Both are optional `AMDMFMALayout` fields
    in 3.6 through `main`; they are now emitted when, and only when, the TTGIR prints them, with a
    self-test on both directions. **The witness is still synthetic**, and not for lack of trying: config
    sweeps of a blockscale GEMM and of a minimal `tl.dot_scaled` MXFP4 kernel both failed to make either
    field print. The reason is mechanical rather than a search problem — `libtriton` exports
    `deduceTilesPerWarpForScale`, so the only route is a genuine `tt.dot_scaled` body: a kernel that
    upcasts to bf16 and issues a plain `tt.dot` cannot reach it, and neither can a chained dot on its own.
    Note also that `AMDMFMALayout` accepts **every** `(version, instr_shape)` pair at construction time and
    validates only at lowering, so constructibility is not an availability signal for a new shape.
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
  - `recover`'s **pipeline verdict** asserted "plain IS software-pipelined" on `max_iter_args >= 2` alone,
    independently of `tt.num_stages`. That is a false positive on any accumulator loop, and it fired on an
    attention champion **in the same report that correctly printed `tt.num_stages=1`** — every
    online-softmax body carries `m`/`l`/`acc` plus an offset. The verdict decides which control to measure
    and therefore whether a shortfall reads as a debt or a defect, so a wrong positive sends the author to
    re-inject a pipeline that was never there. The attribute now decides; a carry count with no attribute
    reports `INCONCLUSIVE` instead of guessing; and the `num_stages=1` branch now points out that a depth
    pinned by a source annotation may still be reachable. Four `--selftest` cases pin both directions.
  - `probe.py` printed its `WGs/CU by LDS<=` line from the module-level `LDS_PER_CU = 65536` — the
    constant its own comment labels "fallback only; prefer `lds_per_cu(<arch>)`" — instead of the
    arch-dispatched helper defined immediately above it, and `plan`'s closing LDS note did the same. This
    is the fourth correction's bug in the newly vendored file, and it lands hardest on the generation the
    probe **defaults** to: `measure` already resolves the arch per kernel and prints it, so on CDNA4 it read
    every kernel above the *CDNA3* divisor as `WGs/CU by LDS<=0` — unschedulable — while naming the right
    arch on the line above. That is the whole class the larger CDNA4 ceiling exists to admit, so the probe
    named a blocker that does not exist, in the one step the runbook says to treat as the whole result
    *before* any timing. Both paths now take the arch (`measure` from the artifact,
    `plan` from `--arch`) and `measure` prints the divisor and its basis so the figure is auditable.
    `probe.py` and `amd_occupancy.py` both still carry the same unreachable-after-`return` remnant
    referencing a `LDS_PER_CU_BY_ARCH` that does not exist; left as found, since reaching it is what
    would turn it into a `NameError`.
    The same file also reported the LDS side as if it were *the* occupancy answer, which is the
    other half of the same mistake: registers and LDS both cap workgroups per CU, and on a kernel
    where they disagree the LDS figure alone points the wrong way. On a gfx950 kernel whose arms were
    uniformly register-bound at 1 WG/CU, the LDS line advertised an order of magnitude more headroom —
    and the arm holding the most of it, having traded a large LDS saving for extra barriers, was the
    **slowest** of the set. `measure` now prints both sides and names which one binds,
    turning the register limit into a workgroup limit the way the hardware does
    (`waves/SIMD x simds_per_cu / num_warps`); `num_warps` was already in the metadata it reads,
    and `simds_per_cu` comes from the same per-arch reference as the divisor.
  - **The LDS/CU figure had three more ways to be confidently wrong, and the first two are the same
    bug as the one above, one level down.** (i) `amd_occupancy.py`'s lookup ended in an unbounded
    recursive `glob` from `HERE/../..`; with the scripts copied to a shallow directory — which
    `USAGE.md`'s own "run the selftests on a new box" step invites — that resolves to `/` and walks
    the whole filesystem. It does not fail, it **hangs**. Replacing it with an unbounded walk *up*
    the ancestors only trades the hang for something quieter: it accepts any `references/hardware/`
    it passes, so a stray `/tmp/references/…` becomes the source of truth — observed while
    reviewing this very fix. The lookup is now bounded at both ends and anchored on the package's
    own `skill.md`. (ii) `probe.py` and `asm_loop_audit.py` both defaulted `lds_per_cu` to **65536**,
    the CDNA3 figure, so with no reference tree reachable a gfx950 kernel was measured against the
    wrong divisor and reported `0 WGs/CU, LDS bind` — a hard blocker on a kernel that runs. Both now
    return `None` and say so; `reference.md` already promised the LDS half reports *nothing* without
    the json. (iii) `measure` collapsed `waves/SIMD` to one value per directory, so it dropped the
    register limiter entirely as soon as a dump held two kernels with different occupancy — which is
    the normal case, since a harness' pack / split-K kernel compiles into the same cache dir. It is
    now joined per kernel by name, and "no LDS at all" is printed distinctly from "no divisor
    known", two states it used to conflate into a `TypeError`-shaped comparison. The row logic is
    factored out and pinned by six `--selftest` cases, and the lookup bound by two more.
  - `dump_ir.sh` accepted `--kernel <bare-name>` and silently did nothing with it. The two flags are
    not interchangeable — `--kernel` is `module.path:object` for the translator, `--kernel-name` is
    the substring that **pins** which compiled kernel gets copied — and `references/phases/transcribe-runbook.md`
    tells you to "pass `--kernel <substring>` to pin it" in the very sidebar that warns a multi-kernel
    op will otherwise hand you the wrong body's layouts *silently*. Verified: `--kernel NOPE` exits 0
    and copies the freshest artifact anyway. So following that line leaves you unpinned inside the
    hazard it is describing. `--kernel` without a colon is now a hard error naming the flag you meant.
    The runbook line itself needs the upstream fix; this is the guard that makes the mistake loud here.

  `ttgir_bridge.py` and the added scripts are `ruff --no-cache` clean; the pre-existing findings in the
  vendored ones are untouched (20 across the older four, 12 in `probe.py` — including the `F821` for the
  dead `LDS_PER_CU_BY_ARCH` above, left unreached on purpose). **All ten offline `--selftest` entry
  points in `scripts/` now run from `scripts/smoke_test_recover.sh`**, so every rule above is guarded
  without a GPU or a `triton` import. **Four** of them were reachable only by hand before — including
  `amd_occupancy.py`, which is where `probe.py` delegates the divisor and whose self-test cross-checks
  `hw_constants.json` — and that is how both the LDS/CU divisor and the pipeline verdict shipped wrong.
  `gluon_swp` self-skips where no AMD backend imports, by design.
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
