# Entry: settle the mode, assert the comparator, then decide whether to start

**What this stage decides.** Three things, in this order, before any transcription and before any
round: (1) whether there is anything here worth the explicit tier at all — the front porch and the
escalation seam; (2) which **entry mode** you are in — a converted anchor that owes a transcription
debt (A), an incumbent that owes none (B), or a resumed run (C) — because that decides which gates
apply and what shape the round loop must have; and (3) whether the comparator every later number is
quoted against can support a claim — `champion_gate.py`, the champion assertion.

**When you are here.** At the start of every run and on every resume. Nothing downstream — the
anchor, the parity gate, the climb — means anything until this stage has passed, and getting the
mode wrong loses a run before the first real round, in opposite directions depending on which way
you got it wrong.

Paths below: `$SKILL` = `perf_knowledge/expert_skills/skills/gluon_authoring`, and
`$KT` = `kernel_workflow/scripts/kernel_tools` (GEAK's shared kernel tools — `dump_ir.sh`, `probe.py`,
`amd_occupancy.py`, `asm_loop_audit.py`, `hw_budget.py`, `capture.sh` live there now; the old
`$SKILL/scripts/<name>` paths remain as shims). Examples use `--arch gfx950`; gfx942 differences are
called out as downgrade notes.

---

## 1. Where entry sits in the stage spine

```
Stage-Entry      assert the champion bundle. No transcription, no rounds, until it passes.
      |
Stage-Anchor     transcribe the champion's OWN TTGIR into explicit Gluon layouts.
      |          Faithful by default, so a regression vs the champion is EXPECTED -- a debt
      |          you took on knowingly, not a floor you climb from silently. Divergence is
      |          allowed in named grades, and an elective one owes the faithful number too.
      |
Stage-Recover    pay that debt back, attributed: split the anchor->champion gap into
      |          lost-pipeline / lost-layout / lost-RA and close each with its own mechanism.
      |          GATE: apply the run's declared parity criterion before any delta is called a win.
      |
Stage-Climb      Layer 1.5 -> 7, one coupled layer per round, CLIMB only.
                 target line = champion_ms. A structural wall returns structure_suspect.
```

Anchor: `transcribe.md`. Recover + parity gate: `recover.md`. Climb: `climb.md`. On a mode-B entry
Stage-Anchor and Stage-Recover are a **defined no-op** (Stage-Entry below).

**Budget comes before authoring.** The roofline/budget read (`budget.md`) and the
bound-class portrait are inputs to the escalation decision below and to the anchor's recalibration,
and every round re-profiles (`profile.md`). An earlier GEAK rule — "do not put a full profiling pass in
front of the transcription; profile once the port has landed" — is **overruled** by that order: the
pre-port profile is the comparator's, it is what the escalation seam reads, and the anchor is
re-profiled anyway as the first act of Stage-Recover.

---

## 2. Before you commit: the front porch (Stage-Diagnose)

Stage-Diagnose answers *"what is this kernel bound by, is there real headroom, and is it worth
optimizing?"* — it does **not** change the kernel. Many real tasks stop here (measure / explain /
decide), and the ones that continue hand a clean bound-class picture to the escalation seam (§3).

### When to use Stage-Diagnose

Use it when the task verb is **measure / profile / explain / compare / verify a hypothesis / decide
if worth optimizing** — not "make it faster". Examples: build an N-arm benchmark, draw a roofline,
explain why a curve is non-uniform, check whether a proposed knob helps, judge whether a region has
headroom.

If the task is "make it faster" and a region is already known to be layout-bound with headroom, skip
Stage-Diagnose and go straight to the escalation seam (§3) and the champion assertion (Stage-Entry, below).

### What Stage-Diagnose includes / skips

Includes (lightweight):
- benchmark hygiene + **measurement validity** (`benchmark-hygiene.md` — repeatability +
  measurement order, same-knob baseline): same-session interleave, min-over-repeats, noise-floor
  gate, no stored cross-run baseline. Timing and correctness are GEAK's
  `e2e_workflow/scripts/harness_lib.py` (CUDA events, per-sample sync, read-evict flush, median);
  run under `kernel_workflow/scripts/gpu_lock.sh`;
- analytical roofline + per-region bound-class portrait (below; `budget.md`);
- **anomaly validation** for any trough/spike before believing it (`records.md`, anomaly
  validation);
- optional **cheap micro-experiments** via the Experiment contract (`records.md` —
  `9. Experiment / hypothesis-test contract`), run by the deep_engineer itself under `gpu_lock.sh`
  (`orchestration.md`);
- presentation rules (below).

Skips (Stage-Diagnose does not optimize): the full Task Contract (`records.md`, `1. Task Contract`),
transcription, the Gluon backbone, and three-evidence layer closure. Those belong to the plain front
end or the Gluon stages once Stage-Diagnose says "escalate".

### Lightweight contract

Lighter than the full Task Contract — pin only what a diagnosis needs:

```text
target:            repo / kernel file / kernel name / wrapper
boundary:          eager | CUDA-graph kernel-only  (match every measurement)
config/served set: expanded across the served range (anti-overfit), not just seeds
per-region metric: compute -> %fp8(/bf16)-peak ; memory -> %HBM-BW-peak  (do NOT mix)
oracle:            correctness reference + tol
profiler status:   available | DEGRADED (counters hang/unavailable -> see below)
question + threshold: the decision this diagnosis must return, and the bar for "worth it"
```

Every "% of roofline" this produces carries a `numerator_basis` (model | counters) and a
`denominator_basis` (datasheet | empirical@tool-version | in-shape probe); a datasheet denominator
ranks but cannot gate or close (`budget.md`).

### Degraded-mode classification (no profiler)

When rocprof / PMC is unavailable (e.g. it hangs under CUDA graph on the build), do **not** stall —
classify bound class from cheaper, still-valid evidence:

- **analytical AI vs ridge**: compute essential-traffic AI (FLOP/byte) and compare to the ridge
  (`../hardware/roofline-models.md`); AI << ridge -> memory-bound, AI >> ridge -> compute-bound,
  straddling -> region-dependent;
- **achieved %peak from latency**: from kernel-only latency + essential traffic, derive achieved
  HBM-BW% (memory regions) or achieved TFLOPS / %compute-peak (compute regions); the gap to the
  ceiling is the headroom;
- **asm hot-loop audit** (`$KT/asm_loop_audit.py`): op-class histogram + `s_waitcnt` / `s_nop` /
  barrier pattern confirms whether the loop is pipelined or serialized and whether MFMA vs memory
  dominates;
- **latency-vs-size scaling shape**: how latency scales with M/token/parallelism separates
  launch/occupancy-bound (flat at small size) from BW-bound (linear in bytes) from compute-bound
  (linear in FLOPs).

Record `profiler status: DEGRADED` and which of the above carried the classification; a
degraded-mode classification is sufficient to set the escalation decision.

### Two regions that should be identical report different op counts

A specific, cheap signal, and worth naming because the symptom points the wrong way. When two loop
regions built from the same source (the two halves of an unrolled body, two structurally identical
compute clusters) report **different non-matrix op counts** to a schedule tool, the difference is
almost always **something being counted that never gets issued** — a value the backend folds into
its consumer as a source modifier, or a reduction step it fuses pairwise.

Why it matters more than the count itself: a scheduling **declaration** that names an instruction
which will not be emitted cannot be satisfied, and the group-pipeline solver then abandons the
schedule for the **whole region** and falls back to instruction-selection order. The observable
result — matrix ops emitted back to back with the vector work stranded around them — is
indistinguishable from "the scheduler did not try", so it gets misread as a scheduler ceiling and
attacked with more kernel edits.

Check: count what the **assembly** issues, not what the IR contains, and compare the two regions
(`$KT/asm_loop_audit.py`). Mechanism and the discipline that avoids it:
`../tile-programming/llir-codesign.md ## Declaring the interleave`.

### Presentation rules

- **Use the right scalar per region** — compute-bound -> %fp8(/bf16)-peak; memory-bound ->
  %HBM-BW-peak. Never average a compute metric and a memory metric into one number.
- **Report ranges, not geomean** — a geomean over a size sweep hides the spread that carries the
  story; show min-max (and the peak cell).
- **Pick the view that matches the AI spread** — a roofline scatter is right when AI varies across
  the region; for a narrow-AI regime (e.g. decode, AI ~ 2-4) all points collapse into one band, so
  use a **per-token (or per-size) speedup / %peak line** instead, where the per-arm gap is readable.
- Split unlike kernels/regions into separate charts rather than overplotting.

### Output + decision flow

Stage-Diagnose returns a **bound-class portrait** (per region: AI, achieved %peak, bound class,
headroom), any **validated** anomalies, cheap negative results, and one decision:

```text
escalate     -> a region is layout/memory/compute-bound with real, validated headroom
                -> the escalation seam (§3); the bound-class portrait is its input
stay-plain   -> headroom is small or owned by wrapper/dispatch/launch/algorithm
                -> front-end.md, no Gluon
not-worth-it -> no validated headroom (or the "anomaly" was measurement/model-explained)
                -> stop; record the negative with numbers
```

---

## 3. The escalation seam: plain Triton vs explicit tile programming

The seam between the **broad-search front end** (`tile-programming-triton` upstream; in GEAK,
GEAK's own plain-Triton rounds) and the **deep-dig back ends** (`tile-programming-gluon`, this
skill; `tile-programming-flydsl`). Both sides read it, from opposite directions:

- **From the front end** — "is there anything here worth handing off, and have I earned the right to
  say so?" That is §3.1 through §3.9.
- **From a back end** — "what did I inherit, what is already disposed, and which plain levers have no
  counterpart in my tier?" That is §3.10 onward.

It follows **EXHAUST-BEFORE-ESCALATE guidance**: normally hand off only when the plain tier is
genuinely exhausted — its useful structural BRANCH bets are measured and its #1-bound plain levers
were tried to a credible wall — or when the budget gap is **provably explicit-layout-only**. Search
completeness is model judgement reviewed by the deep_engineer's closure self-review
(`close.md ## Closure challenge (self-review before a ceiling / keep-baseline / negative close)`), not a
hard gate. The failure this exists to prevent: hand off eagerly, hit the first wall in the
deep-dig tier, then claim "plain was the ceiling" without ever having established that.

### 3.1 What is mechanical and what is judgement

Exactly one thing at this seam is machine-checked, and it is a **fact**, not a search-policy call:
the champion bundle. `champion_gate.py` (run by the receiving pack before its first round, Stage-Entry)
asserts that the champion source still hashes to what was measured, that its `.ttgir` was dumped at
the pinned config, that `champion_ms` beats the kernel's own `default_ms`, and that the config sweep
was fully sampled (and more — §5.2 lists all fourteen checks). Contract: §6.

Everything else here — whether the structural screen was broad enough, whether the #1-bound plain
work reached a real wall, whether the layout-only argument is persuasive — is **judgement**, reviewed
in the closure self-review (`close.md`). Encoding it as a gate would only teach the search to satisfy the gate.

**The one close-side rule that is absolute: a deep-dig result that LOSES to the champion is not the
winner.** A `winning_stage: gluon` claim at `vs_plain < 1.0` is invalid — record
`negative_revert_plain` (keep plain) or `status: partial`. This has been recorded wrongly in practice,
at `vs_plain = 0.33` (3× slower than plain, after eight flat rounds), which is why it is stated as a
rule rather than left to good sense.

### 3.2 Sizing the handoff: TWO measurements, multiplied

**A mechanism measured out of situ is not a prize.** Measure both, and hand off on the product:

```text
prize  =  isolated-path mechanism value   x   in-situ exposure ratio
```

An isolated mechanism measurement must be decomposed at the production boundary: measure its in-situ
exposure and host/launch contribution before ranking a handoff. Do not convert an isolated delta
directly into an end-to-end prize.

**A parity result closes the question when the binding resource is one the deep tier does not own.**
This is an informative outcome, not a failed attempt. If current evidence shows that the remaining
constraint belongs to a resource outside the deep tier's control, record the resource, evidence, and
scope; do not invite an unbounded retry. Record it as `stay_plain_basis: measured_at_parity`.

Also on record from the same arm, and worth knowing before it costs a window: that Gluon path reached
parity via a **process-global monkey-patch** of the backend lowering hook. **Two differently-patched
variants must never be co-loaded in one process** — the second import wins and both then silently
measure the same pipeline. One variant per process (`recover.md`, last-resort section).

### 3.3 Plain-exhaustion precondition (reviewed at close)

Before handing off, prefer ONE of:

- **(a) plain exhausted on the #1 bound:** the useful plain-side BRANCH bets are measured and the #1
  bound's native plain levers were tried to a credible wall; OR
- **(b) provably explicit-layout-only gap:** the budget gap is addressable ONLY by explicit LDS /
  async pipeline / MFMA-operand-layout / register-slicing control that plain cannot express — the
  fast-path in the decision table. (Keep the migration-radius caveats: a fusion that is merely "same
  algorithm, fewer launches" with no layout sub-gap stays plain.) **layout-only is a PROVEN claim, not
  "I think the rest is layout"** — record evidence such as a walled plain lever, or a probe showing
  plain cannot express the layout.

A `winning_stage: plain` close recorded AFTER a handoff should independently establish (a): a wall in
the deep-dig tier does NOT retroactively prove plain is the ceiling. The closure self-review checks
whether the deep-dig wall was mislabeled as plain-is-ceiling.

**GEAK's two state-of-the-source checks belong here, before any authoring cost is spent** — they are
what this skill's entry actually turns on, more than the operator or regime:

- **Is the plain side actually finished?** If a config sweep has not run and its winner is not
  pinned, the first "Gluon win" is the sweep's. Where the shipped `num_stages` is itself a
  pessimisation that can be most of the headline, so measure `plain@ns=1` too and quote both
  (`recover.md`, the `plain@ns=1` control).
- **Is there a layout-shaped residual left?** Read it from the TTGIR: if the champion stages nothing
  through LDS, the two Gluon-only levers — swizzle/padding choice and LDS dedup — have no operand to
  apply to, and the port has to pay for itself some other way.

**Shape class is an admission hint, not a filter.** Decode, skinny-M, paged/indirect access and
"memory-latency-bound with no tile structure worth re-laying-out" are screened with the upstream
**Quick Reject** list (`../pitfalls/negative-patterns.md ## Quick Reject Checklist (= gate skip-transcription conditions)`, and `## Paged / Indirect Decode`). Read a hit there as *expect to
stay plain unless the evidence says otherwise*, not as an exclusion: the same kernel from the same
starting point has both cleared the bar and fallen well below its own anchor depending only on how
the transcription was done.

### 3.4 Gate input (no transcription needed)

Run on **plain Triton aggregates** — read the plain `.ttgir`, compile stats, and ATT; do not
transcribe to "find out" the budget. Roofline suggests the **bound class** (`profile.md`), but it is
advisory: override it with evidence when the live ATT/roofline disagrees. Then read the selected
class's **ideal-vs-as-built budget gap** (`budget.md`).

### 3.5 Bound-class decision

Read the gap **in the relevant bound class** — NOT MFMA-eff alone, which is only the compute-bound
case:

| Bound class | "gap small -> stay plain" signal |
| --- | --- |
| compute | MFMA-eff near-peak (small gap to budget) |
| HBM | achieved BW near the 32 KiB/CU TCP cap |
| LDS | `ds_read` interval ~16 cyc (conflict-free) |
| register | `R_total` within budget, no hot-loop spill |
| latency | pipeline coverage ~complete — and see the note below, because on this axis a *working* auto-pipeline is a stay-plain signal rather than something to reproduce after escalating |

- **Stay plain** when the relevant-class gap is small (the signal above) OR the gap is owned by
  wrapper / dispatch / launch / algorithm.

> **On the latency axis specifically: the compiler auto-pipeline working is a reason to STAY, not a
> thing to re-create on the other side.** The pipeliner runs in `make_ttgir` and not in the Gluon
> lowering, so escalating a kernel whose overlap the pipeliner already handles means arriving
> somewhere it no longer runs and then spending a round putting it back — with plain parity as the
> ceiling of that round. Production draws exactly this line and draws it *within* one kernel family
> — but it does **not** draw it as a monotone threshold, and reading it as one is the trap here.
> In one surveyed gfx950 family the plain shapes are M ∈ {1, 4, 16, 64, 256} with `num_stages` ∈
> {2, 3} and no hand-written ring, while M ∈ {2, 8, 32, 128} and everything above go to Gluon: the
> two alternate bin by bin. In a second family of the same pack the plain set is M ∈ {16, 32, 128,
> 256} and M ∈ {1, 2, 4, 8, 64} are all Gluon — **the same alternation with a different phase, so
> there is no common small-M pattern to carry across families.** The one stable observation is at
> the top: across surveyed production source every M ≥ 512 shape and every prefill shape is Gluon.
> M = 1 is worth its own note: it has no K loop, so it carries no `num_stages` at all rather than a
> low one.
>
> **Do not use token co-occurrence to decide whether the pipeliner is running.** Production leaves
> dead knobs behind: a substantial minority of surveyed Gluon files pass `num_stages`, including
> some in the very family just described — its *large*-M shapes, the ones with the hand-written
> ring. On the Gluon path that argument is inert (`add_stages` runs in `make_ttgir`, and the Gluon
> lowering does not call it; in Triton 3.8.0 no pass on the Gluon path consumes `num_stages`), so the
> token tells you nothing: it neither proves a pipeliner ran nor proves one did not. Only the
> language split does.
>
> So the escalation question on this axis is not "is there a latency gap" but **"is the remaining
> gap one the pipeliner cannot express"** — per-tensor depth, staggered async chains, sub-buffer
> splitting. If it is not, the answer is a `num_stages`/`BLOCK_K` sweep in plain, not a tier change.
>
> **The one mechanism on this axis that plain genuinely cannot reach** is the wave-level phase
> offset: `warp_pipeline_stage`'s grouping pass runs only in the Gluon lowering, so a marker emitted
> from plain Triton is never grouped. That is a positive escalation reason where it applies — and it
> applies narrowly, needing two resident waves (`num_warps >= 8`, the layer-1.5 Gate 0) and a stage
> body free of waits (`../tile-programming/scheduling-model.md ## How to choose`).

- **Hand off (once plain is exhausted)** when the gap is addressable by explicit layout / memory /
  pipeline in ANY bound class (low MFMA-eff is only the compute-bound instance) — but ONLY after the
  plain-exhaustion precondition holds, OR the gap is provably explicit-layout-ONLY (the fast-path).
  This is NOT eager-by-default; a shallow plain stage may not escalate.

> **GEMM: check the dispatch/config-owned gap FIRST (cheapest, often biggest).**
> Before reading this gate toward a layout attack, route dispatch/config through `plain_autotune.py`
> (triton pack; from gluon this is the SWEEP that should already have happened upstream — in GEAK,
> GEAK's plain rounds) as a single SWEEP, not per-round CLIMB work: **split-K** (K large & grid
> < ~a few waves/CU), **`GROUP_SIZE_M`** (L2 reuse), **`BLOCK_M`** (wave-quant / tail). A **shipped
> tuning table may be `tuned=False` and badly under-fill the machine**, so the largest GEMM win is
> frequently a config swap, not layout/pipeline — and a headline quoted against the shipped default
> over-credits the layer work (`recover.md ## Two baselines`, `front-end.md`). Route split-K /
> `BLOCK_M`-shrink via the machine lever cards (`../hardware/lever-cards.json` group `dispatch`);
> their verify signal is a **dispatched-WG / occupancy delta** (grid → CU multiple, MFMA% up), which
> is an acceptable IR/asm-equivalent signal for a structural dispatch win. Only after the dispatch gap
> is closed do you read the remaining gap for a layout/pipeline handoff.

### 3.6 Stay plain -> front-end.md (NOT wrapper-only)

When the guard says stay plain, route to `front-end.md` and apply the *applicable* plain directions
by the 0-15 priority ladder (algorithm / fusion / config-autotune / shape-dispatch / wrapper).
Wrapper-only is just the truly-near-peak sub-case; a stay-plain verdict is **not** "do only wrapper
work". The structural BRANCH fan-out over diverse plain hypotheses is the legitimate search here
(in GEAK, tech_lead's parallel specialist directions in the plain rounds, `orchestration.md`). The plain run also sets the **target line**
(`champion_ms`) that any later deep-dig result must beat.

**Structural stay-plain wins owe the IR leg too.** A stay-plain win is normally closed on two
evidence (budget + profile). But if the win changes a **structural knob** — one that alters the
lowered pipeline / prefetch / vectorization (`num_stages`, `waves_per_eu`, `PRELOAD_V`, a `BLOCK_*`
that flips the software pipeline, async-copy toggles) — its report **claims a structural change**, so
it owes the **third (IR/asm) evidence**: dump the TTGIR/asm (`$KT/dump_ir.sh`) and confirm the
claimed structure is actually present (e.g. `num_stages=2` ⇒ the cross-iteration SW-pipeline /
multi-buffer appears; `PRELOAD_V` ⇒ the hoisted V load) **before recording the win**. A win that is
purely launch / dispatch / shape-selection (no structural claim) is exempt. If the dump is genuinely
unavailable, record the exact reason in the Final Delivery `caveats` — never leave the structural
claim unverified and silent.

**A "plain is the ceiling" verdict requires comparator gap-decomposition evidence.** `stay_plain` is
a claim that the remaining gap is *not* addressable by escalation. That claim is auditable ONLY with,
on disk:

1. the **bound-class profile** of the tuned-plain kernel (rocprof counters → bound class), and
2. a **gap decomposition vs the comparator named in the contract** (the stretch / reference kernel
   the contract sets as the target line — e.g. a vendor asm kernel): the same-boundary A/B latency
   gap, PLUS an **asm/ISA hot-loop audit** of BOTH kernels (`$KT/asm_loop_audit.py`) attributing the
   gap to a named cause (MFMA/VALU overlap, occupancy, LDS conflicts, scheduling) — i.e. *why* plain
   trails and *why* explicit control would or would not close it.

Without both, the decision is **`unproven`, not `stay_plain`** — record it as such and keep the
direction OPEN (escalate or gather the evidence). Rationale: the most expensive failure mode is
declaring "no headroom above plain" from plain aggregates alone, never having measured the comparator
— the gap may be a layout/pipeline sub-gap explicit control *would* close.

A stay-plain / at-ceiling / negative close should carry measured, on-disk evidence and credible
verdicts for the native levers that matter to the #1 bottleneck. One crude knob's failure does not
establish the whole dimension while register-relief bets such as `reduce_accumulator_traffic` /
`tile_slicing` / `lds_dedup` remain plausible. If the comparator kernel is a prebuilt `.hsaco`/`.co`,
audit it via `llvm-objdump` + `asm_loop_audit.py` (objdump label form supported). If no comparator
exists in the contract, say so explicitly in the record — the verdict then rests on the budget gap
alone, and that limitation is named.

Concrete stay-plain signals (full backing: `triton-negative-patterns.md` in the triton pack, and
`../pitfalls/negative-patterns.md ## Quick Reject Checklist (= gate skip-transcription conditions)`):

- `tl.dot` already lowers to MFMA at high efficiency (small compute gap);
- the bottleneck is wrapper / allocation / dispatch / shape selection;
- the kernel is launch-dominated (all shapes ~<=30-50us, latency barely scales with compute — see the
  sub-ms table in `benchmark-hygiene.md`);
- explicit layout would only add `convert_layout` / register pressure;
- tuned plain Triton is more stable across the shape stream.

A stay-plain verdict on an **algorithm / fusion** lever is a **rebuttable default** — re-test it on
the fused budget (§3.7), not a terminal exclusion.

### 3.7 Rebuttable stay-plain for fusion/algorithm

A "stay plain" verdict on an **algorithm / fusion lever** (P0/P2) is a **default to re-test, not a
terminal exclusion**. After the plain lever lands, re-read the *fused configuration's* decomposable
budget and apply one precise criterion:

- **Escalate that lever iff explicit layout would remove a `convert_layout` / LDS round-trip / launch
  that the plain implementation leaves** — decide by the layout/operand/staging sub-gap, **NOT** by
  "plain underperformed". The explicit-tier-superior sub-case is producing a producer's output
  (operand / scale / epilogue input) **already in the consumer's layout**, so the consumer skips a
  GR -> LW -> LR staging that plain cannot avoid (plain does not control the operand layout).
- **Migration-radius caveat — do not over-escalate.** Most producer-consumer fusions still stay plain:
  fusing a producer into an explicit matmul body forces its output into the consumer's
  `DotOperand`/shared layout while masks / atomics / online-state want different parent layouts — a
  whole-kernel rewrite (`../pitfalls/negative-patterns.md ## Fused / Block-Scaled Matrix Kernels`,
  `## Migration Radius Too Large`). Only the small-radius, operand/scale-layout-coupled sub-case
  escalates; a fusion that is merely "same algorithm, fewer launches" with no layout sub-gap stays
  plain.

### 3.8 Decision table (stay plain vs hand off)

| Evidence | Action |
| --- | --- |
| relevant-class gap small (near budget) | stay plain; record near-peak |
| gap owned by wrapper / allocation / dispatch / shape selection | stay plain (`front-end.md` P15/P6) |
| launch-dominated (all shapes <=~30-50us) | stay plain; wrapper / fusion / boundary only |
| algorithmic gap (split-K, fusion, reduction tree) | stay plain (`front-end.md` P0/P2) **as a rebuttable default**; escalate that lever iff explicit layout would remove a `convert_layout` / LDS round-trip / launch the plain implementation leaves (§3.7) |
| `tl.dot` already near-peak MFMA, remaining gap is config | stay plain (P8 config) |
| gap addressable by explicit LDS / async pipeline / MFMA layout / scaled MFMA / register slicing | **hand off** — but ONLY once the plain-exhaustion precondition holds: structural BRANCH screen MEASURED + #1-bound plain levers tried to a wall (a), OR the gap is provably explicit-layout-ONLY (b, the fast-path). A shallow plain stage may NOT escalate |
| explicit layout would only add `convert_layout` / register pressure | stay plain |
| tuned plain more stable across the shape stream | stay plain; keep any explicit attempt as evidence only |

### 3.9 Hand off when the headroom needs explicit control

- bank-conflict-free explicit LDS layout;
- async global->LDS pipelining (`buffer_load_to_shared`; gfx950 main line — on the gfx942 downgrade
  direct-to-LDS lowers only at 32 bits per lane with a destination `order=[1, 0]`);
- explicit MFMA operand/result layout or scaled MFMA (a8w8 / a4w4; scaled MFMA is gfx950-only);
- register / slicing budget control;
- instruction scheduling / compiler-contract to reach MFMA continuity;
- an ordering, a machine-state bit, or a wave-collective the plain abstraction does not spell — the
  explicit tier has a single door for it (`gl.inline_asm_elementwise`, all four versions), so this is
  a handoff to **Gluon** and not a reason to escalate past it to raw HIP. **Only when the need
  survives the "already expressed" check**: cache-scope bits, an L2 writeback before a publish,
  priority around a matrix block and waiting on async copies each have a language spelling already,
  and reaching for asm for one of those is the recorded mistake, not the trigger
  (`../gluon/inline-asm-reference.md`, opening section).

**Which back end.** `tile-programming-gluon` is the same toolchain, so it *recovers* the champion's
layouts from its own TTGIR — the cheapest path, and the default. `tile-programming-flydsl` is a
different toolchain, so it *ports* the champion's structure by hand from `ttgir_facts.json`; choose it
when the target needs a lever Gluon does not expose, or when a FlyDSL implementation already exists.
Both consume the same bundle and both assert it with `champion_gate.py`.

### 3.10 Priority ladder -> layer backbone mapping

The 0-15 priority ladder (`front-end.md`) maps onto the explicit-tile backbone. **Read this from the
back end's side as "what is already disposed"**: everything on the left is settled in the champion.

| plain priority | in the explicit tier |
| --- | --- |
| P0 algorithmic (split-K, decomposition) | decided in the front end; becomes split-K / stream-K scheduling |
| P2 fusion | epilogue / producer-consumer fusion (stays plain by default; escalate only the operand/scale-layout-coupled sub-case — §3.7) |
| **P5 memory/compute reorder** | **the main handoff trigger** -> memory-path + LDS + pipeline layers |
| P6 shape-adaptive | visible dispatch (stays in the front end) |
| P8 autotune/param | within-layer config, but **budget-derived, not a blind sweep** |
| P15 wrapper-only | always stays plain |

#### What does NOT map plain -> Gluon

The table is not a clean 1:1 — some plain levers have **no** counterpart in the explicit tier, or an
**inverted** one. Do not "autotune in Gluon" or "fuse everything in Gluon":

- **P6 shape-dispatch + P15 wrapper are host-tier.** The explicit tier does not touch them; P6
  shape-dispatch is a host-side visible dispatch (`close.md`, multi-shape no-regression + dispatch),
  and wrapper / launch work always stays plain.
- **P8 is inverted, not a knob.** `num_stages` auto-pipelining is a plain-compiler feature with **no
  Gluon equivalent knob** — in Triton 3.8.0 no pass on the Gluon path consumes it, so on this tier it
  is a budget parameter and a champion record only, and the pipeline becomes the **manual pipeline
  backbone layer** (hand-written first: register prefetch → authored LDS ring → `warp_pipeline_stage`
  + scheduling model; `../tile-programming/pipeline.md`). The plain pipeliner passes themselves are
  **re-injectable** with no rebuild and no installed-file edit, but that buys at most **parity
  recovery, not a climb**: its ceiling is plain's own overlap, so it is ranked **last** — a
  diagnostic below the parity gate or a last resort when the hand-written pipeline cannot reach
  parity, never on an incumbent (`recover.md`, last-resort section;
  `../tile-programming/mental-model.md ## Why Gluon is full-explicit`,
  `../tile-programming/pipeline.md ## Where the overlap comes from, and it is not the same question per tier`).
  Two scoped cases where it does **not** pay, and neither generalizes to "re-injection regresses":
  a loop that already **hand-authors its staging** starves the pass, which then rewrites nothing at
  all; and a body whose staging plain puts on a shared layout Gluon has no constructor for, where the
  pass fires perfectly and still loses on the layout it falls back to. `BLOCK_*` is **coupled to the
  recovered layout family** (`warps_per_cta` ties mma / dot / shared / global), so changing a tile
  means re-recovering the whole layout set, not a cheap sweep.
- **Consequence:** the config / tile / stage search runs in **plain first** (cheap), and the winner
  is what gets transcribed. This is the mechanical reason the champion carries a *pinned* config and
  the gate checks that the TTGIR was dumped at it: a back end that re-sweeps is not tuning, it is
  invalidating its own anchor. A tile that needs re-sweeping is a `resweep_request` (§5.6).

### 3.11 After the handoff

The receiving pack's first step is its anchor — **faithful transcription** for Gluon
(`transcribe.md`), an explicit **port** for FlyDSL — not optimization. If, after the round budget, it
cannot beat `champion_ms`, record a negative result and **keep plain**: the guard over-estimated the
headroom. That is a real outcome with a real cost, and reporting it honestly is what keeps the gate's
guidance calibrated.

---

## 4. Which entry mode: A, B or C

The anchor-producing procedure is per pack — gluon transcribes (`transcribe.md`), flydsl picks a rung
and records `anchor_mode`, triton tunes. This section answers the question none of those procedures
do, because it spans them: **does this anchor owe a transcription debt, and therefore which gates
apply and what shape must the round loop be?**

One question decides everything below, and it is not which rung you took: **was the anchor
*converted* from a kernel that had already been measured?** If yes you owe a transcription debt,
whatever the conversion was called. If the anchor **is** that measured kernel, unconverted, you owe
none — and the gate that collects the debt must not be run, because it cannot fail.

| | **A. Converted anchor** | **B. Incumbent anchor** | **C. Re-entry** |
| --- | --- | --- | --- |
| the anchor is | a transcription / port / arch-conversion of a measured kernel | that measured kernel itself | a checkpoint |
| comparator | `champion_ms`, or the converted-from kernel's measured time | the **incumbent**, measured and asserted | whatever the interrupted run used |
| transcription debt | **YES, by construction** | none | already paid, or carried unpaid |
| G1 `champion_gate.py` | REQUIRED before the first read | REQUIRED, on an incumbent bundle | RE-ASSERT on resume |
| G2 `parity_gate.py` | REQUIRED before the first climb | **DOES NOT APPLY** as a gate | only if the debt was carried unpaid |
| G3 occupancy read | every anchor and every staging change | unchanged | unchanged |
| G4 `ab_bench.py` | unchanged | unchanged | unchanged |
| round shape | **PORT** (below) | ORDINARY optimize | as the interrupted run |
| round 1 outcome | expected **below** the comparator | expected at-or-above | continues the trajectory |
| GEAK entry state | **PORT**, `from_backend: triton` | **IN-PLACE**, `from_backend: gluon` | resuming a checkpointed variant |

**Settle the mode against the body, not the task name** — `from triton.experimental import gluon`,
`@gluon.jit`, an explicit `gl.BlockedLayout` / `gl.amd.AMDMFMALayout` / `gl.DotOperandLayout` at the
dot sites mean the source is already explicit Gluon: mode B.

### Which pack lands where

| pack | `anchor_phase` | its modes |
| --- | --- | --- |
| **gluon** (this skill) | `transcribe` | **A and B.** A is the designed path and the one the stage spine is written for: a `plain_champion` bundle from the front end, transcribed, so the debt is real. **B applies whenever the source handed over is already explicit Gluon** — nothing was converted, so there is no debt, G2 does not apply, and TRANSCRIBE / RECOVER / PARITY are a defined no-op. There is still no plain tier and no warm-start rung in either mode: the pack does not tune plain source. What changes on the B path: §5.4 |
| **flydsl** | `author` | **both.** Rung **(b) champion PORT** is A against `champion_ms`; rung (a) warm-start is A too *whenever the kernel had to be converted to become the anchor* (an arch-generation adaptation is the common case) and B when the anchor is the inherited kernel unchanged; rungs (c) template / (d) minimal-explicit have no prior number at all, so neither gate applies and the roofline budget is the only one |
| **triton** | `tune` | none — it **produces** the bundle the others consume, and does not ship G2. Its stake here is one paragraph: the ±1 grid-step obligation in mode A below |

`hip` and `tilelang` receive neither gate. Raw HIP is not a transcription target for any Triton IR
and `tilelang`'s `auto-baseline` transcribes nothing, so in both packs a `plain_champion` is an
artifact no path produces.

### A. Converted anchor — a port, whatever the rung called it

**The debt is real and attributing it is not optional.** A faithful anchor lands below the comparator
by construction. `parity_gate.py` splits the gap across `lost_pipeline` / `lost_layout` / `lost_RA`
from the compiled artifacts and exits **2** while it is unpaid. Climbing from an unpaid anchor caps
the port: the best lever you find gets quoted against a broken starting point, and the run closes
below the champion while reporting a healthy-looking gain against its own anchor. That is
Stage-Recover (`recover.md`), and the tool is what makes the declared parity criterion executable
rather than narrated — `--threshold`, a parameter whose default is `0.95`, compared inclusively
(`>=`).

**Both comparators travel with every result, and `vs_champion` decides.** A result quoted only as
`vs_anchor` is the port's characteristic failure, and it is watched for twice — the deep_engineer's
result must quote `vs_champion` (tech_lead treats a `vs_anchor`-only result as drift), and Director's
arbitration refuses a close quoted against the anchor. If parity is
never reached, close with `parity_unreached` in `caveats[]` and the attributed residual split — the
number is still reportable, but it is a recovery number.

**Before accepting "the plain tier is finished", spot-check the pin at ±1 grid step on each swept
axis.** This is the **producer's** obligation. A sweep's own report that its winner survived is not
evidence about points it never tested, and it is not what `champion_gate.py`'s `SAMPLING` check can
see — that check reads the bundle's own `partially_sampled` claim (and a `local_optimum`
certificate, §5.2) and says so. Measured: a **6.1%** plain win one grid point outside the swept range,
on a kernel whose tier log recorded a completed re-sweep. A port that starts one grid point short of
the real champion inherits that error as a fake escalation gain.

**Round shape.** Tell the caller this is a port. GEAK's `kernel_workflow` round loop defaults to
ordinary optimization, where a candidate under the baseline is worthless and a
non-improving round really is a stall. On a port those defaults delete Stage-Recover outright: the
transcription round produces no candidate, nothing is kept, the round has no winner, and the loop
stops two rounds into a port that is working exactly as designed. What a port needs instead (GEAK's
launch-arg spelling of each: §7.1):

- **candidate floor below 1.0** — low enough for *your measured anchor*, not for a guess. Naive
  anchors between 0.5x and 0.7x are ordinary and one at **0.51x** (a MoE INT4 kernel) has been
  observed, which is a bad window away from falling out of a 0.5 floor.
- **progress delta allowed negative** — on a port an experiment that costs ground is information.
- **no-improve tolerance >= the longest non-improving streak you expect.** Measured on `pa_decode`:
  wins at rounds 1, 8, 8, 9, 10 with **five consecutive** non-improving rounds between them. A
  tolerance of 4 ends that run one round before the payoff.
- **budget counted in the right unit.** If the caller counts directions rather than rounds, and a
  deep direction costs more than one, the round count is budget/cost — not budget.

### B. Incumbent anchor — the anchor IS the kernel that was measured

Nothing was converted, so there is no separate anchor and no debt. Three consequences, and the second
one bites.

**`parity_gate.py` must not be run as a gate.** There is no distinct `anchor_ms` to supply. Passing
the incumbent as both sides yields ratio 1.00 -> CLEARED, which is *true and vacuous*: it records a
gate as satisfied that was never applicable, which is worse than not running it. If you find yourself
reaching for a second number to put on the `--anchor-ms` side, look again — a second number means
something *was* converted, and you are in mode A. Use the **tool** freely as a diagnostic —
`--champion-ms <incumbent> --anchor-ms <a variant that regressed>` is exactly the right way to read a
regression you introduced mid-run, since `lost_layout` / `lost_RA` / `lost_pipeline` are the right
vocabulary for it either way — but say which you are doing, gate or diagnostic, in `decision_log.md`.

**The two-comparator discipline collapses to one, and that removes a safety net.** In mode A,
carrying `vs_anchor` beside `vs_champion` is what stops you selling a recovery as a win. Here there is
only `vs_incumbent`, so nothing structurally reminds you the denominator has to be honest. So the
incumbent must be a **measured, asserted** number — not "the file I started from", and not a figure
inherited from another box or another container. Assert it with G1 against an incumbent bundle
(below), and re-measure it in your own first window. In GEAK the workflow's own frozen
baseline is that floor, and the ≥95%-of-tuned-plain bar does not apply.

**Keep the ORDINARY round shape.** Setting the port knobs here is the mirror-image mistake: a
candidate floor below 1.0 and a negative progress delta keep a genuinely stalled search alive, burning
the budget the port shape exists to protect. But see the depth contract (below) — ordinary shape is about
the candidate *floor*, not about how many rounds one direction may cost.

**No re-injection on an incumbent.** The pipeline on an already-Gluon kernel is authored
(`../tile-programming/pipeline.md ## Authoring the overlap yourself (the climb default on the Gluon path)`);
re-injecting plain's pipeliner is not an option on this entry (`recover.md`, last-resort section).

#### The incumbent bundle: same schema, different meaning per field

`champion_gate.py` keys on `schema: plain_champion` and every check still means something with the
explicit-tile source as `kernel` — but two fields change meaning, and one is where a run in this mode
is most likely to be fooling itself:

| field | incumbent reading |
| --- | --- |
| `SOURCE` / `LIVE` | unchanged, and still the pair that fails silently most often — an incumbent edited since it was measured is not the thing that was measured |
| `CONFIG` | still "the dump came from this source at the pinned launch config". Check it against the **explicit-tile** dump, not a plain one |
| `COMPARATOR` | `incumbent_ms <= default_ms`. If the kernel has no meaningful default, say so — the "not a strawman" claim is then unprovable, not proven |
| **`SAMPLING`** | **the highest-value check in this mode.** An explicit-tile kernel's tile is usually *inherited from a plain champion and never re-swept in the explicit DSL*, where the tile is coupled to the layout family rather than being a free knob. A bundle reporting a completed sweep may be reporting the *plain* sweep. Say which |
| `LOCUS` / `TOOLCHAIN` | unchanged, and load-bearing: cross-GPU and cross-container comparators have drifted 25% on measured hardware |

### C. Re-entry — resuming a checkpointed run

Reload `checkpoint/` and `decision_log.md` as baseline and history, then continue the same loop. Two
things are not optional:

- **Re-assert G1.** The bundle may have been regenerated while you were away, and a resumed run that
  silently changed denominators produces a trajectory nobody can read.
- **If the interrupted run carried its debt unpaid** (`parity_unreached`, or a G2 exit 2 it never
  closed), the debt is still owed. Re-run G2 before the first new climb, not after.

A coupled bundle is always continued whole: resuming half of a memory-path -> layout -> pipeline
change measures neither half.

### What is identical in every mode (G3, G4)

G3 and G4 do not care how you got here.

**G3 — read BOTH occupancy limiters the moment anything compiles**, registers and LDS bytes/WG, and
report a missing one as missing rather than as a zero. *Which tool* is per-pack, so take it from
`scripts/pack_facts.json` (`evidence_tools`) rather than from a name written here: gluon and triton
answer both halves in seconds with `python3 $KT/probe.py measure --dir <ir-dir>`; flydsl ships no
`probe.py` and reads the same two numbers from its `static_audit` binding plus the `occupancy` block
of `round_report.py --parse-rc`. The obligation is the pair of numbers, not the command. Quoting the
LDS side alone hands you generous headroom on a kernel that is register-bound — which is how an arm
with the most apparent LDS slack ends up the slowest of a set.

**G4 — time with the interleaved instrument** (`ab_interleave`, `$SKILL/scripts/ab_bench.py
--module <adapter>.py --permute`, under `kernel_workflow/scripts/gpu_lock.sh`; acceptance timing is
GEAK's `harness_lib.py` in a fresh process per leg): per-arm kernel objects and cache dirs, the oracle
before timing, a delta under the spread reported as `NOT RESOLVED`. Two instrument rules learned from
results that passed every other check:

- **A flat result set is a cache-collision suspect, not a finding.** Two variants differing only by a
  `gl.constexpr` layout constant share a Triton cache entry, and the second silently runs the first's
  binary. It is numerically perfect — every arm computes the right answer, just not with its own code
  — and it yields the most seductive possible artifact: arms that all tie, reading as a clean "the
  layout levers do not move the clock". Give each arm its own kernel object and its own
  `TRITON_CACHE_DIR`, expose `fingerprint()` so `ab_bench.py` can prove the binaries differ, and
  before recording any flat verdict across three or more arms run `--permute` and check whether the
  numbers follow the **code** or the **position**. (A *patched* variant — anything compiled under a
  process-global lowering patch — gets its own process, never a shared one: §3.2.)
- **A tolerance comparison cannot fail on NaN.** `NaN > tol` is False, so an all-NaN output scores
  zero out-of-tolerance elements and prints ALL PASS. `ab_bench.py` fails any non-finite metric and,
  given an `outputs()` hook, scans the tensors itself — one level below the adapter, because that is
  where the trap lives. Check `isfinite` first.

The reversed-intuition traps (`climb.md`) and "the round count is the denominator" (§7.4) apply
unchanged.

---

## Stage-Entry: the champion assertion

**Settle the entry mode (§4) before you run the gate.** The gate runs either way, but on a source
that is *already explicit Gluon* it runs on a different bundle and three of this pack's stages become
a defined no-op (§5.4) — read that first if there is any chance the kernel you were handed is already
explicit.

```bash
python3 "$SKILL/scripts/champion_gate.py" --champion <work>/plain_champion.json [--json]
#   escape flags (each one DECLARES a limitation into caveats[], it does not remove it):
#   [--allow-provisional] [--allow-ungated] [--allow-default-anchor] [--allow-shallow-climb]
```

Exit **0** = `CHAMPION GATE PASS -> may start the deep dig`; exit **2** = a HARD check failed (or the
bundle is unreadable) — `CHAMPION GATE FAIL -> do NOT start`. Every check prints by name as
`[PASS]` / `[FAIL]` / `[WARN]` (WARN = a soft check that did not pass). The bundle's contract and the
rationale for every field are §6.

**Do not start on a failing gate.** Stop, edit nothing, report `blocked` (the deep_engineer returns
`status: "failed"` with `blocked_missing_context: <item>` in its notes and no patch). A gate failure is the front end's bug, not yours to route around:
each failure is a defect that makes this run's numbers unfalsifiable rather than merely inconvenient.

### 5.1 What a failure would cost you

| failure | what it would cost you | escape flag |
| --- | --- | --- |
| `[SOURCE]` — the champion source changed since measurement | you would transcribe a kernel nobody benchmarked, against a number from a different kernel | none |
| `[LIVE]` — `kernel` (the path the run actually loads) no longer hashes to `source_sha` | `[SOURCE]` hashes the frozen copy `source_ref`, which matches essentially forever. Found on a real bundle: `kernel` had been overwritten by a later Gluon track while `source_ref` was pristine; `[SOURCE]` passed and the run's "plain" arm would have been Gluon, reporting a ratio near 1.0 against the wrong denominator. Restore from `source_ref` or re-measure | none |
| `[CONFIG]` — the TTGIR is not from the pinned config | the anchor starts below plain-best **by construction**, so every layer delta is measured from the wrong floor and the run can look like a win while losing to plain | none |
| `[COMPARATOR]` — `champion_ms` is slower than the default (or than `sweep_winner_ms`) | the target line is a strawman inverted; beating it proves nothing | none |
| `[SAMPLING]` — the sweep was capped and the winner is not certified | the target line is provisional, so any speedup here is inflated by whatever the unmeasured configs would have gained. The caveat travels with every number you report | `--allow-provisional` |
| `[GATED]` — the bundle's `trust_level` is not `pinned` | you would measure against a comparator nobody published as final; `ungated` = no oracle ran, so the winner may be wrong-and-fast and may never anchor a transcription | `--allow-ungated` (ungated), `--allow-provisional` (provisional) |
| `[CLIMB]` — the front end's `climb.rounds` is below the floor (10) with no priced `climb_declined` | the handoff is shallow: what looks like layout headroom here may be plain rounds nobody spent | `--allow-shallow-climb` |
| `[ANCHOR]` — the bundle names no `pinned_comparator` anchor | a `production_default_anchor` is not a pinned configuration comparator; it permits plain in-body structural work only and can never start a deep dig | `--allow-default-anchor` (plain-only closes; the bundle still may not begin deep work) |

### 5.2 All fourteen checks, read off the script

`champion_gate.py` runs **fourteen** named checks (count them in its `rec(...)` calls, or read its own
output — it prints every check by name, which is the supported way to recover them without reading
the script). HARD unless marked soft:

| check | passes when | hard / soft |
| --- | --- | --- |
| `SCHEMA` | the bundle loads, `schema: plain_champion`, required fields `kernel`, `source_ref`, `source_sha`, `config`, `champion_ms`, `ttgir` present | hard |
| `SOURCE` | `source_ref` resolves and its sha256 matches `source_sha` | hard |
| `LIVE` | `kernel` resolves and is byte-identical to the measured source | hard |
| `CONFIG` | the `.ttgir` was dumped AT `config` — cross-checked against the TTGIR's own `ttg.num-warps` and a `<ttgir>.config.json` sidecar (every key the two configs share, tile/warp knobs reported first; no fixed knob-name list) | hard; soft (unverifiable, not contradicted) when neither signal exists — have `dump_ir.sh` write the sidecar |
| `COMPARATOR` | `champion_ms <= default_ms` and `<= sweep_winner_ms` | hard; soft when `default_ms` is null (the "not a strawman" claim is then unprovable, and the gate says so) |
| `ANCHOR` | `anchor.type == pinned_comparator` | hard; `--allow-default-anchor` for a plain-only close |
| `GATED` | `trust_level == pinned` | hard for `ungated`/`provisional` without the flag; soft (UNKNOWN) on a bundle that predates the field |
| `SAMPLING` | the grid was sampled whole, or the sweep returned a **certified order-2 INTERIOR local optimum** (`local_optimum.{certified, order, interior, unprobed, pair_coverage_pct}`) — no single-axis and no paired move improved it at the measured noise band, and it sits on no ladder end | hard unless `--allow-provisional`; a certified-interior PASS still carries the certificate's `unprobed` list into `caveats[]` |
| `MEASUREMENT` | reports `n_contaminated` points whose first reading moved past the drift tolerance on re-read — the pin stands, but margins narrower than the re-verification spread are unresolved | soft |
| `RANGE` | `served_range` is non-empty | soft here; the close requires it (`close.md ## Multi-shape no-regression + dispatch`) |
| `CLIMB` | `climb.rounds >= 10`, or `climb_declined` prices the remainder | hard unless `--allow-shallow-climb` |
| `PATH_COVERAGE` | `path_coverage` is a list, each new guarded path / fast path / fallback / masked tail with `path_id`, non-empty `trigger_cases` and a resolvable `correctness_ref` | hard (an absent list fails) |
| `LOCUS` | the execution locus (`locus.docker` / `gpu` / `pythonpath`) is recorded | soft — a host locus says "confirm this box is the one the champion was measured on" |
| `TOOLCHAIN` | `toolchain` is pinned | soft — an unpinned bundle cannot be re-measured after a ROCm/Triton bump |

**`[SAMPLING]` on an uncertified, fully-sampled claim is a PASS reported as unfalsified, not
verified** — the gate can only read the claim, and it tells you to do the ±1 grid-step spot-check on
each swept axis before the first round (mode A above).

### 5.3 `[CLIMB]`: was the plain line climbed before it was published

Every other check asks whether the number is TRUE. This one asks whether the search that produced it
happened. Publishing the champion is the moment the plain run stops — the gate is a hard handoff
boundary, after which only the deep track and arbitration remain — so a bundle emitted after two climb rounds converts
the remaining wall clock into paperwork, and nothing downstream can see that it happened. Not
hypothetical: four measured kernels closed one to three climb rounds deep, two of them with a third
to a half of their budget unspent, and every one passed a strict close audit, because BRANCH width
was counted and CLIMB depth was not.

The floor (10, adopted from the front end's `search_mode.CLIMB_ROUND_FLOOR` when importable, else the
gate's fallback copy — the gate says which) is not a quota. It is the depth at which the pack's own
stopping rule (reconsideration after ten contiguous non-improving rounds) first becomes answerable.
Absent is not zero: the gate distinguishes "the bundle does not say" from "the bundle says two", and
both fail. The alternative is a **priced** `climb_declined`:
`{remaining_prize_pct, evidence_ref: {kind: probe|measurement, artifact}, method, is_upper_bound: true}`
with an artifact that resolves — the same bar a declined BRANCH wave clears, because a digit in a
sentence is not a price. A priced declination passes and is carried into `caveats[]`.

### 5.4 Branch: is the source you were handed already explicit Gluon?

Settle it against the body, not the task name (§4). If it is, **nothing was converted**, so there is
no transcription debt and this run is the *incumbent* entry — mode **B**, which is this pack's second
supported entry and not an exception to it. What changes:

| | port entry (a `plain_champion` from the front end) | already-explicit entry |
| --- | --- | --- |
| `champion_gate.py` | REQUIRED, on the bundle | **REQUIRED**, on an *incumbent* bundle: same `schema: plain_champion`, same checks, but `kernel` / `source_ref` name the explicit-tile source and `[CONFIG]` is checked against the **explicit-tile** dump, not a plain one. Field by field: §4, "The incumbent bundle: same schema, different meaning per field" |
| the comparator | `champion_ms` from the bundle | the **incumbent's own measured, asserted time**, re-measured in your first window on your own box. Not "the file I started from", and not a number inherited from another container — `[LOCUS]` / `[TOOLCHAIN]` are load-bearing here for the same reason. There is no second comparator, so nothing structurally stops a dishonest denominator: name the number, its locus and its date in `decision_log.md` |
| Stage-Anchor / Stage-Recover, and the parity gate | the spine | a **defined no-op, not a blocker**. Record `entry_mode: incumbent` in `decision_log.md` with the evidence that the source is already explicit, and advance to the climb. `recover_gluon.py` / `ttgir_bridge.py recover` have no input — they take a plain `.ttgir` — and there is no distinct `anchor_ms`. Before the first round, audit the incumbent's inline-asm sites (`transcribe.md`, Stage-Anchor) |
| `parity_gate.py` | REQUIRED before the first climb | **must NOT be run as a gate** (mode B above). Use the *tool* as a diagnostic and say in `decision_log.md` which you did |
| Stage-Climb, layers 1.5 → 7 | unchanged | unchanged. This is the whole run |
| at close | `parity_unreached` in `caveats[]` if the gate was never cleared | the parity caveat **does not apply**; say that, with the entry mode, rather than leaving it unaddressed. The incumbent replaces `champion_ms` as the line, and the served range still has to be validated |

If you find yourself reaching for a second number to put on `--anchor-ms`, look again: a second number
means something *was* converted, and you are in the port entry after all.

**Two things this pack does not settle, and naming them is the honest move.** First, nothing in the
tooling records the entry mode — the stage graph has no `entry_mode` field and `champion_gate.py` has
no incumbent flag — so the no-op above is witnessed by your `decision_log.md` and by nothing else.
Second, `[SAMPLING]`: an explicit-tile kernel's tile is usually inherited from a plain sweep and never
re-swept in the explicit DSL, so a bundle claiming a completed sweep may be claiming the *plain* one.
This pack does not define what a sufficient explicit-tile sweep is. What would settle it: a bundle
field that records which DSL the sweep ran in, or a re-sweep of the tile in Gluon — which is a
front-end request (`resweep_request`, §5.6), not something this tier may do. Until then, state which
sweep the `[SAMPLING]` claim is about and carry it as a caveat.

### 5.5 Pin the comparator, then stop moving it

**Before the anchor, settle the comparator.** The kernel you pin must be the plain champion at *its
own best config*, and the `.ttgir` you dump (or the bundle carries) must come from that config: every
number later is relative to it, and the layouts you recover are the ones that config produced. Two
consequences worth knowing up front. A port measured against a shipped default is not a port that
landed, however good the ratio looks — and so is a 95% measured against one. And the anchor is
**bound to the config it was dumped at** — the recovered layouts carry literal `warps_per_cta` and tile
extents, so if the champion's best config differs per shape bucket you dump and transcribe per bucket
rather than expecting one anchor to follow it. Reproduce the comparator's number on your own harness
(GEAK `harness_lib.py`, your GPU, your container) before the anchor exists.

**Then freeze it for the run.** Two directions not to spend while the port is landing: a second arm
on the transcription (the transcription is deterministic — two agents transcribing the same pinned
`.ttgir` land on the same layouts, so a parallel arm measures the same transcription and shares its
bugs), and **anything that re-tunes the plain comparator** — that moves the denominator underneath a
port whose whole definition is a ratio to it. Re-tune plain before the track starts or after it ends,
never during. What is invariant, because each makes the run's numbers unfalsifiable rather than merely
worse: the layout equivalence gate at the end of transcription, not mixing transcription with
optimization in one edit, and **the comparator staying frozen**. Everything else — which residual
owner is worked first, whether the pipeline layer is entered at all, which loop body ships — is a
route chosen from measurement.

### 5.6 Routing out of entry: return to the front end, `resweep_request`, `structure_suspect`

- **If neither entry applies** — no bundle, and no measured incumbent to assert — there is no plain
  tier in this pack to fall back on. That is a **return to the front end**, not something to fix here,
  and so is a residual gap that turns out not to be layout-shaped after all.
- **`resweep_request`** — the champion's sweep did not cover the better config, its tile was never
  re-swept (including in the explicit DSL), or a tile/`num_warps` change is wanted. A tile change means
  re-recovering the whole layout set, so it is not a lever here. Return the request; do not run the
  sweep in this role or claim its possible gain as a Gluon result. It travels in the deep_engineer's
  result; tech_lead hands it to GEAK's plain rounds, which own the sweep.
- **`structure_suspect`** — the structure or the comparator must be reconsidered: an `UNRECOVERABLE`
  layout that survives the probe-build / ns=1 re-dump checks and cannot be carried as a forced
  divergence (`transcribe.md`), or a structural wall during the climb. A `structure_suspect.json`
  accompanies a CLOSE outcome; it is never an outcome value of its own (`close.md`).
- **`blocked`** — `champion_gate.py` exit 2 (above).

### 5.7 The tool entry points you will reach for

| | entry | answers |
| --- | --- | --- |
| 1 | `$KT/hw_budget.py`, no GPU | what is my budget, where is the floor — **and, from the `--workload` archetype you must name to get either, the `../workloads/index.md` row this kernel type makes you read** |
| 2 | `$KT/capture.sh` | where the cycles actually go — ATT rollup + feed split + static audit + IR, one command (profiler entry is GEAK's `kernel_workflow/scripts/profile_kernel.sh`; `profile.md`) |
| 3 | `$KT/probe.py`, compile-only, seconds | did that move register/LDS pressure; can this tile ever reach 2 waves/SIMD |
| 4 | `$SKILL/scripts/ab_bench.py` | is the difference real |
| 5 | `$SKILL/scripts/probe_levers.py --all`, no GPU | which version-sensitive knobs are LIVE on **this** build. `version_disjoint_knobs` separates `live` from `dead-declaration` (accepted, no readers, no IR change) and `absent`. Run it before sweeping one: all four fail silently out of range, so the flat result reads as a fact about the kernel when it is a fact about the build |

Specialists you reach for by name: `champion_gate.py`, `parity_gate.py`, `ttgir_bridge.py`,
`recover_gluon.py`, `ttgir_to_gluon.py`, `$KT/layout_facts.py`, `kernel_workflow/scripts/gpu_lock.sh`.
Full index: `scripts/USAGE.md`.

---

## 6. The champion bundle contract: `plain_champion.json`

The broad-search front end and the deep-dig back ends are separate skills, so the thing they share is
a file, not a conversation. This is that file's contract.

**Producer:** `tile-programming-triton` (the broad-search front end), at close, via
`scripts/champion_emit.py` (triton pack). In GEAK, GEAK's plain rounds must produce an
equivalent bundle (same schema) before this skill's port starts.
**Consumers:** `tile-programming-gluon` (recovers its anchor from the champion's TTGIR) and
`tile-programming-flydsl` (reads `ttgir_facts.json` and ports the champion's structure).
**Gate:** every consumer runs `champion_gate.py --champion <bundle>` before its first round (Stage-Entry, above).

**Scope:** this contract describes the **port** entry — a tuned plain kernel handed to an
explicit-tile pack. A run whose source is *already* explicit-tile has no plain champion, owes no
transcription debt, and needs a different gate set and the opposite harness loop shape. That is a
supported entry, not a disqualification: the bundle is an **incumbent** one, same schema (§4 mode B,
"The incumbent bundle: same schema, different meaning per field"; §5.4).

### Why the bundle carries more than a config

The predecessor artifact, `plain_best_config.json`, carries only the sweep's winning config and its
latency. That is enough to *re-run* a config, and not enough to *stand behind a number*:

- The champion is not the sweep winner. Source-level work (a reduction landing, a fused epilogue, a
  grid remap) lands after the sweep, so the config alone no longer identifies the kernel that was
  measured. Hence `source_ref` + `source_sha`.
- A speedup is only meaningful against a *tuned* baseline, and proving the baseline was not the
  shipped default requires **both** numbers. Hence `default_ms` alongside `champion_ms`.
- A win at one shape can lose over most of the served range's GPU-seconds. Hence `served_range`.
- A latency compared across boxes, containers or toolchains is not a comparison. Hence `locus` +
  `toolchain`.
- Emitting this bundle is what ENDS the plain run — the gate is a hard handoff boundary, after which
  only the deep track and arbitration remain — so a champion published two rounds into the climb converts the
  rest of the budget into paperwork, and nothing downstream can see that it happened. Hence `climb`.

Each field below exists because its absence let a specific wrong claim through.

### Schema

```json
{
  "schema": "plain_champion",
  "kernel": "/repo/op_tests/kernels/foo.py",
  "repo_root": "/repo",
  "family": "gemm_a8w8",

  "source_ref": "champion/foo.py",
  "source_sha": "<sha256 of source_ref>",
  "config": {"BLOCK_SIZE_M": 128, "num_warps": 8, "num_stages": 2},

  "default_ms": 11.16,
  "sweep_winner_ms": 10.02,
  "champion_ms": 9.31,

  "ttgir": "ir/champion/champion.ttgir",
  "ir_dir": "ir/champion/",
  "facts": "ttgir_facts.json",

  "locus": {"docker": "41ac90cd2160", "pythonpath": "/repo", "gpu": "4"},
  "toolchain": {"rocm": "7.1.1", "triton": "<tag/sha>", "torch": "2.x",
                "image": "<docker image>", "arch": "gfx950"},

  "served_weight_kind": "calls",
  "served_range": [{"shape": "M=1,N=4096,K=4096", "ms": 9.31, "vs_default": 1.20, "weight": 1.0}],

  "climb": {"rounds": 14, "kept": 3, "ledger_ref": "exp/rounds.jsonl"},
  "climb_declined": null,

  "anchor": {"type": "pinned_comparator"},
  "path_coverage": [],

  "partially_sampled": false,
  "trust_level": "pinned",
  "caveats": [],
  "provenance": {"sweep_ref": "exp_plain/plain_best_config.json",
                 "checkpoint_ref": "checkpoint/",
                 "branch_arms": ["exp/branch/plain/arm_1"]}
}
```

Every path is **relative to the bundle's own directory** (or absolute), so the whole bundle can be
moved or archived without breaking. `champion_gate.py` resolves them that way.

### Field notes

| field | meaning / why it is checked |
| --- | --- |
| `source_ref` + `source_sha` | the champion source, copied into the bundle. The gate re-hashes it: an edited source means the deep dig would anchor on something that was never measured |
| `kernel` | the path the run actually loads. `[LIVE]` hashes it against `source_sha` too, because `source_ref` is a frozen copy and cannot detect an overwritten live file |
| `config` | the pinned config. Key names are the **kernel's own** (`BLOCK_SIZE_M`, `BLK_M`, …); nothing keys on GEMM-canonical spellings |
| `default_ms` | the kernel's own default (`{}` = no override), measured. `null` is allowed but makes "not a default strawman" unprovable, and the gate says so |
| `sweep_winner_ms` | the config-only winner, so source-level work can be shown not to have regressed it |
| `champion_ms` | the final plain number: SWEEP + source-level CLIMB + structural BRANCH. This is the target line every escalated round is quoted against |
| `ttgir` | dumped from the **champion source at the pinned config**. Write a `<ttgir>.config.json` sidecar next to it so the gate can verify that mechanically instead of reporting UNVERIFIED |
| `facts` | `ttgir_facts.json` — the DSL-neutral parameter extract (tile shape, MFMA atom, swizzle, loop nest). FlyDSL reads this; Gluon reads the TTGIR directly |
| `partially_sampled` | true when the sweep benched a capped subset. The comparator is then *provisional* unless a certified order-2 interior `local_optimum` backs it, and consumers must pass `--allow-provisional` and record it in `caveats[]` |
| `local_optimum`, `grid_coverage_pct` | the sweep's certificate: `certified`, `order` (1 = single-axis ±1 neighbours measured, 2 = plus sensitive-pair diagonals), `interior`, `unprobed`, `pair_coverage_pct`. Read by `[SAMPLING]` |
| `trust_level` | what the champion may be USED as. `pinned` = oracle-gated and fully sampled, the tuned-plain comparator. `provisional` = gated but capped. `ungated` = **no oracle ran**, so the winner may be wrong-and-fast; the gate refuses it unless `--allow-ungated`, and it may never anchor a transcription. A bundle written before this field existed reads as unknown, not as gated |
| `anchor` | `type: pinned_comparator` for deep work. `production_default_anchor` permits plain in-body structural work only |
| `path_coverage` | one row per new guarded control path (`path_id`, `trigger_cases`, `correctness_ref`). A served-range row is not proof that a new guarded path ran |
| `n_contaminated` | sweep points whose first reading measured interference; soft `[MEASUREMENT]` |
| `served_range` | one row per served shape. Soft at the gate, required at CLOSE |
| `climb` | how deep the plain line was climbed before this bundle ended the plain run: `rounds`, optionally `kept` and the `ledger_ref` the count was read from. The gate holds `rounds` against a floor of 10 (§5.3). Absent is not zero |
| `climb_declined` | the price, when the climb stopped below the floor: `{remaining_prize_pct, evidence_ref: {kind: probe|measurement, artifact}, method, is_upper_bound: true}`. Same bar a declined BRANCH wave clears, and for the same reason — the claim ends the search, so its denominator has to have been measured. A digit in a sentence is not a price |
| `served_weight_kind` + per-row `weight` | optional, and the only thing that makes the rows *rankable*. Without weights the table can say "no shape regressed" and nothing more; with them, `scripts/served_envelope.py` turns it into a served speedup and a marginal value per shape. State the kind explicitly — `calls` (traffic share) vs `time` (calls × baseline_ms) — because a time weight consumed as a call weight double-counts the baseline and can invert the shares. Absent weights are absent, never assumed uniform: silently equal-weighting a served mix is a claim about the deployment that the bundle did not measure |

### What the bundle deliberately does not carry

- **Register/spill facts.** They are decided by LLVM after `make_ttgir` and are not in the TTGIR at
  all; a consumer re-derives them from its own build (`$KT/probe.py measure`).
- **A recommended next lever.** The bundle is evidence, not a plan. The front end's opinion about what
  the deep dig should try belongs in its own report, where it cannot be mistaken for a measurement.
- **Anything the front end could not measure.** A field it could not fill is absent or `null`, never
  estimated — the gate distinguishes "unverified" from "verified good", and an invented value would
  collapse that distinction.

---

## The depth contract (every mode; bites hardest in mode B)

No pack here owns a round loop or a budget model, and the caller that does (GEAK's
`kernel_workflow`) measures progress in **directions
closed**. Nothing in either layer says how much a single direction may cost — so the default reading
is one experiment per direction, and under that reading the only work the explicit-tile tier exists
for becomes unreachable. LDS swizzle/padding choice and LDS dedup are the two things plain Triton has
no syntax for; they are also the two that touch several layouts at once. A brief that says "one lever
per round, record the wall, take the next lever" is a breadth-first search that is nominally
compliant with everything here and structurally cannot enter them.

Observed: a mode-B run measured its own residual correctly (both occupancy limiters pinned at 1, MFMA
and LDS at parity, ~3x the overlap floor in exposed latency), identified `warps_per_cta=[4,1]`
replicating the LDS reads 4x as the one structural fix, and then declined it in those words — *"not
attempted, it is a layout rewrite, not a one-lever round"*. It closed 22 directions, kept 8, banked
+6%, and moved nothing structural.

**The tell for a coupled direction: it invalidates more than one layout constructor.** Count the
sites before pricing it — `grep` the source for `warp_bases` / `offset_bases` / the shared-layout
constructors. One site is a lever. Fourteen is a rewrite.

Three rules, and they are cheap to honour:

- **A coupled direction gets a multi-round write budget up front**, sized before the first edit. It
  will be numerically wrong or slower in its intermediate rounds — that is the shape of a coordinated
  layout change, not a stall, and it is the same allowance a port's transcription gets for the same
  reason.
- **A coupled direction is not closed by one probe.** A single compile failure on a partial edit is
  evidence about the partial edit, and nothing else. The measured case: changing one of fourteen
  coupled `warps_per_cta` sites crashed the pass manager with no attribution, and the run recorded
  that as "the arch does not support it". A partial edit crashing is the expected behaviour of a
  coupled change, so it is a false negative and belongs in `../pitfalls/platform-known-issues.md` only
  once it reproduces on a *complete* one. (The full case, with the assert text and the site count:
  `../pitfalls/negative-patterns.md ## warps_per_cta is not an independently tunable knob`.)
- **State the split before the first round.** If the whole budget goes to one-lever peepholes, say so
  and expect a few percent; if a coupled direction is in scope, name it and reserve its rounds.
  Discovering at the deadline that the structural direction was never affordable is a planning
  result, not a measurement.

What this does *not* license: mixing a refactor with an optimization in one edit (that still destroys
attribution), or moving the comparator. A coupled direction is still one direction, still attributed
as a whole, and still gated by G4 at the end — it just is not one round.

**Priced against this skill's own bar.** The observed mode-B run above banked **+6 %** while declining
the one structural fix it had diagnosed; this skill's frontmatter sets `expects.isolated_speedup_min`
to **1.10**, so a run that spends its whole budget on one-lever rounds cannot reach the bar it is
graded against. Reserve the coupled direction's rounds in the brief, before the first edit.

---

## 7. How the entry modes map onto GEAK's launch

GEAK `kernel_workflow` injects this skill (`use_expert_skills: "true"`, `target_language: "gluon"`) into its
existing roles; GEAK owns the round loop, the commit gate (`MIN_IMPROVE` 2%) and acceptance, and the skill
is advisory (`orchestration.md`). Entry differs by mode only in how the round loop is told about it:

| | in GEAK |
| --- | --- |
| who runs the track | tech_lead dispatches one `deep_explore` direction; the deep_engineer runs entry → climb in its own loop |
| how the port shape is set | **the four launch args** (G0, §7.1) |
| champion bundle | GEAK's plain rounds produce it (same schema as the triton pack's `champion_emit.py`); `champion_gate.py` still runs |
| `resweep_request` / `structure_suspect` | returned in the deep_engineer's result; tech_lead turns it into the next round's direction |
| on a gate failure | report `blocked` (no edit); GEAK records it |

### 7.1 GEAK: how these modes map onto GEAK's entry states, and G0

| GEAK entry state (`skill.md ## When to use`, gate table) | mode | G1 bundle |
| --- | --- | --- |
| **PORT** — `from_backend: triton`: a tuned plain Triton kernel exists and is transcribed to Gluon | **A. Converted anchor** — the mode §6 is written for | `plain_champion.json` |
| **IN-PLACE** — `from_backend: gluon`: the source is already explicit Gluon and already measured | **B. Incumbent anchor** | an *incumbent* bundle, read per §4 mode B |
| resuming a checkpointed variant | **C. Re-entry** | re-assert whichever bundle the interrupted run used |

**"Round shape" is GEAK's harness loop shape, and gate G0 checks it before anything is dispatched.**
This skill owns no budget model, but a port has a shape the round loop has to be told about: a
transcription lands **below** the comparator and climbs back, and at the stock defaults that phase is
not representable at all — the transcription round produces no candidate, so no patch is saved, no
verify runs, and the loop stops two rounds in. There is no "port mode" to switch on; there are
**launch args**, and a port is a set of values for four of them. Pass them explicitly on the
`kernel_workflow` launch:

| arg | default | pass on a PORT | why the default breaks a port |
| --- | --- | --- | --- |
| `candidate_floor` | `1.0` | from the measured debt (below) | a faithful anchor is below the comparator **by construction**, so it never enters the candidate list: no patch saved, no verify, `winner = null` |
| `max_no_improve` | `2` | `6` | ends the run two rounds in, which is before the recovery round has finished |
| `budget` | `6` | `20` | `budget` counts **directions**, and a `deep_explore` direction costs 2 — so `budget/2` is the achievable round count. 20 gives 10 rounds; 6 gives 3 |
| `progress_delta` | `+min_improve` | `-0.05` | progress is measured against the best candidate ever seen, so a monotone climb never stalls at any value — but a port that **gives ground** while exploring reads as a stall. On the measured `pa_decode` run, leaving this at the default ends it at round 7 with 1.04× instead of round 10 with 1.25× |

On a PORT all four must be set; on IN-PLACE **none of them may be** — the harness defaults are already
right there, and a candidate floor below 1.0 plus a negative progress delta keep a genuinely stalled
search alive, burning exactly the budget the port shape exists to protect. On a G0 failure, stop
before dispatching: the loop will terminate two rounds into a port that is working exactly as
designed. **G0 has silently cost three runs — all three of them ports.**

**Run the port at `mode: optimize`.** Author mode writes a fresh seed that *replaces* the source —
which would overwrite the very kernel being transcribed, since the port needs that kernel's own
`.ttgir`. So `mode` cannot tell the loop this is a port; nothing does except the four args, and
omitting them is indistinguishable from an ordinary run. `kernel_lane.js`'s own comment on
`CANDIDATE_FLOOR` describes the consequence:

> *"a TRANSCRIPTION (plain Triton -> Gluon / TileLang / HIP) lands BELOW the comparator by
> construction and climbs back, so at 1.0 its whole recovery phase is invisible -- no patch is saved,
> no verify runs, `winner` is null every round, and the loop stalls on a run that is working as
> designed."*

A run that does not pass the four args behaves exactly as it always has. The commit gate is untouched
by all of this and still requires beating the cumulative best by `min_improve`, so a sub-baseline
candidate can be **tracked** but can never be **banked**.

The whole launch, with the two measured values at their opening guesses — replace both once you have
numbers (§7.2):

```js
Workflow({
  scriptPath: "<REPO>/kernel_workflow/kernel_workflow.js",
  args: {
    kernel_path: "<TASK_DIR>",
    workflow_dir: "<REPO>/kernel_workflow",
    mode: "optimize",              // NOT author — author would overwrite the source being transcribed
    target_language: "gluon",
    use_expert_skills: "true",     // this skill is injected only when this is on
    budget: 20,
    max_no_improve: 6,             // measured on pa_decode; 4 ends one round before the payoff
    candidate_floor: 0.5,          // opening guess — reset from the debt parity_gate.py reports
    progress_delta: -0.05,
    gpu_ids: "0",
  },
})
```

**Steer the round toward a single `deep_explore` direction rather than a specialist fan-out** — the
harness cannot infer that from the args, so say it in `task`. In `kernel_workflow` terms the port is
the **`deep_explore` track**: it runs alone in its own round, carries its own long
measure→self-profile→rewrite loop, and has authority over kernel plus wrapper — which is the shape the
transcribe→recover→climb track needs. Two reasons it is one track held by one agent, and they are
different: the porting moves are *deterministic* (parallel arms do not explore alternatives, they all
measure the same transcription, and share a bug in it identically), and the continuation is
*stateful* (what to try next is chosen from the IR and profile of the anchor just built, which the
agent that built it is holding and a fresh agent would have to rebuild). One mismatch to steer around:
that track is documented as a minimally-steered ground-up rewrite, and here the opening is the
opposite. **Do not let a ground-up rewrite replace the transcription** — the transcription *is* the
anchor every later number is attributed against.

### 7.2 Set two of the four from measurement rather than taste

A `max_no_improve` of 4 is short of what a real climb needs — on `pa_decode` the winning levers landed
at rounds 1, 8, 8, 9 and 10 with **five consecutive** non-improving rounds in between, so 4 ends the
run one round before the payoff and **6** is the value that survives the measured trajectory. And a
`candidate_floor` of 0.5 is uncomfortably tight: an observed naive anchor on a MoE INT4 kernel measured
**0.51×**, one bad window from falling out of its own candidate list. Set the floor from the debt you
actually took — `parity_gate.py` reports it — rather than picking a round number that has to guess.
*(The deeper point, worth sending upstream rather than working around: on a port, "is this a
candidate" is the question `parity_gate.py` answers properly, as a recovery verdict with an
attribution. A fixed ratio floor is a proxy for it, and the right proxy value is not knowable before
the anchor is measured.)*

### 7.3 Why the gates are executable

Every exit condition here was once prescribed in prose, and prose did not hold: a run has gone to the
climb from a **0.715×** anchor, reported 1.19× against that anchor, and closed at **0.85×** against the
champion — with every step's *text* obeyed. So each exit condition has a command, and **the round log
must carry the command's output.** "The precondition holds" is not a gate; it is the claim the gate
exists to test. Both gates (champion, parity) are mandatory by instruction, not by interception — the
close is audited against them (`recover.md`).

### 7.4 Budget: the round count is the denominator, not a ceiling

A 20-round budget spent as ~1 round of wall clock is not a 20-round search; it is a 1-round search that
reports a 20-round budget. Size the wall clock to the rounds, and checkpoint every kept win (diff +
metrics + that variant's private IR dir) the moment it lands, so a hard stop is a pause rather than a
discard. For calibration: on `pa_decode` the winning levers landed at rounds 1, 8, 8, 9 and 10 with
**five consecutive negatives at rounds 2–6** in between. A loop that ends at round 1 cannot reach any
of them.

### 7.5 Two comparators, both carried in every result

Correctness and layout equivalence are versus the **anchor**; performance is versus the **champion**.
`vs_anchor` alone hides the whole question — the anchor is a regression you created, so beating it
proves nothing. The same climb scores 1.19× or 0.85× depending only on which denominator is read, and
the honest report carries both with the champion one deciding. (Mode B collapses this to one
comparator — §4.)
