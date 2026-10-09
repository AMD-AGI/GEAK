# Method — the flow end to end

Take an **explicit-tile Gluon** kernel to peak on AMD CDNA — gfx950 (CDNA4, MI350X/MI355X) as the main
line, gfx942 (CDNA3, MI300X/MI325X) as the downgrade. The method is not a decision tree: you **measure the
budget, measure where the cycles go, and close the largest gap you can attribute** — one change per round,
verified.

This is a **deep-dig back end**. It does not tune plain source and it does not search in breadth. It starts
from a measured, sha-pinned bundle — a `plain_champion` produced by the plain-Triton front end
(`tile-programming-triton`; [`front-end.md`](front-end.md)), or an **incumbent** where the source is already
explicit Gluon — asserts that bundle, transcribes it into explicit layouts where there is something to
transcribe, and then climbs. Its value is depth: layout, memory path, pipeline and register slicing are too
fine-grained to screen in parallel, and their payoff only appears several layers in. Structure is not
re-litigated here; the front end settled it far more cheaply than this tier can.

**This page is the map.** It owns the flow, the gates, the Non-Negotiable Rules and the stage → file → tool
table. Every other method file is one stage. Lazy-load only the active stage; do not preload sections
merely because they exist.

## How it runs in GEAK

| | |
| --- | --- |
| who drives | GEAK `kernel_workflow` injects this skill (`use_expert_skills`) into its existing roles; GEAK's phases (Setup → Benchmark → Profile → Optimize → Verify → Merge → Report → Validate) are the stage machine. tech_lead dispatches **one** `deep_explore` direction; the deep_engineer carries entry → climb in its own loop |
| the skill's role | advisory: what to measure, which layer, which gate a round owes, the record contract |
| acceptance timing | `verify_engineer` / `measure_legs` via `e2e_workflow/scripts/harness_lib.py`; commit gate `MIN_IMPROVE` = 2% (a measured noise band only stricter); Director validates and arbitrates; `ab_bench.py` is screening only |
| GPU | `kernel_workflow/scripts/gpu_lock.sh` (lock dir `/tmp/team_gpu_locks`), one GPU per engineer; the pack's broker (`scheduler/`) only with `GEAK_GPU_BROKER=1` |
| requests (`resweep_request`, `structure_suspect`) and the closure self-review | in the deep_engineer's `worker_result`; tech_lead turns a request into the next round's direction, Director quotes the self-review when arbitrating |

The evidence owed is the same in every round: **budget / roofline before authoring**, and a **re-profile
every round**. Detail: [`orchestration.md`](orchestration.md).

---

## 0. The flow, end to end

The whole method on one screen; the right column says which method file owns each box.

```text
BUDGET     hw_budget.py, no GPU. ROOFLINE: intensity vs the SKU ridge -> the bound-class
   |       prior; the MFMA-only floor -> your multiple over it. Sizes the prize.   budget.md
ENTRY      settle the MODE first -- ported plain champion, or a source that is ALREADY
   |       explicit Gluon (incumbent; TRANSCRIBE/RECOVER/PARITY become a defined
   |       no-op). Then champion_gate.py on that bundle. No rounds until it passes:
   |       source hash, TTGIR-at-pinned-config, comparator, sweep sampling.         entry.md
TRANSCRIBE the champion's OWN TTGIR -> explicit Gluon layouts (ttgir_bridge recover;
   |       recover_gluon assembles the anchor). Faithful by default; any divergence
   |       NAMED + its faithful variant measured (the ledger). Equivalence gates:
   |       layout-diff, numerics, determinism, asm parity.                     transcribe.md
RECOVER    close the transcription DEBT -- the pipeline Gluon dropped, a
   |       mis-recovered layout, lost RA -- attributed, repaid by hand-written
   |       mechanisms, not out-climbed. Re-injection: diagnostic / last resort.   recover.md
PARITY     GATE: parity_gate.py. Apply the parity criterion declared for this run
   |       (default 0.95), and while it fails, WHICH of lost_pipeline / lost_layout /
   |       lost_RA owes the gap. Below it you are not optimizing yet; above it a
   |       delta is a real win.                                                 recover.md
PROFILE    profile_kernel.sh / capture.sh (+ rocprof-compute SOL) -> the four dials,
   |       anchor profile, then a re-profile every round.                       profile.md
CLIMB      layers 1.5 -> 7, ONE coupled layer per round, CLIMB only. Target line is
   |       champion_ms. No SWEEP, no BRANCH. Layer table: climb.md ## Layer Backbone
CLOSE      win | partial | negative_revert_plain | negative_keep_baseline | timeout --
   |       the whole enum. A structural wall is one of those PLUS a
   |       structure_suspect.json, never an outcome value of its own. A result under
           champion_ms is a negative, never the winning stage.        close.md ## Stop conditions
```

**How this differs from the front end, as a search algorithm.** The front end runs CENSUS → BRANCH →
CONVERGE → CLIMB: it explores mutually-exclusive structures in parallel and picks one. This pack runs
**TRANSCRIBE → RECOVER → CLIMB** and nothing else. The structure arrived decided; what this tier buys is
depth on layout, memory path, pipeline and register slicing, which is why the first half of the run is
spent *reproducing* a number that already exists rather than searching for a new one. A pack that skips
the recovery accounting and goes straight to climbing spends its rounds re-earning what transcription gave
away and books it as progress.

### The stage spine

Four method stages, strictly in order, each with its own admission rule; the pack's stage cards (optional
bookkeeping, `toolctl`) name six (`entry → transcribe → recover → evidence → climb → close`), the extra two being the evidence
refresh and the close.

```text
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

The framing to carry: a faithful anchor's regression vs the champion is a **debt taken on knowingly**, not
a floor you climb from silently, and Stage-Recover pays it back, attributed, before any delta counts as a
win. On an incumbent entry Stage-Anchor and Stage-Recover are a defined no-op you record, not a blocker.

### Where the plain / Gluon seam sits

Theory / profile / budget for the plain kernel are done on **plain Triton first**; the explicit-tile stage
is entered only after the seam says the residual gap is layout-shaped. The two stages are two skills:

| stage | skill | what it owns |
| --- | --- | --- |
| **Stage-Plain** | `tile-programming-triton` (method: [`front-end.md`](front-end.md)) | Layer 0 structure, the config SWEEP, source-level CLIMB, the structural BRANCH fan-out. Ends by emitting the **champion bundle** |
| **Stage-Gluon** | `tile-programming-gluon` (this skill) | asserts the champion, transcribes it, then CLIMBs layout / memory / pipeline / slicing. No SWEEP, no BRANCH |

**The seam between them is a file, not a phase transition inside one run:** `plain_champion.json`
([`entry.md`](entry.md)), asserted by `champion_gate.py`. So "escalate" means *hand off to the other
skill*, and the phases after it run in that skill's process, against `champion_ms` as the target line.
Whichever pack you are in, the phases before your own stage are already done and their artifacts arrive in
the bundle — do not re-run them. In GEAK the front end is GEAK's own plain-Triton tuning, and the
seam is the pinned champion config plus its dumped `.ttgir`.

Stage-Diagnose ([`entry.md`](entry.md)) is the optional **front porch**: for measure / diagnose /
verify-a-hypothesis tasks, run it first and enter these phases only when it returns `escalate`.

```text
Analyze          -> profile.md (plain front-end analysis)  (workload, target, boundary; worksheet [A-D] + priority ladder)
Harness          -> benchmark-hygiene.md     (isolated baseline + candidate, COMMANDMENT, plain comparator)
Profile (plain)  -> profile.md               (rocprof/ATT/IR + roofline -> bound class)
Tune             -> phases/tune.md           (SWEEP once, PIN the winner: the comparator)   [triton pack]
Budget           -> budget.md                (ideal vs as-built -> gap)
[the seam]       -> entry.md                 (stay plain | hand off to a deep-dig pack)
   stay plain    -> front-end.md (applicable 0-15 directions); emit the champion anyway; done
   hand off:
     Champion    -> entry.md                 (champion_emit.py + ttgir_facts.py)             [triton pack]
     [gate]      -> champion_gate.py         (entry assertion; do NOT start on a failure)    [gluon pack]
     Transcribe  -> transcribe.md            (champion TTGIR -> Gluon equivalence anchor; re-profile)
     Recover     -> recover.md               (attribute + repay the debt; parity gate)
     Layer Loop  -> climb.md                 (one deep_engineer's sequential hill-climb over the backbone)
     [llvm flag] -> ../tile-programming/compiler-contract.md ## Scenario B: sanctioned compiler co-design
Evaluate         -> close.md                 (vs Gluon baseline no-regression + vs champion_ms + IR acceptance)
record winning tier + per-bucket fallback
Report
```

`phases/tune.md` ships only in the triton pack (it is that pack's anchor phase); the transcription stage
ships only in the gluon pack. Everything else on the list is shared.

### The four gates

The upstream method has four gates. GEAK's procedure numbers its executable checks G0–G4; the
mapping is below. **A gate is passed by tool output, not by assertion** — the round log carries the
command's output, because "the precondition holds" is the claim the gate exists to test.

| gate | fires | what it refuses | GEAK |
| --- | --- | --- | --- |
| — | before anything is dispatched | a port launched with the ordinary loop shape: at the stock defaults a transcription lands below the comparator, never becomes a candidate, and the loop stops two rounds in | **G0** — the `kernel_workflow` launch args. PORT: `candidate_floor` (from the measured debt; opening guess 0.5), `max_no_improve` 6, `budget` 20, `progress_delta` −0.05, `mode: optimize`. IN-PLACE (incumbent): leave the defaults — setting them is the mirror-image mistake |
| **champion** (`champion_gate.py`) | before the first round | a bundle whose numbers cannot be re-checked — an unpinned source, a TTGIR from the wrong config, a strawman comparator ([`entry.md`](entry.md) Stage-Entry) | **G1** — on `plain_champion.json` (port) or an incumbent bundle; failure ⇒ stop, edit nothing, report `blocked` |
| **equivalence** | at the end of transcription | an anchor that is not the champion: layout-diff (`ttgir_bridge verify`), numeric oracle at tolerance, determinism (~40 launches, race), asm parity — all four, names-independent ([`transcribe.md`](transcribe.md)) | no G-number: part of the transcription round; its four results are recorded in the round log |
| **parity** (`parity_gate.py`) | between RECOVER and CLIMB | calling a recovery a win, and climbing on top of an unattributed transcription defect ([`recover.md`](recover.md) Stage-Recover) | **G2** — port only, before the FIRST climb; exit 2 ⇒ DO NOT CLIMB, the round is `recovery`. Incumbent: must **not** be run as a gate (equal sides → a vacuous CLEARED); usable as a diagnostic, say which |
| **attribution** | every kept round | a layer landed without all three of budget consistency, profile delta, IR/asm signal ([`climb.md`](climb.md) `## Layer Backbone`) | **G3** — `probe.py measure --dir ir/<tag>/`: read BOTH limiters (registers and LDS cap WGs/CU independently). **G4** — "the number is a number": screening with `scripts/ab_bench.py --module <adapter>.py --permute` (a delta under the spread is `NOT RESOLVED`), acceptance by GEAK verify timing + the 2% commit gate |

**Both upstream gates (champion, parity) are mandatory by instruction, not by interception.** This pack
runs the `tool_first` regime: it ships no per-round write gate, no bound-class decision tree and no JSON
round contract, so nothing stops a round that skipped `champion_gate.py` or ignored a `2` from
`parity_gate.py`. What makes them binding is that the close is audited against them ([`close.md`](close.md)).

## Stage → method file → tools

Moved GEAK shared tools live in `kernel_workflow/scripts/kernel_tools/` (KT); the pack paths under
`scripts/` remain as shims. Examples use `--arch gfx950`; a tool that needs an arch and was not given one
refuses.

| stage (spine id) | method file | what it decides | tools |
| --- | --- | --- | --- |
| budget (before `entry`, refreshed in `evidence`) | [`budget.md`](budget.md) | bound-class prior, floor, ideal vs as-built gap, rank-only vs gate-able numbers | KT `hw_budget.py`, `calc_perf.py`, `mem_bw_probe.py`, `extract_sku.py`; data `perf_knowledge/hardware/data/{sku,hw_constants,workload_models,thresholds}.json` |
| `entry` | [`entry.md`](entry.md) | entry mode (port / incumbent / diagnose), champion assertion, stay-plain vs hand-off | `scripts/champion_gate.py`, `probe_levers.py --all`; the front end's `champion_emit.py`, `ttgir_facts.py` |
| plain front end (other skill / GEAK) | [`front-end.md`](front-end.md) | plain strategies and lever recipes, bucket dispatch | triton pack `knob_probe.py`, `plain_autotune.py`; `scripts/create_harness.py` |
| `transcribe` | [`transcribe.md`](transcribe.md) | the faithful anchor and its divergence ledger; the equivalence gate | `scripts/ttgir_bridge.py recover|verify|view`, `recover_gluon.py` (anchor assembly), `ttgir_to_gluon.py`; KT `dump_ir.sh`, `probe.py` |
| `recover` | [`recover.md`](recover.md) | debt attribution, hand-written repayment, parity disposition; re-injection as diagnostic / last resort | `scripts/parity_gate.py`, `pipeline_survey.py`; last resort / diagnostic only: `gluon_swp.py`, `patch_reinject.py`, `patch_async_reinject.py` |
| `evidence` | [`profile.md`](profile.md) | the four dials A/B/C/D, bound class, degrade ladder | `kernel_workflow/scripts/profile_kernel.sh <gpu_id> <cmd> <out>` under `gpu_lock.sh`; KT `capture.sh`, `rocprofv3_safe.sh`, `rocprof_compute_probe.sh`, `parse_pmc.py`, `parse_rc.py`, `kernel_breakdown.py`, `hotspot_analyzer.py`, `att_*.py`, `mfma_efficiency.py`, `asm_loop_audit.py`, `layout_facts.py`, `amd_occupancy.py` |
| `climb` | [`climb.md`](climb.md) | which layer, which mechanism, keep / mature / revert | KT `probe.py`, `gfx950_isa.py`, `asm_schedule_viz.py`; `scripts/lever_index.py`, `served_envelope.py`, `ab_bench.py` (screening) |
| timing / harness (every stage that measures) | [`benchmark-hygiene.md`](benchmark-hygiene.md) | is the number a number | `e2e_workflow/scripts/harness_lib.py` (acceptance), `scripts/create_harness.py`, `parse_correctness.py`, `ab_bench.py` (screening) |
| `close` | [`close.md`](close.md) | outcome, dispatch, closure self-review (`closure_review.md`), provenance, final report | `scripts/canonical_record.py`, `close_audit.py`, `report_lint.py` (optional bookkeeping) |
| records (all stages) | [`records.md`](records.md) | the ledgers and the experiment contract | `scripts/canonical_record.py`, `recordctl.py`, `round_record.py` |
| failures (any stage) | [`triage.md`](triage.md) | retry / defer / halt; async-handoff worksheet; missing docs | KT `dump_ir.sh`, `asm_loop_audit.py` |
| orchestration | [`orchestration.md`](orchestration.md) | which GEAK role runs which stage, GPU lock, parallelism, records / resume, optional bookkeeping (context lease) | `kernel_workflow/scripts/gpu_lock.sh`, `wait_for.sh`; optional `scripts/toolctl.py`, `locus.sh` (kernel in a separate container) |

Pitfalls catalogues: `../pitfalls/negative-patterns.md` (incl. the decode / paged Quick Reject as an
admission hint), `../pitfalls/platform-known-issues.md`. Layer mechanisms: `../tile-programming/`,
`../gluon/`, `../hardware/`, `../workloads/`.

### Phase outputs (artifacts)

```text
plain_baseline_metrics.json   # the pinned plain comparator                      [triton pack]
plain_champion.json           # the handoff bundle: source+TTGIR+both timings    [triton pack]
ttgir_facts.json              # DSL-neutral parameter extract (flydsl reads it)  [triton pack]
gluon_anchor_metrics.json     # the new Gluon baseline after transcription       [gluon pack]
COMMANDMENT.md                # immutable evaluation contract
budget/<round>.json           # ideal vs as-built per round (with numerator/denominator basis)
profile/<round>/              # trace, ATT, counters
ir/<variant>/                 # .ttgir/.llir/.amdgcn/.s
decision_log.md, checkpoint/  # the round trajectory (deep_engineer's OUTPUT_DIR)
closure_review.md             # closure self-review at a ceiling / keep-baseline / negative close
final_report.json             # see close.md
```

## Non-Negotiable Rules

DSL-neutral hard rules. They bind every round, whether or not this page is resident.

- Do roofline / budget analysis **before** authoring or editing a kernel; do not "optimize to find out" the
  budget. **A ceiling called *calibrated* names the artifact it was measured from.** A hand-typed number and
  a probed one are the same float once they reach the denominator, so a percent-of-ceiling is only as
  credible as the provenance travelling with it: carry the probe's path, or mark the value hand-entered and
  read every percentage off it as an estimate. Every percentage carries `numerator_basis` /
  `denominator_basis`, and only a measured numerator over a probed or calibrated denominator may gate or
  close ([`budget.md`](budget.md)). This is a rule, not a tool behaviour — it binds hardest where the budget
  is prose because no tool ships to check it there.
- The first step is an **anchor**, not an optimization. An anchor's regression versus the roofline / a tuned
  reference is **expected, not a rejection**: it becomes the baseline to climb from. After anchoring,
  re-profile and re-calibrate the theory and budget from its own trace / IR before optimizing — and
  re-profile every round after that.
- Advance **one coupled layer at a time** in the backbone order; a layer is "landed" only with **three
  pieces of evidence**: budget consistent + profile delta positive + IR/asm signal confirmed.
  **Enabling-step exception:** a coupled step that is a declared prerequisite for a later layer may be
  *provisionally* accepted on a flat/negative single-step delta; it is *confirmed kept* only once the
  combination net-beats the checkpoint within the transient-regression budget, else rolled back.
- A host/dispatch-level swap (config / tile / split-K / dispatch heuristic) lands on A/B at the production
  boundary + correctness alone — no IR acceptance required.
- **A compiler-realized lever exploits structure, it never creates it.** A lever whose mechanism is a
  compiler/backend pass needs the paired source structure already in place; without it the knob is neutral
  or regresses. A compiler ceiling is valid only after the IR shows the structure is correct yet the pass
  still clumps/spills — otherwise it is a kernel-structure bug, not a compiler miss.
- Keep a fair comparator throughout; a layer must beat the in-progress baseline, and the final result must
  beat the target line / close the roofline gap without regressing the baseline. **Any speedup claim
  requires the comparator tuned to ITS OWN best config** — a win over an un-tuned / default strawman is
  invalid. When a like-for-like target line is not constructible (different layout/semantics), substitute
  the production config + an external ASM ceiling and say so in the contract.
- **A falsifying probe precedes every scoped-ceiling deferral.** Never defer a lever/feature to
  `deferred_needs_user` on assumption. First run the cheapest falsifying probe and record its exact
  error/number: does the op compile / legalize? does an alternative entry work? is a "disabled" flag
  cosmetic? what does a microbench say? "unproven" / "assumed infeasible" / "the flag defaults off" are not
  acceptable terminal states.
- **Discover tools from their contract, never from a script body.** Use a tool's `--help` (and
  `--describe`, or the `toolctl` stage context when you keep the optional bookkeeping) for the supported
  invocation and emitted fields. `scripts/USAGE.md` is only the staged
  index. Do not invent unadvertised `toolctl` verbs or flags, and do not read a script to recover them. For a
  suspected documentation or tool defect, make a `source_read_request` that names the question and symbol;
  inspect only the resulting `source_excerpt`. This is an operating rule, not a claim that Cursor or Claude
  Code physically blocks other source access. Correct the documentation when the request demonstrates a
  documentation defect. Tools are invoked by the paths in the table above.
- Benchmark hygiene is mandatory: isolate source roots, imports, caches, artifacts, harness, shape stream,
  and timing configuration; match the production boundary. Acceptance timing is GEAK's `harness_lib`
  contract; GPU access goes through `kernel_workflow/scripts/gpu_lock.sh`, never an inline
  `HIP_VISIBLE_DEVICES` ([`benchmark-hygiene.md`](benchmark-hygiene.md)).
- **An arm is not disposed of until its record has been re-read.** When a round fanned work out to parallel
  arms, re-read each arm's record *at the moment you rank them* — the file on disk, not the copy you
  collected earlier — and check its mtime against that moment. An arm still writing when you collected reads
  as absent, and **absent is unfinished, not zero**; scored as a no-win it drops out of the comparison while
  still being counted in it, so the fan-out answers a smaller question than the one it was paid for. An arm
  record says whether it FINISHED separately from what it found, because a crash, a budget cut-off and a
  measured null are three different facts that otherwise look identical.
- **A close records who checked it, and what became of every challenge to it.** Name the auditor
  (`verified_by: self | fleet`): a self-check and an independent check leave the same artifacts behind, so
  without the field a reader cannot tell a cross-examined result from an unexamined one. Where a review was
  run, record each challenge with its disposition — accepted, or rebutted *with the on-disk reason it does
  not apply*. A review whose verdict is recorded but whose challenges are not itemized reads exactly like
  agreement, which is the one reading it must never be able to produce.

**Gluon-specific bindings — the six that decide whether a result is real:**

1. **The champion gate passes before the first round.** Not "read the bundle" — run `champion_gate.py`. A
   bundle whose source hash, config match or comparator cannot be verified makes every number this run
   produces unfalsifiable, and that is discovered at close, when it is expensive.
2. **Every difference between the anchor and the champion's own TTGIR is named, and an elective one carries
   the faithful variant's number too.** Faithful is the default and its regression versus the champion is
   expected. Diverging is allowed — a >100% anchor is a legitimate outcome and `parity_gate.py` already asks
   you to attribute it — but an *unrecorded* improvement destroys the equivalence anchor and with it the
   ability to tell a recovery from a win. Grades and the ledger: [`transcribe.md`](transcribe.md).
3. **The comparator is `champion_ms` from the bundle** — the tuned plain number, not the kernel's shipped
   default, and not the anchor. A win over the anchor is a recovery of what transcription lost, not a win.
4. **Recovery and improvement are separate stages, separated by the declared parity gate.** Not "compare the
   two numbers" — run `parity_gate.py` (threshold declared by the run, default 0.95), which also names the
   suspect. Below the declared criterion, a round is recorded as `recovery` against the named suspect it
   closed (lost pipeline / layout / RA); only above it does a delta count as a win. Climbing before the gap
   is attributed tunes around a transcription defect and mis-attributes the residual for the rest of the
   run. Numbers obtained by re-injecting plain's pipeliner are labelled `injected` and never count as a win.
5. **A Gluon result slower than the champion is a negative, never the winning stage.** Report
   `negative_revert_plain` with the measured gap; Director records the cross-skill
   verdict. A Gluon wall does **not** prove plain was the ceiling — that is a separate claim needing its own
   evidence.
6. **Validate across the served shape range before claiming a win.** Optimizing to the champion's single
   anchor shape can produce a clean-looking win that loses over most of the range's GPU-seconds; bucket the
   dispatch if the crossover does not close. The bundle's `served_range` is the shape list.

**Imports** — supported surface only (Triton 3.8.0; 3.6/3.7 differences are downgrade notes in `../gluon/`):

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

**The anchor recovers the compiler-inferred layouts** from the champion's pinned `.ttgir` (`#blocked` /
`#mma|#amd_mfma` / `#shared` + `convert_layout`, and the recorded `num_stages`, which on the Gluon path is a
budget parameter / champion record only — no 3.8.0 pass consumes it) and re-expresses them explicitly
(`scripts/ttgir_bridge.py recover`, which asks the compiler's own `layoutToGluon()`; `ttgir_to_gluon.py` is
the text-level fallback; `recover_gluon.py` assembles the anchor). Faithful is the default, and its
regression versus the champion is expected — it is the baseline to climb from. Where you diverge on
purpose, the site goes in the ledger and the faithful variant is measured beside it
([`transcribe.md`](transcribe.md)).

## Two version faces, and only one of them ships

Every repo you cite — this pack, and the Triton source under it — has a working tree on some disk and a
commit a `git clone` delivers, and they diverge in one direction: the newest corrections are the ones not
yet committed. So "this pack already documents X" and "the tree says the compiler does Y" are claims about
a disk until they are read off a named commit or tag. Read them off one:

```bash
git show <commit>:<path>        # this pack's distributable face
git diff <commit> -- <path>     # what only this working tree has
git show v3.8.0:<path>          # Triton behaviour — the tag, never the checkout
git merge-base --is-ancestor <commit> v3.8.0   # is this change in the tag at all
```

Anything the `diff` lists is not yet knowledge a reader of this pack can obtain. A commit count or a date
settles nothing: a tree can be newer than a tag and still be on a branch that never reaches it.

## Sources

Merged from: `references/phases.md` (all sections); `references/method-reference.md` preamble
(`# tile-programming-gluon`, incl. "Two version faces"), `## 0. The flow, end to end`, `### The four
gates`, `## Non-Negotiable Rules`, and the `## 4. Stages: assert the champion, transcribe, recover to
parity, then climb` spine diagram; the compressed copies of the same sections in `tile-programming-gluon.md`
(incl. its spine list and "Standing references"); the G0–G4 table and the G0 launch-arg facts from
`skill.md` `### The gates are executable, and a gate is passed by TOOL OUTPUT, not by assertion` (read
only; the lead owns skill.md).

Conflicts resolved:
- Transcription tool: `recover_gluon.py` / `ttgir_to_gluon.py` as *the* recovery → `ttgir_bridge` recovers
  and verifies; `recover_gluon` assembles the anchor (plan "references 内部的重复和矛盾").
- RECOVER box: "the auto-pipeline Gluon dropped … re-inject" → repaid by hand-written mechanisms;
  re-injection is diagnostic / last resort, `injected` (pipeline priority rule).
- Parity threshold: "declared by the run" + default 0.95 (`parity_gate --threshold`).
- `num_stages`: dead on the Gluon path in 3.8.0.
- G4 / `ab_bench.py`: screening only; acceptance = GEAK verify timing + 2% commit gate (tool-conflict rule).
- Phase → tooling table: `scripts/profile_kernel.sh` → GEAK `kernel_workflow/scripts/profile_kernel.sh`;
  `scripts/gpu_lock.sh` → `kernel_workflow/scripts/gpu_lock.sh`; moved tools → `kernel_tools/`.
- Profile timing: GEAK skill.md's "do not put a full profiling pass in front of step 1" is overruled by the
  upstream budget-first / re-profile-every-round rule.
- `## Two run modes` (EMBEDDED vs STANDALONE) → `## How it runs in GEAK` (unpinned; one mode remains): the
  upstream gluon-direction-agent / captain column, "STANDALONE has no G0", the STANDALONE-only tool-discovery rule
  and `locus.sh (STANDALONE)` dropped or reframed as optional bookkeeping; the closure-skeptic agent → the
  deep_engineer's closure self-review (`close.md`).

Dropped: nothing. The old phase-list pointers (`phases/analyze.md`, `phases/harness.md`, `phases/profile.md`,
`phases/budget.md`, `escalation-gate.md`, `plain-strategies.md`, `champion-handoff.md`,
`phases/transcribe.md`, `phases/layer-loop.md`, `phases/evaluate.md`, `orchestration.md`) were rewritten to
the new method files; `phases/tune.md` is kept as a triton-pack path.
