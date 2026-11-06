# Module and stage-skill contract

A module defines a capability and an input/output/acceptance contract. One or more
implementation skills or tools perform that work. Kernel Workflow and native tools
(OOB), for example, are alternative kernel-optimization implementations with the same
frozen-task and verification contract. A module is not a position in an execution order.

## Direct module calls

The root agent follows `SKILL.md`. Each invocation's `PLAN.md` names its question,
module and implementation path, input identities, scope, output directory, constraints,
resource/time budget and decisive evidence. The task and explicit plan govern the scope;
implementation defaults do not silently broaden or narrow it.

Invoke the skill or tool with those inputs and a separate candidate/output directory.
A module may use a relevant part of a larger skill, combine tools, or return a running
handle. Keep the evaluation contract fixed and use the validation appropriate to the
claim: a trace observation, a qualified UT, a cold-verified kernel and a serving comparison
are different deliveries. Do not submit a partial delivery as a completed standard stage.

Return native receipts/logs and `REPORT.md` with execution status, findings, changed
artifacts and uncertainty. A losing experiment remains evidence. `STATE.md` holds pointers
and decisions, not a second copy of measured numbers or worker counters. Existing
`result.json` envelopes may be reused where appropriate; a diagnostic report need not
manufacture a kernel artifact. A new implementation does not redefine the module's gates.

The current stage finalizers remain authoritative for deliveries that meet their full
contract. Applyback verifies kernel integration and delegates the same serving comparison
to Evaluate; direct system experiments use their own effect/quality evidence. Preserve
native invalidity and the configured comparison basis in either route.

## Standard recipe adapter

Each of the five stages in `workload/` (`baseline`, `profile`, `roofline`, `optimize`,
`applyback`) is a **directory slot**. Every subdirectory of a slot is one interchangeable
**implementation**, and each implementation is itself a skill: it has its own `SKILL.md`
and its own code, and can be read or executed on its own, outside this pipeline.

Selecting a stage implementation means pointing `stages.<stage>.skill` at a directory name
in the same slot. Nothing else in the pipeline changes.

```
stages/<stage>/<skill-dir>/
    SKILL.md          required. An agent reads this and knows how to perform the step.
    ...               the skill's own code / references / assets, in any shape.
    run.sh            optional. See "Two execution shapes" below.
```

---

## 1. Calling convention

A stage implementation receives its context **only** through environment variables.
Hardcoding a machine-local absolute path (`/shared_nfs/zihao/...` and the like) inside a
skill is forbidden — read the path from `$PARAMS` instead.

| Variable | Meaning |
| --- | --- |
| `GEAK_ROOT` | GEAK repository root |
| `WORKLOAD_ROOT` | `$GEAK_ROOT/workload` |
| `SKILL_ROOT` | this implementation's own directory, i.e. `stages/<stage>/<skill>/` |
| `STAGE` | stage name |
| `STAGE_SKILL` | the skill directory selected for this stage |
| `RUN_ROOT` | root of this run; `$RUN_ROOT/<stage>/out` holds each stage's artifacts |
| `IN` | the upstream stage's out directory (the preceding entry in `order`); empty for the first stage |
| `OUT` | this stage's artifact directory, already created by the driver |
| `CONFIG` | path to the merged configuration JSON (read-only) |
| `PARAMS` | path to this stage's `params` subtree as JSON (read-only) |
| `RUNNER` | absolute path to the runner script; see `runners/CONTRACT.md` |

Read upstream artifacts through `$RUN_ROOT/<stage>/out`, not by splicing relative paths
onto `$IN`. `applyback`, for example, needs both `$RUN_ROOT/baseline/out/baseline.json`
and `$RUN_ROOT/optimize/out/optimized.json`.

`SKILL_ROOT` is read-only: an implementation must not write artifacts back into its own
directory; everything goes to `$OUT`. This is what makes running several copies of the
same skill in parallel within one run safe.

## 2. Two execution shapes

An implementation must support at least one of these, and may support both.

**A. Script shape (`run.sh` exists)** — for deterministic steps. The driver runs
`bash $SKILL_ROOT/run.sh` to completion and spends no tokens. This is what
`run.py run` invokes.

**B. Agent shape (no `run.sh`, or `"mode": "agent"` in the config)** — for steps that
require judgement (extracting a unit test, reading a trace, deciding a boundary). The
driver does not execute it; it hands an agent the environment table above and the path
to `$SKILL_ROOT/SKILL.md`, and the agent follows that document. The main session's agent
can do this directly; for unattended or long-running work, `$RUNNER` starts a separate
session.

Acceptance is identical for both shapes — section 3. The driver does not care whether a
script or an agent produced the artifacts.

## 3. Finishing: `$OUT/result.json`

Every implementation must end by writing `$OUT/result.json`. This is a stage's only
external interface.

```json
{
  "stage": "roofline",
  "skill": "kernel-roofline",
  "status": "ok",
  "artifacts": {"roofline": "roofline.json", "report": "REPORT.md"},
  "metrics": {"kernels": 7, "sum_expected_e2e_gain_pct": 3.4},
  "notes": "mode=tl-only; 3 kernels lack shape evidence and are marked null in roofline.json"
}
```

- `status` is one of three values:
  - `ok` — the stage completed, artifacts are present, downstream may proceed.
  - `blocked` — an external precondition is missing (no GPUs, no `aiperf`, receipt not
    found). This is **not** a failure, but downstream must not continue. `notes` must
    state what is missing and how to resume once it is supplied. A `blocked` from an
    agent-mode stage costs **one repair session** before the order stops: the agent is
    handed its own note back and asked either to satisfy the precondition — if it is a
    file it can build, a path it can find, a step it skipped — or to report `blocked`
    again naming exactly what is absent. The first note is preserved as
    `$OUT/result.<n>.json` and restored if the repair session leaves nothing valid, so a
    repair can never erase the diagnosis. A second `blocked` is the stage's answer.
  - `fail` — the work was attempted and did not succeed.
- Values in `artifacts` are paths **relative to `$OUT`**, never absolute. The driver
  checks that each one exists.
- `metrics` holds numbers used for reporting and gating; it may be empty.
- `$OUT/REPORT.md`, if present, is **this stage's own section of the run report**. The
  run-level renderer pastes it in verbatim and never summarises it, so a stage says what
  a stage's numbers mean. Write it from the skill if the skill is the one holding the
  numbers (`kernel-roofline` does); otherwise the slot's `finalize.py` writes it, since
  it already has every figure in hand at the moment it decides whether to accept them. A
  stage that writes none still appears in the contract table — nothing downstream breaks,
  it simply has no prose of its own.
- When output is incomplete, report `blocked`/`fail` honestly. Marking a half-finished
  stage `ok` lets downstream burn 48 hours on fabricated inputs — the most expensive
  mistake available in this pipeline.

The driver (`run.py validate`) checks this file and records the outcome in
`$RUN_ROOT/STATE.json`. Before each invocation it records `inputs.json`: the stage,
skill, parameters, target and hashes of preceding stages' `result.json` receipts.
Validation checks that these inputs still match. Resume revalidates retained results
before skipping them, without rerunning GPU tests. Changed upstream receipts or policy
require the original inputs or a fresh stage run; old artifacts remain available.
Older standalone deliveries without `inputs.json` retain envelope validation only.
A script that exits nonzero cannot publish `ok`, and a fresh script invocation preserves
the preceding result and log before it starts.
Declared file artifacts are copied into `$OUT/.previous_runs/<invocation>/` before a
rerun can overwrite them. Working directories stay in place for continuation; their
owning stage must keep native run/cache identities separate when producing new evidence.
Agent session numbering continues across invocations, preserving prompts, logs and
continuation context. Each attempt needs a fresh session or slot-finalizer result;
an empty retry cannot inherit an earlier `ok`. Reports honor driver and native-summary rejections even while
older artifacts remain present.

Write it with `_shared/emit_result.py` (both a library and a CLI), which validates the
structure before it touches disk:

```bash
python3 "$WORKLOAD_ROOT/stages/_shared/emit_result.py" --status ok \
    --artifact baseline=baseline.json --metric throughput=446.87889
```

`_shared/` holds the few helpers that **several slots need in order to judge** — today
the envelope writer above and `task_plan.py`, which reads task ownership out of a
`task_plan.json` whose key names differ between the standalone skill and older slot
deliveries. It belongs to this contract, not to the pipeline that runs it: a slot that
reached up into `engine/` for a tool would tie a replaceable skill to the plumbing, and
the point of the slot is that the plumbing can be swapped out from under it. Criteria
themselves stay in each slot's own `finalize.py` (section 3.1); nothing scenario-specific
or stage-specific goes here.

### 3.0 Saying which stack you ran against

A stage that can observe the stack it ran against should say so, either as an optional
`stack` object inside `result.json`:

```json
"stack": {"model": "amd/GLM-5.2-MXFP4",
          "image": "lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260913",
          "sglang_commit": "7f1f8c706a", "aiter_commit": "4ad998328"}
```

or by writing `$OUT/stack_identity.json` (a document that names the stack in force under
an `active_stack` key is read only there). `run.py validate` compares whatever it finds
against the `target` block of the config and **fails the stage on a contradiction**: if
the config says one model and the stage observed another, one of them is wrong and every
label downstream is wrong with it. A stage that observes nothing is reported as
`unverified`, which is a statement rather than a pass. See `engine/core/target.py` for the
whole gate, and `README.md` for the naming half of it.

### 3.1 The slot's verdict: `finalize.py`

**Every slot has `stages/<stage>/finalize.py`.** It reads `$IN`, `$OUT` and `$PARAMS`,
writes `$OUT/result.json`, and is the stage's conclusion. Whatever the implementation
wrote there — agent session or `run.sh` — is archived and overwritten. The driver runs it
after every stage of either shape; `run.py --dry-run` refuses a slot that does not have
one, so this is enforced before a campaign starts rather than discovered at the end of it.

```
stages/<stage>/finalize.py      judge $OUT -> result.json      (required, one per slot)
```

Two reasons it is the slot's file and not the skill's:

- **Nobody grades their own work.** A profile session once reported `ok` for four tasks
  delivered against six declared, with single-task fields in `metrics`, and nothing
  downstream could tell. Three of the five slots used to leave the verdict to whatever
  was plugged into them.
- **Swapping an implementation must not swap the criterion.** A third-party skill
  vendored as-is (such as `kernel-roofline`) knows nothing about `result.json` and must
  not be modified by a single byte to fit this pipeline (PRINCIPLES R7). The mapping, and
  the acceptance rule with it, live in the slot; a new implementation only has to produce
  the files named in that slot's `README.md` table.

What belongs in it, and what does not:

- **Yes:** did the delivery arrive, is every claim carried by a file somebody can open,
  and **did the configuration actually take effect** — `$PARAMS` is executable policy, and
  a skill's own documented default never overrides it. A run that quietly used the
  skill's default scope instead of the configured one is a `fail`, not a footnote.
- **No:** re-deriving a result the implementation already produced. A previous optimize
  `verify.py` re-judged kernel correctness, grew three output dialects of its own, and
  recorded a task that crashed before comparing any output as a correctness failure. The
  task's own unit test is the authority on the kernel. Check the evidence; do not
  recompute the finding.

Shared slot-level logic lives here too (how expected e2e gain is computed, for instance),
so that replacing the roofline implementation does not silently replace the ranking
criterion along with it. Where the folding or scoring algorithm ships with the skill, call
it during the implementation's work — the roofline README gives the native folding
command. Its finalizer reads the resulting evidence without duplicating the sum.

**The criterion is written once.** Prose that describes it — a slot `README.md`, a
`SKILL.md` — points at `finalize.py` instead of restating its numbers, because a second
copy of a rule is a rule that will drift.

## 4. Per-stage artifact contracts

The `result.json` envelope is generic, but the specific files downstream reads are fixed.
Each stage's `README.md` documents its own mapping; the summary is below. A new
implementation must produce files of the same names and structure to plug in.

This table says **what downstream opens**. It does not say what counts as enough of it —
that is the slot's `finalize.py` and is written only there (section 3.1). The two had
already drifted once: this table named `kernel_workload.json` while the profile slot's
verdict required `kernel_ranking.json`, and each was the authority depending on which one
you happened to read.

| Stage | Required artifacts | Key fields |
| --- | --- | --- |
| `baseline` | `baseline.json` | `throughput`, `unit`, `metric`, `noise_band`, `provenance` |
| `profile` | `kernel_ranking.json`, `selection_targets.json`, `tasks/<name>/` | per-kernel `pct_gpu_time`, `phase`, plus independently runnable unit-test task directories. The census denominator resolves from `kernel_workload.json` or from `kernel_ranking.json.denominators` |
| `roofline` | `roofline.json` | per-kernel `roofline_pct`, `attainable_speedup`, `bound_type`, `expected_e2e_gain_pct` |
| `optimize` | `optimized.json` | cold-verified kernels with `verified_speedup`, selected `kernel_src` directory, `kernel_src_hashes` and frozen `task_meta_sha256`; task-local binding adapters also carry `binding_file_sha256` |
| `applyback` | `ab.json` | `baseline`, `candidate`, `e2e_delta_pct`, `verdict` |

**The ranking key is `expected_e2e_gain_pct = pct_gpu_time × (1 − 1/attainable_speedup)`,
not operator speedup.** This scenario has twice produced an operator win of 1.8× / 1.34×
against a whole-machine change of +0.24% / −0.09%. Operator speedup and end-to-end delta
must be stated separately, and only a qualified serving comparison may support an end-to-end gain.

## 5. Writing a new implementation

1. `mkdir stages/<stage>/<name>/` and add a `SKILL.md` describing inputs, steps, outputs.
2. Read the environment per section 1, write `$OUT/result.json` per section 3, produce
   the downstream files per section 4.
3. Change one line in the config: `"<stage>": {"skill": "<name>"}`.
4. Confirm resolution with
   `python3 workload/run.py --config <cfg> --dry-run`.

No registry, no driver changes. A directory that exists and contains a `SKILL.md` is a
selectable implementation.
