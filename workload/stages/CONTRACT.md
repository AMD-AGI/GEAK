# Stage-skill contract

Each of the five stages in `workload/` (`baseline`, `profile`, `roofline`, `optimize`,
`applyback`) is a **directory slot**. Every subdirectory of a slot is one interchangeable
**implementation**, and each implementation is itself a skill: it has its own `SKILL.md`
and its own code, and can be read or executed on its own, outside this pipeline.

Replacing a module means pointing `stages.<stage>.skill` at a different directory name
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
`stage.py run` invokes.

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
    state what is missing and how to resume once it is supplied.
  - `fail` — the work was attempted and did not succeed.
- Values in `artifacts` are paths **relative to `$OUT`**, never absolute. The driver
  checks that each one exists.
- `metrics` holds numbers used for reporting and gating; it may be empty.
- When output is incomplete, report `blocked`/`fail` honestly. Marking a half-finished
  stage `ok` lets downstream burn 48 hours on fabricated inputs — the most expensive
  mistake available in this pipeline.

The driver (`scripts/stage.py validate`) checks this file and records the outcome in
`$RUN_ROOT/STATE.json`; stages marked `ok` are skipped on resume.

Write it with `scripts/emit_result.py` (both a library and a CLI), which validates the
structure before it touches disk:

```bash
python3 "$WORKLOAD_ROOT/scripts/emit_result.py" --status ok \
    --artifact baseline=baseline.json --metric throughput=446.87889
```

### 3.0 Saying which stack you ran against

A stage that can observe the stack it ran against should say so, either as an optional
`stack` object inside `result.json`:

```json
"stack": {"model": "amd/GLM-5.2-MXFP4",
          "image": "lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260913",
          "sglang_commit": "7f1f8c706a", "aiter_commit": "4ad998328"}
```

or by writing `$OUT/stack_identity.json` (a document that names the stack in force under
an `active_stack` key is read only there). `stage.py validate` compares whatever it finds
against the `target` block of the config and **fails the stage on a contradiction**: if
the config says one model and the stage observed another, one of them is wrong and every
label downstream is wrong with it. A stage that observes nothing is reported as
`unverified`, which is a statement rather than a pass. See `scripts/target.py` for the
whole gate, and `README.md` for the naming half of it.

### 3.1 Slot-level adapters

A third-party skill vendored as-is (such as `agentx-kernel-extraction-standalone`) knows
nothing about `result.json`, and **should not** be modified by a single byte in order to
fit this pipeline. That mapping lives in the **slot directory**, not in the skill:

```
stages/profile/finalize.py      scan $OUT's artifacts -> result.json
stages/roofline/rank.py         fold + compute the ranking key -> roofline.json + result.json
```

Because it belongs to the slot, swapping in another implementation of the same stage
**does not require changing it** — the new implementation only has to produce the files
named in that slot's `README.md` table. Shared slot-level logic (how expected e2e gain is
computed, for instance) lives here too, so that replacing the roofline implementation
does not silently replace the ranking criterion along with it.

## 4. Per-stage artifact contracts

The `result.json` envelope is generic, but the specific files downstream reads are fixed.
Each stage's `README.md` documents its own mapping; the summary is below. A new
implementation must produce files of the same names and structure to plug in.

| Stage | Required artifacts | Key fields |
| --- | --- | --- |
| `baseline` | `baseline.json` | `throughput`, `unit`, `metric`, `noise_band`, `provenance` |
| `profile` | `kernel_workload.json`, `selection_targets.json`, `tasks/<name>/` | per-kernel `pct_gpu_time`, `phase`, plus independently runnable unit-test task directories |
| `roofline` | `roofline.json` | per-kernel `roofline_pct`, `attainable_speedup`, `bound_type`, `expected_e2e_gain_pct` |
| `optimize` | `optimized.json`, `verified.json` | kernels that survived cold re-verification, each with `speedup` and a `kernel_src` path |
| `applyback` | `ab.json` | `baseline`, `candidate`, `e2e_delta_pct`, `verdict` |

**The ranking key is `expected_e2e_gain_pct = pct_gpu_time × (1 − 1/attainable_speedup)`,
not operator speedup.** This scenario has twice produced an operator win of 1.8× / 1.34×
against a whole-machine change of +0.24% / −0.09%. Operator speedup and end-to-end delta
must be stated separately, and only `applyback` may claim an end-to-end gain.

## 5. Writing a new implementation

1. `mkdir stages/<stage>/<name>/` and add a `SKILL.md` describing inputs, steps, outputs.
2. Read the environment per section 1, write `$OUT/result.json` per section 3, produce
   the downstream files per section 4.
3. Change one line in the config: `"<stage>": {"skill": "<name>"}`.
4. Confirm resolution with
   `python3 workload/scripts/stage.py --config <cfg> --dry-run`.

No registry, no driver changes. A directory that exists and contains a `SKILL.md` is a
selectable implementation.
