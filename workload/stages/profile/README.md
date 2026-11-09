# Stage `profile` — decompose an AgentX reproduction into independently runnable kernel unit tests

**Input** `$IN` = baseline's out (`baseline.json`: which serving run is being attributed,
at what throughput, with what noise band).
**Output** `$OUT`: per-kernel share of GPU time, plus a frozen unit-test task directory
for each selected kernel.

## Implementations

| Directory | Notes |
| --- | --- |
| `kernel-extract/` | default. GEAK main's `kernel-extract` skill (`e2e_workflow/knowledge/kernel_extraction_skills/`) with the shared tools it copies into each task bundled under `shared/`; `PROVENANCE.json` records the origin revision, the origin hash of every file and which ones differ |

This skill is **agent-shaped** (contract §2B): it has no `run.sh`, because choosing a
boundary, identifying the callable and deciding which dispatch branch constitutes a whole
operator are judgement calls, not a procedure. The driver only prepares the environment;
an agent reads the skill's `SKILL.md` and follows it.

The skill's own convention has `SKILL_ROOT` pointing at itself and `EVAL_DIR` as its
working directory. Wired into this pipeline (the skill's
`references/pipeline-handoff.md` gives the full mapping):

```
SKILL_ROOT   = $SKILL_ROOT     (already exported by the driver; the names coincide)
SHARED_TOOLS = $SKILL_ROOT/shared
EVAL_DIR     = $OUT
```

## Candidate discovery and task planning

For the bundled standalone extraction skill, pass the upper-level scope through its
existing selector rather than using its narrower default invocation:

```bash
python3 "$WORKLOAD_ROOT/stages/profile/select_candidates.py"
```

The adapter forwards `selection_top_k`, `selection_overall_pct_gt` and
`selection_phases` from `$PARAMS` to `$SKILL_ROOT/scripts/select_targets.py`. The default
is phase Top 10 and overall share >0.5%; these are heuristic discovery cutoffs. Numeric
parameters are the executable policy; `selection` is explanatory prose. The vendored
selector and its validation remain unchanged. Declare other material phases explicitly
in the scenario, such as draft and verify when they are independently measured.
Another profile implementation uses its own selection procedure with the same configured
scope and artifact contract; it need not implement this optional selector CLI.

When the extraction skill has produced reviewed operator attribution, retain it:

```bash
python3 "$WORKLOAD_ROOT/stages/profile/select_candidates.py" --source-map "$OUT/source_map.json"
```

This uses the same configured cutoffs over complete operators. Dropping the map can
exclude a fused operator whose individual kernels each fall below the threshold.
The invocation receipt hashes the map as well as the ranking and selector; a changed
map requires a fresh output path, while identical inputs reuse the existing selection.

`selection_candidates.json` is the broad inventory, not a promise that every row is a
separate UT. Reconcile live identities, complete operator boundaries, available source
and existing delivery evidence. Plan as many useful nonoverlapping tasks as the selected
scope supports; do not stop at the old Top-5/>3% subset, and do not stop at
`stages.optimize.params.task_limit` either — that is the optimize stage's in-flight GPU
capacity, and extraction stops where the evidence stops. Producing more tasks than one
wave can hold is a scheduling fact, recorded as `capacity_deferred` by optimize, not a
reason to extract less. Produce `selection_targets.json` for that explicit extraction
scope, preserving the selector's original candidates and recording every deferred
identity, share and reason in the task plan. A source blocker and a capacity deferral are
different dispositions.
Rank work by expected system savings when supported, and label unknown bounds rather
than inventing them. Never duplicate parent/child work to fill the task count.

Keep `selection_scope`, `candidate_inventory` and `coverage_summary` as required by the
skill's selection-coverage reference, including material unselected rows and phase
leaders. Inventory completeness, selected-task completion and tested case coverage are
separate claims. Reuse compatible census/capture evidence; a scope expansion requires
qualification of the added tasks, not an automatic repeat of the whole serving trace.

## Artifacts → `result.json`

The skill knows nothing about `result.json`, and should not need to: it stays usable
outside this pipeline. That mapping is performed by `finalize.py` in the **slot
directory**. Because it belongs to the slot rather than to any implementation, it keeps
working when the profile skill is replaced, as long as the new skill emits files of the
same names.

```bash
python3 "$WORKLOAD_ROOT/stages/profile/finalize.py"
```

It is the slot's verdict, not a format converter: it rereads the delivery and judges it
against this stage's configuration. It runs after the agent session and its `result.json`
is the one that counts, so a session that reports its own success still fails here if the
evidence disagrees.

| File | Produced by | Consumed by |
| --- | --- | --- |
| `kernel_ranking.json` | §2 | input to `select_targets.py`; required. Must carry `selection_scope`, `candidate_inventory` and `coverage_summary` |
| `selection_targets.json` | §2, `select_targets.py` | **the roofline stage's ranking base** (`targets[].overall_pct` is `pct_gpu_time`); required. Its `policy` block is the receipt that the configured scope reached the selector |
| `kernel_workload.json` | §2, workload census | the denominator for all shares. Either this file (`total_gpu_time_ms`, `total_device_ms` or `total_ms`) or `kernel_ranking.json.denominators` must supply it — a reused census stays where it is, but the number must be nameable |
| `task_plan.json` | §3, boundary declarations | audit, and the fallback owner map before §5 exists |
| `task_gpu_requirements.json` | optional task-name → GPU-count map, when frozen metadata lacks `required_gpus` | mixed-GPU optimize dispatch; does not alter a qualified harness |
| `tasks/<name>/` | §4, frozen harness | **one per GPU in the optimize stage**; must contain `unittest.py`, `meta.json`, `kernel_src/`, `cases.py` |
| `tasks/<name>/run/ut_result.json` | §4, the task's own GPU run | the per-task gate verdict: `ok: true` with no `failed_gates` |
| `tasks/<name>/run/RETURN.json` | §4, `kernel_extraction_delivery.mjs` | the delivery gate's own receipt — it is written only when that report is `ok`, so its absence is a failed gate, not a missing file |
| `REPORT.md` / `results.json` | §5 | reporting, and the authoritative identity disposition. `identities[]` carries each identity's `task` or its blocker; `coverage.delivered_declared_pct_gpu_time` and `coverage.delivered_achieved_pct_gpu_time` carry the two coverage numbers |

`metrics` records the census (`kernels_censused`, `census_denominator_ms`,
`census_denominator_source`), the configured scope that was verified
(`selection_top_k`, `selection_overall_pct_gt`), the selection
(`targets_selected`, `selected_pct_gpu_time`, `identities_owned_by_tasks`,
`identities_blocked`, `owned_pct_gpu_time`, `blocked_pct_gpu_time`), the tasks
(`tasks_built`, `tasks_within_capacity`, `tasks_capacity_deferred`,
`tasks_ut_passed_all_gates`, `tasks_delivery_gate_passed`), both coverage numbers
(`declared_pct_gpu_time`, `achieved_pct_gpu_time`) and the four acceptance claims
(`acceptance_inventory_completion`, `acceptance_task_completion`,
`acceptance_gpu_ut_status`, `acceptance_delivery_consistency`).

## What is judged, and where

`finalize.py` is the verdict (`stages/CONTRACT.md` 3.1) and the rules are written only
there. It rereads the delivery and judges it against `$PARAMS`: that the scope actually
selected is the scope that was configured, that the census denominator is nameable, that
every selected identity has a task or a stated blocker, that each task carries both
receipts, that declared and achieved coverage are both present and consistent, and that
the four acceptance claims stay separate. Read that file — or run it early with the
command above — rather than looking for a second copy of its numbers here.

Two rules worth stating in prose because they shape what you build, not just what passes:

- **The skill's own `SKILL.md` default never overrides `$PARAMS`.** A run configured for
  Top 10 / >0.5% that quietly selected at the documented Top 5 / >3% delivered a quarter
  of the candidate space and reported success. That is why the slot compares the two.
- **Capacity is not an extraction cap.** `stages.optimize.params.task_limit` is the
  optimize stage's in-flight GPU capacity. Extract as far as the evidence goes; the
  surplus is recorded as `tasks_capacity_deferred` and scheduled in a later wave. A
  delivery is never truncated, and an existing extraction is never deleted, to fit it.

## What the delivery must contain

None of this is checked by code; all of it decides whether the evidence means anything.

- A real dispatch boundary or fused chain can cover multiple identities, so
  `targets_selected` and `tasks_built` need not be equal. Group by the mathematical and
  serving boundary, never merely to reduce the task count, and never duplicate
  parent/child work to fill a count. Missing coverage must not be papered over by
  renormalizing the remainder to 100%.
- The delivery gate is a Node script. On a host without a `node` binary, run it in the
  container that has one and say so rather than skipping it. The bundled skill's
  standalone note says no external workflow consumes `RETURN.json`; this slot still
  requires the local receipt. Run the skill's documented delivery preflight with `--out`
  before finalizing: its CPU preflight and the full GPU UT are separate evidence and
  neither replaces the other.
- Declare `meta.json.required_gpus` before freezing and qualifying a task. The serving
  model's TP size does not imply that a per-rank compute UT needs that many devices; a
  collective must retain its actual full rank set. The optimize scheduler reads this
  number per task.
- Place each task in its final directory and finish its selection evidence, observed
  weights, callable binding and metadata **before** the first full GPU test. Moving a task
  or rewriting metadata after a pass can invalidate its evidence and require another full
  qualification; complete those changes before paying for the test.
- Tasks are built from the skill's UT contract families (`kernel-extract/ut_contracts/`):
  the family fixes operand distributions, layout facts and per-output error bounds, and
  `ut_driver.py` runs every gate. A task-authored contract (`--family-file`) is the route
  for an operator no family covers.
- Qualify the editable-source contract as well as the original implementation: a
  semantics-preserving change to a declared candidate leaf must remain runnable, while the
  existing numerical negative controls must still be detected. A pristine guard that
  requires every candidate byte to equal the original makes optimization impossible.
  Freeze the oracle, inputs, harness and undeclared sources; declare every leaf needed by a
  joint operator in metadata before dispatch.
- When GPUs are unavailable it is acceptable to complete the CPU-only portion and report
  `blocked`, but a CPU-only check must never be presented as a complete extraction.

## Pitfalls

- Admit bounded oracle samples only after the intended workload phase starts. Health
  checks and client warmup can exhaust a small case quota before real traffic arrives.
  Record the native phase boundary used; a service-startup message containing
  "profiling" does not establish it. Earlier samples remain useful correctness smokes,
  but their counts are not measured-workload weights or coverage.
- Map logged live batch sizes to the deployed graph's captured shapes before freezing
  cases. Padded rows can still pass through norm, routing and collectives; attention may
  instead give them a distinct context length. Keep sampled call weights separate from
  executed tensor geometry, and record source evidence for each padding rule.
- A just-exited serving container can leave transient GPU activity. If the lease refuses
  before the task starts, begin the configured GPU-idle retry window at that refusal;
  time spent waiting for the lease has its own bound. Retain each refusal.
  Do not disable the idleness check, retry a launched UT automatically, or
  interpret a lease refusal as a kernel correctness failure.
- Keep a launcher unchanged while any process is reading it from shared storage.
  Replacing an NFS shell script can cause a stale-file-handle error in an active Bash
  reader even after its GPU child succeeds. Publish a separate immutable version when
  a change cannot wait, and retain the earlier file until its readers finish. Preserve
  the native task verdict and the outer launcher's exit as separate evidence.
- Run each delivered Python UT through `python3 "$WORKLOAD_ROOT/engine/campaign/run_task_entry.py" "$TASK_DIR"`
  in the pinned runtime, with the normal GPU lease and physical isolation around it.
  Directly launching a file named `unittest.py` can shadow Python's standard-library
  `unittest` package and fail in `unittest.mock` before any kernel is tested. The shared
  entry fixes import setup and preserves the task's own verdict.
- `overall_pct` in `selection_targets.json` is the share of the **whole workload**, not
  the share within the trace window. Window share is a different number: the capture
  window for this scenario once held a prefill:decode ratio of 32:7 against a true ratio
  of 1:1119, and ranking on window share would have promoted prefill kernels while
  end-to-end time was bound in decode.
- The standalone selector ranks before filtering and does not backfill. The slot supplies
  its broader explicit parameters; raising only `max_targets` does not widen discovery,
  and `task_limit` never widens or narrows it at all. Keep the full candidate receipt when
  narrowing the task plan.
- Routed MoE inputs must preserve observed expert occupancy and the local/global routing
  scope. Replacing captured routing with an all-local synthetic case changes the work
  being optimized even if every output comparison passes.
- Inspect the scored call, including rank workers: keep fixture-only staging, control
  traffic and output checks outside it; retain the target's own dispatch, allocation
  and copies. Device events do not exclude host waits inside that call. Timing and
  correctness must use the same computation, preserving sampled outputs for native
  checks before later calls overwrite them.
  For short collectives, qualify timing under the serving eager/graph lifecycle.
  Host-issued single-call timing can hide kernel changes behind launch overhead. Establish
  repeatability with unchanged source and verify that the timer responds to device work;
  if batching is needed, report its count and per-call normalization. A timing repair
  invalidates earlier performance receipts and requires fresh qualification before search.
