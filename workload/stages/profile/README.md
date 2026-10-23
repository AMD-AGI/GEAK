# Stage `profile` — decompose an AgentX reproduction into independently runnable kernel unit tests

**Input** `$IN` = baseline's out (`baseline.json`: which serving run is being attributed,
at what throughput, with what noise band).
**Output** `$OUT`: per-kernel share of GPU time, plus a frozen unit-test task directory
for each selected kernel.

## Implementations

| Directory | Notes |
| --- | --- |
| `agentx-kernel-extraction-standalone/` | default. An external archive vendored byte-for-byte, carrying its own `SKILL.md`, `scripts/`, `references/`, `tests/` and `PROVENANCE.json`. Integrity is enforced by `../MANIFEST.agentx-kernel-extraction-standalone.json` |

This skill is **agent-shaped** (contract §2B): it has no `run.sh`, because choosing a
boundary, identifying the callable and deciding which dispatch branch constitutes a whole
operator are judgement calls, not a procedure. The driver only prepares the environment;
an agent reads the skill's `SKILL.md` and follows it.

The skill's own convention has `SKILL_ROOT` pointing at itself and `EVAL_DIR` as its
working directory. Wired into this pipeline:

```
SKILL_ROOT = $SKILL_ROOT     (already exported by the driver; the names coincide)
EVAL_DIR   = $OUT
```

## Candidate discovery and task planning

For the bundled standalone extraction skill, pass the upper-level scope through its
existing selector rather than using its narrower default invocation:

```bash
python3 "$WORKLOAD_ROOT/stages/profile/select_candidates.py"
```

The adapter only forwards `selection_top_k`, `selection_overall_pct_gt` and
`selection_phases` from `$PARAMS` to `$SKILL_ROOT/scripts/select_targets.py`. The default
is phase Top 10 and overall share >0.5%; these are heuristic discovery cutoffs. Numeric
parameters are the executable policy; `selection` is explanatory prose. The vendored
selector and its validation remain unchanged. Declare other material phases explicitly
in the scenario, such as draft and verify when they are independently measured.
Another profile implementation uses its own selection procedure with the same configured
scope and artifact contract; it need not implement this optional selector CLI.

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

The skill knows nothing about `result.json` — it is an external archive and should not be
edited to fit a pipeline. That mapping is performed by `finalize.py` in the **slot
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

## Acceptance rules

- **What was configured is what must have run.** `selection_targets.json.policy` must
  match `$PARAMS`' `selection_top_k` / `selection_overall_pct_gt` / `selection_phases`;
  `kernel_ranking.json.selection_scope.selector.args` and, when present, the
  `selection_invocation` receipt must agree with it. A disagreement is `fail`, with both
  sets of numbers named. The skill's SKILL.md default never overrides `$PARAMS`: a run
  configured for Top 10 / >0.5% that quietly selected at the documented Top 5 / >3%
  delivered a quarter of the candidate space and reported success.
- **Capacity is not an extraction cap.** `stages.optimize.params.task_limit` is the
  optimize stage's in-flight GPU capacity. Extract as far as the evidence goes; the
  surplus is recorded as `tasks_capacity_deferred` and scheduled in a later wave. A
  delivery is never truncated, and an existing extraction is never deleted, to fit it.
- Every selected identity has a task or a stated blocker. Record the exact mapping in
  `results.json` (`task_plan.json` until §5 exists): a real dispatch boundary or fused
  chain can cover multiple identities, so `targets_selected` and `tasks_built` need not be
  equal. Group by the mathematical and serving boundary, never merely to reduce the task
  count. An identity named to a task directory that does not exist is unaccounted for, not
  covered. Missing coverage must not be papered over by renormalizing the remainder
  to 100%.
- **Each task carries both receipts.** `run/ut_result.json` green, and `run/RETURN.json`
  present and naming this task. Missing or failing either is `blocked` and the task is
  named: "the gates were never run" and "every gate is green" must not produce the same
  `result.json`. The delivery gate is a Node script — on a host without a `node` binary,
  run it in the container that has one and say so rather than skipping it.
- **Declared coverage and achieved coverage are both reported.** Owning a boundary is not
  timing it: one delivery declared 42.92% of GPU time and timed 22.54%, because a task's
  captured decode operands launched neither kernel it owned. A missing achieved number, or
  one exceeding declared, is `blocked`.
- **The four acceptance claims stay separate** (SKILL.md §5): inventory completion, task
  completion, GPU UT status, delivery consistency. None may stand in for another.
- A task directory without `unittest.py` cannot be run by the optimize stage and is
  rejected outright.
- Declare `meta.json.required_gpus` before freezing and qualifying a task, or supply an
  explicit per-task GPU map through the optimize configuration. The serving model's TP
  size does not imply that a per-rank compute UT needs that many devices; a collective
  must retain its actual full rank set.
- Place each task in its final directory and finish its selection evidence, observed
  weights, callable binding and metadata before freezing the first full GPU test.
  Keep later progress and resource updates outside that frozen snapshot. Moving a task
  or rewriting metadata after a pass can invalidate its evidence and require another
  full qualification; complete those changes before paying for the test.
- When GPUs are unavailable it is acceptable to complete the CPU-only portion and report
  `blocked`, but a CPU-only check must never be presented as a complete extraction.
- Qualify the editable-source contract as well as the original implementation: a
  semantics-preserving change to a declared candidate leaf must remain runnable, while
  the existing numerical negative controls must still be detected. A pristine guard
  that requires every candidate byte to equal the original makes optimization impossible.
  Freeze the oracle, inputs, harness and undeclared sources; declare every leaf needed
  by a joint operator in metadata before dispatch.

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
- Run each delivered Python UT through `python3 "$WORKLOAD_ROOT/scripts/run_task_entry.py" "$TASK_DIR"`
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
- For short collectives, qualify device timing under the serving graph/replay lifecycle.
  Host-issued single-call timing can hide kernel changes behind launch overhead. Establish
  repeatability with unchanged source and verify that the timer responds to device work;
  if batching is needed, report its count and per-call normalization. A timing repair
  invalidates earlier performance receipts and requires fresh qualification before search.
