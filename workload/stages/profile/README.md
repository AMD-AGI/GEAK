# Stage `profile` — locate kernel dispatch and workload shapes

**Input** `$IN` = baseline's out (`baseline.json`: which serving run is being attributed,
at what throughput, with what noise band).
**Output** with `kernel-profile`: measured dispatch/shape/source evidence and
`profile_handoff.json`, consumed by the separate [UT stage](../ut/README.md).
The legacy `kernel-extract` implementation also generates and validates tasks here.

## Implementations

| Directory | Notes |
| --- | --- |
| `kernel-profile/` | split workflow: sampling, dispatch, shapes and a hashed evidence handoff; set `params.output_kind="dispatch_evidence"` |
| `kernel-extract/` | default. GEAK main's `kernel-extract` skill (`e2e_workflow/knowledge/kernel_extraction_skills/`) with the shared tools it copies into each task bundled under `shared/`; `PROVENANCE.json` records the origin revision, the origin hash of every file and which ones differ |

The multiconfig scenario uses `profile: kernel-profile` followed by `ut: kernel-ut`.
See [HANDOFF.md](HANDOFF.md) for their interface. In this mode the profile finalizer
checks dispatch evidence without requiring UTs; drafting, scaffolding and the UT wave
described below run under the UT stage's separate output. Existing combined scenarios
retain their task/source qualification checks.

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

## Several configs (`adopt-ci-artifacts` baseline)

When `baseline.json` has `multi_config: true`, the run covers every config of one group
(default: one EP) of an InferenceX CI run, and each task still gets **one** UT. `$OUT` is
the group's eval dir; each config's observations and deployment live under
`$OUT/configs/<pid>/`. Plan the shared capture union before starting probe servers,
then project costs onto each config and merge. One group per run: a second EP group is a sibling
scenario whose baseline `select` names that EP.

1. **Per config**, for every `points[].point_id` in `baseline.json`
   (`EVAL_DIR=$OUT/configs/<pid>`): prepare observations and deployment before profiling.
   `observed_workload.py` takes its inputs from the point's
   `$RUN_ROOT/baseline/out/points/<pid>/observed_inputs.json` (the CI's own scheduler log,
   decode interval, measured window and completed tokens) -- or reuse the baseline's
   validated `observed_workload.json` -- and `serving.json` is that config's deployment.
   Follow [shared sampling](kernel-extract/references/multi-config-sampling.md):
   `multi_config.py plan` groups compatible deployments and deduplicates their observed
   points, `capture` profiles each pool once, and `project --merge` recomputes each
   config's own weights. Pass the scenario's global `sampling.num_points` budget,
   `context_tolerance`, and `target_pct`. Preserve every config's uncovered mass.
2. **Merge** (already done by `project --merge`; for independently captured legacy inputs):

   ```bash
   python3 "$WORKLOAD_ROOT/stages/profile/multi_config.py" merge --group-dir "$OUT" \
       --config <pid1>="$OUT/configs/<pid1>" --config <pid2>="$OUT/configs/<pid2>" ...
   ```

   This writes `$OUT/sampling_points_final.json` (all points, ids `<pid>__<point>`, per-config
   `phase_summary`), the per-point shape and census files, merged `source_map` /
   `adjacency` / `compile_variants`, and `kernel_ranking.json` in which every row's share is
   the equal-weight mean of its share in each config (absent = a measured 0) -- the common
   denominator shares are reported on, not the selection rule (step 3).
3. **Select per config, then take the union**:

   ```bash
   python3 "$WORKLOAD_ROOT/stages/profile/multi_config.py" select --group-dir "$OUT"
   ```

   An operator is a candidate as soon as it passes the configured rule -- phase Top
   `selection_top_k` AND a share of **that single config's own workload** above
   `selection_overall_pct_gt` (the multi-config scenario: Top 10 and >3%) -- in any one
   config; not on its share averaged over the
   configs, which would bury an operator that dominates some concurrencies and is absent at
   others. Each config is selected by the unchanged `select_targets.py` on its own
   `kernel_ranking.json`, by operator (against the merged source map, so operator keys
   agree across configs), with the configured scope; the union is written to
   `$OUT/selection_targets.json` with each target's `selected_in_configs` and
   `config_pct`. Then continue with §3-§5 on `$OUT` unchanged: `draft_task_decls.py`,
   review, `build_task_plan.py`, the UT wave.

   **Communication kernels are listed, not extracted.** Operators whose kernels are all of
   a kind in `selection_list_only_kinds` (default `["collective"]`: all-reduce, all-gather,
   NCCL/RCCL) are taken out of the targets and recorded under `listed_only` with their
   per-config share; they stay in the denominator and the report and get no task and no UT.
   A standalone collective UT reproduces neither the interconnect state, the rank skew nor
   the overlap with compute that decide its serving time; its real levers (fusing the
   all-reduce with the following norm, quantized or fewer collectives) lie outside one
   kernel's seam; and its multi-rank UT costs the whole node and is the flakiest gate.
   Set the list to `[]` to extract them anyway.
4. Every planned case now carries its `config_id` (sig `<pid>/<PHASE>_<shape>`), each config
   planned on its own bands. The UT pools the configs with
   `serving_weight_model.config_weighting` (`harness_lib.serving_weighted_speedup`, which
   also reports `per_config`):
   - `calls` (default): each config's call distribution is normalized to 1 and the configs
     are averaged, `w(M) = sum_c alpha_c * freq_c(M)`, metric `sum_M w(M) * t(M)`. A kernel
     sees M, not the concurrency, so this pools "a launch at M" across concurrencies; a
     config whose launches cost more (larger M) counts for more GPU time.
   - `time`: each config's baseline-time mass is normalized to `alpha_c`, i.e. the
     alpha-weighted harmonic mean of the per-config speedups.
   `alpha_c` is equal unless `serving_weight_model.config_weights` names it. A task whose
   kernels do not run in a config has no cases there and that config drops out of its pool.

### Merging across concurrency: what decides the task boundary

- **"The gain disappears at another concurrency" is usually a path switch, not a shape
  change.** At a larger M the server routes the same operator to a different kernel. So
  **the unit of merging is the operator entry, above the
  path selection**, never one Triton kernel: put `target_callable` above the M-dependent
  dispatch, so every config's cases enter the same seam, reach their own path, and the
  switch threshold itself is something the optimizer can move.
- **Select by operator, per config.** `multi_config.py select` selects each config by
  operator on its own workload and unions the results, so an operator that matters at some
  concurrencies only is not averaged away with zeros from the others.
- **M-dependent paths merge; launch-dependent paths do not.** A UT runs in one environment.
  Paths chosen by M are reached by the cases themselves (the `launches` gate sees all of
  them across the phase's cases). Paths chosen by a launch setting (attention backend, MoE
  backend, KV dtype, speculative settings, ...) cannot be: the baseline splits such configs
  into separate groups (`adopt-ci-artifacts` `split_on`), and `baseline.json`
  `launch_diff` lists what still differs inside a group (e.g. HiCache sizes) for review.
- **Cover every graph bucket.** Under CUDA graph the served M is the padded batch
  (`M = pad(bs) x tokens_per_req`, e.g. DFLASH 8 x the captured bs list). Add the captured
  buckets no point covers as task `envelope_cases` (tested, weight 0), so a regression at an
  M no sampled point hit is still visible.
- **Who decides:** `kernel-extract/scripts/merge_policy.py`, from the evidence; the rules are
  its docstring, the procedure `kernel-extract/references/multi-config-ut-merge.md`.
- **One owner per file.** When two tasks touch one module, one owns it; a whole-module
  replacement also replaces the other functions in that file: rebind the function instead.
- A merged dispatch file that loads `_*.py` helpers from its own directory must ship them in
  the overlay; a missing helper fails at import and the server reports the model as not
  registered. Check imports on one GPU before starting the 8-GPU server.


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
- For short collectives, qualify device timing under the serving graph/replay lifecycle.
  Host-issued single-call timing can hide kernel changes behind launch overhead. Establish
  repeatability with unchanged source and verify that the timer responds to device work;
  if batching is needed, report its count and per-call normalization. A timing repair
  invalidates earlier performance receipts and requires fresh qualification before search.
