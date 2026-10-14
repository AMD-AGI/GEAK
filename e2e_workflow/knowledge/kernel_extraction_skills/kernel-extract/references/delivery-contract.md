# Delivery and workflow acceptance

Read in `prepare`, `extract` and `extract_op`. The workflow consumes finished
UTs, not an intermediate selection return. The shared delivery helper reads
recorded evidence; it does not run GPU tests.

## Complete-stage handoff

Finish selection → source extraction → UT authoring → GPU validation inside
`PHASE=prepare`. Write one delivery file under `EVAL_DIR` and return only
`{"delivery_path":"/absolute/eval/path/ut_delivery.json"}`. Example file:

```json
{
  "workload_json": "kernel_extraction/round_resume/kernel_workload.json",
  "ranking_json": "kernel_extraction/round_resume/kernel_ranking.json",
  "tasks": [{
    "kernel": {
      "short_name": "example",
      "device_kernel": "exact_selected_device_symbol",
      "entity_kind": "gpu_kernel",
      "op_kind": "attn",
      "kernel_extraction": {"selection_pct_gpu_time": 7.2},
      "seam_candidates": []
    },
    "phase": "extract_op",
    "result_path": "kernels/example_task/run/RETURN.json"
  }],
  "failed_tasks": [],
  "deferred_tasks": [],
  "coverage_summary": {},
  "notes": "Coverage and limitations."
}
```

Paths are absolute or relative to `EVAL_DIR` and stay inside it; per-task
`task_dir` is absolute. `kernel` records the task boundary described below,
including real seam evidence and any joint/dispatch members (the empty example
above must be populated). One task owns each selected identity; retain original
per-member cost evidence. `phase` chooses the per-task return convention:
`extract_op` for op UTs, `extract` for other kernel UTs. Only validated tasks enter
optimization queues; no second extractor is called. The reader derives missing
anchor cost from the saved ranking, rather than requesting a new selection.

Record every selected task as completed, failed, or deferred with its reason
and artifact paths. Selection-only plans and bootstrap PASS do not count as
completed UTs. Report the tested scope and unresolved coverage explicitly.
Descriptions such as `source`, `notes` and `regime_summary` can be strings or
structured notes; their types do not reject a completed UT. Executable identity,
actual UT status, frozen baseline and binding/source evidence still matter.

For complete-stage per-task preflight, add `--prepared` to the command below;
it checks execution evidence without applying legacy descriptive-field schemas.
Fix packaging from saved evidence. Rerun GPU work only when execution evidence
is missing, failed, stale, or the source/inputs changed. Do not restart selection
because a descriptive field has a different JSON type.

## Declare the queued boundary

| Boundary | Declaration and selection evidence |
|---|---|
| One kernel or ordinary fused whole op | Anchor verdict; no required-member list; full callable correctness/timing and per-case dispatch checks |
| Mandatory chain, e.g. score → top-k or split-K → reduce | `KERNEL.joint_ut.phases[]`; one `joint_chain` verdict per phase |
| Whole op with alternative selected kernels | `KERNEL.dispatch_ut.phases[]`; separate anchor verdicts for the same whole-op callable |

Every phase/branch declares a unique nonempty `regime`, exact `device_kernel`
and structured `seam_candidates`. Joint phases also declare at least two distinct
`required_device_kernels`, including the anchor. Mandatory members must occur
together in every verified invocation; mutually exclusive variants belong in
separate dispatch branches. Do not combine `joint_ut` and `dispatch_ut`.
Branch labels may subdivide PREFILL/DECODE; retain the workload phase in cases.

Declare these fields on the delivery task's `kernel`, not only in UT metadata.
Within `prepare`, repair incomplete boundary declarations using saved evidence
before final handoff. Legacy per-task calls report `needs_selection_repair=true`
for an undeclared chain; neither path requires recapturing unchanged evidence.

Depth is relative to the whole boundary. A score-only child cannot replace a
joint chain; fused-op selection compares complete `op_seam` entries, while leaves
are verified inside the UT. Probe every relevant allowed candidate, preserving
its classification. Partition variant evidence using operands/state or verified
dispatch markers, not by retaining only traces with a desired kernel name.
Record other branches and their counts. A missing representation is a blocker,
not permission to rename a failing joint verdict to a leaf verdict.

If a deeper complete entry changes `target_callable`, adapt captured arguments
and state to the actual callee and record the conversion. Revalidate binding,
correctness, deployment mode and timing; the old entry's PASS cannot transfer.

## Required acceptance

- **GPU UT:** standard `python3 unittest.py` succeeds with oracle/random
  correctness, graph/compile/input refresh, candidate binding, numerical
  negative controls and complete two-leg timing. Every positive-mass served
  signature has both latencies. Extraction requires neither speedup >1 nor a
  new full-model baseline.
- **Source and isolation:** [source extraction](source-extraction.md) requires
  actual claimed native/JIT leaves, builds and execution proof under `kernel_src/`.
  A wrapper calling installed leaves proves only wrapper editability. Preserve
  the frozen serving baseline and prove separate baseline/candidate environments.
  The JS compatibility check accepts `baseline_frozen=true`, a `candidate_bind`
  object or nonempty `baseline_callable`; that declaration is not independent
  proof of isolation.
- **Selection:** original verdict has `contract="kernel_selection"`, `ok=true`,
  matching callable/kernel, positive observed/marker/matched counts, all relevant
  candidates tested and verified boundary depth. Joint returns contain exactly
  one `phase_selection_validations` row per declared phase, the matching member
  set, positive per-member evidence and complete `joint_call_coverage`. The
  primary callable must be a verified phase entry. Dispatch rows all verify
  the same whole-op callable.
- **Per-task result:** preserve lowercase `smoke="pass"`
  (`extract_op`) or `unittest_smoke="pass"` (`extract`), `task_dir` and
  `editable` not false. Keep phase evidence structured. Legacy per-task agent
  calls still obey their supplied return schema.

The workflow checks declarations and recorded evidence consistency. It does not
independently rerun the GPU assertions; preserve raw results and coverage limits.

## Machine evidence and preflight

Small verdicts can remain inline as `selection_validation`. For large ones use
`selection_validation_ref={"path":"run/selection.json","sha256":"..."}`:
paths must stay inside `task_dir`, and hashes cover exact file bytes. Missing,
changed or invalid files fail even with an inline PASS. Do not use `_same_as`,
prose paths, elided fields or reconstructed summaries as machine references.

Each joint/dispatch `phase_selection_validations[]` row contains `regime`,
`target_callable`, `seam_candidates` and its own verdict/reference. Repeat the
primary phase reference at top level. Keep files available through handoff.

Save the original `KERNEL` in `run/requested_kernel.json` and the schema-correct
return skeleton, with phase rows already present, in `run/draft_return.json`:

```bash
node "$SKILL_DIR/scripts/kernel_extraction_delivery.mjs" \
  --kernel "$TASK/run/requested_kernel.json" \
  --result "$TASK/run/draft_return.json" --phase extract_op \
  --verdict run/selection_DECODE.json \
  --phase-verdict DECODE=run/selection_DECODE.json \
  --phase-verdict PREFILL=run/selection_PREFILL.json \
  --out "$TASK/run/RETURN.json"
```

Use actual phase labels; omit phase options for ordinary tasks. The helper
attaches hashes and writes `--out` only on acceptance. It changes neither the
queued declaration nor verdicts. Omit attachment options to check packed JSON.
Keep the accepted JSON in `result_path`; `prepare` returns the delivery path.
Legacy per-task calls return the accepted JSON unchanged. Repair serialization errors from saved
raw evidence; rerun GPU work only for missing/invalid evidence or changed
code/inputs. Preserve real failures rather than retrying unchanged UTs.

## Source delivery records

When requested, retain `KERNEL.source_delivery_required=true`. The gate requires
all `selected_device_kernels` (or the anchor), plus `source_device_kernels`
when supplied. Return `source_delivery_ref={path,sha256}` for this task-local manifest:

```json
{
  "contract": "kernel_source_delivery/v1",
  "target_callable": "package:complete_op",
  "snapshot_sha256": "...",
  "files": [{"path": "cases.py", "sha256": "..."}],
  "ut_result_ref": {"path": "run/ut_result.json", "sha256": "..."},
  "kernels": [{
    "device_kernel": "exact_selected_symbol",
    "source_files": ["kernel_src/implementation.py"],
    "execution_ref": {"path": "run/leaf_execution.json", "sha256": "..."},
    "negative_control_ref": {"path": "run/leaf_negative.json", "sha256": "..."}
  }]
}
```

Generate `files`/`snapshot_sha256` with
`node "$SKILL_DIR/scripts/kernel_source_delivery.mjs" snapshot "$TASK"`, and
references with `ref TASK FILE`. The snapshot covers `meta.json`, `cases.py`,
`unittest.py`, `kernel_src/` and additional harness/build/input adapters listed
in `meta.kernel_extraction.source_delivery_files`. Freeze before GPU validation.

Full UT JSON reports `ok:true` (or `smoke:"pass"`), `exit_code:0`, the actual
`target_callable` and tested snapshot digest. Per-leaf records carry `kind`
(`candidate_execution` or `numerical_rejection`), `ok:true`, `device_kernel`,
`target_callable`, `snapshot_sha256`, executed `command`, locked `gpu` set and
hashed `log_ref`. Execution needs positive `invocations`; negative controls need
`launch_succeeded:true` and positive `numerical_mismatches`, not build/import
failure. Record the clean snapshot, mutation and mutated-source identity in the
logs; restore clean code and run final UT. Record each requested leaf separately.

Changed source, entry, cases or harness invalidate prior results. These records
prove consistency only when generated from actual outcomes; hashes or field
names alone cannot establish execution. Report delivery, GPU UT, coverage and
e2e outcomes separately.
