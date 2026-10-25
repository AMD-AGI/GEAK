# Compose complete workflow lineages

`interface/runtime/engine/compose_kernel.mjs` runs one complete existing workflow
per lineage, then measures its eligible finalist with one common native evaluator.
A population can have one member. Every member may use `optimizer=greedy` or
`optimizer=evo`; names have no policy meaning. The workflow, its optimizer, prompts,
argument bytes, stopping rules and finalization stages are unchanged.

This is the process and finalist boundary of a prospective population. It does
not implement peer publication/import, archive-wide selection, or an evolutionary
advantage. A lineage keeps its private loop and archive in its own workflow
process. Stock early-stop refusals remain in that workflow; an accepted self-stop
finishes its postlude. The composer calls each workflow once, waits for all members,
and neither retires a member because a peer is faster nor restarts a stopped member.
Every lineage has the same persistence contract: the owner retains its declared
allowance through the deadline; the composer has no retirement mechanism or
persistence switch.

## Declared policies and owned contexts

Schema `geak.kernel-composition/2` requires two policy fields per lineage:

- `finalist: "required"` gives one member first common measurement, tie preference,
  retained-program fallback if its common measurement fails and no common winner
  exists, and a nonzero CLI
  exit when that member cannot be qualified. Exactly one member is required.
- `finalist: "eligible"` admits every other member through the same validation and
  measurement rules. A failed member cannot cancel its peers.
- `imports: "none"` is the only supported import policy. Other values are rejected.
  There is no live peer archive or publication path in this version.

The required policy is generic: it does not imply that the member uses stock
GEAK. A claim about containing stock GEAK needs an actual included unchanged
`greedy` workflow with the intended stock inputs, context and service allowance.

The operational owner supplies isolated execution contexts and reserves their
service through the declared deadline. Concurrency and device grants describe
that allowance; the composer does not provision containers, reserve host capacity,
attest mounts, or prove continuous utilization. Unused allowance is not reassigned
by the composer. External provider limits remain in force.

The composer checks these identities before any workflow starts:

- distinct context IDs, argument realpaths and physical device grants;
- explicit absolute `exp_root`, `eval_dir`, `learned_dir`, `kb_artifacts_dir` and
  `kb_store_dir` for every member, optional `state_dir`, `shared_kb` and `global_kb`,
  all declared native command state roots, output directories and
  recorder paths; these cannot overlap another member's corresponding paths or
  the common evaluator's state root, including through existing symlink parents;
- `resources.gpu_ids`, in positional correspondence with physical `device_ids`,
  matches the workflow's comma-separated `args.gpu_ids`; local device numbers may
  repeat across isolated contexts, but physical grants must be disjoint;
- each workflow's `deadline_epoch` equals the manifest's common absolute deadline,
  and each outer timeout covers that deadline plus seven seconds of cleanup;
- pinned native task/objective identity and existing optimize-mode handoff rules.
- `kb_remote=off` after trimming and case normalization; omitted or `auto` is rejected.

These are declared path and mapping checks, not proof of isolation in a remote
namespace. The owner must ensure that those same paths identify the intended
private storage in each execution context, including the Director's working tree,
runtime sessions, caches, local knowledge stores and learned-card writes. The
returned `eval_dir` is checked against its declaration before retention; that check
cannot prevent a misconfigured child from writing elsewhere during its run.

The single-language lane accepts an optional `learned_dir` sink; standalone calls
keep the default `workflow_dir/knowledge/learned`. `workflow_dir` remains the shared
code, role and helper root. The override passes through optimize/author calls;
the bakeoff interface is unchanged. Compositions require all three explicit knowledge roots
even when warm start or curation is disabled, because independent citation and
writeback paths can still write. `global_kb` is conservatively owned as a memory
hook; these checks do not assert that an active writer uses it.

The caller seeds each private learned sink with the intended common baseline,
including its README, index and cards; an empty replacement changes curation
context. Planner and engineer prior reads remain under shared read-only workflow
knowledge; the override routes curation and citation writes, not those reads.
Do not repurpose `state_dir`: it controls resume behavior and
suppresses ordinary warm start. Private scratch, HOME, caches and logs, immutable
shared inputs, effective container path mappings and deliberate GPU coordination
locks remain caller obligations. These declared path checks do not establish
mount isolation or warm-start compatibility.

## Manifest

Use the task and command protocol in [Native measurements](direct-external-command.md).
The current composer supports `geak-arena-external-task-v1` with
`glm-weight-lib-v2` and the external primary objective. Its native validator checks
task artifacts and binds candidates to task, epoch and source digests.

A manifest has this shape; replace all illustrative paths, device identities,
timeouts, deadline, model and task hash with actual owner bindings:

```json
{
  "schema": "geak.kernel-composition/2",
  "task_root": "/inputs/task",
  "task_sha256": "<exact task.json SHA256>",
  "out_dir": "/exports/comparison-001",
  "deadline_epoch": 1800000000,
  "evaluation": {
    "measurement_command": ["python3", "/workspace/GEAK/scripts/native_runtime/direct_external_command.py", "--task-root", "/inputs/task", "--state-root", "/exports/common-native", "--gpu-lock-file", "/locks/common.lock", "--epoch", "comparison-001", "measure"],
    "result_command": ["python3", "/workspace/GEAK/scripts/native_runtime/direct_external_command.py", "--task-root", "/inputs/task", "--state-root", "/exports/common-native", "--gpu-lock-file", "/locks/common.lock", "--epoch", "comparison-001", "read"],
    "timeout_ms": 3600000
  },
  "lanes": [{
    "id": "reference",
    "policy": {"finalist": "required", "imports": "none"},
    "args_file": "/operator/reference-args.json",
    "runtime_command": ["docker", "exec", "owned-reference", "node", "/workspace/GEAK/interface/runtime/engine/run_workflow.mjs"],
    "runtime_options": {"agent": "codex", "model": "<model>", "concurrency": 4, "agent_timeout_ms": 3600000, "backend_events_file": "/exports/comparison-001/reference/backend_calls.jsonl"},
    "timeout_ms": 32400000,
    "resources": {"context_id": "owned-reference", "device_ids": ["<physical device identity>"], "gpu_ids": ["0"]}
  }]
}
```

Add members with `finalist: "eligible"`, separate private paths and distinct
physical grants. The sole required member can be an independent greedy search or
an evolutionary search; the code does not branch on either its name or optimizer.

Complete argument files must explicitly bind `exp_root`, `eval_dir`, `learned_dir`,
`kb_artifacts_dir`, `kb_store_dir`, `kb_remote=off`, `gpu_ids`,
`deadline_epoch`, positive `search_reserve_s`, the pinned objective and native
commands. Use `mode=optimize` (or omit it), `apply_to_original=false` and an existing
`greedy` or `evo` optimizer. Existing per-workflow reserves remain unchanged:
equality of the final absolute deadline alone does not make search exposure equal.
Native argv uses separate option/value entries and explicit `--task-root`,
`--state-root`, `--gpu-lock-file` and `--epoch`, ending in `measure` or `read`.

Runtime options are `agent`, `profile`, `model`, `registry`, `concurrency`,
`agent_timeout_ms` and `backend_events_file`. The last forwards the existing
[backend-call recorder](../../interface/runtime/engine/README.md#optional-backend-call-intervals)
unchanged; its file must be fresh and its parent directory must exist when the
runtime starts. Optional member and evaluation `cwd`/`env` use the existing process
launcher. Keep credentials in private configuration.

Run `node /workspace/GEAK/interface/runtime/engine/compose_kernel.mjs /operator/composition.json`.
Output directories must be fresh and their parents must exist. Without
`output_dir`, a member writes to `out_dir/<id>`. The composer copies argument bytes
before launch. Runtime launch commands must retain ownership through full child
settlement; a detached remote client is insufficient.

## Selection and evidence

Every member keeps its original result, metrics, output logs, argument snapshot
and an exclusively written finalist patch. The existing native reader checks the
patch and source-bound child evidence. A Director veto or invalid result remains
ineligible. Common measurement starts only after all workflows have settled;
each eligible finalist uses a fresh ORIGINAL-plus-patch workspace and the same
capture/measure/read command prefix. The owner reserves that common service and
allows its full measurement time separately from the search deadlines.

The largest common `primary_speedup` among finalists with a native reproducible
gain and no reproducible regression wins. Ties retain the required member. This
preserves the actual included stock finalist when it qualifies under that same
measurement; it is not a guarantee against a separately sampled stock run, timing
noise, or unvalidated archive entries. Valid no-gain evidence remains explicit.

`result.json` uses `geak.kernel-composition-result/2` and includes
`required_finalist` (with its ID), `comparison_complete`, `selection_basis`, winner
identities and all child records. The required member's valid child gain is
retained as fallback if its common replay fails and no common-qualified winner
exists: `comparison_incomplete_required_retained`,
`retained_required_fallback`, `primary_speedup: null` and
`comparison_unavailable: true`. An earlier child score is never substituted for
a common score. A common-qualified winner takes precedence over that fallback.
If the required child or its common replay is invalid, another member may win,
but the required qualification stays false, the comparison stays incomplete,
and the CLI exits nonzero. Without a
qualified gain or eligible fallback, the original is retained and `winner` is null.

Child records include declared policy/resources/deadline, process start and
settlement timestamps, remaining allowance at settlement and the workflow's
returned stop reason, rounds and budget counts where available. Settlement occurs
after postlude and cleanup; it is not the search-stop timestamp or GPU utilization.
The metrics and optional recorder retain their existing call accounting. Exact
search-stop time, evaluation counts and reserved-service delivery require their
own runtime/evaluator evidence; missing values are not inferred from wall time.

## Migration and remaining import seam

Schema /1 is rejected explicitly. To migrate, add the common deadline, private
paths and local device mappings, and replace the formerly reserved name with an
explicit `required` policy. Result consumers must replace `protected_geak` with
`required_finalist` and the two GEAK-specific fallback labels with the generic
labels above. There is no implicit name-based compatibility branch. Existing
workflow behavior and frozen sources do not change.

Selected immutable publication/import is still missing. The small integration
seam is the existing canonical-selection capture in `01b_args.js` for stock and
retained patch records in `14a_evolution.js` for EVO, followed by the existing
`createEvolution` parent/donor selection and workspace preparation. A publication
must bind lineage, original/base source, immutable content hash and evidence;
only explicitly selected compatible sources may enter another private archive.
Run-local candidate IDs and mutable whole-archive snapshots cannot serve as that
interface. Warm-start bases require equality or normalization to the common
ORIGINAL before import. No new shared mutable archive or scheduler is supplied.

The CPU composer tests use real child processes and existing native source/report
helpers with synthetic timing evidence. Adapter cases use prepared workflow
results; joined one-member cases execute the actual dispatcher and lane through
`createRuntime`, comparing complete raw direct/composed child records through the
postlude. The three joined endings are source improvement, bounded stop refusal
followed by self-stop, and search cutoff followed by a Director veto. They use an
empty local warm-start store, enabled learned reads/writeback, scripted backend
and file effects, and a child-only logical clock with real outer watchdogs.
They establish parity only for those declared mocked cases. They do not qualify
the provider CLI, nonempty warm-start/citation behavior, concurrent directions,
in-flight interruption, native hardware, actual reserved service, peer import use
or a performance benefit. The composer's extra common evaluation is checked
separately from the child's unchanged result.
