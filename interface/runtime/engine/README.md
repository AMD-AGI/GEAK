# engine/ — the standalone workflow runtime

A dependency-free Node re-implementation of the orchestration primitives GEAK's `.js` workflows are
written against (`agent` / `parallel` / `pipeline` / `phase` / `workflow` / `log` / `args` /
`budget`). Claude Code provides those natively; this directory provides them when the workflow runs
on some other CLI. Each `agent()` call becomes one short-lived backend process, so the CLI underneath
never has to support parallel or nested subagents itself.

**It is not codex-specific**, which is why it is not called `codex-engine/`. The scheduler, the
schema layer and the backend contract mention no CLI at all; everything backend-specific is *data* in
`registry.json`. Which CLI actually runs is a resolution result, not a property of this code.

| File | Role |
|---|---|
| `run_workflow.mjs` | the primitives, concurrency semaphore, nesting, script loader, CLI entry, metrics |
| `schema.mjs` | structured-output contract + extraction + validation (incl. enum) |
| `config.mjs` | registry loading, `(agent, model, profile)` resolution, invocation build, env neutralization |
| `registry.json` | the data: agents × models × profiles |
| `backends/base.mjs` | backend contract + `runProcess` (one command in its own process group) + `spawnAgent` + `defaultConcurrency` |
| `backends/generic.mjs` | config-driven backend, good for any CLI describable in `registry.json` |
| `selftest.mjs` | unit tests of the primitives — no GPU, no network, no key |
| `conformance.mjs` | backend capability probes + static contract-drift audit |
| `experiment.mjs` | `(agent × model)` comparison runner |

Two orthogonal axes live in `registry.json`: **agents** (how to drive a CLI) and **models** (which
endpoint). A **profile** pins one `(agent, model)` pair. Setting a provider key is by itself enough
to select an agent — see `SETUP.md` for that rule and for the R1–R7 requirements a candidate CLI has
to satisfy.

Two agents ship: **`claude`** and **`codex`**. Adding a third is a data change, not a code change —
write an `agents` entry with the same fields, then prove it with `conformance.mjs --agent <name>`.
Only agents that have passed that gate belong in the registry; an untested entry reads to the next
person as a supported backend.

```bash
node interface/runtime/engine/selftest.mjs                        # expect 125/125
node interface/runtime/engine/run_workflow.mjs <workflow.js> --agent codex --args '{...}'
```

Setup, knobs and troubleshooting: [`../SETUP.md`](../SETUP.md).

## Optional backend call intervals

Pass `--backend-events-file /absolute/path/backend_calls.jsonl` to record concurrency
for either GEAK or GEAK-EVO. The recorder is off by default. Use a fresh file in an
existing directory, for example beside `--metrics-file` or under `reports/trace/`.
An existing file is never overwritten or appended to by a new runtime.

Each JSONL row has schema `geak.backend-call/1`, an `event` (`start` or `end`),
`backend_call_id`, `agent_call_id`, `schema_attempt`, `backend`, `phase`, `label`,
`timestamp_utc` and `elapsed_ms`. End rows also have `outcome` (`returned` or `threw`)
and optional `error_name`/`error_code` values. Prompts, responses, error messages,
environment values and candidate source are not recorded. Phase and label are the
workflow's existing role attribution, without inspecting prompt text.

One interval brackets one admitted `backend.runAgent()` invocation through promise
settlement. It starts after semaphore wait and deadline checks and ends before
schema parsing. Schema retries have separate backend IDs and the same agent ID;
an outer workflow retry gets a new agent ID. Invalid JSON returned by a backend
therefore has outcome `returned`. Errors are recorded before deadline remapping.
Use overlapping half-open `[start,end)` intervals to count active backend calls.
These are host call boundaries, not GPU activity, OS process lifetime, or the
lifetimes of tools/agents launched internally by a backend. Calls made by nested
workflows through this runtime share the same trace and ID sequence.

`elapsed_ms` uses the runtime's monotonic clock; UTC is derived from its initial
epoch anchor. Tiny synchronous file appends add recording overhead but need no
background writer or shutdown timer. `WORKFLOW_METRICS.backend_call_trace` reports
the path, written event count, open call count, write error and
`complete_at_workflow_return`. Recorder failures leave backend outcomes unchanged.
A crash, hard kill or unresolved call can leave an unmatched start; its end and
duration remain unknown. A call that settles after the workflow returns can still
append its end, while the metrics retain their state at workflow return.

### Optional planner component observation

For prospective audits, set the lane argument `record_planner_inputs:true` and
configure the existing runtime `--backend-events-file` sink. The lane flag defaults
to off. With the flag off no planner observation metadata is attached; without
backend tracing no component summaries are emitted.

At each initial plan or forced replan, the lane synchronously freezes component
`JSON.stringify(value)` strings before `agentT` and later history mutation. The
runtime hashes these in memory and adds `planner_observation` to the existing
backend start/end rows. No raw input/prompt/schema artifact, host writer or extra
subprocess is created. The backend options allowlist excludes this metadata.

Each observation carries the already-known `eval_dir`, round and logical attempt
(`initial` or `replanN`); the enclosing row supplies the role label, agent/backend
IDs and schema attempt. Eval scope separates nested lanes with identical labels.
Scope values must be a nonempty string eval directory, positive safe-integer round
and `initial`/`replanN` attempt; invalid scope emits only an observation error state.
Validate logical-key uniqueness when reading; ambiguous scopes or collisions are
UNKNOWN, never a guessed join. Retries legitimately map one logical call to many
backend dispatches. Calls rejected before backend admission have no backend row.

Components cover the input object, each supplied field plus selected absent
optional fields, injected TASK, role prompt, response schema and prepared reference.
Each summary records state, UTF-8 byte count and SHA-256 of its compact JSON
representation, including JSON quoting for strings. It is not canonical JSON or a
normalized digest. `absent` means no serialized value; `null` means JSON null;
`empty` means an empty string, array or object; other values are `present`.
Serialization/hash failures are `error`, with null byte count/hash, as for absent
values. Zero, false and whitespace strings are present. Observation errors do not
infer failure of a preparation role or missing model evidence.

When backend tracing is enabled, `host_prompt_sha256` and `host_prompt_bytes` also
bind the exact UTF-8 `backend.runAgent` options.prompt string per dispatch, after
schema/backend normalization and runtime timing additions. No argv, environment,
provider configuration or response contents are hashed or recorded by this observer.
`host_prompt_state` is `present` on success; a digest failure records `error` with
null hash/byte count and does not prevent the backend call.
Equal component hashes establish equal supplied representations; equal host-prompt
hashes establish that host string's bytes. Neither establishes referenced file
bytes, actual reads, complete CLI/provider context or equal model conditioning.

Use the existing trace path outside EVAL_DIR where practical. Its ordinary 0600
file and a different pathname do not prove unreadability to same-UID role shells;
the runtime adds no isolation boundary. Serialization, hashing and trace writes
consume ordinary wall time. Fixed-clock parity checks establish unchanged prompts
and calls, not zero overhead. This is prospective only; active packets are unchanged.
This trace-only observer supersedes the a45 raw-snapshot implementation of
`record_planner_inputs`; do not combine the two implementations.
