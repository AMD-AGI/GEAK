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
| `backends/base.mjs` | backend contract + `spawnAgent` + `defaultConcurrency` |
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
node interface/runtime/engine/selftest.mjs                        # expect 105/105
node interface/runtime/engine/run_workflow.mjs <workflow.js> --agent codex --args '{...}'
```

Setup, knobs and troubleshooting: [`../SETUP.md`](../SETUP.md).
