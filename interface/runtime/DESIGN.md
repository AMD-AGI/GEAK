# Standalone agent runtime design

GEAK's JavaScript Workflow is the source of truth. The standalone runtime in this directory re-creates the Workflow execution primitives and changes only how each `agent()` call reaches a coding-agent CLI.

## Preserved contract

The runtime supplies:

- `agent()`
- `parallel()`
- `pipeline()`
- `workflow()` with one nested level
- `phase()` and `log()`
- `args` and `budget`
- a file-scoped Node `require()` for current GEAK workflow code

The workflow script, roles, prompts, schemas, retry behavior, artifacts and evaluator remain unchanged. A provider response is accepted only after the existing schema validator succeeds.

## Provider selection

`registry.json` separates:

- agent recipes: executable, flags, prompt delivery and environment names;
- model endpoints;
- profiles that select an agent and optional model.

Selection order is command line, environment, then `default_profile`.

### Codex

Codex uses its native `workspace-write` sandbox. The runtime can configure the official OpenAI Responses endpoint from `OPENAI_API_KEY`, or an operator-supplied OpenAI-compatible endpoint from `OPENAI_BASE_URL`. `GEAK_CODEX_MODEL` supplies the model ID and `GEAK_CODEX_EXTRA_ARGS` is the final explicit override.

### Hermes

Hermes uses `--safe-mode --toolsets terminal,file,web`. Because Hermes does not supply a native filesystem sandbox, GEAK requires both:

1. `GEAK_HERMES_EXTERNAL_SANDBOX=1`;
2. a concrete container-runtime marker (`/.dockerenv`, `/run/.containerenv`, or a recognized PID-1 cgroup).

The command refuses to start if either condition is absent. The selected Hermes profile owns provider authentication and should have an empty fallback chain. GEAK retains correctness and promotion authority.

## Current-main compatibility

Current GEAK uses a scoped `require('child_process')` for bounded cleanup. `run_workflow.mjs` creates a Node `require` relative to the workflow file with `createRequire()`, then injects it beside the other Workflow primitives.

Claude-specific `StructuredOutput`, `WebSearch` and `WebFetch` names are neutralized for non-Claude backends. Hermes receives corresponding structured-output instructions and the `web` toolset.

## Verification

Run the CPU-only gates after changing the runtime or registry:

```bash
node interface/runtime/selftest.mjs
node interface/runtime/conformance.mjs --audit-only
python -m pytest -q \
  interface/test_run_e2e_recovery.py \
  e2e_workflow/scripts/tests/test_workload_alignment.py
python interface/run_e2e.py ci/fixtures/handoff.dry.json /tmp/geak-dry.json --dry-run
```

The self-test exercises primitives, nesting, schema retries, scoped `require`, provider invocation assembly and Hermes container guards. The conformance audit scans current workflow/role sources for unimplemented primitive, tool, wording or parity drift.

A successful import or dry-run does not prove GPU execution. A physical qualification must preserve exact source/input hashes, execute the original Workflow entrypoint, run a real kernel on the native architecture, validate correctness, preserve the Director verdict and clean up the process/device lease.
