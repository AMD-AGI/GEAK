# Set up the GEAK runtime

GEAK's standalone Node runtime in `interface/runtime/engine/` runs the JavaScript
workflows and dispatches each agent call to the configured Claude or Codex CLI.
The runtime uses Node built-ins and has no npm dependencies of its own.

The standard external command is unchanged:

```bash
python3 interface/run_e2e.py handoff.json result.json
```

This two-JSON entrypoint always uses the standalone runtime. Claude is the default
agent unless an explicit profile/backend or unambiguous provider configuration
selects another agent. An omitted `optimizer` selects `evo`; a handoff containing
`"optimizer": "greedy"` selects the greedy policy in the same kernel worker.
See [the external contract](../run_e2e.md) for handoff and result fields.

For EVO, `evolution_return_development` defaults to `true`: after normal
engineering and evaluation, one concrete unmeasured development attempt can be
retained when there is no measured development source to return. This is source
retention, with correctness and promotion still decided by verification.
Explicit `false` disables the fallback. The E2E handoff also forwards
`evolution_import_stall_sources`, `record_planner_inputs`, and
`case_measurement_view`; the nested worker validates their task requirements.

## Select the release checkout

For an installation managed by Hyperloom, select the release through its existing
GEAK checkout configuration:

```bash
export GEAK_REF=feature/evo-reviewed-release-20261110
```

Run the existing installer/update path with that setting. For a manually managed
checkout, check out the same branch before installing it. The Python package is
still named `geak`, and the entrypoint remains `interface/run_e2e.py`; no separate
EVO package or new command-line flags are required.

Keep the full checkout and point `GEAK_ROOT` and `GEAK_E2E_RUNNER` at the same
build. The runner resolves its workflow files relative to its own location, so a
stale runner path still uses the old checkout even if `GEAK_ROOT` changes.
GEAK's package bootstrap leaves a nonempty `GEAK_HOME` untouched; setting
`GEAK_REF` alone does not switch such a directory.

All commands below assume the selected repository root is the working directory.

## Install prerequisites

Install GEAK's Python dependencies from the selected checkout with the existing
`pip install .` path. No EVO extra is needed. The Python distribution does not
replace the full checkout or install every agent CLI and workload dependency.
If the checkout and agent CLI are already provisioned, `GEAK_SKIP_BOOTSTRAP=1`
disables GEAK's best-effort clone/Claude installation hook during installation.

The runtime requires Node 18 or later. For the Codex setup below, use Node 20 or
later and install the CLI separately:

```bash
node --version
npm install -g @openai/codex@0.146.1
codex --version
```

For Claude, install the Claude Code CLI and make `claude` available on `PATH`, or
set `GEAK_CLAUDE_BIN` to its executable. Configure authentication for the selected
CLI and endpoint. GPU, serving-backend, model, profiler, and benchmark-client
requirements remain those of the workload; see the
[installation guide](../../docs/install/install.md).

## Select the agent and model

For the Python entrypoint, use environment variables. It does not accept Node's
`--agent`, `--profile`, or `--model` flags.

| Environment setting | Effect |
| --- | --- |
| `GEAK_AGENT_PROFILE` | Select a registry profile containing an agent and optional model. |
| `GEAK_AGENT_BACKEND` | Select an agent explicitly; overrides a profile's agent. |
| `GEAK_MODEL` | Select a registry model; overrides a profile's model. |
| `GEAK_AGENT_AUTO=0` | Disable credential-based agent selection. Explicit selections still apply. |

When no profile or backend is selected, the shipped registry uses this rule:

| Credential environment | Agent |
| --- | --- |
| `GEAK_AMDKEY` or `OPENAI_API_KEY`, with no Anthropic-side configuration | Codex |
| Anthropic-side configuration only | Claude |
| Both sides configured | Claude |
| Neither side configured | Claude |

Anthropic-side configuration includes `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`,
`ANTHROPIC_AUTH_TOKEN`, and `CLAUDE_CODE_OAUTH_TOKEN`. Selecting a CLI does not
establish that its authentication or endpoint is usable.

`GEAK_MODEL` names an entry in `engine/registry.json`. To specify a raw model ID,
use `GEAK_CLAUDE_MODEL` or `GEAK_CODEX_MODEL` with the corresponding agent. A model
pinned by a registry selection takes precedence over the backend-specific model
variable. `GEAK_REGISTRY` can select a custom registry.

For example, to keep an E2E run on Claude even when another provider key is present:

```bash
export GEAK_AGENT_BACKEND=claude
python3 interface/run_e2e.py handoff.json result.json
```

## Preserve Claude configuration

The standalone Claude backend honors the existing GEAK environment knobs:

| Variable | Effect |
| --- | --- |
| `GEAK_CLAUDE_BIN` | Claude executable. |
| `GEAK_CLAUDE_MODEL` | Claude model ID. |
| `GEAK_CLAUDE_SETTINGS` | Settings JSON or settings-file path passed as one `--settings` argument. |
| `GEAK_CLAUDE_EFFORT` | Forward `low`, `medium`, `high`, `xhigh`, or `max` through `--effort`. |
| `GEAK_CLAUDE_CONFIG_DIR` | Seeded Claude configuration directory; when nonempty, overrides the child's `CLAUDE_CONFIG_DIR`. Otherwise the inherited `CLAUDE_CONFIG_DIR` or CLI default applies. |

The legacy `ultracode` effort value is not passed as a CLI effort. Workflow
orchestration is provided by GEAK's Node runtime. Existing provider environment
variables remain available to the child CLI; use values supported by that CLI
build and endpoint.

## Configure the Codex provider

The shipped registry can configure Codex with command-line provider overrides.
With auto-configuration enabled, provider resolution uses:

1. An explicit `OPENAI_BASE_URL`, or the selected registry model's `base_url`.
2. Otherwise, the first configured provider key in registry order:
   `GEAK_AMDKEY` for the AMD gateway, then `OPENAI_API_KEY` for OpenAI.

For example, configure one provider in the launching environment:

```bash
export OPENAI_API_KEY="<provider key>"
# Or, for the AMD gateway configured in the registry:
# export GEAK_AMDKEY="<subscription key>"
```

The AMD entry adds the `Ocp-Apim-Subscription-Key` header using `GEAK_AMDKEY`.
`OPENAI_CUSTOM_HEADERS` can supply a JSON map from header names to environment
variable names. Endpoint, authentication, model availability, and certificate
trust must match the deployment; the registry is configuration, not a service
availability guarantee.

The `GEAK_` prefix lets `GEAK_AMDKEY` use Hyperloom's GEAK environment namespace.
Keep the established `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and
`OPENAI_CUSTOM_HEADERS` names when using that provider configuration. Ensure the
chosen variables reach the process that launches GEAK.

| Variable | Default or effect |
| --- | --- |
| `GEAK_CODEX_BIN` | Codex executable; defaults to `codex`. |
| `GEAK_CODEX_MODEL` | Raw model ID. If unset, an automatically selected provider supplies its registry default. |
| `GEAK_CODEX_EFFORT` | Defaults to `xhigh`; accepts `none`, `low`, `medium`, `high`, `xhigh`, and `max` as an alias for `xhigh`. |
| `GEAK_CODEX_AUTOCONFIG` | Defaults to `1`; `0` leaves provider selection to Codex's own configuration. |
| `GEAK_CODEX_EXTRA_ARGS` | Whitespace-separated CLI arguments appended after generated overrides. Shell quoting is not parsed. |
| `OPENAI_BASE_URL` | Explicit endpoint, taking precedence over provider-key lookup. |

A `model_provider=` override in `GEAK_CODEX_EXTRA_ARGS` suppresses generated
provider configuration. A `model_reasoning_effort=` override there takes
precedence over `GEAK_CODEX_EFFORT`. Use a custom registry when arguments need
embedded whitespace. Choose a model ID supported by the selected endpoint's
Responses API.

## Run a standalone kernel workflow

The Node CLI retains its existing flags. They select the runtime backend and are
separate from the workflow arguments in the JSON file:

```bash
node interface/runtime/engine/run_workflow.mjs kernel_workflow/kernel_workflow.js \
  --agent codex --args-file /absolute/path/run-args.json \
  --result-file /absolute/path/result.json
```

Node's `--agent`, `--profile`, `--model`, and `--registry` options override their
corresponding environment settings. For E2E integration, keep using
`run_e2e.py handoff.json result.json` with environment configuration. See the
[EVO run guide](../../docs/evo/stock-workflow.md) for kernel arguments, budgets,
and archive behavior.

## Troubleshoot and verify

| Symptom | Check |
| --- | --- |
| Agent executable cannot be found | `PATH` and the selected `GEAK_CLAUDE_BIN` or `GEAK_CODEX_BIN`. |
| Authentication error | Provider key, header mapping, and selected endpoint. |
| Model rejected | Raw model ID, registry model selection, and endpoint support. |
| TLS error | Certificate trust required by the deployment. |
| Unexpected agent | Explicit profile/backend settings and both sides of the credential environment. |
| Old workflow behavior after an update | Actual checkout revision and the invoked `GEAK_E2E_RUNNER` path. |

The runtime self-test does not need an agent CLI, credentials, network, or GPU:

```bash
node interface/runtime/engine/selftest.mjs
```

The conformance tool checks backend behavior and workflow compatibility:

```bash
node interface/runtime/engine/conformance.mjs --profile codex
```

Use `--fake` to check the conformance harness without an agent CLI, or
`--audit-only` for its static workflow-contract audit. A live probe requires the
selected CLI and provider configuration. The backend requirements are:

| Requirement | Purpose |
| --- | --- |
| R1: Structured output | Return results that GEAK can extract and validate against each agent schema. |
| R2: Headless invocation | Complete a one-shot agent call and return its answer on stdout. |
| R3: Execution permissions | Permit the source edits and build/profiling commands required by the workload. |
| R4: Command timeouts | Support the build and benchmark durations required by the workload. |
| R5: Context capacity | Accommodate role instructions, knowledge, source, and task context. |
| R6: Provider configuration | Accept the selected endpoint and authentication. |
| R7: Working directory and paths | Honor the requested working directory and absolute task paths. |

A failed static audit indicates a mismatch between workflow requirements and
runtime capabilities. Resolve that mismatch before updating the audit baseline.
See [the engine guide](engine/README.md) for the runtime's component files.
