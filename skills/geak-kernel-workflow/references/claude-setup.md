# GEAK checkout and Claude setup

Use this reference during installation, upgrades and CLI/SDK diagnosis.
Commands run on the host/container that will execute GEAK, not just a login host.

## Checkout and Python

Reuse an existing checkout when requested. Inspect its remote, branch and
`git status --short` first. Fetch the requested branch (default upstream
`main`); fast-forward only a clean intended checkout. With local experiments
or edits, create a new worktree from the fetched ref rather than reset/clean it.
For a fresh installation:

```bash
git clone --branch main https://github.com/AMD-AGI/GEAK.git /work/GEAK
export GEAK_ROOT=/work/GEAK
git -C "$GEAK_ROOT" rev-parse HEAD
```

Replace the path and retain the SHA in the run record. Do not update an active
run's source tree. Keep the complete checkout, including both workflows.

Use the workload's existing Python environment when it already has the correct
ROCm/PyTorch/serving packages. Otherwise create a compatible environment first;
Python 3.11+ is a practical choice for current SDK/interface use. The package's
advertised Python 3.8 minimum is not a guarantee for every current dependency.

Set `GEAK_PYTHON` to that interpreter's absolute path. For an existing checkout,
this installs Python dependencies without the build hook cloning another repo
or changing the CLI implicitly:

```bash
export GEAK_PYTHON=/absolute/path/to/environment/bin/python
GEAK_SKIP_BOOTSTRAP=1 "$GEAK_PYTHON" -m pip install "$GEAK_ROOT"
```

Only `GEAK_SKIP_BOOTSTRAP` on this command is needed; do not leave it set and
expect a later install to bootstrap Claude. Ordinary `pip install .` instead
runs a best-effort checkout/CLI bootstrap. It may warn and still exit successfully,
and cached wheels may skip build-time setup. Always verify the actual runtime.
If pip is absent, provision pip in the chosen environment before this step.

## Claude CLI and Python SDK are separate

| Component | Used by | Verification |
| --- | --- | --- |
| Claude Code CLI | Native `Workflow`; agents under the Node host | Actual binary's `--version`, auth and Workflow capability |
| Python `claude-agent-sdk` | `interface/run_e2e.py` native Claude path | Import `ClaudeSDKClient` and `ClaudeAgentOptions` in the launch Python |
| Node.js | `interface/runtime/engine/run_workflow.mjs` | `node --version`; use Node 20+ |

Install/update the CLI if missing or incompatible. The checked GEAK bootstrap
requires at least Claude Code 2.1.177 for native Workflow; version alone is not
proof that a build exposes the tool.

```bash
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
claude --version
```

GEAK's documented fallback is `npm install -g @anthropic-ai/claude-code` when
the native installer fails. Honor a known compatible user-selected CLI version.

**GEAK's pyproject does not depend on the Python SDK.** Install it explicitly
before using the native Python interface, inside the execution container and
with the exact launch interpreter:

```bash
"$GEAK_PYTHON" -m pip install --upgrade claude-agent-sdk
"$GEAK_PYTHON" - <<'PY'
import sys
from importlib.metadata import version
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
print("python:", sys.executable)
print("claude-agent-sdk:", version("claude-agent-sdk"))
print("persistent SDK client import: OK")
PY
```

This is the Python package, not `anthropic`, the npm Claude CLI package, or a
TypeScript SDK. An SDK-only install is not required for native interactive
Workflow or Node-hosted kernel runs, but is required for the reliable e2e
Python launch documented here. Do not upgrade dependencies during an active run.

The SDK can prefer its **bundled Claude** over the one on PATH. Pin GEAK to a
verified CLI so an SDK update cannot silently select a different executable:

```bash
export GEAK_CLAUDE_BIN="$(command -v claude)"
test -x "$GEAK_CLAUDE_BIN"
"$GEAK_CLAUDE_BIN" --version
```

Record SDK version, Python path and CLI path/version. Native Workflow launches
use the same verified CLI. In environments without a global CLI, inspect and
validate the SDK's bundled CLI explicitly before using its absolute path.

## Credentials, settings and lifetime

Reuse configured credentials or a protected environment file. Never ask for a
key in chat, echo it, or put it in a handoff/command argument. Supported patterns:

- Direct Anthropic: `ANTHROPIC_API_KEY`.
- Anthropic-compatible gateway: `ANTHROPIC_BASE_URL` with the provider's
  `ANTHROPIC_AUTH_TOKEN`, API key, or `ANTHROPIC_CUSTOM_HEADERS`.
- Existing Claude login/OAuth configured for the execution user.

Check presence without printing values. Select a model the configured endpoint
actually serves through `GEAK_CLAUDE_MODEL`; do not copy an unavailable model
from a README. Optional small auth probe, when setting up an authorized run:

```bash
"$GEAK_CLAUDE_BIN" -p "Reply with exactly: SETUP OK" \
  --model "$GEAK_CLAUDE_MODEL" </dev/null
```

This checks CLI/model/auth, not the GPU, SDK lifetime or Workflow execution.

Current GEAK enables native Workflow using the settings object
`{"enableWorkflows":true,"ultracode":true}`. The interface injects it itself
unless `GEAK_CLAUDE_SETTINGS` overrides it. Do not strip those settings or
blindly pass `--effort ultracode`: public CLIs reject that effort spelling.
Use supported effort values only when explicitly configuring effort.

For interactive native Workflow, launch the verified CLI from the checkout:

```bash
cd "$GEAK_ROOT"
IS_SANDBOX=1 "$GEAK_CLAUDE_BIN" --dangerously-skip-permissions \
  --settings '{"enableWorkflows":true,"ultracode":true}'
```

These are GEAK's documented unattended execution settings; use them in the
allocated development execution environment. `IS_SANDBOX=1` is a Claude
execution flag, not OS isolation.

Native `run_e2e.py` uses persistent `ClaudeSDKClient` to keep background
workflows alive. If the SDK is unavailable, the source falls back to
`claude -p`, which can terminate a background workflow when the initial turn
ends. Install/fix the SDK before a long run; do not treat a parse failure as
permission to restart a possibly live workflow. A minimal one-shot SDK
`query()` replacement has the same lifetime problem.

For the native SDK e2e path, launch with `GEAK_AGENT_AUTO=0` and no
`GEAK_AGENT_BACKEND`/`GEAK_AGENT_PROFILE`. Any explicit backend, including
`claude`, selects the Node host. Check the dry-run selection. Preserve an
intentionally chosen alternative runtime.

Do not change `HOME` or create an empty Claude config directory to make logs
durable: this can hide working login credentials. `run_e2e.py` mirrors its
trace into the experiment; `GEAK_CLAUDE_CONFIG_DIR` is optional and should point
to an already provisioned config when needed.

## Git identity for workflow commits

GEAK commits temporary workspace baselines/winners, and some role examples
supply `git -c user.name=team -c user.email=team@workflow`. When the user has
specified an identity, pass `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`,
`GIT_COMMITTER_NAME` and `GIT_COMMITTER_EMAIL` with those values in the launch
environment. These variables take precedence over the role's config examples.
Do not invent an identity or overwrite unrelated Git configuration.

Before a commit, verify the effective identities in the committing environment:

```bash
git -C "$GEAK_ROOT" var GIT_AUTHOR_IDENT
git -C "$GEAK_ROOT" var GIT_COMMITTER_IDENT
```

Do not rewrite existing pushed history merely to change attribution.
