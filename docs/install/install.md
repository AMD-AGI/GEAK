---
myst:
    html_meta:
        "description": "Install the GEAK-evo release checkout, Python dependencies and agent CLI, then run workflows with the bundled Node runtime."
        "keywords": "GEAK-evo, install, ROCm, Node, Claude Code, sglang, vLLM, AMD Instinct, setup"
---

# Install GEAK-evo

GEAK-evo runs the repository's JavaScript workflows through the bundled Node runtime.
The Python package installs dependencies and bootstraps Claude Code; keep the full
checkout because it also supplies the workflows, roles, scripts and knowledge files.
The package name `geak`, repository paths and `GEAK_*` environment variables remain
unchanged for compatibility.

## Prerequisites

| Requirement | Detail |
|---|---|
| **AMD Instinct™ MI GPU** | CDNA, gfx942 (MI300X) / gfx950 (MI350X/MI355X). See the [compatibility matrix](../compatibility.md) for other targets. |
| **ROCm 6+** | `rocminfo` / `rocm-smi` must work. |
| **A profiler** | One of `rocprof-compute`, `rocprofv3`, `rocprof` (also `omniperf` or `metrix`). |
| **Python 3.8+** | Package minimum; use a Python version supported by your serving stack. |
| **Node.js 18+** | Required for the bundled runtime. Use Node 20+ for the Codex setup. The runtime itself has no npm dependencies. |
| **Agent CLI and credentials** | Claude Code ≥ 2.1.177 is the bootstrap default. Configure your Claude login/API access, or select another supported backend using the [runtime setup guide](../../interface/runtime/SETUP.md). |
| **Serving backend (E2E)** | A working `sglang` or `vllm` installation, plus model weights on disk. |

## Install the release checkout

Clone the release branch, then install from that checkout:

```bash
git clone --single-branch --branch feature/evo-reviewed-release-20261110 \
  https://github.com/AMD-AGI/GEAK.git
cd GEAK
GEAK_HOME="$PWD" pip install .
```

`GEAK_HOME="$PWD"` points the bootstrap at this checkout so it preserves the selected
branch. Keep using this directory for launches. The install attempts to install or
update Claude Code and prints any remaining setup steps; Node.js and the serving
stack must already be available separately.

Follow the printed PATH instructions for `~/.local/bin`, and configure your agent
credentials. For Claude API access, set `ANTHROPIC_API_KEY` in your environment;
existing Claude login or gateway settings can also be used. See
[runtime setup](../../interface/runtime/SETUP.md) for backend and model selection.

## Verify the environment

```bash
python --version
node --version
claude --version                 # or the CLI selected in your runtime configuration
git branch --show-current
rocminfo | grep -E "Name:|gfx"
command -v rocprof-compute || command -v rocprofv3 || command -v rocprof
```

Check that the branch is `feature/evo-reviewed-release-20261110`, Node is at least
18, the chosen agent CLI is available, and ROCm reports the intended GPU. If
`rocminfo` fails, check the ROCm installation and PATH. If no profiler resolves,
install `rocprof-compute` or `rocprofv3`.

## Start a run

Use [Run a workflow](../how-to/run-agent.md) for a kernel or serving example.
Workflows run through `interface/runtime/engine/run_workflow.mjs`, including when
Claude is the agent backend. Kernel delivery requires the runtime's `runProcess`
primitive, which Claude Code's native `Workflow` tool does not provide.

Existing integrations keep the same command:

```bash
python interface/run_e2e.py /path/to/handoff.json /path/to/result.json
```

The wrapper selects the bundled runtime. Omitting `optimizer` now selects `evo`;
set `"optimizer": "greedy"` in the handoff to use the original search policy.
The [handoff and result contract](../../interface/run_e2e.md) is unchanged.
