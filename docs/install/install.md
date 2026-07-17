---
myst:
    html_meta:
        "description": "Install GEAK 4.0.0: pip install git+ downloads the repo, installs a recent Claude Code, and the Python deps (plus a serving backend for E2E, ROCm required)."
        "keywords": "GEAK, install, ROCm, Claude Code, Workflow, sglang, vLLM, AMD Instinct, setup"
---

# Install GEAK

GEAK 4.0.0 is not a Python package. It is a set of Workflows (`e2e_workflow.js` / `kernel_workflow.js`)
that run inside Claude Code. "Installing" means: get the repo, get a recent Claude Code, and have a
working ROCm environment (plus a serving backend for E2E). For a first run, see
[Run a workflow](../how-to/run-agent.md).

## Prerequisites

GEAK 4.0.0 requires the following software and hardware.

| Requirement | Detail |
|---|---|
| **AMD Instinct™ MI GPU** | CDNA, gfx942 (MI300X) / gfx950 (MI350X/MI355X). Auto-detected. |
| **ROCm 6+** | `rocminfo` / `rocm-smi` must work. |
| **A profiler** | One of `rocprof-compute`, `rocprofv3`, `rocprof` (also `omniperf` or `metrix`). Auto-detected. |
| **Python 3.8+** | Tested on 3.12. |
| **Claude Code ≥ 2.1.177** | Required for the dynamic Workflow feature. Check `claude --version`. |
| **Anthropic API key** | Set as `ANTHROPIC_API_KEY`. Get one at [console.anthropic.com](https://console.anthropic.com). |
| **Serving backend (E2E)** | A running-capable `sglang` or `vllm`, plus model weights on disk. |

## Set up GEAK

Clone the repository and run the setup script.

Installing GEAK installs the `geak` Python package + deps, clones the GEAK repo, and installs the Claude Code CLI.
By default the repo lands in `./GEAK` under the directory you run the command from (override with `GEAK_HOME`).
Pick either method — both end up the same:

**A. One-liner** — run it in the directory where you want GEAK to live:

```bash
pip install "git+https://github.com/AMD-AGI/GEAK"
```

**B. Clone first** — if you'd rather have the checkout up front (e.g. to work on a branch):

```bash
git clone https://github.com/AMD-AGI/GEAK.git
cd GEAK
pip install .
```

It leaves PATH and API access configuration to you. Follow its printed next-steps to add `~/.local/bin` to PATH, then set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=<your-key>
```

Get a key from [console.anthropic.com](https://console.anthropic.com) if you don't have one. Add the export to your shell profile (`~/.bashrc` or `~/.profile`) to avoid setting it each session.

Launch GEAK:

```bash
IS_SANDBOX=1 claude --dangerously-skip-permissions
```

Nothing is compiled at clone time — the workflow `.js` files and their `roles/`, `knowledge/`, `scripts/`
are used directly. Sandbox mode auto-approves the permissions the workflows need.

## Verify the environment

Run these checks before starting a workflow. A misconfigured environment fails deep into a multi-hour run.

```bash
# Claude Code version (must be ≥ 2.1.177)
claude --version

# GPU is visible to ROCm
rocminfo | grep -E "Name:|gfx"

# At least one profiler is on PATH
command -v rocprof-compute || command -v rocprofv3 || command -v rocprof
```

Expected output:

- `claude --version` prints `2.1.177` or higher.
- `rocminfo` lists your GPU name and a `gfx942` or `gfx950` target.
- At least one profiler command resolves without error.

If `rocminfo` fails, your ROCm stack is not installed or not on PATH. If no profiler resolves, install `rocprof-compute` (preferred) or `rocprofv3`.

## Related topics

- [Run a workflow](../how-to/run-agent.md) — start a single-kernel or end-to-end run.
- [Compatibility matrix](../compatibility.md) — verified GPUs, ROCm versions, backends, and dtypes.
