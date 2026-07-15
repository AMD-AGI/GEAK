---
myst:
    html_meta:
        "description": "Install GEAK v4: pip install git+ downloads the repo, installs a recent Claude Code, and the Python deps (plus a serving backend for E2E, ROCm required)."
        "keywords": "GEAK, install, ROCm, Claude Code, Workflow, sglang, vLLM, AMD Instinct, setup"
---

# Install GEAK

GEAK v4 is not a library you `import`. It is a set of **Workflows** (`e2e_workflow.js` / `kernel_workflow.js`)
that run **inside Claude Code**. The `pip install git+...` below is a bootstrapper: it downloads the repo, installs
a recent Claude Code, and installs the Python deps — leaving you a working checkout plus a ROCm environment
(plus a serving backend for E2E). For a first run, see
[Run a workflow](../how-to/run-agent.md).

## 1. Prerequisites

| Requirement | Detail |
|---|---|
| **AMD Instinct MI GPU** | CDNA, gfx942 (MI300X/MI308X) / gfx950 (MI350X/MI355X). Auto-detected. |
| **ROCm 6+** | `rocminfo` / `rocm-smi` must work. |
| **A profiler** | One of `rocprof-compute`, `rocprofv3`, `rocprof` (also `omniperf` / `metrix`). Auto-detected. |
| **Python 3.8+** | Tested on 3.12. |
| **Claude Code ≥ 2.1.177** | Required for the dynamic Workflow feature. Check `claude --version`. |
| **Serving backend (E2E)** | A running-capable `sglang` or `vllm`, plus model weights on disk. |

## 2. Set up

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

Useful env knobs: `GEAK_HOME` (clone target), `GEAK_REF` (branch/tag), `CLAUDE_VERSION`, and
`GEAK_SKIP_BOOTSTRAP=1` to install the Python package without the clone / Claude Code steps (CI, image builds).

It leaves **PATH and API access** setting in Claude Code to you — follow its printed next-steps to add `~/.local/bin` to
PATH and to configure Anthropic API access. 

### Then launch

```bash
IS_SANDBOX=1 claude --dangerously-skip-permissions
```

Nothing is compiled at clone time — the workflow `.js` files and their `roles/`, `knowledge/`, `scripts/`
are used directly. Sandbox mode auto-approves the permissions the workflows need.

## Related topics

- [Run a workflow](../how-to/run-agent.md) — start a single-kernel or end-to-end run.
- [Compatibility matrix](../compatibility.md) — verified GPUs, ROCm versions, backends, and dtypes.
