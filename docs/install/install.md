---
myst:
    html_meta:
        "description": "Install GEAK v4: get the repository, a recent Claude Code, and a working ROCm environment (plus a serving backend for E2E). No pip package, no CLI."
        "keywords": "GEAK, install, ROCm, Claude Code, Workflow, sglang, vLLM, AMD Instinct, setup"
---

# Install GEAK

GEAK v4 is not a Python package. It is a set of **Workflows** (`e2e_workflow.js` / `kernel_workflow.js`)
that run **inside Claude Code**. "Installing" means: get the repo, get a recent Claude Code, and have a
working ROCm environment (plus a serving backend for E2E). For a first run, see
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

```bash
git clone https://github.com/AMD-AGI/GEAK.git && cd GEAK
bash setup.sh
```

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
