# Installation

GEAK v4 is not a Python package. It is a set of **Workflows** (`e2e_workflow.js` / `kernel_workflow.js`)
that run **inside Claude Code**. "Installing" means: get the repo, get a recent Claude Code, and have a
working ROCm environment (plus a serving backend for E2E). For a first run, see [Quick start](quick_start.md).

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
claude update                          # Claude Code >= 2.1.177
git clone https://github.com/AMD-AGI/GEAK.git && cd GEAK
IS_SANDBOX=1 claude --dangerously-skip-permissions
```

Nothing is compiled at clone time — the workflow `.js` files and their `roles/`, `knowledge/`, `scripts/`
are used directly. Sandbox mode auto-approves the permissions the workflows need.
