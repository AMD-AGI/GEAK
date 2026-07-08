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

## 3. Verify the environment

There is no separate installer — the workflow's Setup **preflight** verifies the environment on the first
run and writes `env_report.{md,json}`. To sanity-check yourself:

```bash
rocminfo | grep -i gfx          # GPU arch (gfx942 / gfx950)
rocm-smi                        # GPUs visible
which rocprofv3 rocprof         # at least one profiler
python3 --version               # >= 3.8
```

For E2E, also confirm the backend and weights:

```bash
python3 -c "import sglang; print(sglang.__version__)"   # or: import vllm
ls /path/to/model
```

Optional op-backend probes (E2E head kernels):

```bash
python3 -c "from aiter.ops.flydsl import is_flydsl_available; print('flydsl', is_flydsl_available())"
```

## 4. ROCm library reference

Kernel work draws on ROCm libraries pre-installed under `/opt/rocm*/`
(`ls /opt/rocm-*/include/` shows the version). Source, if needed:

```bash
git clone --depth 1 https://github.com/ROCm/rocm-libraries.git ~/.cache/rocm-libraries
# libraries at ~/.cache/rocm-libraries/projects/<library>/
```

Most relevant: `composablekernel`, `rocwmma`, `rocprim`, `hipcub`, `rocblas`, `hipblaslt`. The full
AMD operator × backend knowledge base ships in-tree under [`perf_knowledge/`](../perf_knowledge/).

## 5. Optional: external orchestrator (Hyperloom)

For programmatic / CI-driven E2E runs, use the stable JSON contract at
[`interface/run_e2e.py`](../interface/run_e2e.md) (`schema_version` 1):

```bash
python interface/run_e2e.py handoff.json result.json [--dry-run]
```

Env knobs: `PERFSKILLS_CLAUDE_MODEL` (`claude-opus-4-8`), `PERFSKILLS_CLAUDE_EFFORT` (`ultracode`),
`PERFSKILLS_E2E_TIMEOUT_S` (`43200` = 12h), `INFERENCEX_PATH`. See the [API reference](api_reference.md).
