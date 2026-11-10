# GEAK-evo

GEAK-evo is an agentic GPU optimization system. It profiles a kernel or serving
workload, develops implementation alternatives, verifies them independently,
and delivers the tested source through the existing GEAK interface.

EVO is the default optimizer for standalone kernels and for kernel lanes started
by the serving workflow or Hyperloom. Its planner can develop retained programs
and combine ideas from their source and measurement evidence. Explicit
`optimizer: "greedy"` selects the original search policy.

## Get started

```sh
git clone --branch feature/evo-reviewed-release-20261110 https://github.com/AMD-AGI/GEAK.git
cd GEAK
GEAK_HOME="$PWD" pip install .
```

Use the [installation guide](docs/install/install.md) to configure ROCm, Node.js,
an authenticated agent backend, and the task's compiler and library dependencies.

| Task | Guide |
| --- | --- |
| Optimize a kernel or complete computation region | [Run GEAK-evo](docs/evo/stock-workflow.md) |
| Profile and optimize a serving workload | [Serving workflow](e2e_workflow/README.md) |
| Configure search and retained-candidate development | [Configuration](docs/evo/configuration.md) |
| Understand planning, verification and delivery | [Architecture](docs/evo/architecture.md) |
| Inspect candidate source and per-case evidence | [Archive](docs/evo/archive.md) |
| Use a task's native evaluator | [Native measurements](docs/evo/direct-external-command.md) |

## Use from Hyperloom

Set this in the environment used by Hyperloom's GEAK installation step:

```sh
export GEAK_REF=feature/evo-reviewed-release-20261110
```

Run the existing Hyperloom installation and workflow commands. Its GEAK installer
resolves the selected branch; the handoff invocation stays:

```sh
python "$GEAK_E2E_RUNNER" handoff.json result.json
```

No optimizer argument is needed. The bundled Node runtime provides orchestration
and measurement-process ownership, while the configured Claude or Codex backend
runs the agents. Existing backend/model settings and the two-JSON interface are
preserved. See [the interface guide](interface/run_e2e.md) for configuration and
CPU-only handoff validation with `--dry-run`.

The repository, Python package `geak`, `GEAK_*` environment variables and artifact
paths retain their names so existing integrations can use this branch. A kernel
measurement and its enclosing serving-workload measurement are separate results;
use the workload's correctness, quality and latency checks when deploying a change.

## How it works

1. Measure the original implementation under the task's numerical and workload contract.
2. Plan and develop alternatives using shared engineers, profiling tools and evaluators.
3. Retain source and evidence; select earlier programs for further development or combination.
4. Independently verify a candidate before promotion, and deliver the exact tested source.
5. For serving tasks, integrate and measure the full workload.

AgentX trace replay uses its measured kernel inventory and prepared standalone
tasks. Fixed-load serving also has an outer strategy, configuration and reprofiling
loop. Both use the same kernel optimizer and serving acceptance contract; see the
[serving workflow](e2e_workflow/README.md) for their phase routing.

EVO starts retained-candidate development after the first stalled round by default.
Budgets and optional evidence/development controls are documented in the
[run guide](docs/evo/stock-workflow.md). Use a fresh evaluation directory for each run.

## Code layout

| Directory | Responsibility |
| --- | --- |
| `interface/` | Stable task/result interface and bundled agent runtime |
| `kernel_workflow/` | Kernel setup, search, archive, verification and delivery |
| `e2e_workflow/` | Serving profiles, kernel integration and workload measurement |
| `geak/` | Installation and native measurement helpers |
| `kb/`, `perf_knowledge/` | Retained knowledge, hardware references and skills |
| `scripts/native_runtime/` | Adapter for supplied task evaluators |

Edit workflow sources under `kernel_workflow/src/` and `e2e_workflow/src/`, then
run `node scripts/build_workflows.mjs`. Generated scripts are checked in for the
runner. Keep shared logic in one place; see [CONTRIBUTING.md](CONTRIBUTING.md).
