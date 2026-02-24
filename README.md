# GEAK-v3

GEAK is an AI-powered framework for automated GPU kernel optimization, built on top of mini-SWE-agent.

It enables systematic, profiling-driven, and scalable optimization of GPU kernels — evolving from single-kernel tuning (v1/v2) to full repository-level autonomous optimization (v3).

## Table of Contents

- [Evolution: From Kernel-Level to Repo-Level Automation](#evolution-from-kernel-level-to-repo-level-automation)
- [Core Architecture](#core-architecture)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Configuration (geak.yaml)](#configuration-geakyaml)
  - [Output & Artifacts](#output--artifacts)
- [Features](#features)
  - [Unit test discovery](#unit-test-discovery)
  - [System Tools (built-in)](#system-tools-built-in)
  - [Best patch selection](#best-patch-selection)
- [Summary](#summary)

---

## Evolution: From Kernel-Level to Repo-Level Automation

### GEAK v1 / v2 — Single Kernel Optimization

Earlier versions focused on optimizing individual GPU kernels through iterative patch generation and performance validation.

They demonstrated that LLM-based agents can:

- Analyze kernel structure
- Propose optimization strategies
- Generate performance-improving patches

### GEAK v3 — Autonomous Repo-Level Optimization

GEAK v3 upgrades the system into a full lifecycle GPU optimization framework.

It operates at repository scale and automates:

- 🔍 Test discovery and generation
- 📊 Baseline performance measurement
- 🧠 Profiling-guided bottleneck diagnosis
- 🎯 Strategy planning and execution
- ✅ Patch validation and regression testing
- 🔁 Multi-round iterative improvement

The system forms a closed-loop optimization engine capable of continuous performance evolution with minimal human intervention.

---

## Core Architecture

### End-to-End Optimization Engine

At its core, GEAK runs a fully autonomous optimization loop:

**Test Detection → Baseline → Profiling → Strategy Planning → (Patch Generation → Validation) × N → Best-performing Kernel**

Each optimization step is:

- Correctness-verified
- Performance-measured
- Version-tracked

### Tool-Augmented Intelligence Layer

GEAK v3 introduces a structured tool ecosystem that enhances agent reasoning and execution quality.

The system integrates:

- **Profiling** for quantitative bottleneck identification
  (memory bandwidth, occupancy, register pressure, execution stalls)

- **Optimization Strategy Management** for tracking explored techniques, marking successful/failed strategies, and prioritizing high-impact directions

- **Version & Patch Management** for automatic diff tracking, benchmarking history, regression detection, and best-patch selection

### Parallel Exploration & Scaling

GEAK v3 supports parallel optimization agents. This parallel scaling:

- Raises the optimization ceiling
- Increases robustness of exploration
- Reduces dependence on single optimization trajectories

---

## Getting Started

### Installation

```bash
git clone https://github.com/AMD-AGI/GEAK
cd GEAK
git switch -c dev origin/dev
pip install -e .

# set LLM API key
export AMD_LLM_API_KEY="YOUR_KEY"
```

### Usage

#### Basic (single-agent) optimization
- Add `--yolo` to run end-to-end without interactive confirmation.

```bash
mini --config geak.yaml \
  --task "Optimize the kernel in src/kernel.cpp" \
  --yolo
```

#### Parallel optimization (multiple agents + best patch selection)
- Each agent works in an isolated git workspace
- Patches and test results are saved separately
- After all runs finish, GEAK automatically selects the best patch based on the specified metric

```bash
mini --config geak.yaml \
  --num-parallel 4 \
  --repo /path/to/kernel/repo \
  --task "Optimize block_reduce kernel" \
  --gpu-ids 0,1,2,3 \
  --metric "Extract Bandwidth in GB/s (higher is better)" \
  --yolo
```

**Notes:**

- `--num-parallel`: number of optimization agents
- `--repo`: required when `--num-parallel > 1` (each agent uses an isolated git worktree)
- `--gpu-ids`: comma-separated GPU IDs for agents
- `--metric`: natural-language instruction for extracting/comparing metrics from test logs
- `--yolo`: run end-to-end without interactive confirmation.

### Configuration (geak.yaml)

`mini` loads configs in layers:

1. base config: `mini.yaml`
2. template: `mini_kernel_strategy_list.yaml` (default)
3. user override: `--config geak.yaml` (**final override**)

This means you can configure tools and parallel defaults directly in `geak.yaml`.

### Output & Artifacts

GEAK saves patches + test logs so results are reproducible.

- **Default output base**: `optimization_logs/`
- **Auto-generated run directory**: `optimization_logs/<kernel_name>_<YYYYmmdd_HHMMSS>/`
- **Parallel runs**: subfolders `parallel_0/`, `parallel_1/`, ...

Typical structure (parallel run):

```bash
optimization_logs/<kernel>_<timestamp>/
├── parallel_0/
│   ├── patch_0.patch
│   ├── patch_0_test.txt
│   ├── patch_1.patch
│   ├── patch_1_test.txt
│   └── agent_0.log
├── parallel_1/
│   └── ...
├── best_results.json
└── select_agent.log
```

**Notes:**

- `patch_N.patch` is a `git diff` of the working tree at that iteration.
- `patch_N_test.txt` is the full stdout/stderr of the test command for that patch.
- `best_results.json` is produced by the patch selection agent after all runs finish.

---

## Features

### Unit test discovery

If you pass `--create-test`, or you **do not** provide `--test-command`, GEAK will run a **UnitTestAgent** that tries to:

1. discover an existing build + correctness test + benchmark flow, or
2. create minimal tests/benchmarks if none exist,

and then returns a single shell command string used for baseline/patch comparisons.

```bash
mini --config geak.yaml \
  --repo /path/to/kernel/repo \
  --create-test \
  --task "Optimize device_batch_memcpy kernel"
```

### System Tools (built-in)

GEAK’s model can call tools defined in `src/minisweagent/tools/tools.json` and implemented in `src/minisweagent/tools/`.

| Tool | Purpose | Key artifacts / outputs |
| --- | --- | --- |
| `profiling` | Profile the workload to identify bottlenecks | rocprofiler-compute summary for the agent |
| `strategy_manager` | Track explored optimization strategies in a markdown file | `.optimization_strategies.md` |
| `test_perf` | Save current diff as a patch and run the configured `test_command` | `patch_N.patch`, `patch_N_test.txt` |

#### Custom Tool Integration

GEAK does not use a decorator-based “ToolRegistry”. To add a new tool:

1. **Define the tool schema** in `src/minisweagent/tools/tools.json` (name, description, JSON parameters).
2. **Implement the tool** in `src/minisweagent/tools/` (return a dict with `output` and `returncode`).
3. **Register it in runtime** by adding it to `ToolRuntime._tool_table` in `src/minisweagent/tools/tools_runtime.py`.

After that, the model can call your tool by name via the tool API.

### Best patch selection

After parallel runs finish, GEAK runs a selection agent that:

- reads all `patch_*_test.txt` logs
- extracts metrics according to `--metric`
- writes:
  - `best_results.json`
  - `select_agent.log`

---

## Summary

GEAK v3 enables reproducible, measurable, and scalable optimization at repository scale — beyond isolated manual tuning. It integrates profiling, strategy management, automated validation, and parallel exploration into a structured performance engineering workflow. 

Contributions, experiments, and feedback are welcome.
