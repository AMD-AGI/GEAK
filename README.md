# GEAK — GPU Evolutionary Agent for Kernels

GEAK is an AI-powered framework for automated GPU kernel optimization. It profiles kernels, identifies bottlenecks, generates optimization strategies, and applies them via an LLM-driven orchestrator.

## Quick Start

### Prerequisites

- Docker with AMD GPU access (`/dev/kfd`, `/dev/dri`)
- `AMD_LLM_API_KEY` environment variable set

### Enter the container

```bash
export AMD_LLM_API_KEY=your-key-here
scripts/run-docker.sh              # interactive shell
scripts/run-docker.sh --rebuild    # rebuild image first
```

### Full pipeline (recommended)

```bash
geak --kernel-url <github_url>                    # single GPU
geak --kernel-url <github_url> --gpu-ids 0,1,2,3  # multi-GPU parallel
```

This runs the three-layer architecture automatically:
1. **Preprocessor**: resolve URL, discover tests, profile, build baselines, generate commandments
2. **Orchestrator**: LLM agent that generates tasks, dispatches to GPUs, iterates until converged or step limit reached
3. **Sub-agents**: `geak --from-task` executes individual optimisation tasks

Output is written to `geak_output/` with independently verifiable stages:
```
geak_output/
├── resolved.json               # Preprocessor: resolved kernel URL
├── CODEBASE_CONTEXT.md         # Preprocessor: repo structure and key files
├── discovery.json              # Preprocessor: discovered tests/benchmarks
├── profile.json                # Preprocessor: GPU profiling results
├── baseline_metrics.json       # Preprocessor: baseline performance
├── COMMANDMENT.md              # Preprocessor: correctness/profiling contract
├── tasks/round_N/*.md          # Orchestrator: generated task files per round
├── results/round_N/<label>/    # Sub-agents: patches, test outputs, logs
└── final_report.json           # Orchestrator: best result summary
```

## High-Level Commands

| Command | Description |
|---------|-------------|
| `geak --kernel-url <url>` | Full pipeline: preprocess + orchestrate + optimize |
| `geak --from-task <task.md>` | Run a single optimization task on one GPU |
| `geak-preprocess <url> -o dir/` | Run only the preprocessor stage |
| `geak-orchestrate --preprocess-dir dir/ --gpu-ids 0,1` | Run only the orchestrator on existing preprocessor output |

## Low-Level Pipeline Commands

Each step reads the previous step's output via `--from-*` flags and can be run independently:

```bash
# 1. Resolve a GitHub kernel URL to a local checkout
resolve-kernel-url <url> --json -o resolved.json

# 2. Discover tests and benchmarks for the kernel
#    (uses the automated-test-discovery MCP tool internally)

# 3. Profile the kernel on GPU
kernel-profile --from-discovery discovery.json --json -o profile.json

# 4. Extract baseline performance metrics
baseline-metrics build --from-profile profile.json --all -o baseline_metrics.json

# 5. Generate the correctness/profiling contract
commandment --from-discovery discovery.json -o COMMANDMENT.md

# 6. Generate optimization tasks (LLM-driven)
task-generator --from-discovery discovery.json \
    --profiling profile.json \
    --commandment COMMANDMENT.md \
    --baseline-metrics baseline_metrics.json \
    -o tasks/round_1/

# 7. Run tasks in parallel across GPUs
run-tasks --task-dir tasks/round_1/ --gpu-ids 0,1,2,3

# 8. Select the best patch from results
select-patch --patch-dir results/round_1/<label>/
```

## Other Tools

| Command | Description |
|---------|-------------|
| `validate-commandment <path>` | Validate a COMMANDMENT.md file |
| `openevolve-worker --from-task <task.md>` | Run an OpenEvolve optimization task |
| `select-patch --patch-dir <dir>` | LLM-driven patch selection from parallel runs |

## GPU Isolation

Every tool an agent uses (bash, profile_kernel, openevolve) inherits the agent's `HIP_VISIBLE_DEVICES` so parallel tasks never contend on the same GPU.

| Entry point | How GPUs are isolated |
|---|---|
| `geak --kernel-url <url> --gpu-ids 0,1,2,3` | Orchestrator dispatches tasks via `ParallelAgent._run_pool()`. Each task acquires N GPU slots from a queue; `HIP_VISIBLE_DEVICES` is set in the agent's environment and propagated to bash, MCP tools, and subprocesses. |
| `geak --from-task <task.md> --gpu-ids 2` | Single-agent mode sets `env.config.env["HIP_VISIBLE_DEVICES"]` before creating the agent. `DefaultAgent.__init__` calls `toolruntime.set_env()` to propagate to all tools. |
| `geak-orchestrate --gpu-ids 0,1,2,3` | Same as full pipeline -- dispatches through `ParallelAgent._run_pool()`. |
| `openevolve-worker --gpu 0,1` | Sets `os.environ["HIP_VISIBLE_DEVICES"]` before spawning the optimizer subprocess. Accepts comma-separated IDs for multi-GPU runs. |
| `task-generator --num-gpus 8` | No GPU compute, but `--num-gpus` tells the LLM planner how many GPUs are available so it generates enough tasks to fill them. |

Multi-GPU tasks (e.g. OpenEvolve with `num_gpus: 3`) acquire multiple GPU slots from the pool and receive a comma-separated `HIP_VISIBLE_DEVICES` (e.g. `"2,5,7"`). Slots are returned when the task finishes.

For implementation details, invariants, and how to add new GPU-aware tools, see [docs/gpu-isolation.md](docs/gpu-isolation.md).

## Configuration

YAML configuration files live in `src/minisweagent/config/`. Key files:

| File | Purpose |
|------|---------|
| `geak.yaml` | Base GEAK agent config (model, environment) |
| `mini_kernel_strategy_list.yaml` | Strategy-based optimization with profiling |
| `mini_unit_test_agent.yaml` | Test/benchmark discovery agent |
| `mini_select_patch.yaml` | Best-patch selection from parallel runs |

## MCP Tool Servers

Standalone MCP servers for specialized tasks:

| Server | Purpose |
|--------|---------|
| `automated-test-discovery` | Content-based test/benchmark discovery |
| `kernel-profiler` | GPU profiling via Metrix and rocprof-compute |
| `kernel-evolve` | LLM-guided kernel optimization |
| `kernel-ercs` | Kernel evaluation, reflection, compatibility checks |
| `openevolve-mcp` | OpenEvolve evolutionary optimizer |
| `mcp-client` | JSON-RPC 2.0 client for MCP communication |

Install individually: `pip install -e mcp_tools/<server-name>/`

## Architecture

```
geak <url> --gpu-ids 0,1,2,3
  │
  ├─ Preprocessor (sequential, no LLM)
  │    resolve_kernel_url → discover → profile_kernel → baseline_metrics → commandment → baseline_benchmarks
  │
  ├─ Orchestrator (LLM agent with tools)
  │    generate_tasks → dispatch_tasks → collect_results → [iterate or finalize]
  │    Auto-finalizes with best result if step limit reached
  │
  └─ Sub-agents (per task, per GPU, isolated worktrees)
       geak --from-task (kernel-ercs, kernel-evolve, openevolve, minisweagent)
```

```
src/minisweagent/
├── run/
│   ├── mini.py              # geak entry point: routes to preprocessor+orchestrator or --from-task
│   ├── preprocessor.py      # Sequential pipeline calling existing Python APIs
│   ├── orchestrator.py      # LLM orchestrator agent with tools + auto-finalization
│   ├── dispatch.py          # Task dispatch via ParallelAgent pool mode
│   ├── task_generator.py    # LLM-driven task generation (agent-based, no rule fallbacks)
│   ├── task_runner.py       # Batch task execution CLI
│   └── task_file.py         # Task file I/O (YAML frontmatter + Markdown)
├── agents/                  # Agent types (default, parallel, strategy, openevolve, select_patch)
├── tools/                   # Built-in tools (bash, editor, profiling, discovery, commandment)
├── models/                  # LLM backends (amd_llm, anthropic, litellm, etc.)
├── kernel_profile.py        # Dual-backend GPU profiling CLI (Metrix + rocprof-compute)
└── baseline_metrics.py      # Format profiler output for downstream tools
mcp_tools/                   # Standalone MCP servers (profiler, kernel-evolve, kernel-ercs, etc.)
```

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AMD_LLM_API_KEY` | (required) | LLM API key |
| `GEAK_MAX_ROUNDS` | `5` | Max orchestration rounds |
| `GEAK_BENCHMARK_EXTRA_ARGS` | `--iterations 50` | Extra args for benchmark commands (set by pipeline) |
| `GEAK_ALLOWED_AGENTS` | (all) | Comma-separated agent type allowlist |
| `GEAK_EXCLUDED_AGENTS` | (none) | Comma-separated agent type blocklist |

See `INSTRUCTIONS.md` Section 6 for the full reference.

## License

MIT License - see LICENSE.md for details