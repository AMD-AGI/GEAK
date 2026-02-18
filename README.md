# GEAK — GPU Evolutionary Agent for Kernels

GEAK is an AI-powered framework for automated GPU kernel optimization. It profiles kernels, identifies bottlenecks, generates optimization strategies, and applies them -- either as a single pipeline or through modular, chainable CLI commands.

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

### All-in-one

```bash
geak <github_url>                  # full optimization pipeline
```

### Modular pipeline

Each step reads the previous step's output via `--from-*` flags:

```bash
# 1. Resolve a GitHub kernel URL to a local clone
resolve-kernel-url <url> --json -o resolved.json

# 2. Discover tests and benchmarks in the repo
test-discovery --from-resolved resolved.json -o discovery.json

# 3. Profile the kernel (Metrix or rocprof-compute backend)
kernel-profile --from-discovery discovery.json --json -o profile.json
kernel-profile --from-discovery discovery.json --backend rocprof-compute --json -o profile.json

# 4. Extract baseline metrics for OpenEvolve
baseline-metrics build --from-profile profile.json --all -o baseline_metrics.json

# 5. Generate a COMMANDMENT (optimization constraints doc)
commandment --from-discovery discovery.json -o COMMANDMENT.md

# 6. Generate optimization tasks
task-generator --from-discovery discovery.json --profiling profile.json \
    --commandment COMMANDMENT.md --baseline-metrics baseline_metrics.json \
    -o tasks/round_1/
```

### Run individual tasks

```bash
openevolve-worker --from-task tasks/round_1/00_openevolve-inner.md --gpu 0
geak --from-task tasks/round_1/10_triton-autotune.md --gpu-ids 2
```

### Iterative refinement

```bash
task-generator ... --from-results results/round_1/ --round 2 -o tasks/round_2/
```

### Other tools

```bash
validate-commandment <path>                     # validate a COMMANDMENT.md
openevolve-worker --kernel-path <p> ...          # run OpenEvolve optimizer (manual)
select-patch --patch-dir <dir> ...               # select best patch from runs
```

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
| `kernel-profiler` | GPU profiling via rocprof-compute |
| `kernel-evolve` | LLM-guided kernel optimization |
| `kernel-ercs` | Kernel evaluation, reflection, compatibility checks |
| `metrix-mcp` | AMD Metrix API for GPU metrics |
| `openevolve-mcp` | OpenEvolve evolutionary optimizer |
| `mcp-client` | JSON-RPC 2.0 client for MCP communication |

Install individually: `pip install -e mcp_tools/<server-name>/`

## Architecture

```
GEAK
├── src/minisweagent/          # Core agent framework
│   ├── agents/                # Agent types (default, parallel, select_patch, etc.)
│   ├── tools/                 # Built-in tools (bash, editor, profiling, discovery, etc.)
│   ├── config/                # YAML configs
│   ├── environments/          # Execution environments (local, docker, singularity)
│   ├── models/                # LLM backends (amd_llm, anthropic, litellm, etc.)
│   ├── optimizer/             # OpenEvolve + Autotune
│   ├── kernel_profile.py      # Dual-backend GPU profiling CLI
│   └── baseline_metrics.py    # Format profiler output for OpenEvolve
├── mcp_tools/                 # Standalone MCP servers + client
├── knowledge_base/            # GPU optimization strategies database
├── eval_suite/                # Regression suite
├── examples/                  # Example kernels
└── Dockerfile                 # Docker image with all tools
```

## License

MIT License - see LICENSE.md for details
