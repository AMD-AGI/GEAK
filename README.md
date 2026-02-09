# GEAK Agent

GPU Evolutionary Agent for Kernels - A simple AI agent for GPU kernel optimization.

Based on [mini-swe-agent](https://github.com/SWE-agent/mini-SWE-agent) architecture: the LLM generates bash commands and MCP tool calls to optimize GPU kernels.

## Quick Start

**Run inside the container** (enter with `./scripts/run-docker.sh`). The repo is mounted at `/workspace` so you can edit on the host and run in the container without rebuilding.

```bash
# Enter container (builds image if needed; requires AMD_LLM_API_KEY)
./scripts/run-docker.sh

# Inside container (you land in /workspace)
pytest geak_agent/tests/ -v
python geak_agent/examples/resolve_kernel_url.py /workspace/geak_agent/examples/resolve_kernel_url.py
python geak_agent/examples/profile_kernel.py 'python3 /path/to/kernel.py --profile'
```

### Run the Agent

```bash
# Inside container
python3 -m geakagent.run.mini -m claude-opus-4-5 \
  -t "Complete GEAK Agent Pipeline for examples/add_kernel/kernel.py
1. DISCOVER: Analyze the kernel
2. TEST GENERATION: Create test cases
3. BENCHMARKING: Profile baseline with Metrix and create COMMANDMENT.md
4. OPTIMIZATION: Use OpenEvolve MCP with max_iterations=10
5. Save optimized kernel and metrics" \
  --yolo
```

### Run on AIG-Eval Kernels

```bash
# Clone AIG-Eval (inside container or mounted)
git clone -b geak-eval-kernels git@github.com:AMD-AGI/AIG-Eval.git

# Run optimization on a single kernel
cd AIG-Eval/tasks/geak_eval
bash run.sh fused_qkv_rope --optimize --gpu 0 --iterations 5

# Run on all 8 kernels
bash run.sh --all --optimize --gpu 0 --iterations 10
```

### Direct OpenEvolve Invocation

```bash
# Auto-build mode (generates COMMANDMENT.md from kernel)
python3 $GEAK_OE_ROOT/examples/geak_eval/run_openevolve.py \
  /path/to/kernel.py \
  --iterations 10 --gpu 0 --output /path/to/output

# Pre-built COMMANDMENT mode (for custom evaluation pipelines)
python3 $GEAK_OE_ROOT/examples/geak_eval/run_openevolve.py \
  /path/to/kernel.py \
  --iterations 10 --gpu 0 --output /path/to/output \
  --commandment /path/to/COMMANDMENT.md \
  --baseline-metrics /path/to/baseline_metrics.json
```

See [docs/RUNTIME_QUICKSTART.md](docs/RUNTIME_QUICKSTART.md) and [docs/RUNTIME_ENV.md](docs/RUNTIME_ENV.md) for runtime options.

**Optional configuration** (e.g. for benchmarks or evals that need stricter behavior):

- **Protected files**: set env `GEAK_PROTECTED_FILES` to a comma-separated list of basenames (e.g. `kernel.py`) to block shell commands from overwriting those files. Alternatively set `env.protected_files` in your config.
- **Summary on cost limit**: when the cost limit is hit, the agent can get one extra step to write a summary. Enable via env `GEAK_SUMMARY_ON_COST_LIMIT=1` (or set `agent.summary_on_cost_limit: true` in config). Optionally set `GEAK_SUMMARY_ON_LIMIT_PROMPT` (or `agent.summary_on_limit_prompt`) to the exact instruction text for that step.

---

## Architecture

```
GEAK-MSA Pipeline (mini-SWE-agent)
====================================
1. DISCOVER    --> Find kernel files
2. TEST GEN    --> Generate test cases / correctness checks
3. BASELINE    --> Profile with Metrix, validate evaluation commands
4. FREEZE      --> Write COMMANDMENT.md (immutable during optimization)
5. OPTIMIZE    --> OpenEvolve evolutionary optimization
6. REPORT      --> Best kernel + speedup metrics

COMMANDMENT.md = Universal Contract
=====================================
  SETUP:        Environment setup, GPU warmup
  CORRECTNESS:  Deterministic correctness check vs baseline
  PROFILE:      Metrix hardware profiling (warm-up + measurement)
  - Written ONLY after all commands validated on baseline
  - Frozen during OpenEvolve evolution (never changes)
  - Each candidate evaluated with exact same commands
  - GPU isolation: one GPU per evaluation, no contention
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `openevolve-mcp` | OpenEvolve optimizer (COMMANDMENT-based, multi-file) |
| `kernel-profiler` | Metrix hardware profiling (rocprofv3) |
| `kernel-evolve` | LLM mutation/crossover strategies |
| `kernel-ercs` | Evaluation/reflection |
| `automated-test-discovery` | Find tests/benchmarks |

### OpenEvolve Integration

OpenEvolve uses this repo as the root: `GEAK_OE_ROOT=/workspace`. When `examples/geak_eval/run_openevolve.py` exists (e.g. on the branch you build from), openevolve-mcp and the optimizer can use it.

**What the optimizer can use:**

| Component | Purpose |
|-----------|---------|
| **GEAK_OE_ROOT** | This repo (`/workspace` in Docker); `run_openevolve.py` at `examples/geak_eval/` when present |
| **openevolve-mcp** | MCP server that invokes `run_openevolve.py`; tool `optimize_kernel(kernel_path, max_iterations, gpu, output_dir, commandment_path, baseline_metrics_path)` |
| **geakagent.optimizer** | Python API: `optimize_kernel(..., optimizer=OptimizerType.OPENEVOLVE)` → calls openevolve-mcp |

**From code:**

```python
from geakagent.optimizer import optimize_kernel, OptimizerType

result = optimize_kernel(
    kernel_code=code,
    kernel_path="/path/to/kernel.py",
    optimizer=OptimizerType.OPENEVOLVE,
    max_iterations=10,
    gpu=0,
    output_dir="/path/to/output",
    commandment_path="/path/to/COMMANDMENT.md",  # optional
)
# result.optimized_code, result.metrics["speedup"], result.iterations
```

**From a run script (e.g. AIG-Eval):** Set `GEAK_OE_ROOT` and either run `run_openevolve.py` directly, or call the optimizer API / openevolve-mcp. COMMANDMENT.md is produced in the output directory when `run_openevolve.py` runs (auto-built if not passed via `--commandment`).

Key features:
- **Multi-file support**: Kernels can span multiple files (no SEPARATOR format)
- **COMMANDMENT.md**: Frozen, deterministic evaluation contract
- **GPU isolation**: GPUPool ensures exclusive GPU per concurrent evaluation
- **Metrix profiler**: Hardware-level metrics (bandwidth, cache hit rates, compute utilization)
- **Warm-up passes**: Stable baseline measurements (not cold-start inflated)
- **Baseline profiling in prompts**: LLM receives hardware metrics to guide optimization

## Installation

### Docker (Recommended)

```bash
git clone -b msa https://github.com/AMD-AGI/GEAK.git
cd GEAK
export AMD_LLM_API_KEY=<your-key>
./scripts/run-docker.sh
```

The Dockerfile installs: this repo (GEAK), Metrix, and all MCP servers; no separate clone (GEAK_OE_ROOT=/workspace).

### Manual Installation

```bash
pip install -e .
export GEAK_OE_ROOT=$(pwd)   # this repo; run_openevolve.py at examples/geak_eval/ when present
pip install -e mcp_tools/openevolve-mcp/ -e mcp_tools/mcp-client/
# See Dockerfile for full MCP list; install Metrix if using profiling
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `AMD_LLM_API_KEY` | API key for AMD LLM gateway (required) |
| `GEAK_OE_ROOT` | Root for OpenEvolve (default: `/workspace` in Docker, or current repo) |
| `HIP_VISIBLE_DEVICES` | GPU devices (default: `0`) |

---

## Project Structure

```
GEAK/
├── src/geakagent/       # Main agent (mini-swe-based)
│   ├── agents/          # Agent implementations
│   ├── config/          # Configuration (mini.yaml, mini_patch_agent.yaml)
│   ├── environments/    # Local, Docker, protected_files
│   ├── models/          # LLM interfaces (AMD LLM gateway)
│   ├── optimizer/       # Optimizer interface (OpenEvolve)
│   └── run/             # CLI (geak, mini)
├── mcp_tools/           # MCP servers & client
│   ├── openevolve-mcp/  # OpenEvolve optimizer
│   ├── kernel-profiler/ # Metrix hardware profiling
│   ├── kernel-evolve/   # LLM mutation strategies
│   ├── kernel-ercs/     # Evaluation/reflection
│   └── mcp-client/      # MCP protocol client
├── geak_agent/          # Discovery, examples, resolve_kernel_url
├── examples/             # Example kernels
├── scripts/              # run-docker.sh
├── docs/                 # Documentation
└── reference/            # optimization_strategies.py, state.py
```

---

## Reference

- `reference/optimization_strategies.py` - GPU optimization strategies
- `reference/state.py` - State management for optimization runs
