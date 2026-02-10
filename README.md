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

### Run with a kernel URL (weblink)

You can pass the kernel as a GitHub URL instead of a local path. The agent will resolve it to a local path (cloning the repo into the workspace if needed) and inject path, optional line number, and kernel name into the task.

```bash
# URL only (no line): discovery identifies kernel(s) in the file
python3 -m geakagent.run.mini -m claude-opus-4-5 \
  --kernel-url "https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/rope/rope.py" \
  -t "Complete GEAK Agent Pipeline: 1. DISCOVER 2. TEST GEN 3. BENCHMARK 4. OPTIMIZE 5. Save results" \
  --yolo

# URL with line (e.g. #L106): resolved kernel name is included in the task for discovery/context
python3 -m geakagent.run.mini -m claude-opus-4-5 \
  --kernel-url "https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/rope/rope.py#L106" \
  -t "Complete GEAK Agent Pipeline: 1. DISCOVER 2. TEST GEN 3. BENCHMARK 4. OPTIMIZE 5. Save results" \
  --yolo
```

- **Line number is optional.** Without `#L123`, the injected task tells the agent that discovery should identify the kernel(s). With `#L123`, the resolved kernel function name is included for context. When profiling, all kernels are reported and the agent chooses which to use.
- Resolving uses `geak_agent.resolve_kernel_url`; when using Docker, the clone is placed under the workspace so the container sees the file.

### Run on AIG-Eval Kernels

```bash
# Clone AIG-Eval (inside container or mounted)
git clone -b geak-eval-kernels git@github.com:AMD-AGI/AIG-Eval.git

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

# Pre-built COMMANDMENT mode
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
  - Frozen during OpenEvolve evolution; GPU isolation per evaluation
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `openevolve-mcp` | OpenEvolve optimizer (COMMANDMENT-based, multi-file) |
| `kernel-profiler` | Metrix hardware profiling (rocprofv3) |
| `kernel-evolve` | LLM mutation/crossover strategies |
| `kernel-ercs` | Evaluation/reflection |
| `automated-test-discovery` | Find tests/benchmarks |

### OpenEvolve Integration (geak-oe)

OpenEvolve is auto-installed in the Docker container from the [`optimizer-geak-openevolve`](https://github.com/AMD-AGI/GEAK/tree/optimizer-geak-openevolve) branch (clone name: **geak-oe**, path: `GEAK_OE_ROOT`).

**What the optimizer can use (from the cherry-pick / newer additions):**

| Component | Purpose |
|-----------|---------|
| **geak-oe** | GEAK repo clone; contains `run_openevolve.py` (builds COMMANDMENT.md, runs evolution) |
| **openevolve-mcp** | MCP server that invokes `run_openevolve.py`; tool `optimize_kernel(kernel_path, max_iterations, gpu, output_dir, commandment_path, baseline_metrics_path)` |
| **geakagent.optimizer** | Python API: `optimize_kernel(..., optimizer=OptimizerType.OPENEVOLVE)` → calls openevolve-mcp (and thus geak-oe) |

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

- **Multi-file support**: Kernels can span multiple files (no SEPARATOR format)
- **COMMANDMENT.md**: Frozen, deterministic evaluation contract
- **GPU isolation**: GPUPool ensures exclusive GPU per concurrent evaluation
- **Metrix profiler**: Hardware-level metrics; warm-up passes for stable baselines

---

## Installation

### Docker (Recommended)

```bash
git clone -b msa https://github.com/AMD-AGI/GEAK-agent.git
cd GEAK-agent
export AMD_LLM_API_KEY=<your-key>
./scripts/run-docker.sh
```

The Dockerfile installs: GEAK-agent, OpenEvolve (optimizer-geak-openevolve), Metrix, and all MCP servers.

### Manual Installation

```bash
pip install -e .
git clone -b optimizer-geak-openevolve https://github.com/AMD-AGI/GEAK.git geak-oe
cd geak-oe && pip install -e . --no-build-isolation && cd ..
export GEAK_OE_ROOT=$(pwd)/geak-oe
pip install -e mcp_tools/openevolve-mcp/ -e mcp_tools/mcp-client/
# See Dockerfile for full MCP list; install Metrix if using profiling
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `AMD_LLM_API_KEY` | API key for AMD LLM gateway (required) |
| `GEAK_OE_ROOT` | Path to OpenEvolve repo (default: `/opt/geak-oe` in Docker so it is not hidden by the `-v REPO:/workspace` mount) |
| `HIP_VISIBLE_DEVICES` | GPU devices (default: `0`) |

---

## Project Structure

```
GEAK-agent/
├── src/geakagent/       # Main agent
├── mcp_tools/           # MCP servers & client (openevolve-mcp, kernel-profiler, etc.)
├── geak_agent/          # Discovery, examples, resolve_kernel_url
├── examples/            # Example kernels
├── scripts/             # run-docker.sh
├── docs/                # Documentation
└── reference/           # optimization_strategies.py, state.py
```

---

## Reference

- `reference/optimization_strategies.py` - GPU optimization strategies
- `reference/state.py` - State management for optimization runs
