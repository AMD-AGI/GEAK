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

### OpenEvolve Integration

OpenEvolve is auto-installed in the Docker container from the [`optimizer-geak-openevolve`](https://github.com/AMD-AGI/GEAK/tree/optimizer-geak-openevolve) branch.

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
| `GEAK_OE_ROOT` | Path to OpenEvolve repo (default: `/workspace/geak-oe`) |
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
