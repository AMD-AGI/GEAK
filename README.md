# GEAK Agent

GPU Evolutionary Agent for Kernels - A simple AI agent for GPU kernel optimization.

Based on [mini-swe-agent](https://github.com/SWE-agent/mini-SWE-agent) architecture: the LLM generates bash commands and MCP tool calls to optimize GPU kernels.

## Quick Start

### Docker Setup (Recommended)

```bash
# Build and run the Docker container
cd GEAK    # this repo (branch: msa)
export AMD_LLM_API_KEY=<your-key>
bash scripts/run-docker.sh
```

This builds a container with all tools pre-installed (OpenEvolve, Metrix, MCP servers).

### Run the Agent

```bash
# Inside Docker container:
python3 -m geakagent.run.mini \
  -m claude-opus-4-5 \
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

OpenEvolve is auto-installed in the Docker container from the
[`optimizer-geak-openevolve`](https://github.com/AMD-AGI/GEAK/tree/optimizer-geak-openevolve) branch.

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
bash scripts/run-docker.sh
```

The Dockerfile handles all dependencies:
- GEAK-agent (this package)
- OpenEvolve (optimizer-geak-openevolve branch)
- Metrix (hardware profiler from AMD intellikit)
- All MCP servers

### Manual Installation

```bash
# 1. Install main package
pip install -e .

# 2. Clone and install OpenEvolve
git clone -b optimizer-geak-openevolve https://github.com/AMD-AGI/GEAK.git geak-oe
cd geak-oe && pip install -e . --no-build-isolation && cd ..
export GEAK_OE_ROOT=$(pwd)/geak-oe

# 3. Install MCP servers
pip install -e mcp_tools/openevolve-mcp/
pip install -e mcp_tools/mcp-client/
pip install -e mcp_tools/kernel-profiler/
pip install -e mcp_tools/kernel-evolve/
pip install -e mcp_tools/kernel-ercs/

# 4. Install Metrix (requires ROCm)
git clone https://github.com/AMDResearch/intellikit.git
cd intellikit/metrix && pip install -e . && cd ../..
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AMD_LLM_API_KEY` | API key for AMD LLM gateway | Yes |
| `GEAK_OE_ROOT` | Path to OpenEvolve repo (default: `/workspace/geak-oe`) | No |
| `OPENAI_API_KEY` | Set to `AMD_LLM_API_KEY` value for OpenEvolve | For optimization |
| `HIP_VISIBLE_DEVICES` | GPU devices to use (default: `0`) | No |

## Project Structure

```
GEAK/  (branch: msa)
|-- src/geakagent/          # Main agent (minswe-based)
|   |-- agents/             # Agent implementations
|   |-- config/             # Configuration files (mini.yaml)
|   |-- environments/       # Execution environments (local, docker)
|   |-- models/             # LLM interfaces (AMD LLM gateway)
|   |-- optimizer/          # Optimizer interface
|   +-- run/                # CLI entry points
|-- mcp_tools/              # MCP servers
|   |-- openevolve-mcp/     # OpenEvolve optimizer (subprocess-based)
|   |-- kernel-profiler/    # Metrix hardware profiling
|   |-- kernel-evolve/      # LLM mutation strategies
|   |-- kernel-ercs/        # Evaluation/reflection
|   +-- mcp-client/         # MCP protocol client
|-- examples/               # Example kernels (add_kernel)
|-- scripts/                # Docker setup (run-docker.sh)
+-- docs/                   # Documentation
```

## Reference

The `reference/` directory contains useful code from the original agent:
- `optimization_strategies.py` - GPU optimization strategies and patterns
- `state.py` - State management for optimization runs
