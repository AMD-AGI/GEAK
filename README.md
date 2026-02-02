# GEAK Agent

GPU Evolutionary Agent for Kernels - A simple AI agent for GPU kernel optimization.

Based on [mini-swe-agent](https://github.com/SWE-agent/mini-SWE-agent) architecture: the LLM generates bash commands and MCP tool calls to optimize GPU kernels.

## 🚀 Quick Start

```bash
# Auto-detect environment and run agent (recommended)
python3 -m geakagent.run.mini -m claude-sonnet-4.5 \
  -t "Optimize kernel at /path/to/kernel.py" \
  --yolo

# Force Docker with default image
python3 -m geakagent.run.mini -m claude-sonnet-4.5 \
  -t "Optimize kernel" \
  --runtime docker \
  --workspace /path/to/kernels \
  --yolo
```

**💡 New:** Auto-detects GPU environment, defaults to Docker `lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x` if needed.

📖 See [RUNTIME_QUICKSTART.md](RUNTIME_QUICKSTART.md) | [RUNTIME_ENV.md](RUNTIME_ENV.md) for details.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      GEAK AGENT                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LLM generates:                                             │
│    - Bash commands (edit files, run tests, benchmark)       │
│    - MCP tool calls (profile, evolve, discover)             │
│                                                             │
│  Execute → Observe → Repeat                                 │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Available MCPs:                                            │
│    - automated-test-discovery  (find tests/benchmarks)      │
│    - kernel-profiler          (rocprof-compute profiling)   │
│    - kernel-evolve            (LLM mutation/crossover)      │
│    - kernel-ercs              (evaluation/reflection)       │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone -b msa2 https://github.com/AMD-AGI/GEAK-agent.git
cd GEAK-agent

# 1. Install main package
pip install -e .

# 2. Clone and install OpenEvolve
git clone -b geak-openevolve https://github.com/AMD-AGI/GEAK-agent.git geak-oe
cd geak-oe && PIP_USER=1 python3 -m pip install -e . --no-build-isolation && cd ..

# 3. Install MCP servers
pip install -e mcp_tools/openevolve-mcp/
python3 -m pip install -e mcp_tools/mcp-client/ --no-build-isolation
```

## Usage

### Example: Full Pipeline Command

```bash
export ANTHROPIC_API_KEY="your-api-key"
python3 -m geakagent.run.mini \
  -m claude-sonnet-4.5 \
  -t "Complete GEAK Agent Pipeline for examples/add_kernel/kernel.py

1. DISCOVER: Analyze the kernel
2. TEST GENERATION: Create test cases in examples/add_kernel/tests/
3. BENCHMARKING: Save baseline metrics to benchmark/baseline/metrics.json
4. OPTIMIZATION: Use OpenEvolve MCP (mcp_tools/openevolve-mcp) with max_iterations=10
5. Save optimized kernel to kernel_optimized.py and metrics to benchmark/optimized/metrics.json" \
  --yolo
```

## Project Structure

```
GEAK-agent/
├── src/geakagent/          # Main agent (minswe-based)
│   ├── agents/             # Agent implementations
│   ├── config/             # Configuration files
│   ├── environments/       # Execution environments
│   ├── models/             # LLM interfaces
│   ├── optimizer/          # Optimizer interface
│   └── run/                # CLI entry points
├── mcp_tools/              # MCP servers & client (consolidated)
│   ├── automated-test-discovery/   # MCP: test discovery
│   ├── kernel-profiler/            # MCP: GPU profiling
│   ├── kernel-evolve/              # MCP: optimization strategies
│   ├── kernel-ercs/                # MCP: evaluation/reflection
│   ├── openevolve-mcp/             # MCP: OpenEvolve optimizer
│   └── mcp-client/                 # MCP: Protocol client (JSON-RPC)
├── geak_agent/             # Discovery pipeline
├── examples/               # Example kernels
├── reference/              # Reference files from old agent
│   ├── optimization_strategies.py
│   └── state.py
└── docs/
```

## Reference

The `reference/` directory contains useful code from the original agent:
- `optimization_strategies.py` - GPU optimization strategies and patterns
- `state.py` - State management for optimization runs
