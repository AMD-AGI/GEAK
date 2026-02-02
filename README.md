# GEAK Agent

GPU Evolutionary Agent for Kernels - A simple AI agent for GPU kernel optimization.

Based on [mini-swe-agent](https://github.com/SWE-agent/mini-SWE-agent) architecture: the LLM generates bash commands and MCP tool calls to optimize GPU kernels.

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
pip install -e .
```

## Usage

```bash
# Run the agent
geak "Optimize the kernel at /path/to/kernel.py"
```

## MCP Servers

### automated-test-discovery
Find tests and benchmarks for a kernel file.
```bash
pip install -e automated-test-discovery/
```

### kernel-profiler
Profile GPU kernels using rocprof-compute.
```bash
pip install -e kernel-profiler/
```

### kernel-evolve
LLM-based kernel mutation and crossover.
```bash
pip install -e kernel-evolve/
```

### kernel-ercs
Kernel evaluation, reflection, and compatibility checking.
```bash
pip install -e kernel-ercs/
```

## Project Structure

```
GEAK-agent/
├── src/geakagent/          # Main agent (minswe-based)
│   ├── agents/             # Agent implementations
│   ├── config/             # Configuration files
│   ├── environments/       # Execution environments
│   ├── models/             # LLM interfaces
│   └── run/                # CLI entry points
├── automated-test-discovery/   # MCP: test discovery
├── kernel-profiler/            # MCP: GPU profiling
├── kernel-evolve/              # MCP: optimization strategies
├── kernel-ercs/                # MCP: evaluation/reflection
├── mini_kernel/                # Core tools (discovery, metrix, cli)
├── reference/                  # Reference files from old agent
│   ├── optimization_strategies.py
│   └── state.py
└── docs/
```

## Documentation

Detailed documentation for core components:

- **[Discovery Pipeline](docs/DISCOVERY_PIPELINE.md)** - Automated test and benchmark discovery
- **[MetrixTool](docs/METRIX_TOOL.md)** - GPU kernel profiler with hardware metrics

## Reference

The `reference/` directory contains useful code from the original agent:
- `optimization_strategies.py` - GPU optimization strategies and patterns
- `state.py` - State management for optimization runs
