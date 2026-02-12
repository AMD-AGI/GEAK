# GEAK — GPU Evolutionary Agent for Kernels

GEAK is an AI-powered framework for automated GPU kernel optimization built on top of mini-SWE-agent. It provides advanced features for systematic kernel optimization through exploration-based approaches, parallel execution, and profiling-guided optimization.

## Features

### 🎯 Core Features

- **Strategy-Based Optimization**: Systematic exploration of multiple optimization strategies with tracking and comparison
- **Unit Test Creation**: Automatic generation and discovery of test commands for kernel validation
- **Parallel Scaling**: Run multiple optimization agents in parallel with automatic best patch selection
- **Profiling Tools**: Deep GPU kernel profiling with rocprof integration for bottleneck identification
- **Patch Management**: Automatic patch saving, testing, and selection from parallel runs
- **Multi-Agent Architecture**: Different specialized agents for different optimization workflows

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Basic kernel optimization
mini --config geak.yaml --task "Optimize the kernel in src/kernel.cpp"

# With parallel scaling
mini --config mini_kernel_strategy_list.yaml --num-parallel 4 --repo /path/to/kernel/repo
```

## Configuration System

GEAK uses YAML configuration files to define agent behaviors, tools, and optimization workflows. Configuration files are located in `src/minisweagent/config/`.

### Key Configuration Files

#### 1. `geak.yaml` - Basic GEAK Configuration

The base configuration for GEAK agents with model and environment settings.

#### 2. `mini_kernel_strategy_list.yaml` - Strategy-Based Optimization

Configuration for systematic optimization with strategy exploration and profiling.

**Key Features:**
- Strategy manager tool for tracking optimization approaches
- Profiling tool for bottleneck identification
- Hardware-aware optimization workflow
- Baseline performance measurement before optimization

**Workflow:**
1. Query hardware information (GPU arch, CU count, memory bandwidth)
2. Establish baseline performance with `test_perf` tool
3. Run `profiling` tool to identify bottlenecks
4. Create strategy list with `strategy_manager` tool
5. Explore strategies one by one, measuring impact
6. Combine successful strategies

**Available Tools:**
- `bash`: Execute shell commands
- `str_replace_editor`: File editing (view, create, str_replace, insert)
- `test_perf`: Save patch and run performance tests
- `submit`: Submit final optimized implementation
- `strategy_manager`: Track and manage optimization strategies
- `profiling`: Profile GPU kernel with rocprof

#### 3. `mini_unit_test_agent.yaml` - Unit Test Creation

Configuration for the specialized agent that finds or creates test commands.

**Purpose:** 
- Discovers existing tests/benchmarks in the repository
- Creates minimal correctness and performance tests if none exist
- Returns a single executable test command as `TEST_COMMAND: <command>`

**Output Format:**
```
MINI_SWE_AGENT_FINAL_OUTPUT
TEST_COMMAND: ./benchmark/benchmark_kernel --trials 20
```

#### 4. `mini_select_patch.yaml` - Patch Selection

Configuration for the agent that selects the best patch from parallel runs.

**Process:**
1. Analyzes patches from multiple parallel agents
2. Compares test results and performance metrics
3. Computes speedup based on user-provided metric
4. Saves selection to `best_results.json`

**Output Format (`best_results.json`):**
```json
{
  "best_patch_id": "parallel_1/patch_2",
  "best_patch_speedup": 1.45,
  "best_patch_file": "/path/to/patch_2.patch",
  "best_patch_test_output": "/path/to/patch_2_test.txt",
  "llm_selection_analysis": "Selected patch 2 from agent 1 because..."
}
```

### Configuration Parameters Reference

#### Agent Configuration

```yaml
agent:
  system_template: |
    # System prompt for the agent
  instance_template: |
    # Task-specific prompt template
  action_observation_template: |
    # Format for tool call results
  format_error_template: |
    # Error message format
  step_limit: 0.0          # Max steps (0 = unlimited)
  cost_limit: 0.0          # Max cost in dollars (0 = unlimited)
  mode: confirm            # confirm | yolo | interactive
  confirm_exit: true       # Ask for confirmation before exiting
```

#### Environment Configuration

```yaml
env:
  env:
    PAGER: cat             # Environment variables
    MANPAGER: cat
    LESS: -R
    PIP_PROGRESS_BAR: 'off'
    TQDM_DISABLE: '1'
  timeout: 3600            # Command timeout in seconds
```

#### Model Configuration

```yaml
model:
  model_class: amd_llm     # LLM backend
  model_name: claude-opus-4.5
  api_key: ""              # API key (or use env var)
  model_kwargs:
    temperature: 0.0       # Sampling temperature
    max_tokens: 16000      # Max output tokens
    reasoning:             # GPT-specific
      effort: high         # none | low | medium | high
    text:                  # GPT-specific
      verbosity: low       # low | high
```

#### Tools Configuration

```yaml
tools:
  profiling: false         # Enable profiling tool
  profiling_type: profiling # profiling | roofline | profiler_analyzer
  strategy_manager: true   # Enable strategy manager tool
```

## Unit Test Creation

The Unit Test Agent automatically finds or creates test commands for kernel validation and benchmarking.

### Workflow

1. **Discovery Phase:**
   - Reads repository README and documentation
   - Enumerates existing tests under `test/`
   - Finds benchmarks under `benchmark/`
   - Checks build configuration (e.g., `-DBUILD_BENCHMARK=ON`)

2. **Reuse Existing Tests:**
   - Prefers running existing benchmark executables
   - Uses appropriate flags (e.g., `--trials 20`)
   - Avoids creating new tests if suitable ones exist

3. **Create New Tests (if needed):**
   - Creates minimal correctness test with trusted reference
   - Creates performance benchmark with multiple iterations
   - Covers multiple dtypes, shapes, and edge cases
   - For fused kernels: tests both unfused and fused variants

4. **Output:**
   - Returns single executable command
   - Command must run correctness test first, then benchmark
   - Correctness failure must produce non-zero exit code

## Strategy-Based Optimization

The Strategy Manager provides systematic tracking and exploration of optimization strategies.

### Strategy Manager Tool

The `strategy_manager` tool provides commands for managing optimization strategies:

#### Available Commands

```bash
# Create strategy list with baseline and strategies
strategy_manager create --baseline-metrics "..." --strategies [...]

# View all strategies or specific strategy
strategy_manager show [--index N]

# Get next recommended strategy (auto-prioritizes high-priority)
strategy_manager next

# Mark strategy status with results
strategy_manager mark --index N --status exploring|successful|failed|partial --result "..." --details "..."

# Add new strategy
strategy_manager add --name "..." --description "..." --expected "..."

# Remove/skip strategy
strategy_manager remove --index N --method skip|delete

# Update strategy fields
strategy_manager update --index N [--name "..."] [--priority high|medium|low] [...]

# Add documentation note
strategy_manager note --text "..."

# Show statistics
strategy_manager summary
```

### Strategy File Format

Strategies are tracked in `.optimization_strategies.md`:

```markdown
# Optimization Strategy Exploration

## Baseline Performance
- Metrics: Bandwidth=45.2 GB/s, Latency=2.3ms, Occupancy=65%
- Log: logs/baseline_test.txt

## Strategy List

### 1. Memory Coalescing [HIGH PRIORITY]
**Status:** [successful]
**Target:** Memory bandwidth
**Expected:** +30% bandwidth improvement
**Result:** +21.1% bandwidth improvement (45.2 -> 54.7 GB/s)
**Details:** Improved memory access pattern by reordering thread indexing
**Log:** logs/strategy_1_test.txt

### 2. Shared Memory Optimization [MEDIUM PRIORITY]
**Status:** [exploring]
**Target:** Memory latency
**Expected:** -20% latency reduction
**Log:** logs/strategy_2_test.txt

### 3. Loop Unrolling [MEDIUM PRIORITY]
**Status:** [pending]
**Target:** Instruction throughput
**Expected:** +10% throughput

## Notes
- Strategy 1 (memory coalescing) showed best results
- Consider combining strategies 1 and 2
```

### Strategy Status Labels

- `[baseline]` - Original performance measurement
- `[pending]` - Not yet implemented
- `[exploring]` - Currently testing
- `[successful]` - Performance improved
- `[failed]` - Performance degraded
- `[partial]` - Some improvement but below expectation
- `[combined]` - Multiple strategies combined

### Optimization Workflow

```python
# 1. Create strategy list
strategy_manager create \
  --baseline-metrics "Bandwidth=45.2GB/s, Latency=2.3ms" \
  --strategies '[
    {"name": "Memory Coalescing", "description": "...", "expected": "+30% BW", "priority": "high"},
    {"name": "Shared Memory", "description": "...", "expected": "-20% latency", "priority": "medium"}
  ]'

# 2. Get next strategy (auto-selects high-priority first)
strategy_manager next
# Output: Strategy 1: Memory Coalescing [HIGH PRIORITY]

# 3. Mark as exploring
strategy_manager mark --index 1 --status exploring

# 4. Implement optimization and test
test_perf --description "memory coalescing"

# 5. Update with results
strategy_manager mark --index 1 --status successful \
  --result "+21.1% improvement" \
  --details "Improved memory access pattern"

# 6. Repeat for next strategy
strategy_manager next
```

## Parallel Scaling & Patch Selection

Run multiple optimization agents in parallel and automatically select the best result.

### Parallel Agent

```bash
# Run 4 parallel agents
mini --config mini_kernel_strategy_list.yaml \
     --num-parallel 4 \
     --repo /path/to/kernel/repo \
     --task "Optimize block_reduce kernel" \
     --gpu-ids 0,1,2,3 \
     --metric "Bandwidth in GB/s (higher is better)"
```

### Workflow

1. **Parallel Execution:**
   - Creates isolated git worktrees for each agent
   - Each agent optimizes independently
   - Patches saved to `patches/parallel_N/` directories
   - Test results saved alongside patches

2. **Automatic Selection:**
   - Launches `SelectPatchAgent` after parallel runs complete
   - Agent analyzes all patches and test results
   - Computes speedup based on user-provided metric
   - Ensures same number of metric points between baseline and patch
   - Saves selection to `best_results.json`

3. **Output Structure:**
   ```
   patches/
   ├── parallel_0/
   │   ├── patch_0.patch (baseline)
   │   ├── patch_0_test.txt
   │   ├── patch_1.patch
   │   ├── patch_1_test.txt
   │   └── agent_0.log
   ├── parallel_1/
   │   ├── patch_0.patch (baseline)
   │   ├── patch_1.patch
   │   └── agent_1.log
   ├── parallel_2/
   │   └── ...
   ├── parallel_3/
   │   └── ...
   ├── best_results.json
   └── select_agent.log
   ```

### Defining Custom Metrics

```bash
--metric "Latency in milliseconds (lower is better)"
--metric "Bandwidth in GB/s (higher is better)"
--metric "FLOPS in TFLOPS (higher is better)"
```

The SelectPatchAgent will:
- Parse the metric from test output logs
- Compute speedup based on whether higher/lower is better
- Ensure equal number of data points between baseline and patch
- Report the best patch with calculated speedup

### GPU Isolation

```bash
# Assign specific GPUs to each agent
--gpu-ids 0,1,2,3

# Each agent gets HIP_VISIBLE_DEVICES set to its assigned GPU
# Agent 0: HIP_VISIBLE_DEVICES=0
# Agent 1: HIP_VISIBLE_DEVICES=1
# etc.
```

## Profiling Tools

GEAK integrates rocprof for deep GPU kernel profiling and bottleneck identification.

### Profiling Tool

```yaml
# Enable in config
tools:
  profiling: true
  profiling_type: profiling  # profiling | roofline | profiler_analyzer
```

### Profiling Types

#### 1. **profiling** (Default)
Full profiling analysis including:
- Hardware information (GPU model, CU count, memory specs)
- Top kernels by execution time
- Performance utilization and bottlenecks
- Compute unit instruction mix
- L1/L2 cache analysis
- Wavefront statistics
- Roofline metrics

#### 2. **roofline**
Focused roofline analysis:
- HBM bandwidth utilization
- Compute utilization (FLOPs, IOPs)
- Arithmetic intensity

#### 3. **profiler_analyzer**
LLM-powered profiling analysis:
- Uses LLM to interpret profiling data
- Provides high-level optimization recommendations
- Identifies key bottlenecks automatically

## Agent Types

### 1. DefaultAgent
Base agent with patch saving and test execution.

### 2. StrategyAgent (extends InteractiveAgent)
Agent with strategy management and UI notifications.

### 3. UnitTestAgent
Specialized agent for finding/creating test commands.

### 4. SelectPatchAgent
Specialized agent for analyzing and selecting best patches from parallel runs.

### 5. ParallelAgent
Orchestrates multiple agents running in parallel with git worktrees.

## Custom Tool Integration

```python
from minisweagent.tools.registry import ToolRegistry

# Register custom tool
@ToolRegistry.register("my_custom_tool")
def my_tool(param1: str, param2: int):
    """Custom tool description."""
    # Tool implementation
    return {"output": "...", "returncode": 0}
```

## Best Practices

### 1. Strategy-Based Optimization
- Always measure baseline first
- Profile to identify bottlenecks before creating strategies
- Mark high-priority strategies for automatic selection
- Try strategies individually before combining
- Document results in strategy file

### 2. Parallel Scaling
- Use parallel mode for exploration of diverse approaches
- Provide clear metric definition for automatic selection
- Assign GPUs to avoid resource contention
- Review select_agent.log to understand selection reasoning

### 3. Profiling
- Profile after each major optimization
- Focus on bottlenecks (memory vs compute bound)
- Use roofline analysis for quick bottleneck identification
- Use full profiling for detailed optimization guidance

### 4. Configuration
- Start with existing configs and customize
- Use `mode: yolo` for parallel agents (no confirmation prompts)
- Set appropriate timeouts for long-running benchmarks

## MCP Tool Servers

GEAK includes standalone MCP (Model Context Protocol) servers that the agent can call for specialized tasks:

| Server | Purpose |
|--------|---------|
| `automated-test-discovery` | Content-based test/benchmark discovery for any project |
| `kernel-profiler` | Hardware-level GPU profiling via rocprof-compute |
| `kernel-evolve` | LLM-guided kernel optimization (generate/mutate/crossover) |
| `kernel-ercs` | Kernel evaluation, reflection, compatibility checks |
| `metrix-mcp` | AMD Metrix API wrapper for GPU metrics |
| `openevolve-mcp` | OpenEvolve evolutionary optimizer |
| `mcp-client` | JSON-RPC 2.0 client for MCP communication |

Install MCP servers individually: `pip install -e mcp_tools/<server-name>/`

## Docker Support

```bash
# Auto-detect environment (uses Docker if no local GPU)
mini --runtime auto --task "Optimize kernel at /path/to/kernel.py"

# Force Docker with specific image
mini --runtime docker --docker-image lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x \
  --workspace /path/to/kernels --task "..."
```

See [RUNTIME_ENV.md](RUNTIME_ENV.md) and [RUNTIME_QUICKSTART.md](RUNTIME_QUICKSTART.md) for details.

## Architecture

```
GEAK Agent (unified)
├── src/minisweagent/          # Core agent framework
│   ├── agents/                # 7 agent types (default, interactive, strategy,
│   │                          #   parallel, select_patch, unit_test, textual)
│   ├── tools/                 # Built-in tools (bash, editor, test_perf,
│   │                          #   profiling, strategy_manager, discovery, submit)
│   ├── config/                # YAML configs (layered: base → template → user)
│   ├── environments/          # Execution environments (local, docker, singularity)
│   ├── models/                # LLM backends (amd_llm, anthropic, litellm, etc.)
│   ├── optimizer/             # Unified optimizer (OpenEvolve + Autotune)
│   ├── mcp_tools/             # MetrixTool for AMD GPU profiling
│   ├── benchmark.py           # Standardized benchmarking framework
│   ├── kernel_profile.py      # CLI for hardware-level GPU profiling
│   └── runtime_env.py         # Runtime environment detection (local/Docker)
├── mcp_tools/                 # 6 standalone MCP servers + client
├── knowledge_base/                 # GPU optimization strategies database (50+)
├── eval_suite/                # Regression suite (10 AITER kernels)
├── examples/                  # Example kernels
└── Dockerfile                 # Docker image with all tools
```

## License

MIT License - see LICENSE.md for details

