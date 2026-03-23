# Kernel Evolve MCP Server

Evolutionary GPU kernel optimization using LLM-guided mutation and crossover.
This is the OpenEvolve brain as an MCP server.

## Tools

| Tool | Description |
|------|-------------|
| `generate_optimization` | LLM generates optimized kernel variant |
| `mutate_kernel` | LLM mutates an existing optimization |
| `crossover_kernels` | LLM combines two kernel optimizations |
| `get_optimization_strategies` | Get strategies for a bottleneck type |
| `suggest_kernel_params` | Get recommended params for kernel type |

## Installation

```bash
cd kernel-evolve
pip install -e .
```

## Usage

### With Cursor

```json
{
  "name": "kernel-evolve",
  "command": "kernel-evolve",
  "env": {
    "AMD_LLM_API_KEY": "your-amd-gateway-key"
  }
}
```

### Standalone

```bash
export AMD_LLM_API_KEY="your-key"
kernel-evolve
```

## Tool Details

### generate_optimization

Generates an optimized kernel targeting a specific bottleneck.

```python
generate_optimization(
    kernel_code="...",
    bottleneck="latency",  # latency, memory, compute, lds, balanced
    strategy="hip_graph"   # optional, auto-selected if not provided
)
# Returns: {optimized_code, strategy_used, ...}
```

### mutate_kernel

Creates a variation of an existing optimization.

```python
mutate_kernel(
    kernel_code="...",
    mutation_type="parameter",  # parameter, algorithm, hybrid
    latency_us=50.0,
    speedup=1.5
)
# Returns: {mutated_code, mutation_type, ...}
```

### crossover_kernels

Combines two optimizations into a hybrid.

```python
crossover_kernels(
    kernel1="...",
    kernel2="...",
    speedup1=1.5,
    speedup2=1.3
)
# Returns: {hybrid_code, ...}
```

### get_optimization_strategies

Returns prioritized strategies for a bottleneck type.

### suggest_kernel_params

Suggests block sizes, num_warps, etc. for a kernel type.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AMD_LLM_API_KEY` | AMD LLM Gateway API key |
| `LLM_GATEWAY_KEY` | Alternative AMD gateway key |
| `OPENAI_API_KEY` | OpenAI API key (fallback) |

## Evolutionary Optimization Workflow

1. **Profile** kernel to identify bottleneck (use kernel-profiler MCP)
2. **Generate** initial optimization with `generate_optimization`
3. **Evaluate** the optimization (run tests, measure speedup)
4. **Mutate** good solutions with `mutate_kernel`
5. **Crossover** two good solutions with `crossover_kernels`
6. **Repeat** until target speedup achieved

## License

MIT License
