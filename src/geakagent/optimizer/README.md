# Optimizer

Compact kernel optimization interface. Wraps existing optimizers (OpenEvolve, AutoTune) with unified API.

## Quick Start

```python
from geakagent.optimizer import optimize_kernel, OptimizerType

# Optimize using OpenEvolve (default)
result = optimize_kernel(
    kernel_code=my_kernel,
    kernel_path="kernel.py",
    optimizer=OptimizerType.OPENEVOLVE,
    max_iterations=50
)

print(f"Speedup: {result.metrics['speedup']}x")
```

## Architecture

```
optimizer/
├── core.py                    # Main API (calls MCP tools)
└── README.md
```

## Adding New Optimizers

1. Add enum to `OptimizerType` in `core.py`
2. Add `_optimize_with_<name>()` function
3. Update router in `optimize_kernel()`

## Usage

```python
from geakagent.optimizer import optimize_kernel, OptimizerType

# Simple - auto-select optimizer
result = optimize_kernel(
    kernel_code=my_kernel,
    bottleneck="latency",
    target_speedup=2.0
)

# Explicit optimizer
result = optimize_kernel(
    kernel_code=my_kernel,
    optimizer=OptimizerType.OPENEVOLVE,
    bottleneck="memory"
)

print(result.optimized_code)
print(f"Speedup: {result.metrics['speedup']}x")
```

## Optimizers

- **OpenEvolve** - LLM-guided optimization (via `mcp_tools/openevolve-mcp`)
- **AutoTune** - Parameter search (future)

## Structure

```
optimizer/
├── __init__.py    # Exports
└── core.py        # Implementation (calls mcp_tools/openevolve-mcp)
```

Clean, compact, modular.
