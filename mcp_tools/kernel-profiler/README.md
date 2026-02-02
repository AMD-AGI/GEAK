# Kernel Profiler MCP Server

Hardware-level GPU kernel profiling with rocprof-compute for AMD GPUs.

## Tools

| Tool | Description |
|------|-------------|
| `profile_kernel` | Profile kernel and identify bottleneck type |
| `benchmark_kernel` | Quick latency benchmark (no counters) |
| `get_roofline_analysis` | Compute roofline model position |
| `get_bottleneck_suggestions` | Get optimization strategies for bottleneck |

## Installation

```bash
cd kernel-profiler
pip install -e .
```

## Usage

### With Cursor

```json
{
  "name": "kernel-profiler",
  "command": "kernel-profiler"
}
```

### Standalone

```bash
kernel-profiler
```

## Tool Details

### profile_kernel

Profiles a kernel file and identifies the performance bottleneck.

```python
profile_kernel(
    kernel_file="/path/to/kernel.py",
    function_name="run_baseline",
    gpu_device="0",
    warmup_iters=100,
    profile_iters=100
)
# Returns: {latency_us, bottleneck, suggestions, ...}
```

### benchmark_kernel

Quick latency measurement without full profiling overhead.

### get_roofline_analysis

Computes roofline model position to determine if kernel is compute-bound or memory-bound.

### get_bottleneck_suggestions

Returns optimization strategies and code hints for a specific bottleneck type.

## Requirements

- Docker with AMD GPU support (ROCm)
- `/dev/kfd` and `/dev/dri` device access

## License

MIT License
