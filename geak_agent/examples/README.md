# GEAK Agent Examples

Example scripts for kernel profiling and resolving kernel specs from URLs. Run these inside the container (after `./scripts/run-docker.sh`; you land in `/workspace`).

## Discovery CLI

```bash
# Discover tests and benchmarks
python -m geak_agent.cli /path/to/kernel.py --discover-only

# With explicit test/bench commands
python -m geak_agent.cli /path/to/kernel.py --test "pytest test.py" --bench "python bench.py"
```

## Python API (Discovery)

```python
from geak_agent.mcp_tools.discovery import discover

result = discover(workspace="/path/to/project")
print(f"Found {len(result.tests)} tests")
```

---

## resolve_kernel_url.py

Resolve a kernel spec to a local path. If the spec is a GitHub URL, the repo is cloned to a temp dir and the file path is returned. Optional fragment `#L106` or `#L106-L108` is parsed; when present, the script also prints `line_number` and `kernel_name` (the function containing that line). Requires `git` and network access for GitHub URLs.

### Basic Usage

```bash
# Local path (returned unchanged)
python geak_agent/examples/resolve_kernel_url.py /path/to/kernel.py
# or run as executable (from repo root):
./geak_agent/examples/resolve_kernel_url.py /path/to/kernel.py

# GitHub blob URL (clones repo to temp, returns local path)
python geak_agent/examples/resolve_kernel_url.py 'https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/moe/moe_op_gelu.py'

# With optional line number (output includes line_number and kernel_name for use with kernel-profile --filter)
python geak_agent/examples/resolve_kernel_url.py 'https://github.com/.../rope.py#L106'
```

---

## profile_kernel.py

Profile a GPU kernel and get optimization suggestions.

### Basic Usage

```bash
# Profile a kernel
python geak_agent/examples/profile_kernel.py 'python3 /path/to/kernel.py --profile'

# Specify GPU device
python geak_agent/examples/profile_kernel.py 'python3 kernel.py --profile' --gpu-devices 3

# Filter specific kernel
python geak_agent/examples/profile_kernel.py 'python3 kernel.py --profile' --filter '*topk*'

# More replays for better statistics
python geak_agent/examples/profile_kernel.py 'python3 kernel.py --profile' --replays 5

# Auto-select main kernel only (default shows all kernels)
python geak_agent/examples/profile_kernel.py 'python3 kernel.py --profile' --auto-select

# Quick profile for speed (3 metrics, 1 pass vs 12 metrics, 2 passes)
python geak_agent/examples/profile_kernel.py 'python3 kernel.py --profile' --quick

# Profile on multiple GPUs (compare performance across devices)
python geak_agent/examples/profile_kernel.py 'python3 kernel.py --profile' --gpu-devices 0,1,2
```

### Example with external kernels

```bash
# Profile a kernel (paths are examples)
cd /workspace
python geak_agent/examples/profile_kernel.py \
    'python3 /path/to/topk/kernel.py --profile' \
    --gpu-devices 3 \
    --filter '*topk*'

python geak_agent/examples/profile_kernel.py \
    'python3 /path/to/gemm/kernel.py --profile' \
    --gpu-devices 3
```

### Output

The script displays (by default):
- **GPU information** (vendor, model, architecture) if available
- **All kernels** with full profiling (12 metrics, 2 passes)
- Bottleneck classifications: memory, compute, latency, LDS, balanced
- Factual observations (not prescriptive suggestions - LLM makes optimization decisions)
- Comprehensive hardware metrics (HBM bandwidth, cache hit rates, coalescing efficiency, LDS conflicts, etc.)
- Use `--auto-select` to show only the main kernel
- Use `--quick` for fast profiling (3 metrics, 1 pass)

### Profile Modes

- **Default (memory profile)**: 12 metrics, 2 passes (~24s)
  - Comprehensive memory system analysis
  - Detects coalescing issues, LDS conflicts, cache behavior
- **Quick mode (`--quick`)**: 3 metrics, 1 pass (~16s)
  - Fast overview for basic memory and latency analysis
  - HBM utilization + L2 hit rate only

## Programmatic Usage

For integration into your own scripts:

### Single or Multiple GPUs
```python
from geak_agent.mcp_tools.metrix import MetrixTool

# Single GPU
tool = MetrixTool(gpu_devices="3")
# Or multiple GPUs
# tool = MetrixTool(gpu_devices=["0", "1", "2"])

# Full profile (default)
result = tool.profile(
    command="python3 kernel.py --profile",
    num_replays=3,
    kernel_filter="*topk*",
    auto_select=False,  # Default: show all kernels
    quick=False  # Default: full (memory) profile
)

# Result always has consistent structure: {"results": [...]}
# Each item in results list represents one GPU
for gpu_result in result["results"]:
    device_id = gpu_result["device_id"]
    gpu_info = gpu_result["gpu_info"]
    
    if gpu_info.get("detected"):
        print(f"\n=== GPU {device_id}: {gpu_info['model']} ===")
    
    # Access kernels for this GPU
    for kernel in gpu_result["kernels"]:
        print(f"Kernel: {kernel['name']}")
        print(f"Duration: {kernel['duration_us']:.2f} μs")
        print(f"Bottleneck: {kernel['bottleneck']}")
        print(f"Observations: {kernel['observations']}")  # Factual, not prescriptive
        print(f"Metrics: {len(kernel['metrics'])} total")
        for name, value in kernel['metrics'].items():
            print(f"  {name}: {value:.2f}")
```
