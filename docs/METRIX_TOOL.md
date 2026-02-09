# MetrixTool - GPU Kernel Profiler

GPU kernel profiling tool with hardware metrics and bottleneck analysis for LLM-driven optimization.

## Concept

MetrixTool provides detailed GPU performance metrics for kernel optimization by:
1. **Profiling kernels** - Using AMD ROCm Metrix profiler for hardware-level metrics
2. **Classifying bottlenecks** - Identifying memory, compute, latency, or LDS limitations
3. **Generating observations** - Providing factual insights for LLM-driven decision making

### Design Philosophy

**Complete information for LLM agents** - Instead of prescriptive suggestions, MetrixTool provides:
- All available hardware metrics from the profiler
- GPU specifications (architecture, bandwidth, compute capability)
- Factual observations about bottlenecks
- Let the LLM agent make optimization decisions based on complete context

This makes it suitable for **autonomous optimization agents** that need comprehensive data.

## Features

### 1. Hardware-Level Profiling

MetrixTool uses AMD's `metrix` profiler to collect detailed GPU metrics:

**Memory Metrics:**
| Metric | Description |
|--------|-------------|
| `memory.hbm_bandwidth_utilization` | HBM bandwidth usage (%) |
| `memory.l2_hit_rate` | L2 cache hit rate (%) |
| `memory.l1_hit_rate` | L1 cache hit rate (%) |
| `memory.read_coalescing_efficiency` | Memory read coalescing (%) |
| `memory.write_coalescing_efficiency` | Memory write coalescing (%) |
| `memory.global_load_efficiency` | Global load efficiency (%) |
| `memory.global_store_efficiency` | Global store efficiency (%) |

**Compute Metrics:**
| Metric | Description |
|--------|-------------|
| `compute.arithmetic_intensity` | FLOPs per byte transferred |
| `compute.total_flops` | Total floating-point operations |
| `compute.fp32_tflops` | FP32 TFLOPS achieved |

**LDS Metrics:**
| Metric | Description |
|--------|-------------|
| `lds.bank_conflicts_per_inst` | LDS bank conflicts per instruction |
| `lds.utilization` | LDS memory utilization (%) |

**Duration:**
| Metric | Description |
|--------|-------------|
| `duration_us` | Kernel execution time (microseconds) |

### 2. Profile Modes

**Default (Memory Profile):**
- 12 comprehensive metrics
- 2 profiling passes (~24s)
- Ideal for optimization work

**Quick Profile (`--quick`):**
- 3 basic metrics (duration, HBM utilization, L2 hit rate)
- 1 profiling pass (~16s)
- Fast overview for initial analysis

### 3. Bottleneck Classification

Automatic classification based on hardware metrics:

| Classification | Criteria |
|----------------|----------|
| **Memory-bound** | HBM bandwidth > 30%, compute intensity low |
| **Compute-bound** | HBM bandwidth < 5%, L2 hit rate > 80% |
| **Latency-bound** | Short duration (< 10μs), low resource utilization |
| **LDS-bound** | High LDS bank conflicts (> 2 per instruction) |
| **Balanced** | No clear bottleneck, relatively efficient |

### 4. GPU Auto-Detection

Automatically detects GPU specifications using Metrix's built-in device info:
- Vendor (AMD/NVIDIA)
- Model name (e.g., "AMD Instinct MI300X OAM")
- Architecture (e.g., "gfx942")
- Compute units
- Peak HBM bandwidth (GB/s)
- Peak L2 bandwidth (GB/s)
- LDS size per CU (KB)
- Peak FP32 TFLOPS

### 5. Multi-GPU Support

Profile the same kernel across multiple GPUs for comparison:
- Supports profiling on GPU devices 0, 1, 2, 3, etc.
- Returns consistent results structure per GPU
- Uses `HIP_VISIBLE_DEVICES` internally to isolate GPU access

## Usage

### Command-Line Interface

Basic profiling:

```bash
# Profile a kernel (default: all kernels, memory profile)
python kernel-profile 'python3 /path/to/kernel.py --profile'

# Specify GPU device
python kernel-profile 'python3 kernel.py --profile' --gpu-devices 3

# Auto-select main kernel only
python kernel-profile 'python3 kernel.py --profile' --auto-select

# Quick profile (faster, fewer metrics)
python kernel-profile 'python3 kernel.py --profile' --quick

# Filter specific kernel by name
python kernel-profile 'python3 kernel.py --profile' --filter '*topk*'

# More replays for better statistics
python kernel-profile 'python3 kernel.py --profile' --replays 5

# Profile on multiple GPUs
python kernel-profile 'python3 kernel.py --profile' --gpu-devices 0,1,2
```

### Programmatic Usage

#### Single GPU

```python
from minisweagent.mcp_tools.metrix import MetrixTool

# Initialize with GPU device
tool = MetrixTool(gpu_devices="3")

# Profile kernel
result = tool.profile(
    command="python3 kernel.py --profile",
    num_replays=3,
    kernel_filter="",  # Empty = all kernels
    auto_select=False,  # False = show all, True = main kernel only
    quick=False  # False = full profile, True = quick profile
)

# Access results (always a list, even for single GPU)
for gpu_result in result["results"]:
    device_id = gpu_result["device_id"]
    gpu_info = gpu_result["gpu_info"]
    
    if gpu_info.get("detected"):
        print(f"GPU {device_id}: {gpu_info['model']}")
        print(f"Architecture: {gpu_info['architecture']}")
        print(f"Peak HBM: {gpu_info['peak_hbm_bandwidth_gb_s']:.1f} GB/s")
        print(f"Peak FP32: {gpu_info['peak_fp32_tflops']:.1f} TFLOPS")
    
    # Process each kernel
    for kernel in gpu_result["kernels"]:
        print(f"\nKernel: {kernel['name']}")
        print(f"Duration: {kernel['duration_us']:.2f} μs")
        print(f"Bottleneck: {kernel['bottleneck']}")
        
        # Observations (factual, not prescriptive)
        for obs in kernel['observations']:
            print(f"  - {obs}")
        
        # All hardware metrics
        for metric_name, metric_value in kernel['metrics'].items():
            print(f"  {metric_name}: {metric_value:.2f}")
```

#### Multiple GPUs

```python
from minisweagent.mcp_tools.metrix import MetrixTool

# Profile across multiple GPUs
tool = MetrixTool(gpu_devices=["0", "1", "2"])

result = tool.profile(
    command="python3 kernel.py --profile",
    num_replays=3
)

# Compare performance across GPUs
for gpu_result in result["results"]:
    device_id = gpu_result["device_id"]
    print(f"\n=== GPU {device_id} ===")
    
    for kernel in gpu_result["kernels"]:
        duration = kernel["duration_us"]
        hbm_util = kernel["metrics"].get("memory.hbm_bandwidth_utilization", 0)
        print(f"{kernel['name']}: {duration:.2f} μs, {hbm_util:.1f}% HBM")
```

### MCP Tool Integration

MetrixTool is also available as an MCP (Model Context Protocol) tool for agent integration:

```python
# Get tool schema
tool_def = MetrixTool.get_tool_definition()
print(tool_def)  # JSON schema for MCP server

# Execute via MCP interface
tool = MetrixTool()
result = tool.execute(
    command="python3 kernel.py --profile",
    gpu_devices="3",
    num_replays=3,
    kernel_filter="*topk*",
    auto_select=False,
    quick=False
)
```

## Examples

### Example 1: Profile TopK Kernel

```bash
cd /home/sdubagun/work/repos/GEAK-agent
python kernel-profile \
    'python3 /home/sdubagun/work/repos/AIG-Eval/tasks/geak_eval/topk/kernel.py --profile' \
    --gpu-devices 3 \
    --filter '*topk*'
```

**Sample Output:**
```
=== GPU 3: AMD Instinct MI300X OAM ===
Architecture: gfx942
Compute Units: 304
Peak HBM Bandwidth: 5300.0 GB/s
Peak FP32 TFLOPS: 163.4

Kernel: topk_stage1_kernel

Duration: 124.58 μs

Bottleneck: memory-bound

Observations:
  - HBM bandwidth utilization is 45.2%, indicating moderate memory pressure
  - L2 cache hit rate is 35.8%, suggesting poor data locality
  - Read coalescing efficiency is 72.3%, room for improvement
  - Achieved 0.8% of peak HBM bandwidth (42.4 GB/s of 5300.0 GB/s)
  - Arithmetic intensity is 0.15 FLOPs/byte (low compute per memory access)

Metrics:
  duration_us: 124.58
  memory.hbm_bandwidth_utilization: 45.2
  memory.l2_hit_rate: 35.8
  memory.l1_hit_rate: 78.9
  memory.read_coalescing_efficiency: 72.3
  memory.write_coalescing_efficiency: 88.1
  memory.global_load_efficiency: 81.2
  memory.global_store_efficiency: 92.4
  compute.arithmetic_intensity: 0.15
  compute.total_flops: 15728640.0
  compute.fp32_tflops: 0.126
  lds.bank_conflicts_per_inst: 0.02
```

### Example 2: Quick Profile for Multiple Kernels

```bash
python kernel-profile \
    'python3 /path/to/multi_kernel.py --profile' \
    --quick
```

Shows all kernels with 3 basic metrics (fast overview).

### Example 3: Compare Across GPUs

```bash
python kernel-profile \
    'python3 kernel.py --profile' \
    --gpu-devices 0,1,2,3
```

Profiles the same kernel on 4 different GPUs for performance comparison.

### Example 4: Focus on Main Kernel

```bash
python kernel-profile \
    'python3 kernel.py --profile' \
    --auto-select
```

Automatically selects and profiles only the longest-running kernel.

## Output Structure

### JSON Result Format

```python
{
    "results": [  # List of GPU results (1 per GPU device)
        {
            "device_id": "3",
            "gpu_info": {
                "detected": True,
                "vendor": "AMD",
                "model": "AMD Instinct MI300X OAM",
                "architecture": "gfx942",
                "compute_units": 304,
                "peak_hbm_bandwidth_gb_s": 5300.0,
                "peak_l2_bandwidth_gb_s": 18432.0,
                "lds_size_per_cu_kb": 64.0,
                "peak_fp32_tflops": 163.4
            },
            "kernels": [  # List of profiled kernels
                {
                    "name": "kernel_name",
                    "duration_us": 124.58,
                    "bottleneck": "memory-bound",
                    "observations": [
                        "HBM bandwidth utilization is 45.2%...",
                        "L2 cache hit rate is 35.8%..."
                    ],
                    "metrics": {
                        "duration_us": 124.58,
                        "memory.hbm_bandwidth_utilization": 45.2,
                        "memory.l2_hit_rate": 35.8,
                        # ... 9 more metrics in full profile
                    }
                }
            ]
        }
    ]
}
```

## Requirements

### Environment

MetrixTool requires:
- AMD ROCm stack with `metrix` profiler installed
- Docker container: `minikernel_sdubagun` (recommended)
- Python 3.8+

### Docker Setup

The tool is designed to run inside a Docker container with the ROCm stack:

```bash
# Run inside Docker container
docker exec -it minikernel_sdubagun bash

# Verify metrix is available
python3 -c "from metrix import Metrix; print('Metrix available!')"

# Run profiling
cd /home/sdubagun/work/repos/GEAK-agent
python3 kernel-profile 'python3 kernel.py --profile'
```

## Integration with GEAK Agent

MetrixTool is a core component of the GEAK (GPU Evolutionary Agent for Kernels) system:

1. **Discovery Pipeline** (`minisweagent.tools.discovery`) finds test and benchmark files
2. **MetrixTool** (`minisweagent.mcp_tools.metrix`) profiles kernel performance
3. **LLM Agent** uses metrics + observations to propose optimizations
4. **Kernel-ERCS** evaluates and reflects on optimization attempts

The tool is designed to provide complete, factual information rather than prescriptive advice, allowing the LLM to reason about optimizations based on full context.

## Advanced Features

### Kernel Filtering

Use wildcard patterns to profile specific kernels:

```python
# Profile only kernels matching pattern
result = tool.profile(
    command="python3 kernel.py --profile",
    kernel_filter="*attention*"
)
```

### Auto-Selection Logic

When `auto_select=True`, MetrixTool selects the "main" kernel using:
- **Longest duration** - Assumes the kernel with the highest execution time is the primary optimization target

This is useful when your script launches multiple small utility kernels but you want to focus on the main computational kernel.

### Statistics Aggregation

The `num_replays` parameter controls statistical robustness:
- Metrix runs the kernel multiple times
- Reports `Statistics` objects with `.avg`, `.min`, `.max`, `.std`
- Higher replays = better statistics, longer profiling time
- Default: 3 replays (good balance)

### Profile Level Tradeoffs

| Aspect | Quick (`--quick`) | Memory (default) |
|--------|-------------------|------------------|
| Metrics | 3 | 12 |
| Passes | 1 | 2 |
| Duration | ~16s | ~24s |
| Use Case | Fast overview | Optimization work |

## Troubleshooting

### "Module metrix not found"

```bash
# Must run inside Docker container
docker exec -it minikernel_sdubagun bash
cd /home/sdubagun/work/repos/GEAK-agent
python3 kernel-profile 'python3 kernel.py --profile'
```

### "Missing expected metrics"

This usually means:
- Using `--quick` mode (only 3 metrics available)
- Or the profiler didn't capture all expected metrics

Solution: Use full profile (remove `--quick` flag).

### All Kernels Classified as "Balanced"

This can happen if:
- Kernel is truly balanced (no clear bottleneck)
- Quick mode doesn't provide enough metrics for classification
- Thresholds need adjustment for your specific GPU/workload

Solution: Use full profile for comprehensive bottleneck analysis.

## Testing

Comprehensive test suite at `tests/tools/test_metrix.py`:

```bash
# Run tests
cd /home/sdubagun/work/repos/GEAK-agent
pytest tests/tools/test_metrix.py -v

# Test coverage includes:
# - Initialization and GPU detection
# - Profile execution and return format
# - Multi-GPU profiling
# - Bottleneck classification (all 5 types)
# - Observation generation
# - Metric validation
# - Auto-selection logic
# - MCP tool interface
```

17 test cases covering ~85% of the codebase.

## Logging

MetrixTool uses Python's `logging` module for visibility:

```python
import logging

# Enable INFO logging
logging.basicConfig(level=logging.INFO)

# Enable DEBUG logging for detailed trace
logging.basicConfig(level=logging.DEBUG)

# Logs include:
# - Initialization with GPU devices
# - Profiling start/end
# - Per-GPU profiling runs
# - Kernel selection (auto-select)
# - Metric validation
# - Bottleneck classification
# - Observation generation
```

## API Reference

### MetrixTool Class

```python
class MetrixTool:
    """GPU kernel profiler with hardware metrics and bottleneck analysis."""
    
    def __init__(self, gpu_devices: Union[str, List[str]] = "0"):
        """Initialize with GPU device(s) to profile on."""
    
    def profile(
        self,
        command: str,
        num_replays: int = 3,
        kernel_filter: str = "",
        auto_select: bool = False,
        quick: bool = False
    ) -> Dict[str, List[Dict]]:
        """
        Profile GPU kernels and return hardware metrics.
        
        Returns:
            {
                "results": [
                    {
                        "device_id": str,
                        "gpu_info": {...},
                        "kernels": [...]
                    }
                ]
            }
        """
    
    @staticmethod
    def get_tool_definition() -> Dict:
        """Get MCP tool schema."""
    
    def execute(self, **kwargs) -> Dict:
        """Execute via MCP interface."""
```

## See Also

- [Discovery Pipeline](./DISCOVERY_PIPELINE.md) - Automated test and benchmark discovery
- [Examples](../examples/README.md) - Sample scripts
- [Tests](../tests/README.md) - Unit test suite
