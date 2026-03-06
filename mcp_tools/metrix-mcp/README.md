# Metrix MCP - GPU Kernel Profiler

GPU kernel profiling using AMD Metrix with hardware metrics and bottleneck analysis.

## Features

- **Hardware-Level Metrics**: HBM bandwidth, L1/L2 cache, coalescing efficiency, LDS conflicts
- **Bottleneck Classification**: Automatic classification (memory-bound, compute-bound, latency-bound, LDS-bound, balanced)
- **Multi-GPU Support**: Profile across multiple GPUs simultaneously
- **Dual Mode**: Use as direct Python API or MCP protocol server
- **GPU Auto-Detection**: Automatically detects GPU specs (arch, bandwidth, compute units, etc.)

## Installation

```bash
# Install package
pip install -e mcp_tools/metrix-mcp/

# Requires AMD Metrix profiler
pip install metrix
```

## Usage

### Mode 1: Direct Python API (Simple & Fast)

```python
from metrix_mcp import MetrixTool

# Initialize
tool = MetrixTool(gpu_devices="0")

# Profile kernel
result = tool.profile(
    command="python3 kernel.py",
    num_replays=3,
    quick=False  # Use full memory profile
)

# Access results
for device_result in result["results"]:
    print(f"GPU {device_result['device_id']}: {device_result['gpu_info']['model']}")
    for kernel in device_result["kernels"]:
        print(f"  Kernel: {kernel['name']}")
        print(f"  Bottleneck: {kernel['bottleneck']}")
        print(f"  Duration: {kernel['duration_us']:.2f} μs")
        print(f"  HBM Utilization: {kernel['metrics']['memory.hbm_bandwidth_utilization']*100:.1f}%")
```

### Mode 2: MCP Protocol (Robust & Isolated)

```python
import asyncio
from mcp_client import call_mcp_tool

async def main():
    result = await call_mcp_tool(
        server_name="metrix-mcp",
        tool_name="profile_kernel",
        arguments={
            "command": "python3 kernel.py",
            "quick": False,
            "auto_select": True,
            "gpu_devices": "0"
        }
    )
    
    if result["success"]:
        print(f"Profiled {len(result['results'][0]['kernels'])} kernel(s)")

asyncio.run(main())
```

### Run as Standalone Server

```bash
# Start MCP server
python3 -m metrix_mcp.server

# Or use script
metrix-mcp
```

## Profile Modes

### Quick Profile (fast)
- 3 metrics: `duration_us`, `hbm_bandwidth_utilization`, `l2_hit_rate`
- 1 profiling pass (~16 seconds)
- Good for quick checks

### Memory Profile (comprehensive, default)
- 12 metrics: duration, HBM bandwidth, L1/L2 hit rates, coalescing, LDS conflicts
- 2 profiling passes (~24 seconds)
- Recommended for optimization

## Bottleneck Classification

| Classification | Criteria |
|----------------|----------|
| **memory-bound** | HBM bandwidth > 30%, compute intensity low |
| **compute-bound** | HBM bandwidth < 5%, L2 hit rate > 80% |
| **latency-bound** | Duration < 10μs, low resource utilization |
| **lds-bound** | LDS bank conflicts > 2 per instruction |
| **balanced** | No clear bottleneck, relatively efficient |

## Hardware Metrics

### Memory Metrics
- `memory.hbm_bandwidth_utilization` - HBM bandwidth usage (%)
- `memory.l1_hit_rate` - L1 cache hit rate (%)
- `memory.l2_hit_rate` - L2 cache hit rate (%)
- `memory.coalescing_efficiency` - Memory coalescing (%)
- `memory.global_load_efficiency` - Global load efficiency (%)
- `memory.global_store_efficiency` - Global store efficiency (%)

### Compute Metrics
- `compute.arithmetic_intensity` - FLOPs per byte transferred
- `compute.fp32_tflops` - FP32 TFLOPS achieved

### LDS Metrics
- `lds.bank_conflicts_per_inst` - LDS bank conflicts per instruction
- `lds.utilization` - LDS memory utilization (%)

### Duration
- `duration_us` - Kernel execution time (microseconds)

## Multi-GPU Profiling

```python
from metrix_mcp import MetrixTool

# Profile on multiple GPUs
tool = MetrixTool(gpu_devices=["0", "1", "2"])

result = tool.profile(
    command="python3 kernel.py",
    auto_select=True
)

# Results contain data for each GPU
for device_result in result["results"]:
    device_id = device_result["device_id"]
    kernels = device_result["kernels"]
    print(f"GPU {device_id}: {len(kernels)} kernels profiled")
```

## Integration with GEAK Agent

```python
# In optimization pipeline
from metrix_mcp import MetrixTool

def profile_and_optimize(kernel_path):
    # Profile baseline
    tool = MetrixTool()
    baseline = tool.profile(
        command=f"python {kernel_path}",
        auto_select=True,
        quick=False
    )
    
    # Get bottleneck
    kernel_data = baseline["results"][0]["kernels"][0]
    bottleneck = kernel_data["bottleneck"]
    observations = kernel_data["observations"]
    
    # Use for LLM-guided optimization
    optimization_prompt = f"""
    Kernel bottleneck: {bottleneck}
    Observations:
    {chr(10).join(f'- {obs}' for obs in observations)}
    
    Optimize this kernel...
    """
    
    return bottleneck, observations
```

## Comparison: MCP vs Direct API

### MCP Protocol
**Pros:**
- Process isolation (GPU profiling can crash)
- Remote execution capability
- Standard protocol for agent communication
- Better timeout control

**Cons:**
- Requires async/await
- Slight overhead from IPC

### Direct API
**Pros:**
- Simpler synchronous code
- No process overhead
- Direct Python imports

**Cons:**
- Crashes can affect main process
- Tight coupling

## Recommendation

- **Use Direct API** for: Development, debugging, simple scripts
- **Use MCP Protocol** for: Production agents, remote profiling, fault tolerance

## Example Output

```json
{
  "success": true,
  "results": [
    {
      "device_id": "0",
      "gpu_info": {
        "vendor": "AMD",
        "model": "AMD Instinct MI300X OAM",
        "architecture": "gfx942",
        "compute_units": 228,
        "peak_hbm_bandwidth_gbs": 5200.0
      },
      "kernels": [
        {
          "name": "fused_topk_kernel",
          "duration_us": 234.56,
          "bottleneck": "memory-bound",
          "observations": [
            "High HBM bandwidth utilization (67.2%)",
            "Low L2 cache hit rate (12.3%)",
            "Memory coalescing could be improved (54.1%)"
          ],
          "metrics": {
            "duration_us": 234.56,
            "memory.hbm_bandwidth_utilization": 0.672,
            "memory.l2_hit_rate": 0.123,
            "memory.coalescing_efficiency": 0.541
          }
        }
      ]
    }
  ]
}
```

## License

Part of the GEAK Agent project.
