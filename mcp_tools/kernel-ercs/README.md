# Kernel ERCS MCP Server

**ERCS = Evaluation, Reflection, Compatibility, Specs**

An MCP (Model Context Protocol) server that provides LLM-based tools for Triton/AMD GPU kernel optimization.

## Tools

| Tool | Description |
|------|-------------|
| `evaluate_kernel_quality` | LLM analyzes kernel across 9 quality criteria |
| `reflect_on_kernel_result` | LLM analyzes test results, suggests next optimization |
| `check_kernel_compatibility` | Quick AMD compatibility scan (no LLM) |
| `get_amd_gpu_specs` | Returns MI350X hardware specs (no LLM) |

## Installation

```bash
cd kernel-ercs
pip install -e .
```

## Usage

### With Cursor

Add to Cursor MCP settings:

```json
{
  "name": "kernel-ercs",
  "command": "kernel-ercs",
  "env": {
    "AMD_LLM_API_KEY": "your-amd-gateway-key"
  }
}
```

### Standalone

```bash
export AMD_LLM_API_KEY="your-key"
kernel-ercs
```

### Programmatic

```python
from kernel_ercs.server import evaluate_kernel_quality, reflect_on_kernel_result

# Evaluate a kernel
result = evaluate_kernel_quality("""
@triton.jit
def my_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs)
    tl.store(y_ptr + offs, x * 2)
""")

print(f"Total Score: {result['total_score']}/9.0")
print(f"Ready to test: {result['ready_to_test']}")
```

## Tool Details

### evaluate_kernel_quality

Scores kernel on 9 AMD-specific criteria (0.0-1.0 each):
1. Fusion Intelligence
2. Block Size Appropriateness  
3. Memory Access Efficiency
4. Algorithmic Complexity
5. Warp/Wavefront Utilization
6. Software Pipelining
7. Numerical Stability
8. Correctness and Portability
9. Optimization Scope

### reflect_on_kernel_result

Analyzes test results and provides:
- Root cause analysis
- Bottleneck identification (memory/compute/correctness)
- Next optimization strategy with code hints
- Confidence level and continue/stop recommendation

### check_kernel_compatibility

Quick scan for AMD compatibility:
- Detects CUDA-only features (tl.libdevice)
- Validates num_warps range (1-16)
- Checks block size limits (max 1024)
- Verifies wavefront alignment (multiples of 64)

### get_amd_gpu_specs

Returns AMD MI350X specifications:
- Hardware specs (CUs, wavefront size, memory)
- Optimization guidelines
- Memory hierarchy details

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AMD_LLM_API_KEY` | AMD LLM Gateway API key |
| `LLM_GATEWAY_KEY` | Alternative AMD gateway key |
| `OPENAI_API_KEY` | OpenAI API key (fallback) |

## License

MIT License
