# Automated Test Discovery MCP

Single-tool MCP server for automatically discovering tests and benchmarks for GPU kernels.

## Features

- **Single tool** - One `discover` call returns everything
- **Content-based detection** - No configuration files needed
- **Auto-pattern learning** - Detects project-specific decorators and functions
- **Multi-language support** - Python (pytest, unittest), C++ (GTest), HIP, CUDA

## Tool: `discover`

Find tests and benchmarks for a GPU kernel file.

### Usage

```python
result = discover(
    kernel_path="/path/to/gemm_a16w16.py",
    max_tests=5,
    max_benchmarks=5
)
```

### Returns

```python
{
    "kernel": {
        "name": "gemm_a16w16",
        "type": "triton",
        "file": "/path/to/gemm_a16w16.py"
    },
    "workspace": "/path/to/project",
    "tests": [
        {
            "name": "test_gemm_a16w16.py",
            "file": "/path/to/test_gemm_a16w16.py",
            "confidence": 1.0,
            "command": "pytest /path/to/test_gemm_a16w16.py -v"
        }
    ],
    "benchmarks": [
        {
            "name": "bench_gemm_a16w16.py",
            "file": "/path/to/bench_gemm_a16w16.py",
            "confidence": 1.0,
            "command": "python /path/to/bench_gemm_a16w16.py"
        }
    ],
    "summary": "Found 1 test(s) and 1 benchmark(s) for gemm_a16w16 (triton kernel). Recommended test: test_gemm_a16w16.py"
}
```

## Installation

```bash
pip install -e .
```

## Usage

### As MCP Server

```bash
automated-test-discovery
```

### In Cursor/Claude

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "automated-test-discovery": {
      "command": "automated-test-discovery"
    }
  }
}
```

## How It Works

### Detection Patterns

**Test files** are scored by content:
- `import pytest`, `@pytest.mark`, `def test_*` 
- `assert`, `.allclose()`, `assertEqual`
- `@perftest()`, `checkAllclose` (custom frameworks)
- GTest: `TEST()`, `EXPECT_*`, `ASSERT_*`

**Benchmark files** are scored by:
- `TFLOPS`, `GFLOPS`, `latency`, `throughput`
- `torch.cuda.Event(enable_timing=True)`
- `triton.testing.do_bench`
- `warmup`, `speedup`, `GB/s`

**Kernel files** are detected by:
- `@triton.jit`, `@triton.autotune` (Triton)
- `__global__ void` (HIP/CUDA)

### Ranking

Files are ranked by:
1. Content score (how many patterns match)
2. Kernel name match boost (files containing the kernel name rank higher)

## Test Results

Tested on 30 GPU kernels from the aiter repository:
- **Match rate: 90%** (27/30 kernels found correct tests)
