# Discovery Pipeline

Automated test and benchmark discovery for GPU kernels. No configuration files needed.

## Concept

The discovery pipeline runs before the agent loop to automatically find:
1. **Kernel files** - Triton, HIP, CUDA kernel definitions
2. **Test files** - Correctness tests for the kernel
3. **Benchmark files** - Performance benchmarks

### Design Philosophy

**Pure content-based detection** - Instead of relying on hardcoded directory names or requiring configuration files, the pipeline:
- Scans all files in the workspace
- Scores each file based on content patterns (keywords, imports, decorators)
- Ranks results by confidence + kernel name relevance

This makes it work for **any project structure** without setup.

## Features

### 1. Content-Based Detection

Files are scored using keyword patterns:

**Test Patterns:**
| Pattern | Score | Description |
|---------|-------|-------------|
| `import pytest` | 0.3 | Pytest import |
| `@pytest.mark` | 0.3 | Pytest decorator |
| `def test_\w+` | 0.4 | Test function |
| `assert` | 0.2 | Assertion |
| `.allclose()` | 0.3 | Tensor comparison |
| `@perftest()` | 0.35 | Custom decorator (auto-detected) |
| `checkAllclose` | 0.3 | Custom function (auto-detected) |

**Benchmark Patterns:**
| Pattern | Score | Description |
|---------|-------|-------------|
| `TFLOPS/GFLOPS` | 0.4 | Performance metric |
| `torch.cuda.Event` | 0.4 | CUDA timing |
| `triton.testing.do_bench` | 0.5 | Triton benchmark |
| `latency` | 0.25 | Latency measurement |
| `throughput` | 0.25 | Throughput measurement |
| `warmup` | 0.25 | Warmup pattern |

### 2. Auto-Pattern Learning

The pipeline scans a sample of test files to learn project-specific patterns:
- Custom decorators (`@perftest`, `@benchmark`)
- Custom assertion functions (`checkAllclose`, `verify_output`)
- Custom imports (`from test_common import`)

These are automatically added to the detection keywords.

### 3. Kernel Name Matching

Files that match the kernel name get a confidence boost:
- **Exact match** (`gemm_a16w16` → `test_gemm_a16w16.py`): +1.0
- **Partial match** (2+ parts match): +0.3 per part

This ensures kernel-specific tests rank higher than generic tests.

### 4. Smart Filtering

The pipeline automatically excludes:
- Kernel definition files (files containing `@triton.jit`, `__global__`)
- Build artifacts (`__pycache__`, `build/`, `dist/`)
- Virtual environments (`.venv`, `node_modules`)

### 5. C++ Support

Detects C++ test frameworks:
- GTest (`TEST()`, `EXPECT_*`, `ASSERT_*`)
- Catch2 (`TEST_CASE()`, `REQUIRE()`, `CHECK()`)
- HIP/CUDA patterns

### 6. Workspace Expansion

When given a single kernel file, automatically expands search to project root:
- Looks for markers: `pyproject.toml`, `.git`, `setup.py`
- Stops at monorepo boundaries: `lerna.json`, `nx.json`

## Usage

### Command Line

```bash
# Discover tests for a kernel
python -m mini_kernel.mcp_tools.discovery /path/to/kernel.py

# Non-interactive mode
python -m mini_kernel.mcp_tools.discovery /path/to/kernel.py --no-confirm

# Discovery only (don't start agent)
python -m mini_kernel.mcp_tools.discovery /path/to/kernel.py --discover-only
```

### Programmatic

```python
from mini_kernel.mcp_tools.discovery import discover

result = discover(
    kernel_path=Path("/path/to/kernel.py"),
    interactive=False
)

# Access results
for test in result.tests:
    print(f"{test.file_path}: {test.confidence:.0%}")
```

### User-Provided Commands

Skip discovery by providing explicit commands:

```bash
python -m mini_kernel.cli /path/to/kernel.py \
    --test "pytest tests/test_kernel.py -v" \
    --bench "python benchmarks/bench_kernel.py"
```

## Test Results

### Comprehensive Test (30 Kernels)

Tested on the `aiter` repository with 30 different Triton kernels.

| Metric | Value |
|--------|-------|
| Kernels tested | 30 |
| Correct matches | 27 |
| **Match rate** | **90%** |

### Matched Tests (27/30)

| Kernel | Discovered Test |
|--------|-----------------|
| `mha.py` | `test_mha.py` |
| `gemm_a8w8.py` | `test_batched_gemm_a8w8.py` |
| `gemm_a8wfp4.py` | `test_gemm_a8wfp4.py` |
| `gemm_afp4wfp4.py` | `test_batched_gemm_afp4wfp4.py` |
| `gemm_a16w16.py` | `test_gemm_a16w16.py` |
| `moe_op_mxfp4.py` | `test_moe_sorting_mxfp4.py` |
| `moe_op_silu_fused.py` | `test_moe_routing_sigmoid_top1_fused.py` |
| `rmsnorm.py` | `test_rmsnorm2d.py` |
| `unified_attention.py` | `test_unified_attention.py` |
| `unified_attention_sparse_mla.py` | `test_unified_attention_sparse_mla.py` |
| `mla_decode_rope.py` | `test_mla_decode_rope.py` |
| `batched_gemm_afp4wfp4.py` | `test_batched_gemm_afp4wfp4.py` |
| `activation.py` | `test_activation.py` |
| `norm.py` | `test_layernorm2dFusedAddQuant.py` |
| `gemm_a16w16_atomic.py` | `test_gemm_a16w16.py` |
| `fused_gemm_a8w8_blockscale_a16w16.py` | `test_fused_gemm_a8w8_blockscale_a16w16.py` |
| `mha_fused_bwd.py` | `test_fmha_bwd.cpp` |
| `pa_mqa_logits.py` | `test_fp8_mqa_logits.py` |
| `gemm_a8w8_blockscale.py` | `test_gemm_a8w8_blockscale_mi350.py` |
| `fused_mxfp4_quant.py` | `test_fused_mxfp4_quant.py` |
| `batched_gemm_a16wfp4.py` | `test_batched_gemm_a16wfp4.py` |
| `gmm.py` | `test_gmm.py` |
| `fused_fp8_quant.py` | `test_fused_fp8_quant.py` |
| `fused_mul_add.py` | `test_fused_mul_add.py` |
| `fused_qk_concat.py` | `test_fused_qk_concat.py` |
| `fused_gemm_afp4wfp4_split_cat.py` | `test_fused_gemm_afp4wfp4_split_cat.py` |
| `quant_moe.py` | `test_moe_smoothquant.cpp` |

### No Matching Tests (3/30)

These kernels don't have dedicated test files in the repository:

| Kernel | Reason |
|--------|--------|
| `lean_atten.py` | No test file exists |
| `moe_op_gelu.py` | No test file exists |
| `moe_op_e2e.py` | No test file exists |

The discovery correctly returns the most confident available result when no matching test exists.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DISCOVERY PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. WORKSPACE EXPANSION                                     │
│     kernel.py → find project root (.git, pyproject.toml)   │
│                                                             │
│  2. AUTO-PATTERN LEARNING                                   │
│     Sample test files → detect custom decorators/functions │
│                                                             │
│  3. KERNEL DISCOVERY                                        │
│     Find @triton.jit, __global__ patterns                  │
│                                                             │
│  4. TEST DISCOVERY                                          │
│     Scan ALL files → score by content → rank by confidence │
│                                                             │
│  5. BENCHMARK DISCOVERY                                     │
│     Scan ALL files → score by content → rank by confidence │
│                                                             │
│  6. USER CONFIRMATION                                       │
│     [y] Yes  [e] Edit  [s] Search more  [c] Create tests   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Future Improvements

- [ ] LLM-assisted analysis for uncertain cases (confidence 0.3-0.6)
- [ ] Interactive test creation when none found
- [ ] Makefile/CMake build system detection
- [ ] Multi-kernel batch discovery
