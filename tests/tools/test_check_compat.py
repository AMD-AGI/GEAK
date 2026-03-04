# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""Tests for check_kernel_compatibility tool."""

import tempfile
from pathlib import Path

from minisweagent.tools.check_compat import CheckKernelCompatibilityTool, check_compatibility

CUDA_CODE = """
#include <cuda_runtime.h>
__global__ void kernel(float* a, float* b) {
    int tid = threadIdx.x;
    float val = __shfl_down_sync(0xFFFFFFFF, a[tid], 1);
    cudaDeviceSynchronize();
    b[tid] = val;
}
"""

TRITON_CODE = """
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
"""


class TestCheckCompatibility:
    def test_cuda_patterns_detected(self):
        issues = check_compatibility(CUDA_CODE)
        assert len(issues) >= 2
        descriptions = [i["description"] for i in issues]
        assert any("__shfl_down_sync" in d for d in descriptions)
        assert any("cudaDeviceSynchronize" in d for d in descriptions)

    def test_clean_triton_code(self):
        issues = check_compatibility(TRITON_CODE)
        assert len(issues) == 0

    def test_comments_skipped(self):
        code = "// cudaMalloc is not used here\n# cudaFree also not used\nint x = 1;"
        issues = check_compatibility(code)
        assert len(issues) == 0

    def test_multiple_patterns(self):
        code = "cudaMalloc(&ptr, size);\ncudaMemcpy(dst, src, size, kind);\ncudaFree(ptr);"
        issues = check_compatibility(code)
        assert len(issues) == 3

    def test_line_numbers(self):
        code = "int x = 0;\ncudaMalloc(&ptr, size);\nint y = 1;"
        issues = check_compatibility(code)
        assert issues[0]["line_number"] == 2


class TestCheckKernelCompatibilityTool:
    def test_cuda_code_returns_issues(self):
        tool = CheckKernelCompatibilityTool()
        result = tool(kernel_code=CUDA_CODE)
        assert result["returncode"] == 0
        assert "CUDA-only" in result["output"]

    def test_clean_code_returns_compatible(self):
        tool = CheckKernelCompatibilityTool()
        result = tool(kernel_code=TRITON_CODE)
        assert result["returncode"] == 0
        assert "AMD-compatible" in result["output"]

    def test_file_path_mode(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(CUDA_CODE)
            path = f.name
        tool = CheckKernelCompatibilityTool()
        result = tool(file_path=path)
        assert result["returncode"] == 0
        assert "CUDA-only" in result["output"]
        Path(path).unlink()

    def test_no_input_returns_error(self):
        tool = CheckKernelCompatibilityTool()
        result = tool()
        assert result["returncode"] == 1
