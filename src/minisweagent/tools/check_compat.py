# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""Tool: check_kernel_compatibility -- scan kernel code for AMD-incompatible patterns.

Regex-based scanner that detects CUDA-only constructs that won't work on AMD GPUs.
No LLM, no GPU -- pure string matching.
"""

from __future__ import annotations

import re
from typing import Any

# Patterns that indicate CUDA-only code (won't work on AMD/ROCm without porting)
_CUDA_PATTERNS: list[tuple[str, str]] = [
    (r"\bcudaMalloc\b", "cudaMalloc (use hipMalloc)"),
    (r"\bcudaMemcpy\b", "cudaMemcpy (use hipMemcpy)"),
    (r"\bcudaFree\b", "cudaFree (use hipFree)"),
    (r"\bcudaDeviceSynchronize\b", "cudaDeviceSynchronize (use hipDeviceSynchronize)"),
    (r"\bcudaGetDevice\b", "cudaGetDevice (use hipGetDevice)"),
    (r"\bcudaSetDevice\b", "cudaSetDevice (use hipSetDevice)"),
    (r"\bcudaStream_t\b", "cudaStream_t (use hipStream_t)"),
    (r"\bcudaEvent_t\b", "cudaEvent_t (use hipEvent_t)"),
    (r"\bcudaError_t\b", "cudaError_t (use hipError_t)"),
    (r"\b__syncwarp\b", "__syncwarp (not available in HIP, use __syncthreads)"),
    (r"\b__ballot_sync\b", "__ballot_sync (use __ballot in HIP)"),
    (r"\b__shfl_sync\b", "__shfl_sync (use __shfl in HIP)"),
    (r"\b__shfl_down_sync\b", "__shfl_down_sync (use __shfl_down in HIP)"),
    (r"\b__shfl_up_sync\b", "__shfl_up_sync (use __shfl_up in HIP)"),
    (r"\b__shfl_xor_sync\b", "__shfl_xor_sync (use __shfl_xor in HIP)"),
    (r"\bcublas", "cuBLAS (use rocBLAS)"),
    (r"\bcusparse", "cuSPARSE (use rocSPARSE)"),
    (r"\bcufft", "cuFFT (use rocFFT)"),
    (r"\bcurand", "cuRAND (use rocRAND)"),
    (r"\bcudnn", "cuDNN (use MIOpen)"),
    (r"\bnvcc\b", "nvcc (use hipcc)"),
    (r"#include\s*<cuda", "CUDA header include (use HIP headers)"),
    (r"#include\s*<cooperative_groups", "cooperative_groups (limited HIP support)"),
    (r"\bthrust::cuda\b", "thrust::cuda (use thrust::hip)"),
]


def check_compatibility(code: str) -> list[dict[str, str]]:
    """Scan code for CUDA-only patterns.

    Returns list of {pattern, description, line_number, line_content} dicts.
    """
    issues = []
    for i, line in enumerate(code.splitlines(), 1):
        # Skip comments
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("#"):
            continue
        for pattern, description in _CUDA_PATTERNS:
            if re.search(pattern, line):
                issues.append(
                    {
                        "pattern": pattern,
                        "description": description,
                        "line_number": i,
                        "line_content": line.rstrip(),
                    }
                )
    return issues


class CheckKernelCompatibilityTool:
    """ToolRuntime-compatible callable for check_kernel_compatibility."""

    def __call__(self, kernel_code: str | None = None, file_path: str | None = None) -> dict[str, Any]:
        """Scan kernel code for CUDA-only / AMD-incompatible patterns.

        Args:
            kernel_code: Kernel source code as a string.
            file_path: Path to a kernel file (used if kernel_code not provided).

        Returns:
            {output: str, returncode: int}
        """
        if not kernel_code and not file_path:
            return {"output": "Provide either kernel_code or file_path", "returncode": 1}

        if not kernel_code:
            try:
                from pathlib import Path

                kernel_code = Path(file_path).read_text()
            except Exception as e:
                return {"output": f"Cannot read file: {e}", "returncode": 1}

        issues = check_compatibility(kernel_code)

        if not issues:
            return {"output": "No CUDA-only patterns detected. Kernel appears AMD-compatible.", "returncode": 0}

        lines = [f"Found {len(issues)} CUDA-only pattern(s):"]
        for issue in issues:
            lines.append(f"  Line {issue['line_number']}: {issue['description']}\n    {issue['line_content']}")
        lines.append("\nThese patterns need to be ported for AMD/ROCm compatibility.")
        return {"output": "\n".join(lines), "returncode": 0}
