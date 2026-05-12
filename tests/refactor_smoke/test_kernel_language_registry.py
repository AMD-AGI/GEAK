"""Unit test: kernel_languages registry + detection.

Replaces the 3 scattered detection functions (mini.py::_normalize_kernel_type,
heterogeneous/task_generator::_infer_kernel_type, preprocess/discovery_types::
_infer_kernel_language) with a single `registry.detect_best()` entry point.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from minisweagent.kernel_languages import registry
from minisweagent.kernel_languages.base import KernelLanguage


def test_registry_has_triton_and_hip() -> None:
    langs = {lang.name for lang in registry.all()}
    assert "triton" in langs, "Triton language must auto-register at import"
    assert "hip" in langs, "HIP language must auto-register at import"


def test_get_by_name() -> None:
    triton = registry.get("triton")
    assert triton is not None
    assert isinstance(triton, KernelLanguage)
    assert triton.kb_namespace == "triton"

    hip = registry.get("hip")
    assert hip is not None
    assert hip.kb_namespace == "hip"

    assert registry.get("nonexistent") is None


def test_aliases() -> None:
    """detect_best_by_name should handle legacy aliases (rocm, rocblas → hip)."""
    assert registry.detect_best_by_name("rocm").name == "hip"
    assert registry.detect_best_by_name("rocblas").name == "hip"
    # Case-insensitive
    assert registry.detect_best_by_name("TRITON").name == "triton"
    assert registry.detect_best_by_name("  hip  ").name == "hip"


def test_detect_triton_by_content() -> None:
    """A .py file with @triton.jit should detect as Triton, not HIP."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "kernel.py"
        p.write_text(
            "import triton\n"
            "import triton.language as tl\n"
            "\n"
            "@triton.jit\n"
            "def kernel(x_ptr, y_ptr, N: tl.constexpr):\n"
            "    offsets = tl.arange(0, N)\n"
            "    x = tl.load(x_ptr + offsets)\n"
            "    tl.store(y_ptr + offsets, x + 1.0)\n"
        )
        lang = registry.detect_best(p)
        assert lang is not None
        assert lang.name == "triton", (
            f"expected triton for @triton.jit content, got {lang.name}"
        )


def test_detect_hip_by_content() -> None:
    """A .cu file with __global__ should detect as HIP."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "kernel.cu"
        p.write_text(
            '#include <hip/hip_runtime.h>\n'
            '\n'
            '__global__ void add_one(float* in, float* out, int n) {\n'
            '    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;\n'
            '    if (idx < n) out[idx] = in[idx] + 1.0f;\n'
            '}\n'
        )
        lang = registry.detect_best(p)
        assert lang is not None
        assert lang.name == "hip", f"expected hip for __global__ content, got {lang.name}"


def test_detect_hip_pybind_wrapper() -> None:
    """A .py file using torch.utils.cpp_extension should detect as HIP."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "wrapper.py"
        p.write_text(
            'import torch\n'
            'from torch.utils.cpp_extension import load\n'
            '\n'
            'kernel = load(name="mykernel", sources=["kernel.hip"])\n'
        )
        lang = registry.detect_best(p)
        # Both Triton and HIP claim .py, but hip's hint (torch.utils.cpp_extension) fires
        assert lang is not None
        assert lang.name == "hip", (
            f"expected hip for cpp_extension wrapper, got {lang.name}"
        )


def test_detect_nothing_matches() -> None:
    """A .rs file with no known hints returns None."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "foo.rs"
        p.write_text("fn main() { println!(\"hi\"); }\n")
        assert registry.detect_best(p) is None


def test_lazy_prompt_loading_returns_empty_when_unset() -> None:
    """Since prompt files haven't been populated yet in PR-1, the property
    accessors should return '' (not raise)."""
    triton = registry.get("triton")
    # These should all be "" until their files are populated in a later commit
    assert isinstance(triton.system_prompt, str)
    assert isinstance(triton.optimization_prompt, str)
    assert isinstance(triton.planner_strategy_hints, str)
    assert isinstance(triton.harness_template, str)
    assert isinstance(triton.commandment_template, str)


def test_tool_set_is_frozenset() -> None:
    """Tool set must be a frozenset (immutable; safe to share between
    OptimizationAgent instances)."""
    for lang in registry.all():
        assert isinstance(lang.tool_set, frozenset)
