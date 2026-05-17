"""Tests for ``minisweagent.run.preprocess_v3.lang``.

These tests use ``tmp_path`` fixtures with tiny synthetic kernel files to
exercise the detection heuristic end-to-end (extension + content scan).
They do not depend on the actual repository contents, so they remain
deterministic across machines.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minisweagent.run.preprocess_v3.lang import (
    FLYDSL,
    UNKNOWN,
    KernelLanguage,
    detect_language,
    detect_language_for_repo,
)

_TRITON_KERNEL_BODY = """
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
"""

_HIP_KERNEL_BODY = """
#include <hip/hip_runtime.h>

__global__ void add_kernel(float* x, float* y, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = x[idx] + y[idx];
}

extern "C" void launch(float* x, float* y, float* out, int n) {
    hipLaunchKernelGGL(add_kernel, dim3((n+255)/256), dim3(256), 0, 0, x, y, out, n);
}
"""

_FLYDSL_KERNEL_BODY = """
import flydsl
from flydsl import kernel, tile


@flydsl.kernel
def add_kernel(x, y, out):
    out[:] = x + y
"""

_PLAIN_PY_BODY = """
def add(a, b):
    return a + b
"""


def test_detect_language_returns_triton_for_triton_kernel(tmp_path: Path) -> None:
    kernel_file = tmp_path / "triton_kernel.py"
    kernel_file.write_text(_TRITON_KERNEL_BODY, encoding="utf-8")

    lang = detect_language(kernel_file)

    assert isinstance(lang, KernelLanguage)
    assert lang.name == "triton"


def test_detect_language_returns_hip_for_hip_kernel(tmp_path: Path) -> None:
    kernel_file = tmp_path / "hip_kernel.cu"
    kernel_file.write_text(_HIP_KERNEL_BODY, encoding="utf-8")

    lang = detect_language(kernel_file)

    assert isinstance(lang, KernelLanguage)
    assert lang.name == "hip"


def test_detect_language_returns_flydsl_for_fdsl_extension(tmp_path: Path) -> None:
    kernel_file = tmp_path / "kernel.fdsl"
    kernel_file.write_text(_FLYDSL_KERNEL_BODY, encoding="utf-8")

    lang = detect_language(kernel_file)

    assert lang is FLYDSL
    assert lang.name == "flydsl"


def test_detect_language_returns_flydsl_for_flydsl_named_py(tmp_path: Path) -> None:
    """``**/*flydsl*`` glob match via filename token (PR plan rule)."""
    kernel_file = tmp_path / "my_flydsl_kernel.py"
    kernel_file.write_text(_FLYDSL_KERNEL_BODY, encoding="utf-8")

    lang = detect_language(kernel_file)

    assert lang is FLYDSL


def test_detect_language_returns_unknown_for_plain_python(tmp_path: Path) -> None:
    kernel_file = tmp_path / "plain.py"
    kernel_file.write_text(_PLAIN_PY_BODY, encoding="utf-8")

    lang = detect_language(kernel_file)

    assert lang is UNKNOWN
    assert lang.name == "unknown"


def test_detect_language_returns_unknown_for_missing_file(tmp_path: Path) -> None:
    """Detection on a non-existent path falls through to UNKNOWN cleanly."""
    nonexistent = tmp_path / "does_not_exist.py"

    lang = detect_language(nonexistent)

    assert lang is UNKNOWN


def test_detect_language_for_repo_picks_majority_triton(tmp_path: Path) -> None:
    """Three triton + one hip → triton wins."""
    (tmp_path / "kernels").mkdir()
    for i in range(3):
        (tmp_path / "kernels" / f"k{i}.py").write_text(_TRITON_KERNEL_BODY, encoding="utf-8")
    (tmp_path / "kernels" / "other.cu").write_text(_HIP_KERNEL_BODY, encoding="utf-8")
    # Plain python files should not vote.
    (tmp_path / "kernels" / "utils.py").write_text(_PLAIN_PY_BODY, encoding="utf-8")

    lang = detect_language_for_repo(tmp_path)

    assert lang.name == "triton"


def test_detect_language_for_repo_returns_unknown_when_no_kernels(tmp_path: Path) -> None:
    """A repo with only non-kernel files falls back to UNKNOWN."""
    (tmp_path / "README.md").write_text("# nothing here", encoding="utf-8")
    (tmp_path / "utils.py").write_text(_PLAIN_PY_BODY, encoding="utf-8")

    lang = detect_language_for_repo(tmp_path)

    assert lang is UNKNOWN


def test_detect_language_for_repo_returns_unknown_for_missing_root(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-dir"

    lang = detect_language_for_repo(missing)

    assert lang is UNKNOWN


def test_detect_language_for_repo_skips_noise_dirs(tmp_path: Path) -> None:
    """``__pycache__`` / ``.git`` / etc. should not vote."""
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.py").write_text(_TRITON_KERNEL_BODY, encoding="utf-8")

    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hooks.py").write_text(_TRITON_KERNEL_BODY, encoding="utf-8")

    (tmp_path / "real_kernel.cu").write_text(_HIP_KERNEL_BODY, encoding="utf-8")

    lang = detect_language_for_repo(tmp_path)

    assert lang.name == "hip"


@pytest.mark.parametrize(
    ("filename", "body", "expected_name"),
    [
        ("kernel.py", _TRITON_KERNEL_BODY, "triton"),
        ("kernel.cu", _HIP_KERNEL_BODY, "hip"),
        ("kernel.fdsl", _FLYDSL_KERNEL_BODY, "flydsl"),
        ("plain.py", _PLAIN_PY_BODY, "unknown"),
    ],
)
def test_detect_language_parametrized(
    tmp_path: Path,
    filename: str,
    body: str,
    expected_name: str,
) -> None:
    kernel_file = tmp_path / filename
    kernel_file.write_text(body, encoding="utf-8")

    assert detect_language(kernel_file).name == expected_name
