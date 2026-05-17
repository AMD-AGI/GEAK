"""Tests for ``minisweagent.run.preprocess_v3.explore``.

The fixtures build small synthetic repos under ``tmp_path`` so the
suite stays deterministic and offline. Each test exercises one
aspect of the wrapper contract:

* The Triton + HIP fixtures verify language coverage via the
  language-specific dependency parsing (Python ``ast.parse`` for
  Triton, ``#include`` regex for HIP).
* Noise filtering is verified explicitly — ``__pycache__`` /
  ``.git`` / ``*.pyc`` must not appear in the rendered markdown.
* ``out_path`` round-trip is checked twice (file written, file
  reread) so a future caller relying on disk-side artifacts has a
  regression net.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minisweagent.kernel_languages import registry
from minisweagent.run.preprocess_v3.explore import CodebaseContext, explore_codebase

_TRITON_KERNEL_BODY = '''"""Add kernel.

Adds two tensors with a Triton kernel."""

import triton
import triton.language as tl

from utils import grid_for


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
'''

_TRITON_UTILS_BODY = '''"""Helpers for the add kernel."""


def grid_for(n_elements: int, block_size: int) -> tuple[int, ...]:
    return ((n_elements + block_size - 1) // block_size,)
'''

_HIP_KERNEL_BODY = """\
// HIP add kernel.

#include <hip/hip_runtime.h>
#include "support.h"

__global__ void add_kernel(float* x, float* y, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = x[idx] + y[idx];
}
"""

_HIP_SUPPORT_HEADER = """\
#pragma once

#include <cstddef>

inline std::size_t grid_for(std::size_t n, std::size_t block) {
    return (n + block - 1) / block;
}
"""


def _make_triton_repo(root: Path) -> Path:
    """Build a tiny Triton-shaped repo with a kernel + one dep + noise."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "kernel.py").write_text(_TRITON_KERNEL_BODY, encoding="utf-8")
    (root / "utils.py").write_text(_TRITON_UTILS_BODY, encoding="utf-8")
    (root / "README.md").write_text("# triton add kernel\n", encoding="utf-8")
    # Noise: must not surface in the rendered tree.
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "kernel.cpython-310.pyc").write_text("garbage", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    return root


def _make_hip_repo(root: Path) -> Path:
    """Build a tiny HIP-shaped repo with a .cu kernel + one .h dep."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "add.cu").write_text(_HIP_KERNEL_BODY, encoding="utf-8")
    (root / "support.h").write_text(_HIP_SUPPORT_HEADER, encoding="utf-8")
    (root / "README.md").write_text("# hip add kernel\n", encoding="utf-8")
    return root


def _triton_lang():
    """Resolve the registered Triton ``KernelLanguage`` (skip if not bundled)."""
    lang = registry.get("triton")
    if lang is None:
        pytest.skip("Triton language bundle not registered")
    return lang


def _hip_lang():
    lang = registry.get("hip")
    if lang is None:
        pytest.skip("HIP language bundle not registered")
    return lang


def test_explore_codebase_returns_context_for_triton(tmp_path: Path) -> None:
    repo = _make_triton_repo(tmp_path / "repo")

    ctx = explore_codebase(repo, repo / "kernel.py", _triton_lang())

    assert isinstance(ctx, CodebaseContext)
    assert ctx.kernel_language is _triton_lang()
    assert ctx.out_path is None
    assert ctx.text.startswith("# Codebase Context")
    assert "kernel.py" in ctx.text
    assert "## Repository Layout" in ctx.text
    assert "## Kernel Dependency Tree" in ctx.text


def test_explore_codebase_returns_context_for_hip(tmp_path: Path) -> None:
    repo = _make_hip_repo(tmp_path / "repo")

    ctx = explore_codebase(repo, repo / "add.cu", _hip_lang())

    assert ctx.kernel_language is _hip_lang()
    assert "add.cu" in ctx.text
    # The dep is the .h header included from add.cu — the C++
    # ``#include`` regex must surface it as an in-repo dependency.
    assert "support.h" in ctx.text
    assert "support.h" in ctx.files


def test_explore_codebase_files_includes_kernel_first(tmp_path: Path) -> None:
    repo = _make_triton_repo(tmp_path / "repo")

    ctx = explore_codebase(repo, repo / "kernel.py", _triton_lang())

    assert ctx.files, "expected at least the kernel itself in files"
    assert ctx.files[0] == "kernel.py", "kernel must lead the files list"


def test_explore_codebase_files_lists_in_repo_dependencies(tmp_path: Path) -> None:
    repo = _make_triton_repo(tmp_path / "repo")

    ctx = explore_codebase(repo, repo / "kernel.py", _triton_lang())

    # ``utils.py`` is the only in-repo dep; ``triton`` is third-party
    # and should NOT appear.
    assert "utils.py" in ctx.files
    assert "triton" not in ctx.files
    # And the markdown render should mention the dependency too.
    assert "utils.py" in ctx.text


def test_explore_codebase_skips_noise_directories(tmp_path: Path) -> None:
    """``__pycache__`` / ``.git`` must not appear in the directory tree section."""
    repo = _make_triton_repo(tmp_path / "repo")

    ctx = explore_codebase(repo, repo / "kernel.py", _triton_lang())

    # The pruned tree should not surface noise dirs.
    assert "__pycache__" not in ctx.text
    assert ".git" not in ctx.text
    assert ".cpython-310.pyc" not in ctx.text


def test_explore_codebase_writes_file_when_out_path_given(tmp_path: Path) -> None:
    repo = _make_triton_repo(tmp_path / "repo")
    out_dir = tmp_path / "geak_output"
    out_path = out_dir / "CODEBASE_CONTEXT.md"

    ctx = explore_codebase(repo, repo / "kernel.py", _triton_lang(), out_path=out_path)

    assert ctx.out_path == out_path.resolve()
    assert ctx.out_path.is_file()
    written = ctx.out_path.read_text(encoding="utf-8")
    assert written == ctx.text
    assert "kernel.py" in written


def test_explore_codebase_writes_to_custom_filename(tmp_path: Path) -> None:
    """The caller can pin a non-default filename."""
    repo = _make_triton_repo(tmp_path / "repo")
    out_path = tmp_path / "geak_output" / "context_alt.md"

    ctx = explore_codebase(repo, repo / "kernel.py", _triton_lang(), out_path=out_path)

    assert ctx.out_path == out_path.resolve()
    assert out_path.is_file()
    # The legacy CODEBASE_CONTEXT.md filename should not be left
    # lingering when we asked for a custom name.
    assert not (out_path.parent / "CODEBASE_CONTEXT.md").exists()


def test_explore_codebase_is_idempotent(tmp_path: Path) -> None:
    """Repeated calls must overwrite the previous file with identical content."""
    repo = _make_triton_repo(tmp_path / "repo")
    out_path = tmp_path / "out" / "CODEBASE_CONTEXT.md"

    ctx_a = explore_codebase(repo, repo / "kernel.py", _triton_lang(), out_path=out_path)
    text_a = ctx_a.out_path.read_text(encoding="utf-8")

    ctx_b = explore_codebase(repo, repo / "kernel.py", _triton_lang(), out_path=out_path)
    text_b = ctx_b.out_path.read_text(encoding="utf-8")

    assert text_a == text_b
    assert ctx_a.files == ctx_b.files


def test_explore_codebase_raises_for_missing_repo(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-repo"
    with pytest.raises(FileNotFoundError, match="repo_root"):
        explore_codebase(missing, missing / "kernel.py", _triton_lang())


def test_explore_codebase_raises_for_missing_kernel(tmp_path: Path) -> None:
    repo = _make_triton_repo(tmp_path / "repo")
    with pytest.raises(FileNotFoundError, match="kernel_path"):
        explore_codebase(repo, repo / "nope.py", _triton_lang())


@pytest.mark.parametrize(
    ("language_name", "kernel_filename", "kernel_body"),
    [
        ("triton", "kernel.py", _TRITON_KERNEL_BODY),
        ("hip", "add.cu", _HIP_KERNEL_BODY),
    ],
)
def test_explore_codebase_per_language(
    tmp_path: Path,
    language_name: str,
    kernel_filename: str,
    kernel_body: str,
) -> None:
    """Smoke test per-language: render must mention the kernel filename."""
    lang = registry.get(language_name)
    if lang is None:
        pytest.skip(f"{language_name} language bundle not registered")

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / kernel_filename).write_text(kernel_body, encoding="utf-8")
    if language_name == "hip":
        (repo / "support.h").write_text(_HIP_SUPPORT_HEADER, encoding="utf-8")
    else:
        (repo / "utils.py").write_text(_TRITON_UTILS_BODY, encoding="utf-8")

    ctx = explore_codebase(repo, repo / kernel_filename, lang)

    assert kernel_filename in ctx.text
    assert ctx.kernel_language is lang
