"""Tests for Triton kernel type inference, including wrapper files.

Covers:
- Direct @triton.jit / @triton.autotune / tl. usage
- Wrapper files that import Triton kernels from submodules
- Depth-limited BFS (max 2 levels of import following)
- Cycle detection and missing modules
- Mixed-language projects
- HIP, CK, CUDA, plain Python classification
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from minisweagent.agents.heterogeneous.task_generator import (
    _check_imported_triton,
    _infer_kernel_type,
)


@pytest.fixture
def tmpdir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


# ── Direct detection (no import following needed) ──────────────────────


class TestDirectDetection:
    def test_triton_jit_decorator(self, tmpdir):
        f = _write(tmpdir / "k.py", "import triton\n@triton.jit\ndef my_kernel(): pass\n")
        assert _infer_kernel_type(f) == "triton"

    def test_triton_autotune_decorator(self, tmpdir):
        f = _write(tmpdir / "k.py", "import triton\n@triton.autotune(configs=[])\n@triton.jit\ndef k(): pass\n")
        assert _infer_kernel_type(f) == "triton"

    def test_tl_usage(self, tmpdir):
        f = _write(tmpdir / "k.py", "import triton.language as tl\nx = tl.load(ptr)\n")
        assert _infer_kernel_type(f) == "triton"

    def test_import_triton_only(self, tmpdir):
        """File that does `import triton` but no @jit or tl. -- still triton."""
        f = _write(tmpdir / "k.py", "import triton\nimport triton.language as tl\ndef wrapper(): pass\n")
        assert _infer_kernel_type(f) == "triton"

    def test_plain_python(self, tmpdir):
        f = _write(tmpdir / "k.py", "import torch\ndef foo(): return 1\n")
        assert _infer_kernel_type(f) == "unknown"

    def test_hip_file(self, tmpdir):
        f = _write(tmpdir / "k.hip", "__global__ void silu() {}")
        assert _infer_kernel_type(f) == "hip"

    def test_cuda_cpp(self, tmpdir):
        f = _write(tmpdir / "k.cu", "__global__ void matmul() {}")
        assert _infer_kernel_type(f) == "hip"

    def test_ck_detection(self, tmpdir):
        """CK detection is only in MCP _get_kernel_type, not _infer_kernel_type."""
        f = _write(tmpdir / "k.cpp", "#include <ck_tile/core.hpp>\nnamespace ck_tile {}")
        assert _infer_kernel_type(f) == "hip"


# ── Import following (wrapper kernels) ─────────────────────────────────


class TestImportFollowing:
    def test_one_level_import(self, tmpdir):
        """Wrapper imports from a local module that has @triton.jit."""
        _write(tmpdir / "ops" / "__init__.py", "")
        _write(tmpdir / "ops" / "kernels.py", "@triton.jit\ndef fused_add(): pass\n")
        f = _write(tmpdir / "wrapper.py", "import triton\nfrom ops.kernels import fused_add\n")
        assert _infer_kernel_type(f) == "triton"

    def test_two_level_import(self, tmpdir):
        """Wrapper -> intermediate -> actual @triton.jit (depth 2)."""
        _write(tmpdir / "pkg" / "__init__.py", "")
        _write(tmpdir / "pkg" / "inner" / "__init__.py", "")
        _write(tmpdir / "pkg" / "inner" / "triton_ops.py", "@triton.jit\ndef matmul_kernel(): pass\n")
        _write(tmpdir / "pkg" / "api.py", "import triton\nfrom pkg.inner.triton_ops import matmul_kernel\n")
        f = _write(tmpdir / "wrapper.py", "import triton\nfrom pkg.api import matmul_kernel\n")
        assert _infer_kernel_type(f) == "triton"

    def test_three_level_too_deep(self, tmpdir):
        """Depth 3 -- beyond the limit, should still return triton via `import triton` heuristic."""
        _write(tmpdir / "a" / "__init__.py", "")
        _write(tmpdir / "a" / "b" / "__init__.py", "")
        _write(tmpdir / "a" / "b" / "c" / "__init__.py", "")
        _write(tmpdir / "a" / "b" / "c" / "deep.py", "@triton.jit\ndef deep_kernel(): pass\n")
        _write(tmpdir / "a" / "b" / "mid.py", "import triton\nfrom a.b.c.deep import deep_kernel\n")
        _write(tmpdir / "a" / "top.py", "import triton\nfrom a.b.mid import deep_kernel\n")
        f = _write(tmpdir / "wrapper.py", "import triton\nfrom a.top import deep_kernel\n")
        # Still returns triton because `import triton` is present; import following
        # may not reach depth 3 but the heuristic catches it
        assert _infer_kernel_type(f) == "triton"

    def test_import_nonexistent_module(self, tmpdir):
        """Import from a module that doesn't exist on disk. Should not crash."""
        f = _write(tmpdir / "wrapper.py", "import triton\nfrom nonexistent.module import kernel\n")
        assert _infer_kernel_type(f) == "triton"

    def test_import_module_without_triton(self, tmpdir):
        """Import from a real module that has NO triton markers."""
        _write(tmpdir / "utils" / "__init__.py", "")
        _write(tmpdir / "utils" / "helpers.py", "def helper(): return 42\n")
        f = _write(tmpdir / "wrapper.py", "import triton\nfrom utils.helpers import helper\n")
        # Still triton because the file itself has `import triton`
        assert _infer_kernel_type(f) == "triton"

    def test_aiter_style_wrapper(self, tmpdir):
        """Simulates the aiter pattern: wrapper imports kernels from deep submodule."""
        _write(tmpdir / "aiter" / "__init__.py", "")
        _write(tmpdir / "aiter" / "ops" / "__init__.py", "")
        _write(tmpdir / "aiter" / "ops" / "triton" / "__init__.py", "")
        _write(tmpdir / "aiter" / "ops" / "triton" / "_triton_kernels" / "__init__.py", "")
        _write(
            tmpdir / "aiter" / "ops" / "triton" / "_triton_kernels" / "fused_quant.py",
            "@triton.jit\ndef _fused_quant_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr): pass\n",
        )
        f = _write(
            tmpdir / "kernel.py",
            (
                "import triton\nimport triton.language as tl\n"
                "from aiter.ops.triton._triton_kernels.fused_quant import _fused_quant_kernel\n"
                "def wrapper(x): return _fused_quant_kernel(x)\n"
            ),
        )
        assert _infer_kernel_type(f) == "triton"
        # Also verify import following actually found the jit
        assert _check_imported_triton(f.read_text(), f)

    def test_relative_import_style(self, tmpdir):
        """from .submodule import ... style (common in packages)."""
        _write(tmpdir / "pkg" / "__init__.py", "")
        _write(tmpdir / "pkg" / "triton_impl.py", "@triton.jit\ndef impl(): pass\n")
        # Relative imports look like `from .triton_impl import impl` in source
        # but our regex matches `from X import ...` -- relative dots aren't a word char
        # so this tests that case doesn't crash
        f = _write(tmpdir / "pkg" / "api.py", "import triton\nfrom .triton_impl import impl\n")
        assert _infer_kernel_type(f) == "triton"


# ── _check_imported_triton specifically ────────────────────────────────


class TestCheckImportedTriton:
    def test_finds_jit_in_imported_module(self, tmpdir):
        _write(tmpdir / "kernels.py", "@triton.jit\ndef k(): pass\n")
        wrapper = _write(tmpdir / "w.py", "from kernels import k\n")
        assert _check_imported_triton(wrapper.read_text(), wrapper) is True

    def test_finds_autotune_in_imported_module(self, tmpdir):
        _write(tmpdir / "kernels.py", "@triton.autotune(configs=[])\ndef k(): pass\n")
        wrapper = _write(tmpdir / "w.py", "from kernels import k\n")
        assert _check_imported_triton(wrapper.read_text(), wrapper) is True

    def test_no_triton_in_imported_module(self, tmpdir):
        _write(tmpdir / "utils.py", "def helper(): return 1\n")
        wrapper = _write(tmpdir / "w.py", "from utils import helper\n")
        assert _check_imported_triton(wrapper.read_text(), wrapper) is False

    def test_missing_module_no_crash(self, tmpdir):
        wrapper = _write(tmpdir / "w.py", "from ghost_module import phantom\n")
        assert _check_imported_triton(wrapper.read_text(), wrapper) is False

    def test_depth_limit_respected(self, tmpdir):
        """At depth 3, should return False (limit is 2)."""
        _write(tmpdir / "d.py", "@triton.jit\ndef deep(): pass\n")
        _write(tmpdir / "c.py", "import triton\nfrom d import deep\n")
        _write(tmpdir / "b.py", "import triton\nfrom c import deep\n")
        wrapper = _write(tmpdir / "a.py", "from b import deep\n")
        # Depth 0->b, 1->c, 2->d -- d has @triton.jit, should find at depth 2
        assert _check_imported_triton(wrapper.read_text(), wrapper) is True

    def test_depth_3_not_reached(self, tmpdir):
        """Four levels: a->b->c->d where only d has @triton.jit. Depth limit stops at 2."""
        _write(tmpdir / "e.py", "@triton.jit\ndef deep(): pass\n")
        _write(tmpdir / "d.py", "import triton\nfrom e import deep\n")
        _write(tmpdir / "c.py", "import triton\nfrom d import deep\n")
        _write(tmpdir / "b.py", "import triton\nfrom c import deep\n")
        wrapper = _write(tmpdir / "a.py", "from b import deep\n")
        # a(0)->b(1)->c(2)->d(3 - exceeds limit) -- should NOT find
        assert _check_imported_triton(wrapper.read_text(), wrapper) is False

    def test_init_py_fallback(self, tmpdir):
        """When from pkg.sub import X, try pkg/sub.py first then pkg/__init__.py."""
        _write(tmpdir / "mypkg" / "__init__.py", "@triton.jit\ndef pkg_kernel(): pass\n")
        wrapper = _write(tmpdir / "w.py", "from mypkg import pkg_kernel\n")
        assert _check_imported_triton(wrapper.read_text(), wrapper) is True

    def test_binary_file_no_crash(self, tmpdir):
        """Importing a module path that resolves to a binary file."""
        bad = _write(tmpdir / "binary_mod.py", "\x00\x01\x02\xff" * 100)
        wrapper = _write(tmpdir / "w.py", "from binary_mod import thing\n")
        assert _check_imported_triton(wrapper.read_text(), wrapper) is False

    def test_large_file_truncated(self, tmpdir):
        """Imported file >8192 bytes: only first 8192 scanned."""
        big_content = "# padding\n" * 1000 + "@triton.jit\ndef late_kernel(): pass\n"
        _write(tmpdir / "big.py", big_content)
        wrapper = _write(tmpdir / "w.py", "from big import late_kernel\n")
        if len(big_content) > 8192 and "@triton.jit" not in big_content[:8192]:
            assert _check_imported_triton(wrapper.read_text(), wrapper) is False
        else:
            assert _check_imported_triton(wrapper.read_text(), wrapper) is True

    def test_multiple_imports_first_hit_wins(self, tmpdir):
        """Multiple from-imports, only one has triton."""
        _write(tmpdir / "utils.py", "def helper(): pass\n")
        _write(tmpdir / "kernels.py", "@triton.jit\ndef k(): pass\n")
        wrapper = _write(tmpdir / "w.py", "from utils import helper\nfrom kernels import k\n")
        assert _check_imported_triton(wrapper.read_text(), wrapper) is True
