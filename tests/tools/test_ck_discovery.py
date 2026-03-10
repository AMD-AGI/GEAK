# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Tests for CK kernel detection and workspace scoping in automated_test_discovery."""

import sys
import textwrap
from pathlib import Path

import pytest

_MCP_SRC = str(Path(__file__).resolve().parent.parent.parent / "mcp_tools" / "automated-test-discovery" / "src")
if _MCP_SRC not in sys.path:
    sys.path.insert(0, _MCP_SRC)

from automated_test_discovery.server import (
    _expand_workspace,
    _get_kernel_type,
    _is_kernel_file,
    _should_skip,
)


# ---------------------------------------------------------------------------
# _get_kernel_type
# ---------------------------------------------------------------------------


class TestGetKernelType:
    def test_ck_namespace(self):
        assert _get_kernel_type("using namespace ck::tensor_operation;") == "ck"

    def test_ck_include_quoted(self):
        assert _get_kernel_type('#include "ck/tensor_operation/gpu/device/device_conv.hpp"') == "ck"

    def test_ck_include_angle(self):
        assert _get_kernel_type("#include <ck/utility/data_type.hpp>") == "ck"

    def test_ck_device_mem(self):
        assert _get_kernel_type("DeviceMem in_dev(sizeof_in);") == "ck"

    def test_ck_tile(self):
        assert _get_kernel_type("ck_tile::foo") == "ck"

    def test_triton_over_ck(self):
        assert _get_kernel_type("@triton.jit\ndef kernel(): tl.load(...)") == "triton"

    def test_hip(self):
        assert _get_kernel_type("__global__ void kern() { hip stuff }") == "hip"

    def test_cuda(self):
        assert _get_kernel_type("__global__ void kern() { stuff }") == "cuda"

    def test_unknown(self):
        assert _get_kernel_type("print('hello')") == "unknown"


# ---------------------------------------------------------------------------
# _is_kernel_file — CK patterns
# ---------------------------------------------------------------------------


class TestIsKernelFileCK:
    def test_recognizes_ck_cpp(self, tmp_path):
        src = tmp_path / "grouped_conv_fwd_xdl_fp16.cpp"
        src.write_text(textwrap.dedent("""\
            #include "ck/tensor_operation/gpu/device/device_conv.hpp"
            using namespace ck;
            int main() {}
        """))
        assert _is_kernel_file(src) is True

    def test_ignores_plain_cpp(self, tmp_path):
        src = tmp_path / "main.cpp"
        src.write_text("int main() { return 0; }\n")
        assert _is_kernel_file(src) is False


# ---------------------------------------------------------------------------
# _expand_workspace — CK scoping
# ---------------------------------------------------------------------------


class TestExpandWorkspaceCK:
    def test_ck_example_dir_scoped(self, tmp_path):
        ck_root = tmp_path / "composablekernel"
        example_dir = ck_root / "example" / "41_grouped_conv_conv_fwd"
        example_dir.mkdir(parents=True)
        (example_dir / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.14)")

        result = _expand_workspace(example_dir)
        assert result == example_dir

    def test_ck_kernel_file_scoped(self, tmp_path):
        ck_root = tmp_path / "composablekernel"
        example_dir = ck_root / "example" / "30_conv_fwd"
        example_dir.mkdir(parents=True)
        kernel = example_dir / "kernel.cpp"
        kernel.write_text('#include "ck/foo.hpp"')

        result = _expand_workspace(kernel)
        assert result == example_dir

    def test_non_ck_goes_to_git_root(self, tmp_path):
        repo = tmp_path / "myrepo"
        repo.mkdir()
        (repo / ".git").mkdir()
        src = repo / "src"
        src.mkdir()
        kernel = src / "kernel.py"
        kernel.write_text("import triton")

        result = _expand_workspace(kernel)
        assert result == repo


# ---------------------------------------------------------------------------
# _should_skip — CK repo dirs
# ---------------------------------------------------------------------------


class TestShouldSkipCK:
    def test_skip_ck_include_dir(self, tmp_path):
        p = tmp_path / "composablekernel" / "include" / "ck" / "utility" / "foo.hpp"
        p.parent.mkdir(parents=True)
        p.write_text("// header")
        assert _should_skip(p) is True

    def test_skip_ck_library_dir(self, tmp_path):
        p = tmp_path / "composablekernel" / "library" / "src" / "gemm.cpp"
        p.parent.mkdir(parents=True)
        p.write_text("// lib")
        assert _should_skip(p) is True

    def test_skip_ck_test_dir(self, tmp_path):
        p = tmp_path / "composablekernel" / "test" / "gemm" / "test_gemm.cpp"
        p.parent.mkdir(parents=True)
        p.write_text("// test")
        assert _should_skip(p) is True

    def test_no_skip_ck_example_dir(self, tmp_path):
        p = tmp_path / "composablekernel" / "example" / "41_conv" / "kernel.cpp"
        p.parent.mkdir(parents=True)
        p.write_text('ck::tensor_operation')
        assert _should_skip(p) is False

    def test_no_skip_non_ck(self, tmp_path):
        p = tmp_path / "myrepo" / "src" / "kernel.py"
        p.parent.mkdir(parents=True)
        p.write_text("import triton")
        assert _should_skip(p) is False
