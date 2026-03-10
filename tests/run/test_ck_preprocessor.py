# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Tests for CK-specific preprocessor functions."""

import textwrap
from pathlib import Path

import pytest

from minisweagent.run.preprocessor import (
    _detect_ck_cmake_target,
    _GEAK_DUMP_SENTINEL,
    _is_ck_kernel,
    _load_ck_template,
    _patch_ck_dump_output,
)


# ---------------------------------------------------------------------------
# _is_ck_kernel
# ---------------------------------------------------------------------------


class TestIsCKKernel:
    def test_true_for_ck_dir(self, tmp_path):
        src = tmp_path / "kernel.cpp"
        src.write_text(textwrap.dedent("""\
            #include "ck/tensor_operation/gpu/device/device_conv.hpp"
            using namespace ck;
            int main() {}
        """))
        assert _is_ck_kernel(tmp_path) is True

    def test_true_for_ck_file(self, tmp_path):
        src = tmp_path / "kernel.cpp"
        src.write_text('DeviceMem in_dev(sizeof_in);\n')
        assert _is_ck_kernel(src) is True

    def test_false_for_triton_dir(self, tmp_path):
        src = tmp_path / "kernel.py"
        src.write_text("import triton\n@triton.jit\ndef kernel(): pass\n")
        assert _is_ck_kernel(tmp_path) is False

    def test_false_for_empty_dir(self, tmp_path):
        assert _is_ck_kernel(tmp_path) is False

    def test_true_for_ck_namespace(self, tmp_path):
        src = tmp_path / "example.hpp"
        src.write_text("ck::tensor_operation::device::DeviceConvFwd foo;\n")
        assert _is_ck_kernel(tmp_path) is True

    def test_ignores_non_cpp_files(self, tmp_path):
        txt = tmp_path / "readme.txt"
        txt.write_text("ck:: is mentioned here but file is not C++\n")
        assert _is_ck_kernel(tmp_path) is False


# ---------------------------------------------------------------------------
# _detect_ck_cmake_target
# ---------------------------------------------------------------------------


class TestDetectCKCMakeTarget:
    def test_parses_add_example_executable(self, tmp_path):
        cmake = tmp_path / "CMakeLists.txt"
        cmake.write_text(textwrap.dedent("""\
            cmake_minimum_required(VERSION 3.14)
            add_example_executable(example_grouped_conv_conv_fwd_xdl_fp16
                grouped_conv_conv_fwd_xdl_fp16.cpp)
        """))
        assert _detect_ck_cmake_target(tmp_path) == "example_grouped_conv_conv_fwd_xdl_fp16"

    def test_parses_add_executable(self, tmp_path):
        cmake = tmp_path / "CMakeLists.txt"
        cmake.write_text(textwrap.dedent("""\
            cmake_minimum_required(VERSION 3.14)
            add_executable(my_kernel kernel.cpp)
        """))
        assert _detect_ck_cmake_target(tmp_path) == "my_kernel"

    def test_returns_none_without_cmake(self, tmp_path):
        assert _detect_ck_cmake_target(tmp_path) is None

    def test_returns_none_for_empty_cmake(self, tmp_path):
        cmake = tmp_path / "CMakeLists.txt"
        cmake.write_text("cmake_minimum_required(VERSION 3.14)\n")
        assert _detect_ck_cmake_target(tmp_path) is None


# ---------------------------------------------------------------------------
# _patch_ck_dump_output
# ---------------------------------------------------------------------------


class TestPatchCKDumpOutput:
    @pytest.fixture
    def example_dir(self, tmp_path):
        inc = tmp_path / "run_example.inc"
        inc.write_text(textwrap.dedent("""\
            #include <iostream>

            void run_example()
            {
                out_device_buf.FromDevice(out_device.mData.data());
                return ck::utils::check_err(out_device, out_host);
            }
        """))
        return tmp_path

    def test_patches_inc_file(self, example_dir):
        modified = _patch_ck_dump_output(example_dir)
        assert len(modified) == 1

        content = modified[0].read_text()
        assert _GEAK_DUMP_SENTINEL in content
        assert "std::getenv" in content
        assert "savetxt" in content

    def test_idempotent(self, example_dir):
        _patch_ck_dump_output(example_dir)
        first_content = (example_dir / "run_example.inc").read_text()

        modified = _patch_ck_dump_output(example_dir)
        assert len(modified) == 0

        second_content = (example_dir / "run_example.inc").read_text()
        assert first_content == second_content

    def test_adds_cstdlib_include(self, example_dir):
        _patch_ck_dump_output(example_dir)
        content = (example_dir / "run_example.inc").read_text()
        assert "#include <cstdlib>" in content

    def test_preserves_original_code(self, example_dir):
        _patch_ck_dump_output(example_dir)
        content = (example_dir / "run_example.inc").read_text()
        assert "FromDevice" in content
        assert "check_err" in content

    def test_no_patch_without_from_device(self, tmp_path):
        inc = tmp_path / "run_example.inc"
        inc.write_text("void run_example() { return 0; }\n")
        modified = _patch_ck_dump_output(tmp_path)
        assert len(modified) == 0


# ---------------------------------------------------------------------------
# _load_ck_template
# ---------------------------------------------------------------------------


class TestLoadCKTemplate:
    def test_loads_template(self):
        template = _load_ck_template()
        assert "__ORIGINAL_BINARY__" in template
        assert "__BUILD_DIR__" in template
        assert "--correctness" in template
        assert "GEAK_DUMP_OUTPUT" in template
