# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Tests for the CK harness template."""

import sys
from pathlib import Path

import pytest

_TEMPLATE_PATH = Path(__file__).resolve().parent.parent.parent / "src" / "minisweagent" / "templates" / "ck_harness_template.py"


class TestCKHarnessTemplateFlags:
    """Verify the template contains all required CLI flags."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = _TEMPLATE_PATH.read_text()

    def test_has_correctness_flag(self):
        assert "--correctness" in self.source

    def test_has_profile_flag(self):
        assert "--profile" in self.source

    def test_has_benchmark_flag(self):
        assert "--benchmark" in self.source

    def test_has_full_benchmark_flag(self):
        assert "--full-benchmark" in self.source

    def test_has_iterations_flag(self):
        assert "--iterations" in self.source

    def test_uses_argparse(self):
        assert "argparse" in self.source or "ArgumentParser" in self.source


class TestCKHarnessTemplateContent:
    """Verify the template uses the right patterns for CK harnesses."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = _TEMPLATE_PATH.read_text()

    def test_has_original_binary(self):
        assert "ORIGINAL_BINARY" in self.source

    def test_has_build_dir(self):
        assert "BUILD_DIR" in self.source

    def test_has_cmake_target(self):
        assert "CMAKE_TARGET" in self.source

    def test_has_cmake_source_dir(self):
        assert "CMAKE_SOURCE_DIR" in self.source

    def test_uses_subprocess(self):
        assert "subprocess" in self.source

    def test_uses_numpy(self):
        assert "numpy" in self.source or "np." in self.source

    def test_uses_geak_dump_output(self):
        assert "GEAK_DUMP_OUTPUT" in self.source

    def test_uses_allclose(self):
        assert "allclose" in self.source

    def test_fp16_tolerances(self):
        assert "RTOL_FP16" in self.source
        assert "ATOL_FP16" in self.source

    def test_fp32_tolerances(self):
        assert "RTOL_FP32" in self.source
        assert "ATOL_FP32" in self.source

    def test_geomean_output(self):
        assert "GEAK_RESULT_LATENCY_MS" in self.source


class TestCKHarnessStaticValidation:
    """Verify the template passes the pipeline's validate_harness check."""

    def test_passes_validate_harness(self):
        from minisweagent.run.pipeline_helpers import validate_harness

        valid, errors = validate_harness(str(_TEMPLATE_PATH))
        assert valid, f"Template failed static validation: {errors}"


class TestCKHarnessNumpyComparison:
    """Test the comparison logic with mock data."""

    def test_allclose_pass(self, tmp_path):
        import numpy as np

        orig = tmp_path / "orig.txt"
        opt = tmp_path / "opt.txt"
        data = np.random.randn(100).astype(np.float32)
        np.savetxt(str(orig), data)
        np.savetxt(str(opt), data)

        orig_loaded = np.loadtxt(str(orig), dtype=np.float32)
        opt_loaded = np.loadtxt(str(opt), dtype=np.float32)
        assert np.allclose(orig_loaded, opt_loaded, rtol=1e-2, atol=1e-2)

    def test_allclose_fail_large_diff(self, tmp_path):
        import numpy as np

        orig = tmp_path / "orig.txt"
        opt = tmp_path / "opt.txt"
        data_orig = np.zeros(100, dtype=np.float32)
        data_opt = np.ones(100, dtype=np.float32)
        np.savetxt(str(orig), data_orig)
        np.savetxt(str(opt), data_opt)

        orig_loaded = np.loadtxt(str(orig), dtype=np.float32)
        opt_loaded = np.loadtxt(str(opt), dtype=np.float32)
        assert not np.allclose(orig_loaded, opt_loaded, rtol=1e-2, atol=1e-2)

    def test_allclose_pass_within_fp16_tolerance(self, tmp_path):
        """Simulate FP16 accumulation differences within tolerance."""
        import numpy as np

        data = np.random.randn(100).astype(np.float32) * 10.0
        noise = np.random.randn(100).astype(np.float32) * 5e-3
        data_noisy = data + noise

        orig = tmp_path / "orig.txt"
        opt = tmp_path / "opt.txt"
        np.savetxt(str(orig), data)
        np.savetxt(str(opt), data_noisy)

        orig_loaded = np.loadtxt(str(orig), dtype=np.float32)
        opt_loaded = np.loadtxt(str(opt), dtype=np.float32)
        assert np.allclose(orig_loaded, opt_loaded, rtol=1e-2, atol=1e-2)


class TestSampleShapes:
    """Test the _sample_shapes helper from the template."""

    def test_sample_fewer_than_n(self):
        sys.path.insert(0, str(_TEMPLATE_PATH.parent))
        from ck_harness_template import _sample_shapes

        shapes = [[1], [2], [3]]
        result = _sample_shapes(shapes, 10)
        assert result == shapes

    def test_sample_exact_n(self):
        sys.path.insert(0, str(_TEMPLATE_PATH.parent))
        from ck_harness_template import _sample_shapes

        shapes = [[i] for i in range(25)]
        result = _sample_shapes(shapes, 25)
        assert result == shapes

    def test_sample_more_than_n(self):
        sys.path.insert(0, str(_TEMPLATE_PATH.parent))
        from ck_harness_template import _sample_shapes

        shapes = [[i] for i in range(100)]
        result = _sample_shapes(shapes, 5)
        assert len(result) == 5
