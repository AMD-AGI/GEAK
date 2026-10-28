"""Hermetic contracts for the normalized AITER benchmark adapter."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "benchmarks" / "run_fp8_a8w8_blockscale.py"
SPEC = importlib.util.spec_from_file_location("run_fp8_a8w8_blockscale", RUNNER)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def test_percentile_interpolates_without_external_statistics_packages():
    assert BENCH.percentile([1.0], 0.1) == 1.0
    assert BENCH.percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert BENCH.percentile([0.0, 10.0], 0.1) == pytest.approx(1.0)


def test_implementation_identity_binds_name_config_and_source():
    first = BENCH.implementation_id("flydsl", {"tile_m": 16}, "a" * 64)
    assert first.startswith("impl_") and len(first) == 21
    assert first == BENCH.implementation_id("flydsl", {"tile_m": 16}, "a" * 64)
    assert first != BENCH.implementation_id("flydsl", {"tile_m": 32}, "a" * 64)
    assert first != BENCH.implementation_id("triton", {"tile_m": 16}, "a" * 64)


def test_manifest_points_at_the_real_runner_and_declares_accounting():
    manifest = yaml.safe_load((HERE / "benchmarks" / "gemm.yaml").read_text())
    suite = next(item for item in manifest["suites"] if item["id"] == "fp8_a8w8_blockscale")
    assert (HERE / suite["normalized_runner"]).is_file()
    assert set(suite["normalized_candidates"]) == set(BENCH.SOURCE_PATHS)
    assert any("preshuffle" in rule for rule in suite["required_accounting"])


def test_runner_removes_task_unittest_from_import_path(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "unittest.py").write_text("raise RuntimeError('shadow')\n")
    old = list(sys.path)
    sys.path.insert(0, str(task))
    try:
        BENCH.remove_unittest_shadow_paths()
        assert str(task) not in sys.path
    finally:
        sys.path[:] = old


def test_fp8_dtype_tracks_architecture_format():
    assert BENCH.fp8_dtype_for_gfx("gfx942:sramecc+") == "fp8_e4m3fnuz_blockscale"
    assert BENCH.fp8_dtype_for_gfx("gfx950") == "fp8_e4m3fn_blockscale"
