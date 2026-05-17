"""Tests for ``minisweagent.run.preprocess_v3.baseline``.

Covers both deterministic primitives:

* :func:`collect_baseline_metrics` — patches
  ``subprocess.run`` to inject deterministic stdout, asserts the
  median / stdev / sample collection, and verifies the wrapper
  command construction.

* :func:`collect_profile` — patches the internal
  ``_invoke_profiler_mcp`` shim to inject a fake profile dict so
  tests don't need profiler-mcp on the host. The single test that
  exercises a real profiler-mcp host is gated behind
  ``pytest.mark.profiler`` and skipped by default.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from minisweagent.run.preprocess_v3 import baseline as baseline_mod
from minisweagent.run.preprocess_v3.baseline import (
    BaselineMetrics,
    ProfileResult,
    collect_baseline_metrics,
    collect_profile,
)


def _make_harness(tmp_path: Path) -> Path:
    """Drop a tiny harness file on disk; contents don't matter — we mock subprocess."""
    harness = tmp_path / "test_harness.py"
    harness.write_text("# fake harness\n", encoding="utf-8")
    return harness


def _ok_proc(stdout: str = "GEAK_RESULT_LATENCY_MS=1.234\n", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=stderr)


def _fail_proc(stderr: str = "boom") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)


# ---------------------------------------------------------------------------
# collect_baseline_metrics
# ---------------------------------------------------------------------------


def test_collect_baseline_metrics_invokes_harness_correct_count(tmp_path: Path) -> None:
    """``repeats`` invocations of the harness, each producing one sample."""
    harness = _make_harness(tmp_path)
    captured_cmds: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        captured_cmds.append(list(cmd))
        return _ok_proc(stdout="GEAK_RESULT_LATENCY_MS=2.5\n")

    with mock.patch("subprocess.run", side_effect=_fake_run):
        result = collect_baseline_metrics(harness, repeats=3)

    assert isinstance(result, BaselineMetrics)
    assert len(captured_cmds) == 3
    # Every call uses the same wrapper command.
    for cmd in captured_cmds:
        assert cmd[0] == "bash"
        assert cmd[1] == "-lc"
        # The inner script invokes the harness with --benchmark.
        inner = cmd[2]
        assert sys.executable in inner
        assert str(harness) in inner
        assert "--benchmark" in inner


def test_collect_baseline_metrics_computes_median_and_stdev(tmp_path: Path) -> None:
    """Median + stdev are computed from per-sample latency_ms values."""
    harness = _make_harness(tmp_path)
    stdouts = iter(
        [
            "GEAK_RESULT_LATENCY_MS=1.0\n",
            "GEAK_RESULT_LATENCY_MS=2.0\n",
            "GEAK_RESULT_LATENCY_MS=3.0\n",
            "GEAK_RESULT_LATENCY_MS=4.0\n",
            "GEAK_RESULT_LATENCY_MS=5.0\n",
        ]
    )

    def _fake_run(cmd, **_kwargs):
        return _ok_proc(stdout=next(stdouts))

    with mock.patch("subprocess.run", side_effect=_fake_run):
        result = collect_baseline_metrics(harness, repeats=5)

    assert result.samples_ms == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert result.median_ms == pytest.approx(3.0)
    # statistics.stdev of 1..5 is sqrt(2.5) ~= 1.5811
    assert result.stdev_ms == pytest.approx(1.5811388300841898)
    assert result.repeats == 5
    assert result.success is True


def test_collect_baseline_metrics_single_sample_has_no_stdev(tmp_path: Path) -> None:
    """``stdev_ms`` is None when fewer than two samples are collected."""
    harness = _make_harness(tmp_path)
    with mock.patch(
        "subprocess.run",
        return_value=_ok_proc(stdout="GEAK_RESULT_LATENCY_MS=7.5\n"),
    ):
        result = collect_baseline_metrics(harness, repeats=1)

    assert result.samples_ms == [7.5]
    assert result.median_ms == pytest.approx(7.5)
    assert result.stdev_ms is None


def test_collect_baseline_metrics_skips_failed_invocations(tmp_path: Path) -> None:
    """Non-zero rc samples are recorded in raw_outputs but not in samples_ms."""
    harness = _make_harness(tmp_path)
    procs = [
        _ok_proc(stdout="GEAK_RESULT_LATENCY_MS=10.0\n"),
        _fail_proc(stderr="harness crashed"),
        _ok_proc(stdout="GEAK_RESULT_LATENCY_MS=20.0\n"),
    ]
    proc_iter = iter(procs)

    def _fake_run(cmd, **_kwargs):
        return next(proc_iter)

    with mock.patch("subprocess.run", side_effect=_fake_run):
        result = collect_baseline_metrics(harness, repeats=3)

    assert result.samples_ms == [10.0, 20.0]
    assert len(result.raw_outputs) == 3
    assert result.raw_outputs[1]["returncode"] == 1
    assert result.raw_outputs[1]["latency_ms"] is None
    assert "harness crashed" in result.raw_outputs[1]["stderr"]


def test_collect_baseline_metrics_when_no_marker_returns_none_median(tmp_path: Path) -> None:
    """No parseable latency in any run -> ``median_ms`` is ``None`` and ``success`` is False."""
    harness = _make_harness(tmp_path)
    with mock.patch(
        "subprocess.run",
        return_value=_ok_proc(stdout="just some chatty output, no marker here"),
    ):
        result = collect_baseline_metrics(harness, repeats=2)

    assert result.median_ms is None
    assert result.samples_ms == []
    assert result.success is False


def test_collect_baseline_metrics_handles_timeout(tmp_path: Path) -> None:
    """Timeouts are recorded as a failed sample (returncode=-1)."""
    harness = _make_harness(tmp_path)

    def _raises_timeout(cmd, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1, output="partial output", stderr="partial err")

    with mock.patch("subprocess.run", side_effect=_raises_timeout):
        result = collect_baseline_metrics(harness, repeats=1)

    assert result.samples_ms == []
    assert result.raw_outputs[0]["returncode"] == -1
    assert "TIMEOUT" in result.raw_outputs[0]["stderr"]


def test_collect_baseline_metrics_clamps_repeats_to_minimum_one(tmp_path: Path) -> None:
    """``repeats <= 0`` is clamped to 1 (still produces output for inspection)."""
    harness = _make_harness(tmp_path)
    with mock.patch(
        "subprocess.run",
        return_value=_ok_proc(stdout="GEAK_RESULT_LATENCY_MS=0.5\n"),
    ) as mocked:
        result = collect_baseline_metrics(harness, repeats=0)

    assert mocked.call_count == 1
    assert result.repeats == 1


def test_collect_baseline_metrics_passes_pythonpath_and_gpu_env(tmp_path: Path) -> None:
    """``work_dir`` is prepended to PYTHONPATH; ``gpu_id`` reaches HIP_VISIBLE_DEVICES."""
    harness = _make_harness(tmp_path)
    captured_envs: list[dict[str, str]] = []

    def _fake_run(cmd, **kwargs):
        captured_envs.append(dict(kwargs.get("env") or {}))
        return _ok_proc(stdout="GEAK_RESULT_LATENCY_MS=1.0\n")

    with mock.patch("subprocess.run", side_effect=_fake_run):
        collect_baseline_metrics(harness, repeats=1, work_dir=tmp_path, gpu_id=2)

    env = captured_envs[0]
    assert env["HIP_VISIBLE_DEVICES"] == "2"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["PYTHONPATH"].split(":")[0] == str(tmp_path)


def test_collect_baseline_metrics_raises_for_missing_harness(tmp_path: Path) -> None:
    missing = tmp_path / "nope.py"
    with pytest.raises(FileNotFoundError, match="harness"):
        collect_baseline_metrics(missing)


# ---------------------------------------------------------------------------
# collect_profile
# ---------------------------------------------------------------------------


_FAKE_PROFILE_PAYLOAD = {
    "backend": "metrix",
    "success": True,
    "results": [
        {
            "device_id": "0",
            "gpu_info": {"vendor": "AMD", "model": "MI300X"},
            "kernels": [
                {
                    "name": "add_kernel",
                    "duration_us": 42.5,
                    "bottleneck": "memory",
                    "metrics": {"duration_us": 42.5, "occupancy": 0.92},
                    "observations": ["memory-bound"],
                }
            ],
        }
    ],
}


def test_collect_profile_invokes_profiler_with_expected_command(tmp_path: Path) -> None:
    """The wrapper builds ``python3 <harness> --profile`` and forwards it to profiler-mcp."""
    harness = _make_harness(tmp_path)
    captured_kwargs: dict = {}

    def _fake_invoke(command, **kwargs):
        captured_kwargs["command"] = command
        captured_kwargs.update(kwargs)
        return _FAKE_PROFILE_PAYLOAD

    with mock.patch.object(baseline_mod, "_invoke_profiler_mcp", side_effect=_fake_invoke):
        result = collect_profile(harness, work_dir=tmp_path, gpu_id=3)

    assert isinstance(result, ProfileResult)
    assert "--profile" in captured_kwargs["command"]
    assert str(harness.resolve()) in captured_kwargs["command"]
    assert captured_kwargs["backend"] == "metrix"
    assert captured_kwargs["gpu_devices"] == "3"
    assert captured_kwargs["workdir"] == str(tmp_path)


def test_collect_profile_returns_profile_dict_on_success(tmp_path: Path) -> None:
    harness = _make_harness(tmp_path)
    with mock.patch.object(
        baseline_mod,
        "_invoke_profiler_mcp",
        return_value=_FAKE_PROFILE_PAYLOAD,
    ):
        result = collect_profile(harness)

    assert result.success is True
    assert result.profile is _FAKE_PROFILE_PAYLOAD
    assert result.backend == "metrix"
    assert result.profile["results"][0]["kernels"][0]["name"] == "add_kernel"


def test_collect_profile_returns_unsuccessful_when_profiler_returns_none(tmp_path: Path) -> None:
    harness = _make_harness(tmp_path)
    with mock.patch.object(baseline_mod, "_invoke_profiler_mcp", return_value=None):
        result = collect_profile(harness)

    assert result.profile is None
    assert result.success is False
    assert result.profile_path is None


def test_collect_profile_writes_out_path(tmp_path: Path) -> None:
    """When ``out_path`` is given, the profile JSON is written and parseable."""
    harness = _make_harness(tmp_path)
    out_path = tmp_path / "geak_output" / "profile.json"

    with mock.patch.object(
        baseline_mod,
        "_invoke_profiler_mcp",
        return_value=_FAKE_PROFILE_PAYLOAD,
    ):
        result = collect_profile(harness, out_path=out_path)

    assert result.profile_path == out_path.resolve()
    assert out_path.is_file()
    parsed = json.loads(out_path.read_text(encoding="utf-8"))
    assert parsed == _FAKE_PROFILE_PAYLOAD


def test_collect_profile_does_not_write_out_path_on_failure(tmp_path: Path) -> None:
    """A failed profile must not leave a stale JSON file behind."""
    harness = _make_harness(tmp_path)
    out_path = tmp_path / "out" / "profile.json"

    with mock.patch.object(baseline_mod, "_invoke_profiler_mcp", return_value=None):
        result = collect_profile(harness, out_path=out_path)

    assert result.profile_path is None
    assert not out_path.exists()


def test_collect_profile_raises_for_missing_harness(tmp_path: Path) -> None:
    missing = tmp_path / "no_harness.py"
    with pytest.raises(FileNotFoundError, match="harness"):
        collect_profile(missing)


def test_collect_profile_propagates_quick_and_replays(tmp_path: Path) -> None:
    """Custom ``num_replays`` / ``quick`` reach the profiler-mcp call."""
    harness = _make_harness(tmp_path)
    captured: dict = {}

    def _fake_invoke(command, **kwargs):
        captured.update(kwargs)
        return _FAKE_PROFILE_PAYLOAD

    with mock.patch.object(baseline_mod, "_invoke_profiler_mcp", side_effect=_fake_invoke):
        collect_profile(harness, num_replays=7, quick=True)

    assert captured["num_replays"] == 7
    assert captured["quick"] is True


# ---------------------------------------------------------------------------
# Live profiler-mcp test (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.profiler
def test_collect_profile_real_profiler_mcp(tmp_path: Path) -> None:  # pragma: no cover
    """End-to-end smoke test that requires a working profiler-mcp host.

    Skipped automatically when profiler-mcp can't be imported (most CI
    environments). Intended for manual / GPU-host runs.
    """
    pytest.importorskip("profiler_mcp.server")
    harness = _make_harness(tmp_path)
    # No subprocess will actually run because the harness body is empty
    # — but importing profiler-mcp end-to-end is the smoke test.
    result = collect_profile(harness, work_dir=tmp_path)
    assert isinstance(result, ProfileResult)
