from __future__ import annotations

import sys
import types

from minisweagent.run import pipeline_helpers
from minisweagent.run.preprocess import harness_utils
from minisweagent.run.utils.metrix_profile import build_metrix_profile_kwargs


def _install_fake_profiler(monkeypatch, calls: list[dict]) -> None:
    server = types.ModuleType("profiler_mcp.server")

    def _fake_profile_kernel(**kwargs):
        calls.append(kwargs)
        return {"results": []}

    server.profile_kernel = types.SimpleNamespace(fn=_fake_profile_kernel)
    package = types.ModuleType("profiler_mcp")
    package.server = server
    monkeypatch.setitem(sys.modules, "profiler_mcp", package)
    monkeypatch.setitem(sys.modules, "profiler_mcp.server", server)


def test_build_metrix_profile_kwargs_defaults_to_full_profile() -> None:
    kwargs = build_metrix_profile_kwargs("python harness.py --profile", 3)

    assert kwargs == {
        "command": "python harness.py --profile",
        "backend": "metrix",
        "num_replays": 3,
        "quick": False,
        "gpu_devices": "3",
    }


def test_build_metrix_profile_kwargs_keeps_optional_workdir() -> None:
    kwargs = build_metrix_profile_kwargs(
        "python harness.py --profile",
        "0",
        quick=True,
        workdir="/tmp/workdir",
    )

    assert kwargs["quick"] is True
    assert kwargs["workdir"] == "/tmp/workdir"


def test_pipeline_helpers_run_baseline_profile_uses_full_metrix(monkeypatch) -> None:
    calls: list[dict] = []
    _install_fake_profiler(monkeypatch, calls)
    monkeypatch.setattr(pipeline_helpers, "_ensure_mcp_importable", lambda: None)
    monkeypatch.setattr(pipeline_helpers, "extract_harness_path", lambda _cmd: "/tmp/test_harness.py")

    pipeline_helpers.run_baseline_profile("python /tmp/test_harness.py --benchmark", gpu_id=5)

    assert len(calls) == 1
    assert calls[0]["command"] == "python /tmp/test_harness.py --profile"
    assert calls[0]["backend"] == "metrix"
    assert calls[0]["quick"] is False
    assert calls[0]["gpu_devices"] == "5"


def test_harness_utils_run_baseline_profile_uses_full_metrix(monkeypatch) -> None:
    calls: list[dict] = []
    _install_fake_profiler(monkeypatch, calls)
    monkeypatch.setattr(harness_utils, "_ensure_mcp_importable", lambda: None)
    monkeypatch.setattr(harness_utils, "extract_harness_path", lambda _cmd: "/tmp/test_harness.py")

    harness_utils.run_baseline_profile("python /tmp/test_harness.py --benchmark", gpu_id=7)

    assert len(calls) == 1
    assert calls[0]["command"] == "python /tmp/test_harness.py --profile"
    assert calls[0]["backend"] == "metrix"
    assert calls[0]["quick"] is False
    assert calls[0]["gpu_devices"] == "7"
