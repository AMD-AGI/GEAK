"""The public entrypoint keeps per-import policy and replaceable recovery hooks."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load(name, monkeypatch, threshold):
    monkeypatch.setenv("GEAK_AGENT_AUTO", "0")
    monkeypatch.setenv("GEAK_SAME_CONFIG_DIVERGENCE_WARN_PCT", str(threshold))
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name("run_e2e.py")
    )
    module = importlib.util.module_from_spec(spec)
    # Callers historically load this script without registering it in sys.modules.
    spec.loader.exec_module(module)
    return module


def test_cached_helpers_do_not_share_entrypoint_thresholds(monkeypatch, tmp_path):
    strict = _load("e2e_strict", monkeypatch, 1.0)
    lenient = _load("e2e_lenient", monkeypatch, 10.0)
    wf = {"eval_dir": str(tmp_path)}

    assert strict._build_baseline_alignment(5.0)["status"] == "warning"
    assert lenient._build_baseline_alignment(5.0)["status"] == "aligned"
    assert strict.normalize_result({}, wf)["baseline_alignment"]["warning_threshold_pct"] == 1.0
    assert lenient.normalize_result({}, wf)["baseline_alignment"]["warning_threshold_pct"] == 10.0

    monkeypatch.setattr(strict, "SAME_CONFIG_DIVERGENCE_WARN_PCT", 2.0)
    assert strict.normalize_result({}, wf)["baseline_alignment"]["warning_threshold_pct"] == 2.0
    assert lenient.normalize_result({}, wf)["baseline_alignment"]["warning_threshold_pct"] == 10.0


def test_normalization_uses_current_entrypoint_recovery_hooks(monkeypatch, tmp_path):
    runner = _load("e2e_recovery_hooks", monkeypatch, 3.0)
    classify = runner._workload_comparability
    calls = []

    def recover(path):
        calls.append(("recover", path))
        return None

    def workload(handoff):
        calls.append(("workload", handoff))
        return classify(handoff)

    monkeypatch.setattr(runner, "_recover_best_intermediate_win", recover)
    monkeypatch.setattr(runner, "_workload_comparability", workload)
    handoff = {}
    runner.normalize_result(handoff, {
        "eval_dir": str(tmp_path),
        "accepted_kernels": [{"e2e_delta_pct": 1.0}],
    })

    assert calls == [("recover", tmp_path), ("workload", handoff)]
