# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-bearing results require the exact selected measurements' receipts."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from interface import run_e2e
from interface import test_source_measurement as measurements
from interface.source_materialization import SourceMaterializationError

raw_source_request = measurements.source_request


@pytest.fixture(autouse=True)
def preserve_environment():
    # The real dispatcher exports protocol/source variables. Keep those exports
    # real within each test and restore newly introduced names afterward too.
    with patch.dict(os.environ):
        yield


@pytest.fixture
def source_request(raw_source_request, tmp_path, monkeypatch):
    for key in run_e2e._SOURCE_RUN_ENV:
        monkeypatch.delenv(key, raising=False)
    args = {
        "eval_dir": str(tmp_path / "eval"),
        "baseline_source_request": json.loads(raw_source_request.read_text()),
    }
    staged = run_e2e.prepare_baseline_source(args)
    assert args["baseline_source_request_path"] == staged["request_path"]
    return Path(staged["request_path"])


def _handoff(request):
    spec = json.loads(request.read_text())
    spec.pop("schema_version")
    return {"baseline_env_spec": spec, "_geak_source_request_path": str(request)}


def _workflow(evaluation, *, final=110):
    launcher = evaluation / "final/final_launch.sh"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text('#!/usr/bin/env bash\nexit 0\n')
    return {
        "eval_dir": str(evaluation), "baseline_throughput_tok_s": 90,
        "final_throughput_tok_s": final, "throughput_speedup": final / 100,
        "accepted_config": {"flags": "--example 1"}, "output_parity": "pass",
    }


@pytest.mark.parametrize("disk_recovery", [False, True])
def test_selected_director_pair_uses_fresh_base_not_setup(source_request, tmp_path, disk_recovery):
    evaluation = measurements._pair(tmp_path, source_request)
    wf = _workflow(evaluation)
    if disk_recovery:
        wf["recovered_from_disk"] = True
    result = run_e2e.normalize_result(_handoff(source_request), wf)
    assert result["status"] == "ok"
    assert result["baseline_throughput_tok_s"] == 100
    assert result["final_throughput_tok_s"] == 110
    assert result["throughput_speedup"] == 1.1
    assert result["source_measurement"]["status"] == "verified"
    assert result["source_measurement"]["replay_status"] == "staged"
    assert Path(result["final_launch_script"]).is_file()


def test_missing_replay_launcher_withholds_even_a_verified_measurement(source_request, tmp_path):
    evaluation = measurements._pair(tmp_path, source_request)
    wf = _workflow(evaluation)
    (evaluation / "final/final_launch.sh").unlink()
    result = run_e2e.normalize_result(_handoff(source_request), wf)
    assert result["status"] == "error"
    assert result["source_measurement"]["status"] == "verified"
    assert result["source_measurement"]["replay_status"] == "unavailable"
    assert result["unverified_source_result"]["final_throughput_tok_s"] == 110


@pytest.mark.parametrize("leg", ["baseline", "final"])
def test_declared_overlay_must_match_the_selected_processes(source_request, tmp_path, leg):
    evaluation = measurements._pair(tmp_path, source_request)
    handoff, wf = _handoff(source_request), _workflow(evaluation)
    claimed = evaluation / "unmeasured_overlay"
    claimed.mkdir()
    (claimed / "sitecustomize.py").write_text("# an unmeasured startup overlay\n")
    if leg == "baseline":
        handoff["baseline_env_spec"]["overlay_pythonpath"] = str(claimed)
    else:
        wf["final_overlay"] = str(claimed)
    result = run_e2e.normalize_result(handoff, wf)
    assert result["status"] == "error"
    assert result["source_measurement"]["reason"] == "selected_overlay_roots_mismatch"


def test_cleanup_barrier_prevents_recovery_of_older_verified_gain(source_request, tmp_path):
    evaluation = measurements._pair(tmp_path, source_request)
    (source_request.parent / "source_cleanup_unverified.json").write_text("{}\n")
    result = run_e2e.normalize_result(_handoff(source_request), _workflow(evaluation))
    assert result["status"] == "error"
    assert result["source_measurement"]["reason"] == "source_cleanup_unverified"
    assert result["unverified_source_result"]["final_throughput_tok_s"] == 110


@pytest.mark.parametrize("damage", ["missing_seal", "wrong_number", "wrong_request", "wrong_manifest", "recovered_intermediate"])
def test_unverified_gain_keeps_diagnostics_without_publishable_adoption(source_request, tmp_path, monkeypatch, damage):
    evaluation = measurements._pair(tmp_path, source_request)
    handoff = _handoff(source_request)
    wf = _workflow(evaluation)
    if damage == "missing_seal":
        (evaluation / "validation/final/source_runtime/measurement.json").unlink()
    elif damage == "wrong_number":
        wf["final_throughput_tok_s"] = 120
    elif damage == "wrong_request":
        handoff.pop("_geak_source_request_path")
        # An inherited source request must not substitute for runner-owned binding.
        monkeypatch.setenv("GEAK_SOURCE_REQUEST", str(source_request))
    elif damage == "wrong_manifest":
        handoff["baseline_env_spec"]["source_materialization"]["manifest_sha256"] = "0" * 64
    else:
        wf["recovered_from_disk"] = True
        wf["recovered_intermediate"] = True
    result = run_e2e.normalize_result(handoff, wf)
    assert result["status"] == "error"
    assert result["error_class"] == "unresolved_baseline_source"
    assert result["source_measurement"]["status"] == "unavailable"
    assert "throughput_speedup" not in result
    assert "accepted_config" not in result
    assert result["unverified_source_result"]["final_throughput_tok_s"] == wf["final_throughput_tok_s"]
    assert result["unverified_source_result"]["accepted_config"]["flags"] == "--example 1"


def test_verified_regression_remains_a_regression(source_request, tmp_path):
    evaluation = measurements._pair(tmp_path, source_request)
    measurements._leaf(evaluation / "validation/final", source_request, 95)
    result = run_e2e.normalize_result(_handoff(source_request), _workflow(evaluation, final=95))
    assert result["status"] == "no_gain"
    assert result["throughput_speedup"] == 0.95
    assert result["source_measurement"]["status"] == "verified"


def test_controls_only_normalization_does_not_require_source_receipts(tmp_path):
    evaluation = tmp_path / "eval"
    evaluation.mkdir()
    result = run_e2e.normalize_result({}, _workflow(evaluation))
    assert result["status"] == "ok"
    assert "source_measurement" not in result


def test_no_source_preparation_clears_stale_private_environment(monkeypatch):
    for key in run_e2e._SOURCE_RUN_ENV:
        monkeypatch.setenv(key, "stale-from-another-run")
    assert run_e2e.prepare_baseline_source({}) == {}
    assert not run_e2e._SOURCE_RUN_ENV.intersection(os.environ)


def test_unverified_source_journey_cannot_advertise_accepted_kernels(tmp_path):
    wf = {"accepted_kernels": [{"short_name": "old-claim", "e2e_delta_pct": 20}]}
    result = {"status": "error", "error_class": "unresolved_baseline_source", "error": "missing_seal"}
    path = run_e2e._write_kernel_journey(tmp_path, wf, result)
    journey = json.loads(Path(path).read_text())
    assert journey["kernels"] == []
    assert journey["error_class"] == "unresolved_baseline_source"


@pytest.mark.parametrize("source_status", ["unavailable", "verified"])
def test_final_source_proof_does_not_invent_per_kernel_source_attribution(tmp_path, source_status):
    wf = {"accepted_kernels": [{"short_name": "old-claim", "e2e_delta_pct": 20}]}
    result = {
        "status": "ok" if source_status == "verified" else "error",
        "error_class": "normalize_failed" if source_status == "unavailable" else None,
        "source_measurement": {"status": source_status},
    }
    path = run_e2e._write_kernel_journey(tmp_path, wf, result)
    journey = json.loads(Path(path).read_text())
    assert journey["kernels"] == []
    assert journey["source_attribution"]["status"] == "unavailable"


def test_source_staging_failure_emits_atomic_interface_error(tmp_path):
    target = tmp_path / "results" / "result.json"
    error = SourceMaterializationError("conflicting_staged_asset")
    assert run_e2e._emit_source_preparation_error(target, error) == 1
    result = json.loads(target.read_text())
    assert result["error"] == str(error)
    assert result["status"] == "error"
    assert list(target.parent.iterdir()) == [target]


def test_source_relative_eval_dir_is_absolute_before_recipe_export(raw_source_request, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    handoff = _handoff(raw_source_request)
    handoff.update(schema_version=2, model_path="/models/example", exp_root=str(tmp_path),
                   eval_dir="relative-evaluation")
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(json.dumps(handoff))
    exported = []

    def launcher(h):
        exported.append(run_e2e._export_recipe_env(h, {"SGLANG_USE_AITER": "1"}, [], "test"))
        return "magpie"

    monkeypatch.setattr(run_e2e, "apply_bench_launcher", launcher)
    assert run_e2e.main([str(handoff_path), str(tmp_path / "result.json"), "--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    recipe = Path(exported[0])
    assert recipe.is_absolute()
    assert output["mapped_args"]["eval_dir"] == str(tmp_path / "relative-evaluation")
    monkeypatch.chdir(tmp_path / "relative-evaluation")
    assert recipe.read_bytes() == b"SGLANG_USE_AITER=1\0"


@pytest.mark.parametrize("existing", [False, True])
def test_unverified_source_report_marks_old_claims_as_diagnostic(tmp_path, existing):
    path = tmp_path / run_e2e.FINAL_REPORT_FILE
    prior = "# Agent report\n\nCandidate claimed +10%; see raw measurement artifacts.\n"
    if existing:
        path.write_text(prior)
    normalized = {
        "status": "error", "error_class": "unresolved_baseline_source",
        "source_measurement": {"status": "unavailable", "reason": "missing_seal"},
    }
    assert run_e2e._write_final_report_fallback(tmp_path, normalized, {}) == str(path)
    first = path.read_text()
    assert "No gain is eligible for adoption" in first
    assert "missing_seal" in first
    if existing:
        assert "Candidate claimed +10%; see raw measurement artifacts." in first
    run_e2e._write_final_report_fallback(tmp_path, normalized, {})
    assert path.read_text() == first
