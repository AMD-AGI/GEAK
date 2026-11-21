"""Unit tests for the GEAK-v4 source-level AVO handoff."""

import importlib.util
from pathlib import Path

import pytest


RUNNER_PATH = Path(__file__).resolve().parents[1] / "run_geak_avo.py"
SPEC = importlib.util.spec_from_file_location("run_geak_avo", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def test_maps_triton_as_source_and_injects_geak_paths(tmp_path):
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    config = RUNNER.make_config({
        "kernel_path": str(kernel),
        "exp_root": str(tmp_path / "out"),
        "target_language": "triton",
        "budget_s": 7200,
        "final_phase_reserve_s": 600,
        "gpu_ids": "1,2",
        "operator": "team",
    })

    args = config["args"]
    assert args["target_language"] == "triton"
    assert args["avo_language_level"] == "source"
    assert args["asm_mode"] == "false"
    assert args["ir_mode"] == "false"
    assert args["ir_feedback"] == "false"
    assert args["budget_s"] == 7200
    assert args["final_phase_reserve_s"] == 600
    assert args["gpu_ids"] == "1,2"
    assert args["operator"] == "team"
    assert args["operator_directions"] == 2
    assert args["start_policy"] == "diverse"
    assert args["start_k"] == 2
    assert Path(args["kernel_workflow_dir"]) == RUNNER.AVO_DIR / "kernel_base"
    assert Path(args["kernel_knowledge_dir"]) == RUNNER.ROOT / "perf_knowledge"
    assert (RUNNER.ROOT / "kernel_workflow" / "kernel_workflow.js").is_file()


def test_normalizes_natural_language_gpu_list_and_defaults(tmp_path):
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    args = RUNNER.make_config({
        "kernel_path": str(kernel),
        "gpu_ids": [1],
    })["args"]

    assert args["gpu_ids"] == "1"
    assert args["operator"] == "team"
    assert args["operator_directions"] == 2
    assert args["start_policy"] == "diverse"
    assert args["start_k"] == 2


@pytest.mark.parametrize(
    ("validation", "expected"),
    [
        ("accepted", "ok"),
        ("no_significant_gain", "no_gain"),
        ("reproduction_mismatch", "reproduction_mismatch"),
        ("validation_unavailable", "error"),
        ("infrastructure_unhealthy", "error"),
        ("correctness_failed", "error"),
    ],
)
def test_result_status_distinguishes_no_gain_from_failed_validation(
        tmp_path, validation, expected):
    exp_root = tmp_path / "exp"
    logs = exp_root / "_shim_logs"
    logs.mkdir(parents=True)
    (logs / "result.json").write_text(
        f'{{"validation_status": "{validation}"}}')
    result = RUNNER.result_for({
        "args": {
            "exp_root": str(exp_root),
            "kernel_path": str(tmp_path),
        },
    }, 0)
    assert result["status"] == expected


def test_result_forwards_artifact_and_roofline_contract(tmp_path):
    exp_root = tmp_path / "exp"
    logs = exp_root / "_shim_logs"
    logs.mkdir(parents=True)
    raw = {
        "validation_status": "accepted",
        "output_bundle": "/tmp/final_artifacts",
        "output_artifacts": ["/tmp/final_patch.diff"],
        "report_path": "/tmp/director_validation.json",
        "formal_report_path": "/tmp/benchmark_report.json",
        "roofline_fraction": 0.81,
        "roofline_regime": "memory-bound",
        "roofline_basis": "analytic",
        "practical_roofline_fraction": 0.994,
        "practical_roofline_regime": "mixed-streaming",
        "practical_roofline_evidence_path": "/tmp/roofline.md",
    }
    import json
    (logs / "result.json").write_text(json.dumps(raw))
    result = RUNNER.result_for({
        "args": {"exp_root": str(exp_root), "kernel_path": str(tmp_path)},
    }, 0)
    for key, value in raw.items():
        assert result[key] == value


@pytest.mark.parametrize(
    ("exit_code", "raw_run_id"),
    [(1, "current-run"), (0, "previous-run")],
)
def test_failed_or_mismatched_shim_does_not_forward_stale_result(
        tmp_path, exit_code, raw_run_id):
    exp_root = tmp_path / "exp"
    logs = exp_root / "_shim_logs"
    logs.mkdir(parents=True)
    (logs / "result.json").write_text(
        '{"validation_status":"accepted","output_bundle":"/stale/bundle",'
        '"output_artifacts":["/stale/patch.diff"],"final_geomean":9.9,'
        f'"_shim_run_id":"{raw_run_id}"}}')

    result = RUNNER.result_for({
        "run_id": "current-run",
        "args": {"exp_root": str(exp_root), "kernel_path": str(tmp_path)},
    }, exit_code)

    assert result["status"] == "error"
    assert result["validation_status"] == "error"
    assert result["output_bundle"] == ""
    assert result["output_artifacts"] == []
    assert result["final_speedup"] == 0


def test_reproduction_mismatch_keeps_director_speedup(tmp_path):
    exp_root = tmp_path / "exp"
    logs = exp_root / "_shim_logs"
    logs.mkdir(parents=True)
    (logs / "result.json").write_text(
        '{"validation_status":"reproduction_mismatch","final_geomean":1.9065}')
    result = RUNNER.result_for({
        "args": {"exp_root": str(exp_root), "kernel_path": str(tmp_path)},
    }, 0)
    assert result["status"] == "reproduction_mismatch"
    assert result["final_speedup"] == 1.9065


@pytest.mark.parametrize("target", ["asm", "amdgcn", "llvm-ir", "ttgir"])
def test_rejects_standalone_low_level_targets(tmp_path, target):
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    with pytest.raises(ValueError, match="source-level"):
        RUNNER.make_config({
            "kernel_path": str(kernel),
            "target_language": target,
        })


def test_rejects_non_source_language_level(tmp_path):
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    with pytest.raises(ValueError, match="source-level"):
        RUNNER.make_config({
            "kernel_path": str(kernel),
            "avo_language_level": "ir",
        })


@pytest.mark.parametrize(
    "nested",
    [
        {"target_language": "asm"},
        {"language_level": "ir"},
        {"asm_mode": "true"},
        {"ir_mode": True},
        {"ir_feedback": "true"},
    ],
)
def test_nested_args_cannot_bypass_source_only_guard(tmp_path, nested):
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    with pytest.raises(ValueError, match="source-level"):
        RUNNER.make_config({
            "kernel_path": str(kernel),
            "args": nested,
        })
