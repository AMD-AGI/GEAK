#!/usr/bin/env python3
"""Tests for the run-local validation environment manifest."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "capture_validation_env", HERE / "capture_validation_env.py",
)
ENV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENV)


def test_manifest_captures_environment_shape_and_decision_outcomes(monkeypatch):
    def package(name):
        if name == "flydsl":
            return {"version": "0.3.0", "module_path": "/opt/flydsl/flydsl/__init__.py"}
        return {"version": "0.1.0", "module_path": "/src/aiter/aiter/__init__.py"}

    monkeypatch.setattr(ENV, "package_probe", package)
    monkeypatch.setattr(
        ENV, "package_git",
        lambda package: {"root": "/src/aiter", "commit": "a" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        ENV, "git_probe",
        lambda path: {"root": str(path), "commit": "b" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        ENV, "rocm_probe",
        lambda: {"version": "7.2.0", "root": "/opt/rocm", "source": "/opt/rocm/.info/version"},
    )
    metadata = {
        "kernel": "gemm",
        "mode": "author",
        "requested_language": "flydsl",
        "observed_language": "flydsl",
        "device": "MI355X / gfx950",
        "gfx": "gfx950",
        "op_spec": {"shapes": {"m": 128, "n": 4096, "k": 7168}, "dtype": "bf16"},
        "measurement": {"method": "same-session paired A/B", "num_test_cases": 3},
        "validation": {"status": "accepted", "correctness": "pass", "speedup": 1.12},
        "decision_outcomes": [{"decision": "flydsl-half-mfma-call-forms", "verified": 1.12}],
    }

    got = ENV.build_manifest(metadata, "/workspace", "/geak/kernel_workflow", "0")
    assert got["flydsl_version"] == "0.3.0"
    assert got["rocm_version"] == "7.2.0"
    assert got["aiter_commit"] == "a" * 40
    assert got["geak_commit"] == "b" * 40
    assert got["provider"] == "standalone_flydsl"
    assert got["op_spec"]["shapes"]["k"] == 7168
    assert got["decision_outcomes"][0]["decision"] == "flydsl-half-mfma-call-forms"
    assert got["not_captured"] == []


def test_unknowns_are_explicit_not_captured(monkeypatch):
    monkeypatch.setattr(
        ENV, "package_probe",
        lambda name: {"version": ENV.NOT_CAPTURED, "module_path": ENV.NOT_CAPTURED},
    )
    monkeypatch.setattr(
        ENV, "package_git",
        lambda package: {"root": ENV.NOT_CAPTURED, "commit": ENV.NOT_CAPTURED,
                         "dirty": ENV.NOT_CAPTURED},
    )
    monkeypatch.setattr(
        ENV, "git_probe",
        lambda path: {"root": ENV.NOT_CAPTURED, "commit": ENV.NOT_CAPTURED,
                      "dirty": ENV.NOT_CAPTURED},
    )
    monkeypatch.setattr(
        ENV, "rocm_probe",
        lambda: {"version": ENV.NOT_CAPTURED, "root": ENV.NOT_CAPTURED,
                 "source": ENV.NOT_CAPTURED},
    )
    got = ENV.build_manifest({"kernel": "x"}, "/workspace", "/workflow", "")
    for field in ("observed_language", "flydsl_version", "rocm_version", "aiter_commit",
                  "geak_commit", "gpu", "gfx"):
        assert field in got["not_captured"]
        assert got[field] == ENV.NOT_CAPTURED


def test_json_fallback_is_still_valid_yaml(tmp_path, monkeypatch):
    # Exercise the output contract without depending on whether PyYAML is installed in CI.
    path = tmp_path / "validation_environment.yaml"
    data = {"schema_version": 1, "not_captured": ["rocm_version"]}
    ENV.dump_yaml(data, path)
    text = path.read_text()
    try:
        import yaml
    except ImportError:
        assert json.loads(text) == data
    else:
        assert yaml.safe_load(text) == data


def test_decision_registry_resolves_curated_and_config_ids(tmp_path):
    decisions = tmp_path / "corpus" / "decisions"
    evidence = tmp_path / "corpus" / "evidence"
    decisions.mkdir(parents=True)
    evidence.mkdir(parents=True)
    (decisions / "gemm.yaml").write_text(
        'cards:\n  - id: "flydsl-half-mfma-call-forms"\n',
    )
    (evidence / "gemm_tuned_configs.yaml").write_text(
        'tuned_configs:\n  - config_id: "cfg_0123456789abcdef"\n',
    )
    (decisions / "reduction.yaml").write_text(
        'cards:\n  - id: "flydsl-wave-reduction"\n',
    )
    (evidence / "attention_tuned_configs.yaml").write_text(
        'tuned_configs:\n  - config_id: "cfg_fedcba9876543210"\n',
    )
    assert ENV.decision_registry(tmp_path) == {
        "flydsl-half-mfma-call-forms", "flydsl-wave-reduction",
        "cfg_0123456789abcdef", "cfg_fedcba9876543210",
    }


def test_probe_resolves_against_the_workspace_pinned_stack(tmp_path, monkeypatch):
    """The manifest must describe the stack the benchmark pinned, not the ambient one.

    Regression: a run whose oracle pinned flydsl 0.3.4.1 + aiter_main was recorded as the box's
    ambient flydsl 0.1.5.dev504 + an older aiter, because the probe read the capture process's own
    sys.path. A provenance block naming the wrong commit reads as verified, so this is load-bearing.
    """
    pinned = tmp_path / "pyoverlay"
    (pinned / "pinnedpkg").mkdir(parents=True)
    (pinned / "pinnedpkg" / "__init__.py").write_text("")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "geak_pythonpath.txt").write_text(f"# task stack\n{pinned}\n")

    monkeypatch.setattr(sys, "path", list(sys.path))  # restored when the test ends
    seen = {}
    monkeypatch.setattr(
        ENV, "package_probe",
        lambda name: seen.setdefault(
            name,
            {"version": "0", "module_path": str(
                (importlib.util.find_spec("pinnedpkg") or SPEC).origin,
            )},
        ),
    )
    manifest = ENV.build_manifest({}, str(workspace), str(tmp_path), "0")

    assert manifest["software_detail"]["stack_pinned"] == [str(pinned)]
    assert manifest["software_detail"]["flydsl_module"].startswith(str(pinned))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
