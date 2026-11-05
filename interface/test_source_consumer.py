"""Source transport through the current interface and actual serving adapter."""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from interface import run_e2e
from interface.effective_config import resolve_effective_config
from interface.source_bundle import stage_source_bundle
from interface.source_materialization import (
    SourceMaterializationError,
    validate_source_materialization,
)
from interface.test_source_materialization import bundle as source_bundle_fixture

ROOT = Path(__file__).resolve().parents[1]
bundle = source_bundle_fixture


def _handoff(baseline, tmp_path):
    return {"schema_version": 2, "model_path": "/models/example", "framework": "sglang",
            "exp_root": str(tmp_path), "eval_dir": str(tmp_path / "evaluation"),
            "baseline_env_spec": baseline}


@pytest.fixture(autouse=True)
def _environment(monkeypatch):
    monkeypatch.delenv("GEAK_SOURCE_REQUEST", raising=False)
    monkeypatch.delenv("GEAK_EVAL_DIR", raising=False)
    monkeypatch.delenv("GEAK_GPU_IDENTITY_JSON", raising=False)


def test_interface_stages_complete_source_and_keeps_overlay_separate(bundle, tmp_path):
    baseline, _, original = bundle
    baseline["overlay_pythonpath"] = str(tmp_path / "authored_overlay")
    handoff = _handoff(baseline, tmp_path)
    args = run_e2e.map_args(handoff, dry_run=True)
    assert args["initial_overlay_pythonpath"] == baseline["overlay_pythonpath"]
    run_e2e.prepare_baseline_source(args)
    staged = validate_source_materialization(args["baseline_source_request"])
    assert staged.bundle_root.is_relative_to(Path(args["eval_dir"]))
    assert staged.required_layer_ids == ("a-first", "b-first", "a-second")
    shutil.rmtree(original)
    assert validate_source_materialization(args["baseline_source_request"]) == staged
    assert Path(os.environ["GEAK_SOURCE_REQUEST"]).is_file()


@pytest.mark.parametrize("candidate", [False, True])
def test_actual_sglang_adapter_uses_accepted_roots_after_overlay_replacement(bundle, tmp_path, candidate):
    baseline, _, original = bundle
    request = {"schema_version": 1, **baseline}
    evaluation = tmp_path / "evaluation"
    staged = stage_source_bundle(request, evaluation)
    shutil.rmtree(original)
    stock = tmp_path / "stock"
    (stock / "sglang").mkdir(parents=True)
    (stock / "sglang/__init__.py").write_text("")
    (stock / "sglang/launch_server.py").write_text(
        "import importlib.util,json,alpha.first,alpha.second,beta.other\n"
        "print(json.dumps({'values':[alpha.first.VALUE,alpha.second.VALUE,beta.other.VALUE],"
        "'origins':[alpha.first.__file__,alpha.second.__file__,beta.other.__file__],"
        "'deleted_absent':importlib.util.find_spec('alpha.old') is None}))\n")
    for name in ("alpha", "beta"):
        (stock / name).mkdir()
        (stock / name / "__init__.py").write_text("")
    for name in ("first", "second", "old"):
        (stock / "alpha" / (name + ".py")).write_text("VALUE='stock'\n")
    (stock / "beta/other.py").write_text("VALUE='stock'\n")
    overlay = tmp_path / "candidate"
    overlay.mkdir()
    if candidate:
        # A normal authored sitecustomize overlay changes no package ownership.
        (overlay / "sitecustomize.py").write_text("import os\nos.environ['OVERLAY_ENGAGED']='1'\n")
    log = tmp_path / "server.log"
    shell = '''set -eu
source "$SOURCE_PATHS"
source "$SGLANG_ADAPTER"
adapter_launch
wait "$SERVER_PID"
'''
    env = dict(os.environ, HERE=str(evaluation), SOURCE_PATHS=str(evaluation / "source_paths.sh"),
               SGLANG_ADAPTER=str(ROOT / "e2e_workflow/scripts/adapters/sglang.sh"),
               SGLANG_SRC_PYTHONPATH=str(stock), PYTHONPATH=str(stock),
               OVERLAY_PYTHONPATH=str(overlay) if candidate else "", GPU_ARCHS="gfx950",
               EXTRA_ENV="", EXTRA_SERVER_ARGS="", GPU="0", MODEL="unused", HOST="localhost",
               PORT="30000", TP="1", MEM_FRACTION="0.8", PROFILE_DIR=str(tmp_path), LOG=str(log),
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    # No rocminfo, torch or serving hardware is used by this miniature module.
    subprocess.run(["bash", "-c", shell], env=env, cwd=tmp_path,
                   capture_output=True, text=True, timeout=10, check=True)
    observed = json.loads(log.read_text())
    assert observed["values"] == ["accepted-first", "accepted-second", "accepted-other"]
    assert observed["deleted_absent"]
    served = validate_source_materialization(staged)
    assert all(Path(path).is_relative_to(served.bundle_root) for path in observed["origins"])


def test_staging_is_repeatable_and_refuses_changed_source(bundle, tmp_path):
    baseline, _, _ = bundle
    request = {"schema_version": 1, **baseline}
    evaluation = tmp_path / "evaluation"
    staged = stage_source_bundle(request, evaluation)
    assert stage_source_bundle(request, evaluation) == staged
    source = validate_source_materialization(staged)
    (source.bundle_root / "trees/a/python/alpha/first.py").write_text("VALUE='changed'\n")
    with pytest.raises(SourceMaterializationError, match="file_digest_mismatch"):
        stage_source_bundle(request, evaluation)


def test_no_source_and_sparse_source_are_distinct(tmp_path, monkeypatch):
    args = run_e2e.map_args(_handoff({}, tmp_path), dry_run=True)
    assert "baseline_source_request" not in args
    monkeypatch.setenv("GEAK_SOURCE_REQUEST", "/stale/request.json")
    run_e2e.prepare_baseline_source(args)
    assert "GEAK_SOURCE_REQUEST" not in os.environ
    assert not Path(args["eval_dir"]).exists()
    sparse = {"source_snapshots": [{"id": "a", "reproducible": True,
                                    "snapshot_dir": "/sparse"}]}
    with pytest.raises(SourceMaterializationError, match="missing_or_invalid_descriptor"):
        run_e2e.map_args(_handoff(sparse, tmp_path), dry_run=True)


def test_bundle_digest_changes_effective_identity(bundle, tmp_path):
    baseline, _, _ = bundle
    first = _handoff(baseline, tmp_path)
    second = copy.deepcopy(first)
    second["baseline_env_spec"]["source_materialization"]["manifest_sha256"] = "0" * 64
    assert resolve_effective_config(first).digest != resolve_effective_config(second).digest


def test_damaged_staged_bundle_refuses_before_adapter_launch(bundle, tmp_path):
    baseline, _, _ = bundle
    evaluation = tmp_path / "evaluation"
    request = stage_source_bundle({"schema_version": 1, **baseline}, evaluation)
    source = validate_source_materialization(request)
    (source.bundle_root / "trees/a/python/alpha/first.py").unlink()
    env = dict(os.environ, HERE=str(evaluation), SOURCE_PATHS=str(evaluation / "source_paths.sh"),
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", "-c", 'source "$SOURCE_PATHS" || exit $?; echo LAUNCHED'],
                            env=env, text=True, capture_output=True, timeout=10, check=False)
    assert result.returncode == 5
    assert "LAUNCHED" not in result.stdout
    assert "unresolved_baseline_source" in result.stderr
