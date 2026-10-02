"""Exercise immutable dispatch-table selection across separate child processes."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from e2e_workflow.scripts.runtime_csv import (
    build_runtime_csv,
    verify_runtime_csv,
    verify_runtime_tuning,
)

ENV = "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE"
KEYS = ["gfx", "cu_num", "M", "N", "K"]
HEADER = "gfx,cu_num,M,N,K,kernelName\n"


def tables(tmp_path):
    baseline = tmp_path / "installed.csv"
    baseline.write_text(HEADER + "gfx950,256,64,32,16,stock_replaced\ngfx950,256,128,32,16,stock_retained\n")
    tuned = tmp_path / "tuned.csv"
    tuned.write_text(HEADER + "gfx950,256,64,32,16,tuned_replacement\ngfx950,256,256,32,16,tuned_new\n")
    return baseline, tuned


def test_complete_tables_isolate_baseline_and_candidate_children(tmp_path):
    baseline, tuned = tables(tmp_path)
    original = baseline.read_bytes(), tuned.read_bytes()
    output = tmp_path / "final" / "tuning" / "a table 'quoted' $literal"
    result = build_runtime_csv(baseline, [tuned], output, ENV, KEYS)
    results = []
    child = "import csv,json,os; print(json.dumps(list(csv.DictReader(open(os.environ['" + ENV + "'])))))"
    for arm in ("baseline_env", "apply_env", "baseline_env"):
        assignment, = shlex.split(result[arm])
        name, value = assignment.split("=", 1)
        proc = subprocess.run([sys.executable, "-S", "-c", child], env={**os.environ, name: value}, capture_output=True, text=True, check=True)
        results.append(json.loads(proc.stdout))
    assert [row["kernelName"] for row in results[0]] == ["stock_replaced", "stock_retained"]
    assert [row["kernelName"] for row in results[1]] == ["tuned_replacement", "stock_retained", "tuned_new"]
    assert results[0] == results[2]
    assert (baseline.read_bytes(), tuned.read_bytes()) == original
    assert result["baseline"]["rows"] == 2 and result["candidate"]["rows"] == 3
    assert result["replaced_rows"] == 1
    assert verify_runtime_csv(output / "runtime_csv.json", tmp_path) == result


@pytest.mark.parametrize("corruption", ["schema", "duplicate", "short_row", "missing_key"])
def test_ambiguous_tuning_is_rejected_before_creating_a_bundle(tmp_path, corruption):
    baseline, tuned = tables(tmp_path)
    if corruption == "schema":
        tuned.write_text("M,kernelName\n64,new\n")
    elif corruption == "duplicate":
        tuned.write_text(HEADER + "gfx950,256,64,32,16,new\n" * 2)
    elif corruption == "short_row":
        tuned.write_text(HEADER + "gfx950,256,64,32,16\n")
    else:
        tuned.write_text(HEADER + "gfx950,256,,32,16,new\n")
    output = tmp_path / "bundle"
    with pytest.raises(ValueError):
        build_runtime_csv(baseline, [tuned], output, ENV, KEYS)
    assert not output.exists()


def test_manifest_rejects_missing_or_changed_candidate_and_external_files(tmp_path):
    baseline, tuned = tables(tmp_path)
    output = tmp_path / "bundle"
    result = build_runtime_csv(baseline, [tuned], output, ENV, KEYS)
    manifest = output / "runtime_csv.json"
    candidate = Path(result["candidate"]["path"])
    candidate.chmod(0o644)
    candidate.write_text(HEADER)
    with pytest.raises(ValueError, match="changed"):
        verify_runtime_csv(manifest, tmp_path)
    candidate.unlink()
    with pytest.raises(FileNotFoundError):
        verify_runtime_csv(manifest, tmp_path)
    with pytest.raises(ValueError, match="outside"):
        verify_runtime_csv(manifest, tmp_path / "different-run")


def test_multiple_tuned_files_must_not_ambiguously_override_each_other(tmp_path):
    baseline, tuned = tables(tmp_path)
    with pytest.raises(ValueError, match="Multiple tuned files"):
        build_runtime_csv(baseline, [tuned, tuned], tmp_path / "bundle", ENV, KEYS)


def test_aiter_file_list_separator_is_rejected(tmp_path):
    baseline, tuned = tables(tmp_path)
    with pytest.raises(ValueError, match="separator"):
        build_runtime_csv(baseline, [tuned], tmp_path / "two:files", ENV, KEYS)


def test_numeric_key_spellings_replace_the_same_runtime_shape(tmp_path):
    baseline, tuned = tables(tmp_path)
    tuned.write_text(HEADER + "gfx950,256.0,6.4e1,32.0,16,selected\n")
    result = build_runtime_csv(baseline, [tuned], tmp_path / "bundle", ENV, KEYS)
    assert result["replaced_rows"] == 1
    assert result["candidate"]["rows"] == 2


def test_numeric_equivalent_duplicate_shapes_are_rejected(tmp_path):
    baseline, tuned = tables(tmp_path)
    tuned.write_text(HEADER + "gfx950,256,64,32,16,first\ngfx950,256.0,64.0,32,16,second\n")
    with pytest.raises(ValueError, match="duplicate dispatch key"):
        build_runtime_csv(baseline, [tuned], tmp_path / "bundle", ENV, KEYS)


@pytest.mark.parametrize("defect", ["missing", "wrong_env", "mixed_installer", "duplicate_selector"])
def test_tuning_admission_rejects_unreproducible_delivery(tmp_path, defect):
    baseline, tuned = tables(tmp_path)
    directory = tmp_path / "bundle"
    table = build_runtime_csv(baseline, [tuned], directory, ENV, KEYS)
    tuning = {"runtime_csv_manifests": [str(directory / "runtime_csv.json")], "apply_env": table["apply_env"]}
    if defect == "missing":
        Path(table["candidate"]["path"]).unlink()
    elif defect == "wrong_env":
        tuning["apply_env"] = table["baseline_env"]
    elif defect == "mixed_installer":
        tuning["live_tree_files"] = ["aiter/configs/shared.csv"]
    else:
        tuning["runtime_csv_manifests"] *= 2
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_runtime_tuning(tuning, tmp_path)


def test_workflow_verification_cli_reads_literal_paths(tmp_path):
    baseline, tuned = tables(tmp_path)
    directory = tmp_path / "a table '$literal'"
    table = build_runtime_csv(baseline, [tuned], directory, ENV, KEYS)
    tuning = {"runtime_csv_manifests": [str(directory / "runtime_csv.json")], "apply_env": table["apply_env"]}
    script = Path(__file__).parents[1] / "runtime_csv.py"
    child = subprocess.run([sys.executable, "-S", str(script), "--verify-tuning", "--eval-dir", str(tmp_path)],
                           input=json.dumps(tuning), text=True, capture_output=True, check=True)
    assert json.loads(child.stdout) == [table]


def test_cli_build_then_verify_preserves_literal_bundle_selection(tmp_path, monkeypatch, capsys):
    import io

    from e2e_workflow.scripts import runtime_csv

    baseline, tuned = tables(tmp_path)
    output = tmp_path / "bundle ' literal"
    monkeypatch.setattr(sys, "argv", [
        "runtime_csv.py", "--baseline", str(baseline), "--tuned", str(tuned),
        "--output", str(output), "--env-name", ENV, "--keys", ",".join(KEYS),
    ])
    runtime_csv.main()
    table = json.loads(capsys.readouterr().out)
    assert table["candidate"]["rows"] == 3
    tuning = {"runtime_csv_manifests": [str(output / "runtime_csv.json")], "apply_env": table["apply_env"]}
    monkeypatch.setattr(sys, "argv", ["runtime_csv.py", "--verify-tuning", "--eval-dir", str(tmp_path)])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(tuning)))
    runtime_csv.main()
    assert json.loads(capsys.readouterr().out) == [table]


@pytest.mark.parametrize("argv", [[], ["--verify-tuning"]])
def test_incomplete_cli_request_fails_before_building(argv, monkeypatch):
    from e2e_workflow.scripts import runtime_csv

    monkeypatch.setattr(sys, "argv", ["runtime_csv.py", *argv])
    with pytest.raises(SystemExit) as exc:
        runtime_csv.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("tuning", [
    {"live_tree_files": ["/workspace/aiter/configs/tuned.csv"]},
    {"cache_invalidation": ["clear aiter_configs cache"]},
])
def test_installed_table_changes_cannot_bypass_runtime_manifest(tuning, tmp_path):
    with pytest.raises(ValueError, match="complete per-process runtime CSVs"):
        verify_runtime_tuning(tuning, tmp_path)


@pytest.mark.parametrize("defect", ["selector", "keys", "empty", "nonfinite", "duplicate_columns"])
def test_invalid_dispatch_contract_never_creates_a_candidate(tmp_path, defect):
    baseline, tuned = tables(tmp_path)
    selector, keys = ENV, KEYS
    if defect == "selector":
        selector = "PATH"
    elif defect == "keys":
        keys = ["missing_column"]
    elif defect == "empty":
        tuned.write_text(HEADER)
    elif defect == "nonfinite":
        tuned.write_text(HEADER + "gfx950,256,NaN,32,16,bad\n")
    else:
        tuned.write_text("M,M\n1,2\n")
    with pytest.raises(ValueError):
        build_runtime_csv(baseline, [tuned], tmp_path / "bundle", selector, keys)
    assert not (tmp_path / "bundle").exists()
