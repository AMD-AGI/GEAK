# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify exact source and measurement bindings, including replica selection."""

import hashlib
import json
import shutil
import statistics
from pathlib import Path

import pytest

from interface.source_measurement import verify_normalized_source_measurements


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return _sha(path)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source_request(tmp_path):
    root = tmp_path / "bundle"
    files = []
    for relative, content in {
        "trees/0/python/alpha/__init__.py": "",
        "trees/0/python/alpha/changed.py": "VALUE='accepted'\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        path.chmod(0o644)
        files.append({"path": relative, "sha256": _sha(path), "mode": 0o644})
    manifest = {
        "schema_version": 1,
        "required_layer_ids": ["first", "second"],
        "pythonpath_prefixes": ["trees/0/python"],
        "trees": [
            {
                "tree_id": "zero",
                "root": "trees/0",
                "accepted_commit": "a" * 40,
                "layer_ids": ["first", "second"],
            }
        ],
        "files": files,
        "deleted_paths": ["trees/0/python/alpha/deleted.py"],
        "modules": [
            {"name": "alpha.changed", "path": "trees/0/python/alpha/changed.py"}
        ],
        "deleted_modules": ["alpha.deleted"],
    }
    digest = _write(root / "manifest.json", manifest)
    descriptor = {
        "schema_version": 1,
        "status": "ready",
        "bundle_root": str(root),
        "manifest_path": "manifest.json",
        "manifest_sha256": digest,
        "required_layer_ids": ["first", "second"],
        "pythonpath_prefixes": ["trees/0/python"],
    }
    path = tmp_path / "source_request.json"
    _write(
        path,
        {
            "schema_version": 1,
            "source_snapshots": [{"id": "first"}, {"id": "second"}],
            "source_materialization": descriptor,
        },
    )
    return path


def _leaf(directory, source_request, value, *, nonce=None):
    descriptor = json.loads(source_request.read_text())["source_materialization"]
    root = Path(descriptor["bundle_root"])
    manifest = json.loads((root / "manifest.json").read_text())
    runtime = directory / "source_runtime"
    owner = {"pid": 1234, "pgid": 1234, "start_ticks": 9876}
    bindings = {
        "status": "verified",
        "launch_nonce": nonce or str(directory),
        "request_sha256": _sha(source_request),
        "manifest_sha256": descriptor["manifest_sha256"],
    }
    gates = {}
    for phase, when in (("ready", 100), ("finished", 300)):
        challenge = phase + "-challenge"
        name = f"process-1234-9876-{challenge}.json"
        row = {
            "name": "alpha.changed",
            "origin": str(root / manifest["modules"][0]["path"]),
            "sha256": manifest["files"][1]["sha256"],
            "source": "accepted_source",
        }
        receipt = {
            "schema": "geak.source_runtime.process.v1",
            **bindings,
            **owner,
            "boot_id": "boot",
            "challenge_id": challenge,
            "observed_at_ns": when - 10,
            "accepted_roots": [str(root / "trees/0/python")],
            "overlay_roots": [],
            "sys_path_sha256": "b" * 64,
            "loaded_modules": [row],
            "resolved_modules": [row],
            "deleted_modules_absent": ["alpha.deleted"],
            "guard": "owned_module_specs_unchanged",
            "worker_topology": "frozen_after_ready",
            "subreaper": True,
            "cache_policy": "source_bytecode_absent",
            "overlay_inventory": "exact_python_manifest_no_symlinks",
        }
        receipt_digest = _write(runtime / name, receipt)
        gate = {
            "schema": "geak.source_runtime.gate.v1",
            **bindings,
            "phase": phase,
            "transport": "unix_datagram_scm_credentials",
            "server_identity": owner,
            "boot_id": "boot",
            "base_url": "http://127.0.0.1:30000",
            "challenge_id": challenge,
            "observed_at_ns": when,
            "processes": [{**owner, "receipt": name, "sha256": receipt_digest}],
            "listener_pids": [1234],
        }
        gates[phase] = {
            "path": f"gate-{phase}.json",
            "sha256": _write(runtime / f"gate-{phase}.json", gate),
        }
    _write(
        directory / "bench_summary.json",
        {
            "throughput_tok_s_median": value,
            "runs": 1,
            "measurement_mode": "warm_server",
            "all_throughput": [value],
        },
    )
    (directory / "bench_runs.jsonl").write_text(
        json.dumps({"output_throughput": value}) + "\n"
    )
    _write(
        runtime / "measurement.json",
        {
            "schema": "geak.source_runtime.measurement.v1",
            "measurement_scope": "hot_timed_rounds",
            **bindings,
            "gates": gates,
            "server_identity": owner,
            "base_url": "http://127.0.0.1:30000",
            "artifacts": {
                name: _sha(directory / name)
                for name in ("bench_summary.json", "bench_runs.jsonl")
            },
        },
    )


def _pair(tmp_path, source_request):
    evaluation = tmp_path / "eval"
    for relative, value in (
        ("baseline", 90),
        ("validation/base", 100),
        ("validation/final", 110),
    ):
        _leaf(evaluation / relative, source_request, value)
    return evaluation


def _verify(source_request, evaluation, **overrides):
    descriptor = json.loads(source_request.read_text())["source_materialization"]
    args = {
        "eval_dir": evaluation,
        "result_source": "workflow_return",
        "baseline_basis_source": "validation_base_bench_summary",
        "setup_tput": 90,
        "baseline_tput": 100,
        "final_tput": 110,
        "expected_manifest_sha256": descriptor["manifest_sha256"],
        "expected_required_layer_ids": descriptor["required_layer_ids"],
    }
    args.update(overrides)
    return verify_normalized_source_measurements(source_request, **args)


def _rewrite_gate(directory, phase, mutate, *, receipt=False):
    runtime = directory / "source_runtime"
    seal = json.loads((runtime / "measurement.json").read_text())
    gate_path = runtime / seal["gates"][phase]["path"]
    gate = json.loads(gate_path.read_text())
    if receipt:
        path = runtime / gate["processes"][0]["receipt"]
        doc = json.loads(path.read_text())
        mutate(doc)
        gate["processes"][0]["sha256"] = _write(path, doc)
    else:
        mutate(gate)
    seal["gates"][phase]["sha256"] = _write(gate_path, gate)
    _write(runtime / "measurement.json", seal)


def test_exact_canonical_measurements_are_verified(source_request, tmp_path):
    result = _verify(source_request, _pair(tmp_path, source_request))
    assert result["status"] == "verified"
    assert result["request_sha256"] == _sha(source_request)
    assert result["required_layer_ids"] == ["first", "second"]
    assert {
        role: row["throughput_tok_s"] for role, row in result["measurements"].items()
    } == {"setup": 90, "baseline": 100, "final": 110}


@pytest.mark.parametrize("phase", ["ready", "finished"])
@pytest.mark.parametrize(
    "field,receipt",
    [
        ("transport", False),
        ("worker_topology", True),
        ("subreaper", True),
        ("cache_policy", True),
        ("overlay_inventory", True),
    ],
)
def test_rehashed_legacy_receipts_cannot_qualify(
    source_request, tmp_path, phase, field, receipt
):
    evaluation = _pair(tmp_path, source_request)
    _rewrite_gate(
        evaluation / "validation/final",
        phase,
        lambda row: row.pop(field),
        receipt=receipt,
    )
    assert _verify(source_request, evaluation)["status"] == "unavailable"


@pytest.mark.parametrize("scope", [None, "preparation", "profiling_only"])
def test_only_hot_timed_rounds_seals_can_qualify(source_request, tmp_path, scope):
    evaluation = _pair(tmp_path, source_request)
    path = evaluation / "validation/final/source_runtime/measurement.json"
    seal = json.loads(path.read_text())
    if scope is None:
        seal.pop("measurement_scope")
    else:
        seal["measurement_scope"] = scope
    _write(path, seal)
    assert _verify(source_request, evaluation)["status"] == "unavailable"


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_manifest_sha256", "0" * 64),
        ("expected_required_layer_ids", ["second", "first"]),
        ("final_tput", 111),
        ("baseline_tput", float("nan")),
    ],
)
def test_other_source_or_published_numbers_cannot_qualify(
    source_request, tmp_path, field, value
):
    result = _verify(source_request, _pair(tmp_path, source_request), **{field: value})
    assert result["status"] == "unavailable"


@pytest.mark.parametrize(
    "damage", ["seal", "ready", "finished", "receipt", "summary", "runs", "symlink"]
)
def test_missing_tampered_or_linked_evidence_is_unavailable(
    source_request, tmp_path, damage
):
    evaluation = _pair(tmp_path, source_request)
    directory = evaluation / "validation/final"
    paths = {
        "seal": directory / "source_runtime/measurement.json",
        "ready": directory / "source_runtime/gate-ready.json",
        "finished": directory / "source_runtime/gate-finished.json",
        "receipt": directory / "source_runtime/process-1234-9876-ready-challenge.json",
        "summary": directory / "bench_summary.json",
        "runs": directory / "bench_runs.jsonl",
    }
    if damage == "symlink":
        original = directory / "source_runtime/measurement.json"
        backup = directory / "backup.json"
        original.rename(backup)
        original.symlink_to(backup)
    else:
        paths[damage].write_text("{}\n")
    assert _verify(source_request, evaluation)["status"] == "unavailable"


@pytest.mark.parametrize(
    "field,value",
    [
        ("manifest_sha256", "0" * 64),
        ("request_sha256", "0" * 64),
        ("server_identity", {"pid": 2222, "pgid": 2222, "start_ticks": 1}),
        ("listener_pids", [9999]),
        ("observed_at_ns", 50),
        ("phase", "ready"),
        ("processes", []),
    ],
)
def test_rehashed_gate_still_requires_correct_bindings(
    source_request, tmp_path, field, value
):
    evaluation = _pair(tmp_path, source_request)
    _rewrite_gate(
        evaluation / "validation/final",
        "finished",
        lambda row: row.update({field: value}),
    )
    assert _verify(source_request, evaluation)["status"] == "unavailable"


@pytest.mark.parametrize(
    "field,value",
    [
        ("deleted_modules_absent", []),
        ("resolved_modules", []),
        ("observed_at_ns", 50),
        ("challenge_id", "ready-challenge"),
        ("accepted_roots", ["/wrong"]),
        ("guard", "disabled"),
    ],
)
def test_rehashed_process_receipt_cannot_drop_source_coverage(
    source_request, tmp_path, field, value
):
    evaluation = _pair(tmp_path, source_request)
    _rewrite_gate(
        evaluation / "validation/final",
        "finished",
        lambda row: row.update({field: value}),
        receipt=True,
    )
    assert _verify(source_request, evaluation)["status"] == "unavailable"


@pytest.mark.parametrize(
    "field,value",
    [
        ("origin", "/outside/stock.py"),
        ("sha256", "0" * 64),
        ("source", "stock"),
        ("name", "alpha.wrong"),
    ],
)
def test_rehashed_module_receipts_require_the_recorded_source(
    source_request, tmp_path, field, value
):
    evaluation = _pair(tmp_path, source_request)

    def mutate(row):
        row["resolved_modules"][0][field] = value

    _rewrite_gate(evaluation / "validation/final", "finished", mutate, receipt=True)
    assert _verify(source_request, evaluation)["status"] == "unavailable"


@pytest.mark.parametrize(
    "provenance",
    [
        "disk_intermediate_win",
        "disk_no_gain_synthesis",
        "disk_e2e_checkpoint_final_pair",
        "disk_tuning_skillset_legacy_provisional",
        "disk_stack_provisional",
    ],
)
def test_unmapped_recovery_cannot_borrow_matching_canonical_numbers(
    source_request, tmp_path, provenance
):
    result = _verify(
        source_request, _pair(tmp_path, source_request), result_source=provenance
    )
    assert result == {
        "status": "unavailable",
        "reason": "unsupported_measurement_provenance",
    }


def test_director_only_baseline_has_no_implicit_measurement_mapping(
    source_request, tmp_path
):
    assert (
        _verify(
            source_request,
            _pair(tmp_path, source_request),
            baseline_basis_source="director_base_block",
        )["status"]
        == "unavailable"
    )


def _isolated(directory, source_request, values):
    replicas = []
    runs = []
    for index, value in enumerate(values, 1):
        selected = directory / f"replica_{index:03d}"
        leaf = selected / "attempt_1"
        _leaf(leaf, source_request, value)
        shutil.copyfile(leaf / "bench_summary.json", selected / "selected_summary.json")
        (selected / "selected_attempt").write_text("1\n")
        replicas.append({"replica": index, "attempt": 1, "throughput_tok_s": value})
        runs.append((leaf / "bench_runs.jsonl").read_bytes())
    _write(
        directory / "bench_summary.json",
        {
            "measurement_mode": "isolated_server",
            "replicas": replicas,
            "requested_replicas": len(values),
            "successful_replicas": len(values),
            "runs": len(values),
            "status": "complete",
            "usable_for_acceptance": True,
            "all_throughput": values,
            "throughput_tok_s_median": round(statistics.median(values), 3),
        },
    )
    (directory / "bench_runs.jsonl").write_bytes(b"".join(runs))


def test_isolated_aggregate_binds_only_selected_replica_measurements(
    source_request, tmp_path
):
    evaluation = _pair(tmp_path, source_request)
    directory = evaluation / "validation/final"
    shutil.rmtree(directory)
    _isolated(directory, source_request, [109, 110, 111])
    _leaf(directory / "replica_001/attempt_2", source_request, 999)
    result = _verify(source_request, evaluation)
    assert result["status"] == "verified"
    assert result["measurements"]["final"]["throughput_tok_s"] == 110
    assert [
        row["attempt"] for row in result["measurements"]["final"]["selected_replicas"]
    ] == [1, 1, 1]


@pytest.mark.parametrize(
    "damage",
    ["selection", "copy", "runs", "median", "missing_seal", "duplicate_replica"],
)
def test_isolated_aggregate_never_falls_back_to_another_attempt(
    source_request, tmp_path, damage
):
    evaluation = _pair(tmp_path, source_request)
    directory = evaluation / "validation/final"
    shutil.rmtree(directory)
    _isolated(directory, source_request, [109, 110, 111])
    _leaf(directory / "replica_001/attempt_2", source_request, 109)
    if damage == "selection":
        (directory / "replica_001/selected_attempt").write_text("2\n")
    elif damage == "copy":
        (directory / "replica_001/selected_summary.json").write_text("{}\n")
    elif damage == "runs":
        (directory / "bench_runs.jsonl").write_text("{}\n")
    elif damage == "missing_seal":
        (directory / "replica_001/attempt_1/source_runtime/measurement.json").unlink()
    else:
        path = directory / "bench_summary.json"
        row = json.loads(path.read_text())
        if damage == "median":
            row["throughput_tok_s_median"] = 999
        else:
            row["replicas"][1] = row["replicas"][0]
        _write(path, row)
    assert _verify(source_request, evaluation)["status"] == "unavailable"


def test_copied_replica_seals_do_not_establish_independent_launches(
    source_request, tmp_path
):
    evaluation = _pair(tmp_path, source_request)
    directory = evaluation / "validation/final"
    shutil.rmtree(directory)
    _isolated(directory, source_request, [110, 110, 110])
    shutil.rmtree(directory / "replica_002/attempt_1")
    shutil.copytree(
        directory / "replica_001/attempt_1", directory / "replica_002/attempt_1"
    )
    result = _verify(source_request, evaluation)
    assert result == {"status": "unavailable", "reason": "duplicate_replica_launch"}


def test_relocated_source_content_keeps_handoff_identity(source_request, tmp_path):
    doc = json.loads(source_request.read_text())
    original = Path(doc["source_materialization"]["bundle_root"])
    destination = tmp_path / "moved"
    shutil.copytree(original, destination)
    shutil.rmtree(original)
    doc["source_materialization"]["bundle_root"] = str(destination)
    _write(source_request, doc)
    assert (
        _verify(source_request, _pair(tmp_path, source_request))["status"] == "verified"
    )
