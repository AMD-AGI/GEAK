# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify exact source and measurement bindings, including replica selection."""

import hashlib
import json
import os
import shutil
import statistics
from pathlib import Path

import pytest

from interface import source_measurement
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


def _leaf(directory, source_request, value, *, nonce=None, overlays=()):
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
    capsule_sha = _write(
        runtime / "launch.json",
        {
            "schema": "geak.source_runtime.launch.v1",
            "launch_nonce": bindings["launch_nonce"],
            "created_at_ns": 1,
            "boot_id": "boot",
            "request": json.loads(source_request.read_text()),
            "request_path": str(source_request),
            "request_sha256": bindings["request_sha256"],
            "manifest_sha256": bindings["manifest_sha256"],
            "accepted_roots": [str(root / "trees/0/python")],
            "overlay_roots": list(map(str, overlays)),
            "overlay_files": {
                str(path): _sha(path)
                for overlay in overlays
                for path in Path(overlay).rglob("*")
                if path.suffix == ".py" or path.name == "_overlay_manifest.json"
            },
            "helper_sha256": "c" * 64,
            "validator_sha256": "d" * 64,
        },
    )
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
            "launch_capsule_sha256": capsule_sha,
            "boot_id": "boot",
            "challenge_id": challenge,
            "observed_at_ns": when - 10,
            "accepted_roots": [str(root / "trees/0/python")],
            "overlay_roots": list(map(str, overlays)),
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
            "launch_capsule_sha256": capsule_sha,
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
            "launch_capsule": {"path": "launch.json", "sha256": capsule_sha},
            "server_identity": owner,
            "base_url": "http://127.0.0.1:30000",
            "artifacts": {
                name: _sha(directory / name)
                for name in ("bench_summary.json", "bench_runs.jsonl")
            },
        },
    )
    _write(
        runtime / "teardown.json",
        {
            "schema": "geak.source_runtime.cleanup.v1",
            **bindings,
            "status": "confirmed",
            "scope": "owned_cohort_teardown",
            "freeze_transport": "unix_datagram_scm_credentials",
            "launch_capsule_sha256": capsule_sha,
            "server_identity": owner,
            "observed_at_ns": 400,
            "processes": [{**owner, "exit_confirmed": True}],
            "unproven_processes": [],
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


def _rewrite_capsule(directory, mutate):
    runtime = directory / "source_runtime"
    path = runtime / "launch.json"
    capsule = json.loads(path.read_text())
    mutate(capsule)
    seal = json.loads((runtime / "measurement.json").read_text())
    seal["launch_capsule"]["sha256"] = _write(path, capsule)
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
@pytest.mark.parametrize("receipt", [False, True])
@pytest.mark.parametrize("digest", [None, "0" * 64])
def test_every_observation_requires_the_original_launch_capsule(
    source_request, tmp_path, phase, receipt, digest
):
    evaluation = _pair(tmp_path, source_request)

    def mutate(row):
        if digest is None:
            row.pop("launch_capsule_sha256")
        else:
            row["launch_capsule_sha256"] = digest

    _rewrite_gate(evaluation / "validation/final", phase, mutate, receipt=receipt)
    assert _verify(source_request, evaluation) == {
        "status": "unavailable",
        "reason": "observation_launch_capsule_binding_mismatch",
    }


def test_replaced_capsule_cannot_reuse_original_process_observations(
    source_request, tmp_path
):
    evaluation = _pair(tmp_path, source_request)
    _rewrite_capsule(
        evaluation / "validation/final",
        lambda row: row.update(helper_sha256="e" * 64),
    )
    assert _verify(source_request, evaluation) == {
        "status": "unavailable",
        "reason": "observation_launch_capsule_binding_mismatch",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "legacy"),
        ("status", "unverified"),
        ("scope", "escaped_descendants"),
        ("freeze_transport", "files"),
        ("launch_nonce", "another-launch"),
        ("launch_capsule_sha256", "0" * 64),
        ("request_sha256", "0" * 64),
        ("manifest_sha256", "0" * 64),
        ("server_identity", {}),
        ("observed_at_ns", 300),
        ("unproven_processes", [9999]),
        ("processes", []),
    ],
)
def test_persisted_measurement_requires_confirmed_launch_bound_teardown(
    source_request, tmp_path, field, value
):
    evaluation = _pair(tmp_path, source_request)
    path = evaluation / "validation/final/source_runtime/teardown.json"
    row = json.loads(path.read_text())
    row[field] = value
    _write(path, row)
    assert _verify(source_request, evaluation) == {
        "status": "unavailable",
        "reason": "unconfirmed_source_teardown",
    }


@pytest.mark.parametrize(
    "damage", ["missing", "symlink", "unexited", "missing_owner", "duplicate"]
)
def test_crash_or_incomplete_cleanup_cannot_leave_an_eligible_measurement(
    source_request, tmp_path, damage
):
    evaluation = _pair(tmp_path, source_request)
    path = evaluation / "validation/final/source_runtime/teardown.json"
    if damage == "missing":
        path.unlink()
    elif damage == "symlink":
        saved = path.with_name("saved-teardown.json")
        path.rename(saved)
        path.symlink_to(saved)
    else:
        row = json.loads(path.read_text())
        if damage == "unexited":
            row["processes"][0]["exit_confirmed"] = False
        elif damage == "missing_owner":
            row["processes"][0]["pid"] += 1
        else:
            row["processes"].append(dict(row["processes"][0]))
        _write(path, row)
    assert _verify(source_request, evaluation)["status"] == "unavailable"


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


@pytest.mark.parametrize(
    "kind", ["file", "symlink", "broken_symlink", "directory", "fifo"]
)
def test_cleanup_barrier_withholds_previously_valid_measurements(
    source_request, tmp_path, kind
):
    evaluation = _pair(tmp_path, source_request)
    assert _verify(source_request, evaluation)["status"] == "verified"
    marker = source_request.parent / "source_cleanup_unverified.json"
    if kind == "file":
        marker.write_text("opaque marker, not trusted JSON")
    elif kind == "symlink":
        marker.symlink_to(source_request)
    elif kind == "broken_symlink":
        marker.symlink_to(tmp_path / "absent")
    elif kind == "directory":
        marker.mkdir()
    else:
        os.mkfifo(marker)
    assert _verify(source_request, evaluation) == {
        "status": "unavailable",
        "reason": "source_cleanup_unverified",
    }


def test_cleanup_barrier_appearing_during_verification_refuses_success(
    source_request, tmp_path, monkeypatch
):
    evaluation = _pair(tmp_path, source_request)
    measurement = source_measurement._measurement

    def check_then_mark(*args, **kwargs):
        result = measurement(*args, **kwargs)
        (source_request.parent / "source_cleanup_unverified.json").touch()
        return result

    monkeypatch.setattr(source_measurement, "_measurement", check_then_mark)
    assert _verify(source_request, evaluation) == {
        "status": "unavailable",
        "reason": "source_cleanup_unverified",
    }


@pytest.mark.parametrize(
    "role,relative",
    [
        ("setup", "baseline"),
        ("baseline", "validation/base"),
        ("final", "validation/final"),
    ],
)
@pytest.mark.parametrize("phase", ["ready", "finished"])
def test_selected_overlay_must_match_every_role_and_gate(
    source_request, tmp_path, role, relative, phase
):
    evaluation = _pair(tmp_path, source_request)
    overlay = tmp_path / "different-overlay"
    overlay.mkdir()
    _rewrite_gate(
        evaluation / relative,
        phase,
        lambda row: row.update(overlay_roots=[str(overlay)]),
        receipt=True,
    )
    assert _verify(source_request, evaluation, **{f"expected_{role}_overlay": ""}) == {
        "status": "unavailable",
        "reason": "selected_overlay_roots_mismatch",
    }


def test_selected_overlay_matches_resolved_ordered_roots(source_request, tmp_path):
    evaluation = _pair(tmp_path, source_request)
    overlays = [tmp_path / "authored-one", tmp_path / "authored-two"]
    for path in overlays:
        path.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(overlays[0], target_is_directory=True)
    _leaf(evaluation / "validation/final", source_request, 110, overlays=overlays)
    expected = os.pathsep.join(map(str, [alias, overlays[1], overlays[0]]))
    result = _verify(
        source_request,
        evaluation,
        expected_setup_overlay="",
        expected_baseline_overlay="",
        expected_final_overlay=expected,
    )
    assert result["status"] == "verified"
    assert result["measurements"]["final"]["overlay_roots"] == list(map(str, overlays))
    assert _verify(
        source_request,
        evaluation,
        expected_final_overlay=os.pathsep.join(map(str, reversed(overlays))),
    ) == {"status": "unavailable", "reason": "selected_overlay_roots_mismatch"}


@pytest.mark.parametrize("expected", ["relative/path", ":/tmp", 42])
def test_invalid_expected_overlay_is_explicitly_unavailable(
    source_request, tmp_path, expected
):
    evaluation = _pair(tmp_path, source_request)
    assert _verify(source_request, evaluation, expected_final_overlay=expected) == {
        "status": "unavailable",
        "reason": "invalid_expected_overlay",
    }


def test_overlay_expected_but_missing_is_not_source_module_evidence(
    source_request, tmp_path
):
    evaluation = _pair(tmp_path, source_request)
    overlay = tmp_path / "expected-overlay"
    overlay.mkdir()
    assert _verify(source_request, evaluation, expected_final_overlay=str(overlay)) == {
        "status": "unavailable",
        "reason": "selected_overlay_roots_mismatch",
    }
    overlay.rmdir()
    assert _verify(source_request, evaluation, expected_final_overlay=str(overlay)) == {
        "status": "unavailable",
        "reason": "missing_expected_overlay",
    }


def test_each_selected_replica_requires_the_claimed_overlay(source_request, tmp_path):
    evaluation = _pair(tmp_path, source_request)
    final = evaluation / "validation/final"
    shutil.rmtree(final)
    _isolated(final, source_request, [110, 110])
    overlay = tmp_path / "different-replica-overlay"
    overlay.mkdir()
    _rewrite_gate(
        final / "replica_002/attempt_1",
        "finished",
        lambda row: row.update(overlay_roots=[str(overlay)]),
        receipt=True,
    )
    assert _verify(source_request, evaluation, expected_final_overlay="") == {
        "status": "unavailable",
        "reason": "selected_overlay_roots_mismatch",
    }


@pytest.mark.parametrize("damage", ["missing", "path", "hash", "symlink"])
def test_launch_capsule_must_be_sealed_with_each_measurement(
    source_request, tmp_path, damage
):
    evaluation = _pair(tmp_path, source_request)
    runtime = evaluation / "validation/final/source_runtime"
    seal = json.loads((runtime / "measurement.json").read_text())
    if damage == "missing":
        seal.pop("launch_capsule")
    elif damage == "path":
        seal["launch_capsule"]["path"] = "../launch.json"
    elif damage == "hash":
        seal["launch_capsule"]["sha256"] = "0" * 64
    else:
        original = runtime / "launch.json"
        moved = runtime / "saved-launch.json"
        original.rename(moved)
        original.symlink_to(moved)
    _write(runtime / "measurement.json", seal)
    assert _verify(source_request, evaluation)["status"] == "unavailable"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "legacy"),
        ("launch_nonce", "other-launch"),
        ("request_sha256", "0" * 64),
        ("manifest_sha256", "0" * 64),
        ("request_path", "/another/request.json"),
        ("request", {}),
        ("accepted_roots", []),
        ("boot_id", "other-boot"),
        ("created_at_ns", 200),
        ("overlay_files", {"/unrecorded/helper.py": "0" * 64}),
    ],
)
def test_rehashed_launch_capsule_cannot_change_source_or_launch_identity(
    source_request, tmp_path, field, value
):
    evaluation = _pair(tmp_path, source_request)
    _rewrite_capsule(
        evaluation / "validation/final", lambda row: row.update({field: value})
    )
    assert _verify(source_request, evaluation)["status"] == "unavailable"


@pytest.mark.parametrize(
    "damage",
    [
        "manifest",
        "startup_hook",
        "unowned_helper",
        "added_file",
        "removed_file",
        "file_symlink",
        "directory_symlink",
        "bytecode",
    ],
)
def test_same_overlay_path_requires_the_complete_measured_content(
    source_request, tmp_path, damage
):
    evaluation = _pair(tmp_path, source_request)
    overlay = tmp_path / "measured-overlay"
    overlay.mkdir()
    contents = {
        "sitecustomize.py": "# measured startup hook\n",
        "_overlay_manifest.json": '{"modules":{"alpha.changed":"patched.py"}}\n',
        "patched.py": "VALUE='measured-authored'\n",
        "helper.py": "VALUE='unowned-measured-helper'\n",
    }
    for name, content in contents.items():
        (overlay / name).write_text(content)
    _leaf(evaluation / "validation/final", source_request, 110, overlays=[overlay])
    assert (
        _verify(source_request, evaluation, expected_final_overlay=str(overlay))[
            "status"
        ]
        == "verified"
    )
    if damage == "manifest":
        (overlay / "_overlay_manifest.json").write_text('{"modules":{}}\n')
    elif damage == "startup_hook":
        (overlay / "sitecustomize.py").write_text(
            "# startup behavior changed after timing\n"
        )
    elif damage == "unowned_helper":
        (overlay / "helper.py").write_text("VALUE='changed-helper'\n")
    elif damage == "added_file":
        (overlay / "later.py").write_text("# absent during timing\n")
    elif damage == "removed_file":
        (overlay / "helper.py").unlink()
    elif damage == "file_symlink":
        (overlay / "linked.data").symlink_to(source_request)
    elif damage == "directory_symlink":
        (overlay / "linked-directory").symlink_to(
            tmp_path / "eval", target_is_directory=True
        )
    else:
        (overlay / "cached.pyc").write_bytes(b"bytecode")
    result = _verify(source_request, evaluation, expected_final_overlay=str(overlay))
    assert result["status"] == "unavailable"
    expected = (
        "symlink_in_overlay_inventory"
        if damage in {"file_symlink", "directory_symlink"}
        else "source_bytecode_present"
        if damage == "bytecode"
        else "sealed_overlay_inventory_mismatch"
    )
    assert result["reason"] == expected


def test_capsule_overlay_inventory_is_required_for_every_selected_replica(
    source_request, tmp_path
):
    evaluation = _pair(tmp_path, source_request)
    final = evaluation / "validation/final"
    shutil.rmtree(final)
    _isolated(final, source_request, [110, 110])
    (final / "replica_002/attempt_1/source_runtime/launch.json").write_text("{}\n")
    assert _verify(source_request, evaluation)["status"] == "unavailable"
