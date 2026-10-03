# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-import and corruption checks for accepted-source bundle consumption."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from interface.source_materialization import (
    SourceMaterializationError,
    validate_source_materialization,
)


def seal(baseline, manifest):
    descriptor = baseline["source_materialization"]
    data = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    (Path(descriptor["bundle_root"]) / "manifest.json").write_bytes(data)
    descriptor["manifest_sha256"] = hashlib.sha256(data).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "accepted"
    contents = {
        "trees/a/python/alpha/__init__.py": "",
        "trees/a/python/alpha/first.py": "VALUE = 'accepted-first'\n",
        "trees/a/python/alpha/second.py": "VALUE = 'accepted-second'\n",
        "trees/a/python/alpha/data.txt": "unchanged package data\n",
        "trees/b/src/beta/__init__.py": "",
        "trees/b/src/beta/other.py": "VALUE = 'accepted-other'\n",
    }
    files = []
    for name, data in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
        path.chmod(0o644)
        files.append({"path": name, "mode": 0o644,
                      "sha256": hashlib.sha256(data.encode()).hexdigest()})
    layers = ["a-first", "b-first", "a-second"]
    prefixes = ["trees/a/python", "trees/b/src"]
    manifest = {
        "schema_version": 1, "required_layer_ids": layers,
        "pythonpath_prefixes": prefixes,
        "trees": [
            {"tree_id": "a", "root": "trees/a", "accepted_commit": "a" * 40,
             "layer_ids": ["a-first", "a-second"]},
            {"tree_id": "b", "root": "trees/b", "accepted_commit": "b" * 40,
             "layer_ids": ["b-first"]},
        ],
        "files": files, "deleted_paths": ["trees/a/python/alpha/old.py"],
        "modules": [
            {"name": "alpha.first", "path": "trees/a/python/alpha/first.py"},
            {"name": "alpha.second", "path": "trees/a/python/alpha/second.py"},
            {"name": "beta.other", "path": "trees/b/src/beta/other.py"},
        ],
        "deleted_modules": ["alpha.old"],
    }
    baseline = {
        "source_snapshots": [{"id": item} for item in layers],
        "source_materialization": {
            "schema_version": 1, "status": "ready", "bundle_root": str(root),
            "manifest_path": "manifest.json", "manifest_sha256": "",
            "required_layer_ids": layers[:], "pythonpath_prefixes": prefixes[:],
        },
    }
    seal(baseline, manifest)
    return baseline, manifest, root


def test_complete_tree_imports_cumulative_edits_and_deletion_after_relocation(bundle, tmp_path):
    baseline, manifest, root = bundle
    original = validate_source_materialization(baseline)
    assert original.required_layer_ids == ("a-first", "b-first", "a-second")
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    shutil.rmtree(root)
    baseline["source_materialization"]["bundle_root"] = str(relocated)
    result = validate_source_materialization(baseline)
    assert result.manifest_sha256 == original.manifest_sha256
    stock = tmp_path / "stock"
    for package in ("alpha", "beta"):
        (stock / package).mkdir(parents=True)
        (stock / package / "__init__.py").write_text("")
    for module in ("first", "second", "old"):
        (stock / "alpha" / (module + ".py")).write_text("VALUE='stock'\n")
    (stock / "beta" / "other.py").write_text("VALUE='stock'\n")
    code = """import importlib.util,json,alpha.first,alpha.second,beta.other
from pathlib import Path
print(json.dumps({'values':[alpha.first.VALUE,alpha.second.VALUE,beta.other.VALUE],
'origins':[alpha.first.__file__,alpha.second.__file__,beta.other.__file__],
'deleted_absent':importlib.util.find_spec('alpha.old') is None,
'data':Path(alpha.__file__).with_name('data.txt').read_text()}))
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((*result.pythonpath_prefixes, str(stock))),
               PYTHONDONTWRITEBYTECODE="1")
    run = subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path,
                         text=True, capture_output=True, timeout=10, check=True)
    observed = json.loads(run.stdout)
    assert observed["values"] == ["accepted-first", "accepted-second", "accepted-other"]
    assert observed["deleted_absent"]
    assert all(Path(path).is_relative_to(relocated) for path in observed["origins"])
    assert observed["data"] == "unchanged package data\n"
    # Actual import must not introduce unrecorded bytecode into the sealed tree.
    assert validate_source_materialization(baseline).manifest == manifest


def test_no_source_is_compatible():
    assert validate_source_materialization({}) is None
    assert validate_source_materialization({"source_snapshots": []}) is None


@pytest.mark.parametrize("snapshot", [
    {"id": "a", "reproducible": True, "snapshot_dir": "/missing"},
    {"id": "a", "reproducible": False},
])
def test_legacy_sparse_layers_are_explicitly_unresolved(snapshot):
    with pytest.raises(SourceMaterializationError, match="missing_or_invalid_descriptor"):
        validate_source_materialization({"source_snapshots": [snapshot]})


@pytest.mark.parametrize("field,value,reason", [
    ("schema_version", True, "unsupported_schema_version"),
    ("schema_version", 2, "unsupported_schema_version"),
    ("status", "partial", "materialization_not_ready"),
    ("required_layer_ids", ["a-second", "b-first", "a-first"], "layer_coverage_mismatch"),
    ("pythonpath_prefixes", ["trees/b/src", "trees/a/python"], "import_prefix_mismatch"),
    ("manifest_path", "../manifest.json", "invalid_manifest_path"),
    ("manifest_sha256", "A" * 64, "invalid_manifest_digest"),
    ("manifest_sha256", "0" * 64, "manifest_digest_mismatch"),
    ("bundle_root", "relative", "invalid_bundle_root"),
])
def test_descriptor_corruption(bundle, field, value, reason):
    baseline, _, _ = bundle
    baseline["source_materialization"][field] = value
    with pytest.raises(SourceMaterializationError, match=reason):
        validate_source_materialization(baseline)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "trees//a", "trees/./a",
                                  "trees/a/../b", "trees/a:else", "trees/a\\b", "trees/a\nb"])
def test_manifest_paths_cannot_escape_or_encode_a_path_list(bundle, path):
    baseline, manifest, _ = bundle
    manifest["files"][0]["path"] = path
    seal(baseline, manifest)
    with pytest.raises(SourceMaterializationError, match="invalid_relative_path"):
        validate_source_materialization(baseline)


@pytest.mark.parametrize("kind,reason", [
    ("missing", "missing_bundle_path"), ("content", "file_digest_mismatch"),
    ("mode", "file_mode_mismatch"), ("extra", "incomplete_file_inventory"),
    ("deletion", "deleted_path_present"), ("symlink", "symlink_in_bundle"),
    ("fifo", "non_regular_bundle_file"),
])
def test_real_filesystem_damage_is_refused(bundle, kind, reason):
    baseline, manifest, root = bundle
    path = root / manifest["files"][1]["path"]
    if kind == "missing":
        path.unlink()
    elif kind == "content":
        path.write_text("VALUE='corrupt'\n")
    elif kind == "mode":
        path.chmod(0o755)
    elif kind == "extra":
        path.with_name("unrecorded.py").write_text("VALUE='extra'\n")
    elif kind == "deletion":
        (root / manifest["deleted_paths"][0]).write_text("VALUE='resurrected'\n")
        # Inventory includes it; the deletion check must still reject it.
        manifest["files"].append({"path": manifest["deleted_paths"][0], "mode": 0o644,
                                  "sha256": hashlib.sha256(b"VALUE='resurrected'\n").hexdigest()})
        seal(baseline, manifest)
        reason = "conflicting_deletion"
    elif kind == "symlink":
        path.unlink()
        path.symlink_to("second.py")
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path)
    with pytest.raises(SourceMaterializationError, match=reason):
        validate_source_materialization(baseline)


@pytest.mark.parametrize("kind,reason", [
    ("duplicate_layer", "layer_coverage_mismatch"),
    ("layer_order", "layer_order_mismatch"),
    ("missing_layer", "layer_coverage_mismatch"),
    ("tree_order", "tree_order_mismatch"),
    ("wrong_module", "module_name_mismatch"),
    ("missing_deleted_module", "deleted_module_coverage_mismatch"),
    ("duplicate_file", "duplicate_file_row"),
])
def test_manifest_cannot_misrepresent_layer_or_module_coverage(bundle, kind, reason):
    baseline, manifest, _ = bundle
    if kind == "duplicate_layer":
        manifest["trees"][1]["layer_ids"].append("a-second")
    elif kind == "layer_order":
        manifest["trees"][0]["layer_ids"].reverse()
    elif kind == "missing_layer":
        manifest["trees"][0]["layer_ids"].pop()
    elif kind == "tree_order":
        manifest["trees"].reverse()
    elif kind == "wrong_module":
        manifest["modules"][0]["name"] = "beta.other"
    elif kind == "missing_deleted_module":
        manifest["deleted_modules"] = []
    else:
        manifest["files"].append(manifest["files"][0])
    seal(baseline, manifest)
    with pytest.raises(SourceMaterializationError, match=reason):
        validate_source_materialization(baseline)


def test_duplicate_json_keys_are_rejected_even_with_matching_hash(bundle):
    baseline, _, root = bundle
    data = b'{"schema_version":1,"schema_version":1}'
    (root / "manifest.json").write_bytes(data)
    baseline["source_materialization"]["manifest_sha256"] = hashlib.sha256(data).hexdigest()
    with pytest.raises(SourceMaterializationError, match="duplicate_manifest_key"):
        validate_source_materialization(baseline)


def _add_file(root, manifest, name, content=b"", mode=0o644):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(mode)
    manifest["files"].append({"path": name, "mode": mode,
                              "sha256": hashlib.sha256(content).hexdigest()})


def test_complete_tree_can_include_unchanged_tooling_and_executable_mode(bundle):
    baseline, manifest, root = bundle
    _add_file(root, manifest, "trees/a/setup.py", b"# packaging only\n", 0o755)
    seal(baseline, manifest)
    assert validate_source_materialization(baseline).manifest == manifest


@pytest.mark.parametrize("kind,reason", [
    ("namespace", "unsupported_namespace_package"),
    ("native", "unsupported_runtime_artifact"),
    ("bytecode", "unsupported_runtime_artifact"),
    ("top_level", "unsupported_top_level_module"),
    ("ownership", "ambiguous_import_ownership"),
    ("empty_directory", "unlisted_bundle_directory"),
    ("directory_symlink", "symlink_in_bundle"),
    ("package_deletion", "unsupported_package_deletion"),
])
def test_unsupported_import_semantics_are_explicit(bundle, kind, reason):
    baseline, manifest, root = bundle
    if kind == "namespace":
        _add_file(root, manifest, "trees/a/python/alpha/namespace/member.py")
    elif kind == "native":
        _add_file(root, manifest, "trees/a/python/alpha/_native.so", b"\x7fELF")
    elif kind == "bytecode":
        _add_file(root, manifest, "trees/a/python/alpha/old.pyc")
    elif kind == "top_level":
        _add_file(root, manifest, "trees/a/python/top.py")
        manifest["modules"].append({"name": "top", "path": "trees/a/python/top.py"})
    elif kind == "ownership":
        _add_file(root, manifest, "trees/b/src/alpha/__init__.py")
    elif kind == "empty_directory":
        (root / "trees/a/python/alpha/old").mkdir()
    elif kind == "directory_symlink":
        (root / "trees/a/python/alpha/alias").symlink_to(root / "trees/b/src/beta",
                                                       target_is_directory=True)
    elif kind == "package_deletion":
        manifest["deleted_paths"].append("trees/a/python/alpha/removed/__init__.py")
        manifest["deleted_modules"].append("alpha.removed")
    seal(baseline, manifest)
    with pytest.raises(SourceMaterializationError, match=reason):
        validate_source_materialization(baseline)


def test_manifest_fifo_is_refused_without_opening_it(bundle):
    baseline, _, root = bundle
    (root / "manifest.json").unlink()
    os.mkfifo(root / "manifest.json")
    with pytest.raises(SourceMaterializationError, match="non_regular_bundle_file"):
        validate_source_materialization(baseline)


def test_reuse_revalidates_instead_of_trusting_previous_ready_status(bundle):
    baseline, _, root = bundle
    assert validate_source_materialization(baseline)
    (root / "trees/a/python/alpha/second.py").write_text("VALUE='later-drift'\n")
    with pytest.raises(SourceMaterializationError, match="file_digest_mismatch"):
        validate_source_materialization(baseline)


def test_truncated_json_with_updated_digest_has_typed_error(bundle):
    baseline, _, root = bundle
    data = b'{"schema_version":'
    (root / "manifest.json").write_bytes(data)
    baseline["source_materialization"]["manifest_sha256"] = hashlib.sha256(data).hexdigest()
    with pytest.raises(SourceMaterializationError, match="unreadable_or_invalid_bundle"):
        validate_source_materialization(baseline)


def test_repository_import_root_can_include_ancillary_top_level_python(bundle):
    baseline, manifest, root = bundle
    _add_file(root, manifest, "trees/a/python/setup.py", b"# unrelated setup script\n")
    _add_file(root, manifest, "trees/a/python/tests/test_tool.py", b"# unrelated test\n")
    seal(baseline, manifest)
    assert validate_source_materialization(baseline)


def test_handoff_mapping_keeps_authored_overlay_separate_and_binds_source_digest(bundle, tmp_path):
    from interface import run_e2e
    from interface.effective_config import resolve_effective_config

    baseline, _, _ = bundle
    baseline["overlay_pythonpath"] = str(tmp_path / "authored-overlay")
    handoff = {"schema_version": 2, "model_path": "/models/model",
               "exp_root": str(tmp_path / "geak"), "baseline_env_spec": baseline}
    mapped = run_e2e.map_args(handoff)
    source = validate_source_materialization(baseline)
    assert mapped["baseline_source_pythonpath"] == os.pathsep.join(source.pythonpath_prefixes)
    assert mapped["initial_overlay_pythonpath"] == baseline["overlay_pythonpath"]
    assert mapped["baseline_source_request"]["source_materialization"] == baseline["source_materialization"]
    assert resolve_effective_config(handoff).manifest["source_materialization_sha256"] == source.manifest_sha256


def test_interface_reports_unresolved_source_before_workflow_dispatch(tmp_path):
    from interface import run_e2e

    handoff = tmp_path / "handoff.json"
    handoff.write_text(json.dumps({"schema_version": 2, "model_path": "/models/model",
                                  "exp_root": str(tmp_path / "geak"),
                                  "baseline_env_spec": {"source_snapshots": [{"id": "lost"}]}}))
    result = tmp_path / "output" / "result.json"
    assert run_e2e.main([str(handoff), str(result)]) == 1
    value = json.loads(result.read_text())
    assert value["status"] == "error"
    assert value["error_class"] == "unresolved_baseline_source"
    assert "missing_or_invalid_descriptor" in value["error"]


@pytest.mark.parametrize("filename", ["kernel.hsaco", "kernel.SO", "kernel.so.1.2", "kernel.pyd",
                                     "kernel.DLL", "kernel.o", "kernel.a", "kernel.CO",
                                     "kernel.so.debug", "kernel.dll.1"])
def test_native_artifact_spellings_cannot_be_accepted_as_package_data(bundle, filename):
    baseline, manifest, root = bundle
    _add_file(root, manifest, "trees/a/python/alpha/" + filename)
    seal(baseline, manifest)
    with pytest.raises(SourceMaterializationError, match="unsupported_runtime_artifact"):
        validate_source_materialization(baseline)


@pytest.mark.parametrize("filename", ["sitecustomize.py", "usercustomize.py",
                                     "sitecustomize.pyc", "sitecustomize/__init__.py"])
def test_bundle_cannot_install_unrecorded_interpreter_startup_hooks(bundle, filename):
    baseline, manifest, root = bundle
    _add_file(root, manifest, "trees/a/python/" + filename)
    seal(baseline, manifest)
    with pytest.raises(SourceMaterializationError, match="unsupported_startup_hook"):
        validate_source_materialization(baseline)


@pytest.mark.parametrize("kind,reason", [
    ("deleted_namespace", "unsupported_namespace_package"),
    ("deleted_package_replacement", "ambiguous_module_ownership"),
    ("live_module_package_collision", "ambiguous_module_ownership"),
    ("empty_tree", "tree_without_source_modules"),
])
def test_remaining_false_ready_boundaries(bundle, kind, reason):
    baseline, manifest, root = bundle
    if kind == "deleted_namespace":
        manifest["deleted_paths"].append("trees/a/python/alpha/gone/old.py")
        manifest["deleted_modules"].append("alpha.gone.old")
    elif kind == "deleted_package_replacement":
        _add_file(root, manifest, "trees/a/python/alpha/old/__init__.py")
    elif kind == "live_module_package_collision":
        _add_file(root, manifest, "trees/a/python/alpha/first/__init__.py")
    elif kind == "empty_tree":
        baseline["source_snapshots"].append({"id": "empty"})
        baseline["source_materialization"]["required_layer_ids"].append("empty")
        baseline["source_materialization"]["pythonpath_prefixes"].append("trees/empty")
        manifest["required_layer_ids"].append("empty")
        manifest["pythonpath_prefixes"].append("trees/empty")
        manifest["trees"].append({"tree_id": "empty", "root": "trees/empty",
                                  "accepted_commit": "c" * 40, "layer_ids": ["empty"]})
        (root / "trees/empty").mkdir()
    seal(baseline, manifest)
    with pytest.raises(SourceMaterializationError, match=reason):
        validate_source_materialization(baseline)


def test_source_bearing_legacy_schema_cannot_bypass_source_validation(bundle, tmp_path):
    from interface import run_e2e

    baseline, _, _ = bundle
    handoff = {"schema_version": 1, "model_path": "/models/model",
               "exp_root": str(tmp_path / "geak"), "baseline_env_spec": baseline}
    with pytest.raises(SourceMaterializationError, match="source_requires_handoff_schema_v2"):
        run_e2e.map_args(handoff)


@pytest.mark.parametrize("suffix", ["support/__init__.py", "support/value.py", "support.py", "setup.py"])
def test_unchanged_dependency_collisions_between_roots_are_unresolved(bundle, suffix):
    baseline, manifest, root = bundle
    for prefix in manifest["pythonpath_prefixes"]:
        _add_file(root, manifest, prefix + "/" + suffix, b"VALUE='different-under-each-root'\n")
    seal(baseline, manifest)
    with pytest.raises(SourceMaterializationError, match="ambiguous_import_ownership"):
        validate_source_materialization(baseline)


@pytest.mark.parametrize("route", ["sglang", "vllm", "magpie"])
@pytest.mark.parametrize("authored_overlay", [False, True])
def test_real_launcher_imports_materialized_source_before_stock(bundle, tmp_path, route, authored_overlay):
    """Actual adapters execute Python children; no GPU/framework is required."""
    baseline, manifest, root = bundle
    scripts = Path(__file__).resolve().parents[1] / "e2e_workflow" / "scripts"
    code = b"""import importlib.util,json
from alpha import first,second
from beta import other
print(json.dumps({'values':[first.VALUE,second.VALUE,other.VALUE],
'origins':[first.__file__,second.__file__,other.__file__],
'deleted_absent':importlib.util.find_spec('alpha.old') is None}))
"""
    backend = "vllm" if route == "vllm" else "sglang"
    entry = "__main__.py" if backend == "vllm" else "launch_server.py"
    _add_file(root, manifest, f"trees/a/python/{backend}/__init__.py")
    _add_file(root, manifest, f"trees/a/python/{backend}/{entry}", code)
    seal(baseline, manifest)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"schema_version": 1, **baseline}))

    stock = tmp_path / "stock"
    (stock / backend).mkdir(parents=True)
    (stock / backend / "__init__.py").write_text("")
    (stock / backend / entry).write_text("raise RuntimeError('stock server selected')\n")
    (stock / "alpha").mkdir()
    (stock / "alpha/__init__.py").write_text("")
    (stock / "alpha/old.py").write_text("VALUE='stock'\n")
    overlay = tmp_path / "overlay"
    if authored_overlay:
        patch = tmp_path / "authored_first.py"
        patch.write_text("VALUE='authored-first'\n")
        subprocess.run([sys.executable, str(scripts / "overlay_setup.py"), "add-module",
                        "--overlay", str(overlay), "--module", "alpha.first",
                        "--patched-file", str(patch)], check=True, capture_output=True, timeout=10)

    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "python").symlink_to(sys.executable)
    vllm = binary / "vllm"
    vllm.write_text('#!/bin/bash\nexec "$TEST_PYTHON" -m vllm "$@"\n')
    vllm.chmod(0o755)
    magpie = tmp_path / "magpie.sh"
    # This fixture completes its import before returning. Keep the launcher's
    # parent shell as a live diagnostic PID; no serving identity is claimed by
    # this test (the source-runtime suite owns that separate contract).
    magpie.write_text('#!/bin/bash\nset -e\npython -m sglang.launch_server > "$SERVER_LOG" 2>&1\necho "$PPID" > "$MAGPIE_SERVER_PID_FILE"\n')
    output = tmp_path / "out"
    output.mkdir()
    log = output / "server.log"
    adapter = scripts / "adapters" / ("launchers/magpie.sh" if route == "magpie" else route + ".sh")
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
               TEST_PYTHON=sys.executable, PYTHONPATH=str(stock),
               PYTHONDONTWRITEBYTECODE="1", GEAK_SOURCE_REQUEST=str(request),
               SGLANG_SRC_PYTHONPATH=str(stock), GPU_ARCHS="gfx000", GPU="0",
               MODEL="unused-model", HOST="127.0.0.1", PORT="39999", TP="1",
               MEM_FRACTION="0.8", EXTRA_SERVER_ARGS="", EXTRA_ENV="", PROFILE="0",
               PROFILE_DIR="", OUT_DIR=str(output), LOG=str(log), BACKEND=backend,
               OVERLAY_PYTHONPATH=str(overlay) if authored_overlay else "",
               MAGPIE_LAUNCH_SCRIPT=str(magpie), ADAPTER=str(adapter))
    for name in ("RECIPE_ENV_FILE", "SERVER_LAUNCH_PREFIX", "GEAK_SOURCE_BOOTSTRAP_PYTHONPATH"):
        env.pop(name, None)
    driver = 'set -eu\nsource "$ADAPTER"\nadapter_launch\n'
    if route != "magpie":
        driver += 'wait "$SERVER_PID"\n'
    subprocess.run(["bash", "-c", driver], env=env, cwd=tmp_path,
                   text=True, capture_output=True, timeout=15, check=True)
    observed = json.loads(log.read_text().splitlines()[-1])
    assert observed["values"] == ["authored-first" if authored_overlay else "accepted-first",
                                   "accepted-second", "accepted-other"]
    assert observed["deleted_absent"]
    assert Path(observed["origins"][1]).is_relative_to(root)
    assert Path(observed["origins"][2]).is_relative_to(root)
    # The real import did not dirty the materialized source with bytecode.
    assert validate_source_materialization(baseline)
