"""Read an existing external evaluation for the authentic GEAK control.

The installed owner injects its existing ArenaEvaluator. This reader launches no
evaluation and grants no measurement, custody or reward qualification.
"""

import argparse
import copy
from pathlib import Path
import subprocess
import tempfile

from . import result
from ._util import canonical_json, require, sha256_hex
from .archive.store import _path, _read
from .evaluate.evaluator import ArenaEvaluator
from .evaluate.external import validate_objective


def _need(value, reason):
    require(value, reason, ValueError)


def _snapshot(root, paths):
    root = Path(root).absolute()
    _need(root.resolve(strict=True) == root, "canonical workspace required")
    return {name: _read(root, _path(name)) for name in paths}


def _git(root, patch, *options):
    # No inherited Git configuration or work-tree override may redirect apply.
    completed = subprocess.run(
        ["git", "apply", *options, "-"], cwd=root, input=patch,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1",
             "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CEILING_DIRECTORIES": str(root.parent)})
    _need(completed.returncode == 0,
          "patch could not be checked/applied: " + completed.stderr.decode("utf-8", errors="replace").strip())
    return completed.stdout


def _patched(source, patch):
    """Apply the captured patch only to a disposable declared source closure."""
    with tempfile.TemporaryDirectory(prefix="geak-control-source-") as directory:
        root = Path(directory)
        for name, (raw, mode) in source.items():
            _need(".git" not in Path(name).parts, "Git metadata is outside the patchable source closure")
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            path.chmod(0o644 | mode)
        if patch:
            # Forward/reverse numstat names the destination/source of renames.
            # NUL records preserve quoted names, spaces and tabs without guessing
            # at the human-readable diff header grammar.
            for direction in ((), ("--reverse",)):
                stats = _git(root, patch, "--numstat", "-z", *direction)
                _need(stats.endswith(b"\0"), "patch has no declared file changes")
                for row in stats[:-1].split(b"\0"):
                    fields = row.split(b"\t", 2)
                    _need(len(fields) == 3, "invalid patch file inventory")
                    name = _path(fields[2].decode("utf-8"))
                    _need(name in source, "patch touches a file outside the source closure: " + name)
            _git(root, patch, "--whitespace=nowarn")
        files = set()
        for path in root.rglob("*"):
            _need(not path.is_symlink(), "patched source contains a symlink")
            if not path.is_dir():
                files.add(path.relative_to(root).as_posix())
        _need(files == set(source), "patched source closure differs")
        return _snapshot(root, source)


def read_control_result(evaluator, candidate_id, *, evaluation_id, workspace, patch=None):
    """Return checked native metrics for the supplied source or source+patch."""
    _need(isinstance(evaluator, ArenaEvaluator), "existing Arena evaluator required")
    record = result.read_evaluation(evaluator, candidate_id, evaluation_id=evaluation_id)
    request, receipt = record["request"], record["receipt"]
    contract = request["contract"]
    _need(contract == evaluator.contract and contract["schema"] == "geak-arena-external-task-v1",
          "external UT contract required")
    _need(request["binding"]["candidate_id"] == candidate_id and request["operation_id"] == evaluation_id,
          "external evaluation identity differs")
    _need(receipt["settled"] is True and receipt["native_suite_pass_observed"] is True
          and not receipt["collection_errors"] and not receipt["host_errors"],
          "external evaluation did not complete a settled passing native suite")
    observed = receipt["observed_runtime"]
    _need(all(observed[key] == contract["runtime"][key] for key in ("image", "config_digest"))
          and observed["physical_devices"], "external runtime/device evidence is missing or differs")
    timing = record["timing_context"]
    _need(timing and timing["timing_basis"] == "external_whole_call_ut"
          and timing["request_id"] == evaluation_id and timing["evidence"],
          "settled external timing evidence required")
    native = timing["diagnostics"]
    objective = {"kind": "external_primary", "primary_regime": native["primary_regime"]}
    validate_objective(contract, objective)

    paths = contract["source_paths"]
    rows = request["source"]["files"]
    _need(len(rows) == len(paths) and {row["path"] for row in rows} == set(paths),
          "evaluated source closure differs")
    source = _snapshot(workspace, paths)
    patch_path = Path(patch).absolute() if patch is not None else None
    if patch_path is not None:
        _need(patch_path.parent.resolve(strict=True) == patch_path.parent, "canonical patch path required")
        patch_snapshot = _read(patch_path.parent, patch_path.name)
        checked = _patched(source, patch_snapshot[0])
    else:
        checked = source
    for row in rows:
        raw, mode = checked[row["path"]]
        _need(sha256_hex(raw) == row["blob_id"] and mode == row["executable"],
              "source differs from evaluated candidate: " + row["path"])
    _need(_snapshot(workspace, paths) == source, "workspace changed during source verification")
    if patch_path is not None:
        _need(_read(patch_path.parent, patch_path.name) == patch_snapshot,
              "patch changed during source verification")

    pairs = timing["paired_cases"]
    return {"objective": objective,
            **{key: copy.deepcopy(native[key]) for key in
               ("primary_regime", "primary_speedup", "native_reproducible_gain",
                "native_reproducible_regression", "phase_speedups")},
            "per_case": [{"name": name, **{key: pairs[name][key] for key in
                          ("baseline_ms", "optimized_ms", "speedup")}}
                         for name in contract["benchmark_case_ids"]],
            "external_evaluation": {"candidate_id": candidate_id, "evaluation_id": evaluation_id},
            "receipt_ref": copy.deepcopy(record["receipt_ref"]),
            "timing_basis": "external_whole_call_ut", "external_measurement_valid": True}


def main(argv=None, *, evaluator):
    """Callable CLI for an installed owner entrypoint with an injected evaluator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--patch", type=Path)
    args = parser.parse_args(argv)
    value = read_control_result(evaluator, args.candidate_id, evaluation_id=args.evaluation_id,
                                workspace=args.workspace, patch=args.patch)
    print(canonical_json(value).decode("utf-8"))
    return 0
