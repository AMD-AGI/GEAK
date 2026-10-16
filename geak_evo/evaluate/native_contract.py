"""Validate native Arena declarations without task imports or tensor loads.

The shared job loader checks small artifacts and exact fixture declarations.
The host owns fixture admission and native execution.
"""
import contextlib
import os
from pathlib import Path
import stat

from .jobs import (_canonical, _decode, _ids, _keys, _path,
                   _pointer, _revision, _sha, _text)
from .._util import require, sha256_hex


SCHEMA = "geak-arena-native-operator-v2"
SMALL_FILE_LIMIT = 8 * 1024 * 1024
SMALL_TOTAL_LIMIT = 32 * 1024 * 1024


def digest(value):
    return sha256_hex(_canonical(value))


@contextlib.contextmanager
def regular(path):
    """Open every path component without following links, including parents."""
    path = Path(path).absolute()
    require(not any(part in (".", "..") for part in path.parts), "noncanonical file path", ValueError)
    with contextlib.ExitStack() as stack:
        fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, fd)
        for part in path.parts[1:-1]:
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            stack.callback(os.close, fd)
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        stream = stack.enter_context(os.fdopen(fd, "rb"))
        require(stat.S_ISREG(os.fstat(fd).st_mode), "regular file required", ValueError)
        yield stream


def _read(path):
    """Stable bounded read for contracts and native JSON, never tensor bodies."""
    with regular(path) as stream:
        before = os.fstat(stream.fileno())
        require(before.st_size <= SMALL_FILE_LIMIT, "oversize small artifact", ValueError)
        raw = stream.read(SMALL_FILE_LIMIT + 1)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")
    require(len(raw) <= SMALL_FILE_LIMIT and len(raw) == after.st_size
          and all(getattr(before, key) == getattr(after, key) for key in fields), "small artifact changed while reading", ValueError)
    return raw


def validate_native_job(contract, artifact_root=None):
    value = _decode(_canonical(contract))
    _keys(value, ("schema", "source", "runtime", "runtime_requirements", "provenance",
                  "artifacts", "aliases", "source_paths", "fixtures", "correctness_case_ids",
                  "benchmark_case_ids", "benchmark_report_ids", "case_bindings", "phases",
                  "numerical_policy", "timing_policy", "report_contract", "measurement_valid",
                  "custody_proven", "reward_eligible"))
    require(value["schema"] == SCHEMA, "unsupported native schema", ValueError)
    for key in ("measurement_valid", "custody_proven", "reward_eligible"):
        require(value[key] is False, "preparation cannot assert " + key, ValueError)
    _keys(value["source"], ("repository", "revision", "task_selector", "snapshot_manifest_sha256"))
    _text(value["source"]["repository"])
    _revision(value["source"]["revision"])
    _path(value["source"]["task_selector"])
    _sha(value["source"]["snapshot_manifest_sha256"])
    _keys(value["runtime"], ("image", "config_digest"))
    image = value["runtime"]["image"]
    _text(image)
    require("@sha256:" in image, "pinned runtime image required", ValueError)
    _sha(image.rsplit("@sha256:", 1)[1])
    require(value["runtime"]["config_digest"].startswith("sha256:"), "config digest required", ValueError)
    _sha(value["runtime"]["config_digest"][7:])
    artifacts, aliases, fixtures = value["artifacts"], value["aliases"], value["fixtures"]
    require(type(artifacts) is dict and artifacts, "small source/metadata closure required", ValueError)
    require(type(aliases) is dict and type(fixtures) is dict, "alias/fixture manifests required", ValueError)
    for name, sha in artifacts.items():
        _path(name)
        _sha(sha)
    sources = value["source_paths"]
    require(type(sources) is list and sources and len(sources) == len(set(sources)), "source closure required", ValueError)
    require(all(name in artifacts for name in sources), "editable files must be registered regular source", ValueError)
    for name, target in aliases.items():
        _path(name)
        _path(target)
        require(target in artifacts and name not in artifacts, "alias must target registered source/support", ValueError)
    for name, row in fixtures.items():
        _path(name)
        _keys(row, ("manifest_path", "sha256", "size_bytes"))
        _path(row["manifest_path"])
        _sha(row["sha256"])
        require(type(row["size_bytes"]) is int and row["size_bytes"] > 0, "fixture bytes required", ValueError)
        require(name not in artifacts and name not in aliases, "tensor payload must stay outside small artifacts", ValueError)
    provenance = value["provenance"]
    _keys(provenance, ("kind", "values", "capture_revision", "model_revision", "framework_revision",
                       "observed_frequencies", "evidence_ref"))
    require(provenance["kind"] == "arena_operator_replay", "operator replay provenance required", ValueError)
    require(provenance["values"] in ("captured_original_fixture", "generated_from_pinned_recipe"),
          "explicit original value provenance required", ValueError)
    require(bool(fixtures) == (provenance["values"] == "captured_original_fixture"),
          "value provenance and declared fixture bodies disagree", ValueError)
    require(provenance["evidence_ref"] in artifacts, "provenance artifact missing", ValueError)
    require(provenance["observed_frequencies"] is None, "no validated per-case serving frequency model", ValueError)
    for name in ("capture_revision", "model_revision", "framework_revision"):
        if provenance[name] is not None:
            _revision(provenance[name])
    correct, benchmark = value["correctness_case_ids"], value["benchmark_case_ids"]
    _ids(correct)
    _ids(benchmark)
    require(set(benchmark) <= set(correct), "timed cases outside correctness declaration", ValueError)
    _keys(value["benchmark_report_ids"], benchmark)
    report_ids = list(value["benchmark_report_ids"].values())
    for name in report_ids:
        _text(name)
    require(len(set(report_ids)) == len(report_ids), "ambiguous/truncated native IDs", ValueError)
    _keys(value["case_bindings"], correct)
    for row in value["case_bindings"].values():
        _keys(row, ("artifact_ref", "json_pointer", "sha256"))
        require(row["artifact_ref"] in artifacts and row["artifact_ref"] not in sources,
              "case metadata must be protected", ValueError)
        _sha(row["sha256"])
    _keys(value["phases"], ("compile", "correctness", "benchmark"))
    for phase in value["phases"].values():
        _keys(phase, ("argv", "timeout_seconds"))
        require(type(phase["argv"]) is list and phase["argv"], "native phase argv required", ValueError)
        for arg in phase["argv"]:
            _text(arg)
        require(type(phase["timeout_seconds"]) is int and phase["timeout_seconds"] > 0,
              "finite native timeout required", ValueError)
    for name in ("numerical_policy", "timing_policy", "report_contract", "runtime_requirements"):
        require(type(value[name]) is dict and value[name], name + " must be explicit", ValueError)
    timing = value["timing_policy"]
    _keys(timing, ("primary", "fallback", "source_refs"))
    require(timing["primary"] in (
          {"kind": "cuda_event_direct_calls", "warmup_calls": 10, "measured_calls": 100,
           "statistic": "mean", "unit": "ms", "graph": False},
          {"kind": "cuda_event_graph_replay", "warmup_calls": 10, "measured_calls": 100,
           "statistic": "mean", "unit": "ms", "graph": True}),
          "supported 10/100 direct-call or explicit graph-replay timing required", ValueError)
    require(timing["fallback"] == {"kind": "native_ut_fallback", "comparable": False,
          "actual_timer": None}, "fallback is retained but has no comparable timing here", ValueError)
    require(all(name in artifacts and name not in sources for name in timing["source_refs"]),
          "timing source must be protected", ValueError)
    if artifact_root is not None:
        root, total, contents = Path(artifact_root).absolute(), 0, {}
        for name, expected in artifacts.items():
            raw = _read(root / name)
            total += len(raw)
            require(total <= SMALL_TOTAL_LIMIT, "planner artifact closure too large", ValueError)
            require(sha256_hex(raw) == expected, "artifact digest mismatch: " + name, ValueError)
            contents[name] = raw
        for name, target in aliases.items():
            alias = root / name
            require(alias.is_symlink() and alias.resolve(strict=True) == root / target,
                  "candidate/support alias differs: " + name, ValueError)
        for case, row in value["case_bindings"].items():
            selected = _pointer(_decode(contents[row["artifact_ref"]]), row["json_pointer"])
            require(selected == case and digest(selected) == row["sha256"], "case binding differs", ValueError)
        meta = _decode(contents[provenance["evidence_ref"]])
        for name in ("tol", "random_draws", "oracle_policy", "math_contract", "case_contracts"):
            require(value["numerical_policy"].get(name) == meta.get(name), "numerical declaration differs from original metadata", ValueError)
        require(value["runtime_requirements"]["candidate_bind"] == meta.get("candidate_bind"),
              "candidate routing differs from original metadata", ValueError)
        if fixtures and meta.get("reference_io_sha256"):
            name = "ut/" + meta.get("reference_io", "reference_io.pt")
            require(fixtures.get(name, {}).get("sha256") == meta["reference_io_sha256"],
                  "oracle fixture differs from original metadata", ValueError)
    return value
