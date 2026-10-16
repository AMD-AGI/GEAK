"""Immutable Arena operator replay contracts, without invented serving evidence.

This is a file/identity reader. It imports no task source and confers no GPU,
measurement, custody, or reward qualification. Artifact roots must be trusted
settled snapshots; their checked bytes must remain bound at execution admission.
"""

import contextlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat

from .jobs_v1 import load_job, validate_job
from .._util import canonical_json, require, sha256_hex




SCHEMA = "geak-arena-operator-replay-v1"




def _keys(value, names):
    require(type(value) is dict and set(value) == set(names), "unexpected contract fields", ValueError)


def _text(value):
    require(type(value) is str and bool(value.strip()) and "\x00" not in value,
          "expected nonempty text", ValueError)


def _sha(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256", ValueError)


def _revision(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value),
          "revision must be a pinned commit or content digest", ValueError)


def _path(value):
    _text(value)
    path = PurePosixPath(value)
    require(not path.is_absolute() and str(path) == value and value != "."
          and not any(part in (".", "..") for part in path.parts) and "\\" not in value,
          "artifact path must be canonical and relative", ValueError)
    return value


def _canonical(value):
    def check(item):
        if type(item) is dict:
            require(all(type(key) is str for key in item), "JSON keys must be strings", ValueError)
            for child in item.values():
                check(child)
        elif type(item) is list:
            for child in item:
                check(child)
        else:
            require(type(item) in (str, int, float, bool, type(None)), "non-JSON value", ValueError)
            require(type(item) is not float or math.isfinite(item), "nonfinite JSON value", ValueError)
    check(value)
    return canonical_json(value, ensure_ascii=False)


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: " + key, ValueError)
        result[key] = value
    return result


def _decode(data):
    value = json.loads(data, object_pairs_hook=_pairs)
    _canonical(value)  # Reject NaN/Infinity as well as non-JSON values.
    return value


def _read(path):
    """Read one stable regular file without following any directory/file symlink."""
    path = Path(path).absolute()
    require(not any(part in (".", "..") for part in path.parts), "noncanonical input path", ValueError)
    try:
        with contextlib.ExitStack() as stack:
            fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            stack.callback(os.close, fd)
            for part in path.parts[1:-1]:
                fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                stack.callback(os.close, fd)
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            stream = stack.enter_context(os.fdopen(fd, "rb"))
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode), "artifact must be a regular file", ValueError)
            data = stream.read()
            after = os.fstat(fd)
            stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")
            require(all(getattr(before, key) == getattr(after, key) for key in stable)
                  and len(data) == after.st_size, "artifact changed during read", ValueError)
            return data
    except OSError as error:
        raise ValueError("artifact unavailable or unsafe: " + str(path)) from error


def _pointer(document, pointer):
    require(type(pointer) is str and (pointer == "" or pointer.startswith("/")),
          "invalid JSON pointer", ValueError)
    value = document
    for token in pointer.split("/")[1:] if pointer else []:
        require(re.search(r"~(?:[^01]|$)", token) is None, "invalid JSON pointer escape", ValueError)
        token = token.replace("~1", "/").replace("~0", "~")
        if type(value) is list:
            require(re.fullmatch(r"0|[1-9][0-9]*", token) is not None, "invalid array pointer", ValueError)
            index = int(token)
            require(index < len(value), "JSON pointer outside array", ValueError)
            value = value[index]
        else:
            require(type(value) is dict and token in value, "unresolved JSON pointer", ValueError)
            value = value[token]
    return value


def _ids(value):
    require(type(value) is list and bool(value), "case domain must be a nonempty list", ValueError)
    require(all(type(item) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", item)
              for item in value), "invalid case ID", ValueError)
    require(len(value) == len(set(value)), "duplicate case ID", ValueError)


def validate_arena_job(contract, artifact_root=None):
    """Return detached validated JSON; artifact_root additionally checks exact bytes."""
    if type(contract) is dict and contract.get("schema") == "geak-arena-generated-task-v1":
        from .generated_task import validate_generated_task
        return validate_generated_task(contract, artifact_root)
    if type(contract) is dict and contract.get("schema") == "geak-arena-native-operator-v2":
        from .native_contract import validate_native_job
        return validate_native_job(contract, artifact_root)
    contract = _decode(_canonical(contract))
    _keys(contract, ("schema", "source", "runtime", "provenance", "artifacts", "source_paths",
                     "correctness_case_ids", "benchmark_case_ids", "benchmark_report_ids",
                     "case_bindings", "case_domains", "phases", "numerical_policy", "graph_policy",
                     "measurement_valid", "custody_proven", "reward_eligible"))
    require(contract["schema"] == SCHEMA, "unsupported Arena schema", ValueError)
    for key in ("measurement_valid", "custody_proven", "reward_eligible"):
        require(contract[key] is False, "Arena research contracts cannot assert " + key, ValueError)

    source = contract["source"]
    _keys(source, ("repository", "revision", "task_selector"))
    _text(source["repository"])
    _revision(source["revision"])
    _path(source["task_selector"])
    runtime = contract["runtime"]
    _keys(runtime, ("image", "config_digest"))
    require(type(runtime["image"]) is str and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}", runtime["image"]),
        "runtime image must pin a registry manifest", ValueError)
    require(type(runtime["config_digest"]) is str and re.fullmatch(
        r"sha256:[0-9a-f]{64}", runtime["config_digest"]), "invalid runtime config digest", ValueError)

    artifacts = contract["artifacts"]
    require(type(artifacts) is dict and bool(artifacts), "empty artifact manifest", ValueError)
    for path, digest in artifacts.items():
        _path(path)
        _sha(digest)
    def ref(path):
        _path(path)
        require(path in artifacts, "unregistered artifact reference: " + path, ValueError)
    paths = contract["source_paths"]
    require(type(paths) is list and bool(paths), "empty source closure", ValueError)
    for path in paths:
        ref(path)
    require(len(paths) == len(set(paths)), "duplicate source path", ValueError)

    provenance = contract["provenance"]
    _keys(provenance, ("kind", "values", "capture_revision", "model_revision",
                       "framework_revision", "observed_frequencies", "evidence_ref"))
    require(provenance["kind"] == "arena_operator_replay", "invalid replay provenance", ValueError)
    require(provenance["values"] == "generated_from_pinned_recipe", "invalid value provenance", ValueError)
    for key in ("capture_revision", "model_revision", "framework_revision"):
        if provenance[key] is not None:
            _revision(provenance[key])
    require(provenance["observed_frequencies"] is None, "observed serving frequencies are unknown", ValueError)
    ref(provenance["evidence_ref"])

    correctness, benchmark = contract["correctness_case_ids"], contract["benchmark_case_ids"]
    _ids(correctness)
    _ids(benchmark)
    require(set(benchmark) <= set(correctness), "benchmark domain is outside correctness domain", ValueError)
    reports = contract["benchmark_report_ids"]
    _keys(reports, benchmark)
    for report in reports.values():
        _text(report)
    require(len(set(reports.values())) == len(reports), "ambiguous native report IDs", ValueError)
    _keys(contract["case_bindings"], correctness)
    _keys(contract["case_domains"], ("correctness", "benchmark"))

    bindings = []
    def binding(value, extra=(), expected=None, compare=False):
        _keys(value, ("artifact_ref", "json_pointer", "sha256") + tuple(extra))
        ref(value["artifact_ref"])
        _sha(value["sha256"])
        pointer = value["json_pointer"]
        require(type(pointer) is str and (not pointer or pointer.startswith("/"))
              and re.search(r"~(?:[^01]|$)", pointer) is None, "invalid JSON pointer", ValueError)
        bindings.append((value, expected, compare))
    for case_id, value in contract["case_bindings"].items():
        binding(value, expected=case_id, compare="case_id")
    binding(contract["case_domains"]["correctness"], expected=correctness, compare=True)
    binding(contract["case_domains"]["benchmark"], expected=benchmark, compare=True)
    for name, kind in (("numerical_policy", "original_arena_policy"),
                       ("graph_policy", "original_arena_graph_policy")):
        policy = contract[name]
        require(type(policy) is dict, "policy must be a JSON object", ValueError)
        binding(policy, ("kind", "declaration"), policy.get("declaration"), True)
        require(policy["kind"] == kind, "original Arena policy must remain explicit", ValueError)
    protected = {provenance["evidence_ref"]} | {record["artifact_ref"] for record, _, _ in bindings}
    require(not set(paths) & protected, "editable source overlaps protected contract metadata", ValueError)
    graph = contract["graph_policy"]["declaration"]
    _keys(graph, ("mode", "warmup_calls", "measured_calls"))
    require(graph["mode"] == "required" and type(graph["warmup_calls"]) is int
          and graph["warmup_calls"] == 10 and type(graph["measured_calls"]) is int
          and graph["measured_calls"] == 100, "original graph 10/100 protocol required", ValueError)
    require(contract["numerical_policy"]["declaration"] is not None,
          "original numerical policy must be declared", ValueError)
    _keys(contract["phases"], ("compile", "correctness", "benchmark"))
    for phase in contract["phases"].values():
        _keys(phase, ("argv", "timeout_seconds"))
        require(type(phase["argv"]) is list and bool(phase["argv"]), "phase argv is empty", ValueError)
        for argument in phase["argv"]:
            _text(argument)
        timeout = phase["timeout_seconds"]
        require(type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0,
              "phase timeout must be finite and positive", ValueError)

    if artifact_root is not None:
        root = Path(artifact_root).absolute()
        contents = {}
        for name, digest in artifacts.items():
            data = _read(root / name)
            require(sha256_hex(data) == digest, "artifact digest mismatch: " + name, ValueError)
            contents[name] = data
        documents = {}
        for record, expected, compare in bindings:
            name = record["artifact_ref"]
            if name not in documents:
                documents[name] = _decode(contents[name])
            value = _pointer(documents[name], record["json_pointer"])
            require(sha256_hex(_canonical(value)) == record["sha256"],
                  "pointed metadata digest mismatch", ValueError)
            if compare == "case_id":
                identities = ([value] if type(value) is str else
                              [value[key] for key in ("sig", "case_id") if key in value]
                              if type(value) is dict else [])
                require(bool(identities) and all(item == expected for item in identities),
                      "pointed case identity differs from declared case ID", ValueError)
            elif compare:
                require(_canonical(value) == _canonical(expected), "declared metadata differs from original", ValueError)
    return contract


def job_digest(validated_contract):
    """Identity of the complete Arena task contract, excluding candidate source changes."""
    return sha256_hex(_canonical(validate_arena_job(validated_contract)))


def load_arena_job(path, artifact_root=None, expected_digest=None):
    contract = validate_arena_job(_decode(_read(path)), artifact_root)
    if expected_digest is not None:
        _sha(expected_digest)
        require(job_digest(contract) == expected_digest, "registered Arena job digest mismatch", ValueError)
    return contract


def load_engineer_job(path, artifact_root=None, expected_digest=None):
    """Dispatch explicitly; malformed or unknown Arena contracts never fall back to jobs.v1."""
    value = _decode(_read(path))
    require(type(value) is dict, "job must be a JSON object", ValueError)
    if "schema" in value:
        require(value["schema"] in (SCHEMA, "geak-arena-native-operator-v2", "geak-arena-generated-task-v1"), "unsupported explicit task schema", ValueError)
        return load_arena_job(path, artifact_root, expected_digest)
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1,
          "unsupported task schema", ValueError)
    return load_job(path, artifact_root=artifact_root, expected_digest=expected_digest)
