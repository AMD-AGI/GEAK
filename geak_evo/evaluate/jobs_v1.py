"""Strict, content-addressed contracts for complete workload-observed callables.

Mappings are editable proposals. Their canonical digest identifies the frozen
contract; callers must retain that digest and reject changed bytes at admission.
Nothing here imports a captured source file or executes an operation.
"""

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

from .._util import canonical_json, require, sha256_hex






def _keys(value, required, optional=()):
    require(type(value) is dict, "expected a JSON object", ValueError)
    require(set(required) <= set(value) <= set(required) | set(optional),
             "missing or unknown fields: " + ", ".join(required), ValueError)


def _string(value):
    require(type(value) is str and bool(value.strip()), "expected nonempty string", ValueError)


def _integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "invalid integer", ValueError)


def _sha(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value),
             "expected lowercase SHA256", ValueError)


def _revision(value):
    require(type(value) is str and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value),
             "revision must be a pinned commit or content digest", ValueError)


def _path(value):
    _string(value)
    path = PurePosixPath(value)
    require(not path.is_absolute() and str(path) == value and
             not any(part in (".", "..") for part in path.parts) and
             "\\" not in value and "\x00" not in value and value != ".",
             "artifact path must be normalized and relative", ValueError)


def _json(value):
    if type(value) is dict:
        require(all(type(key) is str for key in value), "JSON keys must be strings", ValueError)
        for item in value.values():
            _json(item)
    elif type(value) is list:
        for item in value:
            _json(item)
    else:
        require(type(value) in (str, int, float, bool, type(None)), "non-JSON value", ValueError)
        require(type(value) is not float or math.isfinite(value), "non-finite number", ValueError)


def _canonical(value):
    _json(value)
    return canonical_json(value, ensure_ascii=False)


def validate_job(job, artifact_root=None):
    """Return a detached JSON mapping; raise ValueError for invalid contracts.

    With artifact_root, also hash every declared artifact. Without it this is
    structural validation only, never a claim that capture or closure is complete.
    """
    job = json.loads(_canonical(job))
    _keys(job, ("schema_version", "job_version", "workload", "accepted_stack_digest",
                "callable", "baseline", "artifacts", "provenance", "cases",
                "numerical", "timing", "graph"), ("supersedes",))
    _integer(job["schema_version"], 1)
    require(job["schema_version"] == 1, "unsupported schema version", ValueError)
    _integer(job["job_version"], 1)
    if job["job_version"] > 1:
        _sha(job.get("supersedes"))
    else:
        require(job.get("supersedes") is None, "first version cannot supersede a job", ValueError)
    _sha(job["accepted_stack_digest"])
    files = job["artifacts"]
    require(type(files) is dict and bool(files), "empty artifact manifest", ValueError)
    for path, digest in files.items():
        _path(path)
        _sha(digest)

    def ref(value):
        _path(value)
        require(value in files, "missing artifact reference: " + value, ValueError)

    def refs(values, nonempty=False):
        require(type(values) is list and (bool(values) or not nonempty), "invalid reference list", ValueError)
        for value in values:
            ref(value)
        require(len(values) == len(set(values)), "duplicate artifact reference", ValueError)

    workload = job["workload"]
    _keys(workload, ("model", "model_revision", "framework_source", "device", "tp",
                     "isl_tokens", "osl_tokens", "concurrency", "precision"))
    for key in ("model", "device", "precision"):
        _string(workload[key])
    _revision(workload["model_revision"])
    for key in ("tp", "isl_tokens", "osl_tokens", "concurrency"):
        _integer(workload[key], 1)
    framework = workload["framework_source"]
    _keys(framework, ("repository", "version", "revision", "manifest_ref"))
    _string(framework["repository"])
    _string(framework["version"])
    _revision(framework["revision"])
    ref(framework["manifest_ref"])
    provenance = job["provenance"]
    _keys(provenance, ("kind", "capture_ref", "workload_ref"))
    require(provenance["kind"] in ("workload_capture", "test_fixture"), "invalid provenance kind", ValueError)
    ref(provenance["capture_ref"])
    ref(provenance["workload_ref"])
    baseline, binding = job["baseline"], job["callable"]
    _keys(baseline, ("source_paths", "support_paths"))
    refs(baseline["source_paths"], nonempty=True)
    refs(baseline["support_paths"])
    require(not set(baseline["source_paths"]) & set(baseline["support_paths"]),
             "source and support manifest overlap", ValueError)
    _keys(binding, ("source_path", "qualname", "binding_ref", "stateful"))
    require(binding["source_path"] in baseline["source_paths"], "callable outside baseline closure", ValueError)
    _string(binding["qualname"])
    ref(binding["binding_ref"])
    require(binding["binding_ref"] in baseline["support_paths"], "binding missing from closure", ValueError)
    require(type(binding["stateful"]) is bool, "stateful must be boolean", ValueError)
    cases, case_ids = job["cases"], []
    require(type(cases) is list and bool(cases), "job needs observed cases", ValueError)
    for case in cases:
        _keys(case, ("case_id", "phase", "rank", "tensors", "scalars", "inputs_ref",
                     "state_refs", "routing_refs", "observed_frequency"), ("inferred_frequency",))
        _string(case["case_id"])
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case["case_id"]), "invalid case ID", ValueError)
        case_ids.append(case["case_id"])
        _string(case["phase"])
        _integer(case["rank"])
        require(case["rank"] < workload["tp"], "case rank outside TP", ValueError)
        ref(case["inputs_ref"])
        refs(case["state_refs"], nonempty=binding["stateful"])
        refs(case["routing_refs"])
        require(binding["stateful"] or not case["state_refs"], "state needs stateful contract", ValueError)
        require(type(case["tensors"]) is dict and type(case["scalars"]) is dict, "invalid inputs", ValueError)
        for name, tensor in case["tensors"].items():
            _string(name)
            _keys(tensor, ("shape", "dtype", "strides", "stride_unit", "layout"))
            require(type(tensor["shape"]) is list and type(tensor["strides"]) is list,
                     "shape and strides must be lists", ValueError)
            require(len(tensor["shape"]) == len(tensor["strides"]), "shape/stride rank mismatch", ValueError)
            for size in tensor["shape"]:
                _integer(size)
            require(all(type(stride) is int for stride in tensor["strides"]), "invalid stride", ValueError)
            require(tensor["stride_unit"] == "elements", "stride unit must be elements", ValueError)
            _string(tensor["dtype"])
            _string(tensor["layout"])
        for name, value in case["scalars"].items():
            _string(name)
            require(type(value) in (str, int, float, bool, type(None)), "scalar must be scalar JSON", ValueError)
        frequency = case["observed_frequency"]
        _keys(frequency, ("count", "unit", "provenance_ref"))
        _integer(frequency["count"], 1)
        require(frequency["unit"] == "calls", "observed frequency unit must be calls", ValueError)
        ref(frequency["provenance_ref"])
        inferred = case.get("inferred_frequency")
        if inferred is not None:
            _keys(inferred, ("count", "unit", "basis_ref"))
            require(type(inferred["count"]) in (int, float) and inferred["count"] >= 0,
                     "invalid inferred count", ValueError)
            require(inferred["unit"] == "calls", "inferred frequency unit must be calls", ValueError)
            ref(inferred["basis_ref"])
    require(len(case_ids) == len(set(case_ids)), "duplicate case ID", ValueError)
    numerical, timing, graph = job["numerical"], job["timing"], job["graph"]
    _keys(numerical, ("reference", "case_ids", "protocol_ref", "state_contract_ref"))
    require(numerical["reference"] == "baseline", "numerical reference must be captured baseline", ValueError)
    _keys(timing, ("case_ids", "unit", "protocol_ref", "warmup_calls", "measured_calls",
                   "aggregation", "scope"))
    for contract in (numerical, timing):
        ids = contract["case_ids"]
        require(type(ids) is list and all(type(item) is str for item in ids), "invalid coverage", ValueError)
        require(len(ids) == len(set(ids)) and set(ids) == set(case_ids),
                 "numerical and timing coverage must include every observed case exactly once", ValueError)
        ref(contract["protocol_ref"])
    if binding["stateful"]:
        ref(numerical["state_contract_ref"])
    else:
        require(numerical["state_contract_ref"] is None, "stateless job has state contract", ValueError)
    require(timing["unit"] in ("ns", "us", "ms", "s"), "explicit timing unit required", ValueError)
    _integer(timing["warmup_calls"])
    _integer(timing["measured_calls"], 1)
    require(timing["aggregation"] in ("mean", "median", "minimum"), "unknown timing aggregation", ValueError)
    require(timing["scope"] == "complete_callable", "timing must cover complete callable", ValueError)
    _keys(graph, ("mode", "protocol_ref"))
    require(graph["mode"] in ("required", "disabled"), "explicit graph mode required", ValueError)
    ref(graph["protocol_ref"])
    if artifact_root is not None:
        try:
            root = Path(artifact_root).resolve(strict=True)
            for name, digest in files.items():
                path = root.joinpath(*PurePosixPath(name).parts)
                require(not any(parent.is_symlink() for parent in [path] + list(path.parents)
                                 if parent != root and root in parent.parents), "symlink artifact", ValueError)
                resolved = path.resolve(strict=True)
                require(root in resolved.parents and resolved.is_file(), "artifact outside root or not a file", ValueError)
                hasher = hashlib.sha256()
                with resolved.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        hasher.update(block)
                require(hasher.hexdigest() == digest, "artifact digest mismatch: " + name, ValueError)
        except OSError as error:
            raise ValueError("artifact unavailable: " + str(error)) from error
    return job


def job_digest(job):
    """SHA256 of the complete validated canonical contract; no candidate identity."""
    return sha256_hex(_canonical(validate_job(job)))


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key, ValueError)
        result[key] = value
    return result


def load_job(path, artifact_root=None, expected_digest=None):
    """Load strict JSON and optionally bind it to its registered immutable digest."""
    job = validate_job(json.loads(Path(path).read_text(encoding="utf-8"),
                                  object_pairs_hook=_unique_pairs), artifact_root)
    if expected_digest is not None:
        _sha(expected_digest)
        require(job_digest(job) == expected_digest, "registered job digest mismatch", ValueError)
    return job
