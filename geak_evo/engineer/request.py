"""Validate exact job, source, evidence and continuation inputs for an engineer."""

import json
import math
from pathlib import Path
import re
import uuid

from .._util import canonical_json, require
from ..archive.refs import EngineerError, file_ref, read_ref as _read_ref, _source_ref


def _require(condition, message):
    require(condition, message, EngineerError)


def _json(value):
    return canonical_json(value, ensure_ascii=False)


def _sha(value):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256")
    return value

def _directory(value):
    path = Path(value)
    _require(path.is_absolute() and path.resolve(strict=True) == path and path.is_dir(),
             "directory must be canonical and existing")
    return path

def validate_request(request):
    """Bind the job/artifact bytes and explicit exact-state continuation inputs.

    Candidate IDs must already have been resolved by the host ResearchStore.
    File hashes detect changes; only the external worker enforces read-only access.
    """
    from ..evaluate.jobs import load_engineer_job

    request = json.loads(_json(request))
    _require(set(request) == {"schema_version", "direction_id", "job", "measurement_epoch_id",
                             "workspace", "anchor", "tip", "donors", "read_only",
                             "next_question", "budget", "continuation"}, "invalid request fields")
    _require(type(request["schema_version"]) is int and request["schema_version"] == 1,
             "unsupported request version")
    for key in ("direction_id", "next_question"):
        _require(type(request[key]) is str and bool(request[key].strip()), "missing " + key)
    workspace = _directory(request["workspace"])
    job = request["job"]
    _require(type(job) is dict and set(job) == {"path", "job_id", "artifact_root"}, "invalid job binding")
    _sha(job["job_id"])
    artifact_root = _directory(job["artifact_root"])
    _require(workspace != artifact_root and workspace not in artifact_root.parents
             and artifact_root not in workspace.parents, "artifact root and workspace overlap")
    job_path = Path(job["path"])
    file_ref(job_path)
    _require(workspace not in job_path.parents, "job must be outside mutable workspace")
    load_engineer_job(job_path, artifact_root=artifact_root, expected_digest=job["job_id"])
    _require(type(request["measurement_epoch_id"]) is str and request["measurement_epoch_id"].strip(),
             "measurement epoch must be a nonempty exact identifier")
    _require(type(request["donors"]) is list, "donors must be an array")
    sources = [("tip", request["tip"])] + [("donor", item) for item in request["donors"]]
    if request["anchor"] is not None:
        sources.append(("anchor", request["anchor"]))
    for role, source in sources:
        candidate = _source_ref(source)
        _require(workspace not in Path(source["manifest"]["path"]).parents,
                 "archived source is inside mutable workspace")
        if role in ("tip", "anchor"):
            _require(candidate["job_id"] == job["job_id"]
                     and candidate["measurement_epoch_id"] == request["measurement_epoch_id"],
                     role + " candidate belongs to a different job or measurement epoch")
    _require(request["anchor"] is None or type(request["anchor"]) is dict, "invalid anchor")
    _require(type(request["read_only"]) is list and request["read_only"], "read-only state/evidence required")
    for ref in request["read_only"]:
        _require(workspace not in Path(ref["path"]).parents, "evidence is inside mutable workspace")
        _read_ref(ref)
    budget = request["budget"]
    _require(type(budget) is dict and set(budget) == {"wall_seconds", "max_tokens"}, "invalid budget")
    _require(type(budget["wall_seconds"]) in (int, float) and math.isfinite(budget["wall_seconds"])
             and budget["wall_seconds"] > 0, "wall budget must be finite and positive")
    _require(budget["max_tokens"] is None or type(budget["max_tokens"]) is int
             and budget["max_tokens"] > 0, "token budget must be positive or unknown")
    continuation = request["continuation"]
    _require(type(continuation) is dict and set(continuation) == {"mode", "prior_receipt"},
             "invalid continuation")
    _require(continuation["mode"] in ("fresh", "resume", "handoff"), "invalid continuation mode")
    if continuation["mode"] == "fresh":
        _require(continuation["prior_receipt"] is None, "fresh session cannot claim a predecessor")
    else:
        prior = json.loads(_read_ref(continuation["prior_receipt"]))
        _require(prior.get("settlement", {}).get("settled") is True, "predecessor is not settled")
        previous = prior["request"]
        native_handoff = continuation["mode"] == "handoff" and prior.get("backend") == "native_idea"
        fields = ("direction_id", "job", "measurement_epoch_id") + (() if native_handoff else ("workspace",))
        for key in fields:
            _require(previous[key] == request[key], "continuation changed " + key)
        if native_handoff:
            _require(previous["workspace"] != request["workspace"], "native handoff requires a fresh episode workspace")
        _require(prior.get("captured_tip_source_id") == request["tip"]["source_id"],
                 "continuation must start from the exact harvested tip")
        if continuation["mode"] == "resume":
            _thread_id(prior.get("thread_id"))
            _require(prior.get("identity_valid") is True, "predecessor conversation identity is uncertain")
    return request

def _thread_id(value):
    _require(type(value) is str, "missing thread ID")
    try:
        _require(str(uuid.UUID(value)) == value, "thread ID must be an exact UUID")
    except (ValueError, AttributeError) as exc:
        raise EngineerError("thread ID must be an exact UUID") from exc
    return value
