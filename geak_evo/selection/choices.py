"""The list of programs the planner may recommend as final.

A choice is one settled engineer episode of this campaign whose harvest kept a
``selected`` program that passed the native correctness check. Choices are
listed in history order and never ranked. Each choice carries the program's
source files, the checked native trials and the recorded run configuration,
and a ``choice_handle`` (the SHA-256 of everything else in the choice).

The ``episode(snapshot, history_row)`` callback must be the real episode
reader (``NativeEvidence.episode``, i.e. ``committed_native_episode``): it
re-checks the committed assignment, settlement, trial list, saved patch and
candidate roles. This module does not replace that check and never reads model
conversations. Being a choice says nothing about speed on an independent
endpoint.
"""

import copy
from functools import wraps
import json
from pathlib import Path, PurePosixPath

from .._util import EvoError, canonical_json, is_sha256, require, sha256_hex
from ..archive.refs import read_ref
from ..archive.store import _read as _store_read, _sync_directory, _write as _store_write


SCHEMA = "geak-native-recommendation-choice-v1"
_PREPARED_KEYS = {"entry_runner_ref", "entry_input_ref", "backend_binding_ref",
                  "collection_config_ref", "collector_ref", "entry_provenance_ref"}
PUBLIC_CHOICE_KEYS = ("source_manifest_ref", "source_body_refs", "selected_witness")
PUBLIC_PROVENANCE_KEYS = ("packet_ref", "transport_ref", "collection_ref", "raw_manifest_ref")


class RecommendationError(EvoError, ValueError):
    """A choice is missing, changed or belongs to another campaign."""


def _need(value, reason):
    require(value, reason, RecommendationError)


def _shape_errors(operation):
    """Report malformed records as RecommendationError."""
    @wraps(operation)
    def checked(*args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise RecommendationError("malformed recommendation evidence: " + str(exc)) from exc
    return checked


def _sha(value):
    _need(is_sha256(value), "invalid content identity")
    return value


def _text(value):
    _need(type(value) is str and value.strip(), "nonempty identity required")
    return value


def _canonical_name(name):
    return (type(name) is str and name and not PurePosixPath(name).is_absolute()
            and ".." not in PurePosixPath(name).parts and str(PurePosixPath(name)) == name)


def _file_ref(value):
    _need(type(value) is dict and {"path", "sha256"} <= set(value), "file reference required")
    path = value["path"]
    _need(type(path) is str and Path(path).is_absolute(), "absolute file reference required")
    return {"path": path, "sha256": _sha(value["sha256"])}


def _json_ref(ref):
    value = json.loads(read_ref(_file_ref(ref)))
    _need(type(value) is dict, "referenced JSON object required")
    return value


def _object_ref(objects, identity):
    return {"path": str(objects / _sha(identity)), "sha256": identity}


def _evidence_artifact(objects, ref):
    """Read only the named evidence artifact, never its whole group."""
    _need(type(ref) is dict and set(ref) == {"evidence_id", "artifact"}, "evidence reference required")
    name = ref["artifact"]
    _need(_canonical_name(name), "canonical evidence artifact required")
    envelope_ref = _object_ref(objects, ref["evidence_id"])
    envelope = _json_ref(envelope_ref)
    _need(envelope.get("kind") == "evidence" and type(envelope.get("artifacts")) is dict,
          "immutable evidence envelope required")
    _need(name in envelope["artifacts"], "named evidence artifact is unavailable")
    return envelope, _object_ref(objects, envelope["artifacts"][name]), envelope_ref


def _snapshot(store, snapshot):
    _need(type(snapshot) is dict and type(snapshot.get("state")) is dict, "committed campaign snapshot required")
    campaign_id = _text(snapshot.get("campaign_id"))
    revision = snapshot.get("revision")
    _need(type(revision) is int and revision > 0, "positive campaign revision required")
    state = snapshot["state"]
    committed = store.get_revision(campaign_id, revision)
    _need(committed["note"]["evolution"] == state, "recommendation snapshot is not committed")
    binding = state["binding"]
    _need(type(binding) is dict and set(binding) == {"job_id", "measurement_epoch_id"},
          "exact campaign job and measurement epoch required")
    _sha(binding["job_id"])
    _text(binding["measurement_epoch_id"])
    history = state["history"]
    _need(type(history) is list and all(type(row) is dict for row in history), "campaign history required")
    attempts = [row["attempt_id"] for row in history if "attempt_id" in row]
    _need(all(type(value) is str and value for value in attempts)
          and len(set(attempts)) == len(attempts), "native history contains duplicate or invalid attempts")
    return state, Path(store.root).resolve(strict=True) / "objects"


def _source_refs(store, objects, source_id):
    source = store.source(source_id)
    manifest_ref = _object_ref(objects, source_id)
    _need(_json_ref(manifest_ref) == source, "selected source manifest differs from the store")
    files = []
    for row in source["files"]:
        name = row["path"]
        _need(_canonical_name(name), "selected source path is not canonical")
        body_ref = None if row["blob_id"] is None else _object_ref(objects, row["blob_id"])
        if body_ref is not None:
            read_ref(body_ref)
        files.append({"source_path": name, "executable": row["executable"], "body_ref": body_ref,
                      "availability": "absent_in_source_manifest" if body_ref is None else "recorded"})
    _need(files and [row["source_path"] for row in files] == sorted({row["source_path"] for row in files}),
          "selected source closure must be nonempty, sorted and unique")
    return manifest_ref, files


def _configuration(objects, history, material, outer, native_trials):
    """The run configuration actually prepared for this episode, by file ref."""
    prepared_ref = {"evidence_id": history["transport_ref"]["evidence_id"], "artifact": "native_prepared.json"}
    envelope, artifact_ref, _ = _evidence_artifact(objects, prepared_ref)
    _need(envelope["candidate_id"] == history["candidate_id"], "prepared inputs belong to another candidate")
    prepared = _json_ref(artifact_ref)
    _need(set(prepared) == _PREPARED_KEYS, "complete actual native preparation map required")
    for ref in prepared.values():
        read_ref(_file_ref(ref))
    runtime_refs = material["host_provenance"]["runtime_refs"]
    _need(set(runtime_refs) == {"entry_runner_ref", "collector_ref", "entry_provenance_ref"}
          and all(prepared[key] == value for key, value in runtime_refs.items()),
          "prepared runtime differs from the committed native verifier")
    backend = _json_ref(prepared["backend_binding_ref"])
    _need(set(backend) == {"driver_ref", "config_ref"}, "actual native backend binding required")
    config_ref = _file_ref(backend["config_ref"])
    config = _json_ref(config_ref)
    native = config["backend_config"]
    probe = native["probe_config"]
    request = outer["request"]
    _need(config["entry_input_ref"] == prepared["entry_input_ref"]
          and probe["turn_id"] == history["attempt_id"]
          and probe["job_id"] == request["job"]["job_id"]
          and probe["measurement_epoch_id"] == request["measurement_epoch_id"]
          and native["direction_id"] == request["direction_id"],
          "actual host configuration differs from the selected episode")
    clock_ref = _file_ref(native["clock_ref"])
    read_ref(clock_ref)
    read_ref(_file_ref(backend["driver_ref"]))
    requests = [_file_ref(row["request"]["request_ref"]) for row in native_trials]
    for ref in requests:
        read_ref(ref)
    return {"status": "recorded_prepared_bytes", "native_prepared_ref": prepared_ref,
            "native_prepared": copy.deepcopy(prepared), "host_config_ref": config_ref,
            "backend_driver_ref": _file_ref(backend["driver_ref"]), "clock_ref": clock_ref,
            "entry_input_ref": copy.deepcopy(prepared["entry_input_ref"]),
            "collection_config_ref": copy.deepcopy(prepared["collection_config_ref"]),
            "native_request_refs": requests}


def _choice(store, snapshot, state, objects, history_index, episode):
    """Build the choice for one history row, or None if it offers no selected program."""
    history = state["history"][history_index]
    if "attempt_id" not in history or history.get("settled") is not True:
        return None
    _need(callable(episode), "trusted NativeEvidence.episode callback required")
    # Always call the episode check, even when no selected role exists.
    outer, material = episode(snapshot, history)
    provenance = material["host_provenance"]
    _need(outer["attempt_id"] == history["attempt_id"]
          and outer["settlement"].get("settled") is True and outer.get("process_reaped") is True
          and all(provenance[key] == history[key] for key in ("packet_ref", "transport_ref", "collection_ref")),
          "native episode is not the exact committed settled history")
    harvest = material["harvest"]
    roles = provenance["candidate_roles"]
    selected = harvest["selected"]
    if selected is None:
        _need("selected" not in roles, "selected candidate exists without a native selected witness")
        return None
    _need(harvest["status"] == "best_correct_retained" and selected.get("correctness") == "native_pass_observed"
          and material["explanation_check"]["binding_valid"] is True
          and material["native_entry_check"]["binding_valid"] is True,
          "selected source lacks a confirmed native selection binding")
    source_id = _sha(selected["source_id"])
    candidate_id = _sha(roles["selected"])
    candidate = store.get_candidate(candidate_id)
    _need(candidate["source_id"] == source_id
          and all(candidate[key] == value for key, value in state["binding"].items()),
          "selected candidate source or campaign domain differs")
    request = outer["request"]
    _need(request["job"]["job_id"] == state["binding"]["job_id"]
          and request["measurement_epoch_id"] == state["binding"]["measurement_epoch_id"]
          and harvest["binding"]["turn_id"] == history["attempt_id"]
          and all(harvest["binding"][key] == value for key, value in state["binding"].items()),
          "selected witness belongs to another campaign episode")
    native_trials = [row for row in material["native_trials"]
                     if row.get("verified") is not None and row["verified"]["source_id"] == source_id]
    _need(native_trials and all(row["error"] is None and row.get("request") is not None for row in native_trials)
          and selected["feedback_refs"] == [row["feedback_ref"] for row in native_trials]
          and selected["timings"] == [row["verified"]["per_case_times_ms"] for row in native_trials]
          and selected["verified_turn_ids"] == sorted({row["verified"]["turn_id"] for row in native_trials}),
          "selected witness differs from exact checked native passes")
    manifest_ref, source_files = _source_refs(store, objects, source_id)
    config = _configuration(objects, history, material, outer, native_trials)
    choice = {"schema": SCHEMA,
              "binding": dict(state["binding"], campaign_id=snapshot["campaign_id"]),
              "candidate_id": candidate_id, "source_id": source_id,
              "source_manifest_ref": manifest_ref, "source_body_refs": source_files,
              "attempt_id": history["attempt_id"], "history_index": history_index,
              "direction_id": _text(request["direction_id"]),
              "host_provenance": {key: copy.deepcopy(provenance[key]) for key in
                                  ("packet_ref", "transport_ref", "collection_ref", "raw_manifest_ref", "runtime_refs")},
              "selected_witness": {"role": "selected", "harvest_selected": copy.deepcopy(selected),
                                   "native_trials": copy.deepcopy(native_trials)},
              "recorded_configuration": config,
              "unavailable": {"endpoint_eligibility": None, "independent_endpoint_result_ref": None}}
    choice["choice_handle"] = sha256_hex(canonical_json(choice))
    return choice


@_shape_errors
def eligible_choices(store, snapshot, episode):
    """Every settled own episode with a selected program, in history order, unranked.

    Two episodes with the same source stay two choices. Other roles and imported
    candidates are never offered instead. Integrity errors propagate; an episode
    without a selected role simply offers nothing.
    """
    state, objects = _snapshot(store, snapshot)
    choices = []
    for index in range(len(state["history"])):
        choice = _choice(store, snapshot, state, objects, index, episode)
        if choice is not None:
            choices.append(choice)
    return choices


def _check_handle(choice):
    _need(type(choice) is dict and choice.get("schema") == SCHEMA, "native recommendation choice required")
    handle = _sha(choice.get("choice_handle"))
    payload = {key: value for key, value in choice.items() if key != "choice_handle"}
    _need(sha256_hex(canonical_json(payload)) == handle, "recommendation choice handle differs")


@_shape_errors
def authenticate_choice(store, snapshot, choice, episode):
    """Rebuild the choice from the store and return it, or raise if anything differs."""
    _check_handle(choice)
    state, objects = _snapshot(store, snapshot)
    index = choice.get("history_index")
    _need(type(index) is int and 0 <= index < len(state["history"]), "recommendation history index is unavailable")
    actual = _choice(store, snapshot, state, objects, index, episode)
    _need(actual is not None and canonical_json(actual) == canonical_json(choice),
          "recommendation differs from the current authenticated own witness")
    return actual


@_shape_errors
def choice_references(choice):
    """File refs the planner may read for this choice, sorted by path.

    An explicit allowlist: the selected source, its checked native trials and the
    episode's own evidence pairs (envelope + named artifact only). Runtime refs and
    the recorded configuration stay bound by the handle but are not offered, and
    fields added later are excluded by default.
    """
    _check_handle(choice)
    manifest = _file_ref(choice["source_manifest_ref"])
    _need(manifest["sha256"] == choice["source_id"] and Path(manifest["path"]).name == choice["source_id"],
          "selected source manifest is not its immutable object")
    objects = Path(manifest["path"]).parent
    refs = {}

    def add(ref):
        ref = _file_ref(ref)
        prior = refs.get(ref["path"])
        _need(prior is None or prior == ref, "one public path has conflicting content identities")
        read_ref(ref)
        refs[ref["path"]] = ref

    def visit(value):
        if type(value) is dict:
            if {"path", "sha256"} <= set(value):
                add(value)
            elif set(value) == {"evidence_id", "artifact"}:
                _, artifact_ref, envelope_ref = _evidence_artifact(objects, value)
                add(envelope_ref)
                add(artifact_ref)
            for item in value.values():
                visit(item)
        elif type(value) is list:
            for item in value:
                visit(item)

    for key in PUBLIC_CHOICE_KEYS:
        visit(choice[key])
    for key in PUBLIC_PROVENANCE_KEYS:
        visit(choice["host_provenance"][key])
    return [refs[path] for path in sorted(refs)]


# ---------------------------------------------------------------- per-entry eligibility and checkpoints

def _entry_failure(exc):
    # The episode checks raise plain ValueError. Only that and RecommendationError
    # count as "this entry is ineligible"; typed store/file errors, JSON errors and
    # OSError stay fatal.
    return type(exc) in (RecommendationError, ValueError)


def _choice_rejection(index, row, exc):
    return {"history_index": index, "attempt_id": row.get("attempt_id"),
            "candidate_id": row.get("candidate_id"),
            "evidence_refs": {key: copy.deepcopy(row[key]) for key in
                              ("packet_ref", "transport_ref", "collection_ref") if key in row},
            "error_class": type(exc).__name__, "reason": str(exc),
            "status": "rejected_native_eligibility"}


def partition_choices(store, snapshot, episode):
    """Check each entry on its own: returns ``(choices, rejections)``.

    Only explicit choice/episode refusals become rejections; unknown shape errors
    and typed storage errors propagate rather than being mistaken for one bad
    candidate.
    """
    state, objects = _shape_errors(_snapshot)(store, snapshot)
    choices, rejections = [], []
    for index, row in enumerate(state["history"]):
        try:
            choice = _choice(store, snapshot, state, objects, index, episode)
        except ValueError as exc:
            if not _entry_failure(exc):
                raise
            rejections.append(_choice_rejection(index, row, exc))
            continue
        if choice is not None:
            choices.append(choice)
    return choices, rejections


def _checkpoint_identity(choice):
    return {key: value for key, value in choice.items() if key not in ("history_index", "choice_handle")}


@_shape_errors
def _checkpoint_name(choice):
    _check_handle(choice)
    # Keyed by attempt, so the name survives index shifts in the history.
    key = {"binding": choice["binding"], "attempt_id": _text(choice["attempt_id"])}
    return sha256_hex(canonical_json(key)) + ".json"


def _checkpoint_directory(directory, create=False):
    directory = Path(directory)
    _need(directory.is_absolute() and directory.parent.resolve(strict=True) == directory.parent,
          "checkpoint parent must be canonical host storage")
    if create:
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        else:
            _sync_directory(directory.parent)
    _need(directory.resolve(strict=True) == directory and directory.is_dir(),
          "checkpoint directory must be canonical host storage")
    return directory


def checkpoint_choice(directory, choice):
    """Save one choice create-only (0444). An existing file for the same attempt is
    kept as is; conflicting content is an error, never an overwrite."""
    directory = _checkpoint_directory(directory, create=True)
    path = directory / _checkpoint_name(choice)
    try:
        _store_write(path, canonical_json(choice), immutable=True)
    except FileExistsError:
        raw, _ = _store_read(directory, path.name)
        recorded = json.loads(raw)
        _check_handle(recorded)
        _need(_checkpoint_name(recorded) == path.name
              and canonical_json(_checkpoint_identity(recorded)) == canonical_json(_checkpoint_identity(choice)),
              "checkpoint attempt exists with different authenticated evidence")
    return path


def checkpoint_choices(store, snapshot, directory, episode):
    """Checkpoint all current choices (called at each planner freeze).

    A stop before the next freeze may leave the latest episode without a
    checkpoint. This neither certifies timing nor picks a final program.
    """
    choices, rejected = partition_choices(store, snapshot, episode)
    directory = _checkpoint_directory(directory, create=True)
    checkpoints = []
    for choice in choices:
        path = checkpoint_choice(directory, choice)
        raw, _ = _store_read(directory, path.name)
        recorded = json.loads(raw)
        _check_handle(recorded)
        checkpoints.append({"attempt_id": choice["attempt_id"],
                            "checkpoint_choice_handle": recorded["choice_handle"],
                            "current_choice_handle": choice["choice_handle"],
                            "checkpoint_ref": {"path": str(path), "sha256": sha256_hex(raw)}})
    return choices, {"schema": "geak-native-eligibility-snapshot-v1",
                     "campaign_id": snapshot["campaign_id"], "revision": snapshot["revision"],
                     "checkpoints": checkpoints, "rejections": rejected,
                     "checkpoint_trigger": "next_planner_freeze_of_committed_snapshot",
                     "latest_collection_checkpoint_guaranteed_at_stop": False,
                     "endpoint_eligibility": None, "certified_timing": False,
                     "scope": "Native SELECTED witness eligibility only; no ranking, endpoint fallback or certified timing."}


def reauthenticate_checkpoints(store, snapshot, directory, episode):
    """Rebuild every checkpointed choice by attempt id: returns ``(accepted, rejected)``.

    Unreadable or mismatching checkpoint files become explicit rejections with
    their content refs; nothing is repaired. Typed store errors and non-ValueError
    errors from the episode check propagate.
    """
    state, objects = _shape_errors(_snapshot)(store, snapshot)
    directory = _checkpoint_directory(directory)
    where = {row["attempt_id"]: index for index, row in enumerate(state["history"]) if "attempt_id" in row}
    accepted, rejected = [], []
    for path in sorted(directory.iterdir()):
        if path.name.startswith(".pending-"):
            # An interrupted write never published a checkpoint.
            rejected.append({"checkpoint": path.name, "status": "uncommitted_checkpoint_write",
                             "reason": "Atomic checkpoint write did not publish a committed name."})
            continue
        raw, _ = _store_read(directory, path.name)
        source_ref = {"path": str(path), "sha256": sha256_hex(raw)}
        try:
            recorded = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            rejected.append({"checkpoint_ref": source_ref, "error_class": type(exc).__name__,
                             "reason": str(exc), "status": "rejected_checkpoint"})
            continue
        try:
            _check_handle(recorded)
            _need(path.name == _checkpoint_name(recorded), "checkpoint name differs from attempt binding")
            index = where.get(recorded["attempt_id"])
            _need(index is not None, "checkpoint attempt is absent from committed history")
            actual = _choice(store, snapshot, state, objects, index, episode)
            _need(actual is not None
                  and canonical_json(_checkpoint_identity(actual)) == canonical_json(_checkpoint_identity(recorded)),
                  "checkpoint differs from current authenticated own witness")
        except ValueError as exc:
            if not _entry_failure(exc):
                raise
            rejected.append({"checkpoint_ref": source_ref, "error_class": type(exc).__name__,
                             "reason": str(exc), "status": "rejected_checkpoint"})
            continue
        accepted.append(actual)
    return accepted, rejected
