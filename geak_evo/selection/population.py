"""Pick one program across a population of campaigns from their horizon seals.

Each registered member contributes its horizon seal (``frontier.seal_horizon``).
A ``selected`` member is scored by the median, over its distinct native trials,
of the per-trial geometric mean of the benchmark case times (lower is better).
The population result is:

* ``unresolved`` if any member is unresolved (no subset of members is a result);
* ``selected`` with the lowest-scoring member otherwise, ties broken by
  source_id, candidate_id, member_id;
* ``missing`` if no member selected anything.

The ``authenticate_member`` callback is where trust comes from: it must check
the seal's origin and regenerate the exact choice and the full comparison
context of each trial. The caller never supplies a score. The result is written
once to the registered path before any fresh measurement starts; this module
starts no search and no measurement, and its score is a development heuristic,
not an independent endpoint result.
"""

import copy
import datetime
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import statistics

from .._util import EvoError, canonical_json, is_sha256, require, sha256_hex
from .frontier import SUPPORTED_HORIZONS


CONTRACT_SCHEMA = "geak-population-native-selector-contract-v1"
SEAL_SCHEMA = "geak-population-native-selection-v1"
STATISTIC = "median-of-complete-trial-equal-case-geomeans-ms-v1"
UNAVAILABLE_METRIC_SOURCE = "registered metric source unavailable; remeasure"
_CONTRACT_KEYS = {"schema", "population_id", "operator", "job_id", "horizon", "selection_seal_path",
                  "benchmark_case_ids", "benchmark_report_ids", "correctness_case_ids",
                  "benchmark_contract_ref", "native_contract_sha256", "evaluator_ref",
                  "timing_policy", "phases", "runtime_image", "runtime_config_digest",
                  "device_policy", "members", "implementation_ref",
                  "known_missing_policy", "unresolved_policy"}
_MEMBER_KEYS = {"member_id", "campaign_id", "measurement_epoch_id", "horizon_contract_ref", "physical_devices"}
_SOURCE_KEYS = {"source_path", "executable", "body_ref", "availability"}
_SEAL_STAT = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size",
              "st_mtime_ns", "st_ctime_ns")


class SelectorError(EvoError, ValueError):
    pass


def need(value, reason):
    require(value, reason, SelectorError)


def encoded(value):
    return canonical_json(value)


def digest(raw):
    return sha256_hex(raw)


def decoded(raw):
    need(type(raw) is bytes, "exact_bytes_required")

    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    encoded(value)  # Rejects non-finite values everywhere, including context.
    return value


def exact_ref(ref):
    need(type(ref) is dict and set(ref) == {"path", "sha256"}
         and type(ref["path"]) is str and Path(ref["path"]).is_absolute()
         and is_sha256(ref["sha256"]), "exact_absolute_reference_required")
    return copy.deepcopy(ref)


def _file_identity(path):
    path = Path(path).absolute()
    return {"path": str(path), "sha256": digest(path.read_bytes())}


def implementation_ref():
    """Ref of this module's own bytes; a contract is bound to the selector code."""
    return _file_identity(__file__)


def metric_source_ref():
    """Ref of the module that defines the trial statistic (``archive/cells.py``)."""
    return _file_identity(Path(__file__).parent.parent / "archive" / "cells.py")


def check_metric_source(seal, *, registered_source=None):
    """Check a seal's ``metric_source_ref`` against the exact source it registered.

    By default the seal must pin this checkout's ``archive/cells.py``.

    A seal written by another version (for example before the port) pins a
    different path. The host may admit that version by passing
    ``registered_source={"ref": <the pinned {path, sha256}>, "path": <where its
    bytes are now>}``: the seal ref must equal ``ref`` exactly, and the bytes at
    ``path`` must hash to ``ref["sha256"]``. Equal bytes show the file is the
    one that was pinned, not that its statistic matches this version; the host
    must only admit a version it knows to be compatible, or remeasure.

    Any other seal fails closed (``UNAVAILABLE_METRIC_SOURCE`` when the pinned
    source was not admitted or cannot be read). Nothing is re-pinned or guessed.
    Returns the ref.
    """
    need(type(seal) is dict, "exact_selection_seal_required")
    ref = exact_ref(seal.get("metric_source_ref"))
    current = metric_source_ref()
    if ref["path"] == current["path"]:
        need(ref == current, "metric_source_differs")
        return ref
    need(registered_source is not None, UNAVAILABLE_METRIC_SOURCE)
    need(type(registered_source) is dict and set(registered_source) == {"ref", "path"},
         "registered metric source must be {ref, path}")
    need(ref == exact_ref(registered_source["ref"]), "metric_source_differs")
    try:
        raw = Path(registered_source["path"]).read_bytes()
    except (TypeError, OSError):
        raw = None
    need(raw is not None, UNAVAILABLE_METRIC_SOURCE)
    need(digest(raw) == ref["sha256"], "metric_source_differs")
    return ref


def _texts(values, reason):
    need(type(values) is list and values
         and all(type(value) is str and value for value in values)
         and len(set(values)) == len(values), reason)


def _check_members(members, device):
    need(type(members) is list and len(members) >= 2, "at_least_two_registered_members_required")
    for member in members:
        need(type(member) is dict and set(member) == _MEMBER_KEYS, "registered_member_shape_differs")
        for key in ("member_id", "campaign_id", "measurement_epoch_id"):
            need(type(member[key]) is str and member[key], "registered_member_identity_required")
        exact_ref(member["horizon_contract_ref"])
        _texts(member["physical_devices"], "registered_member_devices_required")
        need(set(member["physical_devices"]) <= set(device["physical_devices"]),
             "member_device_outside_registered_pool")
    need(len({m["member_id"] for m in members}) == len(members)
         and len({m["campaign_id"] for m in members}) == len(members)
         and len({encoded(m["horizon_contract_ref"]) for m in members}) == len(members),
         "duplicate_registered_member")


def _contract(raw, ref):
    exact_ref(ref)
    need(digest(raw) == ref["sha256"], "selector_contract_bytes_differ")
    value = decoded(raw)
    need(type(value) is dict and set(value) == _CONTRACT_KEYS
         and value["schema"] == CONTRACT_SCHEMA, "selector_contract_schema_differs")
    need(value["implementation_ref"] == implementation_ref(), "selector_implementation_differs")
    _texts(value["benchmark_case_ids"], "exact_nonempty_unique_benchmark_set_required")
    _texts(value["correctness_case_ids"], "correctness_cases_required")
    need(set(value["benchmark_case_ids"]) <= set(value["correctness_case_ids"]),
         "benchmark_cases_need_correctness_coverage")
    mapping = value["benchmark_report_ids"]
    need(type(mapping) is dict and set(mapping) == set(value["benchmark_case_ids"])
         and all(type(v) is str and v for v in mapping.values())
         and len(set(mapping.values())) == len(mapping), "exact_report_mapping_required")
    for key in ("population_id", "operator", "job_id", "runtime_image", "runtime_config_digest"):
        need(type(value[key]) is str and bool(value[key]), "contract_identity_required")
    need(value["horizon"] in SUPPORTED_HORIZONS, "registered_horizon_required")
    need(type(value["selection_seal_path"]) is str and Path(value["selection_seal_path"]).is_absolute(),
         "registered_selection_seal_path_required")
    need(is_sha256(value["native_contract_sha256"]), "native_contract_hash_required")
    exact_ref(value["benchmark_contract_ref"])
    exact_ref(value["evaluator_ref"])
    need(value["timing_policy"] is not None and value["phases"] is not None, "native_timing_protocol_required")
    need(value["known_missing_policy"] == "exclude_authenticated_clear_or_no_set"
         and value["unresolved_policy"] == "unresolved_population", "missingness_policy_differs")
    device = value["device_policy"]
    need(type(device) is dict and set(device) == {"kind", "physical_devices", "homogeneity_ref"}
         and device["kind"] == "registered_homogeneous_devices_selection_heuristic",
         "explicit_device_heuristic_required")
    _texts(device["physical_devices"], "registered_device_pool_required")
    exact_ref(device["homogeneity_ref"])
    _check_members(value["members"], device)
    return value


def expected_comparison(contract, member):
    """The full comparison context every trial of this member must carry."""
    return {
        "job_id": contract["job_id"], "measurement_epoch_id": member["measurement_epoch_id"],
        "correctness_case_ids": copy.deepcopy(contract["correctness_case_ids"]),
        "benchmark_case_ids": copy.deepcopy(contract["benchmark_case_ids"]),
        "benchmark_report_ids": copy.deepcopy(contract["benchmark_report_ids"]),
        "phases": copy.deepcopy(contract["phases"]),
        "evaluator_sha256": contract["evaluator_ref"]["sha256"],
        "contract_sha256": contract["native_contract_sha256"],
        "timing_policy": copy.deepcopy(contract["timing_policy"]),
        "runtime": {"image": contract["runtime_image"],
                    "config_digest": contract["runtime_config_digest"],
                    "physical_devices": copy.deepcopy(member["physical_devices"])},
        "statistic": STATISTIC,
    }


def _missing_kind(horizon, snapshot, trials):
    need(not trials, "missing_member_must_not_offer_trials")
    recommendation = horizon.get("recommendation")
    if recommendation is None:
        need(horizon.get("missing_reason") == "no_campaign_revision_before_horizon",
             "authenticated_no_campaign_reason_required")
    else:
        record = recommendation["record"]
        need(type(snapshot) is dict and recommendation["status"] == "missing" and recommendation["choice"] is None
             and ((record is None and recommendation["head"] is None)
                  or (type(record) is dict and record["disposition"] == "clear")),
             "authenticated_clear_or_no_set_required")
    return "clear" if recommendation and recommendation["record"] is not None else "no_set"


def _check_sources(choice):
    exact_ref(choice["source_manifest_ref"])
    sources = choice["source_body_refs"]
    need(type(sources) is list and sources, "selected_source_inventory_required")
    for source in sources:
        name = source.get("source_path") if type(source) is dict else None
        need(type(source) is dict and set(source) == _SOURCE_KEYS
             and type(name) is str and name
             and not PurePosixPath(name).is_absolute()
             and ".." not in PurePosixPath(name).parts and str(PurePosixPath(name)) == name
             and type(source["executable"]) is int and source["executable"] & ~0o111 == 0,
             "selected_source_descriptor_differs")
        if source["body_ref"] is None:
            need(source["availability"] == "absent_in_source_manifest", "absent_source_availability_differs")
        else:
            exact_ref(source["body_ref"])
            need(source["availability"] == "recorded", "recorded_source_availability_differs")
    need([row["source_path"] for row in sources] == sorted({row["source_path"] for row in sources}),
         "source_descriptors_must_be_sorted_unique")


def _observations(choice, trials, comparison, cases):
    """Distinct trials (by operation id) with their case-geomean; duplicates must be identical."""
    unique, retained, duplicate_positions = {}, [], []
    for position, row in enumerate(trials):
        need(encoded(row["comparison"]) == encoded(comparison), "unsupported_native_comparison_context")
        observation = row["native_trial"]
        verified = observation["verified"]
        need(observation["error"] is None and observation["request"] is not None
             and type(verified) is dict and verified["source_id"] == choice["source_id"],
             "native_trial_not_authenticated")
        operation = verified["operation_id"]
        need(type(operation) is str and operation, "exact_native_operation_identity_required")
        times = verified["per_case_times_ms"]
        need(type(times) is dict and set(times) == set(cases)
             and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                     for value in times.values()), "exact_positive_finite_benchmark_vector_required")
        identity = (choice["source_id"], operation)
        if identity in unique:
            need(unique[identity]["native_trial_sha256"] == digest(encoded(observation)),
                 "conflicting_duplicate_native_operation")
            duplicate_positions.append({"position": position, "same_as_position": unique[identity]["position"]})
            continue
        geometric = math.exp(math.fsum(math.log(times[case]) for case in cases) / len(cases))
        kept = {"position": position, "source_id": choice["source_id"], "operation_id": operation,
                "native_trial_sha256": digest(encoded(observation)),
                "per_case_times_ms": copy.deepcopy(times), "trial_geomean_ms": geometric,
                "reported_qualification_flags": {key: copy.deepcopy(verified[key]) for key in
                    ("measurement_valid", "custody_proven", "reward_eligible") if key in verified}}
        unique[identity] = kept
        retained.append(kept)
    return retained, duplicate_positions


def _member(contract, member, ref, authenticate_member):
    """Score one member from the callback's regenerated material (no caller numbers)."""
    material = authenticate_member(copy.deepcopy(ref), copy.deepcopy(member), copy.deepcopy(contract))
    need(type(material) is dict and set(material) == {"horizon_bytes", "trials"},
         "exact_authenticated_member_material_required")
    raw = material["horizon_bytes"]
    need(type(raw) is bytes and digest(raw) == ref["sha256"], "authenticated_horizon_bytes_differ")
    horizon = decoded(raw)
    need(horizon["schema"] == "geak-committed-horizon-endpoint-v1"
         and horizon["contract_ref"] == member["horizon_contract_ref"]
         and horizon["horizon"] == contract["horizon"], "authenticated_horizon_identity_differs")
    snapshot = horizon.get("snapshot")
    if snapshot is not None:
        need(type(snapshot) is dict and snapshot["campaign_id"] == member["campaign_id"], "horizon_campaign_differs")
    trials = material["trials"]
    need(type(trials) is list, "complete_authenticated_trial_list_required")
    result = {"member_id": member["member_id"], "horizon_ref": copy.deepcopy(ref),
              "status": horizon["status"], "native_material_sha256": digest(encoded(trials)),
              "comparison": expected_comparison(contract, member)}
    if horizon["status"] == "unresolved":
        need(not trials, "unresolved_member_must_not_offer_trials")
        return result
    if horizon["status"] == "missing":
        result["missing_kind"] = _missing_kind(horizon, snapshot, trials)
        return result
    recommendation = horizon.get("recommendation")
    need(horizon["status"] == "selected" and recommendation["status"] == "selected", "unsupported_member_status")
    choice = recommendation["choice"]
    binding = {"campaign_id": member["campaign_id"], "job_id": contract["job_id"],
               "measurement_epoch_id": member["measurement_epoch_id"]}
    need(choice["binding"] == binding and horizon["snapshot"]["campaign_id"] == member["campaign_id"],
         "selected_member_binding_differs")
    for key in ("source_id", "candidate_id"):
        need(is_sha256(choice[key]), "selected_content_identity_required")
    _check_sources(choice)
    witness = choice["selected_witness"]
    native = witness["native_trials"]
    need(witness["role"] == "selected" and type(native) is list and native,
         "complete_selected_native_witness_required")
    need(len(trials) == len(native) and all(type(row) is dict
         and set(row) == {"native_trial", "comparison"} for row in trials)
         and encoded([row["native_trial"] for row in trials]) == encoded(native),
         "native_trial_inventory_differs_no_subsets")
    need(encoded(witness["harvest_selected"]["timings"]) == encoded(
        [row["verified"]["per_case_times_ms"] for row in native]), "harvest_timing_inventory_differs")
    retained, duplicate_positions = _observations(choice, trials, result["comparison"],
                                                  contract["benchmark_case_ids"])
    score = statistics.median([row["trial_geomean_ms"] for row in retained])
    result.update(source_id=choice["source_id"], candidate_id=choice["candidate_id"],
                  choice=copy.deepcopy(choice), choice_sha256=digest(encoded(choice)),
                  value_ms=score, lower_is_better=True, statistic=STATISTIC,
                  supplied_trial_count=len(trials), observation_count=len(retained),
                  observations=retained, duplicate_positions=duplicate_positions,
                  authenticated_trial_contexts=copy.deepcopy(trials), uncertainty=None)
    return result


def _seal_once(path, body):
    """Write ``body`` once (create-only, mode 0400, fsynced) and read it back.

    The folder must be a directory that group/others cannot write and must not
    be swapped during the write; the saved file must be a single-link 0400
    regular file owned by this process's effective user, unchanged between the
    write and the read-back.
    """
    path = Path(path)
    need(path.is_absolute(), "absolute_new_seal_path_required")
    parent = path.parent.lstat()
    need(stat.S_ISDIR(parent.st_mode) and not parent.st_mode & 0o022,
         "seal_parent_must_be_trusted_directory")
    raw = encoded(body)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _check_parent_unchanged(path.parent, parent)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        saved = stream.read(len(raw) + 1)
        after = os.fstat(stream.fileno())
    named = path.lstat()
    need(stat.S_ISREG(before.st_mode) and stat.S_IMODE(before.st_mode) == 0o400
         and before.st_nlink == 1 and before.st_uid == os.geteuid()
         and all(getattr(before, key) == getattr(after, key) == getattr(named, key) for key in _SEAL_STAT)
         and saved == raw and decoded(saved) == body, "sealed_output_readback_differs")
    return {"path": str(path), "sha256": digest(raw)}


def _check_parent_unchanged(folder, parent):
    """The folder is still the one checked before the write; fsync it."""
    fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        actual = os.fstat(fd)
        need((actual.st_dev, actual.st_ino, actual.st_uid, actual.st_gid, actual.st_mode)
             == (parent.st_dev, parent.st_ino, parent.st_uid, parent.st_gid, parent.st_mode),
             "seal_parent_changed")
        os.fsync(fd)
    finally:
        os.close(fd)


def select_and_seal(contract_ref, contract_raw, member_refs, output_path, *, authenticate_member):
    """Choose across members and write the result once; returns ``(result, seal_ref)``.

    ``authenticate_member(ref, registered_member, contract)`` returns exactly
    ``{horizon_bytes: bytes, trials: [{native_trial: full original row,
    comparison: full context}]}``. It must verify the seal's origin and regenerate
    the exact choice; returning caller numbers, unverified JSON or a truthy flag
    does not implement it. Every member is checked even if one is unresolved.
    """
    contract = _contract(contract_raw, contract_ref)
    need(str(Path(output_path)) == contract["selection_seal_path"], "selection_seal_path_differs")
    need(callable(authenticate_member), "native_member_authenticator_required")
    need(type(member_refs) is dict and set(member_refs) == {m["member_id"] for m in contract["members"]},
         "exact_registered_member_reference_set_required")
    refs = {key: exact_ref(value) for key, value in member_refs.items()}
    results = []
    for member in contract["members"]:
        try:
            row = _member(contract, member, refs[member["member_id"]], authenticate_member)
        except Exception as exc:
            row = {"member_id": member["member_id"], "horizon_ref": refs[member["member_id"]],
                   "status": "unresolved", "error": str(exc) if isinstance(exc, SelectorError)
                   else "member_authentication_or_shape_failure"}
        results.append(row)
    candidates = [row for row in results if row["status"] == "selected"]
    status = ("unresolved" if any(row["status"] == "unresolved" for row in results)
              else "selected" if candidates else "missing")
    winner = (min(candidates, key=lambda row: (row["value_ms"], row["source_id"], row["candidate_id"],
                                               row["member_id"])) if status == "selected" else None)
    selected = None if winner is None else {key: copy.deepcopy(winner[key]) for key in
                                            ("member_id", "source_id", "candidate_id", "choice",
                                             "choice_sha256", "value_ms")}
    result = {"schema": SEAL_SCHEMA, "selector_contract_ref": copy.deepcopy(contract_ref),
              "implementation_ref": copy.deepcopy(contract["implementation_ref"]),
              "metric_source_ref": metric_source_ref(), "population_id": contract["population_id"],
              "operator": contract["operator"], "job_id": contract["job_id"], "horizon": contract["horizon"],
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status": status, "selected": selected, "members": results,
              "input_horizon_refs": refs, "benchmark_case_ids": copy.deepcopy(contract["benchmark_case_ids"]),
              "device_policy": copy.deepcopy(contract["device_policy"]),
              "allowed_member_context_differences": ["measurement_epoch_id", "runtime.physical_devices"],
              "statistic": STATISTIC, "observation_scope": "all trials in each authenticated frozen selected witness",
              "tie_policy": ["value_ms", "source_id", "candidate_id", "member_id"],
              "selection_scope": "native development heuristic across registered homogeneous devices",
              "same_context_performance_evidence_claimed": False, "independent_endpoint_qualification_claimed": False,
              "independent_endpoint_measurements_used": False, "fresh_timing_authorized_by_this_seal": False,
              "endpoint_consumer_must_bind_this_preexisting_seal": True,
              "no_new_search_action": True, "prospective_population_selection_rule": True,
              "no_installed_runtime_change": True, "mechanism_history_completeness": "not_established"}
    return result, _seal_once(output_path, result)
