"""Complete, rebuildable trial views. The saved evidence remains authoritative.

Rows describe supplied collector inventories, not every action an engineer took.
No outcome, stage, comparator or timing is inferred from missing data or prose.
"""

import json

from .._util import sha256_hex
from .store import StateError

SCOPE = ("Every row in the supplied retained trial inventories, including missing sources and feedback. "
         "In-flight work, uncollected episodes and requests absent from those inventories are unknown. "
         "A missing row does not establish that a source or idea was never tested.")


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _feedback_stages(feedback, row):
    """Read the native producer's nested receipt without inventing lifecycle state."""
    stages = {"request": "operation_recorded" if row.get("operation_id") else "unknown",
              "admission": None, "execution_started": None, "correctness": None,
              "benchmark": None, "native_suite_pass_observed": None,
              "host_errors": None, "collection_errors": None,
              "reported_status": None, "error_kind": None}
    if feedback.get("schema") != "geak-native-probe-feedback-v1" or feedback.get("profile") != "native":
        return stages, None
    receipt = feedback.get("receipt")
    if not isinstance(receipt, dict) or receipt.get("schema") != "geak-arena-evaluation-v1":
        return stages, None
    binding = receipt.get("binding")
    if (not isinstance(binding, dict) or receipt.get("evaluation_id") != row.get("operation_id")
            or binding.get("candidate_source_id") != row.get("source_id")
            or any(binding.get(key) != row[key] for key in ("job_id", "measurement_epoch_id"))):
        return stages, "feedback_receipt_binding_conflict"
    if type(receipt.get("native_suite_pass_observed")) is bool:
        stages["native_suite_pass_observed"] = receipt["native_suite_pass_observed"]
    errors = receipt.get("host_errors")
    if isinstance(errors, list) and all(isinstance(item, str) for item in errors):
        stages["host_errors"] = {"count": len(errors), "json_pointer": "/receipt/host_errors"}
    errors = receipt.get("collection_errors")
    if isinstance(errors, list) and all(isinstance(item, dict)
            and all(isinstance(item.get(key), str) for key in ("stage", "error_kind")) for item in errors):
        stages["collection_errors"] = {"count": len(errors), "json_pointer": "/receipt/collection_errors"}
    native = receipt.get("native_observations")
    if isinstance(native, dict):
        for phase in ("correctness", "benchmark"):
            value = native.get(phase)
            status = value.get("status") if isinstance(value, dict) else None
            if isinstance(status, str) and status in ("passed", "failed", "not_run", "unknown"):
                stages[phase] = status
    return stages, None


def trial_attempt(row, attempts):
    """Keep idea handles collection-bound; agreed direction/engineer may carry."""
    matches = [attempts[(row["attempt_id"], identity)] for identity in row["evidence_ids"]
               if (row["attempt_id"], identity) in attempts]
    if matches and all(item == matches[0] for item in matches):
        return matches[0]
    metadata = [{field: value.get(field) for field in ("direction_id", "engineer_id")}
                for key, value in attempts.items() if key == row["attempt_id"]
                or (isinstance(key, tuple) and key[0] == row["attempt_id"])]
    return metadata[0] if metadata and all(item == metadata[0] for item in metadata) else {}


def trial_history(store, evidence_ids, *, attempts=None):
    """Project exact inventories and their linked feedback; never follow external paths.

    ``attempts`` only adds the caller's saved direction/idea/engineer handles.
    Saved projections key them by (attempt_id, collection_evidence_id). An
    attempt_id-only mapping can supply agreed direction/engineer metadata,
    never an idea from another collection.
    Duplicate collection/raw copies merge only through native_evidence_index;
    repeated operations and independent investigations remain separate rows.
    """
    attempts = attempts or {}
    evidence, links, payloads = {}, {}, {}

    def raw_artifact(identity, name):
        blob = evidence[identity]["artifacts"].get(name)
        if blob is None:
            return None
        if blob not in payloads:
            payloads[blob] = store._bytes(blob)
        return payloads[blob]

    def artifact(identity, name):
        raw = raw_artifact(identity, name)
        return json.loads(raw) if raw is not None else None

    pending = sorted(set(evidence_ids))
    while pending:
        identity = pending.pop(0)
        if identity in evidence:
            continue
        # Preserve candidate/ancestry checks; do not load unrelated raw builds,
        # traces or other payloads merely to read an evidence envelope.
        envelope = evidence[identity] = store._bound_evidence(identity)
        for link in artifact(identity, "native_evidence_index.json") or []:
            ref = link["evidence_ref"]
            pending.append(ref["evidence_id"])
            if link.get("original_ref"):
                domain = envelope["binding"]
                links.setdefault((domain["job_id"], domain["measurement_epoch_id"],
                                  _key(link["original_ref"])), []).append(ref)

    links = {key: sorted({_key(ref): ref for ref in refs}.values(), key=_key) for key, refs in links.items()}
    observations, neighbors = {}, {}
    for identity, envelope in sorted(evidence.items()):
        artifacts = envelope["artifacts"]
        binding = envelope["binding"]
        for index, row in enumerate((artifact(identity, "native_idea_context.json") or {}).get("native_observations", [])):
            key = (binding["job_id"], binding["measurement_epoch_id"], row.get("source_id"),
                   row.get("operation_id"), _key(row.get("feedback_ref")))
            observations.setdefault(key, []).append((row, {"evidence_id": identity,
                "artifact": "native_idea_context.json", "field": "native_observations", "index": index}))
        inventory = artifacts.get("native_trial_sources.json")
        if inventory is None:
            continue
        neighbors.setdefault(identity, set())
        for link in artifact(identity, "native_evidence_index.json") or []:
            other = link["evidence_ref"]["evidence_id"]
            target = evidence[other]
            if (target["artifacts"].get("native_trial_sources.json") == inventory
                    and all(target["binding"][key] == binding[key] for key in ("job_id", "measurement_epoch_id"))):
                neighbors[identity].add(other)
                neighbors.setdefault(other, set()).add(identity)

    # A collection can point to several raw copies, and copies can link onward.
    # Canonicalize the entire undirected component, including the collection.
    groups = {}
    for identity in sorted(neighbors):
        if identity in groups:
            continue
        pending, component = [identity], set()
        while pending:
            member = pending.pop()
            if member not in component:
                component.add(member)
                pending.extend(neighbors[member])
        groups.update(dict.fromkeys(component, min(component)))

    rows = {}
    for identity, envelope in sorted(evidence.items()):
        artifacts = envelope["artifacts"]
        inventory = artifact(identity, "native_trial_sources.json")
        if inventory is None:
            continue
        harvest_name = next((name for name in ("native_idea_harvest.json", "archive/manifest.json")
                             if name in artifacts), None)
        harvest = artifact(identity, harvest_name) if harvest_name else None
        roles = artifact(identity, "native_source_roles.json") or {}
        if harvest is not None and len(harvest["trials"]) != len(inventory["trials"]):
            raise StateError("trial inventory and harvest counts differ")
        for position, trial in enumerate(inventory["trials"]):
            index = trial["trial_index"]
            if type(index) is not int or index != position:
                raise StateError("trial index differs from inventory position")
            binding = {key: envelope["binding"][key] for key in ("job_id", "measurement_epoch_id")}
            if trial.get("candidate_id"):
                candidate = store.get_candidate(trial["candidate_id"])
                if candidate["source_id"] != trial.get("source_id") or any(candidate[key] != value for key, value in binding.items()):
                    raise StateError("trial source binding differs from candidate")
            row_key = groups[identity], _key(trial)
            row = rows.setdefault(row_key, dict(trial, **binding, attempt_id=inventory.get("attempt_id"),
                investigation_evidence_id=groups[identity], evidence_id=identity, evidence_ids=[],
                inventory_refs=[], raw_trial_refs=[], source_roles=[], outcomes=[]))
            row["evidence_ids"].append(identity)
            row["inventory_refs"].append({"evidence_id": identity, "artifact": "native_trial_sources.json",
                                           "field": "trials", "index": index})
            row["source_roles"] = sorted(set(row["source_roles"]) | {role for role, candidate_id in roles.items()
                                                   if candidate_id == trial.get("candidate_id")})
            if harvest is not None:
                raw = harvest["trials"][index]
                if raw.get("archive_ref") != trial.get("feedback_ref"):
                    raise StateError("trial inventory differs from retained harvest")
                outcome = {"error": raw.get("error"), "status": raw.get("status"),
                    "source_archive_error": raw.get("source_archive_error"),
                    "verified_status": (raw.get("verified") or {}).get("status"),
                    "diagnostic_status": (raw.get("diagnostic") or {}).get("status")}
                if outcome not in row["outcomes"]:
                    row["outcomes"].append(outcome)
                row["raw_trial_refs"].append({"evidence_id": identity, "artifact": harvest_name,
                                             "field": "trials", "index": index})

    result = []
    for row in rows.values():
        row["trial_id"] = sha256_hex(_key([row["investigation_evidence_id"], row["trial_index"]]).encode())
        handles = trial_attempt(row, attempts)
        row.update({key: handles.get(key)
                    for key in ("direction_id", "idea_id", "engineer_id", "idea_attribution")})
        row["outcome"] = row["outcomes"][0] if len(row["outcomes"]) == 1 else None
        row["raw_trial_ref"] = row["raw_trial_refs"][0] if row["raw_trial_refs"] else None
        key = (row["job_id"], row["measurement_epoch_id"], row.get("source_id"),
               row.get("operation_id"), _key(row.get("feedback_ref")))
        matches = observations.get(key, [])
        unique = {_key(value): value for value, _ in matches}
        row["observation_refs"] = [ref for _, ref in matches]
        conflicts = []
        if row.get("feedback_error"):
            conflicts.append("collector_feedback_error")
        if len(unique) > 1:
            conflicts.append("conflicting_observations")
        if len(row["outcomes"]) > 1:
            conflicts.append("conflicting_outcomes")
        bound = bool(row.get("source_id") and row.get("operation_id") and row.get("feedback_ref"))
        observation = next(iter(unique.values())) if bound and len(unique) == 1 and not conflicts else None
        row["observation"] = observation
        row["times_ms"] = observation.get("per_case_times_ms") if observation else None
        row["timing_context"] = {"units": "ms" if row["times_ms"] is not None else None,
            "case_ids": sorted(row["times_ms"]) if isinstance(row["times_ms"], dict) else [],
            "raw_benchmarks": observation.get("raw_benchmarks", []) if observation else [],
            "comparator_source_id": None, "comparison": "not derived"}
        feedbacks, feedback_refs = {}, []
        feedback_key = row["job_id"], row["measurement_epoch_id"], _key(row.get("feedback_ref"))
        for ref in links.get(feedback_key, []):
            envelope = evidence[ref["evidence_id"]]
            raw = raw_artifact(ref["evidence_id"], ref["artifact"])
            if raw is None or sha256_hex(raw) != row["feedback_ref"].get("sha256"):
                conflicts.append("feedback_reference_conflict")
                continue
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                conflicts.append("invalid_feedback_json")
                continue
            if not isinstance(value, dict):
                conflicts.append("invalid_feedback_object")
                continue
            if (not bound or not row.get("owner_turn_id") or not row.get("attempt_id")
                    or value.get("source_id") != row.get("source_id") or value.get("operation_id") != row.get("operation_id")
                    or value.get("turn_id") != row.get("owner_turn_id")
                    or row.get("owner_turn_id") != row.get("attempt_id")
                    or any(envelope["binding"][key] != row[key] for key in ("job_id", "measurement_epoch_id"))):
                conflicts.append("feedback_binding_conflict")
                continue
            feedbacks[_key(value)] = value
            feedback_refs.append(ref)
        feedback = next(iter(feedbacks.values())) if len(feedbacks) == 1 and not conflicts else {}
        if len(feedbacks) > 1:
            conflicts.append("conflicting_feedback")
        stages, stage_conflict = _feedback_stages(feedback, row)
        if stage_conflict:
            conflicts.append(stage_conflict)
        if conflicts:
            row["observation"], row["times_ms"] = None, None
            row["timing_context"] = {"units": None, "case_ids": [], "raw_benchmarks": [],
                                     "comparator_source_id": None, "comparison": "not derived"}
        row["feedback_refs"] = feedback_refs
        row["feedback_state"] = "conflict" if conflicts else "available" if feedback else "unavailable"
        row["stages"] = stages
        row["association"] = {"status": "conflict" if conflicts else "matched_observation" if observation else "unknown",
                              "issues": sorted(set(conflicts))}
        row.update(kind="native_trial", evaluation_id=row.get("operation_id"),
            status="recorded_observation" if row["observation"] else "outcome_recorded" if row["outcomes"] else "incomplete",
            basis="retained native development trial; no endpoint qualification or idea verdict inferred")
        result.append(row)
    result.sort(key=lambda row: (row["job_id"], row["measurement_epoch_id"], row.get("source_id") or "", row["trial_id"]))
    grouped = {}
    for row in result:
        key = row["job_id"], row["measurement_epoch_id"], row.get("source_id")
        grouped.setdefault(key, []).append(row)
    return {"schema": "geak-evo-trial-history-v1", "scope": SCOPE,
        "identity": {"trial_id": "canonical row ID within this linked-component view; later links may change it",
            "stable_references": "inventory_refs identify exact immutable input rows, independent of grouping"},
        "coverage": {"inventory_count": len(set(groups.values())), "trial_count": len(result),
                     "count_label": "retained trial rows (any state)",
                     "omitted_rows": 0, "outside_inventory": "unknown"},
        "groups": [{"job_id": key[0], "measurement_epoch_id": key[1], "source_id": key[2],
                    "candidate_ids": sorted({row["candidate_id"] for row in values if row.get("candidate_id")}),
                    "trial_ids": [row["trial_id"] for row in values]} for key, values in grouped.items()],
        "trials": result}
