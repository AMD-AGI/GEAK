"""Find what earlier engineer turns did with a saved parent, and pass it on.

Given a candidate, these functions find every settled engineer turn that
produced it (as a retained trial or selected, faithful, unfinished or terminal source), the
committed review of that turn, and its costs. The next engineer receives that
material together with any donors the planner chose.

Nothing here plans, promotes, keeps time or launches work; the evolution loop
and the assignment code do that. The agents choose the sources.
"""

import json

from .._util import require
from .refs import file_ref
from .trials import trial_history

ROLES = ("selected", "faithful", "unfinished", "terminal")



def need(condition, message):
    require(condition, message, ValueError)


def encoded(value):
    """Indented sorted JSON plus a newline (same bytes as the engineer runner's helper)."""
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def episode_review(store, snapshot, history, direction_id):
    """Return the exact committed review of one engineer turn."""
    reviews = [row for row in snapshot["state"]["history"]
               if row.get("review_kind") == "research_review"
               and row["proposal"]["direction_id"] == direction_id
               and all(row["proposal"][key] == history[key] for key in ("transport_ref", "collection_ref"))]
    need(reviews, "retained native episode requires its exact committed scientific review")
    event = reviews[-1]
    proposal = event["proposal"]
    need(event["direction_revision"] == proposal["expected_revision"] + 1,
         "native review revision differs from its committed action")
    state = store.get_revision(direction_id, event["direction_revision"])
    note = state["note"]
    need(state["tip"] == history["candidate_id"]
         and note.get("research_status") == "reported" and note.get("research_review_id")
         and note.get("reviewed_outcome_refs") == [history["transport_ref"], history["collection_ref"]]
         and note.get("research_update_ref") == history["collection_ref"]
         and note.get("latest_transport_ref") == history["transport_ref"]
         and all(note.get(key) == value for key, value in proposal["note"].items()),
         "native episode review differs from its immutable direction checkpoint")
    return {"revision": state["revision"], "note": note}


def retained_trial_candidates(store, artifacts):
    """Candidate IDs and trial indexes backed by the unchanged collected harvest.

    This checks source custody and assigned ancestry, never measurement quality.
    Missing sources remain unavailable; a failed trial with exact code can be
    continued after its owning episode receives the ordinary research review.
    """
    if "native_trial_sources.json" not in artifacts:
        return {}
    inventory = json.loads(artifacts["native_trial_sources.json"])
    need(inventory["schema_version"] == 1 and type(inventory["trials"]) is list,
         "invalid native trial source inventory")
    harvest = json.loads(artifacts["native_idea_harvest.json"] if "native_idea_harvest.json" in artifacts
                         else artifacts["archive/manifest.json"])
    roles = json.loads(artifacts["native_source_roles.json"])
    owner = store.get_candidate(roles["terminal"])
    found = {}
    for row in inventory["trials"]:
        identity = row["candidate_id"]
        if identity is None:
            continue
        index = row["trial_index"]
        need(row["source_retention_error"] is None and type(index) is int
             and 0 <= index < len(harvest["trials"])
             and row.get("owner_turn_id") == inventory.get("attempt_id")
             and type(inventory.get("attempt_id")) is str and inventory["attempt_id"],
             "retained native trial lacks its exact source or owning turn record")
        trial = harvest["trials"][index]
        source = trial.get("source_archive") or trial.get("verified") or trial.get("diagnostic") or {}
        candidate = store.get_candidate(identity)
        need(source.get("source_id") == row["source_id"] == candidate["source_id"]
             and all(candidate[key] == owner[key] for key in
                     ("job_id", "measurement_epoch_id", "parents", "donors")),
             "retained native trial source or assigned ancestry differs")
        found.setdefault(identity, []).append(index)
    return found


def parent_outcomes(store, snapshot, candidate_id, *, require_review=True):
    """List every settled turn that produced ``candidate_id`` in any role or trial.

    A turn may select a source other than its terminal one; all roles count.
    """
    candidate = store.get_candidate(candidate_id)
    need(all(candidate[key] == value for key, value in snapshot["state"]["binding"].items()),
         "native parent belongs to another job/epoch")
    owners = {row["opportunity_id"]: row["direction_id"]
              for row in snapshot["state"]["opportunities"]}
    outcomes = []
    for row in snapshot["state"]["history"]:
        if not (row.get("settled") is True and row.get("collection_ref")):
            continue
        envelope, artifacts = store.evidence(row["collection_ref"]["evidence_id"])
        need(envelope["candidate_id"] == row["candidate_id"], "native collection owner differs")
        if "native_source_roles.json" not in artifacts:
            continue
        roles = json.loads(artifacts["native_source_roles.json"])
        matched = [role for role, identity in roles.items() if identity == candidate_id]
        trial_indexes = retained_trial_candidates(store, artifacts).get(candidate_id, [])
        if not matched and not trial_indexes:
            continue
        harvest = json.loads(artifacts["native_idea_harvest.json"])
        if trial_indexes:
            inventory = json.loads(artifacts["native_trial_sources.json"])
            need(inventory["attempt_id"] == row["attempt_id"], "retained trial belongs to another episode")
        need(set(roles) <= set(ROLES)
             and all(harvest[role]["source_id"] == candidate["source_id"] for role in matched),
             "native parent role differs from its retained source")
        direction_id = owners[row["attempt_id"]]
        review = episode_review(store, snapshot, row, direction_id) if require_review else None
        collected = json.loads(artifacts[row["collection_ref"]["artifact"]])
        outcomes.append({"direction_id": direction_id, "attempt_id": row["attempt_id"],
            "roles": matched, "candidate_id": candidate_id, "source_id": candidate["source_id"],
            "transport_ref": row["transport_ref"], "collection_ref": row["collection_ref"],
            "collection_costs": collected["costs"], "review": review,
            **({"trial_indexes": trial_indexes,
                "trial_outcomes": [{"trial_index": index, "error": harvest["trials"][index].get("error"),
                    "native_trial_verified": harvest["trials"][index].get("verified") is not None,
                    "quality_scope": "native trial record; no accepted fitness implied"} for index in trial_indexes]}
               if trial_indexes else {})})
    return outcomes


def snapshot_trial_history(store, snapshot, evidence_ids=()):
    """Add saved episode/idea handles to the shared complete trial projection."""
    identities, attempts = set(evidence_ids), {}
    owners = {row["opportunity_id"]: row["direction_id"]
              for row in snapshot["state"]["opportunities"]}
    for row in snapshot["state"]["history"]:
        if not row.get("collection_ref"):
            continue
        identities.add(row["collection_ref"]["evidence_id"])
        direction = owners.get(row.get("attempt_id"))
        card = snapshot.get("archive", {}).get("directions", {}).get(direction, {}).get("card") or {}
        attempts[row.get("attempt_id")] = {"direction_id": direction,
            "idea_id": (card.get("idea") or {}).get("idea_id"), "engineer_id": None}
        if row.get("packet_ref"):
            _, data = store.evidence_artifact(row["packet_ref"]["evidence_id"], row["packet_ref"]["artifact"])
            proposal = json.loads(data).get("proposal") or {}
            attempts[row.get("attempt_id")]["engineer_id"] = proposal.get("engineer_id")
    return trial_history(store, identities, attempts=attempts)


def parent_prompt_context(store, snapshot, candidate_id, *, direction_id,
                          search_policy, campaign_root):
    """Build the parent part of an engineer prompt; return ``(text, snapshot_ref)``.

    The full outcome list and same-source trial history are saved as evidence
    on the parent. The prompt carries the latest outcome and complete-view counts.
    """
    from ..planner.context import outcome_prompt_context

    outcomes = parent_outcomes(store, snapshot, candidate_id)
    # Saved as ordinary evidence with exact notes, corrections, roles and costs.
    # No prose or flag in it becomes validated fitness.
    history = snapshot_trial_history(store, snapshot)
    candidate = store.get_candidate(candidate_id)
    history["trials"] = [row for row in history["trials"] if all(
        row[key] == candidate[key] for key in ("job_id", "measurement_epoch_id", "source_id"))]
    history["groups"] = [row for row in history["groups"] if all(
        row[key] == candidate[key] for key in ("job_id", "measurement_epoch_id", "source_id"))]
    history["coverage"]["input_inventory_count"] = history["coverage"]["inventory_count"]
    history["coverage"]["inventory_count"] = len({row["investigation_evidence_id"] for row in history["trials"]})
    history["coverage"]["trial_count"] = len(history["trials"])
    history["coverage"]["filter"] = "exact parent job, measurement epoch and source; all candidate identities retained"
    frozen = dict(snapshot, native_parent_outcomes=outcomes, native_trial_history=history)
    evidence = store.archive_evidence(candidate_id, {"native_parent_snapshot.json": encoded(frozen)})
    envelope, _ = store.evidence(evidence)
    snapshot_ref = file_ref(store.root / "objects" / envelope["artifacts"]["native_parent_snapshot.json"])
    latest = outcomes[-1] if outcomes else None
    owner = latest["direction_id"] if latest else direction_id
    context = outcome_prompt_context(store, snapshot, snapshot_ref,
        direction_id=owner, parent_candidate_id=candidate_id,
        search_policy=search_policy, campaign_root=campaign_root)
    if latest:
        summary = {key: latest[key] for key in ("direction_id", "attempt_id", "roles", "candidate_id",
                                                "source_id", "transport_ref", "collection_ref",
                                                "collection_costs")}
        if latest.get("trial_indexes"):
            summary["trial_indexes"] = latest["trial_indexes"]
            summary["trial_outcomes"] = latest["trial_outcomes"]
    else:
        summary = {"candidate_id": candidate_id,
                   "source_id": store.get_candidate(candidate_id)["source_id"],
                   "native_parent_outcome": None}
    summary["all_parent_outcomes_ref"] = {"file": snapshot_ref, "field": "native_parent_outcomes"}
    summary["exact_role_review"] = latest["review"] if latest else None
    summary["trial_history"] = {"file": snapshot_ref, "field": "native_trial_history",
        "coverage": history["coverage"], "scope": history["scope"], "identity": history["identity"]}
    context = ("Native parent identity and charged prior investigation costs. The selected/faithful/unfinished/terminal "
               "roles and agent review are retained observations/interpretations; no accepted anchor is implied. "
               "Raw parent role/cost history is in the referenced file.\n" + encoded(summary).decode() + "\n" + context)
    # The caller exposes only this returned ref; it holds every outcome.
    return context, snapshot_ref


def _object_ref(store, identity):
    return file_ref(store.root / "objects" / identity)


def _artifact_entry(store, ref, envelope):
    return {"evidence_ref": ref, "file": _object_ref(store, envelope["artifacts"][ref["artifact"]])}


def _claim_context(snapshot, candidate):
    from ..planner.context import current_claim_context  # Planner code; loaded only when needed.
    return current_claim_context(snapshot.get("claims", {}), binding={
        key: candidate[key] for key in ("job_id", "source_id", "measurement_epoch_id")})


def selected_parent_material(store, snapshot, proposal, immigrants=None):
    """Collect the working parent and each planner-chosen semantic donor.

    Donors are named in ``proposal["semantic_donors"]``; their contributions
    stay "unassessed" until a later review. ``immigrants`` maps candidates
    imported from another island to their origin records; their origin reviews
    are foreign observations, never local outcomes.
    """
    immigrants = immigrants or {}
    donors = proposal.get("semantic_donors", [])
    need(type(donors) is list and all(type(value) is str and value for value in donors)
         and len(donors) == len(set(donors)) and proposal["parent_candidate_id"] not in donors,
         "semantic donors must be distinct exact candidate IDs")
    selected = [proposal["parent_candidate_id"]] + donors
    material, sources = [], set()
    for candidate_id in selected:
        outcomes = parent_outcomes(store, snapshot, candidate_id)
        origin = [] if outcomes else immigrants.get(candidate_id, [])
        need(outcomes or origin or candidate_id == store.get(snapshot["campaign_id"])["tip"],
             "native parent must be a reviewed own source, a group immigrant or the exact campaign seed")
        need(not donors or outcomes or origin, "semantic combination needs reviewed native investigations")
        candidate = store.get_candidate(candidate_id)
        need(candidate["source_id"] not in sources, "semantic combination repeats the same source")
        sources.add(candidate["source_id"])
        source = store.source(candidate["source_id"])
        files = [{"path": row["path"], "executable": row["executable"],
                  "source_ref": _object_ref(store, row["blob_id"]) if row["blob_id"] else None}
                 for row in source["files"]]
        evidence = []
        pairs = [(outcome["transport_ref"], outcome["collection_ref"]) for outcome in outcomes]
        for record in origin:
            envelope, _ = store.evidence(record["evidence_id"])
            evidence.extend(_artifact_entry(store, {"evidence_id": record["evidence_id"], "artifact": name},
                                            envelope)
                            for name in sorted(envelope["artifacts"]))
            # Origin outcomes point at their local evidence copies.
            local = record["origin_evidence_map"]
            pairs += [tuple(dict(outcome[key], evidence_id=local[outcome[key]["evidence_id"]])
                            for key in ("transport_ref", "collection_ref")) for outcome in record["outcomes"]]
        for transport_ref, collection_ref in pairs:
            for ref in (transport_ref, collection_ref):
                envelope, _ = store.evidence(ref["evidence_id"])
                evidence.append(_artifact_entry(store, ref, envelope))
            envelope, artifacts = store.evidence(collection_ref["evidence_id"])
            for name in ("native_idea_harvest.json", "native_idea_context.json", "native_evidence_index.json",
                         "native_trial_sources.json"):
                if name in artifacts:
                    evidence.append(_artifact_entry(
                        store, {"evidence_id": collection_ref["evidence_id"], "artifact": name}, envelope))
        material.append({"candidate_id": candidate_id, "source_id": candidate["source_id"],
            "role": "working_parent" if candidate_id == selected[0] else "agent_selected_semantic_donor",
            "candidate_ref": _object_ref(store, candidate_id),
            "source_manifest_ref": _object_ref(store, candidate["source_id"]),
            "source_files": files, "reviewed_outcomes": outcomes, "evidence": evidence,
            "current_claim_context": _claim_context(snapshot, candidate),
            "contribution_status": "unassessed", **({"group_origin": origin} if origin else {}),
            "instruction": "Read actual code, implemented review, raw methods and failures. "
                "Inspect group_origin's newest imported origin_claim_context before relying on an inherited "
                "claim. Older origin publications are historical; their quantities are not local facts. "
                "Reconcile, rederive, modify or reject inherited parts through the native edit/test/profile/repair loop. In the final "
                "report identify which contribution survives, transforms, is rejected (with reason), or remains "
                "unknown in each selected/faithful/unfinished/terminal source. Delivery is not contribution survival."})
    return material
