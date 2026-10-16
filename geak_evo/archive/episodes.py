"""Check one finished engineer turn ("episode") against its saved records.

``committed_native_episode`` re-reads everything a settled turn left behind:
its committed history row and opportunity, the assignment packet, the run
receipt and prepared inputs, the prompt, the time reservation, the collected
evidence and the candidate roles. It returns the turn's material only when all
of them agree with each other and with the store.

``NativeEvidence`` uses that check to build the observation for a protected
(maturation) turn and to attach reviewed progress inventories to the loop.

Only committed history and saved source/evidence objects count here. Agents
still write the scientific reviews; nothing here awards quality credit.
"""

import json
from pathlib import Path

from .._util import canonical_json, require
from .claims import resolve_artifact
from .maturation import read_maturation
from .refs import file_ref, read_ref
from .succession import ROLES, episode_review

# Opportunity fields that must be unchanged between assignment and settlement.
_FIXED_OPPORTUNITY_FIELDS = (
    "job_id", "measurement_epoch_id", "opportunity_id", "direction_id", "island_id", "engineer_id",
    "question", "opened_at", "review_at", "requested", "reserved", "membership_ref")
_PROPOSAL_FIELDS = ("direction_id", "island_id", "engineer_id", "question", "review_at", "membership_ref")



def need(condition, message):
    require(condition, message, ValueError)


def _check_assignment(history, opportunity, packet_envelope, packet, assigned_candidate_id):
    parent = packet["proposal"]["parent_candidate_id"]
    need(history["packet_ref"]["artifact"] == "assignment.json" and packet_envelope["candidate_id"] == parent
         and (assigned_candidate_id is None or parent == assigned_candidate_id)
         and packet["opportunity"]["opportunity_id"] == history["attempt_id"]
         and packet["proposal"]["direction_id"] == opportunity["direction_id"],
         "native assignment owner or opportunity differs")
    # The packet holds the opportunity as it was when assigned (not yet settled).
    admitted = {key: opportunity[key] for key in _FIXED_OPPORTUNITY_FIELDS}
    admitted.update(receipts=[], settled=False, completed_sequence=None, progress_complete=False)
    need(packet["opportunity"] == admitted
         and all(packet["proposal"][key] == opportunity[key] for key in _PROPOSAL_FIELDS)
         and packet["proposal"]["reservation"] == opportunity["requested"] == opportunity["reserved"],
         "native immutable reservation or assignment differs")
    return parent


def _check_reservation(clock, probe, entry, opportunity, request):
    opened_ns = clock["started_monotonic_ns"] + int((opportunity["opened_at"] - clock["started_unix"]) * 1e9)
    need(opened_ns < probe["turn_deadline_ns"] <= probe["budget"]["deadline_ns"]
         and (probe["turn_deadline_ns"] - opened_ns) / 1e9 <= request["budget"]["wall_seconds"]
         and entry["timeout_ms"] / 1000 <= request["budget"]["wall_seconds"]
         and request["budget"] == {"wall_seconds": opportunity["reserved"]["wall_seconds"], "max_tokens": None},
         "native prepared work exceeds its immutable reservation")


def committed_native_episode(store, campaign_root, snapshot, history, assigned_candidate_id=None,
                             *, harvest, episode_material, scratch_root, runtime_refs):
    """Return ``(receipt, material)`` for one settled turn after checking all its records."""
    committed = store.get_revision(snapshot["campaign_id"], snapshot["revision"])
    need(committed["note"]["evolution"] == snapshot["state"], "native evidence snapshot is not committed")
    state = snapshot["state"]
    matches = [row for row in state["history"] if row.get("attempt_id") == history["attempt_id"]]
    need(matches == [history] and history.get("settled") is True, "exact settled native history required")
    opportunities = [row for row in state["opportunities"] if row["opportunity_id"] == history["attempt_id"]]
    need(len(opportunities) == 1 and opportunities[0]["settled"] is True,
         "native opportunity is uncommitted or unsettled")
    opportunity = opportunities[0]
    candidate = store.get_candidate(history["candidate_id"])
    need(all(candidate[key] == value for key, value in state["binding"].items()), "native candidate domain differs")

    # Assignment packet.
    packet_envelope, packet_raw = resolve_artifact(history["packet_ref"], store.evidence)
    packet = json.loads(packet_raw)
    parent = _check_assignment(history, opportunity, packet_envelope, packet, assigned_candidate_id)
    assigned = store.get_candidate(parent)
    attempt = Path(campaign_root) / "attempts" / history["attempt_id"]
    need(attempt.parent == Path(campaign_root) / "attempts" and attempt.resolve(strict=True) == attempt,
         "native attempt path is not exact canonical host storage")

    # Run receipt and request.
    transport_envelope, raw = resolve_artifact(history["transport_ref"], store.evidence)
    _, transport = store.evidence(history["transport_ref"]["evidence_id"])
    receipt_ref = file_ref(attempt / "receipt.json")
    receipt = json.loads(raw)
    need(history["transport_ref"]["artifact"] == "transport.json"
         and transport_envelope["candidate_id"] == history["candidate_id"]
         and json.loads(read_ref(receipt_ref)) == receipt
         and receipt["captured_tip_source_id"] == candidate["source_id"],
         "native transport is not the exact candidate-bound host receipt")
    from ..engineer.request import validate_request  # Engineer code; loaded only when needed.
    request = validate_request(receipt["request"])
    need(request["tip"]["candidate_id"] == parent and request["tip"]["source_id"] == assigned["source_id"]
         and request["next_question"] == packet["proposal"]["question"]
         and request["direction_id"] == opportunity["direction_id"],
         "native request changed assigned source or question")

    # Prepared inputs, prompt and time reservation.
    prepared = json.loads(transport["native_prepared.json"])
    need(json.loads(read_ref(file_ref(attempt / "native_prepared.json"))) == prepared,
         "retained native preparation differs from committed transport")
    need(type(runtime_refs) is dict
         and set(runtime_refs) == {"entry_runner_ref", "collector_ref", "entry_provenance_ref"}
         and all(prepared[key] == value for key, value in runtime_refs.items()),
         "native episode used another admitted entry, collector or provenance")
    for ref in prepared.values():
        read_ref(ref)
    _, assignment_artifacts = store.evidence(history["packet_ref"]["evidence_id"])
    prompt_ref = file_ref(attempt / "prompt.txt")
    need(read_ref(prompt_ref) == transport["prompt.txt"] == assignment_artifacts["engineer_prompt.txt"],
         "native prompt differs from the committed assignment")
    from ..engineer.investigation import inspect_prepared_identity  # Engineer code; loaded only when needed.
    entry, config, collect = inspect_prepared_identity(prepared, request, prompt_ref=prompt_ref,
                                                       operation_id=history["attempt_id"])
    clock = json.loads(read_ref(config["backend_config"]["clock_ref"]))
    _check_reservation(clock, config["backend_config"]["probe_config"], entry, opportunity, request)

    # Collected evidence.
    collection_envelope, _ = resolve_artifact(history["collection_ref"], store.evidence)
    _, artifacts = store.evidence(history["collection_ref"]["evidence_id"])
    need(collection_envelope["candidate_id"] == history["candidate_id"], "native collection owner differs")
    manifest_ref = file_ref(attempt / "collection/archive/manifest.json")
    inventory = json.loads(artifacts["native_evidence_index.json"])
    manifests = [row for row in inventory if row["original_ref"] == manifest_ref]
    need(len(manifests) == 1, "native manifest lacks its exact committed raw evidence index")
    raw_envelope, manifest_raw = resolve_artifact(manifests[0]["evidence_ref"], store.evidence)
    need(raw_envelope["candidate_id"] == history["candidate_id"]
         and manifest_raw == read_ref(manifest_ref)
         and json.loads(artifacts["native_idea_harvest.json"]) == json.loads(manifest_raw),
         "native raw and summary harvest records differ")

    # The material read through the host's episode reader.
    expected = dict(state["binding"], direction_id=opportunity["direction_id"], turn_id=history["attempt_id"],
                    starting_source_id=assigned["source_id"])
    outer, material = episode_material(receipt_ref, manifest_ref, expected, harvest=harvest,
                                       scratch_root=scratch_root)
    refs = material["refs"]["native_entry_binding"]
    need(refs["entry_input_ref"]["sha256"] == prepared["entry_input_ref"]["sha256"]
         and refs["entry_provenance_ref"]["sha256"] == prepared["entry_provenance_ref"]["sha256"]
         and read_ref(refs["entry_result_ref"]) == transport["native_entry_result.json"]
         and json.loads(transport["native_receipt.json"]) == material["native_receipt"],
         "native entry/receipt differs from its admitted and archived producer")

    # Candidate roles.
    roles = json.loads(artifacts["native_source_roles.json"])
    need(set(roles) == {role for role in ROLES if material["harvest"].get(role) is not None}
         and roles.get("terminal") == history["candidate_id"]
         and all(store.get_candidate(identity)["source_id"] == material["harvest"][role]["source_id"]
                 and all(store.get_candidate(identity)[key] == value for key, value in state["binding"].items())
                 for role, identity in roles.items()), "native candidate roles differ from rechecked sources")
    before = {row["path"]: row["blob_id"] for row in store.source(assigned["source_id"])["files"]}
    after = {row["path"]: row["blob_id"] for row in store.source(candidate["source_id"])["files"]}
    material["host_provenance"] = {"packet_ref": history["packet_ref"], "transport_ref": history["transport_ref"],
        "collection_ref": history["collection_ref"], "raw_manifest_ref": manifests[0]["evidence_ref"],
        "runtime_refs": runtime_refs, "candidate_roles": roles, "source_bytes_changed": before != after,
        "scope": "Committed host episode material, not scientific fidelity, gain or endpoint eligibility."}
    return outer, material


class NativeEvidence:
    """Episode checks bound to one evolution loop and campaign directory."""

    def __init__(self, loop, campaign_root, *, harvest, episode_material, scratch_root, runtime_refs):
        self.loop, self.store, self.root = loop, loop.store, Path(campaign_root)
        self.harvest, self.episode_material = harvest, episode_material
        self.scratch_root, self.runtime_refs = scratch_root, runtime_refs

    def episode(self, snapshot, history, assigned_candidate_id=None):
        return committed_native_episode(self.store, self.root, snapshot, history, assigned_candidate_id,
            harvest=self.harvest, episode_material=self.episode_material, scratch_root=self.scratch_root,
            runtime_refs=self.runtime_refs)

    def observation(self, snapshot, direction_id):
        """Describe what the granted protected turn of ``direction_id`` actually did."""
        current = read_maturation(self.store, direction_id)
        need(current == snapshot["maturation"][direction_id],
             "native observation requires the current committed grant")
        grant = current["grant"]
        need(grant is not None, "native protection has no granted opportunity")
        matches = [row for row in snapshot["state"]["history"] if row.get("attempt_id") == grant["opportunity_id"]]
        need(len(matches) == 1, "protected opportunity has no exact native history")
        history = matches[0]
        receipt, material = self.episode(snapshot, history, grant["start_candidate_id"])
        need(receipt["request"]["direction_id"] == direction_id, "protected source delivered to another direction")
        # The protected faithful source and its witnesses must have been given to the engineer.
        seed = grant["faithful_seed"]
        candidate = self.store.get_candidate(seed["candidate_id"])
        identities = [seed["candidate_id"], candidate["source_id"]] + [
            row["blob_id"] for row in self.store.source(candidate["source_id"])["files"] if row["blob_id"]]
        delivered = [file_ref(self.store.root / "objects" / identity) for identity in identities]
        for ref in seed["evidence_refs"]:
            envelope, _ = resolve_artifact(ref, self.store.evidence)
            delivered.append(file_ref(self.store.root / "objects" / envelope["artifacts"][ref["artifact"]]))
        need(all(ref in receipt["request"]["read_only"] for ref in delivered),
             "protected faithful source and witnesses were not delivered")
        # A partial return can still prove delivery and cleanup, but a no-edit
        # turn counts as a review only when the investigation completed.
        complete = material["explanation_check"]["investigation_complete"]
        changed = material["host_provenance"]["source_bytes_changed"]
        work_kind = "implementation_or_diagnosis" if changed else "review_only" if complete else "none"
        value = {"kind": "maturation_observation", "binding": snapshot["state"]["binding"],
            "direction_id": direction_id, "opportunity_id": grant["opportunity_id"],
            "assigned_candidate_id": grant["start_candidate_id"], "candidate_id": history["candidate_id"],
            "delivered": True, "work_kind": work_kind,
            "settled": True, "interrupted": receipt["transport_complete"] is not True}
        provenance = dict(material["host_provenance"], refs=material["refs"],
                          explanation_check=material["explanation_check"], grant=grant)
        return value, provenance

    def review_assignment(self, snapshot, proposal):
        """Save the observation for a protected turn and return its ref and verifier."""
        if proposal["decision"]["action"] == "review":
            need(proposal["decision"].get("outcome") not in ("realize", "refute"),
                 "no scoped native realization/refutation verifier is configured")
            return {}
        value, provenance = self.observation(snapshot, proposal["direction_id"])
        identity = self.store.archive_evidence(value["candidate_id"], {
            "maturation_observation.json": canonical_json(value),
            "native_work_provenance.json": canonical_json(provenance)})
        ref = {"evidence_id": identity, "artifact": "maturation_observation.json"}

        def verify(ref, raw, expected):
            actual, proof = self.observation(self.loop.snapshot(), expected["direction_id"])
            envelope, artifacts = self.store.evidence(ref["evidence_id"])
            return (ref["artifact"] == "maturation_observation.json" and envelope["candidate_id"] == actual["candidate_id"]
                    and raw == canonical_json(actual) and artifacts[ref["artifact"]] == raw
                    and artifacts["native_work_provenance.json"] == canonical_json(proof)
                    and all(actual.get(key) == value for key, value in expected.items()))
        return {"observation_ref": ref, "verify_host": verify}

    def record_reviewed_progress(self, snapshot):
        """Attach each reviewed turn's inventory to the loop; claims no gain coverage."""
        for history in snapshot["state"]["history"]:
            if not history.get("attempt_id") or not history.get("settled"):
                continue
            opportunity = next(row for row in snapshot["state"]["opportunities"]
                               if row["opportunity_id"] == history["attempt_id"])
            if opportunity.get("progress_review_refs"):
                continue
            direction = opportunity["direction_id"]
            if not any(row.get("review_kind") == "research_review" and
                       row["proposal"].get("transport_ref") == history["transport_ref"]
                       for row in snapshot["state"]["history"]):
                continue
            review = episode_review(self.store, snapshot, history, direction)
            _, material = self.episode(snapshot, history)
            value = {"kind": "evolution_progress_review", "binding": snapshot["state"]["binding"],
                     "opportunity_id": history["attempt_id"], "complete": False}
            proof = {"host_provenance": material["host_provenance"], "review": review,
                     "trial_refs": [row["feedback_ref"] for row in material["native_trials"]],
                     "diagnostic_refs": material["diagnostic_refs_not_revalidated"],
                     "scope": "Inventory and explicit review are bound; complete gain coverage remains unavailable. "
                              "Passing development results have no common comparator/uncertainty credit contract; "
                              "failed or incomplete diagnostics are not inferred zero gain. No fitness is created."}
            identity = self.store.archive_evidence(history["candidate_id"], {
                "evolution_progress_review.json": canonical_json(value),
                "native_progress_inventory.json": canonical_json(proof)})
            ref = {"evidence_id": identity, "artifact": "evolution_progress_review.json"}

            def verify(actual_ref, raw, expected):
                return actual_ref == ref and raw == canonical_json(value) and all(
                    value[key] == item for key, item in expected.items())
            self.loop.record_progress_review(history["attempt_id"], ref, verify_review=verify)
            snapshot = self.loop.snapshot()
        return snapshot
