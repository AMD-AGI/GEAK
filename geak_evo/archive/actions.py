"""The fixed set of research actions an agent may take on the archive.

Actions are: accept an anchor, nominate or revive (maturation), write an island
charter, change memberships, record a concept, and record an exchange. Each one
only writes through the existing typed record functions.

``preview_action`` runs the same handlers against an in-memory evidence overlay.
It checks the action and predicts the exact references it would produce without
writing objects or changing direction state; only the evolution loop saves and
adopts the result.
"""

import copy

from .._util import EvoError, canonical_json, require, sha256_hex
from .exchange import resolve_artifact, validate_exchange
from .maturation import checkpoint_maturation
from .records import archive_checkpoint, archive_record, read_record


class ResearchActionError(EvoError, ValueError):
    pass


_json = canonical_json


def _fields(value, required, optional=()):
    require(type(value) is dict and set(required) <= set(value) <= set(required) | set(optional),
            "invalid research action payload fields", ResearchActionError)


def _text(value):
    require(type(value) is str and bool(value.strip()), "research action needs nonempty scientific text",
            ResearchActionError)


def without_timestamp(state):
    return {key: value for key, value in state.items() if key != "updated_at"}


class _Preview:
    """Store stand-in: reads go to the real store, writes stay in memory."""

    def __init__(self, store):
        self.store, self.objects, self.committed = store, {}, None

    def __getattr__(self, name):
        # The handlers below use only these read APIs.
        require(name in ("get", "get_candidate"), "unsupported preview operation", ResearchActionError)
        return getattr(self.store, name)

    def archive_evidence(self, candidate_id, artifacts):
        candidate = self.store.get_candidate(candidate_id)
        envelope = {"kind": "evidence", "candidate_id": candidate_id,
                    "binding": {"job_id": candidate["job_id"], "measurement_epoch_id": candidate["measurement_epoch_id"],
                                "candidate_source_id": candidate["source_id"]},
                    "artifacts": {name: sha256_hex(data) for name, data in artifacts.items()}}
        identity = sha256_hex(_json(envelope))
        self.objects[identity] = (envelope, artifacts)
        return identity

    def evidence(self, identity):
        return self.objects[identity] if identity in self.objects else self.store.evidence(identity)

    def checkpoint(self, direction_id, candidate_id, *, note, expected_revision,
                   status="active", resources=None, evidence=()):
        prior = self.store.get(direction_id)
        require(self.committed is None and prior["revision"] == expected_revision, "invalid preview checkpoint",
                ResearchActionError)
        self.committed = dict(without_timestamp(prior), revision=expected_revision + 1,
                              tip=candidate_id, note=note, status=status,
                              resources=prior["resources"] if resources is None else resources, evidence=list(evidence))
        return self.committed

    def promote(self, direction_id, candidate_id, evidence_id, *, expected_revision):
        prior = self.store.get(direction_id)
        require(self.committed is None and prior["revision"] == expected_revision,
                "invalid preview promotion", ResearchActionError)
        envelope, artifacts = self.evidence(evidence_id)
        candidate = self.store.get_candidate(candidate_id)
        require(envelope["candidate_id"] == candidate_id and callable(self.store.verify_evidence),
                "promotion requires candidate-bound evidence and a host verifier", ResearchActionError)
        checks = self.store.verify_evidence(artifacts, self.store._binding(candidate))
        proposed = dict(without_timestamp(prior), revision=expected_revision + 1,
                        anchor={"candidate_id": candidate_id, "evidence_id": evidence_id, "checks": checks})
        self.store._validate(proposed)
        self.committed = proposed
        return proposed


def preview_maturation(store, direction_id, *, decision, expected_revision, accounting):
    prepared = prepare_maturation(store, direction_id, decision=decision,
                                  expected_revision=expected_revision, accounting=accounting)
    return {key: prepared[key] for key in ("ref", "checkpoint")}


def prepare_maturation(store, direction_id, *, decision, expected_revision, accounting,
                       observation_ref=None, verify_host=None):
    """Check a maturation decision once and keep the exact transition for saving or replay."""
    preview = _Preview(store)
    result = checkpoint_maturation(preview, direction_id, decision=decision,
                                   expected_revision=expected_revision, accounting=accounting,
                                   observation_ref=observation_ref, verify_host=verify_host)
    return {"ref": result["ref"], "checkpoint": preview.committed, "record": result["record"]}


def _current_card(store, direction, frozen_cases):
    ref = direction["note"].get("archive_card_ref")
    require(ref is not None, "membership action requires a current archive card", ResearchActionError)
    return ref, read_record(store, ref, frozen_cases=frozen_cases)


def _card_refs(card):
    return card.get("membership_refs", []) + card.get("source_membership_refs", [])


def _charter_head(store, direction, records, island_id, frozen_cases):
    """Current charter reference for an island: indexed, else found from same-job memberships."""
    indexed = records.get("charters", {}).get(island_id)
    if indexed is not None:
        return indexed
    found = {}
    for existing in store.get().values():
        if existing["job_id"] != direction["job_id"] or existing["measurement_epoch_id"] != direction["measurement_epoch_id"]:
            continue
        if existing["note"].get("archive_card_ref"):
            _, card = _current_card(store, existing, frozen_cases)
            for ref in _card_refs(card):
                member = read_record(store, ref, frozen_cases=frozen_cases)
                if member["island_id"] == island_id:
                    found[_json(member["charter_ref"])] = member["charter_ref"]
    require(len(found) <= 1, "existing charter heads are ambiguous; reconcile island identity first",
            ResearchActionError)
    return next(iter(found.values()), None)


def _accept_anchor(store, proposal, payload):
    _fields(payload, {"nominee_id", "evidence_ref", "reason"})
    _text(payload["nominee_id"])
    _text(payload["reason"])
    _fields(payload["evidence_ref"], {"evidence_id", "artifact"})
    require(payload["evidence_ref"]["artifact"] == "native_evaluation.json",
            "anchor nomination requires the exact native evaluation", ResearchActionError)
    resolve_artifact(store, payload["evidence_ref"])
    store.promote(proposal["direction_id"], payload["nominee_id"], payload["evidence_ref"]["evidence_id"],
                  expected_revision=proposal["expected_revision"])
    return {"anchor_ref": payload["evidence_ref"]}


def _charter(store, direction, candidate_id, payload, records, frozen_cases):
    _fields(payload, {"island_id", "question", "descriptor_spec", "previous_ref", "reason"})
    _text(payload["island_id"])
    _text(payload.pop("reason"))
    previous = _charter_head(store, direction, records, payload["island_id"], frozen_cases)
    require(payload["previous_ref"] == previous, "charter revision requires the current exact head",
            ResearchActionError)
    ref = archive_record(store, candidate_id, dict(payload, kind="evo_island_charter"),
                         frozen_cases=frozen_cases, artifact="charter.json")
    return {"charter_ref": ref}


def _new_membership(store, direction, candidate_id, member, current, records, frozen_cases):
    """Write one membership record after checking its predecessor and charter; return its ref."""
    _fields(member, {"island_id", "charter_ref", "assignment", "previous_ref"}, {"candidate_id"})
    subject = member.pop("candidate_id", candidate_id)
    subject_candidate = store.get_candidate(subject)
    require(all(subject_candidate[key] == direction[key] for key in ("job_id", "measurement_epoch_id")),
            "membership subject belongs to another job/epoch", ResearchActionError)
    _text(member["island_id"])
    members = [read_record(store, ref, frozen_cases=frozen_cases) for ref in current]
    prior = next((ref for ref, value in zip(current, members) if value["island_id"] == member["island_id"]
                  and value["candidate_id"] == subject), None)
    require(member["previous_ref"] == prior, "membership revision requires its current exact predecessor",
            ResearchActionError)
    head = _charter_head(store, direction, records, member["island_id"], frozen_cases)
    require(head is None or member["charter_ref"] == head, "new membership uses a stale charter",
            ResearchActionError)
    return archive_record(store, subject, dict(member, kind="evo_membership"),
                          frozen_cases=frozen_cases, artifact="membership.json")


def _membership(store, proposal, operation_id, direction, payload, records, frozen_cases):
    candidate_id = proposal["candidate_id"]
    _fields(payload, {"retain_refs", "membership", "reason"})
    _text(payload["reason"])
    old_ref, card = _current_card(store, direction, frozen_cases)
    current = _card_refs(card)
    retained = payload["retain_refs"]
    require(type(retained) is list and all(ref in current for ref in retained),
            "retained membership must be in the current card", ResearchActionError)
    require(len({_json(ref) for ref in retained}) == len(retained), "duplicate retained membership",
            ResearchActionError)
    member_ref = None
    if payload["membership"] is not None:
        member_ref = _new_membership(store, direction, candidate_id, payload["membership"], current, records,
                                     frozen_cases)
    tip_refs, source_refs = [], []
    for ref in retained + ([member_ref] if member_ref else []):
        own = read_record(store, ref, frozen_cases=frozen_cases)["candidate_id"] == candidate_id
        (tip_refs if own else source_refs).append(ref)
    card.update(previous_ref=old_ref, membership_refs=tip_refs, source_membership_refs=source_refs)
    committed = archive_checkpoint(
        store, proposal["direction_id"], candidate_id, card=card,
        note=dict(direction["note"], research_action_id=operation_id), expected_revision=proposal["expected_revision"],
        frozen_cases=frozen_cases, status=direction["status"], resources=direction["resources"],
        evidence=direction["evidence"])
    return {"membership_ref": member_ref, "card_ref": committed["note"]["archive_card_ref"]}


def _concept(store, candidate_id, payload, records):
    _fields(payload, {"concept_id", "origin_island_id", "island_id", "derivation", "scope", "evidence_refs"})
    for key in set(payload) - {"evidence_refs"}:
        _text(payload[key])
    require(payload["concept_id"] not in records.get("concepts", {}),
            "concept ID already exists; select a new revision ID", ResearchActionError)
    require(type(payload["evidence_refs"]) is list, "concept evidence must be a list", ResearchActionError)
    for ref in payload["evidence_refs"]:
        resolve_artifact(store, ref)
    record = dict(payload, schema_version=1, kind="exchange_concept")
    identity = store.archive_evidence(candidate_id, {"concept.json": _json(record)})
    return {"concept_ref": {"evidence_id": identity, "artifact": "concept.json"}}


def _exchange(store, proposal, direction, payload, records, frozen_cases):
    candidate_id = proposal["candidate_id"]
    exchange = validate_exchange(store, payload, frozen_cases=frozen_cases)
    recipient = exchange["recipient"]
    require(recipient["direction_id"] == proposal["direction_id"] and recipient["candidate_id"] == candidate_id
            and recipient["revision"] == proposal["expected_revision"], "exchange action recipient mismatch",
            ResearchActionError)
    _, card = _current_card(store, direction, frozen_cases)
    require(recipient["membership_ref"] in card.get("membership_refs", []), "exchange membership is not current",
            ResearchActionError)
    require(exchange["exchange_id"] not in records.get("exchanges", {}), "exchange ID already exists",
            ResearchActionError)
    identity = store.archive_evidence(candidate_id, {"exchange.json": _json(exchange)})
    return {"exchange_ref": {"evidence_id": identity, "artifact": "exchange.json"}}


def perform_action(store, proposal, operation_id, *, accounting, frozen_cases, records):
    """Apply only the named typed action; there is no general state or record writer."""
    action, payload = proposal["action"], copy.deepcopy(proposal["payload"])
    direction = store.get(proposal["direction_id"])
    candidate_id = proposal["candidate_id"]
    require(direction["tip"] == candidate_id and direction["revision"] == proposal["expected_revision"],
            "research action direction changed", ResearchActionError)
    if action == "accept_anchor":
        return _accept_anchor(store, proposal, payload)
    if action in ("nominate", "revive"):
        require(type(payload) is dict and payload.get("action") == action, "maturation action/payload mismatch",
                ResearchActionError)
        result = checkpoint_maturation(store, proposal["direction_id"], decision=payload,
                                       expected_revision=proposal["expected_revision"], accounting=accounting)
        return {"maturation_ref": result["ref"]}
    if action == "charter":
        return _charter(store, direction, candidate_id, payload, records, frozen_cases)
    if action == "membership":
        return _membership(store, proposal, operation_id, direction, payload, records, frozen_cases)
    if action == "concept":
        return _concept(store, candidate_id, payload, records)
    if action == "exchange":
        return _exchange(store, proposal, direction, payload, records, frozen_cases)
    raise ResearchActionError("unknown research action")


def preview_action(store, proposal, operation_id, **context):
    preview = _Preview(store)
    refs = perform_action(preview, proposal, operation_id, **context)
    return {"result_refs": refs, "checkpoint": preview.committed}


def index_result(records, proposal, refs):
    """Add a new charter, concept or exchange reference to the in-memory index."""
    for action, group, key, ref in (("charter", "charters", "island_id", "charter_ref"),
                                   ("concept", "concepts", "concept_id", "concept_ref"),
                                   ("exchange", "exchanges", "exchange_id", "exchange_ref")):
        if proposal["action"] == action:
            records.setdefault(group, {})[proposal["payload"][key]] = refs[ref]
