"""Crossover requests between directions, and the engineer's answer to them.

An exchange asks one recipient direction to consider offers from other
candidates (exact source) or from written concepts. This module only checks
that every reference points at saved, matching records and that the answer has
the right shape. It never selects donors, runs code, or promotes a candidate.
Compatibility statuses and realization text are the agent's claims, not
verified facts.
"""

import copy
import json
from pathlib import PurePosixPath

from .._util import EvoError, is_sha256, require
from .records import ArchiveError, read_record
from .refs import file_ref, source_reference


class ExchangeError(EvoError, ValueError):
    pass


def _need(condition, message):
    require(condition, message, ExchangeError)


def _fields(value, names):
    _need(type(value) is dict and set(value) == set(names.split()), "invalid fields")


def _text(value):
    _need(type(value) is str and bool(value.strip()), "nonempty text required")


def _sha(value):
    _need(is_sha256(value), "invalid digest")


def resolve_artifact(store, ref):
    """Check an {evidence_id, artifact} reference and return the saved bytes."""
    _fields(ref, "evidence_id artifact")
    _sha(ref["evidence_id"])
    name = ref["artifact"]
    _text(name)
    path = PurePosixPath(name)
    _need(not path.is_absolute() and str(path) == name
          and all(part not in (".", "..") for part in path.parts), "unsafe artifact name")
    _, artifacts = store.evidence(ref["evidence_id"])
    _need(name in artifacts, "missing artifact")
    return artifacts[name]


def _document(store, ref):
    value = json.loads(resolve_artifact(store, ref))
    _need(type(value) is dict and type(value.get("schema_version")) is int
          and value["schema_version"] == 1, "invalid document version")
    return value


def _refs(store, refs):
    _need(type(refs) is list, "artifact references must be a list")
    for ref in refs:
        resolve_artifact(store, ref)


def _candidate(store, candidate_id, source_id):
    _sha(candidate_id)
    _sha(source_id)
    candidate = store.get_candidate(candidate_id)
    _need(candidate["source_id"] == source_id, "candidate/source mismatch")
    return candidate


def _membership(store, ref, candidate_id, source_id, island_id, *, frozen_cases=None):
    """Check one saved island-membership snapshot (never the latest membership)."""
    _text(island_id)
    candidate = _candidate(store, candidate_id, source_id)
    try:
        value = read_record(store, ref, frozen_cases=frozen_cases)
    except ArchiveError as exc:
        raise ExchangeError("invalid membership snapshot: %s" % exc) from exc
    binding = {"job_id": candidate["job_id"],
               "measurement_epoch_id": candidate["measurement_epoch_id"],
               "candidate_source_id": source_id}
    envelope, _ = store.evidence(ref["evidence_id"])
    # The archive check covers structure and provenance, not whether the
    # descriptors are true.
    _need(value.get("kind") == "evo_membership" and value.get("candidate_id") == candidate_id
          and value.get("binding") == binding and value.get("island_id") == island_id
          and envelope["candidate_id"] == candidate_id,
          "membership snapshot mismatch")


def _validate_offer(store, offer, seen, frozen_cases):
    _need(type(offer) is dict, "invalid offer")
    common = "offer_id kind origin_island_id selected_island_id claim scope obligations carrier_refs "
    kind = offer.get("kind")
    _need(kind in ("source", "concept"), "explicit source/concept kind required")
    _fields(offer, common + ("candidate_id source_id origin_ref membership_ref" if kind == "source"
                             else "concept_ref"))
    for key in ("offer_id", "origin_island_id", "selected_island_id", "claim", "scope"):
        _text(offer[key])
    _need(offer["offer_id"] not in seen, "duplicate offer ID")
    seen.add(offer["offer_id"])
    if kind == "source":
        _membership(store, offer["origin_ref"], offer["candidate_id"],
                    offer["source_id"], offer["origin_island_id"], frozen_cases=frozen_cases)
        _membership(store, offer["membership_ref"], offer["candidate_id"],
                    offer["source_id"], offer["selected_island_id"], frozen_cases=frozen_cases)
    else:
        concept = _document(store, offer["concept_ref"])
        _need(concept.get("kind") == "exchange_concept"
              and concept.get("origin_island_id") == offer["origin_island_id"]
              and concept.get("island_id") == offer["selected_island_id"], "concept snapshot mismatch")
        _text(concept.get("concept_id"))
    _refs(store, offer["carrier_refs"])
    _need(type(offer["obligations"]) is list, "obligations must be a list")
    for obligation in offer["obligations"]:
        _fields(obligation, "question status evidence_refs")
        _text(obligation["question"])
        _need(obligation["status"] in ("unknown", "supported", "conflict"), "invalid obligation status")
        _refs(store, obligation["evidence_refs"])


def validate_exchange(store, request, *, require_current=True, frozen_cases=None):
    """Check an exchange request's references and claim shapes; return a copy.

    Use require_current=True before giving the exchange to an engineer: the
    recipient must still be the direction's current tip and revision. Audits of
    old answers use False; that never makes an old request valid for new work.
    frozen_cases is the host's case registry, needed when memberships cite
    diagnostics.
    """
    request = copy.deepcopy(request)
    _fields(request, "schema_version exchange_id job_id measurement_epoch_id recipient offers question")
    _need(type(request["schema_version"]) is int and request["schema_version"] == 1,
          "invalid exchange version")
    for key in ("exchange_id", "job_id", "measurement_epoch_id", "question"):
        _text(request[key])
    recipient = request["recipient"]
    _fields(recipient, "direction_id revision candidate_id source_id island_id membership_ref")
    _text(recipient["direction_id"])
    _need(type(recipient["revision"]) is int and recipient["revision"] > 0, "invalid revision")
    candidate = _candidate(store, recipient["candidate_id"], recipient["source_id"])
    for key in ("job_id", "measurement_epoch_id"):
        _need(candidate[key] == request[key], "recipient job/epoch mismatch")
    _membership(store, recipient["membership_ref"], recipient["candidate_id"],
                recipient["source_id"], recipient["island_id"], frozen_cases=frozen_cases)
    if require_current:
        state = store.get(recipient["direction_id"])
        _need(state["revision"] == recipient["revision"]
              and state["tip"] == recipient["candidate_id"], "stale recipient state")
    _need(type(request["offers"]) is list and request["offers"], "offers required")
    seen = set()
    for offer in request["offers"]:
        _validate_offer(store, offer, seen, frozen_cases)
    return request


def exchange_scope(request, offer):
    """"intra" if the offer was selected from the recipient's island, else "inter".

    Uses the island recorded at selection time, not where the offer was born.
    """
    return "intra" if offer["selected_island_id"] == request["recipient"]["island_id"] else "inter"


def _load_exchange(store, ref, require_current, *, frozen_cases=None):
    value = validate_exchange(store, _document(store, ref), require_current=require_current,
                              frozen_cases=frozen_cases)
    envelope, _ = store.evidence(ref["evidence_id"])
    _need(envelope["candidate_id"] == value["recipient"]["candidate_id"], "exchange evidence binding mismatch")
    return value


def _exchange_refs(exchange_ref, exchange):
    """Every saved reference an exchange names, in a stable order."""
    refs = [exchange_ref, exchange["recipient"]["membership_ref"]]
    for offer in exchange["offers"]:
        refs.extend(offer["carrier_refs"])
        refs.extend([offer["origin_ref"], offer["membership_ref"]] if offer["kind"] == "source"
                    else [offer["concept_ref"]])
        for obligation in offer["obligations"]:
            refs.extend(obligation["evidence_refs"])
    return refs


def prepare_engineer_request(store, exchange_ref, request, *, include_sources=True, frozen_cases=None):
    """Add an exchange's donors and read-only files to an existing engineer request.

    The exchange must be saved (exchange_ref) and still current for the
    recipient. include_sources=False leaves out donor source references only;
    it is NOT an information barrier, so host access control is still needed.
    """
    _need(type(include_sources) is bool, "include_sources must be boolean")
    exchange = _load_exchange(store, exchange_ref, True, frozen_cases=frozen_cases)
    request = copy.deepcopy(request)
    recipient = exchange["recipient"]
    _need(request["direction_id"] == recipient["direction_id"]
          and request["tip"]["candidate_id"] == recipient["candidate_id"]
          and request["tip"]["source_id"] == recipient["source_id"]
          and request["job"]["job_id"] == exchange["job_id"]
          and request["measurement_epoch_id"] == exchange["measurement_epoch_id"],
          "engineer/exchange binding mismatch")
    if include_sources:
        for offer in exchange["offers"]:
            if offer["kind"] == "source":
                donor = source_reference(store, offer["candidate_id"])
                if donor not in request["donors"]:
                    request["donors"].append(donor)
    for ref in _exchange_refs(exchange_ref, exchange):
        resolve_artifact(store, ref)
        envelope, _ = store.evidence(ref["evidence_id"])
        artifact = file_ref((store.root / "objects" / envelope["artifacts"][ref["artifact"]]).resolve())
        if artifact not in request["read_only"]:
            request["read_only"].append(artifact)
    request["next_question"] = exchange["question"]
    from ..engineer.request import validate_request  # Engineer module; imported only when needed.
    return validate_request(request)


def _check_dispositions(receipt, offers):
    _need(type(receipt["dispositions"]) is list, "dispositions must be a list")
    decisions = {}
    for item in receipt["dispositions"]:
        _fields(item, "offer_id decision reason")
        _text(item["offer_id"])
        _text(item["reason"])
        _need(item["offer_id"] in offers and item["offer_id"] not in decisions, "invalid disposition ID")
        _need(item["decision"] in ("adopt", "adapt", "reject", "defer", "redundant"), "invalid disposition")
        decisions[item["offer_id"]] = item["decision"]
    _need(set(decisions) == set(offers), "every offer needs a disposition")
    return decisions


def _check_child(store, exchange, child_candidate_id):
    if child_candidate_id is None:
        return None
    _sha(child_candidate_id)
    child = store.get_candidate(child_candidate_id)
    _need(all(child[key] == exchange[key] for key in ("job_id", "measurement_epoch_id")),
          "child job/epoch mismatch")
    _need(exchange["recipient"]["candidate_id"] in child["parents"], "child lacks recipient parent")
    return child


def validate_receipt(store, exchange_ref, receipt, *, child_candidate_id=None, frozen_cases=None):
    """Check the shape of an engineer's answer to an exchange; return a copy.

    The answer may not carry host facts (host_refs, success flags, timings or
    causal verdicts). A source-level realization needs a child candidate that
    the host captured itself (child_candidate_id) and that cites the donor.
    Store the returned claims apart from host evaluation records.
    """
    exchange = _load_exchange(store, exchange_ref, False, frozen_cases=frozen_cases)
    receipt = copy.deepcopy(receipt)
    _fields(receipt, "schema_version exchange_id dispositions realizations bridge_repairs next_test")
    _need(type(receipt["schema_version"]) is int and receipt["schema_version"] == 1
          and receipt["exchange_id"] == exchange["exchange_id"], "receipt exchange mismatch")
    _text(receipt["next_test"])
    offers = {offer["offer_id"]: offer for offer in exchange["offers"]}
    decisions = _check_dispositions(receipt, offers)
    child = _check_child(store, exchange, child_candidate_id)
    _need(type(receipt["realizations"]) is list, "realizations must be a list")
    for item in receipt["realizations"]:
        _fields(item, "offer_id level claim scope semantic_witness evidence_refs")
        _text(item["offer_id"])
        _need(item["offer_id"] in offers and decisions[item["offer_id"]] in ("adopt", "adapt"),
              "realization requires an adopted offer")
        _need(item["level"] in ("concept", "source"), "invalid realization level")
        for key in ("claim", "scope", "semantic_witness"):
            _text(item[key])
        _refs(store, item["evidence_refs"])
        if item["level"] == "source":
            _need(child is not None, "source claim needs host-captured child")
            offer = offers[item["offer_id"]]
            if offer["kind"] == "source":
                _need(offer["candidate_id"] in child["parents"] + child["donors"], "uncited source donor")
    _need(type(receipt["bridge_repairs"]) is list, "bridge repairs must be a list")
    for item in receipt["bridge_repairs"]:
        _fields(item, "offer_ids status claim scope evidence_refs")
        _need(type(item["offer_ids"]) is list and item["offer_ids"]
              and all(identity in offers for identity in item["offer_ids"]), "invalid bridge offers")
        _need(item["status"] in ("proposed", "unfinished", "failed", "realized", "abandoned"), "invalid bridge status")
        _text(item["claim"])
        _text(item["scope"])
        _refs(store, item["evidence_refs"])
    return receipt
