"""Archive records: idea cards, island charters, memberships and decisions.

Each record is a small JSON object saved as evidence in the research store and
bound to one candidate's job, measurement epoch and source. A record only
describes what an agent claimed or decided. It never makes a candidate eligible
for measurement, never marks it as an elite, and never proves novelty.

Records point to each other and to raw evidence with ``{evidence_id,
artifact}`` references. Reading a record checks every reference it carries.
Compilation and diagnostic records name test cases, so the caller passes the
registered case list per job as ``frozen_cases`` when those records appear.
"""

import json
from pathlib import PurePosixPath

from .._util import EvoError, canonical_json, is_sha256, require


class ArchiveError(EvoError, ValueError):
    pass


KINDS = {"evo_archive_card", "evo_island_charter", "evo_membership",
         "evo_compilation", "evo_diagnostic", "evo_decision"}
MAX_METADATA_BYTES = 65536

# Required and optional fields of each record kind (besides the common binding
# fields and the optional ``previous_ref``).
_FIELDS = {
    "evo_island_charter": ({"island_id", "question", "descriptor_spec"}, set()),
    "evo_membership": ({"island_id", "charter_ref", "assignment"}, set()),
    "evo_compilation": ({"case_ids", "recipe_id", "outputs"}, {"target", "runtime"}),
    "evo_diagnostic": ({"case_ids", "compilation_ref", "artifacts", "claim", "unknown_reason"},
                       {"supplemental_refs"}),
    "evo_decision": ({"decision_type", "rationale"}, {"recipient", "offers", "details", "evidence_refs"}),
    "evo_archive_card": ({"direction_id", "idea"},
                         {"author", "membership_refs", "observation_refs", "historical_refs",
                          "decision_refs", "selection_ref", "last_faithful", "protection",
                          "corrections", "source_membership_refs"}),
}
_COMMON = {"schema_version", "kind", "candidate_id", "binding"}


def _text(value):
    require(type(value) is str and bool(value.strip()), "expected nonempty text", ArchiveError)


def _json(value):
    try:
        return canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ArchiveError("metadata must be finite JSON") from exc


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON field", ArchiveError)
        result[key] = value
    return result


def _reject_constant(value):
    raise ArchiveError("nonfinite JSON constant")


def _loads(data):
    try:
        return json.loads(data, object_pairs_hook=_pairs, parse_constant=_reject_constant)
    except (ValueError, UnicodeError) as exc:
        raise ArchiveError("invalid JSON metadata") from exc


def _fields(value, required, optional=()):
    require(type(value) is dict and set(required) <= set(value)
            and set(value) <= set(required) | set(optional), "invalid metadata fields", ArchiveError)


def _is_evo_record(value):
    return isinstance(value, dict) and isinstance(value.get("kind"), str) and value["kind"].startswith("evo_")


def _ref(ref):
    """Check an ``{evidence_id, artifact}`` reference; return it as a tuple key."""
    _fields(ref, ("evidence_id", "artifact"))
    require(is_sha256(ref["evidence_id"]), "invalid SHA256", ArchiveError)
    name = ref["artifact"]
    _text(name)
    require(not PurePosixPath(name).is_absolute() and ".." not in name.split("/")
            and PurePosixPath(name).as_posix() == name and name != "."
            and "\\" not in name and "\0" not in name, "unsafe artifact reference", ArchiveError)
    return ref["evidence_id"], name


def _binding(candidate):
    return {"job_id": candidate["job_id"], "measurement_epoch_id": candidate["measurement_epoch_id"],
            "candidate_source_id": candidate["source_id"]}


class _Reader:
    """Reads and checks records from one store, caching what it has read."""

    def __init__(self, store, frozen_cases=None):
        self.store, self.frozen_cases = store, frozen_cases or {}
        self.raw_cache, self.records, self.active = {}, {}, set()

    def raw(self, ref, candidate_id=None, context=None):
        """Return (envelope, bytes) for a reference, optionally checking its owner."""
        key = _ref(ref)
        if key not in self.raw_cache:
            envelope, artifacts = self.store.evidence(key[0])
            require(key[1] in artifacts, "missing referenced artifact", ArchiveError)
            self.raw_cache[key] = envelope, artifacts[key[1]]
        envelope, data = self.raw_cache[key]
        if candidate_id is not None:
            require(envelope["candidate_id"] == candidate_id, "artifact candidate mismatch", ArchiveError)
        if context is not None:
            require(all(envelope["binding"][k] == context[k] for k in ("job_id", "measurement_epoch_id")),
                    "artifact job/epoch mismatch", ArchiveError)
        return envelope, data

    def refs(self, refs, candidate_id=None, context=None):
        require(type(refs) is list, "references must be a list", ArchiveError)
        for ref in refs:
            self.raw(ref, candidate_id, context)

    def typed_if_present(self, ref):
        """Check a reference as a record only if its bytes look like one."""
        _, data = self.raw(ref)
        try:
            value = _loads(data)
        except ArchiveError:
            return  # A raw log or binary artifact need not be JSON metadata.
        if type(value) is dict and type(value.get("kind")) is str and value["kind"].startswith("evo_"):
            self.record(ref)

    def _nested_records(self, value):
        """References inside ``value`` whose targets are themselves records."""
        found, nodes = [], [value]
        while nodes:
            node = nodes.pop()
            if isinstance(node, dict):
                if set(node) == {"evidence_id", "artifact"}:
                    _, raw = self.raw(node)
                    if len(raw) > MAX_METADATA_BYTES:
                        continue
                    try:
                        target = _loads(raw)
                    except ArchiveError:
                        continue
                    if _is_evo_record(target):
                        found.append(node)
                else:
                    nodes.extend(node.values())
            elif isinstance(node, list):
                nodes.extend(node)
        return found

    def record(self, ref, kind=None):
        """Read one record and every record it references, without recursion.

        Referenced records are validated first (depth first), so long revision
        chains do not hit the Python recursion limit. A reference cycle fails.
        """
        key = _ref(ref)
        pending, staging, entered = [(ref, False)], {}, set()
        try:
            while pending:
                current, finish = pending.pop()
                identity = _ref(current)
                if identity in self.records:
                    continue
                if finish:
                    value, candidate_id = staging[identity]
                    self.validate(value, candidate_id)
                    self.records[identity] = value
                    self.active.remove(identity)
                    continue
                require(identity not in self.active, "metadata reference cycle", ArchiveError)
                envelope, data = self.raw(current)
                require(len(data) <= MAX_METADATA_BYTES, "metadata is too large", ArchiveError)
                value = _loads(data)
                staging[identity] = value, envelope["candidate_id"]
                self.active.add(identity)
                entered.add(identity)
                pending.append((current, True))
                pending.extend((node, False) for node in self._nested_records(value))
        finally:
            self.active.difference_update(entered)
        value = self.records[key]
        require(kind is None or value["kind"] == kind, "unexpected metadata kind", ArchiveError)
        return value

    def same_context(self, candidate_id, binding):
        candidate = self.store.get_candidate(candidate_id)
        require(all(candidate[k] == binding[k] for k in ("job_id", "measurement_epoch_id")),
                "candidate job/epoch mismatch", ArchiveError)
        return candidate

    def cases(self, record):
        cases, job = record["case_ids"], record["binding"]["job_id"]
        require(type(cases) is list and cases and all(type(c) is str and c for c in cases)
                and len(set(cases)) == len(cases), "invalid diagnostic cases", ArchiveError)
        require(job in self.frozen_cases, "diagnostics require the host frozen case registry", ArchiveError)
        require(set(cases) <= set(self.frozen_cases[job]), "unregistered diagnostic case", ArchiveError)

    def membership(self, ref, candidate_id=None, island_id=None):
        value = self.record(ref, "evo_membership")
        require(candidate_id is None or value["candidate_id"] == candidate_id,
                "membership candidate mismatch", ArchiveError)
        require(island_id is None or value["island_id"] == island_id, "membership island mismatch", ArchiveError)
        return value

    def selection(self, record):
        """Check a parent/donor selection decision against the store."""
        recipient = record["recipient"]
        _fields(recipient, ("direction_id", "island_id", "membership_ref"))
        _text(recipient["direction_id"])
        self.membership(recipient["membership_ref"], record["candidate_id"], recipient["island_id"])
        require(type(record["offers"]) is list, "selection offers must be a list", ArchiveError)
        seen = set()
        for offer in record["offers"]:
            _fields(offer, ("role", "candidate_id", "source_id", "selected_from_island_id", "membership_ref"))
            require(offer["role"] in ("parent", "donor"), "unknown ancestry role", ArchiveError)
            candidate = self.store.get_candidate(offer["candidate_id"])
            require(candidate["source_id"] == offer["source_id"], "selected source mismatch", ArchiveError)
            self.membership(offer["membership_ref"], offer["candidate_id"], offer["selected_from_island_id"])
            key = offer["role"], offer["candidate_id"]
            require(key not in seen, "duplicate selected operand", ArchiveError)
            seen.add(key)

    def validate(self, value, candidate_id):
        """Check one record's fields and binding against its owning candidate."""
        require(type(value) is dict and _COMMON <= set(value), "missing metadata binding", ArchiveError)
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "unsupported metadata version", ArchiveError)
        kind = value["kind"]
        require(type(kind) is str and kind in KINDS, "unsupported metadata kind", ArchiveError)
        candidate = self.store.get_candidate(candidate_id)
        binding = _binding(candidate)
        require(value["candidate_id"] == candidate_id and value["binding"] == binding,
                "metadata source/job/epoch binding mismatch", ArchiveError)
        required, optional = _FIELDS[kind]
        _fields(value, _COMMON | required, optional | {"previous_ref"})
        if value.get("previous_ref") is not None:
            prior = self.record(value["previous_ref"], kind)
            self.raw(value["previous_ref"], context=binding)
            identity = "direction_id" if kind == "evo_archive_card" else "island_id"
            if identity in value:
                require(prior[identity] == value[identity], "revision changed identity", ArchiveError)
        if kind == "evo_island_charter":
            _text(value["island_id"])
            _text(value["question"])
            require(type(value["descriptor_spec"]) is dict, "descriptor specification must be an object",
                    ArchiveError)
        elif kind == "evo_membership":
            self._check_membership(value, candidate_id, binding)
        elif kind in ("evo_compilation", "evo_diagnostic"):
            self._check_build_record(value, kind, candidate_id)
        elif kind == "evo_decision":
            _text(value["decision_type"])
            _text(value["rationale"])
            self.refs(value.get("evidence_refs", []))
            if value["decision_type"] == "selection":
                require("recipient" in value and "offers" in value, "selection requires provenance", ArchiveError)
                self.selection(value)
        else:
            self.card(value, candidate)

    def _check_membership(self, value, candidate_id, binding):
        charter = self.record(value["charter_ref"], "evo_island_charter")
        self.raw(value["charter_ref"], context=binding)
        require(charter["island_id"] == value["island_id"], "charter island mismatch", ArchiveError)
        assignment = value["assignment"]
        _fields(assignment, ("status", "descriptors", "evidence_refs", "reason"))
        require(assignment["status"] in ("unknown", "tentative", "supported"), "invalid assignment status",
                ArchiveError)
        require(type(assignment["descriptors"]) is dict, "descriptors must be an object", ArchiveError)
        _text(assignment["reason"])
        self.refs(assignment["evidence_refs"], candidate_id)
        require(assignment["status"] != "supported" or bool(assignment["evidence_refs"]),
                "supported assignment requires source-bound evidence", ArchiveError)

    def _check_build_record(self, value, kind, candidate_id):
        """Compilation and diagnostic records: registered cases and retained outputs."""
        self.cases(value)
        refs = value["outputs"] if kind == "evo_compilation" else value["artifacts"]
        require(type(refs) is dict and all(type(k) is str and k for k in refs), "invalid diagnostic artifacts",
                ArchiveError)
        self.refs(list(refs.values()), candidate_id)
        if kind == "evo_compilation":
            require(is_sha256(value["recipe_id"]), "invalid SHA256", ArchiveError)
            require(bool(refs), "compilation requires retained outputs", ArchiveError)
            require(len(value["case_ids"]) == 1, "v1 compilation requires one case/specialization", ArchiveError)
            return
        _text(value["claim"])
        self.refs(value.get("supplemental_refs", []), candidate_id)
        if value["compilation_ref"] is None:
            _text(value["unknown_reason"])
            return
        compilation = self.record(value["compilation_ref"], "evo_compilation")
        self.raw(value["compilation_ref"], candidate_id)
        require(set(value["case_ids"]) <= set(compilation["case_ids"]), "compilation case mismatch", ArchiveError)
        require(all(ref in compilation["outputs"].values() for ref in refs.values()),
                "diagnostic output not retained by this compilation", ArchiveError)

    def card(self, value, candidate):
        """Check an idea card: its evidence, memberships, selection and corrections."""
        _text(value["direction_id"])
        idea = value["idea"]
        _fields(idea, ("idea_id", "statement", "status"), ("next_test", "applicability", "competing_explanations"))
        for field in ("idea_id", "statement", "status"):
            _text(idea[field])
        if value.get("author") is not None:
            _fields(value["author"], ("engineer_id", "attempt_ref"))
            _text(value["author"]["engineer_id"])
            if value["author"]["attempt_ref"] is not None:
                self.raw(value["author"]["attempt_ref"], context=value["binding"])
        self.refs(value.get("observation_refs", []), value["candidate_id"])
        for ref in value.get("observation_refs", []):
            self.typed_if_present(ref)
        self.refs(value.get("historical_refs", []))
        self.refs(value.get("decision_refs", []))
        for ref in value.get("decision_refs", []):
            self.typed_if_present(ref)
        self._card_memberships(value)
        if value.get("selection_ref") is not None:
            self._card_selection(value, candidate)
        faithful = value.get("last_faithful")
        if faithful is not None:
            _fields(faithful, ("candidate_id", "claim", "evidence_refs"))
            self.same_context(faithful["candidate_id"], value["binding"])
            _text(faithful["claim"])
            require(bool(faithful["evidence_refs"]), "faithfulness requires source evidence", ArchiveError)
            self.refs(faithful["evidence_refs"], faithful["candidate_id"])
        if value.get("protection") is not None:
            self._card_protection(value, faithful)
        corrections = value.get("corrections", [])
        require(type(corrections) is list, "corrections must be a list", ArchiveError)
        for correction in corrections:
            self._card_correction(value, correction)

    def _card_memberships(self, value):
        memberships = value.get("membership_refs", [])
        self.refs(memberships, value["candidate_id"])
        islands = set()
        for ref in memberships:
            member = self.membership(ref, value["candidate_id"])
            require(member["island_id"] not in islands, "duplicate current island membership", ArchiveError)
            islands.add(member["island_id"])
        source_memberships = value.get("source_membership_refs", [])
        self.refs(source_memberships, context=value["binding"])
        seen_sources = set()
        for ref in source_memberships:
            member = self.membership(ref)
            self.same_context(member["candidate_id"], value["binding"])
            key = member["candidate_id"], member["island_id"]
            require(member["candidate_id"] != value["candidate_id"] and key not in seen_sources,
                    "source membership duplicates a current or retained source/island", ArchiveError)
            seen_sources.add(key)

    def _card_selection(self, value, candidate):
        selection = self.record(value["selection_ref"], "evo_decision")
        self.raw(value["selection_ref"], context=value["binding"])
        require(selection["decision_type"] == "selection", "not a selection decision", ArchiveError)
        require(selection["recipient"]["direction_id"] == value["direction_id"], "recipient direction mismatch",
                ArchiveError)
        for offer in selection["offers"]:
            require(offer["candidate_id"] in candidate[offer["role"] + "s"], "selected operand absent from ancestry",
                    ArchiveError)
        require({(offer["role"], offer["candidate_id"]) for offer in selection["offers"]} ==
                {(role, identity) for role in ("parent", "donor") for identity in candidate[role + "s"]},
                "selection does not cover the child's direct ancestry", ArchiveError)

    def _card_protection(self, value, faithful):
        protection = value["protection"]
        _fields(protection, ("candidate_id", "opportunity_ref", "rationale"))
        require(faithful is not None and faithful["candidate_id"] == protection["candidate_id"],
                "protection requires the exact faithful candidate", ArchiveError)
        _text(protection["rationale"])
        _, data = self.raw(protection["opportunity_ref"], context=value["binding"])
        opportunity = _loads(data)
        require(type(opportunity) is dict, "opportunity must be a JSON object", ArchiveError)
        for key in ("job_id", "measurement_epoch_id", "direction_id"):
            expected = value.get(key, value["binding"].get(key))
            require(key not in opportunity or opportunity[key] == expected, "opportunity scope mismatch",
                    ArchiveError)

    def _card_correction(self, value, correction):
        _fields(correction, ("card_ref", "field", "reason", "evidence_refs"))
        prior = self.record(correction["card_ref"], "evo_archive_card")
        require(prior["direction_id"] == value["direction_id"], "correction direction mismatch", ArchiveError)
        self.raw(correction["card_ref"], context=value["binding"])
        _text(correction["field"])
        _text(correction["reason"])
        target = prior
        for field in correction["field"].split("."):
            require(type(target) is dict and field in target, "correction field does not exist", ArchiveError)
            target = target[field]
        self.refs(correction["evidence_refs"])


def archive_record(store, candidate_id, record, *, frozen_cases=None, artifact="record.json"):
    """Check one record, save it as evidence, and return its reference."""
    value = _loads(_json(record))
    require(type(value) is dict, "metadata must be an object", ArchiveError)
    value.setdefault("schema_version", 1)
    value.setdefault("candidate_id", candidate_id)
    value.setdefault("binding", _binding(store.get_candidate(candidate_id)))
    _Reader(store, frozen_cases).validate(value, candidate_id)
    data = _json(value)
    require(len(data) <= MAX_METADATA_BYTES, "metadata is too large", ArchiveError)
    _ref({"evidence_id": "0" * 64, "artifact": artifact})
    return {"evidence_id": store.archive_evidence(candidate_id, {artifact: data}), "artifact": artifact}


def read_record(store, ref, *, frozen_cases=None):
    """Read and check one saved record and the records it references."""
    return _Reader(store, frozen_cases).record(ref)


def _carry_memberships(reader, value, prior, candidate_id):
    """Carry the prior card's source memberships into a card that omits them.

    Cards that use ``source_membership_refs`` keep earlier memberships when the
    tip moves; older cards without that field keep their omission semantics. A
    membership of the new tip candidate moves back to ``membership_refs``.
    """
    carried = list(prior.get("source_membership_refs", []))
    if "source_membership_refs" in prior:
        carried += prior.get("membership_refs", [])
    tip_refs, other_refs = [], []
    for ref in carried:
        target = tip_refs if reader.membership(ref)["candidate_id"] == candidate_id else other_refs
        if ref not in target:
            target.append(ref)
    value["source_membership_refs"] = other_refs
    value["membership_refs"] = list(value.get("membership_refs", []))
    value["membership_refs"].extend(ref for ref in tip_refs if ref not in value["membership_refs"])


def _check_faithful_withdrawal(store, reader, value, previous_ref, candidate_id):
    """Clearing ``last_faithful`` needs an evidence-backed correction of the current card."""
    corrections = value.get("corrections", [])
    require(type(corrections) is list, "corrections must be a list", ArchiveError)
    matches = [item for item in corrections if type(item) is dict
               and item.get("card_ref") == previous_ref and item.get("field") == "last_faithful"]
    require(bool(matches), "withdrawing last faithful requires a correction to the current card", ArchiveError)
    for correction in matches:
        _text(correction.get("reason"))
        require(bool(correction.get("evidence_refs")), "faithfulness correction requires evidence", ArchiveError)
        reader.refs(correction["evidence_refs"], context=_binding(store.get_candidate(candidate_id)))


def archive_checkpoint(store, direction_id, candidate_id, *, card, note, expected_revision,
                       frozen_cases=None, status="active", resources=None, evidence=()):
    """Save a new idea card and checkpoint the direction to ``candidate_id``.

    Leaving ``last_faithful`` out of the new card keeps the previous one. Setting
    it to None withdraws it, which needs an explicit correction with evidence.
    """
    value = _loads(_json(card))
    value.setdefault("kind", "evo_archive_card")
    value.setdefault("direction_id", direction_id)
    require(value["kind"] == "evo_archive_card" and value["direction_id"] == direction_id,
            "checkpoint card direction mismatch", ArchiveError)
    previous = store.get().get(direction_id)
    previous_ref = previous["note"].get("archive_card_ref") if previous else None
    if previous_ref is not None:
        reader = _Reader(store, frozen_cases)
        prior = reader.record(previous_ref, "evo_archive_card")
        if "source_membership_refs" not in value:
            _carry_memberships(reader, value, prior, candidate_id)
        if "last_faithful" not in value and "last_faithful" in prior:
            value["last_faithful"] = prior["last_faithful"]
        elif value.get("last_faithful") is None and prior.get("last_faithful") is not None:
            _check_faithful_withdrawal(store, reader, value, previous_ref, candidate_id)
    ref = archive_record(store, candidate_id, value, frozen_cases=frozen_cases, artifact="archive_card.json")
    checkpoint_note = dict(note, archive_card_ref=ref)
    return store.checkpoint(direction_id, candidate_id, note=checkpoint_note, expected_revision=expected_revision,
                            status=status, resources=resources, evidence=list(evidence) + [ref["evidence_id"]])


def read_archive(store, *, frozen_cases=None):
    """Current view built only from committed directions and their cards."""
    reader, directions, islands = _Reader(store, frozen_cases), {}, {}
    for name, state in store.get().items():
        candidate = store.get_candidate(state["tip"])
        row = {key: state[key] for key in ("direction_id", "revision", "tip", "anchor", "resources", "status")}
        row.update(source_id=candidate["source_id"], card=None, card_ref=state["note"].get("archive_card_ref"),
                   opportunity=None, opportunity_verification="not_verified_by_archive")
        if row["card_ref"] is not None:
            card = reader.record(row["card_ref"], "evo_archive_card")
            require(card["direction_id"] == name and card["candidate_id"] == state["tip"],
                    "stale/misbound current card", ArchiveError)
            require(row["card_ref"]["evidence_id"] in state["evidence"], "card is not checkpoint evidence",
                    ArchiveError)
            row["card"] = card
            if card.get("protection") is not None:
                _, data = reader.raw(card["protection"]["opportunity_ref"])
                row["opportunity"] = _loads(data)
            for ref in card.get("membership_refs", []) + card.get("source_membership_refs", []):
                membership = reader.membership(ref)
                islands.setdefault(membership["island_id"], []).append({
                    "direction_id": name, "membership_ref": ref,
                    "candidate_id": membership["candidate_id"],
                    "source_id": store.get_candidate(membership["candidate_id"])["source_id"],
                    "assignment": membership["assignment"], "validated_elite": None,
                    "charter": reader.record(membership["charter_ref"], "evo_island_charter")})
        directions[name] = row
    return {"schema_version": 1, "directions": directions, "islands": islands,
            "distinct_tip_sources": len({row["source_id"] for row in directions.values()})}


def format_archive(view):
    """Short ASCII summary of ``read_archive``; infers no quality or novelty."""
    lines = ["ARCHIVE %d directions / %d distinct tip sources / %d islands" %
             (len(view["directions"]), view["distinct_tip_sources"], len(view["islands"]))]
    for island, members in sorted(view["islands"].items()):
        lines.append("  ISLAND %s (agent assignments; no inferred elites)" % island)
        for member in members:
            lines.append("    %s: %s; %s" % (member["direction_id"], member["assignment"]["status"],
                                               member["charter"]["question"]))
    for name, row in sorted(view["directions"].items()):
        card = row["card"]
        lines += ["  %s r%d [%s]" % (name, row["revision"], row["status"]),
                  "    anchor %s" % (row["anchor"]["candidate_id"][:12] if row["anchor"] else "unknown"),
                  "    tip    %s / source %s" % (row["tip"][:12], row["source_id"][:12])]
        if card is None:
            lines.append("    research metadata unknown")
            continue
        lines.append("    idea %s: %s" % (card["idea"]["idea_id"], card["idea"]["statement"]))
        lines.append("    engineer %s; %d direct evidence refs" %
                     ((card.get("author") or {}).get("engineer_id", "unknown"), len(card.get("observation_refs", []))))
        lines.append("    novelty/roofline/elite: not established by archive metadata")
        if card.get("protection"):
            lines.append("    protected candidate %s (faithfulness claim)" % card["protection"]["candidate_id"][:12])
            opportunity = row["opportunity"]
            if isinstance(opportunity, dict):
                for key in ("requested", "reserved", "delivered", "actual", "settled"):
                    lines.append("    reported %s: %s" % (key, opportunity.get(key, "unknown")))
            lines.append("    delivery/actual usage not verified by archive")
    return "\n".join(lines).encode("ascii", "backslashreplace").decode("ascii")
