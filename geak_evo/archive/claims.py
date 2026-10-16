"""Numeric claims with their evidence, and append-only corrections.

A claim states a quantity (a value with unit, denominator, population,
aggregation and measurement kind) or a scientific statement, and is bound to one
job, case, source, measurement epoch, hardware and scope. ``ClaimLog`` saves
claims and corrections as a linked chain of evidence records in the research
store; the caller saves the returned head reference with its own
revision-checked state update.

Checks here are mechanical: evidence hashes, extraction of the quoted number,
and exact decimal arithmetic for derived quantities. A quantity counts as a
fact only when an optional host ``verify_quantity`` callback authorizes the
evidence that produced it. Nothing here judges whether an explanation is true.
"""

from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import json
from pathlib import PurePosixPath
import re

from .._util import EvoError, canonical_json, is_sha256, require, sha256_hex


class ClaimError(EvoError, ValueError):
    pass


BINDING_KEYS = {"job_id", "case_id", "source_id", "work_id", "measurement_epoch_id", "hardware_id", "scope"}
QUANTITY_KEYS = {"value", "unit", "denominator", "population", "aggregation", "measurement_kind"}
COMMON_KEYS = {"kind", "statement", "binding", "evidence", "depends_on", "assumptions", "applicability", "next_test"}
REPEAT_KEYS = {"job_id", "case_id", "source_id", "inputs_id", "configuration_id", "measurement_epoch_id", "diagnostic_id"}
_SCALES = {"s": ("time", "1"), "ms": ("time", "0.001"), "us": ("time", "0.000001"),
           "ns": ("time", "0.000000001"), "bytes": ("bytes", "1"),
           "KiB": ("bytes", "1024"), "MiB": ("bytes", "1048576"), "GiB": ("bytes", "1073741824")}
# Fields of each claim kind beyond COMMON_KEYS.
_KIND_FIELDS = {"reported_quantity": {"quantity", "extraction"}, "derived_quantity": {"quantity", "derivation"},
                "scientific_claim": set()}
# Envelope binding field -> claim binding field.
_ENVELOPE_KEYS = (("job_id", "job_id"), ("measurement_epoch_id", "measurement_epoch_id"),
                  ("candidate_source_id", "source_id"))


def _text(value):
    require(type(value) is str and bool(value.strip()), "expected nonempty text", ClaimError)


_json = canonical_json


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key", ClaimError)
        result[key] = value
    return result


def _reject_constant(value):
    raise ClaimError("non-finite JSON number")


def _loads(data):
    try:
        return json.loads(data, object_pairs_hook=_pairs, parse_float=Decimal, parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ClaimError("invalid JSON evidence") from exc


def _number(value):
    """Parse decimal text or an int (never a float or Boolean) into a bounded Decimal."""
    require(type(value) in (str, int, Decimal), "quantity needs decimal text or a JSON number, never Boolean/float",
            ClaimError)
    require(len(str(value)) <= 128, "numeric literal too long", ClaimError)
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ClaimError("invalid decimal number") from exc
    require(number.is_finite() and abs(number.adjusted()) <= 1000
            and len(number.as_tuple().digits) <= 80, "non-finite or excessive decimal", ClaimError)
    return number


def _binding(value):
    require(type(value) is dict and set(value) == BINDING_KEYS, "incomplete claim binding", ClaimError)
    for item in value.values():
        _text(item)
    require(is_sha256(value["source_id"]), "invalid source identity", ClaimError)


def _ref(value):
    """Check an ``{evidence_id, artifact}`` reference; return its canonical JSON text as a key."""
    require(type(value) is dict and set(value) == {"evidence_id", "artifact"}, "invalid evidence reference",
            ClaimError)
    require(is_sha256(value["evidence_id"]), "invalid evidence ID", ClaimError)
    name = value["artifact"]
    _text(name)
    path = PurePosixPath(name)
    require(not path.is_absolute() and str(path) == name and name != "."
            and ".." not in path.parts and "\\" not in name and "\0" not in name, "unsafe artifact name", ClaimError)
    return _json(value).decode()


def resolve_artifact(ref, resolve_evidence):
    """Return (envelope, bytes) for a reference after checking both hashes."""
    _ref(ref)
    envelope, artifacts = resolve_evidence(ref["evidence_id"])
    require(sha256_hex(_json(envelope)) == ref["evidence_id"], "evidence envelope hash mismatch", ClaimError)
    require(ref["artifact"] in artifacts, "missing evidence artifact", ClaimError)
    data = artifacts[ref["artifact"]]
    require(type(data) is bytes and sha256_hex(data) == envelope["artifacts"][ref["artifact"]],
            "artifact hash mismatch", ClaimError)
    return envelope, data


def json_pointer(value, pointer):
    """Select a field by RFC 6901 JSON Pointer; strict array indices."""
    require(type(pointer) is str and (pointer == "" or pointer.startswith("/")), "invalid JSON Pointer", ClaimError)
    for token in pointer.split("/")[1:] if pointer else []:
        require(re.search(r"~(?![01])", token) is None, "invalid JSON Pointer escape", ClaimError)
        token = token.replace("~1", "/").replace("~0", "~")
        if type(value) is list:
            require(re.fullmatch(r"0|[1-9][0-9]*", token) is not None and int(token) < len(value),
                    "missing array element", ClaimError)
            value = value[int(token)]
        else:
            require(type(value) is dict and token in value, "missing JSON Pointer field", ClaimError)
            value = value[token]
    return value


def _quantity(value):
    require(type(value) is dict and set(value) == QUANTITY_KEYS, "incomplete quantity or denominator", ClaimError)
    _number(value["value"])
    for key in QUANTITY_KEYS - {"value"}:
        _text(value[key])


def _same_quantity(actual, proposed):
    _quantity(actual)
    _quantity(proposed)
    require(all(actual[key] == proposed[key] for key in QUANTITY_KEYS - {"value"}),
            "quantity unit/denominator/scope metadata mismatch", ClaimError)
    require(_number(actual["value"]) == _number(proposed["value"]), "quantity value mismatch", ClaimError)


def _arithmetic(operation, values, count, base, target_unit, weight_quantities):
    """Exact decimal result of one supported operation; updates ``base`` units."""
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_HALF_EVEN
        if operation == "sum":
            return sum(values, Decimal(0))
        if operation == "difference":
            require(count == 2, "difference needs two operands", ClaimError)
            return values[0] - values[1]
        if operation == "ratio":
            require(count == 2 and values[1] != 0, "ratio needs a nonzero denominator", ClaimError)
            base.update(unit="1", denominator="second_operand")
            return values[0] / values[1]
        if operation == "convert":
            require(count == 1, "conversion needs one operand", ClaimError)
            old, new = _SCALES.get(base["unit"]), _SCALES.get(target_unit)
            require(old is not None and new is not None and old[0] == new[0], "unsupported unit conversion",
                    ClaimError)
            base["unit"] = target_unit
            return values[0] * Decimal(old[1]) / Decimal(new[1])
        if operation == "weighted_mean":
            weights = values[count:]
            require(len(weights) == count and all(value >= 0 for value in weights) and sum(weights) > 0,
                    "weighted mean requires nonnegative weights with positive total", ClaimError)
            require(all(quantity["unit"] in ("1", "count") for quantity in weight_quantities),
                    "weights must be dimensionless counts", ClaimError)
            return sum(value * weight for value, weight in zip(values[:count], weights)) / sum(weights)
        raise ClaimError("unsupported arithmetic operation")


def _calculate(record, resolve_claim):
    """Recompute a derived quantity from its operand claims; True if all are authorized and active."""
    derivation = record["derivation"]
    require(type(derivation) is dict and set(derivation) == {"operation", "operands", "weights"},
            "invalid derivation", ClaimError)
    require(type(derivation["operands"]) is list and derivation["operands"], "missing derivation operands",
            ClaimError)
    require(type(derivation["weights"]) is list, "invalid weights", ClaimError)
    entries = [resolve_claim(ref) for ref in derivation["operands"] + derivation["weights"]]
    count = len(derivation["operands"])
    for entry in entries:
        require(entry["record"]["kind"] != "scientific_claim", "prose is not a numeric operand", ClaimError)
        bound = entry["record"]["binding"]
        for key in BINDING_KEYS - {"source_id", "work_id"}:
            require(bound[key] == record["binding"][key], "operand binding/scope mismatch", ClaimError)
    require(any(all(entry["record"]["binding"][key] == record["binding"][key] for key in ("source_id", "work_id"))
                for entry in entries[:count]), "output source/work is absent from operands", ClaimError)
    quantities = [entry["record"]["quantity"] for entry in entries]
    base = dict(quantities[0])
    for quantity in quantities[1:count]:
        require(all(quantity[key] == base[key] for key in QUANTITY_KEYS - {"value"}),
                "incompatible operand units or denominator", ClaimError)
    values = [_number(quantity["value"]) for quantity in quantities]
    operation = derivation["operation"]
    require(operation == "weighted_mean" or not derivation["weights"], "unexpected weights", ClaimError)
    result = _arithmetic(operation, values, count, base, record["quantity"]["unit"], quantities[count:])
    base.update(value=str(result), aggregation=operation, measurement_kind="derived")
    _same_quantity(base, record["quantity"])
    return all(entry["checks"]["producer_authorized"] and entry["state"] == "active" for entry in entries)


def validate_claim(record, expected_binding, resolve_evidence, resolve_claim=None, verify_quantity=None):
    """Check one claim and return ``{record, checks, state}``.

    There is no default producer authorization and no scientific review: those
    checks stay False unless a host verifier says otherwise.
    """
    require(type(record) is dict, "claim must be an object", ClaimError)
    record = json.loads(_json(record))
    _binding(expected_binding)
    _binding(record.get("binding"))
    require(record["binding"] == expected_binding, "claim binding mismatch", ClaimError)
    kind = record.get("kind")
    require(type(kind) is str and kind in _KIND_FIELDS and set(record) == COMMON_KEYS | _KIND_FIELDS[kind],
            "invalid claim fields/kind", ClaimError)
    for key in ("statement", "applicability", "next_test"):
        _text(record[key])
    for key in ("evidence", "depends_on", "assumptions"):
        require(type(record[key]) is list, "claim lists required", ClaimError)
    for assumption in record["assumptions"]:
        _text(assumption)
    for ref in record["evidence"]:
        resolve_artifact(ref, resolve_evidence)
    for ref in record["depends_on"]:
        _ref(ref)
        require(callable(resolve_claim), "dependency resolver required", ClaimError)
        resolve_claim(ref)
    checks = dict(evidence_integrity_checked=bool(record["evidence"]), quantity_checked=False,
                  arithmetic_checked=False, producer_authorized=False, scientifically_reviewed=False,
                  native_eligible=False)
    if kind == "reported_quantity":
        checks.update(quantity_checked=True, producer_authorized=_check_extraction(
            record, expected_binding, resolve_evidence, verify_quantity))
    elif kind == "derived_quantity":
        _quantity(record["quantity"])
        require(callable(resolve_claim), "operand resolver required", ClaimError)
        require(type(record["derivation"]) is dict, "derivation must be an object", ClaimError)
        operands = record["derivation"].get("operands", []) + record["derivation"].get("weights", [])
        require(all(ref in record["depends_on"] for ref in operands), "numeric operands must be explicit dependencies",
                ClaimError)
        checks.update(arithmetic_checked=True, producer_authorized=_calculate(record, resolve_claim))
    if record["depends_on"]:
        checks["producer_authorized"] = checks["producer_authorized"] and all(
            resolve_claim(ref)["checks"]["producer_authorized"] and resolve_claim(ref)["state"] == "active"
            for ref in record["depends_on"])
    return {"record": record, "checks": checks, "state": "active"}


def _check_extraction(record, expected_binding, resolve_evidence, verify_quantity):
    """Check a reported quantity against the evidence field it quotes; return producer authorization."""
    extraction = record["extraction"]
    require(type(extraction) is dict and set(extraction) == {"ref", "pointer"}
            and extraction["ref"] in record["evidence"], "extraction must cite declared evidence", ClaimError)
    envelope, data = resolve_artifact(extraction["ref"], resolve_evidence)
    selected = json_pointer(_loads(data), extraction["pointer"])
    require(type(selected) is dict and set(selected) == {"binding", "quantity"},
            "selected field needs binding and quantity", ClaimError)
    require(selected["binding"] == expected_binding, "extracted binding/scope mismatch", ClaimError)
    for key, claim_key in _ENVELOPE_KEYS:
        require(envelope["binding"][key] == expected_binding[claim_key], "evidence envelope binding mismatch",
                ClaimError)
    _same_quantity(selected["quantity"], record["quantity"])
    return bool(verify_quantity and verify_quantity(extraction["ref"], selected, expected_binding) is True)


class ClaimLog:
    """Append-only chain of claim and correction events stored as evidence.

    The caller persists ``head`` with the research store's revision-checked
    checkpoint.
    """

    def __init__(self, store, head=None, *, resolve_evidence=None, verify_quantity=None):
        self.store, self.head = store, head
        self.resolve_evidence = resolve_evidence or store.evidence
        self.verify_quantity = verify_quantity

    def _append(self, candidate_id, event):
        event.update(schema_version=1, previous=self.head, candidate_id=candidate_id)
        name = "claim.json" if event["kind"] == "claim" else "correction.json"
        evidence_id = self.store.archive_evidence(candidate_id, {name: _json(event)})
        self.head = {"evidence_id": evidence_id, "artifact": name}
        return dict(self.head)

    def _check_candidate(self, candidate_id, binding, message):
        candidate = self.store.get_candidate(candidate_id)
        for key in ("job_id", "source_id", "measurement_epoch_id"):
            require(candidate[key] == binding[key], message, ClaimError)

    def append(self, candidate_id, record, expected_binding):
        """Validate a claim against the current view and append it."""
        self._check_candidate(candidate_id, expected_binding, "candidate binding mismatch")
        view = self.view()
        validate_claim(record, expected_binding, self.resolve_evidence,
                       lambda ref: view["claims"][_ref(ref)], self.verify_quantity)
        return self._append(candidate_id, {"kind": "claim", "record": record})

    def correct(self, candidate_id, targets, *, scope, reason, consequence, replacement=None, status="disputed"):
        """Mark claims disputed or superseded; dependent claims become stale."""
        self._check_candidate(candidate_id, scope, "correction candidate binding mismatch")
        view = self.view()
        event = {"kind": "claim_correction", "targets": targets, "scope": scope, "reason": reason,
                 "consequence": consequence, "replacement": replacement, "status": status}
        self._correct(view, event)
        return self._append(candidate_id, event)

    @staticmethod
    def _correct(view, event):
        """Apply one correction event to ``view`` in place."""
        _binding(event["scope"])
        _text(event["reason"])
        _text(event["consequence"])
        require(event["status"] in ("disputed", "superseded") and type(event["targets"]) is list and event["targets"],
                "invalid correction", ClaimError)
        target_keys = {_ref(ref) for ref in event["targets"]}
        for key in target_keys:
            require(key in view["claims"] and view["claims"][key]["record"]["binding"] == event["scope"],
                    "correction target/scope mismatch", ClaimError)
            require(view["claims"][key]["state"] != "superseded", "cannot overwrite an existing supersession",
                    ClaimError)
        if event["status"] == "superseded":
            ClaimLog._check_replacement(view, event, target_keys)
        else:
            require(event["replacement"] is None, "dispute cannot silently select a replacement", ClaimError)
        for key in target_keys:
            view["claims"][key]["state"] = event["status"]
        changed = True
        while changed:
            changed = False
            for entry in view["claims"].values():
                if entry["state"] == "active" and any(view["claims"][_ref(ref)]["state"] != "active"
                                                      for ref in entry["record"]["depends_on"]):
                    entry["state"] = "stale"
                    changed = True
        view["corrections"].append(event)

    @staticmethod
    def _check_replacement(view, event, target_keys):
        """A replacement must exist, share job/case, and not depend on a target."""
        replacement = _ref(event["replacement"])
        require(replacement in view["claims"] and replacement not in target_keys,
                "missing or self-referential replacement", ClaimError)
        require(all(view["claims"][replacement]["record"]["binding"][key] == event["scope"][key]
                    for key in ("job_id", "case_id")), "replacement belongs to another job/case", ClaimError)
        pending, seen = [replacement], set()
        while pending:
            key = pending.pop()
            require(key not in target_keys, "replacement depends on superseded premise", ClaimError)
            if key not in seen:
                seen.add(key)
                pending.extend(_ref(ref) for ref in view["claims"][key]["record"]["depends_on"])

    def view(self):
        """Replay the chain into ``{head, claims, corrections}``.

        If ``verify_quantity`` has a ``verification_view`` context manager, the
        replay runs inside it.
        """
        lifecycle = getattr(self.verify_quantity, "verification_view", None)
        if self.head is not None and callable(lifecycle):
            with lifecycle():
                return self._read_view()
        return self._read_view()

    def _read_events(self):
        """Walk the chain from head to start; return [(ref, event)] newest first."""
        events, seen, head = [], set(), self.head
        while head is not None:
            key = _ref(head)
            require(key not in seen, "claim history cycle", ClaimError)
            seen.add(key)
            envelope, data = resolve_artifact(head, self.resolve_evidence)
            event = _loads(data)
            require(type(event) is dict and event.get("kind") in ("claim", "claim_correction"), "invalid claim event",
                    ClaimError)
            fields = ({"record"} if event["kind"] == "claim" else
                      {"targets", "scope", "reason", "consequence", "replacement", "status"})
            require(set(event) == fields | {"schema_version", "previous", "candidate_id", "kind"}
                    and type(event["schema_version"]) is int and event["schema_version"] == 1
                    and event["candidate_id"] == envelope["candidate_id"], "claim event envelope mismatch", ClaimError)
            binding = event["record"]["binding"] if event["kind"] == "claim" else event["scope"]
            for key, bound_key in _ENVELOPE_KEYS:
                require(envelope["binding"][key] == binding[bound_key], "claim event source/job/epoch mismatch",
                        ClaimError)
            events.append((head, event))
            head = event["previous"]
        return events

    def _read_view(self):
        view = {"head": self.head, "claims": {}, "corrections": []}
        for ref, event in reversed(self._read_events()):
            if event["kind"] == "claim":
                record = event["record"]
                entry = validate_claim(record, record["binding"], self.resolve_evidence,
                                       lambda dependency: view["claims"][_ref(dependency)], self.verify_quantity)
                entry["ref"] = ref
                if any(view["claims"][_ref(item)]["state"] != "active" for item in record["depends_on"]):
                    entry["state"] = "stale"
                view["claims"][_ref(ref)] = entry
            else:
                require(event["kind"] == "claim_correction", "unknown claim event", ClaimError)
                self._correct(view, event)
        return view


def check_premises(premises, expected_binding, view, *, expected_head, expected_case_bindings=None):
    """Sort requested premises into facts, hypotheses and rejected claims.

    A premise used as a fact must be an active, producer-authorized quantity in
    the expected scope. Free-text reasoning is never checked.
    """
    _binding(expected_binding)
    if expected_case_bindings is None:
        expected_case_bindings = {expected_binding["case_id"]: expected_binding}
    require(type(expected_case_bindings) is dict and bool(expected_case_bindings), "missing host case bindings",
            ClaimError)
    for case, binding in expected_case_bindings.items():
        _binding(binding)
        require(case == binding["case_id"] and all(binding[key] == expected_binding[key]
                for key in BINDING_KEYS - {"case_id", "work_id"}), "host case binding changes candidate scope",
                ClaimError)
    require(expected_case_bindings.get(expected_binding["case_id"]) == expected_binding,
            "host case bindings omit or change default scope", ClaimError)
    require(view["head"] == expected_head, "stale claim view; current host head required", ClaimError)
    result = {"facts": [], "hypotheses": [], "rejected": [],
              "coverage": "declared_structured_premises_only" if premises else "no_registered_premises",
              "free_text_rationale_verified": False,
              "limits": "An empty gate does not validate the planner rationale. Only declared structured quantities "
                        "are checked; free-text numbers and scientific explanations remain unverified hypotheses."}
    for premise in premises:
        require(set(premise) == {"claim", "use", "reason", "next_test"}, "invalid premise request", ClaimError)
        entry = view["claims"].get(_ref(premise["claim"]))
        require(entry is not None, "unknown claim reference; use an exact claim ref from the current claims view, "
                "not a raw evidence or quantity ref", ClaimError)
        problems = _premise_problems(entry, expected_case_bindings)
        item = {"claim": premise["claim"], "limitations": problems,
                "quantity": entry["record"].get("quantity"), "binding": entry["record"]["binding"],
                "check_scope": "structured quantity only; statement interpretation and native eligibility are unverified"}
        if premise["use"] == "hypothesis":
            _text(premise["reason"])
            _text(premise["next_test"])
            result["hypotheses"].append(item)
        else:
            require(premise["use"] == "fact", "unknown premise use", ClaimError)
            result["rejected" if problems else "facts"].append(item)
    return result


def _premise_problems(entry, expected_case_bindings):
    problems = []
    bound = entry["record"]["binding"]
    if bound != expected_case_bindings.get(bound["case_id"]):
        problems.append("binding_mismatch")
    if entry["state"] != "active":
        problems.append(entry["state"])
    if entry["record"]["kind"] == "scientific_claim":
        problems.append("scientific_interpretation_unverified")
    if not entry["checks"]["producer_authorized"]:
        problems.append("quantity_producer_not_authorized")
    return problems


def check_repeat(proposed, settled_attempts, delta=None, *, resolve_evidence=None):
    """Allow an exact repeat of a settled attempt only with a stated reason.

    Matching is on the exact attempt identity; source similarity and novelty
    are not judged. A ``new_evidence`` or ``resolved_prerequisite`` repeat must
    cite evidence bytes not already in the earlier attempts' records.
    """
    require(type(proposed) is dict and set(proposed) == REPEAT_KEYS, "incomplete attempt identity", ClaimError)
    for value in proposed.values():
        _text(value)
    require(is_sha256(proposed["source_id"]), "invalid attempt source identity", ClaimError)
    require(type(settled_attempts) is list, "attempt history must be a list", ClaimError)
    for item in settled_attempts:
        require(type(item) is dict and {"attempt_id", "binding", "settled"} <= set(item)
                and type(item["binding"]) is dict and set(item["binding"]) == REPEAT_KEYS,
                "incomplete prior attempt identity", ClaimError)
        _text(item["attempt_id"])
    matches = [item for item in settled_attempts if item["binding"] == proposed]
    require(all(item.get("settled") is True for item in matches), "matching prior attempt is unsettled", ClaimError)
    ids = [item["attempt_id"] for item in matches]
    if not matches:
        return {"admissible": True, "matching_attempts": [], "reason": "no exact match; novelty not established"}
    require(type(delta) is dict and set(delta) == {"kind", "reason", "expected_observation", "reviewed_attempts",
                                                   "evidence"},
            "repeat requires an explicit delta or replication question", ClaimError)
    _text(delta["reason"])
    _text(delta["expected_observation"])
    require(type(delta["reviewed_attempts"]) is list and set(ids) <= set(delta["reviewed_attempts"]),
            "prior attempts were not acknowledged", ClaimError)
    require(delta["kind"] in ("new_evidence", "resolved_prerequisite", "replication", "new_question"),
            "unchanged exact identity is not a changed realization", ClaimError)
    require(type(delta["evidence"]) is list, "repeat evidence must be a list", ClaimError)
    needs_new_bytes = delta["kind"] not in ("replication", "new_question")
    if needs_new_bytes:
        require(delta["evidence"] and callable(resolve_evidence),
                "new evidence needs a resolver and exact references", ClaimError)
    previous_payloads = set()
    for attempt in matches:
        for key in ("transport_ref", "collection_ref"):
            if attempt.get(key) is not None:
                require(callable(resolve_evidence), "retained repeat history requires an evidence resolver",
                        ClaimError)
                envelope, _ = resolve_artifact(attempt[key], resolve_evidence)
                previous_payloads.update(envelope["artifacts"].values())
    new_payloads = set()
    for ref in delta["evidence"]:
        require(callable(resolve_evidence), "evidence resolver required", ClaimError)
        _, data = resolve_artifact(ref, resolve_evidence)
        new_payloads.add(sha256_hex(data))
    if needs_new_bytes:
        require(bool(new_payloads - previous_payloads),
                "unchanged prior payload is not new evidence; supply new analysis or declare replication", ClaimError)
    return {"admissible": True, "matching_attempts": ids, "reason": delta["reason"],
            "limitation": "declared purpose checked; scientific value of repetition unverified"}
