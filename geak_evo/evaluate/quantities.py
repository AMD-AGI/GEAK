"""Recorded timing from the existing admitted host collection boundary.

Only the serialized host observation is attested. No physical custody,
endpoint eligibility, uncertainty, or causal interpretation is established.
"""
from contextlib import contextmanager
from decimal import Decimal
import json

from .._util import canonical_json, require, sha256_hex


SCHEMA = "native-host-recorded-quantities-v1"
SCOPE = "recorded-native-per-case-timing-v1"
UNKNOWN = dict(physical_hbm_bytes=None, sustainable_bandwidth=None,
               achieved_residency=None, roofline_headroom=None, causal_hardware_interpretation=None)
LIMITS = ("Only the structured recorded timing scalar is authorized. Free-text notes, causal explanations, "
          "and undeclared dependencies remain unverified. Physical HBM bytes, achieved residency, sustainable "
          "bandwidth and roofline headroom are unknown to this producer. Native development timing does not "
          "establish an endpoint or noise floor.")




def encoded(value):
    return canonical_json(value)


def digest(value):
    return sha256_hex(encoded(value))


class NumericToken(str):
    pass


def decode(raw, tokens=False):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate quantity evidence JSON field", ValueError)
            result[key] = value
        return result
    def reject(value):
        raise ValueError("nonfinite quantity evidence JSON")
    kwargs = dict(object_pairs_hook=pairs, parse_constant=reject)
    if tokens:
        kwargs.update(parse_int=NumericToken, parse_float=NumericToken)
    return json.loads(raw, **kwargs)


def allocation(admission):
    physical, bdfs = admission.get("physical_ids"), admission.get("bdfs")
    require(type(physical) is list and type(bdfs) is list and len(physical) == len(bdfs) == 1,
         "quantity producer requires one admitted physical mapping", ValueError)
    require(type(physical[0]) is str and type(bdfs[0]) is str
         and physical[0].startswith(bdfs[0].lower() + "@")
         and int(physical[0].split("@", 1)[1], 16) > 0,
         "quantity producer needs full nonzero physical identity, not BDF alone", ValueError)
    return [{"local_ordinal": 0, "bdf": bdfs[0], "physical_id": physical[0]}]


class NativeQuantities:
    def __init__(self, store, campaign_id, *, contract, admission, case_id, producer_ref):
        self.store, self.campaign_id = store, campaign_id
        require(case_id in contract["benchmark_case_ids"], "quantity case is outside complete contract", ValueError)
        self.contract, self.case_id = contract, case_id
        try:
            mapping = allocation(admission)
        except (ValueError, TypeError, IndexError):
            mapping = None  # Unknown device identity permits hypotheses, never timing facts.
        self.descriptors = {case: {"schema": SCHEMA, "producer_ref": producer_ref,
            "job_id": digest(contract), "case_id": case, "case_binding": contract["case_bindings"][case],
            "report_contract": contract["report_contract"], "runtime": contract["runtime"],
            "physical_mapping": mapping, "scope": SCOPE} for case in contract["benchmark_case_ids"]}
        self.descriptor = self.descriptors[case_id]
        self._view = None

    def _descriptor(self, case_id=None):
        case_id = self.case_id if case_id is None else case_id
        require(type(case_id) is str and case_id in self.descriptors, "quantity case is outside complete contract", ValueError)
        return self.descriptors[case_id]

    def binding(self, candidate_id, case_id=None):
        descriptor = self._descriptor(case_id)
        candidate = self.store.get_candidate(candidate_id)
        require(candidate["job_id"] == descriptor["job_id"], "quantity candidate job differs", ValueError)
        return {**{key: candidate[key] for key in ("job_id", "source_id", "measurement_epoch_id")},
            "case_id": descriptor["case_id"],
            "work_id": digest({"schema": "protected-complete-job-case-v1", "job_id": candidate["job_id"],
                               "case_id": descriptor["case_id"], "case_binding": descriptor["case_binding"]}),
            "hardware_id": (descriptor["physical_mapping"][0]["physical_id"]
                            if descriptor["physical_mapping"] else "unknown-physical-identity"), "scope": SCOPE}

    def case_bindings(self, candidate_id):
        """Host-derived claim scopes; the complete native attempt identity is unchanged."""
        return {case: self.binding(candidate_id, case) for case in self.contract["benchmark_case_ids"]}

    def _artifact(self, ref):
        key = (ref["evidence_id"], ref["artifact"])
        cache = self._view["artifacts"] if self._view is not None else {}
        if key not in cache:
            cache[key] = self.store.evidence_artifact(ref["evidence_id"], ref["artifact"])
        return cache[key]

    @contextmanager
    def verification_view(self, state=None):
        """One journal read and one hash read per artifact in this claim view.

        The cache ends with the view. No prior authorization survives a later
        view. No episode replay or process launch occurs. Existing journal read
        costs remain; this is not a hard wall-time bound.
        """
        if self._view is not None:
            yield
            return
        if state is None:
            state = self.store.read_current(self.campaign_id)["note"]["evolution"]
        self._view = {"state": state, "artifacts": {}, "authorized": {}}
        try:
            yield
        finally:
            self._view = None

    def _derive(self, candidate_id, refs, case_id=None):
        from ..archive.claims import _number

        descriptor = self._descriptor(case_id)
        case_id = descriptor["case_id"]
        require(set(refs) == {"evaluation", "host", "request", "plan", "owner_report"},
             "quantity provenance is incomplete", ValueError)
        raw = {key: self._artifact(ref)[1] for key, ref in refs.items()}
        evaluation, host, request, plan, report = [decode(raw[key]) for key in
            ("evaluation", "host", "request", "plan", "owner_report")]
        candidate = self.store.get_candidate(candidate_id)
        bound = {"candidate_source_id": candidate["source_id"], "job_id": candidate["job_id"],
                 "measurement_epoch_id": candidate["measurement_epoch_id"]}
        # Native trial and outer source-role candidate IDs can differ. Exact code,
        # job and epoch must agree; native IDs must join their own request/receipt.
        native_binding = evaluation["binding"]
        require(all(native_binding[key] == value for key, value in bound.items())
             and host["binding"] == request["binding"] == native_binding,
             "quantity source/job/epoch binding differs", ValueError)
        require(evaluation["schema"] == "geak-arena-evaluation-v1"
             and host["schema"] == "geak-arena-settlement-v1"
             and request["schema"] == "geak-arena-evaluation-request-v1"
             and evaluation["evaluation_id"] == host["operation_id"] == request["operation_id"]
             and evaluation["request_sha256"] == host["request_sha256"] == digest(request)
             and request["contract"] == self.contract
             and request["source"] == self.store.source(candidate["source_id"])
             and [row["path"] for row in request["source"]["files"]] == sorted(self.contract["source_paths"])
             and all(request[key] == self.contract[key] for key in
                     ("phases", "correctness_case_ids", "benchmark_case_ids", "benchmark_report_ids")),
             "quantity operation or protected contract differs", ValueError)
        require(evaluation["settled"] is True and evaluation["native_suite_pass_observed"] is True
             and evaluation["collection_errors"] == [] and host["errors"] == []
             and host["execution_started"] is True and host["cleanup"]["settled"] is True
             and host["cleanup"]["evidence"]
             and set(host["phase_exit_codes"]) == set(request["phases"])
             and all(type(code) is int and code == 0 for code in host["phase_exit_codes"].values()),
             "quantity requires positively settled checked native observation", ValueError)
        for key, name in (("plan", "host/plan.json"), ("owner_report", "host/owner-report.json")):
            require(host["artifacts_sha256"].get(name) == sha256_hex(raw[key]),
                 "quantity physical witness is outside host raw inventory", ValueError)
        mapping = descriptor["physical_mapping"]
        require(mapping is not None, "quantity physical identity is unknown", ValueError)
        bdfs, physical = [row["bdf"] for row in mapping], [row["physical_id"] for row in mapping]
        require(plan["allocation_mode"] == "disjoint-physical-v1"
             and plan["physical_ids"] == physical and plan["bdfs"] == bdfs
             and report["smoke"]["physical_devices"] == bdfs and report["smoke"]["device_mapping"] == mapping
             and all(host["runtime"][key] == value for key, value in self.contract["runtime"].items())
             and host["runtime"]["physical_devices"] == bdfs,
             "quantity observed runtime/physical mapping differs from admitted producer", ValueError)
        observations = host["observations"]
        require(evaluation["native_observations"] == observations
             and observations["correctness"]["status"] == observations["benchmark"]["status"] == "passed"
             and observations["correctness"]["case_ids"] == request["correctness_case_ids"]
             and request["benchmark_report_ids"] == self.contract["benchmark_report_ids"]
             and request["benchmark_case_ids"] == self.contract["benchmark_case_ids"],
             "quantity observation/case coverage differs", ValueError)
        require(evaluation["per_case_times_ms"] == {case: observations["benchmark"]["times_ms"][report_id]
              for case, report_id in request["benchmark_report_ids"].items()},
             "quantity scalar differs from recorded host observation", ValueError)
        evaluation_tokens, host_tokens = decode(raw["evaluation"], tokens=True), decode(raw["host"], tokens=True)
        for case, report_id in request["benchmark_report_ids"].items():
            operands = [evaluation_tokens["per_case_times_ms"][case],
                evaluation_tokens["native_observations"]["benchmark"]["times_ms"][report_id],
                host_tokens["observations"]["benchmark"]["times_ms"][report_id]]
            require(all(type(value) is NumericToken for value in operands),
                 "quantity timing must use JSON numeric tokens", ValueError)
            decimals = [_number(str(value)) for value in operands]
            require(decimals[0] > Decimal(0) and all(value == decimals[0] for value in decimals),
                 "quantity decimal scalar differs across host/evaluation fields", ValueError)
        token = evaluation_tokens["per_case_times_ms"][case_id]
        population = {"operation_id": evaluation["evaluation_id"], "case_id": case_id,
            "report_id": request["benchmark_report_ids"][case_id],
            "report_contract": self.contract["report_contract"], "runtime": host["runtime"],
            "statistic": "exact serialized host evaluation per_case_times_ms token"}
        pair = {"binding": self.binding(candidate_id, case_id), "quantity": {
            "value": str(token), "unit": "ms", "denominator": "one bound complete callable",
            "population": encoded(population).decode(), "aggregation": "host_reported_per_case_statistic",
            "measurement_kind": "recorded_native_development_timing"}}
        flags = {key: evaluation.get(key) for key in ("measurement_valid", "custody_proven", "reward_eligible")}
        return pair, flags

    def publish(self, manifest, roles, raw_files, *, operation_id):
        """Called by the admitted collector; publication alone is not authority."""
        sources = {self.store.get_candidate(identity)["source_id"]: identity for identity in roles.values()}
        by_path = {row["original_ref"]["path"]: row for row in raw_files}
        rows, unavailable = [], []
        names = {"evaluation": "arena_evaluation.json", "host": "host_receipt.json", "request": "request.json",
                 "plan": "raw/host/plan.json", "owner_report": "raw/host/owner-report.json"}
        for trial in manifest["trials"]:
            value = trial.get("verified")
            if not value or trial.get("error"):
                continue
            source = value["source_id"]
            if source not in sources:
                unavailable.append({"source_id": source, "reason": "source_has_no_retained_candidate_role"})
                continue
            try:
                refs = {}
                for key, name in names.items():
                    found = [item for item in trial["artifacts"]
                             if item["input_ref"]["name"] == name and item["error"] is None]
                    require(len(found) == 1, "quantity raw artifact missing or ambiguous: " + name, ValueError)
                    original = found[0]["archive_ref"]
                    indexed = by_path[original["path"]]
                    require(indexed["original_ref"] == {key: original[key] for key in ("path", "sha256")},
                         "quantity artifact differs from retained collector inventory", ValueError)
                    refs[key] = indexed["evidence_ref"]
                    require(sha256_hex(self._artifact(refs[key])[1]) == original["sha256"],
                         "quantity archived bytes differ from collector reference", ValueError)
                candidate_id = sources[source]
                for case in self.contract["benchmark_case_ids"]:
                    pair, flags = self._derive(candidate_id, refs, case)
                    proof = {"schema": SCHEMA, "descriptor": self._descriptor(case), "attempt_id": operation_id,
                             "candidate_id": candidate_id, "raw_refs": refs, "native_flags": flags, "limits": LIMITS}
                    identity = self.store.archive_evidence(candidate_id,
                        {"quantity.json": encoded(pair), "quantity_provenance.json": encoded(proof)})
                    rows.append({"ref": {"evidence_id": identity, "artifact": "quantity.json"},
                                 "candidate_id": candidate_id, **pair, "native_flags": flags})
            except (ValueError, KeyError, TypeError, OSError) as exc:
                unavailable.append({"source_id": source, "reason": str(exc)[:400]})
        return {"schema": SCHEMA, "descriptor": self.descriptor, "attempt_id": operation_id,
                **({"case_descriptors": self.descriptors} if len(self.descriptors) > 1 else {}),
                "quantities": rows, "unavailable": unavailable,
                "unknown_hardware_quantities": dict(UNKNOWN), "limits": LIMITS}

    def _indexes(self, state):
        for row in state["history"]:
            if row.get("settled") is not True or not row.get("collection_ref") or not row.get("attempt_id"):
                continue
            ref = dict(row["collection_ref"], artifact="native_quantity_index.json")
            try:
                envelope, raw = self._artifact(ref)
            except KeyError:
                continue  # Historical collections are explicitly unsupported.
            index = decode(raw)
            require(envelope["candidate_id"] == row["candidate_id"]
                 and index["schema"] == SCHEMA and index["descriptor"] == self.descriptor
                 and index["attempt_id"] == row["attempt_id"], "committed quantity producer identity differs", ValueError)
            require(index.get("case_descriptors") == (self.descriptors if len(self.descriptors) > 1 else None),
                 "committed quantity case domain differs", ValueError)
            yield ref, index

    def __call__(self, ref, selected, expected_binding):
        try:
            with self.verification_view():
                key = (ref["evidence_id"], ref["artifact"])
                if key not in self._view["authorized"]:
                    matches = [(entry, index["attempt_id"]) for _, index in self._indexes(self._view["state"])
                               for entry in index["quantities"] if entry["ref"] == ref]
                    require(matches and all(entry == matches[0] for entry in matches),
                         "quantity has no settled host producer in this campaign", ValueError)
                    entry, attempt_id = matches[0]
                    envelope, raw = self._artifact(ref)
                    _, proof_raw = self._artifact(dict(ref, artifact="quantity_provenance.json"))
                    proof = decode(proof_raw)
                    case = entry["binding"]["case_id"]
                    require(ref["artifact"] == "quantity.json" and proof["descriptor"] == self._descriptor(case)
                         and proof["attempt_id"] == attempt_id and proof["schema"] == SCHEMA
                         and proof["candidate_id"] == envelope["candidate_id"] == entry["candidate_id"],
                         "quantity proof or source differs", ValueError)
                    pair, flags = self._derive(entry["candidate_id"], proof["raw_refs"], case)
                    require(raw == encoded(pair) and entry["binding"] == pair["binding"]
                         and entry["quantity"] == pair["quantity"]
                         and entry["native_flags"] == proof["native_flags"] == flags,
                         "quantity does not reproduce from committed raw evidence", ValueError)
                    self._view["authorized"][key] = pair
                pair = self._view["authorized"][key]
                return pair == selected and pair["binding"] == expected_binding
        except (ValueError, KeyError, TypeError, OSError):
            return False

    def catalog(self, snapshot):
        """Planner/engineer navigation; factual use still rechecks raw joins."""
        with self.verification_view(snapshot["state"]):
            indexes = list(self._indexes(snapshot["state"]))
            return {"schema": SCHEMA, "quantities": [row for _, index in indexes for row in index["quantities"]],
                "index_refs": [ref for ref, _ in indexes],
                "unavailable": [row for _, index in indexes for row in index["unavailable"]],
                "unknown_hardware_quantities": dict(UNKNOWN), "free_text_notes_verified": False,
                "limits": LIMITS, "instruction": "A catalog ref names quantity.json evidence, not a registered claim. "
                    "To use a listed scalar in premises, first append a reported_quantity claim through "
                    "research_review.claim_requests with the exact candidate_id, binding, quantity and ref, "
                    "and extraction pointer ''. Then copy the resulting claim.json ref from the current claims "
                    "view into a later proposal's premises. An engineer proposal cannot register claims. "
                    "A reported review can be amended through the same research_review action when nonempty "
                    "claim_requests append or correct claims; preserve its exact outcome refs and current revision. "
                    "This catalog covers admitted timings for retained source roles, not the complete trial inventory. "
                    "Before claiming that a baseline or repeated source was never measured, inspect the native trial "
                    "ledger and exact source bindings; a missing catalog entry establishes no such absence. "
                    "Unregistered numbers and causal notes remain hypotheses. "
                    "Different trial populations cannot be combined by the existing same-population arithmetic."}
