"""Tests for geak_evo.archive.exchange (ported from tests/test_exchange.py).

CPU integrity fixtures only: no model, GPU, or semantic-success qualification.

Run with PYTHONPATH=<repo root>:  python3 -B tests/geak_evo/test_exchange.py
Parity run on the original modules: set EVO_PARITY_ORIGINAL=1 and put the
original ``evolution/src`` on PYTHONPATH instead.

Tests that need ``prepare_engineer_request`` to succeed use the engineer request
validator (``geak_evo.engineer.request``) and ``geak_evo.evaluate.jobs_v1``;
both are required.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

if os.environ.get("EVO_PARITY_ORIGINAL"):
    from geak_evo.engineer import source_reference
    from geak_evo.evo_archive import archive_record
    from geak_evo.exchange import (ExchangeError, exchange_scope, prepare_engineer_request,
                                   resolve_artifact, validate_exchange, validate_receipt)
    from geak_evo.research_state import ResearchStore, StateError
else:
    from geak_evo.archive.exchange import (ExchangeError, exchange_scope, prepare_engineer_request,
                                           resolve_artifact, validate_exchange, validate_receipt)
    from geak_evo.archive.records import archive_record
    from geak_evo.archive.refs import source_reference
    from geak_evo.archive.store import ResearchStore, StateError


if os.environ.get("EVO_PARITY_ORIGINAL"):
    from geak_evo.jobs import job_digest as JOB_DIGEST
else:
    from geak_evo.evaluate.jobs_v1 import job_digest as JOB_DIGEST




def fixture():
    """A complete CPU job contract and its artifact files (copied from test_jobs)."""
    files = {
        "source/baseline.py": b"def complete_call(x):\n    return x\n",
        "binding.json": b'{"entry":"source/baseline.py:complete_call"}',
        "framework.json": b'{"kind":"cpu_test_fixture"}',
        "workload.json": b'{"model":"cpu-fixture","tp":1}',
        "capture.json": b'{"fixture_events":["case-1","case-1"]}',
        "inputs.json": b'{"x":[1,2],"scale":1}',
        "numerical.json": b'{"comparison":"exact against captured baseline"}',
        "timing.json": b'{"timer":"fixture only","synchronize":"before and after"}',
        "graph.json": b'{"mode":"disabled"}',
        "state.json": b'{"counter":0}',
        "state_contract.json": b'{"reset":"before each call","compare":"output and counter"}',
        "inference.json": b'{"basis":"fixture extrapolation only"}',
    }
    job = {
        "schema_version": 1, "job_version": 1,
        "workload": {
            "model": "cpu-fixture/identity", "model_revision": "1" * 40,
            "framework_source": {"repository": "fixture://framework", "version": "fixture-v1",
                                 "revision": "2" * 40, "manifest_ref": "framework.json"},
            "device": "CPU fixture", "tp": 1, "isl_tokens": 2, "osl_tokens": 1,
            "concurrency": 1, "precision": "int64"},
        "accepted_stack_digest": "3" * 64,
        "callable": {"source_path": "source/baseline.py", "qualname": "complete_call",
                     "binding_ref": "binding.json", "stateful": False},
        "baseline": {"source_paths": ["source/baseline.py"], "support_paths": ["binding.json"]},
        "artifacts": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "provenance": {"kind": "test_fixture", "capture_ref": "capture.json",
                       "workload_ref": "workload.json"},
        "cases": [{"case_id": "case-1", "phase": "fixture", "rank": 0,
                   "tensors": {"x": {"shape": [2], "dtype": "int64", "strides": [1],
                                     "stride_unit": "elements", "layout": "contiguous"}},
                   "scalars": {"scale": 1}, "inputs_ref": "inputs.json", "state_refs": [],
                   "routing_refs": [],
                   "observed_frequency": {"count": 2, "unit": "calls", "provenance_ref": "capture.json"},
                   "inferred_frequency": None}],
        "numerical": {"reference": "baseline", "case_ids": ["case-1"],
                      "protocol_ref": "numerical.json", "state_contract_ref": None},
        "timing": {"case_ids": ["case-1"], "unit": "us", "protocol_ref": "timing.json",
                   "warmup_calls": 1, "measured_calls": 2, "aggregation": "median",
                   "scope": "complete_callable"},
        "graph": {"mode": "disabled", "protocol_ref": "graph.json"},
    }
    return job, files


class ExchangeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "work"
        self.workspace.mkdir()
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        job, files = fixture()
        for name, data in files.items():
            path = self.artifacts / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.job_path = self.root / "job.json"
        self.job_path.write_text(json.dumps(job))
        # Without the job module any stable identity works: no test then loads the job.
        self.job_id = JOB_DIGEST(job)
        self.store = ResearchStore(self.root / "store",
                                   allowed_epochs=[(self.job_id, "epoch"), (self.job_id, "old")],
                                   source_paths={self.job_id: ["implementation.py"]})
        self.parent = self.candidate(1)
        self.note = {"claim": "Faithful alternative", "scope": "CPU fixture", "next_test": "Inspect"}
        self.store.checkpoint("receiver", self.parent, expected_revision=0, note=self.note)
        self.donor = self.candidate(2)
        self.request = {
            "schema_version": 1, "exchange_id": "exchange-1", "job_id": self.job_id,
            "measurement_epoch_id": "epoch", "question": "Does a compact representation transfer?",
            "recipient": {"direction_id": "receiver", "revision": 1, "candidate_id": self.parent,
                          "source_id": self.source(self.parent), "island_id": "one",
                          "membership_ref": self.membership(self.parent, "one")},
            "offers": [self.offer("donor-a", self.donor, "two", "two")],
        }
        self.exchange_ref = self.archive(self.parent, self.request)
        self.engineer = {
            "schema_version": 1, "direction_id": "receiver",
            "job": {"path": str(self.job_path), "job_id": self.job_id,
                    "artifact_root": str(self.artifacts)},
            "measurement_epoch_id": "epoch", "workspace": str(self.workspace),
            "anchor": None, "tip": source_reference(self.store, self.parent),
            "donors": [], "read_only": [], "next_question": "Continue",
            "budget": {"wall_seconds": 30, "max_tokens": None},
            "continuation": {"mode": "fresh", "prior_receipt": None},
        }
        # The receiving workspace is restored without executing any candidate.
        (self.workspace / "implementation.py").write_text("value = 1\n")

    def source(self, candidate):
        return self.store.get_candidate(candidate)["source_id"]

    def candidate(self, value, **kwargs):
        (self.workspace / "implementation.py").write_text("value = %s\n" % value)
        source = self.store.capture_source(self.workspace, ["implementation.py"])
        return self.store.candidate(self.job_id, kwargs.pop("epoch", "epoch"), source, **kwargs)

    def archive(self, candidate, payload):
        evidence = self.store.archive_evidence(candidate, {"notes/record.json": json.dumps(payload).encode()})
        return {"evidence_id": evidence, "artifact": "notes/record.json"}

    def membership(self, candidate, island):
        charter = archive_record(self.store, candidate, {"kind": "evo_island_charter",
                                 "island_id": island, "question": "Explore", "descriptor_spec": {}})
        return archive_record(self.store, candidate, {
            "kind": "evo_membership",
            "island_id": island, "charter_ref": charter,
            "assignment": {"status": "tentative", "descriptors": {}, "evidence_refs": [],
                           "reason": "Agent hypothesis, not proven diversity"},
        })

    def offer(self, identity, candidate, origin, selected):
        return {"offer_id": identity, "kind": "source", "candidate_id": candidate,
                "source_id": self.source(candidate), "origin_island_id": origin,
                "selected_island_id": selected, "origin_ref": self.membership(candidate, origin),
                "membership_ref": self.membership(candidate, selected), "claim": "Smaller live state",
                "scope": "Unvalidated source", "carrier_refs": [],
                "obligations": [{"question": "Does rounding preserve the contract?",
                                 "status": "unknown", "evidence_refs": []}]}

    def receipt(self, decision="reject"):
        return {"schema_version": 1, "exchange_id": "exchange-1",
                "dispositions": [{"offer_id": "donor-a", "decision": decision,
                                  "reason": "Requires a different ownership map"}],
                "realizations": [], "bridge_repairs": [], "next_test": "Inspect tails"}

    def realization(self, level="source"):
        return {"offer_id": "donor-a", "level": level, "claim": "Adapted live-state organization",
                "scope": "implementation.py", "semantic_witness": "Accumulator lifetime in loop",
                "evidence_refs": []}

    def test_frozen_intra_inter_classification_survives_migration_and_many_offers(self):
        self.request["offers"][0] = self.offer("donor-a", self.donor, "two", "one")
        for number in (3, 4):
            self.request["offers"].append(self.offer("donor-%s" % number,
                                                      self.candidate(number), "three", "three"))
        frozen = validate_exchange(self.store, self.request)
        self.membership(self.donor, "four")  # Later migration cannot change the frozen request.
        self.assertEqual([exchange_scope(frozen, offer) for offer in frozen["offers"]],
                         ["intra", "inter", "inter"])
        ref = self.archive(self.parent, frozen)
        prepared = prepare_engineer_request(self.store, ref, self.engineer)
        self.assertEqual(len(prepared["donors"]), 3)
        changed = copy.deepcopy(frozen)
        changed["offers"][0]["selected_island_id"] = "four"
        with self.assertRaisesRegex(ExchangeError, "membership"):
            validate_exchange(self.store, changed)

    def test_source_free_concept_uses_note_without_fabricating_candidate(self):
        concept = self.archive(self.parent, {"schema_version": 1, "kind": "exchange_concept",
                                            "concept_id": "streaming-summary", "origin_island_id": "two",
                                            "island_id": "one", "derivation": "An unimplemented merge law"})
        offered = {key: value for key, value in self.request["offers"][0].items()
                   if key not in ("candidate_id", "source_id", "origin_ref", "membership_ref")}
        offered.update(kind="concept", selected_island_id="one", concept_ref=concept)
        self.request["offers"] = [offered]
        ref = self.archive(self.parent, self.request)
        before = set((self.store.root / "objects").iterdir())
        prepared = prepare_engineer_request(self.store, ref, self.engineer)
        self.assertEqual(prepared["donors"], [])
        self.assertEqual(set((self.store.root / "objects").iterdir()), before)
        report = self.receipt("adopt")
        report["realizations"] = [self.realization("concept")]
        validate_receipt(self.store, ref, report)
        offered["candidate_id"] = self.parent
        with self.assertRaises(ExchangeError):
            validate_exchange(self.store, self.request)

    def test_stale_recipient_is_rejected_but_historical_receipt_remains_auditable(self):
        child = self.candidate(8, parents=[self.parent])
        self.store.checkpoint("receiver", child, expected_revision=1, note=self.note, status="unfinished")
        with self.assertRaisesRegex(ExchangeError, "stale"):
            prepare_engineer_request(self.store, self.exchange_ref, self.engineer)
        validate_receipt(self.store, self.exchange_ref, self.receipt())
        self.assertEqual(self.store.get("receiver")["tip"], child)
        self.assertIsNone(self.store.get("receiver")["anchor"])

    def test_wrong_source_and_wrong_membership_envelope_fail(self):
        wrong = copy.deepcopy(self.request)
        wrong["offers"][0]["source_id"] = self.source(self.parent)
        with self.assertRaisesRegex(ExchangeError, "source mismatch"):
            validate_exchange(self.store, wrong)
        original = json.loads(resolve_artifact(self.store, self.request["offers"][0]["membership_ref"]))
        wrong = copy.deepcopy(self.request)
        wrong["offers"][0]["membership_ref"] = self.archive(self.parent, original)
        with self.assertRaisesRegex(ExchangeError, "membership"):
            validate_exchange(self.store, wrong)
        wrong_exchange = self.archive(self.donor, self.request)
        with self.assertRaisesRegex(ExchangeError, "evidence binding"):
            prepare_engineer_request(self.store, wrong_exchange, self.engineer)

    def test_incomplete_memberships_cannot_authorize_exchange_or_receipt(self):
        for role in ("recipient", "origin", "selected"):
            for missing in (("charter_ref",), ("assignment",), ("charter_ref", "assignment")):
                with self.subTest(role=role, missing=missing):
                    request = copy.deepcopy(self.request)
                    item = request["recipient"] if role == "recipient" else request["offers"][0]
                    field = "origin_ref" if role == "origin" else "membership_ref"
                    member = json.loads(resolve_artifact(self.store, item[field]))
                    for key in missing:
                        del member[key]
                    item[field] = self.archive(member["candidate_id"], member)
                    exchange_ref = self.archive(self.parent, request)
                    with self.assertRaisesRegex(ExchangeError, "invalid membership snapshot"):
                        validate_exchange(self.store, request)
                    with self.assertRaisesRegex(ExchangeError, "invalid membership snapshot"):
                        prepare_engineer_request(self.store, exchange_ref, self.engineer)
                    with self.assertRaisesRegex(ExchangeError, "invalid membership snapshot"):
                        validate_receipt(self.store, exchange_ref, self.receipt())
        self.assertEqual(self.engineer["donors"], [])

    def test_membership_charter_and_assignment_use_archive_validation(self):
        original = json.loads(resolve_artifact(self.store, self.request["offers"][0]["membership_ref"]))
        for defect in ("charter_island", "unsupported_assignment"):
            with self.subTest(defect=defect):
                member = copy.deepcopy(original)
                if defect == "charter_island":
                    member["charter_ref"] = archive_record(self.store, self.donor, {
                        "kind": "evo_island_charter", "island_id": "unselected",
                        "question": "Explore", "descriptor_spec": {}})
                else:
                    member["assignment"]["status"] = "supported"
                request = copy.deepcopy(self.request)
                request["offers"][0]["membership_ref"] = self.archive(self.donor, member)
                with self.assertRaisesRegex(ExchangeError, "invalid membership snapshot"):
                    validate_exchange(self.store, request)

    def test_membership_diagnostics_require_host_frozen_cases_through_all_apis(self):
        frozen_cases = {self.job_id: ["fixture-case"]}
        diagnostic = archive_record(self.store, self.donor, {
            "kind": "evo_diagnostic", "case_ids": ["fixture-case"], "compilation_ref": None,
            "artifacts": {}, "claim": "Compiler behavior remains unknown",
            "unknown_reason": "CPU exchange fixture has no compilation evidence"}, frozen_cases=frozen_cases)
        member = json.loads(resolve_artifact(self.store, self.request["offers"][0]["membership_ref"]))
        member["assignment"]["evidence_refs"] = [diagnostic]
        request = copy.deepcopy(self.request)
        request["offers"][0]["membership_ref"] = archive_record(
            self.store, self.donor, member, frozen_cases=frozen_cases)
        exchange_ref = self.archive(self.parent, request)
        calls = (
            lambda cases: validate_exchange(self.store, request, frozen_cases=cases),
            lambda cases: prepare_engineer_request(self.store, exchange_ref, self.engineer, frozen_cases=cases),
            lambda cases: validate_receipt(self.store, exchange_ref, self.receipt(), frozen_cases=cases),
        )
        for index, call in enumerate(calls):
            for cases in (None, {self.job_id: ["other-case"]}):
                with self.subTest(api=index, frozen_cases=cases):
                    with self.assertRaisesRegex(ExchangeError, "invalid membership snapshot"):
                        call(cases)
            call(frozen_cases)
        self.assertIsNone(self.store.get("receiver")["anchor"])

    def test_failed_bridge_and_rejection_do_not_promote_or_erase_tip(self):
        report = self.receipt()
        report["bridge_repairs"] = [{"offer_ids": ["donor-a"], "status": "failed",
                                     "claim": "Local handoff leaves a race", "scope": "CPU analysis",
                                     "evidence_refs": []}]
        before = self.store.get("receiver")
        self.assertEqual(validate_receipt(self.store, self.exchange_ref, report), report)
        self.assertEqual(self.store.get("receiver"), before)
        report["realizations"] = [self.realization()]
        with self.assertRaisesRegex(ExchangeError, "adopted"):
            validate_receipt(self.store, self.exchange_ref, report)

    def test_source_claim_requires_captured_child_with_exact_donor_provenance(self):
        report = self.receipt("adapt")
        report["realizations"] = [self.realization()]
        with self.assertRaisesRegex(ExchangeError, "host-captured"):
            validate_receipt(self.store, self.exchange_ref, report)
        child = self.candidate(5, parents=[self.parent])
        with self.assertRaisesRegex(ExchangeError, "uncited"):
            validate_receipt(self.store, self.exchange_ref, report, child_candidate_id=child)
        child = self.candidate(5, parents=[self.parent], donors=[self.donor])
        validated = validate_receipt(self.store, self.exchange_ref, report, child_candidate_id=child)
        self.assertEqual(validated["realizations"][0]["level"], "source")
        self.assertEqual(self.store.get_candidate(child)["contribution_claims"], [])
        self.assertIsNone(self.store.get("receiver")["anchor"])

    def test_receipts_cannot_supply_host_evidence_or_causal_success(self):
        for key, value in (("host_refs", {}), ("success", True), ("eligible", True), ("latency_ms", 1)):
            with self.subTest(key=key):
                report = self.receipt()
                report[key] = value
                with self.assertRaises(ExchangeError):
                    validate_receipt(self.store, self.exchange_ref, report)
        report = self.receipt("adopt")
        report["realizations"] = [self.realization("causal_benefit")]
        with self.assertRaises(ExchangeError):
            validate_receipt(self.store, self.exchange_ref, report)

    def test_all_offers_need_dispositions_and_unknown_ids_are_rejected(self):
        report = self.receipt()
        report["dispositions"] = []
        with self.assertRaises(ExchangeError):
            validate_receipt(self.store, self.exchange_ref, report)
        report = self.receipt()
        report["dispositions"][0]["offer_id"] = "unoffered"
        with self.assertRaises(ExchangeError):
            validate_receipt(self.store, self.exchange_ref, report)

    def test_prepared_engineer_request_keeps_existing_budget_and_continuation(self):
        prepared = prepare_engineer_request(self.store, self.exchange_ref, self.engineer)
        self.assertEqual(prepared["budget"], self.engineer["budget"])
        self.assertEqual(prepared["continuation"], self.engineer["continuation"])
        self.assertEqual(prepared["tip"], self.engineer["tip"])
        self.assertTrue(prepared["read_only"])
        self.assertEqual(self.engineer["donors"], [])
        withheld = prepare_engineer_request(self.store, self.exchange_ref, self.engineer, include_sources=False)
        self.assertEqual(withheld["donors"], [])
        wrong = copy.deepcopy(self.engineer)
        wrong["tip"] = source_reference(self.store, self.donor)
        with self.assertRaisesRegex(ExchangeError, "binding"):
            prepare_engineer_request(self.store, self.exchange_ref, wrong)

    def test_historical_donor_does_not_transfer_eligibility(self):
        old = self.candidate(9, epoch="old")
        self.request["offers"] = [self.offer("donor-a", old, "two", "two")]
        ref = self.archive(self.parent, self.request)
        prepared = prepare_engineer_request(self.store, ref, self.engineer)
        self.assertEqual(prepared["donors"][0]["candidate_id"], old)
        self.assertIsNone(self.store.get("receiver")["anchor"])

    def test_unsafe_missing_and_tampered_artifacts_are_rejected(self):
        for name in ("../notes/record.json", "/notes/record.json", "notes//record.json", "missing.json"):
            ref = dict(self.exchange_ref, artifact=name)
            with self.subTest(name=name), self.assertRaises(ExchangeError):
                resolve_artifact(self.store, ref)
        envelope, _ = self.store.evidence(self.exchange_ref["evidence_id"])
        path = self.store.root / "objects" / envelope["artifacts"][self.exchange_ref["artifact"]]
        path.chmod(0o644)
        path.write_bytes(b"forged")
        with self.assertRaises(StateError):
            resolve_artifact(self.store, self.exchange_ref)


class CrossoverDonorDriftTests(ExchangeTests):
    """Drift risk 7a: donors reach an engineer request only through a validated exchange_ref."""

    def _rejected(self, exchange_ref, pattern, error=ExchangeError):
        before = copy.deepcopy(self.engineer)
        with self.assertRaisesRegex(error, pattern):
            prepare_engineer_request(self.store, exchange_ref, self.engineer)
        self.assertEqual(self.engineer, before)

    def test_unsaved_or_forged_exchange_ref_adds_no_donor(self):
        self._rejected({"evidence_id": "0" * 64, "artifact": "notes/record.json"}, "", StateError)
        self._rejected(dict(self.exchange_ref, artifact="../notes/record.json"), "unsafe")
        self._rejected(dict(self.exchange_ref, artifact="other.json"), "missing artifact")
        self._rejected({"evidence_id": self.exchange_ref["evidence_id"]}, "")

    def test_exchange_saved_on_a_donor_is_not_the_recipients_exchange(self):
        self._rejected(self.archive(self.donor, self.request), "evidence binding")

    def test_offer_with_wrong_source_or_unsaved_candidate_adds_no_donor(self):
        wrong = copy.deepcopy(self.request)
        wrong["offers"][0]["source_id"] = self.source(self.parent)
        self._rejected(self.archive(self.parent, wrong), "source mismatch")
        wrong = copy.deepcopy(self.request)
        wrong["offers"][0]["candidate_id"] = "1" * 64
        with self.assertRaises((ExchangeError, StateError)):
            prepare_engineer_request(self.store, self.archive(self.parent, wrong), self.engineer)
        self.assertEqual(self.engineer["donors"], [])

    def test_unfrozen_exchange_with_moved_recipient_adds_no_donor(self):
        child = self.candidate(7, parents=[self.parent])
        self.store.checkpoint("receiver", child, expected_revision=1, note=self.note, status="unfinished")
        self._rejected(self.exchange_ref, "stale")

    def test_receipt_child_must_cite_the_validated_donor(self):
        report = self.receipt("adapt")
        report["realizations"] = [self.realization()]
        other = self.candidate(6)
        child = self.candidate(5, parents=[self.parent], donors=[other])
        with self.assertRaisesRegex(ExchangeError, "uncited"):
            validate_receipt(self.store, self.exchange_ref, report, child_candidate_id=child)


# The drift class inherits fixtures, not the base tests.
for _name in [name for name in vars(ExchangeTests) if name.startswith("test_")]:
    setattr(CrossoverDonorDriftTests, _name, None)


if __name__ == "__main__":
    unittest.main()
