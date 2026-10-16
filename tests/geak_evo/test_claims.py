"""CPU evidence fixtures; producer authorization here is synthetic, never native."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from geak_evo.archive.claims import (ClaimError, ClaimLog, check_premises, check_repeat,
                              json_pointer, resolve_artifact, validate_claim)
from geak_evo.archive.store import ResearchStore


class ClaimTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "work"
        self.workspace.mkdir()
        (self.workspace / "operation.py").write_text("value = 1\n")
        self.store = ResearchStore(self.root / "archive", allowed_epochs=[("job", "epoch-1")],
                                   source_paths={"job": ["operation.py"]})
        source = self.store.capture_source(self.workspace, ["operation.py"])
        self.candidate = self.store.candidate("job", "epoch-1", source)
        self.binding = {"job_id": "job", "case_id": "case-1", "source_id": source,
                        "work_id": "work-1", "measurement_epoch_id": "epoch-1",
                        "hardware_id": "CPU fixture", "scope": "complete_callable"}
        self.authorized = set()
        self.log = ClaimLog(self.store, verify_quantity=self.verify)

    def verify(self, ref, selected, expected):
        return ref["evidence_id"] in self.authorized

    @staticmethod
    def quantity(value, unit="us", **changes):
        result = {"value": value, "unit": unit, "denominator": "call", "population": "fixture-inputs",
                  "aggregation": "mean", "measurement_kind": "CPU fixture timing"}
        result.update(changes)
        return result

    def archive(self, data, candidate=None):
        identity = self.store.archive_evidence(candidate or self.candidate, {"metrics.json": data})
        self.authorized.add(identity)
        return {"evidence_id": identity, "artifact": "metrics.json"}

    def report(self, value="12.5", *, binding=None, candidate=None, unit="us", **quantity_changes):
        binding = binding or self.binding
        quantity = self.quantity(value, unit, **quantity_changes)
        ref = self.archive(json.dumps({"metrics": {"latency": {"binding": binding, "quantity": quantity}}}).encode(), candidate)
        record = {"kind": "reported_quantity", "statement": "The fixture artifact reports this complete-call mean.",
                  "binding": dict(binding), "evidence": [ref], "depends_on": [], "assumptions": [],
                  "applicability": "CPU fixture case only", "next_test": "Inspect the paired baseline fixture.",
                  "quantity": quantity, "extraction": {"ref": ref, "pointer": "/metrics/latency"}}
        return record

    def append_report(self, *args, **kwargs):
        record = self.report(*args, **kwargs)
        return self.log.append(kwargs.get("candidate", self.candidate), record, record["binding"])

    def scientific(self, evidence=(), dependencies=()):
        return {"kind": "scientific_claim", "statement": "The cause could be memory traffic; this is unverified.",
                "binding": dict(self.binding), "evidence": list(evidence), "depends_on": list(dependencies),
                "assumptions": ["Executed traffic is not established."], "applicability": "Fixture hypothesis",
                "next_test": "Inspect source-bound traffic and distinguish a dependency stall."}

    def derived(self, operation, operands, quantity, *, binding=None, weights=()):
        return {"kind": "derived_quantity", "statement": "A declared arithmetic result, not a causal explanation.",
                "binding": dict(binding or self.binding), "evidence": [], "depends_on": list(operands) + list(weights),
                "assumptions": ["Applicability still needs scientific interpretation."], "applicability": "Fixture scope",
                "next_test": "Check the independent fixture comparison.", "quantity": quantity,
                "derivation": {"operation": operation, "operands": list(operands), "weights": list(weights)}}

    def premises(self, refs, use="fact", binding=None, view=None):
        requests = [{"claim": ref, "use": use, "reason": "Investigate this exact prior claim.",
                     "next_test": "Compare a discriminating fixture observation."} for ref in refs]
        active = view or self.log.view()
        return check_premises(requests, binding or self.binding, active, expected_head=active["head"])

    def test_structured_extraction_has_separate_authority_and_native_status(self):
        record = self.report()
        checked = validate_claim(record, self.binding, self.store.evidence)
        self.assertTrue(checked["checks"]["quantity_checked"])
        self.assertFalse(checked["checks"]["producer_authorized"])
        ref = self.log.append(self.candidate, record, self.binding)
        self.assertEqual(len(self.premises([ref])["facts"]), 1)
        reopened = ClaimLog(self.store, self.log.head)
        self.assertEqual(len(self.premises([ref], view=reopened.view())["rejected"]), 1)
        entry = next(iter(self.log.view()["claims"].values()))
        self.assertFalse(entry["checks"]["native_eligible"])
        self.assertFalse(entry["checks"]["scientifically_reviewed"])

    def test_wrong_number_unit_denominator_and_scope_are_rejected(self):
        original = self.report()
        for field, wrong in (("value", "12500"), ("unit", "ms"), ("denominator", "wave"),
                             ("population", "active CUs"), ("aggregation", "minimum")):
            record = copy.deepcopy(original)
            record["quantity"][field] = wrong
            with self.subTest(field=field), self.assertRaises(ClaimError):
                self.log.append(self.candidate, record, self.binding)
        for field in ("case_id", "measurement_epoch_id", "hardware_id", "scope"):
            record = copy.deepcopy(original)
            record["binding"][field] = "wrong"
            with self.subTest(field=field), self.assertRaises(ClaimError):
                validate_claim(record, record["binding"], self.store.evidence)
        self.assertIsNone(self.log.head)

    def test_duplicate_keys_nonfinite_and_boolean_quantities_rejected(self):
        record = self.report()
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}'):
            ref = self.archive(raw)
            broken = copy.deepcopy(record)
            broken.update(evidence=[ref], extraction={"ref": ref, "pointer": "/x"})
            with self.subTest(raw=raw), self.assertRaises(ClaimError):
                self.log.append(self.candidate, broken, self.binding)
        record = self.report(True)
        with self.assertRaises(ClaimError):
            self.log.append(self.candidate, record, self.binding)

    def test_json_pointer_handles_escaped_names_and_rejects_ambiguous_indices(self):
        value = {"a/b": {"~value": [{"n": 12}]}}
        self.assertEqual(json_pointer(value, "/a~1b/~0value/0/n"), 12)
        for pointer in ("a/b", "/a~2b", "/a~1b/~0value/00", "/a~1b/~0value/-", "/absent"):
            with self.subTest(pointer=pointer), self.assertRaises(ClaimError):
                json_pointer(value, pointer)

    def test_exact_decimal_difference_and_explicit_unit_conversion(self):
        a, b = self.append_report("0.3"), self.append_report("0.2")
        difference = self.derived("difference", [a, b], self.quantity("0.1", aggregation="difference", measurement_kind="derived"))
        ref = self.log.append(self.candidate, difference, self.binding)
        self.assertEqual(len(self.premises([ref])["facts"]), 1)
        seconds = self.append_report("0.000002", unit="s")
        conversion = self.derived("convert", [seconds], self.quantity("2", "us", aggregation="convert", measurement_kind="derived"))
        self.log.append(self.candidate, conversion, self.binding)
        bad = copy.deepcopy(conversion)
        bad["quantity"]["unit"] = "FLOP/s"
        with self.assertRaisesRegex(ClaimError, "unsupported unit conversion"):
            self.log.append(self.candidate, bad, self.binding)

    def test_ratio_binds_both_source_operands_and_retains_above_one(self):
        baseline = self.append_report("12")
        (self.workspace / "operation.py").write_text("value = 2\n")
        source = self.store.capture_source(self.workspace, ["operation.py"])
        candidate = self.store.candidate("job", "epoch-1", source)
        binding = dict(self.binding, source_id=source, work_id="work-2")
        contender = self.append_report("8", candidate=candidate, binding=binding)
        ratio = self.derived("ratio", [baseline, contender], self.quantity("1.5", "1", denominator="second_operand",
                            aggregation="ratio", measurement_kind="derived"), binding=binding)
        ref = self.log.append(candidate, ratio, binding)
        self.assertEqual(len(self.premises([ref], binding=binding)["facts"]), 1)
        ratio["quantity"]["value"] = "1"
        with self.assertRaisesRegex(ClaimError, "value mismatch"):
            self.log.append(candidate, ratio, binding)
        self.assertEqual(len(self.premises([ref])["rejected"]), 1)

    def test_zero_denominator_and_unreferenced_operand_fail(self):
        a, zero = self.append_report("1"), self.append_report("0")
        ratio = self.derived("ratio", [a, zero], self.quantity("0", "1", denominator="second_operand",
                             aggregation="ratio", measurement_kind="derived"))
        with self.assertRaisesRegex(ClaimError, "nonzero denominator"):
            self.log.append(self.candidate, ratio, self.binding)
        ratio["depends_on"] = []
        with self.assertRaisesRegex(ClaimError, "explicit dependencies"):
            self.log.append(self.candidate, ratio, self.binding)

    def test_weighted_mean_requires_bound_nonnegative_weight_evidence(self):
        a, b = self.append_report("10"), self.append_report("20")
        weights = [self.append_report(value, unit="count", denominator="fixture", measurement_kind="frequency")
                   for value in ("1", "3")]
        mean = self.derived("weighted_mean", [a, b], self.quantity("17.5", aggregation="weighted_mean", measurement_kind="derived"), weights=weights)
        ref = self.log.append(self.candidate, mean, self.binding)
        self.assertEqual(len(self.premises([ref])["facts"]), 1)
        negative = self.append_report("-1", unit="count", denominator="fixture", measurement_kind="frequency")
        mean["derivation"]["weights"] = [negative, weights[1]]
        mean["depends_on"] = [a, b, negative, weights[1]]
        with self.assertRaisesRegex(ClaimError, "nonnegative weights"):
            self.log.append(self.candidate, mean, self.binding)

    def test_real_citation_does_not_upgrade_scientific_interpretation(self):
        report = self.report()
        hypothesis = self.scientific(report["evidence"])
        ref = self.log.append(self.candidate, hypothesis, self.binding)
        self.assertEqual(len(self.premises([ref])["rejected"]), 1)
        self.assertEqual(len(self.premises([ref], use="hypothesis")["hypotheses"]), 1)
        entry = next(iter(self.log.view()["claims"].values()))
        self.assertFalse(entry["checks"]["quantity_checked"])
        self.assertFalse(entry["checks"]["producer_authorized"])

    def test_scoped_correction_preserves_history_and_invalidates_dependencies(self):
        old = self.append_report("12")
        derived = self.derived("sum", [old], self.quantity("12", aggregation="sum", measurement_kind="derived"))
        dependent = self.log.append(self.candidate, derived, self.binding)
        previous_head = self.log.head
        replacement = self.append_report("11")
        correction = self.log.correct(self.candidate, [old], scope=self.binding,
                                      reason="Fixture record was corrected upstream.", consequence="Recompute the dependent assignment premise.",
                                      replacement=replacement, status="superseded")
        result = self.premises([old, dependent, replacement])
        self.assertEqual(len(result["facts"]), 1)
        self.assertEqual({item["limitations"][0] for item in result["rejected"]}, {"superseded", "stale"})
        historical = ClaimLog(self.store, previous_head, verify_quantity=self.verify).view()
        self.assertEqual(len(self.premises([old, dependent], view=historical)["facts"]), 2)
        self.assertEqual(correction["artifact"], "correction.json")
        self.assertEqual(len(self.log.view()["corrections"]), 1)
        self.assertEqual(len(self.premises([old], use="hypothesis")["hypotheses"]), 1)

    def test_dispute_scope_and_circular_replacement_are_checked(self):
        old = self.append_report()
        dependent = self.log.append(self.candidate, self.scientific(dependencies=[old]), self.binding)
        with self.assertRaisesRegex(ClaimError, "depends on superseded"):
            self.log.correct(self.candidate, [old], scope=self.binding, reason="r", consequence="c",
                             replacement=dependent, status="superseded")
        with self.assertRaisesRegex(ClaimError, "target/scope"):
            self.log.correct(self.candidate, [old], scope=dict(self.binding, case_id="other"), reason="r", consequence="c")
        self.log.correct(self.candidate, [old], scope=self.binding, reason="Conflicting fixture records.",
                         consequence="Resolve before using the old quantity.")
        self.assertIn("disputed", self.premises([old])["rejected"][0]["limitations"])

    def test_current_host_head_rejects_an_older_valid_view(self):
        old = self.append_report()
        earlier = self.log.view()
        self.log.correct(self.candidate, [old], scope=self.binding, reason="Conflicting source.", consequence="Investigate first.")
        with self.assertRaisesRegex(ClaimError, "stale claim view"):
            check_premises([{"claim": old, "use": "fact", "reason": "", "next_test": ""}],
                           self.binding, earlier, expected_head=self.log.head)

    def test_wrong_scope_operand_cannot_enter_a_derived_fact(self):
        first = self.append_report("2")
        other = self.append_report("3", binding=dict(self.binding, scope="stage-only"))
        record = self.derived("sum", [first, other], self.quantity("5", aggregation="sum", measurement_kind="derived"))
        with self.assertRaisesRegex(ClaimError, "operand binding/scope"):
            self.log.append(self.candidate, record, self.binding)

    def test_unverified_dependency_keeps_a_quantity_out_of_facts(self):
        hypothesis = self.log.append(self.candidate, self.scientific(), self.binding)
        record = self.report()
        record["depends_on"] = [hypothesis]
        ref = self.log.append(self.candidate, record, self.binding)
        self.assertEqual(len(self.premises([ref])["rejected"]), 1)

    def test_hypothesis_use_needs_an_actual_question(self):
        hypothesis = self.log.append(self.candidate, self.scientific(), self.binding)
        with self.assertRaisesRegex(ClaimError, "nonempty text"):
            check_premises([{"claim": hypothesis, "use": "hypothesis", "reason": "", "next_test": ""}],
                           self.binding, self.log.view(), expected_head=self.log.head)

    def test_artifact_corruption_and_unsafe_reference_are_rejected(self):
        record = self.report()
        ref = record["evidence"][0]
        envelope, artifacts = self.store.evidence(ref["evidence_id"])
        with self.assertRaisesRegex(ClaimError, "artifact hash mismatch"):
            resolve_artifact(ref, lambda _: (envelope, {"metrics.json": b"tampered"}))
        for name in ("../metrics.json", "/metrics.json", "./metrics.json", "dir//metrics.json"):
            with self.subTest(name=name), self.assertRaises(ClaimError):
                resolve_artifact(dict(ref, artifact=name), self.store.evidence)

    def test_repeat_gate_requires_acknowledged_change_or_replication(self):
        proposed = {"job_id": "job", "case_id": "case-1", "source_id": self.binding["source_id"],
                    "inputs_id": "input-digest", "configuration_id": "config-digest", "measurement_epoch_id": "epoch-1",
                    "diagnostic_id": "compile-check"}
        history = [{"attempt_id": "attempt-1", "binding": proposed, "settled": True}]
        with self.assertRaisesRegex(ClaimError, "explicit delta"):
            check_repeat(proposed, history)
        delta = {"kind": "replication", "reason": "Test whether the prior failure was transient.",
                 "expected_observation": "Same fixture either fails again or completes.", "reviewed_attempts": ["attempt-1"], "evidence": []}
        self.assertTrue(check_repeat(proposed, history, delta)["admissible"])
        delta["kind"] = "changed_realization"
        with self.assertRaisesRegex(ClaimError, "unchanged exact identity"):
            check_repeat(proposed, history, delta)
        changed = dict(proposed, source_id="a" * 64)
        self.assertIn("novelty not established", check_repeat(changed, history)["reason"])
        history[0]["settled"] = False
        with self.assertRaisesRegex(ClaimError, "unsettled"):
            check_repeat(proposed, history, delta)

    def test_new_evidence_repeat_needs_resolved_artifacts(self):
        proposed = {key: key for key in ("job_id", "case_id", "source_id", "inputs_id", "configuration_id", "measurement_epoch_id", "diagnostic_id")}
        proposed["source_id"] = self.binding["source_id"]
        history = [{"attempt_id": "old", "binding": proposed, "settled": True}]
        delta = {"kind": "new_evidence", "reason": "A fixture diagnostic changed the explanation.",
                 "expected_observation": "The same run should expose the new diagnostic condition.", "reviewed_attempts": ["old"], "evidence": []}
        with self.assertRaisesRegex(ClaimError, "exact references"):
            check_repeat(proposed, history, delta, resolve_evidence=self.store.evidence)
        delta["evidence"] = [self.archive(b'{"fixture_diagnostic":"new evidence"}')]
        self.assertTrue(check_repeat(proposed, history, delta, resolve_evidence=self.store.evidence)["admissible"])


if __name__ == "__main__":
    unittest.main()
