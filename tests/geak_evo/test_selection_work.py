"""Tests for geak_evo.selection.work (ported from the GEAK-EVO selection tests).

Run from the repo root:  python3 -B -m unittest tests.geak_evo.test_selection_work
"""

import copy
import math
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from geak_evo.selection.work import SelectionError, account_opportunities, advise, record_decision  # noqa: E402


BINDING = {"job_id": "1" * 64, "measurement_epoch_id": "epoch-1"}
LIMITS = {"engineer_active_seconds": 1000, "gpu_device_seconds": 1000, "model_usd": 100}
CONFIG = {"weights": {k: 1 / 3 for k in LIMITS}, "c_ref": 0.05,
          "half_life": 0.25, "epsilon": 0.1, "beta": math.log(1.02)}


def vector(fraction):
    return {k: fraction * value for k, value in LIMITS.items()}


def opportunity(key="a", fraction=0.2, settled=True, cost=0.1, sequence=0):
    return dict(BINDING, opportunity_id=key, direction_id=key, island_id="island-1",
                engineer_id="engineer-1", question="Can a faithful layout remove spilling?",
                opened_at=0, review_at=100, requested=vector(fraction), reserved=vector(fraction),
                receipts=[{"receipt_id": "cost-" + key, "costs": vector(cost)}],
                settled=settled, completed_sequence=sequence if settled else None,
                progress_complete=settled)


def accounting(records):
    return account_opportunities(records, binding=BINDING, limits=LIMITS, deadline=1000, now=20)


def offer(key="a", weight=1):
    return dict(BINDING, direction_id=key, island_id="island-1", engineer_id="engineer-1",
                priority_weight=weight, estimated_costs=vector(0.05))


def progress(key="a", source="2", gain=0.02):
    return dict(BINDING, opportunity_id=key, credit_id="credit-" + key,
                source_id=source * 64, frontier_before_source_id="0" * 64,
                gain_log_lcb=gain, evidence_ref={"evidence_id": source * 64, "artifact": "strict/measurement.json"})


def verified(ref, expected):
    # Deliberately synthetic verifier; no measurement authenticity is asserted.
    return ref == progress(expected["opportunity_id"], expected["source_id"][0])["evidence_ref"]


def calculate(records, credits=(), offers=None, config=None):
    return advise([offer()] if offers is None else offers, accounting(records), list(credits),
                  config=CONFIG if config is None else config, verify_progress=verified)


class SelectionTests(unittest.TestCase):
    def test_receipt_partition_does_not_reset_service_or_change_advice(self):
        whole = opportunity()
        split = copy.deepcopy(whole)
        split["receipts"] = [{"receipt_id": "part-" + str(i), "costs": vector(0.01)} for i in range(10)]
        first = calculate([whole], [progress()])
        second = calculate([split], [progress()])
        for key in ("rate", "service_count", "exploration_gain", "score", "recent_cost"):
            self.assertAlmostEqual(first["rows"][0][key], second["rows"][0][key])
        self.assertAlmostEqual(first["total_service_count"], second["total_service_count"])

    def test_unsettled_consumption_is_not_added_to_original_reservation(self):
        before, after = opportunity(settled=False, cost=0), opportunity(settled=False, cost=0.1)
        first, second = accounting([before]), accounting([after])
        self.assertEqual(second["opportunities"][0]["remaining_reservation"], vector(0.1))
        self.assertEqual(first["available"], second["available"])
        self.assertAlmostEqual(calculate([before])["total_service_count"], calculate([after])["total_service_count"])
        settled = accounting([opportunity()])
        self.assertEqual(settled["remaining_reservation"], vector(0))
        self.assertEqual(settled["available"], vector(0.9))

    def test_expiry_does_not_release_unsettled_work(self):
        item = opportunity(settled=False)
        result = account_opportunities([item], binding=BINDING, limits=LIMITS, deadline=1000, now=200)
        self.assertTrue(result["opportunities"][0]["expired"])
        self.assertEqual(result["available"], vector(0.8))

    def test_missing_cost_remains_unknown_and_reserved_after_settlement(self):
        item = opportunity()
        del item["receipts"][0]["costs"]["model_usd"]
        result = accounting([item])
        self.assertIsNone(result["delivered"]["model_usd"])
        self.assertEqual(result["remaining_reservation"]["model_usd"], 20)
        self.assertEqual(result["available"]["model_usd"], 80)
        advice = calculate([item], [progress()], [offer(), offer("new")])
        self.assertTrue(all(row["score"] is None for row in advice["rows"]))
        self.assertTrue(all(row["status"] == "unavailable" for row in advice["rows"]))
        self.assertTrue(all(row["recent_gain"] is None for row in advice["rows"]))
        self.assertIn("group cost unknown", advice["rows"][0]["unavailable_reasons"])

    def test_unresolved_progress_cannot_disappear_from_rate_denominator(self):
        first = opportunity()
        later = opportunity("a-later", sequence=1)
        later["direction_id"] = "a"
        later["progress_complete"] = False
        result = calculate([first, later], [progress()])
        self.assertIsNone(result["rows"][0]["rate"])
        self.assertAlmostEqual(result["rows"][0]["service_count"], 4)
        later["progress_complete"] = True
        no_gain = calculate([first, later], [progress()])
        self.assertLess(no_gain["rows"][0]["rate"], calculate([first], [progress()])["rows"][0]["rate"])

    def test_unknown_charge_preserves_known_consumption_and_cannot_hide_overrun(self):
        item = opportunity()
        item["receipts"].append({"receipt_id": "pending-bill", "costs": {}})
        result = accounting([item])
        self.assertEqual(result["known_delivered"], vector(0.1))
        self.assertEqual(result["remaining_reservation"], vector(0.1))
        self.assertTrue(all(value is None for value in result["delivered"].values()))
        self.assertEqual(result["available"], vector(0.8))
        item["receipts"][0]["costs"] = vector(0.3)
        with self.assertRaisesRegex(SelectionError, "delivery exceeds reservation"):
            accounting([item])

    def test_settlement_without_cost_receipt_is_unknown_not_explicit_zero(self):
        item = opportunity()
        item["receipts"] = []
        result = accounting([item])
        self.assertTrue(all(value is None for value in result["delivered"].values()))
        self.assertEqual(result["remaining_reservation"], vector(0.2))
        self.assertEqual(result["available"], vector(0.8))
        self.assertIsNone(calculate([item])["rows"][0]["score"])
        item["receipts"] = [{"receipt_id": "explicit-zero", "costs": vector(0)}]
        explicit = accounting([item])
        self.assertEqual(explicit["delivered"], vector(0))
        self.assertEqual(explicit["remaining_reservation"], vector(0))
        self.assertEqual(explicit["available"], vector(1))

    def test_no_credit_without_host_verifier_and_exact_binding(self):
        calls = []

        def check(ref, expected):
            calls.append((ref, expected))
            return expected["gain_log_lcb"] == 0.02

        for verifier in (None, lambda ref, expected: 1, lambda ref, expected: False):
            with self.subTest(verifier=verifier), self.assertRaisesRegex(SelectionError, "host verification"):
                advise([offer()], accounting([opportunity()]), [progress()], config=CONFIG, verify_progress=verifier)
        result = advise([offer()], accounting([opportunity()]), [progress()], config=CONFIG, verify_progress=check)
        self.assertEqual(calls[0][0], progress()["evidence_ref"])
        self.assertEqual(calls[0][1]["source_id"], "2" * 64)
        self.assertEqual(calls[0][1]["opportunity_id"], "a")
        self.assertAlmostEqual(result["rows"][0]["recent_gain"], 0.02)

    def test_dag_credit_is_once_per_produced_source_not_per_ancestor(self):
        first, second = opportunity(), opportunity("b", sequence=1)
        one, two = progress(), progress("b", source="3")
        for field in ("credit_id", "source_id", "evidence_ref"):
            duplicate = copy.deepcopy(two)
            duplicate[field] = one[field]
            with self.subTest(field=field), self.assertRaisesRegex(SelectionError, "DAG gain counts once"):
                calculate([first, second], [one, duplicate], [offer(), offer("b")])
        one["parents"] = ["4" * 64, "5" * 64]
        one["donors"] = ["6" * 64]
        result = calculate([first, second], [one], [offer(), offer("b")])
        self.assertGreater(result["rows"][0]["rate"], 0)
        self.assertEqual(result["rows"][1]["rate"], 0)

    def test_credit_requires_settlement_and_delayed_credit_keeps_episode(self):
        with self.assertRaisesRegex(SelectionError, "settled producing opportunity"):
            calculate([opportunity(settled=False)], [progress()])
        item = opportunity()
        earlier = calculate([item])
        later = calculate([item], [progress()])
        self.assertEqual(earlier["rows"][0]["service_count"], later["rows"][0]["service_count"])
        self.assertGreater(later["rows"][0]["score"], earlier["rows"][0]["score"])

    def test_mixed_jobs_and_epochs_are_not_comparable(self):
        for field, bad in (("job_id", "9" * 64), ("measurement_epoch_id", "epoch-2")):
            for kind in ("opportunity", "offer", "progress"):
                records, offers, credits = [opportunity()], [offer()], [progress()]
                {"opportunity": records, "offer": offers, "progress": credits}[kind][0][field] = bad
                with self.subTest(field=field, kind=kind), self.assertRaisesRegex(SelectionError, "mixed job"):
                    calculate(records, credits, offers)
        retained = accounting([opportunity()])
        retained["opportunities"][0]["measurement_epoch_id"] = "old-epoch"
        with self.assertRaisesRegex(SelectionError, "mixed job"):
            advise([offer()], retained, [], config=CONFIG)

    def test_resource_and_deadline_bounds_are_independent(self):
        too_large = opportunity(fraction=1.1)
        with self.assertRaisesRegex(SelectionError, "reservation exceeds"):
            accounting([too_large])
        for records in ([opportunity(fraction=0.6, settled=False), opportunity("b", fraction=0.6, settled=False)],
                        [opportunity(cost=0.3)]):
            with self.assertRaises(SelectionError):
                accounting(records)
        late = opportunity()
        late["review_at"] = 1001
        with self.assertRaisesRegex(SelectionError, "finite campaign-bounded"):
            accounting([late])
        # A plentiful GPU budget cannot offset an exhausted dollar reservation.
        records = [opportunity(settled=False), opportunity("b", settled=False)]
        for record in records:
            record["requested"]["model_usd"] = record["reserved"]["model_usd"] = 60
        with self.assertRaisesRegex(SelectionError, "aggregate commitments"):
            accounting(records)

    def test_duplicate_receipts_cannot_double_spend_or_reset_history(self):
        first, second = opportunity(), opportunity("b", sequence=1)
        second["receipts"][0]["receipt_id"] = first["receipts"][0]["receipt_id"]
        with self.assertRaisesRegex(SelectionError, "duplicate cost receipt"):
            accounting([first, second])
        with self.assertRaisesRegex(SelectionError, "duplicate opportunity"):
            accounting([first, first])

    def test_explicit_parameters_no_silent_weights_or_probability_claim(self):
        for name in CONFIG:
            config = copy.deepcopy(CONFIG)
            del config[name]
            with self.subTest(name=name), self.assertRaisesRegex(SelectionError, "explicit"):
                calculate([], config=config)
        config = copy.deepcopy(CONFIG)
        config["weights"]["model_usd"] = 0
        with self.assertRaisesRegex(SelectionError, "positive resource weights"):
            calculate([], config=config)
        result = calculate([], offers=[offer("a", 0), offer("b", 0)])
        self.assertEqual([row["prior_mass"] for row in result["rows"]], [0.5, 0.5])
        self.assertTrue(result["advisory_only"])
        self.assertNotIn("selected", result)
        self.assertNotIn("parent", result)
        self.assertEqual([row["direction_id"] for row in result["rows"]], ["a", "b"])

    def test_unknown_next_cost_does_not_become_last_or_zero(self):
        unknown = offer()
        unknown["estimated_costs"] = {}
        result = calculate([], offers=[unknown, offer("b")])
        self.assertIsNone(result["rows"][0]["score"])
        self.assertIsNone(result["rows"][0]["estimated_cost"])
        self.assertIsNotNone(result["rows"][1]["score"])

    def test_override_is_agent_choice_without_host_admission(self):
        advice = calculate([opportunity()], [progress()], [offer(), offer("b")], dict(CONFIG, beta=0))
        selection = {"direction_id": "b", "island_id": "island-1", "engineer_id": "engineer-1",
                     "working_source_id": "7" * 64, "donor_source_ids": ["8" * 64]}
        with self.assertRaisesRegex(SelectionError, "override reason"):
            record_decision(advice, selection, reason="Complete the promised discriminating repair.")
        record = record_decision(advice, selection, reason="Complete the promised repair.",
                                 override_reason="The slow tip tests a prerequisite the score cannot value.",
                                 evidence_refs=[progress()["evidence_ref"]])
        self.assertEqual(record["comparison"], "override")
        self.assertEqual(record["authority"], "agent_choice_only")
        self.assertEqual(record["selection"]["working_source_id"], "7" * 64)
        self.assertNotIn("admitted", record)
        selection["direction_id"] = "new-question"
        invented = record_decision(advice, selection, reason="A new question follows from the compiler evidence.")
        self.assertEqual(invented["comparison"], "not_comparable")

    def test_artifact_refs_and_inputs_remain_exact(self):
        records, credits, offers, config = [opportunity()], [progress()], [offer()], copy.deepcopy(CONFIG)
        original = copy.deepcopy((records, credits, offers, config))
        calculate(records, credits, offers, config)
        self.assertEqual((records, credits, offers, config), original)
        for bad in ("../secret", "/absolute", "a/../b", "a//b", "."):
            credits[0]["evidence_ref"]["artifact"] = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(SelectionError, "relative file"):
                calculate(records, credits)

    def test_other_island_opportunity_is_counted_even_if_not_currently_offered(self):
        first, second = opportunity(), opportunity("b", sequence=1)
        second["island_id"] = "island-2"
        result = calculate([first, second], [progress()])
        self.assertAlmostEqual(result["total_service_count"], 4)
        self.assertAlmostEqual(result["rows"][0]["service_count"], 2)


if __name__ == "__main__":
    unittest.main()
