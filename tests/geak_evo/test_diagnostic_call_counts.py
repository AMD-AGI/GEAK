"""Portable declaration tests; no provider source, process or device access."""
import copy
import tempfile
import unittest

from geak_evo.evaluate import jobs
from test_evaluate_contracts import fixture as native_fixture
from test_generated_task_contract import GeneratedFixture, encoded


class DeclaredCallCountsTests(unittest.TestCase):
    def test_schema_counts_leave_contract_and_digest_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = GeneratedFixture(directory).contract
            direct = native_fixture()[0]
            graph = copy.deepcopy(direct)
            graph["timing_policy"]["primary"].update(kind="cuda_event_graph_replay", graph=True)
            for contract, measured in ((direct, 100), (graph, 100), (generated, 50)):
                with self.subTest(schema=contract["schema"], measured=measured):
                    before, identity = encoded(contract), jobs.job_digest(contract)
                    counts = jobs.declared_call_counts(contract)
                    self.assertEqual(counts, {"measured": measured, "warmup": 10})
                    counts["measured"] = 0
                    self.assertEqual(jobs.declared_call_counts(contract)["measured"], measured)
                    self.assertEqual(encoded(contract), before)
                    self.assertEqual(jobs.job_digest(contract), identity)

    def test_native_integer_valued_floats_preserve_existing_acceptance(self):
        for graph in (False, True):
            with self.subTest(graph=graph):
                native = native_fixture()[0]
                native["timing_policy"]["primary"].update(
                    kind="cuda_event_graph_replay" if graph else "cuda_event_direct_calls",
                    graph=graph, warmup_calls=10.0, measured_calls=100.0)
                before, identity = encoded(native), jobs.job_digest(native)
                jobs.validate_arena_job(native)
                counts = jobs.declared_call_counts(native)
                self.assertEqual(counts, {"warmup": 10, "measured": 100})
                self.assertTrue(all(type(count) is int for count in counts.values()))
                self.assertEqual(encoded(native), before)
                self.assertEqual(jobs.job_digest(native), identity)

    def test_existing_validation_rejects_fractional_counts_and_wrong_layout(self):
        native = native_fixture()[0]
        native["timing_policy"]["primary"]["measured_calls"] = 100.5
        with self.assertRaises(ValueError):
            jobs.declared_call_counts(native)
        with tempfile.TemporaryDirectory() as directory:
            generated = GeneratedFixture(directory).contract
            for change in (lambda value: value.update(schema="unknown"),
                           lambda value: value["timing_policy"].update(primary={}),
                           lambda value: value["timing_policy"]["in_process"].update(inner_calls=2),
                           lambda value: value["timing_policy"]["in_process"].update(inner_calls=True),
                           lambda value: value["timing_policy"]["in_process"].update(measured_calls=50.0)):
                changed = copy.deepcopy(generated)
                change(changed)
                with self.assertRaises(ValueError):
                    jobs.declared_call_counts(changed)


if __name__ == "__main__":
    unittest.main()
