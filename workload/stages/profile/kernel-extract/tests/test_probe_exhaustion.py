"""A finished probe batch cannot become resident later, even with stale metrics."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import serving_probe


class ProbeExhaustionTests(unittest.TestCase):
    def test_finished_batch_fails_without_waiting_or_using_stale_metrics(self):
        journal = []
        with patch.object(serving_probe, "scrape_metrics") as scrape:
            ok, _, _ = serving_probe.wait_for_occupancy(
                "unused", 2, journal, 300, requests_finished=lambda: True)
        self.assertFalse(ok)
        scrape.assert_not_called()
        self.assertEqual(journal[0]["event"], "batch_finished_before_steady_state")
        self.assertEqual(journal[-1]["event"], "steady_state_NOT_confirmed")

    def test_live_batch_still_requires_decode_advance_and_quiet_prefill(self):
        journal = []
        metrics = [
            ({"num_running_reqs": 2, "decode_tokens": 0, "prefill_tokens": 10}, None),
            ({"num_running_reqs": 2, "decode_tokens": 2, "prefill_tokens": 20}, None),
            ({"num_running_reqs": 2, "decode_tokens": 4, "prefill_tokens": 20}, None),
        ]
        with patch.object(serving_probe, "scrape_metrics", side_effect=metrics) as scrape:
            ok, bs, _ = serving_probe.wait_for_occupancy(
                "unused", 2, journal, 300, poll=0, requests_finished=lambda: False)
        self.assertTrue(ok)
        self.assertEqual(bs, 2)
        self.assertEqual(scrape.call_count, 3)
        self.assertEqual(journal[-1]["basis"],
                         "occupancy + decode advancing + prefill quiet")


if __name__ == "__main__":
    unittest.main()
