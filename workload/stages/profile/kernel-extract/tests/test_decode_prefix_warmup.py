"""Decode probes build exact KV prefixes with bounded per-request new token count."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import serving_probe


class DecodePrefixWarmupTests(unittest.TestCase):
    def args(self, skip=False):
        return SimpleNamespace(no_decode_prefix_warmup=skip,
            decode_prefix_chunk_tokens=3, seed=42, request_timeout=1, decode_tokens=100)

    def test_bounded_prefixes_are_exact_and_precede_all_resident_requests(self):
        calls, journal = [], []
        def gen(base, ids, tokens, journal, label, *rest):
            calls.append((label, list(ids), tokens))
            journal.append({'event': 'request_done', 'label': label})
        with patch.object(serving_probe, 'generate', side_effect=gen):
            ex, _ = serving_probe.drive_decode('unused',
                {'point_id': 'de2', 'batch_size': 2, 'context_tokens': 5}, self.args(), journal)
            ex.shutdown(wait=True)
        self.assertEqual([c[0] for c in calls[:4]],
            ['decode_prefix0_3', 'decode_prefix0_5', 'decode_prefix1_3', 'decode_prefix1_5'])
        resident = {label: ids for label, ids, tokens in calls if tokens == 100}
        for i in range(2):
            prior = 0
            for label, ids, tokens in calls[2*i:2*i+2]:
                self.assertEqual(tokens, 1)
                self.assertEqual(ids, resident['decode_req%d' % i][:len(ids)])
                self.assertLessEqual(len(ids) - prior, 3)
                prior = len(ids)
        self.assertTrue(any(e['event'] == 'decode_prefixes_warmed' for e in journal))

    def test_failed_warmup_never_starts_resident_batch(self):
        journal = []
        def fail(base, ids, tokens, journal, label, *rest):
            journal.append({'event': 'request_failed', 'label': label})
        with patch.object(serving_probe, 'generate', side_effect=fail) as generate:
            with self.assertRaisesRegex(RuntimeError, 'warmup failed'):
                serving_probe.drive_decode('unused',
                    {'point_id': 'de2', 'batch_size': 2, 'context_tokens': 5}, self.args(), journal)
        self.assertEqual(generate.call_count, 1)

    def test_explicit_skip_keeps_direct_request_behavior(self):
        labels = []
        def gen(base, ids, tokens, journal, label, *rest):
            labels.append(label)
        with patch.object(serving_probe, 'generate', side_effect=gen):
            ex, _ = serving_probe.drive_decode('unused',
                {'point_id': 'de2', 'batch_size': 2, 'context_tokens': 5}, self.args(True), [])
            ex.shutdown(wait=True)
        self.assertEqual(sorted(labels), ['decode_req0', 'decode_req1'])


if __name__ == '__main__':
    unittest.main()
