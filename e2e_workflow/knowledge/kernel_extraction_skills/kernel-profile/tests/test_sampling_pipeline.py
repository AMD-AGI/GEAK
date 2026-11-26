"""Exercise log-derived sampling without a service, GPU or workflow runner."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


class SamplingPipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='agentx sampling ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.log = self.root / 'server.log'
        self.log.write_text(
            '[2026-10-01 00:00:00 TP0] Prefill batch, #new-seq: 1, #new-token: 9999, #cached-token: 0, #running-req: 0, #queue-req: 0\n'
            '[2026-10-01 00:00:10 TP0] Prefill batch, #new-seq: 1, #new-token: 32, #cached-token: 96, #running-req: 2, #queue-req: 0\n'
            '[2026-10-01 00:00:11 TP1] Prefill batch, #new-seq: 1, #new-token: 32, #cached-token: 96, #running-req: 2, #queue-req: 0\n'
            '[2026-10-01 00:00:12 TP0] Prefill batch, #new-seq: 2, #new-token: 256, #cached-token: 768, #running-req: 4, #queue-req: 0\n'
            '[2026-10-01 00:00:13 TP0] Decode batch, #running-req: 2, #token: 256, accept len: 2.0, cuda graph: True, gen throughput (token/s): 100.0, #queue-req: 0\n'
            '[2026-10-01 00:00:14 TP0] Decode batch, #running-req: 4, #token: 1024, accept len: 2.0, cuda graph: True, gen throughput (token/s): 200.0, #queue-req: 0\n')

    def run_script(self, script, *args):
        return subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
                              cwd=self.root, capture_output=True, text=True)

    def observe(self, *args):
        return self.run_script('observed_workload.py', '--log', self.log,
                               '--window-start', '2026-10-01 00:00:10',
                               '--window-end', '2026-10-01 00:00:20',
                               '--completed-output-tokens', '120',
                               '--out', self.root / 'observed.json', *args)

    def test_sampling_preserves_measured_rank_phase_mass_and_cache_hits(self):
        proc = self.observe('--decode-log-interval', '10')
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        observed = json.loads((self.root / 'observed.json').read_text())
        self.assertEqual(observed['counts'], {
            'prefill_batches': 2, 'decode_rows_logged': 2, 'decode_iterations': 20})
        self.assertEqual(observed['prefill']['total_new_tokens'], 288)
        self.assertEqual(observed['prefill']['prefix_cache_hit_pct'], 75)
        self.assertEqual(observed['decode']['crosscheck']['disagreement_pct'], 0)
        proc = self.run_script('plan_sampling_points.py', '--observed', self.root / 'observed.json',
                               '--num-points', '4', '--out', self.root / 'points.json')
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        plan = json.loads((self.root / 'points.json').read_text())
        self.assertEqual(plan['phase_budget'], {'PREFILL': 2, 'DECODE': 2})
        for phase, mass in [('PREFILL', 2), ('DECODE', 20)]:
            points = [p for p in plan['points'] if p['phase'] == phase]
            self.assertTrue(points)
            self.assertEqual(sum(p['represented']['instances'] for p in points), mass)
            self.assertAlmostEqual(sum(p['represented']['phase_proxy_mass_share_pct']
                                       for p in points), 100, places=2)

    def test_missing_log_interval_cannot_guess_decode_mass(self):
        proc = self.observe()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('decode_log_interval is unknown', proc.stderr)
        self.assertFalse((self.root / 'observed.json').exists())

    def test_unknown_scheduler_dialect_cannot_become_an_empty_workload(self):
        self.log.write_text('[2026-10-01 00:00:12 TP0] Decode batch, incompatible fields\n')
        proc = self.observe('--decode-log-interval', '10')
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('no registered log dialect', proc.stderr)
        self.assertFalse((self.root / 'observed.json').exists())


if __name__ == '__main__':
    unittest.main()


class AtomLogTest(unittest.TestCase):
    """ATOM logs one line per prefill forward and only a periodic throughput row for decode."""

    def test_atom_rows_become_prefill_batches_and_decode_iterations(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'server.log'
            log.write_text(
                '[atom 10:00:01] Scheduled prefill batch: 2 reqs, 16384 new tokens (done: [32768, 0], new: [8192, 8192]), req_ids: (0, 1)\n'
                '[atom 10:00:10] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 200.0 tokens/s, Running: 4 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.0%, Prefix cache hit rate: 0.0%\n'
                '[atom 10:00:20] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%\n')
            out = Path(tmp) / 'ow.json'
            r = subprocess.run([sys.executable, str(SCRIPTS / 'observed_workload.py'), '--log-format', 'atom',
                                '--log', str(log), '--completed-output-tokens', '2000', '--out', str(out)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            d = json.loads(out.read_text())
            self.assertEqual(d['counts']['prefill_batches'], 1)
            self.assertEqual(d['raw_prefill_rows'][0]['cached_token'], 32768)
            # 200 tok/s x 10 s / 4 running = 500 iterations at bs 4; the idle row is not decode
            self.assertEqual([(b['batch_size'], b['iterations']) for b in d['decode']['buckets']], [(4, 500.0)])
            self.assertEqual(d['decode']['crosscheck']['disagreement_pct'], 0.0)
            self.assertIsNone(d['prefill']['prefix_cache_hit_pct'])
