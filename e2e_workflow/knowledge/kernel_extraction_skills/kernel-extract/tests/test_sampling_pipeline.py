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


class BenchLogWindowTest(unittest.TestCase):
    """The canonical aiperf client log delimits the measured leg; parse it, don't ask."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='agentx benchwindow ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.log = self.root / 'server.log'
        self.log.write_text(''.join(
            '[2026-09-14 %s TP0] Decode batch, #running-req: 2, #token: 256, accept len: 2.0, '
            'cuda graph: True, gen throughput (token/s): 100.0, #queue-req: 0\n' % t
            for t in ('01:10:00', '01:21:55', '02:00:00', '02:30:00')))

    def bench(self, lines):
        path = self.root / 'benchmark.log'
        path.write_text(''.join(lines))
        return path

    def observe(self, bench):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / 'observed_workload.py'), '--log', str(self.log),
             '--decode-log-interval', '1', '--bench-log', str(bench),
             '--out', str(self.root / 'observed.json')],
            cwd=self.root, capture_output=True, text=True)

    def test_capitalised_parenthesised_time_only_phase_lines_are_read(self):
        # aiperf's real spelling: 'Phase profiling (profiling) started', time of day only.
        proc = self.observe(self.bench([
            '01:18:17.172 NOTICE   Phase warmup (warmup) started | target: 109 requests\n',
            '01:21:54.543 NOTICE   Phase profiling (profiling) started | phase_index=0\n',
            '02:21:54.544 NOTICE   Phase profiling (profiling) sending complete | sent=1,456\n',
            '02:22:24.544 NOTICE   Phase profiling (profiling) complete | completed=1,456\n']))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        window = json.loads((self.root / 'observed.json').read_text())['measured_window']
        self.assertEqual(window['start'], '2026-09-14T01:21:54.543000')
        # 'sending complete' fires 30 s early and must not be mistaken for the phase end.
        self.assertEqual(window['end'], '2026-09-14T02:22:24.544000')
        self.assertAlmostEqual(window['duration_s'], 3630.001, places=3)

    def test_a_leg_crossing_midnight_rolls_the_end_forward(self):
        self.log.write_text(
            '[2026-09-14 23:45:00 TP0] Decode batch, #running-req: 2, #token: 256, accept len: 2.0, '
            'cuda graph: True, gen throughput (token/s): 100.0, #queue-req: 0\n')
        proc = self.observe(self.bench([
            '23:30:00.000 NOTICE   Phase profiling (profiling) started | x\n',
            '00:30:00.000 NOTICE   Phase profiling (profiling) complete | x\n']))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        window = json.loads((self.root / 'observed.json').read_text())['measured_window']
        self.assertEqual(window['duration_s'], 3600.0)
        self.assertEqual(window['end'][:10], '2026-09-15')

    def test_dated_phase_lines_still_work(self):
        proc = self.observe(self.bench([
            '2026-09-14 01:21:54 NOTICE phase "profiling" started\n',
            '2026-09-14 02:22:24 NOTICE phase "profiling" complete\n']))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        window = json.loads((self.root / 'observed.json').read_text())['measured_window']
        self.assertEqual(window['start'], '2026-09-14T01:21:54')

    def test_an_ambiguous_date_is_refused_rather_than_guessed(self):
        self.log.write_text(self.log.read_text() +
            '[2026-09-15 01:21:55 TP0] Decode batch, #running-req: 2, #token: 256, accept len: 2.0, '
            'cuda graph: True, gen throughput (token/s): 100.0, #queue-req: 0\n')
        proc = self.observe(self.bench([
            '01:21:54.543 NOTICE   Phase profiling (profiling) started | x\n',
            '02:22:24.544 NOTICE   Phase profiling (profiling) complete | x\n']))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('ambiguous', proc.stdout + proc.stderr)

    def test_a_missing_phase_still_reports_how_to_proceed(self):
        proc = self.observe(self.bench(['01:21:54.543 NOTICE   nothing to see here\n']))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('--window-start', proc.stdout + proc.stderr)
