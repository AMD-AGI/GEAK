"""A probe can verify TP1 while the all-rank census rejects graph-hidden TP0.

Uses real trace verification, trace scanning, point_census subprocesses and the
probe_server fallback predicate. Only HTTP capture is replaced by writing traces.
No GPU, serving process or census-threshold override is involved.
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import probe_server
import run_points
import serving_probe
from test_profile_fast_path import decode_events, write_trace


class CensusFallbackTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='rank census fallback ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.points = self.root / 'points.json'
        self.points.write_text(json.dumps({'points': [
            {'point_id': 'de_5', 'phase': 'DECODE', 'batch_size': 5,
             'context_tokens': 64, 'represented': {'instances': 10}}]}))
        self.eval = self.root / 'eval'
        self.command = [str(SCRIPTS / 'run_points.py'), '--points', str(self.points),
                        '--base-url', 'http://capture-is-stubbed', '--trace-root', str(self.root / 'traces'),
                        '--expect-files', '2', '--eval-dir', str(self.eval), '--no-evidence']

    def capture_then_census(self, rank0_mode):
        real_run = run_points.run
        saved_probes = {}

        def capture(cmd, log):
            if Path(cmd[1]).name != 'serving_probe.py':
                return real_run(cmd, log)  # real all-rank point_census subprocess
            directory = Path(cmd[cmd.index('--trace-dir') + 1])
            paths = []
            for rank in [1, 0]:
                events = decode_events(rank, 5, steps=8)
                if rank == 0 and rank0_mode == 'hidden':
                    # Real failure: one visible VERIFY GPU anchor versus8 CPU;
                    # DRAFT GPU spans and other kernels are still visible.
                    seen = False
                    visible = []
                    for event in events:
                        if event[0] == 'gpu_user_annotation' and event[1].startswith('step[VERIFY'):
                            if seen:
                                continue
                            seen = True
                        visible.append(event)
                    events = visible
                elif rank == 0 and rank0_mode == 'no_decode':
                    events = [e for e in events if not e[1].startswith('step[VERIFY')]
                elif rank == 0 and rank0_mode == 'rank_count_mismatch':
                    events = decode_events(rank, 5, steps=7)
                # EP-bearing filenames do not hit the TP-0. preference; the probe
                # verifies first-returned TP1, as happened in the actual c28 run.
                path = directory / ('capture-TP-%d-EP-%d.trace.json.gz' % (rank, rank))
                write_trace(str(path), events)
                paths.append({'path': str(path)})
            journal = []
            result = serving_probe.verify_trace(paths, 'DECODE', 'batch_size', 5, journal)
            self.assertTrue(result['verified'], result)
            self.assertEqual(result['decode_anchors'], {'mode': 'VERIFY', 'gpu': 8, 'cpu': 8})
            self.assertIn('TP-1-EP-1', result['file_scanned'])
            output = Path(cmd[cmd.index('--out') + 1])
            output.write_text(json.dumps({'status': 'ok', 'verify_reason': result['reason'],
                                          'journal': journal}))
            saved_probes[output] = output.read_bytes()
            return 0, 0.0, result['reason']

        with patch.object(sys, 'argv', self.command), patch.object(run_points, 'run', side_effect=capture), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = run_points.main()
        timeline = json.loads((self.eval / 'profile_run.json').read_text())
        for path, data in saved_probes.items():
            self.assertEqual(path.read_bytes(), data)  # retain original per-rank probe verdict
        return rc, timeline

    def test_other_rank_census_hidden_failure_reaches_existing_eager_fallback(self):
        rc, timeline = self.capture_then_census('hidden')
        self.assertNotEqual(rc, 0)
        point = timeline['points']['de_5']
        self.assertNotIn('census', point)  # existing fallback skips any `census` key
        failure = point['census_failure']
        self.assertNotEqual(failure['rc'], 0)
        self.assertIsNone(failure['out'])
        self.assertIn('rank 0: fewer than 2 decode anchors', failure['message'])
        attempt = point['attempts'][0]
        self.assertEqual(attempt['status'], 'ok')  # TP1 probe really did pass
        self.assertTrue(attempt['graph_hidden'])
        self.assertEqual(attempt['graph_hidden_source'], 'point_census')
        self.assertEqual(attempt['graph_hidden_reason'], failure['graph_hidden_reason'])
        self.assertEqual(attempt['trace_dir'], failure['trace_dir'])
        self.assertEqual(probe_server.graph_hidden_points(self.command), ['de_5'])
        self.assertEqual(timeline['points_without_census'], ['de_5'])
        self.assertFalse((self.eval / 'evidence/census/de_5.json').exists())

    def test_ordinary_census_failures_do_not_request_graph_fallback(self):
        for mode, reason in [('no_decode', 'The window holds no complete decode step'),
                             ('rank_count_mismatch', 'ranks disagree on the number')]:
            with self.subTest(mode=mode):
                rc, timeline = self.capture_then_census(mode)
                self.assertNotEqual(rc, 0)
                point = timeline['points']['de_5']
                self.assertIn(reason, point['census']['message'])
                self.assertIsNone(point['census']['out'])
                self.assertFalse(point['attempts'][0]['graph_hidden'])
                self.assertNotIn('census_failure', point)
                self.assertEqual(probe_server.graph_hidden_points(self.command), [])

    def test_classifier_requires_explicit_decode_census_rejection(self):
        message = ('point_census: trace rank 0: fewer than 2 decode anchors on the GPU timeline. '
                   'The replayed kernels are not traced.')
        self.assertEqual(run_points.census_graph_hidden_reason('DECODE', 1, message), message)
        self.assertIsNone(run_points.census_graph_hidden_reason('DECODE', 0, message))
        self.assertIsNone(run_points.census_graph_hidden_reason('PREFILL', 1, message))
        self.assertIsNone(run_points.census_graph_hidden_reason(
            'DECODE', 1, 'point_census: ranks disagree under --enable-profile-cuda-graph'))


if __name__ == '__main__':
    unittest.main()
