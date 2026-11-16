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
        cfg = self.root / 'input'
        (cfg / 'evidence').mkdir(parents=True)
        (cfg / 'evidence/observed_workload.json').write_text(json.dumps(observed))
        (cfg / 'serving.json').write_text(json.dumps({'model': 'test', 'tp': 1, 'gpus': '0'}))
        (cfg / 'runtime_identity.json').write_text(json.dumps({
            'image': 'test', 'model_revision': 'test', 'source_revision': 'test', 'gpu_arch': 'test'}))
        output = self.root / 'profile'
        proc = subprocess.run([sys.executable, str(SCRIPTS.parents[1] / 'multi_config.py'),
                               'plan', '--group-dir', str(output), '--config', 'only=' + str(cfg),
                               '--num-points', '4', '--target-pct', '100'], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        plan = json.loads((output / 'sampling_plan.json').read_text())
        self.assertEqual(plan['capture_count'], 4)
        for phase, mass in [('PREFILL', 2), ('DECODE', 20)]:
            points = [p for p in plan['captures'] if p['point']['phase'] == phase]
            self.assertEqual(sum(p['per_config']['only']['instances'] for p in points), mass)
            self.assertEqual(plan['configs']['only']['phases'][phase]['planned_proxy_coverage_pct'], 100)

    def test_top_n_summary_cannot_be_used_as_a_complete_census(self):
        sys.path.insert(0, str(SCRIPTS))
        from coverage_ledger import load_census
        census = self.root / 'top_n.json'
        census.write_text(json.dumps({'top_kernels': [{'name': 'k', 'total_ms': 1}],
                                     'total_gpu_time_ms': 5, 'num_distinct_kernels': 3}))
        with self.assertRaisesRegex(SystemExit, 'Top-N summaries are not a census'):
            load_census(census, 'DECODE')

    def test_hybrid_swa_dialect_with_cp_rank_prefix(self):
        """sglang 0.5.22 / DeepSeek-V4 (InferenceX run 39258204219): the attention-CP rank
        precedes the TP rank and token usage is reported per KV pool."""
        self.log.write_text(
            '[2026-10-01 00:00:10 ATTN_CP0 TP0] Prefill batch, #new-seq: 1, #new-token: 32, #cached-token: 96, full token usage: 0.00, swa token usage: 0.03, #running-req: 2, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 0.58\n'
            '[2026-10-01 00:00:10 ATTN_CP1 TP1] Prefill batch, #new-seq: 1, #new-token: 32, #cached-token: 96, full token usage: 0.00, swa token usage: 0.03, #running-req: 2, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 0.58\n'
            '[2026-10-01 00:00:13 ATTN_CP0 TP0] Decode batch, #running-req: 2, #full token: 256, full token usage: 0.00, #swa token: 512, swa token usage: 0.06, accept len: 2.00, accept rate: 0.45, cuda graph: True, gen throughput (token/s): 0.29, #queue-req: 0\n')
        proc = self.observe('--decode-log-interval', '10')
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        observed = json.loads((self.root / 'observed.json').read_text())
        self.assertEqual(observed['counts'], {
            'prefill_batches': 1, 'decode_rows_logged': 1, 'decode_iterations': 10})
        self.assertEqual(observed['prefill']['prefix_cache_hit_pct'], 75)

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

    def test_relabel_uses_sent_prefill_context_before_prefix_alignment(self):
        """Request length survives stale context annotations and query-page alignment."""
        census_dir, probe_dir = self.root / 'census', self.root / 'probe'
        census_dir.mkdir()
        probe_dir.mkdir()
        points = []
        journal_bytes = {}
        examples = [
            # A planned context band can be smaller than the no-prefix request.
            ('no_prefix', 16384, 0, 5600, 16384, 16384, True, ''),
            # The prefix's last 13 tokens are re-extended, not added to the prompt.
            ('aligned_prefix', 651, 100349, 101000, 664, 101000, True, '_r2'),
            # Missing request completion still has the exact sent prefill plan.
            ('plan_fallback', 651, 100349, 90000, 664, 101000, False, ''),
        ]
        for pid, new, cached, ctx, achieved, total, completed, suffix in examples:
            points.append(dict(point_id=pid, phase='PREFILL', new_tokens=new,
                               cached_tokens=cached, context_tokens=ctx, running=1))
            (census_dir / (pid + '.json')).write_text(json.dumps(
                dict(achieved_shape=achieved, instances=1)))
            journal = [dict(event='prefill_plan', sequences=1,
                            per_sequence_new=[new], per_sequence_cached=[cached]),
                       dict(event='request_done', label='shape_warm0', prompt_tokens=999)]
            if completed:
                journal.append(dict(event='request_done', label='prefill_seq0',
                                    prompt_tokens=total))
            path = probe_dir / (pid + suffix + '.json')
            path.write_text(json.dumps(dict(status='ok', journal=journal)))
            journal_bytes[path] = path.read_bytes()
        (self.root / 'points.json').write_text(json.dumps(dict(points=points)))
        (self.root / 'observed.json').write_text(json.dumps(dict(raw_prefill_rows=[
            dict(new_token=p['new_tokens'], cached_token=p['cached_tokens'], new_seq=1)
            for p in points])))
        proc = self.run_script('relabel_points.py', '--observed', self.root / 'observed.json',
                               '--points', self.root / 'points.json', '--census-dir', census_dir,
                               '--probe-dir', probe_dir, '--context-tolerance', '1.25',
                               '--out', self.root / 'final.json')
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        result = json.loads((self.root / 'final.json').read_text())
        for actual, example, original in zip(result['points'], examples, points):
            pid, new, cached, ctx, achieved, total, completed, suffix = example
            self.assertEqual(actual['achieved_context'], total)
            self.assertEqual(actual['achieved_shape'], achieved)
            self.assertEqual(actual['planned_shape'], new)
            self.assertEqual(actual['context_tokens'], ctx)
            self.assertEqual(actual['cached_tokens'], cached)
            ev = actual['achieved_context_evidence']
            self.assertEqual(ev['basis'], 'probe_request_prompt_tokens' if completed
                             else 'probe_prefill_plan')
            self.assertEqual(ev['per_sequence_prompt_tokens'], [total])
            self.assertEqual(ev['journal_entries'], [2] if completed else [0])
            self.assertEqual(ev['probe_result'], str(probe_dir / (pid + suffix + '.json')))
        for path, before in journal_bytes.items():
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
