"""Witness search must save work without accepting weak or non-numerical failures."""
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_tensor_lifecycle as lifecycle

D, U = lifecycle.D, lifecycle.U


class NegativeControlSearchTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / 'kernel_src/ops.py'
        self.source.parent.mkdir()
        self.pristine = 'alpha = 1\nbeta = 0\n'
        self.source.write_text(self.pristine)
        self.controls = [{'leaf': 'kernel_src/ops.py', 'find': 'alpha = 1',
                          'replace': 'alpha = 2', 'expect': 'wrong alpha'}]
        cases = [{'sig': chr(97+i), 'random': True, 'm': i+1} for i in range(6)]
        self.task = SimpleNamespace(dir=str(self.root), sig_filter=None, meta={},
            doc={'outputs': {'out': {'bound': {'kind': 'dot_product'}}}},
            manifest={'cases': cases}, U=U,
            mc={'negative_controls': self.controls, 'oracle_batch_cases': 2},
            h=SimpleNamespace(build_candidate_overlay=lambda *a: ('base', 'candidate')))
        self.task.random_specs = lambda: [c for c in cases if self.task.sig_filter is None
                                          or c['sig'] in self.task.sig_filter]
        self.decisions = {}
        self.runs, self.closed = [], []
        self.crash = set()
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(D, 'oracle_output_bytes', side_effect=lambda T,s,*a:s['m']))
        self.patches.enter_context(patch.object(D, '_oracles', side_effect=self.oracle))
        self.patches.enter_context(patch.object(D, 'compare_blobs', side_effect=self.compare))
        self.patches.enter_context(patch.object(D, 'oracle_workers', side_effect=self.workers))

    def control_name(self):
        return 'alpha' if 'alpha = 2' in self.source.read_text() else 'beta'

    @contextmanager
    def workers(self, *args):
        try:
            yield
        finally:
            self.closed.append(self.control_name())
            self.assertNotEqual(self.source.read_text(), self.pristine)

    def oracle(self, T, state, jobs):
        sigs = tuple(c['sig'] for c in T.random_specs())
        self.runs.append((self.control_name(), sigs))
        if self.crash.intersection(sigs):
            raise RuntimeError('fixture launch error')
        return [{}, {}]

    def compare(self, T, *args, **kwargs):
        results = []
        choices = self.decisions.get(self.control_name(), {})
        for spec in T.random_specs():
            value = choices.get(spec['sig'], 0)
            out = ({'ok': False, 'kind': 'dot_product', 'why': 'shape mismatch'} if value == 'shape'
                   else {'ok': value == 0, 'kind': 'dot_product', 'peak_bound_multiple': value})
            results.append({'case': spec['sig']+'|0', 'ok': out['ok'], 'out': out})
        return all(r['ok'] for r in results), results

    def run_gate(self):
        result = D.gate_negative_controls(self.task, SimpleNamespace(seed=0), {'scratch': str(self.root)})
        self.assertEqual(self.source.read_text(), self.pristine)
        return result

    def test_stops_each_control_independently_at_its_witness(self):
        self.controls.append({'leaf': 'kernel_src/ops.py', 'find': 'beta = 0',
                              'replace': 'beta = 1', 'expect': 'wrong beta'})
        self.decisions = {'alpha': {'a': 20}, 'beta': {'a': 10}}
        result = self.run_gate()
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(self.runs, [('alpha', ('a',)), ('beta', ('a',))])
        self.assertEqual(self.closed, ['alpha', 'beta'])
        for row in result['controls']:
            self.assertTrue(row['early_stopped'] and row['restored'])
            self.assertEqual((row['checked_cases'], row['available_cases'], row['witness_case']),
                             (1, 6, 'a|0'))
        self.assertIsNone(self.task.sig_filter)

    def test_weak_rejection_continues_until_margin_is_reached(self):
        self.decisions = {'alpha': {'a': 2, 'c': 20}}
        row = self.run_gate()['controls'][0]
        self.assertTrue(row['ok'])
        self.assertEqual(self.runs, [('alpha', ('a',)), ('alpha', ('b', 'c'))])
        self.assertEqual((row['checked_cases'], row['witness_case']), (3, 'c|0'))

    def test_ineffective_mutation_checks_every_case_and_fails(self):
        row = self.run_gate()['controls'][0]
        self.assertFalse(row['ok'] or row['caught'] or row['early_stopped'])
        self.assertEqual(row['checked_cases'], 6)
        self.assertEqual([s for _, batch in self.runs for s in batch], list('abcdef'))

    def test_shape_errors_are_not_numerical_witnesses(self):
        self.decisions = {'alpha': {s: 'shape' for s in 'abcdef'}}
        row = self.run_gate()['controls'][0]
        self.assertFalse(row['ok'] or row['caught'])
        self.assertEqual(row['checked_cases'], 6)
        self.assertIsNone(row['witness_case'])

    def test_execution_error_is_not_success_and_restores_source(self):
        self.decisions = {'alpha': {'a': 2}}
        self.crash = {'b'}
        row = self.run_gate()['controls'][0]
        self.assertFalse(row['ok'] or row['early_stopped'])
        self.assertTrue(row['restored'])
        self.assertIn('not a numerical control', row['why'])

    def test_probe_hint_orders_search_without_narrowing_fallback(self):
        self.controls[0]['probe_sigs'] = ['f']
        row = self.run_gate()['controls'][0]
        self.assertEqual(self.runs[0], ('alpha', ('f',)))
        self.assertEqual(row['checked_cases'], 6)
        self.assertFalse(row['ok'])

    def test_quick_scope_does_not_expand_to_a_hint_outside_its_subset(self):
        self.controls[0]['probe_sigs'] = ['f']
        self.task.sig_filter = {'b', 'c'}
        row = self.run_gate()['controls'][0]
        self.assertEqual(row['available_cases'], 2)
        self.assertEqual([s for _, batch in self.runs for s in batch], ['b', 'c'])
        self.assertEqual(self.task.sig_filter, {'b', 'c'})

    def test_bad_probe_hints_fail_before_executing_a_mutation(self):
        for probes in [['missing'], ['a', 'a'], 'a']:
            self.controls[0]['probe_sigs'] = probes
            row = self.run_gate()['controls'][0]
            self.assertFalse(row['ok'])
            self.assertIn('probe_sigs', row['why'])
        self.assertEqual(self.runs, [])


@unittest.skipIf(lifecycle.torch is None, 'requires CPU torch')
class RealNegativeControlTests(unittest.TestCase):
    def setUp(self):
        self.f = lifecycle.fixtures.EndToEndTest()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def run_gate(self, replacement=None):
        meta = json.loads(Path(self.f.meta_path).read_text())
        if replacement is not None:
            meta['ut_contract']['negative_controls'][0]['replace'] = replacement
        self.f.write_meta(meta)
        source = Path(self.f.task) / 'kernel_src/ops.py'
        before = source.read_bytes()
        proc, report = self.f.ut('--validation-mode', 'correctness', '--gates', 'negative_controls')
        self.assertEqual(source.read_bytes(), before)
        return proc, report['gates']['negative_controls']

    def test_real_mutation_stops_after_one_numerical_witness(self):
        proc, gate = self.run_gate()
        self.assertEqual(gate['status'], 'pass', proc.stdout+proc.stderr)
        row = gate['controls'][0]
        self.assertEqual((row['checked_cases'], row['available_cases']), (1, 2))
        self.assertTrue(row['early_stopped'] and row['restored'])
        self.assertGreaterEqual(row['margin'], 10)

    def test_real_large_shape_only_mutation_is_found_after_probe_misses(self):
        proc, gate = self.run_gate('(a[:, :-16].float() @ b[:, :-16].float().T '
                                   'if a.shape[0] > 10 else a.float() @ b.float().T)')
        self.assertEqual(gate['status'], 'pass', proc.stdout+proc.stderr)
        row = gate['controls'][0]
        self.assertEqual(row['checked_cases'], 2)
        self.assertFalse(row['early_stopped'])
        self.assertIn('PREFILL', row['witness_case'])


if __name__ == '__main__':
    unittest.main()
