"""Scheduling, isolation, lazy operands and timing-mode resume regressions (CPU)."""
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SKILL = Path(__file__).resolve().parents[1] / 'kernel-ut'
spec = importlib.util.spec_from_file_location('batch_harness', SKILL / 'shared/harness_lib.py')
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


class SweepTest(unittest.TestCase):
    def run_measure(self, runner, **kwargs):
        with patch.object(H, 'build_candidate_overlay', return_value=('base', 'cand')), \
             patch.object(H, 'assert_legs_differ'), patch.object(H, '_run_leg', side_effect=runner):
            return H.measure_legs('/fixture', {}, **kwargs)

    def test_two_processes_per_round_order_pairing_early_exit_and_resume(self):
        calls, emitted = [], []
        def leg(task, overlay, mode, **kw):
            if mode == 'list':
                return {'sigs': ['a', 'b', 'done']}
            sigs = kw['sigs']; calls.append((overlay, list(sigs)))
            return {'cases': [{'sig': s, 'ms': 2 if s == 'a' and overlay == 'base' else 1}
                              for s in sigs], 'process': {'pid': len(calls)}}
        rows = self.run_measure(leg, max_reps=3, on_bucket=emitted.append,
                                done={'done': {'sig': 'done', 'process_mode': 'batched'}})
        self.assertEqual(calls, [('base', ['a', 'b']), ('cand', ['a', 'b']),
                                 ('cand', ['b']), ('base', ['b']), ('base', ['b']), ('cand', ['b'])])
        self.assertEqual([r['sig'] for r in rows], ['a', 'b', 'done'])
        self.assertEqual([r['reps'] for r in rows[:2]], [1, 3])
        self.assertEqual(rows[1]['pair_order'], ['BC', 'CB', 'BC'])
        self.assertEqual([r['sig'] for r in emitted], ['a', 'b'])
        self.assertEqual(rows[0]['speedup'], 2)

    def test_missing_duplicate_and_invalid_measurements_fail_closed(self):
        for cases in [[], [{'sig': 'a', 'ms': 1}] * 2, [{'sig': 'a', 'ms': None}],
                      [{'sig': 'a', 'ms': float('nan')}]]:
            with self.subTest(cases=cases):
                def leg(task, overlay, mode, **kw):
                    return {'sigs': ['a']} if mode == 'list' else {'cases': cases}
                with self.assertRaises(RuntimeError):
                    self.run_measure(leg)

    def test_mode_change_cannot_reuse_a_completed_bucket(self):
        seen = []
        def leg(task, overlay, mode, **kw):
            if mode == 'list': return {'sigs': ['a']}
            seen.append(overlay)
            return {'cases': [{'sig': 'a', 'ms': 1}]}
        rows = self.run_measure(leg, max_reps=1, done={'a': {'sig': 'a', 'process_mode': 'fresh'}})
        self.assertEqual(seen, ['base', 'cand'])
        self.assertEqual(rows[0]['process_mode'], 'batched')

    def test_gpu_shard_exception_reaches_caller(self):
        def leg(task, overlay, mode, **kw):
            if mode == 'list': return {'sigs': ['a', 'b']}
            if kw['env_extra']['HIP_VISIBLE_DEVICES'] == '1': raise RuntimeError('worker failed')
            return {'cases': [{'sig': s, 'ms': 1} for s in kw.get('sigs', [kw.get('bucket')])]}
        for mode in ['batched', 'fresh']:
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, 'worker failed'):
                self.run_measure(leg, gpus=[0, 1], max_reps=1, process_mode=mode)

    def test_driver_reuses_correctness_but_not_timing_from_another_process_mode(self):
        spec = importlib.util.spec_from_file_location('batch_driver', SKILL / 'ut_contracts/ut_driver.py')
        driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'; run.mkdir()
            (run / 'ut_report.json').write_text(json.dumps({'reuse_key': 'same', 'gates': {
                'correctness': {'status': 'pass'}, 'timing': {'status': 'pass', 'process_mode': 'fresh'}}}))
            args = SimpleNamespace(reuse=True, quick=False)
            self.assertEqual(set(driver._reusable(SimpleNamespace(dir=tmp), args, 'same')), {'correctness'})
            (run / 'ut_report.json').write_text(json.dumps({'reuse_key': 'same', 'gates': {
                'correctness': {'status': 'pass'}, 'timing': {'status': 'pass', 'process_mode': 'batched'}}}))
            self.assertEqual(set(driver._reusable(SimpleNamespace(dir=tmp), args, 'same')),
                             {'correctness', 'timing'})

    def test_timer_exception_is_preserved_and_fresh_mode_cannot_silently_drop_it(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        def broken(): raise RuntimeError('missing runtime dependency')
        with patch.object(H, '_torch', return_value=torch):
            result = H.time_op(broken, detail=True)
        self.assertIsNone(result['ms'])
        self.assertIn('missing runtime dependency', result['error'])
        def leg(*args, **kw): return {'cases': [dict(result, sig='a')]}
        with patch.object(H, '_run_leg', side_effect=leg), self.assertRaisesRegex(RuntimeError, 'missing runtime dependency'):
            H._measure_bucket('/fixture', 'base', 'cand', 'a', 5, 1, (.95, 1.1), {})


class PerPhaseTest(unittest.TestCase):
    def test_a_decode_gain_does_not_hide_a_prefill_loss(self):
        per = [{'sig': 'd', 'regime': 'decode', 'm': 1, 'baseline_ms': 3.0, 'optimized_ms': 1.0},
               {'sig': 'p', 'regime': 'prefill', 'm': 64, 'baseline_ms': 1.0, 'optimized_ms': 2.0}]
        w = H.serving_weighted_speedup(per, {})
        self.assertGreater(w['weighted'], 1)
        self.assertEqual(w['per_phase']['decode'], {'weighted': 3.0, 'cases': 1})
        self.assertEqual(w['per_phase']['prefill'], {'weighted': 0.5, 'cases': 1})


dspec = importlib.util.spec_from_file_location('timing_driver', SKILL / 'ut_contracts/ut_driver.py')
D = importlib.util.module_from_spec(dspec)
dspec.loader.exec_module(D)


def receipt(primed=True, timers=('cuda_event_graph',)):
    return {'primed': primed, 'timers': list(timers)}


class DriverTimingTest(unittest.TestCase):
    def task(self, cases, regimes):
        return SimpleNamespace(h=H, manifest={'cases': cases}, phase_regimes=lambda: regimes,
                               dir='/fixture', meta={}, gpus=None)

    def test_case_launch_overrides_the_phase_graph_mode(self):
        T = self.task([{'sig': 'big', 'launch': {'mode': 'eager'}}, {'sig': 'small'}],
                      {'DECODE': {'cuda_graph': True}})
        eager = {'baseline_timing': receipt(timers=('cuda_event',)),
                 'optimized_timing': receipt(timers=('cuda_event',))}
        self.assertEqual(D.timing_basis_defects(T, [dict(eager, sig='big', regime='DECODE')]), [])
        (bad, _) = D.timing_basis_defects(T, [dict(eager, sig='small', regime='DECODE')])
        self.assertIn('deploys under a CUDA graph', bad)

    def test_important_cases_skip_the_reference_unless_a_bound_reads_it(self):
        specs = [{'sig': 'pt', 'axes': {}}, {'sig': 'imp', 'kind': 'important', 'axes': {}}]
        U = SimpleNamespace(BOUNDS_NEEDING_REFERENCE=('dot_product',))
        T = SimpleNamespace(random_specs=lambda: specs, U=U,
                            doc={'outputs': {'out': {'bound': {'kind': 'sqnr'}}}})
        self.assertEqual(D.oracle_case_sigs(T, {'cases': 'all'}), ['pt'])
        T.doc = {'outputs': {'out': {'bound': {'kind': 'dot_product'}}}}
        self.assertFalse(D._important(T, specs[1]))

    def test_host_bound_buckets_are_measured_again_before_refusal(self):
        T = self.task([{'sig': 'a'}, {'sig': 'b'}], {'DECODE': {'cuda_graph': True}})
        calls = []

        def measure(task, meta, done=None, **kw):
            calls.append(sorted(done or {}))
            primed = len(calls) > 1
            row = lambda s, p: {'sig': s, 'regime': 'decode', 'm': 1, 'reps': 1, 'baseline_ms': 2.0,
                                'optimized_ms': 1.0, 'process_mode': 'batched',
                                'baseline_timing': receipt(p), 'optimized_timing': receipt()}
            rows = dict(done or {})
            rows.update({s: row(s, primed or s == 'a') for s in 'ab' if s not in rows})
            return [rows[s] for s in 'ab']
        a = SimpleNamespace(max_reps=1, reuse=False, quick=False, out=None)
        with patch.object(H, 'measure_legs', side_effect=measure):
            out = D.gate_timing(T, a, {})
        self.assertEqual(calls, [[], ['a']])          # only the host-bound bucket again
        self.assertEqual((out['status'], out['host_bound_retried']), ('pass', ['b']))
        self.assertEqual(out['per_phase']['decode']['weighted'], 2.0)


class NativeLegTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.task = self.root / 'task'; self.task.mkdir()
        self.base = self.root / 'base'; self.base.mkdir()
        self.cand = self.root / 'cand'; self.cand.mkdir()
        for directory, value in [(self.base, 1), (self.cand, 2)]:
            (directory / 'live_impl.py').write_text('def entry(): return %d\n' % value)
        shutil.copy2(SKILL / 'shared/leg_runner.py', self.task / 'leg_runner.py')
        (self.task / 'meta.json').write_text(json.dumps({'target_callable': 'live_impl:entry'}))
        (self.task / 'harness_lib.py').write_text('''from types import SimpleNamespace
def _torch(): return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
def sync(torch): pass
def deployment_compile_mode(r): return False
def deployment_graph_mode(r): return False
def time_op(call, **kw): return {'ms':call(), 'wall_ms':1, 'timer':'wall'}
''')
        (self.task / 'cases.py').write_text('''import live_impl
from pathlib import Path
live = 0
class Operand:
    def __init__(self):
        global live
        assert live == 0, 'previous shape operands retained'
        live += 1
        Path(__file__).with_name('allocated').touch()
    def __del__(self):
        global live
        live -= 1
def timing_specs(h, meta): return [{'sig':s} for s in ['a','b','c']]
def make_timing_case(s): return dict(s,args=Operand())
def call(args): return live_impl.entry()
''')

    def test_native_legs_keep_side_identity_reuse_process_and_stream_one_operand(self):
        listing = H._run_leg(str(self.task), str(self.base), 'list')
        self.assertEqual(listing['sigs'], ['a', 'b', 'c'])
        self.assertFalse((self.task / 'allocated').exists())
        b = H._run_leg(str(self.task), str(self.base), 'time', sigs=['c', 'a'])
        c = H._run_leg(str(self.task), str(self.cand), 'time', sigs=['c', 'a'])
        self.assertEqual([r['sig'] for r in b['cases']], ['c', 'a'])
        self.assertEqual([r['ms'] for r in b['cases']], [1, 1])
        self.assertEqual([r['ms'] for r in c['cases']], [2, 2])
        self.assertNotEqual(b['process']['pid'], c['process']['pid'])
        self.assertTrue(all(r['diagnostics']['inputs_s'] >= 0 for r in b['cases']))

    def test_process_diagnostics_must_not_make_identical_bindings_look_different(self):
        b = H._run_leg(str(self.task), str(self.base), 'resolve')
        c = H._run_leg(str(self.task), str(self.base), 'resolve')
        self.assertEqual(b, c)
        with self.assertRaisesRegex(RuntimeError, 'SAME code'):
            H.assert_legs_differ(str(self.task), str(self.base), str(self.base),
                                {'target_callable': 'live_impl:entry'})


if __name__ == '__main__':
    unittest.main()
