"""Scheduling, isolation, lazy operands, timing-mode resume, rank_chain (mono_layer) and
--quick-timing subset regressions (CPU)."""
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


class RankChainTimingTest(unittest.TestCase):
    """mono_layer (meta.timing.kind rank_chain): ONE leg per side per rep times every undecided
    bucket on the whole device set; the first side alternates, both sides of a rep share one
    shuffled bucket order, and the early exit stays per bucket."""
    META = {'timing': {'kind': 'rank_chain', 'world': 8}}

    def run_chain(self, times, sigs=None, **kwargs):
        sigs = list(sigs or times)
        legs = []
        def leg(task, overlay, mode, **kw):
            if mode == 'list':
                return {'sigs': sigs}
            self.assertNotIn('bucket', kw)
            legs.append(('B' if overlay == 'base' else 'C', list(kw['sigs']), kw.get('env_extra')))
            out = []
            for s in kw['sigs']:
                if times.get(s) is None:
                    continue
                ms = times[s][0 if overlay == 'base' else 1]
                out.append({'sig': s, 'ms': ms() if callable(ms) else ms, 'regime': 'decode'})
            return {'cases': out}
        with patch.object(H, 'build_candidate_overlay', return_value=('base', 'cand')), \
             patch.object(H, 'assert_legs_differ'), patch.object(H, '_run_leg', side_effect=leg):
            rows = H.measure_legs('/fixture', self.META, gpus=list(range(8)), **kwargs)
        return rows, legs

    def test_a_decided_bucket_leaves_the_next_reps_legs(self):
        seen = []
        rows, legs = self.run_chain({'d': (2.0, 1.0), 'p': (4.0, 4.0)}, max_reps=3, on_bucket=seen.append)
        self.assertEqual(len(legs), 6)
        self.assertEqual([sorted(s) for _, s, _ in legs[:2]], [['d', 'p']] * 2)
        self.assertEqual([s for _, s, _ in legs[2:]], [['p']] * 4)
        self.assertTrue(all(e['HIP_VISIBLE_DEVICES'] == '0,1,2,3,4,5,6,7' for _, _, e in legs))
        by = {r['sig']: r for r in rows}
        self.assertEqual((by['d']['reps'], by['p']['reps']), (1, 3))
        self.assertEqual(by['d']['speedup'], 2.0)
        self.assertEqual([r['sig'] for r in seen], ['d', 'p'])
        self.assertTrue(all(r['process_mode'] == 'fresh_rank_chain' for r in rows))

    def test_the_first_side_alternates_and_both_sides_share_a_reps_order(self):
        sigs = 'abcde'
        _, legs = self.run_chain({s: (1.0, 1.0) for s in sigs}, max_reps=3)
        self.assertEqual([side for side, _, _ in legs], ['B', 'C', 'C', 'B', 'B', 'C'])
        for i in range(0, 6, 2):
            self.assertEqual(legs[i][1], legs[i + 1][1])
            self.assertEqual(sorted(legs[i][1]), list(sigs))
        self.assertGreater(len({tuple(s) for _, s, _ in legs}), 1)      # reshuffled across reps

    def test_the_bucket_median_is_over_pairs_of_fresh_legs(self):
        it = iter([1.0, 1.02, 9.0])           # undecided after two pairs, then one slow process
        rows, legs = self.run_chain({'d': (lambda: next(it), 1.0)}, max_reps=3)
        self.assertEqual((rows[0]['reps'], rows[0]['baseline_ms'], len(legs)), (3, 1.02, 6))

    def test_an_untimable_bucket_does_not_stop_the_others(self):
        rows, legs = self.run_chain({'d': None, 'p': (4.0, 4.0)}, max_reps=3)
        self.assertEqual([r['sig'] for r in rows], ['p'])
        self.assertEqual([s for _, s, _ in legs[2:]], [['p']] * 4)

    def test_resumed_and_quick_timing_filtered_buckets_are_not_timed(self):
        prev = {'d': {'sig': 'd', 'speedup': 2.0, 'process_mode': 'fresh_rank_chain'}}
        rows, legs = self.run_chain({'d': (2.0, 1.0), 'p': (8.0, 4.0), 'x': (1.0, 1.0)},
                                    done=prev, only={'d', 'p'})
        self.assertEqual([r['sig'] for r in rows], ['d', 'p'])
        self.assertEqual({tuple(s) for _, s, _ in legs}, {('p',)})

    def test_too_few_devices_is_incomplete_not_a_partial_measurement(self):
        with patch.object(H, 'build_candidate_overlay', return_value=('base', 'cand')), \
             patch.object(H, 'assert_legs_differ'), \
             patch.object(H, '_run_leg', return_value={'sigs': ['d']}), \
             self.assertRaises(H.HarnessIncompleteError):
            H.measure_legs('/fixture', self.META, gpus=[0, 1])


class QuickTimingSubsetTest(unittest.TestCase):
    """ut_driver --quick-timing measures a subset of an observed workload's buckets: the weighting
    accepts it only with subset=True, says what it left out, and stays strict otherwise."""
    META = {'workload': {'kind': 'agentx_trace_replay',
                         'serving_weight_model': {'source': 'observed_workload', 'evidence_path': 'x'},
                         'cases': [{'sig': 'small', 'regime': 'decode', 'effective_calls': 100},
                                   {'sig': 'big', 'regime': 'decode', 'effective_calls': 900}]}}
    ROWS = [{'sig': 'small', 'regime': 'decode', 'baseline_ms': 1.0, 'optimized_ms': 0.5},
            {'sig': 'big', 'regime': 'decode', 'baseline_ms': 1.0, 'optimized_ms': 1.25}]

    def test_strict_refuses_a_subset_and_subset_mode_weights_what_was_measured(self):
        part = json.loads(json.dumps(self.ROWS[1:]))
        with self.assertRaises(H.HarnessIncompleteError):
            H.serving_weighted_speedup(json.loads(json.dumps(part)), self.META)
        r = H.serving_weighted_speedup(part, self.META, subset=True)
        self.assertTrue(r['subset'])
        self.assertEqual(r['omitted_positive_mass'], ['small'])
        self.assertAlmostEqual(r['call_mass_coverage'], 0.9)
        self.assertAlmostEqual(r['weighted'], 0.8)

    def test_a_complete_subset_run_equals_the_strict_figure(self):
        full = H.serving_weighted_speedup(json.loads(json.dumps(self.ROWS)), self.META)
        sub = H.serving_weighted_speedup(json.loads(json.dumps(self.ROWS)), self.META, subset=True)
        self.assertAlmostEqual(full['weighted'], sub['weighted'])
        self.assertEqual(sub['omitted_positive_mass'], [])
        self.assertNotIn('subset', full)

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
