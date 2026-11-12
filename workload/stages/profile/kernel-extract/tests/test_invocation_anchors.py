"""CPU regressions for op calls versus mandatory/shared leaf launches.

The compact MoE fixture retains real c10/c16 per-forward counts from Muse Spark
CI40714803265: native gate_up=2/down=2 in PREFILL and DECODE, while the shared
finalize identity is62 in PREFILL (60 A4W4 +2 native) and2 in DECODE. Timings below
are deliberately synthetic, demonstrating how that count error can flip a gate.
"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SKILL / 'scripts'), str(SKILL / 'shared')]
import build_task_plan as B
import harness_lib as H
import plan_ut_cases as P

UP = 'muse_moe_gate_up_mfma_flydsl'
DOWN = 'muse_moe_down_mfma_flydsl'
FINAL = 'muse_moe_finalize_vector_flydsl'
WHY = ('muse_moe_flydsl.py:295-368 invokes one gate_up _launch_gemm on each split_k branch; '
       'muse_moe_gemm.py:211 retains the gate_up launch identity. Exactly once per native call.')


def task(anchor=True):
    out = {'label': 'native_moe', 'kernels': [UP, DOWN],
           'companions': [{'kernel': FINAL, 'why': 'Mandatory shared finalizer, including other operator invocations.'}],
           'phases': ['PREFILL', 'DECODE'], 'ut_family': 'custom', 'ut_family_why': 'Muse norm/gain combine',
           'target_callable': 'sglang.srt.rocm_muse_spark.muse_moe_flydsl:muse_moe_flydsl',
           'case_axes': {'PREFILL': {'m': 'shape'}, 'DECODE': {'m': 'shape'}},
           'phase_regimes': {'PREFILL': {'cuda_graph': False}, 'DECODE': {'cuda_graph': False}}}
    if anchor:
        out['invocation_anchors'] = {'kernels': [UP], 'why': WHY}
    return out


class InvocationAnchorsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='invocation anchors ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.shapes = self.root / 'evidence/shapes'
        self.shapes.mkdir(parents=True)
        self.points = {'points': [
            {'point_id': 'pr_396', 'phase': 'PREFILL', 'achieved_shape': 396,
             'achieved_context': 133100, 'counted': True, 'represented': {'instances': 10}},
            {'point_id': 'de_1', 'phase': 'DECODE', 'achieved_shape': 1,
             'context_tokens': 27000, 'counted': True, 'represented': {'instances': 10}}],
            'phase_summary': {}, 'shape_tolerance': 1.25}
        self.shape('pr_396', {UP: 2, DOWN: 2, FINAL: 62})
        self.shape('de_1', {UP: 2, DOWN: 2, FINAL: 2})

    def shape(self, point, counts):
        doc = {'kernels': {k: {'launches_per_instance': n} for k, n in counts.items()}}
        (self.shapes / (point + '.json')).write_text(json.dumps(doc))

    def plan(self, declaration=None, points=None):
        return P.plan(points or self.points, str(self.shapes), [declaration or task()])['tasks']['native_moe']

    def score(self, plan, mode, factor=1):
        cases, times = [], []
        for c in plan['cases']:
            sig = c['case_id']
            cases.append({'sig': sig, 'regime': c['phase'], 'm': c['shape'],
                          'effective_calls': c['effective_calls'] * factor, 'config_id': 'c10'})
            times.append({'sig': sig, 'regime': c['phase'], 'm': c['shape'],
                          'baseline_ms': 1.0, 'optimized_ms': 0.5 if c['phase'] == 'PREFILL' else 2.0})
        meta = {'workload': {'kind': 'agentx_trace_replay', 'cases': cases,
            'serving_weight_model': {'source': 'observed_workload', 'evidence_path': 'observed.json',
                                    'config_weighting': mode}}}
        return H.serving_weighted_speedup(times, meta)['weighted']

    def test_real_moe_shared_leaf_distorts_weights_but_uniform_factor_cancels(self):
        anchored, legacy = self.plan(), self.plan(task(False))
        self.assertEqual([c['effective_calls'] for c in anchored['cases']], [20, 20])
        self.assertEqual([c['effective_calls'] for c in legacy['cases']], [660, 60])
        self.assertEqual([c['kernel_launches_per_instance'] for c in anchored['cases']], [66, 6])
        self.assertEqual(legacy['invocation_count_basis']['kind'], 'legacy_leaf_launch_sum')
        for mode in ('calls', 'time'):
            with self.subTest(mode=mode):
                correct = self.score(anchored, mode)
                self.assertAlmostEqual(correct, 0.8)
                self.assertAlmostEqual(self.score(anchored, mode, factor=2), correct)
                # Same op timings appear beneficial solely because shared leaf counts
                # overweight PREFILL. This must never be an invocation estimator.
                self.assertAlmostEqual(self.score(legacy, mode), 1.6)

    def test_mutually_exclusive_anchors_sum_calls_even_when_a_point_mixes_branches(self):
        small = 'mfma_moe1_silu_mul_afp4_wfp4_fp4_t32x128x256_pm1_fp4q_sort_async_v33'
        large = 'mfma_moe1_silu_mul_afp4_wfp4_fp4_t64x128x256_pm1_fp4q_sort_async_v33'
        declaration = task()
        declaration.update(kernels=[small, large], phases=['PREFILL'],
            target_callable='sglang.srt.rocm_muse_spark.mxfp4_moe:run_a4w4')
        declaration['invocation_anchors'] = {'kernels': [small, large], 'mutually_exclusive': True,
            'why': 'mxfp4_moe.py:151,166 selects block_m32 for M<=1024 else64; each call launches exactly one stage1. A point may contain both call populations.'}
        self.shape('pr_396', {small: 60, large: 2, FINAL: 62})
        result = self.plan(declaration)
        self.assertEqual(result['cases'][0]['invocations_per_instance'], 62)
        self.assertEqual(result['cases'][0]['effective_calls'], 620)
        self.assertNotEqual(result['cases'][0]['invocations_per_instance'], 60)

    def test_anchor_contract_lint_requires_membership_reason_and_exclusive_alternatives(self):
        valid = task()
        valid['invocation_anchors'] = {'kernels': [FINAL], 'why': 'Exactly once for this isolated scope.'}
        errors, _ = B.lint({'tasks': [valid]}, [UP, DOWN], {})
        self.assertEqual(errors, [])  # a required companion may be the anchor
        invalid = [
            ({'kernels': ['foreign_leaf'], 'why': WHY}, 'selected kernels or companions'),
            ({'kernels': [UP], 'why': ''}, 'exactly-once why'),
            ({'kernels': [UP, DOWN], 'why': WHY}, 'mutually_exclusive=true'),
            ({'kernels': [UP, UP], 'why': WHY}, 'distinct kernel'),
            (None, 'must be an object'),
        ]
        for anchor, text in invalid:
            with self.subTest(anchor=anchor):
                declaration = task(); declaration['invocation_anchors'] = anchor
                errors, _ = B.lint({'tasks': [declaration]}, [UP, DOWN], {})
                self.assertTrue(any(text in e for e in errors), errors)
                with self.assertRaises(ValueError):
                    self.plan(declaration)

    def test_missing_or_invalid_anchor_count_fails_on_an_owned_point(self):
        self.shape('pr_396', {DOWN: 2, FINAL: 62})
        with self.assertRaisesRegex(ValueError, 'pr_396.*no declared invocation anchor'):
            self.plan()
        for bad in [0, -1, float('nan'), True]:
            with self.subTest(count=bad):
                self.shape('pr_396', {UP: bad, DOWN: 2, FINAL: 62})
                with self.assertRaises(ValueError):
                    self.plan()

    def test_shared_companion_does_not_own_points_but_a_companion_anchor_can(self):
        # A shared leaf launched for some other operator, with no selected leaf.
        self.shape('de_1', {FINAL: 62})
        anchored = self.plan()
        self.assertEqual([c['point_id'] for c in anchored['cases']], ['pr_396'])
        legacy = self.plan(task(False))
        self.assertEqual([c['point_id'] for c in legacy['cases']], ['pr_396', 'de_1'])
        self.assertEqual(legacy['cases'][1]['effective_calls'], 620)
        # A source-justified companion anchor may establish ownership of a real
        # alternative branch without a selected leaf in that phase.
        declaration = task()
        declaration['companions'].append({'kernel': 'alternate_entry', 'why': 'Alternative branch entry.'})
        declaration['invocation_anchors'] = {'kernels': [UP, 'alternate_entry'],
            'mutually_exclusive': True, 'why': 'Test dispatcher selects exactly one native entry per full call.'}
        self.shape('de_1', {'alternate_entry': 3, FINAL: 62})
        rows = self.plan(declaration)['cases']
        self.assertEqual([c['point_id'] for c in rows], ['pr_396', 'de_1'])
        self.assertEqual(rows[1]['effective_calls'], 30)

    def test_bucket_interpolates_invocations_and_preserves_instance_partition(self):
        points = {'points': [
            {'point_id': 'small', 'phase': 'PREFILL', 'achieved_shape': 400, 'achieved_context': 1000,
             'counted': True, 'represented': {'instances': 10}},
            {'point_id': 'large', 'phase': 'PREFILL', 'achieved_shape': 800, 'achieved_context': 1000,
             'counted': True, 'represented': {'instances': 10}}],
            'phase_summary': {'PREFILL': {'unrepresented_bands': [
                {'lo': 550, 'hi': 650, 'shape': 600, 'context': 1000, 'instances': 4,
                 'nearest_point': 'small', 'bracketing_points': ['small', 'large'],
                 'proxy_mass_share_of_phase_pct': 10}]}}, 'shape_tolerance': 1.25}
        self.shape('small', {UP: 2, DOWN: 2, FINAL: 62})
        self.shape('large', {UP: 4, DOWN: 4, FINAL: 64})
        before = copy.deepcopy(points)
        rows = self.plan(points=points)['cases']
        self.assertEqual([c['effective_calls'] for c in rows], [12, 40, 12])
        self.assertEqual(rows[-1]['invocations_per_instance'], 3)
        self.assertEqual(rows[-1]['kernel_launches_per_instance'], 69)
        self.assertEqual(sum(c['instances_represented'] for c in rows), 20)
        self.assertEqual(points, before)

    def test_multiconfig_plan_and_manifest_keep_anchor_reason(self):
        merged = {'configs': {'c10': {}, 'c16': {}}, 'points': [], 'shape_tolerance': 1.25}
        for cid in merged['configs']:
            for p in self.points['points']:
                q = dict(p, point_id=cid + '__' + p['point_id'], config_id=cid)
                merged['points'].append(q)
                (self.shapes / (q['point_id'] + '.json')).write_bytes(
                    (self.shapes / (p['point_id'] + '.json')).read_bytes())
        planned = self.plan(points=merged)
        self.assertEqual(planned['invocation_anchors'], task()['invocation_anchors'])
        self.assertEqual({c['config_id'] for c in planned['cases']}, {'c10', 'c16'})
        manifest = B.manifest(task(), planned['cases'])
        self.assertEqual(sum(c['effective_calls'] for c in manifest['cases']), 80)
        self.assertTrue(all(c['invocation_count_basis']['why'] == WHY for c in manifest['cases']))

    def test_build_requested_kernel_retains_anchor_and_basis(self):
        (self.root / 'sampling_points_final.json').write_text(json.dumps(self.points))
        (self.root / 'selection_targets.json').write_text(json.dumps({'targets': [{'id': UP}, {'id': DOWN}]}))
        result = B.build(str(self.root), {'tasks': [task()]}, no_scaffold=True)
        self.assertFalse(result.get('errors'))
        run = self.root / 'tasks/native_moe/run'
        requested = json.loads((run / 'requested_kernel.json').read_text())
        planned = json.loads((run / 'ut_case_plan.json').read_text())
        self.assertEqual(requested['invocation_anchors'], task()['invocation_anchors'])
        self.assertEqual(requested['invocation_count_basis']['why'], WHY)
        self.assertEqual(planned['invocation_anchors'], requested['invocation_anchors'])


if __name__ == '__main__':
    unittest.main()
