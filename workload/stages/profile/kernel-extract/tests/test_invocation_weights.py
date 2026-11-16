"""CPU call-unit regression: observed invocations, launch inventory and frozen handoff."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / 'scripts'))
import plan_ut_cases as planner
import build_task_plan as builder


class InvocationWeightsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.shapes = self.root / 'evidence/shapes'
        self.shapes.mkdir(parents=True)
        self.points = {'points': [
            {'point_id': 'small', 'phase': 'DECODE', 'achieved_shape': 2, 'context_tokens': 20,
             'counted': True, 'represented': {'instances': 10}},
            {'point_id': 'large', 'phase': 'DECODE', 'achieved_shape': 4, 'context_tokens': 40,
             'counted': True, 'represented': {'instances': 20}}],
            'phase_summary': {'DECODE': {'unrepresented_bands': [
                {'shape': 3, 'lo': 3, 'hi': 3, 'instances': 2, 'nearest_point': 'small',
                 'bracketing_points': ['small', 'large'], 'proxy_mass_share_of_phase_pct': 10}],
                'proxy_mass_outside_tolerance_pct': 10}}}
        for name, counts in [('small', {'A': 4, 'B': 4}), ('large', {'A': 4, 'B': 8})]:
            (self.shapes / (name + '.json')).write_text(json.dumps({'point_id': name,
                'kernels': {k: {'launches_per_instance': v} for k,v in counts.items()}}))
        proof = self.root / 'scoped-counts.json'
        proof.write_text('{"fixture": "observed complete entry counts"}\n')
        self.task = {'label': 'op', 'target_callable': 'pkg:op', 'kernels': ['A', 'B'],
                     'phases': ['DECODE'], 'ut_family': 'custom', 'ut_family_why': 'complete op',
                     'invocation_counts': {'target_callable': 'pkg:op', 'points': {
                         p['point_id']: {'ok': True, 'phase': p['phase'], 'shape': p['achieved_shape'],
                             'calls': n, 'instances': 3, 'evidence': {'path': str(proof),
                             'sha256': hashlib.sha256(proof.read_bytes()).hexdigest()}}
                         for p,n in zip(self.points['points'], [6, 12])}}}

    def test_variable_repeated_members_do_not_determine_invocation_rates(self):
        result = planner.plan(self.points, str(self.shapes), [self.task])['tasks']['op']
        rows = result['cases']
        self.assertEqual([c['launches_per_instance'] for c in rows], [8, 12, 10])
        self.assertEqual([c['operator_calls_per_instance'] for c in rows], [2, 4, 3])
        self.assertEqual([c['instances_represented'] for c in rows], [8, 20, 2])
        self.assertEqual([c['effective_calls'] for c in rows], [16, 80, 6])
        self.assertEqual(result['invocation_counts'], self.task['invocation_counts'])

    def test_missing_ambiguous_or_wrong_scope_has_no_launch_fallback(self):
        mutations = [lambda t: t['invocation_counts']['points'].pop('small'),
                     lambda t: t['invocation_counts'].update(target_callable='pkg:another'),
                     lambda t: t['invocation_counts']['points']['small'].update(ok=False),
                     lambda t: t['invocation_counts']['points']['small'].update(calls=None),
                     lambda t: t['invocation_counts']['points']['small'].update(instances=0),
                     lambda t: t['invocation_counts']['points']['small'].update(phase='PREFILL'),
                     lambda t: t['invocation_counts']['points']['small'].update(shape=4),
                     lambda t: t['invocation_counts']['points']['small'].pop('evidence')]
        for mutate in mutations:
            task = copy.deepcopy(self.task)
            mutate(task)
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, 'invocation counts'):
                planner.plan(self.points, str(self.shapes), [task])
        (self.root / 'selection_targets.json').write_text(json.dumps({'targets': [{'id':'A'}, {'id':'B'}]}))
        (self.root / 'sampling_points_final.json').write_text(json.dumps(self.points))
        self.assertTrue(builder.build(str(self.root), {'tasks':[task]}, check_only=True)['errors'])
        self.assertFalse((self.root / 'tasks').exists())

    def test_direct_single_kernel_semantics_remain_unchanged(self):
        task = {k:v for k,v in self.task.items() if k != 'invocation_counts'}
        task['kernels'] = ['A']
        result = planner.plan(self.points, str(self.shapes), [task])['tasks']['op']
        self.assertEqual([c['effective_calls'] for c in result['cases']], [32, 80, 8])
        self.assertNotIn('invocation_counts', result)
        self.assertTrue(all('operator_calls_per_instance' not in c for c in result['cases']))

    def test_native_source_identity_binds_the_complete_count_evidence(self):
        node = shutil.which('node')
        if not node:
            self.skipTest('Node.js required for native source identity')
        task = self.root / 'task'
        (task / 'run').mkdir(parents=True)
        (task / 'kernel_src').mkdir()
        for f in ('cases.py', 'unittest.py', 'kernel_src/op.py'):
            (task / f).write_text('pass\n')
        (task / 'meta.json').write_text('{}\n')
        plan = planner.plan(self.points, str(self.shapes), [self.task])['tasks']['op']
        path = task / 'run/ut_case_plan.json'
        path.write_text(json.dumps(plan))
        builder.stamp_meta(str(task), {'server_preimports': []}, None)
        owner = SKILL / 'shared/kernel_source_delivery.mjs'
        before = json.loads(subprocess.check_output([node, str(owner), 'snapshot', str(task)], text=True))
        plan['invocation_counts']['points']['small']['evidence']['sha256'] = '0' * 64
        path.write_text(json.dumps(plan))
        after = json.loads(subprocess.check_output([node, str(owner), 'snapshot', str(task)], text=True))
        self.assertNotEqual(before['snapshot_sha256'], after['snapshot_sha256'])


if __name__ == '__main__':
    unittest.main()
