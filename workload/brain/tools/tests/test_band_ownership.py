"""Regression for c28's 47-instance band subtracted from a 19-instance point."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import plan_ut_cases as P

spec = importlib.util.spec_from_file_location('relabel_points',
    TOOLS.parents[1] / 'stages/profile/kernel-profile/scripts/relabel_points.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


class BandOwnershipTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.points = [dict(point_id=pid, phase='PREFILL', new_tokens=shape,
                            context_tokens=ctx, sampling_partition=['PREFILL', shape, 1, False])
                       for pid, shape, ctx in [('owner47', 32768, 1000), ('owner19', 31977, 104299)]]
        for p in self.points:
            (self.root / (p['point_id'] + '.json')).write_text(json.dumps({
                'achieved_shape': p['new_tokens'], 'instances': 1,
                'kernels': {'leaf': {'launches_per_instance': 124}}}))
        self.task = {'label': 'op', 'kernels': ['leaf'], 'phases': ['PREFILL']}

    def relabel(self):
        rows = [dict(new_token=p['new_tokens'], cached_token=104299-p['new_tokens'],
                     new_seq=1, cuda_graph=False)
                for p, n in zip(self.points, [47, 19]) for _ in range(n)]
        return R.relabel({'raw_prefill_rows': rows}, {'points': self.points},
                         str(self.root), str(self.root / 'no_probes'), 1.25, 1.25)

    def test_partition_owner_survives_geometric_aggregation(self):
        final = self.relabel()
        bands = final['phase_summary']['PREFILL']['unrepresented_bands']
        self.assertEqual([(b['nearest_point'], b['instances']) for b in bands], [('owner47', 47)])
        self.assertEqual(bands[0]['bracketing_points'][0], 'owner19')
        cases = P.plan(final, str(self.root), [self.task])['tasks']['op']['cases']
        self.assertEqual(sum(c['instances_represented'] for c in cases), 66)
        self.assertEqual(sum(c['effective_calls'] for c in cases), 66 * 124)
        self.assertTrue(all(c['effective_calls'] >= 0 for c in cases))

    def test_legacy_misassigned_band_fails_even_if_bucket_quota_is_zero(self):
        final = self.relabel()
        final['phase_summary']['PREFILL']['unrepresented_bands'][0]['nearest_point'] = 'owner19'
        with self.assertRaisesRegex(ValueError, 'owner19: bands own 47.*19.*rerun relabel'):
            P.plan(final, str(self.root), [self.task], max_buckets=0)

    def test_same_cell_different_owners_have_stable_distinct_case_ids(self):
        final = self.relabel()
        live = [dict(point_id=p['point_id'], achieved_shape=p['new_tokens'],
                     achieved_context=p['context_tokens']) for p in self.points]
        rows = [(32000, 47, 'owner47', 100000), (32000, 19, 'owner19', 100000)]
        bands = R.bands_2d(rows, 1.25, 1.25, 'PREFILL', live)
        self.assertEqual(bands, R.bands_2d(rows[::-1], 1.25, 1.25, 'PREFILL', live))
        for b in bands:
            b['bracketing_points'] = b.pop('bracketing_points_2d')
            b['proxy_mass_share_of_phase_pct'] = 100 * b['instances'] / 66
        final['phase_summary']['PREFILL']['unrepresented_bands'] = bands
        cases = P.plan(final, str(self.root), [self.task])['tasks']['op']['cases']
        self.assertEqual(len({c['case_id'] for c in cases}), len(cases))
        self.assertEqual(sum(c['instances_represented'] for c in cases), 66)
        # A task owning one donor must not acquire the other donor's instances.
        path = self.root / 'owner19.json'
        path.write_text(json.dumps({'kernels': {}}))
        own = P.plan(final, str(self.root), [self.task])['tasks']['op']['cases']
        self.assertEqual(sum(c['instances_represented'] for c in own), 47)


if __name__ == '__main__':
    unittest.main()
