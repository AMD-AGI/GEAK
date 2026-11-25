"""The profile -> UT handoff: what export refuses, and what verify detects."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


class ProfileHandoffTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='profile handoff ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ranking = {
            'ranking': [{'id': 'moe', 'selection_pct_gpu_time': 40.0,
                         'share_of_phase_pct': {'DECODE': 50.0}, 'shapes': [[64, 7168]]},
                        {'id': 'norm', 'selection_pct_gpu_time': 1.0,
                         'share_of_phase_pct': {'DECODE': 2.0}}],
            'denominators': {'DECODE': 'device_ms'},
            'selection_scope': {'source': 'default'}, 'candidate_inventory': [],
            'coverage_summary': {'DECODE': 0.92}}
        self.write('kernel_workload.json', {'phases': ['DECODE']})
        self.write('kernel_ranking.json', self.ranking)
        self.write('selection_targets.json', {
            'policy': {'top_k_per_phase': 5},
            'targets': [{'id': 'moe', 'selected_phases': ['DECODE'], 'phase_ranks': {'DECODE': 1},
                         'overall_pct': 40.0}]})

    def write(self, name, doc):
        (self.root / name).write_text(json.dumps(doc))

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPTS / 'profile_handoff.py'), *map(str, args)],
                              capture_output=True, text=True)

    def export(self):
        return self.run_cli('export', '--eval-dir', self.root)

    def verify(self):
        return self.run_cli('verify', '--handoff', self.root / 'profile_handoff.json')

    def test_export_carries_each_selected_ranking_row_and_verifies(self):
        proc = self.export()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        handoff = json.loads((self.root / 'profile_handoff.json').read_text())
        self.assertEqual([k['id'] for k in handoff['kernels']], ['moe'])
        self.assertEqual(handoff['kernels'][0]['ranking_row']['shapes'], [[64, 7168]])
        self.assertEqual(handoff['coverage_summary'], {'DECODE': 0.92})
        self.assertEqual(self.verify().returncode, 0)

    def test_a_ranking_without_its_inventory_is_refused(self):
        del self.ranking['coverage_summary']
        self.write('kernel_ranking.json', self.ranking)
        proc = self.export()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('coverage_summary', proc.stderr)

    def test_a_selected_target_missing_from_the_ranking_is_refused(self):
        self.write('selection_targets.json', {'targets': [{'id': 'ghost'}]})
        proc = self.export()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('ghost', proc.stderr)

    def test_verify_detects_changed_evidence(self):
        self.assertEqual(self.export().returncode, 0)
        self.ranking['coverage_summary'] = {'DECODE': 0.99}
        self.write('kernel_ranking.json', self.ranking)
        proc = self.verify()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('kernel_ranking.json changed', proc.stderr)


if __name__ == '__main__':
    unittest.main()
