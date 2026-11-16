"""CPU regressions for actual recorded-call replay accounting."""
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'ut_driver.py'
spec = importlib.util.spec_from_file_location('server_call_accounting_driver', SCRIPT)
driver = importlib.util.module_from_spec(spec)
_path = sys.path[:]
try:
    spec.loader.exec_module(driver)
finally:
    sys.path[:] = _path


class ServerCallAccountingTest(unittest.TestCase):
    def run_gate(self, baseline, candidate, call_filter=None, contract=None):
        with tempfile.TemporaryDirectory() as tmp:
            calls = [{'id': name, 'kernel': 'leaf', 'callable': 'module:entry'}
                     for name in ('first', 'second')]
            Path(tmp, 'calls.json').write_text(json.dumps({'calls': calls}))
            harness = SimpleNamespace(build_candidate_overlay=lambda *a: ('base', 'cand'))
            task = SimpleNamespace(dir=tmp, meta={'server_evidence': {'server_calls': 'calls.json'}},
                                   mc=contract or {}, h=harness, gpus=[0], call_filter=call_filter)
            compare = SimpleNamespace(compare_call_outputs=lambda b, c, *a: (b == c, []))
            with patch.object(driver, '_parallel', return_value=((baseline, None), (candidate, None))), \
                 patch.object(driver, '_watch_env', return_value=({}, None)), \
                 patch.object(driver, '_register_watch'), patch.object(driver, '_load', return_value=compare):
                return driver.gate_server_calls(task, None, {'scratch': tmp})

    @staticmethod
    def success():
        return {k: {'status': 'ok', 'outputs': {'return': 3}} for k in ('first', 'second')}

    def test_zero_replays_are_not_a_pass(self):
        skipped = {k: {'status': 'unreplayable', 'why': 'bound instance absent'}
                   for k in ('first', 'second')}
        result = self.run_gate(skipped, skipped)
        self.assertEqual(result['status'], 'fail')
        self.assertEqual(result['replayed_by_baseline'], 0)
        self.assertEqual(result['unreplayed_baseline_calls'], 2)
        self.assertTrue(all(not r['ok'] for r in result['rows']))

    def test_one_missing_baseline_call_remains_unqualified(self):
        baseline = self.success(); del baseline['second']
        result = self.run_gate(baseline, self.success())
        self.assertEqual(result['status'], 'fail')
        self.assertEqual(result['replayed_by_baseline'], 1)

    def test_candidate_failure_is_still_a_failure(self):
        candidate = self.success(); candidate['second'] = {'status': 'error', 'why': 'native failure'}
        result = self.run_gate(self.success(), candidate)
        self.assertEqual(result['status'], 'fail')
        self.assertEqual(result['replayed_by_candidate'], 1)

    def test_every_selected_call_must_replay_and_match(self):
        result = self.run_gate(self.success(), self.success())
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(result['replayed_by_baseline'], result['calls'])
        candidate = self.success(); candidate['second']['outputs'] = {'return': 4}
        self.assertEqual(self.run_gate(self.success(), candidate)['status'], 'fail')

    def test_quick_subset_only_grades_its_selected_calls(self):
        result = self.run_gate({'first': self.success()['first']},
                               {'first': self.success()['first']}, ['first'])
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(result['calls'], 1)
        self.assertEqual(self.run_gate({}, {}, [])['status'], 'fail')

    def test_postprocess_reporting_keeps_strict_replay_accounting(self):
        contract = {'server_call_postprocess': True,
                    'server_call_postprocess_why': 'compare only defined output rows'}
        baseline, candidate = self.success(), self.success()
        for leg in (baseline, candidate):
            for row in leg.values():
                row['postprocessed'] = True
        result = self.run_gate(baseline, candidate, contract=contract)
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(result['postprocess'], {
            'hook': 'cases.server_call_postprocess',
            'why': contract['server_call_postprocess_why'], 'applied_to': 2})
        del baseline['second']
        result = self.run_gate(baseline, candidate, contract=contract)
        self.assertEqual(result['status'], 'fail')
        self.assertEqual(result['postprocess']['applied_to'], 1)
        self.assertEqual(result['unreplayed_baseline_calls'], 1)


if __name__ == '__main__':
    unittest.main()
