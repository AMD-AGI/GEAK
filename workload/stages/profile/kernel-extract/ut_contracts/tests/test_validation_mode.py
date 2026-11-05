"""Exercise real driver orchestration/receipts with CPU gate doubles, never a GPU pass."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('validation_mode_driver', Path(__file__).parents[1] / 'ut_driver.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class ValidationModeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / 'run').mkdir()
        self.task = SimpleNamespace(dir=str(self.root), doc={'family': 'test'},
                                    meta={'target_callable': 'fixture:entry'},
                                    sig_filter=None, call_filter=None,
                                    U=SimpleNamespace(contract_sha256=lambda _: 'c' * 64))
        self.calls = []
        self.fail = None
        self.patches = contextlib.ExitStack()
        self.addCleanup(self.patches.close)
        for name, value in [('Task', self.task), ('_tree_hashes', {}), ('_reuse_key', 'a' * 64),
                            ('_snapshot', 'b' * 64), ('_quick_subset', None)]:
            self.patches.enter_context(patch.object(driver, name, return_value=value))
        for gate in driver.GATES:
            def run(*args, gate=gate):
                self.calls.append(gate)
                return {'status': 'fail' if gate == self.fail else 'pass'}
            self.patches.enter_context(patch.object(driver, 'gate_' + gate, side_effect=run))

    def run_driver(self, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            code = driver.main(argv=['--task', str(self.root), *args])
        suffix = '.quick' if '--quick' in args else ''
        report = json.loads((self.root / ('run/ut_report' + suffix + '.json')).read_text())
        return code, report

    def test_correctness_qualifies_all_non_timing_gates_without_a_measurement(self):
        code, report = self.run_driver('--validation-mode', 'correctness')
        self.assertEqual(code, 0)
        self.assertEqual(report['verdict'], 'pass')
        expected = set(driver.GATES) - {'timing'}
        self.assertEqual(set(self.calls), expected)
        self.assertEqual(set(report['required_gates']), expected)
        self.assertFalse(report['performance_measured'])
        result = json.loads((self.root / 'run/ut_result.json').read_text())
        self.assertTrue(result['ok'])
        self.assertEqual(result['validation_contract'], 'kernel_ut_validation/v1')
        self.assertEqual(result['validation_mode'], 'correctness')
        self.assertFalse(result['performance_measured'])
        self.assertNotIn('timing', result['gates'])

    def test_correctness_does_not_accept_an_omitted_or_failed_gate(self):
        code, report = self.run_driver('--validation-mode', 'correctness', '--skip', 'replay')
        self.assertEqual((code, report['verdict']), (1, 'partial'))
        self.fail = 'binding'
        code, report = self.run_driver('--validation-mode', 'correctness')
        self.assertEqual((code, report['verdict']), (1, 'fail'))

    def test_quick_never_qualifies_a_ut(self):
        code, report = self.run_driver('--validation-mode', 'correctness', '--quick')
        self.assertEqual((code, report['verdict']), (1, 'quick'))
        self.assertFalse((self.root / 'run/ut_result.json').exists())

    def test_default_performance_mode_still_requires_timing(self):
        code, report = self.run_driver('--skip', 'timing')
        self.assertEqual((code, report['verdict']), (1, 'partial'))
        code, report = self.run_driver()
        self.assertEqual(code, 0)
        self.assertIn('timing', self.calls)
        self.assertTrue(report['performance_measured'])

    def test_optimization_reuses_correctness_but_must_measure_performance(self):
        self.run_driver('--validation-mode', 'correctness')
        self.calls.clear()
        code, report = self.run_driver('--reuse')
        self.assertEqual(code, 0)
        self.assertEqual(self.calls, ['restore', 'timing'])
        self.assertTrue(report['performance_measured'])
        self.assertEqual(report['validation_mode'], 'performance')


if __name__ == '__main__':
    unittest.main()
