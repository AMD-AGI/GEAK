"""Current Python UT identities must pass the native full-report delivery verifier."""
import importlib.util
import json
import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SKILL = Path(__file__).resolve().parents[1] / 'kernel-ut'
NODE = os.environ.get('KERNEL_EXTRACT_NODE') or shutil.which('node')
spec = importlib.util.spec_from_file_location('canonical_driver', SKILL / 'ut_contracts/ut_driver.py')
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)


@unittest.skipUnless(NODE, 'native verifier needs Node')
class CanonicalDeliveryTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name); self.task = self.root / 'task'
        (self.task / 'run').mkdir(parents=True); (self.task / 'kernel_src').mkdir()
        shutil.copy2(SKILL / 'shared/kernel_source_delivery.mjs', self.task / 'kernel_source_delivery.mjs')
        for name in ['cases.py', 'unittest.py', 'helper.py', 'kernel_src/leaf.py']:
            (self.task / name).write_text('# synthetic contract fixture, not GPU evidence\n')
        self.write('meta.json', {'target_callable': 'fixture:op', 'kernel_extraction': {
            'source_delivery_files': ['kernel_source_delivery.mjs']}})
        self.write('run/ut_cases.json', {})
        with patch.dict(os.environ, {'KERNEL_EXTRACT_NODE': NODE}):
            digest = D._snapshot(SimpleNamespace(dir=str(self.task)))
            reuse = D._reuse_key(SimpleNamespace(dir=str(self.task)))
        native = json.loads(subprocess.check_output([NODE, str(SKILL / 'shared/kernel_source_delivery.mjs'),
            'ut-driver-source-snapshot', str(self.task)], text=True))
        self.assertEqual(native['snapshot_sha256'], digest)
        self.assertEqual(native['reuse_key'], reuse)
        snapshot = json.loads(subprocess.check_output([NODE, str(self.task / 'kernel_source_delivery.mjs'),
            'snapshot', str(self.task)], text=True))
        gates = [g for g in D.GATES if g != 'timing']
        self.ut = dict(ok=True, verdict='pass', exit_code=0, started=1, family='fixture',
            target_callable='fixture:op', snapshot_sha256=digest, contract_sha256='a'*64,
            validation_contract='kernel_ut_validation/v1', validation_mode='correctness',
            required_gates=gates, performance_measured=False,
            gates={g: {'ok': True, 'status': 'pass'} for g in gates})
        self.report = dict(self.ut, reuse_key=reuse)
        log = self.write('run/log.json', {'synthetic': True})
        common = dict(ok=True, device_kernel='leaf', target_callable='fixture:op',
                      snapshot_sha256=digest, command='synthetic fixture', gpu=[0], log_ref=log)
        ex = self.write('run/execution.json', dict(common, kind='candidate_execution', invocations=1))
        neg = self.write('run/negative.json', dict(common, kind='numerical_rejection',
                                                  launch_succeeded=True, numerical_mismatches=1))
        self.manifest = dict(snapshot, contract='kernel_source_delivery/v1', target_callable='fixture:op',
            ut_snapshot={'contract': 'ut_driver/source_snapshot/v1'},
            kernels=[dict(device_kernel='leaf', source_files=['kernel_src/leaf.py'],
                          execution_ref=ex, negative_control_ref=neg)])

    def write(self, name, doc):
        path = self.task / name; path.write_text(json.dumps(doc))
        return {'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def check(self):
        self.manifest['ut_result_ref'] = self.write('run/ut_result.json', self.ut)
        self.manifest['ut_snapshot']['report_ref'] = self.write('run/ut_report.json', self.report)
        ref = self.write('run/source.json', self.manifest)
        kernel = {'device_kernel': 'leaf', 'source_delivery_required': True}
        result = {'task_dir': str(self.task), 'target_callable': 'fixture:op', 'source_delivery_ref': ref}
        script = ('import {verifySourceDelivery} from %s; console.log(JSON.stringify(await '
                  'verifySourceDelivery(%s,%s)));') % (
                      json.dumps((SKILL / 'shared/kernel_source_delivery.mjs').as_uri()),
                      json.dumps(kernel), json.dumps(result))
        return json.loads(subprocess.check_output([NODE, '--input-type=module', '-e', script], text=True))

    def test_current_python_snapshot_and_reuse_key_deliver_correctness(self):
        self.assertTrue(self.check()['ok'])

    def test_wrong_contract_cannot_reinterpret_current_receipts(self):
        self.manifest['ut_snapshot']['contract'] = 'ut_driver/path_sha256_concat/v1'
        self.assertFalse(self.check()['ok'])

    def test_current_contract_requires_process_state_even_when_both_reports_omit_it(self):
        for d in [self.ut, self.report]:
            d['gates'] = {k:v for k,v in d['gates'].items() if k != 'process_state'}
            d['required_gates'] = [g for g in d['required_gates'] if g != 'process_state']
        self.assertFalse(self.check()['ok'])

    def test_changed_unlisted_helper_is_caught_by_native_reuse_key(self):
        (self.task / 'helper.py').write_text('# changed\n')
        self.assertIn('reuse_key', self.check()['why'])

    def test_changed_required_gate_and_performance_claim_are_rejected(self):
        self.report['performance_measured'] = True
        self.assertFalse(self.check()['ok'])
        self.report['performance_measured'] = False
        self.report['gates'] = dict(self.report['gates'], correctness={'ok': False, 'status': 'fail'})
        self.assertFalse(self.check()['ok'])

    def test_current_contract_accepts_only_the_recorded_positive_infinite_control_margin(self):
        self.report['gates'] = dict(self.report['gates'], negative_controls={
            'status': 'pass', 'ok': True, 'controls': [
                {'caught': True, 'ok': True, 'restored': True, 'margin': float('inf')}]})
        self.assertTrue(self.check()['ok'])
        self.report['gates']['negative_controls']['controls'][0]['margin'] = float('nan')
        self.assertFalse(self.check()['ok'])


if __name__ == '__main__':
    unittest.main()
