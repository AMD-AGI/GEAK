"""CPU regression: generated UT receipts use the delivery verifier's own identity."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


CONTRACTS = Path(__file__).resolve().parents[1]
OWNER = CONTRACTS.parent / 'shared/kernel_source_delivery.mjs'
NODE = shutil.which(os.environ.get('KERNEL_EXTRACT_NODE') or 'node')


@unittest.skipUnless(NODE, 'Node.js 18+ is required for the native source owner')
class SourceIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.task = Path(temporary.name) / 'task with spaces'
        self.env = dict(os.environ, KERNEL_EXTRACT_NODE=NODE)
        self.scaffold()
        (self.task / 'kernel_src/impl.py').write_text('def op():\n    return 1\n')

    def scaffold(self, force=False):
        args = [sys.executable, str(CONTRACTS / 'scaffold_ut.py'), 'new', '--family', 'gemm',
                '--task', str(self.task), '--target', 'pkg:op', '--bind-module', 'pkg',
                '--bind-file', 'kernel_src/impl.py', '--axis', 'n=16', '--axis', 'k=128']
        result = subprocess.run(args + (['--force'] if force else []),
                                capture_output=True, text=True, env=self.env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def owner_snapshot(self):
        return json.loads(subprocess.check_output([NODE, str(OWNER), 'snapshot', str(self.task)], text=True))

    def run_receipt(self, gate='contract', reuse=False):
        # CPU-only subsets stay incomplete; no GPU gates or fake PASS.
        args = [sys.executable, str(self.task / 'unittest.py'), '--gates', gate]
        result = subprocess.run(args + (['--reuse'] if reuse else []),
                                capture_output=True, text=True, env=self.env)
        self.assertNotEqual(result.returncode, 0)
        receipt = self.task / 'run/ut_result.json'
        self.assertTrue(receipt.exists(), result.stdout + result.stderr)
        value = json.loads(receipt.read_text())
        self.assertFalse(value['ok'])
        self.assertEqual(value['snapshot_sha256'], self.owner_snapshot()['snapshot_sha256'])
        return value

    def test_generated_receipt_matches_owner_and_includes_scaffold_helpers(self):
        value = self.run_receipt()
        files = {f['path'] for f in self.owner_snapshot()['files']}
        self.assertTrue({'ut_driver.py', 'ut_contract.py', 'harness_lib.py',
                         'kernel_source_delivery.mjs', 'run/ut_cases.json', 'run/ut_contract.json'} <= files)
        self.assertEqual((self.task / 'kernel_source_delivery.mjs').read_bytes(), OWNER.read_bytes())
        # Writing results must not introduce a source-identity cycle.
        (self.task / 'run/another_result.json').write_text('{"output": true}\n')
        self.assertEqual(value['snapshot_sha256'], self.owner_snapshot()['snapshot_sha256'])

    def test_declared_input_change_reaches_both_receipt_and_verifier(self):
        meta_path = self.task / 'meta.json'
        meta = json.loads(meta_path.read_text())
        meta['kernel_extraction']['source_delivery_files'].append('run/static-input.bin')
        meta_path.write_text(json.dumps(meta))
        operand = self.task / 'run/static-input.bin'
        operand.write_bytes(b'first frozen input')
        first = self.run_receipt('compile_variants')['snapshot_sha256']
        self.assertEqual(self.run_receipt('compile_variants', reuse=True)['reused_gates'], ['compile_variants'])
        operand.write_bytes(b'changed frozen input')
        after = self.run_receipt('compile_variants', reuse=True)
        self.assertNotEqual(first, after['snapshot_sha256'])
        self.assertEqual(after['reused_gates'], [])

    def test_force_regeneration_keeps_the_native_owner_and_driver_together(self):
        (self.task / 'ut_driver.py').write_text('raise RuntimeError("obsolete driver")\n')
        (self.task / 'kernel_source_delivery.mjs').write_text('throw new Error("obsolete owner");\n')
        self.scaffold(force=True)
        self.assertEqual((self.task / 'ut_driver.py').read_bytes(), (CONTRACTS / 'ut_driver.py').read_bytes())
        self.run_receipt()

    def test_missing_runtime_fails_before_gates_without_fallback_receipt(self):
        self.env['KERNEL_EXTRACT_NODE'] = str(self.task / 'missing-node')
        result = subprocess.run([sys.executable, str(self.task / 'unittest.py'), '--gates', 'contract'],
                                capture_output=True, text=True, env=self.env)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('UT_GATE', result.stdout)
        self.assertFalse((self.task / 'run/ut_result.json').exists())

    def test_missing_declared_input_keeps_the_native_owner_error(self):
        meta_path = self.task / 'meta.json'
        meta = json.loads(meta_path.read_text())
        meta['kernel_extraction']['source_delivery_files'].append('missing-input.bin')
        meta_path.write_text(json.dumps(meta))
        result = subprocess.run([sys.executable, str(self.task / 'unittest.py'), '--gates', 'contract'],
                                capture_output=True, text=True, env=self.env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing-input.bin', result.stderr)
        self.assertNotIn('UT_GATE', result.stdout)
        self.assertFalse((self.task / 'run/ut_result.json').exists())


if __name__ == '__main__':
    unittest.main()
