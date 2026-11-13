"""CPU regressions for worker inputs, isolation and driver reporting; no live Docker calls."""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ENGINE = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ENGINE / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scheduler = load('scheduler', 'campaign/scheduler.py')
report = load('driver_report', 'core/report.py')


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='workload-refactor-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_worker_override_accepts_only_positive_decimal_integers(self):
        source = (ENGINE / 'campaign/worker.sh').read_text()
        function = re.search(r'(?ms)^read_override\(\).*?^}', source).group()
        value = self.root / 'override'
        for content, expected in [('900\n', '900'), ('0008', '8'), ('-5', '77'),
                                  ('1e3', '77'), ('invalid 600', '77'), ('0', '77'),
                                  ('12\n34', '77'), ('1234567890', '77'), ('', '77')]:
            with self.subTest(content=content):
                value.write_text(content)
                result = subprocess.run(['bash', '-c', function + '\nread_override "$1" 77',
                                         'test', str(value)], text=True, capture_output=True, check=True)
                self.assertEqual(result.stdout.strip(), expected)

    def test_generated_fixture_rejects_negative_level_and_cold_cheat(self):
        tasks = self.root / 'tasks'
        subprocess.run([sys.executable, str(ENGINE / 'tools/make_fake_tasks.py'),
                        '--dir', str(tasks), '--count', '1'], check=True, capture_output=True)
        task = tasks / 'fake_kernel_0'
        for content, cold, code in [('LEVEL = 2\n', '1', 0), ('LEVEL = -1\n', '1', 1),
                                    ('LEVEL = 2\nCHEAT = 1\n', '0', 0),
                                    ('LEVEL = 2\nCHEAT = 1\n', '1', 1)]:
            with self.subTest(content=content, cold=cold):
                (task / 'kernel_src/mod.py').write_text(content)
                result = subprocess.run([sys.executable, str(task / 'unittest.py')],
                                         env=dict(os.environ, FAKE_COLD=cold),
                                         text=True, capture_output=True)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertIn('CORRECTNESS: ' + ('PASS' if code == 0 else 'FAIL'), result.stdout)

    def test_container_identity_separates_campaigns_and_recovers_only_owned_legacy(self):
        first = scheduler.DockerBackend(str(self.root / 'first'))
        other = scheduler.DockerBackend(str(self.root / 'other'))
        legacy = 'wl_opt_gpu0_task'
        with patch.object(scheduler, 'sh', return_value=(1, 'No such container')):
            scoped = first.name_of('task', [0])
            self.assertNotEqual(scoped, other.name_of('task', [0]))
            self.assertEqual(scoped, scheduler.DockerBackend(first.work_root).name_of('task', [0]))
        with patch.object(scheduler, 'sh', return_value=(0, json.dumps(['WORK_ROOT=' + first.work_root]))):
            self.assertEqual(first.name_of('task', [0]), legacy)
            self.assertNotEqual(other.name_of('task', [0]), legacy)

    def test_topology_and_container_mounts_use_actual_render_devices(self):
        topology = self.root / 'topology'
        for name, minor in [('0', 0), ('2', 137), ('10', 151)]:
            node = topology / name
            node.mkdir(parents=True)
            (node / 'properties').write_text('cpu_cores_count 0\ndrm_render_minor %d\n' % minor)
        self.assertEqual(scheduler.render_devices(topology), ['/dev/dri/renderD137', '/dev/dri/renderD151'])

        backend = scheduler.DockerBackend(str(self.root), image='test-image')
        backend.render_paths = ['/dev/dri/renderD137', '/dev/dri/renderD151']
        calls = []

        def docker(argv, **kwargs):
            calls.append(argv)
            return (1, 'No such container') if argv[:2] == ['docker', 'inspect'] else (0, '')

        with patch.object(scheduler, 'sh', side_effect=docker), \
             patch.object(scheduler.os.path, 'exists', return_value=True), \
             patch('container_runtime.runtime_options', return_value=([], [])):
            self.assertTrue(backend.start('task', [1], {'WORK_ROOT': str(self.root)}))
        command = next(c for c in calls if c[:2] == ['docker', 'run'])
        self.assertIn('/dev/dri/renderD151', command)
        self.assertNotIn('/dev/dri/renderD137', command)
        self.assertIn('GEAK_GPU_RENDER_DEVICES=renderD151', command)
        self.assertIn('GPU_IDS=0', command)

    def test_missing_device_does_not_remove_a_container(self):
        backend = scheduler.DockerBackend(str(self.root))
        backend.render_paths = []
        with patch.object(scheduler, 'sh') as docker:
            self.assertFalse(backend.start('task', [0], {}))
        docker.assert_not_called()

    def test_report_obeys_config_order_and_retains_native_rejection(self):
        self.assertEqual(report.stage_rows(self.root, {'stages': {}, 'target': {'model': 'test'}}), [])
        (self.root / 'config.resolved.json').write_text(json.dumps({'order': ['diagnose', 'baseline']}))
        for name in ['baseline', 'diagnose', 'applyback']:
            out = self.root / name / 'out'
            out.mkdir(parents=True)
            (out / 'result.json').write_text(json.dumps({'status': 'ok', 'metrics': {'gain': 2}}))
        e2e = self.root / 'applyback/out/e2e'
        e2e.mkdir()
        (e2e / 'bench_summary.json').write_text(json.dumps({'independent_arms': False}))
        rows = report.stage_rows(self.root, {'stages': {'diagnose': {'status': 'blocked'}}})
        self.assertEqual([r['stage'] for r in rows], ['diagnose', 'baseline', 'applyback'])
        self.assertEqual([r['status'] for r in rows], ['blocked', 'ok', 'fail'])
        self.assertEqual(rows[0]['metrics'], {})
        self.assertEqual(rows[2]['metrics'], {})


if __name__ == '__main__':
    unittest.main()
