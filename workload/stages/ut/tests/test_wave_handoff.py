"""CPU regressions for shared locks, delivery completion and resume."""
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SLOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SLOT / 'kernel-ut/scripts'))
import ut_wave as W
import ut_gpu_queue as Q
from gpu_lock_files import open_lock


class SharedLocksTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_restrictive_umask_does_not_publish_private_locks(self):
        path = self.root / 'locks/gpu_0.lock'
        old = os.umask(0o077)
        try:
            os.close(open_lock(str(path)))
            self.assertEqual(os.umask(0o077), 0o077)
        finally:
            os.umask(old)
        self.assertEqual(path.stat().st_mode & 0o777, 0o666)
        self.assertEqual(path.parent.stat().st_mode & 0o1777, 0o1777)

    def test_readonly_existing_lock_preserves_inode_contents_and_exclusion(self):
        path = self.root / 'gpu_0.lock'
        path.write_text('old owner lock')
        path.chmod(0o444)
        before = path.stat().st_ino
        fd = open_lock(str(path))
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(Q.try_lease([0], 1, str(self.root), lambda _: True), ([], []))
        finally:
            os.close(fd)
        gpus, fds = Q.try_lease([0], 1, str(self.root), lambda _: True)
        Q.release(fds)
        self.assertEqual(gpus, [0])
        self.assertEqual(path.stat().st_ino, before)
        self.assertEqual(path.read_text(), 'old owner lock')

    def test_busy_or_incomplete_group_releases_partial_lease(self):
        self.assertEqual(Q.try_lease([0, 1], 2, str(self.root), lambda g: g == 0), ([], []))
        gpus, fds = Q.try_lease([0], 1, str(self.root), lambda _: True)
        Q.release(fds)
        self.assertEqual(gpus, [0])


class WaveHandoffTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.task = self.root / 'tasks/op'
        (self.task / 'run').mkdir(parents=True)
        W.save(str(self.task / 'run/ut_result.json'), {'verdict': 'pass', 'ok': True})
        W.save(str(self.root / 'task_plan.json'), {'tasks': [{'label': 'op', 'task_dir': str(self.task)}]})
        W.save(str(self.task / 'run/requested_kernel.json'),
               {'entity_kind': 'gpu_kernel', 'device_kernel': 'leaf', 'source_delivery_required': True})

    def args(self):
        return SimpleNamespace(eval_dir=str(self.root), gpus='0', tasks=None, force=False)

    def test_passing_correctness_without_return_is_planned_for_resume(self):
        self.assertTrue(W.passed(str(self.task)))
        self.assertFalse(W.completed(str(self.task)))
        self.assertEqual([r['label'] for r in W.cmd_plan(self.args(), quiet=True)[2]], ['op'])
        cfg = {'eval_dir': str(self.root)}
        prompt = W.lane_prompt('', dict(label='op', task_dir=str(self.task), need=1), {}, 1, cfg, 1000)
        self.assertIn('existing correctness UT passes', prompt)
        self.assertIn('run/RETURN.json', prompt)

    def test_resume_after_exhausted_attempts_grants_budget_without_reusing_log_names(self):
        wave = str(self.root / 'ut_wave')
        W.set_status(wave, 'op', status='failed', attempts=3, started=1, ended=2)
        rows = W.cmd_plan(self.args(), quiet=True)[2]
        W.resume_rows(wave, rows, 3)
        state = W.status_of(wave, 'op')
        self.assertEqual(state['status'], 'pending')
        self.assertEqual(state['attempts'], 3)
        self.assertEqual(state['attempt_limit'], 6)
        self.assertIsNone(state['started'])

    def test_resume_reconciles_valid_delivery_interrupted_before_terminal_state(self):
        wave = str(self.root / 'ut_wave')
        W.set_status(wave, 'op', status='running', attempts=2, started=1)
        args = self.args()
        args.author_workers = 1
        with patch.object(W, 'completed', return_value=True):
            self.assertEqual(W.cmd_start(args), 0)
        state = W.status_of(wave, 'op')
        self.assertEqual(state['status'], 'passed')
        self.assertEqual(state['attempts'], 2)
        self.assertTrue(state['delivery_ok'])

    def test_wrong_task_return_never_reaches_gate(self):
        W.save(str(self.task / 'run/RETURN.json'), {'short_name': 'another', 'task_dir': str(self.task)})
        with patch.object(W.subprocess, 'run') as run:
            self.assertIn('another task', W.delivery_status(str(self.task))[1])
            run.assert_not_called()

    def test_existing_return_is_verified_natively_without_prepared_bypass(self):
        W.save(str(self.task / 'run/RETURN.json'),
               {'short_name': 'op', 'task_dir': str(self.task), 'unittest_smoke': 'pass'})
        response = SimpleNamespace(returncode=0, stdout='{"ok":true}')
        with patch.object(W.subprocess, 'run', return_value=response) as run:
            self.assertTrue(W.completed(str(self.task)))
            argv = run.call_args.args[0]
            self.assertTrue(argv[1].endswith('kernel_extraction_delivery.mjs'))
            self.assertNotIn('--prepared', argv)
            self.assertNotIn('--out', argv)
            self.assertEqual(W.cmd_plan(self.args(), quiet=True)[2], [])
        response.stdout = '{"ok":false,"checks":{"source_delivery":{"ok":false}}}'
        response.returncode = 1
        with patch.object(W.subprocess, 'run', return_value=response):
            self.assertFalse(W.completed(str(self.task)))
            self.assertEqual(len(W.cmd_plan(self.args(), quiet=True)[2]), 1)

    def test_native_gate_rejects_empty_delivery_evidence(self):
        node = os.environ.get('KERNEL_EXTRACT_NODE', 'node')
        if not shutil.which(node):
            self.skipTest('Node is needed for the native delivery gate')
        W.save(str(self.task / 'run/RETURN.json'), {
            'short_name': 'op', 'task_dir': str(self.task), 'unittest_smoke': 'pass', 'editable': True})
        ok, why = W.delivery_status(str(self.task))
        self.assertFalse(ok)
        self.assertIn('source_delivery', why)

    def test_delivery_only_resume_allows_no_command_but_never_failed_command(self):
        wave = str(self.root / 'ut_wave')
        self.assertTrue(Q.last_command_passed(wave, 'op', allow_missing=True))
        W.save(wave + '/gpu_queue/one/request.json', {'task': 'op', 'submitted': 1})
        W.save(wave + '/gpu_queue/one/result.json', {'returncode': 86})
        self.assertFalse(Q.last_command_passed(wave, 'op', allow_missing=True))

    def test_broker_liveness_uses_lock_and_detects_process_exit(self):
        path = self.root / 'broker.lock'
        self.assertFalse(Q.broker_alive(str(self.root)))
        with path.open('w') as guard:
            self.assertFalse(Q.broker_alive(str(self.root)))
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertTrue(Q.broker_alive(str(self.root)))
        self.assertFalse(Q.broker_alive(str(self.root)))

    def test_drain_in_container_does_not_require_host_pid_visibility(self):
        wave = str(self.root / 'ut_wave')
        W.save(wave + '/gpu_queue/one/request.json', {'owner': 'op'})
        W.save(wave + '/broker.json', {'pid': 999999999, 'identity': 'outside-container'})
        result = wave + '/gpu_queue/one/result.json'
        with open(wave + '/broker.lock', 'w') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            publish = threading.Timer(.1, lambda: W.save(result, {'returncode': 130}))
            publish.start()
            try:
                with patch.object(Q, 'identity', side_effect=AssertionError('host PID lookup')):
                    Q.drain_owner(wave, 'op')
            finally:
                publish.join()
        self.assertTrue(Path(wave + '/gpu_queue/one/cancel').exists())

    def test_native_success_then_changed_source_and_evidence_are_rejected(self):
        if not shutil.which(os.environ.get('KERNEL_EXTRACT_NODE', 'node')):
            self.skipTest('Node is needed for the native delivery gate')
        # Synthetic contract fixture only; these records are never measurement evidence.
        target = 'fixture:op'
        def write(name, obj):
            W.save(str(self.task / name), obj)
            return ref(name)
        def ref(name):
            return {'path': name, 'sha256': hashlib.sha256((self.task / name).read_bytes()).hexdigest()}
        (self.task / 'kernel_src').mkdir()
        for name in ['cases.py', 'unittest.py', 'kernel_src/leaf.py']:
            (self.task / name).write_text('# synthetic fixture\n')
        write('meta.json', {'target_callable': target})
        files = [ref(n) for n in sorted(['meta.json', 'cases.py', 'unittest.py', 'kernel_src/leaf.py'])]
        snapshot = hashlib.sha256(json.dumps(files, separators=(',', ':')).encode()).hexdigest()
        ut = write('run/ut_result.json', {'ok': True, 'verdict': 'pass', 'exit_code': 0,
            'target_callable': target, 'snapshot_sha256': snapshot})
        log = write('run/synthetic_log.json', {'fixture': True})
        common = dict(ok=True, device_kernel='leaf', target_callable=target,
                      snapshot_sha256=snapshot, command='synthetic fixture', gpu=[0], log_ref=log)
        execution = write('run/execution.json', dict(common, kind='candidate_execution', invocations=1))
        negative = write('run/negative.json', dict(common, kind='numerical_rejection',
                                                 launch_succeeded=True, numerical_mismatches=1))
        source = write('run/source.json', {'contract': 'kernel_source_delivery/v1',
            'target_callable': target, 'snapshot_sha256': snapshot, 'files': files, 'ut_result_ref': ut,
            'kernels': [{'device_kernel': 'leaf', 'source_files': ['kernel_src/leaf.py'],
                         'execution_ref': execution, 'negative_control_ref': negative}]})
        selection = write('run/selection.json', {'contract': 'kernel_selection', 'ok': True,
            'target_callable': target, 'device_kernel': 'leaf', 'total_calls_observed': 1,
            'target_marker_calls': 1, 'matched_kernel_calls': 1, 'deepest_verified': True,
            'candidate_targets_tested': [target], 'live_candidate_targets': [target]})
        write('run/RETURN.json', {'short_name': 'op', 'task_dir': str(self.task),
            'editable': True, 'unittest_smoke': 'pass', 'baseline_frozen': True,
            'target_callable': target, 'source_delivery_ref': source, 'selection_validation_ref': selection,
            'seam_candidates': [{'target_callable': target, 'role': 'op_seam', 'depth': 1,
                                 'device_kernels': ['leaf']}]})
        self.assertEqual(W.delivery_status(str(self.task)), (True, None))
        self.assertTrue(W.completed(str(self.task)))
        (self.task / 'kernel_src/leaf.py').write_text('# changed source\n')
        self.assertIn('source_delivery', W.delivery_status(str(self.task))[1])
        (self.task / 'kernel_src/leaf.py').write_text('# synthetic fixture\n')
        write('run/synthetic_log.json', {'fixture': 'changed'})
        self.assertIn('source_delivery', W.delivery_status(str(self.task))[1])


if __name__ == '__main__':
    unittest.main()
