"""CPU subprocess tests of lease lifetime, contention and cancellation. No GPU work."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ut_gpu_queue as Q
import ut_wave as W


class BrokerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gpu_broker_')
        self.addCleanup(self.temp.cleanup)
        self.wave = Path(self.temp.name, 'wave')
        self.tasks = {name: Path(self.temp.name, name) for name in ('a', 'b', 'c')}
        for d in ('logs', 'state', 'gpu_queue'):
            (self.wave / d).mkdir(parents=True, exist_ok=True)
        for path in self.tasks.values():
            path.mkdir()
        self.cfg = {'backend': 'local', 'gpus': [3, 7], 'render_base': 128, 'render_step': 8,
                    'env': {'KERNEL_ENV_KEEP_ARCH': '1'}, 'lock_dir': str(Path(self.temp.name, 'locks')),
                    'rows': [{'label': n, 'need': 1, 'task_dir': str(p)} for n, p in self.tasks.items()]}
        W.save(str(self.wave / 'wave.json'), self.cfg)
        self.local = W.Local(str(self.wave))
        self.sleeper = Path(self.temp.name, 'sleep.py')
        self.sleeper.write_text('import time; time.sleep(600)')
        for name in self.tasks:
            self.local.start(name, [], [str(self.sleeper)], {})
            W.save(str(self.wave / 'state' / (name + '.token')), {'token': name, 'task': name})
        self.parent = subprocess.Popen([sys.executable, str(self.sleeper)])
        self.broker_log = open(self.wave / 'broker.log', 'w')
        self.broker = subprocess.Popen([sys.executable, Q.__file__, 'broker', str(self.wave), str(self.parent.pid)],
                                       env=dict(os.environ, GEAK_GPU_REQUIRE_IDLE='0'),
                                       stdout=self.broker_log, stderr=subprocess.STDOUT)
        self.addCleanup(self.cleanup)
        self.wait(lambda: (self.wave / 'broker.json').exists())

    def cleanup(self):
        if self.broker.poll() is None:
            self.broker.terminate()
        self.broker.wait(20)
        for name in self.tasks:
            self.local.stop(name)
        self.parent.terminate()
        self.parent.wait(5)
        self.broker_log.close()

    def wait(self, predicate, timeout=15):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            result = predicate()
            if result:
                return result
            time.sleep(.03)
        self.fail('condition timed out; broker log: ' + (self.wave / 'broker.log').read_text())

    def submit(self, owner, code, count=1, timeout=20):
        path = self.wave / 'gpu_queue' / uuid.uuid4().hex
        W.save(str(path / 'request.json'), {
            'owner': owner, 'task': owner, 'token': owner, 'count': count,
            'command': [sys.executable, '-c', code], 'cwd': str(self.tasks[owner]),
            'env': {}, 'timeout': timeout, 'submitted': time.time(),
        })
        return path

    def result(self, path):
        return self.wait(lambda: W.load(str(path / 'result.json')))

    def assert_free(self, count=2):
        gpus, fds = Q.try_lease(self.cfg['gpus'], count, self.cfg['lock_dir'], lambda _: True)
        Q.release(fds)
        self.assertEqual(len(gpus), count)

    def test_disjoint_commands_overlap_then_release_before_next_call(self):
        code = 'import os,time; print(os.environ["ROCR_VISIBLE_DEVICES"],os.environ["HIP_VISIBLE_DEVICES"],flush=True);time.sleep(1)'
        a, b = self.submit('a', code), self.submit('b', code)
        ra, rb = self.result(a), self.result(b)
        self.assertEqual((ra['returncode'], rb['returncode']), (0, 0))
        self.assertFalse(set(ra['gpus']) & set(rb['gpus']))
        sa, sb = W.load(str(a / 'lease.json'))['started'], W.load(str(b / 'lease.json'))['started']
        self.assertLess(max(sa, sb), min(sa + ra['run_s'], sb + rb['run_s']))
        self.assertIn('0', (a / 'output.log').read_text())
        self.assert_free()
        # Original CPU authors are still alive, but the full pool is available.
        for n in ('a', 'b'):
            self.assertTrue(self.local.alive(n))
        c = self.submit('c', 'print("whole pool")', count=2)
        self.assertEqual(self.result(c)['gpus'], [3, 7])
        again = self.submit('a', 'print("second validation")')
        self.assertEqual(self.result(again)['returncode'], 0)

    def test_multi_gpu_waiter_releases_partial_set_and_overlaps_serialize(self):
        os.makedirs(self.cfg['lock_dir'])
        with open(Path(self.cfg['lock_dir'], 'gpu_7.lock'), 'w') as foreign:
            fcntl.flock(foreign, fcntl.LOCK_EX)
            big = self.submit('a', 'print("multi")', count=2)
            single = self.submit('b', 'import time;time.sleep(.3)')
            self.assertEqual(self.result(single)['gpus'], [3])
            self.assertFalse((big / 'lease.json').exists())
            self.assert_free(count=1)
        self.assertEqual(self.result(big)['gpus'], [3, 7])
        self.assert_free()

    def test_failure_timeout_and_detached_workers_drain_before_unlock(self):
        failed = self.submit('a', 'raise SystemExit(17)')
        self.assertEqual(self.result(failed)['returncode'], 17)
        self.assert_free()
        child = self.tasks['a'] / 'child.pid'
        child_code = 'import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path(%r).write_text(str(os.getpid()));time.sleep(600)' % str(child)
        # This worker escapes the command's process group. The GPU supervisor's
        # subreaper must adopt and kill it even when the foreground command succeeds.
        code = 'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",%r],start_new_session=True);time.sleep(.3)' % child_code
        detached = self.submit('a', code)
        self.wait(child.exists)
        pid = int(child.read_text())
        self.assertIsNotNone(Q.identity(pid))
        self.assertEqual(self.result(detached)['returncode'], 0)
        self.assertIsNone(Q.identity(pid))
        self.assert_free()
        child.unlink()
        timed = self.submit('a', code + ';time.sleep(600)', timeout=1)
        self.wait(child.exists)
        pid = int(child.read_text())
        self.assertEqual(self.result(timed)['returncode'], 124)
        self.assertIsNone(Q.identity(pid))
        self.assert_free()

    def test_queued_cancel_and_author_death(self):
        os.makedirs(self.cfg['lock_dir'])
        with open(Path(self.cfg['lock_dir'], 'gpu_7.lock'), 'w') as foreign:
            fcntl.flock(foreign, fcntl.LOCK_EX)
            queued = self.submit('a', 'raise RuntimeError("must not run")', count=2)
            (queued / 'cancel').touch()
            self.assertEqual(self.result(queued)['returncode'], 130)
            self.assertFalse((queued / 'lease.json').exists())
        running = self.submit('b', 'import time;time.sleep(600)')
        self.wait(lambda: (running / 'lease.json').exists())
        self.local.stop('b')
        self.assertEqual(self.result(running)['returncode'], 130)
        self.assert_free()

    def test_controller_death_drains_gpu_jobs_and_cpu_authors(self):
        job = self.submit('a', 'import time;time.sleep(600)', count=2)
        self.wait(lambda: (job / 'lease.json').exists())
        self.parent.kill()
        self.parent.wait(5)
        self.assertEqual(self.result(job)['returncode'], 130)
        self.broker.wait(10)
        for name in self.tasks:
            self.assertFalse(self.local.alive(name))
        self.assert_free()

    def test_source_audit_failure_cannot_be_accepted_as_a_passing_command(self):
        job = self.submit('a', 'from pathlib import Path;Path(".geak").mkdir();'
                          'Path(".geak/invalid_measurements.jsonl").write_text("{}\\n");print("PASS")')
        self.assertEqual(self.result(job)['returncode'], 86)
        self.assertFalse(Q.last_command_passed(str(self.wave), 'a'))
        self.assert_free()


if __name__ == '__main__':
    unittest.main()
