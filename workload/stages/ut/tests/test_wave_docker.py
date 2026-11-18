"""Opt-in real Docker regressions. No GPU devices or scheduler jobs are used.

Set GEAK_TEST_DOCKER_IMAGE to the serving image and GEAK_TEST_SHARED_TMP to an
owned directory beneath /shared_nfs. Run as the non-root host broker user.
"""
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

SLOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SLOT / 'kernel-ut/scripts'))
import ut_wave as W
import ut_gpu_queue as Q


@unittest.skipUnless(os.environ.get('GEAK_TEST_DOCKER_IMAGE'), 'opt-in Docker integration')
class DockerHandoffTest(unittest.TestCase):
    def setUp(self):
        self.assertNotEqual(os.getuid(), 0, 'run as the non-root host broker user')
        self.image = os.environ['GEAK_TEST_DOCKER_IMAGE']
        temp = tempfile.TemporaryDirectory(dir=os.environ['GEAK_TEST_SHARED_TMP'], prefix='wave_docker_')
        self.addCleanup(temp.cleanup)
        self.wave = Path(temp.name)
        self.assertTrue(str(self.wave).startswith('/shared_nfs/'))
        for sub in ['logs', 'state', 'gpu_queue', 'task/run']:
            (self.wave / sub).mkdir(parents=True)

    def wait_file(self, path, deadline=60):
        end = time.monotonic() + deadline
        while not path.exists() and time.monotonic() < end:
            time.sleep(.1)
        self.assertTrue(path.exists(), str(path))

    def test_author_submit_broker_finish_and_worker_share_host_identity(self):
        wave = str(self.wave)
        cfg = {'rows': [{'label': 'smoke', 'task_dir': wave + '/task', 'need': 1}],
               'gpus': [0], 'backend': 'docker', 'image': self.image,
               'render_base': 128, 'render_step': 8, 'lock_dir': wave + '/locks', 'env': {},
               'container_prefix': W.default_container_prefix(wave)}
        W.save(wave + '/wave.json', cfg)
        W.save(wave + '/state/smoke.token', {'task': 'smoke', 'token': 'fixture-token'})
        author = self.wave / 'author.py'
        author.write_text('''import os,sys,json,getpass
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, %r)
import ut_gpu_queue as Q
wave=Path(os.environ['UT_WAVE_DIR'])
(wave/'task/run/.ut_writer.lock').touch()
(wave/'identity.json').write_text(json.dumps({'uid':os.getuid(),'gid':os.getgid()}))
assert getpass.getuser() == os.environ['USER']
Path('/tmp/aiter_configs/writable').touch()
assert os.environ['FLYDSL_RUNTIME_CACHE_DIR'].startswith(str(wave))
cache=Path(os.environ['PYTORCH_KERNEL_CACHE_PATH'])
assert not str(cache).startswith('/shared_nfs/')
(cache/'exp2_kernel_archgfx950:sramecc+:xnack-_tmp_12').write_text('cache')
assert (cache/'exp2_kernel_archgfx950:sramecc+:xnack-_tmp_12').read_text() == 'cache'
# The deadline expires before the broker can acquire devices. This exercises
# real submit, liveness, finish and result transport without any GPU access.
rc=Q.submit(SimpleNamespace(count=1,timeout=.000001,command=['true'],env=[],cwd=None))
(wave/'author_result.json').write_text(json.dumps({'rc':rc}))
''' % str(SLOT / 'kernel-ut/scripts'))
        backend = W.Docker(wave, self.image)
        with (self.wave / 'logs/broker.log').open('w') as log:
            broker = subprocess.Popen([sys.executable, Q.__file__, 'broker', wave, str(os.getpid())],
                                      stdout=log, stderr=subprocess.STDOUT)
        try:
            self.wait_file(self.wave / 'broker.json')
            env = dict(UT_WAVE_DIR=wave, UT_WAVE_OWNER='smoke', UT_WAVE_TASK='smoke',
                       UT_WAVE_TOKEN='fixture-token')
            self.assertTrue(backend.start('smoke', [], [str(author)], env))
            self.wait_file(self.wave / 'author_result.json')
            self.assertEqual(W.load(wave + '/author_result.json'), {'rc': 124})
            self.assertEqual(W.load(wave + '/identity.json')['uid'], os.getuid())
            self.assertEqual((self.wave / 'homes/smoke/.cache/aiter-configs/writable').stat().st_uid, os.getuid())
            results = list((self.wave / 'gpu_queue').glob('*/result.json'))
            self.assertEqual(len(results), 1)
            self.assertEqual(W.load(str(results[0]))['gpus'], [])
            for p in [self.wave / 'task/run/.ut_writer.lock', results[0].parent, results[0]]:
                self.assertEqual(p.stat().st_uid, os.getuid())
            self.assertFalse(list((self.wave / 'gpu_queue').glob('*/lease.json')))
            # GPU workers use the same backend/identity. Exercise its file writes
            # without granting devices, since this test only covers permissions.
            self.assertTrue(backend.start('gpu_fixture', [], ['-c',
                'from pathlib import Path; Path(%r).write_text("worker")' % (wave + '/worker.txt')],
                {'UT_WAVE_TASK': 'smoke'}))
            self.wait_file(self.wave / 'worker.txt')
            self.assertEqual((self.wave / 'worker.txt').stat().st_uid, os.getuid())
        finally:
            backend.stop('smoke')
            backend.stop('gpu_fixture')
            broker.terminate()
            broker.wait(timeout=30)

    def test_root_0644_lock_excludes_broker_and_shell_without_inode_replacement(self):
        # Locks are host-local, as in production; task files use the shared mount.
        with tempfile.TemporaryDirectory(prefix='geak_lock_test_') as tmp:
            lock = Path(tmp) / 'gpu_99.lock'
            ready = Path(tmp) / 'ready'
            name = 'geak_lock_test_' + uuid.uuid4().hex[:12]
            code = ('import fcntl,os,time; from pathlib import Path; '
                    'f=open(%r,"w"); os.chmod(%r,0o644); '
                    'fcntl.flock(f,fcntl.LOCK_EX); Path(%r).touch(); time.sleep(90)') % (
                        str(lock), str(lock), str(ready))
            proc = subprocess.run(['docker', 'run', '-d', '--name', name, '--user', '0:0',
                '-v', '%s:%s' % (tmp, tmp), self.image, 'python3', '-c', code], capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            shell = None
            try:
                self.wait_file(ready)
                self.assertEqual(lock.stat().st_uid, 0)
                inode = lock.stat().st_ino
                self.assertEqual(Q.try_lease([99], 1, tmp, lambda _: True), ([], []))
                env = dict(os.environ, KERNEL_EXTRACT_LOCK_DIR=tmp, GEAK_GPU_ALLOWED='99',
                           GEAK_GPU_REQUIRE_IDLE='0', KERNEL_ENV_SKIP_ENUM_REAP='1', KERNEL_ENV_KEEP_ARCH='1')
                marker = self.wave / 'shell_done'
                shell = subprocess.Popen(['bash', str(SLOT.parent / '_shared/gpu_lock.sh'), '99',
                    sys.executable, '-c', 'from pathlib import Path; Path(%r).touch()' % str(marker)],
                    cwd=self.wave, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                time.sleep(.5)
                self.assertFalse(marker.exists())
                subprocess.run(['docker', 'rm', '-f', name], capture_output=True, check=True)
                output, _ = shell.communicate(timeout=15)
                self.assertEqual(shell.returncode, 0, output)
                self.assertTrue(marker.exists())
                gpus, fds = Q.try_lease([99], 1, tmp, lambda _: True)
                Q.release(fds)
                self.assertEqual(gpus, [99])
                self.assertEqual(lock.stat().st_ino, inode)
                self.assertEqual(lock.stat().st_mode & 0o777, 0o644)
            finally:
                subprocess.run(['docker', 'rm', '-f', name], capture_output=True)
                if shell and shell.poll() is None:
                    shell.terminate()
                    shell.communicate(timeout=15)


if __name__ == '__main__':
    unittest.main()
