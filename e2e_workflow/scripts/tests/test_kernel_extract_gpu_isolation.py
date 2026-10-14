"""Exercise kernel-extract's real GPU lock with CPU-only placeholder commands."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest


LOCK = Path(__file__).resolve().parents[3] / "kernel_workflow/scripts/gpu_lock.sh"
SET_LOCK = Path(__file__).resolve().parents[1] / "gpu_lock_set.py"


@unittest.skipUnless(shutil.which("bash") and shutil.which("flock"), "requires bash/flock")
class KernelExtractGpuIsolationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="kernel_extract_lock_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Run the real wrappers against a private lock directory. A shared host's
        # /tmp/team_gpu_locks may belong to another user, even for synthetic IDs.
        lock_dir = self.root / "locks"
        lock_dir.mkdir()
        self.lock = self.root / "repo/kernel_workflow/scripts/gpu_lock.sh"
        self.set_lock = self.root / "repo/e2e_workflow/scripts/gpu_lock_set.py"
        self.lock.parent.mkdir(parents=True)
        self.set_lock.parent.mkdir(parents=True)
        source = LOCK.read_text()
        self.assertEqual(source.count('LOCK_DIR="/tmp/team_gpu_locks"'), 1)
        self.lock.write_text(source.replace('LOCK_DIR="/tmp/team_gpu_locks"',
                                            f'LOCK_DIR="{lock_dir}"'))
        (self.lock.parent / "workspace_sources.py").symlink_to(
            LOCK.parent / "workspace_sources.py")
        shutil.copy2(SET_LOCK, self.set_lock)
        # Synthetic IDs keep these CPU-only tests away from real GPU leases.
        self.gpu = str(1000000 + os.getpid() * 2)
        self.other_gpu = str(int(self.gpu) + 1)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GEAK_")}
        self.env.update(
            GEAK_GPU_ALLOWED=f"{self.gpu},{self.other_gpu}",
            GEAK_GPU_REQUIRE_IDLE="0", KERNEL_ENV_KEEP_ARCH="1",
            KERNEL_ENV_SKIP_ENUM_REAP="1",
        )
        self.worker = self.root / "worker.py"
        self.worker.write_text('''import os, pathlib, sys, time
root, name, status = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
(root / (name + '.started')).write_text(os.environ['HIP_VISIBLE_DEVICES'])
deadline = time.monotonic() + 10
while not (root / (name + '.release')).exists():
    if time.monotonic() >= deadline:
        raise SystemExit(99)
    time.sleep(0.01)
raise SystemExit(status)
''')
        self.processes = []
        self.addCleanup(self.stop_workers)

    def stop_workers(self):
        for name, proc in self.processes:
            (self.root / f"{name}.release").touch()
        for _, proc in self.processes:
            try:
                proc.communicate(timeout=12)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()

    def launch(self, name, gpu, status=0, whole_set=False):
        task = self.root / name
        task.mkdir()
        proc = subprocess.Popen(
            (["python3", str(self.set_lock)] if whole_set else ["bash", str(self.lock)]) +
            [gpu, "python3", str(self.worker), str(self.root), name, str(status)],
            cwd=task, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.processes.append((name, proc))
        return proc

    def wait_started(self, name, gpu):
        marker = self.root / f"{name}.started"
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(marker.exists(), f"{name} did not acquire its GPU")
        self.assertEqual(marker.read_text(), gpu, "command is pinned to the locked GPU")

    def finish(self, name, proc, status=0):
        (self.root / f"{name}.release").touch()
        stdout, stderr = proc.communicate(timeout=5)
        self.assertEqual(proc.returncode, status, stdout + stderr)

    def test_disjoint_gpus_validate_concurrently(self):
        first = self.launch("first", self.gpu)
        self.wait_started("first", self.gpu)
        second = self.launch("second", self.other_gpu)
        self.wait_started("second", self.other_gpu)
        self.assertIsNone(first.poll(), "second starts while the first still holds its lease")
        self.finish("first", first)
        self.finish("second", second)

    def test_shared_gpu_serializes_and_failure_releases_lease(self):
        first = self.launch("first", self.gpu, status=7)
        self.wait_started("first", self.gpu)
        second = self.launch("second", self.gpu)
        # First is deliberately held open: second must not reach its GPU command.
        time.sleep(0.25)
        self.assertFalse((self.root / "second.started").exists())
        self.assertIsNone(second.poll())
        self.finish("first", first, status=7)
        self.wait_started("second", self.gpu)
        self.finish("second", second)

    def test_full_set_excludes_each_single_gpu_and_preserves_requested_order(self):
        requested = f"{self.other_gpu},{self.gpu}"
        group = self.launch("group", requested, status=7, whole_set=True)
        self.wait_started("group", requested)
        first = self.launch("first", self.gpu)
        second = self.launch("second", self.other_gpu)
        time.sleep(0.25)
        self.assertFalse((self.root / "first.started").exists())
        self.assertFalse((self.root / "second.started").exists())
        self.finish("group", group, status=7)
        self.wait_started("first", self.gpu)
        self.wait_started("second", self.other_gpu)
        self.finish("first", first)
        self.finish("second", second)

    def test_set_waits_for_busy_member_before_starting_command(self):
        first = self.launch("first", self.other_gpu)
        self.wait_started("first", self.other_gpu)
        requested = f"{self.gpu},{self.other_gpu}"
        group = self.launch("group", requested, whole_set=True)
        time.sleep(0.25)
        self.assertFalse((self.root / "group.started").exists())
        self.finish("first", first)
        self.wait_started("group", requested)
        self.finish("group", group)

    def test_set_refuses_duplicate_or_unallocated_ids(self):
        for requested in [f"{self.gpu},{self.gpu}", f"{self.gpu},999999999"]:
            proc = self.launch("bad" + requested.replace(",", "_"), requested, whole_set=True)
            stdout, stderr = proc.communicate(timeout=5)
            self.assertEqual(proc.returncode, 2, stdout + stderr)


if __name__ == "__main__":
    unittest.main()
