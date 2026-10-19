#!/usr/bin/env python3
"""Unit tests for run_reaper.py -- what a GEAK run stops when it hands its GPUs over.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_run_reaper.py

WHY THESE EXIST: a search agent abandoned at the reserve boundary keeps running, and so does everything it
launched detached -- a tuner sweep, an A/B leg, a script polling for idle GPUs to start the next leg. The
final phase then measured beside that work, and the caller's rebench, booted after the run returned, failed
its memory-balance check on GPUs a leftover leg still held. The reaper is what stops that work, so the rules
under test are WHICH PROCESSES IT CHOOSES:

  fence   the run's GPU holders, serving-lock holders and detached processes, with their descendants
          -- never the live workflow's own attached tools, never another run's processes
  exit    everything the run owns: its marker, or its eval dir for a process started after the run
  always  never PID 1, never the reaper or any of its ancestors, never a pid whose start time moved

Selection is asserted on process tables built by hand, so no rule depends on how this host reparents
orphans. Stopping is asserted on real processes this test starts, carrying a marker minted for the test,
next to bystanders that must survive.
"""
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("run_reaper", os.path.join(SCRIPTS_DIR, "run_reaper.py"))
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)

HAS_PROC = os.path.isdir("/proc/self/fd")


def P(pid, ppid, *, marked=False, in_eval_dir=False, gpu=False, lock=False, zombie=False, name="x"):
    return rr.Proc(pid, ppid, 1000 + pid, zombie, marked, in_eval_dir, gpu, lock, name)


class SelectionTest(unittest.TestCase):
    # PID 1 is the container init; 5 the caller's orchestrator; 10 the runner; 11 the Claude CLI; 12 the
    # fence agent's shell; 13 the reaper itself. All of them are the reaper's ancestry, so all are kept;
    # the live workflow is the runner and what runs under it.
    CHAIN = (13, 12, 11, 10, 5, 1)
    KEEP = frozenset(CHAIN)
    WORKFLOW = frozenset({10, 11, 12, 13})

    def table(self, *rows):
        base = [P(1, 0), P(5, 1), P(10, 5), P(11, 10, marked=True), P(12, 11, marked=True),
                P(13, 12, marked=True, in_eval_dir=True)]
        return {p.pid: p for p in base + list(rows)}

    def chosen(self, procs, mode):
        return {p.pid: why for p, why in rr.select(procs, mode, self.KEEP, self.WORKFLOW)}

    def test_the_workflow_is_the_chain_up_to_the_runner(self):
        self.assertEqual(rr.workflow_pids(self.CHAIN, 10), self.WORKFLOW)
        self.assertEqual(rr.workflow_pids(self.CHAIN, None), frozenset({5, 10, 11, 12, 13}))
        self.assertEqual(rr.workflow_pids(self.CHAIN, 999), frozenset({5, 10, 11, 12, 13}))
        chain = rr.protected_pids()
        self.assertEqual((chain[0], chain[-1]), (os.getpid(), 1))
        self.assertIn(os.getppid(), chain)

    def test_fence_stops_gpu_lock_and_detached_work_with_its_descendants(self):
        procs = self.table(
            P(20, 1, marked=True, in_eval_dir=True, name="bash wait_then_post.sh"),
            P(21, 20, marked=True, name="sleep"),
            P(22, 20, name="cat"),
            P(30, 11, marked=True, lock=True, name="bash bench_e2e.sh"),
            P(31, 30, marked=True, gpu=True, name="python3 sglang.launch_server"),
            P(32, 31, gpu=True),
            P(33, 30, name="tee"),
            P(40, 11, marked=True, gpu=True, name="python3 op_bench.py"),
            P(50, 1, in_eval_dir=True, name="bash leg.sh"),
            P(55, 5, marked=True, name="bash wait_then_post.sh"),
        )
        self.assertEqual(self.chosen(procs, "fence"), {
            20: "detached", 21: "detached", 22: "descendant",
            30: "serving-lock", 31: "gpu", 32: "gpu", 33: "descendant",
            40: "gpu",
            50: "detached",
            55: "detached",
        })

    def test_fence_stops_a_gpu_worker_that_lost_the_marker_but_not_the_shell_above_it(self):
        # A server started from an agent's own shell; its worker renamed itself and wiped its environment.
        procs = self.table(
            P(35, 11, marked=True, name="python3 sglang.launch_server"),
            P(36, 35, gpu=True, name="sglang::scheduler_TP0"),
        )
        self.assertEqual(self.chosen(procs, "fence"), {36: "gpu"})

    def test_fence_spares_the_live_workflow_attached_tools(self):
        procs = self.table(
            P(60, 11, marked=True, name="sleep"),
            P(61, 11, marked=True, in_eval_dir=True, name="tail"),
            P(62, 61, name="cat"),
            P(80, 1, marked=True, gpu=True, zombie=True),
        )
        self.assertEqual(self.chosen(procs, "fence"), {})

    def test_exit_stops_everything_the_run_owns(self):
        procs = self.table(
            P(60, 11, marked=True, name="sleep"),
            P(61, 60, name="cat"),
            P(20, 1, in_eval_dir=True, name="bash wait_then_post.sh"),
            P(70, 1, gpu=True, name="python3 sglang.launch_server"),
        )
        self.assertEqual(self.chosen(procs, "exit"), {20: "run", 60: "run", 61: "descendant"})

    def test_a_protected_process_is_never_chosen_even_under_a_chosen_one(self):
        procs = self.table(P(90, 1, marked=True, name="bash"))
        procs[5] = procs[5]._replace(ppid=90)
        chosen = self.chosen(procs, "exit")
        self.assertEqual(chosen, {90: "run"})
        self.assertFalse(self.KEEP & set(chosen))

    def test_detached_means_no_workflow_process_above_it(self):
        procs = self.table(P(20, 1), P(21, 20), P(22, 11), P(23, 999), P(24, 21), P(25, 5), P(26, 12))
        self.assertTrue(rr._detached(20, procs, self.WORKFLOW))
        self.assertTrue(rr._detached(24, procs, self.WORKFLOW))
        self.assertTrue(rr._detached(23, procs, self.WORKFLOW), "a vanished parent leaves its child detached")
        self.assertTrue(rr._detached(25, procs, self.WORKFLOW), "adopted by a subreaper above the runner")
        self.assertFalse(rr._detached(22, procs, self.WORKFLOW))
        self.assertFalse(rr._detached(26, procs, self.WORKFLOW))

    def test_a_parent_loop_terminates(self):
        procs = self.table(P(20, 21), P(21, 20))
        self.assertTrue(rr._detached(20, procs, self.WORKFLOW))

    def test_without_the_runner_every_protected_process_but_pid_1_is_the_workflow(self):
        procs = self.table(P(20, 1, marked=True, name="bash leg.sh"), P(55, 5, marked=True, name="bash leg.sh"))
        chosen = {p.pid: why for p, why in rr.select(procs, "fence", self.KEEP)}
        self.assertEqual(chosen, {20: "detached"}, "55 runs under the orchestrator, so it is not detached")


class MatchingTest(unittest.TestCase):
    ROOTS = ("/run_ws/s/geak/e2e_cycle0",)

    def test_eval_dir_mentions_respect_path_boundaries(self):
        self.assertTrue(rr._mentions("bash\0/run_ws/s/geak/e2e_cycle0/tuning/ab/wait_then_post.sh", self.ROOTS))
        self.assertTrue(rr._mentions("x\0--out=/run_ws/s/geak/e2e_cycle0", self.ROOTS))
        self.assertTrue(rr._mentions("bash\0-c\0cd '/run_ws/s/geak/e2e_cycle0' && ./leg.sh", self.ROOTS))
        self.assertFalse(rr._mentions("bash\0/run_ws/s/geak/e2e_cycle01/leg.sh", self.ROOTS))
        self.assertFalse(rr._mentions("bash\0/run_ws/s/geak/e2e_cycle0_old/leg.sh", self.ROOTS))

    def test_the_eval_dir_counts_only_where_a_path_starts(self):
        self.assertFalse(rr._mentions("python3\0/backup/run_ws/s/geak/e2e_cycle0/leg.py", self.ROOTS))
        self.assertFalse(rr._mentions("bash\0-c\0cd /data//run_ws/s/geak/e2e_cycle0 && ./leg.sh", self.ROOTS))
        self.assertTrue(rr._mentions("cp\0/backup/run_ws/s/geak/e2e_cycle0/x\0/run_ws/s/geak/e2e_cycle0/x", self.ROOTS))
        self.assertTrue(rr._mentions("env\0PYTHONPATH=/opt/lib:/run_ws/s/geak/e2e_cycle0/lib\0python3", self.ROOTS))
        self.assertTrue(rr._mentions("bash\0-c\0set -e\n/run_ws/s/geak/e2e_cycle0/leg.sh", self.ROOTS))
        self.assertTrue(rr._mentions("bash\0-c\0./leg.sh >/run_ws/s/geak/e2e_cycle0/leg.log", self.ROOTS))

    def test_a_shallow_eval_dir_matches_nothing(self):
        self.assertEqual(rr.eval_dir_roots("/"), ())
        self.assertEqual(rr.eval_dir_roots("/tmp"), ())
        self.assertEqual(rr.eval_dir_roots("/run_ws/s/geak/e2e_cycle0/"), ("/run_ws/s/geak/e2e_cycle0",))

    def test_names_never_carry_flags_values_or_commands(self):
        self.assertEqual(rr._name(["/usr/bin/python3", "-u", "/x/gemm_tuner.py", "--token", "abc"]),
                         "python3 gemm_tuner.py")
        self.assertEqual(rr._name(["python3", "-m", "sglang.launch_server", "--api-key", "k"]),
                         "python3 sglang.launch_server")
        self.assertEqual(rr._name(["bash", "-c", "curl -H 'Authorization: Bearer k' x"]), "bash curl")
        self.assertEqual(rr._name(["bash", "-c", "TOKEN=k ./leg.sh"]), "bash")
        self.assertEqual(rr._name(["/opt/bin/sglang-server", "--api-key", "k"]), "sglang-server")
        self.assertEqual(rr._name([]), "?")


@unittest.skipUnless(HAS_PROC, "the reaper reads /proc")
class StopTest(unittest.TestCase):
    def setUp(self):
        # An ignored SIGTERM survives exec, and run_e2e's result flush ignores it on purpose: a suite that ran that
        # flush in this process first would otherwise start every process below unable to receive the reaper's TERM.
        self.addCleanup(signal.signal, signal.SIGTERM, signal.signal(signal.SIGTERM, signal.SIG_DFL))
        self.root = os.path.realpath(tempfile.mkdtemp(prefix="run_reaper_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.eval_dir = os.path.join(self.root, "geak", "e2e_cycle0")
        os.makedirs(self.eval_dir)
        saved = os.environ.get(rr.MARK_ENV)
        self.addCleanup(self._restore_mark, saved)
        os.environ[rr.MARK_ENV] = "outer"
        self.run = rr.mark_run(self.eval_dir)
        self.marked = dict(os.environ)
        self.plain = {k: v for k, v in os.environ.items() if k != rr.MARK_ENV}
        # A stand-in /dev/kfd and serving-GPU lock, so holding a GPU needs no GPU.
        for name, value in (("KFD", os.path.join(self.root, "kfd")),
                            ("SERVING_LOCK_PREFIX", os.path.join(self.root, "geak_serving_gpu_"))):
            self.addCleanup(setattr, rr, name, getattr(rr, name))
            setattr(rr, name, value)
        open(rr.KFD, "w").close()

    @staticmethod
    def _restore_mark(saved):
        if saved is None:
            os.environ.pop(rr.MARK_ENV, None)
        else:
            os.environ[rr.MARK_ENV] = saved

    def spawn(self, argv, *, env, cwd=None):
        proc = subprocess.Popen(argv, env=env, cwd=cwd or self.root, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(self._kill, proc)
        self._settle(proc.pid)
        return proc

    def _settle(self, pid):
        """Wait out exec: Popen returns before the kernel publishes the new image's environment, and until
        then /proc/<pid>/environ reads empty."""
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with open(f"/proc/{pid}/environ", "rb") as fh:
                    if fh.read(1):
                        return
            except OSError:
                return
            time.sleep(0.01)
        self.fail(f"pid {pid} never published its environment")

    @staticmethod
    def _kill(proc):
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)

    def _ready_file(self):
        self._ready = getattr(self, "_ready", 0) + 1
        return os.path.join(self.root, f"ready_{self._ready}")

    def _read_when_written(self, path):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path) as fh:
                    return fh.read()
            time.sleep(0.02)
        self.fail(f"{path} was never written")

    HOLD = ("import sys, time\nf = open(sys.argv[1], 'a')\n"
            "with open(sys.argv[2], 'w') as fh: fh.write('1')\ntime.sleep(300)\n")

    def holder(self, path, *, env):
        """A process holding ``path`` open, returned once it does."""
        ready = self._ready_file()
        proc = self.spawn([sys.executable, "-c", self.HOLD, path, ready], env=env)
        self._read_when_written(ready)
        return proc

    def tree(self, *, env, child_marked=True, child_holds=None):
        """A parent with one child, which it does not wait for: a ``sleep``, or a holder of ``child_holds``.
        Returns (parent, child pid) once the child has exec'd and holds what it should."""
        ready, held = self._ready_file(), self._ready_file()
        child = ["sleep", "300"]
        if child_holds:
            child = [sys.executable, "-c", self.HOLD, child_holds, held]
        code = ("import os, subprocess, sys, time\nenv = dict(os.environ)\n"
                "if sys.argv[2] == '0': env.pop(sys.argv[3], None)\n"
                "child = subprocess.Popen(sys.argv[4:], env=env)\n"
                "with open(sys.argv[1], 'w') as fh: fh.write(str(child.pid))\ntime.sleep(300)\n")
        proc = self.spawn([sys.executable, "-c", code, ready, "1" if child_marked else "0", rr.MARK_ENV, *child],
                          env=env)
        pid = int(self._read_when_written(ready))
        self.addCleanup(self._kill_pid, pid, rr._stat(pid))
        self._settle(pid)
        if child_holds:
            self._read_when_written(held)
        return proc, pid

    @staticmethod
    def _kill_pid(pid, stat):
        now = rr._stat(pid)
        if stat and now and now[1] == stat[1] and not now[2]:
            os.kill(pid, signal.SIGKILL)

    def reap(self, mode, **kw):
        return rr.reap(self.eval_dir, mode=mode, term_grace_s=5.0, kill_wait_s=5.0, **kw)

    @staticmethod
    def gone(pid):
        st = rr._stat(pid)
        return st is None or st[2]

    def test_mark_run_appends_to_an_inherited_mark_and_records_the_run(self):
        self.assertEqual(os.environ[rr.MARK_ENV], f"outer:{self.run['mark']}")
        with open(os.path.join(self.eval_dir, rr.RUN_FILE)) as fh:
            on_disk = json.load(fh)
        self.assertEqual(on_disk["mark"], self.run["mark"])
        self.assertEqual(on_disk["runner_pid"], os.getpid())
        self.assertEqual(on_disk["start_ticks"], rr._stat(os.getpid())[1])
        child = self.spawn(["sleep", "300"], env=self.marked)
        self.assertTrue(rr._carries(child.pid, self.run["mark"]))
        self.assertTrue(rr._carries(child.pid, "outer"))
        self.assertFalse(rr._carries(child.pid, self.run["mark"][:-1]))

    def test_fence_stops_gpu_and_lock_holders_and_spares_the_rest(self):
        gpu = self.holder(rr.KFD, env=self.marked)
        lock = self.holder(rr.SERVING_LOCK_PREFIX + "6,7.lock", env=self.marked)
        attached = self.spawn(["sleep", "300"], env=self.marked)
        in_dir = self.spawn(["sleep", "300"], env=self.plain, cwd=self.eval_dir)
        strangers_gpu = self.holder(rr.KFD, env=self.plain)

        receipt = self.reap("fence", stage="finalize")

        self.assertTrue(receipt["ok"], receipt)
        self.assertEqual({(row["pid"], row["why"]) for row in receipt["stopped"]},
                         {(gpu.pid, "gpu"), (lock.pid, "serving-lock")})
        self.assertTrue(self.gone(gpu.pid) and self.gone(lock.pid))
        for spared in (attached, in_dir, strangers_gpu):
            self.assertIsNone(spared.poll())
            self.assertFalse(self.gone(spared.pid))
        with open(os.path.join(self.eval_dir, rr.FENCE_FILE)) as fh:
            fence = json.load(fh)
        self.assertEqual((fence["stage"], fence["stopped"]), ("finalize", 2))
        with open(os.path.join(self.eval_dir, rr.LOG_FILE)) as fh:
            self.assertEqual(json.loads(fh.read().splitlines()[-1])["mode"], "fence")

    def test_fence_stops_an_orphan_whoever_adopted_it(self):
        ready = self._ready_file()
        self.spawn(["bash", "-c", 'sleep 300 & echo $! > "$1"', "orphaner", ready], env=self.marked).wait(timeout=10)
        orphan = int(self._read_when_written(ready))
        self.addCleanup(self._kill_pid, orphan, rr._stat(orphan))
        self._settle(orphan)
        attached = self.spawn(["sleep", "300"], env=self.marked)
        receipt = self.reap("fence", stage="finalize")
        self.assertEqual([(row["pid"], row["why"]) for row in receipt["stopped"]], [(orphan, "detached")])
        self.assertTrue(self.gone(orphan))
        self.assertIsNone(attached.poll())

    def test_fence_stops_a_gpu_worker_under_the_runs_process_even_without_the_marker(self):
        server, worker = self.tree(env=self.marked, child_marked=False, child_holds=rr.KFD)
        receipt = self.reap("fence", stage="finalize")
        self.assertEqual([(row["pid"], row["why"]) for row in receipt["stopped"]], [(worker, "gpu")])
        self.assertTrue(self.gone(worker))
        self.assertIsNone(server.poll())

    def test_exit_stops_everything_the_run_started_and_nothing_else(self):
        tree, child = self.tree(env=self.marked)
        unmarked_tree, unmarked_child = self.tree(env=self.marked, child_marked=False)
        in_dir = self.spawn(["sleep", "300"], env=self.plain, cwd=self.eval_dir)
        bystander = self.spawn(["sleep", "300"], env=self.plain)
        backup_leg = self.spawn([sys.executable, "-c", "import time; time.sleep(300)", f"/backup{self.eval_dir}/leg.py"],
                                env=self.plain)

        receipt = self.reap("exit", run=self.run)

        self.assertTrue(receipt["ok"], receipt)
        why = {row["pid"]: row["why"] for row in receipt["stopped"]}
        self.assertEqual(why, {tree.pid: "run", child: "run", unmarked_tree.pid: "run",
                               unmarked_child: "descendant", in_dir.pid: "run"})
        for pid in why:
            self.assertTrue(self.gone(pid), pid)
        self.assertIsNone(bystander.poll())
        self.assertIsNone(backup_leg.poll())

    def test_a_process_older_than_the_run_is_not_taken_for_its_own(self):
        older = self.spawn(["sleep", "300"], env=self.plain, cwd=self.eval_dir)
        run = dict(self.run, start_ticks=rr._stat(older.pid)[1] + 1)
        receipt = self.reap("exit", run=run)
        self.assertEqual(receipt["stopped"], [])
        self.assertIsNone(older.poll())

    def test_a_recycled_pid_is_never_signalled(self):
        victim = self.spawn(["sleep", "300"], env=self.plain)
        ppid, start, _ = rr._stat(victim.pid)
        recorded = rr.Proc(victim.pid, ppid, start + 1, False, True, False, False, False, "sleep")
        stopped, survivors = rr.stop([(recorded, "run")], term_grace_s=0.2, kill_wait_s=0.2)
        self.assertEqual((stopped, survivors), ([], []))
        self.assertIsNone(victim.poll())

    def test_without_a_run_record_nothing_is_stopped_but_the_fence_still_stands(self):
        os.unlink(os.path.join(self.eval_dir, rr.RUN_FILE))
        lock = self.holder(rr.SERVING_LOCK_PREFIX + "0.lock", env=self.marked)
        receipt = self.reap("fence", stage="validate")
        self.assertEqual(receipt["stopped"], [])
        self.assertIn(rr.RUN_FILE, receipt["note"])
        self.assertIsNone(lock.poll())
        self.assertTrue(os.path.isfile(os.path.join(self.eval_dir, rr.FENCE_FILE)))

    def test_a_fault_is_reported_not_raised(self):
        receipt = self.reap("bogus")
        self.assertTrue(receipt["error"])
        self.assertFalse(receipt["ok"])
        self.assertIn("unknown mode", receipt["note"])

    def test_the_cli_finds_the_run_from_its_eval_dir(self):
        leg = self.spawn(["sleep", "300"], env=self.marked)
        bystander = self.spawn(["sleep", "300"], env=self.plain)
        out = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS_DIR, "run_reaper.py"), "exit", "--eval-dir", self.eval_dir,
             "--term-grace-s", "5", "--kill-wait-s", "5"],
            env=self.plain, capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(out.returncode, 0, out.stderr)
        receipt = json.loads(out.stdout)
        self.assertEqual([(row["pid"], row["why"]) for row in receipt["stopped"]], [(leg.pid, "run")])
        self.assertEqual(leg.wait(timeout=10), -signal.SIGTERM)
        self.assertIsNone(bystander.poll())


class FaultTest(unittest.TestCase):
    """/proc changes under the reaper and its own steps fail; it carries on, and the receipt says what happened."""

    LEG = P(4242, 1, marked=True, gpu=True, name="python3 leg.py")

    def setUp(self):
        self.root = os.path.realpath(tempfile.mkdtemp(prefix="run_reaper_fault_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.blocker = os.path.join(self.root, "not_a_dir")
        open(self.blocker, "w").close()

    @unittest.skipUnless(HAS_PROC, "the reaper reads /proc")
    def test_a_process_gone_mid_scan_is_skipped(self):
        with open("/proc/sys/kernel/pid_max") as fh:
            gone = int(fh.read()) + 1  # above pid_max, so never a live pid
        self.assertEqual(rr._argv(gone), [])
        self.assertEqual(rr._open_files(gone), (False, False))
        with mock.patch.object(rr.os, "listdir", return_value=["self", str(gone)]):
            self.assertEqual(rr.snapshot("m", (), None), {})
        with mock.patch.object(rr.Path, "read_text", return_value="42 (sh) S 1"):
            self.assertIsNone(rr._stat(42), "a stat line cut short")

    def test_a_proc_it_cannot_list_yields_no_processes(self):
        with mock.patch.object(rr.os, "listdir", side_effect=OSError("no /proc")):
            self.assertEqual(rr.snapshot("m", ("/run_ws/s/geak/e2e_cycle0",), 0), {})

    def test_an_fd_closed_mid_scan_is_skipped_and_the_scan_ends_once_both_are_found(self):
        targets = {"3": None, "4": rr.KFD, "5": rr.SERVING_LOCK_PREFIX + "0,1.lock", "6": rr.KFD}

        def readlink(path):
            target = targets[path.rsplit("/", 1)[1]]
            if target is None:
                raise FileNotFoundError(path)
            return target

        with mock.patch.object(rr.os, "listdir", return_value=list(targets)), \
                mock.patch.object(rr.os, "readlink", side_effect=readlink) as read:
            self.assertEqual(rr._open_files(4242), (True, True))
        self.assertEqual(read.call_count, 3)

    def test_the_ancestry_still_ends_at_pid_1_when_a_parent_cannot_be_read(self):
        with mock.patch.object(rr, "_stat", return_value=None):
            self.assertEqual(rr.protected_pids(), [os.getpid(), 1])

    def test_a_process_it_may_not_signal_is_not_counted_as_signalled(self):
        for error in (PermissionError, ProcessLookupError):
            with mock.patch.object(rr, "_alive", return_value=True), \
                    mock.patch.object(rr.os, "kill", side_effect=error) as kill:
                self.assertFalse(rr._signal(self.LEG, signal.SIGTERM))
            kill.assert_called_once_with(self.LEG.pid, signal.SIGTERM)

    def test_a_process_that_outlives_sigkill_is_reported_and_ends_the_rounds(self):
        with mock.patch.object(rr, "_signal", return_value=True), mock.patch.object(rr, "_alive", return_value=True):
            stopped, survivors = rr.stop([(self.LEG, "gpu")], term_grace_s=0, kill_wait_s=0)
        self.assertEqual((stopped, survivors), ([], [{"pid": 4242, "name": "python3 leg.py", "why": "gpu"}]))
        with mock.patch.object(rr, "snapshot", return_value={}), \
                mock.patch.object(rr, "select", return_value=[(self.LEG, "gpu")]) as select, \
                mock.patch.object(rr, "stop", return_value=([], survivors)):
            receipt = rr.reap(self.root, mode="exit", run={"mark": "m", "start_ticks": 1})
        self.assertEqual(select.call_count, 1, "no second round once something survives")
        self.assertEqual(receipt["survivors"], survivors)
        self.assertFalse(receipt["ok"])

    def test_a_receipt_it_cannot_record_is_still_returned(self):
        receipt = rr.reap(self.blocker, mode="exit")
        self.assertTrue(receipt["ok"], receipt)
        self.assertIn(rr.RUN_FILE, receipt["note"])

    def test_the_cli_exits_non_zero_when_the_reaper_could_not_do_its_job(self):
        for argv, rc in ((["exit", "--eval-dir", self.root], 0), (["fence", "--eval-dir", self.blocker], 1)):
            with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                self.assertEqual(rr.main(argv), rc)
            self.assertEqual(json.loads(out.getvalue())["mode"], argv[0])


if __name__ == "__main__":
    unittest.main()
