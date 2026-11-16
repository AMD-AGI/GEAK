"""CPU tests for scripts/ut_wave.py: planning, and a whole wave on the local backend with a
stub runner standing in for the agent (runners/CONTRACT.md interface).

Stub tasks write synthetic receipts; no real GPU is touched. Stdlib only.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts")
sys.path.insert(0, SCRIPTS)
import ut_wave as W  # noqa: E402
import ut_gpu_queue as Q

# A task's UT stand-in records any accidental timing request.
FAKE_UT = textwrap.dedent('''\
    import json, os, sys, time
    here = os.path.dirname(os.path.abspath(__file__))
    gates = ["contract", "correctness", "negative_controls", "restore"]
    measuring = "--skip" not in sys.argv and "correctness" not in sys.argv
    if measuring:
        gates.append("timing")
    with open(os.path.join(here, "run", "ut_calls.log"), "a") as f:
        f.write(" ".join(sys.argv[1:]) + " GPU_IDS=%s\\n" % os.environ.get("GPU_IDS"))
    if measuring:
        with open(os.path.join(here, "run", "timing.trace"), "a") as f:
            f.write("start %s %s\\n" % (time.time(), os.environ.get("ROCR_VISIBLE_DEVICES")))
    verdict = "pass" if "timing" in gates or "correctness" in sys.argv else "partial"
    rep = {"gates": {g: {"status": "pass"} for g in gates}, "verdict": verdict,
           "validation_mode": "performance" if measuring else "correctness" if "correctness" in sys.argv else "performance"}
    time.sleep(0.3)
    json.dump(rep, open(os.path.join(here, "run", "ut_report.json"), "w"))
    json.dump({"verdict": verdict, "exit_code": 0 if verdict == "pass" else 1},
              open(os.path.join(here, "run", "ut_result.json"), "w"))
    if measuring:
        with open(os.path.join(here, "run", "timing.trace"), "a") as f:
            f.write("end %s\\n" % time.time())
    ''')

# The stub agent: behaviour per task label, chosen by a file in the task directory.
STUB_RUNNER = textwrap.dedent('''\
    #!/usr/bin/env bash
    PROMPT="$1"; WORKDIR="$2"; LOG="$3"
    T="$TASK_DIR"
    echo "start $(date +%s.%N) ROCR=$ROCR_VISIBLE_DEVICES GPU_IDS=$GPU_IDS" >> "$T/run/lane.trace"
    mode="$(cat "$T/mode")"
    SKIP="--validation-mode correctness --reuse"
    grep -q "THIS IS ATTEMPT" "$PROMPT" && echo continuation >> "$T/run/lane.trace"
    sleep 1
    run_ut() {
      python3 "$SKILL_ROOT/scripts/ut_wave.py" gpu -- python3 "$T/unittest.py" $SKIP
    }
    case "$mode" in
      ok)      run_ut ;;
      block)   python3 "$SKILL_ROOT/scripts/ut_wave.py" block --task-dir "$T" --reason "no fixture X: ls shows it" ;;
      runner78) exit 78 ;;
      second)  if [ -f "$T/run/tried" ]; then run_ut; else touch "$T/run/tried"; fi ;;
      never)   : ;;
    esac
    echo "end $(date +%s.%N)" >> "$T/run/lane.trace"
    ''')


class WaveTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="utwave_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.eval = os.path.join(self.tmp, "eval")
        self.runner = os.path.join(self.tmp, "stub.sh")
        with open(self.runner, "w") as f:
            f.write(STUB_RUNNER)
        os.makedirs(self.eval)
        with open(os.path.join(self.eval, "serving.json"), "w") as f:
            json.dump({"gpus": "0,1,2,3", "env": {"SGLANG_USE_AITER": "1"}}, f)

    def tasks(self, spec):
        """spec: [(label, family, cases, need, mode)] -> task_plan.json + task dirs."""
        rows = []
        for label, family, cases, need, mode in spec:
            tdir = os.path.join(self.eval, "tasks", label)
            os.makedirs(os.path.join(tdir, "run"))
            os.makedirs(os.path.join(tdir, "kernel_src"))
            with open(os.path.join(tdir, "unittest.py"), "w") as f:
                f.write(FAKE_UT)
            with open(os.path.join(tdir, "mode"), "w") as f:
                f.write(mode)
            with open(os.path.join(tdir, "run", "requested_kernel.json"), "w") as f:
                json.dump({"required_gpu_count": need}, f)
            rows.append({"label": label, "task_dir": tdir, "ut_family": family,
                         "planned_cases": {"total": cases}})
        with open(os.path.join(self.eval, "task_plan.json"), "w") as f:
            json.dump({"tasks": rows}, f)

    def wave(self, *extra):
        p = subprocess.run([sys.executable, os.path.join(SCRIPTS, "ut_wave.py"), "start",
                            "--eval-dir", self.eval, "--runner", self.runner, "--backend", "local",
                            "--foreground", "--tick-sec", "1", "--lane-sec", "120", *extra],
                           capture_output=True, text=True, timeout=300,
                           env=dict(os.environ, KERNEL_EXTRACT_LOCK_DIR=os.path.join(self.tmp, 'locks'),
                                    GEAK_GPU_REQUIRE_IDLE='0', KERNEL_ENV_KEEP_ARCH='1'))
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(os.path.join(self.eval, "ut_wave", "summary.json")) as f:
            return json.load(f)["tasks"], p.stdout

    def trace(self, label):
        with open(os.path.join(self.eval, "tasks", label, "run", "lane.trace")) as f:
            return f.read()

    def test_correctness_wave_finishes_without_any_timing(self):
        self.tasks([("gemm", "gemm", 3, 1, "ok"), ("norm", "rmsnorm", 2, 1, "ok")])
        res, out = self.wave()
        self.assertEqual({r["status"] for r in res.values()}, {"passed"}, out)
        for name in res:
            td = os.path.join(self.eval, "tasks", name)
            self.assertFalse(os.path.exists(os.path.join(td, "run", "timing.trace")))
            self.assertTrue(W.passed(td))

    def test_blockers_runner_78_and_a_second_attempt(self):
        self.tasks([("a", "gemm", 10, 1, "block"), ("b", "gemm", 10, 1, "runner78"),
                    ("c", "gemm", 10, 1, "second"), ("d", "gemm", 10, 1, "never")])
        res, out = self.wave("--attempts", "2")
        self.assertEqual(res["a"]["status"], "blocked")
        self.assertIn("no fixture X", res["a"]["why"])
        self.assertEqual(res["b"]["status"], "blocked")
        self.assertIn("78", res["b"]["why"])
        self.assertEqual((res["c"]["status"], res["c"]["attempts"]), ("passed", 2))
        self.assertIn("continuation", self.trace("c"))
        self.assertEqual((res["d"]["status"], res["d"]["attempts"]), ("failed", 2))


    def planned(self):
        p = subprocess.run([sys.executable, os.path.join(SCRIPTS, "ut_wave.py"), "plan",
                            "--eval-dir", self.eval], capture_output=True, text=True)
        return [ln.split()[0] for ln in p.stdout.splitlines() if ln.startswith("  ")]

    def test_passing_tasks_are_skipped_and_a_stale_receipt_is_not(self):
        self.tasks([("a", "gemm", 10, 1, "ok"), ("b", "gemm", 10, 1, "ok")])
        self.wave()
        self.assertEqual(self.planned(), [])
        time.sleep(0.05)
        os.utime(os.path.join(self.eval, "tasks", "a", "unittest.py"))   # a changed after its receipt
        self.assertEqual(self.planned(), ["a"])

    def test_on_demand_authors_overlap_without_gpu_and_request_only_needed_cards(self):
        W.save(os.path.join(self.eval, 'serving.json'), {'gpus': '3,7'})
        self.tasks([('a', 'gemm', 10, 1, 'ok'), ('b', 'gemm', 10, 1, 'ok'),
                    ('ar', 'all_reduce', 10, 2, 'ok')])
        res, out = self.wave('--gpus', '3,7')
        self.assertEqual({r['status'] for r in res.values()}, {'passed'}, out)
        spans = []
        for name in res:
            self.assertEqual(res[name]['gpus'], [])
            start, end = self.trace(name).splitlines()
            self.assertIn('ROCR= GPU_IDS=', start)
            spans.append((float(start.split()[1]), float(end.split()[1])))
        self.assertLess(max(s for s, _ in spans), min(e for _, e in spans))
        paths = list(Path(self.eval, 'ut_wave', 'gpu_queue').glob('*/request.json'))
        self.assertEqual(len(paths), 3)
        for path in paths:
            req = W.load(str(path))
            result = W.load(str(path.with_name('result.json')))
            self.assertEqual(result['returncode'], 0)
            self.assertTrue(set(result['gpus']) <= {3, 7})
            self.assertEqual(len(result['gpus']), 2 if req['task'] == 'ar' else 1)
            calls = Path(self.eval, 'tasks', req['task'], 'run', 'ut_calls.log').read_text()
            self.assertIn('GPU_IDS=0,1' if req['task'] == 'ar' else 'GPU_IDS=0', calls)
            self.assertFalse(Path(self.eval, 'tasks', req['task'], 'run', 'timing.trace').exists())


    def test_stopped_on_demand_wave_drains_before_resuming_same_task(self):
        self.tasks([('a', 'gemm', 10, 1, 'ok')])
        ut = Path(self.eval, 'tasks', 'a', 'unittest.py')
        ut.write_text('''import os, time
once = os.path.join(os.path.dirname(__file__), 'run', 'once')
if not os.path.exists(once):
    open(once, 'w').close()
    time.sleep(600)
''' + FAKE_UT)
        first, _ = self.wave('--max-wall-sec', '3')
        self.assertEqual(first['a']['status'], 'running')
        queue = Path(self.eval, 'ut_wave', 'gpu_queue')
        old = list(queue.glob('*/result.json'))
        self.assertEqual(len(old), 1)
        self.assertEqual(W.load(str(old[0]))['returncode'], 130)
        res, out = self.wave()
        self.assertEqual(res['a']['status'], 'passed', out)
        results = list(queue.glob('*/result.json'))
        self.assertEqual(len(results), 2)
        self.assertTrue(Q.last_command_passed(str(queue.parent), 'a'))

    def test_failed_gpu_envelope_prevents_receipt_reuse_and_force_retries(self):
        self.tasks([('a', 'gemm', 10, 1, 'ok')])
        self.wave()
        self.assertEqual(self.planned(), [])
        result = next(Path(self.eval, 'ut_wave', 'gpu_queue').glob('*/result.json'))
        state = W.load(str(result))
        state['returncode'] = 86  # simulate a post-command source audit invalidating its receipt
        W.save(str(result), state)
        self.assertEqual(self.planned(), ['a'])
        res, out = self.wave('--force')
        self.assertEqual(res['a']['status'], 'passed', out)
        self.assertTrue(Q.last_command_passed(str(result.parent.parent.parent), 'a'))

    def test_group_request_does_not_hold_a_partial_set(self):
        import fcntl
        locks = Path(self.tmp, 'locks')
        locks.mkdir()
        with open(locks / 'gpu_7.lock', 'w') as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            self.assertEqual(Q.try_lease([3, 7], 2, str(locks), lambda _: True), ([], []))
            gpus, fds = Q.try_lease([3], 1, str(locks), lambda _: True)
            self.assertEqual(gpus, [3])
            Q.release(fds)
        gpus, fds = Q.try_lease([3, 7], 2, str(locks), lambda _: True)
        self.assertEqual(gpus, [3, 7])
        Q.release(fds)

    def test_cpu_docker_has_no_devices_and_gpu_container_has_only_leased_nodes(self):
        backend = W.Docker(self.tmp, 'pinned:image')
        with mock.patch.dict(os.environ, {'UT_WAVE_CONTAINER_PREFIX': ''}):
            self.assertNotEqual(W.Docker(self.tmp, 'image').cname('a'),
                                W.Docker(self.tmp + '/other', 'image').cname('a'))
        with mock.patch.object(W.subprocess, 'run') as run, mock.patch.object(W.os.path, 'exists', return_value=True):
            run.return_value = subprocess.CompletedProcess([], 0, stdout='', stderr='')
            backend.start('author', [], ['worker.py'], {'ROCR_VISIBLE_DEVICES': '0,1,2,3'})
            cpu = run.call_args.args[0]
            self.assertNotIn('--device', cpu)
            self.assertIn('ROCR_VISIBLE_DEVICES=', cpu)
            backend.start('gpu_job', [3, 7], ['worker.py'], {'ROCR_VISIBLE_DEVICES': '0,1,2,3'})
            gpu = run.call_args.args[0]
            devices = [gpu[i + 1] for i, arg in enumerate(gpu) if arg == '--device']
            self.assertEqual(devices, ['/dev/kfd', '/dev/dri/renderD152', '/dev/dri/renderD184'])
            self.assertIn('GPU_IDS=0,1', gpu)
            self.assertNotIn('ROCR_VISIBLE_DEVICES=0,1,2,3', gpu)

    def pool_plan(self, gpus=None):
        args = SimpleNamespace(eval_dir=self.eval, gpus=gpus, tasks=None, force=False)
        return W.cmd_plan(args, quiet=True)[1]

    def test_gpu_pool_inherits_workload_and_can_only_be_narrowed(self):
        self.tasks([('a', 'gemm', 10, 1, 'ok')])
        for allowed in ('', '0,1,2,3,4,5,6,7'):
            with self.subTest(allowed=allowed), mock.patch.dict(
                    os.environ, {'GEAK_GPU_ALLOWED': allowed, 'ROCR_VISIBLE_DEVICES': ''}):
                self.assertEqual(self.pool_plan(), [0, 1, 2, 3])
                self.assertEqual(self.pool_plan('1,3'), [1, 3])
                for gpus in ('0,1,2,3,4', '4,5', '0,7'):
                    with self.assertRaisesRegex(SystemExit, 'exceed workload'):
                        self.pool_plan(gpus)

    def test_scheduler_allocation_can_further_restrict_workload_pool(self):
        self.tasks([('a', 'gemm', 10, 1, 'ok')])
        with mock.patch.dict(os.environ, {'GEAK_GPU_ALLOWED': '0,1', 'ROCR_VISIBLE_DEVICES': ''}):
            self.assertEqual(self.pool_plan('0'), [0])
            with self.assertRaisesRegex(SystemExit, 'GEAK_GPU_ALLOWED'):
                self.pool_plan('2')
        with mock.patch.dict(os.environ, {'GEAK_GPU_ALLOWED': '', 'ROCR_VISIBLE_DEVICES': '2,3'}):
            self.assertEqual(self.pool_plan('2'), [2])
            with self.assertRaisesRegex(SystemExit, 'inherited ROCR'):
                self.pool_plan('1')

    def test_parent_workload_and_gpu_zero_still_bound_explicit_pool(self):
        self.tasks([('a', 'gemm', 10, 1, 'ok')])
        Path(self.eval, 'serving.json').unlink()
        W.save(os.path.join(self.tmp, 'serving.json'), {'gpus': 0})
        with mock.patch.dict(os.environ, {'GEAK_GPU_ALLOWED': '', 'ROCR_VISIBLE_DEVICES': ''}):
            self.assertEqual(self.pool_plan(), [0])
            with self.assertRaisesRegex(SystemExit, 'exceed workload'):
                self.pool_plan('1')

    def test_invalid_workload_allocation_cannot_be_bypassed_with_explicit_gpus(self):
        for value in ('', None, '0,0', '0,,1', [], 'bad'):
            with self.subTest(value=value):
                W.save(os.path.join(self.eval, 'serving.json'), {'gpus': value})
                with self.assertRaisesRegex(SystemExit, 'serving.json'):
                    self.pool_plan('0')

if __name__ == "__main__":
    unittest.main()
