"""CPU tests for scripts/ut_wave.py: planning, and a whole wave on the local backend with a
stub runner standing in for the agent (runners/CONTRACT.md interface).

Each stub task's unittest.py is a fake that writes a receipt, so the timing tail's
`unittest.py --reuse` is exercised as well. Stdlib only.
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
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts")
sys.path.insert(0, SCRIPTS)
import ut_wave as W  # noqa: E402
import ut_gpu_queue as Q

# A task's UT stand-in: --skip timing -> partial receipt without timing; --reuse -> pass.
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
    SKIP="--skip timing"; grep -q "^TIMING .*run every gate, timing included" "$PROMPT" && SKIP=""
    grep -q "^TIMING .*validation-mode correctness" "$PROMPT" && SKIP="--validation-mode correctness --reuse"
    grep -q "THIS IS ATTEMPT" "$PROMPT" && echo continuation >> "$T/run/lane.trace"
    sleep 1
    run_ut() {
      if [ -n "$UT_WAVE_TOKEN" ]; then
        python3 "$SKILL_ROOT/scripts/ut_wave.py" gpu -- python3 "$T/unittest.py" $SKIP
      else
        python3 "$T/unittest.py" $SKIP
      fi
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
                            "--foreground", "--tick-sec", "1", "--lane-sec", "120", "--timing", "tail",
                            "--gpu-mode", "lane", *extra],
                           capture_output=True, text=True, timeout=300,
                           env=dict(os.environ, KERNEL_EXTRACT_LOCK_DIR=os.path.join(self.tmp, 'locks'),
                                    GEAK_GPU_REQUIRE_IDLE='0', KERNEL_ENV_KEEP_ARCH='1'))
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(os.path.join(self.eval, "ut_wave", "summary.json")) as f:
            return json.load(f)["tasks"], p.stdout

    def trace(self, label):
        with open(os.path.join(self.eval, "tasks", label, "run", "lane.trace")) as f:
            return f.read()

    def test_correctness_wave_finishes_without_any_timing_and_can_upgrade(self):
        self.tasks([("gemm", "gemm", 3, 1, "ok"), ("norm", "rmsnorm", 2, 1, "ok")])
        res, out = self.wave("--timing", "none")
        self.assertEqual({r["status"] for r in res.values()}, {"passed"}, out)
        for name in res:
            td = os.path.join(self.eval, "tasks", name)
            self.assertFalse(os.path.exists(os.path.join(td, "run", "timing.trace")))
            self.assertTrue(W.passed(td, require_timing=False))
            self.assertFalse(W.passed(td, require_timing=True))
        res, out = self.wave("--timing", "tail")
        self.assertEqual({r["status"] for r in res.values()}, {"passed"}, out)
        for name in res:
            self.assertTrue(W.passed(os.path.join(self.eval, "tasks", name)))

    def test_allocation_gives_spare_gpus_to_the_costliest_task(self):
        rows = [{"label": "moe", "need": 1, "cost": 300.0}, {"label": "gemm", "need": 1, "cost": 50.0},
                {"label": "ar", "need": 8, "cost": 10.0}]
        share = W.allocate(rows, 4, cap=4)
        self.assertEqual(share, {"moe": 3, "gemm": 1, "ar": 8})
        self.assertEqual(W.allocate(rows[:2], 8, cap=4), {"moe": 4, "gemm": 4})   # none idle
        many = [{"label": str(i), "need": 1, "cost": float(i)} for i in range(10)]
        self.assertEqual(set(W.allocate(many, 4, cap=4).values()), {1})

    def test_lanes_run_concurrently_pinned_and_are_timed_alone(self):
        self.tasks([("moe", "fused_moe", 100, 1, "ok"), ("gemm", "gemm", 50, 1, "ok"),
                    ("norm", "rmsnorm", 40, 1, "ok")])
        res, out = self.wave()
        self.assertEqual({k: v["status"] for k, v in res.items()},
                         {"moe": "passed", "gemm": "passed", "norm": "passed"}, out)
        spans = {}
        for t in res:
            lines = self.trace(t).split("\n")
            spans[t] = (float(lines[0].split()[1]), float(lines[-2].split()[1]))
        # every lane overlaps every other: they ran at the same time
        self.assertLess(max(s for s, _ in spans.values()), min(e for _, e in spans.values()))
        # 4 GPUs, 3 tasks: moe (largest cost) holds two, each lane sees its GPUs as 0..k-1
        self.assertIn("GPU_IDS=0,1", self.trace("moe"))
        gpus = sorted(res[t]["gpus"] for t in res)
        self.assertEqual(sorted(g for gs in gpus for g in gs), [0, 1, 2, 3])
        # the timing tail ran once per task, with --reuse, on all four GPUs
        with open(os.path.join(self.eval, "tasks", "gemm", "run", "ut_calls.log")) as f:
            calls = f.read()
        self.assertIn("--reuse GPU_IDS=0,1,2,3", calls)
        with open(os.path.join(self.eval, "ut_wave", "wave.json")) as f:
            self.assertEqual(json.load(f)["env"]["SGLANG_USE_AITER"], "1")

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

    def test_parallel_timing_waits_for_authoring_and_uses_disjoint_gpus(self):
        self.tasks([("a", "gemm", 10, 1, "ok"), ("b", "gemm", 10, 1, "ok"),
                    ("c", "gemm", 10, 1, "ok")])
        res, out = self.wave("--timing-workers", "2")
        self.assertEqual({v["status"] for v in res.values()}, {"passed"}, out)
        timing = {}
        end_of_authoring = max(float(self.trace(t).splitlines()[-1].split()[1]) for t in res)
        for task in res:
            with open(os.path.join(self.eval, "tasks", task, "run", "timing.trace")) as f:
                start, end = [line.split() for line in f]
            timing[task] = (float(start[1]), float(end[1]), set(start[2].split(",")))
            self.assertGreater(float(start[1]), end_of_authoring)
        overlaps = []
        for a, va in timing.items():
            for b, vb in timing.items():
                if a < b and max(va[0], vb[0]) < min(va[1], vb[1]):
                    overlaps.append((a, b))
                    self.assertFalse(va[2] & vb[2])
        self.assertTrue(overlaps, out)

    def test_timing_allocator_respects_multi_rank_and_worker_limits(self):
        a = {"label": "a", "need": 1, "cost": 100}
        b = {"label": "b", "need": 1, "cost": 20}
        ar = {"label": "ar", "need": 4, "cost": 10}
        self.assertEqual(W.timing_dispatch([ar], [0, 1], 2), [])
        self.assertEqual(W.timing_dispatch([ar], [0, 1, 2, 3], 2, 1), [])
        self.assertEqual(W.timing_dispatch([a], [2, 3], 2, 2), [])
        allocation = W.timing_dispatch([a, b], [0, 1, 2, 3], 2)
        self.assertEqual(len(allocation), 2)
        self.assertEqual(sorted(g for _, gs in allocation for g in gs), [0, 1, 2, 3])

    def test_a_multi_gpu_task_waits_for_the_single_gpu_lanes(self):
        self.tasks([("ar", "all_reduce", 5, 4, "ok"), ("g1", "gemm", 10, 1, "ok"),
                    ("g2", "gemm", 10, 1, "ok")])
        res, out = self.wave("--timing", "lane")
        self.assertEqual({v["status"] for v in res.values()}, {"passed"}, out)
        self.assertEqual(res["ar"]["gpus"], [0, 1, 2, 3])
        ar_start = float(self.trace("ar").split("\n")[0].split()[1])
        for t in ("g1", "g2"):
            self.assertGreater(ar_start, float(self.trace(t).split("\n")[-2].split()[1]))

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
        self.tasks([('a', 'gemm', 10, 1, 'ok'), ('b', 'gemm', 10, 1, 'ok'),
                    ('ar', 'all_reduce', 10, 2, 'ok')])
        res, out = self.wave('--gpu-mode', 'on-demand', '--gpus', '3,7', '--timing', 'none')
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

    def test_on_demand_timing_tail_and_retry(self):
        self.tasks([('a', 'gemm', 10, 1, 'ok'), ('b', 'gemm', 10, 1, 'second')])
        res, out = self.wave('--gpu-mode', 'on-demand', '--gpus', '2,5', '--timing-workers', '2')
        self.assertEqual({r['status'] for r in res.values()}, {'passed'}, out)
        self.assertEqual(res['b']['attempts'], 2)
        end_authoring = max(float(self.trace(n).splitlines()[-1].split()[1]) for n in res)
        for name in res:
            timing = Path(self.eval, 'tasks', name, 'run', 'timing.trace').read_text()
            self.assertGreater(float(timing.splitlines()[0].split()[1]), end_authoring)

    def test_stopped_on_demand_wave_drains_before_resuming_same_task(self):
        self.tasks([('a', 'gemm', 10, 1, 'ok')])
        ut = Path(self.eval, 'tasks', 'a', 'unittest.py')
        ut.write_text('''import os, time
once = os.path.join(os.path.dirname(__file__), 'run', 'once')
if not os.path.exists(once):
    open(once, 'w').close()
    time.sleep(600)
''' + FAKE_UT)
        first, _ = self.wave('--gpu-mode', 'on-demand', '--timing', 'none', '--max-wall-sec', '3')
        self.assertEqual(first['a']['status'], 'running')
        queue = Path(self.eval, 'ut_wave', 'gpu_queue')
        old = list(queue.glob('*/result.json'))
        self.assertEqual(len(old), 1)
        self.assertEqual(W.load(str(old[0]))['returncode'], 130)
        res, out = self.wave('--gpu-mode', 'on-demand', '--timing', 'none')
        self.assertEqual(res['a']['status'], 'passed', out)
        results = list(queue.glob('*/result.json'))
        self.assertEqual(len(results), 2)
        self.assertTrue(Q.last_command_passed(str(queue.parent), 'a'))

    def test_failed_gpu_envelope_prevents_receipt_reuse_and_force_retries(self):
        self.tasks([('a', 'gemm', 10, 1, 'ok')])
        self.wave('--gpu-mode', 'on-demand', '--timing', 'none')
        self.assertEqual(self.planned(), [])
        result = next(Path(self.eval, 'ut_wave', 'gpu_queue').glob('*/result.json'))
        state = W.load(str(result))
        state['returncode'] = 86  # simulate a post-command source audit invalidating its receipt
        W.save(str(result), state)
        self.assertEqual(self.planned(), ['a'])
        res, out = self.wave('--gpu-mode', 'on-demand', '--timing', 'none', '--force')
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

if __name__ == "__main__":
    unittest.main()
