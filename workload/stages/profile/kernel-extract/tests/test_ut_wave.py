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

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts")
sys.path.insert(0, SCRIPTS)
import ut_wave as W  # noqa: E402

# A task's UT stand-in: --skip timing -> partial receipt without timing; --reuse -> pass.
FAKE_UT = textwrap.dedent('''\
    import json, os, sys, time
    here = os.path.dirname(os.path.abspath(__file__))
    gates = ["contract", "correctness", "negative_controls", "restore"]
    if "--skip" not in sys.argv:
        gates.append("timing")
    with open(os.path.join(here, "run", "ut_calls.log"), "a") as f:
        f.write(" ".join(sys.argv[1:]) + " GPU_IDS=%s\\n" % os.environ.get("GPU_IDS"))
    verdict = "pass" if "timing" in gates else "partial"
    rep = {"gates": {g: {"status": "pass"} for g in gates}, "verdict": verdict}
    time.sleep(0.05)
    json.dump(rep, open(os.path.join(here, "run", "ut_report.json"), "w"))
    json.dump({"verdict": verdict, "exit_code": 0 if verdict == "pass" else 1},
              open(os.path.join(here, "run", "ut_result.json"), "w"))
    ''')

# The stub agent: behaviour per task label, chosen by a file in the task directory.
STUB_RUNNER = textwrap.dedent('''\
    #!/usr/bin/env bash
    PROMPT="$1"; WORKDIR="$2"; LOG="$3"
    T="$TASK_DIR"
    echo "start $(date +%s.%N) ROCR=$ROCR_VISIBLE_DEVICES GPU_IDS=$GPU_IDS" >> "$T/run/lane.trace"
    mode="$(cat "$T/mode")"
    SKIP="--skip timing"; grep -q "run every gate, timing included" "$PROMPT" && SKIP=""
    grep -q "THIS IS ATTEMPT" "$PROMPT" && echo continuation >> "$T/run/lane.trace"
    sleep 1
    case "$mode" in
      ok)      python3 "$T/unittest.py" $SKIP ;;
      block)   python3 "$SKILL_ROOT/scripts/ut_wave.py" block --task-dir "$T" --reason "no fixture X: ls shows it" ;;
      runner78) exit 78 ;;
      second)  if [ -f "$T/run/tried" ]; then python3 "$T/unittest.py" $SKIP; else touch "$T/run/tried"; fi ;;
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
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(os.path.join(self.eval, "ut_wave", "summary.json")) as f:
            return json.load(f)["tasks"], p.stdout

    def trace(self, label):
        with open(os.path.join(self.eval, "tasks", label, "run", "lane.trace")) as f:
            return f.read()

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

if __name__ == "__main__":
    unittest.main()
