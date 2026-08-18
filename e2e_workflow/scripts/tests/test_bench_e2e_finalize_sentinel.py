#!/usr/bin/env python3
"""Unit tests for bench_e2e.sh's finalize-deadline stand-down.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_bench_e2e_finalize_sentinel.py

WHY THESE EXIST: the orchestrator reserves a share of its wall-clock budget for the
closing Finalize -> Report -> Validate phases, but the workflow script can only stop
AWAITING an abandoned optimization agent — it cannot kill it. An orphaned integrate
bench therefore keeps running, holds the single serving slot behind the per-GPU lock
(default wait 7200s) and starves the Director's validation, which is how a run
reached its hard kill with no director_e2e_validation.json on disk at all.

run_e2e.py creates $GEAK_FINALIZE_NOW_FILE when that window opens; this dispatcher is
the only place that can make an already-running OPTIONAL leg let go. The closing
phases themselves set GEAK_TAIL_LEG=1 and must stay exempt — they ARE the window.

The stand-down sits before the serving-GPU lock and before any launch, so these tests
need neither a GPU nor a model: they assert the exit code and the stderr reason.
"""
import os
import shutil
import subprocess
import tempfile
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(SCRIPTS_DIR, "bench_e2e.sh")
LIB = os.path.join(SCRIPTS_DIR, "server_teardown.sh")
ADAPTERS = os.path.join(SCRIPTS_DIR, "adapters")

BASH = shutil.which("bash")
STAND_DOWN_RC = 5


@unittest.skipIf(BASH is None, "bash is required to exercise the shell dispatcher")
class BenchFinalizeSentinelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bench_sentinel_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.eval_dir = os.path.join(self.tmp, "eval")
        os.makedirs(self.eval_dir)
        # Stage exactly what roles/director.md stages, so the teardown-contract gate
        # (which sits before this one) passes and we reach the stand-down.
        shutil.copy(BENCH, os.path.join(self.eval_dir, "bench_e2e.sh"))
        shutil.copytree(ADAPTERS, os.path.join(self.eval_dir, "adapters"))
        shutil.copy(LIB, os.path.join(self.eval_dir, "server_teardown.sh"))

        self.adapter = os.path.join(self.tmp, "stub_adapter.sh")
        with open(self.adapter, "w", encoding="utf-8") as fh:
            fh.write(
                "adapter_default_port() { echo 18081; }\n"
                "adapter_launch() { echo 'STUB_LAUNCH_REACHED'; exit 0; }\n"
                "adapter_health() { return 0; }\n"
                "adapter_bench() { return 0; }\n"
            )
        self.sentinel = os.path.join(self.eval_dir, ".geak_finalize_now")

    def arm(self):
        with open(self.sentinel, "w", encoding="utf-8") as fh:
            fh.write("finalize window opened; optional legs must stand down\n")

    def run_bench(self, **env):
        run_env = dict(os.environ)
        run_env.pop("SKILL_DIR", None)
        run_env.pop("WORKFLOW_DIR", None)
        run_env.pop("GEAK_FINALIZE_NOW_FILE", None)
        run_env.pop("GEAK_EVAL_DIR", None)
        run_env.pop("GEAK_TAIL_LEG", None)
        run_env.update(
            ADAPTER=self.adapter,
            BACKEND="vllm",
            MODEL=os.path.join(self.tmp, "model"),
            OUT_DIR=os.path.join(self.tmp, "out"),
            REPEATS="1",
            PROFILE="0",
            # Keep the serving-GPU lock out of the way: these tests are about the
            # stand-down decision, which is asserted independently below.
            SERVING_GPU_LOCK_DISABLE="1",
        )
        run_env.update(env)
        return subprocess.run(
            [BASH, os.path.join(self.eval_dir, "bench_e2e.sh")],
            env=run_env, capture_output=True, text=True, timeout=120,
        )

    # ── inert by default ───────────────────────────────────────────────────────
    def test_no_sentinel_no_change(self):
        """Nothing about the budget is known here, so the run must be untouched."""
        proc = self.run_bench()
        self.assertNotEqual(proc.returncode, STAND_DOWN_RC, proc.stderr[-2000:])
        self.assertIn("STUB_LAUNCH_REACHED", proc.stdout, "never reached the launch")

    def test_unset_env_with_a_file_present_is_still_inert(self):
        """A stale file must not stop a run that was never told where to look — the
        path comes from the runner, so an unrelated dir cannot hijack the decision."""
        self.arm()
        proc = self.run_bench()
        self.assertNotEqual(proc.returncode, STAND_DOWN_RC, proc.stderr[-2000:])
        self.assertIn("STUB_LAUNCH_REACHED", proc.stdout)

    # ── stand-down ────────────────────────────────────────────────────────────
    def test_optional_leg_stands_down_on_the_explicit_path(self):
        self.arm()
        proc = self.run_bench(GEAK_FINALIZE_NOW_FILE=self.sentinel)
        self.assertEqual(proc.returncode, STAND_DOWN_RC, proc.stderr[-2000:])
        self.assertIn("finalize deadline active", proc.stderr)
        self.assertNotIn("STUB_LAUNCH_REACHED", proc.stdout,
                         "an optional leg launched a server inside the reserve")

    def test_eval_dir_fallback_resolves_the_sentinel(self):
        """run_e2e.py exports GEAK_EVAL_DIR too, so the dispatcher resolves the
        sentinel even from a copy that predates GEAK_FINALIZE_NOW_FILE."""
        self.arm()
        proc = self.run_bench(GEAK_EVAL_DIR=self.eval_dir)
        self.assertEqual(proc.returncode, STAND_DOWN_RC, proc.stderr[-2000:])
        self.assertIn("finalize deadline active", proc.stderr)

    def test_armed_but_absent_file_does_not_stand_down(self):
        """The path being KNOWN is not the signal; the file EXISTING is."""
        proc = self.run_bench(GEAK_FINALIZE_NOW_FILE=self.sentinel)
        self.assertNotEqual(proc.returncode, STAND_DOWN_RC, proc.stderr[-2000:])
        self.assertIn("STUB_LAUNCH_REACHED", proc.stdout)

    # ── the closing phases are exempt ─────────────────────────────────────────
    def test_tail_leg_is_exempt(self):
        """Finalize/Validate benches ARE the reserved window; the sentinel must never
        stand THEM down, or the guard would destroy what it exists to protect."""
        self.arm()
        proc = self.run_bench(GEAK_FINALIZE_NOW_FILE=self.sentinel, GEAK_TAIL_LEG="1")
        self.assertNotEqual(proc.returncode, STAND_DOWN_RC, proc.stderr[-2000:])
        self.assertIn("STUB_LAUNCH_REACHED", proc.stdout,
                      "the deliverable bench was blocked by its own reserve")

    # ── the serving-GPU lock wait must be interruptible ───────────────────────
    def test_lock_wait_gives_up_when_the_window_opens(self):
        """The old code did ONE `flock -w 7200`, so an optional leg could sit on the
        budget for 2h. The wait is now sliced, and the sentinel cuts it short."""
        lock = os.path.join(self.tmp, "serving.lock")
        with open(lock, "w", encoding="utf-8") as fh:
            fh.write("")
        self.arm()
        # Hold the lock for the whole test so the wait loop is what we observe.
        holder = subprocess.Popen(
            [BASH, "-c", f'exec 9>"{lock}"; flock 9; sleep 30'],
        )
        self.addCleanup(holder.kill)
        proc = self.run_bench(
            GEAK_FINALIZE_NOW_FILE=self.sentinel,
            GEAK_TAIL_LEG="1",              # exempt at the pre-launch gate ...
            SERVING_GPU_LOCK_DISABLE="0",
            SERVING_GPU_LOCK=lock,
            SERVING_LOCK_WAIT="4",
            SERVING_LOCK_POLL="1",
        )
        # ... so the only way out is the lock timeout, which must be the SLICED
        # budget (~4s), not a 2h block. The timeout=120 above is the real assertion.
        self.assertIn(proc.returncode, (4, 5), proc.stderr[-2000:])
        self.assertNotIn("STUB_LAUNCH_REACHED", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
