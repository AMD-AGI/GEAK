#!/usr/bin/env python3
"""Lifecycle tests for bench_e2e.sh GEAK_REPEAT_MODE=warm_server.

Warm mode is the Hyperloom-aligned protocol: ONE server per leg, one FULL
untimed round to populate the prefix cache, then N timed rounds on that hot
server.  What these tests pin down is the part that is easy to regress
silently -- that the warmup is a full round, that it is discarded, that only
one server is ever launched, and that the summary still carries the same
acceptance contract (status / successful_replicas / usable_for_acceptance)
the isolated-server aggregate publishes.
"""

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest


SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(SCRIPTS_DIR, "bench_e2e.sh")
BASH = shutil.which("bash")


@unittest.skipIf(BASH is None, "bash is required to exercise the shell dispatcher")
class BenchWarmServerLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bench_warm_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.events = os.path.join(self.tmp, "events.jsonl")
        self.counter = os.path.join(self.tmp, "call_counter")
        self.adapter = os.path.join(self.tmp, "fake_adapter.sh")
        # Each bench call gets a distinct throughput (call_index * 100) so a
        # test can tell exactly WHICH calls landed in the timed median.
        with open(self.adapter, "w", encoding="utf-8") as fh:
            fh.write(
                textwrap.dedent(
                    r"""
                    adapter_default_port() { echo 18081; }
                    adapter_launch() {
                      printf '{"event":"launch"}\n' >> "$EVENT_LOG"
                      ${SERVER_LAUNCH_PREFIX:-} sleep 300 &
                      SERVER_PID=$!
                    }
                    adapter_health() { return 0; }
                    adapter_bench() {
                      local nump="$1" maxc="$2" n
                      n=$(( $(cat "$CALL_COUNTER" 2>/dev/null || echo 0) + 1 ))
                      printf '%s' "$n" > "$CALL_COUNTER"
                      printf '{"event":"bench","call":%s,"nump":%s,"conc":%s}\n' \
                        "$n" "$nump" "$maxc" >> "$EVENT_LOG"
                      case ",${FAIL_CALLS:-}," in
                        *",$n,"*) return 9 ;;
                      esac
                      printf '{"output_throughput":%s,"median_ttft_ms":%s,"median_tpot_ms":%s}\n' \
                        "$((n * 100))" "$((n * 10))" "$n" >> "$RESULT_JSONL"
                    }
                    """
                ).lstrip()
            )

    def run_bench(self, *, purpose="search", repeats=None, expect_rc=0, **extra_env):
        out_dir = os.path.join(self.tmp, f"out_{len(os.listdir(self.tmp))}")
        env = dict(os.environ)
        env.update(
            ADAPTER=self.adapter,
            BACKEND="fake",
            MODEL=os.path.join(self.tmp, "model"),
            OUT_DIR=out_dir,
            EVENT_LOG=self.events,
            CALL_COUNTER=self.counter,
            GEAK_REPEAT_MODE="warm_server",
            MEASUREMENT_PURPOSE=purpose,
            NUM_PROMPTS="7",
            CONC="3",
            PROFILE="0",
            REUSE_SERVER="0",
            SERVING_GPU_LOCK_DISABLE="1",
            SERVER_STOP_GRACE_S="0",
        )
        env.pop("REPEATS", None)
        if repeats is not None:
            env["REPEATS"] = str(repeats)
        env.update({key: str(value) for key, value in extra_env.items()})
        proc = subprocess.run(
            [BASH, BENCH], env=env, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(proc.returncode, expect_rc, proc.stderr)
        summary_path = os.path.join(out_dir, "bench_summary.json")
        summary = None
        if os.path.exists(summary_path):
            with open(summary_path, encoding="utf-8") as fh:
                summary = json.load(fh)
        return proc, summary, self.read_events()

    def read_events(self):
        if not os.path.exists(self.events):
            return []
        with open(self.events, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def test_validation_is_hyperloom_two_passes_and_reports_the_second(self):
        # Hyperloom's protocol exactly: one boot, warmup_round discarded,
        # measure_round on the re-attached hot server IS the number.
        proc, summary, events = self.run_bench(purpose="validation")
        # ONE boot -- this is the whole point of the mode.
        self.assertEqual(sum(e["event"] == "launch" for e in events), 1)
        benches = [e for e in events if e["event"] == "bench"]
        # Exactly 2 passes, and the warmup is a FULL round (NUM_PROMPTS, not CONC).
        self.assertEqual([e["nump"] for e in benches], [7, 7])
        self.assertEqual(summary["measurement_mode"], "warm_server")
        self.assertEqual(summary["measurement_purpose"], "validation")
        self.assertEqual(summary["dispersion_basis"], "within_server_rounds")
        # The warmup (call 1 -> 100) must NOT be in the timed results: the
        # SECOND pass (call 2 -> 200) is the reported number.
        self.assertEqual(summary["all_throughput"], [200.0])
        self.assertEqual(summary["throughput_tok_s_median"], 200.0)
        self.assertEqual(summary["observed_median"], 200.0)
        self.assertIn("measurement_mode=warm_server", proc.stdout)

    def test_acceptance_contract_matches_the_isolated_aggregate(self):
        _, summary, _ = self.run_bench(purpose="validation")
        self.assertEqual(summary["requested_rounds"], 1)
        self.assertEqual(summary["successful_rounds"], 1)
        # Same field names the isolated-server aggregate publishes, so
        # director:validate reads one shape regardless of lifecycle.
        self.assertEqual(summary["requested_replicas"], 1)
        self.assertEqual(summary["successful_replicas"], 1)
        self.assertEqual(summary["status"], "complete")
        self.assertTrue(summary["usable_for_acceptance"])

    def test_search_defaults_to_one_timed_round(self):
        _, summary, events = self.run_bench(purpose="search")
        self.assertEqual(sum(e["event"] == "launch" for e in events), 1)
        self.assertEqual(sum(e["event"] == "bench" for e in events), 2)  # warmup + 1
        self.assertEqual(summary["requested_rounds"], 1)
        self.assertEqual(summary["all_throughput"], [200.0])

    def test_explicit_replicas_overrides_purpose_default(self):
        _, summary, events = self.run_bench(purpose="validation", REPLICAS=2)
        self.assertEqual(summary["requested_rounds"], 2)
        self.assertEqual(sum(e["event"] == "bench" for e in events), 3)  # warmup + 2

    def test_explicit_repeats_overrides_purpose_default(self):
        _, summary, _ = self.run_bench(purpose="validation", repeats=2)
        self.assertEqual((summary["requested_rounds"], summary["successful_rounds"]), (2, 2))

    def test_failed_timed_round_is_incomplete_but_keeps_the_observable_median(self):
        # Call 1 is the warmup; fail the second TIMED round (call 3).
        _, summary, events = self.run_bench(
            purpose="validation", REPLICAS=3, FAIL_CALLS="3"
        )
        self.assertEqual(summary["requested_rounds"], 3)
        self.assertEqual(summary["successful_rounds"], 2)
        self.assertEqual(summary["status"], "incomplete")
        # A degraded leg stays observable but must never be silently promoted.
        self.assertFalse(summary["usable_for_acceptance"])
        self.assertEqual(summary["all_throughput"], [200.0, 400.0])
        # Still only one boot: warm mode never re-launches mid-leg.
        self.assertEqual(sum(e["event"] == "launch" for e in events), 1)

    def test_failed_full_warmup_aborts_instead_of_reporting_a_cold_number(self):
        proc, summary, _ = self.run_bench(
            purpose="validation", FAIL_CALLS="1", expect_rc=2
        )
        self.assertIsNone(summary)
        self.assertIn("warmup failed", proc.stderr)

    def test_rejects_a_non_positive_round_count(self):
        proc, _, _ = self.run_bench(purpose="validation", repeats=0, expect_rc=4)
        self.assertIn("positive integer", proc.stderr)

    def test_caller_policy_overrides_the_mode_the_role_forwarded(self):
        # The validating role forwarded isolated_server (e.g. it echoed the
        # globally exported alignment mode); the caller's pin must still win.
        _, summary, events = self.run_bench(
            purpose="validation",
            GEAK_REPEAT_MODE="isolated_server",
            GEAK_VALIDATION_REPEAT_MODE="warm_server",
        )
        self.assertEqual(summary["measurement_mode"], "warm_server")
        self.assertEqual(sum(e["event"] == "launch" for e in events), 1)

    def test_caller_policy_does_not_touch_non_validation_purposes(self):
        _, summary, events = self.run_bench(
            purpose="search",
            GEAK_REPEAT_MODE="isolated_server",
            GEAK_VALIDATION_REPEAT_MODE="warm_server",
            REPEATS=1,
        )
        self.assertEqual(summary["measurement_mode"], "isolated_server")

    def test_naming_no_mode_at_all_still_gets_the_hyperloom_lifecycle(self):
        # A caller that forgot to forward MEASUREMENT_MODE must not silently get
        # a differently-measured number than the rest of the run.
        out_dir = os.path.join(self.tmp, "out_default")
        env = dict(os.environ)
        env.update(
            ADAPTER=self.adapter, BACKEND="fake",
            MODEL=os.path.join(self.tmp, "model"), OUT_DIR=out_dir,
            EVENT_LOG=self.events, CALL_COUNTER=self.counter,
            MEASUREMENT_PURPOSE="search", NUM_PROMPTS="7", CONC="3", PROFILE="0",
            REUSE_SERVER="0", SERVING_GPU_LOCK_DISABLE="1", SERVER_STOP_GRACE_S="0",
        )
        env.pop("GEAK_REPEAT_MODE", None)
        env.pop("REPEATS", None)
        proc = subprocess.run(
            [BASH, BENCH], env=env, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(os.path.join(out_dir, "bench_summary.json"), encoding="utf-8") as fh:
            summary = json.load(fh)
        self.assertEqual(summary["measurement_mode"], "warm_server")
        # warmup + 1 timed, one boot.
        benches = [e for e in self.read_events() if e["event"] == "bench"]
        self.assertEqual([e["nump"] for e in benches], [7, 7])

    def test_repeats_zero_keeps_the_legacy_capture_lifecycle(self):
        # REPEATS=0 is "warmup only, no timed round" -- the shape-capture /
        # profile-window call sites.  The warm-mode default must not turn that
        # into a hard "positive integer" reject.
        out_dir = os.path.join(self.tmp, "out_capture")
        env = dict(os.environ)
        env.update(
            ADAPTER=self.adapter, BACKEND="fake",
            MODEL=os.path.join(self.tmp, "model"), OUT_DIR=out_dir,
            EVENT_LOG=self.events, CALL_COUNTER=self.counter,
            MEASUREMENT_PURPOSE="search", NUM_PROMPTS="7", CONC="3", PROFILE="0",
            REPEATS="0", REUSE_SERVER="0", SERVING_GPU_LOCK_DISABLE="1",
            SERVER_STOP_GRACE_S="0",
        )
        env.pop("GEAK_REPEAT_MODE", None)
        proc = subprocess.run(
            [BASH, BENCH], env=env, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("positive integer", proc.stderr)
        # Short CONC warmup only -- no full round, no timed round.
        benches = [e for e in self.read_events() if e["event"] == "bench"]
        self.assertEqual([e["nump"] for e in benches], [3])

    def test_unknown_purpose_falls_back_to_one_round(self):
        proc, summary, _ = self.run_bench(purpose="nonsense")
        self.assertEqual(summary["requested_rounds"], 1)
        self.assertIn("Unknown MEASUREMENT_PURPOSE", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
