#!/usr/bin/env python3
"""Lifecycle tests for bench_e2e.sh measurement replicas."""

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
class BenchReplicaLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bench_replica_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.events = os.path.join(self.tmp, "events.jsonl")
        self.adapter = os.path.join(self.tmp, "fake_adapter.sh")
        with open(self.adapter, "w", encoding="utf-8") as fh:
            fh.write(
                textwrap.dedent(
                    r"""
                    adapter_default_port() { echo 18080; }
                    adapter_launch() {
                      printf '{"event":"launch","replica":%s,"attempt":%s}\n' \
                        "${REPLICA_INDEX:-0}" "${REPLICA_ATTEMPT:-0}" >> "$EVENT_LOG"
                      ${SERVER_LAUNCH_PREFIX:-} sleep 300 &
                      SERVER_PID=$!
                    }
                    adapter_health() { return 0; }
                    adapter_bench() {
                      local nump="$1" maxc="$2"
                      printf '{"event":"bench","replica":%s,"attempt":%s,"nump":%s,"conc":%s}\n' \
                        "${REPLICA_INDEX:-0}" "${REPLICA_ATTEMPT:-0}" "$nump" "$maxc" >> "$EVENT_LOG"
                      case ",${FAIL_ATTEMPTS:-}," in
                        *",${REPLICA_INDEX:-0}:${REPLICA_ATTEMPT:-0},"*) return 9 ;;
                      esac
                      printf '{"output_throughput":%s,"median_ttft_ms":%s,"median_tpot_ms":%s}\n' \
                        "$((REPLICA_INDEX * 100))" "$((REPLICA_INDEX * 10))" "$REPLICA_INDEX" \
                        >> "$RESULT_JSONL"
                    }
                    """
                ).lstrip()
            )

    def run_bench(self, *, purpose="search", repeats=None, **extra_env):
        out_dir = os.path.join(self.tmp, f"out_{len(os.listdir(self.tmp))}")
        env = dict(os.environ)
        env.update(
            ADAPTER=self.adapter,
            BACKEND="fake",
            MODEL=os.path.join(self.tmp, "model"),
            OUT_DIR=out_dir,
            EVENT_LOG=self.events,
            GEAK_REPEAT_MODE="isolated_server",
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
            [BASH, BENCH],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        with open(os.path.join(out_dir, "bench_summary.json"), encoding="utf-8") as fh:
            summary = json.load(fh)
        return proc, summary, self.read_events()

    def read_events(self):
        if not os.path.exists(self.events):
            return []
        with open(self.events, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def test_search_defaults_to_one_fresh_replica_without_outer_warmup(self):
        proc, summary, events = self.run_bench(purpose="search")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(summary["requested"], 1)
        self.assertEqual(summary["successful"], 1)
        self.assertEqual(summary["status"], "complete")
        self.assertTrue(summary["usable_for_acceptance"])
        self.assertEqual(summary["measurement_mode"], "isolated_server")
        self.assertEqual(summary["observed_median"], 100.0)
        self.assertEqual(sum(e["event"] == "launch" for e in events), 1)
        benches = [e for e in events if e["event"] == "bench"]
        self.assertEqual(len(benches), 1)
        self.assertEqual([e["nump"] for e in benches], [7])

    def test_warm_reuse_replica_discards_full_round_on_one_owned_server(self):
        proc, summary, events = self.run_bench(
            purpose="search", GEAK_REPEAT_MODE="warm_reuse_server"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(summary["measurement_mode"], "warm_reuse_server")
        self.assertEqual(summary["observed_median"], 100.0)
        self.assertEqual(summary["warmup_output_throughput_tok_s"], 100.0)
        self.assertEqual(summary["warmup_runs"], 1)
        lifecycle = summary["measurement_lifecycle"]
        self.assertEqual(lifecycle["owner"], "geak")
        self.assertEqual(lifecycle["launches"], 1)
        self.assertTrue(lifecycle["same_server_for_warmup_and_measure"])
        self.assertTrue(lifecycle["warmup_is_full_round"])
        self.assertEqual(lifecycle["warmup_prompt_count"], 7)
        self.assertEqual((lifecycle["warmup_seed"], lifecycle["timed_seed"]), (0, 1))
        self.assertTrue(lifecycle["port"].isdigit())
        self.assertEqual(sum(e["event"] == "launch" for e in events), 1)
        benches = [e for e in events if e["event"] == "bench"]
        self.assertEqual([e["nump"] for e in benches], [7, 7])

    def test_warm_reuse_validation_keeps_replicas_independent(self):
        proc, summary, events = self.run_bench(
            purpose="validation", GEAK_REPEAT_MODE="warm_reuse_server"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual((summary["requested"], summary["successful"]), (3, 3))
        self.assertEqual(summary["measurement_mode"], "warm_reuse_server")
        self.assertEqual(sum(e["event"] == "launch" for e in events), 3)
        self.assertEqual(sum(e["event"] == "bench" for e in events), 6)

    def test_parity_defaults_to_one_replica(self):
        proc, summary, _ = self.run_bench(purpose="parity")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual((summary["requested"], summary["successful"]), (1, 1))

    def test_validation_defaults_to_three_and_observes_replica_median(self):
        proc, summary, events = self.run_bench(purpose="validation")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual((summary["requested"], summary["successful"]), (3, 3))
        self.assertEqual(summary["observed_median"], 200.0)
        self.assertEqual(sum(e["event"] == "launch" for e in events), 3)
        self.assertEqual(sum(e["event"] == "bench" for e in events), 3)

    def test_explicit_repeats_overrides_purpose_default(self):
        proc, summary, _ = self.run_bench(purpose="validation", repeats=2)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual((summary["requested"], summary["successful"]), (2, 2))

    def test_explicit_replicas_overrides_purpose_default(self):
        proc, summary, _ = self.run_bench(purpose="validation", REPLICAS=2)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual((summary["requested"], summary["successful"]), (2, 2))

    def test_failed_attempt_gets_one_fresh_server_retry(self):
        proc, summary, events = self.run_bench(FAIL_ATTEMPTS="1:1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(summary["successful"], 1)
        self.assertEqual(summary["replicas"][0]["attempt"], 2)
        self.assertEqual(sum(e["event"] == "launch" for e in events), 2)

    def test_exhausted_retry_is_incomplete_but_keeps_successful_median(self):
        proc, summary, events = self.run_bench(
            purpose="validation", repeats=2, FAIL_ATTEMPTS="2:1,2:2"
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual((summary["requested"], summary["successful"]), (2, 1))
        self.assertEqual(summary["status"], "incomplete")
        self.assertFalse(summary["usable_for_acceptance"])
        self.assertEqual(summary["observed_median"], 100.0)
        self.assertEqual(sum(e["event"] == "launch" for e in events), 3)
        self.assertIn("without same-server fallback", proc.stderr)

    def test_isolated_mode_rejects_server_reuse(self):
        env = dict(os.environ)
        env.update(
            GEAK_REPEAT_MODE="isolated_server",
            MODEL="unused",
            OUT_DIR=os.path.join(self.tmp, "reject_reuse"),
            REUSE_SERVER="1",
        )
        proc = subprocess.run(
            [BASH, BENCH], env=env, capture_output=True, text=True, timeout=10
        )
        self.assertEqual(proc.returncode, 4)
        self.assertIn("isolated", proc.stderr.lower())

    def test_warm_reuse_mode_rejects_external_server_reuse(self):
        env = dict(os.environ)
        env.update(
            GEAK_REPEAT_MODE="warm_reuse_server",
            MODEL="unused",
            OUT_DIR=os.path.join(self.tmp, "reject_warm_reuse"),
            REUSE_SERVER="1",
        )
        proc = subprocess.run(
            [BASH, BENCH], env=env, capture_output=True, text=True, timeout=10
        )
        self.assertEqual(proc.returncode, 4)
        self.assertIn("warm_reuse_server", proc.stderr)

    def test_profile_uses_single_server_lifecycle_in_aligned_run(self):
        env = dict(os.environ)
        env.update(
            ADAPTER=self.adapter,
            BACKEND="fake",
            MODEL=os.path.join(self.tmp, "model"),
            OUT_DIR=os.path.join(self.tmp, "profile"),
            EVENT_LOG=self.events,
            GEAK_REPEAT_MODE="isolated_server",
            PROFILE="1",
            REPEATS="1",
            NUM_PROMPTS="7",
            CONC="3",
            SERVING_GPU_LOCK_DISABLE="1",
            SERVER_STOP_GRACE_S="0",
        )
        proc = subprocess.run(
            [BASH, BENCH], env=env, capture_output=True, text=True, timeout=60
        )
        # The fake adapter does not implement the real profiler hooks, but the
        # globally exported alignment mode must not reject PROFILE=1 at the
        # replica scheduler boundary.
        self.assertNotEqual(proc.returncode, 4)
        self.assertEqual(sum(e["event"] == "launch" for e in self.read_events()), 1)
        self.assertIn("single-server profiling lifecycle", proc.stdout)

    def test_inferencex_uses_protocol_warmups_and_seed_for_measured_call(self):
        calls = os.path.join(self.tmp, "inferencex_calls.jsonl")
        fake_client = os.path.join(self.tmp, "fake_inferencex.py")
        with open(fake_client, "w", encoding="utf-8") as fh:
            fh.write(
                textwrap.dedent(
                    """
                    import json
                    import os
                    import sys

                    def value(flag):
                        return sys.argv[sys.argv.index(flag) + 1]

                    call = {
                        "num_prompts": int(value("--num-prompts")),
                        "num_warmups": int(value("--num-warmups")),
                        "seed": int(value("--seed")),
                        "trust_remote_code": "--trust-remote-code" in sys.argv,
                    }
                    with open(os.environ["IX_CALLS"], "a") as out:
                        out.write(json.dumps(call) + "\\n")
                    result_dir = value("--result-dir")
                    os.makedirs(result_dir, exist_ok=True)
                    with open(os.path.join(result_dir, value("--result-filename")), "w") as out:
                        json.dump({
                            "output_throughput": 123,
                            "mean_ttft_ms": 4,
                            "mean_tpot_ms": 5,
                        }, out)
                    """
                ).lstrip()
            )
        proc, summary, _ = self.run_bench(
            BENCH_CLIENT="inferencex",
            INFERENCEX_BENCH_SERVING=fake_client,
            IX_CALLS=calls,
            NUM_WARMUPS="999",
            SEED="999",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(calls, encoding="utf-8") as fh:
            observed = [json.loads(line) for line in fh]
        self.assertEqual(len(observed), 1)
        self.assertEqual([item["num_prompts"] for item in observed], [7])
        self.assertEqual([item["num_warmups"] for item in observed], [6])
        self.assertEqual([item["seed"] for item in observed], [0])
        self.assertEqual([item["trust_remote_code"] for item in observed], [False])
        self.assertEqual(summary["observed_median"], 123.0)

        proc, summary, _ = self.run_bench(
            GEAK_REPEAT_MODE="warm_reuse_server",
            BENCH_CLIENT="inferencex",
            INFERENCEX_BENCH_SERVING=fake_client,
            IX_CALLS=calls,
            NUM_WARMUPS="999",
            SEED="999",
            EXTRA_ENV="BENCH_TRUST_REMOTE_CODE=1",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(calls, encoding="utf-8") as fh:
            observed = [json.loads(line) for line in fh]
        self.assertEqual(len(observed), 3)
        self.assertEqual([item["num_prompts"] for item in observed], [7, 7, 7])
        self.assertEqual([item["num_warmups"] for item in observed], [6, 6, 6])
        # Isolated mode retains seed=0. Warm-reuse's default 0/1 corpus pair
        # prevents the timed pass from directly replaying warmup prefixes.
        self.assertEqual([item["seed"] for item in observed], [0, 0, 1])
        self.assertEqual([item["trust_remote_code"] for item in observed], [False, True, True])
        self.assertEqual(summary["measurement_mode"], "warm_reuse_server")

        proc, _, _ = self.run_bench(
            GEAK_REPEAT_MODE="warm_reuse_server",
            BENCH_CLIENT="inferencex",
            INFERENCEX_BENCH_SERVING=fake_client,
            IX_CALLS=calls,
            GEAK_WARMUP_SEED="17",
            GEAK_TIMED_SEED="19",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(calls, encoding="utf-8") as fh:
            observed = [json.loads(line) for line in fh]
        self.assertEqual([item["seed"] for item in observed[-2:]], [17, 19])


if __name__ == "__main__":
    unittest.main(verbosity=2)
