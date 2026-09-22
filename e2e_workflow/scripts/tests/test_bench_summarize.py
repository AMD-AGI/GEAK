#!/usr/bin/env python3
"""Unit tests for bench_summarize.py -- the writer of bench_summary.json.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v

WHY THESE EXIST: this file used to be two Python heredocs inside bench_e2e.sh, one per
lifecycle, and nothing checked that they agreed. Every acceptance decision in the product
reads their output -- the director's validate step, the integrator's A/B, the orchestrator
handoff -- so a renamed key or a status that says "complete" on a short round is a wrong
accept, not a crash. The keys and the E2E_SUMMARY line are therefore pinned here.
"""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARIZE = os.path.join(SCRIPTS_DIR, "bench_summarize.py")

#: Env this module reads. Scrubbed before every call so a caller's shell cannot change
#: what a test measures; `env=` re-adds the ones a test is actually about.
_LIFECYCLE_ENV = ("E2E_METRIC", "WARM_SERVER_ROUNDS", "GEAK_ISOLATED_REPLICA",
                  "MEASUREMENT_PURPOSE", "EFFECTIVE_CONFIG_DIGEST")


def _load_summarize():
    spec = importlib.util.spec_from_file_location("bench_summarize", SUMMARIZE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


summarizer = _load_summarize()


def _run(args, env=None):
    """Drive the CLI in-process and return its stdout.

    A subprocess reports as zero measured coverage even when every branch runs, which
    under the repo's coverage gate is indistinguishable from a script nobody tested.
    ``main()`` already takes argv, so nothing about the CLI contract needs a fork;
    CliEntrypointTest keeps one real process for what only a process has.
    """
    overrides = dict(env or {})
    saved = {name: os.environ.get(name) for name in set(_LIFECYCLE_ENV) | set(overrides)}
    stdout = io.StringIO()
    try:
        for name in _LIFECYCLE_ENV:
            os.environ.pop(name, None)
        os.environ.update(overrides)
        with contextlib.redirect_stdout(stdout):
            summarizer.main(args)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return stdout.getvalue().strip()


class FromRunsTest(unittest.TestCase):
    """bench_runs.jsonl -> summary. The legacy and warm_server lifecycles."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="summarize_runs_")
        self.runs = os.path.join(self.tmp, "bench_runs.jsonl")
        self.out = os.path.join(self.tmp, "bench_summary.json")

    def write_runs(self, path, tputs):
        with open(path, "w", encoding="utf-8") as fh:
            for i, t in enumerate(tputs):
                fh.write(json.dumps({"output_throughput": t,
                                     "total_token_throughput": t * 9,
                                     "median_ttft_ms": 40.0 + i,
                                     "median_tpot_ms": 8.0 + i}) + "\n")
            # A malformed tail must be skipped, not abort the summary: losing the whole
            # measurement to one truncated line is the expensive failure here.
            fh.write("\n{not json}\n")

    def summarize(self, tputs, cold=None, env=None):
        self.write_runs(self.runs, tputs)
        args = [self.runs, self.out]
        if cold is not None:
            cold_path = os.path.join(self.tmp, "cold.jsonl")
            self.write_runs(cold_path, cold)
            args.append(cold_path)
        line = _run(["from-runs"] + args, env)
        with open(self.out, encoding="utf-8") as fh:
            return json.load(fh), line

    def test_median_spread_and_basis(self):
        s, line = self.summarize([100.0, 103.5, 107.0])
        self.assertEqual(s["throughput_tok_s_median"], 103.5)
        self.assertEqual(s["throughput_tok_s_spread_pct"], 6.76)
        self.assertEqual(s["runs"], 3)
        self.assertEqual(s["all_throughput"], [100.0, 103.5, 107.0])
        self.assertEqual(s["metric_basis"], "aggregate_output_tok_s")
        self.assertEqual(s["measurement_mode"], "legacy_same_server")
        self.assertIn("E2E_SUMMARY aggregate_output_tok_s=103.5", line)
        self.assertIn("measurement_mode=legacy_same_server", line)

    def test_single_round_reports_zero_spread_not_null(self):
        """0.0 means 'no spread'; None would read as 'unknown' and gate differently."""
        s, _ = self.summarize([100.0])
        self.assertEqual(s["throughput_tok_s_spread_pct"], 0.0)

    def test_total_metric_nulls_the_output_named_alias(self):
        """Nobody may read total throughput under an output_* name."""
        s, line = self.summarize([100.0, 110.0], env={"E2E_METRIC": "total"})
        self.assertEqual(s["metric_basis"], "aggregate_total_token_tok_s")
        self.assertEqual(s["throughput_tok_s_median"], 945.0)
        self.assertIsNone(s["output_throughput_tok_s_median"])
        self.assertIsNone(s["output_throughput_tok_s_spread_pct"])
        self.assertIn("aggregate_total_token_tok_s=945.0", line)

    def test_a_missing_cold_file_does_not_abort_the_summary(self):
        """The cold round is discarded evidence; losing it must not cost the timed rounds."""
        self.write_runs(self.runs, [100.0, 110.0])
        _run(["from-runs", self.runs, self.out, os.path.join(self.tmp, "absent.jsonl")])
        with open(self.out, encoding="utf-8") as fh:
            s = json.load(fh)
        self.assertEqual(s["throughput_tok_s_median"], 105.0)
        self.assertEqual(s["cold_runs"], 0)

    def test_cold_round_is_separate_and_absent_by_default(self):
        s, _ = self.summarize([100.0, 110.0])
        self.assertIsNone(s["cold_output_throughput_tok_s"])
        self.assertEqual(s["cold_runs"], 0)
        s, _ = self.summarize([100.0, 110.0], cold=[50.0, 60.0])
        self.assertEqual(s["cold_output_throughput_tok_s"], 55.0)
        self.assertEqual(s["cold_runs"], 2)
        self.assertEqual(s["throughput_tok_s_median"], 105.0, "cold leaked into the hot median")

    def test_isolated_replica_labels_itself(self):
        s, _ = self.summarize([100.0], env={"GEAK_ISOLATED_REPLICA": "1"})
        self.assertEqual(s["measurement_mode"], "isolated_server_replica")

    def test_warm_server_contract_complete(self):
        s, line = self.summarize([100.0], env={
            "WARM_SERVER_ROUNDS": "1", "MEASUREMENT_PURPOSE": "validation",
            "EFFECTIVE_CONFIG_DIGEST": "abc123"})
        self.assertEqual(s["measurement_mode"], "warm_server")
        self.assertEqual(s["measurement_purpose"], "validation")
        self.assertEqual(s["requested_rounds"], 1)
        self.assertEqual(s["successful_rounds"], 1)
        # Rounds and replicas are both "one sample" to the contract; callers read either.
        self.assertEqual(s["requested_replicas"], 1)
        self.assertEqual(s["successful_replicas"], 1)
        self.assertEqual(s["status"], "complete")
        self.assertTrue(s["usable_for_acceptance"])
        self.assertEqual(s["observed_median"], 100.0)
        self.assertEqual(s["dispersion_basis"], "within_server_rounds")
        self.assertEqual(s["effective_config_digest"], "abc123")
        self.assertIn("status=complete usable_for_acceptance=true", line)

    def test_short_warm_run_is_not_usable_for_acceptance(self):
        """2 of 3 rounds still has a median -- accepting on it is the bug."""
        s, line = self.summarize([100.0, 110.0], env={"WARM_SERVER_ROUNDS": "3"})
        self.assertEqual(s["status"], "incomplete")
        self.assertFalse(s["usable_for_acceptance"])
        self.assertEqual(s["successful_rounds"], 2)
        self.assertIn("usable_for_acceptance=false", line)

    def test_unparseable_round_count_fails_closed(self):
        s, _ = self.summarize([100.0], env={"WARM_SERVER_ROUNDS": "notanint"})
        self.assertEqual(s["status"], "incomplete")
        self.assertFalse(s["usable_for_acceptance"])

    def test_no_warm_rounds_emits_no_contract_fields(self):
        """The legacy shape must not grow a status the caller would trust."""
        s, line = self.summarize([100.0])
        for k in ("status", "usable_for_acceptance", "requested_rounds", "observed_median"):
            self.assertNotIn(k, s)
        self.assertNotIn("status=", line)


class FromReplicasTest(unittest.TestCase):
    """replica_*/selected_summary.json -> summary. The isolated_server lifecycle."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="summarize_reps_")

    def add(self, index, tput, basis="aggregate_output_tok_s", attempt=1):
        rdir = os.path.join(self.dir, "replica_%d" % index)
        os.makedirs(rdir, exist_ok=True)
        with open(os.path.join(rdir, "selected_summary.json"), "w", encoding="utf-8") as fh:
            json.dump({"throughput_tok_s_median": tput, "ttft_ms_median": 40.0 + index,
                       "tpot_ms_median": 8.0 + index, "metric_basis": basis}, fh)
        with open(os.path.join(rdir, "selected_attempt"), "w", encoding="utf-8") as fh:
            fh.write(str(attempt))

    def summarize(self, requested, successful, purpose="validation", digest="d1"):
        line = _run(["from-replicas", self.dir, str(requested), str(successful), purpose, digest])
        with open(os.path.join(self.dir, "bench_summary.json"), encoding="utf-8") as fh:
            return json.load(fh), line

    def test_complete_leg(self):
        for i, t in enumerate([104.0, 108.0, 112.0], start=1):
            self.add(i, t, attempt=i)
        s, line = self.summarize(3, 3)
        self.assertEqual(s["measurement_mode"], "isolated_server")
        self.assertEqual(s["measurement_purpose"], "validation")
        self.assertEqual(s["throughput_tok_s_median"], 108.0)
        self.assertEqual(s["output_throughput_tok_s_median"], 108.0)
        self.assertEqual(s["status"], "complete")
        self.assertTrue(s["usable_for_acceptance"])
        self.assertEqual(s["observed_median"], 108.0)
        self.assertEqual([r["replica"] for r in s["replicas"]], [1, 2, 3])
        self.assertEqual([r["attempt"] for r in s["replicas"]], [1, 2, 3])
        self.assertIn("requested=3 successful=3 status=complete", line)

    def test_missing_replica_is_incomplete(self):
        self.add(1, 104.0)
        self.add(2, 108.0)
        s, line = self.summarize(3, 2)
        self.assertEqual(s["status"], "incomplete")
        self.assertFalse(s["usable_for_acceptance"])
        self.assertIn("usable_for_acceptance=false", line)

    def test_stale_replica_dir_from_a_longer_previous_run_is_ignored(self):
        """A leftover replica_9 must not contribute to a 3-replica leg's median."""
        for i, t in enumerate([104.0, 108.0, 112.0], start=1):
            self.add(i, t)
        self.add(9, 9999.0)
        s, _ = self.summarize(3, 3)
        self.assertEqual(s["throughput_tok_s_median"], 108.0)
        self.assertNotIn(9999.0, s["all_throughput"])

    def test_mixed_metric_basis_refuses_to_name_one(self):
        """Two bases in one leg means the median is not comparable; say so."""
        self.add(1, 104.0)
        self.add(2, 108.0, basis="aggregate_total_token_tok_s")
        self.add(3, 112.0)
        s, _ = self.summarize(3, 3)
        self.assertIsNone(s["metric_basis"])
        self.assertIsNone(s["output_throughput_tok_s_median"])

    def test_no_replicas_at_all(self):
        s, _ = self.summarize(3, 0)
        self.assertIsNone(s["throughput_tok_s_median"])
        self.assertEqual(s["status"], "incomplete")
        self.assertFalse(s["usable_for_acceptance"])
        self.assertEqual(s["replicas"], [])


class ContractParityTest(unittest.TestCase):
    """The reason the two emitters live in one file: one contract, spelled once."""

    def test_both_modes_emit_the_same_gate_fields(self):
        gate = {"status", "requested_replicas", "successful_replicas",
                "usable_for_acceptance", "observed_median",
                "throughput_tok_s_median", "metric_basis", "measurement_mode",
                "measurement_purpose", "effective_config_digest"}

        tmp = tempfile.mkdtemp(prefix="summarize_parity_")
        runs = os.path.join(tmp, "bench_runs.jsonl")
        with open(runs, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"output_throughput": 100.0}) + "\n")
        out = os.path.join(tmp, "bench_summary.json")
        _run(["from-runs", runs, out], {"WARM_SERVER_ROUNDS": "1",
                                        "MEASUREMENT_PURPOSE": "validation"})
        with open(out, encoding="utf-8") as fh:
            warm = json.load(fh)

        rep_dir = os.path.join(tmp, "iso")
        rdir = os.path.join(rep_dir, "replica_1")
        os.makedirs(rdir)
        with open(os.path.join(rdir, "selected_summary.json"), "w", encoding="utf-8") as fh:
            json.dump({"throughput_tok_s_median": 100.0,
                       "metric_basis": "aggregate_output_tok_s"}, fh)
        _run(["from-replicas", rep_dir, "1", "1", "validation", ""])
        with open(os.path.join(rep_dir, "bench_summary.json"), encoding="utf-8") as fh:
            iso = json.load(fh)

        self.assertEqual(gate - set(warm), set(), "warm_server is missing contract fields")
        self.assertEqual(gate - set(iso), set(), "isolated_server is missing contract fields")
        for k in ("status", "usable_for_acceptance", "observed_median",
                  "requested_replicas", "successful_replicas"):
            self.assertEqual(warm[k], iso[k], "%s disagrees between lifecycles" % k)


class CliEntrypointTest(unittest.TestCase):
    """The one place a real process is required.

    Every other test drives ``main(argv)`` directly. That covers the parsing and the
    emitters, but not the two things `bench_e2e.sh` actually depends on: that running the
    file as a program works at all, and that a bad invocation exits non-zero instead of
    writing a `bench_summary.json` nobody asked for.
    """

    def _spawn(self, args):
        env = dict(os.environ)
        for name in _LIFECYCLE_ENV:
            env.pop(name, None)
        return subprocess.run([sys.executable, SUMMARIZE] + args,
                              env=env, capture_output=True, text=True, timeout=60)

    def test_running_the_file_as_a_program_emits_the_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = os.path.join(tmp, "bench_runs.jsonl")
            with open(runs, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"output_throughput": 100.0,
                                     "total_token_throughput": 900.0,
                                     "median_ttft_ms": 40.0,
                                     "median_tpot_ms": 8.0}) + "\n")
            out = os.path.join(tmp, "bench_summary.json")
            proc = self._spawn(["from-runs", runs, out])

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("E2E_SUMMARY", proc.stdout)
            self.assertTrue(os.path.exists(out))

    def test_an_unusable_invocation_exits_non_zero(self):
        proc = self._spawn(["no-such-subcommand"])

        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
