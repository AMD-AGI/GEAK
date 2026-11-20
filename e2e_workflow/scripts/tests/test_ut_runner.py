"""CPU tests of ut_runner.py: snapshot parity with the delivery gate, the contract, the timing-receipt
rules of generated tasks, coverage, environment classification, and whole CPU runs of a toy task."""
import contextlib
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import generate_ut as gu  # noqa: E402
import harness_lib as hl  # noqa: E402
import ut_runner as ur  # noqa: E402
import verify_ut as vu  # noqa: E402
from ut_task_fixture import ToyTask, SCRIPTS, have_torch  # noqa: E402
from ut_task_fixture import RankToyTask


class _H:
    """The two harness_lib functions receipt_problems / positive_mass_sigs use."""

    @staticmethod
    def deployment_graph_mode(regime):
        regime = regime or {}
        return False if regime.get("enforce_eager") else bool(regime.get("cuda_graph", True))

    @staticmethod
    def _observed_workload_cases(meta):
        return {c["sig"]: dict(c, effective_calls=float(c["effective_calls"]))
                for c in meta["workload"]["cases"]}

    @staticmethod
    def served_regimes(meta):
        return set()


def _row(sig, b, c, bms=1.0, cms=1.0, regime="decode"):
    return {"sig": sig, "regime": regime, "baseline_ms": bms, "optimized_ms": cms, "legs": {"baseline": b, "current": c}}


GRAPH_META = {"regime": {"cuda_graph": True},
              "workload": {"cases": [{"sig": "a", "regime": "decode", "effective_calls": 3},
                                     {"sig": "z", "regime": "decode", "effective_calls": 0}]}}


class TestSnapshotParity(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_python_snapshot_equals_the_delivery_gate_snapshot(self):
        t = ToyTask()
        self.addCleanup(t.cleanup)
        gu.vendor(t.task)
        t.write("kernel_src/sub/k.hip", "// kernel\n")
        t.write("kernel_src/__pycache__/x.py", "# cached\n")
        t.write("kernel_src/notes.txt", "not a source\n")
        out = subprocess.run(["node", os.path.join(SCRIPTS, "kernel_source_delivery.mjs"), "snapshot", t.task],
                             capture_output=True, text=True, check=True).stdout
        self.assertEqual(ur.source_snapshot(t.task), json.loads(out))


class TestReceiptRules(unittest.TestCase):
    def test_legs_with_different_timers_fail(self):
        per = [_row("a", {"timer": "cuda_event_graph", "primed": True}, {"timer": "cuda_event", "primed": True})]
        fail, inc, inconc = ur.receipt_problems(_H, GRAPH_META, per, ur.build_receipt(per, 1))
        self.assertTrue(fail and "different timers" in fail[0])

    def test_a_graph_phase_timed_eagerly_on_the_baseline_is_incomplete(self):
        per = [_row("a", {"timer": "cuda_event", "primed": True}, {"timer": "cuda_event", "primed": True})]
        fail, inc, inconc = ur.receipt_problems(_H, GRAPH_META, per, ur.build_receipt(per, 1))
        self.assertEqual(fail, [])
        self.assertTrue(any("capture failed" in x for x in inc))

    def test_a_host_bound_positive_mass_case_names_host_ms_and_ms(self):
        per = [_row("a", {"timer": "cuda_event_graph", "primed": False, "host_ms": 0.02},
                    {"timer": "cuda_event_graph", "primed": True, "host_ms": 0.01}, bms=0.005),
               _row("z", {"timer": "cuda_event_graph", "primed": False, "host_ms": 0.02},
                    {"timer": "cuda_event_graph", "primed": False, "host_ms": 0.02})]
        fail, inc, inconc = ur.receipt_problems(_H, GRAPH_META, per, ur.build_receipt(per, 1))
        self.assertEqual(len(inc), 1)
        self.assertIn("a (baseline: host_ms=0.02, ms=0.005)", inc[0])
        self.assertIn("inner", inc[0])
        self.assertNotIn("z (", inc[0])                    # zero call mass: reported, not judged

    def test_a_leg_that_cannot_tell_is_inconclusive(self):
        per = [_row("a", {"timer": "cuda_event_graph"}, {"timer": "cuda_event_graph"})]
        fail, inc, inconc = ur.receipt_problems(_H, GRAPH_META, per, ur.build_receipt(per, 1))
        self.assertTrue(inconc)

    SPIN = {"queued": "gpu_spin_before_start", "us": 200.0, "samples": 150, "late": 0, "retried": False,
            "includes_dispatch": False, "reps": 3}

    def test_an_eager_deployment_only_records_primed(self):
        # A16: the eager phase is timed on the queued spin; a dispatch slower than the GPU time is a receipt
        # flag (scored on GPU time), never a failure.
        meta = dict(GRAPH_META, regime={"cuda_graph": False})
        per = [_row("a", {"timer": "cuda_event", "primed": False, "host_ms": 1.0, "spin": self.SPIN,
                          "latency_ms": 1.2},
                    {"timer": "cuda_event", "primed": False, "host_ms": 0.9, "spin": self.SPIN,
                     "latency_ms": 1.1})]
        self.assertEqual(ur.receipt_problems(_H, meta, per, ur.build_receipt(per, 1)), ([], [], []))
        flags = ur.receipt_flags(_H, meta, per)
        self.assertEqual([(f["leg"], f["kind"]) for f in flags],
                         [("baseline", "dispatch_slower_than_kernel"), ("current", "dispatch_slower_than_kernel")])
        self.assertEqual((flags[0]["host_ms"], flags[0]["ms"], flags[0]["latency_ms"]), (1.0, 1.0, 1.2))
        self.assertIn("host runs ahead of the GPU", flags[0]["note"])

    def test_an_eager_case_timed_without_the_queued_spin_is_incomplete(self):
        meta = dict(GRAPH_META, regime={"cuda_graph": False})
        per = [_row("a", {"timer": "cuda_event", "primed": True, "host_ms": 0.1, "spin": self.SPIN},
                    {"timer": "cuda_event", "primed": True, "host_ms": 0.1})]
        fail, inc, inconc = ur.receipt_problems(_H, meta, per, ur.build_receipt(per, 1))
        self.assertEqual((fail, inconc), ([], []))
        self.assertEqual(len(inc), 1)
        self.assertIn("a: the phase is not deployed under a CUDA graph, but the candidate leg", inc[0])
        self.assertIn("A16", inc[0])

    def test_the_spin_rule_spares_cpu_timing_multi_rank_pools_and_graph_phases(self):
        eager = dict(GRAPH_META, regime={"cuda_graph": False})
        cpu = [_row("a", {"timer": "wall"}, {"timer": "wall"})]
        self.assertEqual(ur.receipt_problems(_H, eager, cpu, ur.build_receipt(cpu, 1)), ([], [], []))
        pool = [_row("a", {"timer": "cuda_event", "primed": True, "host_ms": 0.1, "rank_ms": [1.0, 1.0]},
                     {"timer": "cuda_event", "primed": True, "host_ms": 0.1, "rank_ms": [1.0, 1.0]})]
        self.assertEqual(ur.receipt_problems(_H, eager, pool, ur.build_receipt(pool, 2)), ([], [], []))
        graph = [_row("a", {"timer": "cuda_event_graph", "primed": True, "host_ms": 0.1},
                      {"timer": "cuda_event_graph", "primed": True, "host_ms": 0.1})]
        self.assertEqual(ur.receipt_problems(_H, GRAPH_META, graph, ur.build_receipt(graph, 1)), ([], [], []))
        self.assertEqual(ur.receipt_flags(_H, GRAPH_META, [_row("a", {"primed": False}, {"primed": False})]), [])

    def test_a_spin_that_ran_out_even_doubled_is_flagged_as_including_dispatch(self):
        meta = dict(GRAPH_META, regime={"cuda_graph": False})
        late = dict(self.SPIN, late=40, retried=True, includes_dispatch=True, us=800.0)
        per = [_row("a", {"timer": "cuda_event", "primed": True, "host_ms": 0.1, "spin": late},
                    {"timer": "cuda_event", "primed": True, "host_ms": 0.1, "spin": self.SPIN})]
        self.assertEqual(ur.receipt_problems(_H, meta, per, ur.build_receipt(per, 1)), ([], [], []))
        flags = ur.receipt_flags(_H, meta, per)
        self.assertEqual([(f["leg"], f["kind"], f["late"], f["spin_us"]) for f in flags],
                         [("baseline", "reading_includes_dispatch", 40, 800.0)])

    def test_the_receipt_carries_the_spin_the_latency_and_the_deployment_of_each_case(self):
        per = [_row("p", {"timer": "cuda_event", "primed": False, "host_ms": 0.02, "spin": self.SPIN,
                          "latency_ms": 0.03}, {"timer": "cuda_event", "primed": False, "host_ms": 0.02,
                                                 "spin": self.SPIN, "latency_ms": 0.03}, regime="prefill"),
               _row("d", {"timer": "cuda_event_graph", "primed": True, "host_ms": 0.001},
                    {"timer": "cuda_event_graph", "primed": True, "host_ms": 0.001})]
        r = ur.build_receipt(per, 1, phases={"prefill": False, "decode": True})
        self.assertEqual(r["cases"]["p"]["baseline"]["spin"], self.SPIN)
        self.assertEqual(r["cases"]["p"]["current"]["latency_ms"], 0.03)
        self.assertEqual((r["cases"]["p"]["deployed_graph"], r["cases"]["d"]["deployed_graph"]), (False, True))
        self.assertNotIn("spin", r["cases"]["d"]["baseline"])
        self.assertNotIn("deployed_graph", ur.build_receipt(per, 1)["cases"]["p"])   # additive only

    def test_phase_speedups_are_call_weighted_over_the_included_rows(self):
        rows = [{"regime": "decode", "baseline_ms": 1.0, "optimized_ms": 1.0, "calls": 100, "included": True},
                {"regime": "prefill", "baseline_ms": 2.0, "optimized_ms": 1.0, "calls": 1, "included": True},
                {"regime": "prefill", "baseline_ms": 4.0, "optimized_ms": 4.0, "calls": 3, "included": True},
                {"regime": "prefill", "baseline_ms": 9.0, "optimized_ms": 1.0, "calls": 5, "included": False},
                {"regime": "mixed", "baseline_ms": 1.0, "optimized_ms": 1.0, "calls": 0, "included": True}]
        got = ur.phase_speedups(rows)
        self.assertEqual(sorted(got), ["decode", "prefill"])          # no call mass -> no phase
        self.assertAlmostEqual(got["prefill"]["speedup"], (2.0 + 12.0) / (1.0 + 12.0))
        self.assertEqual((got["decode"]["speedup"], got["prefill"]["calls"]), (1.0, 4))

    def test_the_receipt_is_a_superset_of_the_frozen_oracle_receipt(self):
        per = [_row("a", {"timer": "cuda_event_graph", "primed": True, "host_ms": 0.1,
                          "cache_condition": {"mode": "read-evict"}},
                    {"timer": "cuda_event_graph", "primed": True, "host_ms": 0.1,
                     "cache_condition": {"mode": "read-evict"}})]
        r = ur.build_receipt(per, 1)
        for key in ("all_primed", "timer_unprimed", "cache_mode", "cases", "world_size", "timer", "alignment"):
            self.assertIn(key, r)
        self.assertEqual((r["all_primed"], r["cache_mode"], r["timer"]), (True, "read-evict", "cuda_event_graph"))
        self.assertEqual(r["cases"]["a"]["baseline"], {"ms": 1.0, "timer": "cuda_event_graph",
                                                       "primed": True, "host_ms": 0.1})


class TestCoverage(unittest.TestCase):
    META = {"case_manifest": {"dims": ["regime"], "cases": [
        {"sig": "a", "regime": "decode", "roles": ["timing", "correctness"]},
        {"sig": "b", "regime": "decode", "roles": ["correctness"]}]}}

    def test_the_run_must_time_exactly_the_declared_cases(self):
        self.assertEqual(ur.manifest_problems(self.META, ["a"], {"a", "b"}, {"a"}), [])
        self.assertTrue(ur.manifest_problems(self.META, ["a", "c"], {"a", "b"}, {"a"}))
        self.assertTrue(ur.manifest_problems(self.META, ["a"], {"a"}, {"a"}))           # b never checked
        self.assertTrue(ur.manifest_problems(self.META, ["a"], {"a", "b"}, {"a", "d"}))  # d has call mass

    def test_checked_sigs_reads_every_section(self):
        rep = {"decode": {"eager": [{"case": "a"}, {"case": "output_independence"}],
                          "random": [{"case": "random[0]:b"}], "graph_replay": [{"case": "c"}]}}
        self.assertEqual(ur.checked_sigs(rep), {"a", "b", "c"})

    def test_a_boundary_case_must_still_be_checked_and_is_listed_apart_from_the_buckets(self):
        """A40: a boundary case forms no bucket; coverage.boundary lists it; the UT must have compared it."""
        meta = {"regime": {"cuda_graph": False}, "case_manifest": {"dims": ["regime", "m"], "cases": [
            {"sig": "a", "regime": "decode", "m": 4, "roles": ["timing", "correctness"]},
            {"sig": "a_max", "regime": "decode", "m": 16384, "roles": ["correctness"], "boundary": True}]}}
        self.assertEqual(ur.manifest_problems(meta, ["a"], {"a", "a_max"}, None), [])
        self.assertEqual(ur.manifest_problems(meta, ["a"], {"a"}, None),
                         ["manifest correctness cases never checked: ['a_max']"])
        rep = {"decode": {"eager": [{"case": "a"}, {"case": "a_max"}]}}
        cov = ur.coverage_summary(_H, meta, [rep], ["a"], "unweighted")
        self.assertEqual([b["bucket"] for b in cov["buckets"]], [{"regime": "decode", "m": 4}])
        self.assertEqual(cov["boundary"], [{"sig": "a_max", "regime": "decode",
                                            "dims": {"regime": "decode", "m": 16384}, "checked": True}])
        self.assertFalse(ur.coverage_summary(_H, meta, [{}], ["a"], "unweighted")["boundary"][0]["checked"])
        self.assertNotIn("checked", ur.coverage_summary(_H, meta, None, None, "unweighted")["boundary"][0])

    def test_the_declared_uncovered_calls_are_copied_into_the_coverage(self):
        """A35: coverage.calls.declared_uncovered is meta.workload.uncovered as written ({} without one); the
        signature fields keep their meaning."""
        unc = {"DECODE": {"calls": 1000, "mass_us": 12.5, "why": "decode steps above batch 4 (rare)"}}
        meta = dict(self.META, regime={"cuda_graph": False},
                    workload={"cases": [{"sig": "a", "regime": "decode", "effective_calls": 3}], "uncovered": unc})
        cov = ur.coverage_summary(_H, meta, None, ["a"], "observed")
        self.assertEqual(cov["calls"], {"with_calls": ["a"], "timed": ["a"], "uncovered": [],
                                        "declared_uncovered": unc})
        meta["workload"].pop("uncovered")
        self.assertEqual(ur.coverage_summary(_H, meta, None, ["a"], "observed")["calls"]["declared_uncovered"], {})
        meta.pop("workload")
        self.assertEqual(ur.coverage_summary(_H, meta, None, None, "unweighted")["calls"]["declared_uncovered"], {})


class TestCorrectnessByRank(unittest.TestCase):
    def test_raised_cases_are_not_numerical_mismatches(self):
        judged = {"ok": False, "report": {"decode": {
            "eager": [{"case": "a", "correct": False, "max_rel_err": 0.5},
                      {"case": "b", "correct": True, "max_rel_err": 0.0}],
            "random": [{"case": "random[0]:a", "correct": False, "max_rel_err": None,
                        "note": "value-parity raised: boom"}]}}}
        self.assertEqual(ur.correctness_by_rank(judged),
                         [{"rank": 0, "launch_succeeded": True, "numerical_mismatches": 1, "raised_cases": 1,
                           "failing_cases": ["a"]}])

    def test_every_rank_gets_a_row(self):
        judged = {"ranks": [{"rank": 0, "report": {"x": {"eager": [{"case": "a", "correct": True}]}}},
                            {"rank": 1, "errors": ["boom"], "report": {}}]}
        rows = ur.correctness_by_rank(judged)
        self.assertEqual([(r["rank"], r["launch_succeeded"], r["numerical_mismatches"]) for r in rows],
                         [(0, True, 0), (1, False, 0)])


class TestNativeNoise(unittest.TestCase):
    """A37: the native self-noise over the ranks and the advice of a native self-check failure."""

    def test_the_ranks_worst_noise_rel_is_kept(self):
        ranks = [{"rank": 0, "ok": True, "native_noise": {"max_rel_err": 0.1, "cases": 4, "noise_rel": 0.01}},
                 {"rank": 1, "ok": True, "native_noise": {"max_rel_err": 0.3, "cases": 4, "noise_rel": 0.03,
                                                          "noise_rel_nonfinite": 1}},
                 {"rank": 2, "ok": True, "native_noise": {"max_rel_err": 0.2, "cases": 4, "noise_rel": 0.02}}]
        self.assertEqual(ur.merge_correct(ranks)["native_noise"],
                         {"max_rel_err": 0.3, "noise_rel": 0.03, "noise_rel_nonfinite": 1})
        # a judged (candidate) leg has no native noise: nothing is added
        self.assertEqual(ur.merge_noise([{"rank": 0}, {"rank": 1}]), {"max_rel_err": None})
        self.assertEqual(ur.merge_noise([{"native_noise": {"max_rel_err": None, "cases": 0, "noise_rel": None}}]),
                         {"max_rel_err": None, "noise_rel": None})

    def test_the_advice_names_noise_rel_and_the_tolerance_it_needs(self):
        text = ur.noise_advice({"max_rel_err": 0.5, "noise_rel": 0.0102})
        self.assertIn("noise_rel=0.0102", text)
        self.assertIn("at least 2 x noise_rel = 0.0204", text)
        self.assertEqual(ur.noise_advice({"max_rel_err": 0.5}), "")

    def test_per_output_noise_is_merged_over_the_ranks_and_named_in_the_advice(self):
        """A42: every output's worst noise_rel over the ranks, its tolerance and its non-finite count."""
        ranks = [{"rank": 0, "native_noise": {"max_rel_err": 0.1, "noise_rel": 0.004, "per_output": {
                     "0": {"tol": 0.02, "noise_rel": 0.004}, "1": {"tol": 0.001, "noise_rel": 0.0001}}}},
                 {"rank": 1, "native_noise": {"max_rel_err": 0.2, "noise_rel": 0.003, "per_output": {
                     "0": {"tol": 0.02, "noise_rel": 0.003},
                     "1": {"tol": 0.001, "noise_rel": 0.0004, "noise_rel_nonfinite": 2}}}}]
        merged = ur.merge_correct(ranks)["native_noise"]
        self.assertEqual(merged["per_output"], {"0": {"tol": 0.02, "noise_rel": 0.004},
                                                "1": {"tol": 0.001, "noise_rel": 0.0004, "noise_rel_nonfinite": 2}})
        self.assertEqual(merged["noise_rel"], 0.004)
        text = ur.noise_advice(merged)
        self.assertIn("1: noise_rel=0.0004, tol 0.001, needs at least 0.0008", text)
        self.assertIn("every output's tolerance must be at least 2 x its noise_rel", text)
        self.assertNotIn("per_output", ur.merge_noise([{"native_noise": {"noise_rel": 0.0}}]))


class TestEnvFault(unittest.TestCase):
    def test_environment_failures_are_recognised(self):
        self.assertEqual(ur.env_fault("torch.OutOfMemoryError: HIP out of memory"), "resources")
        self.assertEqual(ur.env_fault("hipIpcOpenMemHandle failed: invalid device pointer"), "ipc")
        self.assertEqual(ur.env_fault("ERROR: GPU 3 still shows foreign work after 30s"), "lock")
        self.assertIsNone(ur.env_fault("ZeroDivisionError in cases.py"))


@unittest.skipUnless(have_torch(), "torch not installed")
class TestToyRuns(unittest.TestCase):
    """Whole UT runs of the toy task on the CPU (legs are real subprocesses)."""

    def setUp(self):
        self.t = ToyTask()
        self.addCleanup(self.t.cleanup)
        self.assertEqual(gu.main(["all", self.t.task]), 0)

    def run_ut(self, *args):
        p = subprocess.run([sys.executable, "unittest.py", *args], cwd=self.t.task, env=self.t.env(),
                           capture_output=True, text=True, timeout=900)
        # the run names the result it wrote (A28: a partial run never writes run/ut_result.json)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("UT_RESULT: ")][-1]
        with open(json.loads(line[len("UT_RESULT: "):])["result"]) as fh:
            return p.returncode, p.stdout + p.stderr, json.load(fh)

    def test_partial_runs_never_write_the_delivered_result(self):
        """A28: a source-mutation run and every partial run leave run/ut_result.json alone; the result says
        whether the last full UT still matches, so a restored source reuses it."""
        code, out, full = self.run_ut()
        self.assertEqual(code, 0, out[-3000:])
        delivered = os.path.join(self.t.task, "run", "ut_result.json")
        before = open(delivered, "rb").read()
        clean = open(os.path.join(self.t.task, "kernel_src", "toyop.py")).read()
        self.t.write("kernel_src/toyop.py", clean.replace("torch.tanh(x @ w)", "torch.tanh(x @ w) * 1.1"))
        code, out, mut = self.run_ut("--mutation", "toy_leaf")
        self.assertEqual(code, 1, out[-3000:])
        self.assertEqual(open(delivered, "rb").read(), before)
        self.assertEqual(mut["gates"]["correctness"], "fail")
        self.assertTrue(os.path.isfile(os.path.join(self.t.task, "run", "mutations", "toy_leaf", "ut_result.json")))
        self.assertGreater(sum(r["numerical_mismatches"] for r in mut["correctness"]["by_rank"]), 0)
        self.assertFalse(mut["full_result"]["current"])            # the mutated snapshot is not the full UT's
        self.assertNotIn("timing", mut)
        self.t.write("kernel_src/toyop.py", clean)                  # restore
        code, out, chk = self.run_ut("--correctness-only")
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(open(delivered, "rb").read(), before)
        self.assertTrue(chk["full_result"]["current"], chk["full_result"])
        self.assertIn("[ut] full UT result run/ut_result.json: current", out)
        self.assertTrue(os.path.isfile(os.path.join(self.t.task, "run", "ut_partial", "correctness-only",
                                                    "ut_result.json")))
        self.assertEqual(chk["snapshot_sha256"], full["snapshot_sha256"])

    def test_without_workload_weights_the_ut_times_unweighted(self):
        """A19: no meta.workload -> main's normal mode: unweighted geomean, no GEAK_WEIGHTED_SPEEDUP line."""
        m = self.t.meta()
        m.pop("workload")
        self.t.write_meta(m)
        self.assertEqual(gu.main(["all", self.t.task]), 0)
        code, out, res = self.run_ut()
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("GEOMEAN_SPEEDUP=", out)
        self.assertNotIn("GEAK_WEIGHTED_SPEEDUP", out)
        self.assertIn("case decode_m1: baseline_ms=", out)
        t = res["timing"]
        self.assertEqual(t["weighting"]["mode"], "unweighted")
        self.assertIsNone(t["weighted"])
        self.assertEqual(t["primary"], t["geomean"])
        self.assertEqual(t["by_phase"]["decode"]["basis"], "unweighted geomean")
        self.assertEqual(res["receipt"]["weighting"], "unweighted")
        self.assertEqual(res["weighting"], "unweighted")
        self.assertEqual(res["coverage"]["weighting"], "unweighted")
        self.assertEqual(res["coverage"]["calls"]["with_calls"], ["decode_m1", "decode_m4"])

    def test_the_result_carries_its_coverage(self):
        """A27: buckets with their timing and correctness cases, calls vs timed, graph and state records."""
        code, out, res = self.run_ut()
        self.assertEqual(code, 0, out[-3000:])
        cov = res["coverage"]
        self.assertEqual(cov["weighting"], "observed")
        self.assertEqual(sorted(json.dumps(b["bucket"], sort_keys=True) for b in cov["buckets"]),
                         ['{"m": 1, "regime": "decode"}', '{"m": 4, "regime": "decode"}'])
        self.assertEqual(cov["boundary"], [])                          # A40: no boundary case declared
        self.assertTrue(all(b["timing"] and b["correctness_checked"] for b in cov["buckets"]))
        self.assertEqual(cov["calls"], {"with_calls": ["decode_m1", "decode_m4"],
                                        "timed": ["decode_m1", "decode_m4"], "uncovered": [],
                                        "declared_uncovered": {}})                 # A35
        self.assertEqual(cov["graph"], {})                         # the toy deploys eagerly
        self.assertEqual(cov["state"]["stateful"], False)
        self.assertNotIn("weighting", res)                          # only an unweighted result is marked

    def test_an_unmodified_extraction_passes_with_a_full_receipt(self):
        code, out, res = self.run_ut()
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("CORRECTNESS: PASS", out)
        self.assertIn("GEAK_WEIGHTED_SPEEDUP=", out)
        self.assertIn("GEAK_TIMING_RECEIPT: ", out)
        self.assertEqual(res["contract_sha256"], self.t.read_json("run/ut_contract.json")["contract_sha256"])
        self.assertEqual(res["snapshot_sha256"], ur.source_snapshot(self.t.task)["snapshot_sha256"])
        self.assertEqual(set(res["receipt"]["cases"]), {"decode_m1", "decode_m4"})
        self.assertEqual(res["identity"]["candidate"]["file"],
                         os.path.realpath(os.path.join(self.t.task, "_cand_overlay", "_patched", "toyop.py")))
        self.assertEqual(res["native_noise"]["noise_rel"], 0.0)     # A37: a bit-reproducible native has none

    def test_a_wrong_candidate_fails_correctness(self):
        self.t.write("kernel_src/toyop.py", "import torch\n\ndef op(x, w):\n    return torch.tanh(x @ w) * 1.1\n")
        code, out, res = self.run_ut("--correctness-only")
        self.assertEqual(code, 1, out[-3000:])
        self.assertEqual(res["gates"]["correctness"], "fail")

    def test_a_changed_test_is_refused_before_anything_runs(self):
        self.t.write("cases.py", open(os.path.join(self.t.task, "cases.py")).read() + "\n# edited\n")
        code, out, res = self.run_ut()
        self.assertEqual(code, 3)
        self.assertIn("cases.py: changed after freeze", res["reason"])

    def test_a_native_that_fails_its_own_reference_makes_the_ut_incomplete(self):
        cases = open(os.path.join(self.t.task, "cases.py")).read().replace(
            '"args": _mk(m, torch.Generator().manual_seed(11))}',
            '"args": _mk(m, torch.Generator().manual_seed(11)), "ref": torch.zeros(m, 8)}')
        self.t.write("cases.py", cases)
        self.assertEqual(gu.main(["all", self.t.task]), 0)
        code, out, res = self.run_ut("--correctness-only")
        self.assertEqual(code, 3, out[-3000:])
        self.assertIn("NATIVE", res["reason"])

    def test_a_native_noisier_than_the_tolerance_stops_the_ut_and_names_the_tolerance_it_needs(self):
        """A37: the native leg ran, its self-check stops the UT (exit 3): run/ut_result.json still has
        native_noise.noise_rel, and the reason names noise_rel and 2 x noise_rel."""
        self.t.write_site("toyop.py", "import os\nimport torch\n\n\ndef op(x, w):\n"
                                      "    g = torch.Generator().manual_seed(os.getpid())   # another run, another noise\n"
                                      "    return torch.tanh(x @ w) + 0.02 * torch.randn(x.shape[0], w.shape[1], "
                                      "generator=g)\n")
        code, out, res = self.run_ut("--correctness-only")
        self.assertEqual(code, 3, out[-3000:])
        self.assertEqual(res["gates"]["native_self_check"], "incomplete")
        nr = res["native_noise"]["noise_rel"]
        self.assertGreater(nr, 0.01)                                  # above the toy task's tol
        self.assertIn(f"noise_rel={nr:.4g}", res["reason"])
        self.assertIn(f"2 x noise_rel = {2 * nr:.4g}", res["reason"])

    def test_a_boundary_case_is_run_and_compared_without_a_timing_case(self):
        """A40: a correctness-only boundary case at a new M passes the generator; the UT compares it and lists it in
        coverage.boundary; dropped from the cases the UT runs, the UT is incomplete."""
        m = self.t.meta()
        m["case_manifest"]["cases"].append({"sig": "decode_m8_max", "regime": "decode", "m": 8,
                                            "roles": ["correctness"], "boundary": True})
        self.t.write_meta(m)
        cases = open(os.path.join(self.t.task, "cases.py")).read()
        self.t.write("cases.py", cases + (
            "\n\n_eager = eager_cases\n\n\ndef eager_cases(h, meta):\n    import torch\n"
            "    return _eager(h, meta) + [{\"sig\": \"decode_m8_max\", \"regime\": \"decode\", "
            "\"args\": _mk(8, torch.Generator().manual_seed(13))}]\n"))
        self.assertEqual(gu.main(["all", self.t.task]), 0)
        code, out, res = self.run_ut()
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(res["coverage"]["boundary"], [{"sig": "decode_m8_max", "regime": "decode",
                                                        "dims": {"regime": "decode", "m": 8}, "checked": True}])
        self.assertNotIn({"regime": "decode", "m": 8}, [b["bucket"] for b in res["coverage"]["buckets"]])
        self.assertNotIn("decode_m8_max", res["receipt"]["cases"])          # never timed
        self.t.write("cases.py", cases)
        self.assertEqual(gu.main(["all", self.t.task]), 0)
        code, out, res = self.run_ut("--correctness-only")
        self.assertEqual(code, 3, out[-3000:])
        self.assertIn("manifest correctness cases never checked: ['decode_m8_max']", res["reason"])

    def test_a_stateful_task_without_a_sequence_is_incomplete(self):
        m = self.t.meta()
        m["capabilities"]["stateful"] = True
        self.t.write_meta(m)
        gu.vendor(self.t.task)
        gu.freeze(self.t.task)                     # bypass the generator's own check on purpose
        code, out, res = self.run_ut("--correctness-only")
        self.assertEqual(code, 3, out[-3000:])
        self.assertIn("sequence_cases", res["reason"])

    def test_an_aa_run_times_the_baseline_against_itself(self):
        code, out, res = self.run_ut("--aa", "--no-early-stop")
        self.assertEqual(code, 0, out[-3000:])
        self.assertTrue(all(c["reps"] == 3 for c in res["timing"]["per_case"]))
        self.assertNotIn("identity", res)


TWO_OUT_OP = """import torch


def op(x, w):
    y = x @ w
    return torch.tanh(y).to(torch.bfloat16), torch.sigmoid(y).float()
"""


@unittest.skipUnless(have_torch(), "torch not installed")
class TestPerOutputToyRuns(TestToyRuns.__base__):
    """A42: whole CPU UT runs of a two-output toy (a bf16 hidden state and an f32 mixing weight), the f32 output at
    its own tolerance 0.001 and the rest at tol 0.02."""

    def setUp(self):
        self.t = ToyTask()
        self.addCleanup(self.t.cleanup)
        self.t.write_site("toyop.py", TWO_OUT_OP)
        self.t.write("kernel_src/toyop.py", TWO_OUT_OP)
        self.tolerance({"1": 0.001})

    run_ut = TestToyRuns.run_ut

    def tolerance(self, per_output):
        m = self.t.meta()
        m["tolerance"] = {"tol": 0.02} if per_output is None else {"tol": 0.02, "per_output": per_output}
        self.t.write_meta(m)
        self.assertEqual(gu.main(["all", self.t.task]), 0)

    def candidate(self, factor):
        self.t.write("kernel_src/toyop.py", TWO_OUT_OP.replace("torch.sigmoid(y).float()",
                                                               f"torch.sigmoid(y).float() * {factor}"))

    def test_an_f32_output_half_a_percent_off_fails_at_its_own_tolerance_and_passes_at_one_tol(self):
        self.candidate(1.005)
        code, out, res = self.run_ut("--correctness-only")
        self.assertEqual(code, 1, out[-3000:])
        self.assertEqual(res["gates"]["correctness"], "fail")
        rows = [r for sec in ("eager", "random") for r in res["correctness"]["candidate"]["report"]["default"][sec]
                if r["case"] != "output_independence"]
        self.assertTrue(rows and all(not r["correct"] and (r["output"], r["tol"]) == ("1", 0.001) for r in rows))
        self.tolerance(None)                                  # the same candidate, every output at tol 0.02
        code, out, res = self.run_ut("--correctness-only")
        self.assertEqual(code, 0, out[-3000:])
        self.assertNotIn("tolerance", res)
        self.assertNotIn("per_output", res["native_noise"])

    def test_the_result_records_every_outputs_tolerance_and_noise(self):
        code, out, res = self.run_ut()
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(res["tolerance"], {"tol": 0.02, "per_output": {"1": 0.001},
                                            "outputs": [{"output": "0", "tol": 0.02}, {"output": "1", "tol": 0.001}]})
        self.assertEqual(res["native_noise"]["per_output"], {"0": {"tol": 0.02, "noise_rel": 0.0},
                                                             "1": {"tol": 0.001, "noise_rel": 0.0}})
        self.assertEqual(res["native_noise"]["noise_rel"], 0.0)
        rows = res["correctness"]["candidate"]["report"]["default"]["random"]
        self.assertTrue(rows and all(r["correct"] and r["output"] in ("0", "1") for r in rows))

    def test_the_controls_move_each_output_by_its_own_tolerance(self):
        """within_tol moves the f32 output by 0.001/10 (by tol/10 = 0.002 it would fail its 0.001); scale_out moves it
        by 4 x 0.001 and is rejected."""
        code, out, res = self.run_ut("--control", "within_tol")
        self.assertEqual(code, 0, out[-3000:])
        code, out, res = self.run_ut("--control", "scale_out", "--reuse-native")
        self.assertEqual(code, 1, out[-3000:])
        verdict, detail = vu.judge_control("scale_out", {}, code, res, 0.02, 1)
        self.assertEqual(verdict, "PASS", detail)

    def test_a_position_past_the_last_output_stops_the_ut(self):
        self.tolerance({"2": 0.001})                          # the generator cannot count the outputs: the UT does
        code, out, res = self.run_ut("--correctness-only")
        self.assertEqual(code, 3, out[-3000:])
        self.assertEqual(res["gates"]["native_self_check"], "incomplete")
        self.assertIn("per_output names the positions ['2'], but the compared output has 2 tensor(s)", res["reason"])

    def test_a_noisy_f32_output_stops_the_ut_and_names_the_tolerance_it_needs(self):
        self.t.write_site("toyop.py", TWO_OUT_OP.replace(
            "torch.sigmoid(y).float()",
            "torch.sigmoid(y).float() * (1 + 0.003 * torch.rand(y.shape, generator=torch.Generator()"
            ".manual_seed(__import__('os').getpid())))"))
        code, out, res = self.run_ut("--correctness-only")
        self.assertEqual(code, 3, out[-3000:])
        per = res["native_noise"]["per_output"]
        self.assertEqual(per["0"]["noise_rel"], 0.0)
        self.assertGreater(per["1"]["noise_rel"], 0.001)
        self.assertIn(f"1: noise_rel={per['1']['noise_rel']:.4g}, tol 0.001", res["reason"])


# --------------------------------------------------------------------------- the driver's helpers (CPU, no torch)
def _patch(case, target, **attrs):
    """Replace `target`'s attributes until the end of the test `case` (undone at its cleanup)."""
    for name, value in attrs.items():
        p = mock.patch.object(target, name, value)
        p.start()
        case.addCleanup(p.stop)


def _touch(path, text=""):
    with open(path, "w") as fh:
        fh.write(text)


class _Clock:
    """ut_runner's `time` in a test: monotonic() reads a counter and sleep() only advances it, so nothing waits; each
    sleep lets `tick` act (the rank processes working while the parent waits)."""

    def __init__(self, t=1000.0, tick=None):
        self.t, self.tick, self.slept = t, tick, []

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s
        if self.tick is not None:
            self.tick()


class TestDriverBasics(unittest.TestCase):
    def test_a_stop_carries_its_exit_code_reason_and_gate(self):
        stop = ur.UTStop(ur.EXIT_ENV, "the GPU lease ran out", "timeout")
        self.assertEqual((stop.code, stop.reason, stop.gate, str(stop)),
                         (2, "the GPU lease ran out", "timeout", "the GPU lease ran out"))
        self.assertIsNone(ur.UTStop(ur.EXIT_FAIL, "wrong numbers").gate)

    def test_no_text_is_no_fault_and_a_missing_device_is_the_environments(self):
        self.assertIsNone(ur.env_fault(""))
        self.assertIsNone(ur.env_fault(None))
        self.assertEqual(ur.env_fault("RuntimeError: HIP error: hipErrorNoDevice"), "device")
        self.assertEqual(ur.env_fault("HIP out of memory\nERROR: Failed to acquire GPU 3 lock after 600s"), "lock")

    def test_a_module_is_loaded_from_its_file_under_the_given_name(self):
        d = tempfile.mkdtemp(prefix="geak_utcore_")
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "anything.py")
        _touch(path, "BASE = 41\n\n\ndef answer():\n    return BASE + 1\n")
        with mock.patch.dict(sys.modules):
            mod = ur._load("geak_utcore_probe", path)
            self.assertIs(sys.modules["geak_utcore_probe"], mod)
            self.assertEqual((mod.answer(), mod.__file__), (42, path))
        self.assertNotIn("geak_utcore_probe", sys.modules)

    def test_the_world_size_must_be_an_integer_and_picks_the_default_time_limits(self):
        self.assertEqual(ur.world_size({}), 1)
        self.assertEqual(ur.world_size({"capabilities": {"multi_rank": {"world_size": 0}}}), 1)
        multi = {"capabilities": {"multi_rank": {"world_size": "4"}},
                 "timeouts": {"command_s": 600, "total_s": None, "build_s": 5}}
        self.assertEqual(ur.world_size(multi), 4)
        self.assertEqual(ur.timeouts(multi), {"setup_s": 900, "command_s": 600.0, "total_s": 3600})
        self.assertEqual(ur.timeouts({"timeouts": {"total_s": 7200}}),
                         {"setup_s": 900, "command_s": 3600, "total_s": 7200.0})
        # a setup_s / command_s of 0 or less keeps the default (a pool cannot run with it); total_s 0 is no limit
        zero = {"capabilities": {"multi_rank": {"world_size": 2}},
                "timeouts": {"setup_s": 0, "command_s": -1, "total_s": 0}}
        self.assertEqual(ur.timeouts(zero), {"setup_s": 900, "command_s": 1800, "total_s": 0.0})
        self.assertEqual(ur.timeouts({"timeouts": {"command_s": 0}}), {"setup_s": 900, "command_s": 3600, "total_s": 0})
        for bad in ("two", [2]):
            with self.assertRaises(ur.UTStop) as cm:
                ur.world_size({"capabilities": {"multi_rank": {"world_size": bad}}})
            self.assertEqual((cm.exception.code, cm.exception.reason),
                             (ur.EXIT_INCOMPLETE, f"capabilities.multi_rank.world_size is not an integer: {bad!r}"))


class TestContractFiles(unittest.TestCase):
    """The frozen test files of a generated task (spec 3.1), the check that they still hash to the frozen contract,
    and the source snapshot recorded with a result -- on the toy task's layout."""

    def setUp(self):
        self.t = ToyTask()
        self.addCleanup(self.t.cleanup)
        for name in ("unittest.py", *ur.RUNTIME_FILES):
            self.t.write(name, f"# {name}\n")
        self.contract = os.path.join(self.t.task, "run", "ut_contract.json")

    def freeze(self, names=None, **over):
        names = ur.contract_file_list(self.t.task, self.t.meta()) if names is None else names
        doc = ur.compute_contract(self.t.task, names)
        doc["contract"] = ur.CONTRACT_NAME
        doc.update(over)
        _touch(self.contract, json.dumps(doc))
        return doc

    def test_the_contract_holds_the_overlay_controls_and_reference_data_but_no_build_cache(self):
        for rel in ("baseline_overlay/pkg/mod.py", "baseline_overlay/pkg/mod.pyc",
                    "baseline_overlay/pkg/__pycache__/mod.cpython-311.pyc", "controls/scale.py", "refdata/a/ref.bin",
                    "refdata/a/ref.pyo", "extra.json"):
            self.t.write(rel, "x\n")
        m = self.t.meta()
        m["controls"] = {"task": [{"name": "scale", "file": "controls/scale.py"}, {"name": "builtin"}, "bogus"]}
        m["ut_contract_files"] = ["refdata", "extra.json"]
        self.assertEqual(ur.contract_file_list(self.t.task, m), sorted([
            "unittest.py", "cases.py", "meta.json", *ur.RUNTIME_FILES, "extra.json",
            os.path.join("baseline_overlay", "pkg", "mod.py"), os.path.join("controls", "scale.py"),
            os.path.join("refdata", "a", "ref.bin")]))
        for rel, bad in (("kernel_src/toyop.py", "kernel_src/toyop.py"), ("run", "run/requested_kernel.json")):
            m["ut_contract_files"] = [rel]
            with self.assertRaises(ValueError) as cm:
                ur.contract_file_list(self.t.task, m)
            self.assertEqual(str(cm.exception),
                             f"contract files may not live in kernel_src/, run/ or _cand_overlay/: {[bad]}")

    def test_a_name_that_only_spells_its_way_into_kernel_src_or_run_is_refused_too(self):
        """The refusal is judged on the normalised path: './kernel_src/x' and 'baseline_overlay/../kernel_src/x' are
        kernel_src/x -- frozen, every candidate edit of the kernel would read as a test changed after freeze."""
        m = self.t.meta()
        msg = "contract files may not live in kernel_src/, run/ or _cand_overlay/: "
        for rel in ("./kernel_src/toyop.py", "baseline_overlay/../kernel_src/toyop.py"):
            m["ut_contract_files"] = [rel]
            with self.assertRaises(ValueError) as cm:
                ur.contract_file_list(self.t.task, m)
            self.assertEqual(str(cm.exception), msg + str([os.path.join("kernel_src", "toyop.py")]))
        m["ut_contract_files"] = []
        m["controls"] = {"task": [{"name": "ctl", "file": "./run/ctl.py"}]}
        with self.assertRaises(ValueError) as cm:
            ur.contract_file_list(self.t.task, m)
        self.assertEqual(str(cm.exception), msg + str([os.path.join("run", "ctl.py")]))
        m["controls"] = {}
        m["ut_contract_files"] = ["./cases.py"]          # a harmless spelling of a contract file: listed once
        names = ur.contract_file_list(self.t.task, m)
        self.assertEqual((names.count("cases.py"), "./cases.py" in names), (1, False))

    def test_a_frozen_contract_holds_until_a_file_changes_or_goes(self):
        doc = self.freeze()
        self.assertEqual(ur.check_contract(self.t.task), (doc["contract_sha256"], []))
        self.t.write("cases.py", "# edited after freeze\n")
        os.remove(os.path.join(self.t.task, "leg_runner.py"))
        self.assertEqual(ur.check_contract(self.t.task),
                         (doc["contract_sha256"], ["cases.py: changed after freeze", "leg_runner.py: missing"]))

    def test_a_contract_without_a_required_file_or_with_an_edited_digest_is_refused(self):
        self.freeze([n for n in ur.contract_file_list(self.t.task, self.t.meta()) if n != "ut_runner.py"])
        self.assertEqual(ur.check_contract(self.t.task)[1], ["ut_runner.py is not in the frozen contract"])
        self.freeze(contract_sha256="0" * 64)
        self.assertEqual(ur.check_contract(self.t.task), ("0" * 64, ["contract_sha256 does not match its file list"]))
        self.freeze(contract="geak_ut_contract/v0")
        self.assertEqual(ur.check_contract(self.t.task), (None, ["run/ut_contract.json is not geak_ut_contract/v1"]))
        os.remove(self.contract)
        self.assertEqual(ur.check_contract(self.t.task),
                         (None, ["run/ut_contract.json missing: freeze the task with generate_ut.py freeze"]))

    def test_the_source_snapshot_is_the_task_files_and_every_kernel_source(self):
        self.t.write("kernel_src/sub/k.hip", "// kernel\n")
        self.t.write("kernel_src/__pycache__/toyop.py", "# cached\n")
        self.t.write("kernel_src/notes.txt", "not a source\n")
        m = self.t.meta()
        m["kernel_extraction"]["source_delivery_files"] = ["harness_lib.py"]
        self.t.write_meta(m)
        snap = ur.source_snapshot(self.t.task)
        self.assertEqual([f["path"] for f in snap["files"]],
                         ["cases.py", "harness_lib.py", os.path.join("kernel_src", "sub", "k.hip"),
                          os.path.join("kernel_src", "toyop.py"), "meta.json", "unittest.py"])
        self.assertEqual(snap["files"][-1],
                         {"path": "unittest.py", "sha256": hashlib.sha256(b"# unittest.py\n").hexdigest()})
        self.assertEqual(snap["snapshot_sha256"],
                         hashlib.sha256(json.dumps(snap["files"], separators=(",", ":")).encode()).hexdigest())
        for rel in ("kernel_src/sub/k.hip", "kernel_src/toyop.py"):
            os.remove(os.path.join(self.t.task, rel))
        with self.assertRaises(ValueError) as cm:
            ur.source_snapshot(self.t.task)
        self.assertEqual(str(cm.exception), "kernel_src contains no source files")

    def test_a_file_reference_never_leaves_the_task(self):
        os.symlink(os.path.join(self.t.site, "toyop.py"), os.path.join(self.t.task, "linked.py"))
        for name in ("../site/toyop.py", "linked.py", "."):
            with self.assertRaises(ValueError) as cm:
                ur.file_ref(self.t.task, name)
            self.assertEqual(str(cm.exception), f"file outside task: {name}")
        self.assertEqual(ur.file_ref(self.t.task, "./cases.py"),
                         {"path": "cases.py", "sha256": ur.sha256_file(os.path.join(self.t.task, "cases.py"))})


class _LegHarness:
    """harness_lib for Legs / run_correctness: `_run_leg` stands in for the leg_runner.py processes (none is started)
    and writes what they write -- `oracle` the native blob (out), `refs` the refs blob beside it and its result
    (--out), `correct` the judged result (--out). `results[mode]` is what a mode reports, or the exception its process
    ends with."""

    def __init__(self, **results):
        self.results, self.calls = results, []

    def _run_leg(self, task, overlay, mode, *, timeout, **kw):
        self.calls.append(dict(kw, overlay=overlay, mode=mode, timeout=timeout))
        res = self.results.get(mode, {"ok": True})
        if isinstance(res, BaseException):
            raise res
        extra = list(kw.get("extra") or [])
        flags = dict(zip(extra[::2], extra[1::2]))
        if mode == "oracle":
            _touch(kw["out"], "native blob")
        if mode == "refs":
            _touch(flags["--blob"] + ".refs.pt", "refs blob")
        if "--out" in flags:
            _touch(flags["--out"], json.dumps(res))
        return res


class TestLegs(unittest.TestCase):
    def test_a_leg_gets_command_s_or_what_the_whole_ut_has_left(self):
        _patch(self, ur, time=_Clock(t=1000.0))
        self.assertEqual(ur.Legs("/t", {}, None, command_s=0, deadline=None).budget(), 3600)
        self.assertEqual(ur.Legs("/t", {}, None, command_s=120, deadline=None).budget(), 120)
        self.assertEqual(ur.Legs("/t", {}, None, command_s=120, deadline=1100.0).budget(), 100.0)
        with self.assertRaises(ur.UTStop) as cm:
            ur.Legs("/t", {}, None, command_s=120, deadline=1005.0).budget()
        self.assertEqual((cm.exception.code, cm.exception.gate, cm.exception.reason),
                         (ur.EXIT_ENV, "timeout", "the whole-UT time limit (meta.timeouts.total_s) ran out"))

    def test_a_leg_runs_under_its_overlay_and_its_failure_says_whose_it_is(self):
        oom = RuntimeError("leg(correct) under /cand exited 1: torch.OutOfMemoryError: HIP out of memory")
        bug = RuntimeError("leg(refs) under /base exited 1: ZeroDivisionError: division by zero")
        h = _LegHarness(resolve={"identity": "site/toyop.py"}, correct=oom, refs=bug,
                        time=subprocess.TimeoutExpired(["leg_runner.py"], 60))
        legs = ur.Legs("/task", {}, h, command_s=60, deadline=None)
        self.assertEqual(legs.run("/base", "resolve", "baseline", target_callable="toyop:op"),
                         {"identity": "site/toyop.py"})
        self.assertEqual(h.calls, [{"overlay": "/base", "mode": "resolve", "timeout": 60,
                                    "target_callable": "toyop:op"}])
        with self.assertRaises(ur.UTStop) as cm:
            legs.run("/cand", "time", "candidate", bucket="decode_m1")
        self.assertEqual((cm.exception.code, cm.exception.gate, cm.exception.reason),
                         (ur.EXIT_ENV, "timeout",
                          "candidate leg (time) exceeded its time limit (meta.timeouts.command_s / total_s)"))
        with self.assertRaises(ur.UTStop) as cm:
            legs.run("/cand", "correct", "candidate")
        self.assertEqual((cm.exception.code, cm.exception.gate, cm.exception.reason),
                         (ur.EXIT_ENV, "resources", f"candidate leg (correct) died of its environment (resources): {oom}"))
        with self.assertRaises(RuntimeError) as cm:
            legs.run("/base", "refs", "baseline")
        self.assertIs(cm.exception, bug)                 # the task's own failure: the caller judges it


class TestRunCorrectness(unittest.TestCase):
    """oracle + refs on the baseline once per contract, then the judged leg -- the leg processes emulated."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="geak_utcore_")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.native_dir = os.path.join(self.root, "run", "native")
        self.prefix = os.path.join(self.native_dir, "native")

    def legs(self, **results):
        h = _LegHarness(**results)
        return h, ur.Legs(self.root, {}, h, command_s=60, deadline=None)

    def judge(self, legs, **kw):
        args = {"native_dir": self.native_dir, "draws": 3, "contract_sha": "c1"}
        args.update(kw)
        return ur.run_correctness(legs, "/base", "/cand", **args)

    def test_the_native_legs_run_once_per_contract_and_the_judged_leg_every_time(self):
        native = {"ok": True, "incomplete": [], "native_noise": {"noise_rel": 0.0}, "report": {"default": {}}}
        judged = {"ok": True, "identity": {"toyop:op": "/cand/toyop.py"}}
        h, legs = self.legs(refs=native, correct=judged)
        seen = {}
        self.assertEqual(self.judge(legs, seen=seen), (native, judged))
        self.assertEqual(seen, {"native": native})
        self.assertEqual([(c["overlay"], c["mode"]) for c in h.calls],
                         [("/base", "oracle"), ("/base", "refs"), ("/cand", "correct")])
        self.assertEqual((h.calls[0]["out"], h.calls[0]["seed"], h.calls[0]["draws"]), (self.prefix + ".pt", 0, 3))
        self.assertEqual(h.calls[1]["extra"], ["--blob", self.prefix, "--out",
                                               os.path.join(self.native_dir, "refs_result.json"), "--draws", "3"])
        self.assertEqual(h.calls[2]["extra"], ["--blob", self.prefix, "--out",
                                               os.path.join(self.native_dir, "judged_candidate.json"), "--draws", "3"])
        with open(os.path.join(self.native_dir, "native.json")) as fh:
            self.assertEqual(json.load(fh), {"contract_sha256": "c1", "refs": native})
        h.calls.clear()
        spec = {"kind": "scale", "factor": 1.1}
        got = self.judge(legs, control="scale_out", control_spec=spec, reuse_native=True, who="control")
        self.assertEqual(got, (native, judged))
        self.assertEqual([c["mode"] for c in h.calls], ["correct"])             # the native results are reused
        self.assertEqual(h.calls[0]["extra"], ["--blob", self.prefix, "--out",
                                               os.path.join(self.native_dir, "judged_scale_out.json"), "--draws", "3",
                                               "--control", "scale_out", "--control-spec", json.dumps(spec)])
        h.calls.clear()
        self.judge(legs, reuse_native=True, contract_sha="c2")              # another contract: computed again
        os.remove(self.prefix + ".refs.pt")
        self.judge(legs, reuse_native=True, contract_sha="c2")              # a native blob is gone: computed again
        self.assertEqual([c["mode"] for c in h.calls], ["oracle", "refs", "correct"] * 2)

    def test_a_native_that_fails_its_own_checks_stops_the_ut_once_its_result_is_recorded(self):
        bad = {"ok": False, "incomplete": [], "errors": ["decode_m1: max_rel_err 0.5 > tol 0.01"],
               "report": {"default": {}}, "native_noise": {"noise_rel": 0.03}}
        h, legs = self.legs(refs=bad)
        seen = {}
        with self.assertRaises(ur.UTStop) as cm:
            self.judge(legs, seen=seen)
        stop = cm.exception
        self.assertEqual((stop.code, stop.gate), (ur.EXIT_INCOMPLETE, "native_self_check"))
        self.assertTrue(stop.reason.startswith("the NATIVE code fails this task's own checks (baseline refs leg)"))
        self.assertIn("so the tolerance must be at least 2 x noise_rel = 0.06", stop.reason)
        self.assertIn("decode_m1: max_rel_err 0.5 > tol 0.01", stop.reason)
        self.assertEqual(seen, {"native": bad})
        self.assertEqual([c["mode"] for c in h.calls], ["oracle", "refs"])      # the candidate never runs
        reasons = ["per_output names the positions ['2']", "a stateful task needs sequence_cases"]
        _h, legs = self.legs(refs={"ok": True, "incomplete": reasons})
        with self.assertRaises(ur.UTStop) as cm:
            self.judge(legs, contract_sha="c2")
        self.assertEqual((cm.exception.code, cm.exception.gate, cm.exception.reason),
                         (ur.EXIT_INCOMPLETE, "native_self_check", "; ".join(reasons)))

    def test_a_crashed_native_refs_leg_is_incomplete_and_a_crashed_judged_leg_a_failed_candidate(self):
        crash = RuntimeError("leg(refs) under /base exited 1: ImportError: cannot import name 'call' from 'cases'")
        _h, legs = self.legs(refs=crash)
        with self.assertRaises(ur.UTStop) as cm:
            self.judge(legs)
        self.assertEqual((cm.exception.code, cm.exception.gate, cm.exception.reason),
                         (ur.EXIT_INCOMPLETE, "native_self_check",
                          f"the NATIVE refs leg crashed running this task's cases: {crash}"))
        _h, legs = self.legs(refs=RuntimeError("leg(refs) under /base exited 1: HIP out of memory"))
        with self.assertRaises(ur.UTStop) as cm:
            self.judge(legs)
        self.assertEqual((cm.exception.code, cm.exception.gate), (ur.EXIT_ENV, "resources"))   # not the task's
        crash = RuntimeError("leg(correct) under /cand exited 1: ValueError: shapes differ")
        _h, legs = self.legs(refs={"ok": True}, correct=crash)
        self.assertEqual(self.judge(legs), ({"ok": True}, {
            "ok": False, "incomplete": [], "crashed": True, "report": {},
            "errors": [f"candidate correctness process crashed: {crash}"]}))


class _Harness:
    """The harness_lib functions weighting_mode / coverage_summary call; each gives its value or raises it."""

    def __init__(self, observed=None, graphs=None):
        self.observed, self.graphs = observed, graphs

    @staticmethod
    def _give(value):
        if isinstance(value, Exception):
            raise value
        return value

    def _observed_workload_cases(self, meta):
        return self._give(self.observed)

    def served_regimes(self, meta):
        return set()

    def deployment_graphs(self, meta):
        return self._give(self.graphs or {})


class TestResultHelpers(unittest.TestCase):
    def test_the_weighting_follows_the_workload_model(self):
        wl = {"workload": {"kind": "agentx_trace_replay"}}
        self.assertEqual(ur.weighting_mode(_Harness(observed={"a": {}}), {}), "unweighted")
        self.assertEqual(ur.weighting_mode(_Harness(observed={"a": {}}), wl), "observed")
        self.assertEqual(ur.weighting_mode(_Harness(observed=None), wl), "analytic")
        self.assertEqual(ur.weighting_mode(_Harness(observed=ValueError("evidence unreadable")), wl), "observed")

    def test_each_phase_gets_the_geomean_of_its_included_cases(self):
        rows = [{"regime": "decode", "speedup": 2.0, "included": True},
                {"regime": "decode", "speedup": 0.5, "included": True},
                {"regime": "decode", "speedup": 8.0, "included": False},
                {"regime": "prefill", "speedup": 4.0, "included": True},
                {"regime": "prefill", "speedup": float("nan"), "included": True},
                {"regime": "prefill", "speedup": 0.0, "included": True},
                {"speedup": 9.0, "included": True}]
        got = ur.phase_geomeans(rows)
        self.assertEqual({ph: (v["calls"], v["basis"]) for ph, v in got.items()},
                         {"decode": (2, "unweighted geomean"), "prefill": (1, "unweighted geomean"),
                          "": (1, "unweighted geomean")})
        for ph, want in (("decode", 1.0), ("prefill", 4.0), ("", 9.0)):
            self.assertAlmostEqual(got[ph]["speedup"], want)
        self.assertEqual(ur.phase_geomeans(None), {})

    def test_a_task_without_a_case_manifest_cannot_show_its_coverage(self):
        for meta in ({}, {"case_manifest": {"dims": ["regime"], "cases": "a"}}):
            self.assertEqual(ur.manifest_problems(meta, ["a"], {"a"}, None),
                             ["meta.case_manifest {dims, cases[]} is missing"])

    def test_the_coverage_records_each_deployment_graph_its_replay_groups_and_the_carried_state(self):
        meta = {"capabilities": {"stateful": True}, "case_manifest": {"dims": ["regime"], "cases": [
            {"sig": "a", "regime": "decode", "roles": ["timing", "correctness"]}, "junk",
            {"regime": "decode", "roles": ["timing"]}]}}
        graphs = {"": {"capture": "full", "dims": ["m"], "graphs": [{"m": 1}, {"m": 4}]},
                  "prefill": {"capture": "piecewise", "dims": ["m"], "graphs": [{"m": 8}],
                              "problems": ["no replay group records m=8"]}}
        groups = [{"group": "g0", "steps": 3, "carry_state": True}, {"group": "g1", "steps": 1}]
        rank0 = {"default": {"eager": [{"case": "a"}], "replay_groups": groups,
                             "graph_coverage": {"missing": [{"m": 4}]}},
                 "sequence": [{"case": "seq[0]:a"}, {"case": "seq[1]:a"}], "sequence_coverage": {"covered": ["warm"]}}
        cov = ur.coverage_summary(_Harness(graphs=graphs), meta, [rank0, {}], ["a"], "unweighted")
        self.assertEqual(cov["graph"], {
            "default": {"capture": "full", "dims": ["m"], "deployment_graphs": [{"m": 1}, {"m": 4}],
                        "groups": groups, "missing": [{"m": 4}], "problems": []},
            "prefill": {"capture": "piecewise", "dims": ["m"], "deployment_graphs": [{"m": 8}], "groups": None,
                        "missing": None, "problems": ["no replay group records m=8"]}})
        self.assertEqual(cov["state"], {"stateful": True, "carry_groups": [{"phase": "default", "group": "g0",
                                                                            "steps": 3}],
                                        "tiers": {"covered": ["warm"]}, "sequence_steps": 2})
        self.assertEqual(cov["ranks"], 2)
        self.assertEqual(cov["buckets"], [{"bucket": {"regime": "decode"}, "timing": ["a"], "correctness": ["a"],
                                           "correctness_checked": ["a"]}])     # the malformed entries form no bucket
        self.assertEqual(cov["calls"]["with_calls"], ["a"])
        broken = ur.coverage_summary(_Harness(graphs=KeyError("phase_regimes")), meta, None, None, "unweighted")
        self.assertEqual((broken["graph"], broken["graph_problem"]), ({}, "KeyError: 'phase_regimes'"))
        self.assertNotIn("ranks", broken)

    def test_the_last_full_result_is_current_only_at_exit_0_for_this_contract_and_snapshot(self):
        task = tempfile.mkdtemp(prefix="geak_utcore_")
        self.addCleanup(shutil.rmtree, task, True)
        path = os.path.join(task, "run", "ut_result.json")
        self.assertEqual(ur.full_result_status(task, "c1", "s1"),
                         {"path": "run/ut_result.json", "current": False, "why": "no full UT result"})
        ur.write_result(path, {"exit_code": 0, "contract_sha256": "c1", "snapshot_sha256": "s1"})
        self.assertEqual(os.listdir(os.path.dirname(path)), ["ut_result.json"])     # no temp file left behind
        self.assertEqual(ur.full_result_status(task, "c1", "s1"),
                         {"path": "run/ut_result.json", "current": True, "why": "exit 0, same contract and snapshot"})
        self.assertEqual(ur.full_result_status(task, "c2", "s2")["why"],
                         "another test contract; another source snapshot")
        ur.write_result(path, {"exit_code": 1, "contract_sha256": "c1", "snapshot_sha256": "s1"})
        self.assertEqual(ur.full_result_status(task, "c1", "s1"),
                         {"path": "run/ut_result.json", "current": False, "why": "its exit code is 1"})
        _touch(path, "{truncated")
        self.assertEqual(ur.full_result_status(task, "c1", "s1")["why"], "no full UT result")

    def test_a_timing_line_per_case_then_the_primary_metric(self):
        per = [{"sig": "a", "baseline_ms": 1.23456789, "optimized_ms": None, "speedup": None, "reps": 2,
                "timer": "cuda_event_graph", "legs": {"baseline": {"primed": True, "host_ms": 0.0123},
                                                      "current": {"primed": False}},
                "rank_ms": {"baseline": [1.0, 1.2], "current": [0.9, 1.1]},
                "inner_raised": {"declared": 1, "final": 16}},
               {"sig": "b", "baseline_ms": 2.0, "optimized_ms": 1.0, "speedup": 2.0, "reps": 1, "timer": "wall"}]
        line_a = ("case a: baseline_ms=1.23457 optimized_ms=None speedup=None reps=2 timer=cuda_event_graph "
                  'primed=True/False host_ms=0.0123/None rank_ms={"baseline": [1.0, 1.2], "current": [0.9, 1.1]} '
                  "inner=1->16")
        line_b = "case b: baseline_ms=2 optimized_ms=1 speedup=2 reps=1 timer=wall primed=None/None host_ms=None/None"
        for unweighted, last in ((True, "GEOMEAN_SPEEDUP=1.5"), (False, "GEAK_WEIGHTED_SPEEDUP=1.5")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ur.print_timing(per, 1.5, unweighted=unweighted)
            self.assertEqual(buf.getvalue().splitlines(), [line_a, line_b, last])

    def test_readings_that_cannot_give_a_floor_and_a_case_already_at_its_cap(self):
        self.assertIsNone(ur.floor_estimate(4, 0.01, 4, 0.01))          # the same inner twice
        self.assertIsNone(ur.floor_estimate(4, 0.01, 2, 0.02))          # the probe below the reading's inner
        self.assertIsNone(ur.floor_estimate(1, None, 16, 0.01))
        self.assertIsNone(ur.floor_estimate(1, 0.02, 16, 0.0))
        row = {"sig": "a", "inner": 4, "baseline_ms": 0.001, "optimized_ms": 0.001}
        asked = []
        self.assertIs(ur.raise_for_floor(row, lambda sig, inner: asked.append(inner), 1, log=asked.append), row)
        self.assertEqual((asked, sorted(row)), ([], ["baseline_ms", "inner", "optimized_ms", "sig"]))

    def test_per_output_noise_skips_what_is_not_a_record_and_a_ranks_tolerance_is_kept(self):
        got = ur.merge_output_noise([{"0": {"tol": 0.02, "noise_rel": 0.01}, "note": "not compared"},
                                     {"0": {"tol": 0.02, "noise_rel": float("inf"), "noise_rel_nonfinite": 1}}])
        self.assertEqual(got, {"0": {"tol": 0.02, "noise_rel": 0.01, "noise_rel_nonfinite": 1}})
        tol = {"tol": 0.02, "per_output": {"1": 0.001},
               "outputs": [{"output": "0", "tol": 0.02}, {"output": "1", "tol": 0.001}]}
        merged = ur.merge_correct([{"rank": 0, "ok": True, "tolerance": None},
                                   {"rank": 1, "ok": True, "tolerance": tol}])
        self.assertEqual((merged["ok"], merged["tolerance"]), (True, tol))
        self.assertNotIn("tolerance", ur.merge_correct([{"rank": 0, "ok": True}]))
        self.assertEqual((ur._median([]), ur._median([3.0, 1.0, 2.0]), ur._median([4.0, 1.0])), (None, 2.0, 2.5))


class _Sockets:
    """ut_runner's `socket` for free_port: binding a port in `taken` fails as an occupied port does, port 0 gets
    `ephemeral`; nothing is opened."""

    def __init__(self, taken, ephemeral=45678):
        self.taken, self.ephemeral, self.binds = set(taken), ephemeral, []

    def socket(self):
        return _Socket(self)


class _Socket:
    def __init__(self, owner):
        self.owner, self.addr = owner, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def bind(self, addr):
        self.owner.binds.append(addr)
        if addr[1] in self.owner.taken:
            raise OSError(98, "Address already in use")
        self.addr = (addr[0], addr[1] or self.owner.ephemeral)

    def getsockname(self):
        return self.addr


class _Draws:
    """ut_runner._PORT_RNG: the scripted ports, then `then` forever; every range drawn from is recorded."""

    def __init__(self, ports, then):
        self.ports, self.then, self.ranges = list(ports), then, []

    def randrange(self, lo, hi):
        self.ranges.append((lo, hi))
        return self.ports.pop(0) if self.ports else self.then


class TestPorts(unittest.TestCase):
    def test_the_ephemeral_floor_is_read_from_the_kernel_else_32768(self):
        for content, want in (("40000\t60999\n", 40000), ("", 32768), ("n/a 60999\n", 32768), (None, 32768)):
            def fake_open(path, *args, content=content, **kw):
                self.assertEqual(path, "/proc/sys/net/ipv4/ip_local_port_range")
                if content is None:
                    raise FileNotFoundError(2, "No such file or directory")
                return io.StringIO(content)
            with self.subTest(content=content), mock.patch.object(ur, "open", fake_open, create=True):
                self.assertEqual(ur._port_floor(), want)

    def test_a_port_is_drawn_below_the_ephemeral_range_never_twice_and_never_while_taken(self):
        sockets, draws = _Sockets(taken={30002, 30004}), _Draws([30001, 30002, 30003], then=30004)
        _patch(self, ur, _port_floor=lambda: 40000, _PORT_RNG=draws, _PORTS_GIVEN={30001}, socket=sockets)
        self.assertEqual(ur.free_port(), 30003)
        self.assertEqual(sockets.binds, [("", 30002), ("", 30003)])     # 30001 was given out before: not even probed
        self.assertEqual(ur._PORTS_GIVEN, {30001, 30003})
        self.assertEqual(set(draws.ranges), {(28000, 40000)})
        sockets.binds.clear()
        self.assertEqual(ur.free_port(), 45678)                         # 500 draws, every one taken: the OS picks
        self.assertEqual((len(sockets.binds), sockets.binds[-1]), (501, ("127.0.0.1", 0)))
        low = _Draws([3000], then=3000)
        _patch(self, ur, _port_floor=lambda: 5000, _PORT_RNG=low)
        self.assertEqual(ur.free_port(), 3000)
        self.assertEqual(low.ranges, [(1024, 5000)])                    # never a privileged port

    def test_the_leased_devices_are_the_lease_visibility_list(self):
        with mock.patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": " 2, 3,,-1", "CUDA_VISIBLE_DEVICES": "7"}):
            self.assertEqual(ur.leased_devices(), ["2", "3"])
            os.environ["HIP_VISIBLE_DEVICES"] = ""
            self.assertEqual(ur.leased_devices(), ["7"])
            del os.environ["HIP_VISIBLE_DEVICES"], os.environ["CUDA_VISIBLE_DEVICES"]
            self.assertEqual(ur.leased_devices(), [])


# ---- the rank side of a RankPool, emulated in-process (leg_runner.serve_main's protocol on the control dir)
PORT_TAKEN = ("torch.distributed.DistNetworkError: The server socket has failed to listen on any local network "
              "address. port: 29501, useIpv6: false, code: -98, name: EADDRINUSE, message: address already in use\n")
ROOT_TB = 'Traceback (most recent call last):\n  File "cases.py", line 9, in call\nValueError: candidate broke on rank 1\n'
PEER_TB = ("Traceback (most recent call last):\nRuntimeError: [../third_party/gloo/gloo/transport/tcp/pair.cc:534] "
           "Connection closed by peer [127.0.0.1]:29501\n")


class _Exit:
    """What a rank does instead of answering: it writes `reply` (if any), appends `log` to its log and exits with
    `code` (negative: killed by that signal)."""

    def __init__(self, code, log="", reply=None):
        self.code, self.log, self.reply = code, log, reply


def _raises(rank, tb):
    """serve_main when a command raises: {rank, error} as the command's reply, the traceback in the log, exit 1."""
    return _Exit(1, tb, {"rank": rank, "error": tb})


def _mode(work):
    return work if work == "setup" else work["mode"]


def _answer(proc, work, attempt):
    """A healthy rank: set up at once, every command answered at once (`list` names two cases)."""
    if work == "setup":
        return {"rank": proc.rank, "pid": proc.pid, "device": "cpu"}
    if work["mode"] == "list":
        return {"rank": proc.rank, "sigs": ["decode_m1", "decode_m4"]}
    return {"rank": proc.rank, "mode": work["mode"], "params": work["params"]}


class _Pipe:
    """A rank's stdin: a line reaches the rank at once; writing to a rank that is gone breaks the pipe."""

    def __init__(self, proc):
        self.proc, self.lines, self.closed, self.broken = proc, [], False, False

    def write(self, text):
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if self.broken or self.proc.returncode is not None:
            raise BrokenPipeError(32, "Broken pipe")
        self.lines.append(json.loads(text))
        self.proc.receive(self.lines[-1])
        return len(text)

    def flush(self):
        """Nothing is buffered."""

    def close(self):
        self.closed = True


class _Rank:
    """One `leg_runner.py --mode serve` process as RankPool sees it (Popen's interface), emulated in-process: nothing
    is started, signalled or waited for. It keeps serve_main's protocol on the pool's control dir -- setup writes
    rank<r>.ready.json, a command line on stdin writes rank<r>.<seq>.json, {"mode": "exit"} ends the rank with 0 --
    and does what the test's `act(rank, work, attempt)` says: a reply, None (still busy: it looks again whenever the
    parent sleeps) or an _Exit."""

    def __init__(self, side, argv, env, log_path):
        self.side, self.argv, self.env, self.log_path = side, argv, env, log_path
        self.rank, self.pid = int(env["RANK"]), 4000 + len(side.spawned)
        self.ctl = argv[argv.index("--ctl") + 1]
        self.stdin, self.returncode, self.todo, self.attempts = _Pipe(self), None, ["setup"], {}
        self.stubborn = self.ignores_term = self.vanished = False
        self.work()

    def receive(self, cmd):
        if cmd.get("mode") != "exit":
            self.todo.append(cmd)
            self.work()
        elif not self.stubborn:
            self.returncode = 0

    def work(self):
        while self.todo and self.returncode is None:
            work = self.todo[0]
            key = "ready" if work == "setup" else str(work["seq"])
            self.attempts[key] = self.attempts.get(key, 0) + 1
            got = self.side.act(self, work, self.attempts[key])
            if got is None:
                return
            self.todo.pop(0)
            if not isinstance(got, _Exit):
                self.reply(key, got)
                continue
            if got.reply is not None:
                self.reply(key, got.reply)
            with open(self.log_path, "a") as fh:
                fh.write(got.log)
            self.returncode = got.code

    def reply(self, key, obj):
        path = os.path.join(self.ctl, f"rank{self.rank}.{key}.json")
        _touch(path + ".tmp", json.dumps(obj))
        os.replace(path + ".tmp", path)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        return self.returncode


class _RankSide:
    """subprocess.Popen and os.killpg for RankPool: every Popen is a _Rank run by `act` (default: healthy ranks);
    `signals` records every killpg as (rank, signal)."""

    def __init__(self, act=None):
        self.act, self.spawned, self.signals = act or _answer, [], []

    def popen(self, argv, **kw):
        proc = _Rank(self, argv, kw["env"], kw["stdout"].name)
        self.spawned.append((proc, kw))
        return proc

    def killpg(self, pgid, sig):
        proc = next(p for p, _ in self.spawned if p.pid == pgid)
        self.signals.append((proc.rank, sig))
        if proc.vanished:
            raise ProcessLookupError(3, "No such process")
        if not (sig == signal.SIGTERM and proc.ignores_term):
            proc.returncode = -sig

    def tick(self):
        for proc, _ in self.spawned:
            proc.work()


class _OsView:
    """ut_runner's `os` with the process-group calls replaced, so a test never signals a real process."""

    def __init__(self, **over):
        self.__dict__.update(over)

    def __getattr__(self, name):
        return getattr(os, name)


class TestRankPool(unittest.TestCase):
    """RankPool, the parent side of the multi-rank UT, against emulated rank processes (_Rank): no process is started,
    signalled or waited for, and the clock only advances."""

    def setUp(self):
        self.t = RankToyTask()
        self.addCleanup(self.t.cleanup)
        m = self.t.meta()
        m["regime"]["env"] = {"VLLM_ROCM_USE_AITER": "1", "ROCR_VISIBLE_DEVICES": "0,1"}
        self.t.write_meta(m)
        self.dirs = os.path.join(self.t.root, "pools")
        os.makedirs(self.dirs)
        self.side, self.clock, ports = None, _Clock(tick=lambda: self.side.tick()), iter(range(29501, 29600))
        env = mock.patch.dict(os.environ, {"PYTHONPATH": "/site"})
        env.start()
        self.addCleanup(env.stop)
        for var in ur.VISIBILITY_VARS:
            os.environ.pop(var, None)
        _patch(self, ur, time=self.clock, free_port=lambda: next(ports),
               os=_OsView(killpg=lambda pgid, sig: self.side.killpg(pgid, sig), getpgid=lambda pid: pid),
               subprocess=types.SimpleNamespace(Popen=lambda argv, **kw: self.side.popen(argv, **kw),
                                                PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT,
                                                TimeoutExpired=subprocess.TimeoutExpired),
               tempfile=types.SimpleNamespace(mkdtemp=lambda prefix: tempfile.mkdtemp(prefix=prefix, dir=self.dirs)))

    def pool(self, act=None, **kw):
        self.side = _RankSide(act)
        args = {"label": "cand", "world": 2, "devices": ["4", "5", "6"], "setup_s": 60, "command_s": 30,
                "deadline": None, "env_extra": {"GEAK_UT_TEST": "1"}}
        args.update(kw)
        return ur.RankPool(self.t.task, "/ovl", **args)

    def test_a_sub_second_time_limit_reaches_the_ranks_as_whole_seconds_of_at_least_one(self):
        pool = self.pool(setup_s=0.4, command_s=29.5)
        self.assertEqual({(proc.env["GEAK_UT_SETUP_S"], proc.env["GEAK_UT_COMMAND_S"]) for proc, _ in self.side.spawned},
                         {("1", "30")})                     # int() made them "0" and "29": a 0 s gloo timeout
        pool.close(kill=True)

    def test_a_pool_starts_its_ranks_runs_numbered_commands_and_exits_cleanly(self):
        pool = self.pool()
        self.assertEqual((pool.port, pool.seq, pool.closed, os.path.dirname(pool.dir)), (29501, 0, False, self.dirs))
        self.assertTrue(os.path.basename(pool.dir).startswith("geak_ut_pool_cand_"))
        for r, (proc, kw) in enumerate(self.side.spawned):
            self.assertEqual(proc.argv, [sys.executable, os.path.join(self.t.task, "leg_runner.py"), "--task",
                                         self.t.task, "--mode", "serve", "--ctl", pool.dir])
            self.assertEqual((kw["stdin"], kw["stderr"], kw["start_new_session"], kw["text"]),
                             (subprocess.PIPE, subprocess.STDOUT, True, True))
            self.assertEqual(kw["stdout"].name, os.path.join(pool.dir, f"rank{r}.log"))
            self.assertTrue(kw["stdout"].closed)                     # the parent keeps no handle on a rank's log
            env = proc.env
            self.assertEqual((env["RANK"], env["LOCAL_RANK"], env["WORLD_SIZE"], env["MASTER_ADDR"],
                              env["MASTER_PORT"]), (str(r), str(r), "2", "127.0.0.1", "29501"))
            self.assertEqual((env["HIP_VISIBLE_DEVICES"], env["CUDA_VISIBLE_DEVICES"]), ("4,5", "4,5"))
            self.assertEqual((env["GEAK_UT_SETUP_S"], env["GEAK_UT_COMMAND_S"], env["GEAK_UT_TEST"]), ("60", "30", "1"))
            self.assertEqual((env["PYTHONPATH"], env["VLLM_ROCM_USE_AITER"]), (os.pathsep.join(["/ovl", "/site"]), "1"))
            self.assertNotIn("ROCR_VISIBLE_DEVICES", env)
        self.assertEqual(pool.wait_ready(), [{"rank": 0, "pid": 4000, "device": "cpu"},
                                             {"rank": 1, "pid": 4001, "device": "cpu"}])
        self.assertEqual(pool.run("list"), [{"rank": 0, "sigs": ["decode_m1", "decode_m4"]},
                                            {"rank": 1, "sigs": ["decode_m1", "decode_m4"]}])
        got = pool.run("time", bucket="decode_m4", inner=4)
        self.assertEqual([(g["rank"], g["params"]) for g in got], [(0, {"bucket": "decode_m4", "inner": 4}),
                                                                   (1, {"bucket": "decode_m4", "inner": 4})])
        for proc, _ in self.side.spawned:
            self.assertEqual(proc.stdin.lines, [{"seq": 1, "mode": "list", "params": {}},
                                                {"seq": 2, "mode": "time", "params": {"bucket": "decode_m4", "inner": 4}}])
        self.assertEqual(self.clock.slept, [])                       # every reply had landed: nothing waited
        pool.close()
        for proc, _ in self.side.spawned:
            self.assertEqual((proc.stdin.lines[-1], proc.stdin.closed, proc.returncode), ({"mode": "exit"}, True, 0))
        self.assertEqual(self.side.signals, [])
        self.assertFalse(os.path.exists(pool.dir))
        pool.close()                                                 # closing twice changes nothing
        self.assertEqual(len(self.side.spawned[0][0].stdin.lines), 3)

    def test_the_parent_waits_for_late_replies_and_a_reply_that_never_comes_ends_the_pool(self):
        def act(proc, work, attempt):
            mode = _mode(work)
            if mode == "hang" or (mode == "time" and attempt < 4):
                return None
            return _answer(proc, work, attempt)
        pool = self.pool(act, command_s=2)
        pool.wait_ready()
        self.assertEqual([g["params"] for g in pool.run("time", bucket="decode_m1")], [{"bucket": "decode_m1"}] * 2)
        self.assertEqual(self.clock.slept, [0.05] * 3)               # answered on their 4th look
        t0 = self.clock.t
        with self.assertRaises(ur.PoolError) as cm:
            pool.run("hang")
        self.assertEqual((cm.exception.kind, str(cm.exception)),
                         ("timeout", "pool cand: hang exceeded its time limit (command_s / total_s)"))
        self.assertAlmostEqual(self.clock.t - t0, 2.0, delta=0.06)
        self.assertEqual(self.side.signals, [(0, signal.SIGTERM), (1, signal.SIGTERM)])
        self.assertTrue(pool.closed)
        self.assertFalse(os.path.exists(pool.dir))

    def test_setup_waits_for_setup_s_or_what_the_whole_ut_has_left_but_at_least_a_second(self):
        pool = self.pool(lambda proc, work, attempt: None if work == "setup" else _answer(proc, work, attempt),
                         deadline=self.clock.t + 0.25)
        self.assertAlmostEqual(pool._limit(60), 0.25)
        t0 = self.clock.t
        with self.assertRaises(ur.PoolError) as cm:
            pool.wait_ready()
        self.assertEqual((cm.exception.kind, str(cm.exception)),
                         ("timeout", "pool cand: setup exceeded its time limit (setup_s / total_s)"))
        self.assertAlmostEqual(self.clock.t - t0, 1.0, delta=0.06)
        self.assertEqual((len(self.side.spawned), pool.port_retries), (2, 0))     # a timeout is not retried
        self.assertEqual([p.returncode for p, _ in self.side.spawned], [-signal.SIGTERM, -signal.SIGTERM])
        pool.deadline = None
        self.assertEqual(pool._limit(60), 60)

    def test_a_rank_that_dies_in_setup_ends_the_pool_and_is_judged_by_its_exit_and_its_log(self):
        for death, kind, msg in (
                (_Exit(1, "Traceback (most recent call last):\nRuntimeError: setup failed on purpose\n"), "task",
                 "pool native: setup: rank 1 exited 1"),
                (_Exit(1, "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 2.00 GiB\n"), "env",
                 "pool native: setup: rank 1 stopped by its environment (resources)"),
                (_Exit(-9), "env", "pool native: setup: rank 1 killed by signal 9")):
            with self.subTest(msg=msg):
                pool = self.pool(lambda proc, work, attempt, death=death: death if work == "setup" and proc.rank == 1
                                 else _answer(proc, work, attempt), label="native")
                with self.assertRaises(ur.PoolError) as cm:
                    pool.wait_ready()
                self.assertEqual((cm.exception.kind, str(cm.exception), cm.exception.rank, cm.exception.tail),
                                 (kind, msg, 1, death.log))
                self.assertEqual((self.side.signals, len(self.side.spawned)), ([(0, signal.SIGTERM)], 2))
                self.assertEqual(self.clock.slept[-1], 0.5)          # the dead ranks' files and logs get time to land
                self.assertFalse(os.path.exists(pool.dir))

    def test_a_taken_rendezvous_port_respawns_the_pool_on_a_new_port_at_most_three_times(self):
        def act(proc, work, attempt):
            if work == "setup" and proc.rank == 0 and proc.env["MASTER_PORT"] == "29501":
                return _Exit(1, PORT_TAKEN)
            return _answer(proc, work, attempt)
        pool = self.pool(act)
        first = pool.dir
        self.assertEqual([r["rank"] for r in pool.wait_ready()], [0, 1])
        self.assertEqual((pool.port, pool.port_retries, pool.closed), (29502, 1, False))
        self.assertEqual([p.env["MASTER_PORT"] for p, _ in self.side.spawned], ["29501", "29501", "29502", "29502"])
        self.assertEqual(self.side.signals, [(1, signal.SIGTERM)])   # the first pool's rank left at the rendezvous
        self.assertFalse(os.path.exists(first))
        self.assertEqual(pool.run("list")[1]["sigs"], ["decode_m1", "decode_m4"])
        pool.close()
        pool = self.pool(lambda proc, work, attempt: _Exit(1, PORT_TAKEN) if work == "setup" and proc.rank == 0
                         else _answer(proc, work, attempt))
        with self.assertRaises(ur.PoolError) as cm:
            pool.wait_ready()
        self.assertEqual((cm.exception.kind, str(cm.exception), cm.exception.rank),
                         ("port", "pool cand: rendezvous port 29506 taken", 0))
        self.assertEqual((pool.port_retries, len(self.side.spawned)), (3, 8))
        self.assertEqual([p.env["MASTER_PORT"] for p, _ in self.side.spawned[::2]], ["29503", "29504", "29505", "29506"])

    def test_a_raising_command_ends_the_pool_and_names_its_root_cause_not_the_peer_that_saw_it(self):
        def act(proc, work, attempt):
            if _mode(work) == "correct":
                return _raises(proc.rank, PEER_TB if proc.rank == 0 else ROOT_TB)
            return _answer(proc, work, attempt)
        pool = self.pool(act)
        pool.wait_ready()
        with self.assertRaises(ur.PoolError) as cm:
            pool.run("correct", blob="/run/native/native", draws=2)
        err = cm.exception
        self.assertEqual((err.kind, err.rank, str(err)), ("task", 1, "pool cand: correct: rank 1 raised"))
        self.assertEqual(err.tail, ROOT_TB + "\n[other ranks] rank 0: " + PEER_TB.strip().splitlines()[-1])
        self.assertEqual((self.side.signals, pool.closed), ([], True))      # both ranks had exited

        def act2(proc, work, attempt):
            if _mode(work) == "correct":
                return _raises(1, ROOT_TB) if proc.rank == 1 else None      # rank 0 waits in the collective
            return _answer(proc, work, attempt)
        pool = self.pool(act2)
        pool.wait_ready()
        with self.assertRaises(ur.PoolError) as cm:
            pool.run("correct")
        self.assertEqual((cm.exception.kind, cm.exception.rank, cm.exception.tail), ("task", 1, ROOT_TB))
        self.assertEqual(self.side.signals, [(0, signal.SIGTERM)])

    def test_a_raised_error_is_the_environments_when_its_text_says_so(self):
        pool = self.pool()
        self.assertIsNone(pool._raised([], "time"))
        self.assertIsNone(pool._raised([{"rank": 0, "cases": []}, {"rank": 1, "cases": []}], "time"))
        oom = pool._raised([{"rank": 0, "error": "Traceback\ntorch.OutOfMemoryError: HIP out of memory"}], "time")
        self.assertEqual((oom.kind, oom.rank, str(oom)), ("env", 0, "pool cand: time: rank 0 raised"))
        peers = pool._raised([{"rank": 0, "error": "x\nConnection reset by peer\n"},
                              {"rank": 1, "error": "y\nNCCL communicator was aborted\n"}], "time")
        self.assertEqual((peers.kind, peers.rank, str(peers)),       # only consequences, and nobody died: no verdict
                         ("env", 0, "pool cand: time: the ranks that raised only saw a peer fail, and no other rank died"))
        self.assertTrue(peers.tail.endswith("\n[ranks that saw it] rank 0: Connection reset by peer; "
                                            "rank 1: NCCL communicator was aborted"))
        pool.close()

    def test_when_every_rank_that_raised_saw_a_peer_fail_the_rank_that_did_not_raise_is_judged(self):
        # e.g. the OOM killer: rank 1 is SIGKILLed with no traceback, ranks 0 and 2 raise "connection closed by
        # peer" and exit 1 -- the run is the environment's (retried), not a candidate failure
        peer = PEER_TB.strip().splitlines()[-1]
        for dead, want in ((_Exit(-9), ("env", "pool cand: correct: rank 1 killed by signal 9")),
                           (_Exit(-6, "HIP out of memory\n"),
                            ("env", "pool cand: correct: rank 1 stopped by its environment (resources)")),
                           (_Exit(3, "SystemExit: 3\n"), ("task", "pool cand: correct: rank 1 exited 3"))):
            with self.subTest(want=want[1]):
                def act(proc, work, attempt, dead=dead):
                    if _mode(work) == "correct":
                        return dead if proc.rank == 1 else _raises(proc.rank, PEER_TB)
                    return _answer(proc, work, attempt)
                pool = self.pool(act, world=3)
                pool.wait_ready()
                with self.assertRaises(ur.PoolError) as cm:
                    pool.run("correct")
                err = cm.exception
                self.assertEqual((err.kind, str(err), err.rank, err.pool), (*want, 1, "cand"))
                self.assertEqual(err.tail, dead.log + f"\n[ranks that saw it] rank 0: {peer}; rank 2: {peer}")

    def test_a_rank_gone_before_a_command_fails_it_and_a_port_error_only_counts_in_setup(self):
        pool = self.pool()
        pool.wait_ready()
        gone = self.side.spawned[1][0]
        gone.returncode = 1
        with open(gone.log_path, "a") as fh:
            fh.write("Traceback (most recent call last):\nValueError: the rank died between commands\n")
        with self.assertRaises(ur.PoolError) as cm:
            pool.run("correct")
        self.assertEqual((cm.exception.kind, str(cm.exception), cm.exception.rank),
                         ("task", "pool cand: correct: rank stdin closed: rank 1 exited 1", 1))
        self.assertIn("the rank died between commands", cm.exception.tail)
        with open(gone.log_path, "a") as fh:
            fh.write(PORT_TAKEN)
        self.assertEqual((pool._failure("time").kind, pool._failure("setup").kind), ("task", "port"))
        self.assertEqual(pool._log(5), "")                           # no such rank: no log
        pool.close(kill=True)
        self.assertEqual(self.side.signals, [(0, signal.SIGTERM)])
        self.assertEqual(pool._log(0), "")                           # the control dir went with the pool

    def test_close_asks_every_rank_to_exit_then_terminates_then_kills_what_is_left(self):
        pool = self.pool(world=3, devices=[])
        pool.wait_ready()
        r0, r1, r2 = (p for p, _ in self.side.spawned)
        self.assertNotIn("HIP_VISIBLE_DEVICES", r0.env)              # CPU ranks: no device selection at all
        r0.stdin.closed = r0.vanished = True     # its stdin is already closed, and it is gone when signalled
        r1.stdin.broken = True                   # its stdin pipe is broken, it lives on
        r2.stubborn = r2.ignores_term = True     # it ignores the exit line and SIGTERM
        t0 = self.clock.t
        pool.close(grace=1.0)
        self.assertAlmostEqual(self.clock.t - t0, 1.0, delta=0.06)
        self.assertEqual(self.side.signals,
                         [(0, signal.SIGTERM), (1, signal.SIGTERM), (2, signal.SIGTERM), (2, signal.SIGKILL)])
        self.assertEqual((r1.returncode, r2.returncode), (-signal.SIGTERM, -signal.SIGKILL))
        self.assertEqual((r0.stdin.lines, r1.stdin.lines, r2.stdin.lines, r2.stdin.closed),
                         ([], [], [{"mode": "exit"}], True))
        self.assertFalse(os.path.exists(pool.dir))


class _TimingPool:
    """measure_pools' view of a RankPool: `list` names the scripted cases; `time` of a bucket gives every rank the
    bucket's next scripted samples (one list per call, the last one repeated; None: the case produced no row)."""

    def __init__(self, script, world=2):
        self.script, self.world, self.calls = script, world, []

    def run(self, mode, **params):
        if mode == "list":
            return [{"rank": r, "sigs": list(self.script)} for r in range(self.world)]
        sig = params["bucket"]
        self.calls.append((sig, params.get("inner")))
        seq = self.script[sig]
        samples = seq[min(sum(c[0] == sig for c in self.calls), len(seq)) - 1]
        if samples is None:
            return [{"rank": r, "cases": []} for r in range(self.world)]
        ms = sorted(samples)[len(samples) // 2] if samples else None
        row = {"sig": sig, "regime": "decode", "m": 4, "timer": "cuda_event_graph",
               "cache_condition": {"mode": "read-evict"}, "ms": ms, "wall_ms": ms, "host_ms": 0.01, "primed": True,
               "samples": samples, "alignment": {"samples": len(samples), "late_releases": 1, "max_late_us": 3.0,
                                                 "device_sync": "rccl_all_reduce_1"}}
        if params.get("inner"):
            row["inner"] = params["inner"]
        return [{"rank": r, "cases": [dict(row)]} for r in range(self.world)]


class TestMeasurePools(unittest.TestCase):
    def test_pairs_stop_once_decisive_or_after_max_reps_and_a_case_without_a_reading_is_dropped(self):
        pn = _TimingPool({"a": [[2.0, 2.0, 2.0]], "b": [[1.0], [1.1], [1.0]], "c": [[]], "d": [None]})
        pc = _TimingPool({"a": [[1.0, 1.0, 1.0]], "b": [[1.0]], "c": [[1.0]], "d": [[1.0]]})
        per_case, aligned = ur.measure_pools(pn, pc, h=hl)
        self.assertEqual([r["sig"] for r in per_case], ["a", "b", "c"])              # d produced no row
        a, b, c = per_case
        self.assertEqual((a["baseline_ms"], a["optimized_ms"], a["speedup"], a["reps"], a["speedup_spread"]),
                         (2.0, 1.0, 2.0, 1, 0.0))                                   # decisive after one pair
        self.assertEqual(a["rank_ms"], {"baseline": [2.0, 2.0], "current": [1.0, 1.0]})
        self.assertEqual(a["legs"]["baseline"], {"timer": "cuda_event_graph", "cache_condition": {"mode": "read-evict"},
                                                 "ms_reps": [2.0], "primed": True, "host_ms": 0.01,
                                                 "rank_ms": [2.0, 2.0]})
        self.assertNotIn("inner", a)
        self.assertEqual((b["reps"], b["baseline_ms"], b["optimized_ms"]), (3, 1.0, 1.0))  # undecided: every pair
        self.assertAlmostEqual(b["speedup_spread"], 0.1)
        self.assertEqual((c["baseline_ms"], c["optimized_ms"], c["speedup"], c["reps"]), (None, 1.0, None, 1))
        self.assertEqual(pn.calls, [("a", None), ("b", None), ("b", None), ("b", None), ("c", None), ("d", None)])
        self.assertEqual(pc.calls, pn.calls)
        self.assertEqual(len(aligned), 2 * 5)
        self.assertEqual(aligned[0], {"samples": 3, "late_releases": 2, "max_late_us": 3.0,
                                      "device_sync": "rccl_all_reduce_1"})

    def test_one_case_at_a_given_inner_times_every_pair_without_the_early_stop(self):
        pn, pc = _TimingPool({"a": [[2.0]], "b": [[1.0]]}), _TimingPool({"a": [[1.0]], "b": [[1.0]]})
        per_case, aligned = ur.measure_pools(pn, pc, h=hl, early_stop=False, sigs=["a"], inner={"a": 16})
        self.assertEqual([(r["sig"], r["reps"], r["inner"]) for r in per_case], [("a", 3, 16)])
        self.assertEqual((pn.calls, pc.calls, len(aligned)), ([("a", 16)] * 3, [("a", 16)] * 3, 6))

    def test_the_alignment_receipt_names_the_device_step_the_ranks_used(self):
        rows = [{"samples": 50, "late_releases": 1, "max_late_us": 12.5, "device_sync": "rccl_all_reduce_1"},
                {"samples": 50, "late_releases": 2, "max_late_us": 40.0, "device_sync": "rccl_all_reduce_1"}]
        env = {"AITER_CUSTOM_AR_RAW_INPUT_POOL": "1 (test-harness setting)"}
        rc = ur.alignment_receipt({"timing": {"device_spin_us": 150, "release_margin_us": 300}}, rows, env)
        self.assertEqual(rc["per_sample"],
                         "harness_lib.flush_cache -> torch.cuda.synchronize -> gloo all_reduce(MAX) barrier agreeing a "
                         "release time (now + 300 us) -> busy-wait to the release time -> GPU spin 150 us on the rank's "
                         "stream -> one-element RCCL all_reduce on the pool's device group -> start event -> kernel -> "
                         "end event")
        self.assertEqual({k: rc[k] for k in ("release_margin_us", "device_spin_us", "device_sync", "samples",
                                             "late_releases", "max_late_us", "harness_env")},
                         {"release_margin_us": 300.0, "device_spin_us": 150.0, "device_sync": "rccl_all_reduce_1",
                          "samples": 100, "late_releases": 3, "max_late_us": 40.0, "harness_env": env})
        rc = ur.alignment_receipt({}, [{"samples": 5}, {"samples": 5, "device_sync": "cases.align"}], {})
        self.assertIn("busy-wait to the release time -> GPU spin 200 us on the rank's stream -> the task's cases.align "
                      "(no tested data) -> start event", rc["per_sample"])
        self.assertEqual((rc["device_spin_us"], rc["release_margin_us"], rc["samples"]), (200.0, 500.0, 10))
        for aligned in ([{"samples": 5, "device_sync": "none (no CUDA)"}], []):
            rc = ur.alignment_receipt({}, aligned, {})
            self.assertIn("busy-wait to the release time -> (no CUDA: no device step) -> start event", rc["per_sample"])
            self.assertIsNone(rc["device_spin_us"])
        self.assertEqual((rc["samples"], rc["late_releases"], rc["max_late_us"], rc["device_sync"]), (0, 0, 0.0, None))

    def test_a_pool_failure_is_the_environments_the_tasks_or_the_candidates(self):
        err = ur.PoolError("env", "pool candidate: setup: rank 0 killed by signal 9")
        self.assertEqual((err.kind, err.rank, err.tail), ("env", None, ""))
        stop = ur._pool_stop(err, "candidate")
        self.assertEqual((stop.code, stop.gate, stop.reason),
                         (ur.EXIT_ENV, "multi_rank", "pool candidate: setup: rank 0 killed by signal 9 (candidate)"))
        for kind in ("timeout", "port"):
            stop = ur._pool_stop(ur.PoolError(kind, f"pool native: {kind}", 0, "x" * 1600 + "TAIL"), "native")
            self.assertEqual((stop.code, stop.gate, stop.reason),
                             (ur.EXIT_ENV, "multi_rank", f"pool native: {kind} (native): " + "x" * 1496 + "TAIL"))
        task = ur.PoolError("task", "pool native_refs: refs: rank 1 raised", 1, "ValueError: a case is malformed")
        for who in ("baseline", "native", "aa", "control"):
            stop = ur._pool_stop(task, who)
            reason = (f"the {who} pool's own code failed on this task's cases: pool native_refs: refs: rank 1 raised: "
                      "ValueError: a case is malformed")
            self.assertEqual((stop.code, stop.gate, stop.reason), (ur.EXIT_INCOMPLETE, "multi_rank", reason))
        stop = ur._pool_stop(task, "candidate")
        self.assertEqual((stop.code, stop.gate, stop.reason),
                         (ur.EXIT_FAIL, "correctness",
                          "the candidate failed: pool native_refs: refs: rank 1 raised: ValueError: a case is malformed"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
