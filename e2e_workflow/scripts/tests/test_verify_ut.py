"""CPU tests of verify_ut.py: case classification (ISSUE-059), recorded errors (ISSUE-060), control
judgements (ISSUE-033 / 028), the static part, and whole CPU verifications of a toy task."""
import contextlib
import errno
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import generate_ut as gu  # noqa: E402
import verify_ut as vu  # noqa: E402
from ut_task_fixture import ToyTask, have_torch  # noqa: E402

TOL = 0.02


def keep_sys_path(test):
    """Put sys.path back, in place, when `test` ends -- passed, failed or errored (a cleanup runs even when setUp
    fails): generate_ut.vendor(), which gu.main runs in this process, puts the scripts dir at the front of sys.path
    for every task it vendors into."""
    saved = list(sys.path)

    def restore():
        sys.path[:] = saved
    test.addCleanup(restore)


def _res(code, report):
    return {"exit_code": code, "correctness": {"candidate": {"ok": code == 0, "report": report}}}


def _rep(eager=(), random=(), sequence=None, replay=()):
    r = {"decode": {"eager": list(eager), "random": list(random), "graph_replay": list(replay)}}
    if sequence is not None:
        r["sequence"] = list(sequence)
    return r


def _c(case, ok, err, note=""):
    return {"case": case, "correct": ok, "max_rel_err": err, "note": note}


class TestClassification(unittest.TestCase):
    def test_raised_calls_are_never_catches_and_nonfinite_outputs_are(self):
        rows = vu.classify_cases(_rep(random=[_c("random[0]:a", False, None, "value-parity raised: boom"),
                                              _c("random[1]:a", False, None, "value-parity vs live baseline"),
                                              _c("random[2]:a", True, 0.0)]), TOL)
        self.assertEqual([r["kind"] for r in rows], ["raised", "nonfinite", "pass"])

    def test_errors_rounded_down_to_tol_are_recorded_rounded_up_and_named(self):
        rows = vu.classify_cases(_rep(eager=[_c("a", False, 0.02), _c("b", False, 0.05), _c("c", True, 0.01)]), TOL)
        rounded = vu.record_errors(rows, TOL)
        self.assertEqual(rounded, ["a"])
        self.assertEqual([r["err_recorded"] for r in rows], [0.02001, 0.05, 0.01])
        self.assertEqual(vu.static_record_problems(rows, TOL), [])

    def test_the_static_rule_refuses_a_record_that_contradicts_its_verdict(self):
        # correct() never reads a failing comparison at or below tol (up to 1): one recorded there failed on something
        # its error does not show. A passing case reading above tol is what correct() gives on a reference near 0.
        rows = vu.classify_cases(_rep(eager=[_c("x", False, 0.0015), _c("y", True, 0.5)]), TOL)
        self.assertEqual(vu.record_errors(rows, TOL), [])
        self.assertEqual([r["err_recorded"] for r in rows], [0.0015, 0.5])
        self.assertEqual(vu.static_record_problems(rows, TOL), ["x: failing at 0.0015 <= tol 0.02"])


class TestJudgeControl(unittest.TestCase):
    def test_noop_must_pass(self):
        v, d = vu.judge_control("noop", {}, 0, _res(0, _rep(eager=[_c("a", True, 0.0)])), TOL, 1)
        self.assertEqual(v, "PASS")
        v, d = vu.judge_control("noop", {}, 1, _res(1, _rep(eager=[_c("a", False, 0.1)])), TOL, 1)
        self.assertEqual(v, "FAIL")

    def test_scale_out_must_be_seen(self):
        v, d = vu.judge_control("scale_out", {}, 1, _res(1, _rep(eager=[_c("a", False, 0.08)])), TOL, 1)
        self.assertEqual(v, "PASS")
        v, d = vu.judge_control("scale_out", {}, 0, _res(0, _rep(eager=[_c("a", True, 0.0)])), TOL, 1)
        self.assertEqual(v, "FAIL")

    def test_a_control_seen_only_through_raising_calls_is_not_caught(self):
        v, d = vu.judge_control("scale_out", {}, 1, _res(1, _rep(random=[_c("random[0]:a", False, None,
                                                                            "value-parity raised: x")])), TOL, 1)
        self.assertEqual(v, "FAIL")
        self.assertIn("raising", d["why"])

    def test_stale_output_must_be_seen_in_the_random_draws(self):
        v, d = vu.judge_control("stale_output", {}, 1, _res(1, _rep(eager=[_c("a", False, 0.5)])), TOL, 1)
        self.assertEqual(v, "FAIL")
        v, d = vu.judge_control("stale_output", {}, 1, _res(1, _rep(random=[_c("random[1]:a", False, 0.5)])), TOL, 1)
        self.assertEqual(v, "PASS")

    def test_skip_state_write_must_be_seen_in_the_sequence(self):
        v, d = vu.judge_control("skip_state_write", {}, 1,
                                _res(1, _rep(sequence=[_c("seq[1]:s", False, 1.0)])), TOL, 1)
        self.assertEqual(v, "PASS")

    def test_an_environment_stop_has_no_verdict(self):
        v, d = vu.judge_control("scale_out", {}, 2, {"exit_code": 2, "reason": "HIP out of memory"}, TOL, 1)
        self.assertEqual(v, "INCONCLUSIVE")

    def test_task_controls_ignore_their_equivalent_cases_and_warn_near_tol(self):
        spec = {"expect": "reject", "equivalent_on": ["a"]}
        v, d = vu.judge_control("mine", spec, 1, _res(1, _rep(eager=[_c("a", False, 0.9)])), TOL, 1)
        self.assertEqual(v, "FAIL")
        self.assertIn("equivalent_on", d["why"])
        v, d = vu.judge_control("mine", spec, 1, _res(1, _rep(eager=[_c("a", False, 0.9),
                                                                    _c("b", False, 0.025)])), TOL, 1)
        self.assertEqual(v, "PASS")
        self.assertIn("1.5 x tol", d["warning"])

    def test_a_below_tol_control_is_verified_and_still_a_gap(self):
        spec = {"expect": "below_tol"}
        v, d = vu.judge_control("tiny", spec, 0, _res(0, _rep(eager=[_c("a", True, 0.003)])), TOL, 1)
        self.assertEqual(v, "GAP")
        v, d = vu.judge_control("tiny", spec, 0, _res(0, _rep(eager=[_c("a", True, 0.0)])), TOL, 1)
        self.assertEqual(v, "FAIL")                         # the bug never acted
        v, d = vu.judge_control("tiny", spec, 1, _res(1, _rep(eager=[_c("a", False, 0.3)])), TOL, 1)
        self.assertEqual(v, "FAIL")                         # not below tol after all

    def test_rank_skew_must_fail_on_the_last_rank(self):
        judged = {"ok": False, "ranks": [{"rank": 0, "report": _rep(eager=[_c("a", True, 0.0)])},
                                         {"rank": 1, "report": _rep(eager=[_c("a", False, 0.4)])}]}
        v, d = vu.judge_control("rank_skew", {}, 1, {"exit_code": 1, "correctness": {"candidate": judged}}, TOL, 2)
        self.assertEqual(v, "PASS")
        judged["ranks"][1]["report"] = _rep(eager=[_c("a", True, 0.0)])
        judged["ranks"][0]["report"] = _rep(eager=[_c("a", False, 0.4)])
        v, d = vu.judge_control("rank_skew", {}, 1, {"exit_code": 1, "correctness": {"candidate": judged}}, TOL, 2)
        self.assertEqual(v, "FAIL")
        # the perturbed rank is world - 1 even when fewer ranks reported (a pool that stopped): rank 1 failing for
        # another reason is no catch on rank 3, and a crashed control's one-rank record names the right rank
        judged["ranks"][1]["report"] = _rep(eager=[_c("a", False, 0.4)])
        v, d = vu.judge_control("rank_skew", {}, 1, {"exit_code": 1, "correctness": {"candidate": judged}}, TOL, 4)
        self.assertEqual((v, d["why"]), ("FAIL", "the perturbed rank 3 has no correctness report (reports: rank 0, 1)"))
        crashed = {"ok": False, "crashed": True, "ranks": [{"rank": 1, "ok": False, "report": {"raised": {"error": "x"}}}]}
        v, d = vu.judge_control("rank_skew", {}, 1, {"exit_code": 1, "correctness": {"candidate": crashed}}, TOL, 4)
        self.assertEqual((v, d["why"]), ("FAIL", "the perturbed rank 3 has no correctness report (reports: rank 1)"))


class TestStatic(unittest.TestCase):
    def setUp(self):
        keep_sys_path(self)
        self.t = ToyTask()
        self.addCleanup(self.t.cleanup)
        self.assertEqual(gu.main(["all", self.t.task]), 0)

    def verify_static(self):
        out = os.path.join(self.t.task, "run", "verify_static.json")
        code = vu.main(["--task", self.t.task, "--static-only", "--out", out])
        return code, self.t.read_json("run/verify_static.json")

    def test_a_generated_task_passes_the_static_part(self):
        code, doc = self.verify_static()
        self.assertEqual((code, doc["verdict"]), (0, "STATIC_PASS"), doc["errors"])

    def test_an_edited_runtime_file_fails(self):
        with open(os.path.join(self.t.task, "leg_runner.py"), "a") as fh:
            fh.write("# local edit\n")
        code, doc = self.verify_static()
        self.assertEqual(doc["verdict"], "FAIL")
        self.assertTrue(any("leg_runner.py differs" in e for e in doc["errors"]))

    def test_a_hand_edited_unittest_fails(self):
        with open(os.path.join(self.t.task, "unittest.py"), "a") as fh:
            fh.write("# edit\n")
        code, doc = self.verify_static()
        self.assertTrue(any("not the generated entry" in e for e in doc["errors"]))


@unittest.skipUnless(have_torch(), "torch not installed")
class TestToyVerification(unittest.TestCase):
    def setUp(self):
        keep_sys_path(self)
        self.t = ToyTask()
        self.addCleanup(self.t.cleanup)

    def _verify(self):
        self.assertEqual(gu.main(["all", self.t.task]), 0)
        p = subprocess.run([sys.executable, "unittest.py"], cwd=self.t.task, env=self.t.env(),
                           capture_output=True, text=True, timeout=900)
        self.assertEqual(p.returncode, 0, p.stdout[-2000:] + p.stderr[-2000:])
        p = subprocess.run([sys.executable, vu.__file__, "--task", self.t.task], cwd=self.t.task,
                           env=self.t.env(), capture_output=True, text=True, timeout=1800)
        return p.returncode, self.t.read_json("run/verify_ut.json")

    def test_a_sound_task_passes_and_reuses_the_ut_timing_as_its_aa(self):
        code, doc = self._verify()
        self.assertEqual((code, doc["verdict"]), (0, "PASS"), doc["errors"])
        self.assertEqual(doc["aa"]["runs"][0]["source"], "run/ut_result.json")
        self.assertEqual({k: v["verdict"] for k, v in doc["controls"].items()},
                         {"noop": "PASS", "within_tol": "PASS", "scale_out": "PASS", "stale_output": "PASS"})
        snap = __import__("ut_runner").source_snapshot(self.t.task)["snapshot_sha256"]
        self.assertEqual(doc["snapshot_sha256"], snap)

    def test_a_test_that_cannot_see_its_outputs_fails(self):
        cases = open(os.path.join(self.t.task, "cases.py")).read().replace(
            "return toyop.op(*args)", "return toyop.op(*args) * 0")
        self.t.write("cases.py", cases)
        code, doc = self._verify()
        self.assertEqual(doc["verdict"], "FAIL")
        self.assertEqual(doc["controls"]["scale_out"]["verdict"], "FAIL")
        self.assertEqual(doc["controls"]["stale_output"]["verdict"], "FAIL")

    def test_a_modified_copy_is_measured_with_aa_and_noted(self):
        # The copied source differs from its origin (same behaviour): the A/A cannot go through the
        # candidate binding with the baseline's code, so both legs run the baseline (--aa) and it is said.
        self.t.write("kernel_src/toyop.py", open(os.path.join(self.t.task, "kernel_src", "toyop.py")).read()
                     + "# a local note\n")
        code, doc = self._verify()
        self.assertEqual(doc["verdict"], "PASS", doc["errors"])
        self.assertTrue(doc["aa"]["runs"][0]["source"].endswith("aa_1.json"))
        self.assertIs(doc["aa"]["through_candidate_binding"]["value"], False)
        self.assertTrue(any("did not go through the candidate binding" in n for n in doc["notes"]))

    def test_declared_binding_files_keep_the_aa_on_the_candidate_binding(self):
        self.t.write("kernel_src/glue.py", "GLUE = 1\n")
        m = self.t.meta()
        m["kernel_extraction"]["binding_files"] = ["kernel_src/glue.py"]
        self.t.write_meta(m)
        code, doc = self._verify()
        self.assertEqual(doc["verdict"], "PASS", doc["errors"])
        self.assertEqual(doc["aa"]["runs"][0]["source"], "run/ut_result.json")
        self.assertIs(doc["aa"]["through_candidate_binding"]["value"], True)
        self.assertIn("1 declared binding file", doc["aa"]["through_candidate_binding"]["why"])

    def test_a_tolerance_sized_from_the_self_noise_passes(self):
        """test3-qwen3-moe-01: an op whose references of 0 move between runs at noise_rel 2 d / (1 + d) = 0.0241 gets
        tol 0.05 from task-authoring.md step 0 (b) (the step holding 2N). Its noop and within_tol cases pass reading
        up to 2 d / (d + tol) = 0.39, above tol, which the static rule of 0b464d2b refused; verify_ut passes it."""
        self.t.write_site("toyop.py", NOISY_TOYOP)
        self.t.write("kernel_src/toyop.py", NOISY_TOYOP)
        m = self.t.meta()
        m["tolerance"] = {"tol": 0.05}
        self.t.write_meta(m)
        code, doc = self._verify()
        self.assertEqual((code, doc["verdict"]), (0, "PASS"), doc["errors"])
        self.assertEqual({k: v["verdict"] for k, v in doc["controls"].items()},
                         {"noop": "PASS", "within_tol": "PASS", "scale_out": "PASS", "stale_output": "PASS"})
        nr = self.t.read_json("run/ut_result.json")["native_noise"]["noise_rel"]
        self.assertAlmostEqual(nr, 2 * NOISY_D / (1 + NOISY_D), places=4)
        self.assertEqual(vu.tol_step(2 * nr), 0.05)
        for name in ("noop", "within_tol"):
            res = self.t.read_json(f"run/verify_ut/control_{name}.json")
            rows = [r for _, rep in vu._rank_reports(res["correctness"]["candidate"])
                    for r in vu.classify_cases(rep, 0.05)]
            self.assertTrue(rows and all(r["kind"] == "pass" for r in rows), name)
            self.assertGreater(max(r["err"] for r in rows), 0.3, name)     # "passing at ... > tol 0.05" before


NOISY_D = 0.0122
NOISY_TOYOP = '''import os

import torch

D = %r


def op(x, w):
    y = torch.tanh(x @ w)
    y[:, :2] = 0.0                     # two output columns whose reference is 0 ...
    rms = float(y.pow(2).mean().sqrt())
    gen = torch.Generator().manual_seed(int.from_bytes(os.urandom(4), "little"))
    sign = torch.randint(0, 2, (y.shape[0], 2), generator=gen) * 2 - 1
    y[:, :2] = sign * D * rms          # ... moved by +-D x RMS from run to run (an atomic order, say)
    return y
''' % NOISY_D


class TestAASource(unittest.TestCase):
    """A16 C': which A/A a task gets, from its kernel_src classes (CPU, no torch)."""

    def setUp(self):
        keep_sys_path(self)
        self.t = ToyTask()
        self.addCleanup(self.t.cleanup)

    def _kind(self):
        self.assertEqual(gu.main(["all", self.t.task]), 0)
        return vu.aa_source(self.t.task, self.t.meta())

    def test_identical_copied_source_runs_through_the_binding(self):
        kind, why = self._kind()
        self.assertEqual(kind, "binding")
        self.assertIn("1 copied source file(s) byte-identical", why)
        self.assertEqual(vu.kernel_src_unmodified(self.t.task)[0], True)

    def test_a_declared_binding_file_does_not_need_an_origin(self):
        self.t.write("kernel_src/bind.cu", "// pybind glue\n")
        m = self.t.meta()
        m["kernel_extraction"]["binding_files"] = ["kernel_src/bind.cu"]
        self.t.write_meta(m)
        self.assertEqual(self._kind()[0], "binding")
        frozen = self.t.read_json("run/ut_contract.json")
        self.assertEqual(list(frozen["binding_files"]), ["kernel_src/bind.cu"])

    def test_a_modified_copy_or_a_moved_origin_is_not_the_baseline_code(self):
        self.assertEqual(self._kind()[0], "binding")
        self.t.write("kernel_src/toyop.py", "def op(x, w):\n    return x @ w\n")
        kind, why = vu.aa_source(self.t.task, self.t.meta())
        self.assertEqual(kind, "modified")
        self.assertIn("differs from its origin", why)
        self.t.write("kernel_src/toyop.py", open(os.path.join(self.t.site, "toyop.py")).read())
        self.assertEqual(vu.aa_source(self.t.task, self.t.meta())[0], "binding")
        self.t.write_site("toyop.py", "def op(x, w):\n    return x\n")
        self.assertIn("changed since freeze", vu.aa_source(self.t.task, self.t.meta())[1])

    def test_an_unclassified_file_is_not_through_the_binding(self):
        self.assertEqual(self._kind()[0], "binding")
        self.t.write("kernel_src/extra.cu", "// written by the agent, undeclared\n")
        kind, why = vu.aa_source(self.t.task, self.t.meta())
        self.assertEqual(kind, "modified")
        self.assertIn("neither copied source", why)


def _timed(by_phase=None, rows=(), weighted=1.0):
    t = {"weighted": weighted, "per_case": list(rows)}
    if by_phase is not None:
        t["by_phase"] = {ph: {"speedup": v, "calls": 1} for ph, v in by_phase.items()}
    return {"exit_code": 0, "gates": {"correctness": "pass"}, "timing": t}


def _case(sig, regime, sp, bms=1.0, host=(0.02, 0.018), lat=(0.03, 0.027)):
    return {"sig": sig, "regime": regime, "baseline_ms": bms, "optimized_ms": bms / sp, "speedup": sp,
            "legs": {"baseline": {"host_ms": host[0], "latency_ms": lat[0]},
                     "current": {"host_ms": host[1], "latency_ms": lat[1]}}}


class TestAAJudgement(unittest.TestCase):
    """A16 C': the weighted A/A AND every phase with call mass must sit in the band."""

    def test_a_phase_out_of_band_fails_the_measurement_and_names_its_worst_case(self):
        rows = [_case("decode_m1", "decode", 1.0), _case("prefill_m1050", "prefill", 1.117),
                _case("prefill_m5065", "prefill", 1.149)]
        ok, corr, probs, detail = vu._aa_ok(_timed({"decode": 1.002, "prefill": 1.125}, rows, 1.004),
                                             0.01, vu.Report("t"), "x")
        self.assertEqual((ok, corr, probs), (False, True, []))
        wc = detail["out_of_band"]["prefill"]["worst_case"]
        self.assertEqual((wc["sig"], wc["baseline"]["host_ms"], wc["candidate"]["latency_ms"]),
                         ("prefill_m5065", 0.02, 0.027))
        self.assertNotIn("decode", detail["out_of_band"])

    def test_every_phase_in_band_passes(self):
        ok, *_ = vu._aa_ok(_timed({"decode": 0.99, "prefill": 1.02}, (), 1.0), 0.01, vu.Report("t"), "x")
        self.assertTrue(ok)

    def test_an_unweighted_result_is_judged_on_its_primary_geomean(self):
        """A19: a task without workload weights has timing.weighted null and its unweighted geomean as primary."""
        res = _timed({"decode": 1.01}, (), None)
        res["timing"]["primary"] = 1.01
        ok, corr, probs, detail = vu._aa_ok(res, 0.01, vu.Report("t"), "x")
        self.assertEqual((ok, probs, detail["weighted"]), (True, [], 1.01))
        res["timing"]["primary"] = 1.05
        self.assertFalse(vu._aa_ok(res, 0.01, vu.Report("t"), "x")[0])
        res["timing"]["primary"] = None
        self.assertIn("no primary", vu._aa_ok(res, 0.01, vu.Report("t"), "x")[2][0])

    def test_an_older_result_without_by_phase_is_recomputed_from_the_call_mass(self):
        meta = {"workload": {"cases": [{"sig": "d", "effective_calls": 1000}, {"sig": "p", "effective_calls": 1}]}}
        rows = [_case("d", "decode", 1.0), _case("p", "prefill", 1.2)]
        self.assertAlmostEqual(vu.phase_values(_timed(None, rows), meta)["prefill"], 1.2)
        ok, *_ = vu._aa_ok(_timed(None, rows, 1.0002), 0.01, vu.Report("t"), "x", meta)
        self.assertFalse(ok)


class TestAAFlow(unittest.TestCase):
    """run_aa: reuse -> out of band -> --aa diagnostic -> measured once more -> FAIL only when both are out."""

    def setUp(self):
        self.calls = []
        for name in ("aa_source", "_reusable_ut", "_measure"):
            self.addCleanup(setattr, vu, name, getattr(vu, name))
        self.rep = vu.Report("/t")
        self.out = _timed({"decode": 1.0, "prefill": 1.125}, [_case("p1", "prefill", 1.125)], 1.004)
        self.inb = _timed({"decode": 1.0, "prefill": 1.004}, [_case("p1", "prefill", 1.004)], 1.0)

    def _wire(self, kind, reused, fresh):
        vu.aa_source = lambda task, meta: (kind, "why")
        vu._reusable_ut = lambda task, rep: reused
        seq = list(fresh)

        def measure(task, rep, work, label, native_dir, through):
            self.calls.append((label, through))
            return f"/t/run/verify_ut/{label}.json", 0, seq.pop(0), False
        vu._measure = measure

    def test_a_reused_ut_result_in_band_passes_without_a_new_run(self):
        self._wire("binding", self.inb, [])
        self.assertEqual(vu.run_aa("/t", {}, self.rep, "/t/run/verify_ut", None)[0], "PASS")
        self.assertEqual((self.calls, self.rep.doc["aa"]["receipt_from"]), ([], "run/ut_result.json"))

    def test_out_of_band_twice_fails_after_the_diagnostic_and_names_the_phase(self):
        self._wire("binding", self.out, [self.inb, self.out])      # diagnostic (--aa) in band, re-measure out
        verdict, _ = vu.run_aa("/t", {}, self.rep, "/t/run/verify_ut", None)
        self.assertEqual(verdict, "FAIL")
        self.assertEqual(self.calls, [("aa_diagnostic", False), ("aa_2", True)])
        diag = self.rep.doc["aa"]["diagnostic"]
        self.assertTrue(diag["in_band"])
        self.assertIn("candidate side", diag["reading"])
        self.assertTrue(any("phase prefill A/A 1.125" in e and "worst case p1" in e for e in self.rep.doc["errors"]))

    def test_out_of_band_once_is_measured_again_and_passes_in_band(self):
        self._wire("binding", self.out, [self.out, self.inb])      # diagnostic also out, re-measure in band
        verdict, res = vu.run_aa("/t", {}, self.rep, "/t/run/verify_ut", None)
        self.assertEqual((verdict, self.rep.doc["aa"]["receipt_from"]), ("PASS", "run/verify_ut/aa_2.json"))
        self.assertIn("harness", self.rep.doc["aa"]["diagnostic"]["reading"])

    def test_a_modified_copy_runs_aa_twice_without_a_diagnostic(self):
        self._wire("modified", self.inb, [self.out, self.out])
        verdict, _ = vu.run_aa("/t", {}, self.rep, "/t/run/verify_ut", None)
        self.assertEqual((verdict, self.calls), ("FAIL", [("aa_1", False), ("aa_2", False)]))
        self.assertNotIn("diagnostic", self.rep.doc["aa"])
        self.assertIs(self.rep.doc["aa"]["through_candidate_binding"]["value"], False)


def _noisy(nr, mre=None, gates=None):
    """An in-band A/A result whose native self-noise is noise_rel `nr` (None: a result written before A37)."""
    res = _timed({"decode": 1.0}, (), 1.0)
    res["native_noise"] = {"max_rel_err": mre, "cases": 6}
    if nr is not None:
        res["native_noise"]["noise_rel"] = nr
    res["gates"] = gates or {"correctness": "pass"}
    return res


def _qwen3_moe_noise():
    """review-17 tol_metric_demo A: main's qwen3_moe self-noise -- an output of RMS 0.383 whose one zero reference
    element moves by 0.0039 (2^-8) between runs."""
    import torch
    g = torch.Generator().manual_seed(0)
    ref = torch.randn(8192, generator=g, dtype=torch.float32)
    ref = ref / ref.pow(2).mean().sqrt() * 0.383
    ref[123] = 0.0
    out = ref.clone()
    out[123] = 0.0039
    return out, ref


class TestSelfNoiseRule(unittest.TestCase):
    """A37: the A/A self-noise rule is noise_rel <= 0.5 x tol, noise_rel on correct()'s own scale."""

    def test_the_rule_the_record_and_the_message(self):
        ok, corr, probs, detail = vu._aa_ok(_noisy(0.0102, 0.5091), 0.02, vu.Report("t"), "x")
        self.assertEqual((ok, corr), (True, False))
        self.assertIn("noise_rel 0.0102 is above 0.5 x tol 0.02", probs[0])
        self.assertIn("2 x noise_rel = 0.0204", probs[0])
        self.assertEqual(detail["self_noise"], {"noise_rel": 0.0102, "max_rel_err": 0.5091, "tol": 0.02,
                                                "twice_noise_rel": 0.0204, "rule": "noise_rel"})
        ok, corr, probs, _ = vu._aa_ok(_noisy(0.0102, 0.3394), 0.03, vu.Report("t"), "x")
        self.assertEqual((ok, corr, probs), (True, True, []))

    def test_a_result_without_noise_rel_keeps_the_max_rel_err_rule(self):
        ok, corr, probs, detail = vu._aa_ok(_noisy(None, 0.1018), 0.1, vu.Report("t"), "x")
        self.assertFalse(corr)
        self.assertIn("native run-to-run error 0.1018", probs[0])
        self.assertEqual(detail["self_noise"]["rule"], "max_rel_err (a result without noise_rel)")
        self.assertTrue(vu._aa_ok(_noisy(None, 0.0679), 0.15, vu.Report("t"), "x")[1])

    def test_a_non_finite_self_noise_fails(self):
        res = _noisy(None, None)
        res["native_noise"].update(noise_rel=None, noise_rel_nonfinite=2)
        _ok, corr, probs, _ = vu._aa_ok(res, 0.02, vu.Report("t"), "x")
        self.assertFalse(corr)
        self.assertIn("2 comparison(s) with a non-finite noise_rel", probs[0])
        _ok, corr, probs, _ = vu._aa_ok(_noisy(None, None) | {"native_noise": {"noise_rel": None, "cases": 0}},
                                        0.02, vu.Report("t"), "x")
        self.assertEqual((corr, probs), (True, []))          # nothing measured, nothing to judge

    def test_the_tolerance_steps(self):
        self.assertEqual([vu.tol_step(x) for x in (0.0204, 0.0306, 0.03, 0.0151, 0.002, 0.45, 7.0)],
                         [0.03, 0.05, 0.03, 0.02, 0.002, 0.5, 10.0])


class TestSelfNoiseInTheAARecord(unittest.TestCase):
    """A37: verify_ut.json's aa section records noise_rel, max_rel_err, tol and 2 x noise_rel per measurement, and
    notes a tolerance looser than the step holding 3 x noise_rel -- without changing the verdict."""

    def setUp(self):
        self.calls = []
        for name in ("aa_source", "_reusable_ut", "_measure"):
            self.addCleanup(setattr, vu, name, getattr(vu, name))

    _wire = TestAAFlow._wire

    def run_with(self, tol, nr=0.0102, mre=0.0509):
        self.rep = vu.Report("/t")
        self._wire("binding", _noisy(nr, mre), [])
        verdict, _ = vu.run_aa("/t", {"tolerance": {"tol": tol}}, self.rep, "/t/run/verify_ut", None)
        return verdict, self.rep.doc

    def test_every_measurement_records_the_self_noise(self):
        verdict, doc = self.run_with(0.2)
        self.assertEqual(verdict, "PASS")
        rec = doc["aa"]["runs"][0]["self_noise"]
        self.assertEqual({k: rec[k] for k in ("noise_rel", "max_rel_err", "tol", "twice_noise_rel")},
                         {"noise_rel": 0.0102, "max_rel_err": 0.0509, "tol": 0.2, "twice_noise_rel": 0.0204})

    def test_a_tolerance_looser_than_the_noise_needs_is_noted_and_still_passes(self):
        verdict, doc = self.run_with(0.2)                    # 3 x 0.0102 -> step 0.05 < 0.2
        self.assertEqual(verdict, "PASS")
        self.assertEqual(len([n for n in doc["notes"] if n.startswith(vu.LOOSE_NOTE)]), 1, doc["notes"])
        for tol in (0.03, 0.05):                             # the step of 2 x noise_rel; the step of 3 x noise_rel
            verdict, doc = self.run_with(tol)
            self.assertEqual(verdict, "PASS")
            self.assertFalse([n for n in doc["notes"] if n.startswith(vu.LOOSE_NOTE)], (tol, doc["notes"]))
        verdict, doc = self.run_with(0.2, nr=0.0)            # bit-exact native: nothing to say about the tolerance
        self.assertFalse([n for n in doc["notes"] if n.startswith(vu.LOOSE_NOTE)])

    def test_too_tight_a_tolerance_fails_with_the_one_it_needs(self):
        verdict, doc = self.run_with(0.02)
        self.assertEqual(verdict, "FAIL")
        self.assertTrue(any("2 x noise_rel = 0.0204" in e for e in doc["errors"]), doc["errors"])
        self.assertEqual(doc["aa"]["runs"][0]["self_noise"]["tol"], 0.02)


@unittest.skipUnless(have_torch(), "torch not installed")
class TestNoiseRelOnCorrectsScale(unittest.TestCase):
    """A37 with the real correct(): the equivalence the rule rests on, the qwen3_moe numbers, and that the new rule
    passes whatever the old one passed."""

    def setUp(self):
        import harness_lib
        import torch
        self.h, self.torch = harness_lib, torch

    def test_correct_passes_exactly_when_noise_rel_is_within_the_tolerance(self):
        torch, h = self.torch, self.h
        g = torch.Generator().manual_seed(7)
        checked = 0
        for i in range(400):
            ref = torch.randn(int(torch.randint(2, 64, (1,), generator=g)), 3, generator=g) * float(
                torch.rand(1, generator=g) * 3 + 1e-3)
            ref[ref.abs() < 0.3] = 0.0                                        # zero references
            out = ref + torch.randn(ref.shape, generator=g) * float(torch.rand(1, generator=g)) * 0.1
            ref[0, 0] = out[0, 0] = float("-inf")                            # the same -inf on both sides
            if i % 2:
                ref, out = (ref, ref[:, :1] * 2), (out, ref[:, :1] * 2 + 0.01)  # a multi-output return
            nr = h.noise_rel(out, ref)
            for t in (0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5):
                self.assertEqual(h.correct(out, ref, t)[0], nr <= t, (i, t, nr))
                checked += 1
        self.assertEqual(checked, 2800)
        x = torch.randn(4, 4, generator=g)
        self.assertEqual(h.noise_rel(x, x.clone()), 0.0)                     # bit-identical: 0

    def test_the_qwen3_moe_noise_needs_0_03_and_the_old_rule_0_15(self):
        out, ref = _qwen3_moe_noise()
        nr = self.h.noise_rel(out, ref)
        self.assertAlmostEqual(nr, 0.01018, places=5)
        for tol, passes in ((0.02, False), (0.03, True)):
            _ok, corr, _p, _d = vu._aa_ok(_noisy(nr, self.h.correct(out, ref, tol)[1]), tol, vu.Report("t"), "x")
            self.assertEqual(corr, passes, tol)
        for tol, passes in ((0.1, False), (0.15, True)):                     # a result without noise_rel
            _ok, corr, _p, _d = vu._aa_ok(_noisy(None, self.h.correct(out, ref, tol)[1]), tol, vu.Report("t"), "x")
            self.assertEqual(corr, passes, tol)

    def test_the_new_rule_passes_whatever_the_old_rule_passed(self):
        torch, h = self.torch, self.h
        old_passed = 0
        for i in range(2000):
            g = torch.Generator().manual_seed(1000 + i)
            r = torch.randn(256, generator=g) * float(torch.rand(1, generator=g) * 3 + 0.01)
            k = int(torch.randint(0, 256, (1,), generator=g))
            r[:k // 8] *= 1e-3                                                 # near-zero references
            o = r + torch.randn(256, generator=g) * float(torch.rand(1, generator=g)) * 0.02 * float(
                r.pow(2).mean().sqrt())
            tol = (0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)[i % 9]
            if h.correct(o, r, tol)[1] <= vu.NOISE_FRACTION * tol:
                old_passed += 1
                self.assertLessEqual(h.noise_rel(o, r), vu.NOISE_FRACTION * tol * (1 + 1e-6), (i, tol))
        self.assertGreater(old_passed, 100)


def _correct_row(h, case, out, ref, tol):
    """A row as harness_lib's checks write it: correct()'s verdict, its err rounded to 5 decimals (None when not
    finite) and the fields a per-output err carries (A42)."""
    ok, err = h.correct(out, ref, tol)
    return dict({"case": case, "correct": ok, "max_rel_err": round(err, 5) if math.isfinite(err) else None},
                **h._row_extra(err))


def _control(rows):
    """A single-rank control result whose candidate report holds `rows` as random draws."""
    return _res(0 if all(r["correct"] for r in rows) else 1, _rep(random=rows))


class TestStaticRecordRule(unittest.TestCase):
    """ISSUE-060's static rule on correct()'s own scale (test3-qwen3-moe-01 section 3): a failing case reads above
    failing_floor = tol / max(1, tol, the largest tolerance of the task); only an error rounded onto it is lifted."""

    def test_a_failing_case_is_held_above_tol_and_only_one_rounded_onto_it_is_lifted(self):
        rows = vu.classify_cases(_rep(eager=[_c("a", False, 0.02), _c("b", False, 0.01999), _c("c", False, 0.05),
                                             _c("d", True, 0.9), _c("e", True, 0.02)]), TOL)
        self.assertEqual(vu.record_errors(rows, TOL), ["a"])
        self.assertEqual([r["err_recorded"] for r in rows], [0.02001, 0.01999, 0.05, 0.9, 0.02])
        self.assertEqual(vu.static_record_problems(rows, TOL), ["b: failing at 0.01999 <= tol 0.02"])
        self.assertEqual(vu.static_record_problems([{"case": "w", "kind": "fail", "err_recorded": 0.02}], TOL),
                         ["w: failing at 0.02 <= tol 0.02"])                  # at the floor is not above it
        rows = vu.classify_cases(_rep(eager=[_c("u", False, 0.0)]), 0.0)      # tol 0: a difference under 5e-6
        self.assertEqual((vu.record_errors(rows, 0.0), rows[0]["err_recorded"]), (["u"], 1e-05))
        self.assertEqual(vu.static_record_problems(rows, 0.0), [])
        rows = vu.classify_cases(_rep(eager=[_c("v", False, 0.0)]), 1e-6)     # a tolerance finer than 5 decimals
        self.assertEqual((vu.record_errors(rows, 1e-6), rows[0]["err_recorded"]), (["v"], 1e-05))

    def test_above_1_a_failing_case_is_held_above_1(self):
        rows = vu.classify_cases(_rep(eager=[_c("x", False, 1.5), _c("y", False, 1.0), _c("z", False, 0.5)]), 2.0)
        self.assertEqual(vu.record_errors(rows, 2.0), ["y"])
        self.assertEqual([r["err_recorded"] for r in rows], [1.5, 1.00001, 0.5])
        self.assertEqual(vu.static_record_problems(rows, 2.0), ["z: failing at 0.5 <= 1 (tol 2.0)"])
        self.assertEqual([vu.failing_floor(t) for t in (0.0, 0.02, 1.0, 2.0)], [0.0, 0.02, 1.0, 1.0])
        self.assertEqual((vu.failing_floor(0.5, 4.0), vu.failing_floor(0.02, 0.05)), (0.125, 0.02))

    def test_the_largest_tolerance_of_a_task(self):
        self.assertEqual(vu.largest_tolerance({"tolerance": {"tol": 0.02}}, 0.02), 0.02)
        self.assertEqual(vu.largest_tolerance({"tolerance": {"tol": 0.02, "per_output": {"1": 0.001}}}, 0.02), 0.02)
        self.assertEqual(vu.largest_tolerance({"tolerance": {"tol": 0.02, "per_output": {"0": 0.5, "1": 4}}}, 0.02),
                         4.0)
        self.assertEqual(vu.largest_tolerance({"tol": 0.3}, 0.3), 0.3)

    def test_the_controls_are_judged_with_the_largest_tolerance_of_the_task(self):
        """run_controls passes largest_tolerance: a failing row of output 0 (0.5) that reads 0.2 -- the row of a
        comparison whose output 1 (4.0) failed, output 0 being further over its own tolerance -- is a catch."""
        fail = {"case": "random[0]:p", "correct": False, "max_rel_err": 0.2, "output": "0", "tol": 0.5}
        ok = {"case": "random[0]:p", "correct": True, "max_rel_err": 0.0, "output": "0", "tol": 0.5}
        self.addCleanup(setattr, vu, "run_with_retries", vu.run_with_retries)
        vu.run_with_retries = lambda rep, label, fn: ((0, _control([ok]), False)
                                                      if label in ("control:noop", "control:within_tol")
                                                      else (1, _control([fail]), False))
        rep = vu.Report("/t")
        meta = {"tolerance": {"tol": 0.02, "per_output": {"0": 0.5, "1": 4.0}}, "capabilities": {}}
        got = vu.run_controls("/t", meta, rep, "/t/run/verify_ut", 1, "/t/native")
        self.assertEqual({k: v["verdict"] for k, v in got.items()},
                         {"noop": "PASS", "within_tol": "PASS", "scale_out": "PASS", "stale_output": "PASS"},
                         rep.doc["errors"])
        v, d = vu.judge_control("scale_out", {}, 1, _control([fail]), 0.02, 1)       # held to 0.5 itself
        self.assertEqual(v, "FAIL")
        self.assertIn("failing at 0.2 <= tol 0.5", d["why"])
        at_floor = dict(fail, max_rel_err=0.125)                    # 0.5 / 4, where only the rounding can put it
        v, d = vu.judge_control("scale_out", {}, 1, _control([at_floor]), 0.02, 1, 4.0)
        self.assertEqual((v, d["rounded_up"]), ("PASS", ["random[0]:p"]), d.get("why"))

    def test_a_passing_case_of_a_rejected_control_may_read_above_tol(self):
        """ut_neg_glm47_mono, round 2, run/attempt1 (native noise 0): rank_skew failed 29 cases on each of the 4 ranks
        and passed eager_moe_L61856_blockend on every rank reading 0.64758 at tol 0.02 -- a change within the band
        (noise_rel at most 0.02), which 0b464d2b's rule refused."""
        judged = {"ok": False, "ranks": [{"rank": k, "report": _rep(
            eager=[_c("eager_moe_L61856_blockend", True, 0.64758)],
            replay=[_c("replay_carry_step0_L61856", False, 26.9694)])} for k in range(4)]}
        v, d = vu.judge_control("rank_skew", {}, 1, {"exit_code": 1, "correctness": {"candidate": judged}}, TOL, 4)
        self.assertEqual(v, "PASS", d.get("why"))


@unittest.skipUnless(have_torch(), "torch not installed")
class TestStaticRecordRuleOnCorrectsScale(unittest.TestCase):
    """The static rule with the real correct() (test3-qwen3-moe-01 section 3). correct() fails an element when
    |d| > t (|ref| + R) and reads it as |d| / (|ref| + t R): a failing comparison reads above min(t, 1), a passing one
    up to max(t, 1) -- on a reference element of 0 it reads |d| / (t R), above t once |d| > t^2 R. So the rule holds
    a failing case above failing_floor and does not hold a passing case to tol."""

    def setUp(self):
        import harness_lib
        import torch
        self.h, self.torch = harness_lib, torch

    def test_the_rule_never_refuses_what_correct_writes(self):
        torch, h = self.torch, self.h
        g = torch.Generator().manual_seed(60)
        tols = (0.0, 1e-6, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 5.0)
        seen = {"fail": 0, "pass": 0, "pass_above_tol": 0}
        for i in range(1300):
            n = int(torch.randint(2, 96, (1,), generator=g))
            ref = torch.randn(n, generator=g) * float(torch.rand(1, generator=g) * 3 + 1e-3)
            ref[torch.rand(n, generator=g) < 0.2] = 0.0                          # references of 0
            scale = float(torch.rand(1, generator=g)) ** 3 * 4                    # tiny to large differences
            out = ref + torch.randn(n, generator=g) * scale * float(ref.pow(2).mean().sqrt())
            if i % 3 == 0:
                ref, out = (ref, ref[:3] * 2), (out, ref[:3] * 2 * (1 + scale / 10))   # a multi-output return
            tol = tols[i % len(tols)]
            row = _correct_row(h, f"c{i}", out, ref, tol)
            rows = vu.classify_cases(_rep(eager=[row]), tol)
            vu.record_errors(rows, tol)
            self.assertEqual(vu.static_record_problems(rows, tol), [], (i, tol, row))
            e = h.correct(out, ref, tol)[1]
            if not math.isfinite(e):
                continue
            if row["correct"]:
                seen["pass"] += 1
                seen["pass_above_tol"] += e > tol
                self.assertLessEqual(e, max(tol, 1.0) * (1 + 1e-6), (i, tol))
            else:
                seen["fail"] += 1
                self.assertGreater(e, vu.failing_floor(tol) * (1 - 1e-6), (i, tol))
        self.assertGreater(min(seen["fail"], seen["pass"]), 300, seen)
        self.assertGreater(seen["pass_above_tol"], 50, seen)       # 0b464d2b's rule refused each ("passing at")

    def test_the_qwen3_moe_tolerances_2n_and_one_step_more_pass(self):
        """test3-qwen3-moe-01: native noise_rel up to N = 0.0244, so tol 0.05 (the step holding 2N) or 0.1 (one step
        more). Its noop and within_tol cases pass reading far above tol on the references of 0 (0.07-0.27 on the GPU)
        and 0b464d2b's rule refused them; the controls and the A/A now pass at both."""
        torch, h = self.torch, self.h
        g = torch.Generator().manual_seed(3)
        refs, noisy = [], []
        for m in (1, 2, 3, 4, 4):
            ref = torch.randn(m, 2048, generator=g)
            ref = ref / ref.pow(2).mean().sqrt() * 0.383
            ref[:, ::7] = 0.0                                                     # routed nowhere: exactly 0
            rms = float(ref.pow(2).mean().sqrt())
            move = (torch.randint(0, 2, ref.shape, generator=g) * 2 - 1) * 0.0244 * rms
            refs.append(ref)
            noisy.append(torch.where(ref == 0, move, ref * (1 + 1e-3 * torch.randn(ref.shape, generator=g))))
        nr = max(h.noise_rel(o, r) for o, r in zip(noisy, refs))
        self.assertAlmostEqual(nr, 0.0244, places=4)
        self.assertEqual((vu.tol_step(2 * nr), vu.tol_step(2 * vu.tol_step(2 * nr))), (0.05, 0.1))
        for tol in (0.05, 0.1):
            ctl = {"noop": noisy, "within_tol": [o * (1 + tol / 10) for o in noisy],
                   "scale_out": [o * (1 + 4 * tol) for o in noisy],
                   "stale_output": noisy[:4] + noisy[3:4]}                # the first output of each shape
            for name, outs in ctl.items():
                rows = [_correct_row(h, f"random[{i}]:m{r.shape[0]}", o, r, tol)
                        for i, (o, r) in enumerate(zip(outs, refs))]
                if name in ("noop", "within_tol"):
                    self.assertTrue(all(r["correct"] for r in rows), (tol, name))
                    self.assertGreater(max(r["max_rel_err"] for r in rows), 2 * tol, (tol, name))   # N / tol
                code = 0 if all(r["correct"] for r in rows) else 1
                v, d = vu.judge_control(name, {}, code, _res(code, _rep(random=rows)), tol, 1)
                self.assertEqual(v, "PASS", (tol, name, d.get("why")))
            _ok, corr, probs, _d = vu._aa_ok(_noisy(nr, 0.5), tol, vu.Report("t"), "x")
            self.assertEqual((corr, probs), (True, []), tol)

    def test_a_per_output_row_is_held_above_its_tolerance_over_the_largest(self):
        """A42, PF-03: the row of a failing comparison records a failing output, the one furthest over its own
        tolerance. When output 1 (4.0) fails reading 1.2 and output 0 (0.5) passes reading 0.2 -- further over its own
        tolerance, 0.4 of it against 0.3 -- the row is output 1's (28cb6431 recorded output 0's 0.2), failing at 1.2:
        above 4.0 / 4, where failing_floor holds it."""
        import leg_runner
        torch, h = self.torch, self.h
        ref = torch.tensor([0.0, 1.0, -1.0, 1.0])
        rms = float(ref.pow(2).mean().sqrt())
        o0, o1 = ref.clone(), ref.clone()
        o0[0], o1[0] = 0.1 * rms, 4.8 * rms
        per = {"0": 0.5, "1": 4.0}
        self.assertEqual([(ok, round(e, 5)) for ok, e in (h.correct(o0, ref, 0.5), h.correct(o1, ref, 4.0))],
                         [(True, 0.2), (False, 1.2)])
        compare = leg_runner.per_output_correct(h, h.correct)
        ok, err = compare((o0, o1), (ref, ref.clone()), leg_runner.TaskTol(0.02, per, h=h))
        row = dict(_c("p", ok, round(err, 5)), **h._row_extra(err))
        self.assertEqual((row["correct"], row["max_rel_err"], row["output"], row["tol"]), (False, 1.2, "1", 4.0))
        tol_max = vu.largest_tolerance({"tolerance": {"tol": 0.02, "per_output": per}}, 0.02)
        self.assertEqual(vu.failing_floor(4.0, tol_max), 1.0)
        rows = vu.classify_cases(_rep(eager=[row]), 0.02)
        self.assertEqual(vu.record_errors(rows, 0.02, tol_max), [])
        self.assertEqual(vu.static_record_problems(rows, 0.02, tol_max), [])

    def test_a_failing_row_names_a_failing_output_not_a_passing_one_further_over_its_tolerance(self):
        """PF-03, review-33 section 2.1 C (no tolerance 0): on a reference element near 0 a passing output reads up to
        1, up to 1 / t times its own tolerance -- post, moved by 0.8 t RMS there, reads 0.8 at 0.005, 160 times over --
        while out, 3% off at 0.02, fails about 1.5 times over. The row is out's (28cb6431 named post, as if post had
        failed); of several failing outputs it is the one furthest over its own tolerance, first or last in order."""
        import leg_runner
        torch, h = self.torch, self.h
        g = torch.Generator().manual_seed(0)
        ref = {k: torch.randn(64, generator=g) for k in ("comb", "out", "post", "pre")}
        ref["post"][:8] = 0.0
        out = {k: v.clone() for k, v in ref.items()}
        out["post"][0] = 0.8 * 0.005 * float(ref["post"].pow(2).mean().sqrt())     # |d| = 0.8 t RMS on a reference of 0
        out["out"] = ref["out"] * 1.03
        out["comb"], out["pre"] = ref["comb"] * 1.02, ref["pre"] * 1.02           # 2% off at 0.005: about 4 times over
        tols = {"comb": 0.005, "out": 0.02, "post": 0.005, "pre": 0.005}
        alone = {k: h.correct(out[k], ref[k], t) for k, t in tols.items()}
        self.assertEqual({k: ok for k, (ok, _) in alone.items()},
                         {"comb": False, "out": False, "post": True, "pre": False})
        share = {k: e / tols[k] for k, (_, e) in alone.items()}
        self.assertGreater(share["post"], 100)
        self.assertLess(share["out"], min(share["comb"], share["pre"]))
        compare = leg_runner.per_output_correct(h, h.correct)
        rows = []
        for names, named in ((("out", "post"), "out"), (("out", "post", "pre"), "pre"),
                             (("comb", "out", "post"), "comb")):
            per = {k: tols[k] for k in names if k != "out"}                       # out at the task's 0.02
            ok, err = compare({k: out[k] for k in names}, {k: ref[k] for k in names},
                              leg_runner.TaskTol(0.02, per, h=h))
            row = dict(_c("+".join(names), ok, round(err, 5)), **h._row_extra(err))
            self.assertEqual((row["correct"], row["max_rel_err"], row["output"], row["tol"]),
                             (False, round(alone[named][1], 5), named, tols[named]), names)
            rows.append(row)
        tol_max = vu.largest_tolerance({"tolerance": {"tol": 0.02, "per_output": tols}}, 0.02)
        rows = vu.classify_cases(_rep(eager=rows), 0.02)
        self.assertEqual(vu.record_errors(rows, 0.02, tol_max), [])
        self.assertEqual(vu.static_record_problems(rows, 0.02, tol_max), [])

    def test_a_bit_identical_native_keeps_its_records_and_verdicts(self):
        """Native noise 0 (a kernel that reproduces bit for bit): noop reads 0 and within_tol about tol / 10 at most,
        so no passing case of either reads above tol, and the controls get the records and verdicts 0b464d2b's rule
        gave them (_rule_of_0b464d2b: a passing case held to tol as well, every failing case at or below tol lifted)."""
        torch, h = self.torch, self.h
        for dtype, tols in ((torch.float32, (1e-6, 0.001, 0.01, 0.02, 0.05)), (torch.bfloat16, (0.01, 0.02, 0.05))):
            g = torch.Generator().manual_seed(5)
            refs = [torch.randn(m, 64, generator=g).to(dtype) for m in (1, 4, 7, 7)]
            for r in refs:
                r[:, ::9] = 0
            for tol in tols:
                ctl = {"noop": [r.clone() for r in refs], "within_tol": [r * (1.0 + tol / 10.0) for r in refs],
                       "scale_out": [r * (1.0 + 4.0 * tol) for r in refs],
                       "stale_output": refs[:3] + refs[2:3]}                   # the first output of each shape
                for name, outs in ctl.items():
                    rows = [_correct_row(h, f"random[{i}]:m", o, r, tol) for i, (o, r) in enumerate(zip(outs, refs))]
                    if name in ("noop", "within_tol"):
                        self.assertTrue(all(r["correct"] and r["max_rel_err"] <= tol for r in rows), (dtype, tol))
                    new = vu.classify_cases(_rep(random=rows), tol)
                    lifted = vu.record_errors(new, tol)
                    old_lifted, old_records, old_bad = _rule_of_0b464d2b(vu.classify_cases(_rep(random=rows), tol), tol)
                    got = (lifted, [r.get("err_recorded") for r in new], vu.static_record_problems(new, tol))
                    self.assertEqual(got, (old_lifted, old_records, old_bad), (dtype, tol, name))
                    code = 0 if all(r["correct"] for r in rows) else 1
                    v, d = vu.judge_control(name, {}, code, _res(code, _rep(random=rows)), tol, 1)
                    self.assertEqual(v, "PASS", (dtype, tol, name, d.get("why")))


def _rule_of_0b464d2b(rows, tol):
    """The record and static rule of 0b464d2b, for comparison: (lifted cases, recorded errors, refusals). Every failing
    case at or below tol was lifted above it, and a passing case was held not to read above tol."""
    lifted, records, bad = [], [], []
    for r in rows:
        e = r.get("err")
        rec = None
        if r["kind"] == "fail" and _finite(e) and e <= tol:
            rec = vu._above_tol_5dp(tol)
            lifted.append(r["case"])
        elif _finite(e):
            rec = round(e, 5)
        records.append(rec)
        if rec is not None and r["kind"] == "fail" and not rec > tol:
            bad.append(f"{r['case']}: failing at {rec} <= tol {tol}")
        if rec is not None and r["kind"] == "pass" and rec > tol + 1e-12:
            bad.append(f"{r['case']}: passing at {rec} > tol {tol}")
    return lifted, records, bad


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


class TestPerOutputTolerances(unittest.TestCase):
    """A42: the self-noise rule holds each output to half its own tolerance; a row compared at its own output's
    tolerance is recorded and judged against that tolerance."""

    def per_output(self, **outputs):
        return {"native_noise": {"max_rel_err": 0.0, "cases": 6,
                                 "noise_rel": max(v["noise_rel"] for v in outputs.values()),
                                 "per_output": outputs}}

    def test_each_output_is_held_to_half_its_own_tolerance(self):
        res = self.per_output(**{"0": {"tol": 0.05, "noise_rel": 0.02}, "1": {"tol": 0.001, "noise_rel": 0.0004}})
        rec, problem, loose = vu.self_noise(res, 0.02)
        self.assertIsNone(problem)                      # 0.02 > 0.5 x tol 0.02 would fail; its own 0.05 holds it
        self.assertFalse(loose)
        self.assertEqual(rec["rule"], "noise_rel per output")
        self.assertEqual(rec["per_output"]["1"], {"noise_rel": 0.0004, "tol": 0.001, "twice_noise_rel": 0.0008})
        res = self.per_output(**{"0": {"tol": 0.05, "noise_rel": 0.0}, "1": {"tol": 0.001, "noise_rel": 0.0006}})
        rec, problem, loose = vu.self_noise(res, 0.02)
        self.assertIn("output 1: native self-noise noise_rel 0.0006 is above 0.5 x its tolerance 0.001", problem)
        self.assertIn("at least 2 x noise_rel = 0.0012", problem)
        self.assertNotIn("output 0", problem)

    def test_a_loose_output_is_noted_and_a_non_finite_one_fails(self):
        res = self.per_output(**{"0": {"tol": 0.05, "noise_rel": 0.001}, "1": {"tol": 0.001, "noise_rel": 0.0003}})
        rec, problem, loose = vu.self_noise(res, 0.02)
        self.assertIsNone(problem)
        self.assertTrue(loose)
        self.assertEqual(rec["per_output"]["0"]["step_for_3x_noise_rel"], 0.003)
        self.assertNotIn("step_for_3x_noise_rel", rec["per_output"]["1"])
        rep = vu.Report("/nowhere")
        res.update(timing={"primary": 1.0}, gates={"correctness": "pass"})
        in_band, corr, probs, detail = vu._aa_ok(res, 0.02, rep, "run/ut_result.json")
        self.assertTrue(in_band and corr, probs)
        self.assertEqual(rep.doc["notes"], [f"{vu.LOOSE_NOTE}: output 0 tol 0.05, native noise_rel 0.001, the step "
                                            "holding 3 x noise_rel 0.003 (run/ut_result.json; task-authoring.md step "
                                            "0 (b))"])
        res = self.per_output(**{"0": {"tol": 0.05, "noise_rel": 0.0}})
        res["native_noise"]["per_output"]["1"] = {"tol": 0.001, "noise_rel": None, "noise_rel_nonfinite": 2}
        _, problem, _ = vu.self_noise(res, 0.02)
        self.assertIn("output 1: native self-noise has 2 comparison(s) with a non-finite noise_rel", problem)

    def test_a_row_is_recorded_and_judged_at_its_own_outputs_tolerance(self):
        rep = _rep(eager=[dict(_c("a", False, 0.0015), output="1", tol=0.001),
                          dict(_c("b", False, 0.001), output="1", tol=0.001),
                          dict(_c("c", True, 0.015), output="0", tol=0.05)])
        rows = vu.classify_cases(rep, TOL)
        self.assertEqual([(r["output"], r["tol"]) for r in rows], [("1", 0.001), ("1", 0.001), ("0", 0.05)])
        self.assertEqual(vu.record_errors(rows, TOL), ["b"])            # failing at exactly its tolerance: up
        self.assertEqual([r["err_recorded"] for r in rows], [0.0015, 0.00101, 0.015])
        self.assertEqual(vu.static_record_problems(rows, TOL), [])
        # judged at the task's tol instead, the failing f32 rows would read "failing at ... <= tol 0.02"
        plain = vu.classify_cases(_rep(eager=[_c("a", False, 0.0015)]), TOL)
        self.assertEqual(vu.record_errors(plain, TOL), [])
        self.assertEqual(plain[0]["err_recorded"], 0.0015)
        self.assertEqual(vu.static_record_problems(plain, TOL), ["a: failing at 0.0015 <= tol 0.02"])
        # scale_out seen at the f32 output's own tolerance: a catch, no ISSUE-033 warning for a task control either
        res = _res(1, _rep(eager=[dict(_c("a", False, 0.004), output="1", tol=0.001)]))
        self.assertEqual(vu.judge_control("scale_out", {}, 1, res, TOL, 1)[0], "PASS")
        v, d = vu.judge_control("mine", {"expect": "reject"}, 1, res, TOL, 1)
        self.assertEqual(v, "PASS")
        self.assertNotIn("warning", d)


class TestReceiptCheck(unittest.TestCase):
    SPIN = {"queued": "gpu_spin_before_start", "us": 200.0, "late": 0, "samples": 150}

    def _res(self, b, c, deployed_graph=False):
        return {"receipt": {"cache_mode": "read-evict", "world_size": 1, "timer": b.get("timer"),
                            "flags": [{"sig": "p", "kind": "dispatch_slower_than_kernel"}],
                            "cases": {"p": {"baseline": b, "current": c, "deployed_graph": deployed_graph}}}}

    def test_an_eager_case_needs_the_queued_spin_on_both_legs(self):
        rep = vu.Report("t")
        ok = vu.check_receipt({}, self._res({"timer": "cuda_event", "spin": self.SPIN}, {"timer": "cuda_event"}), rep)
        self.assertFalse(ok)
        self.assertTrue(any("current eager timing lacks the queued GPU spin" in e for e in rep.doc["errors"]))

    def test_spun_legs_pass_and_the_flags_are_kept(self):
        rep = vu.Report("t")
        res = self._res({"timer": "cuda_event", "spin": self.SPIN}, {"timer": "cuda_event", "spin": self.SPIN})
        self.assertTrue(vu.check_receipt({}, res, rep))
        self.assertEqual(rep.doc["receipt"]["flags"][0]["kind"], "dispatch_slower_than_kernel")

    def test_a_graph_case_and_cpu_timing_need_no_spin(self):
        for b in ({"timer": "cuda_event_graph"}, {"timer": "wall"}):
            rep = vu.Report("t")
            self.assertTrue(vu.check_receipt({}, self._res(b, dict(b), deployed_graph=b["timer"] != "wall"), rep))


# --------------------------------------------------------------------------- the GPU part, on the CPU (no torch)
SPIN_REC = {"queued": "gpu_spin_before_start", "us": 200.0, "late": 0, "samples": 150}
TOY_SIGS = ("decode_m1", "decode_m4")
GENERIC_PASS = {"noop": "PASS", "within_tol": "PASS", "scale_out": "PASS", "stale_output": "PASS"}


class FakeUTRunner:
    """Stands in for verify_ut's `subprocess` module: every `python3 <task>/ut_runner.py ...` verify_ut starts is
    recorded and answered from `script`, {what: [(exit code, result), ...]} -- `what` is the result file's stem (aa_1,
    aa_diagnostic, native_refs) or control:<name>, one answer per attempt, the last one repeating. The result is written
    where ut_runner.py writes it (--out) with its exit_code; None writes nothing (a run that died before writing it) and
    an exception is raised (a timeout). native_stamp: written into --native-dir/native.json by a run that measures
    native references (no --reuse-native), as ut_runner.py stamps them."""

    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self, script, native_stamp=None):
        self.script = {k: list(v) for k, v in script.items()}
        self.native_stamp = native_stamp
        self.calls = []

    def run(self, cmd, cwd=None, env=None, capture_output=False, text=False, timeout=None):
        args = cmd[2:]
        out = args[args.index("--out") + 1]
        what = (f"control:{args[args.index('--control') + 1]}" if "--control" in args
                else os.path.splitext(os.path.basename(out))[0])
        self.calls.append({"what": what, "cmd": cmd, "args": args, "cwd": cwd, "timeout": timeout,
                           "capture": (capture_output, text), "parallel": (env or {}).get("GEAK_UT_POOL_PARALLEL")})
        answers = self.script[what]
        code, res = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(res, BaseException):
            raise res
        if self.native_stamp is not None and "--native-dir" in args and "--reuse-native" not in args:
            native = args[args.index("--native-dir") + 1]
            os.makedirs(native, exist_ok=True)
            with open(os.path.join(native, "native.json"), "w", encoding="utf-8") as fh:
                json.dump(self.native_stamp, fh)
        if res is not None:
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(dict(res, exit_code=code), fh)
        return subprocess.CompletedProcess(cmd, code, stdout=f"UT_RESULT: exit {code}\n",
                                           stderr="[ut] the stderr tail\n")


def _receipt(cache_mode="read-evict"):
    """A single-rank eager receipt as ut_runner.build_receipt writes it: both legs timed on the queued GPU spin."""
    leg = {"ms": 1.0, "timer": "cuda_event", "primed": True, "host_ms": 0.02, "spin": SPIN_REC, "latency_ms": 0.03}
    return {"version": 1, "all_primed": True, "timer_unprimed": False, "cache_mode": cache_mode, "world_size": 1,
            "timer": "cuda_event", "alignment": None, "flags": [],
            "cases": {s: {"baseline": dict(leg), "current": dict(leg), "cache_mode": cache_mode, "deployed_graph": False}
                      for s in TOY_SIGS}}


def _toy_ut_result(task, receipt=None, primary=1.004):
    """A passing UT result of the toy task inside the A/A band, bound to the task's current contract and snapshot."""
    res = _timed({"decode": primary}, [_case("decode_m1", "decode", 1.002), _case("decode_m4", "decode", primary)],
                 primary)
    res["timing"]["primary"] = primary
    res.update(contract_sha256=vu.ut_runner.check_contract(task)[0],
               snapshot_sha256=vu.ut_runner.source_snapshot(task)["snapshot_sha256"],
               native_noise={"noise_rel": 0.0, "max_rel_err": 0.0, "cases": 4}, receipt=receipt or _receipt())
    return res


def _toy_aa(task, m1, m4):
    """An A/A result of the toy task as ut_runner.py writes one, bound to its contract and snapshot: decode_m1 and
    decode_m4 timed at speedups m1 and m4 (baseline 1 ms each); the decode phase -- the toy's only one -- and the
    weighted A/A are their call-weighted speedup (meta.workload.cases: 10 and 30 effective calls)."""
    res = _toy_ut_result(task)
    sp = (10 + 30) / (10 / m1 + 30 / m4)
    res["timing"].update(weighted=sp, primary=sp, by_phase={"decode": {"speedup": sp, "calls": 40}},
                         per_case=[_case("decode_m1", "decode", m1), _case("decode_m4", "decode", m4)])
    return res


def _toy_controls(over=None):
    """FakeUTRunner answers for the generic controls of the toy task (tol 0.01) as a sound UT gives them: noop and
    within_tol pass, scale_out is rejected on its eager cases, stale_output on a random draw. over: {name: answers}."""
    reports = {"noop": (0, _rep(eager=[_c("decode_m1", True, 0.0), _c("decode_m4", True, 0.0)],
                                random=[_c("random[0]:decode_m1", True, 0.0)])),
               "within_tol": (0, _rep(eager=[_c("decode_m1", True, 0.001), _c("decode_m4", True, 0.00098)])),
               "scale_out": (1, _rep(eager=[_c("decode_m1", False, 0.04), _c("decode_m4", False, 0.04)])),
               "stale_output": (1, _rep(random=[_c("random[0]:decode_m1", True, 0.0),
                                                _c("random[1]:decode_m4", False, 0.9)]))}
    script = {f"control:{name}": [(code, _res(code, report))] for name, (code, report) in reports.items()}
    script.update({f"control:{name}": answers for name, answers in (over or {}).items()})
    return script


class TestVerifyRun(unittest.TestCase):
    """verify_ut.main end to end on a generated toy task (CPU, no torch) with the ut_runner.py processes it starts
    answered by FakeUTRunner: the A/A, the timing receipt, the native references, every control and the verdict."""

    def setUp(self):
        keep_sys_path(self)
        self.t = ToyTask()
        self.addCleanup(self.t.cleanup)
        self.task = os.path.realpath(self.t.task)
        self.work = os.path.join(self.task, "run", "verify_ut")
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("GEAK_UT_POOL_PARALLEL", None)

    def generate(self, ut_result=True, ut_native=True, receipt=None):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gu.main(["all", self.task]), 0)
        self.contract = vu.ut_runner.check_contract(self.task)[0]
        if ut_result:                     # the UT's own run of this contract and snapshot (A08.2)
            self.t.write("run/ut_result.json", json.dumps(_toy_ut_result(self.task, receipt)))
        if ut_native:                     # ... and its native references
            self.t.write("run/ut_native/native.json", json.dumps({"contract_sha256": self.contract, "world": 1}))

    def verify(self, script, native_stamp=None):
        fake = FakeUTRunner(script, native_stamp)
        buf = io.StringIO()
        with mock.patch.object(vu, "subprocess", fake), contextlib.redirect_stdout(buf):
            code = vu.main(["--task", self.task])
        doc = self.t.read_json("run/verify_ut.json")
        self.assertTrue(buf.getvalue().startswith(f"FREEZE_VERIFY: {doc['verdict']}\n"), buf.getvalue()[:300])
        return code, doc, fake

    @staticmethod
    def verdicts(doc):
        return {k: v["verdict"] for k, v in doc["controls"].items()}

    @staticmethod
    def native_dirs(fake):
        return {c["args"][c["args"].index("--native-dir") + 1] for c in fake.calls}

    def test_a_sound_task_passes_on_its_own_ut_result_and_native_references(self):
        self.generate()
        self.t.write("run/verify_ut/stale.json", "{}")                 # a previous verification's leftovers
        code, doc, fake = self.verify(_toy_controls())
        self.assertEqual((code, doc["verdict"]), (0, "PASS"), doc["errors"])
        self.assertEqual((doc["aa"]["runs"][0]["source"], doc["aa"]["receipt_from"]), ("run/ut_result.json",) * 2)
        self.assertTrue(doc["receipt"]["ok"])
        self.assertEqual(self.verdicts(doc), GENERIC_PASS)
        self.assertEqual(doc["controls"]["scale_out"]["result"], "run/verify_ut/control_scale_out.json")
        ut_native = os.path.join(self.task, "run", "ut_native")
        self.assertEqual([c["args"][4:] for c in fake.calls],
                         [["--native-dir", ut_native, "--reuse-native", "--control", n] for n in GENERIC_PASS])
        self.assertEqual({(c["cwd"], c["parallel"], c["capture"]) for c in fake.calls}, {(self.task, None, (True, True))})
        self.assertFalse(os.path.exists(os.path.join(self.work, "stale.json")))
        with open(os.path.join(self.work, "control_scale_out.log"), encoding="utf-8") as fh:
            log = fh.read()
        self.assertTrue(log.startswith(f"$ {sys.executable} {os.path.join(self.task, 'ut_runner.py')} --task "
                                       f"{self.task} --out {os.path.join(self.work, 'control_scale_out.json')} "),
                        log[:400])
        self.assertTrue(log.endswith("\n[exit 1]\nUT_RESULT: exit 1\n\n--- stderr ---\n[ut] the stderr tail\n"), log)
        self.assertEqual((doc["contract_sha256"], doc["snapshot_sha256"]),
                         (self.contract, vu.ut_runner.source_snapshot(self.task)["snapshot_sha256"]))

    def test_a_receipt_off_the_timing_contract_fails_with_every_control_passing(self):
        self.generate(receipt=_receipt("write-evict"))
        code, doc, _fake = self.verify(_toy_controls())
        self.assertEqual((code, doc["verdict"]), (1, "FAIL"))
        self.assertEqual(doc["errors"], ["receipt: cache_mode 'write-evict' is not read-evict"])
        self.assertEqual(self.verdicts(doc), GENERIC_PASS)

    def test_without_a_reusable_ut_result_the_aa_is_measured_through_the_candidate_binding(self):
        self.generate(ut_result=False, ut_native=False)
        native = os.path.join(self.work, "native")
        script = dict(_toy_controls(), aa_1=[(0, _toy_ut_result(self.task))])
        code, doc, fake = self.verify(script, native_stamp={"contract_sha256": self.contract, "world": 1})
        self.assertEqual((code, doc["verdict"]), (0, "PASS"), doc["errors"])
        self.assertEqual((fake.calls[0]["what"], fake.calls[0]["args"][4:]),
                         ("aa_1", ["--no-early-stop", "--reps", "3", "--native-dir", native]))      # no --aa
        self.assertIs(doc["aa"]["through_candidate_binding"]["value"], True)
        self.assertEqual(doc["aa"]["receipt_from"], "run/verify_ut/aa_1.json")
        self.assertEqual(self.native_dirs(fake), {native})     # the A/A's own native references serve the controls
        self.assertEqual(self.verdicts(doc), GENERIC_PASS)

    def test_native_references_are_measured_when_the_ut_left_none_for_this_contract(self):
        self.generate(ut_native=False)
        native = os.path.join(self.work, "native")
        refs = ["--aa", "--correctness-only", "--native-dir", native]
        code, doc, fake = self.verify(dict(_toy_controls(), native_refs=[(0, {"gates": {"correctness": "pass"}})]))
        self.assertEqual((code, doc["verdict"]), (0, "PASS"), doc["errors"])
        self.assertEqual((fake.calls[0]["what"], fake.calls[0]["args"][4:]), ("native_refs", refs))
        self.assertEqual(self.native_dirs(fake), {native})
        # references stamped for another contract do not count; a failed reference run is a FAIL, no control runs
        self.t.write("run/ut_native/native.json", json.dumps({"contract_sha256": "0" * 64, "world": 1}))
        code, doc, fake = self.verify(dict(_toy_controls(), native_refs=[(1, {"reason": "native self-check failed"})]))
        self.assertEqual((code, doc["verdict"], doc["controls"]), (1, "FAIL", {}))
        self.assertEqual(doc["errors"], ["native reference run failed (exit 1): native self-check failed"])
        self.assertEqual([c["what"] for c in fake.calls], ["native_refs"])
        # stopped from outside every time: rerun with half as many pools alive, then INCONCLUSIVE without an error
        lock = (2, {"reason": "ERROR: Failed to acquire GPU 3 lock after 600s", "multi_rank": {"pool_parallel": 4}})
        code, doc, fake = self.verify(dict(_toy_controls(), native_refs=[lock]))
        self.assertEqual((code, doc["verdict"], doc["errors"], doc["controls"]), (2, "INCONCLUSIVE", [], {}))
        self.assertEqual([c["parallel"] for c in fake.calls], [None, "2", "1"])
        self.assertEqual([(a["what"], a["pool_parallel"]) for a in doc["attempts"]],
                         [("native_refs", None), ("native_refs", 2), ("native_refs", 1)])

    def test_a_control_the_ut_cannot_see_fails_the_verification(self):
        self.generate()
        blind = _res(0, _rep(eager=[_c("decode_m1", True, 0.0), _c("decode_m4", True, 0.0)]))
        code, doc, _fake = self.verify(_toy_controls({"scale_out": [(0, blind)]}))
        self.assertEqual((code, doc["verdict"], self.verdicts(doc)["scale_out"]), (1, "FAIL", "FAIL"))
        self.assertEqual(doc["errors"],
                         [("control scale_out: never rejected; must fail in one of ['eager', 'random', 'graph_replay', "
                           "'sequence']")])

    def test_a_below_tol_task_control_is_a_gap_and_one_seen_near_tol_a_warning(self):
        for name in ("tiny", "near"):
            self.t.write(f"controls/{name}.py", f"# task control {name}\n")
        m = self.t.meta()
        m["controls"] = {"task": [{"name": "tiny", "file": "controls/tiny.py", "expect": "below_tol"},
                                  {"name": "near", "file": "controls/near.py"}]}
        self.t.write_meta(m)
        self.generate()
        tiny = _res(0, _rep(eager=[_c("decode_m1", True, 0.004), _c("decode_m4", True, 0.0)]))
        near = _res(1, _rep(eager=[_c("decode_m1", False, 0.012), _c("decode_m4", True, 0.0)]))
        code, doc, fake = self.verify(_toy_controls({"tiny": [(0, tiny)], "near": [(1, near)]}))
        gap = "control tiny: verified below tol on every case: a coverage gap the test cannot see"
        self.assertEqual((code, doc["verdict"]), (1, "FAIL"))
        self.assertEqual(self.verdicts(doc), dict(GENERIC_PASS, tiny="GAP", near="PASS"))
        self.assertEqual((doc["gaps"], doc["errors"]), ([gap], [f"coverage gaps (ISSUE-028/012): {gap}"]))
        self.assertEqual(doc["warnings"], ["control near: seen only below 1.5 x tol (max error 0.012) (ISSUE-033)"])
        specs = {c["what"]: json.loads(c["args"][c["args"].index("--control-spec") + 1])
                 for c in fake.calls if "--control-spec" in c["args"]}
        self.assertEqual(specs, {"control:tiny": {"file": "controls/tiny.py", "expect": "below_tol"},
                                 "control:near": {"file": "controls/near.py"}})

    def test_a_control_stopped_by_its_environment_has_no_verdict(self):
        self.generate()
        oom = (1, {"reason": "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 2.00 GiB"})
        code, doc, fake = self.verify(_toy_controls({"stale_output": [oom]}))
        self.assertEqual((code, doc["verdict"], doc["errors"]), (2, "INCONCLUSIVE", []))
        self.assertEqual(self.verdicts(doc), dict(GENERIC_PASS, stale_output="INCONCLUSIVE"))
        self.assertEqual([c["parallel"] for c in fake.calls if c["what"] == "control:stale_output"], [None, "1", "1"])

    def test_a_task_without_meta_is_a_named_fail_and_the_report_is_still_written(self):
        os.remove(os.path.join(self.task, "meta.json"))
        code, doc, fake = self.verify({})
        self.assertEqual((code, doc["verdict"], fake.calls), (1, "FAIL", []))
        self.assertTrue(doc["errors"][0].startswith("verify_ut raised FileNotFoundError: "), doc["errors"])
        self.assertNotIn("snapshot_sha256", doc)


class _DiskFullOnReplace:
    """generate_ut's `os` on a full disk: os.replace fails as rename(2) does with ENOSPC (and records what it refused);
    every other name is the real os module."""

    def __init__(self):
        self.refused = []

    def __getattr__(self, name):
        return getattr(os, name)

    def replace(self, src, dst, **_kw):
        self.refused.append((src, dst))
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), src, None, dst)


class TestStaticProblems(unittest.TestCase):
    """The static part names every problem it finds: missing files, check() problems / warnings / exceptions, and a
    task whose source snapshot cannot be taken."""

    def setUp(self):
        keep_sys_path(self)
        self.t = ToyTask()
        self.addCleanup(self.t.cleanup)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gu.main(["all", self.t.task]), 0)

    def verify_static(self):
        out = os.path.join(self.t.task, "run", "verify_static.json")
        with contextlib.redirect_stdout(io.StringIO()):
            code = vu.main(["--task", self.t.task, "--static-only", "--out", out])
        return code, self.t.read_json("run/verify_static.json")

    def test_missing_runtime_and_entry_files_are_named(self):
        for name in ("leg_runner.py", "unittest.py"):
            os.remove(os.path.join(self.t.task, name))
        code, doc = self.verify_static()
        self.assertEqual((code, doc["verdict"]), (1, "FAIL"))
        for msg in ("runtime file leg_runner.py missing from the task", "unittest.py missing",
                    "contract: leg_runner.py: missing", "contract: unittest.py: missing"):
            self.assertIn(msg, doc["errors"])

    def test_a_check_that_raises_is_named_not_a_crash(self):
        # The real check(): it raises where it writes run/ut_check.json -- the disk is full while (and only while) the
        # static part runs it. verify_ut names the exception and fails the static part instead of crashing.
        full = _DiskFullOnReplace()
        with mock.patch.object(vu.generate_ut, "os", full):
            code, doc = self.verify_static()
        dst = os.path.join(os.path.realpath(self.t.task), "run", "ut_check.json")
        tmp = f"{dst}.tmp{os.getpid()}"
        self.assertEqual(full.refused, [(tmp, dst)])
        named = f"check: check raised OSError: [Errno {errno.ENOSPC}] {os.strerror(errno.ENOSPC)}: '{tmp}' -> '{dst}'"
        self.assertEqual((code, doc["verdict"], doc["errors"]), (1, "FAIL", [named]))
        self.assertEqual(doc["static"], {"ok": False, "world_size": 1})

    def test_check_problems_fail_and_its_warnings_are_kept(self):
        os.remove(os.path.join(self.t.task, "run", "requested_kernel.json"))
        self.t.write("kernel_src/extra.cu", "// written for the task, undeclared\n")
        code, doc = self.verify_static()
        self.assertEqual((code, doc["verdict"], len(doc["errors"])), (1, "FAIL", 1))
        self.assertTrue(doc["errors"][0].startswith("check: kernel_src/ files are either copied source"), doc["errors"])
        self.assertTrue(doc["errors"][0].endswith("unclassified: kernel_src/extra.cu"), doc["errors"])
        self.assertEqual(doc["warnings"], [("check: run/requested_kernel.json not found: the GPU-count declaration is "
                                            "checked at delivery")])

    def test_a_task_whose_kernel_src_holds_no_source_has_no_snapshot(self):
        os.remove(os.path.join(self.t.task, "kernel_src", "toyop.py"))
        code, doc = self.verify_static()
        self.assertEqual((code, doc["verdict"]), (1, "FAIL"))
        self.assertIn("snapshot: kernel_src contains no source files", doc["errors"])
        self.assertNotIn("snapshot_sha256", doc)


class TestDriver(unittest.TestCase):
    """_driver: one ut_runner.py process (answered by FakeUTRunner) -> (exit code, result), its log written beside the
    result."""

    def setUp(self):
        self.task = tempfile.mkdtemp(prefix="geak_vu_driver_")
        self.addCleanup(shutil.rmtree, self.task, True)
        self.out = os.path.join(self.task, "aa_1.json")

    def drive(self, answer, out=None, env_extra=None, timeout=None):
        out = out or self.out
        fake = FakeUTRunner({os.path.splitext(os.path.basename(out))[0]: [answer]})
        with mock.patch.object(vu, "subprocess", fake):
            code, res = vu._driver(self.task, ["--aa", "--correctness-only"], out, env_extra, timeout)
        return code, res, fake.calls[0]

    @staticmethod
    def read(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def test_a_retry_that_writes_no_result_is_judged_on_its_own_exit_not_on_the_last_attempts_file(self):
        """Every attempt of a run writes the same --out. A retry whose ut_runner.py ends without writing it (exit 1)
        has a verdict of its own; reading back the previous attempt's environment failure from that file would turn
        it into one more run "without a verdict" (and the A/A into INCONCLUSIVE instead of FAIL)."""
        fake = FakeUTRunner({"aa_1": [(2, {"reason": "ERROR: Failed to acquire GPU 3 lock after 600s"}), (1, None)]})
        rep = vu.Report(self.task)
        with mock.patch.object(vu, "subprocess", fake):
            out, code, res, nv = vu._measure(self.task, rep, self.task, "aa_1", os.path.join(self.task, "native"), True)
        self.assertEqual((out, code, res["exit_code"], nv), (self.out, 1, 1, False))
        self.assertNotIn("Failed to acquire GPU 3 lock", res["reason"])      # not the first attempt's result
        self.assertIn("UT_RESULT: exit 1", res["reason"])                  # the retry's own output tail
        self.assertEqual([(a["exit_code"], a["pool_parallel"]) for a in rep.doc["attempts"]], [(2, None), (1, 1)])
        self.assertEqual([c["parallel"] for c in fake.calls], [None, "1"])
        self.assertFalse(os.path.exists(self.out))                          # the retry wrote none

    def test_the_result_the_run_wrote_is_returned_with_its_log(self):
        code, res, call = self.drive((1, {"reason": "correctness failed on some rank"}),
                                     env_extra={"GEAK_UT_POOL_PARALLEL": "1"}, timeout=60)
        log = os.path.join(self.task, "aa_1.log")
        self.assertEqual((code, res), (1, {"reason": "correctness failed on some rank", "exit_code": 1, "log": log}))
        self.assertEqual(call["cmd"], [sys.executable, os.path.join(self.task, "ut_runner.py"), "--task", self.task,
                                       "--out", self.out, "--aa", "--correctness-only"])
        self.assertEqual((call["cwd"], call["timeout"], call["capture"], call["parallel"]),
                         (self.task, 60, (True, True), "1"))
        self.assertEqual(self.read(log), f"$ {' '.join(call['cmd'])}\n[exit 1]\nUT_RESULT: exit 1\n\n--- stderr ---\n"
                                         "[ut] the stderr tail\n")

    def test_a_run_that_wrote_no_result_is_read_from_its_output(self):
        code, res, _call = self.drive((3, None))
        self.assertEqual((code, res["exit_code"]), (3, 3))
        self.assertEqual(res["reason"], "UT_RESULT: exit 3\n\n--- stderr ---\n[ut] the stderr tail\n")
        self.assertFalse(vu._no_verdict(code, res))

    def test_a_run_killed_by_a_signal_has_no_verdict(self):
        code, res, _call = self.drive((-9, None))
        self.assertEqual((code, res["killed_by_signal"]), (vu.ut_runner.EXIT_ENV, 9))
        self.assertTrue(vu._no_verdict(code, res))

    def test_a_run_past_its_timeout_has_no_verdict(self):
        code, res, _call = self.drive((0, subprocess.TimeoutExpired(["ut_runner.py"], 5)), timeout=5)
        self.assertEqual((code, res["exit_code"]), (vu.ut_runner.EXIT_ENV, vu.ut_runner.EXIT_ENV))
        self.assertTrue(res["reason"].startswith("verify_ut: driver timed out after 5s: "), res["reason"])
        self.assertIn("\n[exit 2]\nverify_ut: driver timed out after 5s: ", self.read(res["log"]))
        self.assertTrue(vu._no_verdict(code, res))

    def test_an_out_path_without_json_gets_its_log_beside_it(self):
        out = os.path.join(self.task, "probe")
        code, res, _call = self.drive((0, {"timing": {}}), out=out)
        self.assertEqual((code, res["log"]), (0, out + ".log"))
        self.assertTrue(os.path.isfile(out + ".log"))


class TestRunsWithoutAVerdict(unittest.TestCase):
    """ISSUE-029: a run stopped from outside its code has no verdict and is run again (at most twice more) with half
    as many pools alive each time; _measure is one A/A timing run through that rule."""

    def test_a_run_is_rerun_until_it_has_a_verdict(self):
        answers = [(2, {"reason": "ERROR: Failed to acquire GPU 3 lock after 600s", "multi_rank": {"pool_parallel": 4}}),
                   (1, {"reason": "hipIpcOpenMemHandle failed: invalid device pointer"}),
                   (1, {"reason": "correctness failed on some rank"})]
        seen = []

        def fn(attempt, parallel):
            seen.append((attempt, parallel))
            return answers[attempt]
        rep = vu.Report("/t")
        code, res, nv = vu.run_with_retries(rep, "aa_1", fn)
        self.assertEqual((code, res["reason"], nv), (1, "correctness failed on some rank", False))
        self.assertEqual(seen, [(0, None), (1, 2), (2, 1)])
        self.assertEqual([(a["what"], a["exit_code"], a["pool_parallel"]) for a in rep.doc["attempts"]],
                         [("aa_1", 2, None), ("aa_1", 1, 2), ("aa_1", 1, 1)])

    def test_three_runs_without_a_verdict_stay_without_one(self):
        rep = vu.Report("/t")
        code, res, nv = vu.run_with_retries(rep, "control:noop", lambda attempt, parallel: (1, {"reason": "hipErrorNoDevice"}))
        self.assertEqual((code, res["reason"], nv, len(rep.doc["attempts"])), (1, "hipErrorNoDevice", True, 3))

    def test_only_the_environment_takes_the_verdict_away(self):
        self.assertTrue(vu._no_verdict(vu.ut_runner.EXIT_ENV, {}))
        self.assertTrue(vu._no_verdict(1, {"reason": "RuntimeError: HIP out of memory"}))
        self.assertFalse(vu._no_verdict(1, {"reason": "AssertionError: the candidate raised"}))
        self.assertFalse(vu._no_verdict(vu.ut_runner.EXIT_INCOMPLETE, {"reason": None}))

    def test_measure_reruns_the_aa_with_fewer_pools(self):
        # The real _driver: each attempt is a `python3 <task>/ut_runner.py` process (answered by FakeUTRunner) whose
        # result file and log land in the work dir.
        task = tempfile.mkdtemp(prefix="geak_vu_measure_")
        self.addCleanup(shutil.rmtree, task, True)
        work, native = os.path.join(task, "run", "verify_ut"), os.path.join(task, "run", "verify_ut", "native")
        os.makedirs(work)
        out, log = os.path.join(work, "aa_1.json"), os.path.join(work, "aa_1.log")
        foreign = (2, {"reason": "ERROR: GPU 3 still shows foreign work after 600s", "multi_rank": {"pool_parallel": 2}})
        fake = FakeUTRunner({"aa_1": [foreign, (0, {"timing": {"primary": 1.0}})]})
        rep = vu.Report(task)
        with mock.patch.dict(os.environ), mock.patch.object(vu, "subprocess", fake):
            os.environ.pop("GEAK_UT_POOL_PARALLEL", None)
            got = vu._measure(task, rep, work, "aa_1", native, False)
        self.assertEqual(got, (out, 0, {"timing": {"primary": 1.0}, "exit_code": 0, "log": log}, False))
        cmd = [sys.executable, os.path.join(task, "ut_runner.py"), "--task", task, "--out", out,
               "--no-early-stop", "--reps", "3", "--native-dir", native, "--aa"]
        self.assertEqual([c["cmd"] for c in fake.calls], [cmd, cmd])
        self.assertEqual([c["parallel"] for c in fake.calls], [None, "1"])      # the retry: half of its 2 pools
        self.assertEqual([(a["what"], a["attempt"], a["exit_code"], a["pool_parallel"], a["log"])
                          for a in rep.doc["attempts"]], [("aa_1", 0, 2, None, log), ("aa_1", 1, 0, 1, log)])
        with open(log, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), f"$ {' '.join(cmd)}\n[exit 0]\nUT_RESULT: exit 0\n\n--- stderr ---\n"
                                        "[ut] the stderr tail\n")


class TestAAWithoutAVerdict(unittest.TestCase):
    """run_aa on a generated toy task when a measurement has no verdict or fails, and when the --aa diagnostic has none:
    the real aa_source / _reusable_ut / _measure / _driver, the ut_runner.py processes answered by FakeUTRunner."""

    def setUp(self):
        keep_sys_path(self)
        self.t = ToyTask()
        self.addCleanup(self.t.cleanup)
        self.task = os.path.realpath(self.t.task)
        self.work = os.path.join(self.task, "run", "verify_ut")
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("GEAK_UT_POOL_PARALLEL", None)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gu.main(["all", self.task]), 0)

    def run_aa(self, script):
        """What verify_ut.main does up to the A/A -- the static part binds the report to the task's contract and
        snapshot, the work dir is made -- then run_aa with every ut_runner.py process answered from `script`."""
        self.rep = vu.Report(self.task)
        self.assertTrue(vu.static_checks(self.task, self.rep), self.rep.doc["errors"])
        os.makedirs(self.work)
        self.fake = FakeUTRunner(script)
        with mock.patch.object(vu, "subprocess", self.fake):
            return vu.run_aa(self.task, self.t.meta(), self.rep, self.work, None)

    def started(self):
        """[(what, through the candidate binding)] of every ut_runner.py process started, in order."""
        return [(c["what"], "--aa" not in c["args"]) for c in self.fake.calls]

    def test_an_aa_stopped_from_outside_is_inconclusive(self):
        # The copied source edited after freeze is not the baseline's code: both legs run the baseline (--aa), and the
        # UT's own passing result of this very contract and snapshot is no A/A.
        with open(os.path.join(self.task, "kernel_src", "toyop.py"), "a", encoding="utf-8") as fh:
            fh.write("# a local note\n")
        self.t.write("run/ut_result.json", json.dumps(_toy_ut_result(self.task)))
        oom = "OutOfMemoryError: HIP out of memory. Tried to allocate 2.00 GiB"
        verdict, res = self.run_aa({"aa_1": [(2, {"reason": oom})]})
        self.assertEqual((verdict, res), ("INCONCLUSIVE", None))
        self.assertIsNotNone(vu._reusable_ut(self.task, self.rep))      # reusable by its binding: the class keeps it out
        self.assertEqual(self.started(), [("aa_1", False)] * 3)         # no verdict: run twice more, then given up
        self.assertEqual([c["parallel"] for c in self.fake.calls], [None, "1", "1"])
        why = f"kernel_src/toyop.py differs from its origin {os.path.join(self.t.site, 'toyop.py')}"
        self.assertEqual(self.rep.doc["aa"]["through_candidate_binding"], {"value": False, "why": why})
        self.assertIn(f"A/A did not go through the candidate binding (both legs ran the baseline, --aa): {why}",
                      self.rep.doc["notes"])
        self.assertEqual(self.rep.doc["aa"]["runs"], [{"source": os.path.join(self.work, "aa_1.json"), "exit_code": 2,
                                                       "no_verdict": True, "reason": oom}])
        self.assertEqual((self.rep.doc["aa"]["verdict"], self.rep.doc["errors"]), ("INCONCLUSIVE", []))

    def test_an_aa_run_that_fails_is_a_fail_naming_its_reason(self):
        # The pristine copy goes through the binding; the UT's own run of this contract and snapshot failed, so it is
        # no measurement to reuse: a fresh run, which fails.
        self.t.write("run/ut_result.json", json.dumps(dict(_toy_ut_result(self.task), exit_code=1)))
        failed = {"reason": "correctness failed on some rank"}
        verdict, res = self.run_aa({"aa_1": [(1, failed)]})
        self.assertEqual((verdict, res), ("FAIL", dict(failed, exit_code=1, log=os.path.join(self.work, "aa_1.log"))))
        self.assertEqual(self.started(), [("aa_1", True)])
        self.assertEqual(self.rep.doc["aa"]["through_candidate_binding"],
                         {"value": True, "why": "1 copied source file(s) byte-identical to their origins, "
                                                "0 declared binding file(s)"})
        self.assertEqual(self.rep.doc["aa"]["runs"], [{"source": os.path.join(self.work, "aa_1.json"), "exit_code": 1,
                                                       "reason": "correctness failed on some rank"}])
        self.assertEqual(self.rep.doc["aa"]["verdict"], "FAIL")
        self.assertEqual(self.rep.doc["errors"], ["A/A run failed (exit 1): correctness failed on some rank"])

    def test_a_diagnostic_without_a_verdict_is_recorded_as_such(self):
        # The UT's own result of this contract and snapshot is the first A/A, out of band on decode (decode_m4 at
        # 1.125): the --aa diagnostic is stopped from outside every time, then the second A/A is in band.
        self.t.write("run/ut_result.json", json.dumps(_toy_aa(self.task, 1.002, 1.125)))
        lock = "ERROR: GPU 3 still shows foreign work after 600s"
        verdict, res = self.run_aa({"aa_diagnostic": [(2, {"reason": lock})],
                                    "aa_2": [(0, _toy_aa(self.task, 0.998, 1.004))]})
        self.assertEqual((verdict, res["log"]), ("PASS", os.path.join(self.work, "aa_2.log")))
        self.assertEqual(self.started(), [("aa_diagnostic", False)] * 3 + [("aa_2", True)])
        self.assertEqual([c["parallel"] for c in self.fake.calls], [None, "1", "1", None])
        self.assertEqual(self.rep.doc["aa"]["diagnostic"], {"source": "run/verify_ut/aa_diagnostic.json",
                                                            "exit_code": 2, "no_verdict": True, "reason": lock})
        runs = self.rep.doc["aa"]["runs"]
        self.assertEqual([(r["source"], r["in_band"]) for r in runs],
                         [("run/ut_result.json", False), ("run/verify_ut/aa_2.json", True)])
        self.assertEqual([(ph, v["worst_case"]["sig"], v["worst_case"]["speedup"])
                          for ph, v in runs[0]["out_of_band"].items()], [("decode", "decode_m4", 1.125)])
        self.assertEqual((self.rep.doc["aa"]["receipt_from"], self.rep.doc["errors"]), ("run/verify_ut/aa_2.json", []))


class TestAAInputs(unittest.TestCase):
    """What the A/A reads from the task: a reusable UT result, the kernel_src classes, the phase's worst case."""

    def setUp(self):
        keep_sys_path(self)
        self.t = ToyTask()
        self.addCleanup(self.t.cleanup)

    def generate(self, step="all"):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gu.main([step, self.t.task]), 0)

    def test_only_a_passing_ut_result_of_the_same_contract_and_snapshot_is_reused(self):
        rep = vu.Report(self.t.task)
        rep.doc.update(contract_sha256="c1", snapshot_sha256="s1")
        self.assertIsNone(vu._reusable_ut(self.t.task, rep))              # no UT result yet
        self.t.write("run/ut_result.json", '{"exit_code": 0,')            # cut short
        self.assertIsNone(vu._reusable_ut(self.t.task, rep))
        good = {"exit_code": 0, "contract_sha256": "c1", "snapshot_sha256": "s1", "timing": {"primary": 1.0}}
        for over, reused in (({}, True), ({"exit_code": 1}, False), ({"contract_sha256": "c0"}, False),
                             ({"snapshot_sha256": "s0"}, False)):
            self.t.write("run/ut_result.json", json.dumps(dict(good, **over)))
            self.assertEqual(vu._reusable_ut(self.t.task, rep), dict(good, **over) if reused else None, over)

    def test_without_a_frozen_contract_both_legs_run_the_baseline(self):
        self.generate("vendor")
        self.assertEqual(vu.aa_source(self.t.task, self.t.meta()), ("modified", "no frozen contract"))
        self.t.write("run/ut_contract.json", "{")
        self.assertEqual(vu.aa_source(self.t.task, self.t.meta()), ("modified", "no frozen contract"))

    def test_an_origin_removed_after_freeze_is_unreadable(self):
        self.generate()
        origin = os.path.join(self.t.site, "toyop.py")
        os.remove(origin)
        self.assertEqual(vu.aa_source(self.t.task, self.t.meta()), ("modified", f"origin {origin} unreadable"))

    def test_binding_files_alone_are_not_the_baselines_code(self):
        m = self.t.meta()
        m["kernel_extraction"].update(source_origins={}, binding_files=["kernel_src/toyop.py"])
        self.t.write_meta(m)
        self.generate()
        self.assertEqual(vu.kernel_src_unmodified(self.t.task),
                         (False, "kernel_src has no copied source with a recorded origin"))

    def test_a_phase_without_a_timed_case_has_no_worst_case(self):
        res = _timed({"prefill": 1.2}, [_case("d", "decode", 1.0), {"sig": "p", "regime": "prefill", "speedup": None}])
        self.assertIsNone(vu.worst_case(res, "prefill"))
        _ok, _corr, _probs, detail = vu._aa_ok(res, 0.01, vu.Report("t"), "x")
        self.assertEqual(detail["out_of_band"], {"prefill": {"speedup": 1.2, "worst_case": None}})


class TestReceiptProblems(unittest.TestCase):
    """check_receipt: what makes the A/A's timing receipt unusable."""

    @staticmethod
    def leg(timer="cuda_event", **kw):
        return dict({"ms": 1.0, "timer": timer}, **kw)

    @staticmethod
    def check(receipt):
        rep = vu.Report("t")
        ok = vu.check_receipt({}, {"receipt": receipt}, rep)
        return ok, rep.doc["errors"], rep.doc["receipt"]

    def test_a_result_without_a_receipt_fails(self):
        for res in (None, {"timing": {}}, {"receipt": "read-evict"}):
            rep = vu.Report("t")
            self.assertFalse(vu.check_receipt({}, res, rep))
            self.assertEqual((rep.doc["errors"], rep.doc["receipt"]),
                             (["timing receipt missing from the A/A result"], {"ok": False}))

    def test_the_cache_mode_the_timers_and_the_queued_spins_must_match(self):
        other = dict(SPIN_REC, queued="host_sleep_before_start")
        ok, errors, rec = self.check({"cache_mode": "write-evict", "world_size": 1, "timer": "mixed", "cases": {
            "p": {"baseline": self.leg(spin=SPIN_REC), "current": self.leg(spin=other), "deployed_graph": False},
            "q": {"baseline": self.leg("cuda_event_graph"), "current": self.leg("wall"), "deployed_graph": True}}})
        self.assertFalse(ok)
        self.assertEqual(errors, ["receipt: cache_mode 'write-evict' is not read-evict",
                                  ("receipt: p: the legs queued different spins (gpu_spin_before_start / "
                                   "host_sleep_before_start)"),
                                  "receipt: q: legs used different timers"])
        self.assertEqual((rec["ok"], rec["cache_mode"], rec["timer"], rec["world_size"]), (False, "write-evict", "mixed", 1))

    def test_a_multi_rank_receipt_needs_a_time_per_rank_and_its_alignment(self):
        cases = {"p": {"baseline": self.leg(rank_ms=[1.0, 1.1]), "current": self.leg(rank_ms=[1.0]),
                       "deployed_graph": False},
                 "q": {"baseline": self.leg(rank_ms=[1.0, None]), "current": self.leg(rank_ms=[0.9, 1.0]),
                       "deployed_graph": False}}
        ok, errors, _rec = self.check({"cache_mode": "read-evict", "world_size": 2, "timer": "cuda_event", "cases": cases})
        self.assertFalse(ok)
        self.assertEqual(errors, ["receipt: p: current lacks a time for each of the 2 ranks",
                                  "receipt: q: baseline lacks a time for each of the 2 ranks",
                                  "receipt: multi-rank receipt lacks the alignment method"])
        cases["p"]["current"]["rank_ms"] = [1.0, 1.05]
        cases["q"]["baseline"]["rank_ms"] = [1.0, 1.2]
        ok, errors, rec = self.check({"cache_mode": "read-evict", "world_size": 2, "timer": "cuda_event", "cases": cases,
                                      "alignment": {"method": "barrier_then_event"}})
        self.assertEqual((ok, errors, rec["world_size"]), (True, [], 2))     # no queued spin asked of multi-rank eager


class TestControlReports(unittest.TestCase):
    """Report shapes the control judgement meets: stray entries, a phase that raised, a run that could not be evaluated,
    the carry_state replay groups of skip_state_write (A18.3), and the plan of controls."""

    def test_stray_entries_are_skipped_and_a_phase_that_raised_is_a_raised_row(self):
        report = {"decode": {"eager": ["a stray string", {"case": "output_independence", "correct": False,
                                                          "max_rel_err": None}, _c("decode_m1", True, 0.0)]},
                  "prefill": {"error": "Traceback (most recent call last): ... RuntimeError: boom"},
                  "sequence": {"incomplete": "capabilities.stateful is declared but cases.py has no sequence_cases"}}
        rows = vu.classify_cases(report, TOL)
        self.assertEqual([(r["section"], r["case"], r["kind"]) for r in rows],
                         [("eager", "decode_m1", "pass"), ("error", None, "raised"), ("error", None, "raised")])
        self.assertIn("RuntimeError: boom", rows[1]["note"])
        v, d = vu.judge_control("scale_out", {}, 1, _res(1, report), TOL, 1)
        self.assertEqual(v, "FAIL")
        self.assertIn("never rejected (only raising calls, which compare no output: 2)", d["why"])
        v, d = vu.judge_control("noop", {}, 1, _res(1, report), TOL, 1)
        self.assertEqual((v, d["why"]), ("FAIL", "must PASS but the UT rejected it (2 failing/raised case(s))"))

    def test_a_control_that_could_not_be_evaluated_fails(self):
        v, d = vu.judge_control("noop", {}, vu.ut_runner.EXIT_INCOMPLETE,
                                {"exit_code": 3, "reason": "coverage (ISSUE-006): decode_m4 never compared"}, TOL, 1)
        self.assertEqual((v, d["why"]),
                         ("FAIL", "the control could not be evaluated: coverage (ISSUE-006): decode_m4 never compared"))

    @staticmethod
    def carry_report(replay):
        groups = [{"group": 0, "carry_state": False, "steps": 2}, {"group": 1, "carry_state": True, "steps": 3}, "x"]
        return {"decode": {"eager": [], "random": [], "graph_replay": replay, "replay_groups": groups},
                "prefill": {"eager": [], "graph_replay": [], "replay_groups": [{"group": 0, "carry_state": False}]},
                "sequence": [_c("seq[1]:s", False, 0.5)]}

    def test_skip_state_write_is_seen_in_the_replay_of_every_carry_state_group(self):
        report = self.carry_report([dict(_c("g0", False, 0.3), group=0), dict(_c("g1a", False, 0.4), group=1),
                                    dict(_c("g1b", False, None, "replay raised: x"), group=1),
                                    dict(_c("g1c", False, None, "graph capture failed"), group=1),
                                    dict(_c("g1d", True, 0.0), group=1), "not a row"])
        self.assertEqual(vu.carry_groups(report), {"decode": {1}})
        self.assertEqual(vu.carry_rows_failed(report, TOL), ["decode:group1:g1a"])
        v, d = vu.judge_control("skip_state_write", {}, 1, _res(1, report), TOL, 1)
        self.assertEqual((v, d["carry_state_rows"]), ("PASS", ["decode:group1:g1a"]))

    def test_rejected_in_the_sequence_only_fails_when_a_group_carries_state(self):
        report = self.carry_report([dict(_c("g0", False, 0.3), group=0), dict(_c("g1a", True, 0.0), group=1)])
        self.assertEqual(vu.carry_rows_failed(report, TOL), [])
        v, d = vu.judge_control("skip_state_write", {}, 1, _res(1, report), TOL, 1)
        self.assertEqual((v, d["carry_state_rows"]), ("FAIL", []))
        self.assertIn("not in the graph replay of the carry_state group(s)", d["why"])

    def test_the_plan_follows_the_capabilities_the_disabled_and_the_task_controls(self):
        rep = vu.Report("/t")
        why = "every output is written on every call: nothing stale to read"
        task_ctl = {"name": "drop_bias", "file": "controls/drop_bias.py", "expect": "reject"}
        meta = {"capabilities": {"stateful": True}, "controls": {"disabled": {"stale_output": why}, "task": [task_ctl]}}
        self.assertEqual(vu.control_plan(meta, 2, rep),
                         [("noop", {}), ("within_tol", {}), ("scale_out", {}), ("skip_state_write", {}),
                          ("rank_skew", {}), ("drop_bias", task_ctl)])
        self.assertEqual(rep.doc["controls"], {"stale_output": {"verdict": "DISABLED", "why": why}})
        self.assertEqual(vu.control_plan({}, 1, vu.Report("/t")),
                         [("noop", {}), ("within_tol", {}), ("scale_out", {}), ("stale_output", {})])


if __name__ == "__main__":
    unittest.main(verbosity=2)
