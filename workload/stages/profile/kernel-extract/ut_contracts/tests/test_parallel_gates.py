"""GPU tests for the UT driver's parallel paths, on the toy Triton GEMM task of
test_server_evidence.py: correctness sharded by case over two GPUs, timing spread by bucket,
--quick and --reuse.

Run with at least two visible GPUs:
  HIP_VISIBLE_DEVICES=6,7 python3 .../ut_contracts/tests/test_parallel_gates.py
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_server_evidence as S  # noqa: E402

TWO = S.GPU and S.torch.cuda.device_count() >= 2


@unittest.skipUnless(TWO, "needs torch, triton and two GPUs")
class ParallelGatesTest(S.ServerEvidenceTest):

    def scaffold(self):
        super().scaffold()
        # more cases, so a shard is more than one case
        path = os.path.join(self.task, "run", "ut_cases.json")
        with open(path) as f:
            man = json.load(f)
        for m in (8, 16, 33, 64, 100):
            man["cases"].append({"sig": "PREFILL_m%d" % m, "regime": "PREFILL", "axes": {"m": m},
                                 "effective_calls": 1, "timed": True, "random": True})
        with open(path, "w") as f:
            json.dump(man, f)
        p = self.run_py([os.path.join(S.UTC, "scaffold_ut.py"), "sync-meta", "--task", self.task])
        self.assertEqual(p.returncode, 0, p.stderr)
        self.evidence(server_preimports=["toyfw.common"])

    def ut(self, *extra, gpus="0,1"):
        self.env["UT_GPUS"] = gpus
        p = self.run_py([os.path.join(self.task, "unittest.py"), *extra])
        name = "ut_report.quick.json" if "--quick" in extra else "ut_report.json"
        with open(os.path.join(self.task, "run", name)) as f:
            return p, json.load(f)

    # the inherited server-evidence tests are covered there
    test_server_records_the_warmup_variant_and_its_call = None
    test_without_the_server_call_the_m1_variant_is_reported_missing = None
    test_a_waiver_with_a_reason_accepts_the_missing_variant = None
    test_replayed_server_calls_cover_every_variant_and_the_ut_passes = None
    test_a_candidate_broken_only_on_the_warmup_path_fails_server_calls = None
    test_without_preimports_the_env_differs_and_with_them_next_pow2_fails = None

    def test_two_gpus_pass_every_gate_with_the_same_cases_as_one(self):
        p, two = self.ut("--max-reps", "1")
        self.assertEqual(two["verdict"], "pass", json.dumps(two["gates"], indent=1, default=str)[:4000])
        self.assertEqual(two["gpus"], ["0", "1"])
        p, one = self.ut("--max-reps", "1", gpus="0")
        self.assertEqual(one["verdict"], "pass")
        self.assertEqual([r["case"] for r in two["gates"]["correctness"]["cases"]],
                         [r["case"] for r in one["gates"]["correctness"]["cases"]])
        self.assertEqual(len(two["gates"]["timing"]["per_case"]), 7)
        self.assertEqual(two["gates"]["compile_variants"]["status"], "pass")

    def test_a_wrong_candidate_still_fails_when_sharded(self):
        self.write_src(S.TOY_GEMM.replace("acc += tl.dot(a, tl.trans(b))",
                                          "acc += tl.dot(a, tl.trans(b)) * 1.01"))
        p, rep = self.ut("--skip", "timing")
        self.assertEqual(rep["gates"]["correctness"]["status"], "fail")

    def test_quick_runs_the_boundary_cases_and_writes_no_receipt(self):
        p, rep = self.ut("--quick")
        self.assertEqual(p.returncode, 1)
        self.assertEqual(rep["verdict"], "quick")
        self.assertEqual(sorted(rep["quick"]["cases"]),
                         ["DECODE_m4", "PREFILL_m200", "PREFILL_m64", "PREFILL_m8"])
        self.assertNotIn("timing", rep["gates"])
        self.assertEqual(rep["gates"]["correctness"]["status"], "pass")
        self.assertFalse(os.path.exists(os.path.join(self.task, "run", "ut_result.json")))

    def test_reuse_runs_only_what_did_not_pass(self):
        p, part = self.ut("--skip", "timing")
        self.assertEqual(part["verdict"], "partial", json.dumps(part["gates"], default=str)[:3000])
        p, full = self.ut("--reuse", "--max-reps", "1")
        self.assertEqual(full["verdict"], "pass", json.dumps(full["gates"], default=str)[:3000])
        reused = sorted(g for g, r in full["gates"].items() if r.get("reused"))
        self.assertNotIn("timing", reused)
        self.assertIn("correctness", reused)
        with open(os.path.join(self.task, "run", "ut_result.json")) as f:
            self.assertTrue(json.load(f)["ok"])
        # any change to the task invalidates reuse
        with open(os.path.join(self.task, "kernel_src", "ops.py"), "a") as f:
            f.write("\n# touched\n")
        p, again = self.ut("--reuse", "--skip", "timing")
        self.assertFalse(any(r.get("reused") for r in again["gates"].values()))


@unittest.skipUnless(S.GPU, "needs torch, triton and a GPU")
class LaunchesGateTest(S.ServerEvidenceTest):
    """meta.expected_launches (build_task_plan) against what the baseline launches."""
    test_server_records_the_warmup_variant_and_its_call = None
    test_without_the_server_call_the_m1_variant_is_reported_missing = None
    test_a_waiver_with_a_reason_accepts_the_missing_variant = None
    test_replayed_server_calls_cover_every_variant_and_the_ut_passes = None
    test_a_candidate_broken_only_on_the_warmup_path_fails_server_calls = None
    test_without_preimports_the_env_differs_and_with_them_next_pow2_fails = None

    def expect(self, want, waivers=None):
        path = os.path.join(self.task, "meta.json")
        with open(path) as f:
            meta = json.load(f)
        meta["expected_launches"] = want
        if waivers is not None:
            meta.setdefault("ut_contract", {})["launch_waivers"] = waivers
        with open(path, "w") as f:
            json.dump(meta, f)

    def gates(self):
        self.env["UT_GPUS"] = "0"
        p = self.run_py([os.path.join(self.task, "unittest.py"), "--gates",
                         "contract,inputs,binding,launches"])
        with open(os.path.join(self.task, "run", "ut_report.json")) as f:
            return json.load(f)["gates"]["launches"]

    def test_the_kernels_the_server_launched_are_seen_per_phase(self):
        self.expect({"PREFILL": ["_gemm_kernel"], "DECODE": ["_gemm_kernel"]})
        r = self.gates()
        self.assertEqual(r["status"], "pass", json.dumps(r, default=str)[:2000])
        self.assertIn("_gemm_kernel", r["seen"]["PREFILL"])

    def test_a_stage_the_ut_never_reaches_fails_and_is_named(self):
        self.expect({"DECODE": ["_gemm_kernel", "gemm2_stage_the_ut_skips"]})
        r = self.gates()
        self.assertEqual(r["status"], "fail")
        self.assertEqual(r["missing"], {"DECODE": ["gemm2_stage_the_ut_skips"]})

    def test_a_reasoned_waiver_accepts_it(self):
        self.expect({"DECODE": ["_gemm_kernel", "gemm2_stage_the_ut_skips"]},
                    [{"kernel": "gemm2_stage_the_ut_skips", "phase": "DECODE",
                      "why": "launched by the server's warmup only"}])
        self.assertEqual(self.gates()["status"], "pass")

    def test_without_evidence_the_gate_is_not_required(self):
        self.assertEqual(self.gates()["status"], "not_required")


if __name__ == "__main__":
    unittest.main(verbosity=2)
