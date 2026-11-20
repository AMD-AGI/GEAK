"""CPU tests of generate_ut.py: vendor, check, freeze and the step-invalidation rule (ISSUE-019/UT-001)."""
import json
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import generate_ut as gu  # noqa: E402
import ut_runner  # noqa: E402
from ut_task_fixture import ToyTask, SCRIPTS  # noqa: E402


class _Case(unittest.TestCase):
    def setUp(self):
        # vendor() puts the scripts dir on sys.path for every new task: put the content back after each test,
        # in place and also when the test fails (a cleanup runs either way).
        self.addCleanup(sys.path.__setitem__, slice(None), list(sys.path))
        self.t = ToyTask()
        self.addCleanup(self.t.cleanup)
        self.task = self.t.task

    def run_gen(self, *args):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = gu.main(list(args) + [self.task])
        return code, json.loads(buf.getvalue())


class TestVendor(_Case):
    def test_vendor_copies_the_runtime_unchanged_and_writes_the_marked_entry(self):
        code, rep = self.run_gen("vendor")
        self.assertEqual(code, 0, rep)
        for name in ut_runner.RUNTIME_FILES:
            self.assertEqual(ut_runner.sha256_file(os.path.join(self.task, name)),
                             ut_runner.sha256_file(os.path.join(SCRIPTS, name)))
        with open(os.path.join(self.task, "unittest.py")) as fh:
            self.assertTrue(fh.readline().startswith(ut_runner.GENERATED_MARKER))
        files = self.t.meta()["kernel_extraction"]["source_delivery_files"]
        self.assertEqual(sorted(files), sorted(ut_runner.RUNTIME_FILES))
        self.assertTrue(os.path.isfile(os.path.join(self.task, "baseline_overlay", "sitecustomize.py")))

    def test_a_hand_written_unittest_is_never_overwritten_without_force(self):
        self.t.write("unittest.py", "print('mine')\n")
        with self.assertRaises(SystemExit):
            gu.vendor(self.task)
        gu.vendor(self.task, force=True)
        with open(os.path.join(self.task, "run", "unittest.handwritten.py")) as fh:
            self.assertIn("mine", fh.read())


class TestCheck(_Case):
    def setUp(self):
        super().setUp()
        gu.vendor(self.task)

    def problems(self):
        return gu.check(self.task)["problems"]

    def test_the_toy_task_passes(self):
        self.assertEqual(self.problems(), [])

    def test_missing_hooks_are_named(self):
        self.t.write("cases.py", "def call(args):\n    return args\n")
        p = " ".join(self.problems())
        for hook in ("timing_cases", "random_shapes", "eager_cases"):
            self.assertIn(hook, p)

    def test_a_graph_deployment_needs_replay(self):
        m = self.t.meta()
        m["regime"] = {"cuda_graph": True}
        self.t.write_meta(m)
        self.assertTrue(any("replay" in p for p in self.problems()))

    def test_a_stateful_task_needs_sequence_cases(self):
        m = self.t.meta()
        m["capabilities"]["stateful"] = True
        self.t.write_meta(m)
        self.assertTrue(any("sequence_cases" in p for p in self.problems()))

    def test_a_coverage_bucket_without_a_correctness_case_is_named(self):
        m = self.t.meta()
        m["case_manifest"]["cases"].append({"sig": "decode_m8", "regime": "decode", "m": 8, "roles": ["timing"]})
        self.t.write_meta(m)
        p = self.problems()
        self.assertTrue(any("regime=decode, m=8" in x and "correctness" in x for x in p), p)

    def test_a_positive_mass_case_without_a_timing_case_is_named(self):
        m = self.t.meta()
        m["workload"]["cases"].append({"sig": "decode_m2", "regime": "decode", "m": 2, "effective_calls": 5})
        self.t.write_meta(m)
        self.assertTrue(any("decode_m2" in x for x in self.problems()))

    def test_the_gpu_count_must_match_the_declaration(self):
        self.t.write("run/requested_kernel.json", json.dumps({"required_gpu_count": 4}))
        self.assertTrue(any("required_gpu_count=4" in x for x in self.problems()))

    def test_a_single_rank_kernel_with_ipc_symbols_needs_a_justification(self):
        self.t.write("kernel_src/peer.hip", "hipIpcOpenMemHandle(&p, h, 0);\n")
        self.assertTrue(any("cross-device" in x for x in self.problems()))
        m = self.t.meta()
        m["capabilities"]["single_rank_justification"] = "a copied module's unused all-reduce helper"
        m["kernel_extraction"]["binding_files"] = ["kernel_src/peer.hip"]
        self.t.write_meta(m)
        self.assertEqual(self.problems(), [])

    def test_multi_rank_declarations_need_comm_inputs_and_evidence(self):
        m = self.t.meta()
        m["capabilities"]["multi_rank"] = {"world_size": 2}
        self.t.write_meta(m)
        self.t.write("run/requested_kernel.json", json.dumps({"required_gpu_count": 2}))
        p = " ".join(self.problems())
        self.assertIn("comm", p)
        self.assertIn("inputs", p)
        self.assertIn("evidence", p)

    def test_a_cases_file_that_does_not_parse_is_a_named_problem_not_a_crash(self):
        self.t.write("cases.py", "def call(:\n")
        self.assertTrue(any("does not parse" in x for x in self.problems()))

    def test_a_disabled_control_needs_a_reason(self):
        m = self.t.meta()
        m["controls"] = {"disabled": {"stale_output": ""}}
        self.t.write_meta(m)
        self.assertTrue(any("stale_output" in x for x in self.problems()))

    def test_declared_coverage_gaps_block_the_task(self):
        m = self.t.meta()
        m["coverage"] = {"gaps": ["prefill not covered"]}
        self.t.write_meta(m)
        self.assertTrue(any("ISSUE-012" in x for x in self.problems()))


class TestKernelSrcClasses(_Case):
    """A16 C': every kernel_src/ source file is copied source with an origin or a declared binding file."""

    def setUp(self):
        super().setUp()
        gu.vendor(self.task)

    def problems(self):
        return gu.check(self.task)["problems"]

    def _bindings(self, files):
        m = self.t.meta()
        m["kernel_extraction"]["binding_files"] = files
        self.t.write_meta(m)

    def test_an_undeclared_file_written_for_the_task_is_named(self):
        self.t.write("kernel_src/sub/bind.cu", "// glue\n")
        self.t.write("kernel_src/notes.txt", "not a source\n")
        got = [x for x in self.problems() if "unclassified" in x]
        self.assertEqual(len(got), 1)
        self.assertIn("kernel_src/sub/bind.cu", got[0])
        self.assertNotIn("notes.txt", got[0])
        self._bindings(["kernel_src/sub/bind.cu"])
        self.assertEqual(self.problems(), [])

    def test_binding_files_must_exist_under_kernel_src_and_not_be_copied_source(self):
        self._bindings(["kernel_src/missing.py", "cases.py", "kernel_src/toyop.py"])
        got = self.problems()
        self.assertTrue(any("missing.py does not exist" in x for x in got))
        self.assertTrue(any("cases.py is not under kernel_src/" in x for x in got))
        self.assertTrue(any("listed both as copied source" in x for x in got))
        self._bindings("kernel_src/toyop.py")
        self.assertTrue(any("must be a list" in x for x in self.problems()))

    def test_a_new_kernel_src_file_reruns_the_check(self):
        code, rep = self.run_gen("all")
        self.assertEqual(code, 0, rep)
        self.t.write("kernel_src/late.py", "X = 1\n")
        code, rep = self.run_gen("all")
        self.assertEqual((code, rep["failed"]), (1, "check"))

    def test_the_eager_spin_settings_must_be_positive_numbers(self):
        m = self.t.meta()
        m["timing"] = {"eager_spin_us": 400, "eager_spin_factor": "x", "eager_spin_min_us": -1,
                       "prefill": "free text stays allowed"}
        self.t.write_meta(m)
        got = [x for x in self.problems() if "eager spin" in x]
        self.assertEqual(len(got), 2)
        self.assertTrue(any("eager_spin_factor" in x for x in got) and any("eager_spin_min_us" in x for x in got))


# A35 fixture (review-16 1.4): the round-2 ut_pos_qwen3_attn task's meta, reduced to what the phase check reads
# (its kernel name, served_regimes, the observed decode cases, the PREFILL calls it wrote into uncovered and its
# decode-only case manifest), and the kernel's row of the extraction's ranking (ut_pos_qwen3_moe
# round_baseline/kernel_ranking.json, rank 1) next to the split-K reduce row (rank 11), reduced.
PA = ("void paged_attention_ll4mi_QKV_mfma16_kernel<__hip_bfloat16, __hip_bfloat16, (vllm::Fp8KVCacheDataType)0, "
      "16, 128, 256, false, 8, 1, ck_tile::ComposedAttention<0u, false>, false>(__hip_bfloat16 const*, "
      "__hip_bfloat16 const*, __hip_bfloat16 const*, float, int const*, int const*, int const*, int, float const*, "
      "int, int, int, int, float*, float*, __hip_bfloat16*, float, float, float const*, float const*, float const*, "
      "ck_tile::ComposedAttention<0u, false> const*, int)")
REDUCE = ("void paged_attention_ll4mi_reduce_kernel<__hip_bfloat16, __hip_bfloat16, 128, 128, 256, 16>("
          "__hip_bfloat16*, float const*, float const*, __hip_bfloat16 const*, int const*, int const*, int, "
          "float const*)")
PA_ROW = {"rank": 1, "name": PA, "device_kernel": PA, "phase": "both", "served_regimes": ["prefill", "decode"],
          "calls": {"PREFILL": 27408, "DECODE": 2943696}, "share_of_phase_pct": {"PREFILL": 1.823, "DECODE": 46.151},
          "pct_gpu_time": 39.21}
REDUCE_ROW = {"rank": 11, "name": REDUCE, "device_kernel": REDUCE, "phase": "both",
              "served_regimes": ["prefill", "decode"], "calls": {"PREFILL": 27408, "DECODE": 2943696},
              "share_of_phase_pct": {"PREFILL": 0.074, "DECODE": 1.602}, "pct_gpu_time": 1.36}
DECODE_CASES = [("decode_bs1", 1, 154320), ("decode_bs2", 2, 497760), ("decode_bs3", 3, 1086096),
                ("decode_bs4", 4, 1205520)]
QWEN3_ATTN = {
    "device_kernel": PA,
    "served_regimes": ["decode"],
    "workload": {
        "kind": "agentx_trace_replay", "weighting": "observed", "served_regimes": ["decode"],
        "serving_weight_model": {"source": "observed_workload", "evidence_path": "kernel_workload.json"},
        "cases": [{"sig": s, "regime": "decode", "m": m, "effective_calls": n} for s, m, n in DECODE_CASES],
        "uncovered": {"PREFILL": {"calls": 27408, "mass_us": 2369109.0, "share_of_kernel_mass_pct": 0.73,
                                  "why": "decode rows of mixed prefill steps (eager, piecewise graphs exclude "
                                         "attention) are outside this decode-graph-scoped UT"}}},
    "case_manifest": {"dims": ["regime", "m"], "cases": [
        {"sig": s, "regime": "decode", "m": m, "roles": ["timing", "correctness"]} for s, m, _n in DECODE_CASES]},
}


class TestMeasuredPhases(_Case):
    """A35: every phase the kernel has calls in (ranking row, observed cases, declared uncovered calls) is served
    and has timing and correctness cases; uncovered only holds the rest of a covered phase."""

    def setUp(self):
        super().setUp()
        gu.vendor(self.task)
        self.t.write_ranking({"phases": {"PREFILL": {}, "DECODE": {}}, "ranking": [PA_ROW, REDUCE_ROW]})
        self.edit(**json.loads(json.dumps(QWEN3_ATTN)))

    def edit(self, **over):
        m = self.t.meta()
        m.update(over)
        self.t.write_meta(m)
        return m

    def problems(self):
        return gu.check(self.task)["problems"]

    def cover_prefill(self, label="prefill", **phase_regime):
        """served_regimes decode + `label`, a prefill timing and correctness case and its observed calls."""
        m = self.t.meta()
        m["served_regimes"] = m["workload"]["served_regimes"] = ["decode", label]
        m["workload"]["cases"].append({"sig": f"{label}_mixed_m4", "regime": label, "m": 4, "effective_calls": 27408})
        m["case_manifest"]["cases"].append({"sig": f"{label}_mixed_m4", "regime": label, "m": 4,
                                            "roles": ["timing", "correctness"]})
        m["workload"].pop("uncovered", None)
        if phase_regime:
            m["phase_regimes"] = {"decode": {"cuda_graph": False}, label: dict({"cuda_graph": False}, **phase_regime)}
            m.pop("regime", None)
        self.t.write_meta(m)
        return m

    def test_the_round_2_qwen3_attn_task_is_refused_for_its_prefill_calls(self):
        p = self.problems()
        phase = [x for x in p if x.startswith("phase prefill:")]
        self.assertEqual(len(phase), 1, p)
        for need in ("27408", "1.823", "no timing case", "no correctness case", "served_regimes ['decode']",
                     self.t.ranking, "row 1", "phase_regimes.<label>.phase", "Every phase the kernel has calls in"):
            self.assertIn(need, phase[0])
        self.assertTrue(any(x.startswith("meta.workload.uncovered.PREFILL declares 27408 calls") for x in p), p)
        self.assertEqual(len(p), 2, p)

    def test_serving_and_casing_the_prefill_phase_passes(self):
        self.cover_prefill()
        self.assertEqual(self.problems(), [])

    def test_a_row_without_prefill_calls_and_only_decode_cases_passes(self):
        self.t.write_ranking({"ranking": [dict(PA_ROW, calls={"PREFILL": 0, "DECODE": 2943696},
                                               share_of_phase_pct={"PREFILL": 0.0, "DECODE": 46.151})]})
        m = self.t.meta()
        m["workload"].pop("uncovered")
        self.t.write_meta(m)
        self.assertEqual(self.problems(), [])

    def test_uncovered_calls_of_a_covered_phase_pass(self):
        m = self.cover_prefill()
        m["workload"]["uncovered"] = {"DECODE": {"calls": 1000, "why": "decode steps above batch 4 (rare)"}}
        self.t.write_meta(m)
        self.assertEqual(self.problems(), [])

    def test_uncovered_calls_on_a_phase_without_cases_are_refused(self):
        # the ranking row says nothing about PREFILL here: the declared uncovered calls alone make it a phase
        self.t.write_ranking({"ranking": [dict(PA_ROW, calls={"DECODE": 2943696},
                                               share_of_phase_pct={"DECODE": 46.151})]})
        p = self.problems()
        self.assertTrue(any(x.startswith("phase prefill:") and "meta.workload.uncovered.PREFILL calls=27408" in x
                            for x in p), p)
        self.assertTrue(any(x.startswith("meta.workload.uncovered.PREFILL declares 27408 calls") for x in p), p)

    def test_zero_uncovered_calls_may_sit_on_a_phase_without_cases_with_calls_and_why(self):
        self.t.write_ranking({"ranking": [dict(PA_ROW, calls={"PREFILL": 0, "DECODE": 2943696},
                                               share_of_phase_pct={"PREFILL": 0.0, "DECODE": 46.151})]})
        m = self.t.meta()
        m["workload"]["uncovered"] = {"PREFILL": {"calls": 0, "mass_us": 0.0, "why": "not launched in PREFILL"}}
        self.t.write_meta(m)
        self.assertEqual(self.problems(), [])
        m["workload"]["uncovered"] = {"PREFILL": {"calls": 0}}
        self.t.write_meta(m)
        self.assertEqual(self.problems(), ["meta.workload.uncovered.PREFILL needs calls and why "
                                           f"({gu.PHASES_DOC})"])

    def test_an_agentx_task_needs_a_ranking_and_the_normal_mode_does_not(self):
        m = self.t.meta()
        m["kernel_extraction"].pop("ranking_json")
        self.t.write_meta(m)
        p = self.problems()
        self.assertTrue(any("no kernel ranking is given" in x and "--ranking" in x for x in p), p)
        m["kernel_extraction"]["ranking_json"] = os.path.join(self.t.root, "missing.json")
        self.t.write_meta(m)
        self.assertTrue(any("missing.json (meta.kernel_extraction.ranking_json) does not exist" in x
                            for x in self.problems()))
        m["kernel_extraction"].pop("ranking_json")
        m.pop("workload")                               # A19 normal mode: no observed workload, no phase check
        m.pop("served_regimes")
        self.t.write_meta(m)
        self.assertEqual(self.problems(), [])

    def test_a_row_not_found_is_named_and_ranking_row_finds_it(self):
        self.cover_prefill()
        other = PA.replace("ComposedAttention<0u, false>, false>", "ComposedAttention<0u, false>, true>", 1)
        self.edit(device_kernel=other)
        p = self.problems()
        self.assertEqual(len(p), 1, p)
        self.assertIn("has no row whose device_kernel, name or id is", p[0])
        self.assertIn("meta.kernel_extraction.ranking_row", p[0])
        m = self.t.meta()
        m["kernel_extraction"]["ranking_row"] = {"rank": 1, "pct_gpu_time": 39.21, "share_of_phase_pct": {}}
        self.t.write_meta(m)                            # a row's figures do not name it
        self.assertEqual(len(self.problems()), 1)
        m["kernel_extraction"]["ranking_row"] = " ".join(PA.split())     # whitespace is normalised
        self.t.write_meta(m)
        self.assertEqual(self.problems(), [])
        m["kernel_extraction"]["ranking_row"] = {"name": PA.replace(", ", ",  ")}
        self.t.write_meta(m)
        self.assertEqual(self.problems(), [])

    def test_the_kernel_name_falls_back_to_the_selected_kernels_then_the_request(self):
        self.cover_prefill()
        m = self.t.meta()
        m.pop("device_kernel")
        m["kernel_extraction"]["selected_device_kernels"] = [PA]
        self.t.write_meta(m)
        self.assertEqual(self.problems(), [])
        m["kernel_extraction"].pop("selected_device_kernels")
        self.t.write_meta(m)
        self.t.write("run/requested_kernel.json", json.dumps({"device_kernel": PA, "required_gpu_count": 1}))
        self.assertEqual(self.problems(), [])
        self.t.write("run/requested_kernel.json", json.dumps({"device_kernel": "other", "required_gpu_count": 1}))
        self.assertTrue(any("'other' (from run/requested_kernel.json device_kernel)" in x for x in self.problems()))

    def test_every_name_of_a_class_needs_a_row_and_their_phases_are_joined(self):
        # the anchor has no PREFILL calls; the reduce kernel of the same task does
        self.t.write_ranking({"ranking": [dict(PA_ROW, calls={"PREFILL": 0, "DECODE": 2943696},
                                               share_of_phase_pct={"PREFILL": 0.0, "DECODE": 46.151}), REDUCE_ROW]})
        m = self.t.meta()
        m["workload"].pop("uncovered")
        m["joint_ut"] = {"phases": [{"regime": "decode", "device_kernel": REDUCE}]}
        self.t.write_meta(m)
        p = self.problems()
        self.assertEqual(len(p), 1, p)
        self.assertTrue(p[0].startswith("phase prefill:") and "row 11" in p[0], p)
        m["joint_ut"]["phases"].append({"regime": "decode", "device_kernel": "not ranked"})
        self.t.write_meta(m)
        self.assertTrue(any("'not ranked' (from meta.device_kernel / meta.joint_ut.phases[].device_kernel)" in x
                            for x in self.problems()))

    def test_sub_labels_count_for_their_phase(self):
        # a phase split by deployment (gptoss: prefill_graph PIECEWISE, prefill_eager above the largest graph)
        for phase_regime in ({"phase": "PREFILL"}, {}):
            self.t.write_meta(self.t.meta() | {"device_kernel": PA})
            m = self.cover_prefill("prefill_graph", **phase_regime)
            m["served_regimes"].append("prefill_eager")
            m["workload"]["cases"].append({"sig": "prefill_eager_m2048", "regime": "prefill_eager", "m": 2048,
                                           "effective_calls": 100})
            m["case_manifest"]["cases"].append({"sig": "prefill_eager_m2048", "regime": "prefill_eager", "m": 2048,
                                                "roles": ["timing", "correctness"]})
            if phase_regime:
                m["phase_regimes"]["prefill_eager"] = {"cuda_graph": False, "phase": "prefill"}
            self.t.write_meta(m)
            self.assertEqual(self.problems(), [], phase_regime)
            self.setUp()

    def test_phase_regimes_phase_comes_before_the_name(self):
        # decode rows of a mixed prefill step: the ranking counts them as PREFILL; the label starts with decode_
        m = self.cover_prefill("decode_mixed")
        self.assertTrue(any(x.startswith("phase prefill:") for x in self.problems()))
        m["phase_regimes"] = {"decode": {"cuda_graph": False}, "decode_mixed": {"cuda_graph": False, "phase": "Prefill"}}
        m.pop("regime", None)
        self.t.write_meta(m)
        self.assertEqual(self.problems(), [])

    def test_a_phase_regimes_phase_may_name_several_phases(self):
        # A35 second supplement, from the round-2 glm47 gemm task: the LM head is the same M=1 eager launch in decode
        # and prefill steps, so one label `logits` whose phase is "DECODE+PREFILL" covers both phases
        for phase in ("DECODE+PREFILL", " Decode + prefill ", ["decode", "PREFILL"], ["DECODE+PREFILL"]):
            self.cover_prefill("logits", phase=phase)
            self.assertEqual(self.problems(), [], phase)
            self.setUp()
        # one phase named (or a phase that names none, like the gemm task's "DECODE (sibling branches)"): the
        # other phase is refused as before
        for phase in ("DECODE", ["decode"], "DECODE (sibling branches)", "DECODE+guard"):
            self.cover_prefill("logits", phase=phase)
            p = self.problems()
            self.assertEqual(len(p), 1, (phase, p))
            self.assertTrue(p[0].startswith("phase prefill:"), (phase, p))
            self.assertIn("joined with '+'", p[0])
            self.setUp()

    def test_label_phases_reads_one_or_several_phases(self):
        phases = {"prefill", "decode"}
        meta = {"phase_regimes": {"logits": {"phase": "DECODE+PREFILL"}, "both": {"phase": ["Decode", " prefill "]},
                                  "guard": {"phase": "DECODE (sibling branches)"}, "prefill_x": {"phase": "Decode"},
                                  "prefill_g": {"phase": "DECODE (sibling branches)"},
                                  "decode_y": {"phase": []}, "decode_z": {"phase": "  "}}}

        def lp(label):
            return gu.label_phases(meta, label, phases)
        self.assertEqual(lp("logits"), {"decode", "prefill"})
        self.assertEqual(lp("both"), {"decode", "prefill"})
        self.assertEqual(lp("guard") & phases, set())        # names no phase: in no phase, no fallback
        self.assertEqual(lp("prefill_g") & phases, set())    # ... not even to the label's prefix
        self.assertEqual(lp("prefill_x"), {"decode"})        # .phase comes before the prefix
        self.assertEqual(lp("decode_y"), {"decode"})         # an empty .phase is no .phase
        self.assertEqual(lp("decode_z"), {"decode"})
        self.assertEqual((lp("PREFILL"), lp("prefill_graph"), lp("other")), ({"prefill"}, {"prefill"}, set()))

    def test_ranking_is_written_into_meta_and_verify_ut_reads_the_same(self):
        import verify_ut as vu
        m = self.cover_prefill()
        m["kernel_extraction"].pop("ranking_json")
        self.t.write_meta(m)
        code, rep = self.run_gen("all")
        self.assertEqual(code, 1)
        self.assertTrue(any("no kernel ranking is given" in x for x in rep["problems"]))
        rel = os.path.relpath(self.t.ranking)
        code, rep = self.run_gen("all", "--ranking", rel)
        self.assertEqual(code, 0, rep)
        self.assertEqual(self.t.meta()["kernel_extraction"]["ranking_json"], os.path.abspath(rel))
        state = self.t.read_json("run/ut_generate.json")
        self.assertEqual(state["steps"]["check"]["inputs"]["<ranking>"], ut_runner.sha256_file(self.t.ranking))
        # the same path again writes nothing: every step stays current
        meta_sha = ut_runner.sha256_file(os.path.join(self.task, "meta.json"))
        code, rep = self.run_gen("all", "--ranking", self.t.ranking)
        self.assertEqual((code, rep["ran"]), (0, []))
        self.assertEqual(ut_runner.sha256_file(os.path.join(self.task, "meta.json")), meta_sha)
        report = vu.Report(self.task)
        self.assertTrue(vu.static_checks(self.task, report), report.doc["errors"])
        # a changed ranking reruns the check (and freeze after it)
        self.t.write_ranking({"ranking": [PA_ROW, REDUCE_ROW], "note": "re-profiled"})
        code, rep = self.run_gen("all")
        self.assertEqual((code, rep["kept"], rep["ran"]), (0, ["vendor"], ["check", "freeze"]))
        # another path is a changed meta.json: the test contract changes with it
        contract = self.t.read_json("run/ut_contract.json")["contract_sha256"]
        other = os.path.join(self.t.root, "ranking_copy.json")
        with open(self.t.ranking) as src, open(other, "w") as dst:
            dst.write(src.read())
        code, rep = self.run_gen("all", "--ranking", other)
        self.assertEqual(code, 0, rep)
        self.assertIn("freeze", rep["ran"])
        self.assertNotEqual(self.t.read_json("run/ut_contract.json")["contract_sha256"], contract)


class TestBoundaryCases(_Case):
    """A40: a correctness-only boundary case ("boundary": true) at an end of the observed range forms no coverage
    bucket of its own: no timing case, no observed row."""

    def setUp(self):
        super().setUp()
        gu.vendor(self.task)

    def add(self, **case):
        m = self.t.meta()
        m["case_manifest"]["cases"].append(dict({"sig": "decode_m16384_max", "regime": "decode", "m": 16384,
                                                 "roles": ["correctness"], "boundary": True}, **case))
        self.t.write_meta(m)

    def test_a_boundary_case_at_a_new_m_needs_no_timing_case(self):
        self.add()
        self.assertEqual(gu.check(self.task)["problems"], [])
        m = self.t.meta()
        m["case_manifest"]["cases"][-1].pop("boundary")
        self.t.write_meta(m)
        self.assertEqual(gu.check(self.task)["problems"],
                         ["coverage bucket [regime=decode, m=16384] has no timing case"])

    def test_a_boundary_case_with_the_timing_role_is_refused(self):
        self.add(roles=["timing", "correctness"])
        self.assertTrue(any("a boundary case (boundary: true) is a correctness case only" in x
                            for x in gu.check(self.task)["problems"]))
        m = self.t.meta()
        m["case_manifest"]["cases"][-1].update(roles=["correctness"], boundary="yes")
        self.t.write_meta(m)
        self.assertTrue(any("boundary must be true or false" in x for x in gu.check(self.task)["problems"]))

    def test_a_boundary_case_still_fills_in_every_dimension(self):
        self.add()
        m = self.t.meta()
        m["case_manifest"]["cases"][-1].pop("m")
        self.t.write_meta(m)
        self.assertEqual(gu.check(self.task)["problems"], ["case_manifest case decode_m16384_max lacks dimension(s) ['m']"])


class TestPerOutputTolerances(_Case):
    """A42: meta.tolerance.per_output is checked for its format (the UT checks it against the outputs); a comparison
    plugin is refused."""

    def setUp(self):
        super().setUp()
        gu.vendor(self.task)

    def problems(self, per_output="absent", **meta):
        m = self.t.meta()
        if per_output != "absent":
            m["tolerance"] = {"tol": 0.02, "per_output": per_output}
        m.update(meta)
        self.t.write_meta(m)
        return gu.check(self.task)["problems"]

    def test_positions_or_keys_with_finite_tolerances_pass(self):
        for per in ({"0": 0.05, "1": 0.001, "12": 0}, {"post_out": 0.001, "kv_insert": 0.07}, {}, None):
            with self.subTest(per=per):
                self.assertEqual(self.problems(per), [])

    def test_a_malformed_per_output_is_named(self):
        for per, words in (({"1": -0.1}, "must be a finite non-negative number"),
                           ({"1": True}, "must be a finite non-negative number"),
                           ({"1": "0.1"}, "must be a finite non-negative number"),
                           ({"1": float("inf")}, "must be a finite non-negative number"),
                           ({"01": 0.1}, "write an output position as plain digits"),
                           ({"-1": 0.1}, "write an output position as plain digits"),
                           ({"1.0": 0.1}, "write an output position as plain digits"),
                           ({"0": 0.1, "out": 0.2}, "names outputs both by position ['0'] and by key ['out']"),
                           ({" ": 0.1}, "must name an output"),
                           ([0.1], "must map outputs to tolerances")):
            with self.subTest(per=per):
                probs = self.problems(per)
                self.assertEqual(len(probs), 1, probs)
                self.assertIn(words, probs[0])

    def test_a_comparison_plugin_is_refused_and_per_output_named_instead(self):
        self.t.write("cmp.py", "def compare(out, ref, tol):\n    return True, 0.0\n")
        caps = dict(self.t.meta()["capabilities"], per_output_compare=True)
        probs = self.problems(capabilities=caps, compare={"file": "cmp.py", "attr": "compare"})
        self.assertEqual(len(probs), 1, probs)
        self.assertTrue(probs[0].startswith("capabilities.per_output_compare and meta.compare: a comparison plugin "
                                            "is not supported"), probs)
        self.assertIn("meta.tolerance.per_output", probs[0])
        caps["per_output_compare"] = False
        probs = self.problems(capabilities=caps)
        self.assertEqual(len(probs), 1, probs)
        self.assertTrue(probs[0].startswith("meta.compare: a comparison plugin is not supported"), probs)
        caps["per_output_compare"] = True
        m = self.t.meta()
        m.pop("compare")
        m["capabilities"] = caps
        self.t.write_meta(m)
        self.assertTrue(gu.check(self.task)["problems"][0].startswith("capabilities.per_output_compare: a comparison"))
        caps["per_output_compare"] = False                # the delivered tasks' declaration: passes
        self.assertEqual(self.problems(capabilities=caps, compare={}), [])


class TestFreeze(_Case):
    def test_the_contract_covers_the_test_but_not_the_candidate(self):
        code, rep = self.run_gen("all")
        self.assertEqual(code, 0, rep)
        doc = self.t.read_json("run/ut_contract.json")
        paths = [f["path"] for f in doc["files"]]
        for need in ("unittest.py", "cases.py", "meta.json", *ut_runner.RUNTIME_FILES):
            self.assertIn(need, paths)
        self.assertFalse(any(p.startswith(("kernel_src", "run")) for p in paths))
        self.assertEqual(doc["kernel_src_origin"]["kernel_src/toyop.py"]["sha256"],
                         ut_runner.sha256_file(os.path.join(self.t.site, "toyop.py")))
        self.assertEqual(doc["binding_files"], {})
        sha, problems = ut_runner.check_contract(self.task)
        self.assertEqual((sha, problems), (doc["contract_sha256"], []))
        # editing only the candidate keeps the contract; editing the test breaks it
        self.t.write("kernel_src/toyop.py", "def op(x, w):\n    return x @ w\n")
        self.assertEqual(ut_runner.check_contract(self.task)[1], [])
        self.t.write("cases.py", open(os.path.join(self.task, "cases.py")).read() + "\n# edited\n")
        self.assertTrue(any("cases.py" in p for p in ut_runner.check_contract(self.task)[1]))


class TestStepInvalidation(_Case):
    def test_all_reruns_only_what_changed_and_every_later_step(self):
        code, rep = self.run_gen("all")
        self.assertEqual(rep["ran"], ["vendor", "check", "freeze"])
        code, rep = self.run_gen("all")
        self.assertEqual((rep["ran"], rep["kept"]), ([], ["vendor", "check", "freeze"]))
        self.t.write("cases.py", open(os.path.join(self.task, "cases.py")).read() + "\n# edit\n")
        code, rep = self.run_gen("all")
        self.assertEqual((rep["kept"], rep["ran"]), (["vendor"], ["check", "freeze"]))
        # a tampered runtime file reruns vendor (which restores it) and everything after it
        with open(os.path.join(self.task, "harness_lib.py"), "a") as fh:
            fh.write("# tampered\n")
        code, rep = self.run_gen("all")
        self.assertEqual(rep["ran"], ["vendor", "check", "freeze"])
        self.assertEqual(ut_runner.sha256_file(os.path.join(self.task, "harness_lib.py")),
                         ut_runner.sha256_file(os.path.join(SCRIPTS, "harness_lib.py")))

    def test_a_failing_check_drops_the_later_records(self):
        self.run_gen("all")
        self.t.write("cases.py", "def call(args):\n    return args\n")
        code, rep = self.run_gen("all")
        self.assertEqual(code, 1)
        self.assertEqual(rep["failed"], "check")
        state = self.t.read_json("run/ut_generate.json")
        self.assertNotIn("freeze", state["steps"])

    def test_running_one_step_alone_invalidates_the_later_ones(self):
        self.run_gen("all")
        self.run_gen("check")
        state = self.t.read_json("run/ut_generate.json")
        self.assertNotIn("freeze", state["steps"])


class TestMetaFields(_Case):
    """check_meta names each malformed field of meta.json (ISSUE-014: a named problem, never a crash). Every case
    starts from the vendored toy meta, which passes."""

    def setUp(self):
        super().setUp()
        gu.vendor(self.task)
        self.base = self.t.meta()

    def problems(self, drop=(), **over):
        m = json.loads(json.dumps(self.base))
        for k in drop:
            m.pop(k, None)
        m.update(over)
        self.t.write_meta(m)
        return gu.check(self.task)["problems"]

    def test_the_target_and_the_candidate_binding(self):
        self.assertEqual(self.problems(target_callable="toyop.op"),
                         ["meta.target_callable must be an exact 'module:attr' spec"])
        self.assertEqual(self.problems(candidate_bind={"kind": "rebind", "file": "kernel_src/toyop.py"}),
                         [f"meta.candidate_bind (rebind) lacks {k}" for k in ("target", "impl_module", "impl_attr")])
        self.assertEqual(self.problems(candidate_bind={"kind": "module"}),
                         ["meta.candidate_bind (module) lacks module", "meta.candidate_bind (module) lacks file"])
        p = self.problems(candidate_bind={})
        self.assertEqual(len(p), 1, p)
        self.assertTrue(p[0].startswith("meta.candidate_bind must be {kind: module, module, file} or {kind: rebind"), p)
        self.assertEqual(self.problems(candidate_bind={"kind": "module", "module": "toyop", "file": "cases.py"}),
                         ["meta.candidate_bind.file must live under kernel_src/"])
        self.assertEqual(self.problems(candidate_bind={"module": "toyop", "file": "kernel_src/gone.py"}),
                         ["meta.candidate_bind.file kernel_src/gone.py does not exist"])

    def test_the_tolerance_must_be_a_finite_number(self):
        for tol in ("0.01", True, float("inf"), -0.01):
            self.assertEqual(self.problems(tolerance={"tol": tol}),
                             ["meta.tolerance.tol must be a finite non-negative number"], tol)
        # an integer too large for a float is a named problem too, not a crash of the check
        self.assertEqual(self.problems(tolerance={"tol": 10 ** 400}), ["meta.tolerance: int too large to convert to float"])

    def test_the_regime_and_the_workload(self):
        self.assertIn("meta.phase_regimes must map every phase label to a regime object", self.problems(phase_regimes=[]))
        self.assertIn("meta.regime (or meta.phase_regimes) is required: the deployed graph/compile settings",
                      self.problems(drop=("regime",)))
        self.assertIn("meta.workload, when given, needs cases (observed case model with sig, regime, effective_calls)",
                      self.problems(workload=dict(self.base["workload"], cases=[])))

    def test_the_capabilities_are_required_and_typed(self):
        self.assertEqual(self.problems(drop=("capabilities",)),
                         ["meta.capabilities is required: {stateful, multi_rank: {world_size, ...}}",
                          "meta.capabilities.stateful must be true or false",
                          "meta.capabilities.multi_rank.world_size must be a positive integer (1 for a single-GPU kernel)"])
        caps = self.base["capabilities"]
        self.assertEqual(self.problems(capabilities=dict(caps, stateful=None)),
                         ["meta.capabilities.stateful must be true or false"])
        for mr in ({"world_size": True}, {"world_size": 0}, {"world_size": "2"}, 2):
            self.assertEqual(self.problems(capabilities=dict(caps, multi_rank=mr)),
                             [("meta.capabilities.multi_rank.world_size must be a positive integer (1 for a single-GPU "
                               "kernel)")], mr)

    def test_a_multi_rank_task_is_refused_where_multi_rank_execution_is_not_available(self):
        caps = dict(self.base["capabilities"], multi_rank={
            "world_size": 2, "comm": "collective_api", "inputs": {"x": "per_rank", "w": "replicated"},
            "evidence": {"serving_tp": 2}})
        timeouts = {"setup_s": 60, "command_s": 600, "total_s": 900}
        self.t.write("run/requested_kernel.json", json.dumps({"device_kernel": "toy", "required_gpu_count": 2}))
        self.assertEqual(self.problems(capabilities=caps, timeouts=timeouts), [])
        with mock.patch.object(gu, "MULTI_RANK_SUPPORTED", False):
            p = self.problems(capabilities=caps, timeouts=timeouts)
        self.assertEqual(len(p), 1, p)
        self.assertTrue(p[0].startswith("world_size > 1: multi-rank execution is not available in this GEAK version"))

    def test_the_timeouts(self):
        for tm in ({"setup_s": 60, "wall_s": 10}, [600]):
            self.assertEqual(self.problems(timeouts=tm), ["meta.timeouts may only hold setup_s, command_s, total_s"], tm)
        self.assertEqual(self.problems(timeouts={"setup_s": -1, "command_s": "600"}),
                         ["meta.timeouts.setup_s must be a positive number of seconds",
                          "meta.timeouts.command_s must be a positive number of seconds"])
        # 0 would give a rank pool a 1 s setup wait and a 0 s gloo timeout; total_s 0 is no whole-UT limit
        self.assertEqual(self.problems(timeouts={"setup_s": 0, "command_s": 0, "total_s": 0}),
                         ["meta.timeouts.setup_s must be a positive number of seconds",
                          "meta.timeouts.command_s must be a positive number of seconds"])
        self.assertEqual(self.problems(timeouts={"total_s": -5}),
                         ["meta.timeouts.total_s must be a non-negative number of seconds (0: no whole-UT limit)"])
        self.assertEqual(self.problems(timeouts={"command_s": 900, "total_s": 600}),
                         ["meta.timeouts.command_s exceeds total_s"])
        self.assertEqual(self.problems(timeouts={"setup_s": 60, "command_s": 600, "total_s": 900}), [])

    def test_task_controls_need_a_file_and_a_known_expectation(self):
        self.t.write("controls/drop_bias.py", "# a task control\n")
        ctl = {"name": "drop_bias", "file": "controls/drop_bias.py"}
        self.assertEqual(self.problems(controls={"task": [{"name": "no_file"}, "drop_bias", dict(ctl, expect="pass")]}),
                         ["meta.controls.task entry {'name': 'no_file'} needs name and an existing file",
                          "meta.controls.task entry 'drop_bias' needs name and an existing file",
                          "task control drop_bias: expect must be 'reject' or 'below_tol'"])
        self.assertEqual(self.problems(controls={"task": [dict(ctl, expect="below_tol"), dict(ctl, name="other")]}), [])

    def test_the_graph_settings(self):
        for v in (0, True, 1.5):
            self.assertEqual(self.problems(timing={"graph_inner_max_factor": v}),
                             [(f"meta.timing.graph_inner_max_factor must be a positive integer (A20: how far the UT may "
                               f"raise a host-dominated graph case's inner), got {v!r}")], v)
        self.assertEqual(self.problems(timing={"graph_inner_max_factor": 4}), [])
        # a manifest the deployment-graph reader cannot read is named, not a crash
        p = self.problems(case_manifest=["decode_m1"])
        self.assertIn("graph capture declaration could not be checked: AttributeError: 'list' object has no attribute "
                      "'get'", p)
        self.assertIn("meta.case_manifest {dims: [...], cases: [{sig, <dims>, roles: [timing|correctness]}]} is required", p)

    def test_source_origins_must_be_a_map(self):
        ke = dict(self.base["kernel_extraction"], source_origins=["kernel_src/toyop.py"])
        p = self.problems(kernel_extraction=ke)
        self.assertEqual(p[0], "meta.kernel_extraction.source_origins must map kernel_src/ paths to the installed files "
                               "they were copied from")
        self.assertTrue(p[1].endswith("unclassified: kernel_src/toyop.py"), p)
        self.assertEqual(len(p), 2, p)

    def test_the_shape_of_the_case_manifest(self):
        man = self.base["case_manifest"]
        self.assertIn("meta.case_manifest {dims: [...], cases: [{sig, <dims>, roles: [timing|correctness]}]} is required",
                      self.problems(drop=("case_manifest",)))
        self.assertEqual(self.problems(case_manifest=dict(man, dims="regime")),
                         ["meta.case_manifest.dims must be a list of field names"])
        cases = [*man["cases"], "decode_m8", {"roles": ["timing"]}, dict(man["cases"][0]),
                 dict(man["cases"][1], sig="decode_m4b", roles=["timing", "bench"])]
        self.assertEqual(self.problems(case_manifest=dict(man, cases=cases)),
                         ["case_manifest case without sig: 'decode_m8'",
                          "case_manifest case without sig: {'roles': ['timing']}",
                          "case_manifest has duplicate sig decode_m1",
                          "case_manifest case decode_m4b: roles must be a non-empty subset of [timing, correctness]"])

    def test_an_observed_case_without_a_numeric_call_mass_has_no_positive_mass(self):
        wl = self.base["workload"]
        extra = {"sig": "decode_m2", "regime": "decode", "m": 2}
        self.assertEqual(self.problems(workload=dict(wl, cases=[*wl["cases"], dict(extra, effective_calls="lots")])), [])
        self.assertEqual(self.problems(workload=dict(wl, cases=[*wl["cases"], dict(extra, effective_calls=5)])),
                         ["observed case decode_m2 has positive call mass but no timing case in the manifest"])
        self.assertEqual([gu._positive(x) for x in (True, "lots", None, [1], "3", 0, -1, 2.5)],
                         [False, False, False, False, True, False, False, True])

    def test_a_meta_the_checks_cannot_read_is_a_named_problem(self):
        self.assertEqual(self.problems(controls=["noop"]),
                         ["meta.json could not be checked: AttributeError: 'list' object has no attribute 'get'"])


class TestCheckInputs(_Case):
    """The files check() and vendor() read: meta.json, unittest.py, cases.py, run/requested_kernel.json, kernel_src/."""

    def test_vendor_needs_meta_and_a_missing_unittest_is_not_generated(self):
        self.assertFalse(gu.is_generated(self.task))
        os.remove(os.path.join(self.task, "meta.json"))
        with self.assertRaises(SystemExit) as cm:
            gu.vendor(self.task)
        self.assertEqual(str(cm.exception), "vendor: write meta.json first (task-authoring.md)")
        self.assertFalse(os.path.exists(os.path.join(self.task, "unittest.py")))
        self.assertEqual(gu.check(self.task), {"ok": False, "problems": ["meta.json missing"], "warnings": []})

    def test_a_missing_or_undecodable_cases_file_is_named(self):
        gu.vendor(self.task)
        os.remove(os.path.join(self.task, "cases.py"))
        self.assertEqual(gu.check(self.task)["problems"], ["cases.py missing"])
        with open(os.path.join(self.task, "cases.py"), "wb") as fh:
            fh.write(b"def call(args):\n    return '\xff\xfe'\n")
        p = gu.check(self.task)["problems"]
        self.assertEqual(len(p), 1, p)
        self.assertTrue(p[0].startswith("cases.py could not be read: UnicodeDecodeError: "), p)

    def test_a_multi_rank_task_warns_until_its_gpu_count_is_written(self):
        from ut_task_fixture import RankToyTask
        t = RankToyTask()
        self.addCleanup(t.cleanup)
        gu.vendor(t.task)
        self.assertEqual(gu.check(t.task), {"ok": True, "problems": [], "warnings": [], "world_size": 2})
        t.write("run/requested_kernel.json", json.dumps({"device_kernel": "toy"}))
        self.assertEqual(gu.check(t.task)["warnings"],
                         [("run/requested_kernel.json writes no required_gpu_count: RETURN.json must declare "
                           "required_gpu_count=2 (world_size 2)")])
        os.remove(os.path.join(t.task, "run", "requested_kernel.json"))
        res = gu.check(t.task)
        self.assertEqual((res["ok"], res["warnings"]),
                         (True, ["run/requested_kernel.json not found: the GPU-count declaration is checked at delivery"]))

    def test_the_cross_rank_scan_skips_a_file_it_cannot_open(self):
        self.t.write("kernel_src/peer.hip", "hipIpcOpenMemHandle(&p, h, 0);\n")
        os.symlink(os.path.join(self.t.root, "nowhere.cu"), os.path.join(self.task, "kernel_src", "gone.cu"))
        self.assertEqual(gu.scan_cross_rank(self.task), [("kernel_src/peer.hip", 1, "hipIpcOpenMemHandle")])

    def test_hooks_are_read_from_the_syntax_tree_and_eager_regimes_need_no_replay(self):
        hooks = gu._hooks("import os.path as osp\nfrom x import y as z\nclass Group:\n    pass\nN: int = 3\nA = B = 1\n"
                          "def call(args, extra=None, /):\n    pass\nasync def replay(h, meta, phase=None):\n    pass\n")
        self.assertEqual(hooks, {"osp": {"kind": "import"}, "z": {"kind": "import"}, "Group": {"kind": "class"},
                                 "N": {"kind": "assign"}, "A": {"kind": "assign"}, "B": {"kind": "assign"},
                                 "call": {"kind": "def", "positional": 2, "required": 1},
                                 "replay": {"kind": "def", "positional": 3, "required": 2}})
        self.assertEqual([gu._graph(r) for r in (None, {}, {"cuda_graph": False}, {"cuda_graph": True, "enforce_eager": True})],
                         [True, True, False, False])


class TestRankingRows(_Case):
    """A35: how the kernel's row of the extraction's ranking is found and read."""

    def setUp(self):
        super().setUp()
        gu.vendor(self.task)

    def test_set_ranking_writes_the_path_only_into_a_meta_object(self):
        path = os.path.join(self.t.root, "ranking.json")
        m = self.t.meta()
        m.pop("kernel_extraction")
        self.t.write_meta(m)
        self.assertTrue(gu.set_ranking(self.task, path))
        self.assertEqual(self.t.meta()["kernel_extraction"], {"ranking_json": path})
        self.assertFalse(gu.set_ranking(self.task, path))                  # the same path: meta.json is not rewritten
        self.t.write("meta.json", "[]")
        self.assertFalse(gu.set_ranking(self.task, path))
        with open(os.path.join(self.task, "meta.json")) as fh:
            self.assertEqual(fh.read(), "[]")

    def test_without_any_kernel_name_the_row_cannot_be_found(self):
        m = self.t.meta()
        m.pop("device_kernel")
        self.t.write_meta(m)
        self.t.write("run/requested_kernel.json", json.dumps({"required_gpu_count": 1}))
        self.assertEqual(gu.ranking_identities(self.task, self.t.meta()), (None, []))
        self.assertEqual(gu.check(self.task)["problems"],
                         [(f"no kernel name to find this kernel's row of the ranking {self.t.ranking}: write "
                           "meta.device_kernel, or name the row with meta.kernel_extraction.ranking_row (its id or name)")])

    def test_a_ranking_that_cannot_be_read_or_has_no_rows_is_named(self):
        with open(self.t.ranking, "w") as fh:
            fh.write("{not json")
        p = gu.check(self.task)["problems"]
        self.assertTrue(any(x.startswith(f"kernel ranking {self.t.ranking} (meta.kernel_extraction.ranking_json) cannot "
                                         "be read: JSONDecodeError: ") for x in p), p)
        self.t.write_ranking({"rows": []})
        self.assertIn(f"kernel ranking {self.t.ranking} has no `ranking` list of kernel rows", gu.check(self.task)["problems"])

    def test_a_row_is_matched_on_its_id_or_names(self):
        self.assertFalse(gu._row_matches("toy", {"name": "toy"}))
        self.assertTrue(gu._row_matches({"id": "k  7", "name": "x"}, {"id": "k 7", "name": "other"}))
        self.assertFalse(gu._row_matches({"id": "k7", "name": "toy"}, {"id": "k8"}))      # an id alone names no name
        self.assertTrue(gu._row_matches({"id": "toy", "name": "row 3"}, {"name": "toy"}))  # a kernel name as the row id
        self.assertFalse(gu._row_matches({"id": "toy"}, {"id": "k8", "name": "toy"}))     # a named id is not a name
        self.t.write_ranking({"phases": {}, "ranking": [{"id": "r7", "name": "toy kernel", "calls": {"DECODE": 40}}]})
        toy = self.t.meta()
        toy["kernel_extraction"]["ranking_row"] = {"id": "r7", "rank": 1}
        self.t.write_meta(toy)
        self.assertEqual(gu.check(self.task)["problems"], [])

    def test_the_phases_of_a_ranking_and_the_uncovered_entries(self):
        self.assertEqual(gu.ranking_phases({"phases": ["PREFILL", " Decode ", 3]}, [{"calls": {"MIXED": 1}}]),
                         {"prefill", "decode", "mixed"})
        p = gu.Problems()
        entries = [{"phase": "PREFILL", "calls": 5, "why": "mixed steps"}, {"regime": "decode", "calls": 0, "why": "-"}]
        self.assertEqual(gu._uncovered_entries({"uncovered": entries}, p), [("PREFILL", entries[0]), ("decode", entries[1])])
        self.assertEqual(p, [])
        self.assertEqual(gu._uncovered_entries({"uncovered": ["PREFILL"]}, p), [])
        self.assertEqual(p, [("meta.workload.uncovered must map each phase to {calls, why} (the calls of a covered phase "
                              f"this UT does not time, with the reason; {gu.PHASES_DOC})")])


class TestStepRecords(_Case):
    """run/ut_generate.json: each step's recorded inputs, and a changed shared runtime re-vendoring the task."""

    def test_the_inputs_of_freeze_and_vendor_are_what_those_steps_record(self):
        code, rep = self.run_gen("all")
        self.assertEqual(code, 0, rep)
        self.assertEqual(gu._step_inputs(self.task, "freeze"), gu.freeze(self.task)["inputs"])
        self.assertEqual(gu._step_inputs(self.task, "vendor"), gu.vendor(self.task)["inputs"])
        m = self.t.meta()
        m["ut_contract_files"] = ["kernel_src/toyop.py"]                  # not a contract file: nothing to list
        self.t.write_meta(m)
        self.assertEqual(gu._step_inputs(self.task, "freeze"), {"<unlisted>": None})

    def test_a_changed_shared_runtime_reruns_vendor_and_every_later_step(self):
        self.assertEqual(self.run_gen("all")[0], 0)
        state = self.t.read_json("run/ut_generate.json")
        state["steps"]["vendor"]["inputs"]["<shared>/ut_runner.py"] = "0" * 64      # vendored from another GEAK
        self.t.write("run/ut_generate.json", json.dumps(state))
        code, rep = self.run_gen("all")
        self.assertEqual((code, rep["ran"], rep["kept"]), (0, ["vendor", "check", "freeze"], []))
        state = self.t.read_json("run/ut_generate.json")
        self.assertEqual(state["steps"]["vendor"]["inputs"]["<shared>/ut_runner.py"],
                         ut_runner.sha256_file(os.path.join(SCRIPTS, "ut_runner.py")))


class TestMalformedMetaIsANamedProblem(_Case):
    """ISSUE-014 holds past check_meta too: a capabilities or regime that is not an object, or a meta.json that is
    not an object at all, is a named problem -- check() never dies with an AttributeError, and a failing check
    still records run/ut_check.json and makes the `check` step exit 1 with the problems."""

    def setUp(self):
        super().setUp()
        gu.vendor(self.task)
        self.base = self.t.meta()

    def check_with(self, **over):
        m = json.loads(json.dumps(self.base))
        m.update(over)
        self.t.write_meta(m)
        out = gu.check(self.task)
        self.assertFalse(out["ok"], out)
        self.assertEqual(self.t.read_json("run/ut_check.json"), out)
        return out["problems"]

    def test_capabilities_that_are_not_an_object(self):
        problems = self.check_with(capabilities="single")
        self.assertEqual(problems[0], "meta.capabilities is required: {stateful, multi_rank: {world_size, ...}}")
        self.assertFalse([p for p in problems if "sequence_cases" in p], problems)   # not read as stateful
        code, rep = self.run_gen("check")
        self.assertEqual((code, rep["ok"], rep["failed"], rep["problems"]), (1, False, "check", problems))

    def test_a_regime_that_is_not_an_object(self):
        self.assertEqual(self.check_with(regime="eager")[0],
                         "meta.regime (or meta.phase_regimes) is required: the deployed graph/compile settings")
        self.assertEqual(self.check_with(phase_regimes={"decode": "eager"}),
                         ["meta.phase_regimes must map every phase label to a regime object"])

    def test_a_meta_json_that_is_not_an_object(self):
        self.t.write("meta.json", json.dumps([1, 2]))
        self.assertEqual(gu.check(self.task),
                         {"ok": False, "problems": ["meta.json must be a JSON object"], "warnings": []})
        code, rep = self.run_gen("check")
        self.assertEqual((code, rep["failed"], rep["problems"]), (1, "check", ["meta.json must be a JSON object"]))

    def test_a_check_that_fails_on_meta_json_itself_replaces_the_last_report(self):
        """run/ut_check.json is where check names its problems (task-authoring.md): a meta.json that turns into an
        array, or goes missing, after a passing check must leave this run's failure there, never the earlier pass."""
        for label, breaks, problem in (
                ("array", lambda: self.t.write("meta.json", json.dumps([1, 2])), "meta.json must be a JSON object"),
                ("missing", lambda: os.remove(os.path.join(self.task, "meta.json")), "meta.json missing")):
            with self.subTest(meta=label):
                self.t.write_meta(self.base)
                passed = gu.check(self.task)
                self.assertTrue(passed["ok"], passed)
                self.assertEqual(self.t.read_json("run/ut_check.json"), passed)
                breaks()
                failed = gu.check(self.task)
                self.assertEqual(failed, {"ok": False, "problems": [problem], "warnings": []})
                self.assertEqual(self.t.read_json("run/ut_check.json"), failed)
                code, rep = self.run_gen("check")
                self.assertEqual((code, rep["failed"], rep["problems"]), (1, "check", [problem]))
                self.assertEqual(self.t.read_json("run/ut_check.json"), failed)
                state = self.t.read_json("run/ut_generate.json")["steps"]["check"]
                self.assertEqual((state["ok"], state["outputs"]["run/ut_check.json"]),
                                 (False, gu._digest(self.task, "run/ut_check.json")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
