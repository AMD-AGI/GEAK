#!/usr/bin/env python3
"""Unit tests for probe_postprocess.py -- per-pid probe files -> per_shape_probe.{json,md} (stdlib).

This is the merge step between the in-server probe and every human/agent that reads the shape
distribution. Two things it produces are load-bearing and easy to get quietly wrong:

  - the SHAPE HISTOGRAM: vLLM runs APIServer and EngineCore as separate processes, so a shape's real
    traffic weight only exists after per-pid files are summed. A dedup key that ignored dtypes, or a
    sum that dropped a pid, understates exactly the buckets tuning should target.
  - the DATA-SEMANTICS block: the same JSON means completely different things depending on whether
    the probe ran under CUDA graph (counts are capture-phase artefacts, NOT traffic) or enforce-eager
    (counts and cuda.Event latencies are real). Mislabelling that turns a "which shapes exist" report
    into a fake traffic-weight table, which is how a run tunes the wrong bucket.

Also pinned: the profile join is name-based with NO hard-coded kernel map (the stripped-token variant
is what lets a Python launcher like `invoke_fused_moe_triton_kernel` reach the profiled
`fused_moe_kernel`'s %GPU), and %GPU renders without crashing in all three of its states -- float,
the "unknown" sentinel (no profile), and None (profile given but no name match).

Run: python3 -m pytest e2e_workflow/scripts/tests/test_probe_postprocess.py -v
"""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(mod_name, filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pp = _load("probe_postprocess", "probe_postprocess.py")

TARGET = "triton_kernels.matmul_ogs:matmul_ogs"


def _pid_file(dims, count, dtypes=("torch.bfloat16",), labels=("arg0",),
              gpu_us_avg=None, timed_count=0):
    case = {"dims": dims, "dtypes": list(dtypes), "arg_labels": list(labels), "count": count}
    if gpu_us_avg is not None:
        case["gpu_us_avg"] = gpu_us_avg
        case["timed_count"] = timed_count
    return case


class _PostprocessTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="probe_postprocess_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.probe_dir = os.path.join(self.tmp, "probe")
        os.makedirs(self.probe_dir)
        self._argv = sys.argv
        self.addCleanup(setattr, sys, "argv", self._argv)

    def _write_pid(self, pid, cases, target=TARGET, total_calls=None, timing=False):
        payload = {"target": target, "pid": pid, "timing": timing,
                   "total_calls": total_calls if total_calls is not None
                   else sum(c["count"] for c in cases),
                   "num_distinct_shapes": len(cases), "cases": cases}
        with open(os.path.join(self.probe_dir, f"probe_{pid}_x.json"), "w") as fh:
            json.dump(payload, fh)

    def _write_profile(self, top_kernels):
        path = os.path.join(self.tmp, "profile_topN.json")
        with open(path, "w") as fh:
            json.dump({"top_kernels": top_kernels}, fh)
        return path

    def _build(self, profile="", workload=None):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            summ = pp.build(self.probe_dir, profile, workload=workload)
        self.warned = err.getvalue()
        return summ


# --------------------------------------------------------------------------- #
# Name derivation -- how a target string joins to a profiled kernel symbol
# --------------------------------------------------------------------------- #
class NameDerivation(unittest.TestCase):
    def test_label_is_the_attr(self):
        self.assertEqual(pp.label_from_target(TARGET), "matmul_ogs")

    def test_label_of_a_target_without_a_colon_is_the_whole_string(self):
        self.assertEqual(pp.label_from_target("fused_moe_kernel"), "fused_moe_kernel")

    def test_substrs_cover_attr_and_leaf_module(self):
        self.assertEqual(pp.match_substrs_from_target(TARGET), ["matmul_ogs"])
        self.assertEqual(
            pp.match_substrs_from_target("aiter.ops.triton.unified_attention:unified_attention"),
            ["unified_attention"])

    def test_launcher_name_strips_wrapper_and_backend_tokens(self):
        """invoke_fused_moe_triton_kernel must reach the profiled fused_moe_kernel."""
        subs = pp.match_substrs_from_target("vllm.moe:invoke_fused_moe_triton_kernel")
        self.assertEqual(subs, ["invoke_fused_moe_triton_kernel", "moe",
                                "fused_moe_kernel", "fused_moe"])

    def test_a_bare_module_target_falls_back_to_the_leaf(self):
        self.assertEqual(pp.match_substrs_from_target("pkg.sub.gemm"), ["gemm"])

    def test_an_empty_target_returns_itself_rather_than_an_empty_matcher(self):
        """An empty substr list would match EVERY profiled kernel."""
        self.assertEqual(pp.match_substrs_from_target(""), [""])

    def test_match_profile_is_case_insensitive_over_name_and_short_name(self):
        top = [{"name": "void _MatMul_OGS_kernel<...>", "short_name": ""},
               {"name": "", "short_name": "unified_attention"},
               {"name": "elementwise_add", "short_name": "add"}]
        hits = pp.match_profile(top, ["matmul_ogs"])
        self.assertEqual([k["name"] for k in hits], ["void _MatMul_OGS_kernel<...>"])


# --------------------------------------------------------------------------- #
# load_probe -- the cross-pid merge
# --------------------------------------------------------------------------- #
class LoadProbe(_PostprocessTestCase):
    def test_same_shape_across_pids_is_one_summed_case(self):
        """APIServer + EngineCore each flush; the traffic weight only exists after the sum."""
        self._write_pid(11, [_pid_file([[64, 1024]], 300)])
        self._write_pid(22, [_pid_file([[64, 1024]], 700)])
        merged = pp.load_probe(self.probe_dir)
        cases = list(merged[TARGET]["cases"].values())
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["count"], 1000)
        self.assertEqual(merged[TARGET]["total_calls"], 1000)

    def test_same_dims_with_different_dtypes_stay_distinct(self):
        """bf16 and fp8 at one shape are different kernels to tune; merging them hides one."""
        self._write_pid(11, [_pid_file([[64, 1024]], 10, dtypes=("torch.bfloat16",))])
        self._write_pid(22, [_pid_file([[64, 1024]], 10, dtypes=("torch.float8_e4m3fn",))])
        self.assertEqual(len(pp.load_probe(self.probe_dir)[TARGET]["cases"]), 2)

    def test_different_targets_are_kept_apart(self):
        self._write_pid(11, [_pid_file([[64, 1024]], 10)])
        self._write_pid(12, [_pid_file([[1, 8, 128]], 10)], target="aiter:unified_attention")
        self.assertEqual(sorted(pp.load_probe(self.probe_dir)),
                         sorted([TARGET, "aiter:unified_attention"]))

    def test_measured_latency_is_averaged_weighted_by_timed_count(self):
        self._write_pid(11, [_pid_file([[64, 1024]], 100, gpu_us_avg=10.0, timed_count=90)])
        self._write_pid(22, [_pid_file([[64, 1024]], 100, gpu_us_avg=20.0, timed_count=10)])
        c = list(pp.load_probe(self.probe_dir)[TARGET]["cases"].values())[0]
        self.assertEqual(c["_timed_count"], 100)
        self.assertAlmostEqual(c["_gpu_us_weighted"] / c["_timed_count"], 11.0)

    def test_a_gpu_avg_with_no_timed_count_is_not_counted(self):
        self._write_pid(11, [_pid_file([[64, 1024]], 5, gpu_us_avg=10.0, timed_count=0)])
        c = list(pp.load_probe(self.probe_dir)[TARGET]["cases"].values())[0]
        self.assertEqual(c["_timed_count"], 0)

    def test_an_empty_probe_dir_merges_to_nothing(self):
        self.assertEqual(pp.load_probe(self.probe_dir), {})


# --------------------------------------------------------------------------- #
# build -- the report body
# --------------------------------------------------------------------------- #
class BuildWithProfile(_PostprocessTestCase):
    def test_pct_gpu_and_avg_us_are_joined_from_matched_call_sites(self):
        self._write_pid(11, [_pid_file([[64, 1024]], 100)])
        prof = self._write_profile([
            {"name": "_matmul_ogs", "calls": 300, "avg_us": 10.0, "pct_gpu_time": 40.0},
            {"name": "_matmul_ogs_epilogue", "calls": 100, "avg_us": 30.0, "pct_gpu_time": 5.0},
            {"name": "unrelated_kernel", "calls": 999, "avg_us": 1.0, "pct_gpu_time": 50.0},
        ])
        k = self._build(prof)["kernels"][0]
        self.assertEqual(k["pct_gpu"], 45.0)              # summed over matched call sites
        self.assertEqual(k["kernel_avg_us"], 15.0)        # calls-weighted, not arithmetic
        self.assertEqual(k["profile_calls"], 400)

    def test_unmatched_kernel_reports_none_not_unknown(self):
        """A profile WAS supplied and simply did not name this kernel -- distinct from 'no profile'."""
        self._write_pid(11, [_pid_file([[64, 1024]], 100)])
        prof = self._write_profile([{"name": "something_else", "calls": 1, "avg_us": 1.0,
                                     "pct_gpu_time": 1.0}])
        k = self._build(prof)["kernels"][0]
        self.assertIsNone(k["pct_gpu"])
        self.assertIsNone(k["kernel_avg_us"])
        self.assertIsNone(k["cases"][0]["latency_us_approx"])

    def test_a_missing_profile_path_is_treated_as_no_profile(self):
        self._write_pid(11, [_pid_file([[64, 1024]], 100)])
        summ = self._build(os.path.join(self.tmp, "nope.json"))
        self.assertEqual(summ["kernels"][0]["pct_gpu"], pp.GPU_UNKNOWN)
        self.assertIsNone(summ["profile_source"])
        self.assertIn("WARNING", self.warned)

    def test_no_profile_warns_loudly_and_marks_gpu_unknown(self):
        self._write_pid(11, [_pid_file([[64, 1024]], 100)])
        summ = self._build("")
        self.assertEqual(summ["kernels"][0]["pct_gpu"], pp.GPU_UNKNOWN)
        self.assertIn("%GPU is UNKNOWN", self.warned)

    def test_approx_latency_is_the_profile_average_spread_across_shapes(self):
        self._write_pid(11, [_pid_file([[64, 1024]], 90), _pid_file([[4096, 1024]], 10)])
        prof = self._write_profile([{"name": "_matmul_ogs", "calls": 100, "avg_us": 12.3456,
                                     "pct_gpu_time": 40.0}])
        cases = self._build(prof)["kernels"][0]["cases"]
        self.assertTrue(all(c["latency_us_approx"] == 12.346 for c in cases))
        self.assertNotIn("latency_us_measured", cases[0])


class BuildSemantics(_PostprocessTestCase):
    def test_cases_are_sorted_by_count_with_fractions(self):
        self._write_pid(11, [_pid_file([[1, 1024]], 25), _pid_file([[64, 1024]], 75)])
        cases = self._build()["kernels"][0]["cases"]
        self.assertEqual([c["dims"] for c in cases], [[[64, 1024]], [[1, 1024]]])
        self.assertEqual([c["count_frac"] for c in cases], [0.75, 0.25])

    def test_measured_latency_flips_the_semantics_block_to_enforce_eager(self):
        """With cuda.Event samples present, counts ARE the steady-state traffic weight."""
        self._write_pid(11, [_pid_file([[64, 1024]], 100, gpu_us_avg=8.0, timed_count=99)],
                        timing=True)
        summ = self._build()
        k = summ["kernels"][0]
        self.assertEqual(k["cases"][0]["latency_us_measured"], 8.0)
        self.assertEqual(k["cases"][0]["timed_count"], 99)
        self.assertIn("per_shape_measured", k["latency_basis"])
        self.assertIn("enforce-eager", summ["data_semantics"]["mode"])
        self.assertIn("REAL per-shape call frequency", summ["data_semantics"]["count"])

    def test_untimed_probe_is_labelled_graph_capture_not_traffic(self):
        """The dangerous case: counts here are capture artefacts and must say so."""
        self._write_pid(11, [_pid_file([[64, 1024]], 100)])
        summ = self._build()
        self.assertIn("CUDA graph ON", summ["data_semantics"]["mode"])
        self.assertIn("NOT real serving frequency", summ["data_semantics"]["count"])
        self.assertIn("count_vs_profile", summ["data_semantics"])
        self.assertIn("profile_per_kernel_avg", summ["kernels"][0]["latency_basis"])

    def test_workload_defaults_to_nulls_and_is_echoed_when_given(self):
        self._write_pid(11, [_pid_file([[64, 1024]], 1)])
        self.assertEqual(self._build()["workload"], {"isl": None, "osl": None, "conc": None})
        wl = {"isl": 1024, "osl": 1024, "conc": 64}
        self.assertEqual(self._build(workload=wl)["workload"], wl)

    def test_kernels_are_discovered_from_the_probe_files_alone(self):
        """MODEL-AGNOSTIC: a target nobody hard-coded still appears, sorted deterministically."""
        self._write_pid(11, [_pid_file([[8, 8]], 1)], target="zzz.mod:zeta_kernel")
        self._write_pid(12, [_pid_file([[8, 8]], 1)], target="aaa.mod:alpha_kernel")
        summ = self._build()
        self.assertEqual([k["label"] for k in summ["kernels"]],
                         ["alpha_kernel", "zeta_kernel"])
        self.assertTrue(all(k["probe_status"] == "captured" for k in summ["kernels"]))

    def test_an_empty_probe_dir_still_produces_a_valid_report(self):
        summ = self._build()
        self.assertEqual(summ["kernels"], [])
        self.assertEqual(summ["schema"], "per-shape-probe-v1")
        self.assertIn("CUDA graph ON", summ["data_semantics"]["mode"])


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
class Markdown(_PostprocessTestCase):
    def test_pct_gpu_renders_in_all_three_states(self):
        self.assertEqual(pp._fmt_pct_gpu(42.5), "42.50%")
        self.assertEqual(pp._fmt_pct_gpu(7), "7.00%")
        self.assertEqual(pp._fmt_pct_gpu(pp.GPU_UNKNOWN), "unknown (no profile)")
        self.assertEqual(pp._fmt_pct_gpu(None), "n/a")

    def test_md_leads_with_the_semantics_warning_then_a_shape_table(self):
        self._write_pid(11, [_pid_file([[64, 1024]], 100, gpu_us_avg=8.0, timed_count=99)],
                        timing=True)
        md = pp.to_md(self._build(workload={"isl": 1024, "osl": 1024, "conc": 64}))
        self.assertIn("# per-shape probe — per-shape-probe-v1", md)
        self.assertIn("ISL=1024 OSL=1024 conc=64", md)
        self.assertLess(md.index("数据语义与局限"), md.index("## matmul_ogs"))
        self.assertIn("latency_us (measured)", md)
        self.assertIn("| `[[64, 1024]]` | torch.bfloat16 | 100 | 100.00% | 8.0 |", md)

    def test_md_marks_an_untimed_table_as_approx(self):
        self._write_pid(11, [_pid_file([[64, 1024]], 100)])
        md = pp.to_md(self._build())
        self.assertIn("~latency_us (approx)", md)
        self.assertIn("| n/a |", md)

    def test_md_caps_the_table_at_50_shapes(self):
        self._write_pid(11, [_pid_file([[i, 1024]], 100 - i) for i in range(60)])
        md = pp.to_md(self._build())
        self.assertEqual(md.count("| torch.bfloat16 |"), 50)

    def test_md_renders_a_non_captured_kernel_as_a_note(self):
        summ = {"schema": "per-shape-probe-v1", "workload": None, "profile_source": None,
                "data_semantics": {"mode": "n/a"}, "note_coverage": "",
                "kernels": [{"label": "pack_bitmatrix", "target": "m:pack_bitmatrix",
                             "probe_status": "unhookable", "note": "JITFunction"}]}
        md = pp.to_md(summ)
        self.assertIn("probe_status: **unhookable** — JITFunction", md)
        self.assertNotIn("| shape (dims) |", md)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
class Main(_PostprocessTestCase):
    def _run(self, *extra):
        sys.argv = ["probe_postprocess.py", *extra]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            pp.main()
        return out.getvalue(), err.getvalue()

    def test_writes_both_json_and_md_and_summarises_on_stdout(self):
        self._write_pid(11, [_pid_file([[64, 1024]], 100)])
        prof = self._write_profile([{"name": "_matmul_ogs", "calls": 100, "avg_us": 9.0,
                                     "pct_gpu_time": 33.0}])
        out_prefix = os.path.join(self.tmp, "per_shape_probe")
        out, err = self._run("--probe-dir", self.probe_dir, "--profile-topn", prof,
                             "--out", out_prefix, "--isl", "1024", "--osl", "1024",
                             "--conc", "64")
        with open(out_prefix + ".json") as fh:
            summ = json.load(fh)
        self.assertEqual(summ["workload"], {"isl": 1024, "osl": 1024, "conc": 64})
        self.assertTrue(os.path.exists(out_prefix + ".md"))
        self.assertIn("matmul_ogs: 1 shapes, probe_calls=100 profile_calls=100", out)
        self.assertIn("wrote", err)

    def test_workload_is_omitted_when_no_flag_is_given(self):
        self._write_pid(11, [_pid_file([[64, 1024]], 1)])
        out_prefix = os.path.join(self.tmp, "p")
        self._run("--probe-dir", self.probe_dir, "--out", out_prefix)
        with open(out_prefix + ".json") as fh:
            self.assertEqual(json.load(fh)["workload"],
                             {"isl": None, "osl": None, "conc": None})

    def test_non_captured_kernels_print_their_status(self):
        self._write_pid(11, [_pid_file([[64, 1024]], 1)])
        real_build = pp.build

        def patched(*a, **k):
            summ = real_build(*a, **k)
            summ["kernels"][0]["probe_status"] = "unhookable"
            return summ

        pp.build = patched
        self.addCleanup(setattr, pp, "build", real_build)
        out, _ = self._run("--probe-dir", self.probe_dir,
                           "--out", os.path.join(self.tmp, "p"))
        self.assertIn("matmul_ogs: unhookable", out)


if __name__ == "__main__":
    unittest.main()
