#!/usr/bin/env python3
"""Unit tests for parse_profile.py -- the Profile phase's trace -> canonical Top-N contract.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_parse_profile.py

This script is the single place the whole e2e workflow learns WHICH kernel is the bottleneck and
WHAT shapes/dtypes it runs at. Every downstream agent (extractor, harness, attribute_weights) reads
its JSON and trusts it blindly, so a misparse does not fail loudly -- it silently points the entire
optimization run at the wrong kernel, or benchmarks the right kernel at the wrong shape. These tests
pin the parsing and the emitted schema, with emphasis on the branches that decide those two things:

  - classify / short_name / norm_key   : name -> class/backend/editable + the loose key that joins a
                                         rocprof HW name to a torch op name (a bad join = no shapes)
  - _seg / _classify_step /
    _collect_step_spans               : the gpu_user_annotation step windows that separate PREFILL
                                         from DECODE. Mis-windowing attributes decode launches to
                                         prefill and inverts the per-phase latency the unittest
                                         weights by.
  - snap_capture_size /
    analytic_regime_calls             : the decode-M snap and the per-phase call denominator, incl.
                                         the fallback rung taken when the shared attribute_weights
                                         impl is not importable.
  - parse_torch_trace                 : cpu_op External-id -> shape linking, the 5-shape display cap
                                         vs the UNCAPPED by_case distribution, launches that fall
                                         outside every step window, gzip input, and malformed input.
  - parse_rocprof_dir                 : CSV column-name dialects and the skip-and-continue guards
                                         for unreadable / empty / column-less files.
  - build_summary / build_workload    : the emitted JSON shape, shape enrichment from the torch agg,
                                         and the steady/NOT-steady serving verdict.
  - to_markdown / main                : the human table and the CLI wiring (argv -> files on disk).

Stdlib only, no GPU, no network -- everything is driven from synthetic traces in tempdirs.
"""
import contextlib
import gzip
import importlib.util
import io
import json
import os
import runpy
import shutil
import sys
import tempfile
import types
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(mod_name, filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pp = _load("parse_profile", "parse_profile.py")
MODULE_PATH = os.path.join(SCRIPTS_DIR, "parse_profile.py")

_MISSING = object()


@contextlib.contextmanager
def _shared_impl(func):
    """Swap the optional `attribute_weights` dependency that analytic_regime_calls imports lazily.

    Both rungs (shared impl importable / not importable) must be reachable from a test regardless of
    whether scripts/ happens to be on sys.path in the runner -- otherwise the fallback formula, which
    is what actually runs in CI, would go unverified.
    """
    mod = types.ModuleType("attribute_weights")
    if func is not None:
        mod.estimate_serving_regime_calls = func
    saved = sys.modules.get("attribute_weights", _MISSING)
    sys.modules["attribute_weights"] = mod
    try:
        yield
    finally:
        if saved is _MISSING:
            del sys.modules["attribute_weights"]
        else:
            sys.modules["attribute_weights"] = saved


# --------------------------------------------------------------------------- #
# Synthetic vLLM-style trace: one prefill step window [100,200), one decode
# window [300,350), cpu_ops carrying the shapes, and kernel launches placed
# both inside and outside those windows.
# --------------------------------------------------------------------------- #
GEMM = "void gemm_kernel<float>(int)"


def _trace_events():
    return [
        # step annotations -> the phase windows
        {"cat": "gpu_user_annotation", "name": "execute_model context_2(512) generation_0(0)",
         "ts": 100, "dur": 100},
        {"cat": "gpu_user_annotation", "name": "execute_model context_0() generation_4(sq1sk513)",
         "ts": 300, "dur": 50},
        # annotations that are NOT step windows
        {"cat": "gpu_user_annotation", "name": "ProfilerStep#1", "ts": 0, "dur": 1000},
        {"cat": "gpu_user_annotation", "name": "execute_model context_1(8)", "ts": 400, "dur": 5},
        {"cat": "gpu_user_annotation", "name": "execute_model context_1(8) generation_1(1)",
         "ts": 500},
        "not-a-dict",
        # cpu_ops -> the External id -> shape/dtype table
        {"cat": "cpu_op", "name": "aten::mm",
         "args": {"External id": 7, "Input Dims": [[4, 512], [512, 512], []],
                  "Input type": ["c10::BFloat16", "c10::BFloat16", ""]}},
        {"cat": "cpu_op", "name": "aten::mm_again",
         "args": {"External id": 7, "Input Dims": [[9, 9]], "Input type": ["float"]}},
        {"cat": "cpu_op", "name": "aten::empty", "args": {"External id": 8, "Input Dims": [[], []]}},
        {"cat": "cpu_op", "name": "aten::no_dims", "args": {"External id": 9}},
        {"cat": "cpu_op", "name": "aten::no_ext", "args": {"Input Dims": [[1]]}},
        {"cat": "cpu_op", "name": "aten::relu",
         "args": {"External id": 11, "Input Dims": [[4, 512]], "Input type": [""]}},
        {"cat": "cpu_op", "name": "aten::silu", "args": {"External id": 12,
                                                         "Input Dims": [[4, 512]]}},
        # kernel launches
        {"cat": "kernel", "name": GEMM, "ts": 120, "dur": 200.0, "args": {"External id": 7}},
        {"cat": "kernel", "name": GEMM, "ts": 310, "dur": 100.0, "args": {"External id": 7}},
        {"cat": "kernel", "name": "triton_rms_norm_kernel", "ts": 130, "dur": 50.0,
         "args": {"External id": 11}},
        {"cat": "kernel", "name": "silu_and_mul_kernel", "ts": 140, "dur": 25.0,
         "args": {"External id": 12}},
        {"cat": "kernel", "name": "graph_replay_kernel", "ts": 50, "dur": 10.0},
        {"cat": "kernel", "name": "graph_replay_kernel", "ts": 250, "dur": 10.0},
        {"cat": "gpu_memcpy", "name": "Memcpy DtoH", "dur": 5.0},
        {"cat": "gpu_memset", "name": "Memset", "ts": 320, "dur": None},
        # a CPU-side launch stub that must NOT be counted as GPU time
        {"cat": "cuda_runtime", "name": "hipLaunchKernel", "ts": 119, "dur": 3.0},
    ]


KERNEL_STATS_CSV = (
    '"Name","Calls","TotalDurationNs","AverageNs"\n'
    '"void gemm_kernel<float>(int) [clone .kd]",4,4000000,1000000\n'
    '"triton_rms_norm_kernel",2,1000000,500000\n'
)


class _TmpMixin:
    """Per-test tempdirs, so each test states only the files it cares about."""

    def setUp(self):
        self._dirs = []

    def tearDown(self):
        for d in self._dirs:
            shutil.rmtree(d, ignore_errors=True)

    def _dir(self):
        d = tempfile.mkdtemp(prefix="parse_profile_test_")
        self._dirs.append(d)
        return d

    def _write(self, name, text, root=None, binary=False):
        root = root or self._dir()
        path = os.path.join(root, name)
        with open(path, "wb" if binary else "w") as fh:
            fh.write(text)
        return path

    def _trace_file(self, events=None, gz=False, raw=None):
        payload = raw if raw is not None else json.dumps(
            {"schemaVersion": 1, "traceEvents": _trace_events() if events is None else events})
        path = os.path.join(self._dir(), "trace.json.gz" if gz else "trace.json")
        opener = (lambda: gzip.open(path, "wt")) if gz else (lambda: open(path, "w"))
        with opener() as fh:
            fh.write(payload)
        return path

    def _rocprof_dir(self, csv_text=KERNEL_STATS_CSV, name="results_kernel_stats.csv"):
        d = self._dir()
        self._write(name, csv_text, root=d)
        return d


def _agg_entry(total_us, calls=1, shapes=(), dtypes=(), by_case=None, by_phase=None):
    return {"calls": calls, "total_us": total_us, "shapes": set(shapes), "dtypes": set(dtypes),
            "by_case": dict(by_case or {}), "by_phase": dict(by_phase or {})}


# --------------------------------------------------------------------------- #
# classify / short_name / norm_key
# --------------------------------------------------------------------------- #
class TestClassify(unittest.TestCase):
    def test_first_matching_rule_wins(self):
        # 'triton' is checked before the GEMM rule, so a Triton GEMM stays source-editable.
        cls, backend, editable, hint = pp.classify("triton_gemm_fp8_kernel")
        self.assertEqual((cls, backend, editable), ("triton", "triton", True))
        self.assertIn("Triton", hint)

    def test_library_gemm_is_not_editable(self):
        cls, backend, editable, _ = pp.classify(GEMM)
        self.assertEqual((cls, backend, editable), ("library_gemm", "hipblaslt", False))

    def test_rule_table_routes_each_family(self):
        expected = {
            "Cijk_Alik_Bljk_HHS_BH": ("library_gemm", "hipblaslt", False),
            "aiter::fused_moe": ("fused_custom", "aiter", True),
            "flash_fwd_splitkv": ("library_attn", "ck", False),
            "ck_tile_grouped_conv": ("fused_custom", "ck", True),
            "selective_scan_fwd": ("fused_custom", "triton", True),
            "rms_norm_forward": ("reduction_norm", "triton", True),
            "vectorized_elementwise_add": ("elementwise_overhead", "torch_native", True),
            "Memcpy DtoH": ("memory", "torch_native", False),
        }
        for name, want in expected.items():
            with self.subTest(name=name):
                cls, backend, editable, _ = pp.classify(name)
                self.assertEqual((cls, backend, editable), want)

    def test_snake_case_kernel_falls_back_to_triton(self):
        # An unmatched snake_case *_kernel symbol is almost always a JIT kernel -> editable.
        cls, backend, editable, hint = pp.classify("graph_replay_kernel")
        self.assertEqual((cls, backend, editable), ("triton", "triton", True))
        self.assertIn("Snake_case", hint)
        self.assertEqual(pp.classify("attn_bwd_kernel_v2")[0], "library_attn")
        self.assertEqual(pp.classify("MyThing_fwd_kernel")[:3], ("triton", "triton", True))

    def test_unclassified_symbol_is_other_but_still_editable(self):
        # The default must stay editable=True: an unknown kernel should be inspected, not skipped.
        cls, backend, editable, hint = pp.classify("MyOpLauncher::run")
        self.assertEqual((cls, backend, editable), ("other", "unknown", True))
        self.assertIn("Unclassified", hint)
        self.assertEqual(pp.classify("custom_op_0")[0], "other")

    def test_classification_is_case_insensitive(self):
        self.assertEqual(pp.classify("TRITON_FOO")[0], "triton")


class TestShortNameAndNormKey(unittest.TestCase):
    def test_strips_void_template_and_namespace(self):
        self.assertEqual(pp.short_name(GEMM), "gemm_kernel")
        self.assertEqual(pp.short_name("void at::native::vectorized_elementwise<4>(int)"),
                         "vectorized_elementwise")
        self.assertEqual(pp.short_name("MyOpLauncher::run"), "run")

    def test_truncates_to_60_chars(self):
        self.assertEqual(pp.short_name("k" * 70), "k" * 60)

    def test_non_identifier_name_is_returned_as_is(self):
        self.assertEqual(pp.short_name("<unknown>"), "<unknown>")

    def test_norm_key_joins_hw_name_to_torch_name(self):
        # This is the only bridge from a rocprof HW symbol to the torch op that carries the shapes;
        # if the two do not normalize to the same key the merged report loses all shapes.
        self.assertEqual(pp.norm_key(GEMM), "gemmkernel")
        self.assertEqual(pp.norm_key("void gemm_kernel<float>(int) [clone .kd]"), "gemmkernel")
        self.assertEqual(pp.norm_key("at::native::Fill_Kernel"), "fillkernel")


# --------------------------------------------------------------------------- #
# serving-phase step windows
# --------------------------------------------------------------------------- #
class TestStepSpans(unittest.TestCase):
    def test_seg_without_the_tag_is_all_zeros(self):
        self.assertEqual(pp._seg("execute_model", "context"), (0, 0, 0))
        self.assertEqual(pp._seg("context_abc(12)", "context"), (0, 0, 0))

    def test_seg_geak_dialect_counts_tokens(self):
        self.assertEqual(pp._seg("execute_model context_2(512)", "context"), (2, 512, 0))

    def test_seg_sq_sk_dialect_reports_query_and_kv(self):
        self.assertEqual(pp._seg("generation_4(sq1sk513)", "generation"), (4, 1, 513))
        self.assertEqual(pp._seg("generation_4(sq8)", "generation"), (4, 8, 0))

    def test_seg_non_numeric_inner_is_zero_tokens(self):
        self.assertEqual(pp._seg("context_0()", "context"), (0, 0, 0))

    def test_classify_step_needs_both_halves(self):
        self.assertIsNone(pp._classify_step("execute_model context_1(8)"))
        self.assertIsNone(pp._classify_step("execute_model generation_1(1)"))
        self.assertIsNone(pp._classify_step("ProfilerStep#1"))

    def test_classify_step_prefill_and_decode(self):
        self.assertEqual(
            pp._classify_step("execute_model context_2(512) generation_0(0)"), (True, 512, 0, 0))
        self.assertEqual(
            pp._classify_step("execute_model context_0() generation_4(sq1sk513)"),
            (False, 0, 4, 513))

    def test_collect_step_spans_sorts_and_skips_non_spans(self):
        spans = pp._collect_step_spans(_trace_events())
        self.assertEqual(spans, [(100, 200, "P", 512, 0), (300, 350, "D", 0, 4)])

    def test_collect_step_spans_sorts_out_of_order_input(self):
        events = [
            {"cat": "gpu_user_annotation", "name": "execute_model context_0() generation_2(2)",
             "ts": 900, "dur": 10},
            {"cat": "gpu_user_annotation", "name": "execute_model context_1(64) generation_0(0)",
             "ts": 10, "dur": 20},
        ]
        self.assertEqual([s[0] for s in pp._collect_step_spans(events)], [10, 900])

    def test_collect_step_spans_requires_ts_and_dur(self):
        # A span with no duration cannot bound a window; it must be dropped, not defaulted to 0.
        events = [{"cat": "gpu_user_annotation",
                   "name": "execute_model context_1(8) generation_1(1)", "ts": 5}]
        self.assertEqual(pp._collect_step_spans(events), [])

    def test_collect_step_spans_ignores_other_categories(self):
        events = [{"cat": "cpu_op", "name": "execute_model context_1(8) generation_1(1)",
                   "ts": 5, "dur": 5}]
        self.assertEqual(pp._collect_step_spans(events), [])


# --------------------------------------------------------------------------- #
# snap_capture_size
# --------------------------------------------------------------------------- #
class TestSnapCaptureSize(unittest.TestCase):
    def test_no_sizes_or_no_value_is_a_passthrough(self):
        self.assertEqual(pp.snap_capture_size(5, []), 5)
        self.assertEqual(pp.snap_capture_size(5, None), 5)
        self.assertEqual(pp.snap_capture_size(0, [1, 2, 4]), 0)

    def test_rounds_up_to_the_next_capture_size(self):
        self.assertEqual(pp.snap_capture_size(3, [1, 2, 4, 8]), 4)
        self.assertEqual(pp.snap_capture_size(8, [8, 1, 4, 2]), 8)

    def test_above_the_largest_size_clamps_to_the_largest(self):
        # Production cannot pad past the biggest captured graph, so neither may the estimate.
        self.assertEqual(pp.snap_capture_size(99, [1, 2, 4, 8]), 8)


# --------------------------------------------------------------------------- #
# analytic_regime_calls -- shared impl and the local fallback
# --------------------------------------------------------------------------- #
class TestAnalyticRegimeCalls(unittest.TestCase):
    def test_prefers_the_shared_impl_and_forwards_the_chunk(self):
        seen = {}

        def fake(isl, osl, conc, prefill_chunk=None):
            seen.update(isl=isl, osl=osl, conc=conc, prefill_chunk=prefill_chunk)
            return {"prefill": 111, "decode": 222}

        with _shared_impl(fake):
            self.assertEqual(pp.analytic_regime_calls(512, 8, 4, 256),
                             {"prefill": 111, "decode": 222})
        self.assertEqual(seen, {"isl": 512, "osl": 8, "conc": 4, "prefill_chunk": 256})

    def test_shared_impl_gets_conc_of_one_when_unset(self):
        with _shared_impl(lambda isl, osl, conc, prefill_chunk=None: {"conc": conc}):
            self.assertEqual(pp.analytic_regime_calls(512, 8, 0), {"conc": 1})

    def test_fallback_matches_the_documented_formula(self):
        # prefill carries CONC in the launch COUNT; decode carries it in the SHAPE, so decode==OSL.
        with _shared_impl(None):
            self.assertEqual(pp.analytic_regime_calls(512, 8, 4, 256),
                             {"prefill": 8, "decode": 8})

    def test_fallback_chunk_defaults_to_a_single_prefill_pass(self):
        with _shared_impl(None):
            self.assertEqual(pp.analytic_regime_calls(512, 8, 0), {"prefill": 1, "decode": 8})
            self.assertEqual(pp.analytic_regime_calls(1000, 4, 2, 256),
                             {"prefill": 8, "decode": 4})

    def test_fallback_without_both_isl_and_osl_is_empty(self):
        with _shared_impl(None):
            self.assertEqual(pp.analytic_regime_calls(0, 8, 4), {})
            self.assertEqual(pp.analytic_regime_calls(512, 0, 4), {})
            self.assertEqual(pp.analytic_regime_calls(None, None, 1), {})

    def test_shared_impl_exception_falls_back_instead_of_crashing(self):
        def boom(*a, **k):
            raise RuntimeError("shared impl is broken")

        with _shared_impl(boom):
            self.assertEqual(pp.analytic_regime_calls(512, 8, 1, 512),
                             {"prefill": 1, "decode": 8})


# --------------------------------------------------------------------------- #
# parse_torch_trace
# --------------------------------------------------------------------------- #
class TestParseTorchTrace(_TmpMixin, unittest.TestCase):
    def _parse(self, **kw):
        return pp.parse_torch_trace(self._trace_file(**kw))

    def test_only_gpu_categories_are_counted(self):
        agg, total_us, launches, _ = self._parse()
        self.assertEqual(sorted(agg), ["Memcpy DtoH", "Memset", "graph_replay_kernel",
                                       "silu_and_mul_kernel", "triton_rms_norm_kernel", GEMM])
        self.assertNotIn("hipLaunchKernel", agg)   # cuda_runtime is host time, not GPU time
        self.assertEqual(launches, 8)
        self.assertAlmostEqual(total_us, 400.0)

    def test_launches_of_one_kernel_are_aggregated(self):
        agg, _, _, _ = self._parse()
        self.assertEqual(agg[GEMM]["calls"], 2)
        self.assertAlmostEqual(agg[GEMM]["total_us"], 300.0)

    def test_missing_duration_counts_as_zero_not_a_crash(self):
        agg, _, _, _ = self._parse()
        self.assertEqual(agg["Memset"]["calls"], 1)
        self.assertAlmostEqual(agg["Memset"]["total_us"], 0.0)

    def test_shapes_and_dtypes_come_from_the_linked_cpu_op(self):
        agg, _, _, _ = self._parse()
        self.assertEqual(sorted(agg[GEMM]["shapes"]), ["[[4, 512], [512, 512]]"])
        self.assertEqual(sorted(agg[GEMM]["dtypes"]), ["c10::BFloat16"])

    def test_first_cpu_op_for_an_external_id_wins(self):
        # Two cpu_ops share External id 7; the later [[9, 9]] must not overwrite the real GEMM dims.
        agg, _, _, _ = self._parse()
        self.assertNotIn("[[9, 9]]", agg[GEMM]["shapes"])

    def test_all_empty_input_dims_do_not_register_a_shape(self):
        # aten::empty (id 8) reports [[], []]; a kernel linked to it must stay shape-unknown rather
        # than claim a bogus zero-dim shape.
        events = _trace_events() + [
            {"cat": "kernel", "name": "empty_like_kernel", "ts": 150, "dur": 1.0,
             "args": {"External id": 8}}]
        agg, _, _, _ = pp.parse_torch_trace(self._trace_file(events=events))
        self.assertEqual(agg["empty_like_kernel"]["shapes"], set())
        self.assertEqual(list(agg["empty_like_kernel"]["by_case"]), [("", "")])

    def test_blank_input_types_yield_no_dtypes(self):
        agg, _, _, _ = self._parse()
        self.assertEqual(agg["triton_rms_norm_kernel"]["shapes"], {"[[4, 512]]"})
        self.assertEqual(agg["triton_rms_norm_kernel"]["dtypes"], set())

    def test_absent_input_type_key_yields_no_dtypes(self):
        agg, _, _, _ = self._parse()
        self.assertEqual(agg["silu_and_mul_kernel"]["shapes"], {"[[4, 512]]"})
        self.assertEqual(agg["silu_and_mul_kernel"]["dtypes"], set())

    def test_unlinked_launches_collapse_into_one_shape_unknown_case(self):
        agg, _, _, _ = self._parse()
        d = agg["graph_replay_kernel"]
        self.assertEqual(d["shapes"], set())
        self.assertEqual(d["by_case"][("", "")]["count"], 2)
        self.assertAlmostEqual(d["by_case"][("", "")]["total_us"], 20.0)

    def test_display_shapes_cap_at_five_but_by_case_is_uncapped(self):
        # The Top-N table shows at most 5 distinct shapes; the workload model needs ALL of them, so
        # by_case must keep growing past the display cap.
        events = []
        for i in range(7):
            ext = 100 + i
            events.append({"cat": "cpu_op", "name": "aten::mm",
                           "args": {"External id": ext, "Input Dims": [[i + 1, 8]],
                                    "Input type": ["float"]}})
            events.append({"cat": "kernel", "name": "many_shape_kernel", "dur": 1.0,
                           "args": {"External id": ext}})
        agg, _, launches, _ = pp.parse_torch_trace(self._trace_file(events=events))
        self.assertEqual(launches, 7)
        self.assertEqual(len(agg["many_shape_kernel"]["shapes"]), 5)
        self.assertEqual(len(agg["many_shape_kernel"]["by_case"]), 7)

    def test_launches_are_attributed_to_the_step_window_they_fall_in(self):
        agg, _, _, _ = self._parse()
        gemm = agg[GEMM]["by_phase"]
        self.assertEqual(sorted(gemm), ["decode", "prefill"])
        self.assertEqual(gemm["prefill"], {"count": 1, "total_us": 200.0, "m": {512: 1}})
        self.assertEqual(gemm["decode"], {"count": 1, "total_us": 100.0, "m": {4: 1}})
        self.assertEqual(agg["Memset"]["by_phase"]["decode"]["count"], 1)

    def test_prefill_step_m_is_every_token_in_the_step(self):
        agg, _, _, _ = self._parse()
        self.assertEqual(agg["triton_rms_norm_kernel"]["by_phase"]["prefill"]["m"], {512: 1})

    def test_launches_outside_every_window_have_no_phase(self):
        # ts=50 precedes the first window and ts=250 sits in the gap between windows; a launch with
        # no ts at all (Memcpy) also has nowhere to land. Guessing a phase here would fabricate
        # decode latency out of idle time.
        agg, _, _, _ = self._parse()
        self.assertEqual(agg["graph_replay_kernel"]["by_phase"], {})
        self.assertEqual(agg["Memcpy DtoH"]["by_phase"], {})
        self.assertEqual(agg["graph_replay_kernel"]["by_case"][("", "")].get("phase"), {})

    def test_per_case_phase_is_recorded(self):
        agg, _, _, _ = self._parse()
        case = agg[GEMM]["by_case"][('[[4, 512], [512, 512]]',
                                     '["c10::BFloat16", "c10::BFloat16"]')]
        self.assertEqual(case["count"], 2)
        self.assertEqual(case["phase"], {"prefill": 1, "decode": 1})

    def test_phase_meta_summarizes_the_windows(self):
        _, _, _, meta = self._parse()
        self.assertEqual(meta, {"has_annotations": True, "n_prefill_steps": 1,
                                "n_decode_steps": 1, "prefill_tokens": 512,
                                "decode_batches": [4]})

    def test_trace_without_annotations_has_empty_phase_meta(self):
        events = [{"cat": "kernel", "name": "k_kernel", "ts": 1, "dur": 2.0}]
        agg, total, launches, meta = pp.parse_torch_trace(self._trace_file(events=events))
        self.assertEqual(meta, {})
        self.assertEqual(agg["k_kernel"]["by_phase"], {})
        self.assertEqual((total, launches), (2.0, 1))

    def test_gzipped_trace_is_transparently_read(self):
        agg, total, launches, meta = self._parse(gz=True)
        self.assertEqual(launches, 8)
        self.assertAlmostEqual(total, 400.0)
        self.assertTrue(meta["has_annotations"])

    def test_trace_without_trace_events_key_is_empty(self):
        agg, total, launches, meta = pp.parse_torch_trace(
            self._trace_file(raw=json.dumps({"schemaVersion": 1})))
        self.assertEqual((agg, total, launches, meta), ({}, 0.0, 0, {}))

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            pp.parse_torch_trace(os.path.join(self._dir(), "nope.json"))

    def test_malformed_json_raises_a_decode_error(self):
        # A truncated trace (profiler killed mid-flush) must fail loudly, not parse to an empty
        # profile that would report "no bottleneck".
        with self.assertRaises(json.JSONDecodeError):
            pp.parse_torch_trace(self._trace_file(raw='{"traceEvents": [{"cat": "kernel"'))

    def test_empty_file_raises_a_decode_error(self):
        with self.assertRaises(json.JSONDecodeError):
            pp.parse_torch_trace(self._trace_file(raw=""))

    def test_non_gzip_content_behind_a_gz_suffix_raises(self):
        path = os.path.join(self._dir(), "trace.json.gz")
        with open(path, "w") as fh:
            fh.write('{"traceEvents": []}')
        with self.assertRaises(gzip.BadGzipFile):
            pp.parse_torch_trace(path)

    def test_undecodable_bytes_raise_a_unicode_error(self):
        path = self._write("trace.json", b"\xff\xfe\x00{", binary=True)
        with self.assertRaises(UnicodeDecodeError):
            pp.parse_torch_trace(path)

    def test_bare_list_trace_is_not_actually_supported(self):
        # SOURCE BUG (asserted, not fixed): line 219 evaluates `data.get(...)` before the
        # `isinstance(data, list)` default, so a top-level-list trace (chrome's other legal shape)
        # raises AttributeError and the list fallback is dead code.
        with self.assertRaises(AttributeError):
            pp.parse_torch_trace(self._trace_file(
                raw=json.dumps([{"cat": "kernel", "name": "k", "dur": 1}])))


# --------------------------------------------------------------------------- #
# parse_rocprof_dir
# --------------------------------------------------------------------------- #
class TestParseRocprofDir(_TmpMixin, unittest.TestCase):
    def test_kernel_stats_csv_is_read_as_the_authoritative_aggregate(self):
        agg, total_us, launches = pp.parse_rocprof_dir(self._rocprof_dir())
        self.assertEqual(sorted(agg), ["triton_rms_norm_kernel",
                                       "void gemm_kernel<float>(int) [clone .kd]"])
        self.assertEqual(agg["triton_rms_norm_kernel"]["calls"], 2)
        self.assertAlmostEqual(agg["triton_rms_norm_kernel"]["total_us"], 1000.0)
        self.assertAlmostEqual(total_us, 5000.0)
        self.assertEqual(launches, 6)

    def test_rocprof_agg_carries_no_shapes(self):
        # rocprof sees HW kernels only; shapes must stay empty so build_summary knows to enrich.
        agg, _, _ = pp.parse_rocprof_dir(self._rocprof_dir())
        self.assertEqual(agg["triton_rms_norm_kernel"]["shapes"], set())
        self.assertEqual(agg["triton_rms_norm_kernel"]["dtypes"], set())

    def test_nested_subdirectory_is_found(self):
        root = self._dir()
        sub = os.path.join(root, "run_1")
        os.makedirs(sub)
        self._write("out_kernel_stats.csv", KERNEL_STATS_CSV, root=sub)
        _, total_us, launches = pp.parse_rocprof_dir(root)
        self.assertAlmostEqual(total_us, 5000.0)
        self.assertEqual(launches, 6)

    def test_missing_directory_is_empty_not_an_error(self):
        self.assertEqual(pp.parse_rocprof_dir(os.path.join(self._dir(), "absent")),
                         ({}, 0.0, 0))

    def test_directory_without_csvs_is_empty(self):
        d = self._dir()
        self._write("results.json", "{}", root=d)
        self.assertEqual(pp.parse_rocprof_dir(d), ({}, 0.0, 0))

    def test_unreadable_empty_and_columnless_files_are_skipped(self):
        # rocprofv3 litters its output dir with sibling CSVs (and sometimes directories) that are
        # not kernel stats. Any of them must be skipped, not abort the scan before the real file.
        d = self._dir()
        os.makedirs(os.path.join(d, "a1_is_a_dir.csv"))
        self._write("a2_binary.csv", b"Name,TotalDurationNs\n\xff\xfe,1\n", root=d, binary=True)
        self._write("a3_empty.csv", "", root=d)
        self._write("a4_no_useful_columns.csv", "Foo,Bar\n1,2\n", root=d)
        self._write("a5_header_only.csv", "Name,TotalDurationNs\n", root=d)
        self._write("z_kernel_stats.csv", KERNEL_STATS_CSV, root=d)
        agg, total_us, launches = pp.parse_rocprof_dir(d)
        self.assertEqual(len(agg), 2)
        self.assertAlmostEqual(total_us, 5000.0)
        self.assertEqual(launches, 6)

    def test_only_the_first_valid_stats_file_is_used(self):
        # Two aggregates would double-count total GPU time, so the scan stops at the first one.
        d = self._dir()
        self._write("a_kernel_stats.csv",
                    "Name,Calls,TotalDurationNs\nonly_kernel,1,1000\n", root=d)
        self._write("b_kernel_stats.csv",
                    "Name,Calls,TotalDurationNs\nother_kernel,1,9000\n", root=d)
        agg, total_us, launches = pp.parse_rocprof_dir(d)
        self.assertEqual(list(agg), ["only_kernel"])
        self.assertAlmostEqual(total_us, 1.0)
        self.assertEqual(launches, 1)

    def test_kernel_name_and_total_duration_column_dialect(self):
        d = self._rocprof_dir("Kernel_Name,Total_Duration\nfoo_kernel,2500\n",
                              name="stats_kernel_x.csv")
        agg, total_us, launches = pp.parse_rocprof_dir(d)
        self.assertAlmostEqual(agg["foo_kernel"]["total_us"], 2.5)
        self.assertEqual(agg["foo_kernel"]["calls"], 1)   # no Calls column -> one launch
        self.assertEqual(launches, 1)
        self.assertAlmostEqual(total_us, 2.5)

    def test_kernelname_and_count_column_dialect(self):
        d = self._rocprof_dir("KernelName,TotalDurationNs,Count\nbar_kernel,3000,3\n")
        agg, _, launches = pp.parse_rocprof_dir(d)
        self.assertEqual(agg["bar_kernel"]["calls"], 3)
        self.assertEqual(launches, 3)

    def test_float_call_count_is_truncated_to_an_int(self):
        d = self._rocprof_dir("Name,Calls,TotalDurationNs\nbaz_kernel,3.0,3000\n")
        agg, _, launches = pp.parse_rocprof_dir(d)
        self.assertEqual(agg["baz_kernel"]["calls"], 3)
        self.assertEqual(launches, 3)

    def test_blank_cells_default_to_zero_duration_and_one_call(self):
        d = self._rocprof_dir("Name,Calls,TotalDurationNs\nqux_kernel,,\n")
        agg, total_us, launches = pp.parse_rocprof_dir(d)
        self.assertEqual(agg["qux_kernel"], {"calls": 1, "total_us": 0.0,
                                             "shapes": set(), "dtypes": set()})
        self.assertEqual((total_us, launches), (0.0, 1))

    def test_repeated_kernel_rows_accumulate(self):
        d = self._rocprof_dir("Name,Calls,TotalDurationNs\nk_kernel,1,1000\nk_kernel,2,3000\n")
        agg, total_us, launches = pp.parse_rocprof_dir(d)
        self.assertEqual(agg["k_kernel"]["calls"], 3)
        self.assertAlmostEqual(agg["k_kernel"]["total_us"], 4.0)
        self.assertEqual(launches, 3)

    def test_non_numeric_duration_raises(self):
        # SOURCE BUG (asserted, not fixed): a corrupt duration cell escapes as an unhandled
        # ValueError, unlike every other malformation in this function which is skipped.
        d = self._rocprof_dir("Name,Calls,TotalDurationNs\nk_kernel,1,not-a-number\n")
        with self.assertRaises(ValueError):
            pp.parse_rocprof_dir(d)


# --------------------------------------------------------------------------- #
# build_summary
# --------------------------------------------------------------------------- #
class TestBuildSummary(_TmpMixin, unittest.TestCase):
    def _torch(self):
        return pp.parse_torch_trace(self._trace_file())

    def test_top_n_is_ranked_by_total_time(self):
        agg, total, launches, meta = self._torch()
        summ = pp.build_summary(agg, total, launches, "torch-trace", 3, phase_meta=meta)
        self.assertEqual(summ["source"], "torch-trace")
        self.assertEqual(summ["total_gpu_time_ms"], 0.4)
        self.assertEqual(summ["num_kernel_launches"], 8)
        self.assertEqual(summ["num_distinct_kernels"], 6)
        self.assertEqual([k["short_name"] for k in summ["top_kernels"]],
                         ["gemm_kernel", "triton_rms_norm_kernel", "silu_and_mul_kernel"])
        self.assertEqual([k["rank"] for k in summ["top_kernels"]], [1, 2, 3])

    def test_entry_carries_the_documented_fields(self):
        agg, total, launches, meta = self._torch()
        top = pp.build_summary(agg, total, launches, "torch-trace", 1, phase_meta=meta,
                               conc=4, isl=512, osl=8, chunk=256,
                               capture_sizes=[1, 2, 4, 8])["top_kernels"][0]
        self.assertEqual(top["name"], GEMM)
        self.assertEqual(top["calls"], 2)
        self.assertEqual(top["total_ms"], 0.3)
        self.assertEqual(top["avg_us"], 150.0)
        self.assertEqual(top["pct_gpu_time"], 75.0)
        self.assertEqual(top["shapes"], [[[4, 512], [512, 512]]])
        self.assertEqual(top["dtypes"], ["c10::BFloat16"])
        self.assertEqual(top["classification"], "library_gemm")
        self.assertEqual(top["backend_guess"], "hipblaslt")
        self.assertFalse(top["editable"])

    def test_phase_annotation_is_measured_per_kernel(self):
        agg, total, launches, meta = self._torch()
        top = pp.build_summary(agg, total, launches, "torch-trace", 1, phase_meta=meta,
                               conc=4, isl=512, osl=8, chunk=256,
                               capture_sizes=[1, 2, 4, 8])["top_kernels"][0]
        self.assertEqual(top["phase"], "both")
        self.assertEqual(top["served_regimes"], ["prefill", "decode"])
        self.assertEqual(top["phase_calls_measured"], {"prefill": 1, "decode": 1})
        self.assertEqual(top["calls_per_step"], {"prefill": 1.0, "decode": 1.0})
        self.assertEqual(top["base_latency_ms"], {"prefill": 0.2, "decode": 0.1})
        self.assertEqual(top["est_calls"], {"prefill": 8, "decode": 8})

    def test_decode_m_is_snapped_to_a_capture_size(self):
        # Decode M is hidden behind the CUDA graph at runtime, so it is the padded capture size the
        # harness must benchmark -- not the raw concurrency.
        agg, total, launches, meta = self._torch()
        top = pp.build_summary(agg, total, launches, "torch-trace", 1, phase_meta=meta,
                               conc=3, capture_sizes=[1, 2, 4, 8])["top_kernels"][0]
        self.assertEqual(top["est_shape"]["prefill"], {"M": 512, "M_dist": {512: 1}})
        self.assertEqual(top["est_shape"]["decode"]["M"], 4)
        self.assertIn("cudagraph", top["est_shape"]["decode"]["M_note"])

    def test_decode_m_is_unknown_without_a_concurrency(self):
        agg, total, launches, meta = self._torch()
        top = pp.build_summary(agg, total, launches, "torch-trace", 1,
                               phase_meta=meta)["top_kernels"][0]
        self.assertIsNone(top["est_shape"]["decode"]["M"])
        self.assertNotIn("est_calls", top)

    def test_prefill_only_kernel_serves_one_regime(self):
        agg, total, launches, meta = self._torch()
        by_name = {k["name"]: k for k in
                   pp.build_summary(agg, total, launches, "torch-trace", 9,
                                    phase_meta=meta)["top_kernels"]}
        self.assertEqual(by_name["triton_rms_norm_kernel"]["phase"], "prefill")
        self.assertEqual(by_name["triton_rms_norm_kernel"]["served_regimes"], ["prefill"])
        self.assertEqual(by_name["Memset"]["phase"], "decode")
        self.assertNotIn("phase", by_name["graph_replay_kernel"])

    def test_enrichment_recovers_shapes_for_rocprof_names(self):
        # The merged path is the important one: HW durations from rocprof, shapes from the trace,
        # joined on norm_key. If this join breaks the harness benchmarks the wrong shape.
        torch_agg, _, _, meta = self._torch()
        rp_agg, rp_total, rp_launch = pp.parse_rocprof_dir(self._rocprof_dir())
        summ = pp.build_summary(rp_agg, rp_total, rp_launch, "merged", 5, enrich=torch_agg,
                                phase_meta=meta, conc=4, isl=512, osl=8, chunk=256,
                                capture_sizes=[1, 2, 4, 8])
        top = summ["top_kernels"][0]
        self.assertEqual(top["name"], "void gemm_kernel<float>(int) [clone .kd]")
        self.assertEqual(top["calls"], 4)                     # HW call count, not the trace's
        self.assertEqual(top["total_ms"], 4.0)                # HW duration
        self.assertEqual(top["shapes"], [[[4, 512], [512, 512]]])   # enriched from the trace
        self.assertEqual(top["dtypes"], ["c10::BFloat16"])
        self.assertEqual(top["phase"], "both")                # by_phase borrowed via norm_key
        self.assertEqual(top["base_latency_ms"], {"prefill": 0.2, "decode": 0.1})

    def test_unmatched_name_stays_shape_less_after_enrichment(self):
        torch_agg, _, _, _ = self._torch()
        rp_agg = {"unrelated_hw_kernel": _agg_entry(100.0, calls=1)}
        top = pp.build_summary(rp_agg, 100.0, 1, "merged", 5,
                               enrich=torch_agg)["top_kernels"][0]
        self.assertEqual(top["shapes"], [])
        self.assertEqual(top["dtypes"], [])
        self.assertNotIn("phase", top)

    def test_steady_decode_capture_is_trusted(self):
        meta = {"has_annotations": True, "n_prefill_steps": 1, "n_decode_steps": 4,
                "prefill_tokens": 512, "decode_batches": [4, 4, 4, 2]}
        sv = pp.build_summary({}, 0.0, 0, "torch-trace", 5, phase_meta=meta, conc=4,
                              isl=512, osl=8, chunk=256,
                              capture_sizes=[1, 2, 4, 8])["serving"]
        self.assertEqual(sv["decode_batch_captured"], 4)      # modal batch, not the max
        self.assertEqual(sv["decode_batch_steady"], 4)
        self.assertTrue(sv["steady"])
        self.assertIn("trustworthy", sv["note"])
        self.assertEqual(sv["analytic_calls"], {"prefill": 8, "decode": 8})

    def test_under_sampled_decode_is_flagged_not_silently_used(self):
        # This is the whole point of the serving block: a 4-deep decode capture at concurrency 64
        # biases per-launch decode time low, and the report must say so out loud.
        meta = {"has_annotations": True, "n_prefill_steps": 1, "n_decode_steps": 1,
                "prefill_tokens": 512, "decode_batches": [4]}
        sv = pp.build_summary({}, 0.0, 0, "torch-trace", 5, phase_meta=meta,
                             conc=64, isl=512, osl=8, capture_sizes=[])["serving"]
        self.assertFalse(sv["steady"])
        self.assertIn("COUNTS ok", sv["note"])
        self.assertIn("biased ", sv["note"])
        self.assertEqual(sv["decode_batch_steady"], 64)

    def test_no_annotations_means_no_serving_block(self):
        agg, total, launches, meta = self._torch()
        summ = pp.build_summary(agg, total, launches, "rocprofv3", 2, phase_meta={})
        self.assertNotIn("serving", summ)

    def test_empty_profile_reports_zero_instead_of_dividing_by_zero(self):
        summ = pp.build_summary({"k_kernel": _agg_entry(0.0, calls=0)}, 0.0, 0, "torch-trace", 5)
        self.assertEqual(summ["total_gpu_time_ms"], 0.0)
        self.assertEqual(summ["top_kernels"][0]["pct_gpu_time"], 0.0)
        self.assertEqual(summ["top_kernels"][0]["avg_us"], 0.0)

    def test_dtypes_are_capped_at_eight(self):
        agg = {"k_kernel": _agg_entry(10.0, dtypes=[f"dt{i}" for i in range(12)])}
        self.assertEqual(len(pp.build_summary(agg, 10.0, 1, "torch-trace", 1)
                             ["top_kernels"][0]["dtypes"]), 8)


# --------------------------------------------------------------------------- #
# build_workload
# --------------------------------------------------------------------------- #
class TestBuildWorkload(_TmpMixin, unittest.TestCase):
    def _torch(self):
        return pp.parse_torch_trace(self._trace_file())

    def test_cases_are_weighted_by_measured_time(self):
        agg, total, _, _ = self._torch()
        wl = pp.build_workload(agg, total, 2)
        self.assertEqual(wl["schema"], "workload-v1")
        self.assertEqual(wl["total_gpu_time_ms"], 0.4)
        self.assertEqual(wl["num_kernels"], 2)
        k = wl["kernels"][0]
        self.assertEqual(k["name"], GEMM)
        self.assertEqual(k["num_cases"], 1)
        case = k["cases"][0]
        self.assertEqual(case["dims"], [[4, 512], [512, 512]])
        self.assertEqual(case["dtypes"], ["c10::BFloat16", "c10::BFloat16"])
        self.assertEqual(case["count"], 2)
        self.assertEqual(case["baseline_latency_ms"], 0.15)
        self.assertEqual(case["weight"], 300.0)
        self.assertEqual(case["weight_norm"], 1.0)
        self.assertEqual(case["weight_source"], "trace")
        self.assertEqual(case["regime"], "prefill")   # tie -> first-seen phase wins
        self.assertEqual(k["phase"], "both")
        self.assertEqual(k["served_regimes"], ["prefill", "decode"])

    def test_shape_unknown_cases_are_labelled_regime_prior(self):
        # A graph-replay launch has real count/time but no recoverable shape; attribute_weights
        # must be able to tell those apart from measured shapes.
        agg, total, _, _ = self._torch()
        by_name = {k["name"]: k for k in pp.build_workload(agg, total, 9)["kernels"]}
        case = by_name["graph_replay_kernel"]["cases"][0]
        self.assertEqual(case["dims"], [])
        self.assertEqual(case["dtypes"], [])
        self.assertEqual(case["weight_source"], "regime_prior")
        self.assertNotIn("regime", case)
        self.assertNotIn("phase", by_name["graph_replay_kernel"])

    def test_cases_are_sorted_by_weight_and_normalized(self):
        by_case = {("[[1]]", ""): {"count": 1, "total_us": 10.0, "phase": {}},
                   ("[[2]]", ""): {"count": 3, "total_us": 90.0, "phase": {"decode": 3}}}
        agg = {"k_kernel": _agg_entry(100.0, calls=4, by_case=by_case)}
        cases = pp.build_workload(agg, 100.0, 1)["kernels"][0]["cases"]
        self.assertEqual([c["dims"] for c in cases], [[[2]], [[1]]])
        self.assertEqual([c["weight_norm"] for c in cases], [0.9, 0.1])
        self.assertEqual(cases[0]["regime"], "decode")

    def test_target_filter_keeps_only_the_kernel_under_optimization(self):
        agg, total, _, _ = self._torch()
        wl = pp.build_workload(agg, total, 9, target="RMS_Norm")
        self.assertEqual([k["name"] for k in wl["kernels"]], ["triton_rms_norm_kernel"])
        self.assertEqual(wl["num_kernels"], 1)
        self.assertEqual(wl["total_gpu_time_ms"], 0.4)   # denominator stays the whole profile

    def test_target_matching_nothing_yields_no_kernels(self):
        agg, total, _, _ = self._torch()
        self.assertEqual(pp.build_workload(agg, total, 9, target="nosuchkernel"),
                         {"schema": "workload-v1", "total_gpu_time_ms": 0.4,
                          "num_kernels": 0, "kernels": []})

    def test_kernel_without_cases_still_reports_its_time(self):
        agg = {"hw_only_kernel": _agg_entry(2500.0, calls=5)}
        k = pp.build_workload(agg, 2500.0, 1)["kernels"][0]
        self.assertEqual(k["num_cases"], 0)
        self.assertEqual(k["cases"], [])
        self.assertEqual(k["total_ms"], 2.5)
        self.assertEqual(k["pct_gpu_time"], 100.0)

    def test_zero_total_time_does_not_divide_by_zero(self):
        agg = {"k_kernel": _agg_entry(0.0, calls=0)}
        self.assertEqual(pp.build_workload(agg, 0.0, 1)["kernels"][0]["pct_gpu_time"], 0.0)


# --------------------------------------------------------------------------- #
# to_markdown
# --------------------------------------------------------------------------- #
class TestToMarkdown(_TmpMixin, unittest.TestCase):
    def _md(self, **kw):
        agg, total, launches, meta = pp.parse_torch_trace(self._trace_file())
        return pp.to_markdown(pp.build_summary(agg, total, launches, "torch-trace",
                                               kw.pop("top_n", 3), phase_meta=meta, **kw))

    def test_header_reports_source_and_totals(self):
        md = self._md()
        self.assertIn("# Profile Top-3 — standardized summary", md)
        self.assertIn("- source: `torch-trace`", md)
        self.assertIn("total GPU time: **0.40 ms** over 8 launches, 6 distinct kernels", md)

    def test_table_row_carries_the_routing_decision(self):
        md = self._md()
        row = [l for l in md.splitlines() if "`gemm_kernel`" in l][0]
        self.assertIn("library_gemm", row)
        self.assertIn("hipblaslt", row)
        self.assertIn("| N |", row)       # library GEMM is not source-editable
        self.assertIn("both", row)
        self.assertIn("75.0", row)

    def test_kernel_without_a_phase_renders_a_question_mark(self):
        md = self._md(top_n=9)
        row = [l for l in md.splitlines() if "`graph_replay_kernel`" in l][0]
        self.assertIn("| ? |", row)
        self.assertIn("``", row)          # no shapes recovered

    def test_serving_verdict_is_rendered(self):
        md = self._md(conc=4, isl=512, osl=8, chunk=256, capture_sizes=[1, 2, 4, 8])
        self.assertIn("- serving phase: 1 prefill + 1 decode steps", md)
        self.assertIn("decode batch captured=4 steady=4", md)
        self.assertIn("**STEADY**", md)
        self.assertIn("analytic_calls={'prefill': 8, 'decode': 8}", md)

    def test_not_steady_verdict_is_shouted(self):
        md = self._md(conc=64, isl=512, osl=8)
        self.assertIn("NOT steady — decode %/latency biased low", md)

    def test_no_serving_line_without_annotations(self):
        agg, total, launches, _ = pp.parse_torch_trace(self._trace_file())
        md = pp.to_markdown(pp.build_summary(agg, total, launches, "rocprofv3", 2))
        self.assertNotIn("serving phase", md)

    def test_long_shape_cell_is_truncated(self):
        agg = {"k_kernel": _agg_entry(10.0, shapes=["[[123456, 123456], [123456, 123456]]",
                                                    "[[654321, 654321], [654321, 654321]]"])}
        md = pp.to_markdown(pp.build_summary(agg, 10.0, 1, "torch-trace", 1))
        row = [l for l in md.splitlines() if l.startswith("| 1 |")][0]
        cell = row.split("`")[-2]
        self.assertTrue(cell.endswith("…"))
        self.assertEqual(len(cell), 61)

    def test_opt_hints_are_capped_at_twelve(self):
        agg = {f"custom_op_{i}": _agg_entry(float(20 - i)) for i in range(14)}
        md = pp.to_markdown(pp.build_summary(agg, 100.0, 14, "torch-trace", 14))
        self.assertIn("## Opt hints (top entries)", md)
        self.assertEqual(len([l for l in md.splitlines() if l.startswith("- **")]), 12)
        self.assertIn("Unclassified — inspect source to route.", md)

    def test_output_ends_with_a_newline(self):
        self.assertTrue(self._md().endswith("\n"))


# --------------------------------------------------------------------------- #
# main -- the CLI wiring the workflow actually invokes
# --------------------------------------------------------------------------- #
class TestMain(_TmpMixin, unittest.TestCase):
    def _main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        saved = sys.argv
        sys.argv = ["parse_profile.py"] + argv
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                pp.main()
        finally:
            sys.argv = saved
        return out.getvalue(), err.getvalue()

    def test_no_input_source_is_a_usage_error(self):
        err = io.StringIO()
        saved = sys.argv
        sys.argv = ["parse_profile.py", "--top", "5"]
        try:
            with contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as cm:
                    pp.main()
        finally:
            sys.argv = saved
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("--torch-trace", err.getvalue())

    def test_torch_trace_writes_json_md_and_workload(self):
        trace = self._trace_file()
        d = self._dir()
        prefix = os.path.join(d, "profile")
        wl_path = os.path.join(d, "workload.json")
        stdout, stderr = self._main([
            "--torch-trace", trace, "--out", prefix, "--workload-out", wl_path,
            "--top", "3", "--isl", "512", "--osl", "8", "--conc", "4",
            "--prefill-chunk", "256", "--capture-sizes", "1,2, 4 ,8"])

        with open(prefix + ".json") as fh:
            summ = json.load(fh)
        self.assertEqual(summ["source"], "torch-trace")
        self.assertEqual(summ["total_gpu_time_ms"], 0.4)
        self.assertEqual(summ["num_kernel_launches"], 8)
        self.assertEqual(summ["num_distinct_kernels"], 6)
        self.assertEqual(len(summ["top_kernels"]), 3)
        self.assertEqual(summ["top_kernels"][0]["shapes"], [[[4, 512], [512, 512]]])
        self.assertEqual(summ["top_kernels"][0]["est_calls"], {"prefill": 8, "decode": 8})
        self.assertEqual(summ["top_kernels"][0]["est_shape"]["decode"]["M"], 4)
        self.assertTrue(summ["serving"]["steady"])

        with open(prefix + ".md") as fh:
            md = fh.read()
        self.assertIn("# Profile Top-3", md)

        with open(wl_path) as fh:
            wl = json.load(fh)
        self.assertEqual(wl["schema"], "workload-v1")
        self.assertEqual(wl["num_kernels"], 3)

        self.assertIn("# Profile Top-3", stdout)          # md also goes to stdout for the agent
        self.assertIn(f"wrote {prefix}.json", stderr)
        self.assertIn("wrote " + wl_path, stderr)
        self.assertIn("(3 kernels)", stderr)

    def test_gzipped_trace_via_cli(self):
        stdout, _ = self._main(["--torch-trace", self._trace_file(gz=True), "--top", "1"])
        self.assertIn("# Profile Top-1", stdout)
        self.assertIn("`gemm_kernel`", stdout)

    def test_stdout_only_when_no_out_prefix(self):
        d = self._dir()
        stdout, stderr = self._main(["--torch-trace", self._trace_file(), "--top", "2"])
        self.assertIn("# Profile Top-2", stdout)
        self.assertEqual(stderr, "")
        self.assertEqual(os.listdir(d), [])

    def test_rocprof_only_reports_hw_time_and_no_shape_cases(self):
        # rocprof alone has no cpu_op links, so every workload case is shape-unknown. The workflow
        # relies on that being visible rather than silently emitting empty dims as real shapes.
        wl_path = os.path.join(self._dir(), "wl.json")
        stdout, _ = self._main(["--rocprof-dir", self._rocprof_dir(), "--top", "2",
                                "--workload-out", wl_path])
        self.assertIn("- source: `rocprofv3`", stdout)
        self.assertIn("total GPU time: **5.00 ms** over 6 launches", stdout)
        with open(wl_path) as fh:
            wl = json.load(fh)
        self.assertEqual(wl["num_kernels"], 2)
        self.assertEqual([k["num_cases"] for k in wl["kernels"]], [0, 0])

    def test_merged_takes_durations_from_rocprof_and_shapes_from_the_trace(self):
        prefix = os.path.join(self._dir(), "merged")
        self._main(["--torch-trace", self._trace_file(), "--rocprof-dir", self._rocprof_dir(),
                    "--out", prefix, "--top", "2", "--conc", "4", "--isl", "512", "--osl", "8"])
        with open(prefix + ".json") as fh:
            summ = json.load(fh)
        self.assertEqual(summ["source"], "merged")
        self.assertEqual(summ["total_gpu_time_ms"], 5.0)              # HW time wins
        self.assertEqual(summ["num_kernel_launches"], 6)
        top = summ["top_kernels"][0]
        self.assertEqual(top["name"], "void gemm_kernel<float>(int) [clone .kd]")
        self.assertEqual(top["shapes"], [[[4, 512], [512, 512]]])     # shapes from the trace
        self.assertEqual(top["phase"], "both")

    def test_merged_workload_prefers_the_trace_agg_for_shapes(self):
        wl_path = os.path.join(self._dir(), "wl.json")
        self._main(["--torch-trace", self._trace_file(), "--rocprof-dir", self._rocprof_dir(),
                    "--workload-out", wl_path, "--top", "1", "--target", "gemm"])
        with open(wl_path) as fh:
            wl = json.load(fh)
        self.assertEqual(wl["kernels"][0]["name"], GEMM)              # trace name, not the HW name
        self.assertEqual(wl["kernels"][0]["cases"][0]["dims"], [[4, 512], [512, 512]])

    def test_empty_rocprof_dir_falls_through_to_the_trace(self):
        stdout, _ = self._main(["--torch-trace", self._trace_file(),
                                "--rocprof-dir", self._dir(), "--top", "1"])
        self.assertIn("- source: `torch-trace`", stdout)

    def test_missing_trace_propagates_the_io_error(self):
        with self.assertRaises(FileNotFoundError):
            self._main(["--torch-trace", os.path.join(self._dir(), "absent.json")])

    def test_module_entrypoint_invokes_main(self):
        # The workflow shells out to `python parse_profile.py ...`, so the __main__ guard has to run.
        trace = self._trace_file()
        out = io.StringIO()
        saved = sys.argv
        sys.argv = ["parse_profile.py", "--torch-trace", trace, "--top", "1"]
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                runpy.run_path(MODULE_PATH, run_name="__main__")
        finally:
            sys.argv = saved
        self.assertIn("# Profile Top-1", out.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
