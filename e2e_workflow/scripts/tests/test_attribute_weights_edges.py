#!/usr/bin/env python3
"""Residual edge/error branches of attribute_weights.py (stdlib only; no pytest needed).

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_attribute_weights_edges.py

test_workload_alignment.py already covers the happy paths of this module. This file picks up what it
leaves: the guard clauses and degraded-input fallbacks. Those matter more than they look, because
attribute_weights decides WHICH kernel is worth optimizing and how an isolated kernel speedup is
projected to end-to-end. When its inputs are odd -- a profile window that saw zero decode time, a
launch name whose grid cannot be parsed, a meta with no regime split -- the module does not fail
loudly, it silently picks a fallback. Whatever that fallback is becomes the weight the optimizer
chases, so each one is pinned here.

The concrete failure mode this guards is the one a recent production run hit: a tuned kernel that was
1.67x faster in isolation moved end-to-end throughput by +0.06%, because the weighting did not
reflect the shapes the live server actually ran.
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(mod_name, filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


aw = _load("attribute_weights", "attribute_weights.py")


def _entry(name, weight, dims=None, count=None):
    """One profiled launch: a specialized kernel name plus its summed case weight."""
    case = {"weight": weight}
    if dims is not None:
        case["dims"] = dims
    if count is not None:
        case["count"] = count
    return {"name": name, "cases": [case]}


# --------------------------------------------------------------------------- #
# estimate_serving_regime_calls -- the analytic denominator
# --------------------------------------------------------------------------- #
class TestServingCallsGuards(unittest.TestCase):
    def test_non_numeric_lengths_yield_no_model_rather_than_crashing(self):
        # The extractor can hand through a string straight from a recipe; returning {} makes the
        # caller skip the serving model instead of aborting the whole attribution.
        self.assertEqual(aw.estimate_serving_regime_calls("not-a-number", 128), {})

    def test_non_numeric_concurrency_yields_no_model(self):
        self.assertEqual(aw.estimate_serving_regime_calls(1024, 128, conc="many"), {})

    def test_float_string_is_also_rejected(self):
        # int("12.5") raises ValueError -- a float-ish string is not silently truncated.
        self.assertEqual(aw.estimate_serving_regime_calls("12.5", 128), {})


# --------------------------------------------------------------------------- #
# attribute_gemm -- degraded classification paths
# --------------------------------------------------------------------------- #
class TestGemmBucketFallback(unittest.TestCase):
    def test_meta_without_a_regime_split_treats_every_bucket_as_prefill(self):
        # No decode_m_buckets/prefill_m_buckets: the flat m_buckets list is the only shape signal,
        # and every bucket is emitted prefill-like rather than being dropped.
        notes = []
        meta = {"a_shape": ["M", 512], "b_shape": [256, 512], "m_buckets": [16, 4096]}
        cases = aw.attribute_gemm(meta, [_entry("k_GRID_MN_8_BLOCK_SIZE_N_128", 10.0)], notes)
        self.assertEqual({c["regime"] for c in cases}, {"prefill"})
        self.assertEqual(sorted(c["m"] for c in cases), [16, 4096])

    def test_a_regime_with_no_buckets_is_skipped_entirely(self):
        # decode has measured time but meta declares no decode bucket -> nothing to emit it onto.
        notes = []
        meta = {"a_shape": ["M", 512], "b_shape": [256, 512], "prefill_m_buckets": [4096]}
        cases = aw.attribute_gemm(meta, [_entry("k_GRID_MN_4_BLOCK_SIZE_N_128", 10.0)], notes)
        self.assertEqual({c["regime"] for c in cases}, {"prefill"})


class TestGemmMedianSplit(unittest.TestCase):
    """When N/BLOCK_SIZE_N cannot be parsed from a launch name, the regime is decided by a median
    split over GRID_MN. This is the least precise path in the module and the one most likely to
    mis-assign time between decode and prefill, so its arithmetic is pinned exactly."""

    META = {"a_shape": ["M", 512], "b_shape": [None, 512],
            "decode_m_buckets": [8], "prefill_m_buckets": [4096]}

    def test_unparseable_launches_split_at_the_grid_median(self):
        notes = []
        entries = [_entry("k_GRID_MN_2", 1.0), _entry("k_GRID_MN_4", 2.0),
                   _entry("k_GRID_MN_64", 4.0)]
        cases = aw.attribute_gemm(dict(self.META), entries, notes)
        by = {c["regime"]: c["weight"] for c in cases}
        # median of [2,4,64] is 4 -> grids <= 4 are decode (1.0+2.0), the rest prefill (4.0).
        self.assertEqual(by["decode"], 3.0)
        self.assertEqual(by["prefill"], 4.0)

    def test_the_median_split_is_announced_in_the_notes(self):
        notes = []
        aw.attribute_gemm(dict(self.META), [_entry("k_GRID_MN_2", 1.0)], notes)
        self.assertTrue(any("GRID_MN median split" in n for n in notes))
        self.assertTrue(any("N/BLOCK_N not parseable" in n for n in notes))

    def test_launches_with_no_grid_at_all_count_as_prefill(self):
        # grid parses to 0/None -> `grid and grid <= med` is False -> prefill side.
        notes = []
        cases = aw.attribute_gemm(dict(self.META), [_entry("kernel_without_fields", 5.0)], notes)
        by = {c["regime"]: c["weight"] for c in cases}
        self.assertEqual(by["prefill"], 5.0)
        self.assertEqual(by["decode"], 0.0)


# --------------------------------------------------------------------------- #
# size / split priors
# --------------------------------------------------------------------------- #
class TestCaseSize(unittest.TestCase):
    def test_symbolic_dims_fall_back_to_one(self):
        # A meta case whose shapes are still symbolic ("M") has no usable size proxy; 1 keeps it in
        # the split with minimal share instead of removing it or raising.
        self.assertEqual(aw._case_size([["M", 512], [None, 512]]), 1)

    def test_no_dims_at_all_falls_back_to_one(self):
        self.assertEqual(aw._case_size([]), 1)

    def test_first_fully_integer_operand_wins(self):
        self.assertEqual(aw._case_size([["M", 4], [8, 16]]), 128)


class TestMembersSplit(unittest.TestCase):
    def test_zero_total_size_splits_evenly(self):
        # Every member had a symbolic shape, so size-proportional weighting is undefined; an even
        # split keeps all cases visible rather than zeroing them.
        members = [{"size": 0}, {"size": 0}, {"size": 0}]
        fracs, used_count = aw._members_split(members)
        self.assertEqual([f for _, f in fracs], [1 / 3, 1 / 3, 1 / 3])
        self.assertFalse(used_count)

    def test_size_proportional_when_sizes_are_known(self):
        fracs, used_count = aw._members_split([{"size": 1}, {"size": 3}])
        self.assertEqual([f for _, f in fracs], [0.25, 0.75])
        self.assertFalse(used_count)


# --------------------------------------------------------------------------- #
# _distribute / cross-check
# --------------------------------------------------------------------------- #
class TestDistributeZeroTime(unittest.TestCase):
    def test_a_regime_meta_declares_but_the_profile_timed_at_zero_warns_and_keeps_it(self):
        # This is the graph-hidden-decode case. Dropping it would let the optimizer conclude decode
        # is free; emitting weight 0 with a loud note keeps it benchmarkable and floorable.
        notes = []
        mcases = [{"name": "d", "dims": [[8, 512]], "dtypes": [], "regime": "decode", "size": 4096}]
        out = aw._distribute(mcases, {"decode": 0.0}, {}, notes)
        self.assertEqual(out[0]["weight"], 0.0)
        self.assertEqual(out[0]["weight_source"], "prior")
        self.assertTrue(any("ZERO profiled time" in n for n in notes))

    def test_an_untagged_regime_is_not_warned_about(self):
        # regime "" is "unknown", not "declared but unmeasured" -- warning on it would be noise.
        notes = []
        mcases = [{"name": "c", "dims": [[8, 512]], "dtypes": [], "regime": "", "size": 4096}]
        out = aw._distribute(mcases, {}, {}, notes)
        self.assertEqual(out[0]["weight"], 0.0)
        self.assertEqual(notes, [])


class TestCountTimeCrosscheck(unittest.TestCase):
    def test_a_large_count_versus_time_divergence_is_surfaced(self):
        # Capture says decode is most of the CALLS, the profile says prefill is most of the TIME.
        # The weight uses time; the note tells a reader the two signals disagree.
        notes = []
        mcases = [{"regime": "decode", "count": 100}, {"regime": "prefill", "count": 1}]
        aw._count_time_crosscheck(mcases, {"decode": 1.0, "prefill": 99.0}, notes)
        self.assertTrue(any("CROSS-CHECK" in n for n in notes))
        self.assertTrue(any("weight uses TIME" in n for n in notes))

    def test_agreeing_signals_produce_no_note(self):
        notes = []
        mcases = [{"regime": "decode", "count": 50}, {"regime": "prefill", "count": 50}]
        aw._count_time_crosscheck(mcases, {"decode": 50.0, "prefill": 50.0}, notes)
        self.assertEqual(notes, [])

    def test_absent_counts_skip_the_crosscheck(self):
        notes = []
        aw._count_time_crosscheck([{"regime": "decode"}], {"decode": 5.0}, notes)
        self.assertEqual(notes, [])


# --------------------------------------------------------------------------- #
# case-based op_kinds with no meta cases -> pass-through
# --------------------------------------------------------------------------- #
class TestPassthroughDegradation(unittest.TestCase):
    def test_attn_without_meta_cases_passes_the_profiled_shapes_through(self):
        notes = []
        out = aw.attribute_attn({}, [_entry("attn_fwd", 7.0, dims=[[32, 128]])], notes)
        self.assertEqual([c["weight"] for c in out], [7.0])
        self.assertEqual([c["weight_source"] for c in out], ["trace"])

    def test_generic_without_meta_cases_passes_the_profiled_shapes_through(self):
        notes = []
        out = aw.attribute_generic({}, [_entry("rmsnorm", 3.0, dims=[[16, 64]])], notes)
        self.assertEqual([c["weight"] for c in out], [3.0])

    def test_no_meta_cases_and_no_profile_is_an_explicit_note_not_a_crash(self):
        notes = []
        self.assertEqual(aw.attribute_generic({}, [], notes), [])
        self.assertTrue(any("nothing to weight" in n for n in notes))


class TestBestShapeMatch(unittest.TestCase):
    def test_empty_dims_match_nothing(self):
        self.assertIsNone(aw._best_shape_match([], [{"dims": [[8, 16]]}]))

    def test_leading_none_operand_matches_nothing(self):
        self.assertIsNone(aw._best_shape_match([[]], [{"dims": [[8, 16]]}]))


# --------------------------------------------------------------------------- #
# decode floor guards
# --------------------------------------------------------------------------- #
class TestAutoDecodeFloor(unittest.TestCase):
    CASES = [{"regime": "decode", "weight": 1.0}, {"regime": "prefill", "weight": 9.0}]

    def test_no_analytic_decode_calls_means_no_floor(self):
        self.assertEqual(aw._auto_decode_floor(self.CASES, {"decode": 0}, 0.0, []), 0.0)

    def test_no_decode_case_in_meta_means_no_floor(self):
        # The kernel does not run in decode at all, so flooring would invent weight for a regime it
        # never executes -- exactly the mistake the served-regimes gate exists to prevent.
        cases = [{"regime": "prefill", "weight": 9.0}]
        self.assertEqual(aw._auto_decode_floor(cases, {"decode": 128}, 0.0, []), 0.0)

    def test_an_already_healthy_decode_share_is_left_alone(self):
        cases = [{"regime": "decode", "weight": 5.0}, {"regime": "prefill", "weight": 5.0}]
        self.assertEqual(aw._auto_decode_floor(cases, {"decode": 128}, 0.0, []), 0.0)

    def test_an_under_captured_decode_share_is_floored_and_explained(self):
        notes = []
        floor = aw._auto_decode_floor(self.CASES, {"decode": 128}, 0.0, notes)
        self.assertEqual(floor, aw._DECODE_AUTOFLOOR)
        self.assertTrue(any("auto decode-floor" in n for n in notes))

    def test_an_explicit_floor_above_the_automatic_one_wins(self):
        self.assertEqual(aw._auto_decode_floor(self.CASES, {"decode": 128}, 0.5, []), 0.5)


class TestApplyRegimeFloorGuards(unittest.TestCase):
    def test_a_single_regime_is_left_untouched(self):
        # With one regime the floor is meaningless: that regime already holds 100% of the weight.
        cases = [{"regime": "prefill", "weight": 1.0, "m": 4096}]
        aw._apply_regime_floor(cases, 0.3, [])
        self.assertEqual(cases[0]["weight"], 1.0)

    def test_regimes_already_above_the_floor_are_left_untouched(self):
        cases = [{"regime": "decode", "weight": 5.0, "m": 8},
                 {"regime": "prefill", "weight": 5.0, "m": 4096}]
        aw._apply_regime_floor(cases, 0.3, [])
        self.assertEqual([c["weight"] for c in cases], [5.0, 5.0])

    def test_an_impossible_floor_is_skipped_with_a_note(self):
        # 0.6 x 2 floored regimes > 1.0 total weight -- unsatisfiable, so it is refused loudly
        # rather than producing weights that sum above the total.
        notes = []
        cases = [{"regime": "decode", "weight": 0.0, "m": 8},
                 {"regime": "prefill", "weight": 0.0, "m": 4096}]
        aw._apply_regime_floor(cases, 0.6, notes)
        self.assertTrue(any("skipped" in n for n in notes))


# --------------------------------------------------------------------------- #
# regime guards -- the "isolated win, e2e loss" pre-checks
# --------------------------------------------------------------------------- #
class TestRegimeWarnings(unittest.TestCase):
    def test_fp8_kv_cache_warns_for_attention_kernels(self):
        # A kernel hardcoding a bf16 KV stride faults when the server serves fp8 KV; the oracle must
        # be built on the same layout.
        notes = []
        msg = aw._regime_warnings({"kv_cache_dtype": "fp8"}, "attn", [], 100.0, 2.0, notes)
        self.assertIn("kv-cache-dtype=fp8", msg)
        self.assertTrue(notes)

    def test_fp8_kv_cache_is_not_warned_about_for_a_gemm(self):
        self.assertEqual(aw._regime_warnings({"kv_cache_dtype": "fp8"}, "gemm", [], 100.0, 2.0, []), "")

    def test_a_near_zero_gpu_share_flags_the_seam_as_not_live(self):
        notes = []
        msg = aw._regime_warnings({}, "gemm", [_entry("k", 1.0)], 0.4, 2.0, notes)
        self.assertIn("probably NOT the live kernel", msg)

    def test_no_profile_entries_means_no_live_seam_warning(self):
        # Without entries there is no measured share to judge, so the guard stays silent instead of
        # firing on a 0% that only means "not measured".
        self.assertEqual(aw._regime_warnings({}, "gemm", [], 0.0, 2.0, []), "")


# --------------------------------------------------------------------------- #
# main() -- op_kind dispatch and the kernel->regime gate
# --------------------------------------------------------------------------- #
class TestMainDispatchAndGate(unittest.TestCase):
    """The served-regimes gate is the fix for the failure this whole module exists to prevent:
    weighting decode shapes onto a kernel that only ever runs in prefill, optimizing that, and
    watching an isolated speedup turn into an end-to-end regression."""

    def _run(self, meta, profile, *argv):
        d = tempfile.mkdtemp()
        mp, pp, op = (os.path.join(d, n) for n in ("meta.json", "prof.json", "out.json"))
        for path, obj in ((mp, meta), (pp, profile)):
            with open(path, "w") as fh:
                json.dump(obj, fh)
        args = ["attribute_weights.py", "--meta", mp, "--profile-weights", pp, "--out", op]
        args.extend(argv)
        with mock.patch.object(sys, "argv", args), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            aw.main()
        with open(op) as fh:
            return json.load(fh)

    @staticmethod
    def _prof(kernels):
        return {"kernels": kernels}

    @staticmethod
    def _gemm_meta(short_name="my_gemm"):
        return {"op_kind": "gemm", "short_name": short_name, "a_shape": ["M", 512],
                "b_shape": [256, 512], "decode_m_buckets": [8], "prefill_m_buckets": [4096]}

    @staticmethod
    def _case_meta(op_kind, short_name="my_kernel"):
        return {"op_kind": op_kind, "short_name": short_name, "cases": [
            {"sig": "d", "input_shapes": [[8, 512]], "regime": "decode"},
            {"sig": "p", "input_shapes": [[4096, 512]], "regime": "prefill"}]}

    def test_a_name_that_matches_no_profile_entry_yields_prior_only_weights(self):
        out = self._run(self._gemm_meta(), self._prof([{"name": "some_other_kernel", "cases": []}]))
        self.assertIn("weights are prior only", out["notes"])

    def test_op_kind_moe_routes_through_the_gemm_engine_with_a_confidence_note(self):
        out = self._run({"op_kind": "moe", "short_name": "my_moe", "a_shape": ["M", 512],
                         "b_shape": [256, 512], "prefill_m_buckets": [4096]},
                        self._prof([{"name": "my_moe_GRID_MN_64_BLOCK_SIZE_N_128",
                                     "cases": [{"weight": 10.0}]}]))
        self.assertIn("op_kind=moe", out["notes"])
        self.assertIn("lower-confidence", out["notes"])

    def test_op_kind_attn_uses_the_case_based_engine(self):
        out = self._run(self._case_meta("attn", "my_attn"),
                        self._prof([{"name": "my_attn_fwd", "cases": [{"weight": 10.0}]}]))
        self.assertEqual(out["op_kind"], "attn")
        self.assertEqual({c["regime"] for c in out["cases"]}, {"decode", "prefill"})

    def test_an_unrecognized_op_kind_falls_back_to_the_generic_engine(self):
        out = self._run(self._case_meta("rmsnorm", "my_norm"),
                        self._prof([{"name": "my_norm_kernel", "cases": [{"weight": 10.0}]}]))
        self.assertEqual(out["num_cases"], 2)

    def test_the_gate_defaults_to_the_phase_measured_in_the_trace(self):
        # No --served-regimes: parse_profile's per-kernel served_regimes is authoritative, so the
        # gate is data-driven instead of depending on the extractor remembering a flag.
        out = self._run(self._case_meta("attn", "my_attn"),
                        self._prof([{"name": "my_attn_fwd", "served_regimes": ["prefill"],
                                     "cases": [{"weight": 10.0}]}]))
        self.assertEqual(out["served_regimes"], ["prefill"])
        self.assertIn("derived from trace phase", out["notes"])

    def test_an_explicit_gate_drops_cases_in_unserved_regimes(self):
        out = self._run(self._case_meta("attn", "my_attn"),
                        self._prof([{"name": "my_attn_fwd", "cases": [{"weight": 10.0}]}]),
                        "--served-regimes", "prefill")
        self.assertEqual({c["regime"] for c in out["cases"]}, {"prefill"})
        self.assertIn("dropped 1 case(s)", out["notes"])

    def test_an_explicit_gate_wins_over_the_trace_derived_one(self):
        out = self._run(self._case_meta("attn", "my_attn"),
                        self._prof([{"name": "my_attn_fwd", "served_regimes": ["decode"],
                                     "cases": [{"weight": 10.0}]}]),
                        "--served-regimes", "prefill")
        self.assertEqual(out["served_regimes"], ["prefill"])
        self.assertNotIn("derived from trace phase", out["notes"])

    def test_a_gate_matching_every_case_drops_nothing_and_stays_quiet(self):
        out = self._run(self._case_meta("attn", "my_attn"),
                        self._prof([{"name": "my_attn_fwd", "cases": [{"weight": 10.0}]}]),
                        "--served-regimes", "prefill,decode")
        self.assertEqual(out["num_cases"], 2)
        self.assertNotIn("dropped", out["notes"])

    def test_a_gate_that_would_drop_everything_is_refused_rather_than_obeyed(self):
        # An empty case set is useless and almost certainly a mis-set flag, so the original set is
        # kept and the contradiction is reported instead of silently producing nothing.
        meta = {"op_kind": "attn", "short_name": "my_attn", "cases": [
            {"sig": "p", "input_shapes": [[4096, 512]], "regime": "prefill"}]}
        out = self._run(meta, self._prof([{"name": "my_attn_fwd", "cases": [{"weight": 10.0}]}]),
                        "--served-regimes", "decode")
        self.assertEqual(out["num_cases"], 1)
        self.assertIn("would drop ALL cases", out["notes"])

    def test_an_ungated_regime_specific_kernel_name_raises_a_loud_advisory(self):
        # "_fwd" plus cases spanning both regimes is the exact shape of the bug: a prefill wrapper
        # with a separate decode kernel, weighted as if it served both.
        out = self._run(self._case_meta("attn", "attn_fwd_kernel"),
                        self._prof([{"name": "attn_fwd_kernel", "cases": [{"weight": 10.0}]}]))
        self.assertIn("served-regimes NOT set", out["notes"])
        self.assertIn("may be weighted onto a kernel that does not run it", out["notes"])

    def test_a_single_regime_kernel_without_a_gate_is_not_warned_about(self):
        meta = {"op_kind": "attn", "short_name": "attn_fwd_kernel", "cases": [
            {"sig": "p", "input_shapes": [[4096, 512]], "regime": "prefill"}]}
        out = self._run(meta, self._prof([{"name": "attn_fwd_kernel", "cases": [{"weight": 10.0}]}]))
        self.assertNotIn("served-regimes NOT set", out["notes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
