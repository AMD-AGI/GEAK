#!/usr/bin/env python3
"""Regression tests for machine-verified live kernel selection."""

import importlib.util
import json
import os
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location(
    "kernel_selection", os.path.join(SCRIPTS, "kernel_selection.py"))
ks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ks)

# parse_profile owns the display-name limit that canonical_kernel_name's prefix rule depends on.
PP_SPEC = importlib.util.spec_from_file_location(
    "parse_profile", os.path.join(SCRIPTS, "parse_profile.py"))
pp = importlib.util.module_from_spec(PP_SPEC)
PP_SPEC.loader.exec_module(pp)

TARGET = (
    "vllm.v1.attention.ops.chunked_prefill_paged_decode:"
    "chunked_prefill_paged_decode"
)
KERNEL = "kernel_paged_attention_2d"


def trace(kernel=KERNEL, under_target=True):
    marker_start = 100
    kernel_start = 120 if under_target else 250
    events = [
        {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + TARGET,
         "ph": "X", "pid": 1, "tid": 2, "ts": 90, "dur": 1},
        {"cat": "cpu_op", "name": ks.MARKER_PREFIX + TARGET,
         "ph": "X", "pid": 1, "tid": 2, "ts": marker_start, "dur": 100},
        {"cat": "kernel", "name": f"void vllm::{kernel}<bf16>(int)",
         "ph": "X", "ts": kernel_start, "dur": 25, "args": {"External id": 7}},
    ]
    if under_target:
        events.insert(1, {"cat": "cpu_op", "name": "launch", "ph": "X",
                          "pid": 1, "tid": 2, "ts": 110, "dur": 5,
                          "args": {"External id": 7}})
    return events


class TestCallableSpec(unittest.TestCase):
    def test_only_exact_machine_specs_are_accepted(self):
        self.assertTrue(ks.valid_callable_spec(TARGET))
        self.assertFalse(ks.valid_callable_spec(TARGET + " -> inner kernel"))
        self.assertFalse(ks.valid_callable_spec("vllm/path.py:launcher"))


class TestKernelMatching(unittest.TestCase):
    def test_demangled_kernel_matches_profile_identity(self):
        self.assertTrue(ks.kernel_matches(
            KERNEL, "void vllm::kernel_paged_attention_2d<bf16>(int)"))
        self.assertFalse(ks.kernel_matches(KERNEL, "unrelated_attention_kernel"))

    def test_nested_template_arguments_reduce_to_the_bare_kernel_name(self):
        """Demangled C++ symbols nest their template arguments, and the argument list can carry both
        '::' and '(' -- the two delimiters the rest of the canonicalization splits on. Leaving any of
        it behind produces a token that never matches the profile's own name for the same kernel."""
        for decorated in (
            "at::native::vectorized_elementwise_kernel<4, at::native::AddFunctor<float>, "
            "at::detail::Array<char*, 3> >(int, float)",
            "void ns::vectorized_elementwise_kernel<std::pair<int, float>>(void*)",
            "vectorized_elementwise_kernel<float> [clone .isra.0]",
        ):
            with self.subTest(decorated=decorated):
                self.assertEqual(ks.canonical_kernel_name(decorated),
                                 "vectorized_elementwise_kernel")
                self.assertTrue(ks.kernel_matches("vectorized_elementwise_kernel", decorated))

    def test_a_symbol_elided_mid_template_answers_with_the_name_not_a_fragment(self):
        """Profile artifacts store long kernel names elided mid-template, so the opener never closes
        and no amount of balanced stripping removes it. Reading the last whitespace token then lifts
        a fragment out of the template arguments -- this real elision answered 'gpu_k' -- and states
        it as confidently as a real name, which matches the wrong kernel instead of refusing."""
        self.assertEqual(
            ks.canonical_kernel_name(
                "void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_k..."),
            "elementwise_kernel_manual_unroll")
        self.assertEqual(
            ks.canonical_kernel_name("void aiter::grouped_topk_kernel<c10::BFloat16, float "
                                     "__vector(4), true>(c10::BFloat1"),
            "grouped_topk_kernel")
        self.assertEqual(ks.canonical_kernel_name("gemm_kernel<half"), "gemm_kernel")
        self.assertEqual(ks.canonical_kernel_name("<>"), "")

    def test_an_elided_name_still_matches_the_full_symbol_it_was_cut_from(self):
        elided = "_ZN7ck_tile6kentryILi2ENS_15MoeFlatmmKernelINS_33GemmSpatiallyLocalTilePart..."
        full = ("_ZN7ck_tile6kentryILi2ENS_15MoeFlatmmKernelINS_33GemmSpatiallyLocalTilePartitioner"
                "INS_13TileGemmShapeINS_8sequenceIJLi16")
        self.assertTrue(ks.kernel_matches(elided, full))


# Symbols lifted verbatim from ROCm captures under /shared_nfs/hyperloom-claw. They are here because
# hand-written C++ never produced the shapes that broke this: an unnamed namespace putting a '(' in
# front of the kernel, and a bare kernel with no namespace at all to separate it from 'void'.
REAL_ROCM_SYMBOLS = {
    "void (anonymous namespace)::kda_packed_decode_kernel<8, false>"
    "((anonymous namespace)::KdaPackedDecodeParams)": "kda_packed_decode_kernel",
    "void aiter::greedy_sample_kernel<float, 1024, 16>(float const*, int*, int, int)":
        "greedy_sample_kernel",
    "void paged_attention_ll4mi_QKV_mfma16_kernel<0, __hip_bfloat16, "
    "(vllm::Fp8KVCacheDataType)0, 256>(float*, int*)": "paged_attention_ll4mi_qkv_mfma16_kernel",
    "void wvSplitKrc_<__hip_bfloat16, 8, 4>(void*, int)": "wvsplitkrc_",
    "_ZN5aiter24add_rmsnorm_quant_kernelIDF16bLi256EEEvPT_i": "_zn5aiter24add_rmsnorm_quant_kernelidf16bli256eeevpt_i",
    "reshape_and_cache_shuffle_5d": "reshape_and_cache_shuffle_5d",
}


class TestRealRocmSymbolsCanonicalizeToTheirKernel(unittest.TestCase):
    def test_each_symbol_reduces_to_the_kernel_it_names(self):
        for symbol, token in REAL_ROCM_SYMBOLS.items():
            with self.subTest(symbol=symbol[:48]):
                self.assertEqual(ks.canonical_kernel_name(symbol), token)

    def test_an_unnamed_namespace_does_not_collapse_the_symbol_to_its_return_type(self):
        """ROCm spells the unnamed namespace '(anonymous namespace)', which puts a parenthesis BEFORE
        the kernel. Cutting the symbol at its first '(' left the return type as the whole token, so
        every such kernel reduced to 'void' -- and two unrelated ones then certified each other,
        which is exactly the unearned credit this verdict exists to refuse."""
        one = "void (anonymous namespace)::clamp_position_kernel<long>(long*, long)"
        two = "void (anonymous namespace)::kda_packed_decode_kernel<8, false>(int)"
        self.assertEqual(ks.canonical_kernel_name(one), "clamp_position_kernel")
        self.assertEqual(ks.canonical_kernel_name(two), "kda_packed_decode_kernel")
        self.assertFalse(ks.kernel_matches(one, two))
        self.assertFalse(ks.kernel_matches(two, one))

    def test_a_kernel_with_no_namespace_is_not_fused_with_its_return_type(self):
        """Stripping separators rather than splitting on them glued 'void' onto any kernel that had
        no '::' to fall back on, so the bare name never matched its own decorated spelling."""
        bare = "paged_attention_ll4mi_QKV_mfma4_kernel"
        decorated = f"void {bare}<__hip_bfloat16, 128, 256>(float*, int)"
        self.assertEqual(ks.canonical_kernel_name(decorated), bare.lower())
        self.assertTrue(ks.kernel_matches(bare, decorated))

    def test_a_name_truncated_by_the_display_limit_still_matches_its_full_symbol(self):
        """parse_profile.short_name caps display names, and mangled and Tensile symbols run well past
        the cap. The truncated name ends mid-token, so no word boundary can follow it -- without an
        explicit prefix rule our own shortening reads as a different kernel."""
        symbol = "_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi16ELb1ELb0ELb1ELi1EEEvPT0_i"
        declared = symbol[:ks.SHORT_NAME_LIMIT]
        self.assertGreater(len(symbol), ks.SHORT_NAME_LIMIT)
        self.assertTrue(ks.kernel_matches(declared, symbol))
        self.assertFalse(ks.kernel_matches(declared[:ks.SHORT_NAME_LIMIT - 1], symbol))

    def test_a_name_embedded_inside_another_is_a_different_kernel(self):
        """The first two pairs are real neighbours inside one sglang capture. Accepting a name
        because it appears inside the other certified _fwd_kernel as _fwd_kernel_stage2 -- a
        different kernel, which is precisely the credit this verdict exists to withhold."""
        for a, b in (
            ("_fwd_kernel", "_fwd_kernel_stage2"),
            ("reshape_and_cache_kernel", "reshape_and_cache_kernel_flash"),
            ("gemm_kernel", "fused_gemm_kernel_v2"),
            ("gemm", "gemm_kernel_v2"),
            ("attention_kernel", "attention_kernelbackward"),
        ):
            with self.subTest(a=a, b=b):
                self.assertFalse(ks.kernel_matches(a, b))
                self.assertFalse(ks.kernel_matches(b, a))

    def test_two_instantiations_of_one_template_are_not_one_kernel(self):
        """The base token drops template arguments so a bare declared name can match its decorated
        spelling. When both sides carry those arguments the information is present on both, and one
        capture here held 20 distinct kernels whose only difference was the functor."""
        fill = ("void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, "
                "std::array<char*, 1ul> >(int, at::native::FillFunctor<bool>)")
        power = ("void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous "
                 "namespace)::pow_tensor_scalar_kernel_impl<float>, std::array<char*, 2ul> >(int)")
        self.assertEqual(ks.canonical_kernel_name(fill), ks.canonical_kernel_name(power))
        self.assertFalse(ks.kernel_matches(fill, power))
        self.assertTrue(ks.kernel_matches(fill, fill))

    def test_a_bare_declared_name_still_matches_a_templated_symbol(self):
        """The tightening above applies only when BOTH sides carry template arguments. A head that
        declares the bare name has none to compare, and refusing it would reject every real head."""
        self.assertTrue(ks.kernel_matches(
            "vectorized_elementwise_kernel",
            "void at::native::vectorized_elementwise_kernel<4, AddFunctor<float> >(int)"))

    def test_arguments_are_read_past_the_return_type_and_namespace(self):
        """Only one side spells the return type and namespace. This pair is a real head row whose
        stored short_name had been cut inside the ARGUMENT list while its template stayed whole --
        comparing the symbols entire refused a kernel against itself."""
        self.assertTrue(ks.kernel_matches(
            "clamp_position_kernel<long>(long*, long const*, unsigned lon",
            "void (anonymous namespace)::clamp_position_kernel<long>(long*, long const*, unsigned "
            "long, int)"))

    def test_one_kernel_matches_across_builds_that_name_its_namespace_differently(self):
        """Both spellings occur across the captures: one build leaves the namespace unnamed, another
        declares it. Same kernel, same instantiation."""
        self.assertTrue(ks.kernel_matches(
            "void (anonymous namespace)::store_kvcache<512l, 512l, 1, false, long>(int)",
            "void sglang::store_kvcache<512l, 512l, 1, false, long>(sglang::StoreKVCacheParams)"))

    def test_an_elided_argument_list_still_has_to_agree_as_far_as_it_is_spelled(self):
        """Artifacts elide long names mid-template, so neither list is complete. Skipping the check
        entirely would let every elided instantiation of one template certify the others -- these two
        differ only in the visible '128, 4' against '128, 8'."""
        four = "void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_k..."
        eight = "void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_k..."
        self.assertEqual(ks.canonical_kernel_name(four), ks.canonical_kernel_name(eight))
        self.assertFalse(ks.kernel_matches(four, eight))
        self.assertTrue(ks.kernel_matches(four, four))

    def test_a_separator_cannot_make_one_argument_list_a_prefix_of_another(self):
        """Deleting separators instead of folding them read '<128, 4,' as a prefix of '<128, 48,'."""
        self.assertFalse(ks.kernel_matches("k<128, 4, at::native::gpu_k",
                                           "k<128, 48, at::native::gpu_k"))

    def test_the_display_limit_agrees_with_the_module_that_applies_it(self):
        """The prefix rule above is only sound while both sides mean the same number of characters."""
        self.assertEqual(ks.SHORT_NAME_LIMIT, pp.SHORT_NAME_LIMIT)


class TestTheSharedFixtureHoldsOnThisSide(unittest.TestCase):
    """kernel_symbols.json is the one place both canonicalizers are pinned. The node half of this is
    scripts/test_kernel_canonicalization_parity.js; a rule added here without adding it there is
    exactly the drift that lets a kernel pass the JS gate and be refused by this verdict."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "kernel_symbols.json")) as handle:
            cls.fixture = json.load(handle)

    def test_every_symbol_reduces_to_its_recorded_token(self):
        for case in self.fixture["canonical"]:
            with self.subTest(symbol=case["symbol"][:48]):
                self.assertEqual(ks.canonical_kernel_name(case["symbol"]), case["canonical"])

    def test_every_recorded_pair_gets_the_recorded_verdict(self):
        for case in self.fixture["matches"]:
            with self.subTest(why=case["why"]):
                self.assertEqual(ks.kernel_matches(case["a"], case["b"]), case["match"])
                self.assertEqual(ks.kernel_matches(case["b"], case["a"]), case["match"])

    def test_the_fixture_still_covers_symbols_taken_from_real_captures(self):
        """Hand-written C++ is what let the unnamed-namespace and bare-kernel defects through."""
        self.assertGreaterEqual(sum(1 for c in self.fixture["canonical"] if c.get("real")), 5)


class TestSelectionVerdict(unittest.TestCase):
    def meta(self, calls=7, target=TARGET):
        module, attr = target.split(":", 1)
        return {"module": module, "attr": attr, "total_calls_observed": calls}

    def test_0720_live_inner_launcher_is_selection_success(self):
        verdict = ks.verify(TARGET, KERNEL, self.meta(), trace())
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["matched_kernel_calls"], 1)
        self.assertEqual(verdict["failed"], [])

    def test_capture_and_seam_markers_count_one_logical_call(self):
        """capture_shapes and seam_trace both emit GEAK_TARGET for the selected callable. The trace
        therefore nests the same marker inside itself, but the report must count the invocation once."""
        events = trace()
        events.insert(2, {
            "cat": "cpu_op", "name": ks.MARKER_PREFIX + TARGET,
            "ph": "X", "pid": 1, "tid": 2, "ts": 105, "dur": 80,
        })
        verdict = ks.verify(TARGET, KERNEL, self.meta(calls=1), events)
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(verdict["target_marker_calls"], 1)
        self.assertEqual(verdict["matched_kernel_calls"], 1)

    def test_nested_markers_on_other_threads_and_later_calls_stay_distinct(self):
        marker = ks.MARKER_PREFIX + TARGET
        spans = ks._outermost_spans([
            (100.0, 200.0, {"pid": 1, "tid": 2, "name": marker}),
            (110.0, 190.0, {"pid": 1, "tid": 2, "name": marker}),
            (120.0, 180.0, {"pid": 1, "tid": 3, "name": marker}),
            (300.0, 320.0, {"pid": 1, "tid": 2, "name": marker}),
        ])
        self.assertEqual([(start, end) for start, end, _ in spans],
                         [(100.0, 200.0), (120.0, 180.0), (300.0, 320.0)])

    def test_kernel_seen_elsewhere_is_not_enough(self):
        verdict = ks.verify(TARGET, KERNEL, self.meta(), trace(under_target=False))
        self.assertFalse(verdict["ok"])
        self.assertIn("device_kernel_not_under_target", verdict["failed"])

    def test_async_kernel_after_marker_is_linked_by_external_id(self):
        events = [
            {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + TARGET,
             "ph": "X", "pid": 1, "tid": 2, "ts": 90, "dur": 1},
            {"cat": "cpu_op", "name": ks.MARKER_PREFIX + TARGET,
             "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 40},
            {"cat": "cpu_op", "name": "launch", "ph": "X", "pid": 1, "tid": 2,
             "ts": 110, "dur": 5,
             "args": {"External id": 42}},
            {"cat": "kernel", "name": KERNEL, "ph": "X", "ts": 300, "dur": 20,
             "args": {"External id": 42}},
        ]
        verdict = ks.verify(TARGET, KERNEL, self.meta(), events)
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(verdict["correlated_external_ids"], 1)

    def test_triton_kernel_without_external_id_is_linked_by_launch_correlation(self):
        # ROCm/kineto emits Triton device rows (hipModuleLaunchKernel) with only `correlation`;
        # `External id` is absent, so the External-id bridge alone reports a false negative.
        events = [
            {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + TARGET,
             "ph": "X", "pid": 1, "tid": 2, "ts": 90, "dur": 1},
            {"cat": "cpu_op", "name": ks.MARKER_PREFIX + TARGET,
             "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 40},
            {"cat": "cuda_runtime", "name": "hipModuleLaunchKernel", "ph": "X",
             "pid": 1, "tid": 2, "ts": 110, "dur": 5, "args": {"correlation": 25}},
            {"cat": "kernel", "name": KERNEL, "ph": "X", "pid": 2, "tid": 0,
             "ts": 300, "dur": 20, "args": {"correlation": 25, "stream": 0}},
        ]
        verdict = ks.verify(TARGET, KERNEL, self.meta(), events)
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(verdict["matched_kernel_calls"], 1)
        self.assertEqual(verdict["correlated_launch_correlations"], 1)

    def test_launch_correlation_outside_the_marker_span_is_not_enough(self):
        events = [
            {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + TARGET,
             "ph": "X", "pid": 1, "tid": 2, "ts": 90, "dur": 1},
            {"cat": "cpu_op", "name": ks.MARKER_PREFIX + TARGET,
             "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 40},
            {"cat": "cuda_runtime", "name": "hipModuleLaunchKernel", "ph": "X",
             "pid": 1, "tid": 2, "ts": 500, "dur": 5, "args": {"correlation": 25}},
            {"cat": "kernel", "name": KERNEL, "ph": "X", "pid": 2, "tid": 0,
             "ts": 600, "dur": 20, "args": {"correlation": 25, "stream": 0}},
        ]
        verdict = ks.verify(TARGET, KERNEL, self.meta(), events)
        self.assertFalse(verdict["ok"])
        self.assertIn("device_kernel_not_under_target", verdict["failed"])

    def test_correlation_ids_do_not_collide_across_merged_call_traces(self):
        # Every per-call trace restarts `correlation` at 1, so an unprefixed merge would let call-2's
        # kernel be attributed to call-1's in-span launch.
        def one(marked, corr, kernel_ts):
            evs = [
                {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + TARGET,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 90, "dur": 1},
                {"cat": "kernel", "name": KERNEL, "ph": "X", "pid": 2, "tid": 0,
                 "ts": kernel_ts, "dur": 20, "args": {"correlation": corr}},
            ]
            if marked:
                evs += [
                    {"cat": "cpu_op", "name": ks.MARKER_PREFIX + TARGET,
                     "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 40},
                    {"cat": "cuda_runtime", "name": "hipModuleLaunchKernel", "ph": "X",
                     "pid": 1, "tid": 2, "ts": 110, "dur": 5, "args": {"correlation": corr}},
                ]
            return {"traceEvents": evs}

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            # call-1: marked launch, correlation 1.  call-2: NO marker at all, correlation 1 again.
            for i, doc in enumerate((one(True, 1, 300), one(False, 1, 900))):
                p = os.path.join(tmp, "t%d.json" % i)
                json.dump(doc, open(p, "w"))
                paths.append(p)
            merged = ks.merge_process_traces(paths)
            verdict = ks.verify(TARGET, KERNEL, self.meta(), merged)
            self.assertEqual(verdict["matched_kernel_calls"], 1, verdict)

    def test_0802_outer_wrapper_fails_when_0720_launcher_is_marked(self):
        outer = "vllm.v1.attention.layer:unified_attention_with_output"
        inner = TARGET
        events = [
            {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + outer,
             "ph": "X", "pid": 1, "tid": 2, "ts": 80, "dur": 1},
            {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + inner,
             "ph": "X", "pid": 1, "tid": 2, "ts": 82, "dur": 1},
            {"cat": "cpu_op", "name": ks.MARKER_PREFIX + outer,
             "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 100},
            {"cat": "cpu_op", "name": ks.MARKER_PREFIX + inner,
             "ph": "X", "pid": 1, "tid": 2, "ts": 120, "dur": 50},
            {"cat": "cpu_op", "name": "launch", "ph": "X", "pid": 1, "tid": 2,
             "ts": 130, "dur": 5, "args": {"External id": 9}},
            {"cat": "kernel", "name": KERNEL, "ph": "X", "ts": 300, "dur": 20,
             "args": {"External id": 9}},
        ]
        outer_meta = {"module": "vllm.attention", "attr": "outer", "total_calls_observed": 2}
        outer_verdict = ks.verify(outer, KERNEL, outer_meta, events, [outer, inner])
        self.assertFalse(outer_verdict["ok"])
        self.assertIn("deeper_live_candidate_exists", outer_verdict["failed"])
        self.assertEqual(outer_verdict["deeper_live_candidates"], [inner])

        inner_verdict = ks.verify(inner, KERNEL, self.meta(), events, [outer, inner])
        self.assertTrue(inner_verdict["ok"], inner_verdict)
        self.assertTrue(inner_verdict["deepest_verified"])

    def test_every_declared_probe_candidate_must_have_an_installed_marker(self):
        missing = "vllm.attention:missing_candidate"
        verdict = ks.verify(TARGET, KERNEL, self.meta(), trace(), [TARGET, missing])
        self.assertFalse(verdict["ok"])
        self.assertIn("candidate_marker_not_installed", verdict["failed"])
        self.assertEqual(verdict["missing_candidate_markers"], [missing])

    def test_installed_but_inactive_alternative_branch_does_not_fail(self):
        alternative = "vllm.attention:prefill_only"
        events = trace()
        events.insert(1, {
            "cat": "cpu_op", "name": ks.INSTALL_PREFIX + alternative,
            "ph": "X", "pid": 1, "tid": 2, "ts": 92, "dur": 1,
        })
        verdict = ks.verify(
            TARGET, KERNEL, self.meta(), events, [TARGET, alternative])
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(
            sorted(verdict["candidate_targets_tested"]),
            sorted([TARGET, alternative]),
        )

    def test_a_probed_candidate_that_never_fired_is_reported_not_silently_dropped(self):
        # A marker rebinds one module attribute. A callable dispatched through `torch.ops.*`, or
        # reached through an alias a caller imported before installation, runs with the wrapper
        # bypassed and produces zero spans while being fully live. Reading that as "not deeper"
        # would let a shallower seam collect `deepest_verified`, so the verdict has to say which
        # candidates were probed and never observed, apart from the ones never probed at all.
        invisible = "vllm.model_executor.layers.attention.attention:unified_attention_with_output"
        unprobed = "vllm.attention:never_installed"
        events = trace()
        events.insert(1, {
            "cat": "cpu_op", "name": ks.INSTALL_PREFIX + invisible,
            "ph": "X", "pid": 1, "tid": 2, "ts": 92, "dur": 1,
        })
        verdict = ks.verify(
            TARGET, KERNEL, self.meta(), events, [TARGET, invisible, unprobed])
        self.assertEqual(verdict["installed_but_never_live_candidates"], [invisible])
        self.assertEqual(verdict["missing_candidate_markers"], [unprobed])
        self.assertNotIn(invisible, verdict["deeper_live_candidates"])

    def test_device_projected_annotation_does_not_invert_call_nesting(self):
        # The profiler re-emits each marker on the GPU timeline as `gpu_user_annotation`, where an
        # OUTER seam's short device span can land inside the INNER launcher's long one. Only the
        # host spans describe the real call nesting; the selected inner launcher must still pass.
        outer = "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe:_fused_moe_kernel_sequence"
        events = trace()
        events.insert(1, {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + outer,
                          "ph": "X", "pid": 1, "tid": 2, "ts": 92, "dur": 1})
        events.insert(2, {"cat": "cpu_op", "name": ks.MARKER_PREFIX + outer,
                          "ph": "X", "pid": 1, "tid": 2, "ts": 95, "dur": 200})
        # device projections: outer's is a tiny span inside the target's long one, same pid/tid
        events += [
            {"cat": "gpu_user_annotation", "name": ks.MARKER_PREFIX + TARGET,
             "ph": "X", "pid": 9, "tid": 9, "ts": 500, "dur": 400},
            {"cat": "gpu_user_annotation", "name": ks.MARKER_PREFIX + outer,
             "ph": "X", "pid": 9, "tid": 9, "ts": 600, "dur": 5},
        ]
        verdict = ks.verify(TARGET, KERNEL, self.meta(), events, [TARGET, outer])
        self.assertEqual(verdict["deeper_live_candidates"], [])
        self.assertTrue(verdict["ok"], verdict)
        self.assertTrue(verdict["deepest_verified"])

    def test_capture_of_a_different_callable_is_rejected(self):
        wrong = "vllm.model_executor.layers.attention.attention:outer_wrapper"
        verdict = ks.verify(TARGET, KERNEL, self.meta(target=wrong), trace())
        self.assertFalse(verdict["ok"])
        self.assertIn("capture_target_mismatch", verdict["failed"])

    def test_zero_live_calls_is_rejected(self):
        verdict = ks.verify(TARGET, KERNEL, self.meta(calls=0), trace())
        self.assertFalse(verdict["ok"])
        self.assertIn("target_not_observed", verdict["failed"])

    def test_cli_writes_the_same_machine_verdict(self):
        with tempfile.TemporaryDirectory() as root:
            meta_path = os.path.join(root, "meta.json")
            trace_path = os.path.join(root, "trace.json")
            out_path = os.path.join(root, "selection.json")
            with open(meta_path, "w") as fh:
                json.dump(self.meta(), fh)
            with open(trace_path, "w") as fh:
                json.dump({"traceEvents": trace()}, fh)
            rc = ks.main([
                "--target", TARGET,
                "--device-kernel", KERNEL,
                "--capture-meta", meta_path,
                "--torch-trace", trace_path,
                "--out", out_path,
            ])
            self.assertEqual(rc, 0)
            with open(out_path) as fh:
                self.assertTrue(json.load(fh)["ok"])

    def test_cli_fails_closed_when_capture_pid_has_no_trace(self):
        with tempfile.TemporaryDirectory() as root:
            meta_path = os.path.join(root, "capture.pid-111.rank-0", "meta.json")
            os.makedirs(os.path.dirname(meta_path))
            trace_path = os.path.join(root, "selection.pid-999.rank-0.json")
            out_path = os.path.join(root, "selection.json")
            meta = self.meta()
            meta["process_id"] = 111
            with open(meta_path, "w") as fh:
                json.dump(meta, fh)
            with open(trace_path, "w") as fh:
                json.dump({"traceEvents": trace()}, fh)
            rc = ks.main([
                "--target", TARGET,
                "--device-kernel", KERNEL,
                "--capture-meta", meta_path,
                "--torch-trace", trace_path,
                "--out", out_path,
                "--no-reclaim-captures",
            ])
            self.assertEqual(rc, 1)
            with open(out_path) as fh:
                verdict = json.load(fh)
            self.assertFalse(verdict["ok"])
            self.assertIn("capture_process_trace_missing", verdict["failed"])
            self.assertEqual(verdict["trace_file"], "")

    def test_cli_merges_calls_before_deciding_deepest_candidate(self):
        mid = "vllm.attention:mid"
        inner = TARGET
        with tempfile.TemporaryDirectory() as root:
            meta_path = os.path.join(root, "capture.pid-111.rank-0", "meta.json")
            os.makedirs(os.path.dirname(meta_path))
            meta = self.meta(target=mid)
            meta["process_id"] = 111
            with open(meta_path, "w") as fh:
                json.dump(meta, fh)

            common_installs = [
                {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + mid,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 80, "dur": 1},
                {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + inner,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 82, "dur": 1},
            ]
            call_one = common_installs + [
                {"cat": "cpu_op", "name": ks.MARKER_PREFIX + mid,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 50},
                {"cat": "cpu_op", "name": "launch", "ph": "X",
                 "pid": 1, "tid": 2, "ts": 110, "dur": 5,
                 "args": {"External id": 7}},
                {"cat": "kernel", "name": KERNEL, "ph": "X", "ts": 180, "dur": 10,
                 "args": {"External id": 7}},
            ]
            call_two = common_installs + [
                {"cat": "cpu_op", "name": ks.MARKER_PREFIX + mid,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 80},
                {"cat": "cpu_op", "name": ks.MARKER_PREFIX + inner,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 120, "dur": 40},
                {"cat": "cpu_op", "name": "launch", "ph": "X",
                 "pid": 1, "tid": 2, "ts": 130, "dur": 5,
                 "args": {"External id": 9}},
                {"cat": "kernel", "name": KERNEL, "ph": "X", "ts": 200, "dur": 10,
                 "args": {"External id": 9}},
            ]
            trace_paths = []
            for index, events in enumerate((call_one, call_two), 1):
                path = os.path.join(
                    root, f"selection.pid-111.rank-0.call-{index}.json")
                with open(path, "w") as fh:
                    json.dump({"traceEvents": events}, fh)
                trace_paths.append(path)
            out_path = os.path.join(root, "selection.json")
            rc = ks.main([
                "--target", mid,
                "--device-kernel", KERNEL,
                "--capture-meta", meta_path,
                "--torch-trace", *trace_paths,
                "--candidate-target", mid,
                "--candidate-target", inner,
                "--out", out_path,
                "--no-reclaim-captures",
            ])
            self.assertEqual(rc, 1)
            with open(out_path) as fh:
                verdict = json.load(fh)
            self.assertIn("deeper_live_candidate_exists", verdict["failed"])
            self.assertEqual(verdict["deeper_live_candidates"], [inner])

    def test_cli_requires_deepest_selection_on_every_capture_process(self):
        mid = "vllm.attention:mid"
        inner = TARGET

        def process_events(with_inner):
            events = [
                {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + mid,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 80, "dur": 1},
                {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + inner,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 82, "dur": 1},
                {"cat": "cpu_op", "name": ks.MARKER_PREFIX + mid,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 80},
            ]
            if with_inner:
                events.append({
                    "cat": "cpu_op", "name": ks.MARKER_PREFIX + inner,
                    "ph": "X", "pid": 1, "tid": 2, "ts": 120, "dur": 40,
                })
            events.extend([
                {"cat": "cpu_op", "name": "launch", "ph": "X",
                 "pid": 1, "tid": 2, "ts": 130, "dur": 5,
                 "args": {"External id": 9}},
                {"cat": "kernel", "name": KERNEL, "ph": "X", "ts": 200, "dur": 10,
                 "args": {"External id": 9}},
            ])
            return events

        with tempfile.TemporaryDirectory() as root:
            meta_paths, trace_paths = [], []
            for pid, with_inner in ((111, False), (222, True)):
                meta_path = os.path.join(
                    root, f"capture.pid-{pid}.rank-0", "meta.json")
                os.makedirs(os.path.dirname(meta_path))
                meta = self.meta(target=mid)
                meta["process_id"] = pid
                with open(meta_path, "w") as fh:
                    json.dump(meta, fh)
                meta_paths.append(meta_path)
                trace_path = os.path.join(
                    root, f"selection.pid-{pid}.rank-0.call-1.json")
                with open(trace_path, "w") as fh:
                    json.dump({"traceEvents": process_events(with_inner)}, fh)
                trace_paths.append(trace_path)
            out_path = os.path.join(root, "selection.json")
            rc = ks.main([
                "--target", mid,
                "--device-kernel", KERNEL,
                "--capture-meta", *meta_paths,
                "--torch-trace", *trace_paths,
                "--candidate-target", mid,
                "--candidate-target", inner,
                "--out", out_path,
                "--no-reclaim-captures",
            ])
            self.assertEqual(rc, 1)
            with open(out_path) as fh:
                verdict = json.load(fh)
            self.assertFalse(verdict["ok"])
            self.assertEqual(verdict["deeper_live_candidates"], [inner])
            self.assertEqual(
                sorted(verdict["live_candidate_targets"]), sorted([mid, inner]))


class TestTheVerdictRefusesMalformedInput(unittest.TestCase):
    """These are the fail-closed edges. Each one is a way an extraction could arrive incomplete and
    still be read as "nothing to check here", which is the vacuous pass the contract exists to stop."""

    def meta(self, calls=7, target=TARGET):
        module, attr = target.split(":", 1)
        return {"module": module, "attr": attr, "total_calls_observed": calls}

    def test_a_prose_target_is_named_as_such_rather_than_probed(self):
        verdict = ks.verify("the attention wrapper", KERNEL, self.meta(), trace())
        self.assertFalse(verdict["ok"])
        self.assertIn("invalid_target_callable", verdict["failed"])

    def test_a_head_with_no_device_symbol_cannot_certify_anything(self):
        verdict = ks.verify(TARGET, "", self.meta(), trace())
        self.assertFalse(verdict["ok"])
        self.assertIn("missing_device_kernel", verdict["failed"])

    def test_one_malformed_candidate_sinks_the_whole_probe(self):
        """The coverage check compares the declared candidate set against what was probed. A member
        that cannot be probed at all must fail rather than quietly shrink the set being compared."""
        verdict = ks.verify(TARGET, KERNEL, self.meta(), trace(),
                            candidate_targets=["live_call_seam (see notes)"])
        self.assertFalse(verdict["ok"])
        self.assertIn("invalid_candidate_target", verdict["failed"])

    def test_an_empty_kernel_name_never_matches_by_accident(self):
        self.assertFalse(ks.kernel_matches("", KERNEL))
        self.assertFalse(ks.kernel_matches(KERNEL, ""))
        self.assertFalse(ks.kernel_matches("<>", KERNEL))


class TestSpansAreReadFromEitherTraceShape(unittest.TestCase):
    def test_begin_end_pairs_bound_a_span_the_way_a_complete_event_does(self):
        """Kineto writes a marker as a complete `X` event, but a trace that was cut short (or came
        from a different exporter) carries the same span as a B/E pair. Reading only `X` would report
        a live seam as never having run."""
        events = [
            {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + TARGET,
             "ph": "X", "pid": 1, "tid": 2, "ts": 90, "dur": 1},
            {"cat": "cpu_op", "name": ks.MARKER_PREFIX + TARGET,
             "ph": "B", "pid": 1, "tid": 2, "ts": 100},
            {"cat": "cpu_op", "name": "launch", "ph": "X", "pid": 1, "tid": 2,
             "ts": 110, "dur": 5, "args": {"External id": 7}},
            {"cat": "cpu_op", "name": ks.MARKER_PREFIX + TARGET,
             "ph": "E", "pid": 1, "tid": 2, "ts": 200},
            {"cat": "kernel", "name": KERNEL, "ph": "X", "ts": 300, "dur": 10,
             "args": {"External id": 7}},
        ]
        module, attr = TARGET.split(":", 1)
        verdict = ks.verify(
            TARGET, KERNEL,
            {"module": module, "attr": attr, "total_calls_observed": 1}, events)
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(verdict["matched_kernel_calls"], 1)

    def test_an_end_with_no_begin_is_dropped_instead_of_inventing_a_span(self):
        spans = ks._complete_spans([
            {"cat": "cpu_op", "name": "m", "ph": "E", "pid": 1, "tid": 2, "ts": 200},
        ], "m")
        self.assertEqual(spans, [])

    def test_an_event_with_no_timestamp_is_not_inside_any_span(self):
        self.assertFalse(ks._within_any_span({"cat": "cpu_op"}, [(0.0, 10.0, {})]))

    def test_a_device_row_does_not_donate_its_own_id_to_the_launch_set(self):
        """Kernel rows sit inside the marker span on the device timeline too. Harvesting ids from
        them would make a kernel vouch for itself, so only host events may establish causality."""
        events = [
            {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + TARGET,
             "ph": "X", "pid": 1, "tid": 2, "ts": 90, "dur": 1},
            {"cat": "cpu_op", "name": ks.MARKER_PREFIX + TARGET,
             "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 100},
            {"cat": "kernel", "name": KERNEL, "ph": "X", "pid": 1, "tid": 2,
             "ts": 120, "dur": 10, "args": {"External id": 7}},
        ]
        module, attr = TARGET.split(":", 1)
        verdict = ks.verify(
            TARGET, KERNEL,
            {"module": module, "attr": attr, "total_calls_observed": 1}, events)
        self.assertFalse(verdict["ok"])
        self.assertIn("device_kernel_not_under_target", verdict["failed"])


class TestMergingProcessTraces(unittest.TestCase):
    def test_a_trace_file_carrying_non_event_entries_is_merged_without_raising(self):
        """`traceEvents` from a truncated export can contain nulls/strings. The verifier reads other
        people's files, so a malformed entry must be skipped rather than abort every rank's merge."""
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "selection.pid-1.rank-0.call-1.json")
            with open(path, "w") as fh:
                json.dump({"traceEvents": [
                    None,
                    "not an event",
                    {"cat": "cpu_op", "name": "m", "ph": "X", "pid": 1, "tid": 2,
                     "ts": 1, "dur": 1, "args": {"External id": 5, "correlation": 6}},
                ]}, fh)
            merged = ks.merge_process_traces([path])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["pid"], "trace-0:1")
        self.assertEqual(merged[0]["args"]["External id"], "trace-0:5")
        self.assertEqual(merged[0]["args"]["correlation"], "trace-0:6")


class TestCaptureStorageReclaim(unittest.TestCase):
    """Issue #429: promote one authoritative oracle and delete process-local capture.pid-* dirs."""

    def meta(self, calls=7, target=TARGET, process_id=111):
        module, attr = target.split(":", 1)
        return {
            "module": module, "attr": attr, "total_calls_observed": calls,
            "process_id": process_id, "target": target,
        }

    def _write_capture(self, root, pid, payload=b"ORACLE", rank=0):
        cap = os.path.join(root, f"capture.pid-{pid}.rank-{rank}")
        os.makedirs(cap)
        meta_path = os.path.join(cap, "meta.json")
        with open(meta_path, "w") as fh:
            json.dump(self.meta(process_id=pid), fh)
        with open(os.path.join(cap, "reference_io.pt"), "wb") as fh:
            fh.write(payload)
        trace_path = os.path.join(root, f"selection.pid-{pid}.rank-{rank}.call-1.json")
        with open(trace_path, "w") as fh:
            json.dump({"traceEvents": trace()}, fh)
        return meta_path, trace_path

    def test_infer_task_dir_from_process_local_metas(self):
        with tempfile.TemporaryDirectory() as root:
            meta_a, _ = self._write_capture(root, 111, b"A" * 100)
            meta_b, _ = self._write_capture(root, 222, b"B" * 200)
            self.assertEqual(ks.infer_task_dir([meta_a, meta_b]), root)

    def test_successful_selection_promotes_one_oracle_and_reclaims_ranks(self):
        with tempfile.TemporaryDirectory() as root:
            meta_a, trace_a = self._write_capture(root, 111, b"KEEP-ME-ORACLE")
            meta_b, trace_b = self._write_capture(root, 222, b"DROP-ME-ORACLE-XXXX")
            # nested retry leftover
            nested = os.path.join(root, "_selcap", "capture.pid-333.rank-0")
            os.makedirs(nested)
            with open(os.path.join(nested, "reference_io.pt"), "wb") as fh:
                fh.write(b"NESTED-HEAVY")
            out_path = os.path.join(root, "selection.json")
            rc = ks.main([
                "--target", TARGET,
                "--device-kernel", KERNEL,
                "--capture-meta", meta_a, meta_b,
                "--torch-trace", trace_a, trace_b,
                "--task-dir", root,
                "--out", out_path,
            ])
            self.assertEqual(rc, 0)
            with open(out_path) as fh:
                verdict = json.load(fh)
            self.assertTrue(verdict["ok"])
            self.assertTrue(verdict["capture_storage"]["promoted"])
            self.assertTrue(os.path.isfile(os.path.join(root, "reference_io.pt")))
            self.assertTrue(os.path.isfile(os.path.join(root, "meta.json")))
            with open(os.path.join(root, "reference_io.pt"), "rb") as fh:
                # best process is max matched calls; both equal so max() picks one stably
                self.assertIn(fh.read(), (b"KEEP-ME-ORACLE", b"DROP-ME-ORACLE-XXXX"))
            remaining = [
                p for p in os.listdir(root) if p.startswith("capture.pid-")]
            self.assertEqual(remaining, [])
            self.assertFalse(os.path.exists(nested))
            self.assertGreater(verdict["capture_storage"]["bytes_reclaimed"], 0)

    def test_failed_selection_still_reclaims_heavy_captures(self):
        with tempfile.TemporaryDirectory() as root:
            meta_path = os.path.join(root, "capture.pid-111.rank-0", "meta.json")
            os.makedirs(os.path.dirname(meta_path))
            with open(meta_path, "w") as fh:
                json.dump(self.meta(), fh)
            with open(os.path.join(os.path.dirname(meta_path), "reference_io.pt"),
                      "wb") as fh:
                fh.write(b"X" * 4096)
            trace_path = os.path.join(root, "selection.pid-999.rank-0.json")
            with open(trace_path, "w") as fh:
                json.dump({"traceEvents": trace()}, fh)
            out_path = os.path.join(root, "selection.json")
            rc = ks.main([
                "--target", TARGET,
                "--device-kernel", KERNEL,
                "--capture-meta", meta_path,
                "--torch-trace", trace_path,
                "--task-dir", root,
                "--out", out_path,
            ])
            self.assertEqual(rc, 1)
            self.assertFalse(os.path.exists(os.path.dirname(meta_path)))
            self.assertFalse(os.path.isfile(os.path.join(root, "reference_io.pt")))

    def test_reclaim_exception_is_recorded_without_failing_selection(self):
        with tempfile.TemporaryDirectory() as root:
            meta_a, trace_a = self._write_capture(root, 111, b"OK")
            out_path = os.path.join(root, "selection.json")
            real = ks.sys.modules.get("capture_shapes")
            # Force promote_and_reclaim to raise while keeping import successful.
            import capture_shapes as _cs
            previous = _cs.promote_and_reclaim

            def boom(*_a, **_k):
                raise RuntimeError("reclaim exploded")

            _cs.promote_and_reclaim = boom
            try:
                rc = ks.main([
                    "--target", TARGET,
                    "--device-kernel", KERNEL,
                    "--capture-meta", meta_a,
                    "--torch-trace", trace_a,
                    "--task-dir", root,
                    "--out", out_path,
                ])
            finally:
                _cs.promote_and_reclaim = previous
            self.assertEqual(rc, 0)
            with open(out_path) as fh:
                verdict = json.load(fh)
            self.assertTrue(verdict["ok"])
            self.assertIn("reclaim exploded", verdict["capture_storage"]["errors"][0])


if __name__ == "__main__":
    unittest.main()
