"""Token tagging, catalog hygiene and observed_in_trace in fusion_catalog.py."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fusion_catalog as fc


class TokenTagTest(unittest.TestCase):
    def test_tags_come_from_whole_tokens(self):
        # 2026-10-11 Qwen3.5: "gemma" matched "gemm", "gating" matched topk.
        self.assertEqual(fc._op_tags("_fused_qk_gemma_rmsnorm_kernel"), ["norm"])
        self.assertEqual(fc._op_tags("gemma_qkv_rmsnorm"), ["norm"])
        for name in ("fused_gdn_gating", "fused_sigmoid_gating_delta_rule_update",
                     "ff_a16w16_fused_gated"):
            self.assertNotIn("topk", fc._op_tags(name), name)
        self.assertNotIn("norm", fc._op_tags("cutlass_w4a8_moe_deepep_normal"))
        self.assertIn("gemm", fc._op_tags("wv_splitk_small_fp16_bf16_kernel"))
        self.assertIn("topk", fc._op_tags("moe_fused_gate"))
        self.assertIn("topk", fc._op_tags("_ZN4vllm3moe17topkGatingSoftmaxIDF16bLi32EEEvPKT_"))

    def test_mangled_and_camel_names_are_tokenized(self):
        self.assertEqual(
            fc._op_tags("_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256EEEvPT0_"),
            ["add_residual", "norm", "quant"])
        self.assertEqual(fc._op_tags("RoPECached"), ["rope"])
        self.assertIn("kv_cache", fc._op_tags("fused_qk_norm_rope_cache_quant_shuffle"))
        self.assertNotIn("kv_cache", fc._op_tags("rope_cached_fwd"))

    def test_granularity_and_gemm_precisions(self):
        self.assertEqual(fc.quant_granularity("act_mul_and_fp8_group_quant"), "group")
        self.assertEqual(fc.quant_granularity("fused_silu_mul_fp8_per_tensor_static_quant"), "tensor")
        self.assertEqual(fc.quant_granularity("rmsnorm2d_fwd_with_dynamicquant"), "token")
        self.assertEqual(fc.gemm_precisions("fused_gemm_a8w8_blockscale_a16w16"), ["a8w8", "a16w16"])


class HygieneTest(unittest.TestCase):
    def test_predicates_fakes_and_device_fallbacks_are_not_kernels(self):
        for name in ("is_activation_quantization_format", "should_quick_allreduce",
                     "can_use_custom_all_reduce_with_nvlink", "has_x", "check_quantized",
                     "fused_moe_fake", "fused_topk_cpu", "fused_topk_npu", "all_reduce_fake",
                     "init_quick_all_reduce", "gemm_a8w8_blockscale_tune"):
            self.assertFalse(fc.is_kernel_name(name), name)
        for name in ("act_mul_and_fp8_group_quant", "fused_qk_norm_rope_cache_quant_shuffle",
                     "moe_sorting_fwd", "set_mla_kv_buffer_triton"):
            self.assertTrue(fc.is_kernel_name(name), name)

    def test_legacy_catalog_is_retagged_and_cleaned(self):
        legacy = [
            {"name": "fused_gdn_gating", "op_tags": ["topk"], "dtype_tags": []},
            {"name": "is_moe", "op_tags": ["moe"], "dtype_tags": []},
            {"name": "fused_allreduce_rmsnorm", "op_tags": ["allreduce", "norm"], "dtype_tags": [],
             "signatures": ["(input, residual_inp, weight, eps)"]},
        ]
        kept, dropped = fc.sanitize_kernels(legacy, tagger=None)
        names = {k["name"]: k for k in kept}
        self.assertNotIn("fused_gdn_gating", names)  # no op tag left after re-tagging
        self.assertIn("is_moe", {d["name"] for d in dropped})
        self.assertEqual(names["fused_allreduce_rmsnorm"]["op_tags"],
                         ["add_residual", "allreduce", "norm"])
        # A catalog from the current tagger is trusted as written.
        current, _ = fc.sanitize_kernels([{"name": "x_fused", "op_tags": ["norm"], "dtype_tags": []}],
                                         tagger=fc.TAGGER_VERSION)
        self.assertEqual(current[0]["op_tags"], ["norm"])


class ObservedTest(unittest.TestCase):
    def test_trace_kernels_mark_catalog_entries(self):
        kernels, _ = fc.sanitize_kernels([
            {"name": "add_rmsnorm", "op_tags": [], "dtype_tags": []},
            {"name": "add_rmsnorm_quant", "op_tags": [], "dtype_tags": []},
            {"name": "fused_allreduce_rmsnorm", "op_tags": [], "dtype_tags": []},
            {"name": "fused_allreduce_rmsnorm_quant", "op_tags": [], "dtype_tags": []},
            {"name": "reshape_and_cache_with_block_quant", "op_tags": [], "dtype_tags": []},
        ])
        found = fc.mark_observed(kernels, [
            "_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256EEEvPT0_",
            "_ZN5aiter30allreduce_fusion_kernel_1stageIDF16bDF16bLi8EEEvPNS_8RankDataE",
            "_ZN7ck_tile6kentryILi1ENS_38FmhaBatchPrefillWithPagedKVCacheKernelEEEvDpT1_",
        ])
        by_name = {k["name"]: k for k in kernels}
        self.assertTrue(by_name["add_rmsnorm_quant"]["observed_in_trace"])
        self.assertFalse(by_name["add_rmsnorm"]["observed_in_trace"])  # shorter stem loses
        self.assertTrue(by_name["fused_allreduce_rmsnorm"]["observed_in_trace"])
        self.assertFalse(by_name["fused_allreduce_rmsnorm_quant"]["observed_in_trace"])
        self.assertFalse(by_name["reshape_and_cache_with_block_quant"]["observed_in_trace"])
        self.assertEqual(len(found), 2)


if __name__ == "__main__":
    unittest.main()
