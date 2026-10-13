"""Gates in fusion_screen.py."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fusion_screen as screen

COPY = "at::native::elementwise_kernel<direct_copy_kernel_cuda>"
QK_CATALOG = {"kernels": [
    {"name": "fused_qk_norm_mrope_3d_cache_pts_quant_shuffle",
     "op_tags": ["kv_cache", "layout", "norm", "quant", "rope"], "dtype_tags": [], "sources": ["aiter"]},
    {"name": "fused_qk_norm_rope_cache_quant_shuffle",
     "op_tags": ["kv_cache", "layout", "norm", "quant", "rope"], "dtype_tags": [], "sources": ["aiter"]},
]}


def _row(row_id, name, duration, **extra):
    item = {
        "row_id": row_id,
        "raw_name": name,
        "short_name": name,
        "duration_us": duration,
        "phase": "decode",
        "pattern_id": "p0",
        "pattern_layer_count": 40,
        "stream": 0,
        "layer_id": 1,
        "shape": {"input_dims": [], "input_types": ["bf16"]},
    }
    item.update(extra)
    return item


def _table(rows, layer_total):
    return {
        "tables": [{
            "phase": "decode",
            "pattern_id": "p0",
            "pattern_layer_count": 40,
            "layer_total_us": layer_total,
            "rows": rows,
        }]
    }


class ScreenTest(unittest.TestCase):
    def test_small_chain_passes_and_large_epilogue_needs_115(self):
        # One layer: four 40us quant kernels (7% of the step) plus a long GEMM
        # whose epilogue is too small to clear 1.15x.
        quants = [_row("q%d" % i, "quant_cast", 40, device_seq_index=i) for i in range(4)]
        gemm = _row("g", "gemm_bf16", 2000, device_seq_index=4, shape={
            "input_dims": [[8192, 5120, 4096]], "input_types": ["bf16"]})
        epi = _row("e", "quant_cast", 80, device_seq_index=5)
        rows = screen.load_rows(_table(quants + [gemm, epi], 2000 + 160 + 80))
        result = screen.screen(rows, gfx="gfx942")
        names = [item["kernel_names"] for item in result["initial_topk"]]
        self.assertIn(["quant_cast", "quant_cast", "quant_cast", "quant_cast"], names)
        self.assertNotIn(["gemm_bf16", "quant_cast"], names)
        self.assertTrue(result["kernel_class"]["g"]["large"])
        self.assertFalse(result["kernel_class"]["q0"]["large"])

    def test_below_gain_gate_goes_to_the_pool_not_the_topk(self):
        # Two 10us kernels beside a 5000us barrier: one launch saved is 0.2% of
        # the request, under the 0.3% gate but above the 0.1% pool floor.
        rows = screen.load_rows(_table([
            _row("a", "silu", 10, device_seq_index=0),
            _row("b", "quant_cast", 10, device_seq_index=1),
            _row("rest", "kernel_moe_gemm", 5000, device_seq_index=2),
        ], 5020))
        result = screen.screen(rows)
        self.assertEqual(result["initial_topk"], [])
        self.assertEqual(result["execution_list"], [])
        pool = result["candidate_pool"]
        self.assertEqual([item["row_ids"] for item in pool], [["a", "b"]])
        self.assertTrue(screen.POOL_GAIN_PCT <= pool[0]["expected_e2e_gain_pct"] < screen.GAIN_GATE_PCT)

    def test_conflict_keeps_higher_score_and_caps_at_8(self):
        rows = []
        # First four kernels are one adjacent run, so the 4-kernel window shares
        # rows with its shorter subsets. The rest are disjoint pairs.
        for index in range(4):
            rows.append(_row("h%d" % index, "quant_cast" if index % 2 == 0 else "silu",
                             40, device_seq_index=index, timestamp=index * 40))
        for index in range(8):
            base = 10000 + index * 1000
            rows.append(_row("a%d" % index, "quant_cast", 40,
                             device_seq_index=10 + index * 2, timestamp=base))
            rows.append(_row("b%d" % index, "silu", 40,
                             device_seq_index=11 + index * 2, timestamp=base + 40))
        layer_total = sum(row["duration_us"] for row in rows)
        result = screen.screen(screen.load_rows(_table(rows, layer_total)), top_k=8)
        self.assertEqual(len(result["execution_list"]), 8)
        taken = []
        for item in result["execution_list"]:
            self.assertFalse(set(item["row_ids"]) & set(taken))
            taken.extend(item["row_ids"])
        kept_heads = [item["row_ids"] for item in result["execution_list"]
                      if "h0" in item["row_ids"]]
        self.assertEqual(kept_heads, [["h0", "h1", "h2", "h3"]])
        self.assertTrue(all(item["difficulty"] == 1.3 for item in result["execution_list"]))
        self.assertTrue(all(item["pass_ranks"] == [item["pass_ranks"][0]] * 2
                            for item in result["execution_list"]))

    def test_existing_kernel_uses_difficulty_1(self):
        rows = screen.load_rows(_table([
            _row("a", "rmsnorm", 40, device_seq_index=0, timestamp=0),
            _row("b", "quant_fp8", 40, device_seq_index=1, timestamp=40),
            _row("pad", "copy_layout", 2000, device_seq_index=2, timestamp=5000),
        ], 2080))
        catalog = {"kernels": [{"name": "fused_rmsnorm_quant_fp8"}]}
        result = screen.screen(rows, catalog=catalog)
        chain = next(item for item in result["initial_topk"] if item["kernel_names"] == ["rmsnorm", "quant_fp8"])
        self.assertEqual(chain["difficulty"], 1.0)
        self.assertTrue(chain["has_existing_kernel"])
        final = next(item for item in result["execution_list"] if item["kernel_names"] == ["rmsnorm", "quant_fp8"])
        self.assertTrue(final["compare_existing"])
        self.assertTrue(final["self_author"])

    def test_merge_prefers_both_passes_then_rank_sum(self):
        def chain(name, score, row_ids):
            return {"candidate_id": name, "score": score, "row_ids": row_ids}
        a = chain("a", 9, ["r1"])
        b = chain("b", 8, ["r2"])
        c = chain("c", 7, ["r3"])
        d = chain("d", 6, ["r3", "r4"])
        merged = screen.merge_passes([[a, c, b], [d, b]], top_k=3)
        # b is in both passes (3+2). Single-pass rank sums, missing = K+1 = 4:
        # a 1+4, d 4+1, c 2+4. c then shares r3 with d and is dropped.
        self.assertEqual([item["candidate_id"] for item in merged], ["b", "a", "d"])
        self.assertEqual(merged[0]["pass_ranks"], [3, 2])
        self.assertEqual(merged[1]["pass_ranks"], [1, 4])

    def test_parallel_window_requires_distinct_streams(self):
        rows = screen.load_rows(_table([
            _row("a", "quant_cast", 30, stream=1, timestamp=0, device_seq_index=0),
            _row("b", "silu", 30, stream=2, timestamp=5, device_seq_index=1),
            _row("c", "quant_cast", 30, stream=1, timestamp=1000, device_seq_index=2),
            _row("pad", "copy_layout", 800, stream=0, timestamp=2000, device_seq_index=3),
        ], 890))
        result = screen.screen(rows)
        kinds = {tuple(item["kernel_names"]): item["kind"] for item in result["initial_topk"]}
        self.assertEqual(kinds.get(("quant_cast", "silu")), "parallel")

    def test_qk_norm_rope_grows_to_kv_write_with_existing_kernel(self):
        # Qwen3 decode attention prologue: the covering aiter kernel also writes the
        # KV cache, so the chain grows past 4 compute kernels to store_kvcache and
        # never swallows the paged-attention barrier behind it.
        names = [("c0", COPY, 4.5), ("qn", "aiter::add_rmsnorm_quant_kernel", 4.2),
                 ("c1", COPY, 4.3), ("kn", "aiter::add_rmsnorm_quant_kernel", 4.4),
                 ("rope", "rotary_embedding_kernel<bf16>", 5.0), ("kv", "store_kvcache<2048>", 4.4),
                 ("pa", "paged_attention_ll4mi_qkv_kernel", 24.0),
                 ("g", "gemm_a8w8_blockscale_kernel", 583.2)]
        rows = screen.load_rows(_table(
            [_row(rid, name, dur, device_seq_index=i) for i, (rid, name, dur) in enumerate(names)], 634.0))
        result = screen.screen(rows, catalog=QK_CATALOG)
        top = result["execution_list"][0]
        self.assertIn("kv", top["row_ids"])
        self.assertIn("rope", top["row_ids"])
        self.assertTrue(top["has_existing_kernel"])
        self.assertEqual(top["difficulty"], 1.0)
        self.assertEqual(top["matched_existing_kernels"][0]["name"],
                         "fused_qk_norm_rope_cache_quant_shuffle")
        for item in result["initial_topk"]:
            self.assertNotIn("pa", item["row_ids"])
            self.assertNotIn("g", item["row_ids"])

    def test_barrier_never_inside_a_chain(self):
        rows = screen.load_rows(_table([
            _row("n", "rmsnorm", 40, device_seq_index=0),
            _row("pa", "paged_attention_kernel", 40, device_seq_index=1),
            _row("q", "quant_cast", 40, device_seq_index=2),
            _row("r", "rotary_embedding_kernel", 40, device_seq_index=3),
            _row("pad", "copy_layout", 500, device_seq_index=4, timestamp=9000),
        ], 660))
        result = screen.screen(rows)
        for item in result["initial_topk"]:
            self.assertNotIn("pa", item["row_ids"])
        self.assertIn(["q", "r"], [item["row_ids"] for item in result["initial_topk"]])
        self.assertGreater(result["rejected"].get("barrier_without_existing_kernel", 0), 0)

    def test_existing_kernel_lowers_gain_gate_to_015_percent(self):
        # SiLU*Mul + fp8 group quant saves one launch, ~0.23% of the request:
        # below 0.3% but above the 0.15% gate for an existing kernel.
        def rows():
            return screen.load_rows(_table([
                _row("act", "sgl_hip::activation::act_and_mul_kernel<silu>", 8.6, device_seq_index=0),
                _row("q", "aiter::dynamic_per_group_scaled_quant_kernel", 4.6, device_seq_index=1),
                _row("g", "gemm_a8w8_blockscale_kernel", 2000.6, device_seq_index=2),
            ], 2013.8))
        self.assertEqual(screen.screen(rows())["initial_topk"], [])
        catalog = {"kernels": [
            {"name": "is_activation_quantization_format", "op_tags": ["activation", "quant"],
             "dtype_tags": [], "sources": ["sglang"]},
            {"name": "act_mul_and_fp8_group_quant", "op_tags": ["activation", "quant"],
             "dtype_tags": ["fp8", "fp8_blockscale"], "sources": ["aiter"]},
        ]}
        result = screen.screen(rows(), catalog=catalog)
        chain = next(item for item in result["execution_list"] if item["row_ids"] == ["act", "q"])
        self.assertEqual(chain["gate_min"], screen.EXISTING_GAIN_GATE_PCT / 100.0)
        self.assertLess(chain["expected_e2e_gain_pct"], screen.GAIN_GATE_PCT)
        self.assertEqual([m["name"] for m in chain["matched_existing_kernels"]],
                         ["act_mul_and_fp8_group_quant"])

    def test_same_seam_in_prefill_and_decode_is_one_entry(self):
        def phase_rows(prefix, phase, extra):
            return [_row(prefix + rid, name, dur, phase=phase, device_seq_index=i, **extra)
                    for i, (rid, name, dur) in enumerate([
                        ("qn", "rmsnorm", 4.0), ("kn", "rmsnorm", 4.0),
                        ("r", "rotary_embedding_kernel", 5.0), ("g", "gemm_bf16", 500.0)])]
        table = {"tables": [
            {"phase": "prefill", "pattern_id": "p0", "pattern_layer_count": 40, "layer_total_us": 513.0,
             "rows": phase_rows("p", "prefill", {"step_input_tokens": 2048})},
            {"phase": "decode", "pattern_id": "p0", "pattern_layer_count": 40, "layer_total_us": 634.0,
             "rows": phase_rows("d", "decode", {"step_batch_size": 8})},
        ]}
        catalog = {"kernels": [{"name": "fused_qk_norm_rope_2way", "op_tags": ["norm", "rope"],
                                "dtype_tags": [], "sources": ["aiter"]}]}
        result = screen.screen(screen.load_rows(table), catalog=catalog,
                               workload={"isl": 1024, "osl": 1024, "conc": 64})
        self.assertEqual(len(result["execution_list"]), 1)
        entry = result["execution_list"][0]
        self.assertEqual(sorted(entry["phases"]), ["decode", "prefill"])
        self.assertEqual(len(entry["candidate_ids"]), 2)
        self.assertEqual({entry["row_ids"][0][0], entry["partner_row_ids"][0][0]}, {"p", "d"})

    def test_rope_is_elementwise_and_copies_do_not_set_difficulty(self):
        rows = screen.load_rows(_table([
            _row("c", COPY, 40, device_seq_index=0),
            _row("r", "rotary_embedding_kernel", 40, device_seq_index=1),
            _row("q", "quant_cast", 40, device_seq_index=2),
            _row("pad", "copy_layout", 500, device_seq_index=3, timestamp=9000),
        ], 620))
        result = screen.screen(rows)
        chain = next(item for item in result["initial_topk"] if item["row_ids"] == ["c", "r", "q"])
        self.assertEqual(chain["difficulty"], 1.3)
        self.assertEqual(chain["data_move_row_ids"], ["c"])
        self.assertEqual(chain["unknown_kernels"], [])

    def test_baseline_fused_kernel_with_follow_up_quant_is_flagged(self):
        rows = screen.load_rows(_table([
            _row("n", "aiter::add_rmsnorm_quant_kernel", 4.7, device_seq_index=0),
            _row("q", "aiter::dynamic_per_group_scaled_quant_kernel", 4.6, device_seq_index=1),
            _row("g", "gemm_a8w8_blockscale_kernel", 624.7, device_seq_index=2),
        ], 634.0))
        catalog = {"kernels": [{"name": "add_rmsnorm_quant",
                                "op_tags": ["add_residual", "norm", "quant"],
                                "dtype_tags": [], "sources": ["aiter"]}]}
        result = screen.screen(rows, catalog=catalog)
        fused = result["baseline_fused"]
        self.assertEqual([item["catalog_kernel"] for item in fused], ["add_rmsnorm_quant"])
        self.assertTrue(fused[0]["still_in_candidates"])

    def test_cli_writes_both_boards(self):
        import tempfile
        table = _table([
            _row("a", "quant_cast", 40, device_seq_index=0, timestamp=0),
            _row("b", "silu", 40, device_seq_index=1, timestamp=40),
            _row("pad", "copy_layout", 2000, device_seq_index=2, timestamp=5000),
        ], 2080)
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "table.json")
            with open(src, "w") as handle:
                json.dump(table, handle)
            out = os.path.join(tmp, "fusion")
            rc = screen.main(["--semantic-table", src, "--out-dir", out,
                              "--md", os.path.join(tmp, "03_FUSION_TOPK.md"),
                              "--isl", "", "--osl", "", "--conc", "", "--top-k", "8",
                              "--passes", "2", "--gfx", "gfx942", "--catalog", ""])
            self.assertEqual(rc, 0)
            with open(os.path.join(out, "fusion_topk.json")) as handle:
                topk = json.load(handle)
            self.assertEqual(len(topk["execution_list"]), 1)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "03_FUSION_TOPK.md")))
            with open(os.path.join(out, "fusion_candidates.json")) as handle:
                board = json.load(handle)
            for key in ("seam_entries", "appendix", "region_coverage", "priors",
                        "classification_conflicts", "catalog_hygiene"):
                self.assertIn(key, board)
            entry = topk["execution_list"][0]
            for key in ("seam_key", "request_saved_us", "merged_members", "forced_by_prior",
                        "pass_ranks", "row_ids", "candidate_ids"):
                self.assertIn(key, entry)
            # Apply-back (fusion_integrator / fusion_applyback_harness) acts only on tier-A/B rows.
            self.assertEqual(entry["tier"], "B")



def _srow(row_id, name, duration, seq, stage=None, op=None, module=None, dims=None, types=None,
          phase="decode", pattern="P0", layers=45, **extra):
    """A semantic-table row as the real pipeline writes it (stage + parent operator)."""
    item = {
        "row_id": row_id, "raw_name": name, "short_name": name, "duration_us": duration,
        "phase": phase, "pattern_id": pattern, "pattern_layer_count": layers, "stream": 8,
        "layer_id": 8, "device_seq_index": seq, "pos": seq,
        "parent_operator": {"canonical_op": op or "unresolved"},
        "shape": {"input_dims": dims or [], "input_types": types or []},
    }
    if stage:
        item["stage"] = stage
    if module:
        item["semantic_evidence"] = {"contained_by": "model.layers.8." + module}
    item.update(extra)
    return item


def _stable(rows, phase="decode", pattern="P0", layers=45, total=None):
    return {"phase": phase, "pattern_id": pattern, "pattern_layer_count": layers,
            "layer_total_us": total if total is not None else sum(r["duration_us"] for r in rows),
            "rows": rows}


QUANT = "_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDF16bLi128EEEvPT_"
QUANT_DIMS = [[4, 4096], [128, 128], [4, 32], [], [], [], []]
QUANT_TYPES = ["Float8_e4m3fnuz", "BFloat16", "float", "", "Scalar", "", "Scalar"]


def _router_rows(prefix, pattern, seq0):
    """Qwen3.5 decode MoE router seam: router GEMM .. moe_sorting .. quant, then the expert GEMM."""
    spec = [
        ("gate", "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x256", 7.0, "gemm", "aten::mm",
         "mlp.gate", [[4, 4096], [4096, 512]], ["BFloat16", "BFloat16"]),
        ("topk", "_ZN4vllm3moe17topkGatingSoftmaxIDF16bLi32ELi512EEEvPKT_", 10.1, "topk",
         "aiter::topk_softmax", "mlp.topk", None, None),
        ("sgate", "Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x512", 8.5, "gemm", "aten::mm",
         "mlp.shared_expert_gate", [[4, 4096], [4096, 1]], ["BFloat16", "BFloat16"]),
        ("sig", "void at::native::vectorized_elementwise_kernel<4, sigmoid>", 4.6, "elementwise",
         "aten::sigmoid", "mlp", None, None),
        ("copy", "void at::native::vectorized_elementwise_kernel<4, copy>", 5.2, "elementwise",
         "aten::copy_", "mlp", None, None),
        ("append", "_fused_append_shared_experts_with_weights_kernel", 4.7, "moe", "model.layers.8.mlp",
         "mlp", None, None),
        ("sort", "void ck_tile::kentry<2, ck_tile::MoeSortingKernel>", 13.5, "moe",
         "aiter::moe_sorting_fwd", "mlp.experts", None, None),
        ("quant", QUANT, 5.1, "quant", "aiter::dynamic_per_token_scaled_quant", "mlp.experts",
         QUANT_DIMS, QUANT_TYPES),
        ("stage1", "kernel_moe_gemm", 8.6, "moe", "aiter::ck_moe_stage1", "mlp.experts", None, None),
    ]
    return [_srow(prefix + rid, name, dur, seq0 + i, stage=stage, op=op, module=module, dims=dims,
                  types=types, pattern=pattern, layers=45 if pattern == "P0" else 15)
            for i, (rid, name, dur, stage, op, module, dims, types) in enumerate(spec)]


class ClassificationTest(unittest.TestCase):
    def test_gemma_rmsnorm_is_a_norm_and_wv_splitk_is_a_gemm(self):
        # The 2026-10-11 accidents: "gemma" contained "gemm"; "wvsplit" never
        # matched "wv_splitk"; the stage column said linear_attn.
        norm = _srow("n", "_fused_qk_gemma_rmsnorm_kernel", 5.4, 0, stage="norm")
        bare_norm = _srow("n2", "_fused_qk_gemma_rmsnorm_kernel", 5.4, 1)
        wv = _srow("w", "void aiter::wv_splitk_small_fp16_bf16_kernel<64>", 5.0, 2,
                   stage="linear_attn", op="aiter::wv_splitk_small_fp16_bf16")
        bare_wv = _srow("w2", "wv_splitk_small_fp16_bf16_kernel", 5.0, 3)
        self.assertEqual(screen.classify_row(norm)["cls"], "norm")
        self.assertEqual(screen.classify_row(bare_norm)["cls"], "norm")
        self.assertEqual(screen.classify_row(wv)["cls"], "gemm")
        self.assertIn("parent operator wins", screen.classify_row(wv)["conflict"])
        self.assertEqual(screen.classify_row(bare_wv)["cls"], "gemm")

    def test_data_move_comes_from_the_parent_operator(self):
        copy = _srow("c", "void at::native::elementwise_kernel_manual_unroll<128, 4>", 5.0, 0,
                     stage="elementwise", op="aten::copy_")
        sigmoid = _srow("s", "void at::native::elementwise_kernel_manual_unroll<128, 4>", 5.0, 1,
                        stage="elementwise", op="aten::sigmoid")
        split = _srow("x", "fused_qkvzba_split_reshape_cat_contiguous_kernel", 5.4, 2,
                      stage="linear_attn", op="model.layers.8")
        self.assertEqual(screen.classify_row(copy)["cls"], "data_move")
        self.assertEqual(screen.classify_row(sigmoid)["cls"], "elementwise")
        self.assertEqual(screen.classify_row(split)["cls"], "layout")

    def test_linear_attention_body_and_moe_expert_gemm_are_barriers(self):
        for name, stage in (("_causal_conv1d_update_kernel", "linear_attn"),
                            ("fused_recurrent_gated_delta_rule_packed_decode_kernel", "linear_attn"),
                            ("chunk_gated_delta_rule_fwd_kernel_h_blockdim64", "linear_attn"),
                            ("chunk_local_cumsum_scalar_kernel", "linear_attn"),
                            ("chunk_fwd_kernel_o", "linear_attn"),
                            ("recompute_w_u_fwd_kernel", "linear_attn"),
                            ("fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256", "moe"),
                            ("kernel_moe_gemm", "moe")):
            row = _srow("r", name, 5.0, 0, stage=stage)
            self.assertIn(screen._barrier(row, {}), ("linear_attn", "moe_gemm"), name)


class SmallGemmTest(unittest.TestCase):
    def _qkvz_table(self):
        rows = [
            _srow("ar", "_ZN5aiter30allreduce_fusion_kernel_1stageIDF16bDF16bLi8EEEvPNS_8RankDataE",
                  6.6, 0, stage="communication"),
            _srow("q", QUANT, 5.7, 1, stage="quant", op="aiter::dynamic_per_token_scaled_quant",
                  module="linear_attn.in_proj_qkvz", dims=QUANT_DIMS, types=QUANT_TYPES),
            _srow("qkvz", "_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128", 66.0, 2,
                  stage="gemm", module="linear_attn.in_proj_qkvz",
                  op="model.layers.8.linear_attn.in_proj_qkvz",
                  dims=[[4, 4096], [2560, 4096], [20, 32], [4, 2560]],
                  types=["bfloat16", "float8_e4m3fnuz", "float32", "bfloat16"]),
            _srow("ba", "void aiter::wv_splitk_small_fp16_bf16_kernel<64>", 5.0, 3,
                  stage="linear_attn", op="aiter::wv_splitk_small_fp16_bf16",
                  module="linear_attn.in_proj_ba",
                  dims=[[16, 4096], [4, 4096], [4, 16], [], []],
                  types=["BFloat16", "BFloat16", "BFloat16", "Scalar", "Scalar"]),
            _srow("split", "fused_qkvzba_split_reshape_cat_contiguous_kernel", 5.4, 4,
                  stage="linear_attn", op="model.layers.8"),
            _srow("conv", "_causal_conv1d_update_kernel", 5.5, 5, stage="linear_attn",
                  module="linear_attn.attn"),
            _srow("rec", "fused_recurrent_gated_delta_rule_packed_decode_kernel", 7.8, 6,
                  stage="linear_attn", module="linear_attn.attn"),
            _srow("pad", "kernel_moe_gemm", 100.0, 7, stage="moe"),
        ]
        return {"tables": [_stable(rows)]}

    def test_small_gemm_is_not_a_barrier_and_siblings_are_found(self):
        catalog = {"kernels": [{"name": "fused_gemm_a8w8_blockscale_a16w16",
                                "op_tags": ["gemm", "quant"], "dtype_tags": ["fp8", "bf16"],
                                "sources": ["aiter"]}], "tagger": "token-v1"}
        result = screen.screen(screen.load_rows(self._qkvz_table()), catalog=catalog,
                               workload={"isl": 8192, "osl": 1024})
        self.assertFalse(result["kernel_class"]["qkvz"]["large"])
        sibling = [r for r in result["all_records"] if r["kind"] == "sibling"]
        self.assertEqual([r["row_ids"] for r in sibling], [["qkvz", "ba"]])
        record = sibling[0]
        self.assertEqual(record["model"], "sibling_merge")
        # qkvz streams 10.5 MB in 66 us (eff 0.07): its headroom, recovered at
        # rho 0.8 by the existing dual-GEMM kernel, dwarfs the saved launch.
        self.assertGreater(record["saved_us"], LAUNCH_ONLY_FLOOR)
        self.assertGreater(record["eff_saved_us"], 40.0)
        self.assertEqual(record["rho"], screen.RHO_EXISTING)
        self.assertEqual(record["n_after"], 1)
        self.assertEqual([m["name"] for m in record["matched_existing_kernels"]],
                         ["fused_gemm_a8w8_blockscale_a16w16"])
        for item in result["all_records"]:
            if item["passed"]:
                self.assertNotIn("conv", item["row_ids"])
                self.assertNotIn("rec", item["row_ids"])
        self.assertTrue(any(r["kind"] == "sibling" for r in result["initial_topk"]))

    def test_sibling_without_dual_gemm_kernel_has_difficulty_2(self):
        result = screen.screen(screen.load_rows(self._qkvz_table()),
                               workload={"isl": 8192, "osl": 1024})
        sibling = next(r for r in result["all_records"] if r["kind"] == "sibling")
        self.assertEqual(sibling["difficulty"], screen.SIBLING_DIFFICULTY)

    def test_large_gemm_is_a_barrier(self):
        rows = screen.load_rows({"tables": [_stable([
            _srow("q", "quant_cast", 40, 0),
            _srow("g", "gemm_bf16", 2000, 1, dims=[[8192, 5120, 4096]], types=["bf16"]),
            _srow("r", "rotary_embedding_kernel", 40, 2),
        ], layers=40)]})
        result = screen.screen(rows)
        self.assertEqual(result["kernel_class"]["g"]["large_basis"], "time+compute")
        self.assertTrue(all("g" not in r["row_ids"] or r["invalid"] for r in result["all_records"]))


LAUNCH_ONLY_FLOOR = 4.0


class SeamTest(unittest.TestCase):
    def _router_table(self):
        p0 = _router_rows("a", "P0", 0)
        p1 = _router_rows("b", "P1", 100)
        return {"tables": [_stable(p0, total=230.0), _stable(p1, pattern="P1", layers=15, total=267.0)]}

    def test_router_seam_chain_grows_to_8_inside_one_module_and_merges_patterns(self):
        result = screen.screen(screen.load_rows(self._router_table()),
                               workload={"isl": 8192, "osl": 1024})
        top = result["execution_list"][0]
        self.assertEqual(top["pattern_ids"], ["P0", "P1"])
        compute = [c for c in top["classes"] if c != "data_move"]
        self.assertEqual(len(compute), 7)
        self.assertEqual(top["kernel_names"][0][:4], "Cijk")
        self.assertIn("aquant", top["row_ids"])
        self.assertIn("bquant", top["partner_row_ids"])
        self.assertNotIn("astage1", top["row_ids"])
        self.assertAlmostEqual(top["request_saved_us"],
                               sum(m["request_saved_us"] for m in top["merged_members"]), 3)
        # Seven compute kernels with no shared module would be too long.
        self.assertEqual(top["code_seam"], "mlp")

    def test_dedupe_keeps_longest_and_core_and_lists_the_rest(self):
        result = screen.screen(screen.load_rows(self._router_table()),
                               workload={"isl": 8192, "osl": 1024})
        self.assertTrue(result["appendix"])
        for item in result["appendix"]:
            self.assertTrue(item["appendix_reason"])
        p0 = [c for c in result["initial_topk"] if c["pattern_id"] == "P0" and c["kind"] == "serial"]
        self.assertLessEqual(len(p0), 2)

    def test_unknown_member_blocks_existing_kernel(self):
        rows = screen.load_rows({"tables": [_stable([
            _srow("n", "rmsnorm", 40, 0),
            _srow("x", "mystery_kernel", 40, 1),
            _srow("q", "quant_fp8", 40, 2),
            _srow("pad", "kernel_moe_gemm", 1000, 3, stage="moe"),
        ], layers=40)]})
        catalog = {"kernels": [{"name": "fused_rmsnorm_quant_fp8", "op_tags": ["norm", "quant"],
                                "dtype_tags": ["fp8"], "sources": ["aiter"]}], "tagger": "token-v1"}
        result = screen.screen(rows, catalog=catalog)
        chain = next(r for r in result["all_records"] if r["row_ids"] == ["n", "x", "q"])
        self.assertFalse(chain["has_existing_kernel"])
        self.assertEqual(chain["difficulty"], screen.UNKNOWN_DIFFICULTY)
        self.assertEqual(chain["unknown_kernels"], ["mystery_kernel"])
        self.assertIn("unclassified", chain["existing_match_reason"])

    def test_quant_granularity_mismatch_is_not_a_cover(self):
        rows = screen.load_rows({"tables": [_stable([
            _srow("act", "_ZN5aiter18act_and_mul_kernelIfDF16bEEvPT_", 5.1, 0, stage="activation"),
            _srow("q", QUANT, 5.3, 1, stage="quant", dims=QUANT_DIMS, types=QUANT_TYPES),
            _srow("pad", "kernel_moe_gemm", 200, 2, stage="moe"),
        ], layers=60)]})
        catalog = {"kernels": [
            {"name": "fused_silu_mul_fp8_per_tensor_static_quant", "op_tags": ["activation", "quant"],
             "dtype_tags": ["fp8"], "sources": ["aiter"]},
            {"name": "act_mul_and_fp8_group_quant", "op_tags": ["activation", "quant"],
             "dtype_tags": ["fp8", "fp8_blockscale"], "sources": ["aiter"]}], "tagger": "token-v1"}
        result = screen.screen(rows, catalog=catalog)
        chain = next(r for r in result["all_records"] if r["row_ids"] == ["act", "q"])
        self.assertEqual([m["name"] for m in chain["matched_existing_kernels"]],
                         ["act_mul_and_fp8_group_quant"])

    def test_request_level_score_prefers_decode_over_a_bigger_prefill_step(self):
        prefill = [_srow("pn", "rmsnorm", 400.0, 0, phase="prefill", step_input_tokens=16384),
                   _srow("pq", "quant_cast", 400.0, 1, phase="prefill", step_input_tokens=16384),
                   _srow("pg", "kernel_moe_gemm", 5000.0, 2, stage="moe", phase="prefill",
                         step_input_tokens=16384)]
        decode = [_srow("dn", "rmsnorm", 5.0, 0, step_batch_size=4),
                  _srow("dq", "quant_cast", 5.0, 1, step_batch_size=4),
                  _srow("dg", "kernel_moe_gemm", 100.0, 2, stage="moe", step_batch_size=4)]
        table = {"tables": [_stable(prefill, phase="prefill"), _stable(decode)]}
        result = screen.screen(screen.load_rows(table), workload={"isl": 8192, "osl": 1024})
        pre = next(r for r in result["all_records"] if r["row_ids"] == ["pn", "pq"])
        dec = next(r for r in result["all_records"] if r["row_ids"] == ["dn", "dq"])
        self.assertGreater(pre["step_saved_us"], dec["step_saved_us"])
        self.assertGreater(dec["request_saved_us"], pre["request_saved_us"])
        self.assertEqual(result["execution_list"][0]["phase"], "decode")


class PriorTest(unittest.TestCase):
    PRIOR = {"prior_id": "xb-router", "title": "router", "origin": "measured_priors",
             "source": "test", "model": "Qwen3.5", "model_family": "qwen35", "gfx": ["gfx942"],
             "measured_e2e_gain_pct": 6.0,
             "seam": {"kinds": ["serial"], "max_span": 10, "members": [
                 {"cls": "gemm", "module": "mlp.gate"}, {"cls": "topk"},
                 {"cls": "gemm", "module": "mlp.shared_expert_gate"}]}}

    def _rows(self):
        rows = _router_rows("a", "P0", 0)
        # An unrelated, higher-scoring elementwise seam elsewhere in the layer.
        # Shaped and slow (80us on a [4, 4096] tensor): a large efficiency headroom.
        dims, types = [[4, 4096]], ["BFloat16"]
        rows += [_srow("e1", "rmsnorm", 80.0, 50, dims=dims, types=types),
                 _srow("e2", "quant_cast", 80.0, 51, dims=dims, types=types),
                 _srow("e3", "rotary_embedding_kernel", 80.0, 52, dims=dims, types=types),
                 _srow("e4", "silu", 80.0, 53, dims=dims, types=types),
                 _srow("e5", "kernel_moe_gemm", 100.0, 54, stage="moe")]
        return screen.load_rows({"tables": [_stable(rows)]})

    def test_second_pass_is_a_different_ranking(self):
        result = screen.screen(self._rows(), workload={"isl": 8192, "osl": 1024},
                               priors=[self.PRIOR], model="Qwen3.5-397B", top_k=2)
        first, second = result["passes"]
        self.assertTrue(second[0].get("prior_matches"))
        self.assertNotEqual([e["entry_id"] for e in first], [e["entry_id"] for e in second])

    def test_forced_prior_does_not_take_a_k_slot(self):
        result = screen.screen(self._rows(), workload={"isl": 8192, "osl": 1024},
                               priors=[self.PRIOR], model="Qwen3.5-397B", top_k=1)
        forced = [e for e in result["execution_list"] if e.get("forced_by_prior")]
        ranked = [e for e in result["execution_list"] if not e.get("forced_by_prior")]
        self.assertEqual([e["forced_by_prior"] for e in forced], ["xb-router"])
        self.assertLessEqual(len(ranked), 1)
        self.assertLessEqual(len(result["execution_list"]), 1 + screen.FORCED_PRIOR_EXTRA)

    def test_forced_cap_counts_the_whole_list(self):
        # The K+4 cap is on rows, not on forced entries: a short board still takes
        # every measured prior (a flat cap of 4 dropped post-attn AR+Norm+dual
        # quant from a 7-row Qwen3.5 board).
        saved = screen.FORCED_PRIOR_EXTRA
        screen.FORCED_PRIOR_EXTRA = 0
        try:
            result = screen.screen(self._rows(), workload={"isl": 8192, "osl": 1024},
                                   priors=[self.PRIOR], model="Qwen3.5-397B", top_k=8)
        finally:
            screen.FORCED_PRIOR_EXTRA = saved
        rows = result["execution_list"]
        self.assertIn("xb-router", [e.get("forced_by_prior") for e in rows])
        self.assertLessEqual(len(rows), 8)

    def test_entries_carry_request_level_expected_gain(self):
        result = screen.screen(self._rows(), workload={"isl": 8192, "osl": 1024},
                               priors=[self.PRIOR], model="Qwen3.5-397B", top_k=2)
        for entry in result["execution_list"]:
            self.assertGreater(entry["expected_e2e_gain_pct"], 0.0)
            self.assertLess(entry["expected_e2e_gain_pct"], 100.0)

    def test_adjacent_priors_trim_the_earlier_window(self):
        # The router window runs on into moe_sorting + the expert-input quant; a
        # second measured prior (post-attn AR+Norm+dual quant) owns that quant.
        # Both must be forced in, the router entry shrunk to its own seam.
        ar = _srow("ar", "_ZN5aiter30allreduce_fusion_kernel_1stageIDF16bDF16bLi8EEEvP", 7.0, -1,
                   stage="communication")
        rows = screen.load_rows({"tables": [_stable([ar] + _router_rows("a", "P0", 0))]})
        dual = {"prior_id": "xb-dual", "title": "dual", "origin": "measured_priors",
                "source": "test", "model": "Qwen3.5", "model_family": "qwen35", "gfx": ["gfx942"],
                "measured_e2e_gain_pct": 1.08,
                "seam": {"kinds": ["serial"], "max_span": 12, "members": [
                    {"cls": "allreduce"}, {"cls": "gemm", "module": "mlp.gate", "role": "context"},
                    {"cls": "quant", "module": "mlp.experts"}]}}
        result = screen.screen(rows, workload={"isl": 8192, "osl": 1024},
                               priors=[self.PRIOR, dual], model="Qwen3.5-397B", top_k=8)
        by_prior = {e.get("forced_by_prior"): e for e in result["execution_list"]}
        self.assertIn("xb-router", by_prior)
        self.assertIn("xb-dual", by_prior)
        quant_rows = {r["row_id"] for r in rows if r["raw_name"] == QUANT}
        self.assertFalse(quant_rows & screen._entry_rows(by_prior["xb-router"]))
        self.assertTrue(quant_rows & screen._entry_rows(by_prior["xb-dual"]))

    def test_module_named_prior_needs_module_evidence(self):
        # A decode-blind table: router GEMM + shared-gate GEMM rows with no module
        # evidence must not be taken for a GDN in_proj_qkvz + in_proj_ba sibling.
        rows = _router_rows("a", "P0", 0)
        for row in rows:
            row.pop("semantic_evidence", None)
            row["parent_operator"] = {"canonical_op": "aten::mm"} if row["raw_name"].startswith("Cijk") \
                else row["parent_operator"]
        rows = screen.load_rows({"tables": [_stable(rows, phase="prefill")]})
        qkvz = {"prior_id": "xb-qkvz", "title": "qkvz+ba", "origin": "measured_priors",
                "source": "test", "model": "Qwen3.5", "model_family": "qwen35", "gfx": ["gfx942"],
                "measured_e2e_gain_pct": 17.72,
                "seam": {"kinds": ["sibling"], "max_span": 3, "members": [
                    {"cls": "gemm", "module": "in_proj_qkvz"}, {"cls": "gemm", "module": "in_proj_ba"}]}}
        self.assertEqual(screen.seam_occurrences(qkvz, rows), [])
        result = screen.screen(rows, workload={"isl": 8192, "osl": 1024},
                               priors=[qkvz], model="Qwen3.5-397B", top_k=8)
        self.assertFalse(any(e.get("forced_by_prior") == "xb-qkvz" for e in result["execution_list"]))

    def test_prior_for_another_model_family_is_ignored(self):
        result = screen.screen(self._rows(), workload={"isl": 8192, "osl": 1024},
                               priors=[self.PRIOR], model="Llama-3-70B", top_k=1)
        self.assertFalse(any(e.get("forced_by_prior") for e in result["execution_list"]))

    def test_below_noise_band_is_not_forced(self):
        prior = dict(self.PRIOR, measured_e2e_gain_pct=0.5)
        result = screen.screen(self._rows(), workload={"isl": 8192, "osl": 1024},
                               priors=[prior], model="Qwen3.5-397B", top_k=1)
        self.assertFalse(any(e.get("forced_by_prior") for e in result["execution_list"]))

    def test_default_measured_priors_file_loads(self):
        priors = screen.load_all_priors()
        ids = {p["prior_id"] for p in priors}
        self.assertIn("xb-qwen35-qkvz-ba-dual-gemm", ids)
        self.assertIn("xb-qwen35-router-topk", ids)


class RegionCoverageTest(unittest.TestCase):
    def test_every_fusible_region_has_a_candidate_or_a_reason(self):
        rows = [
            _srow("s", "void at::native::vectorized_elementwise_kernel<4, sigmoid>", 5.0, 0,
                  stage="elementwise", op="aten::sigmoid"),
            _srow("c", "void at::native::vectorized_elementwise_kernel<4, copy>", 5.0, 1,
                  stage="elementwise", op="aten::copy_"),
            _srow("g", "kernel_moe_gemm", 100.0, 2, stage="moe"),
            _srow("n", "rmsnorm", 30.0, 3, stage="norm", dims=[[4, 4096], [4096]],
                  types=["BFloat16", "BFloat16"]),
            _srow("q", "quant_cast", 30.0, 4, stage="quant", dims=QUANT_DIMS, types=QUANT_TYPES),
            _srow("g2", "kernel_moe_gemm", 100.0, 5, stage="moe"),
        ]
        table = {"tables": [_stable(rows)]}
        result = screen.screen(screen.load_rows(table), table=table)
        coverage = {tuple(r["row_ids"]): r for r in result["region_coverage"]}
        self.assertEqual(coverage[("s", "c")]["status"], "rejected")
        self.assertIn("only 1 compute", coverage[("s", "c")]["reason"])
        self.assertEqual(coverage[("n", "q")]["status"], "covered")
        import fusion_candidate_harness as harness
        self.assertEqual(harness.check_screen_region_coverage(
            table, {"region_coverage": result["region_coverage"]}), [])
        errors = harness.check_screen_region_coverage(table, {"region_coverage": []})
        self.assertTrue(any("neither a covering candidate" in e for e in errors))


AR = "_ZN5aiter30allreduce_fusion_kernel_1stageIDF16bDF16bLi8EEEvPNS_8RankDataE"
QKVZ_DIMS = [[4, 4096], [2560, 4096], [20, 32], [4, 2560]]
QKVZ_TYPES = ["bfloat16", "float8_e4m3fnuz", "float32", "bfloat16"]


def _schema(*names):
    return {"kernel_shape": {"operator_schema": {"arguments": [
        {"index": index, "name": name} for index, name in enumerate(names)]}}}


def _moe_table(with_ar=True):
    """Decode MoE block: post-attention all-reduce, router seam, expert block."""
    rows = []
    if with_ar:
        rows.append(_srow("ar", AR, 7.3, -1, stage="communication"))
    rows += _router_rows("a", "P0", 0)
    for row in rows:
        if row["row_id"] == "asort":
            # moe_sorting writes the sorted ids and zero-fills a [M, H] output
            # buffer: a workspace, not the hidden state the quant reads.
            row["shape"] = dict({"input_dims": [[4, 11], [4, 11], [8241], [8241], [516], [2], [4, 4096]],
                                 "input_types": ["int", "float", "int", "float", "int", "int", "BFloat16"]},
                                **_schema("topk_ids", "topk_weights", "sorted_token_ids", "sorted_weights",
                                          "sorted_expert_ids", "num_valid_ids", "moe_buf"))
        if row["row_id"] == "aquant":
            row["shape"] = dict(row["shape"], **_schema("out", "input", "scales", "scale_ub"))
    spec = _stable(rows, total=230.0)
    spec["layer_io"] = {"input": {"shape": [4, 4096], "dtype": "bfloat16"}}
    return {"tables": [spec]}


class AlgorithmV4Test(unittest.TestCase):
    """Constants, dataflow graph, efficiency, new window kinds, alt groups (items 0-9)."""

    def test_hardware_defaults_and_overrides(self):
        hw = screen.hardware("gfx950")
        self.assertEqual(hw["hbm_tbps"], 8.0)
        self.assertEqual(hw["peak_tflops"], {"bf16": 2500.0, "fp8": 5000.0})
        self.assertEqual(hw["source"]["hbm"], "datasheet")
        hw = screen.hardware("gfx942", hbm_tbps="4.1")
        self.assertEqual((hw["hbm_tbps"], hw["source"]["hbm"]), (4.1, "cli"))
        old = os.environ.get("GEAK_PEAK_TFLOPS_FP8")
        os.environ["GEAK_PEAK_TFLOPS_FP8"] = "2000"
        try:
            hw = screen.hardware("gfx942")
        finally:
            if old is None:
                os.environ.pop("GEAK_PEAK_TFLOPS_FP8")
            else:
                os.environ["GEAK_PEAK_TFLOPS_FP8"] = old
        self.assertEqual((hw["peak_tflops"]["fp8"], hw["source"]["fp8"]), (2000.0, "env"))
        self.assertEqual(hw["bw_bytes_per_us"], 5.3e6)

    def test_c0_is_the_median_of_the_shortest_five_percent(self):
        rows = [_srow("r%d" % i, "silu", float(d), i) for i, d in enumerate(range(1, 41))]
        rows.append(_srow("p", "silu", 50.0, 0, phase="prefill"))
        c0 = screen.phase_c0_us(screen.load_rows({"tables": [_stable(rows[:40]),
                                                             _stable(rows[40:], phase="prefill")]}))
        self.assertEqual(c0["decode"], 1.5)   # shortest 2 of 40: median(1, 2)
        self.assertEqual(c0["prefill"], 50.0)

    def test_dataflow_infers_edges_by_numel_and_dtype(self):
        table = _moe_table()
        rows = screen.load_rows(table)
        graph = screen.build_dataflow(rows, table)
        by_dst = {}
        for edge in graph["edges"]:
            by_dst.setdefault(edge["dst"], set()).add(edge["src"])
            self.assertTrue(edge["inferred"])
        # The expert-input quant reads [128, 128] bf16: the all-reduce's [4, 4096]
        # output reshaped. Nothing in between writes a tensor of that size.
        self.assertEqual(by_dst["aquant"], {"ar"})
        self.assertEqual(by_dst["agate"], {"ar"})
        self.assertEqual(by_dst["asgate"], {"ar"})
        self.assertGreaterEqual(graph["stats"]["assumed_collective_outputs"], 1)

    def test_explicit_tensor_ids_are_exact_edges(self):
        rows = [_srow("p", "rmsnorm", 5.0, 0, output_tensor_ids=["t1"]),
                _srow("x", "silu", 5.0, 1, output_tensor_ids=["t2"]),
                _srow("k", "quant_cast", 5.0, 2, input_tensor_ids=["t1"])]
        table = {"tables": [_stable(rows)]}
        graph = screen.build_dataflow(screen.load_rows(table), table)
        self.assertEqual([(e["src"], e["dst"], e["inferred"]) for e in graph["edges"]],
                         [("p", "k", False)])

    def test_unshaped_kernel_of_the_readers_module_blocks_inference(self):
        rows = [_srow("w", "rmsnorm", 5.0, 0, module="mlp.experts", dims=[[4, 4096]],
                      types=["BFloat16"], op="aten::mul"),
                _srow("u", "mystery_kernel", 5.0, 1, module="mlp.experts"),
                _srow("q", QUANT, 5.0, 2, stage="quant", module="mlp.experts",
                      dims=QUANT_DIMS, types=QUANT_TYPES)]
        table = {"tables": [_stable(rows)]}
        graph = screen.build_dataflow(screen.load_rows(table), table)
        self.assertNotIn("q", graph["producers"])
        self.assertGreater(graph["stats"]["unresolved_inputs"], 0)

    def test_inefficient_gemm_has_headroom(self):
        row = _srow("g", "_gemm_a8w8_blockscale_kernel_GROUP_K_128", 66.0, 0, stage="gemm",
                    dims=QKVZ_DIMS, types=QKVZ_TYPES)
        cost = screen.kernel_cost(row, screen.hardware("gfx942"), 4.6)
        self.assertTrue(cost["inefficient"])
        self.assertAlmostEqual(cost["ideal_us"], 4.6)
        self.assertLess(cost["eff"], 0.1)
        self.assertAlmostEqual(cost["headroom_us"], 66.0 - 4.6)
        self.assertGreater(cost["bytes"], 2560 * 4096)

    def test_fan_out_and_seam_rewrite_windows(self):
        table = _moe_table()
        result = screen.screen(screen.load_rows(table), table=table,
                               workload={"isl": 8192, "osl": 1024})
        kinds = {}
        for record in result["all_records"]:
            kinds.setdefault(record["kind"], []).append(record)
        fan = [r for r in kinds.get("fan_out", []) if r["row_ids"] == ["ar", "aquant"]]
        self.assertEqual(len(fan), 1, [r["row_ids"] for r in kinds.get("fan_out", [])])
        self.assertEqual(fan[0]["elim_row_ids"], ["aquant"])
        self.assertEqual(fan[0]["n_after"], 1)
        rewrite = kinds["seam_rewrite"]
        self.assertEqual([r["row_ids"] for r in rewrite],
                         [["agate", "atopk", "asgate", "asig", "acopy", "aappend"]])
        self.assertEqual(rewrite[0]["n_after"], 2)
        self.assertEqual(rewrite[0]["code_seam"], "mlp")
        self.assertGreaterEqual(rewrite[0]["difficulty"], screen.SEAM_REWRITE_DIFFICULTY)

    def test_epilogue_forward_skips_non_writers(self):
        rows = [_srow("n", "rmsnorm", 6.0, 0, dims=[[4, 1024], [1024]], types=["BFloat16"] * 2,
                      op="aten::rms_norm"),
                _srow("g", "Cijk_router", 7.0, 1, stage="gemm", op="aten::mm",
                      dims=[[4, 4096], [4096, 512]], types=["BFloat16", "BFloat16"]),
                _srow("q", QUANT, 5.0, 2, stage="quant", dims=[[4, 1024], [32, 128], [4, 8]],
                      types=["Float8_e4m3fnuz", "BFloat16", "float"]),
                _srow("pad", "kernel_moe_gemm", 200.0, 3, stage="moe")]
        table = {"tables": [_stable(rows)]}
        result = screen.screen(screen.load_rows(table), table=table)
        forward = [r for r in result["all_records"] if r["kind"] == "epilogue_forward"]
        self.assertIn(["n", "q"], [r["row_ids"] for r in forward])

    def test_alternatives_for_one_quant_share_a_group(self):
        table = _moe_table()
        result = screen.screen(screen.load_rows(table), table=table,
                               workload={"isl": 8192, "osl": 1024})
        holders = [e for e in result["execution_list"] + result["candidate_pool"]
                   if any("aquant" in rung["all_row_ids"] for rung in e["alt_ladder"])]
        self.assertTrue(holders)
        ladder = holders[0]["alt_ladder"]
        kinds = {rung["kind"] for rung in ladder if "aquant" in rung["all_row_ids"]}
        self.assertIn("fan_out", kinds)
        self.assertTrue(kinds & {"serial", "seam_rewrite"}, ladder)
        self.assertEqual(ladder[0]["entry_id"], holders[0]["entry_id"])
        groups = [e["alt_group"] for e in result["execution_list"]]
        self.assertEqual(len(groups), len(set(groups)))
        taken = set()
        for entry in result["execution_list"]:
            self.assertFalse(taken & set(entry["elim_row_ids"]))
            taken |= set(entry["elim_row_ids"])

    def test_baseline_fused_member_keeps_only_residual_benefit(self):
        rows = [_srow("n", "_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256EEEvPT0_", 60.0, 0,
                      stage="norm", dims=[[4, 4096]] * 4 + [[4096]], types=["BFloat16"] * 5),
                _srow("q", QUANT, 5.1, 1, stage="quant", dims=QUANT_DIMS, types=QUANT_TYPES),
                _srow("pad", "kernel_moe_gemm", 300.0, 2, stage="moe")]
        catalog = {"kernels": [{"name": "add_rmsnorm_quant", "op_tags": ["add_residual", "norm", "quant"],
                                "dtype_tags": [], "sources": ["aiter"]}], "tagger": "token-v1"}
        table = {"tables": [_stable(rows)]}
        result = screen.screen(screen.load_rows(table), table=table, catalog=catalog)
        record = next(r for r in result["all_records"] if r["compute_row_ids"] == ["n", "q"])
        self.assertEqual(record["baseline_fused_row_ids"], ["n"])
        self.assertEqual(record["eff_saved_us"], 0.0)   # n is inefficient, but already fused
        self.assertGreater(result["kernel_costs"]["n"]["headroom_us"], 30.0)

    def test_lower_bound_window_without_speedup_waits_in_the_pool(self):
        rows = screen.load_rows({"tables": [_stable([
            _srow("n", "rmsnorm", 30.0, 0), _srow("q", "quant_cast", 30.0, 1),
            _srow("pad", "kernel_moe_gemm", 100.0, 2, stage="moe"),
            _srow("tiny", "kernel_moe_gemm", 5.0, 3, stage="moe")])]})
        result = screen.screen(rows)
        record = next(r for r in result["all_records"] if r["row_ids"] == ["n", "q"])
        self.assertTrue(record["lower_bound"])
        self.assertEqual(record["gate"], "pool")
        self.assertEqual([e["row_ids"] for e in result["candidate_pool"]], [["n", "q"]])

    def test_isolated_inefficient_kernel_is_handed_off(self):
        rows = [_srow("g", "_gemm_a8w8_blockscale_kernel_GROUP_K_128", 66.0, 0, stage="gemm",
                      dims=QKVZ_DIMS, types=QKVZ_TYPES),
                _srow("pad", "kernel_moe_gemm", 300.0, 1, stage="moe"),
                _srow("tiny", "kernel_moe_gemm", 4.6, 2, stage="moe")]
        table = {"tables": [_stable(rows)]}
        result = screen.screen(screen.load_rows(table), table=table)
        record = next(r for r in result["inefficient_coverage"] if r["row_id"] == "g")
        self.assertEqual((record["status"], record["reason"]),
                         ("handoff", "handoff:HeadKernel/TuningSkillset"))
        import fusion_candidate_harness as harness
        self.assertEqual(harness.check_screen_item_coverage(
            {"inefficient_coverage": result["inefficient_coverage"],
             "seam_coverage": result["seam_coverage"]}), [])
        bad = dict(record, reason="")
        self.assertTrue(harness.check_screen_item_coverage({"inefficient_coverage": [bad]}))

    def test_seam_coverage_lists_every_seam_with_three_kernels(self):
        table = _moe_table()
        result = screen.screen(screen.load_rows(table), table=table,
                               workload={"isl": 8192, "osl": 1024})
        seams = {r["code_seam"]: r for r in result["seam_coverage"]}
        self.assertIn(seams["mlp"]["status"], ("covered", "covered_by_pool"))
        self.assertTrue(seams["mlp"]["covered_by"])

    def test_adaptive_k(self):
        self.assertEqual(screen.adaptive_k(None, 45, 8), 8)
        self.assertEqual(screen.adaptive_k(None, 45, 11), 11)
        self.assertEqual(screen.adaptive_k(3600, 45), 8)                  # floor(0.93) -> 8
        self.assertEqual(screen.adaptive_k(12 * 3600, 45), 11)           # floor(11.2)
        self.assertEqual(screen.adaptive_k(48 * 3600, 45), 16)
        self.assertEqual(screen.adaptive_k(12 * 3600, 30), 16)

    def test_negative_prior_triples_difficulty_and_is_never_forced(self):
        negative = {"prior_id": "neg-silu", "title": "silu+quant regressed", "origin": "measured_priors",
                    "source": "test", "model": "Qwen3.5", "model_family": "qwen35", "gfx": ["gfx942"],
                    "polarity": "negative", "measured_e2e_gain_pct": -2.0,
                    "seam": {"kinds": ["serial", "epilogue_forward"], "max_span": 2,
                             "members": [{"cls": "activation"}, {"cls": "quant"}]}}
        rows = [_srow("act", "_ZN5aiter18act_and_mul_kernelIfDF16bEEvPT_", 5.1, 0, stage="activation"),
                _srow("q", QUANT, 5.3, 1, stage="quant", dims=QUANT_DIMS, types=QUANT_TYPES),
                _srow("pad", "kernel_moe_gemm", 30.0, 2, stage="moe"),
                _srow("n", "rmsnorm", 5.0, 3), _srow("r", "rotary_embedding_kernel", 5.0, 4),
                _srow("pad2", "kernel_moe_gemm", 30.0, 5, stage="moe")]
        result = screen.screen(screen.load_rows({"tables": [_stable(rows)]}),
                               workload={"isl": 8192, "osl": 1024}, priors=[negative],
                               model="Qwen3.5-397B", noise_band_pct=0.1)
        board = result["execution_list"]
        neg = [e for e in board if e["negative_prior"]]
        self.assertEqual(len(neg), 1)
        self.assertEqual(neg[0]["prior_polarity"], "negative")
        self.assertAlmostEqual(neg[0]["difficulty"], neg[0]["base_difficulty"] * 3)
        self.assertIsNone(neg[0]["forced_by_prior"])
        self.assertIs(board[-1], neg[0])

    def test_phase_blind_table_forces_no_prior_without_phases(self):
        table = _moe_table()
        table["phase_coverage"] = {"required_phases": ["decode", "prefill"],
                                   "missing_required_phases": ["prefill"]}
        prior = dict(PriorTest.PRIOR)
        result = screen.screen(screen.load_rows(table), table=table, priors=[prior],
                               model="Qwen3.5-397B", workload={"isl": 8192, "osl": 1024})
        self.assertEqual(result["phase_blind"], ["prefill"])
        self.assertFalse(any(e.get("forced_by_prior") for e in result["execution_list"]))
        status = result["prior_status"][0]
        self.assertTrue(status["forced"].startswith("phase_blind"))
        prior["phases"] = ["decode"]
        result = screen.screen(screen.load_rows(_moe_table()), table=table, priors=[prior],
                               model="Qwen3.5-397B", workload={"isl": 8192, "osl": 1024})
        self.assertIn("xb-router", [e.get("forced_by_prior") for e in result["execution_list"]])

    def test_cli_board_contract(self):
        import tempfile
        table = _moe_table()
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "table.json")
            with open(src, "w") as handle:
                json.dump(table, handle)
            empty = os.path.join(tmp, "priors.json")
            with open(empty, "w") as handle:
                json.dump({"schema_version": 2, "priors": []}, handle)
            out = os.path.join(tmp, "fusion")
            rc = screen.main(["--semantic-table", src, "--out-dir", out, "--isl", "8192",
                              "--osl", "1024", "--conc", "4", "--fusion-budget-s", str(12 * 3600),
                              "--measured-priors", empty, "--priors-index", ""])
            self.assertEqual(rc, 0)
            with open(os.path.join(out, "fusion_topk.json")) as handle:
                topk = json.load(handle)
            with open(os.path.join(out, "fusion_candidates.json")) as handle:
                board = json.load(handle)
        self.assertEqual(topk["k_effective"], 11)
        for key in ("candidate_pool", "k_effective", "noise_band_pct"):
            self.assertIn(key, topk)
            self.assertIn(key, board)
        self.assertTrue(topk["execution_list"])
        for entry in topk["execution_list"] + topk["candidate_pool"]:
            for key in ("alt_group", "alt_ladder", "expected_e2e_gain_pct", "lower_bound",
                        "negative_prior", "prior_polarity"):
                self.assertIn(key, entry)
            self.assertIsNotNone(entry["expected_e2e_gain_pct"])
            self.assertEqual(entry["alt_ladder"][0]["rung"], 0)
            self.assertEqual(entry["alt_ladder"][0]["row_ids"], entry["row_ids"])
            for rung in entry["alt_ladder"]:
                for key in ("rung", "kind", "kernel_names", "row_ids", "matched_existing_kernels",
                            "expected_e2e_gain_pct", "difficulty", "seam_key"):
                    self.assertIn(key, rung)
        for key in ("seam_coverage", "inefficient_coverage", "dataflow", "kernel_efficiency"):
            self.assertIn(key, board)


class NAfterTest(unittest.TestCase):
    """Kernels left after a fusion: GEMMs in a window stay their own launches."""

    def test_window_without_gemm_fuses_to_one_kernel(self):
        rows = [_srow("q1", "rmsnorm", 5.0, 0, stage="norm"),
                _srow("q2", QUANT, 5.0, 1, stage="quant")]
        self.assertEqual(screen._n_after("serial", rows, {}, False), 1)

    def test_each_gemm_stays_plus_one_fused_kernel(self):
        rows = [_srow("g1", "Cijk_Alik_Bljk_router", 7.0, 0, stage="gemm", op="aten::mm"),
                _srow("t1", "topkGatingSoftmax", 10.0, 1, stage="topk", op="aiter::topk_softmax"),
                _srow("g2", "Cijk_Ailk_Bljk_gate", 8.0, 2, stage="gemm", op="aten::mm"),
                _srow("s1", "sigmoid", 4.6, 3, stage="elementwise", op="aten::sigmoid")]
        self.assertEqual(screen._n_after("serial", rows, {}, False), 3)

    def test_a_covering_existing_kernel_is_one_launch(self):
        rows = [_srow("q1", QUANT, 5.0, 0, stage="quant"),
                _srow("g1", "_gemm_a8w8_blockscale_kernel", 20.0, 1, stage="gemm")]
        self.assertEqual(screen._n_after("serial", rows, {}, True), 1)


if __name__ == "__main__":
    unittest.main()
