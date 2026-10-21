"""CPU tests for the UT contracts: families, distributions, bounds, references, and one full
scaffolded task driven through every gate by ut_driver.py.

Run: python3 workload/stages/profile/kernel-extract/ut_contracts/tests/test_ut_contracts.py
The torch-dependent tests skip when torch is absent; the lint and family tests do not need it.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
UTC = os.path.dirname(HERE)
sys.path.insert(0, UTC)
import ut_contract as U  # noqa: E402

try:
    import torch
except ImportError:          # the lint and the family checks still run
    torch = None

needs_torch = unittest.skipIf(torch is None, "torch is not installed")


class FamilyTest(unittest.TestCase):
    def test_every_family_validates_and_names_a_known_reference(self):
        if torch is not None:
            import reference_f64 as R
        for f in sorted(os.listdir(U.FAMILIES_DIR)):
            doc = U.load_family(f[:-5])
            self.assertEqual(doc["family"], f[:-5])
            for key in ("hooks", "call", "timing", "negative_controls", "pitfalls"):
                self.assertIn(key, doc, "%s lacks %s" % (f, key))
            self.assertTrue(doc["negative_controls"], f)
            if doc.get("reference") and torch is not None:
                self.assertIn(doc["reference"], R.REFERENCES, f)
            for name, s in doc["inputs"].items():
                hook = (s.get("dist") or {}).get("hook") or (s.get("layout") or {}).get("hook")
                if hook:
                    self.assertIn(hook, doc["hooks"], "%s: %s uses undeclared hook %s" % (f, name, hook))

    def test_expressions_are_sandboxed(self):
        self.assertEqual(U.evaluate("ceil_div(k, 128)", {"k": 7168}), 56)
        self.assertTrue(U.evaluate("topk <= n and mode == 'K'", {"topk": 2, "n": 4, "mode": "K"}))
        for bad in ("__import__('os')", "k.real", "[x for x in (1,)]", "open('f')"):
            with self.assertRaises(U.ContractError):
                U.evaluate(bad, {"k": 1})

    def test_freeze_pins_attrs_and_axes_and_refuses_unjustified_overrides(self):
        fam = U.load_family("gemm_a8w8_blockscale")
        doc = U.freeze_task_contract(fam, attrs={"fp8_dtype": "float8_e4m3fn"},
                                     const_axes={"n": 7168, "k": 2048})
        sym = U.symbols_for(doc, {"m": 3})
        self.assertEqual((sym["scale_n"], sym["scale_k"], sym["scale_major"]), (56, 16, "K"))
        with self.assertRaises(U.ContractError):
            U.symbols_for(doc, {"m": 0})                      # below lo
        with self.assertRaises(U.ContractError):
            U.freeze_task_contract(fam, attrs={}, const_axes={"k": 100})   # not a multiple of 128
        with self.assertRaises(U.ContractError):
            U.freeze_task_contract(fam, attrs={}, const_axes={"n": 16, "k": 128},
                                   overrides=[{"path": "compare.max_violation_fraction", "value": 0.1}])
        ok = U.freeze_task_contract(fam, attrs={}, const_axes={"n": 16, "k": 128},
                                    overrides=[{"path": "compare.max_violation_fraction", "value": 0.1,
                                                "why": "measured baseline violation on gfx942"}])
        self.assertEqual(ok["compare"]["max_violation_fraction"], 0.1)

    def test_constraints_are_enforced_per_case(self):
        doc = U.freeze_task_contract(U.load_family("moe_grouped_topk"), attrs={},
                                     const_axes={"n_experts": 256, "n_group": 8, "topk_group": 4, "topk": 8})
        U.symbols_for(doc, {"n_tokens": 4})
        bad = U.freeze_task_contract(U.load_family("moe_grouped_topk"), attrs={},
                                     const_axes={"n_experts": 250, "n_group": 8, "topk_group": 4, "topk": 8})
        with self.assertRaises(U.ContractError):
            U.symbols_for(bad, {"n_tokens": 4})


@needs_torch
class DistributionTest(unittest.TestCase):
    def doc(self, family, attrs, axes):
        return U.freeze_task_contract(U.load_family(family), attrs=attrs, const_axes=axes)

    def test_draws_are_reproducible_and_shaped(self):
        doc = self.doc("gemm", {"act_dtype": "bfloat16"}, {"n": 64, "k": 128})
        make = lambda g: U.build_inputs(doc, {"m": 5}, g, "cpu")[0]
        same, _ = U.check_reproducible(make, 7, "cpu")
        self.assertTrue(same)
        args = make(torch.Generator().manual_seed(7))
        self.assertEqual(tuple(args["a"].shape), (5, 128))
        self.assertEqual(tuple(args["b"].shape), (64, 128))
        self.assertEqual(args["a"].dtype, torch.bfloat16)
        self.assertNotIn("bias", args)
        self.assertAlmostEqual(float(args["b"].float().std()), 1 / math.sqrt(128), delta=0.02)

    def test_nondeterministic_builder_is_caught(self):
        state = {"n": 0}

        def make(g):
            state["n"] += 1
            return {"x": torch.full((3,), float(state["n"]))}
        same, _ = U.check_reproducible(make, 0, "cpu")
        self.assertFalse(same)

    def test_routing_ids_are_distinct_and_skew_follows_counts(self):
        doc = self.doc("fused_moe", {"quant": "bf16"},
                       {"model_dim": 128, "inter_dim": 64, "n_experts": 8, "topk": 2})
        spec = dict(doc["inputs"]["topk_ids"])
        sym = U.symbols_for(doc, {"n_tokens": 4000})
        ids, _ = U.draw("topk_ids", spec, sym, torch.Generator().manual_seed(0), "cpu")
        self.assertTrue(U.check_input_values(doc, {"topk_ids": ids}, sym)[0])
        spec["dist"] = {"kind": "routing", "experts": "n_experts", "load": "observed",
                        "counts": [1000, 1, 1, 1, 1, 1, 1, 1]}
        ids, _ = U.draw("topk_ids", spec, sym, torch.Generator().manual_seed(0), "cpu")
        hist = torch.bincount(ids.flatten().long(), minlength=8)
        self.assertGreater(int(hist.max()), 3900)               # the hot expert is in nearly every row
        self.assertTrue(400 < int(hist.min()) < 750)            # the other 7 share the second slot

    def test_layout_hooks_keep_logical_row_major(self):
        doc = self.doc("gemm", {"layout": "b_preshuffled_16x16"}, {"n": 32, "k": 64})

        def shuffle(t, name, sym):
            s = t.flip(-1).contiguous()
            s.is_shuffled = True
            return s
        args, logical, sym = U.build_inputs(doc, {"m": 2}, torch.Generator().manual_seed(1), "cpu",
                                            hooks={"preshuffle_16x16": shuffle})
        self.assertTrue(U.check_layout_facts(doc, args, sym)[0])
        self.assertTrue(torch.equal(logical["b"], args["b"].flip(-1).double()))
        plain = U.build_inputs(doc, {"m": 2}, torch.Generator().manual_seed(1), "cpu",
                               hooks={"preshuffle_16x16": lambda t, name, sym: t})[0]
        self.assertFalse(U.check_layout_facts(doc, plain, sym)[0])   # the dropped is_shuffled marker

    def test_sparse_indices_are_distinct_and_padded(self):
        doc = self.doc("sparse_mla", {"kv_dtype": "bfloat16"},
                       {"heads": 2, "d_v": 16, "d_rope": 8, "topk": 64})
        args, _, sym = U.build_inputs(doc, {"n_tokens": 3, "context": 40, "pool": 50},
                                      torch.Generator().manual_seed(0), "cpu")
        idx = args["indices"].reshape(3, -1)
        self.assertTrue(bool((idx[:, 40:] == -1).all()))
        valid = idx[:, :40]
        self.assertTrue(bool((valid >= 0).all()) and bool((valid < 40).all()))
        self.assertEqual(int(valid.sort(-1).values.diff(dim=-1).eq(0).sum()), 0)

    def test_inputs_unmodified_catches_in_place_writes(self):
        args = {"x": torch.ones(4), "_sig": "s"}
        ok, changed, _ = U.inputs_unmodified(lambda a: a["x"].mul_(2), args)
        self.assertEqual((ok, changed), (False, ["x"]))
        ok, _, _ = U.inputs_unmodified(lambda a: a["x"].clone().mul_(2), {"x": torch.ones(4)})
        self.assertTrue(ok)


@needs_torch
class BoundTest(unittest.TestCase):
    def gemm_case(self, m=64, n=48, k=512, seed=0):
        import reference_f64 as R
        doc = U.freeze_task_contract(U.load_family("gemm"), attrs={"act_dtype": "bfloat16"},
                                     const_axes={"n": n, "k": k})
        args, logical, sym = U.build_inputs(doc, {"m": m}, torch.Generator().manual_seed(seed), "cpu")
        refs, mags = R.reference("gemm", logical, sym)
        return doc, args, sym, refs, mags

    def test_dot_product_accepts_fp32_accumulation_and_rejects_a_dropped_tile(self):
        doc, args, sym, refs, mags = self.gemm_case()
        good = (args["a"].float() @ args["b"].float().T).to(torch.bfloat16)
        split = (args["a"][:, :256].float() @ args["b"][:, :256].float().T
                 + args["a"][:, 256:].float() @ args["b"][:, 256:].float().T).to(torch.bfloat16)
        bad = (args["a"][:, :496].float() @ args["b"][:, :496].float().T).to(torch.bfloat16)
        ok = U.compare_outputs(doc, good, good, sym=sym, refs=refs, magnitudes=mags)
        self.assertTrue(ok["ok"], ok)
        self.assertTrue(U.compare_outputs(doc, split, good, sym=sym, refs=refs, magnitudes=mags)["ok"])
        res = U.compare_outputs(doc, bad, good, sym=sym, refs=refs, magnitudes=mags)
        self.assertFalse(res["ok"])
        self.assertGreater(U.control_margin(res), 10)

    def test_dot_product_rejects_a_bf16_accumulator(self):
        doc, args, sym, refs, mags = self.gemm_case(k=4096)
        acc = torch.zeros(args["a"].shape[0], args["b"].shape[0], dtype=torch.bfloat16)
        for k0 in range(0, 4096, 64):
            acc = (acc.float() + args["a"][:, k0:k0 + 64].float() @ args["b"][:, k0:k0 + 64].float().T).to(torch.bfloat16)
        res = U.compare_outputs(doc, acc, acc, sym=sym, refs=refs, magnitudes=mags)
        self.assertFalse(res["ok"])

    def test_baseline_relative_scales_with_the_baselines_own_error(self):
        y = torch.randn(32, 64, dtype=torch.float64)
        noisy = (y + 1e-2 * torch.randn(32, 64, dtype=torch.float64)).float()
        twice = (y + 3e-2 * torch.randn(32, 64, dtype=torch.float64)).float()
        b = {"kind": "baseline_relative"}
        self.assertTrue(U.compare_one(b, noisy, noisy, ref=y, out_dtype="float32")["ok"])
        self.assertFalse(U.compare_one(b, twice, noisy, ref=y, out_dtype="float32")["ok"])
        biased = (y + 5e-3).float()        # within 2x the error, but systematically shifted
        self.assertFalse(U.compare_one(b, biased, noisy, ref=y, out_dtype="float32")["ok"])

    def test_noise_relative_uses_two_baseline_runs(self):
        g = torch.Generator().manual_seed(3)
        b0 = torch.randn(16, 128, generator=g)
        b1 = b0 + 1e-3 * torch.randn(16, 128, generator=g)       # atomics reorder: run-to-run noise
        within = b0 + 1e-3 * torch.randn(16, 128, generator=g)   # a candidate as noisy as the baseline
        beyond = b0 + 5e-2 * torch.randn(16, 128, generator=g)
        bound = {"kind": "noise_relative", "multiple": 4.0, "rounding_multiple": 0.0}
        kw = dict(base_noise=b1, out_dtype="float32", max_violation_fraction=5e-4)
        self.assertTrue(U.compare_one(bound, within, b0, **kw)["ok"])
        self.assertFalse(U.compare_one(bound, beyond, b0, **kw)["ok"])

    def test_topk_equivalent_accepts_tie_order_and_rejects_a_wrong_pick(self):
        scores = torch.tensor([[5.0, 3.0, 3.0, 1.0, 0.0, 9.0]])
        base = torch.tensor([[5, 0, 1, -1]])          # 9, 5, 3 (one of the tied 3s), pad
        tie = torch.tensor([[0, 2, 5, -1]])            # same scores, other tie, other order
        wrong = torch.tensor([[5, 0, 3, -1]])          # picks the 1.0
        dup = torch.tensor([[5, 5, 0, -1]])
        bound = {"kind": "topk_equivalent", "scores": "scores", "pad": -1}
        self.assertTrue(U.compare_one(bound, tie, base, scores=scores)["ok"])
        self.assertEqual(U.compare_one(bound, wrong, base, scores=scores)["reasons"], {"different_scores": 1})
        self.assertEqual(U.compare_one(bound, dup, base, scores=scores)["reasons"], {"duplicate": 1})
        # the legacy relative-tol compare accepts index 1001 for 1000; this does not
        big = torch.arange(2000.0).flip(0)[None]
        self.assertFalse(U.compare_one(bound, torch.tensor([[1001]]), torch.tensor([[1000]]), scores=big)["ok"])
        self.assertTrue(U.compare_one({"kind": "harness", "tol": 0.01},
                                      torch.tensor([[1001]]), torch.tensor([[1000]]))["ok"])

    def test_bytes_exact_and_multiset_and_sort_by(self):
        a = torch.tensor([[1, 2, 3]], dtype=torch.uint8)
        self.assertFalse(U.compare_one({"kind": "bytes_exact"}, a, a + 1)["ok"])
        self.assertTrue(U.compare_one({"kind": "multiset_rows"}, torch.tensor([[3, 1, 2]]), torch.tensor([[1, 2, 3]]))["ok"])
        doc = U.freeze_task_contract(U.load_family("moe_grouped_topk"), attrs={},
                                     const_axes={"n_experts": 8, "n_group": 2, "topk_group": 2, "topk": 2})
        sym = U.symbols_for(doc, {"n_tokens": 1})
        cand = (torch.tensor([[0.3, 0.7]]), torch.tensor([[4, 1]]))
        base = (torch.tensor([[0.7, 0.3]]), torch.tensor([[1, 4]]))
        self.assertTrue(U.compare_outputs(doc, cand, base, sym=sym)["ok"])
        swapped = (torch.tensor([[0.7, 0.3]]), torch.tensor([[4, 1]]))
        self.assertFalse(U.compare_outputs(doc, swapped, base, sym=sym)["ok"])

    def test_sqnr(self):
        y = torch.randn(1000, dtype=torch.float64)
        self.assertTrue(U.compare_one({"kind": "sqnr", "min_db": 20}, (y * 1.01).float(), y.float(), ref=y)["ok"])
        self.assertFalse(U.compare_one({"kind": "sqnr", "min_db": 20}, (y * 1.5).float(), y.float(), ref=y)["ok"])


@needs_torch
class ReferenceTest(unittest.TestCase):
    def setUp(self):
        import reference_f64 as R
        self.R = R
        self.g = torch.Generator().manual_seed(0)

    def rn(self, *shape):
        return torch.randn(*shape, generator=self.g, dtype=torch.float64)

    def test_e2m1_and_e8m0_decoding(self):
        packed = torch.tensor([[0x21, 0xF7]], dtype=torch.uint8)     # lo nibble first
        self.assertEqual(self.R.decode_e2m1(packed).tolist(), [[0.5, 1.0, 6.0, -6.0]])
        self.assertEqual(self.R.decode_e8m0(torch.tensor([127, 128, 125], dtype=torch.uint8)).tolist(), [1.0, 2.0, 0.25])

    def test_blockscale_gemm_matches_explicit_dequant(self):
        a, b = self.rn(4, 256), self.rn(130, 256)
        a_s, b_s = self.rn(4, 2).abs(), self.rn(2, 2).abs()
        out, _ = self.R.gemm_a8w8_blockscale({"a": a, "b": b, "a_scale": a_s, "b_scale": b_s}, {})
        da = a * a_s.repeat_interleave(128, 1)
        db = b * b_s.repeat_interleave(128, 0)[:130].repeat_interleave(128, 1)
        self.assertTrue(torch.allclose(out["out"], da @ db.T))

    def test_moe_matches_a_dense_loop(self):
        T, D, I, E, K = 5, 8, 6, 4, 2
        x, w1, w2 = self.rn(T, D), self.rn(E, 2 * I, D), self.rn(E, D, I)
        ids = torch.tensor([[0, 1], [1, 2], [3, 0], [2, 3], [1, 0]])
        w = torch.rand(T, K, generator=self.g, dtype=torch.float64)
        out, _ = self.R.fused_moe({"hidden_states": x, "w1": w1, "w2": w2, "topk_ids": ids, "topk_weights": w},
                                  {"activation": "silu"})
        want = torch.zeros(T, D, dtype=torch.float64)
        for t in range(T):
            for j in range(K):
                h = w1[ids[t, j]] @ x[t]
                g, u = h[:I], h[I:]
                want[t] += w[t, j] * (w2[ids[t, j]] @ (g * torch.sigmoid(g) * u))
        self.assertTrue(torch.allclose(out["out"], want))
        rows = torch.tensor([1, 3])
        sub, _ = self.R.fused_moe({"hidden_states": x, "w1": w1, "w2": w2, "topk_ids": ids, "topk_weights": w},
                                  {"activation": "silu"}, rows)
        self.assertTrue(torch.allclose(sub["out"], want[rows]))

    def test_sparse_mla_equals_dense_attention_over_the_selected_rows(self):
        T, H, dv, dr, pool = 3, 2, 8, 4, 20
        qn, qr, kv = self.rn(T, H, dv), self.rn(T, H, dr), self.rn(pool, 1, dv + dr)
        idx = torch.tensor([[[3, 7, -1, 1]], [[-1, -1, -1, -1]], [[0, 19, 5, 6]]])
        out, _ = self.R.sparse_mla({"q_nope": qn, "q_rope": qr, "kv": kv, "indices": idx},
                                   {"d_v": dv, "sm_scale": 0.3})
        q = torch.cat([qn, qr], -1)
        rows = kv[[3, 7, 1], 0]
        p = torch.softmax(q[0] @ rows.T * 0.3, -1)
        self.assertTrue(torch.allclose(out["output"][0], p @ rows[:, :dv]))
        self.assertTrue(bool((out["output"][1] == 0).all()) and bool(torch.isinf(out["lse"][1]).all()))

    def test_paged_decode_equals_contiguous_attention(self):
        B, H, Hkv, D, bs, nb = 2, 4, 2, 8, 4, 6
        q, kc, vc = self.rn(B, H, D), self.rn(nb, bs, Hkv, D), self.rn(nb, bs, Hkv, D)
        bt = torch.tensor([[5, 2, 0], [1, 4, 3]])
        lens = torch.tensor([9, 5])
        out, _ = self.R.paged_attention_decode({"query": q, "k_cache": kc, "v_cache": vc,
                                                "block_table": bt, "seq_lens": lens}, {"scale": 0.5})
        k = kc[bt[0]].reshape(-1, Hkv, D)[:9]
        v = vc[bt[0]].reshape(-1, Hkv, D)[:9]
        h = 3                                            # kv head 1
        p = torch.softmax(q[0, h] @ k[:, 1].T * 0.5, -1)
        self.assertTrue(torch.allclose(out["out"][0, h], p @ v[:, 1]))

    def test_varlen_causal_is_bottom_right_aligned(self):
        q, k, v = self.rn(3, 1, 4), self.rn(5, 1, 4), self.rn(5, 1, 4)
        out, _ = self.R.varlen_attention({"q": q, "k": k, "v": v, "cu_seqlens_q": torch.tensor([0, 3]),
                                          "cu_seqlens_k": torch.tensor([0, 5])}, {"scale": 1.0, "causal": True})
        p = torch.softmax(q[0, 0] @ k[:3, 0].T, -1)      # query 0 of 3 over 5 keys sees keys 0..2
        self.assertTrue(torch.allclose(out["out"][0, 0], p @ v[:3, 0]))

    def test_rope_neox_rotates_halves(self):
        x = self.rn(2, 8)
        cache = torch.cat([torch.cos(torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.float64)),
                           torch.sin(torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.float64))], -1)
        out, _ = self.R.rope({"positions": torch.tensor([0, 1]), "query": x, "cos_sin_cache": cache},
                             {"head_size": 8, "style": "neox"})
        self.assertTrue(torch.allclose(out["query_out"][0], x[0]))     # position 0: identity
        x1, x2 = x[1, :2], x[1, 2:4]
        c, s = torch.cos(torch.tensor([1.0, 2.0], dtype=torch.float64)), torch.sin(torch.tensor([1.0, 2.0], dtype=torch.float64))
        self.assertTrue(torch.allclose(out["query_out"][1, :4], torch.cat([x1 * c - x2 * s, x2 * c + x1 * s])))

    def test_grouped_topk_respects_groups(self):
        logits = torch.tensor([[9.0, 8.0, 0.0, 0.0, 7.5, 0.0, 0.0, 0.0]])
        out, _ = self.R.moe_grouped_topk({"gating_output": logits},
                                         {"n_group": 4, "topk_group": 1, "topk": 2, "routed_scaling_factor": 2.0})
        self.assertEqual(sorted(out["topk_ids"][0].tolist()), [0, 1])
        self.assertAlmostEqual(float(out["topk_weights"].sum()), 2.0)


# Small legal points for every family: (attrs, const axes, case axes).
SMALL = {
    "gemm": ({}, {"n": 32, "k": 64}, {"m": 3}),
    "gemm_a8w8": ({"q_dtype": "float8_e4m3fn"}, {"n": 32, "k": 128}, {"m": 3}),
    "gemm_a8w8_blockscale": ({}, {"n": 160, "k": 256}, {"m": 3}),
    "gemm_a4w4_mxfp4": ({}, {"n": 32, "k": 64}, {"m": 3}),
    "fused_moe": ({"quant": "bf16"}, {"model_dim": 128, "inter_dim": 64, "n_experts": 4, "topk": 2}, {"n_tokens": 5}),
    "moe_grouped_topk": ({}, {"n_experts": 16, "n_group": 4, "topk_group": 2, "topk": 4}, {"n_tokens": 5}),
    "rmsnorm": ({}, {"hidden": 64}, {"rows": 4}),
    "rmsnorm_quant": ({"quant": "fp8"}, {"hidden": 256, "group": 128}, {"rows": 4}),
    "act_mul": ({}, {"width": 64}, {"rows": 4}),
    "rope": ({}, {"q_heads": 2, "kv_heads": 1, "head_size": 16, "rot_dim": 8, "max_pos": 32}, {"n_tokens": 5}),
    "kv_cache_write": ({"cache_dtype": "bfloat16"}, {"pool_slots": 32, "kv_heads": 2, "head_size": 8}, {"n_tokens": 5}),
    "topk": ({}, {"topk": 4}, {"n_rows": 3, "n_candidates": 20}),
    "dsa_indexer": ({"has_k_scale": False}, {"heads": 2, "dim": 16, "topk": 4}, {"n_tokens": 3, "n_keys": 20}),
    "sparse_mla": ({"kv_dtype": "bfloat16"}, {"heads": 2, "d_v": 16, "d_rope": 8, "topk": 8}, {"n_tokens": 3, "context": 12, "pool": 16}),
    "paged_attention_decode": ({}, {"heads": 4, "kv_heads": 2, "head_size": 16, "block_size": 4},
                               {"batch": 2, "max_context": 10, "num_blocks": 8}),
    "varlen_attention": ({}, {"heads": 2, "kv_heads": 1, "head_size": 16, "v_head_size": 16},
                         {"batch": 2, "total_q": 6, "total_k": 10}),
    "all_reduce": ({"act_dtype": "float32"}, {"hidden": 16, "world": 4}, {"n_tokens": 3}),
}


def _generic_hooks(doc):
    """Stand-ins for the hooks a task writes from the installed stack: right shapes and dtypes."""
    def draw_hook(name, spec, shape, dtype, sym, gen, device):
        names = [name] + [n for n, s in doc["inputs"].items()
                          if (s.get("dist") or {}).get("kind") == "produced" and s["dist"]["by"] == name]
        out = {}
        for n in names:
            s = doc["inputs"][n]
            if not U.present(s, sym):
                continue
            shp = U.shape_of(s, sym)
            if n in ("ks", "cu_seqlens_q"):
                continue
            dt = U.dtype_of(s, sym)
            if dt in U.INTEGER_DTYPES:
                t = torch.randint(0, 16, shp, generator=gen).to(U.torch_dtype(dt))
            else:
                t = torch.randn(shp, generator=gen).to(U.torch_dtype(dt))
            out[n] = (t, None)
        if name == "ks":
            n_keys = sym["n_keys"]
            ke = torch.randint(1, n_keys + 1, shape, generator=gen)
            out = {"ks": (torch.zeros(shape, dtype=torch.int32), None), "ke": (ke.to(torch.int32), None)}
        if name == "cu_seqlens_q":
            cq = torch.tensor([0, 2, sym["total_q"]], dtype=torch.int32)
            ck = torch.tensor([0, 5, sym["total_k"]], dtype=torch.int32)
            out = {"cu_seqlens_q": (cq, None), "cu_seqlens_k": (ck, None)}
        if name == "slot_mapping":
            out = {"slot_mapping": (torch.randperm(sym["pool_slots"], generator=gen)[: shape[0]], None)}
        if name == "cos_sin_cache":
            ang = torch.rand(shape[0], shape[1] // 2, generator=gen)
            out = {"cos_sin_cache": (torch.cat([ang.cos(), ang.sin()], -1), None)}
        if name in ("a", "b") and "scale_input" in spec["dist"]:          # mxfp4: logical = float
            packed, scales = out[name][0], out[spec["dist"]["scale_input"]][0]
            import reference_f64 as R
            out[name] = (packed, R.dequant_mxfp4(packed, (scales % 4 + 125).to(torch.uint8)))
        if name == "w1":                                                   # moe: logical float weights
            out = {k: (v[0], torch.randn(U.shape_of(doc["inputs"][k], sym), generator=gen, dtype=torch.float64))
                   for k, v in out.items()}
        return out

    def layout_hook(t, name, sym):
        t = t.clone()
        t.is_shuffled = True
        return t
    hooks = {}
    for h in doc.get("hooks", {}):
        is_layout = any((s.get("layout") or {}).get("hook") == h for s in doc["inputs"].values())
        hooks[h] = layout_hook if is_layout else draw_hook
    return hooks


@needs_torch
class EveryFamilyBuildsTest(unittest.TestCase):
    def test_contract_draws_and_reference_agree_on_shapes(self):
        import reference_f64 as R
        self.assertEqual(set(SMALL), {f[:-5] for f in os.listdir(U.FAMILIES_DIR)})
        for family, (attrs, axes, case) in SMALL.items():
            with self.subTest(family=family):
                doc = U.freeze_task_contract(U.load_family(family), attrs=attrs, const_axes=axes)
                args, logical, sym = U.build_inputs(doc, case, torch.Generator().manual_seed(0), "cpu",
                                                    hooks=_generic_hooks(doc))
                for name, s in doc["inputs"].items():
                    if s.get("kind") == "scalar" or not U.present(s, sym):
                        continue
                    self.assertIn(name, args)
                    self.assertEqual(tuple(args[name].shape), U.shape_of(s, sym), name)
                    self.assertEqual(str(args[name].dtype), "torch." + U.dtype_of(s, sym), name)
                self.assertTrue(U.check_layout_facts(doc, args, sym)[0])
                if not doc.get("reference"):
                    continue
                refs, _ = R.reference(doc["reference"], logical, sym)
                for name, s in doc["outputs"].items():
                    if U.present(s, sym) and name in refs:
                        self.assertEqual(tuple(refs[name].shape), U.shape_of(s, sym), name)


TOY_KERNEL = textwrap.dedent('''\
    import torch


    def gemm(a, b):
        """toy installed GEMM: fp32 accumulation, bf16 out"""
        return (a.float() @ b.float().T).to(a.dtype)
    ''')


@needs_torch
class EndToEndTest(unittest.TestCase):
    """Scaffold a gemm task around a toy installed kernel and run ut_driver.py's gates on the CPU.

    The baseline is `toyk.ops:gemm` from a site directory outside the task; the candidate is the
    task's kernel_src copy bound through the overlay, exactly as on a GPU node."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="utc_e2e_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        site = os.path.join(self.tmp, "site", "toyk")
        os.makedirs(site)
        open(os.path.join(site, "__init__.py"), "w").close()
        with open(os.path.join(site, "ops.py"), "w") as f:
            f.write(TOY_KERNEL)
        self.env = dict(os.environ, PYTHONPATH=os.path.join(self.tmp, "site"), HARNESS_CACHE_FLUSH_MB="1")
        self.task = os.path.join(self.tmp, "eval", "tasks", "toy_gemm")
        manifest = {"cases": [
            {"sig": "DECODE_m4", "regime": "DECODE", "axes": {"m": 4}, "effective_calls": 900, "timed": True, "random": True},
            {"sig": "PREFILL_m200", "regime": "PREFILL", "axes": {"m": 200}, "effective_calls": 3, "timed": True, "random": True},
        ], "replay_sigs": []}
        regimes = {"DECODE": {"cuda_graph": False}, "PREFILL": {"cuda_graph": False}}
        for name, obj in (("m.json", manifest), ("r.json", regimes)):
            with open(os.path.join(self.tmp, name), "w") as f:
                json.dump(obj, f)
        self.run_py([os.path.join(UTC, "scaffold_ut.py"), "new", "--family", "gemm", "--task", self.task,
                     "--target", "toyk.ops:gemm", "--bind-module", "toyk.ops", "--bind-file", "kernel_src/ops.py",
                     "--attr", "act_dtype=bfloat16", "--axis", "n=48", "--axis", "k=256",
                     "--manifest", os.path.join(self.tmp, "m.json"),
                     "--phase-regimes", os.path.join(self.tmp, "r.json")])
        with open(os.path.join(self.task, "kernel_src", "ops.py"), "w") as f:
            f.write(TOY_KERNEL)
        cases = os.path.join(self.task, "cases.py")
        with open(cases) as f:
            text = f.read()
        text = text.replace('''    raise NotImplementedError("FILL call(): bind toyk.ops:gemm with the deployed caller's "
                              "arguments and return (out)")''', '''    return _entry()(args["a"], args["b"])''')
        with open(cases, "w") as f:
            f.write(text)
        self.meta_path = os.path.join(self.task, "meta.json")
        with open(self.meta_path) as f:
            meta = json.load(f)
        meta["workload"]["serving_weight_model"]["evidence_path"] = "run/observed_workload.json"
        meta["ut_contract"]["negative_controls"] = [{
            "leaf": "kernel_src/ops.py", "find": "a.float() @ b.float().T",
            "replace": "a[:, :-16].float() @ b[:, :-16].float().T",
            "expect": "the last 16 of K dropped: error ~|a_k b_k|, far outside dot_product"}]
        self.write_meta(meta)

    def write_meta(self, meta):
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def run_py(self, argv, cwd=None):
        p = subprocess.run([sys.executable] + argv, env=self.env, cwd=cwd, capture_output=True, text=True, timeout=900)
        return p

    def ut(self, *extra):
        p = self.run_py([os.path.join(self.task, "unittest.py"), *extra], cwd=self.task)
        with open(os.path.join(self.task, "run", "ut_report.json")) as f:
            return p, json.load(f)

    def test_unmodified_extraction_passes_every_gate(self):
        lint = self.run_py([os.path.join(self.task, "ut_contract.py"), self.task])
        self.assertEqual(lint.returncode, 0, lint.stdout + lint.stderr)
        p, rep = self.ut("--max-reps", "1")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr + json.dumps(rep, indent=1, default=str)[:4000])
        self.assertEqual(rep["verdict"], "pass")
        self.assertEqual(rep["gates"]["replay"]["status"], "not_required")
        self.assertIn("GEAK_WEIGHTED_SPEEDUP=", p.stdout)
        ctrl = rep["gates"]["negative_controls"]["controls"][0]
        self.assertTrue(ctrl["caught"] and ctrl["restored"] and ctrl["margin"] >= 10, ctrl)
        last = json.loads(p.stdout.strip().splitlines()[-1])           # worker.sh's verdict line
        self.assertEqual(last["exit"], 0)
        self.assertAlmostEqual(last["weighted"], rep["gates"]["timing"]["weighted_speedup"])
        # the optimize stage's entry runner accepts the receipt as a passing, fresh result
        entry = os.path.join(UTC, *[".."] * 4, "scripts", "run_task_entry.py")
        if os.path.exists(entry):
            r = self.run_py([entry, self.task, "unittest.py", "--skip", "timing,negative_controls"])
            self.assertEqual(r.returncode, 1, r.stdout[-2000:] + r.stderr[-2000:])   # partial run
            r = self.run_py([entry, self.task, "unittest.py", "--max-reps", "1"])
            docs = [json.loads(ln) for ln in r.stdout.splitlines() if ln.startswith('{"ut_result"')]
            self.assertTrue(docs and docs[-1]["ok"], r.stdout[-3000:] + r.stderr[-2000:])

    def test_wrong_candidate_fails_correctness_and_stays_failed(self):
        with open(os.path.join(self.task, "kernel_src", "ops.py"), "w") as f:
            f.write(TOY_KERNEL.replace("(a.float() @ b.float().T)", "(a.float() @ b.float().T * 1.01)"))
        p, rep = self.ut("--skip", "timing")
        self.assertEqual(p.returncode, 1)
        self.assertEqual(rep["gates"]["correctness"]["status"], "fail")

    def test_a_control_that_changes_nothing_fails_the_gate(self):
        with open(self.meta_path) as f:
            meta = json.load(f)
        meta["ut_contract"]["negative_controls"][0]["replace"] = "(a.float() @ b.float().T) + 0.0 * a.sum()"
        self.write_meta(meta)
        p, rep = self.ut("--skip", "timing", "--keep-going")
        nc = rep["gates"]["negative_controls"]
        self.assertEqual(nc["status"], "fail")
        self.assertFalse(nc["controls"][0]["caught"])
        self.assertTrue(nc["controls"][0]["restored"])
        self.assertEqual(rep["gates"]["restore"]["status"], "pass")

    def test_graph_phase_without_bundle_is_harness_incomplete_on_gpu_or_skipped_on_cpu(self):
        with open(self.meta_path) as f:
            meta = json.load(f)
        meta["phase_regimes"]["DECODE"] = {"cuda_graph": True}
        self.write_meta(meta)
        lint = self.run_py([os.path.join(self.task, "ut_contract.py"), self.task])
        self.assertEqual(lint.returncode, 1)
        self.assertIn("replay_sigs", lint.stdout)

    def test_lint_rejects_meta_out_of_sync_with_manifest(self):
        with open(self.meta_path) as f:
            meta = json.load(f)
        meta["workload"]["cases"][0]["effective_calls"] = 1
        self.write_meta(meta)
        lint = self.run_py([os.path.join(self.task, "ut_contract.py"), self.task])
        self.assertEqual(lint.returncode, 1)
        self.assertIn("sync-meta", lint.stdout)
        self.run_py([os.path.join(UTC, "scaffold_ut.py"), "sync-meta", "--task", self.task])
        self.assertEqual(self.run_py([os.path.join(self.task, "ut_contract.py"), self.task]).returncode, 0)


TOY_RMS_QUANT = textwrap.dedent('''\
    import torch


    def rms_quant(x, weight, eps):
        """toy installed rmsnorm + per-token fp8 quant"""
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
        scale = y.abs().amax(-1, keepdim=True) / 448.0
        return (y / scale).to(torch.float8_e4m3fn), scale
    ''')


@needs_torch
class EndToEndNoiseRelativeTest(EndToEndTest):
    """The quantized-output path: noise_relative needs a second baseline run, bitwise
    determinism needs a second candidate run, and the controls compare against both."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="utc_e2e_q_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        site = os.path.join(self.tmp, "site", "toyk")
        os.makedirs(site)
        open(os.path.join(site, "__init__.py"), "w").close()
        with open(os.path.join(site, "norm.py"), "w") as f:
            f.write(TOY_RMS_QUANT)
        self.env = dict(os.environ, PYTHONPATH=os.path.join(self.tmp, "site"), HARNESS_CACHE_FLUSH_MB="1")
        self.task = os.path.join(self.tmp, "eval", "tasks", "toy_rmsq")
        manifest = {"cases": [{"sig": "DECODE_r8", "regime": "DECODE", "axes": {"rows": 8},
                               "effective_calls": 50, "timed": True, "random": True}], "replay_sigs": []}
        with open(os.path.join(self.tmp, "m.json"), "w") as f:
            json.dump(manifest, f)
        with open(os.path.join(self.tmp, "r.json"), "w") as f:
            json.dump({"DECODE": {"cuda_graph": False}}, f)
        self.run_py([os.path.join(UTC, "scaffold_ut.py"), "new", "--family", "rmsnorm_quant", "--task", self.task,
                     "--target", "toyk.norm:rms_quant", "--bind-module", "toyk.norm", "--bind-file", "kernel_src/norm.py",
                     "--attr", "quant=fp8", "--attr", "has_residual=false", "--attr", "fp8_dtype=float8_e4m3fn",
                     "--axis", "hidden=256", "--axis", "group=256",
                     "--manifest", os.path.join(self.tmp, "m.json"), "--phase-regimes", os.path.join(self.tmp, "r.json")])
        with open(os.path.join(self.task, "kernel_src", "norm.py"), "w") as f:
            f.write(TOY_RMS_QUANT)
        cases = os.path.join(self.task, "cases.py")
        with open(cases) as f:
            text = f.read()
        old = text[text.index('    raise NotImplementedError("FILL call()'):text.index("# ------", text.index('    raise NotImplementedError("FILL call()'))]
        text = text.replace(old, '    return _entry()(args["x"], args["weight"], args["eps"])\n\n\n')
        with open(cases, "w") as f:
            f.write(text)
        self.meta_path = os.path.join(self.task, "meta.json")
        with open(self.meta_path) as f:
            meta = json.load(f)
        meta["workload"]["serving_weight_model"]["evidence_path"] = "run/observed_workload.json"
        meta["ut_contract"]["negative_controls"] = [{
            "leaf": "kernel_src/norm.py", "find": "/ 448.0", "replace": "/ 400.0",
            "expect": "every per-token scale off by 12%: out_scale fails its 1e-6 rtol"}]
        self.write_meta(meta)

    def test_unmodified_extraction_passes_every_gate(self):
        p, rep = self.ut("--max-reps", "1")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr + json.dumps(rep, indent=1, default=str)[:4000])
        corr = rep["gates"]["correctness"]
        self.assertTrue(corr["bitwise_repeat"]["ok"])
        out_q = corr["cases"][0]["out_q"]
        self.assertEqual(out_q["kind"], "noise_relative")
        self.assertEqual(out_q["baseline_run_to_run_max"], 0.0)
        ctrl = rep["gates"]["negative_controls"]["controls"][0]
        self.assertTrue(ctrl["caught"] and ctrl["margin"] >= 10, ctrl)

    test_wrong_candidate_fails_correctness_and_stays_failed = None
    test_a_control_that_changes_nothing_fails_the_gate = None
    test_graph_phase_without_bundle_is_harness_incomplete_on_gpu_or_skipped_on_cpu = None
    test_lint_rejects_meta_out_of_sync_with_manifest = None


if __name__ == "__main__":
    unittest.main()
