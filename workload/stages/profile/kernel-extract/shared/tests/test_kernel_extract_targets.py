"""Behavioral checks for the extraction scope, independent of optimizer budgets."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/select_targets.py"
spec = importlib.util.spec_from_file_location("extract_targets", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
select_targets = module.select_targets


def row(name, overall, prefill=0, decode=0):
    return {"name": name, "selection_pct_gpu_time": overall,
            "share_of_phase_pct": {"PREFILL": prefill, "DECODE": decode}}


class ExtractionTargetsTest(unittest.TestCase):
    def test_phase_top_five_then_strict_threshold_without_backfill(self):
        result = select_targets({"ranking": [
            row("p1", 8, prefill=30), row("p2", 3, prefill=20),
            row("p3", 2, prefill=15), row("p4", 6, prefill=12),
            row("shared", 7, prefill=10, decode=30),
            row("p6", 9, prefill=5),
            row("d2", 6, decode=20), row("d3", 5, decode=15),
            row("d4", 4, decode=10), row("d5", 3.01, decode=8),
            row("d6", 10, decode=7),
        ]})
        targets = {r["id"]: r for r in result["targets"]}
        self.assertEqual(set(targets), {"p1", "p4", "shared", "d2", "d3", "d4", "d5"})
        self.assertEqual(targets["shared"]["selected_phases"], ["PREFILL", "DECODE"])
        self.assertEqual(targets["shared"]["phase_ranks"], {"PREFILL": 5, "DECODE": 1})
        self.assertEqual(result["target_count"], 7)
        self.assertFalse(result["per_phase"]["PREFILL"]["top_k"][1]["selected"])

    def test_less_than_five_and_no_qualifying_kernels_are_valid(self):
        result = select_targets({"ranking": [row("tiny", 3, prefill=100)]})
        self.assertEqual(result["target_count"], 0)
        self.assertEqual(len(result["per_phase"]["PREFILL"]["top_k"]), 1)
        self.assertEqual(result["per_phase"]["DECODE"]["top_k"], [])

    def test_do_not_hide_selected_collectives_or_optimization_drops(self):
        collective = {**row("collective", 8, decode=30), "editable": False}
        low_headroom = {**row("compute", 4, decode=20), "expected_e2e_gain": 0}
        result = select_targets({"ranking": [collective, low_headroom], "budget": 1})
        self.assertEqual({r["id"] for r in result["targets"]}, {"collective", "compute"})

    def test_ties_use_identity_and_ignore_old_global_rank(self):
        rows = [{**row(name, 4, decode=10), "rank": 99-i}
                for i, name in enumerate(["f", "e", "d", "c", "b", "a"])]
        self.assertEqual([r["id"] for r in select_targets({"ranking": rows})["targets"]],
                         ["a", "b", "c", "d", "e"])

    def test_invalid_or_missing_costs_and_duplicate_identity_fail(self):
        valid = row("a", 4, decode=10)
        for bad in [None, "4", True, float("nan"), float("inf"), -1, 101]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                select_targets({"ranking": [{**valid, "selection_pct_gpu_time": bad}]})
        for invalid in [[], [valid, valid], [{**valid, "share_of_phase_pct": {}}],
                        [{**valid, "share_of_phase_pct": {"DECODE": float("nan")}}]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                select_targets({"ranking": invalid})


def krow(name, overall, prefill=0, decode=0, kind="compute"):
    return {"id": name, "name": name, "kind": kind, "selection_pct_gpu_time": overall,
            "share_of_phase_pct": {"PREFILL": prefill, "DECODE": decode}}


def smap(**ops):
    """kernel -> (entry, module class) as source_map.py writes them."""
    return {"kernels": {k: {"op": {"key": "%s@%s" % v, "module_class": v[1],
                                   "entry": {"target_callable": v[0]}}} for k, v in ops.items()}}


class OperatorSelectionTest(unittest.TestCase):
    """The GLM-5.2 TP8 run of 2026-10-23, shares rounded: by kernel name six kernels
    cleared 3% and the fused MoE's stage-2 and sort kernels, the sparse MLA reduce and the
    indexer's top-k fell out of the operators they belong to."""
    RANK = {"ranking": [
        krow("gemm1", 9.85, decode=12.5), krow("split_k", 4.71, decode=6.0),
        krow("sparse_fused", 3.65, prefill=17.4), krow("logits_hist", 3.18, decode=4.0),
        krow("mqa_logits", 2.45, prefill=11.6), krow("gemm2_a", 2.31, decode=2.9),
        krow("k_scatter_topk", 2.15, decode=2.7), krow("gemm2_b", 2.12, decode=2.7),
        krow("grouped_topk", 2.12, decode=2.6), krow("sparse_reduce", 2.09, decode=2.6),
        krow("moe_sort", 1.84, decode=2.3), krow("hgemm_bf16_wk", 1.0, prefill=1, decode=1),
        krow("Memcpy DtoD", 4.0, prefill=4, decode=4, kind="memory"),
        krow("unresolved_k", 0.5, decode=0.6)]}
    SM = smap(gemm1=("moe:fused_moe_wrapper", "FusedMoE"), gemm2_a=("moe:fused_moe_wrapper", "FusedMoE"),
              gemm2_b=("moe:fused_moe_wrapper", "FusedMoE"), moe_sort=("moe:fused_moe_wrapper", "FusedMoE"),
              split_k=("mla:fwd", "RadixAttention"), sparse_fused=("mla:fwd", "RadixAttention"),
              sparse_reduce=("mla:fwd", "RadixAttention"),
              logits_hist=("idx:run", "Indexer"), k_scatter_topk=("idx:run", "Indexer"),
              mqa_logits=("idx:prefill_score", "Indexer"), hgemm_bf16_wk=("gemm:mm", "Indexer"),
              grouped_topk=("router:topk", "Gate"))

    def test_operators_carry_their_members_and_other_phase_branch(self):
        res = module.select_ops(self.RANK, self.SM)
        ops = {o["id"]: o for o in res["ops"]}
        # a device copy stays a selected row of its own (the draft excludes it with a reason)
        self.assertEqual(set(ops), {"moe:fused_moe_wrapper@FusedMoE", "mla:fwd@RadixAttention",
                                    "idx:run@Indexer", "idx:prefill_score@Indexer",
                                    "kernel:Memcpy DtoD"})
        self.assertEqual(sorted(ops["moe:fused_moe_wrapper@FusedMoE"]["members"]),
                         ["gemm1", "gemm2_a", "gemm2_b", "moe_sort"])
        self.assertAlmostEqual(ops["moe:fused_moe_wrapper@FusedMoE"]["overall_pct"], 16.12)
        # the prefill score has 2.45% and is selected only as the decode op's other branch
        self.assertEqual(ops["idx:prefill_score@Indexer"]["phase_branch_of"], "idx:run@Indexer")
        targets = {t["id"]: t for t in res["targets"]}
        self.assertEqual(targets["moe_sort"]["selected_via"], "op")
        # same module, other family, both phases: never pulled in as a branch
        self.assertNotIn("hgemm_bf16_wk", targets)
        self.assertNotIn("grouped_topk", targets)

    def test_kernel_without_stack_is_its_own_operator(self):
        rank = {"ranking": [krow("lonely", 5, decode=50), krow("other", 1, decode=10)]}
        res = module.select_ops(rank, {"kernels": {}})
        self.assertEqual([o["id"] for o in res["ops"]], ["kernel:lonely"])

    def test_candidates_cover_phase_leaders_and_the_tail_floor(self):
        res = module.candidates(self.RANK, top=2, min_pct=2.2)
        ids = {t["id"] for t in res["targets"]}
        self.assertIn("mqa_logits", ids)            # prefill rank 2
        self.assertIn("gemm2_a", ids)               # above the floor
        self.assertNotIn("moe_sort", ids)
        self.assertNotIn("Memcpy DtoD", ids)


SM_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/source_map.py"
_sm_spec = importlib.util.spec_from_file_location("source_map_t", SM_SCRIPT)
source_map = importlib.util.module_from_spec(_sm_spec)
_sm_spec.loader.exec_module(source_map)
PK = ("sglang", "aiter")


class OperatorKeyTest(unittest.TestCase):
    def test_library_entry_below_the_model_tree(self):
        moe = ["<built-in method fused_moe_ of pybind11 object>", "torch/_ops.py(1244): __call__",
               "torch_guard.py(292): wrapper_custom", "aiter/fused_moe.py(671): fused_moe",
               "sglang/kernels/ops/moe/moe_sorting_small.py(437): fused_moe_wrapper",
               "sglang/srt/layers/moe/moe_runner/aiter.py(256): run",
               "torch/nn/modules/module.py(1779): _call_impl", "nn.Module: FusedMoE_1",
               "nn.Module: DeepseekV2MoE_1"]
        e, cls = source_map.op_of_sample(moe, PK)
        self.assertEqual(e[2], "fused_moe_wrapper")
        self.assertEqual(cls, "FusedMoE")

    def test_jit_wrappers_are_not_an_entry(self):
        ar = ["aiter/jit/core.py(1925): wrapper", "aiter/jit/core.py(2216): custom_wrapper",
              "aiter/dist/device_communicators/custom_all_reduce.py(1564): custom_fused_ar_rms",
              "sglang/srt/distributed/parallel_state.py(814): fused_allreduce_rmsnorm",
              "nn.Module: DeepseekV2DecoderLayer_1"]
        self.assertEqual(source_map.op_of_sample(ar, PK)[0][2], "custom_fused_ar_rms")
        only_jit = ["aiter/jit/core.py(1925): wrapper",
                    "sglang/srt/layers/layernorm.py(201): forward"]
        self.assertEqual(source_map.op_of_sample(only_jit, PK)[0][2], "forward")

    def test_kernel_launched_by_model_code_is_its_own_entry(self):
        st = ["triton/runtime/jit.py(720): run",
              "sglang/srt/layers/moe/ep_moe/layer.py(88): _append_shared",
              "nn.Module: FusedMoE_3"]
        e, cls = source_map.op_of_sample(st, PK)
        self.assertEqual((e[2], cls), ("_append_shared", "FusedMoE"))


if __name__ == "__main__":
    unittest.main()
