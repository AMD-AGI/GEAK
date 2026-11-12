"""merge_policy.py: which tasks of a multi-config group are one UT (offline, synthetic evidence)."""
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HERE.parent / "shared"))
import merge_policy  # noqa: E402
import compile_watch  # noqa: E402

SRC = textwrap.dedent('''\
    def dispatch(x, kernels):
        n = x.shape[0]
        if "a" in kernels and n <= 8:
            return small_op(x)
        if "b" in kernels and n > 8:
            return big_op(x)
        return None


    def layer_forward(self, x):
        y = pre_op(x)
        z = dispatch(y, frozenset({"a", "b"}))
        if self.sliding:
            w = swa_op(z)
        else:
            w = full_op(z)
        return post_op(w)
    ''')
MOD = "pkg.mod"
DEF = {"dispatch": 1, "layer_forward": 10, "small_op": 30, "big_op": 31, "pre_op": 32,
       "post_op": 33, "swa_op": 34, "full_op": 35}
T = lambda x: {"__tensor__": 1, "shape": [x, 64], "dtype": "torch.bfloat16"}  # noqa: E731
FROZEN = {"__object__": "builtins.frozenset", "repr": "frozenset({'a', 'b'})"}


def frame(fn, path):
    return {"target_callable": "%s:%s" % (MOD, fn), "path": path, "line": DEF[fn]}


class MergePolicyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="merge policy ")
        self.addCleanup(self.tmp.cleanup)
        g = self.g = Path(self.tmp.name)
        src = g / "src" / "mod.py"
        src.parent.mkdir()
        src.write_text(SRC)
        self.src = str(src)
        # kernel -> stack (innermost first)
        under_dispatch = lambda op: [frame(op, self.src), frame("dispatch", self.src),  # noqa: E731
                                     frame("layer_forward", self.src)]
        direct = lambda op: [frame(op, self.src), frame("layer_forward", self.src)]  # noqa: E731
        stacks = {"k_small": under_dispatch("small_op"), "k_big": under_dispatch("big_op"),
                  "k_pre": direct("pre_op"), "k_post": direct("post_op"),
                  "k_swa": direct("swa_op"), "k_full": direct("full_op")}
        tasks = {"small": ("small_op", ["k_small"]), "big": ("big_op", ["k_big"]),
                 "pre": ("pre_op", ["k_pre"]), "post": ("post_op", ["k_post"]),
                 "swa": ("swa_op", ["k_swa"]), "full": ("full_op", ["k_full"])}
        plan = {"tasks": [{"label": n, "target_callable": "%s:%s" % (MOD, fn),
                           "selected_identities": ks, "phases": ["DECODE"],
                           "task_dir": str(g / "tasks" / n),
                           "owned_point_ids": ["c1__de_1", "c2__de_1"]}
                          for n, (fn, ks) in tasks.items()]}
        (g / "task_plan.json").write_text(json.dumps(plan))
        for n in tasks:
            (g / "tasks" / n / "kernel_src").mkdir(parents=True)
        (g / "tasks" / "big" / "kernel_src" / "big.py").write_text("import os\nX = os.environ.get('BIG_TUNE')\n")
        # c1 serves bs 4 (small branch), c2 bs 16 (big branch); every layer type everywhere
        for cfg, bs, branch in (("c1", 4, "k_small"), ("c2", 16, "k_big")):
            ev = g / "configs" / cfg / "evidence"
            (ev / "census").mkdir(parents=True)
            (ev / "source_map.json").write_text(json.dumps(
                {"kernels": {k: {"frames": v} for k, v in stacks.items()}}))
            row = {k: 1.0 for k in ("k_pre", "k_post", "k_swa", "k_full", branch)}
            (ev / "census" / "de_1.json").write_text(json.dumps(
                {"point_id": "de_1", "phase": "DECODE", "achieved_shape": bs, "per_rank": {"0": row}}))
            (ev / "compile_variants.json").write_text(json.dumps({"variants": [
                {"frames": [{"callable": "%s:dispatch" % MOD, "args": {"x": T(bs), "kernels": FROZEN}},
                            {"callable": "%s:layer_forward" % MOD,
                             "args": {"self": {"__object__": "Layer"}, "x": T(bs)}}]}]}))
        self.baseline = g / "baseline.json"
        self.baseline.write_text(json.dumps({"launch_diff": {"ep4": {
            "env:BIG_TUNE": {"c1": None, "c2": "1"},
            "env:UNRELATED": {"c1": None, "c2": "1"},
            "chunked-prefill-size": {"c1": 8192, "c2": 16384}}}}))

    def run_policy(self, *extra):
        out = self.g / "merge_policy.json"
        p = subprocess.run([sys.executable, str(SCRIPTS / "merge_policy.py"), "--group-dir", str(self.g),
                            "--baseline", str(self.baseline), "--out", str(out), *extra],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return json.loads(out.read_text())

    def group(self, d, members):
        (g,) = [g for g in d["branch_groups"] if set(g["members"]) == set(members)]
        return g

    def test_size_dispatched_branches_under_a_replayable_caller_are_one_task(self):
        d = self.run_policy()
        g = self.group(d, ["big", "small"])
        self.assertEqual((g["merged"], g["how"], g["seam"]), (True, "dispatcher", "%s:dispatch" % MOD))
        self.assertEqual(g["call_lines"], {"small": [4], "big": [6]})
        self.assertTrue(g["threshold_movable"])
        self.assertEqual(g["boundaries"]["DECODE"],
                         [{"below": "small", "max_m": 4, "above": "big", "min_m": 16}])
        self.assertEqual(g["expected_kernels"], {"big": ["k_big"], "small": ["k_small"]})
        self.assertEqual({r["m"] for r in g["case_sources"]["big"]}, {16})
        self.assertIn("merged_into", d["per_task"]["big"])

    def test_sequential_operators_and_layer_types_are_never_grouped(self):
        d = self.run_policy()
        grouped = {m for g in d["branch_groups"] for m in g["members"]}
        # pre/post run at every point with either branch; swa/full co-occur at one M
        self.assertFalse(grouped & {"pre", "post", "swa", "full"})

    def test_a_setting_splits_a_task_only_if_its_source_reads_it(self):
        d = self.run_policy()
        self.assertEqual(d["per_task"]["big"]["decision"], "split_by_config_value")
        self.assertEqual(sorted(d["per_task"]["big"]["reads_config_difference"]), ["env:BIG_TUNE"])
        self.assertEqual(d["per_task"]["small"]["decision"], "merge_across_configs")

    def test_a_bound_method_seam_is_merged_on_a_rebuilt_instance(self):
        # Move the branches up into the method: same evidence, but the caller has `self`.
        src = Path(self.src)
        src.write_text(SRC.replace("def dispatch(x, kernels):", "def dispatch(self, kernels):"))
        for cv in self.g.glob("configs/*/evidence/compile_variants.json"):
            d = json.loads(cv.read_text())
            d["variants"][0]["frames"][0]["args"]["self"] = {"__object__": "Backend"}
            cv.write_text(json.dumps(d))
        g = self.group(self.run_policy(), ["big", "small"])
        # an engineering obstacle changes HOW it is merged, never whether
        self.assertEqual((g["merged"], g["how"], g["threshold_movable"]), (True, "rebuilt_instance", True))
        self.assertIn("attribute_manifest", g)

    def test_condition_not_read_as_a_size_is_still_merged_and_says_so(self):
        Path(self.src).write_text(SRC.replace('"a" in kernels and n <= 8', '"a" in kernels')
                                  .replace('"b" in kernels and n > 8', '"b" in kernels'))
        g = self.group(self.run_policy(), ["big", "small"])
        self.assertTrue(g["merged"])
        self.assertEqual(g["conditions_read_size"], {"big": False, "small": False})

    def test_an_unreplayable_argument_merges_by_scope_equalization(self):
        for cv in self.g.glob("configs/*/evidence/compile_variants.json"):
            d = json.loads(cv.read_text())
            d["variants"][0]["frames"][0]["args"]["fb"] = {"__object__": "pkg.ForwardBatch", "repr": "FB"}
            cv.write_text(json.dumps(d))
        g = self.group(self.run_policy(), ["big", "small"])
        self.assertEqual((g["merged"], g["how"], g["threshold_movable"]), (True, "scope_equalized", False))
        self.assertIn("scope_region", g)

    def test_unconditional_calls_are_not_a_dispatch(self):
        # big is called unconditionally: same caller, exclusive in M, but no branch in the source
        Path(self.src).write_text(SRC.replace('    if "b" in kernels and n > 8:\n        return big_op(x)',
                                              '    y = big_op(x)'))
        d = self.run_policy()
        self.assertFalse([g for g in d["branch_groups"] if "big" in g["members"]])
        self.assertTrue(any(set(r["pair"]) == {"big", "small"} and "not under any conditional" in r["why"]
                            for r in d["refused_pairs"]))


class PerBranchTimingTest(unittest.TestCase):
    def test_a_gain_on_one_branch_cannot_hide_a_loss_on_another(self):
        import harness_lib
        pc = [{"sig": "a1", "regime": "decode_0", "m": 8, "baseline_ms": 1.0, "optimized_ms": 0.5, "branch": "small"},
              {"sig": "a2", "regime": "decode_0", "m": 16, "baseline_ms": 1.0, "optimized_ms": 0.6, "branch": "small"},
              {"sig": "b1", "regime": "decode_0", "m": 64, "baseline_ms": 2.0, "optimized_ms": 2.4, "branch": "big"}]
        r = harness_lib.serving_weighted_speedup(pc, {})
        self.assertGreater(r["weighted"], 1.0)
        self.assertEqual(r["worst_branch"]["branch"], "big")
        self.assertLess(r["worst_branch"]["weighted"], 1.0)


class LiteralReplayTest(unittest.TestCase):
    def test_option_sets_replay_and_truncated_reprs_do_not(self):
        self.assertEqual(compile_watch.literal_object(FROZEN), (True, frozenset({"a", "b"})))
        self.assertEqual(merge_policy.literal(FROZEN), (True, frozenset({"a", "b"})))
        cut = {"__object__": "builtins.frozenset", "repr": "frozenset({" + "'x', " * 40}
        self.assertFalse(compile_watch.literal_object(cut)[0])
        self.assertFalse(compile_watch.literal_object({"__object__": "pkg.ForwardBatch",
                                                       "repr": "ForwardBatch(...)"})[0])


if __name__ == "__main__":
    unittest.main()
