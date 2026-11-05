"""CPU tests for compile_watch.py: identity, collection, task slicing and the coverage diff.

The hook itself, call replay and the UT gates need Triton and a GPU; they are covered by
ut_contracts/tests/test_server_evidence.py.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.dirname(HERE)
sys.path.insert(0, SHARED)
sys.path.insert(0, os.path.join(os.path.dirname(SHARED), "scripts"))
import compile_watch as cw  # noqa: E402


class FakeFn:
    module, name = "pkg.attn", "_attn_kernel"


def compile_info(m="i32", m_const=None, warps=4, div=True):
    sig = {"q_ptr": "*bf16", "M": m, "BLOCK": "constexpr"}
    consts = {(2,): 64}
    if m_const is not None:
        consts[(1,)] = m_const
    attrs = {(0,): [["tt.divisibility", 16]]}
    if div and m == "i32":
        attrs[(1,)] = [["tt.divisibility", 16]]
    return {"signature": sig, "constants": consts, "configs": [attrs], "num_warps": warps,
            "num_stages": 2}


def event(info, pid, t, stack=("pkg.attn:attention", "pkg.model:Layer.forward"), frames=None,
          capture=False):
    ident = cw.variant_identity("pkg.attn._attn_kernel", info)
    ev = dict(ident, id=cw.variant_id(ident), name="_attn_kernel", pid=pid, t=t,
              in_graph_capture=capture, stack=list(stack))
    if frames:
        ev["frames"] = frames
    return ev


class IdentityTest(unittest.TestCase):

    def test_constexpr_one_is_a_different_variant_with_named_fields(self):
        a = cw.variant_identity("f", compile_info())
        b = cw.variant_identity("f", compile_info(m="constexpr", m_const=1))
        self.assertNotEqual(cw.variant_id(a), cw.variant_id(b))
        self.assertEqual(b["constexprs"], {"BLOCK": 64, "M": 1})
        self.assertEqual(a["attrs"]["M"], [["tt.divisibility", 16]])
        diff = {d["field"]: (d["server"], d["ut"]) for d in cw.diff_fields(b, a)}
        self.assertEqual(diff["signature[M]"], ("constexpr", "i32"))
        self.assertEqual(diff["constexprs[M]"], (1, None))
        self.assertEqual(diff["attrs[M]"], (None, [["tt.divisibility", 16]]))

    def test_identity_ignores_time_process_and_stack(self):
        e1 = event(compile_info(), 1, 10.0)
        e2 = event(compile_info(), 2, 99.0, stack=("other:fn",))
        self.assertEqual(e1["id"], e2["id"])
        self.assertNotEqual(e1["id"], event(compile_info(warps=8), 1, 10.0)["id"])


class FrameRecordingTest(unittest.TestCase):

    @staticmethod
    def frame(module, path, name="invoke"):
        scope = {"__name__": module, "sys": sys}
        exec(compile("def %s(q, max_seqlen_q, window_size):\n"
                     "    return sys._getframe()\n" % name, path, "exec"), scope)
        return scope[name](7, 151, (2046, 0))

    def test_records_aiter_triton_entry_before_backend_with_limited_frames(self):
        """AITER's /triton/ directory contains the native caller we must replay."""
        native = self.frame("aiter.ops.triton.attention.unified_attention",
                            "/sgl-workspace/aiter/aiter/ops/triton/attention/unified_attention.py",
                            "unified_attention")
        frames = [self.frame("triton", "<generated>"),
                  self.frame("triton.runtime.jit", "/runtime/jit.py"), native,
                  self.frame("sglang.srt.layers.attention.aiter_backend", "/backend.py")]
        rows = cw._record_frames(frames, 1, [], False)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["callable"],
                         "aiter.ops.triton.attention.unified_attention:unified_attention")
        self.assertEqual(rows[0]["positional"], ["q", "max_seqlen_q", "window_size"])
        self.assertEqual(rows[0]["args"]["max_seqlen_q"], 151)
        self.assertEqual(rows[0]["args"]["window_size"],
                         {"__seq__": "tuple", "items": [2046, 0]})

    def test_keeps_namespace_lookalikes_and_filters_torch_plumbing(self):
        wrappers = [
            ("torch._dynamo.eval_frame", "/site/torch/_dynamo/eval_frame.py"),
            ("torch._inductor.runtime", "/site/torch/_inductor/runtime.py"),
            ("torch._library.custom_ops", "/site/torch/_library/custom_ops.py"),
            ("torch._ops", "/site/torch/_ops.py"),
            ("torch.utils._contextlib", "/site/torch/utils/_contextlib.py"),
            ("torch.nn.modules.module", "/site/torch/nn/modules/module.py"),
            ("compile_watch", "/overlay/compile_watch.py"),
        ]
        frames = [self.frame(module, path) for module, path in wrappers]
        frames += [self.frame("triton_helpers.attention", "/project/triton/attention.py"),
                   self.frame("pkg.triton.attention", "/project/attention.py")]
        rows = cw._record_frames(frames, 2, [], False)
        self.assertEqual([r["callable"] for r in rows],
                         ["triton_helpers.attention:invoke", "pkg.triton.attention:invoke"])


class CollectTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="cw_")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def write(self, pid, events):
        with open(os.path.join(self.dir, "variants.pid%d.jsonl" % pid), "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

    def test_merges_ranks_keeps_frames_and_tags_startup_and_capture(self):
        fr = [{"callable": "pkg.attn:attention", "positional": ["q"], "args": {}}]
        warm = compile_info(m="constexpr", m_const=1)
        self.write(1, [dict(event(warm, 1, 100.0), env={"triton.next_power_of_2": "function:x"}),
                       event(compile_info(), 1, 300.0, capture=True)])
        self.write(2, [event(warm, 2, 101.0, frames=fr), event(compile_info(), 2, 301.0)])
        doc = cw.collect(self.dir, ready_epoch=200.0, server_mode="profile_cuda_graph")
        self.assertEqual(doc["processes"], 2)
        self.assertEqual(doc["env"], {"triton.next_power_of_2": "function:x"})
        by_m = {r["signature"]["M"]: r for r in doc["variants"]}
        self.assertTrue(by_m["constexpr"]["before_ready"])
        self.assertEqual(by_m["constexpr"]["frames"], fr)
        self.assertEqual(by_m["constexpr"]["processes"], 2)
        self.assertFalse(by_m["i32"]["before_ready"])
        self.assertTrue(by_m["i32"]["in_graph_capture"] and by_m["i32"]["eager"])

    def test_belongs_by_kernel_name_or_entry_on_the_stack(self):
        rec = event(compile_info(), 1, 1.0)
        self.assertTrue(cw.belongs(rec, {"_attn_kernel"}))
        self.assertTrue(cw.belongs(dict(rec, name="_helper"), set(), "pkg.attn:attention"))
        self.assertFalse(cw.belongs(dict(rec, name="_helper"), set(), "pkg.attn:other"))
        self.assertTrue(cw.belongs(dict(rec, name="_h"), set(), "pkg.model:Layer.forward"))


class CoverageTest(unittest.TestCase):

    def test_multiconfig_ids_share_semantics_but_keep_record_accounting(self):
        raw = event(compile_info(), 9, 5.0)
        server = [dict(raw, id=cid + "__" + raw["id"], config_id=cid)
                  for cid in ("conc10", "conc40")]
        before = json.dumps(server, sort_keys=True)
        cov = cw.coverage(server, [raw])
        self.assertEqual((cov["server_variants"], cov["covered"]), (2, 2))
        self.assertEqual(cov["missing"], [])
        self.assertEqual(json.dumps(server, sort_keys=True), before)
        # UT IDs also need not be raw hashes; metadata does not select compilation.
        cov = cw.coverage([raw], [dict(raw, id="ut__" + raw["id"], pid=123)])
        self.assertEqual(cov["covered"], 1)

    def test_matching_stored_id_cannot_hide_any_changed_identity_field(self):
        server = event(compile_info(), 1, 1.0)
        changes = {
            "fn": "other.module._attn_kernel",
            "signature": dict(server["signature"], q_ptr="*fp16"),
            "constexprs": dict(server["constexprs"], BLOCK=128),
            "attrs": dict(server["attrs"], M=[["tt.divisibility", 8]]),
            "num_warps": 8,
            "num_stages": 3,
        }
        for field, value in changes.items():
            for prefix in ("", "conc10__"):
                with self.subTest(field=field, prefix=prefix):
                    rec = dict(server, id=prefix + server["id"])
                    ut = dict(rec, **{field: value})
                    cov = cw.coverage([rec], [ut])
                    self.assertEqual(cov["covered"], 0)
                    self.assertEqual(cov["missing"][0]["id"], rec["id"])
                    self.assertTrue(any(d["field"] == field or
                                        d["field"].startswith(field + "[")
                                        for d in cov["missing"][0]["differs"]))

    def test_same_short_name_in_another_module_does_not_match(self):
        server = event(compile_info(), 1, 1.0)
        ut = dict(server, fn="other.module._attn_kernel")
        ut["id"] = cw.variant_id(ut)
        cov = cw.coverage([server], [ut])
        self.assertEqual(cov["covered"], 0)
        self.assertEqual(cov["missing"][0]["differs"], [
            {"field": "fn", "server": server["fn"], "ut": ut["fn"]}])

    def test_incompatible_ut_events_with_one_stored_id_are_not_deduplicated(self):
        a = event(compile_info(), 1, 1.0)
        b = event(compile_info(warps=8), 1, 2.0)
        cov = cw.coverage([a, b], [a, dict(b, id=a["id"])])
        self.assertEqual((cov["covered"], cov["missing"]), (2, []))

    def test_waivers_remain_config_specific(self):
        raw = event(compile_info(), 1, 1.0)
        server = [dict(raw, id=cid + "__" + raw["id"]) for cid in ("conc10", "conc40")]
        ut = event(compile_info(warps=8), 9, 5.0)
        cov = cw.coverage(server, [ut], [{"id": server[0]["id"], "why": "capture only"}])
        self.assertEqual(cov["covered"], 0)
        self.assertEqual([r["id"] for r in cov["waived"]], [server[0]["id"]])
        self.assertEqual([r["id"] for r in cov["missing"]], [server[1]["id"]])
        self.assertEqual(cw.coverage(server, [ut], [
            {"id": raw["id"], "why": "cannot waive all configs by raw hash"}])["waived"], [])

    def test_missing_variant_names_the_nearest_ut_variant_and_waivers_need_a_reason(self):
        server = [dict(event(compile_info(), 1, 1.0), eager=True),
                  dict(event(compile_info(m="constexpr", m_const=1), 1, 2.0), eager=True)]
        ut = [event(compile_info(), 9, 5.0)]
        cov = cw.coverage(server, ut)
        self.assertEqual(cov["covered"], 1)
        self.assertEqual(len(cov["missing"]), 1)
        fields = {d["field"] for d in cov["missing"][0]["differs"]}
        self.assertEqual(fields, {"signature[M]", "constexprs[M]", "attrs[M]"})
        vid = cov["missing"][0]["id"]
        self.assertEqual(len(cw.coverage(server, ut, [{"id": vid, "why": ""}])["missing"]), 1)
        cov = cw.coverage(server, ut, [{"id": vid, "why": "graph-only bs, replay covers it"}])
        self.assertEqual((len(cov["missing"]), len(cov["waived"])), (0, 1))
        self.assertIn("never compiled", cw.coverage(server, [])["missing"][0]["differs"])

    def test_env_diff_reports_rebinding_only(self):
        server = {"triton.next_power_of_2": "function:sglang.srt.utils.common", "triton.cdiv": "x"}
        ut = {"triton.next_power_of_2": "ConstexprFunction:triton.runtime.jit", "triton.cdiv": "x"}
        self.assertEqual([d["attr"] for d in cw.env_diff(server, ut)], ["triton.next_power_of_2"])
        self.assertEqual(cw.env_diff(server, {}), [])


class TaskSliceTest(unittest.TestCase):
    """build_task_plan.server_evidence on a synthetic compile_variants.json."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cw_task_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.eval = os.path.join(self.tmp, "eval")
        os.makedirs(os.path.join(self.eval, "evidence"))
        os.makedirs(os.path.join(self.eval, "tasks", "attn", "run"))
        fr = [{"callable": "sglang.attn:attention", "file": "a.py", "line": 3,
               "positional": ["q", "max_seqlen_q"],
               "args": {"q": {"__tensor__": 1, "shape": [1, 8], "data": None},
                        "max_seqlen_q": 1}}]
        recs = [dict(event(compile_info(m="constexpr", m_const=1), 1, 1.0,
                           stack=("sglang.attn:attention",), frames=fr), eager=True),
                dict(event(compile_info(), 1, 2.0, stack=("sglang.attn:attention",),
                           frames=[dict(fr[0], callable="sglang.other:x")]), eager=True),
                dict(event(compile_info(), 1, 3.0), name="_unrelated", eager=True,
                     stack=["sglang.moe:fused"])]
        with open(os.path.join(self.eval, "evidence", "compile_variants.json"), "w") as f:
            json.dump({"variants": recs, "watch_dir": self.tmp, "env": {"a": "b"},
                       "server_mode": "profile_cuda_graph"}, f)

    def test_slices_calls_and_defaults_preimports(self):
        import build_task_plan as B
        tdir = os.path.join(self.eval, "tasks", "attn")
        se = B.server_evidence(self.eval, {"kernels": ["_attn_kernel"],
                                           "target_callable": "sglang.attn:attention"}, tdir)
        self.assertEqual(se["server_preimports"], ["sglang.srt.utils.common"])
        self.assertEqual(se["server_evidence"]["variants"], 2)
        self.assertEqual(se["server_evidence"]["calls"], 1)
        self.assertEqual(se["server_evidence"]["variants_without_entry_frame"], 1)
        with open(os.path.join(tdir, "run", "server_calls.json")) as f:
            calls = json.load(f)["calls"]
        self.assertEqual(calls[0]["args"]["max_seqlen_q"], 1)
        self.assertTrue(os.path.exists(os.path.join(tdir, "compile_watch.py")))
        with open(os.path.join(tdir, "meta.json"), "w") as f:
            json.dump({"ut_contract": {}}, f)
        B.stamp_meta(tdir, se, [{"id": "x", "why": "y"}])
        with open(os.path.join(tdir, "meta.json")) as f:
            meta = json.load(f)
        self.assertEqual(meta["server_evidence"]["server_calls"], "run/server_calls.json")
        self.assertEqual(meta["ut_contract"]["compile_variant_waivers"], [{"id": "x", "why": "y"}])

    def test_large_recorded_call_is_referenced_not_copied(self):
        import build_task_plan as B
        path = os.path.join(self.eval, "evidence", "compile_variants.json")
        with open(path) as f:
            doc = json.load(f)
        os.makedirs(os.path.join(self.tmp, "calls"))
        for size, name in ((4096, "big.pt"), (16, "small.pt")):
            with open(os.path.join(self.tmp, "calls", name), "wb") as f:
                f.write(b"\0" * size)
        doc["variants"][0]["tensors"] = "calls/big.pt"
        with open(path, "w") as f:
            json.dump(doc, f)
        tdir = os.path.join(self.eval, "tasks", "attn")
        old, B.SERVER_CALL_COPY_MAX = B.SERVER_CALL_COPY_MAX, 1024
        try:
            se = B.server_evidence(self.eval, {"kernels": ["_attn_kernel"],
                                               "target_callable": "sglang.attn:attention"}, tdir)
        finally:
            B.SERVER_CALL_COPY_MAX = old
        with open(os.path.join(tdir, "run", "server_calls.json")) as f:
            call = json.load(f)["calls"][0]
        self.assertEqual(call["tensors"], os.path.join(self.tmp, "calls", "big.pt"))
        self.assertFalse(os.path.exists(os.path.join(tdir, "run", "server_calls")))
        self.assertEqual(se["server_evidence"]["calls_referenced_in_place"], 1)
        # leg_runner resolves it with os.path.join(task, tensors): absolute wins
        self.assertEqual(os.path.join(tdir, call["tensors"]), call["tensors"])
        doc["variants"][0]["tensors"] = "calls/small.pt"
        with open(path, "w") as f:
            json.dump(doc, f)
        B.server_evidence(self.eval, {"kernels": ["_attn_kernel"],
                                      "target_callable": "sglang.attn:attention"}, tdir)
        with open(os.path.join(tdir, "run", "server_calls.json")) as f:
            call = json.load(f)["calls"][0]
        self.assertEqual(call["tensors"], os.path.join("run", "server_calls", call["id"] + ".pt"))

    def test_explicit_preimports_win_and_no_evidence_is_reported(self):
        import build_task_plan as B
        tdir = os.path.join(self.eval, "tasks", "attn")
        os.remove(os.path.join(self.eval, "evidence", "compile_variants.json"))
        se = B.server_evidence(self.eval, {"kernels": [], "target_callable": "sglang.a:b",
                                           "server_preimports": ["x.y"]}, tdir)
        self.assertEqual(se["server_preimports"], ["x.y"])
        self.assertIsNone(se["server_evidence"])
        self.assertIn("compile_variants.json", se["why"])


if __name__ == "__main__":
    unittest.main()
