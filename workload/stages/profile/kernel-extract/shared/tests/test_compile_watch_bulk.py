#!/usr/bin/env python3
"""Recorded server calls keep structure, not bulk payload.

A MoE entry's packed MXFP4 expert weights (uint8) were stored with every recorded call
(0.86 GB a call, 185 GB for one task). Bulk payloads -- float operands and large integer
ones -- are now a descriptor plus a small sample; routing ids, index tables and small
integer tensors keep their contents.

Run:  python3 tests/test_compile_watch_bulk.py   (needs torch; CPU is enough)
"""
import importlib.util
import io
import os
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

os.environ["COMPILE_WATCH_STRUCT_CAP"] = "4096"      # small, so the test tensors stay small
os.environ["COMPILE_WATCH_SAMPLE"] = "2048"
spec = importlib.util.spec_from_file_location(
    "compile_watch_bulk", Path(__file__).resolve().parents[1] / "compile_watch.py")
cw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cw)


def _record(args):
    tensors, budget = [], [cw.EVENT_DATA_CAP]
    desc = {n: cw._describe(v, tensors, budget, False, name=n) for n, v in args.items()}
    return {"args": desc, "positional": list(args)}, tensors


def _saved_bytes(tensors):
    buf = io.BytesIO()
    torch.save(tensors, buf)
    return buf.tell()


@unittest.skipIf(torch is None, "torch not installed")
class BulkPayloads(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.ids = torch.randint(0, 32, (64, 8), generator=g, dtype=torch.int32)
        self.x = (torch.randn(256, 64, generator=g) * 0.5 + 3.0).to(torch.bfloat16)
        # packed fp4 pairs: any byte; e8m0 scales: a narrow exponent band, never 255 (NaN)
        self.w13 = torch.randint(0, 256, (4, 64, 128), generator=g, dtype=torch.uint8)
        self.s13 = torch.randint(118, 128, (4, 64, 32), generator=g, dtype=torch.uint8)
        self.req_to_token = torch.zeros(64, 256, dtype=torch.int32)
        self.req_to_token[:3, :40] = torch.arange(120, dtype=torch.int32).view(3, 40)
        self.call, self.tensors = _record({"x": self.x, "ids": self.ids, "w13": self.w13,
                                           "s13": self.s13, "req_to_token": self.req_to_token})

    def test_structure_is_kept_and_bulk_is_not(self):
        a = self.call["args"]
        self.assertIsNotNone(a["ids"]["data"])                 # routing: real contents
        self.assertIsNotNone(a["req_to_token"]["data"])        # index-named: kept over the cap
        for n in ("w13", "s13", "x"):
            self.assertIsNone(a[n]["data"], n)
        self.assertEqual(a["w13"]["gen"]["kind"], "resample")
        self.assertEqual(a["x"]["gen"]["kind"], "normal")
        self.assertAlmostEqual(a["x"]["gen"]["mean"], 3.0, delta=0.1)
        self.assertIn("bulk integer payload", a["w13"]["why_no_data"])
        # what is stored does not grow with the weights: 32x the expert weights, same bytes
        big = torch.randint(0, 256, (4, 64, 128 * 32), dtype=torch.uint8)
        _, t_big = _record({"x": self.x, "ids": self.ids, "w13": big, "s13": self.s13,
                            "req_to_token": self.req_to_token})
        self.assertEqual(_saved_bytes(t_big), _saved_bytes(self.tensors))
        self.assertLess(_saved_bytes(t_big), big.numel() // 10)

    def test_replay_keeps_structure_and_value_distribution(self):
        args, kwargs, built, notes = cw.materialize_call(self.call, self.tensors, seed=7)
        self.assertTrue(torch.equal(built["ids"], self.ids))
        self.assertTrue(torch.equal(built["req_to_token"], self.req_to_token))
        for n, ref in (("w13", self.w13), ("s13", self.s13), ("x", self.x)):
            self.assertEqual(tuple(built[n].shape), tuple(ref.shape), n)
            self.assertEqual(built[n].dtype, ref.dtype, n)
            self.assertEqual(built[n].stride(), ref.stride(), n)
        s = built["s13"]
        self.assertGreaterEqual(int(s.min()), 118)               # never outside the recorded band
        self.assertLessEqual(int(s.max()), 127)
        self.assertGreater(int(torch.unique(built["w13"]).numel()), 200)
        self.assertAlmostEqual(float(built["x"].float().mean()), 3.0, delta=0.1)
        self.assertAlmostEqual(float(built["x"].float().std()), 0.5, delta=0.1)
        self.assertTrue(any("resampled" in n for n in notes))

    def test_replay_is_deterministic_per_seed(self):
        a = cw.materialize_call(self.call, self.tensors, seed=3)[2]
        b = cw.materialize_call(self.call, self.tensors, seed=3)[2]
        c = cw.materialize_call(self.call, self.tensors, seed=4)[2]
        self.assertTrue(torch.equal(a["w13"], b["w13"]))
        self.assertFalse(torch.equal(a["w13"], c["w13"]))

    def test_strided_view_samples_its_own_span(self):
        base = torch.full((8, 4096), 200, dtype=torch.uint8)
        base[2:6] = torch.randint(0, 16, (4, 4096), dtype=torch.uint8)
        call, tensors = _record({"w": base[2:6]})
        g = call["args"]["w"]["gen"]
        self.assertLessEqual(g["max"], 15)

    def test_sample_indices_stay_in_range_past_float32_precision(self):
        # A float32 linspace over a > 2**24-element storage rounds its last positions past
        # hi - 1, and the out-of-range index_select faulted every rank of the Muse Spark probe
        # server on the vocab embedding (2026-11-02). Indices must stay inside the span.
        n = (1 << 24) + 12346          # hi - 1 odd: float32 rounds it up to hi
        x = torch.arange(n, dtype=torch.int64) % 251
        smp = cw._sample(x.to(torch.uint8), 65536)
        self.assertEqual(int(smp.numel()), 65536)
        flat = torch.arange(n, dtype=torch.int64)
        pos = cw._sample(flat, 65536)
        self.assertLessEqual(int(pos.max()), n - 1)
        self.assertEqual(int(pos.max()), n - 1)        # the last element is reachable
        self.assertEqual(int(pos.min()), 0)

    def test_old_records_without_gen_replay_as_before(self):
        call, tensors = _record({"x": self.x})
        del call["args"]["x"]["gen"]
        built = cw.materialize_call(call, tensors, seed=1)[2]
        self.assertAlmostEqual(float(built["x"].float().mean()), 0.0, delta=0.1)
        old_int = {"args": {"w": dict(call["args"]["x"], dtype="torch.uint8")}, "positional": ["w"]}
        with self.assertRaises(cw.Unreplayable):
            cw.materialize_call(old_int, tensors)

    def test_inside_graph_capture_nothing_is_read(self):
        tensors, budget = [], [cw.EVENT_DATA_CAP]
        d = cw._describe(self.w13, tensors, budget, True, name="w13")
        self.assertIsNone(d["data"])
        self.assertNotIn("gen", d)
        self.assertEqual(tensors, [])


@unittest.skipIf(torch is None, "torch not installed")
class InlineCalls(unittest.TestCase):
    """The recorded call travels in JSON; the task directory gets no tensor file."""

    def test_record_collect_slice_and_replay_without_a_pt_file(self):
        import json
        import shutil
        import sys
        import tempfile
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        sys.modules["compile_watch"] = cw          # build_task_plan imports it by name
        import build_task_plan as B
        tmp = tempfile.mkdtemp(prefix="cw_inline_")
        self.addCleanup(shutil.rmtree, tmp, True)
        watch = os.path.join(tmp, "watch")
        os.makedirs(watch)
        ids = torch.randint(0, 32, (64, 8), dtype=torch.int32)
        bt = torch.zeros(64, 512, dtype=torch.int32)
        bt[:2, :10] = torch.arange(20, dtype=torch.int32).view(2, 10)
        w13 = torch.randint(0, 256, (4, 64, 4096), dtype=torch.uint8)
        x = torch.randn(64, 64).to(torch.bfloat16)

        def compact_a4_experts(x, ids, w13, block_table):
            # what the hook sees: called from inside Triton's JIT on a cache miss
            cw._STATE.update(dir=watch, frames=1, installed=True)
            cw._hook(fn=type("F", (), {"module": "k", "name": "_moe_kernel"})(),
                     compile={"signature": {"x": "*bf16"}, "constexprs": {}})
        old = dict(cw._STATE)
        try:
            compact_a4_experts(x, ids, w13, bt)
        finally:
            fh = cw._STATE.get("fh")
            if fh:
                fh.close()
            cw._STATE.clear()
            cw._STATE.update(old)
        self.assertFalse(os.path.exists(os.path.join(watch, "calls")))   # nothing written as .pt
        doc = cw.collect(watch)
        rec = doc["variants"][0]
        self.assertIn("tensors_b64", rec)
        ev = os.path.join(tmp, "eval")
        os.makedirs(os.path.join(ev, "evidence"))
        tdir = os.path.join(ev, "tasks", "moe")
        os.makedirs(os.path.join(tdir, "run"))
        with open(os.path.join(ev, "evidence", "compile_variants.json"), "w") as f:
            json.dump(doc, f)
        frame = rec["frames"][0]["callable"]
        se = B.server_evidence(ev, {"kernels": ["_moe_kernel"], "target_callable": frame}, tdir)
        self.assertEqual(se["server_evidence"]["calls"], 1)
        self.assertFalse(os.path.exists(os.path.join(tdir, "run", "server_calls")))
        pts = [f for _r, _d, fs in os.walk(tdir) for f in fs if f.endswith(".pt")]
        self.assertEqual(pts, [])
        with open(os.path.join(tdir, "run", "server_calls.json")) as f:
            call = json.load(f)["calls"][0]
        with open(os.path.join(tdir, "run", "server_variants.json")) as f:
            self.assertNotIn("tensors_b64", f.read())
        data = cw.call_data(call)
        args, kwargs, built, notes = cw.materialize_call(call, data, seed=1)
        self.assertTrue(torch.equal(built["ids"], ids))
        self.assertTrue(torch.equal(built["block_table"], bt))
        self.assertEqual(tuple(built["w13"].shape), tuple(w13.shape))
        self.assertLess(len(call["tensors_b64"]), w13.numel() // 4)


if __name__ == "__main__":
    unittest.main()
