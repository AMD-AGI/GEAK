#!/usr/bin/env python3
"""A pool over the replay budget is compacted through the token slots that address it.

A cache writer's recorded call (an FP4 indexer key store: keys [9, 128], the merged MLA
cache [13.4M pages, 64, 68] uint8, slots [9] int64 = page * 64 + offset) carries no
block table, only the slots. The pool shrinks to the pages the slots touch and each slot
keeps its offset inside its page.

Run:  python3 tests/test_compile_watch_slots.py   (needs torch; CPU is enough)
"""
import importlib.util
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

spec = importlib.util.spec_from_file_location(
    "compile_watch_slots", Path(__file__).resolve().parents[1] / "compile_watch.py")
cw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cw)

PAGES = 13446007


def _tensor(shape, dtype, data=None, gen=None, storage=None):
    stride, acc = [], 1
    for n in reversed(shape):
        stride.insert(0, acc)
        acc *= n
    item = {"torch.uint8": 1, "torch.int64": 8, "torch.bfloat16": 2, "torch.int32": 4}[dtype]
    d = {"__tensor__": 1, "shape": list(shape), "stride": stride, "dtype": dtype,
         "device": "cpu", "offset": 0, "nbytes": acc * item,
         "storage_nbytes": storage or acc * item, "data": data}
    if gen:
        d["gen"] = gen
    return d


def _call(slot_name, extra=None):
    args = {"keys": _tensor([9, 128], "torch.bfloat16", data=0),
            "cache": _tensor([PAGES, 64, 68], "torch.uint8",
                             gen={"kind": "resample", "sample": 2},
                             storage=PAGES * 4352 + 20 * 2**20),
            slot_name: _tensor([9], "torch.int64", data=1)}
    args.update(extra or {})
    return {"args": args, "positional": list(args)}


@unittest.skipIf(torch is None, "torch not installed")
class SlotCompaction(unittest.TestCase):
    def setUp(self):
        self.keys = torch.randn(9, 128).to(torch.bfloat16)
        self.slots = torch.tensor([7675200 + i for i in range(8)] + [-1], dtype=torch.int64)
        self.sample = torch.randint(0, 223, (64,), dtype=torch.uint8)

    def test_slots_compact_by_page_and_keep_offsets(self):
        data = [self.keys, self.slots.clone(), self.sample]
        args, _, built, notes = cw.materialize_call(_call("slots"), data, budget=16 << 30)
        cache, slots = args[1], args[2]
        self.assertEqual(list(cache.shape), [1, 64, 68])
        # still a > 2 GiB storage, so the replay keeps the server's no-pointer_range variant
        self.assertGreater(cache.untyped_storage().nbytes(), 2**31 - 1)
        self.assertEqual(slots.tolist(), list(range(8)) + [-1])
        self.assertTrue(any("slots ['slots'] (by page of 64)" in n for n in notes), notes)

    def test_slot_mapping_with_block_table_shares_one_renumbering(self):
        table = torch.tensor([[119925, 3]], dtype=torch.int32)
        data = [self.keys, self.slots.clone(), self.sample, table]
        call = _call("slot_mapping", {"block_table": _tensor([1, 2], "torch.int32", data=3)})
        args, _, built, _ = cw.materialize_call(call, data, budget=16 << 30)
        self.assertEqual(list(built["cache"].shape), [2, 64, 68])
        self.assertEqual(built["block_table"].tolist(), [[1, 0]])
        self.assertEqual(built["slot_mapping"].tolist(), [64 + i for i in range(8)] + [-1])

    def test_unnamed_index_still_refuses(self):
        data = [self.keys, self.slots.clone(), self.sample]
        with self.assertRaises(cw.Unreplayable) as ctx:
            cw.materialize_call(_call("positions"), data, budget=16 << 30)
        self.assertIn("no pool/index pair", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
