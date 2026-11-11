#!/usr/bin/env python3
"""CPU tests for rank_timing.py (rank_chain timing of whole-layer monokernels). No GPU, no torch:
the rank workers themselves need both; here the spec, the device lease, the fold over ranks and the
failure path (a rank that dies must fail the bucket, never hang it) are pinned.

Run: python3 shared/tests/test_rank_timing.py
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("rank_timing", os.path.join(SCRIPTS_DIR, "rank_timing.py"))
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)


class _Env:
    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.kw}
        for k, v in self.kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class Spec(unittest.TestCase):
    def test_only_rank_chain_tasks_get_a_spec(self):
        self.assertIsNone(rt.chain_spec({}))
        self.assertIsNone(rt.chain_spec({"timing": {"kind": "other"}}))

    def test_defaults_and_overrides(self):
        s = rt.chain_spec({"timing": {"kind": "rank_chain", "world": "4", "iters": 7}})
        self.assertEqual((s["world"], s["iters"], s["reps"]), (4, 7, rt.DEFAULTS["reps"]))

    def test_world_must_be_positive(self):
        with self.assertRaises(ValueError):
            rt.chain_spec({"timing": {"kind": "rank_chain", "world": 0}})


class Devices(unittest.TestCase):
    def test_the_leg_lease_is_split_one_device_a_rank_in_order(self):
        with _Env(HIP_VISIBLE_DEVICES="4,5,6,7,0,1,2,3", CUDA_VISIBLE_DEVICES=None):
            self.assertEqual(rt.visible_devices(8), ["4", "5", "6", "7", "0", "1", "2", "3"])

    def test_too_few_devices_is_refused(self):
        with _Env(HIP_VISIBLE_DEVICES="0", CUDA_VISIBLE_DEVICES=None):
            with self.assertRaises(RuntimeError):
                rt.visible_devices(8)

    def test_unpinned_means_the_first_world_devices(self):
        with _Env(HIP_VISIBLE_DEVICES=None, CUDA_VISIBLE_DEVICES=None):
            self.assertEqual(rt.visible_devices(2), ["0", "1"])


class Fold(unittest.TestCase):
    def _rank(self, r, ms, host=0.01):
        return {"rank": r, "ms": ms, "samples_ms": [ms], "host_ms": host, "length": 8, "short": 0}

    def test_the_bucket_is_its_slowest_rank(self):
        spec = rt.chain_spec({"timing": {"kind": "rank_chain"}})
        res = rt.summarize([self._rank(0, 0.13), self._rank(1, 0.15), self._rank(2, 0.14)], spec)
        self.assertEqual(res["ms"], 0.15)
        self.assertEqual(res["per_rank_ms"], [0.13, 0.15, 0.14])
        self.assertEqual(res["timer"], rt.TIMER)
        self.assertIs(res["primed"], True)
        self.assertEqual(res["cache_condition"]["mode"], "distinct-layer-chain")

    def test_host_slower_than_device_is_not_primed(self):
        spec = rt.chain_spec({"timing": {"kind": "rank_chain"}})
        res = rt.summarize([self._rank(0, 0.15, host=0.2), self._rank(1, 0.15)], spec)
        self.assertIs(res["primed"], False)


class Bucket(unittest.TestCase):
    def setUp(self):
        self.task = tempfile.mkdtemp(prefix="rank_chain_task_")
        self.addCleanup(shutil.rmtree, self.task, True)
        os.makedirs(os.path.join(self.task, "run"))
        with open(os.path.join(self.task, "run", "ut_cases.json"), "w") as f:
            json.dump({"cases": [{"sig": "a", "timed": True, "regime": "DECODE", "m": 8},
                                 {"sig": "b", "timed": False}]}, f)

    def test_timed_rows_follow_the_manifest_and_bucket(self):
        self.assertEqual([c["sig"] for c in rt.timed_rows(self.task)], ["a"])
        self.assertEqual(rt.timed_rows(self.task, "b"), [])

    def test_a_dead_rank_fails_the_bucket_promptly_and_leaves_nothing_behind(self):
        """The workers here cannot start (no cases.py, and no torch on a CPU box): the bucket must
        raise, not wait for its timeout, and the scratch directory must be gone."""
        spec = dict(rt.chain_spec({"timing": {"kind": "rank_chain", "world": 2}}), worker_timeout_s=60)
        before = set(os.listdir(tempfile.gettempdir()))
        t0 = time.time()
        with _Env(HIP_VISIBLE_DEVICES="0,1", CUDA_VISIBLE_DEVICES=None):
            with self.assertRaises(RuntimeError):
                rt.time_bucket(self.task, {"sig": "a"}, {}, spec)
        self.assertLess(time.time() - t0, 50)
        left = {d for d in set(os.listdir(tempfile.gettempdir())) - before if d.startswith("rank_chain_")}
        self.assertEqual(left, set())


if __name__ == "__main__":
    unittest.main()
