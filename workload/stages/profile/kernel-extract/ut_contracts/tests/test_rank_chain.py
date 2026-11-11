"""CPU tests for whole-layer monokernel UTs (family `mono_layer`, timing kind `rank_chain`):
scaffolding, the lint rules, and the timing gate's refusal of a number that is not the deployed
device time (eager fallback in a graph phase, host-bound, or unreceipted).

Run: python3 workload/stages/profile/kernel-extract/ut_contracts/tests/test_rank_chain.py
"""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
UTC = os.path.dirname(HERE)
sys.path.insert(0, UTC)
import scaffold_ut  # noqa: E402
import ut_contract as U  # noqa: E402

_spec = importlib.util.spec_from_file_location("rank_chain_driver", os.path.join(UTC, "ut_driver.py"))
driver = importlib.util.module_from_spec(_spec)
_path = sys.path[:]
try:
    _spec.loader.exec_module(driver)
finally:
    sys.path[:] = _path


def _scaffold(root, world=8):
    task = os.path.join(root, "muse_mono_layer")
    with contextlib.redirect_stdout(io.StringIO()):
        scaffold_ut.main(["new", "--family", "mono_layer", "--task", task,
                          "--target", "pkg.runner:Runner.launch", "--bind-module", "pkg.gfx",
                          "--bind-file", "kernel_src/gfx/__init__.py",
                          "--axis", "hidden=8192", "--axis", "world=%d" % world])
    return task


class Scaffold(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="rank_chain_scaffold_")
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_mono_layer_tasks_are_timed_by_rank_chain(self):
        task = _scaffold(self.root, world=4)
        with open(os.path.join(task, "meta.json")) as f:
            meta = json.load(f)
        self.assertEqual(meta["timing"], {"kind": "rank_chain", "world": 4, "chain_length": 8})
        self.assertTrue(os.path.exists(os.path.join(task, "rank_timing.py")))
        with open(os.path.join(task, "cases.py")) as f:
            src = f.read()
        compile(src, "cases.py", "exec")
        self.assertIn("def rank_chain(rank, world, group, case)", src)
        self.assertIn("starts 4 rank processes", src)

    def test_other_families_are_untouched(self):
        task = os.path.join(self.root, "gemm")
        with contextlib.redirect_stdout(io.StringIO()):
            scaffold_ut.main(["new", "--family", "gemm_a8w8_blockscale", "--task", task,
                              "--attr", "fp8_dtype=float8_e4m3fn", "--axis", "n=7168", "--axis", "k=2048"])
        with open(os.path.join(task, "meta.json")) as f:
            self.assertNotIn("timing", json.load(f))
        with open(os.path.join(task, "cases.py")) as f:
            self.assertNotIn("rank_chain", f.read())


class Lint(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="rank_chain_lint_")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.task = _scaffold(self.root)

    def errors(self):
        return [e for e in U.lint_task(self.task)["errors"] if "rank_chain" in e or "timing" in e]

    def _meta(self, fn):
        p = os.path.join(self.task, "meta.json")
        with open(p) as f:
            meta = json.load(f)
        fn(meta)
        with open(p, "w") as f:
            json.dump(meta, f)

    def test_the_unfilled_stub_is_an_error(self):
        self.assertEqual(self.errors(), ["cases.py rank_chain is still the scaffold's FILL stub"])

    def test_a_filled_chain_passes(self):
        p = os.path.join(self.task, "cases.py")
        with open(p) as f:
            src = f.read().replace('NotImplementedError("FILL _Chain', 'NotImplementedError("written _Chain')
        with open(p, "w") as f:
            f.write(src)
        self.assertEqual(self.errors(), [])

    def test_meta_that_would_time_call_is_an_error(self):
        self._meta(lambda m: m.pop("timing"))
        self.assertTrue(any("would time call()" in e for e in self.errors()))

    def test_world_must_match_the_task(self):
        self._meta(lambda m: m["timing"].update(world=4))
        self.assertTrue(any("meta.timing.world 4" in e for e in self.errors()))


class TimingBasis(unittest.TestCase):
    """gate_timing refuses a speedup whose `ms` is not the deployed device time."""

    def _task(self, graph=True):
        h = SimpleNamespace(deployment_graph_mode=lambda r: bool(r.get("cuda_graph", True)))
        return SimpleNamespace(h=h, phase_regimes=lambda: {"DECODE": {"cuda_graph": graph}})

    @staticmethod
    def _row(timer="cuda_event_graph", primed=True, otimer=None, oprimed=None):
        return {"sig": "DECODE_r8", "regime": "DECODE",
                "baseline_timing": {"timers": [timer], "primed": primed, "host_ms": [0.01]},
                "optimized_timing": {"timers": [otimer or timer],
                                     "primed": primed if oprimed is None else oprimed, "host_ms": [0.01]}}

    def test_graph_replay_and_rank_chain_pass(self):
        self.assertEqual(driver.timing_basis_defects(self._task(), [self._row()]), [])
        self.assertEqual(driver.timing_basis_defects(self._task(), [self._row("rank_chain_graph")]), [])

    def test_eager_fallback_in_a_graph_phase_fails(self):
        d = driver.timing_basis_defects(self._task(), [self._row(otimer="cuda_event")])
        self.assertEqual(len(d), 1)
        self.assertIn("deploys under a CUDA graph", d[0])

    def test_eager_timing_is_fine_where_the_phase_is_eager(self):
        self.assertEqual(driver.timing_basis_defects(self._task(graph=False), [self._row("cuda_event")]), [])

    def test_host_bound_and_unreceipted_fail(self):
        d = driver.timing_basis_defects(self._task(), [self._row(primed=False)])
        self.assertTrue(d and all("host-bound" in x for x in d))
        d = driver.timing_basis_defects(self._task(), [{"sig": "x", "regime": "DECODE"}])
        self.assertTrue(d and all("no primed receipt" in x for x in d))

    def test_the_gate_reports_the_defects_instead_of_a_speedup(self):
        T = self._task()
        T.dir, T.meta, T.gpus = "/t", {}, None
        T.h.measure_legs = lambda *a, **k: [self._row(timer="cuda_event", primed=False)]
        T.h.serving_weighted_speedup = lambda per, meta: {"primary": 1.04}
        T.h.cache_policy = lambda: {}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r = driver.gate_timing(T, SimpleNamespace(max_reps=3), {})
        self.assertEqual(r["status"], "fail")
        self.assertTrue(r["defects"])
        self.assertNotIn("GEAK_WEIGHTED_SPEEDUP", out.getvalue())


class MultiRankLegs(unittest.TestCase):
    """Correctness / control legs of a multi-rank task must not be spread one GPU each."""

    def _T(self, meta):
        h = SimpleNamespace(rank_chain_world=lambda m: (int((m.get("timing") or {}).get("world", 8))
                                                        if (m.get("timing") or {}).get("kind") == "rank_chain" else None))
        return SimpleNamespace(h=h, meta=meta, rank_group=None)

    def test_rank_chain_task_gets_one_slot_with_the_whole_set(self):
        T = self._T({"timing": {"kind": "rank_chain", "world": 8}})
        got = driver.group_multi_rank(T, [str(i) for i in range(8)])
        self.assertEqual(got, ["0,1,2,3,4,5,6,7"])
        self.assertEqual(T.rank_group, [str(i) for i in range(8)])
        seen = []
        driver._parallel(["a", "b", "c"], got, lambda it, g: seen.append((it, g)))
        self.assertEqual([g for _, g in seen], ["0,1,2,3,4,5,6,7"] * 3)   # one at a time, every device

    def test_single_rank_tasks_keep_spreading(self):
        T = self._T({})
        self.assertEqual(driver.group_multi_rank(T, ["0", "1"]), ["0", "1"])
        self.assertIsNone(T.rank_group)
        self.assertIsNone(driver.group_multi_rank(self._T({"timing": {"kind": "rank_chain"}}), None))

    def test_too_few_devices_is_refused(self):
        with self.assertRaises(SystemExit):
            driver.group_multi_rank(self._T({"timing": {"kind": "rank_chain", "world": 8}}), ["0", "1"])


if __name__ == "__main__":
    unittest.main()
