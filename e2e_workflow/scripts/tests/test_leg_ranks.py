#!/usr/bin/env python3
"""Multi-rank legs (meta.world_size > 1), end to end on CPU: torchrun with the gloo backend, 2 ranks.

A toy region whose output on each rank depends on every rank (an all_reduce), as an EP MegaKernel's does:
- the time mode reports the slowest rank and every rank's time;
- the candidate leg's correctness is checked rank by rank against the baseline leg's per-rank outputs;
- a candidate wrong on ONE rank only fails, and names that rank;
- a list `candidate_bind` binds every module of the region.

Run: python3 -m pytest e2e_workflow/scripts/tests/test_leg_ranks.py -v
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import torch  # noqa: F401
    import torch.distributed  # noqa: F401
    HAVE_TORCH = True
except Exception:  # pragma: no cover - image without torch
    HAVE_TORCH = False

LIVE_OP = textwrap.dedent('''
    """The live stack's region: y = x * (rank + 1) + sum over ranks of x."""
    import torch.distributed as dist
    from toyregion import helper

    def op(x):
        s = x.clone()
        dist.all_reduce(s)
        return helper.scale(x) + s
''')
LIVE_HELPER = textwrap.dedent('''
    import torch.distributed as dist

    def scale(x):
        return x * (dist.get_rank() + 1)
''')
CASES = textwrap.dedent('''
    from toyregion.op import op

    def call(args):
        return op(args["x"])

    def timing_cases(h, meta):
        import torch
        import torch.distributed as dist
        for n in (8, 64):   # inputs are built per rank, so listing the cases needs the process group too
            yield {"sig": f"n{n}", "regime": "decode", "args": {"x": torch.ones(n) * (dist.get_rank() + 1)}, "m": n}

    def random_shapes(h, meta):
        import torch
        for n in (8, 64):
            yield {"sig": f"n{n}", "regime": "decode",
                   "make_inputs": lambda g, n=n: {"x": torch.randn(n, generator=g, device=g.device).cpu()}}

    def setup_dist(h, meta):
        import torch.distributed as dist
        assert dist.is_initialized()
''')


def _load_harness():
    spec = importlib.util.spec_from_file_location("harness_lib_ranks", os.path.join(SCRIPTS, "harness_lib.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class MultiRankLegs(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="legranks_")
        live = os.path.join(self.root, "site", "toyregion")
        os.makedirs(live)
        open(os.path.join(live, "__init__.py"), "w").close()
        with open(os.path.join(live, "op.py"), "w") as fh:
            fh.write(LIVE_OP)
        with open(os.path.join(live, "helper.py"), "w") as fh:
            fh.write(LIVE_HELPER)
        self.task = os.path.join(self.root, "task")
        os.makedirs(os.path.join(self.task, "kernel_src"))
        os.makedirs(os.path.join(self.task, "baseline_overlay"))
        for f in ("harness_lib.py", "leg_runner.py", "overlay_setup.py"):
            shutil.copy(os.path.join(SCRIPTS, f), self.task)
        with open(os.path.join(self.task, "cases.py"), "w") as fh:
            fh.write(CASES)
        # kernel_src/: the editable copy of both modules of the region.
        shutil.copy(os.path.join(live, "op.py"), os.path.join(self.task, "kernel_src", "op.py"))
        shutil.copy(os.path.join(live, "helper.py"), os.path.join(self.task, "kernel_src", "helper.py"))
        self.meta = {
            "target_callable": "toyregion.op:op", "world_size": 2, "dist_backend": "gloo", "tol": 1e-4,
            "random_draws": 2, "regime": {"cuda_graph": False},
            "candidate_bind": [
                {"kind": "module", "module": "toyregion.op", "file": "kernel_src/op.py"},
                {"kind": "module", "module": "toyregion.helper", "file": "kernel_src/helper.py"}],
        }
        with open(os.path.join(self.task, "meta.json"), "w") as fh:
            json.dump(self.meta, fh)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [os.path.join(self.root, "site")] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
        subprocess.run([sys.executable, os.path.join(self.task, "overlay_setup.py"), "add-marker",
                        "--overlay", os.path.join(self.task, "baseline_overlay"), "--target", "toyregion.op:op"],
                       capture_output=True, text=True)
        self.h = _load_harness()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _edit(self, name, text):
        with open(os.path.join(self.task, "kernel_src", name), "w") as fh:
            fh.write(text)

    def test_world_size_is_read_from_meta(self):
        self.assertEqual(self.h.world_size(self.task), 2)

    def test_time_reports_the_slowest_rank_and_every_rank(self):
        base = os.path.join(self.task, "baseline_overlay")
        res = self.h._run_leg(self.task, base, "time", bucket="n8", timeout=300)
        self.assertEqual(res["world_size"], 2)
        row = res["cases"][0]
        self.assertEqual(len(row["per_rank_ms"]), 2)
        self.assertEqual(row["ms"], max(row["per_rank_ms"]))

    def test_an_unchanged_candidate_passes_on_every_rank(self):
        res = self.h.run_correctness_ranks(self.task, self.meta, timeout=300)
        self.assertTrue(res["pass"], res)
        self.assertEqual([r["rank"] for r in res["per_rank"]], [0, 1])

    def test_a_candidate_wrong_on_one_rank_fails_and_names_it(self):
        self._edit("helper.py", LIVE_HELPER.replace(
            "return x * (dist.get_rank() + 1)",
            "return x * (dist.get_rank() + 1) * (1.01 if dist.get_rank() == 1 else 1.0)"))
        res = self.h.run_correctness_ranks(self.task, self.meta, timeout=300)
        self.assertFalse(res["pass"])
        self.assertEqual([r["rank"] for r in res["per_rank"] if not r["pass"]], [1])

    def test_every_bound_module_reaches_the_candidate_leg(self):
        base, cand = self.h.build_candidate_overlay(self.task, self.meta)
        man = json.load(open(os.path.join(cand, "_overlay_manifest.json")))
        self.assertEqual(sorted(m["module"] for m in man["modules"]), ["toyregion.helper", "toyregion.op"])

    def test_a_rerun_dir_of_symlinks_keeps_the_task_dir_off_sys_path(self):
        # fusion_handback re-runs a hand-back in a dir that symlinks the task's files; Python puts the
        # REAL dir (the task, with its unittest.py) on sys.path, which must not shadow stdlib unittest.
        with open(os.path.join(self.task, "unittest.py"), "w") as fh:
            fh.write("raise SystemExit('the task unittest.py shadowed stdlib unittest')\n")
        rerun = os.path.join(self.root, "rerun")
        os.makedirs(rerun)
        for name in os.listdir(self.task):
            os.symlink(os.path.join(self.task, name), os.path.join(rerun, name))
        res = self.h._run_leg(rerun, os.path.join(rerun, "baseline_overlay"), "time", bucket="n8", timeout=300)
        self.assertEqual(len(res["cases"][0]["per_rank_ms"]), 2)

    def test_measure_legs_scores_the_slowest_rank(self):
        per_case = self.h.measure_legs(self.task, self.meta, timeout=300, max_reps=1)
        self.assertEqual(sorted(r["sig"] for r in per_case), ["n64", "n8"])
        self.assertTrue(all(r.get("world_size") == 2 and r["speedup"] for r in per_case))


if __name__ == "__main__":
    unittest.main()
