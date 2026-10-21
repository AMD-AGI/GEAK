"""Real PyTorch CPU integration tests; opt in with GEAK_REAL_TORCH_TESTS=1.

Run in the serving image without GPU devices. The ordinary stdlib-only suite can
still run on a login node without torch. All generated test files are temporary.
"""
import atexit
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(os.environ.get("GEAK_REAL_TORCH_TESTS") == "1", "requires real CPU torch")
class ReferenceStorageCPU(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.h = load("harness_lib")
        tmp = tempfile.TemporaryDirectory(prefix="geak-reference-test-")
        self.addCleanup(tmp.cleanup)
        self.task = Path(tmp.name)
        for name in ("harness_lib.py", "leg_runner.py"):
            shutil.copyfile(SCRIPTS / name, self.task / name)
        (self.task / "baseline_overlay").mkdir()
        self.meta = {"target_callable": "live_baseline:op", "regime": {}}
        (self.task / "meta.json").write_text(json.dumps(self.meta))
        (self.task / "cases.py").write_text('''
import torch
def random_shapes(h, meta):
    for m in (2, 5):
        yield {"sig": str(m), "make_inputs": lambda rng, m=m:
               torch.randn((m, 3), generator=rng, dtype=torch.bfloat16).t()}
def call(args):
    from live_baseline import op
    return op(args)
''')
        self.baseline = self.task / "baseline_overlay/live_baseline.py"
        self.baseline.write_text('''
import torch
def op(x):
    return {"out": x * 2, "nested": (torch.arange(4, dtype=torch.int32),
            torch.arange(8, dtype=torch.uint8), None, 7)}
''')

    def refs(self, **kw):
        paths = []
        original = self.h._run_leg
        def run(*args, **kwargs):
            paths.append(kwargs["out"])
            return original(*args, **kwargs)
        try:
            with patch.object(self.h, "_run_leg", run):
                return self.h.baseline_random_outputs(str(self.task), self.meta, **kw)
        finally:
            self.assertTrue(paths)
            for path in paths:
                self.assertFalse(os.path.exists(path), "memfd descriptor leaked")
            self.assertEqual(list(self.task.rglob("*.pt")), [])

    def test_real_subprocess_seeded_nested_strided_outputs(self):
        refs = self.refs(seed=19, draws=2)
        self.assertEqual(set(refs), {"2|0", "2|1", "5|0", "5|1"})
        for m in (2, 5):
            for draw in range(2):
                x = self.torch.randn((m, 3), dtype=self.torch.bfloat16,
                                     generator=self.torch.Generator().manual_seed(19 + draw)).t()
                got = refs[f"{m}|{draw}"]
                self.assertTrue(self.torch.equal(got["out"], x * 2))
                self.assertEqual(got["out"].stride(), x.stride())
                self.assertTrue(self.torch.equal(got["nested"][0], self.torch.arange(4, dtype=self.torch.int32)))
                self.assertTrue(self.torch.equal(got["nested"][1], self.torch.arange(8, dtype=self.torch.uint8)))
                self.assertEqual(got["nested"][2:], (None, 7))

    def test_concurrent_baseline_calls_do_not_overwrite_each_other(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            def run(seed):
                return self.h.baseline_random_outputs(str(self.task), self.meta,
                                                      bucket="2", draws=1, seed=seed)
            a, b = list(pool.map(run, [7, 19]))
        self.assertFalse(self.torch.equal(a["2|0"]["out"], b["2|0"]["out"]))
        self.assertEqual(list(self.task.rglob("*.pt")), [])

    def test_real_over_budget_and_child_failure_clean_up(self):
        with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
            self.refs(max_output_bytes=1)
        self.baseline.write_text("def op(x): raise RuntimeError('intentional failure')\n")
        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            self.refs()

    def test_real_timeout_closes_reference_buffer(self):
        self.baseline.write_text("import time\ndef op(x): time.sleep(60)\n")
        with self.assertRaises(subprocess.TimeoutExpired):
            self.refs(timeout=5)

    def test_metadata_capture_and_promotion_never_save_tensors(self):
        cs = load("capture_shapes")
        # The test module's pure Python callable is a real wrappable seam.
        with patch.dict(os.environ, {"CAPTURE_PERSIST_POLICY": "metadata_only",
                                     "CAPTURE_PROCESS_UNIQUE": "1"}):
            cs.install(__name__ + ":toy_op", str(self.task), max_cases=1)
        self.addCleanup(atexit.unregister, cs._flush)
        self.addCleanup(setattr, sys.modules[__name__], "toy_op", cs._STATE["orig"])
        with patch.object(self.torch, "save", side_effect=AssertionError("tensor dump")):
            for x in (self.torch.zeros(2, 3), self.torch.zeros(2, 3).t(),
                      self.torch.zeros(5, 3)):
                toy_op(x, epsilon=1e-5)
            cs._flush()
        cap = Path(cs._STATE["out_dir"]) / "meta.json"
        result = cs.promote_and_reclaim(str(self.task), str(cap))
        self.assertFalse(result["authoritative_oracle_present"])
        meta = json.loads((self.task / "meta.json").read_text())
        self.assertEqual(meta["num_distinct_shapes"], 3)
        self.assertIsNone(meta["reference_io"])
        self.assertFalse(meta["oracle_complete"])
        self.assertEqual(list(self.task.rglob("*.pt")), [])


def toy_op(x, epsilon):
    return x + epsilon


if __name__ == "__main__":
    unittest.main()
