"""The process_state gate: a candidate that changes process-wide state fails, the stack's own
changes cancel, and a declared change is allowed. CPU only, with a stand-in torch module."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    saved = sys.path[:]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


driver = _load("process_state_driver", HERE / "ut_driver.py")
leg = _load("process_state_leg", HERE.parent / "shared" / "leg_runner.py")


def fake_torch():
    """Just enough of torch for leg_runner._process_state to read."""
    t = ModuleType("torch")
    flags = SimpleNamespace(allow_tf32=False, allow_fp16_reduced_precision_reduction=True,
                            allow_bf16_reduced_precision_reduction=True)
    t.get_default_dtype = lambda: "torch.bfloat16"
    t.get_float32_matmul_precision = lambda: "highest"
    t.are_deterministic_algorithms_enabled = lambda: False
    t.get_num_threads = lambda: 8
    t.backends = SimpleNamespace(
        cuda=SimpleNamespace(matmul=flags, preferred_blas_library=lambda: "hipblaslt",
                             preferred_linalg_library=lambda: "default"),
        cudnn=SimpleNamespace(enabled=True, benchmark=False, deterministic=False, allow_tf32=True))
    tun = {"on": False, "file": ""}
    t.cuda = SimpleNamespace(tunable=SimpleNamespace(
        is_enabled=lambda: tun["on"], tuning_is_enabled=lambda: False,
        record_untuned_is_enabled=lambda: False, get_filename=lambda: tun["file"],
        get_results=lambda: [1] * (5 if tun["on"] else 0)))
    t.mm = lambda *a: None
    t.matmul = lambda *a: None
    t.nn = SimpleNamespace(functional=SimpleNamespace(linear=lambda *a: None))
    t.Tensor = SimpleNamespace(matmul=lambda *a: None)
    return t, tun


class ProcessStateTests(unittest.TestCase):
    def snap(self, act):
        torch, tun = fake_torch()
        with patch.dict(sys.modules, {"torch": torch}), patch.dict(os.environ, {}, clear=False):
            before = leg._process_state()
            act(torch, tun)
            after = leg._process_state()
        return {"before": before, "after": after}

    def test_a_candidate_that_enables_tunableop_fails(self):
        base = self.snap(lambda torch, tun: None)

        def candidate(torch, tun):
            tun.update(on=True, file="/tmp/aiter_gemm_a8w8_tuned_x.csv")
        cand = self.snap(candidate)
        changed, allowed = driver.candidate_only_changes(base, cand)
        self.assertIn("torch.cuda.tunable.is_enabled", changed)
        self.assertIn("torch.cuda.tunable.get_filename", changed)
        self.assertEqual(allowed, {})

    def test_what_the_serving_stack_itself_changes_cancels(self):
        def stack(torch, tun):
            os.environ["SGLANG_SET_BY_IMPORT"] = "1"
            torch.backends.cuda.matmul.allow_tf32 = True
        changed, _ = driver.candidate_only_changes(self.snap(stack), self.snap(stack))
        self.assertEqual(changed, {})

    def test_rebinding_a_shared_torch_callable_fails(self):
        def candidate(torch, tun):
            torch.mm = lambda *a: "patched"
        changed, _ = driver.candidate_only_changes(self.snap(lambda t, u: None),
                                                   self.snap(candidate))
        self.assertIn("callable:torch.mm", changed)

    def test_an_environment_variable_set_by_the_candidate_fails(self):
        def candidate(torch, tun):
            os.environ["HIPBLASLT_TUNING_FILE"] = "/tmp/x"
        changed, _ = driver.candidate_only_changes(self.snap(lambda t, u: None),
                                                   self.snap(candidate))
        self.assertIn("env:HIPBLASLT_TUNING_FILE", changed)

    def test_a_declared_change_is_allowed_with_its_reason(self):
        def candidate(torch, tun):
            tun.update(on=True)
        changed, allowed = driver.candidate_only_changes(
            self.snap(lambda t, u: None), self.snap(candidate),
            allow=[("torch.cuda.tunable.", "the tuned table is part of the deliverable")])
        self.assertEqual(changed, {})
        self.assertIn("torch.cuda.tunable.is_enabled", allowed)

    def test_the_gate_is_required(self):
        self.assertIn("process_state", driver.GATES)
        self.assertLess(driver.GATES.index("isolation"), driver.GATES.index("process_state"))


if __name__ == "__main__":
    unittest.main()
