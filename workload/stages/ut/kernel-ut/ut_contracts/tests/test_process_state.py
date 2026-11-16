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


CACHE_KEY = "FLYDSL_RUNTIME_CACHE_DIR"
HARNESS = """import hashlib, os

def _cache_dir():
    with open(os.environ["LEGENV_TEST_PKG"], "rb") as f:
        return "/cache/" + hashlib.sha256(f.read()).hexdigest()[:20]

def leg_env():
    return {"FLYDSL_RUNTIME_CACHE_DIR": _cache_dir()}
"""


class LegEnvTests(unittest.TestCase):
    """meta.leg_env: a value that differs per leg by design (the Muse mono harness keys FlyDSL's
    cache directory by the bytes of the package the leg resolved, so the two legs' values differ)
    is set by leg_runner before the first snapshot and is no candidate-only change; what a leg
    sets while it runs, that key included, still is (G1b, 2026-11-11)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        with open(os.path.join(self.dir, "legenv_harness.py"), "w") as f:
            f.write(HARNESS)
        for leg in ("baseline", "candidate"):
            with open(os.path.join(self.dir, leg + ".pkg"), "w") as f:
                f.write("# the sgl_amd_mono bytes the %s leg resolves\n" % leg)

    def tearDown(self):
        sys.modules.pop("legenv_harness", None)
        self.tmp.cleanup()

    def leg(self, name, body, declare=True):
        """leg_runner.main for one leg, its _run replaced by `body` (the harness + kernel)."""
        task = os.path.join(self.dir, "task_" + name)
        os.makedirs(task, exist_ok=True)
        with open(os.path.join(task, "legenv_harness.py"), "w") as f:
            f.write(HARNESS)
        meta = {"target_callable": "pkg.mod:fn"}
        if declare:
            meta["leg_env"] = "legenv_harness:leg_env"
        with open(os.path.join(task, "meta.json"), "w") as f:
            json.dump(meta, f)
        out = os.path.join(task, "process_state.json")
        torch, _ = fake_torch()
        env = {"PROCESS_STATE_OUT": out, "LEGENV_TEST_PKG": os.path.join(self.dir, name + ".pkg")}
        with patch.dict(sys.modules, {"torch": torch}), patch.dict(os.environ, env), \
                patch.object(sys, "argv", ["leg_runner.py", "--task", task, "--mode", "oracle"]), \
                patch.object(leg, "_run", lambda a, task, meta: body()):
            os.environ.pop(CACHE_KEY, None)
            os.environ.pop("COMPILE_WATCH_OUT", None)
            leg.main()
            sys.modules.pop("legenv_harness", None)
        with open(out) as f:
            return json.load(f)

    def harness(self, extra=None):
        """What rank_harness.Group / rank_chain.Chain do: make sure FlyDSL builds into the leg's
        source-keyed cache (a no-op when meta.leg_env already set it), then the kernel runs."""
        def body():
            h = _load("legenv_harness_body", os.path.join(self.dir, "legenv_harness.py"))
            if os.environ.get(CACHE_KEY) != h._cache_dir():
                os.environ[CACHE_KEY] = h._cache_dir()
            os.environ.update(extra or {})
        return body

    def test_per_leg_cache_dirs_do_not_fail_the_gate(self):
        base = self.leg("baseline", self.harness())
        cand = self.leg("candidate", self.harness())
        self.assertNotEqual(base["before"]["env:" + CACHE_KEY], cand["before"]["env:" + CACHE_KEY],
                            "the two legs resolve different source: their cache keys must differ")
        self.assertEqual(base["leg_env"], [CACHE_KEY])
        changed, allowed = driver.candidate_only_changes(base, cand)
        self.assertEqual((changed, allowed), ({}, {}))

    def test_the_same_harness_without_leg_env_fails_as_g1b_did(self):
        base = self.leg("baseline", self.harness(), declare=False)
        cand = self.leg("candidate", self.harness(), declare=False)
        changed, _ = driver.candidate_only_changes(base, cand)
        self.assertEqual(sorted(changed), ["env:" + CACHE_KEY])

    def test_a_candidate_that_sets_another_variable_still_fails(self):
        base = self.leg("baseline", self.harness())
        cand = self.leg("candidate", self.harness({"HIPBLASLT_TUNING_FILE": "/tmp/x"}))
        changed, _ = driver.candidate_only_changes(base, cand)
        self.assertEqual(sorted(changed), ["env:HIPBLASLT_TUNING_FILE"])

    def test_a_candidate_that_moves_the_cache_while_it_runs_still_fails(self):
        base = self.leg("baseline", self.harness())
        cand = self.leg("candidate", self.harness({CACHE_KEY: "/tmp/elsewhere"}))
        changed, _ = driver.candidate_only_changes(base, cand)
        self.assertEqual(sorted(changed), ["env:" + CACHE_KEY])

    def test_a_malformed_declaration_stops_the_leg(self):
        os.makedirs(os.path.join(self.dir, "t"))
        with open(os.path.join(self.dir, "t", "h.py"), "w") as f:
            f.write("def bad():\n    return {'X': 1}\n")
        for decl in ("h", "h:", "../h:bad", "h:bad"):
            with self.subTest(decl=decl), patch.dict(os.environ, {}):
                with self.assertRaises(SystemExit):
                    leg.apply_leg_env(os.path.join(self.dir, "t"), {"leg_env": decl})
                self.assertNotIn("X", os.environ)
        sys.modules.pop("h", None)
        self.assertEqual(leg.apply_leg_env(self.dir, {}), {})

    def test_lint(self):
        U = driver._load("ut_contract", str(HERE / "ut_contract.py"))
        lint = U._lint_leg_env
        self.assertEqual(lint(self.dir, {"leg_env": "legenv_harness:leg_env"}), [])
        self.assertEqual(len(lint(self.dir, {"leg_env": "legenv_harness:nope"})), 1)
        self.assertEqual(len(lint(self.dir, {"leg_env": "missing:leg_env"})), 1)
        self.assertEqual(len(lint(self.dir, {"leg_env": "legenv_harness"})), 1)
        allow = {"ut_contract": {"process_state_allow": [{"prefix": "env:FLYDSL_RUNTIME_CACHE_DIR"}]}}
        self.assertEqual(len(lint(self.dir, allow)), 1, "an allow entry without a why is an error")


if __name__ == "__main__":
    unittest.main()
