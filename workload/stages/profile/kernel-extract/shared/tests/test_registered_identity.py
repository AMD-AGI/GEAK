"""CPU integration against the installed AITER adapter and real torch dispatcher."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
import uuid

try:
    import torch
except ImportError:
    torch = None

GUARD = Path(os.environ.get("AITER_TORCH_GUARD_SOURCE",
                          "/sgl-workspace/aiter/aiter/jit/utils/torch_guard.py"))
SPEC = importlib.util.spec_from_file_location(
    "identity_leg", Path(__file__).resolve().parents[1] / "leg_runner.py")
lr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lr)


@unittest.skipUnless(torch is not None and GUARD.is_file(), "requires pinned torch/AITER image")
class RegisteredIdentity(unittest.TestCase):
    def setUp(self):
        lr._observe_dispatch_registration()
        spec = importlib.util.spec_from_file_location("identity_test_torch_guard", GUARD)
        self.guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.guard)
        sys.modules[self.guard.__name__] = self.guard
        self.name = "geak_identity_" + uuid.uuid4().hex
        self.mods = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def tearDown(self):
        if self.guard.aiter_lib is not None:
            self.guard.aiter_lib._destroy()
        for mod in self.mods + [self.guard.__name__]:
            sys.modules.pop(mod, None)
        for key in list(lr._DISPATCH_IMPLS):
            if key[0] == "aiter::" + self.name:
                lr._DISPATCH_IMPLS.pop(key)

    def body(self, suffix, offset):
        name = self.name + suffix
        mod = types.ModuleType(name)
        path = Path(self.tmp.name) / (name + ".py")
        source = ("import torch\n"
                  f"def {self.name}(x: torch.Tensor) -> torch.Tensor:\n"
                  f"    return x + {offset}\n")
        path.write_text(source)
        mod.__file__ = str(path)
        exec(compile(source, str(path), "exec"), mod.__dict__)
        raw = getattr(mod, self.name)
        mod.op = self.guard.torch_compile_guard(mutates_args=[])(raw)
        sys.modules[name] = mod
        self.mods.append(name)
        return mod, raw

    def test_identity_is_registered_body_and_cpu_dispatch_executes_it(self):
        mod, raw = self.body("base", 3)
        ident = lr._identity(mod.__name__ + ":op")
        self.assertEqual(ident["file"], raw.__code__.co_filename)
        self.assertEqual(ident["qualname"], self.name)
        self.assertEqual(ident["dispatch"], "aiter::" + self.name + "/CUDA")
        torch.testing.assert_close(mod.op(torch.tensor([2.])), torch.tensor([5.]))

    def test_candidate_wrapper_cannot_claim_discarded_candidate_body(self):
        base, _ = self.body("base", 3)
        cand, _ = self.body("cand", 19)
        self.assertEqual(lr._identity(base.__name__ + ":op"),
                         lr._identity(cand.__name__ + ":op"))
        torch.testing.assert_close(cand.op(torch.tensor([2.])), torch.tensor([5.]))

    def test_missing_observation_is_error_not_candidate_module_fallback(self):
        mod, _ = self.body("base", 3)
        lr._DISPATCH_IMPLS.pop(("aiter::" + self.name, "CUDA"))
        self.assertIn("no observed live CUDA registration", lr._identity(mod.__name__ + ":op")["error"])

    def test_removed_registration_is_rejected(self):
        mod, _ = self.body("base", 3)
        self.guard.aiter_lib._destroy()
        self.assertIn("error", lr._identity(mod.__name__ + ":op"))

    def test_unrecognized_registered_body_is_rejected(self):
        mod, _ = self.body("base", 3)
        self.guard.aiter_lib.impl(self.name, lambda x: x + 99, "CUDA", allow_override=True)
        self.assertIn("unrecognized registered AITER body", lr._identity(mod.__name__ + ":op")["error"])

    def test_plain_callable_identity_stays_unchanged(self):
        mod, raw = self.body("base", 3)
        mod.op = raw
        self.assertEqual(lr._identity(mod.__name__ + ":op"), {
            "module": mod.__name__, "file": mod.__file__, "qualname": self.name})


if __name__ == "__main__":
    unittest.main()
