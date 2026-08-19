#!/usr/bin/env python3
"""Unit tests for leg_runner.py -- the ONE executable both measurement legs run (stdlib only).

Pinned here: the sys.path scrub (the task dir holds a file literally named unittest.py, so it MUST be
off sys.path before torch imports stdlib unittest); _resolve/_identity, which h.assert_legs_differ
depends on to prove the two legs import different code; and the four modes list / resolve / time /
oracle.

torch, harness_lib and cases.py are all stubbed: the module imports them lazily/by path, so every
mode runs on CPU against fake tensors. Nothing here needs a GPU.

Run: python3 -m pytest e2e_workflow/scripts/tests/test_leg_runner.py -v
"""
import collections
import contextlib
import importlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import textwrap
import types
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MISSING = object()


def _load(mod_name, filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lr = _load("leg_runner", "leg_runner.py")


# --------------------------------------------------------------------------- #
# Stub task dir: a real harness_lib.py + cases.py on disk, because leg_runner
# loads both BY PATH out of the task dir (that is the anti-tamper contract).
# --------------------------------------------------------------------------- #
_HARNESS_STUB = textwrap.dedent('''
    """Stand-in for the task dir's harness_lib.py."""
    import sys
    calls = []

    def deployment_compile_mode(regime):
        return regime.get("compile_mode", "")

    def compiled_op(call, regime):
        calls.append(("compiled_op", regime.get("compile_mode")))
        def wrapped(args):
            calls.append(("wrapped", args.get("m")))
            return call(args)
        return wrapped

    def deployment_graph_mode(regime):
        return regime.get("graph_mode", False)

    def time_op(fn, graph=False, detail=False):
        calls.append(("time_op", graph, detail))
        fn()
        return {"ms": 1.25, "wall_ms": 1.5, "timer": "event"}

    def _torch():
        return sys.modules["torch"]
''')

_CASES_STUB = textwrap.dedent('''
    """Stand-in for the task dir's cases.py."""
    seen = []

    class _Out:
        """Minimal stand-in for a torch tensor: only what leg_runner touches."""
        def detach(self): return self
        def clone(self): return self
        def cpu(self): return self

    def timing_cases(h, meta):
        return [
            {"sig": "decode", "args": {"m": 1}, "regime": "decode", "m": 1},
            {"sig": "prefill", "args": {"m": 4096}, "regime": "prefill", "m": 4096},
        ]

    def random_shapes(h, meta):
        return [{"sig": "decode", "make_inputs": lambda rng: {"seed": rng.seed,
                                                              "device": rng.device}}]

    def call(args):
        seen.append(args)
        return _Out()
''')


def _fake_torch():
    torch = types.ModuleType("torch")
    torch.saved = []
    torch.cuda_available = False
    torch.cuda = types.SimpleNamespace(is_available=lambda: torch.cuda_available)

    class Generator:
        def __init__(self, device=None):
            self.device = device
            self.seed = None

        def manual_seed(self, s):
            self.seed = s
            return self

    torch.Generator = Generator
    torch.save = lambda obj, path: torch.saved.append((obj, path))
    return torch


class _LegRunnerTestCase(unittest.TestCase):
    """A temp task dir + a stub torch, with every global the module mutates restored."""

    def setUp(self):
        self.task = tempfile.mkdtemp(prefix="leg_runner_task_")
        self.addCleanup(shutil.rmtree, self.task, True)
        self.torch = _fake_torch()
        self._prev_torch = sys.modules.get("torch", _MISSING)
        sys.modules["torch"] = self.torch
        self.addCleanup(self._restore_torch)
        # _load() plants harness_lib/cases into sys.modules under fixed names.
        for name in ("harness_lib", "cases"):
            self.addCleanup(sys.modules.pop, name, None)
        self._argv = sys.argv
        self.addCleanup(setattr, sys, "argv", self._argv)

    def _restore_torch(self):
        if self._prev_torch is _MISSING:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self._prev_torch

    def _write(self, name, text):
        with open(os.path.join(self.task, name), "w") as fh:
            fh.write(text)

    def _task_dir(self, meta=None, harness=_HARNESS_STUB, cases=_CASES_STUB):
        self._write("meta.json", json.dumps(meta if meta is not None else {
            "target_callable": "fake_live_stack:op", "regime": {}}))
        self._write("harness_lib.py", harness)
        self._write("cases.py", cases)
        return self.task

    def _target_module(self, name="fake_live_stack", fn=None, file_attr=_MISSING):
        mod = types.ModuleType(name)
        mod.op = fn if fn is not None else (lambda args: None)
        if file_attr is not _MISSING:
            mod.__file__ = file_attr
        else:
            mod.__file__ = os.path.join(self.task, "..", f"{name}.py")
        sys.modules[name] = mod
        self.addCleanup(sys.modules.pop, name, None)
        return mod

    def _main(self, *argv):
        sys.argv = ["leg_runner.py", *argv]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            lr.main()
        return out.getvalue()


# --------------------------------------------------------------------------- #
# The sys.path scrub -- the task dir must not shadow stdlib unittest
# --------------------------------------------------------------------------- #
class PathScrub(unittest.TestCase):
    def test_module_dir_is_dropped_from_sys_path_at_import(self):
        """A task dir holds a file named unittest.py; leaving its dir on sys.path shadows stdlib.

        Re-run the scrub rather than asserting the process-global: other test modules in the same
        pytest session put SCRIPTS_DIR back on sys.path at import time (test_op_bench,
        test_run_correctness_gate), so a bare assertion here passes alone and fails in the suite.
        What actually needs pinning is that EXECUTING leg_runner removes its own dir.
        """
        here = os.path.abspath(SCRIPTS_DIR)
        saved = list(sys.path)
        self.addCleanup(sys.path.__setitem__, slice(None), saved)
        sys.path.insert(0, here)
        self.addCleanup(sys.modules.pop, "leg_runner_scrub_probe", None)
        _load("leg_runner_scrub_probe", "leg_runner.py")
        self.assertNotIn(here, [os.path.abspath(p or ".") for p in sys.path])


# --------------------------------------------------------------------------- #
# _load / _resolve / _identity
# --------------------------------------------------------------------------- #
class Resolve(_LegRunnerTestCase):
    def test_load_executes_the_file_and_registers_it(self):
        self._write("mod_under_test.py", "VALUE = 7\n")
        mod = lr._load("mod_under_test", os.path.join(self.task, "mod_under_test.py"))
        self.addCleanup(sys.modules.pop, "mod_under_test", None)
        self.assertEqual(mod.VALUE, 7)
        self.assertIs(sys.modules["mod_under_test"], mod)

    def test_resolve_walks_a_dotted_attr_path(self):
        mod = self._target_module()
        mod.layer = types.SimpleNamespace(inner=types.SimpleNamespace(op="DEEP"))
        self.assertEqual(lr._resolve("fake_live_stack:layer.inner.op"), "DEEP")

    def test_resolve_with_empty_attr_returns_the_module(self):
        mod = self._target_module()
        self.assertIs(lr._resolve("fake_live_stack:"), mod)

    def test_identity_reports_module_realpath_and_qualname(self):
        real = os.path.join(self.task, "live.py")
        self._write("live.py", "")
        self._target_module(file_attr=real)

        def op(args):
            return None

        op.__module__ = "fake_live_stack"
        sys.modules["fake_live_stack"].op = op
        ident = lr._identity("fake_live_stack:op")
        self.assertEqual(ident["module"], op.__module__)
        self.assertTrue(ident["qualname"].endswith("op"))
        self.assertEqual(ident["file"], os.path.realpath(real))

    def test_identity_of_an_unresolvable_callable_is_an_error_record(self):
        """assert_legs_differ reads this dict; a raise here would abort the whole measurement."""
        ident = lr._identity("no_such_module_at_all:op")
        self.assertIn("error", ident)
        self.assertIn("ModuleNotFoundError", ident["error"])

    def test_identity_falls_back_to_the_target_module_file(self):
        """A callable whose __module__ is not importable still reports the target module's file."""
        real = os.path.join(self.task, "live.py")
        self._write("live.py", "")
        mod = self._target_module(file_attr=real)

        def op(args):
            return None

        op.__module__ = "module_that_does_not_exist"
        mod.op = op
        ident = lr._identity("fake_live_stack:op")
        self.assertEqual(ident["module"], "module_that_does_not_exist")
        self.assertEqual(ident["file"], os.path.realpath(real))

    def test_identity_tolerates_a_callable_with_no_file_anywhere(self):
        mod = self._target_module(file_attr=None)
        mod.op = object()  # no __module__, no __qualname__
        ident = lr._identity("fake_live_stack:op")
        self.assertEqual(ident["file"], "")
        self.assertTrue(ident["qualname"])

    def test_identity_survives_a_module_whose_file_attribute_explodes(self):
        """Both file lookups are best-effort: a hostile/lazy module must not abort the leg."""
        class Booby:
            op = staticmethod(lambda args: None)

            @property
            def __file__(self):
                raise RuntimeError("lazy attribute blew up")

        sys.modules["booby_stack"] = Booby()
        self.addCleanup(sys.modules.pop, "booby_stack", None)
        Booby.op.__module__ = "booby_stack"
        ident = lr._identity("booby_stack:op")
        self.assertEqual(ident["file"], "")
        self.assertNotIn("error", ident)

    def test_two_legs_resolving_different_code_report_different_tuples(self):
        """The precondition assert_legs_differ enforces, exercised end to end."""
        self._write("base.py", "def op(args):\n    return 'BASE'\n")
        self._write("cand.py", "def op(args):\n    return 'CAND'\n")
        base = lr._load("leg_base_mod", os.path.join(self.task, "base.py"))
        cand = lr._load("leg_cand_mod", os.path.join(self.task, "cand.py"))
        for name in ("leg_base_mod", "leg_cand_mod"):
            self.addCleanup(sys.modules.pop, name, None)
        self.assertNotEqual(lr._identity("leg_base_mod:op"), lr._identity("leg_cand_mod:op"))
        self.assertEqual((base.op(None), cand.op(None)), ("BASE", "CAND"))


# --------------------------------------------------------------------------- #
# main() -- the four modes
# --------------------------------------------------------------------------- #
class ModeResolve(_LegRunnerTestCase):
    def test_resolve_prints_the_identity_without_loading_the_harness(self):
        """resolve runs BEFORE harness_lib/cases are loaded, so a bare task dir is enough."""
        self._write("meta.json", json.dumps({"target_callable": "fake_live_stack:op"}))
        self._target_module()
        ident = json.loads(self._main("--task", self.task, "--mode", "resolve"))
        self.assertTrue(ident["qualname"].endswith("<lambda>"))
        self.assertNotIn("harness_lib", sys.modules)

    def test_task_path_is_absolutised(self):
        self._task_dir()
        self._target_module()
        rel = os.path.relpath(self.task, os.getcwd())
        ident = json.loads(self._main("--task", rel, "--mode", "resolve"))
        self.assertNotIn("error", ident)


class ModeList(_LegRunnerTestCase):
    def test_list_prints_the_bucket_sigs_from_the_task_own_cases(self):
        self._task_dir()
        self._target_module()
        self.assertEqual(json.loads(self._main("--task", self.task, "--mode", "list")),
                         {"sigs": ["decode", "prefill"]})


class ModeTime(_LegRunnerTestCase):
    def test_time_reports_every_case_plus_the_leg_identity(self):
        self._task_dir()
        self._target_module()
        out = json.loads(self._main("--task", self.task, "--mode", "time"))
        self.assertEqual([c["sig"] for c in out["cases"]], ["decode", "prefill"])
        self.assertEqual(out["cases"][0]["ms"], 1.25)
        self.assertEqual(out["cases"][1]["m"], 4096)
        self.assertEqual(out["cases"][0]["regime"], "decode")
        self.assertTrue(out["identity"]["qualname"].endswith("<lambda>"))

    def test_bucket_selects_exactly_one_case(self):
        self._task_dir()
        self._target_module()
        out = json.loads(self._main("--task", self.task, "--mode", "time", "--bucket", "prefill"))
        self.assertEqual([c["sig"] for c in out["cases"]], ["prefill"])

    def test_unknown_bucket_times_nothing_rather_than_everything(self):
        self._task_dir()
        self._target_module()
        out = json.loads(self._main("--task", self.task, "--mode", "time", "--bucket", "nope"))
        self.assertEqual(out["cases"], [])

    def test_deployment_compile_and_graph_modes_are_applied(self):
        """Timing the eager callable when the deployment compiles/graphs it is a fake number."""
        self._task_dir(meta={"target_callable": "fake_live_stack:op",
                             "regime": {"compile_mode": "max-autotune", "graph_mode": True}})
        self._target_module()
        self._main("--task", self.task, "--mode", "time")
        h = sys.modules["harness_lib"]
        self.assertIn(("compiled_op", "max-autotune"), h.calls)
        self.assertTrue(all(c[1] is True for c in h.calls if c[0] == "time_op"))

    def test_a_timer_returning_none_degrades_to_null_fields(self):
        self._task_dir(harness=_HARNESS_STUB.replace(
            'return {"ms": 1.25, "wall_ms": 1.5, "timer": "event"}', "return None"))
        self._target_module()
        out = json.loads(self._main("--task", self.task, "--mode", "time"))
        self.assertIsNone(out["cases"][0]["ms"])
        self.assertIsNone(out["cases"][0]["timer"])


class ModeOracle(_LegRunnerTestCase):
    def test_oracle_saves_one_entry_per_shape_and_draw(self):
        self._task_dir()
        self._target_module()
        dest = os.path.join(self.task, "leg_out.pt")
        out = json.loads(self._main("--task", self.task, "--mode", "oracle",
                                    "--out", dest, "--draws", "3", "--seed", "100"))
        self.assertEqual(out, {"out": dest, "n": 3})
        blob, path = self.torch.saved[-1]
        self.assertEqual(path, dest)
        self.assertEqual(sorted(blob), ["decode|0", "decode|1", "decode|2"])

    def test_draw_seeds_are_distinct_and_anchored_on_seed(self):
        """Same seed for every draw would make the parity check pass on one value draw only."""
        self._task_dir()
        self._target_module()
        self._main("--task", self.task, "--mode", "oracle",
                   "--out", os.path.join(self.task, "o.pt"), "--draws", "3", "--seed", "40")
        self.assertEqual([a["seed"] for a in sys.modules["cases"].seen], [40, 41, 42])

    def test_draw_count_defaults_to_meta_random_draws(self):
        self._task_dir(meta={"target_callable": "fake_live_stack:op", "regime": {},
                             "random_draws": 2})
        self._target_module()
        out = json.loads(self._main("--task", self.task, "--mode", "oracle",
                                    "--out", os.path.join(self.task, "o.pt")))
        self.assertEqual(out["n"], 2)

    def test_zero_draws_still_records_one(self):
        """max(1, draws) -- an oracle with no entries would make parity vacuously pass."""
        self._task_dir(meta={"target_callable": "fake_live_stack:op", "regime": {},
                             "random_draws": 0})
        self._target_module()
        out = json.loads(self._main("--task", self.task, "--mode", "oracle",
                                    "--out", os.path.join(self.task, "o.pt")))
        self.assertEqual(out["n"], 1)

    def test_generator_device_follows_cuda_availability(self):
        for available, want in ((True, "cuda"), (False, "cpu")):
            with self.subTest(cuda=available):
                self.torch.cuda_available = available
                self._task_dir()
                self._target_module()
                self._main("--task", self.task, "--mode", "oracle",
                           "--out", os.path.join(self.task, "o.pt"), "--draws", "1")
                self.assertEqual(sys.modules["cases"].seen[-1]["device"], want)


class OracleSnapshot(_LegRunnerTestCase):
    """What the oracle can record.

    `out.detach()` on a bare return value assumes every op returns ONE tensor. Attention entries
    routinely return `(out, lse)`; that raised AttributeError inside the leg and reached the caller
    through `_run_leg` as an anonymous non-zero-exit RuntimeError, which names neither the op nor the
    real problem."""

    class _T:
        """Duck-typed tensor: records that the full detach/clone/cpu chain ran."""
        def __init__(self, tag):
            self.tag = tag
            self.detached = False
        def detach(self):
            self.detached = True
            return self
        def clone(self):
            return self
        def cpu(self):
            return self

    def test_a_bare_tensor_is_detached_cloned_and_moved_to_host(self):
        t = self._T("out")
        got = lr._snapshot(t)
        self.assertIs(got, t)
        self.assertTrue(t.detached)

    def test_a_tuple_return_is_recorded_elementwise(self):
        """The #416 review case: attention returning (out, lse)."""
        out, lse = self._T("out"), self._T("lse")
        got = lr._snapshot((out, lse))
        self.assertEqual([x.tag for x in got], ["out", "lse"])
        self.assertTrue(all(x.detached for x in got))

    def test_a_list_return_keeps_its_type(self):
        got = lr._snapshot([self._T("a")])
        self.assertIsInstance(got, list)

    def test_a_namedtuple_return_keeps_its_fields(self):
        """type(out)(genexpr) would build a namedtuple with ONE positional arg and raise."""
        Attn = collections.namedtuple("Attn", "out lse")
        got = lr._snapshot(Attn(self._T("out"), self._T("lse")))
        self.assertIsInstance(got, Attn)
        self.assertEqual(got.lse.tag, "lse")

    def test_a_dict_return_is_recorded_by_key(self):
        got = lr._snapshot({"out": self._T("out"), "lse": self._T("lse")})
        self.assertEqual(sorted(got), ["lse", "out"])

    def test_nesting_is_recorded_all_the_way_down(self):
        got = lr._snapshot({"pair": [(self._T("deep"),)]})
        self.assertTrue(got["pair"][0][0].detached)

    def test_a_scalar_alongside_a_tensor_survives(self):
        """Some entries return (out, num_tokens); dropping the int would corrupt the blob."""
        self.assertEqual(lr._snapshot((self._T("out"), 7))[1], 7)

    def test_an_unrecordable_return_names_the_type_it_got(self):
        with self.assertRaises(TypeError) as cm:
            lr._snapshot(object())
        self.assertIn("object", str(cm.exception))
        self.assertIn("cases.call", str(cm.exception))

    def test_the_oracle_mode_records_a_tuple_returning_op_end_to_end(self):
        self._task_dir(cases=_CASES_STUB.replace("return _Out()", "return (_Out(), _Out())"))
        self._target_module()
        out = json.loads(self._main("--task", self.task, "--mode", "oracle",
                                    "--out", os.path.join(self.task, "o.pt"), "--draws", "1"))
        self.assertEqual(out["n"], 1)
        blob, _ = self.torch.saved[-1]
        self.assertEqual(len(blob["decode|0"]), 2)


class Cli(_LegRunnerTestCase):
    def test_unknown_mode_is_rejected_by_argparse(self):
        self._task_dir()
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self._main("--task", self.task, "--mode", "benchmark")

    def test_task_is_required(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self._main("--mode", "list")


if __name__ == "__main__":
    unittest.main()
