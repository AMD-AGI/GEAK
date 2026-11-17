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
import copy
import importlib
import importlib.util
import io
import json
import math
import os
import pickle
import shutil
import sys
import tempfile
import textwrap
import types
import unittest
from unittest import mock

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
    def test_an_additional_phase_entry_can_be_resolved(self):
        self._write("meta.json", json.dumps({"target_callable": "absent_primary:op"}))
        self._target_module(name="prefill_stack").op.__module__ = "prefill_stack"
        ident = json.loads(self._main("--task", self.task, "--mode", "resolve",
                                      "--target-callable", "prefill_stack:op"))
        self.assertNotIn("error", ident)
        self.assertTrue(ident["file"].endswith("prefill_stack.py"))

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
    def test_phase_graph_and_compile_modes_are_independent(self):
        self._task_dir(meta={"target_callable": "fake_live_stack:op",
                             "regime": {"compile_mode": "wrong_global", "graph_mode": True},
                             "phase_regimes": {
                                 "decode": {"graph_mode": True},
                                 "prefill": {"graph_mode": False, "compile_mode": "max-autotune"},
                             }})
        self._target_module()
        self._main("--task", self.task, "--mode", "time")
        calls = sys.modules["harness_lib"].calls
        self.assertEqual([c[1] for c in calls if c[0] == "time_op"], [True, False])
        self.assertEqual([c for c in calls if c[0] == "compiled_op"],
                         [("compiled_op", "max-autotune")])

    def test_missing_phase_mode_does_not_fall_back_to_global(self):
        self._task_dir(meta={"target_callable": "fake_live_stack:op",
                             "phase_regimes": {"decode": {"graph_mode": True}}})
        self._target_module()
        with self.assertRaisesRegex(ValueError, "missing phase_regimes entry"):
            self._main("--task", self.task, "--mode", "time", "--bucket", "prefill")

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


# A harness stub whose time_op records every keyword it was given and answers like a spun eager timing
# when `spin` is passed (A16).
_HARNESS_KW_STUB = _HARNESS_STUB.replace(
    'def time_op(fn, graph=False, detail=False):\n'
    '    calls.append(("time_op", graph, detail))\n'
    '    fn()\n'
    '    return {"ms": 1.25, "wall_ms": 1.5, "timer": "event"}\n',
    'def time_op(fn, graph=False, **kw):\n'
    '    calls.append(("time_op", graph, dict(kw)))\n'
    '    fn()\n'
    '    r = {"ms": 1.25, "wall_ms": 1.5, "timer": "cuda_event_graph" if graph else "cuda_event",\n'
    '         "primed": True, "host_ms": 0.5}\n'
    '    if kw.get("spin") is not None:\n'
    '        r.update(spin={"queued": "gpu_spin_before_start", "us": 200.0, "samples": 3, "late": 0,\n'
    '                       "retried": False, "includes_dispatch": False}, latency_ms=1.75)\n'
    '    return r\n')
assert _HARNESS_KW_STUB != _HARNESS_STUB

_PHASED = {"target_callable": "fake_live_stack:op",
           "phase_regimes": {"decode": {"graph_mode": True}, "prefill": {"graph_mode": False}}}


class ModeTimeSpin(_LegRunnerTestCase):
    """A16: in a GENERATED task an eager-deployed phase is timed on a queued GPU spin; graph phases and
    hand-written (old) tasks keep the original call."""

    def _generated(self):
        self._write("unittest.py", lr.GENERATED_MARKER + " (generate_ut.py v1): do not edit\n")

    def _time_calls(self):
        return [c for c in sys.modules["harness_lib"].calls if c[0] == "time_op"]

    def test_only_the_eager_phase_of_a_generated_task_gets_the_spin(self):
        self._task_dir(meta=dict(_PHASED), harness=_HARNESS_KW_STUB)
        self._generated()
        self._target_module()
        out = json.loads(self._main("--task", self.task, "--mode", "time"))
        (dec_graph, dec_kw), (pre_graph, pre_kw) = [(c[1], c[2]) for c in self._time_calls()]
        self.assertEqual((dec_graph, pre_graph), (True, False))
        self.assertNotIn("spin", dec_kw)
        self.assertEqual(pre_kw["spin"], {})                       # the defaults
        rows = {c["sig"]: c for c in out["cases"]}
        self.assertNotIn("spin", rows["decode"])
        self.assertEqual(rows["prefill"]["spin"]["queued"], "gpu_spin_before_start")
        self.assertEqual(rows["prefill"]["latency_ms"], 1.75)

    def test_meta_timing_tunes_the_spin_and_ignores_non_numbers(self):
        meta = dict(_PHASED, timing={"eager_spin_us": 500, "eager_spin_factor": 3, "eager_spin_min_us": "big",
                                     "prefill": "eager event timing (free text)"})
        self._task_dir(meta=meta, harness=_HARNESS_KW_STUB)
        self._generated()
        self._target_module()
        self._main("--task", self.task, "--mode", "time", "--bucket", "prefill")
        (call,) = self._time_calls()
        self.assertEqual(call[2]["spin"], {"us": 500.0, "factor": 3.0})

    def test_a_hand_written_task_is_timed_exactly_as_before(self):
        self._task_dir(meta=dict(_PHASED), harness=_HARNESS_KW_STUB)
        self._write("unittest.py", "# hand-written UT\n")
        self._target_module()
        self._main("--task", self.task, "--mode", "time")
        self.assertEqual([c[2] for c in self._time_calls()], [{"detail": True}, {"detail": True}])

    def test_eager_spin_reads_only_positive_numbers(self):
        self.assertEqual(lr.eager_spin({}), {})
        self.assertEqual(lr.eager_spin({"timing": "text"}), {})
        self.assertEqual(lr.eager_spin({"timing": {"eager_spin_us": 0, "eager_spin_late_limit": 0.1,
                                                   "eager_spin_min_us": True}}), {"late_limit": 0.1})


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


class NativeNoiseRel(unittest.TestCase):
    """A37: the refs leg records noise_rel for every random-draw comparison it makes, through the unchanged
    h.correct, and native_noise carries the worst finite one."""

    def test_every_random_comparison_is_recorded_and_correct_is_restored(self):
        seen = []
        h = types.SimpleNamespace(noise_rel=lambda out, ref: abs(out - ref))

        def correct(out, ref, tol):
            seen.append(tol)
            return abs(out - ref) <= tol, 0.0
        h.correct = correct

        def random_check(pairs, tol):              # stands for check_random_vs_baseline: calls h.correct per draw
            return all(h.correct(o, r, tol)[0] for o, r in pairs), []
        h.check_random_vs_baseline = random_check
        noise = []
        ok, _ = lr._recording_noise(h, noise)([(1.0, 1.25), (2.0, 2.0)], 0.5)
        self.assertEqual((ok, noise, seen), (True, [0.25, 0.0], [0.5, 0.5]))
        self.assertIs(h.correct, correct)           # restored
        h2 = types.SimpleNamespace(correct=correct, check_random_vs_baseline=random_check)   # an older harness
        self.assertEqual(lr._recording_noise(h2, noise)([(1.0, 1.0)], 0.5)[0], True)
        self.assertEqual(noise, [0.25, 0.0])

    def test_native_noise_keeps_the_worst_finite_noise_rel_and_counts_the_rest(self):
        report = {"decode": {"random": [{"case": "random[0]:a", "max_rel_err": 0.4},
                                        {"case": "random[1]:a", "max_rel_err": None}]}}
        self.assertEqual(lr._native_noise(report, [0.01, 0.03, float("inf")]),
                         {"max_rel_err": 0.4, "cases": 1, "noise_rel": 0.03, "noise_rel_nonfinite": 1})
        self.assertEqual(lr._native_noise(report, []), {"max_rel_err": 0.4, "cases": 1, "noise_rel": None})
        self.assertEqual(lr._native_noise(report), {"max_rel_err": 0.4, "cases": 1})


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


# =========================================================================== #
# Per-output tolerances (A42), the native references (RefBook) and the verify_ut
# controls, on CPU: 1-D fake tensors and the real, torch-free harness_lib helpers
# (flatten_outputs / is_output_position / per_output_tolerances / _row_extra).
# =========================================================================== #
hl = _load("harness_lib_for_test_leg_runner", "harness_lib.py")      # not registered in sys.modules

_FLOAT_DTYPES = ("float16", "bfloat16", "float32", "float64")
_INT_DTYPES = {"uint8": (8, 0), "int8": (8, -(1 << 7)), "int32": (32, -(1 << 31)), "int64": (64, -(1 << 63))}
_DTYPES = _FLOAT_DTYPES + ("bool", "complex64") + tuple(_INT_DTYPES)


class _FT:
    """A 1-D fake tensor carrying exactly what leg_runner touches: shape, dtype, device, is_floating_point,
    is_complex, data_ptr, numel, detach/clone/to/cpu, abs/max, * + ^ ~ ==, all, copy_. As in torch, the truth value
    of a multi-element tensor raises, so no assertion can compare two of them by accident (compare _vals)."""

    def __init__(self, values, dtype="float32", device="cpu"):
        self.values = list(values)
        self.dtype = dtype
        self.device = device

    @property
    def shape(self):
        return (len(self.values),)

    def is_floating_point(self):
        return self.dtype in _FLOAT_DTYPES

    def is_complex(self):
        return self.dtype.startswith("complex")

    def data_ptr(self):
        return id(self)

    def numel(self):
        return len(self.values)

    def detach(self):
        return self

    def clone(self):
        return _FT(self.values, self.dtype, self.device)

    def cpu(self):
        return self.to("cpu")

    def to(self, where):
        if where in _DTYPES:
            cast = float if where in _FLOAT_DTYPES else {"bool": bool, "complex64": complex}.get(where, int)
            return _FT([cast(v) for v in self.values], where, self.device)
        return _FT(self.values, self.dtype, where)

    def abs(self):
        return _FT([abs(v) for v in self.values], self.dtype, self.device)

    def max(self):
        return _FT([max(self.values)], self.dtype, self.device)

    def __mul__(self, k):
        return _FT([v * k for v in self.values], self.dtype, self.device)

    def __add__(self, other):
        o = other.values if isinstance(other, _FT) else [other]
        o = o * len(self.values) if len(o) == 1 else o
        return _FT([a + b for a, b in zip(self.values, o)], self.dtype, self.device)

    def __invert__(self):
        return _FT([not v for v in self.values], self.dtype, self.device)

    def __xor__(self, k):
        # Python's ^ on ints is two's complement, so within the dtype's range it equals torch's fixed-width ^
        return _FT([v ^ k for v in self.values], self.dtype, self.device)

    def __eq__(self, other):
        if len(other.values) != len(self.values):
            raise RuntimeError("The size of tensor a must match the size of tensor b")
        return _FT([a == b for a, b in zip(self.values, other.values)], "bool", self.device)

    __hash__ = object.__hash__

    def __bool__(self):
        if len(self.values) != 1:
            raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")
        return bool(self.values[0])

    def all(self):
        return all(self.values)

    def copy_(self, src):
        self.values[:] = src.values
        return self


def _vals(x):
    """The values of a (nested) output of fake tensors, as plain lists / dicts, for assertEqual."""
    if isinstance(x, _FT):
        return x.values
    if isinstance(x, dict):
        return {k: _vals(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_vals(v) for v in x]
    return x


def _restore_module(name, prev):
    if prev is _MISSING:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = prev


def _keep_modules(test, *names):
    """Restore sys.modules[name] (or its absence) after `test`, whatever the test puts there."""
    for name in names:
        test.addCleanup(_restore_module, name, sys.modules.get(name, _MISSING))


def _install_dtype_torch(test):
    """sys.modules["torch"] = a fake carrying only what _off_by_much reads (torch.bool, torch.iinfo); restored
    after `test`."""
    _keep_modules(test, "torch")
    fake = types.ModuleType("torch")
    fake.bool = "bool"
    fake.iinfo = lambda dtype: types.SimpleNamespace(bits=_INT_DTYPES[dtype][0], min=_INT_DTYPES[dtype][1])
    sys.modules["torch"] = fake
    return fake


class TaskTolerance(unittest.TestCase):
    """task_tol / task_tolerance: the one plain float exactly as before, a TaskTol only with per_output (A42)."""

    def test_task_tol_reads_tolerance_tol_then_meta_tol(self):
        self.assertEqual(lr.task_tol({"tolerance": {"tol": 0.02}, "tol": 0.5}), 0.02)
        self.assertEqual(lr.task_tol({"tolerance": {"rtol": 1}, "tol": "0.05"}), 0.05)   # no tolerance.tol
        self.assertEqual(lr.task_tol({"tolerance": 0.3, "tol": 0.04}), 0.04)            # not a dict: ignored
        self.assertEqual(lr.task_tol({"tol": 0}), 0.0)

    def test_task_tol_rejects_a_tolerance_that_is_not_finite_and_non_negative(self):
        for bad in (float("nan"), -0.01, float("inf")):
            with self.subTest(tol=bad), self.assertRaisesRegex(ValueError, "not a finite non-negative number"):
                lr.task_tol({"tolerance": {"tol": bad}})

    def test_without_per_output_the_tolerance_is_a_plain_float(self):
        for per in (_MISSING, None, {}):
            tolerance = {"tol": 0.02} if per is _MISSING else {"tol": 0.02, "per_output": per}
            with self.subTest(per_output=per):
                tol = lr.task_tolerance({"tolerance": tolerance}, hl, native=True)
                self.assertIs(type(tol), float)
                self.assertEqual(tol, 0.02)
        # an older harness_lib without per_output_tolerances never yields a TaskTol
        tol = lr.task_tolerance({"tolerance": {"tol": 0.02, "per_output": {"1": 0.001}}}, types.SimpleNamespace())
        self.assertIs(type(tol), float)

    def test_with_per_output_the_tolerance_is_a_task_tol_of_the_leg(self):
        meta = {"tolerance": {"tol": 0.02, "per_output": {"1": 0.001}}}
        native = lr.task_tolerance(meta, hl, native=True)
        self.assertIsInstance(native, lr.TaskTol)
        self.assertEqual((float(native), native.per_output, native.native, native.problems),
                         (0.02, {"1": 0.001}, True, []))
        self.assertIs(native.h, hl)
        self.assertFalse(lr.task_tolerance(meta, hl).native)
        # a malformed per_output: a TaskTol carrying harness_lib's problem and no per-output tolerance
        bad = lr.task_tolerance({"tolerance": {"tol": 0.02, "per_output": {"1": -1}}}, hl)
        self.assertEqual((bad.per_output, bad.problems),
                         ({}, ["meta.tolerance.per_output['1'] must be a finite non-negative number, not -1"]))

    def test_a_task_tol_or_output_err_copies_and_pickles_as_a_plain_float(self):
        tol = lr.TaskTol(0.02, None, h=hl, problems=("earlier problem",))
        self.assertEqual((tol.per_output, tol.problems, tol.outputs, tol.noise), ({}, ["earlier problem"], None, {}))
        err = lr.OutputErr(1.5, "1", 0.001)
        self.assertEqual((float(err), err.row_fields), (1.5, {"output": "1", "tol": 0.001}))
        self.assertEqual(hl._row_extra(err), {"output": "1", "tol": 0.001})    # what every check copies to its row
        for value, want in ((tol, 0.02), (err, 1.5)):
            for twin in (copy.copy(value), copy.deepcopy(value), pickle.loads(pickle.dumps(value))):
                self.assertIs(type(twin), float)
                self.assertEqual(twin, want)


class TaskTolLayout(unittest.TestCase):
    """TaskTol.layout / fit: the outputs of a reference, and the problems of fitting per_output to it (A42)."""

    def setUp(self):
        self.a, self.b, self.c = _FT([1.0]), _FT([2.0]), _FT([3.0])

    def test_outputs_are_positions_or_dict_keys_in_flatten_outputs_order(self):
        a, b, c = self.a, self.b, self.c
        self.assertEqual(lr.TaskTol(0.02, {"1": 0.001}, h=hl).layout((a, [b])),
                         ([("0", 0.02), ("1", 0.001)], []))
        self.assertEqual(lr.TaskTol(0.02, {"mix": 0.001}, h=hl).layout({"mix": (a, b), "hid": c}),
                         ([("hid", 0.02), ("mix[0]", 0.001), ("mix[1]", 0.001)], []))

    def test_a_per_output_that_does_not_fit_names_the_problem(self):
        a, b = self.a, self.b
        cases = [
            ({"2": 0.001}, (a, b), "per_output names the positions ['2'], but the compared output has 2 tensor(s)"),
            ({"mix": 0.001}, (a, b), ("the compared output is not a dict but 2 tensor(s): name its outputs by "
                                      "position (\"0\" .. \"1\"), not by key ['mix']")),
            ({"mix": 0.001}, None, ("the compared output is not a dict but 0 tensor(s): name its outputs by "
                                    "position (\"0\" .. \"0\"), not by key ['mix']")),
            ({"0": 0.001}, {"mix": a, "hid": b}, ("the compared output is a dict (keys ['hid', 'mix']): name its "
                                                  "outputs by key, not by position ['0']")),
            ({"gate": 0.001}, {"mix": a, "hid": b}, ("per_output names ['gate'], which the compared dict (keys "
                                                     "['hid', 'mix']) does not have")),
        ]
        for per, out, problem in cases:
            with self.subTest(per=per, out=type(out).__name__):
                self.assertEqual(lr.TaskTol(0.02, per, h=hl).layout(out)[1], [problem])
        # the rows are still every output, at meta.tolerance.tol where per_output does not name it
        self.assertEqual(lr.TaskTol(0.02, {"0": 0.001}, h=hl).layout({"mix": a, "hid": b})[0],
                         [("hid", 0.02), ("mix", 0.02)])

    def test_fit_pins_the_first_layout_and_only_the_native_leg_blames_the_ut(self):
        a, b, c = self.a, self.b, self.c
        native = lr.TaskTol(0.02, {"0": 0.001}, h=hl, native=True)
        candidate = lr.TaskTol(0.02, {"0": 0.001}, h=hl)
        for t in (native, candidate):
            self.assertEqual(t.fit((a, b)), [("0", 0.001), ("1", 0.02)])
            self.assertEqual(t.fit([c, a]), [("0", 0.001), ("1", 0.02)])        # the same outputs again
            self.assertIsNone(t.fit((a, b, c)))                                   # other outputs
            self.assertIsNone(t.fit((a, b, c)))
            self.assertEqual(t.outputs, [("0", 0.001), ("1", 0.02)])
        self.assertEqual(native.problems, [(                                      # once, however often it recurs
            "meta.tolerance.per_output: one comparison compares the outputs ['0', '1', '2'], an earlier one "
            "['0', '1']: per-output tolerances need every comparison of the task to compare the same outputs")])
        self.assertEqual(candidate.problems, [])
        # a per_output that does not fit the very first reference pins nothing
        t = lr.TaskTol(0.02, {"5": 0.001}, h=hl, native=True)
        self.assertIsNone(t.fit((a, b)))
        self.assertIsNone(t.outputs)
        self.assertEqual(t.problems, [("meta.tolerance.per_output: per_output names the positions ['5'], but the "
                                       "compared output has 2 tensor(s)")])

    def test_noise_is_recorded_per_output_and_the_worst_finite_one_reported(self):
        t = lr.TaskTol(0.02, {"1": 0.001}, h=hl, native=True)
        self.assertEqual((t.noise_record(), t.record()["outputs"]), ({}, []))      # nothing fitted yet

        def measure(o, r):
            return o.values[0] - r.values[0]
        ref = (_FT([1.0]), _FT([2.0]))
        t.record_noise((_FT([1.25]), _FT([math.inf])), ref, measure)
        t.record_noise((_FT([1.5]), _FT([-math.inf])), ref, measure)
        t.record_noise((_FT([9.0]),), ref, measure)                                 # a count that differs
        t.record_noise((_FT([9.0]),) * 3, ref + (_FT([3.0]),), measure)             # other outputs
        self.assertEqual(t.noise, {"0": [0.25, 0.5], "1": [math.inf, -math.inf]})
        self.assertEqual(t.noise_record(), {"0": {"tol": 0.02, "noise_rel": 0.5},
                                            "1": {"tol": 0.001, "noise_rel": None, "noise_rel_nonfinite": 2}})
        self.assertEqual(t.record(), {"tol": 0.02, "per_output": {"1": 0.001},
                                      "outputs": [{"output": "0", "tol": 0.02}, {"output": "1", "tol": 0.001}]})


class PerOutputCorrect(unittest.TestCase):
    """per_output_correct (A42): a comparison at the task's TaskTol is made output by output with correct() at each
    output's own tolerance; its err is that of the output furthest over its own tolerance (a failing one if any)."""

    def setUp(self):
        self.calls, self.verdicts = [], {}

    def correct(self, out, ref, tol):
        self.calls.append((_vals(out), _vals(ref), tol))
        return self.verdicts[id(out)]

    def compare(self, verdicts, per, tol=0.02):
        """Two outputs [1.0], [2.0], for which correct() answers verdicts[0] / verdicts[1]."""
        outs = (_FT([1.0]), _FT([2.0]))
        self.verdicts = {id(o): v for o, v in zip(outs, verdicts)}
        ok, err = lr.per_output_correct(hl, self.correct)(outs, tuple(o.clone() for o in outs),
                                                          lr.TaskTol(tol, per, h=hl))
        return ok, float(err), hl._row_extra(err)

    def test_a_plain_tolerance_is_correct_itself_on_the_whole_output(self):
        out, ref, answer = (_FT([1.0]), _FT([2.0])), (_FT([1.0]), _FT([2.0])), (True, 0.125)
        self.verdicts = {id(out): answer}
        self.assertIs(lr.per_output_correct(hl, self.correct)(out, ref, 0.02), answer)
        self.assertEqual(self.calls, [([[1.0], [2.0]], [[1.0], [2.0]], 0.02)])

    def test_every_output_is_compared_at_its_own_tolerance(self):
        self.assertEqual(self.compare([(True, 0.01), (True, 0.0008)], {"1": 0.001}),
                         (True, 0.0008, {"output": "1", "tol": 0.001}))           # 0.8 of its tolerance > 0.5
        self.assertEqual([tol for _, _, tol in self.calls], [0.02, 0.001])

    def test_a_passing_output_reading_nan_at_tolerance_0_reads_0(self):
        """correct() reads 0/0 = NaN on an exact match at tolerance 0: the row keeps a finite 0.0."""
        self.assertEqual(self.compare([(True, math.nan), (True, 0.0)], {"0": 0.0}),
                         (True, 0.0, {"output": "0", "tol": 0.0}))

    def test_a_failing_comparison_names_the_failing_output_furthest_over_its_tolerance(self):
        # output 0 passes 160x over its 0.005 (a reference element near 0); output 1 fails 1.5x over its 0.02
        self.assertEqual(self.compare([(True, 0.8), (False, 0.03)], {"0": 0.005}),
                         (False, 0.03, {"output": "1", "tol": 0.02}))
        # two failing outputs: the one further over its own tolerance, not the larger err
        self.assertEqual(self.compare([(False, 0.01), (False, 0.03)], {"0": 0.005}),
                         (False, 0.01, {"output": "0", "tol": 0.005}))
        self.assertEqual(self.compare([(False, 0.01), (False, math.inf)], {"0": 0.005}),
                         (False, math.inf, {"output": "1", "tol": 0.02}))

    def test_nothing_to_compare_or_outputs_that_do_not_pair_up_fail(self):
        a, b = _FT([1.0]), _FT([2.0])
        compare = lr.per_output_correct(hl, self.correct)
        for out, ref, per in (((a, b), None, {"1": 0.001}),           # no reference
                              (None, (a, b), {"1": 0.001}),           # nothing compared
                              ((a,), (a, b), {"1": 0.001}),           # a tensor count that differs
                              ((a, b), (a, b), {"mix": 0.001})):      # per_output does not fit (candidate leg)
            with self.subTest(out=_vals(out), ref=_vals(ref), per=per):
                self.assertEqual(compare(out, ref, lr.TaskTol(0.02, per, h=hl)), (False, math.inf))
        self.assertEqual(self.calls, [])

    def test_tol_share(self):
        self.assertEqual([lr._tol_share(e, t) for e, t in ((math.nan, 0.1), (math.inf, 0.1), (0.05, 0.1),
                                                           (0.3, 0.0), (0.0, 0.0))],
                         [math.inf, math.inf, 0.5, math.inf, 0.0])


class NativeRefBook(unittest.TestCase):
    """RefBook: the native refs leg records every compared output; the candidate leg is judged against them."""

    def setUp(self):
        self.calls = []

        def correct(out, ref, tol):
            self.calls.append((_vals(out), _vals(ref), tol, getattr(ref, "device", None)))
            ok = _vals(out) == _vals(ref)
            return ok, (0.0 if ok else 1.0)
        self.orig = correct
        self.h = types.SimpleNamespace(flatten_outputs=hl.flatten_outputs, correct=correct)

    def check(self, pairs, tol):
        """A harness check: h.correct once per (output, case ref)."""
        return [self.h.correct(o, r, tol) for o, r in pairs]

    def test_the_native_leg_records_every_compared_output_and_checks_its_own_ref(self):
        book = lr.RefBook(self.h)
        o1, o2 = _FT([1.0, 2.0], device="cuda:0"), _FT([3.0], device="cuda:0")
        res = book.wrap("eager", self.check)([(o1, _FT([1.0, 2.0])), (o2, None)], 0.02)
        self.assertEqual(res, [(True, 0.0), (True, 0.0)])
        self.assertEqual(self.calls, [([1.0, 2.0], [1.0, 2.0], 0.02, "cpu")])   # only the case carrying a ref
        self.assertIs(self.h.correct, self.orig)
        rec = book.dump()
        self.assertEqual(sorted(rec), ["eager", "graph_replay", "sequence"])
        self.assertEqual([(t.values, t.device) for t in rec["eager"]], [([1.0, 2.0], "cpu"), ([3.0], "cpu")])
        o1.values[0] = 99.0
        self.assertEqual(rec["eager"][0].values, [1.0, 2.0])                       # a snapshot of the output
        self.assertEqual(book.mismatches(), [])

    def test_per_output_must_fit_a_native_output_even_where_nothing_is_compared(self):
        tol = lr.TaskTol(0.02, {"2": 0.001}, h=hl, native=True)
        res = lr.RefBook(self.h).wrap("sequence", self.check)([((_FT([1.0]), _FT([2.0])), None)], tol)
        self.assertEqual(res, [(True, 0.0)])
        self.assertEqual(tol.problems, [("meta.tolerance.per_output: per_output names the positions ['2'], but the "
                                         "compared output has 2 tensor(s)")])

    def test_the_candidate_leg_is_judged_against_the_nth_native_output_on_its_device(self):
        native = lr.RefBook(self.h)
        native.wrap("graph_replay", self.check)([(_FT([1.0]), None), (_FT([2.0]), None)], 0.02)
        book = lr.RefBook(self.h, {"graph_replay": native.dump()["graph_replay"], "sequence": [_FT([5.0])] * 2})
        self.assertEqual(book.dump()["eager"], [])                    # a kind the native leg did not record
        cand = [_FT([1.0], device="cuda:1"), _FT([2.5], device="cuda:1"), _FT([3.0], device="cuda:1")]
        res = book.wrap("graph_replay", self.check)([(c, _FT([7.0])) for c in cand], 0.02)    # the case's ref: unused
        self.assertEqual(res, [(True, 0.0), (False, 1.0), (False, math.inf)])                 # 3rd: nothing recorded
        self.assertEqual(self.calls, [([1.0], [1.0], 0.02, "cuda:1"), ([2.5], [2.0], 0.02, "cuda:1")])
        self.assertEqual(book.mismatches(), [
            "graph_replay: candidate made 3 comparisons, the native leg recorded 2",
            "sequence: candidate made 0 comparisons, the native leg recorded 2"])

    def test_a_candidate_output_without_a_device_is_compared_on_the_cpu(self):
        book = lr.RefBook(self.h, {"eager": [_FT([1.0], device="meta")]})
        deviceless = types.SimpleNamespace(shape=(1,))
        book.wrap("eager", self.check)([(deviceless, None)], 0.02)
        self.assertEqual(self.calls[0][3], "cpu")
        self.assertEqual(book.mismatches(), [])

    def test_h_correct_is_restored_when_the_check_raises(self):
        def check():
            raise RuntimeError("check blew up")
        for book in (lr.RefBook(self.h), lr.RefBook(self.h, {})):
            with self.assertRaisesRegex(RuntimeError, "check blew up"):
                book.wrap("eager", check)()
            self.assertIs(self.h.correct, self.orig)

    def test_to_device_moves_every_tensor_of_a_container(self):
        moved = lr._to_device({"a": (_FT([1.0]), [_FT([2.0]), 3]), "b": None}, "cuda:2")
        self.assertIsInstance(moved["a"], tuple)
        self.assertIsInstance(moved["a"][1], list)
        self.assertEqual((moved["a"][0].device, moved["a"][1][0].device, moved["a"][1][1], moved["b"]),
                         ("cuda:2", "cuda:2", 3, None))
        self.assertEqual(_vals(moved), {"a": [[1.0], [[2.0], 3]], "b": None})

    def test_a_namedtuple_native_output_is_judged_on_the_candidate_device(self):
        """An op returning a namedtuple (an attention's (out, lse)) is recorded as one by _snapshot; the candidate
        leg rebuilds it field by field on its own device -- type(obj)(genexpr) would raise TypeError there, so a
        correct kernel could never pass its native-reference comparison."""
        Attn = collections.namedtuple("Attn", "out lse")
        moved = lr._to_device(Attn(_FT([1.0]), 3), "cuda:2")
        self.assertIsInstance(moved, Attn)
        self.assertEqual((moved.out.device, moved.lse), ("cuda:2", 3))
        native = lr.RefBook(self.h)
        native.wrap("eager", self.check)(
            [(Attn(_FT([1.0], device="cuda:0"), _FT([2.0], device="cuda:0")), None)] * 2, 0.02)
        recorded = native.dump()["eager"]
        self.assertEqual([type(r) for r in recorded], [Attn, Attn])
        refs = []

        def correct(out, ref, tol):
            refs.append(ref)
            ok = _vals(out) == _vals(ref)
            return ok, (0.0 if ok else 1.0)
        self.h.correct = correct
        book = lr.RefBook(self.h, {"eager": recorded})
        cand = [Attn(_FT([1.0], device="cuda:1"), _FT([2.0], device="cuda:1")),
                Attn(_FT([1.0], device="cuda:1"), _FT([2.5], device="cuda:1"))]
        res = book.wrap("eager", self.check)([(c, None) for c in cand], 0.02)
        self.assertEqual(res, [(True, 0.0), (False, 1.0)])          # judged against the native (out, lse)
        self.assertEqual([type(r) for r in refs], [Attn, Attn])
        self.assertEqual([(r.out.device, r.lse.device) for r in refs], [("cuda:1", "cuda:1")] * 2)
        self.assertEqual(_vals(refs[1]), [[1.0], [2.0]])
        self.assertEqual(book.mismatches(), [])


class ControlHelpers(unittest.TestCase):
    """_map / _map_flat / _off_by_much / _tensor_args: what the verify_ut controls apply to."""

    def test_map_applies_to_floats_and_optionally_the_rest_keeping_the_structure(self):
        Pair = collections.namedtuple("Pair", "out lse")
        f, i = _FT([1.0, 2.0]), _FT([3, 4], dtype="int64")
        out = {"p": Pair(f, i), "l": [f, 7, None], "t": (i,)}
        doubled = lr._map(out, lambda t: t * 2.0)
        self.assertIsInstance(doubled["p"], Pair)
        self.assertIsInstance(doubled["l"], list)
        self.assertIsInstance(doubled["t"], tuple)
        self.assertEqual(_vals(doubled), {"p": [[2.0, 4.0], [3, 4]], "l": [[2.0, 4.0], 7, None], "t": [[3, 4]]})
        self.assertIs(doubled["p"].lse, i)                       # no f_other: a non-float output is untouched
        self.assertEqual(_vals(lr._map(out, lambda t: t, lambda t: t * -1)["t"]), [[-3, -4]])

    def test_map_flat_numbers_the_tensors_in_flatten_outputs_order(self):
        Pair = collections.namedtuple("Pair", "first second")
        a, b, c = _FT([1.0]), _FT([2.0]), _FT([3.0])
        shaped = types.SimpleNamespace(shape=(1,))         # a tensor without is_floating_point: counted, untouched
        seen = []

        def f(t, i):
            seen.append((t.values[0], i))
            return t * 10.0
        got = lr._map_flat({"mix": Pair(shaped, a), "hid": [b, 5], "aux": (c,)}, f)
        self.assertEqual(seen, [(3.0, 0), (2.0, 1), (1.0, 3)])      # by sorted key; `shaped` took position 2
        self.assertEqual(list(got), ["mix", "hid", "aux"])          # the output's own key order
        self.assertIsInstance(got["mix"], Pair)
        self.assertIs(got["mix"].first, shaped)
        self.assertEqual(_vals(got["hid"]), [[20.0], 5])
        self.assertEqual(_vals(got["aux"]), [[30.0]])

    def test_off_by_much_moves_a_non_float_output_off_its_values(self):
        _install_dtype_torch(self)
        self.assertEqual(lr._off_by_much(_FT([True, False], dtype="bool")).values, [False, True])
        moved = lr._off_by_much(_FT([3, -5, 0], dtype="int64"))
        self.assertEqual((moved.dtype, moved.values), ("int64", [3 - 2**63, -5 + 2**63, -2**63]))   # top bit flipped
        # a uint8 output holding 255 (packed MXFP4, e8m0 scales): max + 1 would wrap to 0 and move nothing
        self.assertEqual(lr._off_by_much(_FT([0, 255, 128, 3], dtype="uint8")).values, [128, 127, 0, 131])
        self.assertEqual(lr._off_by_much(_FT([-128, 127, 0], dtype="int8")).values, [0, -1, -128])
        self.assertEqual(lr._off_by_much(_FT([1 + 1j, -2j], dtype="complex64")).values, [4 + 1j, 3 - 2j])
        empty = _FT([], dtype="int64")
        self.assertIs(lr._off_by_much(empty), empty)

    def test_tensor_args_finds_every_tensor_argument_once(self):
        a, b, c, d, e, hidden, cls_attr = (_FT([float(k)]) for k in range(7))
        holder = types.SimpleNamespace(state=d, n=3, nested=[hidden])           # one attribute level only

        class Holder:
            weight = cls_attr                                                   # a class is not an argument
        found = lr._tensor_args((a, [b, a], {"k": c}, holder, Holder, "text", 4), {"kw": e})
        self.assertEqual([id(t) for t in found], [id(a), id(b), id(c), id(d), id(e)])
        self.assertEqual(lr._tensor_args([], {}), [])


class Controls(unittest.TestCase):
    """The generic verify_ut controls (Control) around cases.call and the replay bundle's read_out."""

    def setUp(self):
        _install_dtype_torch(self)          # scale_out moves non-float outputs with _off_by_much

    def test_noop_and_rank_skew_hand_back_cases_call_and_the_replay_bundle(self):
        def call(args):
            return args
        replay = {"read_out": call}
        for name in ("noop", "rank_skew"):
            ctl = lr.Control(name, 0.02)
            self.assertIs(ctl.wrap_call(call), call)
            self.assertIs(ctl.wrap_replay(replay), replay)
            self.assertIsNone(ctl.outputs)
        self.assertEqual(lr.Control("noop", 0.02)._perturb("output", ("call", ())), "output")

    def test_within_tol_and_scale_out_move_every_float_output_by_the_one_tolerance(self):
        def call(args):
            x, n = args
            return (x * 1.0, n.clone())
        x, n = _FT([2.0, -4.0]), _FT([3, -5], dtype="int64")
        w = lr.Control("within_tol", 0.02).wrap_call(call)((x, n))
        self.assertEqual(_vals(w), [[2.0 * (1.0 + 0.02 / 10.0), -4.0 * (1.0 + 0.02 / 10.0)], [3, -5]])
        s = lr.Control("scale_out", 0.02).wrap_call(call)((x, n))
        self.assertEqual(_vals(s), [[2.0 * (1.0 + 4.0 * 0.02), -4.0 * (1.0 + 4.0 * 0.02)], [3 - 2**63, -5 + 2**63]])
        self.assertEqual(_vals((x, n)), [[2.0, -4.0], [3, -5]])        # the inputs are untouched

    def test_with_per_output_tolerances_each_output_moves_by_its_own(self):
        tol = lr.TaskTol(0.02, {"1": 0.001}, h=hl)
        f0, f1, n = _FT([1.0]), _FT([2.0]), _FT([3], dtype="int64")
        ctl = lr.Control("within_tol", tol)
        self.assertEqual((type(ctl.tol), ctl.tol), (float, 0.02))
        self.assertIs(ctl.outputs, tol)
        self.assertEqual(_vals(ctl._perturb(((f0, f1), n), None)),
                         [[[1.0 * (1.0 + 0.02 / 10.0)], [2.0 * (1.0 + 0.001 / 10.0)]], [3]])
        self.assertEqual(_vals(lr.Control("scale_out", tol)._perturb(((f0, f1), n), None)),
                         [[[1.0 * (1.0 + 4.0 * 0.02)], [2.0 * (1.0 + 4.0 * 0.001)]], [3 - 2**63]])
        # a dict by key, its keys kept in the output's order
        w = lr.Control("within_tol", lr.TaskTol(0.02, {"mix": 0.001}, h=hl))._perturb({"mix": f0, "hid": f1}, None)
        self.assertEqual(list(w), ["mix", "hid"])
        self.assertEqual(_vals(w), {"mix": [1.0 * (1.0 + 0.001 / 10.0)], "hid": [2.0 * (1.0 + 0.02 / 10.0)]})
        # a per_output that does not fit the output: every output by meta.tolerance.tol
        w = lr.Control("within_tol", lr.TaskTol(0.02, {"mix": 0.001}, h=hl))._perturb((f0, f1), None)
        self.assertEqual(_vals(w), [[1.0 * (1.0 + 0.02 / 10.0)], [2.0 * (1.0 + 0.02 / 10.0)]])

    def test_stale_output_answers_every_call_with_the_first_output_of_its_input_shapes(self):
        def call(args):
            return {"y": args["x"] * 2.0, "n": _FT([len(args["x"].values)], dtype="int64")}
        stale = lr.Control("stale_output", 0.02).wrap_call(call)
        first = stale({"x": _FT([1.0, 2.0])})
        again = stale({"x": _FT([5.0, 6.0])})                # the same shape: the first output again
        other = stale({"x": _FT([7.0])})                      # another shape: its own first output
        self.assertEqual([_vals(r) for r in (first, again, other)],
                         [{"y": [2.0, 4.0], "n": [2]}, {"y": [2.0, 4.0], "n": [2]}, {"y": [14.0], "n": [1]}])
        again["y"].values[0] = -1.0                          # every answer is a fresh copy
        self.assertEqual(_vals(stale({"x": _FT([0.0, 0.0])})["y"]), [2.0, 4.0])

    def test_the_replay_bundle_read_out_is_perturbed_and_the_rest_kept(self):
        buf = _FT([1.0, 2.0])

        def read_out():
            return buf.clone()
        replay = {"fill": "FILL", "run": "RUN", "read_out": read_out}
        w = lr.Control("within_tol", 0.02).wrap_replay(replay)
        self.assertIs(replay["read_out"], read_out)                     # the bundle itself is not changed
        self.assertEqual((w["fill"], w["run"]), ("FILL", "RUN"))
        self.assertEqual(_vals(w["read_out"]()), [1.0 * (1.0 + 0.02 / 10.0), 2.0 * (1.0 + 0.02 / 10.0)])
        stale = lr.Control("stale_output", 0.02).wrap_replay(replay)
        self.assertEqual(_vals(stale["read_out"]()), [1.0, 2.0])
        buf.values[:] = [8.0, 9.0]
        self.assertEqual(_vals(stale["read_out"]()), [1.0, 2.0])       # every replay reads the first output
        self.assertEqual(_vals(read_out()), [8.0, 9.0])
        for name, rp in (("within_tol", None), ("scale_out", {}), ("skip_state_write", replay)):
            self.assertIs(lr.Control(name, 0.02).wrap_replay(rp), rp)  # no bundle / no carry_state: as is

    def test_skip_state_write_drops_the_in_place_writes_of_the_call(self):
        state, x = _FT([0.0, 0.0]), _FT([1.0, 2.0])

        def call(args):
            args["state"].values[:] = [v + 1.0 for v in args["x"].values]      # an in-place state write
            return args["state"]                                              # the written buffer itself
        out = lr.Control("skip_state_write", 0.02).wrap_call(call)({"state": state, "x": x})
        self.assertIsNot(out, state)
        self.assertEqual(_vals(out), [2.0, 3.0])            # the output was read before the write was dropped
        self.assertEqual((_vals(state), _vals(x)), ([0.0, 0.0], [1.0, 2.0]))

    def test_skip_state_write_goes_on_past_an_argument_it_cannot_compare(self):
        grown, state = _FT([1.0]), _FT([0.0])

        def call(args):
            args[0].values.append(5.0)                      # resized in place: == against the copy raises
            args[1].values[0] = 7.0
            return args[1] * 1.0
        out = lr.Control("skip_state_write", 0.02).wrap_call(call)([grown, state])
        self.assertEqual((_vals(out), _vals(state), _vals(grown)), ([7.0], [0.0], [1.0, 5.0]))

    def test_skip_state_write_on_a_carry_state_group_restores_the_state_after_every_replay(self):
        acc, static_in, static_out = _FT([0.0]), _FT([0.0]), _FT([0.0])

        def fill(case):
            static_in.values[:] = case["x"]

        def run():                                          # the graph: a state write and the output
            acc.values[0] += static_in.values[0]
            static_out.values[0] = acc.values[0]

        def read_out():
            return static_out
        group = {"carry_state": True, "state": lambda: [acc], "fill": fill, "run": run, "read_out": read_out}
        g = lr.Control("skip_state_write", 0.02).wrap_replay(group)
        self.assertIs(g["run"], run)
        reads = []
        for x in ([1.0], [2.0]):
            g["fill"]({"x": x})
            g["run"]()
            reads.append(_vals(g["read_out"]()))
        self.assertEqual(reads, [[1.0], [2.0]])             # each step reads the stale state (carried: 1.0, 3.0)
        self.assertEqual(_vals(acc), [0.0])
        g["run"]()
        self.assertEqual(_vals(g["read_out"]()), [2.0])     # a read with no fill before it restores nothing
        self.assertEqual(_vals(acc), [2.0])


class TaskControl(unittest.TestCase):
    """A task control (meta.controls.task[]): `file` + `attr`, attr(call, h, meta, ctx) -> call."""

    def setUp(self):
        self.task = tempfile.mkdtemp(prefix="leg_runner_ctl_")
        self.addCleanup(shutil.rmtree, self.task, True)
        _keep_modules(self, "_geak_ut_control_permute", "_geak_ut_control_tagged")
        with open(os.path.join(self.task, "ctl.py"), "w") as fh:
            fh.write(textwrap.dedent('''
                def wrap(call, h, meta, ctx):
                    return ("wrapped", call, h, meta, ctx)

                def reverse(call, h, meta, ctx):
                    return lambda args: call(args[::-1])
            '''))

    def test_the_named_attr_of_the_named_file_wraps_cases_call(self):
        h, meta, ctx = object(), {"target_callable": "x:y"}, object()
        ctl = lr.Control("permute", 0.02, task=self.task, spec={"file": "ctl.py", "attr": "reverse"},
                         h=h, meta=meta, ctx=ctx)
        self.assertEqual(ctl.wrap_call(lambda args: list(args))([1, 2, 3]), [3, 2, 1])
        replay = {"read_out": list}
        self.assertIs(ctl.wrap_replay(replay), replay)          # a task control acts on cases.call only

        def call(args):
            return args
        tagged = lr.Control("tagged", 0.02, task=self.task, spec={"file": "ctl.py"}, h=h, meta=meta, ctx=ctx)
        self.assertEqual(tagged.wrap_call(call), ("wrapped", call, h, meta, ctx))     # attr defaults to `wrap`


# A harness stub whose time_op also answers every sample when asked (--raw-samples).
_HARNESS_SAMPLES_STUB = _HARNESS_KW_STUB.replace(
    '    return r\n', '    if kw.get("samples"):\n        r["samples"] = [1.25, 1.5]\n    return r\n')
assert _HARNESS_SAMPLES_STUB != _HARNESS_KW_STUB

# cases.py of a generated task: one case raises `inner`, and graph capture enters capture_context.
_CASES_CAPTURE_STUB = textwrap.dedent('''
    import contextlib

    def timing_cases(h, meta):
        return [
            {"sig": "decode", "args": {"m": 1}, "regime": "decode", "m": 1, "inner": 4},
            {"sig": "prefill", "args": {"m": 4096}, "regime": "prefill", "m": 4096},
        ]

    def capture_context():
        return contextlib.nullcontext()

    def call(args):
        return None
''')

# cases.py whose seeded factory takes the rank context: make_inputs(rng, ctx).
_CASES_RANKED_STUB = textwrap.dedent('''
    seen = []

    class _Out:
        def detach(self): return self
        def clone(self): return self
        def cpu(self): return self

    def random_shapes(h, meta):
        return [{"sig": "decode", "make_inputs": lambda rng, ctx: {
            "seed": rng.seed, "rank": ctx.rank, "world": ctx.world, "data_rank": ctx.data_rank,
            "draw_seed": ctx.seed, "device": ctx.device}}]

    def call(args):
        seen.append(args)
        return _Out()
''')

_POOL_ENV = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "GEAK_UT_DATA_RANK_SHIFT")


class MainModes(_LegRunnerTestCase):
    """main(): the serve dispatch, its usage errors, refs/correct, generated-task timing and the rank context."""

    def setUp(self):
        _keep_modules(self, "harness_lib", "cases")          # runs after the base class pops them
        super().setUp()

    def _run(self, *argv):
        """(exit code or None, stdout, stderr) of one main()."""
        sys.argv = ["leg_runner.py", *argv]
        out, err = io.StringIO(), io.StringIO()
        code = None
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                lr.main()
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_serve_needs_its_ctl_dir(self):
        """The usage error comes from argparse, before any rank is set up; main() serving a real rank (the real
        serve_main, through main) is pinned in test_rank_pool.py TestServeMain."""
        code, out, err = self._run("--task", self.task, "--mode", "serve")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("--mode serve needs --ctl", err)
        self.assertEqual(os.listdir(self.task), [])                # nothing was set up or written

    def test_target_callable_is_only_valid_with_resolve(self):
        self._task_dir()
        code, out, err = self._run("--task", self.task, "--mode", "list", "--target-callable", "fake_live_stack:op")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("--target-callable is only valid with --mode resolve", err)

    def test_refs_and_correct_need_blob_and_out(self):
        """A usage error, before the correctness worker runs: no result file is written. main() running the real
        correctness worker in these modes is pinned in test_replay_groups.py (TestCorrectModeCPU)."""
        self._task_dir()
        blob, dest = os.path.join(self.task, "native"), os.path.join(self.task, "res.json")
        task_files = sorted(f for f in os.listdir(self.task) if f != "__pycache__")   # main() imports the harness
        for mode in ("refs", "correct"):
            for extra in ((), ("--blob", blob), ("--out", dest)):
                with self.subTest(mode=mode, extra=extra):
                    code, out, err = self._run("--task", self.task, "--mode", mode, *extra)
                    self.assertEqual((code, out), (2, ""))
                    self.assertIn(f"--mode {mode} needs --blob and --out", err)
                    self.assertEqual(sorted(f for f in os.listdir(self.task) if f != "__pycache__"), task_files)
                    self.assertFalse(os.path.exists(dest))

    def test_a_generated_task_times_with_inner_capture_context_and_raw_samples(self):
        self._task_dir(meta=dict(_PHASED), harness=_HARNESS_SAMPLES_STUB, cases=_CASES_CAPTURE_STUB)
        self._write("unittest.py", lr.GENERATED_MARKER + " (generate_ut.py v1): do not edit\n")
        self._target_module()

        def timed(*extra):
            out = json.loads(self._main("--task", self.task, "--mode", "time", *extra))
            kws = [c[2] for c in sys.modules["harness_lib"].calls if c[0] == "time_op"]
            return {r["sig"]: r for r in out["cases"]}, kws, sys.modules["cases"].capture_context
        rows, (dec_kw, pre_kw), capture = timed("--raw-samples")
        self.assertEqual(dec_kw, {"detail": True, "samples": True, "inner": 4, "capture_context": capture})
        self.assertEqual(pre_kw, {"detail": True, "samples": True, "inner": 1, "capture_context": capture,
                                  "spin": {}})
        self.assertEqual((rows["decode"]["inner"], rows["decode"]["samples"]), (4, [1.25, 1.5]))
        self.assertEqual(rows["prefill"]["samples"], [1.25, 1.5])
        self.assertNotIn("inner", rows["prefill"])               # a case's own inner only when it raised one
        # --inner overrides every case's inner and is reported on every row; no samples unless asked
        rows, kws, _ = timed("--inner", "8")
        self.assertEqual([kw["inner"] for kw in kws], [8, 8])
        self.assertEqual([(r["inner"], "samples" in r) for r in rows.values()], [(8, False), (8, False)])
        self.assertTrue(all("samples" not in kw for kw in kws))
        # a hand-written task: the samples it asked for, nothing else
        self._write("unittest.py", "# hand-written UT\n")
        rows, kws, _ = timed("--raw-samples", "--inner", "8")
        self.assertEqual(kws, [{"detail": True, "samples": True}] * 2)
        self.assertEqual([("inner" in r, r["samples"]) for r in rows.values()], [(False, [1.25, 1.5])] * 2)

    def test_oracle_builds_each_draw_from_the_rank_context_of_the_pool_environment(self):
        shutil.copy(os.path.join(SCRIPTS_DIR, "harness_lib.py"), os.path.join(self.task, "harness_lib.py"))
        self._write("meta.json", json.dumps({"target_callable": "fake_live_stack:op", "regime": {}}))
        self._write("cases.py", _CASES_RANKED_STUB)
        base = {k: v for k, v in os.environ.items() if k not in _POOL_ENV}
        dest = os.path.join(self.task, "o.pt")
        for env, rank, world in (({"WORLD_SIZE": "2", "RANK": "1"}, 1, 2), ({}, 0, 1)):
            with self.subTest(env=env), mock.patch.dict(os.environ, dict(base, **env), clear=True):
                out = json.loads(self._main("--task", self.task, "--mode", "oracle", "--out", dest,
                                            "--draws", "2", "--seed", "10"))
                self.assertEqual(out, {"out": dest, "n": 2})
                self.assertEqual(sys.modules["cases"].seen, [
                    {"seed": s, "rank": rank, "world": world, "data_rank": rank, "draw_seed": s, "device": "cpu"}
                    for s in (10, 11)])
                self.assertEqual(sorted(self.torch.saved[-1][0]), ["decode|0", "decode|1"])


if __name__ == "__main__":
    unittest.main()
