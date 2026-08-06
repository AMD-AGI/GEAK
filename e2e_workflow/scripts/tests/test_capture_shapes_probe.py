#!/usr/bin/env python3
"""Unit tests for capture_shapes_probe.py -- the in-server per-shape CALL-COUNT probe (no torch).

This probe recovers what the profiler cannot see: kernels replayed inside a CUDA graph report
`dims=[]` for ~99% of GPU time, so the shape histogram that decides WHICH shapes to tune has to come
from a Python-level wrapper instead. That wrapper sits on a hot serving path inside a live vLLM/SGLang
process, which makes its failure modes expensive and quiet:

  - it must NEVER change behaviour: the real kernel runs first, its return value is passed through
    untouched, and any bookkeeping error is swallowed rather than raised into the server
  - it must NEVER eagerly import its target's module (importing aiter on the EngineCore handshake
    path blocks startup) -- hence the lazy meta-path hook
  - it must refuse a triton @jit JITFunction: those are launched as fn[grid](...), and wrapping one
    with a plain function crashes the server
  - it must scan KWARGS: vLLM calls unified_attention(q=..., k=..., v=...) entirely by keyword, and a
    positional-only scan records dims=[] -- observed, and indistinguishable from graph capture
  - persistence must not depend on the exit path: vLLM installs its own SIGTERM handler on EngineCore
    and atexit does not reliably run in that child, so a daemon thread snapshots periodically, and
    per-pid filenames keep APIServer and EngineCore from clobbering each other

torch is stubbed into sys.modules -- the module imports it lazily inside _torch(), so the whole probe
runs on CPU against fake tensors exposing only .shape/.dtype. Each test restores the module globals
the probe mutates (_TARGETS, _PENDING, the finder/atexit/flusher flags) plus sys.meta_path, so nothing
leaks between tests or writes at interpreter exit.

Run: python3 -m pytest e2e_workflow/scripts/tests/test_capture_shapes_probe.py -v
"""
import atexit
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import time
import types
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MISSING = object()


@contextlib.contextmanager
def _env(**kw):
    old = {k: os.environ.get(k, _MISSING) for k in kw}
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is _MISSING:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def _stderr():
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        yield buf


def _load(mod_name, filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# PROBE_TIME is read at import time; load with it clear so the shared instance starts untimed.
with _env(PROBE_TIME=None):
    P = _load("capture_shapes_probe", "capture_shapes_probe.py")


# --------------------------------------------------------------------------- #
# Fake torch / tensors
# --------------------------------------------------------------------------- #
class FakeTensor:
    def __init__(self, *shape, dtype="torch.bfloat16"):
        self.shape = tuple(shape)
        self.dtype = dtype


class FakeEvent:
    """Stand-in for torch.cuda.Event: `done` controls whether elapsed_time is readable yet."""

    def __init__(self, enable_timing=False, ms=1.0, done=True, explode=False):
        self.enable_timing = enable_timing
        self.ms = ms
        self.done = done
        self.explode = explode
        self.recorded = 0

    def record(self):
        self.recorded += 1

    def query(self):
        if self.explode:
            raise RuntimeError("event query blew up")
        return self.done

    def elapsed_time(self, other):
        return self.ms


def _fake_torch():
    torch = types.ModuleType("torch")
    torch.is_tensor = lambda o: isinstance(o, FakeTensor)
    torch.events = []

    def make_event(enable_timing=False):
        ev = FakeEvent(enable_timing=enable_timing)
        torch.events.append(ev)
        return ev

    torch.cuda = types.SimpleNamespace(Event=make_event)
    return torch


class _ProbeTestCase(unittest.TestCase):
    """Stub torch, pristine module globals, a temp out_dir, and no leaked hook/thread/atexit."""

    def setUp(self):
        self.torch = _fake_torch()
        self._prev_torch = sys.modules.get("torch", _MISSING)
        sys.modules["torch"] = self.torch
        self.addCleanup(self._restore_torch)

        self._meta_path = list(sys.meta_path)
        self.addCleanup(self._restore_meta_path)
        self.addCleanup(atexit.unregister, P._flush_all)
        self._reset_globals()
        self.addCleanup(self._reset_globals)

        self.out_dir = os.path.join(tempfile.mkdtemp(prefix="probe_test_"), "out")
        self.addCleanup(shutil.rmtree, os.path.dirname(self.out_dir), True)

    def _restore_torch(self):
        if self._prev_torch is _MISSING:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self._prev_torch

    def _restore_meta_path(self):
        sys.meta_path[:] = self._meta_path

    def _reset_globals(self):
        P._TARGETS.clear()
        P._PENDING.clear()
        P._FINDER_INSTALLED = False
        P._ATEXIT_REGISTERED = False
        # Left SET by default: only the flusher's own test may spawn the daemon thread.
        P._FLUSHER_STARTED = True
        P._TIME = False

    def _target_module(self, fn=None, name="fake_serving_layer", attr="op"):
        mod = types.ModuleType(name)
        setattr(mod, attr, fn if fn is not None else (lambda *a, **k: "OUT"))
        sys.modules[name] = mod
        self.addCleanup(sys.modules.pop, name, None)
        return mod

    def _install(self, target="fake_serving_layer:op", out_dir=_MISSING):
        with _stderr() as err:
            P.install(target, self.out_dir if out_dir is _MISSING else out_dir)
        return err.getvalue()

    def _flushed(self, target="fake_serving_layer:op"):
        safe = target.replace(":", "__").replace(".", "_")
        with open(os.path.join(self.out_dir, f"probe_{os.getpid()}_{safe}.json")) as fh:
            return json.load(fh)

    def _only_case(self, target="fake_serving_layer:op"):
        return next(iter(P._TARGETS[target]["cases"].values()))


# --------------------------------------------------------------------------- #
# _iter_tensor_args / _shape_sig -- the histogram key
# --------------------------------------------------------------------------- #
class ShapeSignature(_ProbeTestCase):
    def test_positional_tensors_are_labelled_by_index(self):
        sig = P._shape_sig((FakeTensor(64, 1024), FakeTensor(1024, 4096)), {})
        self.assertEqual(sig, "arg0=T(64, 1024):torch.bfloat16|arg1=T(1024, 4096):torch.bfloat16")

    def test_kwargs_are_scanned_and_visited_in_sorted_order(self):
        """vLLM calls unified_attention(q=,k=,v=) by keyword; a positional-only scan records dims=[]."""
        sig = P._shape_sig((), {"v": FakeTensor(3), "q": FakeTensor(1), "k": FakeTensor(2)})
        self.assertEqual(sig, "k=T(2,):torch.bfloat16|q=T(1,):torch.bfloat16|v=T(3,):torch.bfloat16")

    def test_tensors_nested_one_level_in_lists_and_tuples_are_found(self):
        sig = P._shape_sig(([FakeTensor(8)],), {"caches": (FakeTensor(9),)})
        self.assertEqual(sig, "arg0[0]=T(8,):torch.bfloat16|caches[0]=T(9,):torch.bfloat16")

    def test_non_tensor_arguments_are_ignored(self):
        sig = P._shape_sig((7, "x", None, [1, 2]), {"flag": True, "opts": {"a": 1}, "empty": ()})
        self.assertEqual(sig, "<no-tensor-args>")

    def test_dtype_is_part_of_the_key(self):
        a = P._shape_sig((FakeTensor(64, dtype="torch.bfloat16"),), {})
        b = P._shape_sig((FakeTensor(64, dtype="torch.float8_e4m3fn"),), {})
        self.assertNotEqual(a, b)


# --------------------------------------------------------------------------- #
# install / _try_hook -- the lazy, never-importing hook
# --------------------------------------------------------------------------- #
class Install(_ProbeTestCase):
    def test_an_already_imported_module_is_hooked_immediately(self):
        mod = self._target_module()
        orig = mod.op
        err = self._install()
        self.assertIsNot(mod.op, orig)
        self.assertIs(P._TARGETS["fake_serving_layer:op"]["orig"], orig)
        self.assertIn("hooked now", err)

    def test_install_never_imports_the_target_module(self):
        """Eagerly importing a heavy lib on the EngineCore handshake path blocks startup."""
        target = "aiter.ops.triton.unified_attention:unified_attention"
        err = self._install(target)
        self.assertNotIn("aiter.ops.triton.unified_attention", sys.modules)
        self.assertIn("pending lazy hook", err)
        self.assertIsNone(P._TARGETS[target]["orig"])

    def test_a_late_import_is_hooked_by_the_meta_path_finder(self):
        self._install("late_module:op")
        finder = sys.meta_path[0]
        self.assertIsInstance(finder, P._HookFinder)
        mod = self._target_module(name="late_module")
        orig = mod.op
        with _stderr():
            self.assertIsNone(finder.find_spec("late_module"))
        self.assertIsNot(mod.op, orig)

    def test_the_finder_never_claims_an_import(self):
        self._install("late_module:op")
        finder = sys.meta_path[0]
        self.assertIsNone(finder.find_spec("some.unrelated.module"))
        self.assertIsNone(finder.find_module("some.unrelated.module"))

    def test_the_finder_sweeps_targets_imported_as_a_side_effect(self):
        """A target's module can appear while a DIFFERENT module is being imported."""
        self._install("side_effect_module:op")
        finder = sys.meta_path[0]
        mod = self._target_module(name="side_effect_module")
        orig = mod.op
        with _stderr():
            finder.find_spec("something.else.entirely")
        self.assertIsNot(mod.op, orig)

    def test_a_finder_sweep_error_is_swallowed(self):
        """The finder runs inside every import in the server; it must never raise."""
        self._install("late_module:op")
        finder = sys.meta_path[0]
        P._PENDING["boom"] = "not-a-list-of-pairs"
        with _stderr() as err:
            self.assertIsNone(finder.find_spec("boom"))
        self.assertIn("finder sweep error", err.getvalue())

    def test_install_is_idempotent_per_target(self):
        mod = self._target_module()
        self._install()
        wrapped = mod.op
        self._install()
        self.assertIs(mod.op, wrapped)          # not double-wrapped
        self.assertEqual(len(P._PENDING["fake_serving_layer"]), 1)

    def test_only_one_finder_is_ever_inserted(self):
        self._target_module()
        self._install()
        self._install("other_module:op")
        self.assertEqual(sum(isinstance(f, P._HookFinder) for f in sys.meta_path), 1)

    def test_a_missing_attr_leaves_the_target_pending(self):
        self._target_module(name="partial_module", attr="something_else")
        err = self._install("partial_module:op")
        self.assertIsNone(P._TARGETS["partial_module:op"]["orig"])
        self.assertIn("pending lazy hook", err)

    def test_a_target_whose_module_is_absent_is_not_hooked(self):
        self._install("never_imported_module:op")
        self.assertFalse(P._try_hook("never_imported_module:op"))

    def test_a_jit_function_is_refused_rather_than_wrapped(self):
        """A triton @jit kernel is launched as fn[grid](...); wrapping it crashes the server."""
        class JITFunction:
            def __getitem__(self, grid):
                return lambda *a, **k: "LAUNCHED"

        jit = JITFunction()
        mod = self._target_module(fn=jit, name="triton_mod", attr="kernel")
        err = self._install("triton_mod:kernel")
        self.assertIs(mod.kernel, jit)                      # untouched
        self.assertTrue(P._TARGETS["triton_mod:kernel"]["unhookable"])
        self.assertIn("SKIP triton_mod:kernel", err)
        self.assertEqual(mod.kernel[(1,)](), "LAUNCHED")    # launch syntax still works

    def test_a_non_callable_attr_is_refused(self):
        self._target_module(fn=[1, 2, 3], name="const_mod", attr="table")
        err = self._install("const_mod:table")
        self.assertTrue(P._TARGETS["const_mod:table"]["unhookable"])
        self.assertIn("not a plain callable", err)

    def test_a_refused_target_stops_being_swept(self):
        self._target_module(fn=[1], name="const_mod", attr="table")
        self._install("const_mod:table")
        self.assertTrue(P._try_hook("const_mod:table"))     # resolved -> no retry


class InstallFromEnv(_ProbeTestCase):
    def test_env_targets_are_installed(self):
        self._target_module()
        self._target_module(name="second_layer")
        with _env(PROBE_TARGETS="fake_serving_layer:op, second_layer:op",
                  PROBE_OUT=self.out_dir), _stderr():
            P.install_from_env()
        self.assertEqual(sorted(P._TARGETS), ["fake_serving_layer:op", "second_layer:op"])

    def test_missing_env_is_a_no_op(self):
        for tgts, out in (("", self.out_dir), ("a:b", ""), (None, None)):
            with self.subTest(targets=tgts):
                with _env(PROBE_TARGETS=tgts, PROBE_OUT=out), _stderr():
                    P.install_from_env()
                self.assertEqual(P._TARGETS, {})

    def test_a_malformed_target_does_not_stop_the_others(self):
        self._target_module()
        with _env(PROBE_TARGETS="no_colon_here,fake_serving_layer:op",
                  PROBE_OUT=self.out_dir), _stderr() as err:
            P.install_from_env()
        self.assertIn("install failed for no_colon_here", err.getvalue())
        self.assertIn("fake_serving_layer:op", P._TARGETS)


# --------------------------------------------------------------------------- #
# The wrapper on the hot path
# --------------------------------------------------------------------------- #
class Wrapper(_ProbeTestCase):
    def test_the_real_kernel_runs_once_and_its_result_passes_through(self):
        seen = []

        def op(*a, **k):
            seen.append((a, k))
            return "REAL"

        mod = self._target_module(fn=op)
        self._install()
        t = FakeTensor(64, 1024)
        self.assertEqual(mod.op(t, scale=2), "REAL")
        self.assertEqual(seen, [((t,), {"scale": 2})])

    def test_every_call_is_counted_at_every_shape_uncapped(self):
        """Unlike capture_shapes there is no max_cases: the whole distribution is the product."""
        mod = self._target_module()
        self._install()
        for i in range(40):
            mod.op(FakeTensor(i % 4, 1024))
        st = P._TARGETS["fake_serving_layer:op"]
        self.assertEqual(st["calls"], 40)
        self.assertEqual(len(st["cases"]), 4)
        self.assertTrue(all(c["count"] == 10 for c in st["cases"].values()))

    def test_dims_dtypes_and_labels_stay_parallel_per_tensor(self):
        """Downstream needs each tensor's OWN (shape, dtype): bf16 act + fp32 scale must not merge."""
        mod = self._target_module()
        self._install()
        mod.op(FakeTensor(64, 1024), scale=FakeTensor(64, dtype="torch.float32"))
        case = self._only_case()
        self.assertEqual(case["dims"], [[64, 1024], [64]])
        self.assertEqual(case["dtypes"], ["torch.bfloat16", "torch.float32"])
        self.assertEqual(case["arg_labels"], ["arg0", "scale"])

    def test_the_probe_never_snapshots_tensor_values(self):
        """Only .shape/.dtype are read -- a detach/clone on a hot path would be ruinous."""
        class Exploding(FakeTensor):
            def detach(self):
                raise AssertionError("probe must not snapshot I/O")

            def clone(self):
                raise AssertionError("probe must not snapshot I/O")

        mod = self._target_module()
        self._install()
        mod.op(Exploding(64, 1024))
        self.assertEqual(P._TARGETS["fake_serving_layer:op"]["calls"], 1)

    def test_a_bookkeeping_error_never_breaks_the_server(self):
        mod = self._target_module()
        self._install()
        P._TARGETS["fake_serving_layer:op"]["cases"] = None   # force an internal failure
        with _stderr() as err:
            self.assertEqual(mod.op(FakeTensor(1)), "OUT")
        self.assertIn("capture error", err.getvalue())

    def test_an_exception_from_the_real_kernel_propagates(self):
        """The probe swallows ITS OWN errors, not the kernel's."""
        def boom(*a, **k):
            raise ValueError("kernel failed")

        mod = self._target_module(fn=boom)
        self._install()
        with self.assertRaises(ValueError):
            mod.op(FakeTensor(1))


class WrapperTiming(_ProbeTestCase):
    """PROBE_TIME=1: cuda.Event pairs, drained lazily off the hot path."""

    def setUp(self):
        super().setUp()
        P._TIME = True

    def test_events_bracket_the_call_and_are_stashed_not_read(self):
        mod = self._target_module()
        self._install()
        mod.op(FakeTensor(64))
        case = self._only_case()
        self.assertEqual(len(case["_pending"]), 1)
        self.assertEqual(case["gpu_ms_sum"], 0.0)     # never synchronises on the hot path
        self.assertEqual([e.recorded for e in self.torch.events], [1, 1])

    def test_timing_unavailable_falls_back_to_an_untimed_call(self):
        def no_events(enable_timing=False):
            raise RuntimeError("no cuda")

        self.torch.cuda.Event = no_events
        mod = self._target_module()
        self._install()
        self.assertEqual(mod.op(FakeTensor(64)), "OUT")
        self.assertEqual(self._only_case()["_pending"], [])

    def test_the_first_sample_per_shape_is_dropped_as_warmup(self):
        """First call at a new shape pays JIT/autotune; keeping it skews a small sample badly."""
        mod = self._target_module()
        self._install()
        for _ in range(3):
            mod.op(FakeTensor(64))
        P._drain_timing(P._TARGETS["fake_serving_layer:op"])
        case = self._only_case()
        self.assertEqual(case["timed_count"], 2)      # 3 calls, 1 warmup dropped
        self.assertEqual(case["gpu_ms_sum"], 2.0)
        self.assertEqual(case["_pending"], [])

    def test_unfinished_events_stay_pending(self):
        mod = self._target_module()
        self._install()
        mod.op(FakeTensor(64))
        case = self._only_case()
        case["_pending"] = [(FakeEvent(), FakeEvent(done=False))]
        P._drain_timing(P._TARGETS["fake_serving_layer:op"])
        self.assertEqual(len(case["_pending"]), 1)
        self.assertEqual(case["timed_count"], 0)

    def test_an_unreadable_event_pair_is_dropped(self):
        mod = self._target_module()
        self._install()
        mod.op(FakeTensor(64))
        case = self._only_case()
        case["_pending"] = [(FakeEvent(), FakeEvent(explode=True))]
        P._drain_timing(P._TARGETS["fake_serving_layer:op"])
        self.assertEqual(case["_pending"], [])
        self.assertEqual(case["timed_count"], 0)

    def test_drain_skips_cases_with_nothing_pending(self):
        st = {"cases": {"a": {"_pending": []}, "b": {}}}
        P._drain_timing(st)                            # must not raise
        self.assertEqual(st["cases"]["a"]["_pending"], [])

    def test_flush_reports_measured_micros_per_call(self):
        mod = self._target_module()
        self._install()
        for _ in range(3):
            mod.op(FakeTensor(64))
        with _stderr():
            P._flush_one("fake_serving_layer:op")
        payload = self._flushed()
        self.assertTrue(payload["timing"])
        self.assertEqual(payload["cases"][0]["gpu_us_avg"], 1000.0)   # 1.0 ms/call
        self.assertEqual(payload["cases"][0]["timed_count"], 2)

    def test_a_drain_failure_does_not_stop_the_flush(self):
        mod = self._target_module()
        self._install()
        mod.op(FakeTensor(64))
        self._only_case()["_pending"] = "not-a-list-of-event-pairs"
        with _stderr():
            P._flush_one("fake_serving_layer:op")
        payload = self._flushed()
        self.assertEqual(payload["total_calls"], 1)
        self.assertNotIn("gpu_us_avg", payload["cases"][0])


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
class Flush(_ProbeTestCase):
    def test_the_filename_is_per_pid_and_per_target(self):
        """APIServer and EngineCore both flush; a shared name loses one of them."""
        self._target_module(name="pkg.sub")
        self._install("pkg.sub:op")
        sys.modules["pkg.sub"].op(FakeTensor(1))
        with _stderr() as err:
            P._flush_one("pkg.sub:op")
        name = f"probe_{os.getpid()}_pkg_sub__op.json"
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, name)))
        self.assertIn(name, err.getvalue())

    def test_the_out_dir_is_created_on_demand(self):
        self.assertFalse(os.path.exists(self.out_dir))
        mod = self._target_module()
        self._install()
        mod.op(FakeTensor(1))
        with _stderr():
            P._flush_one("fake_serving_layer:op")
        self.assertTrue(os.path.isdir(self.out_dir))

    def test_cases_are_written_hottest_first_with_the_totals(self):
        mod = self._target_module()
        self._install()
        for _ in range(3):
            mod.op(FakeTensor(1))
        mod.op(FakeTensor(4096))
        with _stderr():
            P._flush_one("fake_serving_layer:op")
        payload = self._flushed()
        self.assertEqual(payload["target"], "fake_serving_layer:op")
        self.assertEqual(payload["pid"], os.getpid())
        self.assertEqual(payload["total_calls"], 4)
        self.assertEqual(payload["num_distinct_shapes"], 2)
        self.assertEqual([c["count"] for c in payload["cases"]], [3, 1])
        self.assertEqual([c["dims"] for c in payload["cases"]], [[[1]], [[4096]]])
        self.assertFalse(payload["timing"])
        self.assertNotIn("gpu_us_avg", payload["cases"][0])

    def test_flush_is_an_idempotent_overwrite(self):
        mod = self._target_module()
        self._install()
        mod.op(FakeTensor(1))
        with _stderr():
            P._flush_one("fake_serving_layer:op")
            mod.op(FakeTensor(1))
            P._flush_one("fake_serving_layer:op")
        self.assertEqual(self._flushed()["total_calls"], 2)
        self.assertEqual(len(os.listdir(self.out_dir)), 1)

    def test_flush_all_skips_targets_that_never_ran(self):
        """46k+ empty files were observed from short-lived helper processes; never write those."""
        self._target_module()
        self._target_module(name="idle_layer")
        self._install()
        self._install("idle_layer:op")
        sys.modules["fake_serving_layer"].op(FakeTensor(1))
        with _stderr():
            P._flush_all()
        self.assertEqual(len(os.listdir(self.out_dir)), 1)

    def test_a_flush_error_on_one_target_does_not_stop_the_others(self):
        self._target_module()
        self._target_module(name="broken_layer")
        self._install()
        self._install("broken_layer:op")
        sys.modules["fake_serving_layer"].op(FakeTensor(1))
        sys.modules["broken_layer"].op(FakeTensor(1))
        P._TARGETS["broken_layer:op"]["out_dir"] = None       # makedirs will raise
        with _stderr() as err:
            P._flush_all()
        self.assertIn("flush error on broken_layer:op", err.getvalue())
        self.assertEqual(self._flushed()["total_calls"], 1)

    def test_nothing_is_written_when_no_target_ran(self):
        self._target_module()
        self._install()
        with _stderr():
            P._flush_all()
        self.assertFalse(os.path.exists(self.out_dir))


class PeriodicFlusher(_ProbeTestCase):
    def test_the_daemon_thread_snapshots_without_an_exit_hook(self):
        """vLLM overwrites the SIGTERM handler on EngineCore and atexit may never run."""
        mod = self._target_module()
        self._install()
        mod.op(FakeTensor(64))
        path = os.path.join(self.out_dir, f"probe_{os.getpid()}_fake_serving_layer__op.json")
        P._FLUSHER_STARTED = False
        with _stderr():
            P._start_periodic_flush(interval=0.01)
            P._start_periodic_flush(interval=0.01)   # idempotent: no second thread
            deadline = time.monotonic() + 10
            while not os.path.exists(path) and time.monotonic() < deadline:
                time.sleep(0.02)
        self.assertTrue(os.path.exists(path), "periodic flusher never wrote a snapshot")

    def test_install_registers_the_exit_hook_and_the_flusher_once(self):
        self._target_module()
        self._target_module(name="second_layer")
        self._install()
        self.assertTrue(P._ATEXIT_REGISTERED)
        self._install("second_layer:op")
        self.assertTrue(P._ATEXIT_REGISTERED)

    def test_the_flusher_thread_survives_a_failing_snapshot(self):
        """One corrupt target must not silently kill the thread that is the PRIMARY persistence."""
        class Hostile:
            attempts = 0

            def __getitem__(self, key):
                type(self).attempts += 1
                raise RuntimeError("corrupt target state")

        P._TARGETS["hostile:op"] = Hostile()
        P._FLUSHER_STARTED = False
        with _stderr():
            P._start_periodic_flush(interval=0.01)
            deadline = time.monotonic() + 10
            while Hostile.attempts < 3 and time.monotonic() < deadline:
                time.sleep(0.02)
        self.assertGreaterEqual(Hostile.attempts, 3, "flusher thread died on the first failure")


if __name__ == "__main__":
    unittest.main()
