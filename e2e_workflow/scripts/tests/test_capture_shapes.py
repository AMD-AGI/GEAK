#!/usr/bin/env python3
"""Unit tests for capture_shapes.py -- the in-server shape/oracle recorder (stdlib only; no torch).

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_capture_shapes.py

capture_shapes.py is what tells us WHICH shapes production traffic actually hits: it monkeypatches a
hot callable inside a live server and records the complete shape histogram plus an immutable I/O
oracle. When that histogram is wrong, tuning optimizes shapes the deployment never runs -- the
"isolated win, e2e loss" failure (a run that landed +0.06% end-to-end because the tuned shape set did
not match the live distribution). The behaviours pinned here are the ones that silently poison a run:

  - _sig / _shapes_dtypes / _lead_regime : the histogram key, its shape/dtype payload, and the
                                          decode-vs-prefill tag that later splits profiled TIME
  - _wrapper                            : EVERY call counted at EVERY shape (uncapped), oracle
                                          capture capped by max_cases yet never frozen on a single
                                          regime, and a capture failure that must not break serving
  - _capturing / _maybe_flush           : never snapshot or write while a CUDA graph is captured
  - _flush                              : the atexit dump -- meta.json contents + oracle sha256
  - _wrappable / _make_wrapper / install: refuse native callables (the mxfp4 matmul_ogs SIGSEGV),
                                          stay transparent to introspection-driven dispatch, and
                                          install idempotently

torch is stubbed into sys.modules: the module imports it lazily inside _torch(), so the whole recorder
runs on CPU against fake tensors that expose only .shape/.dtype. Each test restores sys.modules, the
module's global _STATE, and any atexit hook install() registered, so nothing writes to the repo.
"""
import atexit
import contextlib
import functools
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import types
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_MISSING = object()


@contextlib.contextmanager
def _env(**kw):
    """Set/clear env vars for the duration of the block (None clears)."""
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
    """Capture the module's sys.stderr progress lines instead of spraying them into the test log."""
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        yield buf


def _load(mod_name, filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The module self-installs on import when CAPTURE_TARGET/CAPTURE_OUT are set; load it with a clean
# env so the shared instance is inert regardless of the ambient environment.
with _env(CAPTURE_TARGET=None, CAPTURE_OUT=None, CAPTURE_DECODE_LEAD_MAX=None):
    cs = _load("capture_shapes", "capture_shapes.py")


def _reset_state(mod):
    """Restore _STATE to its post-import defaults, keeping the existing lock object."""
    mod._STATE.update(
        target=None, out_dir=None, max_cases=5, num_steps=0,
        records=[], seen=set(), orig=None, mod=None, attr=None,
        installed=False, calls=0,
        regime_seen=set(), decode_lead_max=256,
        sequence=[], seq_cap=256, in_graph_calls=0,
        shape_counts={}, shape_meta={},
        flush_every=64, oracle_written=False, oracle_sha=None, oracle_records=0,
        byte_budget=0, case_byte_limit=0, persist_policy="share_large",
        share_min_bytes=16 << 20, shared_tensors={}, shared_bytes_est=0,
        oracle_bytes_est=0, budget_exceeded=False,
        budget_skip_count=0, oracle_save_count=0, meta_flush_count=0,
    )


class FakeTensor:
    """Just enough torch.Tensor surface for the recorder: shape/dtype/device + the detach/clone walk."""

    def __init__(self, shape, dtype="torch.float16", device="cuda:0", contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self._contiguous = contiguous

    def dim(self):
        return len(self.shape)

    def numel(self):
        n = 1
        for dim in self.shape:
            n *= int(dim)
        return n

    def element_size(self):
        name = str(self.dtype).split(".")[-1].lower()
        return {
            "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2,
            "int64": 8, "int32": 4, "int8": 1, "uint8": 1, "bool": 1,
        }.get(name, 2)

    def detach(self):
        return self

    def to(self, device):
        return FakeTensor(self.shape, self.dtype, device, self._contiguous)

    def clone(self):
        return FakeTensor(self.shape, self.dtype, self.device, self._contiguous)

    def is_contiguous(self):
        return self._contiguous


class ExplodingTensor:
    """A tensor whose .shape read raises -- stands in for an exotic tensor subclass mid-serving."""

    dtype = "torch.float16"

    @property
    def shape(self):
        raise RuntimeError("shape read failed")


def _fake_torch():
    torch = types.ModuleType("torch")
    torch.capturing = False
    torch.saved = []
    torch.is_tensor = lambda o: isinstance(o, (FakeTensor, ExplodingTensor))
    torch.cuda = types.SimpleNamespace(
        is_current_stream_capturing=lambda: torch.capturing)

    def save(obj, path):
        torch.saved.append((obj, path))
        with open(path, "wb") as fh:
            fh.write(b"fake-oracle:%d" % len(obj.get("records", [])))

    torch.save = save
    return torch


class _RecorderTestCase(unittest.TestCase):
    """Gives every test a stub torch, a pristine _STATE, a temp out_dir, and no leaked atexit hook."""

    def setUp(self):
        self.torch = _fake_torch()
        self._prev_torch = sys.modules.get("torch", _MISSING)
        sys.modules["torch"] = self.torch
        self.addCleanup(self._restore_torch)
        _reset_state(cs)
        self.addCleanup(_reset_state, cs)
        # install() registers _flush with atexit; drop it so no test writes at interpreter exit.
        self.addCleanup(atexit.unregister, cs._flush)
        self.out_dir = tempfile.mkdtemp(prefix="capture_shapes_test_")
        self.addCleanup(shutil.rmtree, self.out_dir, True)

    def _restore_torch(self):
        if self._prev_torch is _MISSING:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self._prev_torch

    def _target_module(self, fn=None, name="fake_serving_layer"):
        mod = types.ModuleType(name)
        mod.op = fn if fn is not None else (lambda *a, **k: "OUT")
        sys.modules[name] = mod
        self.addCleanup(sys.modules.pop, name, None)
        return mod

    def _hook(self, fn=None, out_dir=None, max_cases=5, name="fake_serving_layer"):
        """Install the recorder over a stub module's `op`, returning (module, install stderr)."""
        mod = self._target_module(fn, name)
        with _stderr() as err:
            cs.install(f"{name}:op", self.out_dir if out_dir is None else out_dir,
                       max_cases=max_cases)
        return mod, err.getvalue()

    def _meta(self, out_dir=None):
        with open(os.path.join(out_dir or self.out_dir, "meta.json")) as fh:
            return json.load(fh)


# --------------------------------------------------------------------------- #
# _sig -- the histogram key
# --------------------------------------------------------------------------- #
class TestSig(_RecorderTestCase):
    def test_tensor_args_carry_shape_and_dtype(self):
        sig = cs._sig((FakeTensor((4, 8)), FakeTensor((8,), dtype="torch.bfloat16")), {})
        self.assertEqual(sig, "T(4, 8):torch.float16|T(8,):torch.bfloat16")

    def test_scalars_are_verbatim_and_objects_by_type(self):
        self.assertEqual(cs._sig((7, 0.5, True, None, object()), {}),
                         "7|0.5|True|None|object")

    def test_kwargs_are_sorted_so_the_key_is_call_order_independent(self):
        t = FakeTensor((2, 3))
        a = cs._sig((), {"scale": 0.5, "bias": t})
        b = cs._sig((), {"bias": t, "scale": 0.5})
        self.assertEqual(a, b)
        self.assertEqual(a, "bias=T(2, 3):torch.float16|scale=0.5")

    def test_non_scalar_kwarg_is_reduced_to_its_type(self):
        self.assertEqual(cs._sig((), {"opts": {"a": 1}}), "opts=dict")

    def test_same_shape_with_a_different_scalar_is_a_DIFFERENT_key(self):
        # Pinned as-is because it is a real fragmentation risk: the histogram key mixes tensor shapes
        # with scalar VALUES, so a varying scalar (layer_idx, seq_len, ...) splits one hot shape into
        # many low-count entries and inflates num_distinct_shapes.
        t = FakeTensor((4, 8))
        self.assertNotEqual(cs._sig((t, 0), {}), cs._sig((t, 1), {}))


# --------------------------------------------------------------------------- #
# _shapes_dtypes -- the per-key shape/dtype payload
# --------------------------------------------------------------------------- #
class TestShapesDtypes(_RecorderTestCase):
    def test_walks_nested_containers_in_args_and_kwargs(self):
        got = cs._shapes_dtypes(
            (FakeTensor((4, 8)), [FakeTensor((8, 8)), 3], "str"),
            {"extra": {"w": FakeTensor((8, 2), dtype="torch.float32")}, "scale": 0.5})
        self.assertEqual(got["input_shapes"], [[4, 8], [8, 8], [8, 2]])
        self.assertEqual(got["input_dtypes"], ["torch.float16", "torch.float32"])

    def test_dtypes_are_deduped_and_sorted(self):
        got = cs._shapes_dtypes((FakeTensor((1,)), FakeTensor((2,)),
                                 FakeTensor((3,), dtype="torch.bfloat16")), {})
        self.assertEqual(got["input_dtypes"], ["torch.bfloat16", "torch.float16"])

    def test_no_tensors_is_empty(self):
        self.assertEqual(cs._shapes_dtypes((1, "x"), {"k": None}),
                         {"input_shapes": [], "input_dtypes": []})


# --------------------------------------------------------------------------- #
# _lead_regime -- the decode/prefill tag that later splits profiled TIME
# --------------------------------------------------------------------------- #
class TestLeadRegime(_RecorderTestCase):
    def test_small_leading_dim_is_decode(self):
        self.assertEqual(cs._lead_regime((FakeTensor((4, 8)),), {}), "decode")

    def test_batched_decode_up_to_the_cutoff_is_still_decode(self):
        # The documented bug: a cutoff of 8 called batched decode "prefill" and never captured a
        # decode oracle case under load. At the 256 default, a full max_num_seqs batch stays decode.
        self.assertEqual(cs._lead_regime((FakeTensor((256, 8)),), {}), "decode")

    def test_large_leading_dim_is_prefill(self):
        self.assertEqual(cs._lead_regime((FakeTensor((512, 8)),), {}), "prefill")

    def test_cutoff_is_configurable(self):
        cs._STATE["decode_lead_max"] = 8
        self.assertEqual(cs._lead_regime((FakeTensor((256, 8)),), {}), "prefill")

    def test_first_tensor_is_found_through_nested_sequences_and_kwargs(self):
        self.assertEqual(cs._lead_regime(([], [None, [FakeTensor((999, 4))]]), {}), "prefill")
        self.assertEqual(cs._lead_regime((), {"x": FakeTensor((1024, 4))}), "prefill")

    def test_no_tensor_operand_defaults_to_decode(self):
        self.assertEqual(cs._lead_regime((1, "x"), {"k": None}), "decode")

    def test_zero_dim_tensor_defaults_to_decode(self):
        self.assertEqual(cs._lead_regime((FakeTensor(()),), {}), "decode")


# --------------------------------------------------------------------------- #
# _capturing -- graph-capture detection must never raise into the serving thread
# --------------------------------------------------------------------------- #
class TestCapturing(_RecorderTestCase):
    def test_reports_capture_when_torch_says_so(self):
        self.torch.capturing = True
        self.assertTrue(cs._capturing())

    def test_false_when_not_capturing(self):
        self.assertFalse(cs._capturing())

    def test_older_torch_without_the_query_is_treated_as_eager(self):
        self.torch.cuda = types.SimpleNamespace()
        self.assertFalse(cs._capturing())

    def test_torch_absent_is_treated_as_eager(self):
        sys.modules["torch"] = None
        self.assertFalse(cs._capturing())


# --------------------------------------------------------------------------- #
# _wrapper -- the recording hot path
# --------------------------------------------------------------------------- #
class TestWrapperRecording(_RecorderTestCase):
    def test_call_is_transparent_and_records_the_shape_key(self):
        mod, _ = self._hook(fn=lambda *a, **k: FakeTensor((4, 8)))
        with _stderr():
            out = mod.op(FakeTensor((4, 8)), scale=0.5)
        self.assertEqual(out.shape, (4, 8))
        s = cs._STATE
        self.assertEqual(s["calls"], 1)
        self.assertEqual(s["shape_counts"], {"T(4, 8):torch.float16|scale=0.5": 1})
        self.assertEqual(s["shape_meta"]["T(4, 8):torch.float16|scale=0.5"],
                         {"input_shapes": [[4, 8]], "input_dtypes": ["torch.float16"]})
        self.assertEqual(len(s["records"]), 1)
        self.assertEqual(s["records"][0]["regime"], "decode")
        # Shapes/dtypes only -- no tensor is retained anywhere in the record.
        self.assertEqual(s["records"][0]["input_shapes"], [[4, 8]])
        self.assertEqual(s["records"][0]["input_dtypes"], ["torch.float16"])
        self.assertNotIn("args", s["records"][0])
        self.assertNotIn("output", s["records"][0])

    def test_repeated_calls_aggregate_counts_but_record_the_case_once(self):
        mod, _ = self._hook()
        t = FakeTensor((4, 8))
        with _stderr():
            for _ in range(5):
                mod.op(t)
        s = cs._STATE
        self.assertEqual(s["calls"], 5)
        self.assertEqual(s["shape_counts"]["T(4, 8):torch.float16"], 5)
        self.assertEqual(len(s["records"]), 1)
        self.assertEqual(s["seen"], {"T(4, 8):torch.float16"})

    def test_distinct_shapes_each_get_their_own_histogram_entry(self):
        mod, _ = self._hook()
        with _stderr():
            mod.op(FakeTensor((4, 8)))
            mod.op(FakeTensor((4, 8)))
            mod.op(FakeTensor((16, 8)))
        self.assertEqual(cs._STATE["shape_counts"],
                         {"T(4, 8):torch.float16": 2, "T(16, 8):torch.float16": 1})

    def test_oracle_is_capped_by_max_cases_but_never_frozen_on_one_regime(self):
        # The "single-case oracle" bug: with max_cases exhausted by decode shapes, a prefill shape
        # must STILL be recorded, otherwise the immutable oracle never exercises the big-M path.
        mod, _ = self._hook(max_cases=1)
        with _stderr():
            mod.op(FakeTensor((4, 8)))
            mod.op(FakeTensor((8, 8)))
            mod.op(FakeTensor((2048, 8)))
        s = cs._STATE
        self.assertEqual(len(s["shape_counts"]), 3)
        self.assertEqual([r["regime"] for r in s["records"]], ["decode", "prefill"])
        self.assertEqual(s["regime_seen"], {"decode", "prefill"})

    def test_second_prefill_shape_does_not_overshoot_further(self):
        mod, _ = self._hook(max_cases=1)
        with _stderr():
            mod.op(FakeTensor((2048, 8)))
            mod.op(FakeTensor((4096, 8)))
        self.assertEqual(len(cs._STATE["records"]), 1)
        self.assertEqual(len(cs._STATE["shape_counts"]), 2)

    def test_in_graph_calls_are_counted_but_never_snapshotted(self):
        # A clone during CUDA-graph capture is an illegal sync and would record placeholder data.
        mod, _ = self._hook()
        self.torch.capturing = True
        with _stderr():
            mod.op(FakeTensor((4, 8)))
        s = cs._STATE
        self.assertEqual(s["in_graph_calls"], 1)
        self.assertEqual(s["shape_counts"]["T(4, 8):torch.float16"], 1)
        self.assertEqual(s["records"], [])
        self.assertEqual(s["sequence"], [{"sig": "T(4, 8):torch.float16", "in_graph": True}])

    def test_the_same_shape_is_captured_on_a_later_eager_call(self):
        mod, _ = self._hook()
        self.torch.capturing = True
        with _stderr():
            mod.op(FakeTensor((4, 8)))
        self.torch.capturing = False
        with _stderr():
            mod.op(FakeTensor((4, 8)))
        s = cs._STATE
        self.assertEqual(len(s["records"]), 1)
        self.assertEqual(s["shape_counts"]["T(4, 8):torch.float16"], 2)
        self.assertEqual([e["in_graph"] for e in s["sequence"]], [True, False])

    def test_call_sequence_is_ordered_with_repeats_and_capped(self):
        mod, _ = self._hook()
        cs._STATE["seq_cap"] = 2
        with _stderr():
            mod.op(FakeTensor((4, 8)))
            mod.op(FakeTensor((16, 8)))
            mod.op(FakeTensor((4, 8)))
        self.assertEqual([e["sig"] for e in cs._STATE["sequence"]],
                         ["T(4, 8):torch.float16", "T(16, 8):torch.float16"])
        self.assertEqual(cs._STATE["calls"], 3)

    def test_a_capture_failure_is_swallowed_so_serving_continues(self):
        mod, _ = self._hook()
        with _stderr() as err:
            out = mod.op(ExplodingTensor())
        self.assertEqual(out, "OUT")
        self.assertIn("capture error (ignored)", err.getvalue())
        self.assertEqual(cs._STATE["calls"], 1)
        self.assertEqual(cs._STATE["shape_counts"], {})

    def test_missing_torch_does_not_break_the_hooked_call(self):
        mod, _ = self._hook()
        sys.modules["torch"] = None
        with _stderr() as err:
            out = mod.op(FakeTensor((4, 8)))
        self.assertEqual(out, "OUT")
        self.assertIn("capture error (ignored)", err.getvalue())
        self.assertEqual(cs._STATE["records"], [])

    def test_recorded_case_is_announced_on_stderr(self):
        mod, _ = self._hook()
        with _stderr() as err:
            mod.op(FakeTensor((4, 8)))
        self.assertIn("recorded case 1 (decode): T(4, 8):torch.float16", err.getvalue())


# --------------------------------------------------------------------------- #
# thread safety -- a live server calls the hook from many serving threads
# --------------------------------------------------------------------------- #
class TestThreadSafety(_RecorderTestCase):
    def test_concurrent_calls_aggregate_under_the_lock(self):
        mod, _ = self._hook()
        cs._STATE["flush_every"] = 10 ** 9   # keep disk writes out of the contended loop
        cs._STATE["seq_cap"] = 10 ** 9
        t = FakeTensor((4, 8))

        def drive():
            for _ in range(50):
                mod.op(t)

        threads = [threading.Thread(target=drive) for _ in range(4)]
        with _stderr():
            for th in threads:
                th.start()
            for th in threads:
                th.join()
        s = cs._STATE
        self.assertEqual(s["shape_counts"]["T(4, 8):torch.float16"], 200)
        self.assertEqual(len(s["sequence"]), 200)
        self.assertEqual(len(s["records"]), 1)

    def test_flush_snapshots_state_that_another_thread_is_still_growing(self):
        # _flush iterates shape_counts OUTSIDE the wrapper lock; it must copy under the lock or a
        # serving thread adding a shape raises "dict changed size during iteration".
        mod, _ = self._hook()
        cs._STATE["flush_every"] = 10 ** 9
        stop = threading.Event()

        def drive():
            i = 0
            while not stop.is_set():
                mod.op(FakeTensor((i % 64 + 1, 8)))
                i += 1

        th = threading.Thread(target=drive)
        with _stderr():
            mod.op(FakeTensor((1, 8)))   # so the first flush always has something to write
            th.start()
            try:
                for _ in range(20):
                    cs._flush()
            finally:
                stop.set()
                th.join()
        meta = self._meta()
        self.assertEqual(meta["num_distinct_shapes"], len(meta["shape_counts"]))


# --------------------------------------------------------------------------- #
# _flush / _maybe_flush -- the atexit dump and the crash-resilient partial dump
# --------------------------------------------------------------------------- #
class TestFlush(_RecorderTestCase):
    def _drive(self):
        mod, _ = self._hook(max_cases=5)
        with _stderr():
            mod.op(FakeTensor((4, 8)), scale=0.5)
            mod.op(FakeTensor((4, 8)), scale=0.5)
            mod.op(FakeTensor((2048, 8)), scale=0.5)
        return mod

    def test_atexit_dump_writes_the_expected_meta_json(self):
        mod = self._drive()
        with _stderr() as err:
            cs._flush()
        meta = self._meta()
        self.assertEqual(meta["target"], "fake_serving_layer:op")
        self.assertEqual(meta["module"], "fake_serving_layer")
        self.assertEqual(meta["attr"], "op")
        self.assertEqual(meta["num_cases"], 2)
        self.assertEqual(meta["total_calls_observed"], 3)
        self.assertEqual(meta["regimes_covered"], ["decode", "prefill"])
        self.assertEqual(meta["num_distinct_shapes"], 2)
        self.assertEqual(len(meta["call_sequence"]), 3)
        self.assertFalse(meta["graph_replayed"])
        self.assertEqual(meta["in_graph_calls"], 0)
        self.assertNotIn("reference_io", meta)
        self.assertNotIn("reference_io_sha256", meta)
        self.assertNotIn("oracle_complete", meta)
        self.assertFalse(meta["build"])
        self.assertIn("Do NOT edit", meta["note"])
        self.assertIn("flushed 2 case(s)", err.getvalue())

    def test_cases_carry_shapes_dtypes_and_their_real_call_count(self):
        self._drive()
        with _stderr():
            cs._flush()
        cases = {c["sig"]: c for c in self._meta()["cases"]}
        decode = cases["T(4, 8):torch.float16|scale=0.5"]
        self.assertEqual(decode["regime"], "decode")
        self.assertEqual(decode["input_shapes"], [[4, 8]])
        self.assertEqual(decode["input_dtypes"], ["torch.float16"])
        self.assertEqual(decode["count"], 2)
        self.assertEqual(cases["T(2048, 8):torch.float16|scale=0.5"]["count"], 1)

    def test_histogram_is_sorted_by_call_weight(self):
        self._drive()
        with _stderr():
            cs._flush()
        hist = self._meta()["shape_counts"]
        self.assertEqual([e["count"] for e in hist], [2, 1])
        self.assertEqual(hist[0]["sig"], "T(4, 8):torch.float16|scale=0.5")
        self.assertEqual(hist[0]["input_shapes"], [[4, 8]])

    def test_nested_tensor_args_are_walked_into_case_input_shapes(self):
        mod, _ = self._hook()
        with _stderr():
            mod.op([FakeTensor((4, 8)), FakeTensor((8, 8))],
                   extra={"w": FakeTensor((8, 2), dtype="torch.float32")})
            cs._flush()
        case = self._meta()["cases"][0]
        self.assertEqual(sorted(case["input_shapes"]), [[4, 8], [8, 2], [8, 8]])
        self.assertEqual(case["input_dtypes"], ["torch.float16", "torch.float32"])

    def test_flush_with_nothing_captured_writes_no_files(self):
        cs._STATE["out_dir"] = self.out_dir
        with _stderr() as err:
            cs._flush()
        self.assertIn("nothing to flush", err.getvalue())
        self.assertEqual(os.listdir(self.out_dir), [])

    def test_periodic_flush_lands_a_usable_capture_before_atexit(self):
        # An OOM/SIGKILL never fires atexit, so a small workload must already be on disk.
        mod, _ = self._hook()
        cs._STATE["flush_every"] = 1
        with _stderr():
            mod.op(FakeTensor((4, 8)))
        self.assertEqual(sorted(os.listdir(self.out_dir)), ["meta.json"])
        self.assertEqual(self._meta()["total_calls_observed"], 1)

    def test_periodic_flush_only_fires_on_the_boundary(self):
        mod, _ = self._hook()
        cs._STATE["flush_every"] = 3
        with _stderr():
            mod.op(FakeTensor((4, 8)))
            mod.op(FakeTensor((4, 8)))
        self.assertEqual(os.listdir(self.out_dir), [])
        with _stderr():
            mod.op(FakeTensor((4, 8)))
        self.assertIn("meta.json", os.listdir(self.out_dir))

    def test_no_flush_while_a_cuda_graph_is_being_captured(self):
        mod, _ = self._hook()
        cs._STATE["flush_every"] = 1
        self.torch.capturing = True
        with _stderr():
            mod.op(FakeTensor((4, 8)))
        self.assertEqual(os.listdir(self.out_dir), [])

    def test_maybe_flush_is_a_noop_before_any_call(self):
        self._hook()
        cs._STATE["flush_every"] = 1
        with _stderr():
            cs._maybe_flush()
        self.assertEqual(os.listdir(self.out_dir), [])

    def test_a_broken_out_dir_never_breaks_the_hooked_call(self):
        fd, blocker = tempfile.mkstemp()
        os.close(fd)
        self.addCleanup(os.unlink, blocker)
        mod, _ = self._hook(out_dir=os.path.join(blocker, "capture"))
        cs._STATE["flush_every"] = 1
        with _stderr() as err:
            out = mod.op(FakeTensor((4, 8)))
        self.assertEqual(out, "OUT")
        self.assertIn("periodic flush error (ignored)", err.getvalue())
        self.assertEqual(cs._STATE["shape_counts"]["T(4, 8):torch.float16"], 1)


# --------------------------------------------------------------------------- #
# _wrappable -- refusing native callables is what stops the mid-run SIGSEGV
# --------------------------------------------------------------------------- #
class TestWrappable(_RecorderTestCase):
    def test_pure_python_callables_are_wrappable(self):
        class _Layer:
            def op(self):
                return None

        def fn():
            return None

        self.assertTrue(cs._wrappable(fn))
        self.assertTrue(cs._wrappable(lambda: None))
        self.assertTrue(cs._wrappable(_Layer().op))
        self.assertTrue(cs._wrappable(functools.partial(fn)))

    def test_native_builtins_are_refused(self):
        self.assertFalse(cs._wrappable(len))
        self.assertFalse(cs._wrappable("".join))

    def test_triton_jit_internals_are_refused(self):
        class _JitFunction:
            cache = {}

            def fn(self):
                return None

            def __call__(self):
                return None

        self.assertFalse(cs._wrappable(_JitFunction()))

    def test_a_python_callable_instance_is_wrappable(self):
        class _CallableOp:
            def __call__(self):
                return None

        self.assertTrue(cs._wrappable(_CallableOp()))

    def test_a_builtin_type_used_as_a_factory_is_refused(self):
        self.assertFalse(cs._wrappable(list))

    def test_a_non_callable_is_refused(self):
        self.assertFalse(cs._wrappable(object()))


# --------------------------------------------------------------------------- #
# _make_wrapper -- must be transparent to introspection-driven native dispatch
# --------------------------------------------------------------------------- #
class TestMakeWrapper(_RecorderTestCase):
    def test_mirrors_identity_signature_and_extra_attributes(self):
        class _CallableOp:
            heuristic = "class-level"

            def __call__(self, a, b=1, *, c=2):
                return (a, b, c)

        def orig(a, b=1, *, c=2):
            return (a, b, c)

        orig.tuning_hint = "keep-me"
        w = cs._make_wrapper(orig)
        self.assertEqual(w.__name__, "orig")
        self.assertIs(w.__wrapped__, orig)
        self.assertEqual(str(w.__signature__), "(a, b=1, *, c=2)")
        self.assertEqual(w.tuning_hint, "keep-me")
        self.assertEqual(cs._make_wrapper(_CallableOp()).heuristic, "class-level")

    def test_wrapper_delegates_to_the_recorder(self):
        cs._STATE["orig"] = lambda *a, **k: "OUT"
        w = cs._make_wrapper(cs._STATE["orig"])
        with _stderr():
            self.assertEqual(w(FakeTensor((4, 8))), "OUT")
        self.assertEqual(cs._STATE["calls"], 1)

    def test_unresolvable_signature_and_faulting_attributes_are_tolerated(self):
        class _AttrTrap:
            @property
            def landmine(self):
                raise AttributeError("native attribute read faults")

        w = cs._make_wrapper(_AttrTrap())
        self.assertFalse(hasattr(w, "__signature__"))
        self.assertFalse(hasattr(w, "landmine"))
        self.assertIs(w.__wrapped__.__class__, _AttrTrap)


# --------------------------------------------------------------------------- #
# install / install_from_env
# --------------------------------------------------------------------------- #
class TestInstall(_RecorderTestCase):
    def test_install_replaces_the_attribute_and_records_state(self):
        def op(x):
            return "OUT"

        mod, err = self._hook(fn=op, max_cases=3)
        s = cs._STATE
        self.assertTrue(s["installed"])
        self.assertIs(s["orig"], op)
        self.assertIs(s["mod"], mod)
        self.assertEqual(s["attr"], "op")
        self.assertEqual(s["max_cases"], 3)
        self.assertEqual(s["out_dir"], self.out_dir)
        self.assertIsNot(mod.op, op)
        self.assertIs(mod.op.__wrapped__, op)
        self.assertIn("hooked fake_serving_layer:op", err)
        self.assertIn("up to 3 shape case(s)", err)
        self.assertIn("no tensor oracle", err)

    def test_install_is_idempotent(self):
        mod, _ = self._hook()
        wrapper = mod.op
        other = self._target_module(name="other_serving_layer")
        with _stderr() as err:
            cs.install("other_serving_layer:op", self.out_dir)
        self.assertIs(mod.op, wrapper)
        self.assertEqual(cs._STATE["target"], "fake_serving_layer:op")
        self.assertFalse(hasattr(other.op, "__wrapped__"))
        self.assertEqual(err.getvalue(), "")

    def test_restoring_the_original_stops_recording(self):
        # There is no uninstall(); _STATE["orig"] (== wrapper.__wrapped__) is the documented way back.
        mod, _ = self._hook()
        with _stderr():
            mod.op(FakeTensor((4, 8)))
        setattr(mod, cs._STATE["attr"], cs._STATE["orig"])
        with _stderr():
            self.assertEqual(mod.op(FakeTensor((16, 8))), "OUT")
        self.assertEqual(cs._STATE["calls"], 1)
        self.assertEqual(list(cs._STATE["shape_counts"]), ["T(4, 8):torch.float16"])

    def test_a_missing_attribute_fails_at_install(self):
        self._target_module()
        with self.assertRaises(AttributeError):
            cs.install("fake_serving_layer:not_an_op", self.out_dir)
        self.assertFalse(cs._STATE["installed"])

    def test_a_missing_module_fails_at_install(self):
        with self.assertRaises(ImportError):
            cs.install("no_such_serving_module_xyz:op", self.out_dir)
        self.assertFalse(cs._STATE["installed"])

    def test_a_native_target_is_refused_at_startup_not_mid_run(self):
        self._target_module(fn=len)
        with _env(CAPTURE_WRAP_UNSAFE=None):
            with self.assertRaises(RuntimeError) as ctx:
                cs.install("fake_serving_layer:op", self.out_dir)
        self.assertIn("refusing to wrap non-Python callable", str(ctx.exception))
        self.assertIn("SIGSEGV", str(ctx.exception))
        self.assertFalse(cs._STATE["installed"])

    def test_capture_wrap_unsafe_forces_a_native_target(self):
        mod = self._target_module(fn=len)
        with _env(CAPTURE_WRAP_UNSAFE="1"):
            with _stderr():
                cs.install("fake_serving_layer:op", self.out_dir)
        self.assertIsNot(mod.op, len)
        with _stderr():
            self.assertEqual(mod.op([1, 2, 3]), 3)
        self.assertEqual(cs._STATE["shape_counts"], {"list": 1})

    def test_install_registers_the_atexit_flush(self):
        registered = []
        real = cs.atexit
        cs.atexit = types.SimpleNamespace(register=registered.append)
        self.addCleanup(setattr, cs, "atexit", real)
        self._hook()
        self.assertEqual(registered, [cs._flush])

    def test_install_from_env_needs_both_target_and_out_dir(self):
        self._target_module()
        for target, out in (("fake_serving_layer:op", None), (None, self.out_dir), (None, None)):
            with _env(CAPTURE_TARGET=target, CAPTURE_OUT=out):
                cs.install_from_env()
            self.assertFalse(cs._STATE["installed"])

    def test_install_from_env_reads_target_out_and_max(self):
        mod = self._target_module()
        with _env(CAPTURE_TARGET="fake_serving_layer:op", CAPTURE_OUT=self.out_dir,
                  CAPTURE_MAX="2"):
            with _stderr():
                cs.install_from_env()
        self.assertTrue(cs._STATE["installed"])
        self.assertEqual(cs._STATE["max_cases"], 2)
        self.assertEqual(cs._STATE["out_dir"], self.out_dir)
        self.assertIsNot(mod.op, cs._STATE["orig"])

    def test_selection_capture_uses_a_process_local_artifact_directory(self):
        self._target_module()
        with _env(GEAK_SELECTION_TRACE=os.path.join(self.out_dir, "selection.json")):
            with _stderr():
                cs.install("fake_serving_layer:op", self.out_dir)
        expected_prefix = os.path.join(
            self.out_dir, f"capture.pid-{os.getpid()}.rank-")
        self.assertTrue(cs._STATE["out_dir"].startswith(expected_prefix))


# --------------------------------------------------------------------------- #
# import-time self-install (the overlay PYTHONPATH / sitecustomize entry point)
# --------------------------------------------------------------------------- #
class TestImportTimeInstall(_RecorderTestCase):
    def _fresh(self, name, **env):
        with _env(**env):
            with _stderr() as err:
                mod = _load(name, "capture_shapes.py")
        self.addCleanup(atexit.unregister, mod._flush)
        return mod, err.getvalue()

    def test_env_configured_overlay_hooks_the_target_on_import(self):
        target = self._target_module(name="overlay_serving_layer")
        orig = target.op
        mod, err = self._fresh("capture_shapes_overlay",
                               CAPTURE_TARGET="overlay_serving_layer:op",
                               CAPTURE_OUT=self.out_dir, CAPTURE_MAX="2")
        self.assertTrue(mod._STATE["installed"])
        self.assertEqual(mod._STATE["max_cases"], 2)
        self.assertEqual(mod._STATE["out_dir"], self.out_dir)
        self.assertIs(target.op.__wrapped__, orig)
        self.assertIn("hooked overlay_serving_layer:op", err)
        with _stderr():
            self.assertEqual(target.op(FakeTensor((4, 8))), "OUT")
        self.assertEqual(mod._STATE["shape_counts"], {"T(4, 8):torch.float16": 1})

    def test_a_bad_target_is_reported_and_never_kills_server_startup(self):
        mod, err = self._fresh("capture_shapes_badtarget",
                               CAPTURE_TARGET="no_such_serving_module_xyz:op",
                               CAPTURE_OUT=self.out_dir)
        self.assertFalse(mod._STATE["installed"])
        self.assertIn("install_from_env failed", err)

    def test_no_env_means_no_hook_on_import(self):
        mod, err = self._fresh("capture_shapes_inert",
                               CAPTURE_TARGET=None, CAPTURE_OUT=None)
        self.assertFalse(mod._STATE["installed"])
        self.assertEqual(err, "")

    def test_decode_cutoff_is_env_overridable_at_import(self):
        mod, _ = self._fresh("capture_shapes_cutoff", CAPTURE_TARGET=None, CAPTURE_OUT=None,
                             CAPTURE_DECODE_LEAD_MAX="8")
        self.assertEqual(mod._STATE["decode_lead_max"], 8)


class TestReclaim(_RecorderTestCase):
    """Reclaim: promote the selected process meta, drop every process-local capture dir."""

    def test_promote_and_reclaim_leaves_one_authoritative_meta(self):
        task = self.out_dir
        keep = None
        for pid in (111, 222):
            cap = os.path.join(task, f"capture.pid-{pid}.rank-0")
            os.makedirs(cap)
            meta = os.path.join(cap, "meta.json")
            with open(meta, "w") as fh:
                json.dump({"process_id": pid, "target": "m:op", "pad": "x" * 400}, fh)
            if pid == 111:
                keep = meta
        # leftover atomic tmp
        with open(os.path.join(task, "meta.json.tmp-1-2"), "wb") as fh:
            fh.write(b"TMP")
        tel = cs.promote_and_reclaim(task, keep_meta_path=keep, promote=True)
        self.assertTrue(tel["promoted"])
        with open(os.path.join(task, "meta.json")) as fh:
            self.assertEqual(json.load(fh)["process_id"], 111)
        self.assertEqual(list(cs.iter_capture_dirs(task)), [])
        self.assertFalse(os.path.exists(os.path.join(task, "meta.json.tmp-1-2")))
        self.assertTrue(os.path.isfile(os.path.join(task, "capture_telemetry.json")))
        self.assertGreater(tel["bytes_reclaimed"], 0)

    def test_cleanup_without_promote_removes_all_capture_dirs(self):
        task = self.out_dir
        for pid in (1, 2):
            cap = os.path.join(task, "_selcap", f"capture.pid-{pid}.rank-0")
            os.makedirs(cap)
            with open(os.path.join(cap, "reference_io.pt"), "wb") as fh:
                fh.write(b"Z" * 1024)
        tel = cs.cleanup_task_capture_artifacts(task, promote=False)
        self.assertEqual(list(cs.iter_capture_dirs(task)), [])
        self.assertFalse(os.path.isfile(os.path.join(task, "reference_io.pt")))
        self.assertGreaterEqual(tel["bytes_reclaimed"], 2048)

    def test_dir_bytes_skips_missing_links_and_link_children(self):
        self.assertEqual(cs._dir_bytes(""), 0)
        missing = os.path.join(self.out_dir, "nope")
        self.assertEqual(cs._dir_bytes(missing), 0)
        file_path = os.path.join(self.out_dir, "one.bin")
        with open(file_path, "wb") as fh:
            fh.write(b"abcd")
        self.assertEqual(cs._dir_bytes(file_path), 4)
        link = os.path.join(self.out_dir, "alink")
        os.symlink(file_path, link)
        self.assertEqual(cs._dir_bytes(link), 0)
        tree = os.path.join(self.out_dir, "tree")
        os.makedirs(tree)
        with open(os.path.join(tree, "a.bin"), "wb") as fh:
            fh.write(b"12345")
        os.symlink(file_path, os.path.join(tree, "skip.link"))
        self.assertEqual(cs._dir_bytes(tree), 5)

    def test_iter_capture_dirs_and_remove_path_edge_cases(self):
        self.assertEqual(list(cs.iter_capture_dirs(os.path.join(self.out_dir, "missing"))), [])
        tel = {"removed_paths": [], "bytes_reclaimed": 0}
        self.assertEqual(cs._remove_path("", tel), 0)
        self.assertEqual(cs._remove_path(os.path.join(self.out_dir, "gone"), tel), 0)
        lonely = os.path.join(self.out_dir, "lonely.bin")
        with open(lonely, "wb") as fh:
            fh.write(b"ZZ")
        self.assertEqual(cs._remove_path(lonely, tel), 2)
        self.assertFalse(os.path.exists(lonely))

        bad = os.path.join(self.out_dir, "capture.pid-9.rank-0")
        os.makedirs(bad)
        with open(os.path.join(bad, "x"), "wb") as fh:
            fh.write(b"abc")
        real_rmtree = shutil.rmtree

        def boom(path, ignore_errors=False):
            raise OSError("denied")

        shutil.rmtree = boom
        try:
            tel2 = {"removed_paths": [], "bytes_reclaimed": 0, "errors": []}
            self.assertEqual(cs._remove_path(bad, tel2), 0)
            self.assertTrue(tel2["errors"])
        finally:
            shutil.rmtree = real_rmtree

    def test_promote_overwrites_an_existing_task_root_meta(self):
        task = self.out_dir
        with open(os.path.join(task, "meta.json"), "w") as fh:
            json.dump({"process_id": 0}, fh)
        cap = os.path.join(task, "capture.pid-1.rank-0")
        os.makedirs(cap)
        meta = os.path.join(cap, "meta.json")
        with open(meta, "w") as fh:
            json.dump({"process_id": 1, "num_cases": 3}, fh)
        got = cs.promote_selected_meta(task, meta)
        self.assertTrue(got["promoted_meta"])
        with open(os.path.join(task, "meta.json")) as fh:
            self.assertEqual(json.load(fh)["process_id"], 1)

    def test_cleanup_missing_task_dir_and_telemetry_write_failure(self):
        missing = os.path.join(self.out_dir, "no-task")
        tel = cs.cleanup_task_capture_artifacts(missing)
        self.assertIn("missing task_dir", tel["errors"][0])

        task = os.path.join(self.out_dir, "task")
        os.makedirs(task)
        real_write = cs._write_json

        def boom(_path, _payload):
            raise OSError("ro fs")

        cs._write_json = boom
        try:
            tel2 = cs.cleanup_task_capture_artifacts(task, promote=False)
        finally:
            cs._write_json = real_write
        self.assertTrue(any("telemetry write failed" in e for e in tel2["errors"]))

    def test_reclaim_workspace_captures_sweeps_every_task(self):
        eval_dir = self.out_dir
        kernels = os.path.join(eval_dir, "kernels")
        os.makedirs(kernels)
        # non-task noise
        with open(os.path.join(kernels, "README"), "w") as fh:
            fh.write("x")
        task = os.path.join(kernels, "demo_task")
        os.makedirs(task)
        leftover = os.path.join(task, "capture.pid-77.rank-0")
        os.makedirs(leftover)
        with open(os.path.join(leftover, "meta.json"), "w") as fh:
            fh.write("{}" + " " * 500)
        tel = cs.reclaim_workspace_captures(eval_dir)
        self.assertGreater(tel["bytes_reclaimed"], 0)
        self.assertNotIn("over_budget", tel)
        self.assertFalse(os.path.isdir(leftover))
        self.assertTrue(os.path.isfile(
            os.path.join(eval_dir, "capture_workspace_telemetry.json")))

        empty = os.path.join(self.out_dir, "empty_eval")
        os.makedirs(empty)
        tel2 = cs.reclaim_workspace_captures(empty)
        self.assertIn("missing kernels dir", tel2["errors"][0])

    def test_wrapper_uses_record_function_when_available(self):
        entered = []

        @contextlib.contextmanager
        def rf(name):
            entered.append(name)
            yield

        self.torch.profiler = types.SimpleNamespace(record_function=rf)
        mod, _ = self._hook()
        with _stderr():
            mod.op(FakeTensor((2, 2)))
        self.assertTrue(any(n.startswith("GEAK_TARGET::") for n in entered))

    def test_install_honors_flush_every(self):
        self._target_module(name="flush_every_mod")
        with _env(CAPTURE_FLUSH_EVERY="3"):
            with _stderr():
                cs.install("flush_every_mod:op", self.out_dir)
        self.assertEqual(cs._STATE["flush_every"], 3)

    def test_cli_cleanup_task_dir_and_reclaim_workspace(self):
        task = os.path.join(self.out_dir, "cli_task")
        os.makedirs(task)
        cap = os.path.join(task, "capture.pid-5.rank-0")
        os.makedirs(cap)
        meta = os.path.join(cap, "meta.json")
        with open(meta, "w") as fh:
            json.dump({"process_id": 5}, fh)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cs._cli([
                "--cleanup-task-dir", task,
                "--keep-meta", meta,
                "--promote",
            ])
        self.assertEqual(rc, 0)
        with open(os.path.join(task, "meta.json")) as fh:
            self.assertEqual(json.load(fh)["process_id"], 5)

        eval_dir = os.path.join(self.out_dir, "cli_eval")
        kernels = os.path.join(eval_dir, "kernels", "x_task")
        os.makedirs(kernels)
        leftover = os.path.join(kernels, "capture.pid-8.rank-0")
        os.makedirs(leftover)
        with open(os.path.join(leftover, "meta.json"), "w") as fh:
            fh.write("{}")
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            rc2 = cs._cli([
                "--reclaim-workspace", eval_dir,
            ])
        self.assertEqual(rc2, 0)
        self.assertFalse(os.path.isdir(leftover))

        with self.assertRaises(SystemExit):
            cs._cli([])

    def test_dir_bytes_tolerates_getsize_oserror(self):
        tree = os.path.join(self.out_dir, "getsize_tree")
        os.makedirs(tree)
        victim = os.path.join(tree, "a.bin")
        with open(victim, "wb") as fh:
            fh.write(b"abc")
        real_getsize = os.path.getsize

        def flaky(path):
            if path == victim:
                raise OSError("vanished")
            return real_getsize(path)

        os.path.getsize = flaky
        try:
            self.assertEqual(cs._dir_bytes(tree), 0)
        finally:
            os.path.getsize = real_getsize

    def test_reclaim_workspace_telemetry_write_failure(self):
        eval_dir = os.path.join(self.out_dir, "tel_fail_eval")
        kernels = os.path.join(eval_dir, "kernels", "y_task")
        os.makedirs(kernels)
        real_write = cs._write_json

        def boom(_path, _payload):
            raise OSError("disk full")

        cs._write_json = boom
        try:
            tel = cs.reclaim_workspace_captures(eval_dir)
        finally:
            cs._write_json = real_write
        self.assertTrue(any("disk full" in e for e in tel["errors"]))

    def test_flush_meta_tmp_cleanup_paths(self):
        """A failing meta replace must raise AND leave no .tmp- litter behind."""
        mod, _ = self._hook(name="fake_serving_meta_cleanup")
        with _stderr():
            mod.op(FakeTensor((2, 2)))
        real_replace = os.replace

        def fail_meta(src, dst):
            if os.path.basename(dst) == "meta.json":
                raise OSError("meta replace failed")
            return real_replace(src, dst)

        os.replace = fail_meta
        try:
            with self.assertRaises(OSError):
                cs._flush()
        finally:
            os.replace = real_replace
        leftovers = [n for n in os.listdir(self.out_dir) if ".tmp-" in n]
        self.assertEqual(leftovers, [])

    def test_flush_tolerates_a_failing_tmp_unlink(self):
        """If even the tmp unlink raises, the original OSError still propagates."""
        mod, _ = self._hook(name="fake_serving_meta_unlink")
        with _stderr():
            mod.op(FakeTensor((2, 2)))
        real_replace, real_unlink, real_exists = os.replace, os.unlink, os.path.exists

        def fail_meta(src, dst):
            if os.path.basename(dst) == "meta.json":
                raise OSError("meta replace failed")
            return real_replace(src, dst)

        def exists_tmp(path):
            return True if "meta.json.tmp-" in path else real_exists(path)

        def unlink_meta_tmp(path):
            if "meta.json.tmp-" in path:
                raise OSError("meta unlink busy")
            return real_unlink(path)

        os.replace, os.path.exists, os.unlink = fail_meta, exists_tmp, unlink_meta_tmp
        try:
            with self.assertRaises(OSError):
                cs._flush()
        finally:
            os.replace, os.path.exists, os.unlink = real_replace, real_exists, real_unlink


if __name__ == "__main__":
    unittest.main(verbosity=2)
