#!/usr/bin/env python3
"""Unit tests for harness_lib.py -- the measurement library the CI judge's verdict is computed from.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_harness_lib.py

harness_lib is vendored into every extracted `<short_name>_task/` and is the SINGLE source of truth for
how an isolated kernel task decides "this candidate is faster and still correct". A wrong number here is
not a crash, it is a wrong ACCEPT/REJECT on every kernel GEAK produces. So what is pinned here is the
arithmetic and the verdict shape, not GPU behaviour:

  - regime-driven synthesis : _DTYPE_BYTES / pack_x / regime_spec / regime_dtype -- the paged-KV inner
                             factor `x` and the MI300(fnuz)-vs-MI355(OCP fn) fp8 fork, which is the one
                             hardware-specific axis in the operand builder
  - timing                  : time_op's three timers (captured-graph replay / cuda-event / wall) with
                             SCRIPTED event times and a scripted perf_counter, so the reported median is
                             exact; plus the cache flush that makes a memory-bound kernel read cold
  - correctness             : `correct`'s RMS noise floor (a max-scaled floor would mask the small-element
                             error on a spiky tensor -- pinned with a hand-computed spiky case), the
                             shared-`static_out` defeats (assert_independent_outputs /
                             check_correct_multi / check_correct_sequence / check_graph_replay), and
                             run_correctness's FAIL-CLOSED replay gate
  - the primary metric      : amdahl_ceiling / amdahl_check and serving_weighted_speedup -- the
                             workload-weighted speedup, its served-regimes gate and its pseudo-identity
                             guard, which is the number the A/B judge actually reads

torch is injected into sys.modules as a fake: harness_lib imports it lazily inside `_torch()` (and inside
`compiled_op`), so the whole library runs on a CPU-only box. The fake tensor carries REAL element values
over an explicit shape, which is what makes the statistical assertions exact (hand-computed RMS, medians,
max relative error) instead of shape-only smoke checks. CUDA-event times and perf_counter are scripted
lists, so `time_op` returns a median we chose. `time.perf_counter` is replaced on the module, so nothing
ever sleeps or polls.
"""
import contextlib
import importlib.util
import io
import itertools
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
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


hl = _load("harness_lib", "harness_lib.py")


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
def _patched(obj, **kw):
    """Swap attributes on a module/object for the duration of the block."""
    old = {k: getattr(obj, k) for k in kw}
    try:
        for k, v in kw.items():
            setattr(obj, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


# --------------------------------------------------------------------------- #
# fake torch: dtypes
# --------------------------------------------------------------------------- #
class _Dtype:
    def __init__(self, name, itemsize, is_floating_point=True, vmax=None):
        self.name = name
        self.itemsize = itemsize
        self.is_floating_point = is_floating_point
        self.max = vmax

    def __repr__(self):
        return "torch.%s" % self.name


BF16 = _Dtype("bfloat16", 2, True, 3.3895313892515355e38)
FP16 = _Dtype("float16", 2, True, 65504.0)
FP32 = _Dtype("float32", 4, True, 3.4028234663852886e38)
FP64 = _Dtype("float64", 8, True, 1.7976931348623157e308)
INT8 = _Dtype("int8", 1, False, 127)
UINT8 = _Dtype("uint8", 1, False, 255)
INT32 = _Dtype("int32", 4, False, 2147483647)
# The fp8 set REAL torch exposes: e4m3 has an OCP "fn" spelling, e5m2 does not.
FP8_E4M3FN = _Dtype("float8_e4m3fn", 1, True, 448.0)
FP8_E5M2 = _Dtype("float8_e5m2", 1, True, 57344.0)
FP8_E4M3FNUZ = _Dtype("float8_e4m3fnuz", 1, True, 240.0)
FP8_E5M2FNUZ = _Dtype("float8_e5m2fnuz", 1, True, 57344.0)

# Every fake randn() cycles this pattern, so amax/RMS over a synthesized operand is a known number.
# 4.0 (not 3.0) because 4.0 * 0.1 is exactly representable and 3.0 * 0.1 is not.
RANDN_PATTERN = (0.0, 1.0, -4.0, 2.0)

_PTRS = itertools.count(0x7F0000001000, 0x100)


# --------------------------------------------------------------------------- #
# fake torch: the tensor
# --------------------------------------------------------------------------- #
class _T:
    """A fake tensor with REAL element values, so every statistic harness_lib computes is exact.

    Elements are materialized lazily from `fill` when no explicit data is given, so flush_cache's
    512MB buffer can be asserted (numel/zero_) without ever allocating 134M floats.
    """

    def __init__(self, shape, data=None, fill=0.0, dtype=FP32, device="cpu"):
        self.shape = tuple(int(s) for s in shape)
        self._data = None if data is None else [float(v) for v in data]
        self._fill = float(fill)
        self.dtype = dtype
        self.device = device
        self._ptr = next(_PTRS)

    # ---- structure
    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    def tolist(self):
        return [self._fill] * self.numel() if self._data is None else list(self._data)

    def data_ptr(self):
        return self._ptr

    def element_size(self):
        return self.dtype.itemsize

    def detach(self):
        """A VIEW: shares storage, so data_ptr is unchanged -- what the independence check keys off."""
        view = _T(self.shape, self._data, self._fill, self.dtype, self.device)
        view._data = self._data
        view._ptr = self._ptr
        return view

    def clone(self):
        return _T(self.shape, self.tolist(), dtype=self.dtype, device=self.device)

    def to(self, *args, **kw):
        dtype, device = kw.get("dtype"), kw.get("device")
        for a in args:
            if isinstance(a, _Dtype):
                dtype = a
            elif isinstance(a, str):
                device = a
        return _T(self.shape, self.tolist(), dtype=dtype or self.dtype, device=device or self.device)

    def float(self):
        return self.to(FP32)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _T(tuple(int(s) for s in shape), self.tolist(),
                  dtype=self.dtype, device=self.device)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            data = self.tolist()[idx]
            return _T((len(data),), data, dtype=self.dtype, device=self.device)
        raise TypeError("unsupported fake index %r" % (idx,))

    def zero_(self):
        self._data = None
        self._fill = 0.0
        return self

    def overwrite_(self, values):
        """Test-side in-place write, standing in for a kernel writing into a persistent buffer."""
        self._data = [float(v) for v in values]
        return self

    # ---- elementwise / reductions
    def _bin(self, other, op):
        a = self.tolist()
        if isinstance(other, _T):
            b, shape = other.tolist(), self.shape
            if len(a) != len(b):
                if len(b) == 1:
                    b = b * len(a)
                elif len(a) == 1:
                    a, shape = a * len(b), other.shape
                else:
                    raise RuntimeError("fake broadcast %s vs %s" % (self.shape, other.shape))
            return _T(shape, [op(x, y) for x, y in zip(a, b)], dtype=self.dtype, device=self.device)
        return _T(self.shape, [op(x, float(other)) for x in a],
                  dtype=self.dtype, device=self.device)

    def __add__(self, other):
        return self._bin(other, lambda x, y: x + y)

    __radd__ = __add__

    def __sub__(self, other):
        return self._bin(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self._bin(other, lambda x, y: y - x)

    def __mul__(self, other):
        return self._bin(other, lambda x, y: x * y)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._bin(other, lambda x, y: x / y)

    def __neg__(self):
        return _T(self.shape, [-v for v in self.tolist()], dtype=self.dtype, device=self.device)

    def __le__(self, other):
        return self._bin(other, lambda x, y: float(x <= y))

    def div(self, other):
        return self.__truediv__(other)

    def abs(self):
        return _T(self.shape, [abs(v) for v in self.tolist()], dtype=self.dtype, device=self.device)

    def pow(self, n):
        return _T(self.shape, [v ** n for v in self.tolist()], dtype=self.dtype, device=self.device)

    def sqrt(self):
        return _T(self.shape, [math.sqrt(v) for v in self.tolist()],
                  dtype=self.dtype, device=self.device)

    def clamp(self, lo, hi):
        return _T(self.shape, [min(max(v, lo), hi) for v in self.tolist()],
                  dtype=self.dtype, device=self.device)

    def clamp_min(self, lo):
        return _T(self.shape, [max(v, lo) for v in self.tolist()],
                  dtype=self.dtype, device=self.device)

    def _reduce(self, fn):
        return _T((), [fn(self.tolist())], dtype=self.dtype, device=self.device)

    def mean(self):
        return self._reduce(lambda v: sum(v) / len(v))

    def amax(self, dim=None):
        return self._reduce(max)

    def max(self):
        return self._reduce(max)

    def all(self):
        return all(v != 0.0 for v in self.tolist())

    def item(self):
        return self.tolist()[0]

    def __repr__(self):
        return "_T(shape=%s, dtype=%s, data=%s)" % (self.shape, self.dtype, self.tolist())


# --------------------------------------------------------------------------- #
# fake torch: the module
# --------------------------------------------------------------------------- #
class _Stack:
    """A fake torch for harness_lib: real tensor math, SCRIPTED cuda-event times, and a recording
    stream/graph surface, so timing and capture decisions are deterministic on a CPU box."""

    def __init__(self, cuda=False, arch="gfx942:sramecc+:xnack-", event_ms=(1.0,)):
        self.cuda = cuda
        self.arch = arch
        self.available_raises = False
        self.props_raises = False
        self.capture_raises = False
        self.event_ms = list(event_ms)
        self._event_i = 0
        self.syncs = 0
        self.replays = 0
        self.capturing = False      # True inside torch.cuda.graph(g)
        self.recording = []         # device work registered during the current capture
        self.events = []            # ("record"|"synchronize", ...) in order
        self.seeds = []             # (device, seed) per Generator.manual_seed
        self.stream_ops = []        # wait_stream / stream-context / graph-context, in order
        self.mod = self._build()

    def _next_event_ms(self):
        ms = self.event_ms[self._event_i % len(self.event_ms)]
        self._event_i += 1
        return ms

    def _build(self):
        stack = self
        torch = types.ModuleType("torch")
        for dt in (BF16, FP16, FP32, FP64, INT8, UINT8, INT32,
                   FP8_E4M3FN, FP8_E5M2, FP8_E4M3FNUZ, FP8_E5M2FNUZ):
            setattr(torch, dt.name, dt)

        def _is_available():
            if stack.available_raises:
                raise RuntimeError("HIP runtime not initialized")
            return stack.cuda

        def _synchronize():
            stack.syncs += 1

        def _get_device_properties(idx):
            if stack.props_raises:
                raise RuntimeError("invalid device ordinal")
            return types.SimpleNamespace(gcnArchName=stack.arch, index=idx)

        class _Event:
            def __init__(self, enable_timing=False):
                self.enable_timing = enable_timing

            def record(self):
                stack.events.append(("record", id(self)))

            def synchronize(self):
                stack.events.append(("synchronize", id(self)))

            def elapsed_time(self, other):
                return stack._next_event_ms()

        class _Stream:
            def __init__(self, name="side"):
                self.name = name

            def wait_stream(self, other):
                stack.stream_ops.append(("wait_stream", self.name, other.name))

        default_stream = _Stream("default")

        @contextlib.contextmanager
        def _stream_ctx(s):
            stack.stream_ops.append(("enter_stream", s.name))
            try:
                yield
            finally:
                stack.stream_ops.append(("exit_stream", s.name))

        class _CUDAGraph:
            def __init__(self):
                if stack.capture_raises:
                    raise RuntimeError("cuda graph capture unsupported on this image")
                self.work = []

            def replay(self):
                # A real replay re-issues the recorded DEVICE work without re-entering the Python
                # launcher, which is exactly what makes a static-buffer kernel go stale.
                stack.replays += 1
                for fn in self.work:
                    fn()

        @contextlib.contextmanager
        def _graph_ctx(g):
            stack.stream_ops.append(("enter_graph", id(g)))
            stack.capturing = True
            stack.recording = []
            try:
                yield
            finally:
                stack.capturing = False
                g.work = list(stack.recording)
                stack.recording = []
                stack.stream_ops.append(("exit_graph", id(g)))

        class _Generator:
            def __init__(self, device="cpu"):
                self.device = device
                self.seed = None

            def manual_seed(self, s):
                self.seed = int(s)
                stack.seeds.append((self.device, self.seed))
                return self

        def _randn(*shape, **kw):
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = tuple(shape[0])
            t = _T(shape, dtype=kw.get("dtype") or FP32, device=kw.get("device") or "cpu")
            n = t.numel()
            t._data = [RANDN_PATTERN[i % len(RANDN_PATTERN)] for i in range(n)]
            return t

        def _filled(val):
            def make(*shape, **kw):
                if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                    shape = tuple(shape[0])
                return _T(shape, fill=val, dtype=kw.get("dtype") or FP32,
                          device=kw.get("device") or "cpu")
            return make

        def _randperm(pool, generator=None, device=None):
            # A deterministic, NON-contiguous permutation of range(pool) so the block table under
            # test is asserted on exact values rather than on "some shuffle happened".
            pool = int(pool)
            seed = getattr(generator, "seed", 0) or 0
            return _T((pool,), [(pool - 1 - i + seed) % pool for i in range(pool)],
                      dtype=INT32, device=device or "cpu")

        def _equal(a, b):
            return tuple(a.shape) == tuple(b.shape) and a.tolist() == b.tolist()

        torch.cuda = types.SimpleNamespace(
            is_available=_is_available, synchronize=_synchronize,
            get_device_properties=_get_device_properties, Event=_Event, Stream=_Stream,
            current_stream=lambda: default_stream, stream=_stream_ctx,
            CUDAGraph=_CUDAGraph, graph=_graph_ctx)
        torch.finfo = lambda dt: types.SimpleNamespace(max=dt.max, min=-dt.max)
        torch.iinfo = lambda dt: types.SimpleNamespace(max=dt.max, min=-dt.max - 1)
        torch.Generator = _Generator
        torch.randn = _randn
        torch.ones = _filled(1.0)
        torch.zeros = _filled(0.0)
        torch.empty = _filled(0.0)
        torch.randperm = _randperm
        torch.equal = _equal
        torch.tensor = lambda data, dtype=None: _T((len(data),), list(data), dtype=dtype or FP32)
        return torch


class _Clock:
    """Deterministic stand-in for time.perf_counter: call i returns the running sum of the deltas
    consumed so far, so a timed sample's wall ms is exactly the delta between its two reads."""

    def __init__(self, deltas=(0.001,)):
        self.deltas = list(deltas)
        self.t = 0.0
        self.i = 0

    def perf_counter(self):
        now = self.t
        self.t += self.deltas[self.i % len(self.deltas)]
        self.i += 1
        return now


def _wall_deltas(walls_ms):
    """Deltas that make the k-th timed sample measure walls_ms[k] (each sample reads the clock twice)."""
    out = []
    for w in walls_ms:
        out += [w / 1e3, 0.0]
    return out


def _echo_call(args):
    """A well-behaved launcher: fn(args) -> a FRESH output tensor (the documented contract)."""
    return _T((len(args),), list(args))


def _case(sig, args, ref_values=None):
    return {"sig": sig, "args": tuple(args),
            "ref": _T((len(args),), list(args if ref_values is None else ref_values))}


class _StaticOutCall:
    """The graph-replay `static_out` cheat: ONE persistent buffer, overwritten on every call."""

    def __init__(self, n=2):
        self.buf = _T((n,), [0.0] * n)
        self.calls = 0

    def __call__(self, args):
        self.calls += 1
        self.buf.overwrite_(args)
        return self.buf


class _HarnessTestCase(unittest.TestCase):
    """Fake torch in sys.modules, a scripted clock in place of time.perf_counter, and a pristine
    _CACHE_FLUSH_BUF (a module global the flush path mutates and would leak between tests)."""

    CUDA = False
    EVENT_MS = (1.0,)
    WALL_MS = None                      # per-sample wall ms; None -> a flat 1.0 ms every sample

    def setUp(self):
        self.stack = _Stack(cuda=self.CUDA, event_ms=self.EVENT_MS)
        self.torch = self.stack.mod
        self._prev_torch = sys.modules.get("torch", _MISSING)
        sys.modules["torch"] = self.torch
        self.addCleanup(self._restore_torch)
        self.clock = _Clock(_wall_deltas(self.WALL_MS) if self.WALL_MS else (0.001,))
        self.addCleanup(setattr, hl, "time", hl.time)
        hl.time = self.clock
        hl._CACHE_FLUSH_BUF = None
        self.addCleanup(setattr, hl, "_CACHE_FLUSH_BUF", None)

    def _restore_torch(self):
        if self._prev_torch is _MISSING:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self._prev_torch

    def _counting_call(self):
        box = {"n": 0}

        def call():
            box["n"] += 1
        return box, call


class _CudaTestCase(_HarnessTestCase):
    CUDA = True


# --------------------------------------------------------------------------- #
# _torch / sync -- the lazy import that lets the library load on a CPU box
# --------------------------------------------------------------------------- #
class TestTorchAndSync(_HarnessTestCase):
    def test_torch_is_imported_lazily_from_sys_modules(self):
        self.assertIs(hl._torch(), self.torch)

    def test_sync_is_a_noop_without_a_device(self):
        hl.sync(self.torch)
        hl.sync()
        self.assertEqual(self.stack.syncs, 0)

    def test_sync_synchronizes_when_a_device_is_visible(self):
        self.stack.cuda = True
        hl.sync(self.torch)
        hl.sync()
        self.assertEqual(self.stack.syncs, 2)


# --------------------------------------------------------------------------- #
# detect_arch / fp8_is_fnuz -- the MI300(fnuz) vs MI355(OCP fn) fork
# --------------------------------------------------------------------------- #
class TestArch(_HarnessTestCase):
    def test_no_visible_device_is_the_empty_arch(self):
        self.assertEqual(hl.detect_arch(self.torch), "")
        self.assertEqual(hl.detect_arch(), "")

    def test_arch_string_is_stripped_of_target_features_and_lowercased(self):
        self.stack.cuda = True
        self.stack.arch = "GFX950:SRAMECC+:XNACK-"
        self.assertEqual(hl.detect_arch(self.torch), "gfx950")

    def test_a_raising_device_query_is_the_empty_arch_not_a_crash(self):
        self.stack.cuda = True
        self.stack.props_raises = True
        self.assertEqual(hl.detect_arch(self.torch), "")

    def test_cdna3_and_gfx90a_use_the_amd_fnuz_fp8(self):
        for arch in ("gfx942", "gfx942:sramecc+", "gfx940", "gfx941", "gfx90a", "GFX942"):
            self.assertTrue(hl.fp8_is_fnuz(arch), arch)

    def test_cdna4_and_unknown_archs_use_the_ocp_fp8(self):
        for arch in ("gfx950", "gfx1100", "", None):
            self.assertFalse(hl.fp8_is_fnuz(arch), arch)


# --------------------------------------------------------------------------- #
# regime_dtype -- a regime dtype STRING to a torch dtype
# --------------------------------------------------------------------------- #
class TestRegimeDtype(_HarnessTestCase):
    def test_non_fp8_names_map_to_their_torch_dtype(self):
        expect = {"bf16": BF16, "bfloat16": BF16, "fp16": FP16, "float16": FP16, "half": FP16,
                  "fp32": FP32, "float32": FP32, "float": FP32, "int8": INT8, "uint8": UINT8}
        for name, dt in expect.items():
            self.assertIs(hl.regime_dtype(name, self.torch), dt, name)

    def test_names_are_case_insensitive(self):
        self.assertIs(hl.regime_dtype("BF16", self.torch), BF16)

    def test_unknown_name_falls_back_to_the_compute_dtype(self):
        self.assertIs(hl.regime_dtype("float4_exotic", self.torch), BF16)

    def test_integer_names_fall_back_when_the_image_lacks_them(self):
        del self.torch.int8
        self.assertIs(hl.regime_dtype("int8", self.torch), BF16)

    def test_bare_fp8_follows_the_arch_not_a_hardcoded_variant(self):
        self.assertIs(hl.regime_dtype("fp8", self.torch, arch="gfx942"), FP8_E4M3FNUZ)
        self.assertIs(hl.regime_dtype("fp8", self.torch, arch="gfx950"), FP8_E4M3FN)
        self.assertIs(hl.regime_dtype("fp8_e4m3", self.torch, arch="gfx942"), FP8_E4M3FNUZ)
        self.assertIs(hl.regime_dtype("e5m2", self.torch, arch="gfx942"), FP8_E5M2FNUZ)

    def test_bare_fp8_detects_the_arch_when_none_is_supplied(self):
        self.stack.cuda = True
        self.stack.arch = "gfx942"
        self.assertIs(hl.regime_dtype("fp8", self.torch), FP8_E4M3FNUZ)
        self.stack.arch = "gfx950"
        self.assertIs(hl.regime_dtype("fp8", self.torch), FP8_E4M3FN)

    def test_an_explicit_variant_wins_over_arch_detection(self):
        # A pre-quantized checkpoint that declares its own format must be honoured literally, even on
        # the other arch -- otherwise the synthesized operand does not match the checkpoint.
        self.assertIs(hl.regime_dtype("fp8_e4m3fnuz", self.torch, arch="gfx950"), FP8_E4M3FNUZ)
        self.assertIs(hl.regime_dtype("fp8_e4m3fn", self.torch, arch="gfx942"), FP8_E4M3FN)
        self.assertIs(hl.regime_dtype("fp8_e5m2fnuz", self.torch, arch="gfx950"), FP8_E5M2FNUZ)

    def test_bare_e5m2_on_an_ocp_arch_degrades_to_bf16(self):
        # SOURCE GAP, pinned as-is: the OCP fp8 pair torch exposes is float8_e4m3fn + float8_e5m2 --
        # there is NO float8_e5m2fn. The suffix rule appends "fn" to both mantissa forms, so a bare
        # "fp8_e5m2"/"e5m2" KV dtype on gfx950 resolves to bf16 (silently 2x the intended KV bytes,
        # which also doubles pack_x) instead of torch.float8_e5m2.
        self.assertIs(hl.regime_dtype("fp8_e5m2", self.torch, arch="gfx950"), BF16)
        self.assertIs(hl.regime_dtype("e5m2", self.torch, arch="gfx950"), BF16)


# --------------------------------------------------------------------------- #
# _bytes_of / pack_x -- the paged-KV inner factor, keyed off the KV dtype
# --------------------------------------------------------------------------- #
class TestPackX(_HarnessTestCase):
    def test_byte_widths_come_from_the_table_without_touching_torch(self):
        sys.modules["torch"] = None            # the table path must not import torch at all
        for name, want in (("fp8", 1), ("fp8_e4m3fnuz", 1), ("int8", 1), ("uint8", 1),
                           ("bf16", 2), ("fp16", 2), ("half", 2),
                           ("fp32", 4), ("float", 4), ("fp64", 8), ("float64", 8)):
            self.assertEqual(hl._bytes_of(name), want, name)

    def test_pack_x_is_16_bytes_over_the_element_size(self):
        self.assertEqual(hl.pack_x("fp8"), 16)
        self.assertEqual(hl.pack_x("int8"), 16)
        self.assertEqual(hl.pack_x("bf16"), 8)
        self.assertEqual(hl.pack_x("fp16"), 8)
        self.assertEqual(hl.pack_x("fp32"), 4)
        self.assertEqual(hl.pack_x("fp64"), 2)

    def test_pack_bytes_is_overridable(self):
        self.assertEqual(hl.pack_x("bf16", pack_bytes=32), 16)
        self.assertEqual(hl.pack_x("fp8", pack_bytes=8), 8)

    def test_a_torch_dtype_is_measured_through_element_size(self):
        self.assertEqual(hl._bytes_of(FP32, self.torch), 4)
        self.assertEqual(hl.pack_x(BF16, torch=self.torch), 8)
        self.assertEqual(hl.pack_x(FP8_E4M3FN, torch=self.torch), 16)

    def test_a_name_outside_the_table_is_resolved_through_regime_dtype(self):
        # "float8_e4m3fn" is the TORCH spelling, not a regime one, so it misses the byte table and
        # has to round-trip through regime_dtype -> element_size. Still 1 byte, so x is still 16.
        self.assertEqual(hl._bytes_of("float8_e4m3fn", self.torch), 1)
        self.assertEqual(hl.pack_x("float8_e4m3fn", torch=self.torch), 16)

    def test_an_unresolvable_name_packs_like_the_bf16_fallback(self):
        self.assertEqual(hl._bytes_of("float4_exotic", self.torch), 2)
        self.assertEqual(hl.pack_x("float4_exotic", torch=self.torch), 8)


# --------------------------------------------------------------------------- #
# regime_spec -- the parsed regime folded into what a synthesizer needs
# --------------------------------------------------------------------------- #
class TestRegimeSpec(unittest.TestCase):
    def test_an_empty_regime_is_unquantized_bf16(self):
        for regime in ({}, None):
            self.assertEqual(hl.regime_spec(regime), {
                "compute_dtype": "bf16", "kv_dtype": "bf16", "kv_x": 8, "kv_quant": False,
                "quant_method": "none", "operand_dtype": "bf16", "needs_scales": False})

    def test_auto_kv_follows_the_compute_dtype(self):
        for raw in ("auto", "AUTO", "", "none", None):
            spec = hl.regime_spec({"kv_cache_dtype": raw})
            self.assertEqual((spec["kv_dtype"], spec["kv_x"], spec["kv_quant"]), ("bf16", 8, False))

    def test_a_one_byte_kv_dtype_is_quantized_and_packs_16(self):
        spec = hl.regime_spec({"kv_cache_dtype": "FP8_E4M3"})
        self.assertEqual(spec["kv_dtype"], "fp8_e4m3")
        self.assertEqual(spec["kv_x"], 16)
        self.assertTrue(spec["kv_quant"])
        self.assertTrue(spec["needs_scales"])       # a quantized KV cache needs k_scale/v_scale

    def test_weight_quant_drives_the_operand_dtype_and_scales(self):
        spec = hl.regime_spec({"quant": {"method": "FP8_BLOCKSCALE", "weight_dtype": "fp8_e4m3fnuz"}})
        self.assertEqual(spec["quant_method"], "fp8_blockscale")
        self.assertEqual(spec["operand_dtype"], "fp8_e4m3fnuz")
        self.assertTrue(spec["needs_scales"])
        self.assertFalse(spec["kv_quant"])          # weight quant says nothing about the KV cache

    def test_quant_without_a_declared_weight_dtype_defaults_to_ocp_e4m3(self):
        self.assertEqual(hl.regime_spec({"quant": {"method": "fp8"}})["operand_dtype"], "fp8_e4m3")

    def test_method_none_is_not_quantized(self):
        for method in ("none", "NONE", "", None):
            spec = hl.regime_spec({"quant": {"method": method}})
            self.assertEqual(spec["operand_dtype"], "bf16")
            self.assertFalse(spec["needs_scales"])


class TestRegimeSpecUnknownKvDtype(_HarnessTestCase):
    def test_an_unrecognized_kv_dtype_is_treated_as_two_byte(self):
        # SOURCE/DOC MISMATCH, pinned as-is: regime_spec is documented "PURE, no torch", but an
        # unrecognized kv_cache_dtype misses _DTYPE_BYTES and pack_x then IMPORTS torch to measure it.
        # On a torch-free box that is an ImportError inside a function advertised as torch-free.
        spec = hl.regime_spec({"kv_cache_dtype": "float4_exotic"})
        self.assertEqual(spec["kv_dtype"], "float4_exotic")
        self.assertFalse(spec["kv_quant"])
        self.assertEqual(spec["kv_x"], 8)


# --------------------------------------------------------------------------- #
# synth_kv_cache -- the paged K/V layout in the live regime's dtype
# --------------------------------------------------------------------------- #
class TestSynthKvCache(_HarnessTestCase):
    def test_unquantized_cache_uses_the_vllm_paged_layout_and_unit_scales(self):
        got = hl.synth_kv_cache(2, 2, 32, 4, {}, torch=self.torch, seed=11)
        self.assertEqual(got["x"], 8)                       # bf16 -> 16 // 2
        self.assertEqual(got["kv_dtype"], "bf16")
        self.assertEqual(got["key_cache"].shape, (2, 2, 4, 4, 8))    # head_size // x, block, x
        self.assertEqual(got["value_cache"].shape, (2, 2, 32, 4))
        self.assertIs(got["key_cache"].dtype, BF16)
        self.assertIs(got["value_cache"].dtype, BF16)
        self.assertEqual(got["k_scale"].item(), 1.0)        # scalar 1.0, not a real scale
        self.assertEqual(got["v_scale"].item(), 1.0)
        self.assertIs(got["k_scale"].dtype, FP32)
        self.assertEqual(self.stack.seeds, [("cpu", 11)])

    def test_operands_are_scaled_into_the_bf16_range(self):
        got = hl.synth_kv_cache(1, 1, 8, 2, {}, torch=self.torch)
        # randn * 0.1 over the fake pattern (0, 1, -4, 2)
        self.assertEqual(got["value_cache"].tolist()[:4], [0.0, 0.1, -0.4, 0.2])

    def test_a_quantized_kv_dtype_produces_real_non_unit_per_tensor_scales(self):
        regime = {"kv_cache_dtype": "fp8_e4m3fnuz"}
        got = hl.synth_kv_cache(2, 2, 32, 4, regime, torch=self.torch)
        self.assertEqual(got["x"], 16)                      # 1-byte KV -> 16 // 1
        self.assertEqual(got["key_cache"].shape, (2, 2, 2, 4, 16))
        self.assertIs(got["key_cache"].dtype, FP8_E4M3FNUZ)
        # max|operand| is 4.0 * 0.1; the scale maps it exactly onto the fnuz fp8 max (240.0).
        self.assertAlmostEqual(got["k_scale"].item(), 0.4 / 240.0, places=12)
        self.assertAlmostEqual(got["v_scale"].item(), 0.4 / 240.0, places=12)
        self.assertNotEqual(got["k_scale"].item(), 1.0)
        self.assertLessEqual(max(abs(v) for v in got["key_cache"].tolist()), 240.0)

    def test_the_fp8_variant_follows_the_requested_arch(self):
        got = hl.synth_kv_cache(1, 1, 16, 2, {"kv_cache_dtype": "fp8"},
                                torch=self.torch, arch="gfx950")
        self.assertIs(got["key_cache"].dtype, FP8_E4M3FN)
        self.assertAlmostEqual(got["k_scale"].item(), 0.4 / 448.0, places=12)

    def test_an_integer_kv_dtype_is_scaled_by_its_iinfo_max(self):
        got = hl.synth_kv_cache(1, 1, 16, 2, {"kv_cache_dtype": "int8"}, torch=self.torch)
        self.assertIs(got["key_cache"].dtype, INT8)
        self.assertAlmostEqual(got["k_scale"].item(), 0.4 / 127.0, places=12)

    def test_the_generator_is_seeded_on_the_device_it_will_synthesize_on(self):
        self.stack.cuda = True
        hl.synth_kv_cache(1, 1, 16, 2, {}, torch=self.torch, seed=7)
        self.assertEqual(self.stack.seeds, [("cuda", 7)])


# --------------------------------------------------------------------------- #
# deployment_graph_mode / deployment_compile_mode -- the timing CONTEXT
# --------------------------------------------------------------------------- #
class TestDeploymentModes(unittest.TestCase):
    def test_decode_is_graph_captured_by_default(self):
        for regime in ({}, None, {"cuda_graph": True}):
            self.assertTrue(hl.deployment_graph_mode(regime))

    def test_enforce_eager_and_an_explicit_off_switch_disable_the_graph(self):
        self.assertFalse(hl.deployment_graph_mode({"enforce_eager": True}))
        self.assertFalse(hl.deployment_graph_mode({"cuda_graph": False}))
        # enforce_eager wins even when the descriptor still says cuda_graph
        self.assertFalse(hl.deployment_graph_mode({"enforce_eager": True, "cuda_graph": True}))

    def test_compile_mode_is_off_unless_the_regime_says_compiled(self):
        for regime in ({}, None, {"compile": "eager"}, {"compile": None}):
            self.assertFalse(hl.deployment_compile_mode(regime))

    def test_every_compiled_spelling_is_recognized(self):
        for value in ("torch_compile", "TORCH_COMPILE", "compile", "inductor", "true", "1"):
            self.assertTrue(hl.deployment_compile_mode({"compile": value}), value)

    def test_enforce_eager_opts_out_of_compilation(self):
        self.assertFalse(hl.deployment_compile_mode(
            {"enforce_eager": True, "compile": "torch_compile"}))


# --------------------------------------------------------------------------- #
# compiled_op -- fusion parity, and a fallback that is never silent
# --------------------------------------------------------------------------- #
class TestCompiledOp(_HarnessTestCase):
    COMPILED = {"compile": "torch_compile"}

    def _fn(self):
        def fn(x):
            return x
        return fn

    def test_an_eager_regime_returns_the_callable_untouched(self):
        fn = self._fn()
        self.assertIs(hl.compiled_op(fn, {}), fn)
        self.assertFalse(hasattr(fn, "_geak_compile_error"))

    def test_a_compiled_regime_wraps_the_callable_with_the_server_s_settings(self):
        self.stack.cuda = True
        seen = []

        def compile_(fn, **kw):
            seen.append(kw)
            return lambda *a, **k: ("compiled", fn(*a, **k))
        self.torch.compile = compile_
        fn = self._fn()
        wrapped = hl.compiled_op(fn, self.COMPILED)
        self.assertIsNot(wrapped, fn)
        self.assertEqual(seen, [{"fullgraph": True, "dynamic": False}])
        self.assertEqual(wrapped(3), ("compiled", 3))

    def test_compile_knobs_are_forwarded_and_mode_is_optional(self):
        self.stack.cuda = True
        seen = []
        self.torch.compile = lambda fn, **kw: seen.append(kw) or fn
        hl.compiled_op(self._fn(), self.COMPILED, fullgraph=False, dynamic=True,
                       mode="max-autotune")
        self.assertEqual(seen, [{"fullgraph": False, "dynamic": True, "mode": "max-autotune"}])

    def test_a_missing_torch_degrades_to_eager_and_records_why(self):
        sys.modules["torch"] = None
        fn = self._fn()
        self.assertIs(hl.compiled_op(fn, self.COMPILED), fn)
        self.assertIn("torch import failed", fn._geak_compile_error)

    def test_torch_without_compile_degrades_to_eager_and_records_why(self):
        fn = self._fn()
        self.assertIs(hl.compiled_op(fn, self.COMPILED), fn)
        self.assertEqual(fn._geak_compile_error, "torch.compile unavailable (torch<2.0)")

    def test_no_device_degrades_to_eager_and_records_why(self):
        self.torch.compile = lambda fn, **kw: fn
        fn = self._fn()
        self.assertIs(hl.compiled_op(fn, self.COMPILED), fn)
        self.assertEqual(fn._geak_compile_error, "no cuda; compiled path skipped")

    def test_an_unqueryable_device_still_attempts_the_compile(self):
        self.stack.available_raises = True
        self.torch.compile = lambda fn, **kw: ("compiled", fn)
        fn = self._fn()
        self.assertEqual(hl.compiled_op(fn, self.COMPILED), ("compiled", fn))

    def test_a_failing_compile_degrades_to_eager_and_records_the_exception(self):
        self.stack.cuda = True

        def boom(fn, **kw):
            raise RuntimeError("inductor backend not available")
        self.torch.compile = boom
        fn = self._fn()
        self.assertIs(hl.compiled_op(fn, self.COMPILED), fn)
        self.assertEqual(fn._geak_compile_error,
                         "RuntimeError: inductor backend not available")

    def test_a_long_compile_error_is_truncated_so_the_report_stays_readable(self):
        self.stack.cuda = True
        self.torch.compile = lambda fn, **kw: (_ for _ in ()).throw(RuntimeError("x" * 500))
        fn = self._fn()
        hl.compiled_op(fn, self.COMPILED)
        self.assertEqual(fn._geak_compile_error, "RuntimeError: " + "x" * 200)

    def test_a_callable_that_refuses_attributes_still_degrades_quietly(self):
        # A native/builtin op (len stands in) cannot carry _geak_compile_error. Recording the failure
        # must never itself become the failure -- parity is kept by returning fn unchanged.
        self.stack.cuda = True

        sys.modules["torch"] = None
        self.assertIs(hl.compiled_op(len, self.COMPILED), len)

        sys.modules["torch"] = self.torch
        self.assertIs(hl.compiled_op(len, self.COMPILED), len)          # no torch.compile attr

        self.torch.compile = lambda fn, **kw: (_ for _ in ()).throw(RuntimeError("nope"))
        self.assertIs(hl.compiled_op(len, self.COMPILED), len)          # compile raised
        self.assertFalse(hasattr(len, "_geak_compile_error"))


# --------------------------------------------------------------------------- #
# time_op -- the timer selection and the median it reports
# --------------------------------------------------------------------------- #
class TestTimeOpWall(_HarnessTestCase):
    WALL_MS = (3.0, 1.0, 2.0)

    def test_a_cudaless_box_reports_the_wall_median_as_the_device_time(self):
        box, call = self._counting_call()
        got = hl.time_op(call, warmup=2, repeats=3, detail=True)
        self.assertEqual(got["timer"], "wall")
        self.assertAlmostEqual(got["ms"], 2.0)              # median of 3.0, 1.0, 2.0
        self.assertEqual(got["ms"], got["wall_ms"])         # no device timeline to differ from
        self.assertEqual(box["n"], 5)                       # warmup 2 + repeats 3

    def test_without_detail_only_the_median_ms_is_returned(self):
        _, call = self._counting_call()
        self.assertAlmostEqual(hl.time_op(call, warmup=1, repeats=3), 2.0)

    def test_inner_amortizes_the_per_sample_cost(self):
        box, call = self._counting_call()
        got = hl.time_op(call, warmup=1, repeats=3, inner=2)
        self.assertAlmostEqual(got, 1.0)                    # the 2.0ms median split over 2 launches
        self.assertEqual(box["n"], 7)                       # 1 warmup + 3 samples x 2 launches

    def test_nonpositive_warmup_repeats_and_inner_are_clamped_to_one(self):
        box, call = self._counting_call()
        got = hl.time_op(call, warmup=0, repeats=0, inner=0)
        self.assertAlmostEqual(got, 3.0)                    # a single sample IS the median
        self.assertEqual(box["n"], 2)

    def test_graph_timing_falls_back_to_wall_without_a_device(self):
        _, call = self._counting_call()
        got = hl.time_op(call, warmup=1, repeats=1, graph=True, detail=True)
        self.assertEqual(got["timer"], "wall")
        self.assertEqual(self.stack.replays, 0)

    def test_a_raising_closure_is_reported_as_untimeable(self):
        def boom():
            raise RuntimeError("kernel launch failed")
        self.assertIsNone(hl.time_op(boom, warmup=1, repeats=1))
        self.assertIsNone(hl.time_op(boom, warmup=1, repeats=1, detail=True))

    def test_an_unqueryable_device_makes_the_op_untimeable(self):
        # SOURCE GAP, pinned as-is: time_op guards its is_available() probe so a broken runtime
        # still gets WALL timing, but the wall path immediately calls sync(), whose own
        # is_available() is UNGUARDED. So the guard buys nothing -- the sample is lost (ms=None)
        # and the bucket looks untimeable rather than wall-timed.
        self.stack.available_raises = True
        _, call = self._counting_call()
        self.assertIsNone(hl.time_op(call, warmup=1, repeats=1, detail=True))


class TestTimeOpEvents(_CudaTestCase):
    EVENT_MS = (6.0, 2.0, 4.0)
    WALL_MS = (30.0, 10.0, 20.0)

    def test_device_time_is_the_cuda_event_median_and_wall_is_reported_alongside(self):
        box, call = self._counting_call()
        got = hl.time_op(call, warmup=2, repeats=3, detail=True)
        self.assertEqual(got["timer"], "cuda_event")
        self.assertEqual(got["ms"], 4.0)                    # median of the scripted 6, 2, 4
        self.assertAlmostEqual(got["wall_ms"], 20.0)        # host+device reference, measured alongside
        self.assertEqual(box["n"], 5)

    def test_device_and_wall_are_both_divided_by_inner(self):
        got = hl.time_op(lambda: None, warmup=1, repeats=3, inner=2, detail=True)
        self.assertEqual(got["ms"], 2.0)
        self.assertAlmostEqual(got["wall_ms"], 10.0)

    def test_the_cache_is_flushed_once_per_sample_outside_the_event_window(self):
        hl.time_op(lambda: None, warmup=1, repeats=3)
        self.assertIsNotNone(hl._CACHE_FLUSH_BUF)
        # sync() before each sample plus the post-warmup sync
        self.assertEqual(self.stack.syncs, 4)

    def test_flush_can_be_disabled_for_a_deliberately_hot_measurement(self):
        hl.time_op(lambda: None, warmup=1, repeats=2, flush_cache=False)
        self.assertIsNone(hl._CACHE_FLUSH_BUF)

    def test_each_sample_records_start_and_end_and_waits_on_the_end_event(self):
        hl.time_op(lambda: None, warmup=1, repeats=2)
        self.assertEqual([k for k, _ in self.stack.events],
                         ["record", "record", "synchronize"] * 2)


class TestTimeOpGraph(_CudaTestCase):
    EVENT_MS = (9.0, 3.0, 6.0)
    WALL_MS = (90.0, 30.0, 60.0)

    def test_a_captured_graph_is_replayed_and_timed_with_the_same_event_method(self):
        box, call = self._counting_call()
        got = hl.time_op(call, warmup=2, repeats=3, graph=True, detail=True)
        self.assertEqual(got["timer"], "cuda_event_graph")
        self.assertEqual(got["ms"], 6.0)
        self.assertAlmostEqual(got["wall_ms"], 60.0)
        # capture warms up on a side stream (3 launches) then records `inner` launches
        self.assertEqual(box["n"], 4)
        self.assertEqual(self.stack.replays, 5)             # 2 warmup replays + 3 timed replays

    def test_capture_runs_on_a_side_stream_that_is_joined_before_recording(self):
        hl.time_op(lambda: None, warmup=1, repeats=1, graph=True)
        self.assertEqual(self.stack.stream_ops[:4],
                         [("wait_stream", "side", "default"), ("enter_stream", "side"),
                          ("exit_stream", "side"), ("wait_stream", "default", "side")])
        self.assertEqual([op[0] for op in self.stack.stream_ops[4:]],
                         ["enter_graph", "exit_graph"])

    def test_inner_divides_the_replay_time_it_amortizes(self):
        box, call = self._counting_call()
        got = hl.time_op(call, warmup=1, repeats=3, inner=3, graph=True, detail=True)
        self.assertEqual(got["ms"], 2.0)                    # 6.0 scripted / inner 3
        self.assertAlmostEqual(got["wall_ms"], 20.0)
        self.assertEqual(box["n"], 6)                       # 3 side-stream warmups + 3 captured

    def test_an_uncapturable_op_falls_back_to_eager_event_timing(self):
        self.stack.capture_raises = True
        box, call = self._counting_call()
        got = hl.time_op(call, warmup=2, repeats=3, graph=True, detail=True)
        self.assertEqual(got["timer"], "cuda_event")        # the `timer` field is how the UT sees it
        self.assertEqual(got["ms"], 6.0)
        self.assertEqual(self.stack.replays, 0)
        self.assertEqual(box["n"], 8)                       # 3 failed-capture warmups + 2 + 3

    def test_graph_replay_can_also_be_timed_hot(self):
        hl.time_op(lambda: None, warmup=1, repeats=2, graph=True, flush_cache=False)
        self.assertIsNone(hl._CACHE_FLUSH_BUF)


class TestTimingResult(unittest.TestCase):
    def test_detail_exposes_both_clocks_and_which_timer_produced_them(self):
        self.assertEqual(hl._timing_result(1.5, 2.25, "cuda_event", True),
                         {"ms": 1.5, "wall_ms": 2.25, "timer": "cuda_event"})

    def test_the_bare_form_is_the_device_ms_the_speedup_is_scored_on(self):
        self.assertEqual(hl._timing_result(1.5, 2.25, "cuda_event", False), 1.5)


# --------------------------------------------------------------------------- #
# flush_cache -- reading a memory-bound kernel's weights COLD from HBM
# --------------------------------------------------------------------------- #
class TestFlushCache(_HarnessTestCase):
    def test_without_a_device_there_is_nothing_to_evict(self):
        hl.flush_cache(self.torch)
        hl.flush_cache()
        self.assertIsNone(hl._CACHE_FLUSH_BUF)

    def test_an_unqueryable_device_is_a_noop_not_a_crash(self):
        self.stack.available_raises = True
        hl.flush_cache(self.torch)
        self.assertIsNone(hl._CACHE_FLUSH_BUF)


class TestFlushCacheOnDevice(_CudaTestCase):
    def test_the_default_buffer_is_larger_than_mi300_s_infinity_cache(self):
        hl.flush_cache(self.torch)
        buf = hl._CACHE_FLUSH_BUF
        self.assertEqual(buf.numel(), (512 << 20) // 4)     # 512MB of fp32 > the 256MB LLC
        self.assertIs(buf.dtype, FP32)
        self.assertEqual(buf.device, "cuda")

    def test_the_size_is_env_overridable_for_other_cache_hierarchies(self):
        with _env(HARNESS_CACHE_FLUSH_MB="64"):
            hl.flush_cache(self.torch)
        self.assertEqual(hl._CACHE_FLUSH_BUF.numel(), (64 << 20) // 4)

    def test_an_explicit_size_overrides_the_environment(self):
        with _env(HARNESS_CACHE_FLUSH_MB="64"):
            hl.flush_cache(self.torch, mb=8)
        self.assertEqual(hl._CACHE_FLUSH_BUF.numel(), (8 << 20) // 4)

    def test_a_zero_size_still_allocates_something_writable(self):
        hl.flush_cache(self.torch, mb=0)
        self.assertEqual(hl._CACHE_FLUSH_BUF.numel(), 1)

    def test_a_big_enough_buffer_is_reused_and_only_rezeroed(self):
        hl.flush_cache(self.torch, mb=8)
        first = hl._CACHE_FLUSH_BUF
        first.overwrite_([1.0] * 4)                          # pretend a prior flush left data behind
        hl.flush_cache(self.torch, mb=4)
        self.assertIs(hl._CACHE_FLUSH_BUF, first)
        self.assertEqual(first.tolist()[:4], [0.0, 0.0, 0.0, 0.0])

    def test_a_larger_request_reallocates(self):
        hl.flush_cache(self.torch, mb=4)
        first = hl._CACHE_FLUSH_BUF
        hl.flush_cache(self.torch, mb=8)
        self.assertIsNot(hl._CACHE_FLUSH_BUF, first)
        self.assertEqual(hl._CACHE_FLUSH_BUF.numel(), (8 << 20) // 4)


# --------------------------------------------------------------------------- #
# correct -- the RMS noise floor that stops a spiky tensor hiding a real error
# --------------------------------------------------------------------------- #
class TestCorrect(_HarnessTestCase):
    def test_an_identical_output_is_correct_with_zero_error(self):
        ref = _T((2,), [2.0, 2.0])
        ok, err = hl.correct(ref.clone(), ref, 0.5)
        self.assertTrue(ok)
        self.assertEqual(err, 0.0)

    def test_error_within_the_mixed_tolerance_passes_with_the_exact_ratio(self):
        # ref RMS = 2.0 -> atol = tol*RMS = 1.0; bound = atol + tol*|ref| = 1.0 + 1.0 = 2.0.
        # diff = [0, 2] is exactly on the bound; err = max(diff / (|ref| + atol)) = 2 / 3.
        ok, err = hl.correct(_T((2,), [2.0, 4.0]), _T((2,), [2.0, 2.0]), 0.5)
        self.assertTrue(ok)
        self.assertAlmostEqual(err, 2.0 / 3.0, places=12)

    def test_error_past_the_bound_fails_with_the_exact_ratio(self):
        # diff = [0, 3] against the same bound of 2.0; err = 3 / (2 + 1) = 1.0.
        ok, err = hl.correct(_T((2,), [2.0, 5.0]), _T((2,), [2.0, 2.0]), 0.5)
        self.assertFalse(ok)
        self.assertEqual(err, 1.0)

    def test_the_floor_is_rms_scaled_so_a_spiky_tensor_cannot_hide_a_small_element_error(self):
        # ref = [4, 0, 0, 0]: RMS = 2.0 but max|ref| = 4.0. The RMS floor is tol*2 = 1.0, so a 1.5
        # absolute error on a near-zero element FAILS. A max-scaled floor would have been tol*4 = 2.0
        # and would have passed it -- the unbounded-relative-error hole this design closes.
        ref = _T((4,), [4.0, 0.0, 0.0, 0.0])
        ok, err = hl.correct(_T((4,), [4.0, 1.5, 0.0, 0.0]), ref, 0.5)
        self.assertFalse(ok)
        self.assertEqual(err, 1.5)

    def test_a_single_sample_reference_is_handled(self):
        # RMS of one element IS that element: atol = 0.5 * 2.0 = 1.0, bound = 1.0 + 1.0 = 2.0.
        self.assertEqual(hl.correct(_T((1,), [4.0]), _T((1,), [2.0]), 0.5), (True, 2.0 / 3.0))
        self.assertEqual(hl.correct(_T((1,), [5.0]), _T((1,), [2.0]), 0.5), (False, 1.0))

    def test_an_all_zero_reference_falls_back_to_the_absolute_floor(self):
        # Zero variance and zero magnitude: clamp_min(1e-6) keeps the pure-relative term from
        # dividing by zero, so atol = tol * 1e-6 = 5e-7 is the whole budget.
        ref = _T((2,), [0.0, 0.0])
        self.assertEqual(hl.correct(ref.clone(), ref, 0.5), (True, 0.0))
        ok, err = hl.correct(_T((2,), [1e-6, 0.0]), ref, 0.5)
        self.assertFalse(ok)
        self.assertAlmostEqual(err, 2.0, places=9)          # 1e-6 / 5e-7

    def test_a_shape_mismatch_is_incorrect_with_an_infinite_error(self):
        ok, err = hl.correct(_T((3,), [1.0, 1.0, 1.0]), _T((2,), [1.0, 1.0]), 0.5)
        self.assertFalse(ok)
        self.assertEqual(err, float("inf"))

    def test_a_non_tensor_output_is_incorrect_rather_than_an_exception(self):
        self.assertEqual(hl.correct(None, _T((2,), [1.0, 1.0]), 0.5), (False, float("inf")))

    def test_negative_values_are_compared_on_magnitude(self):
        ok, err = hl.correct(_T((2,), [-2.0, -4.0]), _T((2,), [-2.0, -2.0]), 0.5)
        self.assertTrue(ok)
        self.assertAlmostEqual(err, 2.0 / 3.0, places=12)


class TestCorrectOnMultiTensorReturns(_HarnessTestCase):
    """Attention entries return `(out, lse)`.

    Comparing only the first component would let a candidate corrupt `lse` and still pass; falling
    into the except would report a CORRECT candidate as incorrect. Both are silent, so both are
    tested here rather than left to the shape of whatever the op happens to return."""

    def test_a_tuple_passes_only_when_every_component_passes(self):
        ref = (_T((2,), [2.0, 2.0]), _T((2,), [1.0, 1.0]))
        ok, err = hl.correct((_T((2,), [2.0, 2.0]), _T((2,), [1.0, 1.0])), ref, 0.5)
        self.assertTrue(ok)
        self.assertEqual(err, 0.0)

    def test_a_corrupt_second_component_fails_the_whole_comparison(self):
        """The first component matching exactly must not carry the verdict."""
        ref = (_T((2,), [2.0, 2.0]), _T((2,), [2.0, 2.0]))
        ok, err = hl.correct((_T((2,), [2.0, 2.0]), _T((2,), [2.0, 9.0])), ref, 0.5)
        self.assertFalse(ok)
        self.assertGreater(err, 1.0)

    def test_the_reported_error_is_the_worst_component(self):
        ref = (_T((2,), [2.0, 2.0]), _T((2,), [2.0, 2.0]))
        ok, err = hl.correct((_T((2,), [2.0, 4.0]), _T((2,), [2.0, 2.0])), ref, 0.5)
        self.assertTrue(ok)
        self.assertAlmostEqual(err, 2.0 / 3.0, places=12)

    def test_a_component_count_mismatch_is_incorrect(self):
        """Zipping the shorter side would compare a 2-tuple against a 3-tuple and pass."""
        ok, err = hl.correct((_T((1,), [1.0]),), (_T((1,), [1.0]), _T((1,), [1.0])), 0.5)
        self.assertFalse(ok)
        self.assertEqual(err, float("inf"))

    def test_a_dict_return_is_paired_by_key_not_by_insertion_order(self):
        a = {"lse": _T((1,), [1.0]), "out": _T((1,), [8.0])}
        b = {"out": _T((1,), [8.0]), "lse": _T((1,), [1.0])}
        self.assertEqual(hl.correct(a, b, 0.5), (True, 0.0))

    def test_a_scalar_riding_along_with_a_tensor_is_not_compared(self):
        """(out, num_tokens) must not fail because an int has no .shape."""
        ok, _ = hl.correct((_T((1,), [1.0]), 7), (_T((1,), [1.0]), 7), 0.5)
        self.assertTrue(ok)

    def test_an_empty_return_is_incorrect_rather_than_vacuously_correct(self):
        self.assertEqual(hl.correct((), (), 0.5), (False, float("inf")))

    def test_flatten_is_stable_across_nesting(self):
        self.assertEqual(len(hl.flatten_outputs([{"a": _T((1,), [1.0])}, (_T((1,), [2.0]),)])), 2)


# --------------------------------------------------------------------------- #
# assert_independent_outputs -- catching the shared/persistent `static_out`
# --------------------------------------------------------------------------- #
class TestAssertIndependentOutputs(_HarnessTestCase):
    def test_a_fresh_output_per_call_passes_with_no_reason(self):
        self.assertEqual(hl.assert_independent_outputs(_echo_call, (1.0, 2.0), (3.0, 4.0)),
                         (True, ""))

    def test_a_shared_return_buffer_is_named_with_its_storage_address(self):
        call = _StaticOutCall()
        ok, reason = hl.assert_independent_outputs(call, (1.0, 2.0), (3.0, 4.0))
        self.assertFalse(ok)
        self.assertIn("shared_output_buffer", reason)
        self.assertIn(hex(call.buf.data_ptr()), reason)
        self.assertIn("FRESH out", reason)

    def test_a_call_that_overwrites_its_previous_return_is_caught(self):
        # Distinct storage each time, but the second launch writes into the first return -- the
        # aliasing form a data_ptr comparison alone cannot see.
        returned = []

        def call(args):
            if returned:
                returned[0].overwrite_([99.0, 99.0])
            out = _T((2,), list(args))
            returned.append(out)
            return out

        ok, reason = hl.assert_independent_outputs(call, (1.0, 2.0), (3.0, 4.0))
        self.assertFalse(ok)
        self.assertIn("mutated_prior_output", reason)

    def test_a_raising_launcher_is_reported_not_propagated(self):
        def call(args):
            raise RuntimeError("illegal memory access")

        ok, reason = hl.assert_independent_outputs(call, (1.0,), (2.0,))
        self.assertFalse(ok)
        self.assertIn("independence_check_raised", reason)
        self.assertIn("illegal memory access", reason)


# --------------------------------------------------------------------------- #
# check_correct_multi / check_correct_sequence
# --------------------------------------------------------------------------- #
class TestCheckCorrectMulti(_HarnessTestCase):
    def test_all_cases_correct_plus_the_independence_verdict(self):
        cases = [_case("m1", (1.0, 2.0)), _case("m8", (3.0, 4.0))]
        ok, per = hl.check_correct_multi(_echo_call, cases, 0.01)
        self.assertTrue(ok)
        self.assertEqual(per, [
            {"case": "m1", "correct": True, "max_rel_err": 0.0},
            {"case": "m8", "correct": True, "max_rel_err": 0.0},
            {"case": "output_independence", "correct": True, "max_rel_err": None, "note": ""},
        ])

    def test_a_single_case_skips_the_independence_check(self):
        ok, per = hl.check_correct_multi(_echo_call, [_case("only", (1.0, 2.0))], 0.01)
        self.assertTrue(ok)
        self.assertEqual([e["case"] for e in per], ["only"])

    def test_a_shared_buffer_return_fails_the_EARLIER_case_it_overwrote(self):
        # This is the whole point of holding every output live before comparing: the second launch
        # overwrites the first return, so case m1 is compared against case m8's values.
        call = _StaticOutCall()
        cases = [_case("m1", (1.0, 1.0)), _case("m8", (2.0, 2.0))]
        ok, per = hl.check_correct_multi(call, cases, 0.01)
        self.assertFalse(ok)
        self.assertFalse(per[0]["correct"])
        self.assertEqual(per[0]["max_rel_err"], round(1.0 / 1.01, 5))
        self.assertTrue(per[1]["correct"])                  # the LAST case still matches
        self.assertFalse(per[2]["correct"])
        self.assertIn("shared_output_buffer", per[2]["note"])

    def test_a_wrong_shape_output_records_a_null_error_rather_than_infinity(self):
        cases = [{"sig": "m1", "args": (1.0, 2.0), "ref": _T((3,), [1.0, 2.0, 3.0])}]
        ok, per = hl.check_correct_multi(_echo_call, cases, 0.01)
        self.assertFalse(ok)
        self.assertIsNone(per[0]["max_rel_err"])

    def test_a_case_without_a_label_is_reported_with_an_empty_one(self):
        _, per = hl.check_correct_multi(_echo_call,
                                        [{"args": (1.0,), "ref": _T((1,), [1.0])}], 0.01)
        self.assertEqual(per[0]["case"], "")


class TestCheckCorrectSequence(_HarnessTestCase):
    def test_the_literal_call_order_is_replayed_and_indexed(self):
        ordered = [_case("big", (1.0, 2.0, 3.0)), _case("m1", (4.0,)),
                   _case("big", (5.0, 6.0, 7.0))]
        ok, per = hl.check_correct_sequence(_echo_call, ordered, 0.01)
        self.assertTrue(ok)
        self.assertEqual([e["case"] for e in per], ["seq[0]:big", "seq[1]:m1", "seq[2]:big"])
        self.assertEqual([e["max_rel_err"] for e in per], [0.0, 0.0, 0.0])

    def test_a_kernel_that_stashes_the_FIRST_shape_s_state_fails_on_the_second(self):
        # The stale-state bug a deduped single-shape check misses: the workspace is sized on the
        # first call and every later, differently-shaped call is truncated into it.
        state = {}

        def call(args):
            state.setdefault("n", len(args))
            return _T((state["n"],), list(args[:state["n"]]) + [0.0] * (state["n"] - len(args)))

        ordered = [_case("m1", (1.0,)), _case("big", (2.0, 3.0))]
        ok, per = hl.check_correct_sequence(call, ordered, 0.01)
        self.assertFalse(ok)
        self.assertTrue(per[0]["correct"])
        self.assertFalse(per[1]["correct"])
        self.assertIsNone(per[1]["max_rel_err"])            # shape mismatch -> infinite error

    def test_an_empty_sequence_is_vacuously_ok(self):
        self.assertEqual(hl.check_correct_sequence(_echo_call, [], 0.01), (True, []))


# --------------------------------------------------------------------------- #
# check_graph_replay -- capture once, replay many, ONE static buffer
# --------------------------------------------------------------------------- #
class _ReplayBundle:
    """Static input/output buffers plus the three closures check_graph_replay drives, mirroring the
    server's capture-once/replay-many contract (fill copies IN, run writes the static output).

    `run` registers its device work with the fake stack while a capture is open, so a later
    graph.replay() re-issues that work without re-entering `run` -- the deployment behaviour.
    """

    def __init__(self, stack, n=3, corrupt_on=None):
        self.stack = stack
        self.static_in = _T((n,), [0.0] * n)
        self.static_out = _T((n,), [0.0] * n)
        self.n = n
        self.corrupt_on = corrupt_on
        self.filled = []

    def fill(self, case):
        args = list(case["args"])
        if self.corrupt_on == case.get("sig"):
            raise RuntimeError("out-of-bounds write padding %r into the static buffer" % (args,))
        self.filled.append(case.get("sig"))
        self.static_in.overwrite_(args + [0.0] * (self.n - len(args)))

    def _device_work(self):
        self.static_out.overwrite_(self.static_in.tolist())

    def run(self):
        if self.stack.capturing:
            self.stack.recording.append(self._device_work)
        self._device_work()

    def read_out(self):
        return self.static_out


def _replay_case(sig, args, n=3, ref_values=None):
    padded = list(args) + [0.0] * (n - len(args))
    return {"sig": sig, "args": tuple(args),
            "ref": _T((n,), list(padded if ref_values is None else ref_values))}


class TestCheckGraphReplayWithoutDevice(_HarnessTestCase):
    def test_a_cudaless_box_records_a_pass_no_op_instead_of_a_false_failure(self):
        ok, per = hl.check_graph_replay(lambda c: None, lambda: None, lambda: None,
                                        [_replay_case("m1", (1.0,))], 0.01)
        self.assertTrue(ok)
        self.assertEqual(per, [{"case": "graph_replay", "correct": True, "max_rel_err": None,
                                "note": "skipped: no CUDA / no cases"}])


class TestCheckGraphReplay(_CudaTestCase):
    def _cases(self):
        return [_replay_case("big", (1.0, 2.0, 3.0)), _replay_case("m1", (4.0,))]

    def test_every_case_is_filled_and_replayed_through_the_captured_buffers(self):
        b = _ReplayBundle(self.stack)
        ok, per = hl.check_graph_replay(b.fill, b.run, b.read_out, self._cases(), 0.01)
        self.assertTrue(ok)
        self.assertEqual([e["case"] for e in per], ["big", "m1"])
        self.assertEqual([e["note"] for e in per], ["graph_replay", "graph_replay"])
        self.assertEqual([e["max_rel_err"] for e in per], [0.0, 0.0])
        self.assertEqual(self.stack.replays, 2)
        self.assertEqual(b.filled, ["big", "big", "m1"])     # capture fill, then one per case

    def test_capture_happens_on_the_named_case_after_a_side_stream_warmup(self):
        b = _ReplayBundle(self.stack)
        hl.check_graph_replay(b.fill, b.run, b.read_out, self._cases(), 0.01,
                              capture_idx=1, warmup=2)
        self.assertEqual(b.filled[0], "m1")
        self.assertEqual(self.stack.stream_ops[:2],
                         [("wait_stream", "side", "default"), ("enter_stream", "side")])

    def test_a_stale_output_under_replay_is_recorded_incorrect(self):
        # The smaller case's oracle differs from what the padded static buffer holds.
        b = _ReplayBundle(self.stack)
        cases = [_replay_case("big", (1.0, 2.0, 3.0)),
                 _replay_case("m1", (4.0,), ref_values=[4.0, 9.0, 9.0])]
        ok, per = hl.check_graph_replay(b.fill, b.run, b.read_out, cases, 0.01)
        self.assertFalse(ok)
        self.assertTrue(per[0]["correct"])
        self.assertFalse(per[1]["correct"])

    def test_a_faulting_replay_is_recorded_not_swallowed(self):
        b = _ReplayBundle(self.stack, corrupt_on="m1")
        ok, per = hl.check_graph_replay(b.fill, b.run, b.read_out, self._cases(), 0.01)
        self.assertFalse(ok)
        self.assertTrue(per[0]["correct"])
        self.assertFalse(per[1]["correct"])
        self.assertIsNone(per[1]["max_rel_err"])
        self.assertIn("graph_replay_raised (OOB/stale under replay)", per[1]["note"])
        self.assertIn("out-of-bounds write", per[1]["note"])

    def test_an_image_without_capture_support_records_a_pass_no_op(self):
        self.stack.capture_raises = True
        b = _ReplayBundle(self.stack)
        ok, per = hl.check_graph_replay(b.fill, b.run, b.read_out, self._cases(), 0.01)
        self.assertTrue(ok)
        self.assertEqual(len(per), 1)
        self.assertTrue(per[0]["correct"])
        self.assertIn("skipped: capture unavailable", per[0]["note"])
        self.assertIn("capture unsupported", per[0]["note"])

    def test_no_cases_is_a_pass_no_op(self):
        ok, per = hl.check_graph_replay(lambda c: None, lambda: None, lambda: None, [], 0.01)
        self.assertTrue(ok)
        self.assertEqual(per[0]["note"], "skipped: no CUDA / no cases")


# --------------------------------------------------------------------------- #
# check_random_vs_baseline -- value parity against the frozen live baseline
# --------------------------------------------------------------------------- #
class TestCheckRandomVsBaseline(_HarnessTestCase):
    def _shape(self, sig="m1:bf16", n=2):
        def make_inputs(rng):
            self.rngs.append(rng.seed)
            return tuple(float(rng.seed + i) for i in range(n))
        return {"sig": sig, "make_inputs": make_inputs}

    def setUp(self):
        super().setUp()
        self.rngs = []

    def test_every_draw_is_a_fresh_seeded_input_set_gated_on_correctness(self):
        ok, per = hl.check_random_vs_baseline(_echo_call, _echo_call, [self._shape()], 0.01,
                                              draws=3, warmup=1, repeats=1, seed=100)
        self.assertTrue(ok)
        self.assertEqual([e["case"] for e in per],
                         ["random[0]:m1:bf16", "random[1]:m1:bf16", "random[2]:m1:bf16"])
        self.assertEqual(self.rngs, [100, 101, 102])        # seed + draw index, reproducible
        self.assertEqual([e["max_rel_err"] for e in per], [0.0, 0.0, 0.0])
        self.assertIn("correctness gates; speedup reports", per[0]["note"])

    def test_the_speedup_is_report_only_and_rounded(self):
        _, per = hl.check_random_vs_baseline(_echo_call, _echo_call, [self._shape()], 0.01,
                                             draws=1, warmup=1, repeats=1)
        self.assertEqual(per[0]["speedup"], 1.0)            # same scripted clock on both sides

    def test_an_unmeasurable_pair_reports_no_speedup_rather_than_a_division(self):
        self.clock.deltas = [0.0]                           # both sides time as 0ms
        _, per = hl.check_random_vs_baseline(_echo_call, _echo_call, [self._shape()], 0.01,
                                             draws=1, warmup=1, repeats=1)
        self.assertIsNone(per[0]["speedup"])
        self.assertTrue(per[0]["correct"])

    def test_a_candidate_that_disagrees_with_the_baseline_fails_the_hard_gate(self):
        def wrong(args):
            return _T((len(args),), [v * 2.0 for v in args])

        ok, per = hl.check_random_vs_baseline(_echo_call, wrong, [self._shape()], 0.01,
                                              draws=1, warmup=1, repeats=1, seed=1)
        self.assertFalse(ok)
        self.assertFalse(per[0]["correct"])
        self.assertIsNotNone(per[0]["max_rel_err"])

    def test_a_candidate_that_aliases_the_baseline_storage_is_caught_by_the_snapshot(self):
        # The baseline output is cloned BEFORE the candidate runs, so a candidate that writes into
        # the baseline's storage and returns it cannot make itself look correct.
        def cheat(args):
            cheat.last.overwrite_([0.0] * len(args))
            return cheat.last

        def baseline(args):
            cheat.last = _T((len(args),), list(args))
            return cheat.last

        ok, per = hl.check_random_vs_baseline(baseline, cheat, [self._shape()], 0.01,
                                              draws=1, warmup=1, repeats=1, seed=1)
        self.assertFalse(ok)
        self.assertFalse(per[0]["correct"])

    def test_a_raising_draw_is_recorded_and_the_remaining_draws_still_run(self):
        def make_inputs(rng):
            if rng.seed == 0:
                raise RuntimeError("operand synthesis ran out of memory")
            return (1.0, 2.0)

        ok, per = hl.check_random_vs_baseline(
            _echo_call, _echo_call, [{"sig": "m1", "make_inputs": make_inputs}], 0.01,
            draws=2, warmup=1, repeats=1)
        self.assertFalse(ok)
        self.assertFalse(per[0]["correct"])
        self.assertIsNone(per[0]["speedup"])
        self.assertIn("value-parity raised:", per[0]["note"])
        self.assertIn("out of memory", per[0]["note"])
        self.assertTrue(per[1]["correct"])

    def test_nonpositive_draws_still_takes_one(self):
        _, per = hl.check_random_vs_baseline(_echo_call, _echo_call, [self._shape()], 0.01,
                                             draws=0, warmup=1, repeats=1)
        self.assertEqual(len(per), 1)

    def test_no_shapes_is_vacuously_ok(self):
        self.assertEqual(hl.check_random_vs_baseline(_echo_call, _echo_call, [], 0.01), (True, []))

    def test_an_unlabelled_shape_is_reported_by_draw_index_alone(self):
        _, per = hl.check_random_vs_baseline(
            _echo_call, _echo_call, [{"make_inputs": lambda rng: (1.0,)}], 0.01,
            draws=1, warmup=1, repeats=1)
        self.assertEqual(per[0]["case"], "random[0]:")


class TestCheckRandomVsBaselineOnDevice(_CudaTestCase):
    def test_the_generator_and_timing_follow_the_deployment_graph_context(self):
        shapes = [{"sig": "m1", "make_inputs": lambda rng: (1.0, 2.0)}]
        ok, _ = hl.check_random_vs_baseline(_echo_call, _echo_call, shapes, 0.01,
                                            draws=1, warmup=1, repeats=1, graph=True)
        self.assertTrue(ok)
        self.assertEqual(self.stack.seeds, [("cuda", 0)])
        self.assertGreater(self.stack.replays, 0)           # both sides timed under a graph replay


# --------------------------------------------------------------------------- #
# amdahl_ceiling / amdahl_check -- can this e2e delta come from this kernel?
# --------------------------------------------------------------------------- #
class TestAmdahlCeiling(unittest.TestCase):
    def test_the_documented_worked_example(self):
        # 20% of GPU time halved -> 10% of total saved -> 1/0.9 - 1 = +11.111% throughput.
        self.assertAlmostEqual(hl.amdahl_ceiling(0.2, 2.0), 100.0 * (1.0 / 0.9 - 1.0), places=12)

    def test_a_percent_and_a_fraction_are_the_same_input(self):
        self.assertEqual(hl.amdahl_ceiling(20.0, 2.0), hl.amdahl_ceiling(0.2, 2.0))

    def test_a_one_x_speedup_buys_nothing(self):
        self.assertEqual(hl.amdahl_ceiling(0.5, 1.0), 0.0)

    def test_a_zero_or_negative_speedup_is_no_ceiling_at_all(self):
        self.assertEqual(hl.amdahl_ceiling(0.5, None), 0.0)   # unmeasured -> no attributable ceiling
        self.assertEqual(hl.amdahl_ceiling(0.5, 0.0), 0.0)
        self.assertEqual(hl.amdahl_ceiling(0.5, -3.0), 0.0)

    def test_a_regression_cannot_produce_a_negative_ceiling(self):
        self.assertEqual(hl.amdahl_ceiling(0.5, 0.5), 0.0)

    def test_a_zero_or_negative_gpu_share_buys_nothing(self):
        self.assertEqual(hl.amdahl_ceiling(0.0, 10.0), 0.0)
        self.assertEqual(hl.amdahl_ceiling(-5.0, 10.0), 0.0)

    def test_an_over_100_percent_share_is_clamped_to_the_whole_workload(self):
        self.assertEqual(hl.amdahl_ceiling(300.0, 2.0), hl.amdahl_ceiling(1.0, 2.0))
        self.assertAlmostEqual(hl.amdahl_ceiling(1.0, 2.0), 100.0, places=12)

    def test_the_saved_fraction_is_capped_so_the_ceiling_stays_finite(self):
        # 100% of GPU time at 1e9x would save 99.9999999% of the time; the 0.999 cap keeps the
        # reciprocal from exploding into a meaningless number.
        self.assertAlmostEqual(hl.amdahl_ceiling(1.0, 1e9), 99900.0, places=6)

    def test_a_string_share_and_speedup_are_coerced(self):
        self.assertEqual(hl.amdahl_ceiling("20.0", "2.0"), hl.amdahl_ceiling(0.2, 2.0))


class TestAmdahlCheck(unittest.TestCase):
    CEILING = 100.0 * (1.0 / 0.9 - 1.0)                     # 20% of GPU time at 2x

    def test_an_observed_delta_inside_the_ceiling_is_attributable(self):
        got = hl.amdahl_check(5.0, 0.2, 2.0)
        self.assertEqual(got["verdict"], "ok")
        self.assertTrue(got["plausible"])
        self.assertEqual(got["ceiling_pct"], round(self.CEILING, 3))
        self.assertEqual(got["allowed_pct"], round(self.CEILING * 1.5 + 0.5, 3))
        self.assertIn("within Amdahl ceiling", got["note"])
        self.assertIn("observed +5.00%", got["note"])

    def test_a_delta_far_above_the_ceiling_is_box_drift_not_the_kernel(self):
        got = hl.amdahl_check(50.0, 0.2, 2.0)
        self.assertEqual(got["verdict"], "implausible")
        self.assertFalse(got["plausible"])
        self.assertIn("EXCEEDS the Amdahl ceiling", got["note"])
        self.assertIn("re-measure interleaved", got["note"])
        self.assertIn("2.000x", got["note"])

    def test_the_slack_boundary_is_inclusive(self):
        allowed = self.CEILING * 1.5 + 0.5
        self.assertTrue(hl.amdahl_check(allowed, 0.2, 2.0)["plausible"])
        self.assertFalse(hl.amdahl_check(allowed * 1.0001, 0.2, 2.0)["plausible"])

    def test_the_noise_band_alone_covers_a_null_kernel(self):
        # A kernel with no isolated win has a zero ceiling, so only the noise band is allowed.
        got = hl.amdahl_check(0.5, 0.2, 1.0)
        self.assertEqual(got["ceiling_pct"], 0.0)
        self.assertEqual(got["allowed_pct"], 0.5)
        self.assertTrue(got["plausible"])
        self.assertFalse(hl.amdahl_check(0.51, 0.2, 1.0)["plausible"])

    def test_the_slack_and_band_are_tunable(self):
        got = hl.amdahl_check(1.0, 0.2, 1.0, noise_band_pct=2.0, slack=1.0)
        self.assertEqual(got["allowed_pct"], 2.0)
        self.assertTrue(got["plausible"])

    def test_a_regression_is_trivially_plausible(self):
        self.assertTrue(hl.amdahl_check(-3.0, 0.2, 2.0)["plausible"])


# --------------------------------------------------------------------------- #
# served_regimes -- the gate that stops a decode case being weighted onto prefill
# --------------------------------------------------------------------------- #
class TestServedRegimes(unittest.TestCase):
    def test_no_metadata_is_ungated(self):
        for meta in (None, {}, {"served_regimes": []}, {"workload": {}}):
            self.assertEqual(hl.served_regimes(meta), set())

    def test_the_extractor_override_wins_over_the_trace_derived_set(self):
        meta = {"served_regimes": ["Decode"],
                "workload": {"served_regimes": ["prefill", "decode"]}}
        self.assertEqual(hl.served_regimes(meta), {"decode"})

    def test_the_trace_derived_set_is_the_fallback(self):
        self.assertEqual(hl.served_regimes({"workload": {"served_regimes": ["PREFILL"]}}),
                         {"prefill"})

    def test_entries_are_normalized_and_blanks_dropped(self):
        self.assertEqual(hl.served_regimes({"served_regimes": [" Decode ", "", "  ", "prefill"]}),
                         {"decode", "prefill"})


# --------------------------------------------------------------------------- #
# serving_weighted_speedup -- the PRIMARY metric the A/B judge reads
# --------------------------------------------------------------------------- #
class TestServingWeightedSpeedup(unittest.TestCase):
    META = {"served_regimes": ["decode"],
            "workload": {"serving_weight_model": {"analytic_calls": {"decode": 100}}}}

    def test_the_worked_weighting_over_two_decode_buckets(self):
        # The largest-M bucket carries the regime's analytic passes (100), the transient one gets 1:
        #   weights  = 2.0*1 = 2, 4.0*100 = 400   -> W = 402
        #   harmonic = 2/2.0 + 400/4.0 = 1 + 100  -> D = 101
        #   weighted = 402 / 101
        per_case = [{"sig": "m1", "regime": "decode", "m": 1, "baseline_ms": 2.0,
                     "optimized_ms": 1.0},
                    {"sig": "m256", "regime": "decode", "m": 256, "baseline_ms": 4.0,
                     "optimized_ms": 1.0}]
        got = hl.serving_weighted_speedup(per_case, self.META)
        self.assertAlmostEqual(got["weighted"], 402.0 / 101.0, places=12)
        self.assertEqual(got["primary"], got["weighted"])
        self.assertAlmostEqual(got["geomean"], math.sqrt(8.0), places=12)   # geomean(2x, 4x)
        self.assertEqual(got["included"], 2)
        self.assertEqual(got["reason"], "")
        self.assertEqual([r["calls"] for r in got["per_case"]], [1, 100])
        self.assertEqual([r["weight"] for r in got["per_case"]], [2.0, 400.0])
        self.assertEqual([r["speedup"] for r in got["per_case"]], [2.0, 4.0])

    def test_an_unserved_regime_is_dropped_before_it_can_be_weighted(self):
        per_case = [{"sig": "dec", "regime": "decode", "m": 8, "baseline_ms": 2.0,
                     "optimized_ms": 1.0},
                    {"sig": "pre", "regime": "prefill", "m": 4096, "baseline_ms": 50.0,
                     "optimized_ms": 1.0}]
        got = hl.serving_weighted_speedup(per_case, self.META)
        self.assertEqual(got["dropped_unserved"], ["pre"])
        self.assertEqual(got["included"], 1)
        self.assertAlmostEqual(got["weighted"], 2.0, places=12)

    def test_an_unregimed_case_survives_the_gate_and_carries_one_pass(self):
        got = hl.serving_weighted_speedup(
            [{"name": "unlabelled", "baseline_ms": 2.0, "optimized_ms": 1.0}], self.META)
        self.assertEqual(got["dropped_unserved"], [])
        self.assertEqual(got["per_case"][0]["sig"], "unlabelled")
        self.assertEqual(got["per_case"][0]["calls"], 1)
        self.assertEqual(got["included"], 1)

    def test_an_ungated_meta_weights_every_regime(self):
        per_case = [{"sig": "dec", "regime": "decode", "m": 8, "baseline_ms": 2.0,
                     "optimized_ms": 1.0},
                    {"sig": "pre", "regime": "prefill", "m": 4096, "baseline_ms": 2.0,
                     "optimized_ms": 1.0}]
        got = hl.serving_weighted_speedup(per_case, {})
        self.assertEqual(got["dropped_unserved"], [])
        self.assertEqual(got["included"], 2)
        self.assertEqual([r["calls"] for r in got["per_case"]], [1, 1])   # no analytic call model

    def test_a_pseudo_identity_bucket_is_flagged_and_excluded(self):
        per_case = [{"sig": "warm_jit", "regime": "decode", "m": 1, "baseline_ms": 1.0,
                     "optimized_ms": 1.00000001},
                    {"sig": "m256", "regime": "decode", "m": 256, "baseline_ms": 4.0,
                     "optimized_ms": 2.0}]
        got = hl.serving_weighted_speedup(per_case, self.META)
        self.assertEqual(got["suspect_identity"], ["warm_jit"])
        self.assertEqual(got["included"], 1)
        self.assertFalse(got["per_case"][0]["included"])
        self.assertTrue(got["per_case"][0]["identity"])
        self.assertAlmostEqual(got["weighted"], 2.0, places=12)

    def test_the_identity_window_is_relative_and_tunable(self):
        per_case = [{"sig": "b", "regime": "decode", "m": 1, "baseline_ms": 100.0,
                     "optimized_ms": 99.0}]
        self.assertEqual(hl.serving_weighted_speedup(per_case, self.META)["suspect_identity"], [])
        self.assertEqual(
            hl.serving_weighted_speedup(per_case, self.META, identity_eps=0.02)["suspect_identity"],
            ["b"])

    def test_an_all_identity_result_is_untrusted_rather_than_a_fabricated_1x(self):
        per_case = [{"sig": "a", "regime": "decode", "m": 1, "baseline_ms": 1.0,
                     "optimized_ms": 1.0}]
        got = hl.serving_weighted_speedup(per_case, self.META)
        self.assertIsNone(got["weighted"])
        self.assertIsNone(got["geomean"])
        self.assertIsNone(got["primary"])
        self.assertEqual(got["included"], 0)
        self.assertEqual(got["suspect_identity"], ["a"])
        self.assertIn("no measurable non-identity bucket survived", got["reason"])
        self.assertIn("fresh subprocess", got["reason"])

    def test_an_empty_case_list_is_untrusted(self):
        got = hl.serving_weighted_speedup([], self.META)
        self.assertIsNone(got["primary"])
        self.assertEqual(got["per_case"], [])
        self.assertIn("UNTRUSTED", got["reason"])

    def test_a_zero_weight_bucket_is_excluded_without_being_called_identity(self):
        # calls=0 in the analytic model zeroes the bucket's weight; it must not silently contribute.
        meta = {"workload": {"serving_weight_model": {"analytic_calls": {"decode": 0}}}}
        got = hl.serving_weighted_speedup(
            [{"sig": "only", "regime": "decode", "m": 8, "baseline_ms": 2.0, "optimized_ms": 1.0}],
            meta)
        self.assertEqual(got["per_case"][0]["calls"], 0)
        self.assertEqual(got["per_case"][0]["weight"], 0.0)
        self.assertFalse(got["per_case"][0]["included"])
        self.assertEqual(got["suspect_identity"], [])
        self.assertIsNone(got["primary"])

    def test_an_unmeasured_bucket_is_excluded(self):
        got = hl.serving_weighted_speedup(
            [{"sig": "no_ms", "regime": "decode", "m": 8},
             {"sig": "real", "regime": "decode", "m": 256, "baseline_ms": 4.0, "optimized_ms": 2.0}],
            self.META)
        self.assertFalse(got["per_case"][0]["included"])
        self.assertIsNone(got["per_case"][0]["speedup"])
        self.assertEqual(got["included"], 1)

    def test_non_numeric_timings_are_dropped_not_coerced(self):
        got = hl.serving_weighted_speedup(
            [{"sig": "junk", "regime": "decode", "m": 1, "baseline_ms": "n/a",
              "optimized_ms": None, "speedup": "fast"}], self.META)
        row = got["per_case"][0]
        self.assertIsNone(row["baseline_ms"])
        self.assertIsNone(row["optimized_ms"])
        self.assertIsNone(row["speedup"])
        self.assertFalse(row["included"])

    def test_measured_times_override_a_reported_speedup(self):
        got = hl.serving_weighted_speedup(
            [{"sig": "a", "regime": "decode", "m": 1, "baseline_ms": 4.0, "optimized_ms": 2.0,
              "speedup": 99.0}], self.META)
        self.assertEqual(got["per_case"][0]["speedup"], 2.0)

    def test_a_reported_speedup_is_used_when_the_optimized_time_is_unusable(self):
        got = hl.serving_weighted_speedup(
            [{"sig": "a", "regime": "decode", "m": 1, "baseline_ms": 4.0, "optimized_ms": 0.0,
              "speedup": 3.0}], self.META)
        self.assertEqual(got["per_case"][0]["speedup"], 3.0)
        self.assertAlmostEqual(got["weighted"], 3.0, places=12)

    def test_the_dominant_bucket_falls_back_to_baseline_ms_when_m_is_absent(self):
        # Without a recorded M the slowest bucket stands in as "largest", so the regime's analytic
        # passes still land on one bucket instead of raising on float(None).
        per_case = [{"sig": "cheap", "regime": "decode", "baseline_ms": 1.0, "optimized_ms": 0.5},
                    {"sig": "dear", "regime": "decode", "m": None, "baseline_ms": 8.0,
                     "optimized_ms": 4.0}]
        got = hl.serving_weighted_speedup(per_case, self.META)
        self.assertEqual([r["calls"] for r in got["per_case"]], [1, 100])
        self.assertAlmostEqual(got["weighted"], 2.0, places=12)

    def test_geomean_can_be_switched_off(self):
        got = hl.serving_weighted_speedup(
            [{"sig": "a", "regime": "decode", "m": 1, "baseline_ms": 4.0, "optimized_ms": 2.0}],
            self.META, geomean=False)
        self.assertIsNone(got["geomean"])
        self.assertAlmostEqual(got["weighted"], 2.0, places=12)


# --------------------------------------------------------------------------- #
# run_correctness -- the single entrypoint, and its FAIL-CLOSED replay gate
# --------------------------------------------------------------------------- #
class TestRunCorrectness(_HarnessTestCase):
    EAGER = {"enforce_eager": True}

    def setUp(self):
        super().setUp()
        self.eager_cases = [_case("m1", (1.0, 2.0)), _case("m256", (3.0, 4.0))]

    def _current(self):
        def current_call(args):
            return _echo_call(args)
        return current_call

    def _run(self, regime, **kw):
        kw.setdefault("eager_cases", self.eager_cases)
        kw.setdefault("baseline_call", _echo_call)
        kw.setdefault("current_call", self._current())
        kw.setdefault("random_shapes", [])
        kw.setdefault("tol", 0.01)
        return hl.run_correctness(regime, **kw)

    def _replay(self, n_cases=2, **kw):
        bundle = _ReplayBundle(self.stack)
        cases = [_replay_case("big", (1.0, 2.0, 3.0)), _replay_case("m1", (4.0,))][:n_cases]
        out = {"fill": bundle.fill, "run": bundle.run, "read_out": bundle.read_out,
               "cases": cases}
        out.update(kw)
        return out

    def test_an_eager_regime_runs_only_the_eager_and_random_legs(self):
        ok, report = self._run(self.EAGER)
        self.assertTrue(ok)
        self.assertEqual(set(report), {"eager", "random"})
        self.assertEqual([e["case"] for e in report["eager"]],
                         ["m1", "m256", "output_independence"])
        self.assertEqual(report["random"], [])

    def test_an_eager_correctness_failure_fails_the_whole_suite(self):
        def wrong(args):
            return _T((len(args),), [v + 10.0 for v in args])

        ok, report = self._run(self.EAGER, current_call=wrong)
        self.assertFalse(ok)
        self.assertFalse(report["eager"][0]["correct"])

    def test_the_random_leg_runs_against_the_frozen_baseline(self):
        shapes = [{"sig": "m1", "make_inputs": lambda rng: (1.0, 2.0)}]
        ok, report = self._run(self.EAGER, random_shapes=shapes, draws=1)
        self.assertTrue(ok)
        self.assertEqual([e["case"] for e in report["random"]], ["random[0]:m1"])

    def test_a_random_leg_failure_fails_the_whole_suite(self):
        shapes = [{"sig": "m1", "make_inputs": lambda rng: (1.0, 2.0)}]

        def wrong(args):
            return _T((len(args),), [v * 5.0 for v in args])

        ok, report = self._run(self.EAGER, random_shapes=shapes, draws=1, current_call=wrong)
        self.assertFalse(ok)
        self.assertFalse(report["random"][0]["correct"])

    def _assert_fails_closed(self, regime, expect, **kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with self.assertRaises(hl.HarnessIncompleteError) as cm:
                self._run(regime, **kw)
        reason = str(cm.exception)
        self.assertIn(expect, reason)
        self.assertIn("UT-GENERATION defect", reason)
        self.assertIn("NOT a kernel-correctness failure", reason)
        # the sentinel must come from HERE, exactly once, so the smoke test can regenerate the UT
        self.assertEqual(buf.getvalue().count(hl.UT_HARNESS_INCOMPLETE_SENTINEL), 1)
        self.assertTrue(buf.getvalue().startswith(hl.UT_HARNESS_INCOMPLETE_SENTINEL + ": "))
        return reason

    def test_a_graph_deploy_kernel_with_no_replay_bundle_fails_closed(self):
        self._assert_fails_closed({"cuda_graph": True}, "no replay bundle")

    def test_a_replay_bundle_missing_its_closures_fails_closed(self):
        bundle = self._replay()
        bundle["run"] = None
        bundle["read_out"] = "not callable"
        self._assert_fails_closed({"cuda_graph": True}, "missing closures ['run', 'read_out']",
                                  replay=bundle)

    def test_a_single_shape_replay_bundle_fails_closed(self):
        # One shape cannot expose a static-buffer-reuse OOB, so it is treated as no bundle at all.
        self._assert_fails_closed({"cuda_graph": True},
                                  "only 1 replay case(s); need >=2 boundary shapes",
                                  replay=self._replay(n_cases=1))

    def test_a_bundle_with_no_cases_fails_closed(self):
        self._assert_fails_closed({"cuda_graph": True}, "only 0 replay case(s)",
                                  replay=self._replay(n_cases=0))

    def test_a_complete_bundle_reaches_the_replay_leg(self):
        ok, report = self._run({"cuda_graph": True}, replay=self._replay())
        self.assertTrue(ok)
        self.assertEqual(set(report), {"eager", "random", "graph_replay"})
        # no device here, so the replay leg records its documented PASS no-op
        self.assertEqual(report["graph_replay"][0]["note"], "skipped: no CUDA / no cases")

    def test_the_capture_index_is_forwarded_and_coerced(self):
        captured = []
        bundle = self._replay(capture_idx="1")
        real_fill = bundle["fill"]
        bundle["fill"] = lambda case: captured.append(case["sig"]) or real_fill(case)
        self.stack.cuda = True
        ok, report = self._run({"cuda_graph": True}, replay=bundle)
        self.assertTrue(ok)
        self.assertEqual(captured[0], "m1")
        self.assertEqual([e["case"] for e in report["graph_replay"]], ["big", "m1"])

    def test_a_compiled_regime_adds_the_fusion_parity_leg(self):
        ok, report = self._run({"enforce_eager": False, "cuda_graph": False,
                                "compile": "torch_compile"})
        self.assertTrue(ok)
        self.assertEqual(set(report), {"eager", "random", "compile_parity"})

    def test_the_harness_error_is_a_distinct_exception_type(self):
        self.assertTrue(issubclass(hl.HarnessIncompleteError, Exception))
        self.assertEqual(hl.UT_HARNESS_INCOMPLETE_SENTINEL, "UT_HARNESS_INCOMPLETE")


class TestRunCorrectnessOnDevice(_CudaTestCase):
    def test_a_replay_fault_fails_the_suite_the_eager_legs_passed(self):
        # The h2 paged_attention shape: eager per-call testing is clean and only the deployment
        # replay path faults. run_correctness must return ok=False on it.
        bundle = _ReplayBundle(self.stack, corrupt_on="m1")
        replay = {"fill": bundle.fill, "run": bundle.run, "read_out": bundle.read_out,
                  "cases": [_replay_case("big", (1.0, 2.0, 3.0)), _replay_case("m1", (4.0,))]}
        ok, report = hl.run_correctness(
            {"cuda_graph": True}, eager_cases=[_case("m1", (1.0, 2.0))],
            baseline_call=_echo_call, current_call=_echo_call, random_shapes=[], tol=0.01,
            replay=replay)
        self.assertFalse(ok)
        self.assertTrue(all(e["correct"] for e in report["eager"]))
        self.assertFalse(report["graph_replay"][1]["correct"])
        self.assertIn("graph_replay_raised", report["graph_replay"][1]["note"])


# --------------------------------------------------------------------------- #
# _compile_parity -- fusion drift is a FAIL, a compile error is a surfaced note
# --------------------------------------------------------------------------- #
class TestCompileParity(_HarnessTestCase):
    COMPILED = {"compile": "torch_compile"}

    def setUp(self):
        super().setUp()
        self.cases = [_case("m1", (1.0, 2.0)), _case("m256", (3.0, 4.0))]

    def _current(self):
        def current_call(args):
            return _echo_call(args)
        return current_call

    def test_a_compile_failure_is_surfaced_as_a_non_fatal_note_not_a_rejection(self):
        # An isolated bare-op fullgraph compile is not the server's whole-model compile, so this
        # must stay visible-and-soft rather than auto-rejecting the candidate.
        self.torch.compile = lambda fn, **kw: fn
        ok, per = hl._compile_parity(self._current(), self.cases, self.COMPILED, 0.01)
        self.assertTrue(ok)
        self.assertEqual(len(per), 1)
        self.assertEqual(per[0]["case"], "compile_parity")
        self.assertTrue(per[0]["correct"])
        self.assertIn("compile_soft_degrade (NON-FATAL)", per[0]["note"])
        self.assertIn("no cuda; compiled path skipped", per[0]["note"])
        self.assertIn("Verify at the e2e gate", per[0]["note"])

    def test_a_compile_error_already_recorded_on_the_callable_is_surfaced(self):
        current = self._current()
        current._geak_compile_error = "TorchRuntimeError: graph break in the epilogue"
        self.torch.compile = lambda fn, **kw: fn
        self.stack.cuda = True
        ok, per = hl._compile_parity(current, self.cases, self.COMPILED, 0.01)
        self.assertTrue(ok)
        self.assertIn("graph break in the epilogue", per[0]["note"])

    def test_an_opaque_custom_op_compiles_to_identical_numerics(self):
        self.stack.cuda = True
        self.torch.compile = lambda fn, **kw: (lambda *a, **k: fn(*a, **k))
        ok, per = hl._compile_parity(self._current(), self.cases, self.COMPILED, 0.01)
        self.assertTrue(ok)
        self.assertEqual([e["case"] for e in per], ["m1", "m256"])
        self.assertEqual([e["max_rel_err"] for e in per], [0.0, 0.0])
        self.assertEqual(per[0]["note"], "compile_parity(compiled vs eager)")

    def test_fusion_induced_numeric_drift_is_a_real_correctness_failure(self):
        self.stack.cuda = True

        def compile_(fn, **kw):
            return lambda args: _T((len(args),), [v * 2.0 for v in args])
        self.torch.compile = compile_
        ok, per = hl._compile_parity(self._current(), self.cases, self.COMPILED, 0.01)
        self.assertFalse(ok)
        self.assertFalse(per[0]["correct"])
        self.assertIsNotNone(per[0]["max_rel_err"])

    def test_a_raising_compiled_call_is_soft_skipped_and_named(self):
        self.stack.cuda = True

        def compile_(fn, **kw):
            def compiled(args):
                raise RuntimeError("compiled region hit an unsupported op")
            return compiled
        self.torch.compile = compile_
        ok, per = hl._compile_parity(self._current(), self.cases, self.COMPILED, 0.01)
        self.assertTrue(ok)
        self.assertTrue(per[0]["correct"])
        self.assertIn("compile_parity soft-skip (compiled call raised)", per[0]["note"])
        self.assertIn("unsupported op", per[0]["note"])

    def test_an_eager_regime_parity_check_compares_the_callable_against_itself(self):
        ok, per = hl._compile_parity(self._current(), self.cases, {}, 0.01)
        self.assertTrue(ok)
        self.assertEqual([e["max_rel_err"] for e in per], [0.0, 0.0])


# --------------------------------------------------------------------------- #
# boundary-shape helpers -- the ragged shapes a uniform capture never exercises
# --------------------------------------------------------------------------- #
class TestBoundaryDecodeSeqLens(unittest.TestCase):
    def test_the_default_geometry_straddles_block_and_partition_boundaries(self):
        self.assertEqual(hl.boundary_decode_seq_lens({}, 4096),
                         [1, 15, 16, 17, 255, 256, 257, 511, 512, 4095, 4096])

    def test_an_explicit_geometry_moves_every_boundary(self):
        self.assertEqual(hl.boundary_decode_seq_lens({"block_size": 32, "partition_size": 64}, 300),
                         [1, 31, 32, 33, 63, 64, 65, 127, 128, 299, 300])

    def test_candidates_beyond_the_context_are_dropped(self):
        self.assertEqual(hl.boundary_decode_seq_lens({}, 20), [1, 15, 16, 17, 19, 20])

    def test_a_zero_or_missing_context_degenerates_to_the_single_token_case(self):
        for ctx in (0, None, ""):
            self.assertEqual(hl.boundary_decode_seq_lens({}, ctx), [1])

    def test_a_missing_or_zero_geometry_falls_back_to_the_documented_defaults(self):
        for geo in (None, {}, {"block_size": 0, "partition_size": None}):
            self.assertEqual(hl.boundary_decode_seq_lens(geo, 600),
                             [1, 15, 16, 17, 255, 256, 257, 511, 512, 599, 600])

    def test_the_result_is_sorted_and_deduplicated(self):
        got = hl.boundary_decode_seq_lens({"block_size": 16, "partition_size": 16}, 40)
        self.assertEqual(got, sorted(set(got)))
        self.assertEqual(got, [1, 15, 16, 17, 31, 32, 39, 40])


class TestShuffledBlockTable(_HarnessTestCase):
    def test_the_table_is_a_scattered_mapping_not_a_contiguous_arange(self):
        got = hl.shuffled_block_table(2, 3, seed=0, torch=self.torch, device="cpu")
        self.assertEqual(got.shape, (2, 3))
        self.assertIs(got.dtype, INT32)
        # the pool is padded to need+16 = 22 blocks, and the mapping walks it non-contiguously
        self.assertEqual([int(v) for v in got.tolist()], [21, 20, 19, 18, 17, 16])
        self.assertNotEqual([int(v) for v in got.tolist()], list(range(6)))

    def test_an_explicit_pool_is_honoured_when_it_is_larger_than_needed(self):
        got = hl.shuffled_block_table(2, 3, pool_blocks=100, torch=self.torch, device="cpu")
        self.assertEqual([int(v) for v in got.tolist()], [99, 98, 97, 96, 95, 94])

    def test_a_too_small_pool_is_padded_rather_than_overrun(self):
        got = hl.shuffled_block_table(2, 3, pool_blocks=4, torch=self.torch, device="cpu")
        self.assertEqual(len(set(int(v) for v in got.tolist())), 6)

    def test_the_seed_is_applied_on_the_target_device(self):
        hl.shuffled_block_table(1, 2, seed=5, torch=self.torch)
        self.assertEqual(self.stack.seeds, [("cuda", 5)])   # device defaults to the accelerator

    def test_the_seed_changes_the_mapping_reproducibly(self):
        a = hl.shuffled_block_table(1, 2, seed=1, torch=self.torch, device="cpu").tolist()
        b = hl.shuffled_block_table(1, 2, seed=2, torch=self.torch, device="cpu").tolist()
        self.assertNotEqual(a, b)
        self.assertEqual(
            a, hl.shuffled_block_table(1, 2, seed=1, torch=self.torch, device="cpu").tolist())


# --------------------------------------------------------------------------- #
# (d) two-leg measurement
#
# Every leg is a subprocess, so `subprocess` is swapped for a recorder that answers by mode. No GPU,
# no real leg_runner.py -- these tests pin the ORCHESTRATION, not the timing.
# --------------------------------------------------------------------------- #
class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakePopen:
    """The handle `_run_leg` drives: communicate() delivers the scripted _Proc, or raises a timeout."""

    def __init__(self, proc, call, timeout_once=False):
        self._proc = proc
        self._call = call
        self._timeout_once = timeout_once
        self.pid = os.getpid()          # a real pid, so os.getpgid() in the killer resolves
        self.returncode = None
        self.killed = False

    def communicate(self, timeout=None):
        self._call["timeout"] = timeout
        if self._timeout_once:
            self._timeout_once = False
            raise subprocess.TimeoutExpired(self._call["cmd"], timeout)
        self.returncode = self._proc.returncode
        return self._proc.stdout, self._proc.stderr

    def wait(self, timeout=None):
        """Reaped only if a signal actually took; the killpg fake is what decides that."""
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self._call["cmd"], timeout)
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


class _FakeSubprocess:
    """Stands in for the `subprocess` module inside harness_lib; records every leg invocation."""

    PIPE = subprocess.PIPE
    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self, handler, timeout_once=False):
        self._handler = handler
        self._timeout_once = timeout_once
        self.calls = []
        self.procs = []

    def run(self, cmd, env=None, capture_output=False, text=False, timeout=None):
        self.calls.append({"cmd": list(cmd), "env": env or {}, "timeout": timeout})
        return self._handler(list(cmd), env or {})

    def Popen(self, cmd, env=None, stdout=None, stderr=None, text=False, start_new_session=False):
        call = {"cmd": list(cmd), "env": env or {}, "timeout": None,
                "start_new_session": start_new_session}
        self.calls.append(call)
        p = _FakePopen(self._handler(list(cmd), env or {}), call, self._timeout_once)
        self.procs.append(p)
        return p

    def flag(self, name, call=-1):
        """The value the recorded call passed for `--name`, or None if it was omitted."""
        cmd = self.calls[call]["cmd"]
        return cmd[cmd.index(name) + 1] if name in cmd else None

    def modes(self):
        return [c["cmd"][c["cmd"].index("--mode") + 1] for c in self.calls if "--mode" in c["cmd"]]


class _LegTestCase(_HarnessTestCase):
    """A real task dir on disk (leg_runner.py / kernel_src) + a scripted subprocess."""

    BASE_IDENT = {"module": "live_stack.moe", "qualname": "matmul_ogs",
                  "file": "/opt/live/stack/moe.py"}
    CAND_IDENT = {"module": "live_stack.moe", "qualname": "matmul_ogs",
                  "file": "/tmp/task/_cand_overlay/live_stack/moe.py"}

    def setUp(self):
        super().setUp()
        self.task = tempfile.mkdtemp(prefix="harness_legs_")
        self.addCleanup(shutil.rmtree, self.task, True)
        self.base = os.path.join(self.task, "baseline_overlay")
        self.cand = os.path.join(self.task, "_cand_overlay")
        os.makedirs(os.path.join(self.task, "kernel_src"))
        for rel in ("leg_runner.py", "overlay_setup.py", "kernel_src/moe.py"):
            with open(os.path.join(self.task, rel), "w") as fh:
                fh.write("# stub\n")
        os.makedirs(self.base)
        self.meta = {"target_callable": "live_stack.moe:matmul_ogs",
                     "candidate_bind": {"kind": "module", "module": "live_stack.moe",
                                        "file": "kernel_src/moe.py"}}
        self.addCleanup(setattr, hl, "subprocess", hl.subprocess)

    def _script(self, handler):
        self.sub = _FakeSubprocess(handler)
        hl.subprocess = self.sub
        return self.sub

    def _legs(self, base_ident=None, cand_ident=None, timings=None, sigs=("decode", "prefill")):
        """The standard happy-path server: resolve answers per overlay, list/time answer per bucket."""
        base_ident = self.BASE_IDENT if base_ident is None else base_ident
        cand_ident = self.CAND_IDENT if cand_ident is None else cand_ident
        timings = timings or {"decode": (2.0, 1.0), "prefill": (8.0, 4.0)}

        def handler(cmd, env):
            if "overlay_setup.py" in " ".join(cmd):
                return _Proc(0)
            mode = cmd[cmd.index("--mode") + 1]
            is_base = env.get("PYTHONPATH", "").split(os.pathsep)[0] == self.base
            if mode == "resolve":
                return _Proc(0, json.dumps(base_ident if is_base else cand_ident))
            if mode == "list":
                return _Proc(0, json.dumps({"sigs": list(sigs)}))
            if mode == "time":
                sig = cmd[cmd.index("--bucket") + 1]
                got = timings.get(sig)
                if got is None:
                    return _Proc(0, json.dumps({"cases": []}))
                ms = got[0] if is_base else got[1]
                return _Proc(0, json.dumps({"cases": [{"sig": sig, "ms": ms, "regime": "decode",
                                                       "m": 64}]}))
            return _Proc(0, json.dumps({"out": self.flag_out, "n": 1}))
        self.flag_out = ""
        return self._script(handler)


class TestRunLeg(_LegTestCase):
    def test_the_overlay_goes_first_on_pythonpath(self):
        """Last wins on PYTHONPATH would make the overlay shadow nothing and both legs identical."""
        with _env(PYTHONPATH="/pre/existing"):
            self._script(lambda cmd, env: _Proc(0, '{"ok": true}'))
            hl._run_leg(self.task, self.base, "list")
        self.assertEqual(self.sub.calls[0]["env"]["PYTHONPATH"],
                         os.pathsep.join([self.base, "/pre/existing"]))

    def test_an_absent_pythonpath_is_not_turned_into_an_empty_entry(self):
        """A trailing ':' on PYTHONPATH puts CWD on sys.path -- where the task dir's unittest.py is."""
        with _env(PYTHONPATH=None):
            self._script(lambda cmd, env: _Proc(0, '{"ok": true}'))
            hl._run_leg(self.task, self.base, "list")
        self.assertEqual(self.sub.calls[0]["env"]["PYTHONPATH"], self.base)

    def test_the_leg_runs_the_task_dirs_own_leg_runner(self):
        self._script(lambda cmd, env: _Proc(0, '{"ok": true}'))
        hl._run_leg(self.task, self.base, "list", seed=7)
        cmd = self.sub.calls[0]["cmd"]
        self.assertEqual(cmd[:2], [sys.executable, os.path.join(self.task, "leg_runner.py")])
        self.assertEqual(self.sub.flag("--task"), self.task)
        self.assertEqual(self.sub.flag("--seed"), "7")

    def test_optional_flags_are_omitted_when_unset(self):
        self._script(lambda cmd, env: _Proc(0, "{}"))
        hl._run_leg(self.task, self.base, "list")
        for flag in ("--bucket", "--out", "--draws"):
            self.assertIsNone(self.sub.flag(flag), flag)

    def test_optional_flags_are_passed_when_set(self):
        self._script(lambda cmd, env: _Proc(0, "{}"))
        hl._run_leg(self.task, self.base, "oracle", bucket="decode", out="/tmp/o.pt", draws=5)
        self.assertEqual(self.sub.flag("--bucket"), "decode")
        self.assertEqual(self.sub.flag("--out"), "/tmp/o.pt")
        self.assertEqual(self.sub.flag("--draws"), "5")

    def test_the_timeout_is_forwarded_to_the_subprocess(self):
        self._script(lambda cmd, env: _Proc(0, "{}"))
        hl._run_leg(self.task, self.base, "list", timeout=42)
        self.assertEqual(self.sub.calls[0]["timeout"], 42)

    def test_only_the_last_json_line_is_read(self):
        """A leg prints warnings and progress; the payload is the final JSON object, not the first."""
        self._script(lambda cmd, env: _Proc(
            0, 'loading weights...\n{"sigs": ["stale"]}\nwarming up\n{"sigs": ["decode"]}\n'))
        self.assertEqual(hl._run_leg(self.task, self.base, "list"), {"sigs": ["decode"]})

    def test_a_nonzero_exit_raises_with_the_tail_of_stderr(self):
        self._script(lambda cmd, env: _Proc(3, "", "x" * 900 + "SEGFAULT in leg"))
        with self.assertRaises(RuntimeError) as cm:
            hl._run_leg(self.task, self.base, "time")
        self.assertIn("leg(time)", str(cm.exception))
        self.assertIn("exited 3", str(cm.exception))
        self.assertIn("SEGFAULT in leg", str(cm.exception))

    def test_a_leg_that_printed_no_json_raises_rather_than_returning_nothing(self):
        """Exit 0 with no payload must not degrade into an empty measurement."""
        self._script(lambda cmd, env: _Proc(0, "everything is fine\n"))
        with self.assertRaises(RuntimeError) as cm:
            hl._run_leg(self.task, self.base, "list")
        self.assertIn("produced no JSON", str(cm.exception))


class TestLegTimeoutReleasesTheGpu(_LegTestCase):
    """A timed-out leg must take everything it spawned with it.

    Killing only the direct child leaves a helper holding the GPU allocation, and the NEXT leg then
    times against a busy device -- a slowdown attributable to nothing, in the one code path whose
    whole purpose is to compare two legs fairly."""

    def test_the_leg_is_started_in_its_own_session(self):
        """No new session => the pgid is OURS, and there is no group to kill but our own."""
        self._script(lambda cmd, env: _Proc(0, "{}"))
        hl._run_leg(self.task, self.base, "list")
        self.assertTrue(self.sub.calls[0]["start_new_session"])

    def _timing_out_leg(self):
        self.sub = _FakeSubprocess(lambda cmd, env: _Proc(0, "{}"), timeout_once=True)
        hl.subprocess = self.sub
        return self.sub

    def test_a_timeout_signals_the_whole_group_and_still_raises(self):
        killed = []
        sub = self._timing_out_leg()

        def _killpg(pgid, sig):
            killed.append((pgid, sig))
            sub.procs[0].returncode = -int(sig)      # SIGTERM is honoured
        with _patched(os, killpg=_killpg, getpgid=lambda pid: 999 if pid else 1):
            with self.assertRaises(subprocess.TimeoutExpired):
                hl._run_leg(self.task, self.base, "time", timeout=1)
        self.assertEqual(killed, [(999, signal.SIGTERM)])

    def test_a_group_that_ignores_sigterm_is_escalated_to_sigkill(self):
        """A leg wedged in a kernel launch will not act on SIGTERM; the GPU is freed by SIGKILL."""
        killed = []
        sub = self._timing_out_leg()

        def _killpg(pgid, sig):
            killed.append((pgid, sig))
            if sig == signal.SIGKILL:
                sub.procs[0].returncode = -9
        with _patched(os, killpg=_killpg, getpgid=lambda pid: 999 if pid else 1):
            with self.assertRaises(subprocess.TimeoutExpired):
                hl._run_leg(self.task, self.base, "time", timeout=1)
        self.assertEqual([s for _, s in killed], [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual({pgid for pgid, _ in killed}, {999})

    def test_our_own_group_is_never_signalled(self):
        """If start_new_session did not take, killpg would take the harness down with the leg."""
        killed = []
        self._timing_out_leg()
        with _patched(os, killpg=lambda pgid, sig: killed.append((pgid, sig)),
                      getpgid=lambda pid: 4242):     # child's group == our own
            with self.assertRaises(subprocess.TimeoutExpired):
                hl._run_leg(self.task, self.base, "time", timeout=1)
        self.assertEqual(killed, [])
        self.assertTrue(self.sub.procs[0].killed)

    def test_a_group_that_is_already_gone_is_not_an_error(self):
        self._timing_out_leg()

        def _gone(pid):
            raise ProcessLookupError(pid)
        with _patched(os, getpgid=_gone):
            with self.assertRaises(subprocess.TimeoutExpired):
                hl._run_leg(self.task, self.base, "time", timeout=1)

    def test_a_real_grandchild_does_not_outlive_the_timeout(self):
        """The end-to-end claim, against the real OS: reaped, not merely signalled."""
        marker = os.path.join(self.task, "grandchild.pid")
        with open(os.path.join(self.task, "leg_runner.py"), "w") as fh:
            fh.write(
                "import os, subprocess, sys, time\n"
                "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                f"open({marker!r}, 'w').write(str(p.pid))\n"
                "time.sleep(120)\n")
        with self.assertRaises(subprocess.TimeoutExpired):
            hl._run_leg(self.task, self.base, "list", timeout=5)
        with open(marker) as fh:
            gpid = int(fh.read())
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(gpid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        self.fail(f"grandchild {gpid} survived the leg timeout and still holds the device")


class TestBuildCandidateOverlay(_LegTestCase):
    def test_the_candidate_is_the_baseline_plus_one_module_entry(self):
        self._script(lambda cmd, env: _Proc(0))
        base, cand = hl.build_candidate_overlay(self.task, self.meta)
        self.assertEqual((base, cand), (self.base, self.cand))
        cmd = self.sub.calls[0]["cmd"]
        self.assertEqual(cmd[:2], [sys.executable, os.path.join(self.task, "overlay_setup.py")])
        self.assertIn("add-module", cmd)
        self.assertEqual(self.sub.flag("--module"), "live_stack.moe")
        self.assertEqual(self.sub.flag("--patched-file"), os.path.join(self.task, "kernel_src/moe.py"))
        self.assertEqual(self.sub.flag("--overlay"), self.cand)
        self.assertEqual(self.sub.flag("--from"), self.base)

    def test_a_rebind_bind_uses_the_rebind_subcommand(self):
        """Same {kind,target,file} the e2e Integrator uses, so an isolated win is rebindable e2e."""
        self.meta["candidate_bind"] = {"kind": "rebind", "target": "live_stack.moe:matmul_ogs",
                                       "impl_module": "geak_kernels.moe", "impl_attr": "matmul_ogs",
                                       "file": "kernel_src/moe.py"}
        self._script(lambda cmd, env: _Proc(0))
        hl.build_candidate_overlay(self.task, self.meta)
        self.assertIn("add-rebind", self.sub.calls[0]["cmd"])
        self.assertEqual(self.sub.flag("--target"), "live_stack.moe:matmul_ogs")
        self.assertEqual(self.sub.flag("--impl-module"), "geak_kernels.moe")
        self.assertEqual(self.sub.flag("--impl-attr"), "matmul_ogs")

    def test_a_stale_candidate_overlay_is_rebuilt_from_scratch(self):
        """kernel_src/ is the optimizer's live workspace; a reused overlay measures yesterday's code."""
        os.makedirs(self.cand)
        stale = os.path.join(self.cand, "yesterday.py")
        with open(stale, "w") as fh:
            fh.write("# previous iteration\n")
        self._script(lambda cmd, env: _Proc(0))
        hl.build_candidate_overlay(self.task, self.meta)
        self.assertFalse(os.path.exists(stale))

    def test_a_missing_candidate_bind_refuses_to_build(self):
        for bind in ({}, None, {"kind": "module", "module": "m"}):
            with self.subTest(bind=bind):
                self.meta["candidate_bind"] = bind
                self._script(lambda cmd, env: _Proc(0))
                with self.assertRaises(RuntimeError) as cm:
                    hl.build_candidate_overlay(self.task, self.meta)
                self.assertIn("meta.candidate_bind missing", str(cm.exception))

    def test_a_candidate_bind_pointing_at_an_absent_file_refuses_to_build(self):
        self.meta["candidate_bind"]["file"] = "kernel_src/never_written.py"
        self._script(lambda cmd, env: _Proc(0))
        with self.assertRaises(RuntimeError) as cm:
            hl.build_candidate_overlay(self.task, self.meta)
        self.assertIn("its file is absent", str(cm.exception))

    def test_an_overlay_tool_failure_raises_with_the_tail_of_stderr(self):
        self._script(lambda cmd, env: _Proc(1, "", "SyntaxError in kernel_src/moe.py"))
        with self.assertRaises(RuntimeError) as cm:
            hl.build_candidate_overlay(self.task, self.meta)
        self.assertIn("candidate overlay build failed", str(cm.exception))
        self.assertIn("SyntaxError", str(cm.exception))


class TestAssertLegsDiffer(_LegTestCase):
    def test_two_legs_importing_different_code_are_accepted(self):
        self._legs()
        bi, ci = hl.assert_legs_differ(self.task, self.base, self.cand, self.meta)
        self.assertEqual((bi, ci), (self.BASE_IDENT, self.CAND_IDENT))
        self.assertEqual(self.sub.modes(), ["resolve", "resolve"])

    def test_identical_identities_are_refused_as_unmeasurable(self):
        """Both legs on the same code: correctness still passes and only the RATIO is meaningless."""
        self._legs(cand_ident=self.BASE_IDENT)
        with self.assertRaises(RuntimeError) as cm:
            hl.assert_legs_differ(self.task, self.base, self.cand, self.meta)
        self.assertIn("SAME code", str(cm.exception))
        self.assertIn("meta.candidate_bind", str(cm.exception))

    def test_a_baseline_resolving_inside_the_task_dir_is_refused(self):
        """A baseline the optimizer can edit is the old self-referential bug; never measure it."""
        inside = dict(self.BASE_IDENT, file=os.path.join(self.task, "baseline_src", "moe.py"))
        self._legs(base_ident=inside)
        with self.assertRaises(RuntimeError) as cm:
            hl.assert_legs_differ(self.task, self.base, self.cand, self.meta)
        self.assertIn("resolved INSIDE the task dir", str(cm.exception))

    def test_a_task_dir_prefix_that_is_not_a_path_boundary_is_not_a_false_positive(self):
        outside = dict(self.BASE_IDENT, file=self.task + "_other/moe.py")
        self._legs(base_ident=outside)
        bi, _ = hl.assert_legs_differ(self.task, self.base, self.cand, self.meta)
        self.assertEqual(bi["file"], outside["file"])

    def test_either_leg_failing_to_import_the_target_is_named(self):
        for who, base_bad in (("baseline", True), ("candidate", False)):
            with self.subTest(leg=who):
                bad = {"error": "ModuleNotFoundError: live_stack"}
                self._legs(base_ident=bad if base_bad else None,
                           cand_ident=None if base_bad else bad)
                with self.assertRaises(RuntimeError) as cm:
                    hl.assert_legs_differ(self.task, self.base, self.cand, self.meta)
                self.assertIn(f"{who} leg cannot import live_stack.moe:matmul_ogs", str(cm.exception))
                self.assertIn("ModuleNotFoundError", str(cm.exception))


class TestMeasureLegs(_LegTestCase):
    def test_every_bucket_is_timed_in_a_fresh_subprocess_per_leg(self):
        """A warm interpreter shares JIT/autotune state between legs and reports a fake 1.0x."""
        self._legs()
        per = hl.measure_legs(self.task, self.meta)
        self.assertEqual([c["sig"] for c in per], ["decode", "prefill"])
        self.assertEqual([c["speedup"] for c in per], [2.0, 2.0])
        self.assertEqual(per[0]["baseline_ms"], 2.0)
        self.assertEqual(per[0]["optimized_ms"], 1.0)
        self.assertEqual((per[0]["regime"], per[0]["m"]), ("decode", 64))
        # overlay build, 2x resolve, list, then one time-leg per (bucket, leg)
        self.assertEqual(self.sub.modes(), ["resolve", "resolve", "list",
                                            "time", "time", "time", "time"])

    def test_measurement_is_refused_before_the_legs_are_proven_different(self):
        """assert_legs_differ runs BEFORE any timing."""
        self._legs(cand_ident=self.BASE_IDENT)
        with self.assertRaises(RuntimeError):
            hl.measure_legs(self.task, self.meta)
        self.assertNotIn("time", self.sub.modes())

    def test_a_bucket_with_no_cases_on_either_leg_is_skipped(self):
        self._legs(timings={"decode": (2.0, 1.0)}, sigs=("decode", "vanished"))
        per = hl.measure_legs(self.task, self.meta)
        self.assertEqual([c["sig"] for c in per], ["decode"])

    def test_an_unmeasurable_bucket_reports_a_null_speedup_not_a_ratio(self):
        """ms=None (timer unavailable) must never silently become 1.0x."""
        self._legs(timings={"decode": (None, 1.0), "prefill": (8.0, 0.0)})
        per = hl.measure_legs(self.task, self.meta)
        self.assertEqual([c["speedup"] for c in per], [None, None])

    def test_an_unmeasurable_bucket_does_not_burn_the_rep_budget(self):
        """No timer means no pair can decide it; retrying is pure cost."""
        self._legs(timings={"decode": (None, None)}, sigs=("decode",))
        hl.measure_legs(self.task, self.meta)
        self.assertEqual(self.sub.modes().count("time"), 2)


class TestMeasureLegsResolution(_LegTestCase):
    """How many pairs a bucket costs, and in what order the legs run.

    Fresh-per-bucket is deliberate, but it is also what leaves one B,C pair exposed to fresh-process
    variance -- ~0.1-0.5% on prefill, up to ~14% on small-M decode on gfx950. A 1.05x candidate is
    inside that noise, and 1.05x is precisely the size of win the direction fix exists to recover."""

    def _sequenced(self, seq, sigs=("decode",)):
        """A leg server whose `time` answers come from a per-sig LIST, one entry consumed per pair."""
        state = {s: 0 for s in sigs}
        order = []

        def handler(cmd, env):
            if "overlay_setup.py" in " ".join(cmd):
                return _Proc(0)
            mode = cmd[cmd.index("--mode") + 1]
            is_base = env.get("PYTHONPATH", "").split(os.pathsep)[0] == self.base
            if mode == "resolve":
                return _Proc(0, json.dumps(self.BASE_IDENT if is_base else self.CAND_IDENT))
            if mode == "list":
                return _Proc(0, json.dumps({"sigs": list(sigs)}))
            sig = cmd[cmd.index("--bucket") + 1]
            order.append(("B" if is_base else "C", sig))
            pair = seq[sig][min(state[sig], len(seq[sig]) - 1)]
            if not is_base:
                state[sig] += 1                      # a pair completes on the candidate leg
            ms = pair[0] if is_base else pair[1]
            return _Proc(0, json.dumps({"cases": [{"sig": sig, "ms": ms, "regime": "decode",
                                                   "m": 64}]}))
        self._script(handler)
        return order

    def test_a_decisive_bucket_still_costs_exactly_one_pair(self):
        """The common case must not get 3x slower to buy resolution it does not need."""
        self._sequenced({"decode": [(2.0, 1.0)]})
        per = hl.measure_legs(self.task, self.meta)
        self.assertEqual(self.sub.modes().count("time"), 2)
        self.assertEqual(per[0]["reps"], 1)
        self.assertEqual(per[0]["speedup"], 2.0)
        self.assertEqual(per[0]["speedup_spread"], 0.0)

    def test_a_decisive_slowdown_also_stops_after_one_pair(self):
        """Below the band is as decisive as above it."""
        self._sequenced({"decode": [(1.0, 2.0)]})
        hl.measure_legs(self.task, self.meta)
        self.assertEqual(self.sub.modes().count("time"), 2)

    def test_an_undecided_bucket_spends_the_full_budget(self):
        """1.05x is the motivating case: inside the noise, so it is the one worth re-measuring."""
        self._sequenced({"decode": [(1.055, 1.0)] * 3})
        per = hl.measure_legs(self.task, self.meta)
        self.assertEqual(self.sub.modes().count("time"), 6)
        self.assertEqual(per[0]["reps"], 3)

    def test_the_legs_are_interleaved_rather_than_run_all_baseline_then_all_candidate(self):
        """B,B,B,C,C,C lets drift over the sweep accrue to the candidate; B,C,B,C makes it common-mode."""
        order = self._sequenced({"decode": [(1.02, 1.0)] * 3})
        hl.measure_legs(self.task, self.meta)
        self.assertEqual([leg for leg, _ in order], ["B", "C", "B", "C", "B", "C"])

    def test_the_reported_time_is_the_median_pair_not_the_first(self):
        """One unlucky cold process must not carry the bucket."""
        self._sequenced({"decode": [(1.0, 1.0), (1.04, 1.0), (9.0, 1.0)]})
        per = hl.measure_legs(self.task, self.meta)
        self.assertEqual(per[0]["baseline_ms"], 1.04)
        self.assertEqual(per[0]["speedup"], 1.04)

    def test_the_spread_across_pairs_is_reported(self):
        """A bucket that disagrees with itself must be visible, not averaged into silence."""
        self._sequenced({"decode": [(1.0, 1.0), (1.04, 1.0), (1.08, 1.0)]})
        per = hl.measure_legs(self.task, self.meta)
        self.assertAlmostEqual(per[0]["speedup_spread"], (1.08 - 1.0) / 1.04, places=12)

    def test_a_bucket_that_turns_decisive_on_a_later_pair_stops_there(self):
        """The check is on the RUNNING median, so the budget is released as soon as it is not needed."""
        self._sequenced({"decode": [(1.02, 1.0), (3.0, 1.0), (3.0, 1.0)]})
        hl.measure_legs(self.task, self.meta)
        self.assertEqual(self.sub.modes().count("time"), 4)

    def test_max_reps_one_restores_the_single_pair_behaviour(self):
        self._sequenced({"decode": [(1.055, 1.0)] * 3})
        per = hl.measure_legs(self.task, self.meta, max_reps=1)
        self.assertEqual(self.sub.modes().count("time"), 2)
        self.assertEqual(per[0]["reps"], 1)

    def test_the_undecided_band_is_caller_controlled(self):
        """A widened band makes a 1.5x bucket worth re-measuring; the default would not."""
        self._sequenced({"decode": [(1.5, 1.0)] * 3})
        hl.measure_legs(self.task, self.meta, undecided=(0.5, 2.0))
        self.assertEqual(self.sub.modes().count("time"), 6)

    def test_the_budget_is_per_bucket_not_shared(self):
        """A noisy decode bucket must not consume the reps a second bucket would have used."""
        self._sequenced({"decode": [(1.02, 1.0)] * 3, "prefill": [(1.03, 1.0)] * 3},
                        sigs=("decode", "prefill"))
        per = hl.measure_legs(self.task, self.meta)
        self.assertEqual([c["reps"] for c in per], [3, 3])
        self.assertEqual(self.sub.modes().count("time"), 12)


class TestBaselineRandomOutputs(_LegTestCase):
    def test_the_oracle_is_recorded_by_the_baseline_leg_in_its_own_process(self):
        """Same seed => same inputs on both sides, so the legs never have to be co-resident."""
        loaded = []
        self.torch.load = lambda path, map_location=None: loaded.append((path, map_location)) or {
            "decode|0": _T((2,), [1.0, 2.0])}
        self._script(lambda cmd, env: _Proc(0, '{"out": "x", "n": 2}'))
        got = hl.baseline_random_outputs(self.task, self.meta, seed=100, draws=2)

        dest = os.path.join(self.task, "_baseline_random.pt")
        self.assertEqual(self.sub.calls[0]["env"]["PYTHONPATH"].split(os.pathsep)[0], self.base)
        self.assertEqual(self.sub.modes(), ["oracle"])
        self.assertEqual(self.sub.flag("--out"), dest)
        self.assertEqual(self.sub.flag("--seed"), "100")
        self.assertEqual(self.sub.flag("--draws"), "2")
        self.assertEqual(loaded, [(dest, "cpu")])          # CPU-side, so either leg can compare it
        self.assertEqual(list(got), ["decode|0"])


# --------------------------------------------------------------------------- #
# The recorded-oracle arm of check_random_vs_baseline / run_correctness
# --------------------------------------------------------------------------- #
class TestCheckRandomVsBaselineRecorded(_HarnessTestCase):
    """`baseline_outputs=` replaces the in-process `baseline_call` closure entirely. Timing then
    belongs to measure_legs, so speedup must be None."""

    def _shape(self, sig="m1"):
        return {"sig": sig, "make_inputs": lambda rng: (1.0, 2.0)}

    def test_the_recorded_output_is_compared_and_no_speedup_is_reported(self):
        ok, per = hl.check_random_vs_baseline(
            None, _echo_call, [self._shape()], 0.01, draws=2,
            baseline_outputs={"m1|0": _T((2,), [1.0, 2.0]), "m1|1": _T((2,), [1.0, 2.0])})
        self.assertTrue(ok)
        self.assertEqual([e["case"] for e in per], ["random[0]:m1", "random[1]:m1"])
        self.assertEqual([e["speedup"] for e in per], [None, None])
        self.assertEqual([e["max_rel_err"] for e in per], [0.0, 0.0])

    def test_the_recorded_output_is_moved_onto_the_candidates_device(self):
        """The oracle is loaded map_location='cpu'; comparing it cross-device would raise."""
        ref = _T((2,), [1.0, 2.0], device="cpu")
        ok, _ = hl.check_random_vs_baseline(
            None, lambda args: _T((2,), list(args), device="cuda"), [self._shape()], 0.01,
            draws=1, baseline_outputs={"m1|0": ref})
        self.assertTrue(ok)
        self.assertEqual(ref.device, "cpu")                # the recorded oracle is not mutated

    def test_a_drift_from_the_recorded_baseline_still_fails(self):
        ok, per = hl.check_random_vs_baseline(
            None, lambda args: _T((2,), [v + 10.0 for v in args]), [self._shape()], 0.01,
            draws=1, baseline_outputs={"m1|0": _T((2,), [1.0, 2.0])})
        self.assertFalse(ok)
        self.assertFalse(per[0]["correct"])

    def test_a_missing_recorded_draw_is_a_failure_not_a_silent_skip(self):
        """A short oracle would otherwise shrink the parity check without anyone noticing."""
        ok, per = hl.check_random_vs_baseline(
            None, _echo_call, [self._shape()], 0.01, draws=2,
            baseline_outputs={"m1|0": _T((2,), [1.0, 2.0])})
        self.assertFalse(ok)
        self.assertTrue(per[0]["correct"])
        self.assertFalse(per[1]["correct"])
        self.assertIn("no recorded baseline output for m1|1", per[1]["note"])

    def test_a_tuple_returning_op_is_compared_component_wise_end_to_end(self):
        """The oracle now records `(out, lse)` as a tuple; the parity leg must pair it up, not
        report the candidate incorrect because the return value is not one tensor."""
        ok, per = hl.check_random_vs_baseline(
            None, lambda args: (_T((2,), list(args)), _T((1,), [9.0])), [self._shape()], 0.01,
            draws=1, baseline_outputs={"m1|0": (_T((2,), [1.0, 2.0]), _T((1,), [9.0]))})
        self.assertTrue(ok, per[0].get("note"))
        self.assertEqual(per[0]["max_rel_err"], 0.0)

    def test_every_component_of_a_recorded_tuple_is_moved_onto_the_candidates_device(self):
        """Moving only the first would raise a cross-device compare on the second."""
        ref = (_T((2,), [1.0, 2.0], device="cpu"), _T((1,), [9.0], device="cpu"))
        ok, per = hl.check_random_vs_baseline(
            None, lambda args: (_T((2,), list(args), device="cuda"),
                                _T((1,), [9.0], device="cuda")),
            [self._shape()], 0.01, draws=1, baseline_outputs={"m1|0": ref})
        self.assertTrue(ok, per[0].get("note"))
        self.assertEqual([t.device for t in ref], ["cpu", "cpu"])   # oracle not mutated

    def test_a_candidate_that_returns_no_tensor_is_named_rather_than_compared(self):
        ok, per = hl.check_random_vs_baseline(
            None, lambda args: None, [self._shape()], 0.01,
            draws=1, baseline_outputs={"m1|0": _T((2,), [1.0, 2.0])})
        self.assertFalse(ok)
        self.assertIn("no tensor", per[0]["note"])


class TestRunCorrectnessNeedsABaseline(_HarnessTestCase):
    def test_neither_a_recorded_oracle_nor_a_baseline_call_is_a_hard_error(self):
        """Running the random leg with no baseline at all would pass vacuously."""
        with self.assertRaises(ValueError) as cm:
            hl.run_correctness({"enforce_eager": True}, eager_cases=[], current_call=_echo_call,
                               random_shapes=[], tol=0.01)
        self.assertIn("baseline_outputs", str(cm.exception))
        self.assertIn("baseline_random_outputs", str(cm.exception))


def _swm_meta(**kw):
    swm = {"isl": 16384, "osl": 1024, "conc": 64, "prefill_chunk": 8192,
           "analytic_calls": {"prefill": 64, "decode": 1024}}
    swm.update(kw.pop("swm", {}))
    meta = {"m_buckets": [64, 1024, 8192], "workload": {"serving_weight_model": swm}}
    meta.update(kw)
    return meta


class ServedBucketsTest(unittest.TestCase):
    """`served_buckets` replaces `max(m_buckets)` as the answer to "what shape do we bench?".

    The old premise -- largest bucket == the GPU-time mass -- is false: mass is per-launch time x
    LAUNCHES, and decode runs OSL passes at the small bucket against prefill's CONC*ceil(ISL/chunk)
    passes at the large one. These pin the mapping from the serving model to the shapes to bench."""

    def test_both_served_phases_are_returned_ordered_by_pass_count(self):
        self.assertEqual(hl.served_buckets(_swm_meta()),
                         [("decode", 64, 1024), ("prefill", 8192, 64)])

    def test_the_largest_bucket_is_not_the_most_served_one(self):
        # the exact defect: max(m_buckets) is 8192, but 1024 of the 1088 served passes are at M=64.
        buckets = hl.served_buckets(_swm_meta())
        self.assertEqual(max(m for _, m, _ in buckets), 8192)
        self.assertEqual(buckets[0][1], 64)

    def test_a_phase_m_snaps_to_the_nearest_profiled_bucket(self):
        # conc=100 was never profiled; the closest shape the profile actually saw is 64.
        m = _swm_meta(swm={"conc": 100})
        self.assertEqual(dict((p, b) for p, b, _ in hl.served_buckets(m))["decode"], 64)

    def test_a_phase_the_kernel_does_not_serve_carries_no_weight(self):
        m = _swm_meta(served_regimes=["decode"])
        self.assertEqual(hl.served_buckets(m), [("decode", 64, 1024)])

    def test_prefill_falls_back_to_isl_when_the_server_does_not_chunk(self):
        m = _swm_meta(swm={"prefill_chunk": None}, m_buckets=[64, 16384])
        self.assertEqual(dict((p, b) for p, b, _ in hl.served_buckets(m))["prefill"], 16384)

    def test_one_shape_serving_both_phases_is_reported_once(self):
        # conc == chunk: both phases land on the same bucket; it must not be benched twice.
        m = _swm_meta(swm={"conc": 8192}, m_buckets=[8192])
        self.assertEqual(hl.served_buckets(m), [("decode", 8192, 1024)])

    def test_no_serving_model_leaves_the_caller_on_its_old_behaviour(self):
        self.assertEqual(hl.served_buckets({"m_buckets": [8192]}), [])
        self.assertEqual(hl.served_buckets({}), [])
        self.assertEqual(hl.served_buckets(None), [])

    def test_unusable_call_counts_are_dropped_not_guessed(self):
        m = _swm_meta(swm={"analytic_calls": {"prefill": 0, "decode": "x", "edge": 5}})
        self.assertEqual(hl.served_buckets(m), [])

    def test_buckets_are_used_only_to_snap_never_to_invent_a_phase(self):
        # a profiled bucket with no serving phase behind it is not benched.
        m = _swm_meta(m_buckets=[64, 512, 8192])
        self.assertNotIn(512, [b for _, b, _ in hl.served_buckets(m)])


class TestOracleSharedAndLazy(_HarnessTestCase):
    """Issue #429: shared weight refs + per-case lazy correctness keep MoE oracles from OOM'ing."""

    def test_resolve_oracle_shared_expands_refs(self):
        shared = {"w1": {"__tensor__": True, "data": _T((2, 2), fill=3.0)}}
        obj = {"w1": {"__shared__": "w1"}, "hidden": 1}
        out = hl.resolve_oracle_shared(obj, shared)
        self.assertIs(out["w1"], shared["w1"])
        self.assertEqual(out["hidden"], 1)

    def test_resolve_oracle_shared_missing_key_raises(self):
        with self.assertRaises(KeyError):
            hl.resolve_oracle_shared({"w1": {"__shared__": "missing"}}, {})

    def test_reconstruct_captured_tensor_leaf(self):
        leaf = {"__tensor__": True, "data": _T((2,), fill=1.5)}
        out = hl.reconstruct_captured(leaf, device="cpu")
        self.assertTrue(hasattr(out, "shape"))
        self.assertEqual(tuple(out.shape), (2,))

    def test_check_correct_multi_lazy_runs(self):
        """Smoke: lazy checker iterates cases and returns per-case rows."""
        def call(args):
            return args["ref"]

        cases = [
            {"args": {"ref": _T((2,), fill=1.0)}, "ref": _T((2,), fill=1.0), "sig": "a"},
        ]
        _ok, per = hl.check_correct_multi_lazy(
            call, iter(cases), {"rtol": 1e-5, "atol": 1e-5}, max_keep_live=1)
        self.assertEqual(len(per), 1)
        self.assertIn("correct", per[0])
        self.assertIn("case", per[0])

    def test_to_device_like_walks_containers(self):
        leaf = _T((2,), fill=1.0)
        out = hl.to_device_like({"a": leaf, "b": [leaf]}, "cpu")
        self.assertIn("a", out)
        self.assertEqual(len(out["b"]), 1)

    def test_reconstruct_captured_repr_and_containers(self):
        self.assertEqual(hl.reconstruct_captured({"__repr__": "x"}, "cpu"), "x")
        nested = hl.reconstruct_captured(
            {"t": {"__tensor__": True, "data": _T((1,), fill=2.0)},
             "xs": [{"__tensor__": True, "data": _T((1,), fill=3.0)}]},
            "cpu")
        self.assertEqual(tuple(nested["t"].shape), (1,))
        self.assertEqual(tuple(nested["xs"][0].shape), (1,))

    def test_load_reference_io_supports_legacy_torch_load_signature(self):
        path = "oracle.pt"
        calls = []

        def load(p, map_location=None, weights_only=None):
            calls.append({"weights_only": weights_only})
            if weights_only is not None:
                raise TypeError("old torch")
            return {"records": [], "shared": {}}

        self.torch.load = load
        blob = hl.load_reference_io(path)
        self.assertEqual(blob["records"], [])
        self.assertEqual(calls[0]["weights_only"], False)
        self.assertIsNone(calls[1]["weights_only"])

        self.torch.load = lambda *a, **k: [1, 2, 3]
        with self.assertRaises(TypeError):
            hl.load_reference_io(path)

    def test_iter_eager_cases_from_oracle_resolves_shared_pool(self):
        shared_w = {"__tensor__": True, "data": _T((2, 2), fill=4.0)}
        blob = {
            "shared": {"w1": shared_w},
            "records": [{
                "sig": "s0",
                "regime": "decode",
                "args": (),
                "kwargs": {
                    "w1": {"__shared__": "w1"},
                    "hidden_states": {"__tensor__": True, "data": _T((2,), fill=1.0)},
                },
                "output": {"__tensor__": True, "data": _T((2,), fill=2.0)},
            }],
        }
        self.torch.load = lambda *a, **k: blob
        cases = hl.eager_cases_from_oracle("ref.pt", device="cpu")
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["sig"], "s0")
        self.assertEqual(cases[0]["regime"], "decode")
        self.assertTrue(hasattr(cases[0]["args"]["w1"], "shape"))
        self.assertEqual(tuple(cases[0]["ref"].shape), (2,))

    def test_iter_eager_cases_merges_positional_moe_names_into_kwargs(self):
        blob = {
            "shared": {},
            "records": [{
                "sig": "moe",
                "args": (
                    {"__tensor__": True, "data": _T((2,), fill=1.0)},
                    {"__tensor__": True, "data": _T((2,), fill=2.0)},
                ),
                "kwargs": {"scale": 1.0},
                "output": {"__tensor__": True, "data": _T((2,), fill=3.0)},
            }],
        }
        self.torch.load = lambda *a, **k: blob
        case = next(hl.iter_eager_cases_from_oracle("ref.pt"))
        self.assertIn("hidden_states", case["args"])
        self.assertIn("w1", case["args"])
        self.assertEqual(case["args"]["scale"], 1.0)

    def test_check_correct_multi_lazy_runs_independence_with_two_cases(self):
        def call(args):
            # Return the oracle tensor itself so the value check passes; independence is a
            # separate synthetic row that still exercises the two-case branch.
            return args["ref"]

        cases = [
            {"args": {"ref": _T((2,), fill=1.0)}, "ref": _T((2,), fill=1.0), "sig": "a"},
            {"args": {"ref": _T((2,), fill=2.0)}, "ref": _T((2,), fill=2.0), "sig": "b"},
        ]
        _ok, per = hl.check_correct_multi_lazy(
            call, iter(cases), {"rtol": 1e-5, "atol": 1e-5}, max_keep_live=1)
        self.assertEqual(per[-1]["case"], "output_independence")
        self.assertIn("correct", per[-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
