#!/usr/bin/env python3
"""Unit tests for op_bench.py -- the single-op multi-backend bake-off DRIVER (stdlib only; no torch).

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_op_bench.py

op_bench.py produces the isolated GEMM/attention number that the Op Benchmarker routes on and that the
A/B judge consumes. Everything asserted here is DRIVER LOGIC, not GPU math:

  - operand construction : _resolve_shape (symbolic "M" -> m_buckets), _load_or_synth_gemm (recorded
                           oracle vs synthesized), _synth_blockscale_case (fp8 block-scale shapes)
  - op classification    : _is_blockscale_gemm / _is_grouped_or_quant_gemm -- which heads must NOT go
                           through the dense torch-BLAS bake-off at all
  - backend dispatch     : bench_gemm's per-backend ladder (hipblaslt / tunableop / rocblas / ck /
                           aiter / flydsl / triton), including the BLAS-preference restore order and
                           every "backend unavailable on this image" branch
  - measurement plumbing : _time_call (harness_lib path + the no-harness_lib fallback loop) and
                           _correct's scale-relative error
  - the emitted JSON     : main()'s winner/baseline selection, isolated_speedup, harness-self-fault
                           signal, Amdahl ceiling, and the per-winner DEPLOYABLE recipe (apply_env)

Why it matters operationally: a mis-marshalled operand or a mis-attributed winner here is not a crash,
it is a WRONG NUMBER that silently routes the whole kernel campaign -- e.g. a symbolic "M" dim reaching
randn() as a string, a grouped MoE GEMM reported as a harness self-fault, or a tunableop win emitted
without the PYTORCH_TUNABLEOP_FILENAME env that makes it deployable.

torch / triton / aiter are injected into sys.modules as fakes for the duration of each test (op_bench
imports all three LAZILY, inside functions and try: blocks, so the module itself imports on a CPU-only
box). The fake tensor tracks SHAPE + DTYPE + a single scalar value, which is exactly the resolution the
driver logic is written against: shapes/dtypes let us assert what each backend was handed, and the
scalar makes _correct's verdict deterministic (a stub that mirrors the oracle's contract compares
equal; one that does not is recorded incorrect). The fakes are installed per-test and removed in
tearDown so the other test modules in this directory keep seeing a torch-free image.
"""
import contextlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import tempfile
import types
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# op_bench.py does a bare `import harness_lib`. It self-inserts its own directory on sys.path at import
# time (op_bench.py line 32), but that only helps once op_bench itself is being executed -- and it is a
# global side effect we do not want to depend on for ordering. So put scripts/ on sys.path here,
# explicitly, BEFORE loading anything: scripts/ is not a package, so this is the only way the sibling
# `harness_lib` import inside op_bench resolves.
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


def _load(mod_name, filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_without_harness_lib(mod_name, filename):
    """Load op_bench with `import harness_lib` forced to fail, exercising its guarded fallback (`_hlib
    is None`) -- the old-checkout path that times with a naive wall-clock loop and resolves dtypes from
    its own local table. `None` in sys.modules is the documented way to make an import raise."""
    sentinel = object()
    prev = sys.modules.get("harness_lib", sentinel)
    sys.modules["harness_lib"] = None
    try:
        return _load(mod_name, filename)
    finally:
        if prev is sentinel:
            del sys.modules["harness_lib"]
        else:
            sys.modules["harness_lib"] = prev


# --------------------------------------------------------------------------- #
# fake torch tensor: tracks shape + dtype + one scalar value
# --------------------------------------------------------------------------- #
class _Dtype:
    def __init__(self, name, itemsize=2, fmax=65504.0):
        self.name = name
        self.itemsize = itemsize
        self.fmax = fmax

    @property
    def element_ty(self):        # triton reads c_ptr.dtype.element_ty
        return self

    def __repr__(self):
        return "torch.%s" % self.name


BF16 = _Dtype("bfloat16", 2, 3.3895e38)
FP16 = _Dtype("float16", 2, 65504.0)
FP32 = _Dtype("float32", 4, 3.4028e38)
INT8 = _Dtype("int8", 1, 127.0)
UINT8 = _Dtype("uint8", 1, 255.0)
FP8_E4M3FNUZ = _Dtype("float8_e4m3fnuz", 1, 240.0)
FP8_E5M2FNUZ = _Dtype("float8_e5m2fnuz", 1, 57344.0)
FP8_E4M3FN = _Dtype("float8_e4m3fn", 1, 448.0)
FP8_E5M2 = _Dtype("float8_e5m2", 1, 57344.0)

RANDN_VALUE = 1.0        # every element of a fake randn(); makes derived values reproducible


def _bcast(a, b):
    """Right-aligned elementwise broadcast. Permissive (takes the larger extent per dim) because the
    fake is also used for triton POINTER arithmetic, where a base pointer tensor is added to a tile of
    offsets. Real shape contracts are asserted explicitly by the tests, not inferred from a raise."""
    ra, rb = list(a), list(b)
    n = max(len(ra), len(rb))
    ra = [1] * (n - len(ra)) + ra
    rb = [1] * (n - len(rb)) + rb
    return tuple(max(x, y) for x, y in zip(ra, rb))


class _T:
    """A fake tensor: shape, dtype, device and a single scalar `val` shared by every element."""

    def __init__(self, shape, dtype=FP32, val=0.0, device="cpu"):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.val = float(val)
        self.device = device

    # ---- structure
    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    def dim(self):
        return len(self.shape)

    def element_size(self):
        return self.dtype.itemsize

    def stride(self, i=None):
        strides, acc = [], 1
        for s in reversed(self.shape):
            strides.append(acc)
            acc *= s
        strides.reverse()
        return tuple(strides) if i is None else strides[i]

    def _like(self, shape=None, dtype=None, val=None, device=None):
        return _T(self.shape if shape is None else shape,
                  self.dtype if dtype is None else dtype,
                  self.val if val is None else val,
                  self.device if device is None else device)

    def to(self, *args, **kw):
        dtype, device = kw.get("dtype"), kw.get("device")
        for a in args:
            if isinstance(a, str):
                device = a
            elif isinstance(a, _Dtype):
                dtype = a
            elif isinstance(a, _T):
                dtype, device = a.dtype, a.device
        return self._like(dtype=dtype or self.dtype, device=device or self.device)

    def float(self):
        return self.to(FP32)

    def contiguous(self):
        return self._like()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = [int(s) for s in shape]
        if -1 in shape:
            known = 1
            for s in shape:
                if s != -1:
                    known *= s
            shape[shape.index(-1)] = self.numel() // max(1, known)
        return self._like(shape=tuple(shape))

    def t(self):
        if self.dim() < 2:
            return self._like()
        if self.dim() > 2:
            raise RuntimeError("t() expects a tensor with <= 2 dimensions, but self is %dD" % self.dim())
        return self._like(shape=(self.shape[1], self.shape[0]))

    def repeat_interleave(self, n, dim=0):
        shape = list(self.shape)
        shape[dim] *= int(n)
        return self._like(shape=tuple(shape))

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        shape, pos = [], 0
        for it in idx:
            if it is None:
                shape.append(1)
                continue
            if pos >= len(self.shape):
                raise IndexError("too many indices for shape %s" % (self.shape,))
            n = self.shape[pos]
            pos += 1
            if isinstance(it, slice):
                shape.append(len(range(*it.indices(n))))
            elif isinstance(it, int):
                continue                        # integer index drops the dim
            else:
                raise TypeError("unsupported fake index %r" % (it,))
        shape.extend(self.shape[pos:])
        return self._like(shape=tuple(shape))

    def __setitem__(self, idx, value):
        self.val = value.val if isinstance(value, _T) else float(value)

    # ---- reductions / elementwise
    def abs(self):
        return self._like(val=abs(self.val))

    def amax(self, dim=None):
        dims = (dim,) if isinstance(dim, int) else tuple(dim or ())
        keep = [s for i, s in enumerate(self.shape) if i not in {d % self.dim() for d in dims}]
        return self._like(shape=tuple(keep))

    def max(self):
        return self._like(shape=())

    def min(self):
        return self._like(shape=())

    def item(self):
        return self.val

    def clamp(self, lo, hi):
        return self._like(val=min(max(self.val, lo), hi))

    def clamp_min(self, lo):
        return self._like(val=max(self.val, lo))

    def div(self, other):
        return self.__truediv__(other)

    def all(self):
        return bool(self.val)

    def _binop(self, other, op):
        if isinstance(other, _T):
            return _T(_bcast(self.shape, other.shape), self.dtype, op(self.val, other.val), self.device)
        return self._like(val=op(self.val, float(other)))

    def __add__(self, other):
        return self._binop(other, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binop(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binop(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._binop(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binop(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._binop(other, lambda a, b: b / a)

    def __neg__(self):
        return self._like(val=-self.val)

    def __lt__(self, other):
        return self._binop(other, lambda a, b: float(a < b))

    def __le__(self, other):
        return self._binop(other, lambda a, b: float(a <= b))

    def __gt__(self, other):
        return self._binop(other, lambda a, b: float(a > b))

    def __and__(self, other):
        return self._binop(other, lambda a, b: float(bool(a) and bool(b)))

    def __matmul__(self, other):
        a, b = self.shape, other.shape
        if len(a) < 2 or len(b) < 2:
            raise RuntimeError("matmul needs >=2D operands, got %s @ %s" % (a, b))
        if a[-1] != b[-2]:
            raise RuntimeError("matmul shape mismatch %s @ %s" % (a, b))
        shape = _bcast(a[:-2], b[:-2]) + (a[-2], b[-1])
        return _T(shape, self.dtype, self.val * other.val * a[-1], self.device)

    def __repr__(self):
        return "_T(shape=%s, dtype=%s, val=%r)" % (self.shape, self.dtype, self.val)


def _dequant_matmul(x, w, x_scale, w_scale, dtype=None):
    """The fp8 block-scale backend CONTRACT, mirrored: dequantize x with its [M, K/blk] per-token-block
    scales and w with its [N/blk, K/blk] per-weight-block scales, then out = x@w.T. Because it performs
    the same operations as _synth_blockscale_case's oracle, a driver that hands over the right operands
    in the right order compares EQUAL -- and one that swaps x_scale/w_scale cannot even broadcast."""
    blk_k = x.shape[-1] // x_scale.shape[-1]
    blk_n = w.shape[0] // w_scale.shape[0]
    x_deq = x.float() * x_scale.repeat_interleave(blk_k, dim=1)[:, :x.shape[-1]]
    w_full = w_scale.repeat_interleave(blk_n, dim=0).repeat_interleave(blk_k, dim=1)
    w_deq = w.float() * w_full[:w.shape[0], :w.shape[1]]
    return (x_deq @ w_deq.t()).to(dtype or BF16)


# --------------------------------------------------------------------------- #
# the fake torch / triton / aiter trio
# --------------------------------------------------------------------------- #
class _Stack:
    """Builds a fresh fake GPU stack and installs it in sys.modules for one test.

    Behaviour switches (set before the call under test) model the branches that actually fire in
    production: a backend missing from this image, a callable that raises, a BLAS switch the build
    does not support.
    """

    MODULES = ("torch", "torch.nn", "torch.nn.functional", "triton", "triton.language",
               "aiter", "aiter.ops", "aiter.ops.triton", "aiter.ops.flydsl",
               "aiter.ops.flydsl.utils", "aiter.ops.flydsl.gemm_kernels", "aiter.tuned_gemm")

    def __init__(self, cuda=False):
        self.cuda = cuda
        self.calls = []              # every fake kernel/backend invocation, in order
        self.launches = []           # triton kernel launches: {grid, args, meta}
        self.stores = []             # triton tl.store targets
        self.blas = []               # preferred_blas_library() arguments, in order
        self.tunable_calls = []
        self.syncs = 0
        self.blas_unsupported = ("ck",)
        self.have_tunable = True
        self.set_filename_raises = False
        self.write_file_raises = False
        self.flydsl_available = True
        self.aiter_all_fail = False
        self.bpreshuffle_raises = False
        self.blockscale_raises = False
        self.loaded_blob = None      # what fake torch.load() returns
        self._saved = {}
        self._build()

    # ---- torch
    def _build(self):
        stack = self
        torch = types.ModuleType("torch")
        for dt in (BF16, FP16, FP32, INT8, UINT8, FP8_E4M3FNUZ, FP8_E5M2FNUZ, FP8_E4M3FN, FP8_E5M2):
            setattr(torch, dt.name, dt)

        class _Finfo:
            def __init__(self, dt):
                self.max = dt.fmax
                self.min = -dt.fmax

        class _Generator:
            def __init__(self, device="cpu"):
                self.device = device
                self.seed = None

            def manual_seed(self, s):
                self.seed = int(s)
                stack.calls.append(("manual_seed", self.device, self.seed))
                return self

        def _randn(*shape, **kw):
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = tuple(shape[0])
            return _T(shape, kw.get("dtype") or FP32, RANDN_VALUE, kw.get("device") or "cpu")

        def _filled(val):
            def make(*shape, **kw):
                if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                    shape = tuple(shape[0])
                return _T(shape, kw.get("dtype") or FP32, val, kw.get("device") or "cpu")
            return make

        def _load_blob(path, map_location=None):
            stack.calls.append(("torch.load", os.path.basename(path), map_location))
            if stack.loaded_blob is None:
                raise RuntimeError("fake torch.load has no blob registered for %s" % path)
            return stack.loaded_blob

        cuda = types.SimpleNamespace()
        cuda.is_available = lambda: stack.cuda

        def _synchronize():
            stack.syncs += 1
        cuda.synchronize = _synchronize

        tunable = types.SimpleNamespace()
        tunable.enable = lambda flag: stack.tunable_calls.append(("enable", flag))
        tunable.tuning_enable = lambda flag: stack.tunable_calls.append(("tuning_enable", flag))

        def _set_filename(name):
            stack.tunable_calls.append(("set_filename", name))
            if stack.set_filename_raises:
                raise RuntimeError("tunableop filename rejected")
        tunable.set_filename = _set_filename

        def _write_file(name):
            stack.tunable_calls.append(("write_file", name))
            if stack.write_file_raises:
                raise RuntimeError("tunableop CSV not writable")
        tunable.write_file = _write_file
        if self.have_tunable:
            cuda.tunable = tunable

        def _prefer(lib):
            stack.blas.append(lib)
            if lib in stack.blas_unsupported:
                raise RuntimeError("preferred_blas_library(%r) unsupported in this build" % lib)
        torch.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(preferred_blas_library=_prefer))
        torch.cuda = cuda
        torch.finfo = _Finfo
        torch.Generator = _Generator
        torch.randn = _randn
        torch.zeros = _filled(0.0)
        torch.empty = _filled(0.0)
        torch.ones = _filled(1.0)
        torch.load = _load_blob
        torch.tensor = lambda data, dtype=None: _T((len(data),), dtype or FP32, 0.0)

        def _matmul(a, b):
            stack.calls.append(("torch.matmul", a.shape, b.shape))
            return a @ b
        torch.matmul = _matmul

        def _addmm(bias, a, b):
            stack.calls.append(("torch.addmm", bias.shape, a.shape, b.shape))
            return (a @ b) + bias
        torch.addmm = _addmm

        functional = types.ModuleType("torch.nn.functional")

        def _linear(a, w, bias=None):
            stack.calls.append(("F.linear", a.shape, w.shape, None if bias is None else bias.shape,
                                w.dtype.name))
            out = a @ w.t()
            return out if bias is None else out + bias
        functional.linear = _linear
        nn = types.ModuleType("torch.nn")
        nn.functional = functional
        torch.nn = nn
        self.torch, self.nn, self.functional = torch, nn, functional
        self._build_triton()
        self._build_aiter()

    # ---- triton
    def _build_triton(self):
        stack = self
        triton = types.ModuleType("triton")
        tl = types.ModuleType("triton.language")
        tl.constexpr = "constexpr"
        tl.float32 = FP32
        tl._pid = 0
        tl.program_id = lambda axis: tl._pid
        tl.cdiv = lambda a, b: -(-int(a) // int(b))
        tl.minimum = lambda a, b: min(a, b)
        tl.arange = lambda lo, hi: _T((hi - lo,), FP32, 0.0)
        tl.zeros = lambda shape, dtype=None: _T(tuple(shape), dtype or FP32, 0.0)

        def _tl_load(ptr, mask=None, other=0.0):
            return _T(ptr.shape, ptr.dtype, 1.0, ptr.device)
        tl.load = _tl_load

        def _tl_store(ptr, value, mask=None):
            stack.stores.append({"ptr_shape": ptr.shape, "tile_shape": value.shape,
                                 "tile_dtype": value.dtype})
        tl.store = _tl_store
        tl.dot = lambda a, b: _T(a.shape[:-1] + b.shape[-1:], FP32, a.val * b.val * a.shape[-1])

        class _Config:
            def __init__(self, kwargs, num_warps=4, num_stages=2):
                self.kwargs = dict(kwargs)
                self.num_warps = num_warps
                self.num_stages = num_stages

        class _Jit:
            def __init__(self, fn):
                self.fn = fn

        class _Autotuned:
            def __init__(self, kernel, configs, key):
                self.kernel, self.configs, self.key = kernel, configs, key

            def __getitem__(self, grid):
                def launch(*args):
                    meta = dict(self.configs[0].kwargs)
                    dims = grid(meta)
                    stack.launches.append({"grid": dims, "args": args, "meta": meta})
                    n = dims[0] if isinstance(dims, tuple) else dims
                    for pid in range(max(1, min(int(n), 2))):
                        tl._pid = pid
                        self.kernel.fn(*args, **meta)
                return launch

        triton.Config = _Config
        triton.jit = lambda fn: _Jit(fn)
        triton.cdiv = tl.cdiv
        triton.autotune = lambda configs, key: (lambda kernel: _Autotuned(kernel, configs, key))
        triton.language = tl
        self.triton, self.tl = triton, tl

    # ---- aiter
    def _build_aiter(self):
        stack = self
        aiter = types.ModuleType("aiter")

        def _blockscale(x, w, x_scale, w_scale, dtype=None):
            stack.calls.append(("gemm_a8w8_blockscale", x.shape, w.shape, x_scale.shape,
                                w_scale.shape, x.dtype.name, dtype.name))
            if stack.blockscale_raises:
                raise RuntimeError("blockscale kernel not compiled for this arch")
            return _dequant_matmul(x, w, x_scale, w_scale, dtype)

        def _bpreshuffle(x, w, x_scale, w_scale, dtype=None):
            stack.calls.append(("gemm_a8w8_blockscale_bpreshuffle", x.shape, w.shape, x_scale.shape,
                                w_scale.shape, x.dtype.name, dtype.name))
            if stack.bpreshuffle_raises:
                raise RuntimeError("weight must be preshuffled for this entrypoint")
            return _dequant_matmul(x, w, x_scale, w_scale, dtype)

        def _blockscale_v2(x, w, x_scale, w_scale, dtype=None):
            # Same signature and result as _blockscale but a DISTINCT object, so the
            # entrypoint scan has a genuine second candidate to keep -- unlike
            # _blockscale_alias below, which is the same object and must be deduped away.
            stack.calls.append(("gemm_a8w8_blockscale_v2", x.shape, w.shape, x_scale.shape,
                                w_scale.shape, x.dtype.name, dtype.name))
            return _dequant_matmul(x, w, x_scale, w_scale, dtype)

        def _probe(label, weight_is_nk):
            def fn(a, b):
                stack.calls.append((label, a.shape, b.shape))
                if stack.aiter_all_fail:
                    raise RuntimeError("%s: no kernel for this arch" % label)
                want_k = a.shape[-1]
                got_k = b.shape[-1] if weight_is_nk else b.shape[-2]
                if got_k != want_k:
                    raise RuntimeError("%s expects K=%d, got operand %s" % (label, want_k, b.shape))
                return a @ (b.t() if weight_is_nk else b)
            return fn

        def _returns_none(a, b):
            stack.calls.append(("aiter.gemm_a16_returns_none", a.shape, b.shape))
            return None

        aiter.gemm_a8w8_blockscale = _blockscale
        aiter.gemm_a8w8_blockscale_alias = _blockscale        # same object -> must be deduped
        aiter.gemm_a8w8_blockscale_bpreshuffle = _bpreshuffle
        aiter.gemm_a8w8_blockscale_v2 = _blockscale_v2
        aiter.gemm_a16_returns_none = _returns_none           # probed first, must not be accepted
        aiter.gemm_nk = _probe("aiter.gemm_nk", True)
        aiter.matmul_helper = lambda *a: None                 # name does not match the probe filter

        ops = types.ModuleType("aiter.ops")
        ops.linear = _probe("aiter.ops.linear", True)
        ops_triton = types.ModuleType("aiter.ops.triton")
        ops_triton.gemm_a16w16 = _probe("aiter.ops.triton.gemm_a16w16", True)
        flydsl = types.ModuleType("aiter.ops.flydsl")
        utils = types.ModuleType("aiter.ops.flydsl.utils")
        utils.is_flydsl_available = lambda: stack.flydsl_available
        kernels = types.ModuleType("aiter.ops.flydsl.gemm_kernels")

        def _flydsl_hgemm(a, b, bias=None, b_preshuffle=False, auto_shuffle_b=False):
            stack.calls.append(("flydsl_hgemm", a.shape, b.shape,
                                None if bias is None else bias.shape, b_preshuffle, auto_shuffle_b))
            out = a @ b.t()
            return out if bias is None else out + bias
        kernels.flydsl_hgemm = _flydsl_hgemm
        tuned = types.ModuleType("aiter.tuned_gemm")
        tuned.tgemm = _probe("aiter.tuned_gemm.tgemm", True)

        flydsl.utils, flydsl.gemm_kernels = utils, kernels
        ops.triton, ops.flydsl = ops_triton, flydsl
        aiter.ops, aiter.tuned_gemm = ops, tuned
        self.aiter = aiter
        self._aiter_modules = {"aiter": aiter, "aiter.ops": ops, "aiter.ops.triton": ops_triton,
                               "aiter.ops.flydsl": flydsl, "aiter.ops.flydsl.utils": utils,
                               "aiter.ops.flydsl.gemm_kernels": kernels, "aiter.tuned_gemm": tuned}

    # ---- sys.modules management
    def install(self):
        mapping = {"torch": self.torch, "torch.nn": self.nn, "torch.nn.functional": self.functional,
                   "triton": self.triton, "triton.language": self.tl}
        mapping.update(self._aiter_modules)
        sentinel = object()
        for name in self.MODULES:
            self._saved[name] = sys.modules.get(name, sentinel)
            sys.modules[name] = mapping[name]
        self._sentinel = sentinel
        return self

    def uninstall(self):
        for name, prev in self._saved.items():
            if prev is self._sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
        self._saved = {}

    def kernel_names(self):
        return [c[0] for c in self.calls]


class _FakeHlib:
    """harness_lib with a DETERMINISTIC time_op: a fixed device/wall ms plus a record of the timing
    knobs it was handed, so tests can assert the driver passes its own --warmup/--repeats through and
    times under the deployment graph mode. Everything else delegates to the real library."""

    def __init__(self, real, ms=1.5, wall_ms=2.25):
        self._real = real
        self._ms = ms
        self._wall = wall_ms
        self.calls = []

    def time_op(self, call, warmup=10, repeats=50, graph=False, detail=False, **kw):
        self.calls.append({"warmup": warmup, "repeats": repeats, "graph": graph, "detail": detail})
        call()                       # a timed closure must be callable; exercise it once
        if self._ms is None:
            return None
        return {"ms": self._ms, "wall_ms": self._wall, "timer": "fake"}

    def __getattr__(self, name):
        return getattr(self._real, name)


ob = _load("op_bench", "op_bench.py")
ob_nolib = _load_without_harness_lib("op_bench_no_harness_lib", "op_bench.py")


# --------------------------------------------------------------------------- #
# mixins
# --------------------------------------------------------------------------- #
class _ObStateMixin:
    """Restores every piece of op_bench module state a test may touch."""

    def setUp(self):
        self._restore = []
        for mod in (ob, ob_nolib):
            for attr in ("_GRAPH_MODE", "_TRITON_MM", "_hlib", "bench_gemm", "bench_attn"):
                self._restore.append((mod, attr, getattr(mod, attr)))
        ob._GRAPH_MODE = ob_nolib._GRAPH_MODE = False
        ob._TRITON_MM = ob_nolib._TRITON_MM = None

    def tearDown(self):
        for mod, attr, val in self._restore:
            setattr(mod, attr, val)

    def _task_dir(self, meta=None):
        d = tempfile.mkdtemp(prefix="op_task_")
        self.addCleanup(shutil.rmtree, d, True)
        if meta is not None:
            with open(os.path.join(d, "meta.json"), "w") as fh:
                json.dump(meta, fh)
        return d

    def _args(self, task="", **kw):
        ns = types.SimpleNamespace(task=task, backends="", repeats=3, warmup=2, tol=2e-2, seed=7,
                                   out="", triton_autotune=False)
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns


class _FakeStackMixin(_ObStateMixin):
    """Installs the fake torch/triton/aiter trio for one test, plus the deterministic timer."""

    CUDA = False

    def setUp(self):
        super().setUp()
        self.stack = _Stack(cuda=self.CUDA).install()
        self.addCleanup(self.stack.uninstall)
        self.hlib = _FakeHlib(ob._hlib)
        ob._hlib = self.hlib

    def _by_backend(self, results):
        return {r["backend"]: r for r in results}


# --------------------------------------------------------------------------- #
# _sync / _time_call -- the measurement plumbing
# --------------------------------------------------------------------------- #
class TestSyncAndTiming(_FakeStackMixin, unittest.TestCase):
    def test_sync_is_a_noop_without_cuda(self):
        ob._sync(self.stack.torch)
        self.assertEqual(self.stack.syncs, 0)

    def test_sync_synchronizes_when_cuda_is_available(self):
        self.stack.cuda = True
        ob._sync(self.stack.torch)
        self.assertEqual(self.stack.syncs, 1)

    def test_time_call_forwards_knobs_and_graph_mode_to_harness_lib(self):
        # Both baseline and candidate must be timed under the SAME deployment graph context; the
        # module-level _GRAPH_MODE is what carries it into every measurement.
        ob._GRAPH_MODE = True
        ms, wall = ob._time_call(lambda: None, 4, 9)
        self.assertEqual((ms, wall), (1.5, 2.25))
        self.assertEqual(self.hlib.calls, [{"warmup": 4, "repeats": 9, "graph": True, "detail": True}])

    def test_time_call_returns_none_pair_when_harness_lib_reports_failure(self):
        ob._hlib = _FakeHlib(ob._hlib, ms=None)
        self.assertEqual(ob._time_call(lambda: None, 1, 1), (None, None))

    def test_time_call_real_harness_lib_produces_two_floats(self):
        ob._hlib = self.hlib._real            # the actual harness_lib.time_op, on the fake torch
        calls = []
        ms, wall = ob._time_call(lambda: calls.append(1), 2, 3)
        self.assertEqual(len(calls), 5)       # warmup=2 + repeats=3, wall-only on a CUDA-less box
        self.assertIsInstance(ms, float)
        self.assertIsInstance(wall, float)

    def test_fallback_loop_runs_warmup_then_repeats_when_harness_lib_is_absent(self):
        self.assertIsNone(ob_nolib._hlib)
        calls = []
        ms, wall = ob_nolib._time_call(lambda: calls.append(1), 2, 5)
        self.assertEqual(len(calls), 7)
        self.assertEqual(ms, wall)            # no device timeline -> event ms IS the wall median
        self.assertGreaterEqual(ms, 0.0)

    def test_fallback_loop_clamps_nonpositive_warmup_and_repeats(self):
        calls = []
        ob_nolib._time_call(lambda: calls.append(1), 0, 0)
        self.assertEqual(len(calls), 2)       # max(1, ...) on both loops

    def test_fallback_loop_reports_none_when_the_closure_raises(self):
        def boom():
            raise RuntimeError("kernel launch failed")
        self.assertEqual(ob_nolib._time_call(boom, 1, 1), (None, None))


# --------------------------------------------------------------------------- #
# _correct -- the scale-relative correctness gate
# --------------------------------------------------------------------------- #
class TestCorrect(_FakeStackMixin, unittest.TestCase):
    def test_identical_output_is_correct_with_zero_error(self):
        ref = _T((8, 32), BF16, 0.5)
        ok, err = ob._correct(self.stack.torch, ref._like(), ref, 2e-2)
        self.assertTrue(ok)
        self.assertEqual(err, 0.0)

    def test_error_is_bounded_by_the_scale_relative_atol(self):
        # err = |out-ref| / (|ref| + tol*max|ref|): a 2x-off output is ~1/(1+tol), never unbounded.
        ref = _T((4, 4), BF16, 1.0)
        ok, err = ob._correct(self.stack.torch, ref._like(val=2.0), ref, 2e-2)
        self.assertFalse(ok)
        self.assertAlmostEqual(err, 1.0 / 1.02, places=6)

    def test_near_zero_reference_does_not_blow_up_the_metric(self):
        ref = _T((4, 4), BF16, 0.0)
        ok, err = ob._correct(self.stack.torch, ref._like(), ref, 2e-2)
        self.assertTrue(ok)
        self.assertTrue(math.isfinite(err))

    def test_within_tolerance_output_passes(self):
        ref = _T((4, 4), BF16, 1.0)
        ok, _ = ob._correct(self.stack.torch, ref._like(val=1.001), ref, 2e-2)
        self.assertTrue(ok)

    def test_shape_mismatch_is_incorrect_with_infinite_error(self):
        ok, err = ob._correct(self.stack.torch, _T((8, 16), BF16, 1.0), _T((8, 32), BF16, 1.0), 2e-2)
        self.assertFalse(ok)
        self.assertEqual(err, float("inf"))

    def test_non_tensor_output_is_incorrect_not_an_exception(self):
        ok, err = ob._correct(self.stack.torch, None, _T((8, 32), BF16, 1.0), 2e-2)
        self.assertFalse(ok)
        self.assertEqual(err, float("inf"))


# --------------------------------------------------------------------------- #
# _dtype
# --------------------------------------------------------------------------- #
class TestDtype(_FakeStackMixin, unittest.TestCase):
    def test_named_dtypes_resolve_through_harness_lib(self):
        t = self.stack.torch
        self.assertIs(ob._dtype(t, "bf16"), BF16)
        self.assertIs(ob._dtype(t, "float16"), FP16)
        self.assertIs(ob._dtype(t, "fp32"), FP32)

    def test_bare_fp8_is_arch_driven_not_hardcoded_fnuz(self):
        # No visible GPU -> not a CDNA3 fnuz arch -> the OCP variant. This is the MI300/MI355 fork the
        # shared resolver exists for; op_bench must NOT resolve fp8 from its own table when it has one.
        self.assertIs(ob._dtype(self.stack.torch, "fp8"), FP8_E4M3FN)

    def test_explicit_fnuz_name_is_honoured_literally(self):
        self.assertIs(ob._dtype(self.stack.torch, "fp8_e4m3fnuz"), FP8_E4M3FNUZ)

    def test_local_table_is_used_when_harness_lib_is_absent(self):
        t = self.stack.torch
        self.assertIs(ob_nolib._dtype(t, "bfloat16"), BF16)
        self.assertIs(ob_nolib._dtype(t, "fp16"), FP16)
        self.assertIs(ob_nolib._dtype(t, "float32"), FP32)
        self.assertIs(ob_nolib._dtype(t, "fp8"), FP8_E4M3FNUZ)      # fnuz default, pre-resolver
        self.assertIs(ob_nolib._dtype(t, "fp8_e5m2"), FP8_E5M2)

    def test_local_table_falls_back_to_bf16_for_unknown_names(self):
        self.assertIs(ob_nolib._dtype(self.stack.torch, "float4_exotic"), BF16)


# --------------------------------------------------------------------------- #
# _resolve_shape -- the symbolic-dim guard
# --------------------------------------------------------------------------- #
class TestResolveShape(unittest.TestCase):
    def test_concrete_int_shape_passes_through(self):
        self.assertEqual(ob._resolve_shape([8, 4096], {}), [8, 4096])

    def test_numeric_strings_are_coerced(self):
        self.assertEqual(ob._resolve_shape(["8", "4096"], {}), [8, 4096])

    def test_symbolic_m_resolves_to_the_dominant_bucket(self):
        # The representative M is the LARGEST profiled bucket -- that is where the GPU-time mass is.
        self.assertEqual(ob._resolve_shape(["M", 4096], {"m_buckets": [1, 64, 2048]}), [2048, 4096])

    def test_symbolic_aliases_all_resolve(self):
        meta = {"m_buckets": ["1", 512]}
        for dim in ("M", "m", "m_tokens", "-1", "none", ""):
            self.assertEqual(ob._resolve_shape([dim, 16], meta), [512, 16])

    def test_empty_shape_is_rejected(self):
        for bad in ([], None, ()):
            with self.assertRaises(ValueError):
                ob._resolve_shape(bad, {})

    def test_bool_dim_is_rejected_before_it_reaches_randn(self):
        # bool is an int subclass; letting it through would silently synthesize a 0/1-row operand.
        with self.assertRaises(ValueError) as cm:
            ob._resolve_shape([True, 16], {})
        self.assertIn("bool dim", str(cm.exception))

    def test_symbolic_dim_without_buckets_names_the_missing_metadata(self):
        with self.assertRaises(ValueError) as cm:
            ob._resolve_shape(["M", 4096], {})
        self.assertIn("m_buckets", str(cm.exception))

    def test_unresolvable_symbol_is_rejected_with_guidance(self):
        with self.assertRaises(ValueError) as cm:
            ob._resolve_shape(["hidden", 4096], {"m_buckets": [8]})
        self.assertIn("unresolvable symbolic dim", str(cm.exception))

    def test_non_numeric_buckets_are_ignored(self):
        with self.assertRaises(ValueError):
            ob._resolve_shape(["M", 16], {"m_buckets": ["auto", None]})


# --------------------------------------------------------------------------- #
# _resolve_callable
# --------------------------------------------------------------------------- #
class TestResolveCallable(_FakeStackMixin, unittest.TestCase):
    def test_module_colon_attr_resolves(self):
        self.assertIs(ob._resolve_callable("aiter:gemm_a8w8_blockscale"),
                      self.stack.aiter.gemm_a8w8_blockscale)

    def test_dotted_submodule_resolves(self):
        self.assertIs(ob._resolve_callable("aiter.ops.flydsl.gemm_kernels:flydsl_hgemm"),
                      self.stack.aiter.ops.flydsl.gemm_kernels.flydsl_hgemm)

    def test_specs_without_a_colon_are_none(self):
        for spec in ("", None, "aiter.gemm"):
            self.assertIsNone(ob._resolve_callable(spec))

    def test_unimportable_module_is_none(self):
        self.assertIsNone(ob._resolve_callable("no_such_backend_module:gemm"))

    def test_missing_attribute_is_none(self):
        self.assertIsNone(ob._resolve_callable("aiter:gemm_that_does_not_exist"))


# --------------------------------------------------------------------------- #
# op classification -- which heads must NOT enter the dense bake-off
# --------------------------------------------------------------------------- #
class TestOpClassification(unittest.TestCase):
    def test_blockscale_needs_both_fp8_and_a_block_size(self):
        self.assertTrue(ob._is_blockscale_gemm({"dtype": "fp8", "weight_block_size": [128, 128]}))
        self.assertTrue(ob._is_blockscale_gemm({"dtype": "fp8_e4m3fnuz", "quant_scheme": "blockwise"}))
        self.assertTrue(ob._is_blockscale_gemm({"dtype": "e5m2", "weight_block_size": [128, 128]}))
        self.assertFalse(ob._is_blockscale_gemm({"dtype": "fp8"}))
        self.assertFalse(ob._is_blockscale_gemm({"dtype": "bf16", "weight_block_size": [128, 128]}))
        self.assertFalse(ob._is_blockscale_gemm({}))

    def test_grouped_is_detected_from_kernel_class(self):
        for kc in ("fused_moe", "grouped_gemm", "experts_mlp"):
            self.assertTrue(ob._is_grouped_or_quant_gemm({"kernel_class": kc}))

    def test_grouped_is_detected_from_packed_weight_dtype(self):
        for dt in ("int4", "int8", "uint4", "awq", "gptq", "w4a16", "w8a16"):
            self.assertTrue(ob._is_grouped_or_quant_gemm({"dtype": dt}))

    def test_grouped_is_detected_from_quant_scheme(self):
        for qs in ("awq", "gptq", "int4", "compressed_tensors"):
            self.assertTrue(ob._is_grouped_or_quant_gemm({"quant_scheme": qs}))

    def test_three_dimensional_weight_is_grouped(self):
        # [E, N, K] expert weights would make the dense path call .t() on a 3D tensor.
        self.assertTrue(ob._is_grouped_or_quant_gemm({"b_shape": [8, 2048, 4096]}))

    def test_structured_moe_shape_block_is_grouped(self):
        self.assertTrue(ob._is_grouped_or_quant_gemm({"shape": {"E": 8, "N": 2048, "K": 4096}}))

    def test_dense_bf16_gemm_is_not_grouped(self):
        self.assertFalse(ob._is_grouped_or_quant_gemm(
            {"dtype": "bf16", "kernel_class": "linear", "b_shape": [2048, 4096],
             "shape": {"N": 2048}}))


# --------------------------------------------------------------------------- #
# _synth_blockscale_case -- fp8 a8w8 operand + oracle construction
# --------------------------------------------------------------------------- #
class TestSynthBlockscaleCase(_FakeStackMixin, unittest.TestCase):
    META = {"dtype": "fp8", "out_dtype": "bf16", "b_shape": [256, 512],
            "weight_block_size": [128, 128]}

    def test_shapes_and_dtypes_match_the_extracted_unittest_contract(self):
        case = ob._synth_blockscale_case(self.stack.torch, self.META, 64, "cpu", 7)
        self.assertEqual(case["x"].shape, (64, 512))          # [M, K] fp8 activations
        self.assertEqual(case["w"].shape, (256, 512))         # [N, K] fp8 weights
        self.assertEqual(case["x_scale"].shape, (64, 4))      # [M, K/BLK_K] per-token-block
        self.assertEqual(case["w_scale"].shape, (2, 4))       # [N/BLK_N, K/BLK_K] per-weight-block
        self.assertEqual(case["ref"].shape, (64, 256))
        self.assertIs(case["x"].dtype, FP8_E4M3FN)
        self.assertIs(case["w"].dtype, FP8_E4M3FN)
        self.assertIs(case["x_scale"].dtype, FP32)
        self.assertIs(case["w_scale"].dtype, FP32)
        self.assertIs(case["ref"].dtype, BF16)
        self.assertIs(case["out_dt"], BF16)
        self.assertEqual(case["M"], 64)

    def test_quantized_operands_are_clamped_to_the_fp8_max(self):
        case = ob._synth_blockscale_case(self.stack.torch, self.META, 8, "cpu", 0)
        self.assertLessEqual(abs(case["x"].val), FP8_E4M3FN.fmax)
        self.assertLessEqual(abs(case["w"].val), FP8_E4M3FN.fmax)

    def test_generator_is_seeded_on_the_target_device(self):
        ob._synth_blockscale_case(self.stack.torch, self.META, 8, "cuda", 1234)
        self.assertIn(("manual_seed", "cuda", 1234), self.stack.calls)

    def test_ragged_k_pads_up_to_a_whole_number_of_blocks(self):
        meta = dict(self.META, b_shape=[130, 300])            # neither dim is a multiple of 128
        case = ob._synth_blockscale_case(self.stack.torch, meta, 8, "cpu", 0)
        self.assertEqual(case["x"].shape, (8, 300))           # sliced back to the real K
        self.assertEqual(case["w"].shape, (130, 300))
        self.assertEqual(case["x_scale"].shape, (8, 3))       # ceil(300/128)
        self.assertEqual(case["w_scale"].shape, (2, 3))       # ceil(130/128)

    def test_explicit_block_size_is_honoured(self):
        meta = dict(self.META, weight_block_size=[64, 128])
        case = ob._synth_blockscale_case(self.stack.torch, meta, 8, "cpu", 0)
        self.assertEqual(case["w_scale"].shape, (4, 4))       # 256/64 x 512/128


# --------------------------------------------------------------------------- #
# bench_blockscale_gemm
# --------------------------------------------------------------------------- #
class TestBenchBlockscaleGemm(_FakeStackMixin, unittest.TestCase):
    META = {"op_kind": "gemm", "dtype": "fp8", "out_dtype": "bf16", "b_shape": [256, 512],
            "weight_block_size": [128, 128], "m_buckets": [1, 64, 512],
            "baseline_callable": "aiter:gemm_a8w8_blockscale"}

    def test_baseline_and_bpreshuffle_are_benched_at_the_dominant_bucket(self):
        res = ob.bench_blockscale_gemm(self._args(), self.META)
        by = self._by_backend(res)
        self.assertEqual(set(by), {"aiter_blockscale", "aiter_bpreshuffle"})
        for name in by:
            self.assertTrue(by[name]["available"], name)
            self.assertTrue(by[name]["correct"], name)
            self.assertEqual(by[name]["ms"], 1.5)
            self.assertEqual(by[name]["wall_ms"], 2.25)
            self.assertEqual(by[name]["max_rel_err"], 0.0)
            self.assertFalse(by[name]["raised"])
        self.assertIn("M=512 (dominant bucket)", by["aiter_blockscale"]["note"])

    def test_operands_and_out_dtype_reach_the_backend_in_the_documented_order(self):
        ob.bench_blockscale_gemm(self._args(), self.META)
        calls = [c for c in self.stack.calls if c[0] == "gemm_a8w8_blockscale"]
        # warmup + correctness launch + one timed sample
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0], ("gemm_a8w8_blockscale", (512, 512), (256, 512), (512, 4), (2, 4),
                                    "float8_e4m3fn", "bfloat16"))

    def test_an_alias_of_the_same_callable_is_not_benched_twice(self):
        meta = dict(self.META, target_callable="aiter:gemm_a8w8_blockscale_alias")
        res = ob.bench_blockscale_gemm(self._args(), meta)
        self.assertEqual([r["backend"] for r in res], ["aiter_blockscale", "aiter_bpreshuffle"])

    def test_a_distinct_target_callable_is_benched_as_its_own_candidate(self):
        # Live baseline vs the task's target entrypoint: two genuinely different callables, so the
        # bake-off races both, plus the bpreshuffle lever -> three candidates.
        meta = dict(self.META, target_callable="aiter:gemm_a8w8_blockscale_v2")
        res = ob.bench_blockscale_gemm(self._args(), meta)
        self.assertEqual([r["backend"] for r in res],
                         ["aiter_blockscale", "aiter_blockscale_target", "aiter_bpreshuffle"])
        self.assertTrue(all(r["correct"] for r in res))
        self.assertIn(("gemm_a8w8_blockscale_v2", (512, 512), (256, 512), (512, 4), (2, 4),
                       "float8_e4m3fn", "bfloat16"), self.stack.calls)

    def test_a_target_that_is_already_the_bpreshuffle_lever_is_benched_once(self):
        # The bpreshuffle leg is appended unconditionally, so a task whose target_callable already IS
        # that entrypoint must not have it raced twice under two different backend names.
        meta = dict(self.META, target_callable="aiter:gemm_a8w8_blockscale_bpreshuffle")
        res = ob.bench_blockscale_gemm(self._args(), meta)
        self.assertEqual([r["backend"] for r in res], ["aiter_blockscale", "aiter_blockscale_target"])

    def test_target_callable_alone_supplies_the_baseline(self):
        meta = {k: v for k, v in self.META.items() if k != "baseline_callable"}
        meta["target_callable"] = "aiter:gemm_a8w8_blockscale"
        res = ob.bench_blockscale_gemm(self._args(), meta)
        self.assertEqual([r["backend"] for r in res], ["aiter_blockscale", "aiter_bpreshuffle"])

    def test_unimportable_callable_is_a_recorded_skip_not_a_crash(self):
        meta = dict(self.META, baseline_callable="no_such_module:gemm_a8w8_blockscale")
        res = self._by_backend(ob.bench_blockscale_gemm(self._args(), meta))
        entry = res["aiter_blockscale"]
        self.assertFalse(entry["available"])
        self.assertFalse(entry["raised"])
        self.assertIn("callable not importable: no_such_module:gemm_a8w8_blockscale", entry["note"])

    def test_a_raising_candidate_is_flagged_as_raised_for_the_self_fault_signal(self):
        self.stack.bpreshuffle_raises = True
        entry = self._by_backend(ob.bench_blockscale_gemm(self._args(), self.META))["aiter_bpreshuffle"]
        self.assertTrue(entry["available"])
        self.assertTrue(entry["raised"])
        self.assertIsNone(entry["ms"])
        self.assertIn("call raised:", entry["note"])
        self.assertIn("preshuffled", entry["note"])

    def test_untimeable_candidate_records_a_null_ms(self):
        ob._hlib = _FakeHlib(self.hlib._real, ms=None)
        entry = self._by_backend(ob.bench_blockscale_gemm(self._args(), self.META))["aiter_blockscale"]
        self.assertTrue(entry["correct"])
        self.assertIsNone(entry["ms"])
        self.assertIsNone(entry["wall_ms"])

    def test_m_falls_back_to_a_shape_when_no_buckets_are_profiled(self):
        meta = {k: v for k, v in self.META.items() if k != "m_buckets"}
        meta["a_shape"] = [128, 512]
        res = ob.bench_blockscale_gemm(self._args(), meta)
        self.assertIn("M=128", res[0]["note"])

    def test_symbolic_a_shape_without_buckets_raises_a_named_error(self):
        # Surfaces as main()'s top-level ERROR result rather than a cryptic randn(str, int) TypeError.
        meta = {k: v for k, v in self.META.items() if k != "m_buckets"}
        meta["a_shape"] = ["M", 512]
        with self.assertRaises(ValueError) as cm:
            ob.bench_blockscale_gemm(self._args(), meta)
        self.assertIn("m_buckets", str(cm.exception))

    def test_device_follows_cuda_availability(self):
        self.stack.cuda = True
        ob.bench_blockscale_gemm(self._args(), self.META)
        self.assertIn(("manual_seed", "cuda", 7), self.stack.calls)


# --------------------------------------------------------------------------- #
# _load_or_synth_gemm
# --------------------------------------------------------------------------- #
class TestLoadOrSynthGemm(_FakeStackMixin, unittest.TestCase):
    def _task_with_io(self, blob):
        d = self._task_dir()
        open(os.path.join(d, "reference_io.pt"), "w").close()
        self.stack.loaded_blob = blob
        return d

    def test_recorded_oracle_is_preferred_over_synthesis(self):
        blob = {"A": _T((8, 16), FP32, 0.25), "B": _T((32, 16), FP32, 0.5),
                "bias": None, "output": _T((8, 32), FP32, 4.0)}
        d = self._task_with_io(blob)
        A, B, bias, tb, ref = ob._load_or_synth_gemm(self.stack.torch, d, {"dtype": "bf16"}, "cpu", 0)
        self.assertIs(A.dtype, BF16)
        self.assertIs(B.dtype, BF16)
        self.assertIsNone(bias)
        self.assertTrue(tb)
        self.assertIs(ref.dtype, FP32)        # the oracle is compared in fp32
        self.assertEqual(ref.val, 4.0)
        self.assertEqual([c for c in self.stack.kernel_names() if c == "torch.load"], ["torch.load"])

    def test_oracle_without_a_recorded_output_is_recomputed_with_bias(self):
        blob = {"A": _T((8, 16), FP32, 0.5), "B": _T((32, 16), FP32, 2.0),
                "bias": _T((32,), FP32, 1.0), "output": None}
        d = self._task_with_io(blob)
        A, B, bias, tb, ref = ob._load_or_synth_gemm(self.stack.torch, d, {}, "cpu", 0)
        self.assertEqual(ref.shape, (8, 32))
        self.assertEqual(ref.val, 0.5 * 2.0 * 16 + 1.0)
        self.assertIs(bias.dtype, BF16)

    def test_recomputed_oracle_honours_transpose_b_false(self):
        blob = {"A": _T((8, 16), FP32, 1.0), "B": _T((16, 32), FP32, 1.0)}
        d = self._task_with_io(blob)
        _, _, _, tb, ref = ob._load_or_synth_gemm(self.stack.torch, d, {"transpose_b": False},
                                                  "cpu", 0)
        self.assertFalse(tb)
        self.assertEqual(ref.shape, (8, 32))

    def test_unrecognised_blob_falls_back_to_synthesis(self):
        d = self._task_with_io(["not", "a", "dict"])
        meta = {"a_shape": [4, 16], "b_shape": [32, 16]}
        A, B, bias, tb, ref = ob._load_or_synth_gemm(self.stack.torch, d, meta, "cpu", 3)
        self.assertEqual((A.shape, B.shape, ref.shape), ((4, 16), (32, 16), (4, 32)))

    def test_synthesis_resolves_symbolic_dims_and_seeds_the_generator(self):
        meta = {"a_shape": ["M", 16], "b_shape": [32, 16], "m_buckets": [1, 64]}
        A, B, bias, tb, ref = ob._load_or_synth_gemm(self.stack.torch, self._task_dir(), meta,
                                                     "cuda", 11)
        self.assertEqual(A.shape, (64, 16))
        self.assertEqual(A.device, "cuda")
        self.assertIsNone(bias)
        self.assertIn(("manual_seed", "cpu", 11), self.stack.calls)   # host generator, then .to(device)

    def test_bias_width_follows_the_weight_layout(self):
        meta = {"a_shape": [8, 16], "b_shape": [32, 16], "bias": True}
        _, _, bias, _, ref = ob._load_or_synth_gemm(self.stack.torch, self._task_dir(), meta, "cpu", 0)
        self.assertEqual(bias.shape, (32,))                  # N from b_shape[0] under F.linear layout
        self.assertEqual(ref.val, 0.1 * 0.1 * 16 + 0.1)

        meta_nt = {"a_shape": [8, 16], "b_shape": [16, 48], "bias": True, "transpose_b": False}
        _, _, bias_nt, _, _ = ob._load_or_synth_gemm(self.stack.torch, self._task_dir(), meta_nt,
                                                     "cpu", 0)
        self.assertEqual(bias_nt.shape, (48,))               # N from b_shape[-1] for a plain matmul

    def test_missing_oracle_and_missing_shapes_is_a_named_error(self):
        with self.assertRaises(ValueError) as cm:
            ob._load_or_synth_gemm(self.stack.torch, self._task_dir(), {"dtype": "bf16"}, "cpu", 0)
        self.assertIn("neither reference_io.pt nor a_shape/b_shape", str(cm.exception))


# --------------------------------------------------------------------------- #
# _gemm_fn / BLAS + TunableOp switches
# --------------------------------------------------------------------------- #
class TestGemmClosureAndSwitches(_FakeStackMixin, unittest.TestCase):
    def test_transpose_b_uses_f_linear_with_the_nk_weight(self):
        A, B, bias = _T((8, 16), BF16, 0.5), _T((32, 16), BF16, 2.0), _T((32,), BF16, 1.0)
        out = ob._gemm_fn(self.stack.torch, A, B, bias, True)()
        self.assertEqual(self.stack.calls[-1],
                         ("F.linear", (8, 16), (32, 16), (32,), "bfloat16"))
        self.assertEqual(out.shape, (8, 32))
        self.assertEqual(out.val, 0.5 * 2.0 * 16 + 1.0)

    def test_plain_matmul_when_not_transposed_and_unbiased(self):
        out = ob._gemm_fn(self.stack.torch, _T((8, 16), BF16, 1.0), _T((16, 32), BF16, 1.0),
                          None, False)()
        self.assertEqual(self.stack.calls[-1][0], "torch.matmul")
        self.assertEqual(out.shape, (8, 32))

    def test_addmm_fuses_the_bias_for_a_2d_operand(self):
        ob._gemm_fn(self.stack.torch, _T((8, 16), BF16, 1.0), _T((16, 32), BF16, 1.0),
                    _T((32,), BF16, 1.0), False)()
        self.assertEqual(self.stack.calls[-1][0], "torch.addmm")

    def test_batched_operand_adds_the_bias_after_matmul(self):
        out = ob._gemm_fn(self.stack.torch, _T((2, 8, 16), BF16, 1.0), _T((16, 32), BF16, 1.0),
                          _T((32,), BF16, 1.0), False)()
        self.assertEqual(self.stack.calls[-1][0], "torch.matmul")
        self.assertEqual(out.shape, (2, 8, 32))

    def test_prefer_blas_reports_whether_the_switch_applied(self):
        self.assertTrue(ob._set_prefer_blas(self.stack.torch, "hipblaslt"))
        self.assertEqual(self.stack.blas, ["hipblaslt"])
        self.assertFalse(ob._set_prefer_blas(self.stack.torch, "ck"))   # unsupported in this build

    def test_prefer_blas_is_false_when_the_api_is_missing(self):
        del self.stack.torch.backends.cuda.preferred_blas_library
        self.assertFalse(ob._set_prefer_blas(self.stack.torch, "hipblaslt"))

    def test_tunableop_enables_tuning_and_sets_the_csv(self):
        self.assertTrue(ob._tunableop(self.stack.torch, True, True, "/tmp/t.csv"))
        self.assertEqual(self.stack.tunable_calls,
                         [("enable", True), ("tuning_enable", True), ("set_filename", "/tmp/t.csv")])

    def test_tunableop_without_a_filename_skips_set_filename(self):
        self.assertTrue(ob._tunableop(self.stack.torch, False, False))
        self.assertEqual([c[0] for c in self.stack.tunable_calls], ["enable", "tuning_enable"])

    def test_tunableop_survives_a_rejected_filename(self):
        self.stack.set_filename_raises = True
        self.assertTrue(ob._tunableop(self.stack.torch, True, True, "/nope/t.csv"))

    def test_tunableop_is_false_when_the_api_is_absent(self):
        del self.stack.torch.cuda.tunable
        self.assertFalse(ob._tunableop(self.stack.torch, True, True))


# --------------------------------------------------------------------------- #
# bench_gemm -- the dispatch ladder
# --------------------------------------------------------------------------- #
class TestBenchGemmDispatch(_FakeStackMixin, unittest.TestCase):
    META = {"op_kind": "gemm", "dtype": "bf16", "a_shape": [8, 16], "b_shape": [32, 16]}

    def _run(self, backends="", meta=None, **kw):
        args = self._args(task=self._task_dir(), backends=backends, **kw)
        return args, ob.bench_gemm(args, meta or self.META)

    def test_blockscale_head_is_delegated_to_the_dedicated_path(self):
        meta = {"op_kind": "gemm", "dtype": "fp8", "out_dtype": "bf16", "b_shape": [256, 512],
                "weight_block_size": [128, 128], "m_buckets": [64],
                "baseline_callable": "aiter:gemm_a8w8_blockscale"}
        _, res = self._run(meta=meta)
        self.assertEqual([r["backend"] for r in res], ["aiter_blockscale", "aiter_bpreshuffle"])
        self.assertNotIn("F.linear", self.stack.kernel_names())

    def test_grouped_moe_gemm_is_a_clean_skip_not_a_harness_fault(self):
        meta = {"op_kind": "gemm", "kernel_class": "fused_moe", "dtype": "int4_w4a16"}
        _, res = self._run(meta=meta)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["backend"], "grouped_quant_gemm")
        self.assertFalse(res[0]["available"])
        self.assertFalse(res[0]["raised"])           # must NOT trip the harness-self-fault signal
        self.assertIn("requires a Tier-C authored fused-experts grouped GEMM", res[0]["note"])
        self.assertIn("fused_moe", res[0]["note"])
        self.assertIn("int4_w4a16", res[0]["note"])

    def test_default_backend_set_excludes_the_retired_triton_stub(self):
        _, res = self._run()
        self.assertEqual([r["backend"] for r in res],
                         ["hipblaslt", "tunableop", "rocblas", "ck", "aiter", "flydsl"])

    def test_hipblaslt_is_the_timed_default_path(self):
        args, res = self._run(backends="hipblaslt")
        entry = res[0]
        self.assertEqual(entry["backend"], "hipblaslt")
        self.assertTrue(entry["available"])
        self.assertTrue(entry["correct"])
        self.assertEqual(entry["ms"], 1.5)
        self.assertEqual(entry["max_rel_err"], 0.0)
        self.assertEqual(entry["note"], "torch default Lt path")
        self.assertEqual(self.stack.blas, ["hipblaslt"])
        # warmup launch + correctness launch + the timed sample, all through F.linear
        self.assertEqual(self.stack.kernel_names().count("F.linear"), 3)
        self.assertEqual(self.hlib.calls[0]["warmup"], args.warmup)
        self.assertEqual(self.hlib.calls[0]["repeats"], args.repeats)

    def test_tunableop_persists_and_freezes_a_csv_artifact(self):
        args, res = self._run(backends="tunableop")
        entry = res[0]
        self.assertTrue(entry["available"])
        csv = os.path.join(args.task, "tunableop.csv")
        self.assertEqual(entry["artifact"], csv)
        self.assertIn("CSV deployable at startup", entry["note"])
        seq = [c for c in self.stack.tunable_calls if c[0] in ("tuning_enable", "write_file")]
        # tune -> persist -> freeze (tuning off) -> disable entirely
        self.assertEqual(seq, [("tuning_enable", True), ("write_file", csv),
                               ("tuning_enable", False), ("tuning_enable", False)])

    def test_tunableop_survives_an_unwritable_csv(self):
        self.stack.write_file_raises = True
        _, res = self._run(backends="tunableop")
        self.assertTrue(res[0]["available"])

    def test_tunableop_absent_api_is_recorded_unavailable(self):
        del self.stack.torch.cuda.tunable
        _, res = self._run(backends="tunableop")
        self.assertEqual(res[0]["backend"], "tunableop")
        self.assertFalse(res[0]["available"])
        self.assertEqual(res[0]["note"], "torch.cuda.tunable API unavailable")

    def test_rocblas_switches_to_the_non_lt_path_then_restores_hipblaslt(self):
        _, res = self._run(backends="rocblas")
        self.assertEqual(res[0]["note"], "torch non-Lt path")
        self.assertEqual(self.stack.blas, ["cublas", "hipblaslt"])

    def test_rocblas_notes_an_unconfirmed_switch(self):
        self.stack.blas_unsupported = ("cublas",)
        _, res = self._run(backends="rocblas")
        self.assertIn("(switch unconfirmed)", res[0]["note"])

    def test_ck_is_recorded_unsupported_when_the_build_lacks_it(self):
        _, res = self._run(backends="ck")
        self.assertFalse(res[0]["available"])
        self.assertIn("preferred_blas_library('ck') unsupported", res[0]["note"])
        self.assertEqual(self.stack.blas, ["ck", "hipblaslt"])       # always restore the default

    def test_ck_is_benched_when_the_build_exposes_it(self):
        self.stack.blas_unsupported = ()
        _, res = self._run(backends="ck")
        self.assertTrue(res[0]["available"])
        self.assertEqual(res[0]["note"], "torch preferred_blas=ck")

    def test_hipblaslt_tuned_is_always_reported_as_not_a_pytorch_lever(self):
        _, res = self._run(backends="hipblaslt_tuned")
        self.assertEqual(res[0]["backend"], "hipblaslt_tuned")
        self.assertFalse(res[0]["available"])
        self.assertIn("HIPBLASLT_TUNING_OVERRIDE_FILE is consume-only", res[0]["note"])
        self.assertEqual(self.stack.blas, [])                       # nothing was even attempted

    def test_aiter_probe_result_is_timed_like_any_other_backend(self):
        _, res = self._run(backends="aiter")
        entry = res[0]
        self.assertTrue(entry["available"])
        self.assertTrue(entry["correct"])
        self.assertEqual(entry["note"], "aiter fused gemm (auto-probed)")
        self.assertIn(("aiter.gemm_nk", (8, 16), (32, 16)), self.stack.calls)

    def test_aiter_failure_is_recorded_as_a_failed_call(self):
        self.stack.aiter_all_fail = True
        _, res = self._run(backends="aiter")
        self.assertFalse(res[0]["available"])
        self.assertIn("call failed:", res[0]["note"])
        self.assertIn("no working aiter gemm entrypoint", res[0]["note"])

    def test_flydsl_is_gated_by_is_flydsl_available(self):
        self.stack.flydsl_available = False
        _, res = self._run(backends="flydsl")
        self.assertFalse(res[0]["available"])
        self.assertIn("is_flydsl_available()==False", res[0]["note"])
        self.assertEqual(res[0]["artifact"], "")
        self.assertNotIn("flydsl_hgemm", self.stack.kernel_names())

    def test_flydsl_is_a_first_class_candidate_when_installed(self):
        _, res = self._run(backends="flydsl")
        entry = res[0]
        self.assertTrue(entry["available"])
        self.assertTrue(entry["correct"])
        self.assertIn("flydsl_hgemm", entry["note"])
        self.assertIn(("flydsl_hgemm", (8, 16), (32, 16), None, False, False), self.stack.calls)

    def test_flydsl_import_failure_is_recorded_not_raised(self):
        sys.modules["aiter.ops.flydsl.utils"] = None
        _, res = self._run(backends="flydsl")
        self.assertFalse(res[0]["available"])
        self.assertIn("flydsl unavailable:", res[0]["note"])

    def test_triton_stub_transposes_the_weight_once_outside_the_timed_loop(self):
        _, res = self._run(backends="triton")
        entry = res[0]
        self.assertTrue(entry["available"])
        self.assertIn("RETIRED", entry["note"])
        # 3 launches (warmup + correctness + timed sample) but only ONE [N,K]->[K,N] transpose.
        self.assertEqual(len(self.stack.launches), 3)
        launch = self.stack.launches[0]
        a2, bm, cbuf, Mr, Nr, Kr = launch["args"][:6]
        self.assertEqual((a2.shape, bm.shape, cbuf.shape), ((8, 16), (16, 32), (8, 32)))
        self.assertEqual((Mr, Nr, Kr), (8, 32, 16))
        self.assertEqual(launch["args"][6:], (16, 1, 32, 1, 32, 1))   # row-major strides of a2/Bm/c
        self.assertEqual(launch["grid"], (1,))
        # the retired stub never writes the oracle values, so it is recorded available-but-incorrect
        self.assertFalse(entry["correct"])
        self.assertIsNotNone(entry["max_rel_err"])

    def test_triton_unavailable_is_recorded_not_raised(self):
        sys.modules["triton"] = None
        _, res = self._run(backends="triton")
        self.assertFalse(res[0]["available"])
        self.assertIn("triton unavailable:", res[0]["note"])
        self.assertEqual(res[0]["artifact"], "")

    def test_explicit_backend_list_is_honoured_verbatim(self):
        _, res = self._run(backends=" rocblas , , hipblaslt ")
        self.assertEqual([r["backend"] for r in res], ["hipblaslt", "rocblas"])

    def test_unknown_backend_names_produce_no_results(self):
        _, res = self._run(backends="cutlass")
        self.assertEqual(res, [])

    def test_incorrect_backend_keeps_a_finite_error_and_a_timing(self):
        # A candidate that runs but returns the wrong values must stay AVAILABLE with a real ms:
        # "ran but incorrect" is not a harness fault, it just cannot win.
        self.stack.functional.linear = lambda a, w, bias=None: (a @ w.t()) * 2.0
        _, res = self._run(backends="hipblaslt")
        self.assertTrue(res[0]["available"])
        self.assertFalse(res[0]["correct"])
        self.assertEqual(res[0]["ms"], 1.5)
        self.assertAlmostEqual(res[0]["max_rel_err"], round(1.0 / 1.02, 5), places=5)

    def test_wrong_shape_output_records_a_null_error(self):
        self.stack.functional.linear = lambda a, w, bias=None: _T((1, 1), a.dtype, 0.0)
        _, res = self._run(backends="hipblaslt")
        self.assertFalse(res[0]["correct"])
        self.assertIsNone(res[0]["max_rel_err"])

    def test_device_is_cuda_when_a_gpu_is_visible(self):
        self.stack.cuda = True
        _, res = self._run(backends="hipblaslt")
        self.assertTrue(res[0]["available"])
        self.assertGreater(self.stack.syncs, 0)


# --------------------------------------------------------------------------- #
# the aiter / flydsl / triton entrypoint helpers
# --------------------------------------------------------------------------- #
class TestBackendHelpers(_FakeStackMixin, unittest.TestCase):
    def test_aiter_probe_prefers_the_nk_weight_for_a_linear_style_op(self):
        A, B = _T((8, 16), BF16, 0.5), _T((32, 16), BF16, 2.0)
        out = ob._aiter_gemm(A, B, None, True)
        self.assertEqual(out.shape, (8, 32))
        self.assertEqual(out.val, 0.5 * 2.0 * 16)
        # the [N,K] weight is offered first, so the first accepting entrypoint wins on argset 1
        self.assertEqual([c for c in self.stack.calls if c[0] == "aiter.gemm_nk"],
                         [("aiter.gemm_nk", (8, 16), (32, 16))])

    def test_aiter_probe_adds_the_bias_the_entrypoint_did_not_take(self):
        out = ob._aiter_gemm(_T((8, 16), BF16, 1.0), _T((32, 16), BF16, 1.0), _T((32,), BF16, 5.0), True)
        self.assertEqual(out.val, 16 + 5.0)

    def test_aiter_probe_transposes_for_a_matmul_style_weight(self):
        # transpose_b=False means B is already [K,N]; the probe must offer B.t() == [N,K] too.
        out = ob._aiter_gemm(_T((8, 16), BF16, 1.0), _T((16, 32), BF16, 1.0), None, False)
        self.assertEqual(out.shape, (8, 32))

    def test_aiter_probe_scans_module_and_submodules_before_giving_up(self):
        self.stack.aiter_all_fail = True
        with self.assertRaises(RuntimeError) as cm:
            ob._aiter_gemm(_T((8, 16), BF16, 1.0), _T((32, 16), BF16, 1.0), None, True)
        msg = str(cm.exception)
        self.assertIn("no working aiter gemm entrypoint", msg)
        self.assertIn("aiter.tuned_gemm.tgemm", msg)          # last probed candidate is reported
        probed = {c[0] for c in self.stack.calls}
        # The three blockscale entrypoints are counted as tried but never appear here: their
        # 5-arg signature raises TypeError before the body runs, so nothing is recorded.
        # gemm_a16_returns_none DOES appear -- a 2-arg entrypoint is genuinely invoked, and is
        # rejected for handing back None rather than for being filtered out. That distinction
        # is the point of the fixture: the probe must not accept a None-returning candidate.
        self.assertEqual(probed, {"aiter.gemm_a16_returns_none", "aiter.gemm_nk", "aiter.ops.linear",
                                  "aiter.ops.triton.gemm_a16w16", "aiter.tuned_gemm.tgemm"})
        # 8 = every distinct gemm-named callable across aiter + ops + ops.triton + tuned_gemm.
        # gemm_a8w8_blockscale_alias is the same object as gemm_a8w8_blockscale and is deduped
        # by id, and matmul_helper is excluded by the name filter.
        self.assertIn("tried 8", msg)

    def test_flydsl_gemm_calls_hgemm_without_preshuffle_and_restores_the_batch_shape(self):
        A, B = _T((2, 4, 16), BF16, 0.5), _T((32, 16), BF16, 2.0)
        out = ob._flydsl_gemm(A, B, None, True)
        self.assertEqual(out.shape, (2, 4, 32))
        self.assertEqual(self.stack.calls[-1], ("flydsl_hgemm", (8, 16), (32, 16), None, False, False))

    def test_flydsl_gemm_normalises_a_kn_weight_to_nk(self):
        out = ob._flydsl_gemm(_T((8, 16), BF16, 1.0), _T((16, 32), BF16, 1.0), None, False)
        self.assertEqual(self.stack.calls[-1][2], (32, 16))
        self.assertEqual(out.shape, (8, 32))

    def test_flydsl_gemm_forwards_the_bias(self):
        out = ob._flydsl_gemm(_T((8, 16), BF16, 1.0), _T((32, 16), BF16, 1.0), _T((32,), BF16, 3.0), True)
        self.assertEqual(self.stack.calls[-1][3], (32,))
        self.assertEqual(out.val, 16 + 3.0)

    def test_flydsl_refuses_fp8_operands_with_routing_guidance(self):
        # A fabricated scale would be a WRONG number; the graceful skip is the correct behaviour.
        for fp8 in (FP8_E4M3FNUZ, FP8_E5M2FNUZ, FP8_E4M3FN, FP8_E5M2):
            with self.assertRaises(RuntimeError) as cm:
                ob._flydsl_gemm(_T((8, 16), fp8, 1.0), _T((32, 16), fp8, 1.0), None, True)
            self.assertIn("flydsl_preshuffle_gemm_a8", str(cm.exception))
        self.assertNotIn("flydsl_hgemm", self.stack.kernel_names())

    def test_triton_kernel_is_built_once_and_cached_at_module_scope(self):
        first = ob._get_triton_mm()
        self.assertIs(ob._get_triton_mm(), first)
        self.assertIs(ob._TRITON_MM, first)
        triton, mm = first
        self.assertIs(triton, self.stack.triton)
        self.assertEqual(sorted(mm.configs[0].kwargs),
                         ["BLOCK_K", "BLOCK_M", "BLOCK_N", "GROUP_M"])
        self.assertEqual(mm.key, ["M", "N", "K"])

    def test_triton_matmul_launches_with_flattened_rows_and_row_major_strides(self):
        A, B, bias = _T((2, 4, 16), BF16, 1.0), _T((32, 16), BF16, 1.0), _T((32,), BF16, 2.0)
        out = ob._triton_matmul(self.stack.torch, A, B, bias, True, False)
        self.assertEqual(out.shape, (2, 4, 32))
        self.assertEqual(len(self.stack.launches), 1)
        args = self.stack.launches[0]["args"]
        self.assertEqual((args[0].shape, args[1].shape, args[2].shape), ((8, 16), (16, 32), (8, 32)))
        self.assertEqual(args[3:], (8, 32, 16, 16, 1, 32, 1, 32, 1))
        # the grouped-pid kernel body must store a full BLOCK_M x BLOCK_N tile in the output dtype
        meta = self.stack.launches[0]["meta"]
        self.assertEqual(self.stack.stores[0]["tile_shape"], (meta["BLOCK_M"], meta["BLOCK_N"]))
        self.assertIs(self.stack.stores[0]["tile_dtype"], BF16)

    def test_triton_matmul_without_bias_returns_the_raw_tile(self):
        out = ob._triton_matmul(self.stack.torch, _T((8, 16), BF16, 1.0), _T((16, 32), BF16, 1.0),
                                None, False, True)
        self.assertEqual(out.shape, (8, 32))
        self.assertEqual(self.stack.launches[0]["args"][1].shape, (16, 32))   # already [K,N]

    def test_triton_matmul_raises_when_triton_is_missing(self):
        sys.modules["triton"] = None
        with self.assertRaises(Exception):
            ob._triton_matmul(self.stack.torch, _T((8, 16), BF16, 1.0), _T((32, 16), BF16, 1.0),
                              None, True, False)


# --------------------------------------------------------------------------- #
# bench_attn
# --------------------------------------------------------------------------- #
class TestBenchAttn(_FakeStackMixin, unittest.TestCase):
    def test_missing_capture_is_reported_as_unavailable(self):
        res = ob.bench_attn(self._args(task=self._task_dir()), {"op_kind": "attn"})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["backend"], "current")
        self.assertFalse(res[0]["available"])
        self.assertIsNone(res[0]["ms"])
        self.assertIn("needs reference_io.pt", res[0]["note"])

    def test_captured_oracle_is_validated_and_backend_swaps_are_delegated(self):
        d = self._task_dir()
        io_path = os.path.join(d, "reference_io.pt")
        open(io_path, "w").close()
        res = ob.bench_attn(self._args(task=d), {"op_kind": "attn"})
        self.assertTrue(res[0]["available"])
        self.assertTrue(res[0]["correct"])
        self.assertIsNone(res[0]["ms"])                     # op-level attention is not raced here
        self.assertEqual(res[0]["artifact"], io_path)
        self.assertIn("--attention-backend", res[0]["note"])
        self.assertIn("Config Tuner", res[0]["note"])


# --------------------------------------------------------------------------- #
# main() -- winner selection, the deployable recipe, and the emitted JSON
# --------------------------------------------------------------------------- #
class TestMainSummary(_ObStateMixin, unittest.TestCase):
    def _run_main(self, meta, results=None, raises=None, extra_argv=(), capture_stdout=False):
        d = self._task_dir(meta)
        out_path = "" if capture_stdout else os.path.join(d, "result.json")

        def fake_bench(args, m):
            self.seen = (args, m)
            if raises is not None:
                raise raises
            return list(results or [])
        ob.bench_gemm = fake_bench
        ob.bench_attn = fake_bench
        argv = ["op_bench.py", "--task", d] + list(extra_argv)
        if out_path:
            argv += ["--out", out_path]
        old_argv = sys.argv
        sys.argv = argv
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                ob.main()
        finally:
            sys.argv = old_argv
        text = buf.getvalue()
        if out_path:
            with open(out_path) as fh:
                summary = json.load(fh)
        else:
            summary = json.loads(text[:text.rindex("}") + 1])
        return summary, text

    @staticmethod
    def _res(backend, ms=None, correct=False, **kw):
        entry = {"backend": backend, "available": ms is not None, "correct": correct, "ms": ms}
        entry.update(kw)
        return entry

    def test_default_arguments_reach_the_bench_function(self):
        self._run_main({"op_kind": "gemm"}, results=[])
        args, meta = self.seen
        self.assertEqual((args.repeats, args.warmup, args.tol, args.seed), (50, 10, 2e-2, 0))
        self.assertEqual(args.backends, "")
        self.assertFalse(args.triton_autotune)

    def test_cli_overrides_are_parsed(self):
        self._run_main({"op_kind": "gemm"}, results=[],
                       extra_argv=["--backends", "aiter,triton", "--repeats", "7", "--warmup", "2",
                                   "--tol", "0.05", "--seed", "9", "--triton-autotune"])
        args, _ = self.seen
        self.assertEqual((args.backends, args.repeats, args.warmup, args.tol, args.seed),
                         ("aiter,triton", 7, 2, 0.05, 9))
        self.assertTrue(args.triton_autotune)

    def test_fastest_correct_backend_wins_and_speedup_is_against_hipblaslt(self):
        summary, text = self._run_main(
            {"op_kind": "gemm", "pct_gpu_time": 20.0},
            results=[self._res("hipblaslt", ms=2.0, correct=True),
                     self._res("aiter", ms=0.5, correct=True),
                     self._res("rocblas", ms=1.0, correct=True)])
        self.assertEqual(summary["winner_backend"], "aiter")
        self.assertEqual(summary["winner_ms"], 0.5)
        self.assertEqual(summary["baseline_backend"], "hipblaslt")
        self.assertEqual(summary["baseline_ms"], 2.0)
        self.assertEqual(summary["isolated_speedup"], 4.0)
        self.assertFalse(summary["winner_editable"])
        self.assertEqual(summary["winner_kind"], "none")
        self.assertFalse(summary["harness_suspect"])
        self.assertEqual(summary["harness_error"], "")
        self.assertEqual(summary["op_kind"], "gemm")
        self.assertIn("OPBENCH winner=aiter speedup=4.0x", text)

    def test_amdahl_ceiling_bounds_the_e2e_delta_the_win_can_produce(self):
        summary, _ = self._run_main(
            {"op_kind": "gemm", "pct_gpu_time": 20.0},
            results=[self._res("hipblaslt", ms=2.0, correct=True),
                     self._res("aiter", ms=1.0, correct=True)])
        # 20% of GPU time halved -> 10% of total time saved -> 1/0.9 - 1 = +11.111%
        self.assertAlmostEqual(summary["amdahl_ceiling_e2e_pct"], 11.111, places=3)
        self.assertEqual(summary["pct_gpu_time"], 20.0)

    def test_amdahl_ceiling_is_omitted_without_a_gpu_time_share(self):
        summary, _ = self._run_main({"op_kind": "gemm"},
                                    results=[self._res("hipblaslt", ms=1.0, correct=True)])
        self.assertIsNone(summary["amdahl_ceiling_e2e_pct"])
        self.assertIsNone(summary["pct_gpu_time"])

    def test_amdahl_ceiling_accepts_the_pct_gpu_alias(self):
        summary, _ = self._run_main({"op_kind": "gemm", "pct_gpu": 0.5},
                                    results=[self._res("hipblaslt", ms=2.0, correct=True),
                                             self._res("ck", ms=1.0, correct=True)])
        self.assertEqual(summary["pct_gpu_time"], 0.5)
        self.assertIsNotNone(summary["amdahl_ceiling_e2e_pct"])

    def test_unparseable_gpu_time_share_omits_the_ceiling_instead_of_crashing(self):
        summary, _ = self._run_main({"op_kind": "gemm", "pct_gpu_time": "unknown"},
                                    results=[self._res("hipblaslt", ms=1.0, correct=True)])
        self.assertIsNone(summary["amdahl_ceiling_e2e_pct"])

    def test_tunableop_winner_emits_the_env_that_survives_cuda_graph_capture(self):
        summary, _ = self._run_main(
            {"op_kind": "gemm"},
            results=[self._res("hipblaslt", ms=2.0, correct=True),
                     self._res("tunableop", ms=1.0, correct=True, artifact="/task/tunableop.csv")])
        self.assertEqual(summary["winner_kind"], "env")
        self.assertEqual(summary["tuning_artifact"], "/task/tunableop.csv")
        self.assertIn("PYTORCH_TUNABLEOP_ENABLED=1", summary["apply_env"])
        self.assertIn("PYTORCH_TUNABLEOP_TUNING=0", summary["apply_env"])
        self.assertIn("PYTORCH_TUNABLEOP_FILENAME=/task/tunableop.csv", summary["apply_env"])
        self.assertIn("captured into the cuda-graph", summary["deployable_note"])

    def test_rocblas_winner_emits_the_blas_preference_env(self):
        summary, _ = self._run_main({"op_kind": "gemm"},
                                    results=[self._res("rocblas", ms=1.0, correct=True)])
        self.assertEqual(summary["apply_env"], "TORCH_BLAS_PREFER_HIPBLASLT=0")
        self.assertEqual(summary["winner_kind"], "env")

    def test_ck_winner_is_flagged_for_deployability_verification(self):
        summary, _ = self._run_main({"op_kind": "gemm"},
                                    results=[self._res("ck", ms=1.0, correct=True)])
        self.assertEqual(summary["winner_kind"], "flag")
        self.assertEqual(summary["apply_env"], "")
        self.assertIn("verify deployability at the e2e gate", summary["deployable_note"])

    def test_editable_winner_is_routed_to_a_kernel_rewrite(self):
        for backend in ("triton", "hip"):
            summary, _ = self._run_main({"op_kind": "gemm"},
                                        results=[self._res(backend, ms=1.0, correct=True)])
            self.assertTrue(summary["winner_editable"])
            self.assertEqual(summary["winner_kind"], "patch_candidate")

    def test_hipblaslt_winner_has_nothing_to_deploy(self):
        summary, _ = self._run_main({"op_kind": "gemm"},
                                    results=[self._res("hipblaslt", ms=1.0, correct=True)])
        self.assertEqual(summary["winner_kind"], "none")
        self.assertEqual(summary["isolated_speedup"], 1.0)      # it is its own baseline
        self.assertIn("nothing to deploy", summary["deployable_note"])

    def test_library_winner_without_a_baseline_reports_a_neutral_speedup(self):
        summary, _ = self._run_main({"op_kind": "gemm"},
                                    results=[self._res("flydsl", ms=1.0, correct=True)])
        self.assertEqual(summary["winner_backend"], "flydsl")
        self.assertIsNone(summary["baseline_backend"])
        self.assertEqual(summary["isolated_speedup"], 1.0)
        self.assertEqual(summary["winner_kind"], "none")
        self.assertIn("verify deployability", summary["deployable_note"])

    def test_no_correct_backend_means_no_winner_and_zero_speedup(self):
        summary, text = self._run_main(
            {"op_kind": "gemm"},
            results=[self._res("hipblaslt", ms=1.0, correct=False),
                     self._res("aiter", ms=None, correct=False, note="slow but fine")])
        self.assertIsNone(summary["winner_backend"])
        self.assertIsNone(summary["winner_ms"])
        self.assertEqual(summary["isolated_speedup"], 0.0)
        self.assertFalse(summary["winner_editable"])
        # something RAN (it was merely incorrect) -> not a harness self-fault
        self.assertFalse(summary["harness_suspect"])
        self.assertNotIn("harness_error=", text)

    def test_every_candidate_raising_trips_the_harness_self_fault_signal(self):
        summary, text = self._run_main(
            {"op_kind": "gemm"},
            results=[self._res("aiter_blockscale", ms=None, note="call raised: TypeError()",
                               raised=True),
                     self._res("aiter_bpreshuffle", ms=None, note="call failed: RuntimeError()")])
        self.assertTrue(summary["harness_suspect"])
        self.assertIn("call raised: TypeError()", summary["harness_error"])
        self.assertIn("harness_suspect=True", text)
        self.assertIn("harness_error=", text)

    def test_a_recorded_skip_alone_is_not_a_harness_self_fault(self):
        summary, _ = self._run_main(
            {"op_kind": "gemm"},
            results=[self._res("grouped_quant_gemm", ms=None, raised=False,
                               note="not a dense torch-BLAS bake-off candidate")])
        self.assertFalse(summary["harness_suspect"])
        self.assertEqual(summary["harness_error"], "")

    def test_empty_result_list_is_not_a_harness_self_fault(self):
        summary, _ = self._run_main({"op_kind": "gemm"}, results=[])
        self.assertFalse(summary["harness_suspect"])
        self.assertEqual(summary["results"], [])

    def test_a_raising_bench_becomes_a_traced_error_result(self):
        summary, _ = self._run_main({"op_kind": "gemm"},
                                    raises=ValueError("empty/None shape"))
        self.assertEqual(len(summary["results"]), 1)
        entry = summary["results"][0]
        self.assertEqual(entry["backend"], "ERROR")
        self.assertIn("empty/None shape", entry["note"])
        self.assertIn("Traceback", entry["trace"])
        self.assertTrue(summary["harness_suspect"])
        self.assertIn("empty/None shape", summary["harness_error"])

    def test_attn_task_is_routed_to_the_attention_bench(self):
        summary, _ = self._run_main({"op_kind": "ATTN"},
                                    results=[self._res("current", ms=None, correct=True,
                                                       note="delegated to config track")])
        self.assertEqual(summary["op_kind"], "attn")
        self.assertIsNone(summary["winner_backend"])

    def test_attention_baseline_is_the_current_captured_path(self):
        summary, _ = self._run_main({"op_kind": "attn"},
                                    results=[self._res("current", ms=3.0, correct=True)])
        self.assertEqual(summary["baseline_backend"], "current")
        self.assertEqual(summary["winner_backend"], "current")
        self.assertEqual(summary["isolated_speedup"], 1.0)

    def test_missing_op_kind_defaults_to_gemm(self):
        summary, _ = self._run_main({}, results=[])
        self.assertEqual(summary["op_kind"], "gemm")

    def test_regime_sets_the_deployment_graph_timing_context(self):
        self._run_main({"op_kind": "gemm", "regime": {"cuda_graph": True}}, results=[])
        self.assertTrue(ob._GRAPH_MODE)

    def test_enforce_eager_regime_keeps_timing_eager(self):
        self._run_main({"op_kind": "gemm", "regime": {"enforce_eager": True}}, results=[])
        self.assertFalse(ob._GRAPH_MODE)

    def test_regimeless_task_stays_eager_amortized(self):
        self._run_main({"op_kind": "gemm"}, results=[])
        self.assertFalse(ob._GRAPH_MODE)

    def test_summary_is_printed_when_no_out_file_is_requested(self):
        summary, text = self._run_main({"op_kind": "gemm"},
                                       results=[self._res("hipblaslt", ms=1.25, correct=True)],
                                       capture_stdout=True)
        self.assertEqual(summary["winner_ms"], 1.25)
        self.assertIn("OPBENCH winner=hipblaslt", text)
        self.assertIn("editable=False kind=none", text)

    def test_out_file_carries_the_task_dir_and_full_result_list(self):
        summary, _ = self._run_main({"op_kind": "gemm"},
                                    results=[self._res("hipblaslt", ms=1.0, correct=True)])
        args, _ = self.seen
        self.assertEqual(summary["task"], args.task)
        self.assertEqual(summary["results"][0]["backend"], "hipblaslt")
        self.assertEqual(summary["apply_flags"], "")


def _r(backend, ms, correct=True, **kw):
    d = {"backend": backend, "available": True, "correct": correct, "ms": ms}
    d.update(kw)
    return d


class MergeServedTest(unittest.TestCase):
    """`_merge_served` collapses a per-bucket sweep into ONE served-weighted number.

    The point of the sweep is that a candidate's speedup is shape-dependent: a large-M prefill
    tile can be much faster at M=8192 and no better -- or worse -- at the M=64 the workload spends
    94% of its passes on. Benching one shape reported the first number and hid the second."""

    def test_ms_is_the_pass_weighted_mean_not_the_largest_buckets(self):
        merged = ob._merge_served([("decode", 64, 1024, [_r("cand", 1.0)]),
                                   ("prefill", 8192, 64, [_r("cand", 10.0)])])
        self.assertEqual(len(merged), 1)
        # (1.0*1024 + 10.0*64) / 1088 == 1.5294 ; the single-bucket bench would have said 10.0
        self.assertAlmostEqual(merged[0]["ms"], 1.5294, places=3)

    def test_a_win_only_at_the_unserved_shape_no_longer_wins(self):
        """The regression this fix exists for: `cand` is 5x faster at the big prefill bucket and
        2x SLOWER at the decode bucket the workload runs 1024 of its 1088 passes on. Benching
        max(m_buckets) alone reports 5x. Served-weighted it is a net LOSS, because the decode
        penalty (1.0ms x 1024 passes) dwarfs the prefill saving (8.0ms x 64 passes)."""
        sweep = [("decode", 64, 1024, [_r("base", 1.0), _r("cand", 2.0)]),
                 ("prefill", 8192, 64, [_r("base", 10.0), _r("cand", 2.0)])]
        merged = {r["backend"]: r for r in ob._merge_served(sweep)}
        big_bucket_speedup = 10.0 / 2.0
        served_speedup = merged["base"]["ms"] / merged["cand"]["ms"]
        self.assertEqual(big_bucket_speedup, 5.0)
        self.assertLess(served_speedup, 1.0)

    def test_the_per_bucket_numbers_stay_visible_for_audit(self):
        merged = ob._merge_served([("decode", 64, 1024, [_r("cand", 1.0)]),
                                   ("prefill", 8192, 64, [_r("cand", 10.0)])])
        self.assertEqual(merged[0]["ms_by_bucket"], {"decode:M=64": 1.0, "prefill:M=8192": 10.0})
        self.assertIn("served-weighted over 2 buckets", merged[0]["note"])

    def test_a_candidate_that_breaks_at_one_served_shape_is_not_correct_anywhere(self):
        merged = ob._merge_served([("decode", 64, 1024, [_r("cand", 1.0)]),
                                   ("prefill", 8192, 64, [_r("cand", None, correct=False,
                                                             note="wrong at large M")])])
        self.assertFalse(merged[0]["correct"])
        self.assertIsNone(merged[0]["ms"])                 # never selectable as a winner
        self.assertIn("[prefill M=8192] wrong at large M", merged[0]["note"])

    def test_a_bucket_that_raised_propagates_and_does_not_poison_the_mean(self):
        merged = ob._merge_served([("decode", 64, 1024, [_r("cand", 1.0)]),
                                   ("prefill", 8192, 64, [_r("cand", None, correct=False,
                                                             raised=True)])])
        self.assertTrue(merged[0]["raised"])
        self.assertIsNone(merged[0]["ms"])

    def test_backend_order_is_preserved_so_the_baseline_lookup_still_works(self):
        merged = ob._merge_served([("decode", 64, 1024, [_r("hipblaslt", 1.0), _r("ck", 2.0)]),
                                   ("prefill", 8192, 64, [_r("hipblaslt", 3.0), _r("ck", 4.0)])])
        self.assertEqual([r["backend"] for r in merged], ["hipblaslt", "ck"])

    def test_a_backend_missing_from_one_bucket_is_still_merged(self):
        merged = {r["backend"]: r for r in
                  ob._merge_served([("decode", 64, 1024, [_r("a", 1.0), _r("b", 2.0)]),
                                    ("prefill", 8192, 64, [_r("a", 3.0)])])}
        self.assertEqual(merged["b"]["ms"], 2.0)
        self.assertEqual(set(merged["a"]["ms_by_bucket"]), {"decode:M=64", "prefill:M=8192"})


class ServedSweepDispatchTest(unittest.TestCase):
    """bench_gemm sweeps the served buckets; with nothing to sweep it must behave exactly as before."""

    def setUp(self):
        self.calls = []
        self._orig = ob._bench_gemm_at
        ob._bench_gemm_at = lambda args, meta, m=None: (self.calls.append(m) or [_r("x", 1.0)])
        self.addCleanup(lambda: setattr(ob, "_bench_gemm_at", self._orig))

    def test_two_served_buckets_are_each_benched(self):
        ob.bench_gemm(None, _swm_meta_ob())
        self.assertEqual(sorted(c for c in self.calls if c), [64, 8192])

    def test_no_serving_model_benches_once_at_the_old_shape(self):
        ob.bench_gemm(None, {"m_buckets": [8192]})
        self.assertEqual(self.calls, [None])

    def test_a_single_served_bucket_is_not_wrapped_in_a_sweep(self):
        ob.bench_gemm(None, _swm_meta_ob(served_regimes=["decode"]))
        self.assertEqual(self.calls, [None])


def _swm_meta_ob(**kw):
    meta = {"m_buckets": [64, 1024, 8192],
            "workload": {"serving_weight_model": {
                "isl": 16384, "osl": 1024, "conc": 64, "prefill_chunk": 8192,
                "analytic_calls": {"prefill": 64, "decode": 1024}}}}
    meta.update(kw)
    return meta


if __name__ == "__main__":
    unittest.main(verbosity=2)
