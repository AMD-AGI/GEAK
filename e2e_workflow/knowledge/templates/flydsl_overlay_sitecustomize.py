"""REFERENCE TEMPLATE (e2e_workflow e2e_integrator) — a PROVEN capture-safe overlay seam.

This is the working seam from the +14.17% Qwen3.5-27B-FP8 FlyDSL run (CUDA-graph capture OK, engaged on
the live decode path). REUSE it when overlaying an authored/JIT kernel into a CUDA-graph-captured sglang
server; adapt only: `_TARGET`/`_ALIAS`/`_ATTR` (the op's source module+attr), `_FP8_UTILS`+
`_FP8_UTILS_NAMES` (the live call-site module + the names it import-copied), and `_IMPL_MOD` (your
authored impl module). The capture-safety machinery (lazy meta-path finder = no eager import/fork-storm;
`flydsl_overlay_precompile` = JIT all decode M-buckets BEFORE capture; data_ptr weight cache = NO per-call
host sync; one-shot engagement proof = no GPU readback) is the GENERAL pattern — keep all of it. The two
capture-killers it defeats: (1) a per-call host sync (`.item()`), (2) JIT/alloc inside the captured region.
See e2e_integrator.md "CUDA-graph-safe overlay" and knowledge/gemm_attention_backends.md "CUDA-graph
capture safety".

----------------------------------------------------------------------------------------------------
Author-mode overlay activator (FlyDSL) — PASSIVE / LAZY / CUDA-graph-safe.

ROUND-2 host_runtime (r2_d2): COMPLETE the live-engagement seam and remove the
per-call host sync from the weight-cache key.

Python imports `sitecustomize` automatically at interpreter startup for any
directory on `sys.path` / `PYTHONPATH`. This module is the ENGAGEMENT SEAM that
routes the FlyDSL fused block-scale GEMM onto two surfaces WITHOUT editing any
aiter/sglang file on disk and WITHOUT eager-importing aiter/flydsl/torch/sglang
at process start:

  (A) The IMMUTABLE isolated unittest.
      It resolves its target via `importlib.import_module(
      'aiter.ops.triton.gemm_a8w8_blockscale')` then `getattr`s the
      `gemm_a8w8_blockscale` attribute. We rebind that attribute (on both the
      file module `aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale` and the
      public alias `aiter.ops.triton.gemm_a8w8_blockscale`) to the FlyDSL impl.
      The unittest does a FRESH getattr per call, so rebinding the attribute is
      sufficient for surface (A).

  (B) The LIVE sglang serving DECODE/PREFILL path (the REAL objective).
      `sglang.srt.layers.quantization.fp8_utils` does, at ITS OWN import time
      (inside `if _use_aiter:`, fp8_utils.py:88-97):

          from aiter import (gemm_a8w8_blockscale_bpreshuffle, ...)
          from aiter.ops.triton.gemm_a8w8_blockscale import (
              gemm_a8w8_blockscale as triton_gemm_a8w8_blockscale,
          )

      Those are `from ... import ... as ...` statements — each binds a NEW
      module-global name *inside fp8_utils* pointing at whatever the attribute is
      at that instant. `aiter_w8a8_block_fp8_linear` (fp8_utils.py:760-804, the
      decode/prefill call site) then reads those module globals every call:

          if use_triton:   gemm_a8w8_blockscale_op = triton_gemm_a8w8_blockscale
          else:            gemm_a8w8_blockscale_op = gemm_a8w8_blockscale_bpreshuffle
          output = gemm_a8w8_blockscale_op(q_input, weight, x_scale, weight_scale, dtype=...)

      `use_triton` is True by default (only the gfx95-tuned (n,k) table flips it),
      so the LIVE path almost always calls `triton_gemm_a8w8_blockscale`. If
      fp8_utils imports that symbol BEFORE we patch the aiter attribute, rebinding
      the aiter attribute later does NOT reach fp8_utils' already-captured global —
      surface (A) alone silently masks ZERO live engagement (confirmed in R1 +
      MEMORY 'aiter FP8 GEMM has no overlay seam'). We therefore ALSO wrap the
      loader of fp8_utils and, AFTER it executes, rebind its captured globals
      (`triton_gemm_a8w8_blockscale` AND `gemm_a8w8_blockscale_bpreshuffle`,
      following the import-time COPY, not the source module attribute) to the
      FlyDSL impl. Both wraps happen lazily via a meta-path finder.

CUDA-GRAPH SAFETY (decode):
  sglang captures decode batches into a CUDA graph during warmup. The FlyDSL
  kernel must be graph-safe: M-agnostic JIT precompile BEFORE capture (so a
  first-decode-shape JIT does not happen inside the captured region), and NO host
  syncs (.item()/.cpu()/sync) in the hot path. We install an M-agnostic warmup
  hook `flydsl_overlay_precompile(weight, weight_scale, m_buckets=...)` on the
  FlyDSL impl module. The e2e runner calls it ONCE during warmup BEFORE capture.
  It (1) JITs the down-proj kernel at all decode M buckets and (2) PRE-REGISTERS
  the persistent weight into the FlyDSL weight cache so the per-call lookup is a
  pure data_ptr hit with NO fingerprint host sync (see HOST-SYNC REMOVAL below).

HOST-SYNC REMOVAL (per-call fingerprint -> warmup-time fingerprint):
  The FlyDSL weight cache (gemm_a8w8_blockscale_flydsl.py:_get_weight /
  _weight_fingerprint) keys on (data_ptr, shape, dtype, w_scale fingerprint).
  The fingerprint does a small host sync (w_scale.sum().item()) PER CALL — needed
  in the isolated unittest where transient tensors recycle data_ptr, but a pure
  cost in LIVE serving where weights are persistent (data_ptr is a stable key).
  Because the math file is OWNED by r2_d0/r2_d1 this round and I may NOT edit it,
  I move the sync out of the hot path COOPERATIVELY:
    * The precompile/warmup hook pays the fingerprint ONCE (it calls the impl on
      the real weight, populating _WEIGHT_CACHE), AND
    * it records the weight's data_ptr in a seam-owned registry
      `flydsl_overlay_warm_ptrs` (a set on the impl module) + a fast-path table
      `flydsl_overlay_weight_cache` (data_ptr -> prepped (wq_shuf, w_col_scale)).
  MERGE INSTRUCTION for r2_d0/d1 (documented, not edited here): in `_get_weight`,
  FIRST try a pure data_ptr lookup against `flydsl_overlay_weight_cache` (or check
  `data_ptr in flydsl_overlay_warm_ptrs` and skip `_weight_fingerprint`); only
  fall back to the fingerprint path on a miss. That makes the CUDA-graph decode
  path a pure data_ptr hit with ZERO host sync, while the unittest (no warmup,
  recycled ptrs) keeps the safe fingerprint key. If the merge keeps the current
  `_get_weight` unchanged, the seam is still correct (it just leaves the existing
  per-call fingerprint in place); the registry is additive and harmless.

ENGAGEMENT PROOF:
  A one-shot atomic flag + banner (`_engagement`) is flipped the FIRST time the
  FlyDSL impl runs on the live path. Written ONCE (not per-call), no GPU readback,
  no host sync — safe inside a captured decode region. A monotone host-side call
  counter is kept for honest log accounting (plain int increment, no GPU touch).

SERVER-LAUNCH NOTE (NOT baked here, by requirement #2 — keep the seam passive):
  The first prefill JIT-compiles the FlyDSL kernel, which can exceed sglang's
  default scheduler watchdog. The e2e runner MUST launch with:
      --watchdog-timeout 600
  and call `flydsl_overlay_precompile(down_proj_weight, down_proj_weight_scale)`
  ONCE during warmup BEFORE CUDA-graph capture. Do not bake server flags here.

Activate by putting this directory on PYTHONPATH, e.g.:
    cd <workspace> && PYTHONPATH="$PWD/kernel_src" \
        bash <workflow>/scripts/gpu_lock.sh <gpu> python3 unittest.py
For e2e: OVERLAY_PYTHONPATH=<workspace>/kernel_src (the bench dispatcher prepends it).
"""

import importlib
import importlib.abc
import importlib.util
import sys

# --- Surface (A): aiter attribute rebind (drives the isolated unittest) -------
# The file-backed module that actually defines `gemm_a8w8_blockscale`. The public
# alias `aiter.ops.triton.gemm_a8w8_blockscale` resolves to this same object.
_TARGET = "aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale"
_ALIAS = "aiter.ops.triton.gemm_a8w8_blockscale"
_ATTR = "gemm_a8w8_blockscale"

# --- Surface (B): live sglang call-site rebind (drives the decode/prefill path)
# fp8_utils captures these as module globals at its own import time (fp8_utils.py
# lines 88-97, inside `if _use_aiter:`). The live `aiter_w8a8_block_fp8_linear`
# reads them per call. We follow the import-time COPY, not the source attribute.
_FP8_UTILS = "sglang.srt.layers.quantization.fp8_utils"
_FP8_UTILS_NAMES = (
    "triton_gemm_a8w8_blockscale",       # use_triton=True path (dominant)
    "gemm_a8w8_blockscale_bpreshuffle",  # use_triton=False (gfx95-tuned) path
)

_IMPL_MOD = "gemm_a8w8_blockscale_flydsl"

_state = {"aiter_done": False, "fp8utils_done": False}


# ---------------------------------------------------------------------------
# Engagement proof: one-shot, host-sync-free, never per-call GPU touch.
# ---------------------------------------------------------------------------
class _Engagement:
    """Atomic one-shot proof the FlyDSL kernel ran on the LIVE path.

    `mark()` flips a Python bool exactly once (prints a banner once) and bumps a
    plain Python counter. No .item()/.cpu()/sync — safe inside a CUDA-graph
    capture. The counter is a pure host-side int, never read from device memory.
    """

    def __init__(self):
        self.engaged = False
        self.calls = 0

    def mark(self):
        self.calls += 1
        if not self.engaged:
            self.engaged = True
            try:
                print(
                    "[flydsl-overlay] ENGAGED: FlyDSL gemm_a8w8_blockscale ran on "
                    "the LIVE call site (first call observed)",
                    flush=True,
                )
            except Exception:
                pass


_engagement = _Engagement()


def _impl():
    """Lazily import the FlyDSL drop-in. Wrapped so the FIRST live call trips the
    one-shot engagement proof without adding any per-call host sync. The import
    of torch/flydsl happens only when this is first CALLED (i.e. when the live
    server or the unittest actually invokes the GEMM), never at seam import."""
    from gemm_a8w8_blockscale_flydsl import gemm_a8w8_blockscale as fn

    def _wrapped(*args, **kwargs):
        _engagement.mark()
        return fn(*args, **kwargs)

    try:
        _wrapped.__name__ = getattr(fn, "__name__", "gemm_a8w8_blockscale")
        _wrapped.__wrapped__ = fn
        _wrapped.flydsl_overlay = True
    except Exception:
        pass
    return _wrapped


def _install_warmup_hook():
    """Install the M-agnostic precompile + weight-preregister hook on the FlyDSL
    impl module so the e2e runner (or sglang warmup) can JIT the decode down-proj
    shapes and pre-register persistent weights BEFORE CUDA-graph capture.

    Idempotent; does NO GPU work itself (only attaches callables + registries).
    Returns silently if the impl module is not importable yet."""
    try:
        impl_mod = importlib.import_module(_IMPL_MOD)
    except Exception:
        return

    # Seam-owned registries the math owner (r2_d0/d1) can consult by PURE data_ptr
    # to skip the per-call fingerprint host sync once a weight is warmed. Additive
    # and harmless if the math file ignores them.
    if getattr(impl_mod, "flydsl_overlay_warm_ptrs", None) is None:
        impl_mod.flydsl_overlay_warm_ptrs = set()
    if getattr(impl_mod, "flydsl_overlay_weight_cache", None) is None:
        impl_mod.flydsl_overlay_weight_cache = {}

    if getattr(impl_mod, "flydsl_overlay_precompile", None) is not None:
        return

    def flydsl_overlay_precompile(weight, weight_scale, m_buckets=None, device=None):
        """M-agnostic warmup: JIT-compile the FlyDSL down-proj kernel for every M
        bucket the decode CUDA graph may capture, AND pre-register the persistent
        weight so the per-call cache lookup is a pure data_ptr hit (no host sync).

        Call this ONCE during sglang warmup, BEFORE CUDA-graph capture, for each
        down-proj weight, at ALL M (not prefill-only). Host syncs HERE are fine —
        warmup is OUTSIDE the graph. Best-effort: failures are swallowed so a
        warmup hiccup never crashes the server (the kernel then JITs lazily, which
        is why --watchdog-timeout must be raised)."""
        import torch

        fn = getattr(impl_mod, "gemm_a8w8_blockscale", None)
        if fn is None:
            return
        if device is None:
            device = weight.device if hasattr(weight, "device") else "cuda"
        n, k = int(weight.shape[0]), int(weight.shape[1])

        # (1) Pre-register the weight prep ONCE (pays fingerprint+preshuffle here,
        #     not per call). Populate the seam fast-path table keyed by data_ptr.
        try:
            _prep = getattr(impl_mod, "_prep_weight", None)
            _getw = getattr(impl_mod, "_get_weight", None)
            ptr = int(weight.data_ptr())
            if _getw is not None:
                # Warms the math file's own _WEIGHT_CACHE (and pays its fingerprint
                # exactly once, at warmup) so a later hot-path call hits the cache.
                prepped = _getw(weight, weight_scale, n, k)
            elif _prep is not None:
                prepped = _prep(weight, weight_scale, n, k)
            else:
                prepped = None
            impl_mod.flydsl_overlay_warm_ptrs.add(ptr)
            if prepped is not None:
                impl_mod.flydsl_overlay_weight_cache[ptr] = prepped
        except Exception:
            pass

        # (2) JIT the kernel for every decode M bucket BEFORE capture.
        if m_buckets is None:
            m_buckets = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128,
                         256, 512, 1024, 2048)
        blk = 128
        sk = (k + blk - 1) // blk
        fp8 = getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn)
        for m in m_buckets:
            try:
                x = torch.zeros((m, k), device=device, dtype=fp8)
                xs = torch.ones((m, sk), device=device, dtype=torch.float32)
                fn(x, weight, xs, weight_scale, dtype=torch.bfloat16)
            except Exception:
                continue
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    try:
        impl_mod.flydsl_overlay_precompile = flydsl_overlay_precompile
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Surface (A) patch: rebind the aiter attribute (unittest).
# ---------------------------------------------------------------------------
def _patch_aiter(mod):
    # Idempotent and ALWAYS-ON: aiter's backward-compat finder re-executes the
    # target module a second time under the public alias name, so we must patch
    # on EVERY exec (not just the first) or the alias copy keeps the stock fn.
    try:
        fn = _impl()
    except Exception as exc:  # pragma: no cover
        print(f"[flydsl-overlay] import of flydsl impl failed: {exc!r}")
        return
    setattr(mod, _ATTR, fn)
    alias = sys.modules.get(_ALIAS)
    if alias is not None and alias is not mod:
        setattr(alias, _ATTR, fn)
    _install_warmup_hook()
    if not _state["aiter_done"]:
        _state["aiter_done"] = True
        print("[flydsl-overlay] bound FlyDSL gemm_a8w8_blockscale over aiter triton symbol")


# ---------------------------------------------------------------------------
# Surface (B) patch: rebind fp8_utils' captured module globals (live path).
# ---------------------------------------------------------------------------
def _patch_fp8_utils(mod):
    # Only rebind names fp8_utils actually captured (it does so only inside
    # `if _use_aiter:` — i.e. when SGLANG_USE_AITER is set on HIP). If the names
    # are absent, aiter is not in use and there is nothing to engage.
    present = [n for n in _FP8_UTILS_NAMES if hasattr(mod, n)]
    if not present:
        if not _state["fp8utils_done"]:
            _state["fp8utils_done"] = True
            print(
                "[flydsl-overlay] fp8_utils imported without aiter blockscale "
                "symbols (SGLANG_USE_AITER off?) — nothing to engage on live path"
            )
        return
    try:
        fn = _impl()
    except Exception as exc:  # pragma: no cover
        print(f"[flydsl-overlay] import of flydsl impl failed (fp8_utils): {exc!r}")
        return
    # Follow the import-time COPY: rebind the module GLOBALS, not the source attr.
    for name in present:
        setattr(mod, name, fn)
    _install_warmup_hook()
    if not _state["fp8utils_done"]:
        _state["fp8utils_done"] = True
        print(
            "[flydsl-overlay] rebound fp8_utils live call-site globals "
            f"{present} -> FlyDSL (decode/prefill down-proj will engage)"
        )


# ---------------------------------------------------------------------------
# Meta-path finder wrapping the genuine loaders of both targets.
# ---------------------------------------------------------------------------
class _WrapLoader(importlib.abc.Loader):
    def __init__(self, inner, after):
        self._inner = inner
        self._after = after

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            self._after(module)
        except Exception as exc:  # pragma: no cover
            print(f"[flydsl-overlay] post-exec patch failed: {exc!r}")


class _Finder(importlib.abc.MetaPathFinder):
    # fullname -> post-exec patch fn
    _HOOKS = {
        _TARGET: _patch_aiter,
        _FP8_UTILS: _patch_fp8_utils,
    }

    def find_spec(self, fullname, path=None, target=None):
        after = self._HOOKS.get(fullname)
        if after is None:
            return None
        # Keep our finder ahead of aiter's compat finder and avoid self-recursion
        # while resolving the genuine spec.
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        try:
            spec = importlib.util.find_spec(fullname)
        except Exception:
            spec = None
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _WrapLoader(spec.loader, after)
        return spec


# If a target is somehow already imported, patch immediately; else hook it. This
# does NOT import anything that is not already in sys.modules (stays lazy: no
# aiter/flydsl/torch/sglang import is triggered at seam import time).
_existing_aiter = sys.modules.get(_TARGET) or sys.modules.get(_ALIAS)
if _existing_aiter is not None:
    _patch_aiter(_existing_aiter)

_existing_fp8 = sys.modules.get(_FP8_UTILS)
if _existing_fp8 is not None:
    _patch_fp8_utils(_existing_fp8)

if _existing_aiter is None or _existing_fp8 is None:
    sys.meta_path.insert(0, _Finder())
