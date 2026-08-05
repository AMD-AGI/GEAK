#!/usr/bin/env python3
"""Run plain's software pipeliner on a Gluon kernel WITHOUT editing any Triton file.

No upstream `gluon_to_ttgir` calls `add_schedule_loops` / `add_pipeline` -- checked on
3.6.0, 3.7.0, 3.7.1 and 3.8.0. The passes themselves ship in `libtriton` on all four; only
the Python pass list omits them. So the pipeline is reachable, and the question is only how
you reach it.

WHY NOT THE ENV VAR THE PACK USED TO NAME. `TRITON_GLUON_SWP_PIPELINE`,
`TRITON_GLUON_COOP_LDS` and `TRITON_GLUON_PINGPONG` are additions to a VENDOR FORK's
`GetEnv.h`; no upstream version reads them. Measured on clean 3.7.1 and 3.8.0: setting them
is **tolerated and inert** -- so is a knob invented on the spot -- which is the worst
outcome available. Nothing errors, nothing changes, and a null result reads as "this
technique does not work here" instead of "that variable does not exist in this build".

WHY NOT PATCH `compiler.py`. It works (`patch_reinject.py` does exactly that and is kept
for when you want the pass list itself on disk to read), but it edits an installed file:
it needs write access to site-packages, it does not survive `pip install --force-reinstall`,
it leaks into every other process sharing that environment, and a crash between apply and
revert leaves the tree patched.

WHAT THIS DOES INSTEAD. Wraps `HIPBackend.gluon_to_ttgir` in-process and runs the two
passes as a second pass manager over the module the stock function returns. Nothing on disk
changes, a read-only or system-wide install is fine, and the effect ends with the process.

    import gluon_swp
    with gluon_swp.pipelined(2):            # or gluon_swp.enable(2) / .disable()
        out = my_gluon_kernel[grid](...)

TWO CONDITIONS THE KERNEL MUST MEET, or this changes nothing at all:

  1. the loop must be a `tl.range(...)`. Gluon has no `range` of its own -- only
     `static_range`, which unrolls -- and a `for` over the builtin lowers to an `scf.for`
     with no `tt.num_stages` for `add_schedule_loops` to read. `tl.range(num_stages=None)`
     inherits the launch value, so `None` is fine; a bare `range` is not.
  2. the loads must be `gl.load`, not `gl.amd.cdna3.buffer_load`. Plain's own `make_ttgir`
     runs the pipeliner at pass #15/#16 and `add_convert_to_buffer_ops` at #28, so plain's
     pipeliner only ever sees `tt.load`. Pass `buffer_ops=True` to restore that conversion
     after the pipeliner and get both.

And on a dot kernel, do NOT hand-write the staging: building the LDS path is what the
pipeliner is for, and an explicit `allocate_shared_memory` + `gl.barrier()` body starves it.
`references/gluon/pipeline-reference.md` has the measured 2x2 and the per-shape numbers.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys

_ORIGINAL = None
_STATE: dict = {}


def _backend():
    from triton.backends.amd import compiler as C
    return C


def capabilities() -> dict:
    """What this build can and cannot do, probed rather than inferred from the version."""
    C = _backend()
    tg = getattr(C.amd.passes, "ttgpuir", None)
    have = {p: bool(tg is not None and hasattr(tg, p)) for p in (
        "add_schedule_loops", "add_pipeline", "add_optimize_dot_operands",
        "add_convert_to_buffer_ops", "add_canonicalize_pointers", "add_warp_pipeline",
        "add_block_pingpong")}
    import inspect
    stock = inspect.getsource(C.HIPBackend.gluon_to_ttgir)
    try:
        import triton
        ver = triton.__version__
    except Exception:  # noqa: BLE001
        ver = "unknown"
    return {
        "triton": ver,
        "passes_in_libtriton": have,
        "gluon_to_ttgir_already_pipelines": ("add_schedule_loops" in stock
                                             and "add_pipeline" in stock),
        "gluon_to_ttgir_calls_warp_pipeline": "add_warp_pipeline" in stock,
        "can_reinject": have["add_schedule_loops"] and have["add_pipeline"],
        "can_restore_buffer_ops": have["add_convert_to_buffer_ops"],
        "installed": _ORIGINAL is not None,
    }


def enable(num_stages: int, *, buffer_ops: bool = False, pingpong: bool | None = None,
           async_copy: bool | None = None, verbose: bool = False) -> None:
    """Install the wrapper. `num_stages` is what add_schedule_loops is given.

    `pingpong` / `async_copy` default to whatever this backend decides for the arch, which
    is what plain would have used; pass False to hold one fixed while measuring.
    """
    global _ORIGINAL
    if num_stages is not None and num_stages < 2:
        raise ValueError("num_stages must be >= 2; below that the pipeliner is a no-op "
                         "and installing the wrapper would only add confusion")
    caps = capabilities()
    if not caps["can_reinject"]:
        raise RuntimeError(
            "this build has no add_schedule_loops/add_pipeline in libtriton: "
            + json.dumps(caps["passes_in_libtriton"]))
    if caps["gluon_to_ttgir_already_pipelines"]:
        raise RuntimeError(
            "gluon_to_ttgir ALREADY calls the pipeliner in this tree -- you are on a fork "
            "that has it spliced in. Use its own knob; wrapping would run it twice.")
    if buffer_ops and not caps["can_restore_buffer_ops"]:
        raise RuntimeError("this build has no add_convert_to_buffer_ops")

    C = _backend()
    if _ORIGINAL is None:
        _ORIGINAL = C.HIPBackend.gluon_to_ttgir
    _STATE.update(num_stages=num_stages, buffer_ops=buffer_ops, pingpong=pingpong,
                  async_copy=async_copy, verbose=verbose)
    orig = _ORIGINAL

    def wrapped(src, metadata, options):
        mod = orig(src, metadata, options)
        ns = _STATE["num_stages"]
        if not ns or ns < 2:
            return mod
        arch = getattr(options, "arch", None)
        ac = _STATE["async_copy"]
        pp = _STATE["pingpong"]
        if ac is None:
            ac = bool(getattr(C, "is_async_copy_enabled", lambda a: False)(arch))
        if pp is None:
            pp = bool(getattr(C, "is_pingpong_schedule_enabled",
                              lambda a, b: False)(arch, ac))
        pm = C.ir.pass_manager(mod.context)
        pm.enable_debug()
        C.amd.passes.ttgpuir.add_optimize_dot_operands(pm, arch)
        C.amd.passes.ttgpuir.add_schedule_loops(pm, ns)
        C.amd.passes.ttgpuir.add_pipeline(pm, ac, pp)
        if _STATE["buffer_ops"]:
            # plain's ORDER: pipeline first, buffer conversion after. Restoring it is what
            # lets an anchor be written with gl.load (which the pipeliner can see) and still
            # end up on buffer ops (which the memory path wants).
            from triton import knobs as _kn
            C.passes.common.add_canonicalizer(pm)
            C.amd.passes.ttgpuir.add_canonicalize_pointers(pm)
            C.passes.common.add_canonicalizer(pm)
            C.amd.passes.ttgpuir.add_convert_to_buffer_ops(
                pm, arch, _kn.amd.use_buffer_atomics,
                _kn.amd.buffer_ops_analyze_small_tensor_range)
        pm.run(mod, "gluon_swp_reinject")
        if _STATE["verbose"]:
            print(f"[gluon_swp] ns={ns} async_copy={ac} pingpong={pp} "
                  f"buffer_ops={_STATE['buffer_ops']} arch={arch}", file=sys.stderr)
        return mod

    C.HIPBackend.gluon_to_ttgir = staticmethod(wrapped)


def disable() -> None:
    global _ORIGINAL
    if _ORIGINAL is None:
        return
    _backend().HIPBackend.gluon_to_ttgir = _ORIGINAL
    _ORIGINAL = None
    _STATE.clear()


@contextlib.contextmanager
def pipelined(num_stages: int, **kw):
    """Scoped form. Compile INSIDE the block -- Triton caches, so a kernel already compiled
    without the wrapper will be served from cache and silently not pipelined. Use a distinct
    `triton.knobs.cache.dir` per arm."""
    enable(num_stages, **kw)
    try:
        yield
    finally:
        disable()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        fails = []

        def ck(name, cond, detail=""):
            print(f"  {'ok  ' if cond else 'FAIL'} {name}"
                  + (f"  -- {detail}" if detail and not cond else ""))
            if not cond:
                fails.append(name)

        print("gluon_swp selftest")
        try:
            caps = capabilities()
        except Exception as e:  # noqa: BLE001
            print(f"  skip: no AMD backend importable ({type(e).__name__})")
            raise SystemExit(0)
        print("  caps:", json.dumps(caps))
        ck("the two passes are present in libtriton", caps["can_reinject"],
           json.dumps(caps["passes_in_libtriton"]))
        ck("stock gluon_to_ttgir does NOT already pipeline",
           not caps["gluon_to_ttgir_already_pipelines"],
           "this is a fork with the splice already in -- use its own knob")
        # install / uninstall must restore the exact original object
        C = _backend()
        before = C.HIPBackend.gluon_to_ttgir
        enable(2)
        ck("enable() replaces gluon_to_ttgir", C.HIPBackend.gluon_to_ttgir is not before)
        disable()
        ck("disable() restores the ORIGINAL object, not a copy",
           C.HIPBackend.gluon_to_ttgir is before)
        with pipelined(2):
            ck("pipelined() installs inside the block",
               C.HIPBackend.gluon_to_ttgir is not before)
        ck("pipelined() restores on exit", C.HIPBackend.gluon_to_ttgir is before)
        try:
            enable(1)
            ck("enable(1) is refused rather than silently doing nothing", False)
        except ValueError:
            ck("enable(1) is refused rather than silently doing nothing", True)
        finally:
            disable()
        # the fork-only env knobs the pack used to name are inert here, and saying so is
        # the point of this check existing
        inert = [v for v in ("TRITON_GLUON_SWP_PIPELINE", "TRITON_GLUON_COOP_LDS",
                             "TRITON_GLUON_PINGPONG")
                 if v not in os.environ]
        ck("fork-only knobs are not required by this module", len(inert) == 3,
           "they are read by a vendor fork only; this module does not consult them")
        print(f"SELFTEST {'PASS' if not fails else 'FAIL'}"
              + (f" ({len(fails)} failed: {', '.join(fails)})" if fails else ""))
        raise SystemExit(1 if fails else 0)
    print(json.dumps(capabilities(), indent=2))
