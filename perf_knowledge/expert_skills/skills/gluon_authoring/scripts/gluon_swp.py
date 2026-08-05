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

THREE CONDITIONS THE KERNEL MUST MEET, or this changes nothing at all:

  1. the loop must be a pipelining CANDIDATE, which depends on the DOT, not on loop syntax.
     `add_schedule_loops(pm, ns)` takes the depth as a pass OPTION and never reads a
     `tt.num_stages` attribute off the loop -- plain's own `scf.for` carries none and
     pipelines anyway. So a **bare `range` with a `tt.dot` in the body pipelines**; verified by
     dumping the same bare-`range` source armed and unarmed and counting `ttg.memdesc_index`.
     Do NOT rewrite a loop to `tl.range` for this shim's sake -- the annotation only matters
     on the no-dot path, where it is the sole way to request the transform.
  2. the loads must be `gl.load`, not `gl.amd.cdna<N>.buffer_load`. Plain's own `make_ttgir`
     runs the pipeliner before `add_convert_to_buffer_ops`, so plain's pipeliner only ever
     sees `tt.load`; a body already on buffer ops is skipped in silence. `buffer_ops=True`
     restores that conversion after the pipeliner -- but only for a body with NO `amdg.buffer_*`
     left of its own, including in early-exit branches. Converting one that does have them
     fails as `failed to legalize operation 'amdg.buffer_load/store'`, or as a
     `TritonAMDGPUCanonicalizePointers` / `PassManager::run failed` abort. Mixed body:
     `buffer_ops=False`.
  3. the staging must be the pipeliner's own. It multi-buffers an immutable
     `tt.load -> ttg.local_alloc` chain, whereas `allocate_shared_memory` yields a MUTABLE
     `ttg.local_alloc` that it refuses with
     `'ttg.local_alloc' op pipeliner doesn't know how to predicate this op`.
     Read that as "the hand staging is still there", NOT as a language limit -- a transcription
     concluded the latter and had to retract it. Partial removal does not count: one
     `ttg.barrier` left in the loop makes the pass skip it entirely and in silence. Remove the
     allocation, the barriers and the sliced loads together.

De-staging is NOT free, and the injection does not always pay it back: removing hand staging
is a large regression on its own, and the net can stay NEGATIVE on some toolchain versions
even though the injection itself is worth a consistent speedup everywhere. Judge it on the
same-window per-rep ratio of armed-and-de-staged over the original anchor; differencing two
percentages against a noisy shipped baseline cannot resolve an effect this small. A cheap
pre-check tells you whether to attempt it at all: compare the PLAIN kernel's own
`num_stages=1` against its `num_stages=2` -- where plain itself gains nothing from pipelining
on a version, recovering the pipeline for a transcription did not pay off there either.
`references/gluon/pipeline-reference.md` has the measured 2x2; the trial report has the
per-kernel numbers.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys

_ORIGINAL = None
_STATE: dict = {}
_APPLIED: list = []

# `add_pipeline` is NOT the end of what plain does. Plain's make_ttgir keeps going, and two of
# those passes are load-bearing rather than cosmetic:
#   * without `remove_layout_conversions` the pipeliner's `local_load` stays in a blocked layout
#     with a separate `convert_layout` to the dot operand hanging off it, so every operand takes
#     a SECOND trip through LDS on top of the multi-buffer staging. On a tile whose staging
#     already fills the budget the arm then fails to LAUNCH on shared memory -- the injection
#     looks broken when it actually worked.
#   * without `in_thread_transpose` (arch-gated upstream by `is_in_thread_transpose_enabled`)
#     `optimize_dot_operands` picks a degenerate shared layout and a wide LDS read collapses
#     into many narrow ones. It also decides layout FIDELITY: with it, the pipeliner emits the
#     rotating shared layout that Gluon has no constructor for, so an operand staging that is
#     UNRECOVERABLE by hand comes back for free.
# Both were found the hard way, and neither symptom looks like "a pass is missing".
POST_RECIPES = ("none", "minimal", "plain", "plain_pp", "plain_itt")
_DEFAULT_POST = "plain_itt"


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
    # Inspect the ORIGINAL, never whatever is currently installed. Once enable() has swapped
    # in the wrapper, its source contains add_schedule_loops/add_pipeline, so inspecting the
    # live attribute made the tree look like a fork that already splices them -- and a second
    # enable() at a different depth was refused outright. Re-arming to sweep num_stages is
    # exactly what a depth probe does, so that bug would have hit every one of them.
    _o = _ORIGINAL
    if isinstance(_o, staticmethod):
        _o = _o.__func__
    stock = inspect.getsource(_o or C.HIPBackend.gluon_to_ttgir)
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


def cache_tag() -> str:
    """A string encoding the CURRENT arming, for keying your `TRITON_CACHE_DIR` on.

    Per-arm cache dirs are not enough on their own. Triton's cache key -- in process AND on
    disk -- knows nothing about this wrapper, so two arms that differ only by the injection,
    or by its DEPTH, collide on one artefact and the second silently gets the first one's
    code. A trial hit the disk-side version of this while probing num_stages=3: reusing the
    ns=2 arm's directory served the ns=2 binary back and read as "depth does nothing".

        triton.knobs.cache.dir = f"/tmp/run_{arm}_{gluon_swp.cache_tag()}"

    Distinct kernel objects fix the in-process collision; this fixes the on-disk one. Use
    both, and confirm `ttg.memdesc_index` in the arm's own dumped IR.
    """
    if _ORIGINAL is None:
        return "swp_off"
    return (f"swp{_STATE.get('num_stages')}_post{_STATE.get('post')}"
            f"_buf{int(bool(_STATE.get('buffer_ops')))}"
            f"_pp{_STATE.get('pingpong')}_ac{_STATE.get('async_copy')}")


def applied() -> list:
    """Pass names spliced on the last compile. A leading '-' means the pass does not exist on
    this build, so a recipe degrades between Triton minors unless you read this."""
    return list(_APPLIED)


def _post_pipeline_tail(C, pm, recipe, ns, pingpong, arch):
    """Append `recipe`'s post-pipeline passes to `pm`; return the names applied."""
    tg, agt = C.passes.ttgpuir, C.amd.passes.ttgpuir
    done = []

    def opt(owner, name, *args):
        fn = getattr(owner, name, None)
        if fn is None:
            done.append(f"-{name}")   # absent on this build: recorded, never silent
            return
        fn(pm, *args)
        done.append(name)

    if recipe == "none":
        return done
    if recipe == "minimal":
        C.passes.common.add_canonicalizer(pm)
        done.append("canonicalizer")
        opt(tg, "add_remove_layout_conversions")
        return done
    opt(agt, "add_convert_to_tensor_ops")
    C.passes.common.add_canonicalizer(pm)
    done.append("canonicalizer")
    opt(tg, "add_remove_layout_conversions")
    opt(tg, "add_reduce_data_duplication")
    if recipe == "plain_itt":
        # Plain runs this between reduce_data_duplication and move_up_prologue_loads, gated on
        # the arch. Probe the gate rather than assuming: a static namespace read cannot see it.
        gate = getattr(C, "is_in_thread_transpose_enabled", None)
        if gate is None or bool(gate(arch)):
            opt(agt, "add_in_thread_transpose")
            opt(tg, "add_remove_layout_conversions")
        else:
            done.append("-in_thread_transpose(arch-gated off)")
    opt(agt, "add_move_up_prologue_loads")
    if recipe in ("plain_pp", "plain_itt") and pingpong and ns and ns > 1:
        # Can only fire on staging the pipeliner built: it collects `local_load`s whose source
        # is a loop-carried BlockArgument, and hand staging reaches it via `memdesc_index`.
        opt(agt, "add_block_pingpong", ns)
    if recipe == "plain_itt":
        opt(agt, "add_fold_true_cmpi")
        opt(agt, "add_prepare_if_combining")
        C.passes.common.add_canonicalizer(pm)
        done.append("canonicalizer")
        opt(C.passes.common, "add_cse")
        opt(C.passes.common, "add_symbol_dce")
    return done


def enable(num_stages: int, *, buffer_ops: bool = False, pingpong: bool | None = None,
           async_copy: bool | None = None, post: str = _DEFAULT_POST,
           verbose: bool = False) -> None:
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
        # Capture the DESCRIPTOR from the class __dict__, not the resolved attribute.
        # `C.HIPBackend.gluon_to_ttgir` resolves a staticmethod to a plain function, so
        # assigning that back in disable() turned it into an instance method and every
        # subsequent Gluon compile in the process died with "gluon_to_ttgir() takes 3
        # positional arguments but 4 were given". That breaks the one-process interleaved-arms
        # protocol this module exists to serve -- a trial lost three arms to it -- and the old
        # selftest could not see it, because it compared the RESOLVED attribute, which is the
        # same function object whether or not the descriptor survived.
        _ORIGINAL = C.HIPBackend.__dict__["gluon_to_ttgir"]
    if post not in POST_RECIPES:
        raise ValueError(f"post must be one of {POST_RECIPES}, got {post!r}")
    _STATE.update(post=post, num_stages=num_stages, buffer_ops=buffer_ops, pingpong=pingpong,
                  async_copy=async_copy, verbose=verbose)
    orig = _ORIGINAL.__func__ if isinstance(_ORIGINAL, staticmethod) else _ORIGINAL

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
        _APPLIED[:] = _post_pipeline_tail(C, pm, _STATE["post"], ns, pp, arch)
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
            # "requested", not "enabled": add_block_pingpong has its own constraints and
            # refuses far more often than it accepts. A reader took pingpong=True for
            # "ping-pong is on" while s_setprio in the resulting .amdgcn was still 0.
            print(f"[gluon_swp] ns={ns} buffer_ops={_STATE['buffer_ops']} arch={arch} "
                  f"| REQUESTED async_copy={ac} pingpong={pp} -- whether either fired is "
                  f"only readable from the artefacts (s_setprio / async_copy in the ISA/IR), "
                  f"not from this line", file=sys.stderr)
        return mod

    C.HIPBackend.gluon_to_ttgir = staticmethod(wrapped)


def disable() -> None:
    global _ORIGINAL
    if _ORIGINAL is None:
        return
    # Assign the descriptor back verbatim, preserving staticmethod-ness.
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
        before_desc = C.HIPBackend.__dict__["gluon_to_ttgir"]
        before_kind = type(before_desc).__name__
        enable(2)
        ck("enable() replaces gluon_to_ttgir", C.HIPBackend.gluon_to_ttgir is not before)
        disable()
        ck("disable() restores the ORIGINAL object, not a copy",
           C.HIPBackend.gluon_to_ttgir is before)
        # The check that matters, and the one the previous version could not make: the
        # DESCRIPTOR has to come back as a staticmethod. Restoring the resolved function
        # leaves an instance method, and then every later compile fails with an arity error
        # while this comparison still passes.
        ck(f"disable() preserves the descriptor kind ({before_kind})",
           type(C.HIPBackend.__dict__["gluon_to_ttgir"]).__name__ == before_kind,
           f"became {type(C.HIPBackend.__dict__['gluon_to_ttgir']).__name__}")
        # and it must still be CALLABLE with the 3-arg signature after a round trip
        enable(2); disable()
        import inspect as _i
        _sig = _i.signature(C.HIPBackend.gluon_to_ttgir)
        ck("gluon_to_ttgir still takes (src, metadata, options) after enable/disable",
           len(_sig.parameters) == 3, str(_sig))
        with pipelined(2):
            ck("pipelined() installs inside the block",
               C.HIPBackend.gluon_to_ttgir is not before)
        ck("pipelined() restores on exit", C.HIPBackend.gluon_to_ttgir is before)
        # Re-arming at a different depth must work: a depth sweep does exactly that, and it
        # was broken because capabilities() inspected the installed wrapper instead of the
        # original and concluded the tree was an already-spliced fork.
        try:
            enable(2)
            enable(3)
            ck("re-arming at a new depth works (depth sweeps need it)",
               _STATE["num_stages"] == 3, str(_STATE.get("num_stages")))
            tags = set()
            for _ns in (2, 3, 4):
                enable(_ns)
                tags.add(cache_tag())
            ck("cache_tag() distinguishes depths (the on-disk cache key does not)",
               len(tags) == 3, str(sorted(tags)))
        finally:
            disable()
        ck("cache_tag() says off when nothing is installed", cache_tag() == "swp_off")
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

        # --- post-pipeline tail ---
        refused = False
        try:
            enable(2, post="nonsuch")
        except ValueError:
            refused = True
        except Exception:
            pass
        finally:
            disable()
        ck("an unknown post recipe is refused, not silently ignored", refused,
           "a typo'd recipe falling through to no tail would read as 'the tail did nothing'")

        with pipelined(2, post="minimal"):
            t_min = cache_tag()
        with pipelined(2, post=_DEFAULT_POST):
            t_def = cache_tag()
        ck("cache_tag() distinguishes post recipes", t_min != t_def,
           f"{t_min} vs {t_def}: two recipes sharing a cache dir serve each other's code")

        ck("the default recipe is not the tail-free one", _DEFAULT_POST != "none",
           "add_pipeline alone can leave a dot-operand loop over the shared-memory budget")
        ck("the default recipe carries in_thread_transpose", _DEFAULT_POST == "plain_itt",
           "without it a wide LDS read degenerates and the rotating shared layout is lost")
        print(f"SELFTEST {'PASS' if not fails else 'FAIL'}"
              + (f" ({len(fails)} failed: {', '.join(fails)})" if fails else ""))
        raise SystemExit(1 if fails else 0)
    print(json.dumps(capabilities(), indent=2))
