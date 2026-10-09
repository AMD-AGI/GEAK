#!/usr/bin/env python3
"""Sentinel v3: a marker that SURVIVES torch.compile.

v2 used `torch.profiler.record_function` and found nothing for
`UnquantizedLinearMethod.apply`. That result was NOT interpretable: dynamo drops
record_function from a compiled graph, so an absent marker cannot distinguish
"the patch never ran" from "the patch ran, was inlined, and the marker was elided".
An instrument whose negative result is ambiguous is not an instrument.

A registered custom op is opaque to dynamo -- it is preserved as a call in the graph
rather than traced through -- so it survives compilation and shows up in the trace under
its own name. This registers `geak::sentinel_mark`, a no-op on a 1-element tensor, and
calls it from inside each patched seam.

Reading the result:
  `geak::sentinel_mark` present  -> the patched code IS on the executed path (compiled or not)
  absent                          -> it is not, and now that IS conclusive
"""
import os
import sys

_WRAPPED = []
_REGISTERED = [False]


def _register_marker():
    if _REGISTERED[0]:
        return True
    import torch

    def sentinel_mark(t: torch.Tensor) -> None:
        return None

    def sentinel_mark_fake(t: torch.Tensor) -> None:
        return None

    try:
        lib = torch.library.Library("geak", "FRAGMENT")
        lib.define("sentinel_mark(Tensor t) -> ()")
        lib.impl("sentinel_mark", sentinel_mark, "CUDA")
        lib.impl("sentinel_mark", sentinel_mark, "CPU")
        torch.library.register_fake("geak::sentinel_mark", sentinel_mark_fake)
        _REGISTERED[0] = True
        globals()["_LIB"] = lib          # keep it alive; a GC'd Library deregisters
        return True
    except Exception as exc:
        sys.stderr.write("[overlay-sentinel3] marker registration failed: %r\n" % (exc,))
        return False


def _wrap_attr(owner, attr, tag):
    fn = getattr(owner, attr, None)
    if fn is None or getattr(fn, "_geak_sentinel", False):
        return False
    import torch

    def wrapped(*a, __fn=fn, **kw):
        out = __fn(*a, **kw)
        try:
            probe = out if isinstance(out, torch.Tensor) else None
            if probe is None:
                for v in a:
                    if isinstance(v, torch.Tensor):
                        probe = v
                        break
            if probe is not None:
                torch.ops.geak.sentinel_mark(probe)
        except Exception:
            pass
        return out

    wrapped._geak_sentinel = True
    setattr(owner, attr, wrapped)
    _WRAPPED.append(tag)
    return True


def install():
    if os.environ.get("GEAK_SENTINEL", "0") not in ("1", "true", "True"):
        return False
    if not _register_marker():
        return False
    try:
        from vllm.model_executor.layers import linear as _l
    except Exception as exc:
        sys.stderr.write("[overlay-sentinel3] import failed: %r\n" % (exc,))
        return False
    cls = getattr(_l, "UnquantizedLinearMethod", None)
    if cls is not None:
        _wrap_attr(cls, "apply", "METHOD_APPLY")
    sys.stderr.write("[overlay-sentinel3] ENGAGED wrapped=%s\n" % (_WRAPPED or "NONE",))
    return True
