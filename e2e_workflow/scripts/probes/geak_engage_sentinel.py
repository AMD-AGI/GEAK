#!/usr/bin/env python3
"""Numerically-inert sentinel that answers ONE question: did the overlay actually run?

`fusion_integrator.md` proves engagement with a startup `[overlay-...] ENGAGED` banner plus
the fused kernel in a reprofile trace. On vLLM the banner alone can LIE: it only proves the
Python rebind happened, and what the GPU executes is a torch.compile artifact that may have
been captured -- or loaded from ~/.cache/vllm/torch_compile_cache -- without it.

So this wraps two DIFFERENT seam kinds with a uniquely-named record_function and changes
nothing else. Whichever names show up in the trace is the answer:

  custom op impl  : vllm::rocm_unquantized_gemm's implementation. Dynamo treats a custom op
                    as opaque and calls through torch.ops, so a rebind here should survive
                    compilation.
  inlined function: a plain Python callable dynamo traces THROUGH and bakes into the graph.
                    A rebind is only honoured if it happened before the graph was captured
                    -- and a cache hit means it was captured on some earlier run.
"""
import os
import sys

_WRAPPED = []


def _wrap(mod, attr, tag):
    fn = getattr(mod, attr, None)
    if fn is None or getattr(fn, "_geak_sentinel", False):
        return False
    import torch

    def wrapped(*a, __fn=fn, __tag=tag, **kw):
        with torch.profiler.record_function("GEAK_SENTINEL_" + __tag):
            return __fn(*a, **kw)

    wrapped._geak_sentinel = True
    setattr(mod, attr, wrapped)
    _WRAPPED.append(tag)
    return True


def install():
    if os.environ.get("GEAK_SENTINEL", "0") not in ("1", "true", "True"):
        return False
    try:
        from vllm.model_executor.layers import utils as _u
    except Exception as exc:
        sys.stderr.write("[overlay-sentinel] import failed: %r\n" % (exc,))
        return False
    # The seam roles/kernel_extractor.md names for vLLM unquantized bf16/fp16 on ROCm.
    for attr, tag in (("rocm_unquantized_gemm_impl", "CUSTOMOP_IMPL"),
                      ("rocm_unquantized_gemm", "PY_WRAPPER"),
                      ("dispatch_unquantized_gemm", "DISPATCH")):
        _wrap(_u, attr, tag)
    # The banner fusion_integrator.md currently accepts as proof. Printed on purpose even
    # when the rebind will turn out to be inert, because demonstrating that it can be
    # printed and MEAN NOTHING is the point of this probe.
    sys.stderr.write("[overlay-sentinel] ENGAGED wrapped=%s\n" % (_WRAPPED or "NONE",))
    return True
