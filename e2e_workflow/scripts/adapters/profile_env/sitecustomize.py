"""Auto-import shim that keeps GPU activity in a spawned engine's torch-profiler trace.

Observed on ROCm 10.0.0 (pip SDK), torch 2.12.0+rocm10.0.0 and vLLM 0.27.1 on gfx1201:
the parent process has ``ROCPROFILER_REGISTER_LIBRARY`` set once it has initialized the
GPU, the EngineCore child spawned afterwards inherits it, and that child's torch profiler
records no ``kernel`` or ``cuda_runtime`` events. It was not reproduced on ROCm 7.2 with
torch 2.9.1, whose profiler uses the ROCTracer backend. The variable must be gone before
torch loads the HIP runtime, so it is dropped at interpreter start-up; elsewhere this is
a no-op.
"""

from __future__ import annotations

import os

os.environ.pop("ROCPROFILER_REGISTER_LIBRARY", None)


def _chain_following_sitecustomize() -> None:
    """Import the next ``sitecustomize`` after this one on ``sys.path``, if there is one.

    Only entries after this directory are searched, and a second run in the same
    process chains nothing, so a hook that chains back to this one cannot loop.
    """
    import importlib.util
    import sys

    if getattr(sys, "_geak_profile_env_chained", False):
        return
    sys._geak_profile_env_chained = True
    here = os.path.dirname(os.path.abspath(__file__))
    seen_here = False
    for entry in sys.path:
        try:
            resolved = os.path.abspath(entry or ".")
        except (OSError, ValueError):
            continue
        if resolved == here:
            seen_here = True
            continue
        if not seen_here:
            continue
        candidate = os.path.join(resolved, "sitecustomize.py")
        if not os.path.isfile(candidate):
            continue
        spec = importlib.util.spec_from_file_location("_geak_following_sitecustomize", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return


# An overlay hook that raises reaches site.execsitecustomize, which reports it, as it would without this shim.
_chain_following_sitecustomize()
