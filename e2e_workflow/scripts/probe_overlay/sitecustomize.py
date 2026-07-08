"""Overlay entry point: registers the per-shape probe at interpreter startup.

Placed on OVERLAY_PYTHONPATH so Python auto-imports it before the server imports anything (this is
why the function-local `from triton_kernels.matmul_ogs import matmul_ogs` in vLLM binds to the
wrapped version — see plan §6.2 constraint 1). Runs in every process that starts with this overlay
on the path (APIServer, EngineCore, compile workers); each records/flushes independently by pid.

Config via env:
  PROBE_OUT       (required) output dir for probe_<pid>_<target>.json
  PROBE_TARGETS   (optional) override the default target list, comma-separated module:attr
"""
import os
import sys

_OUT = os.environ.get("PROBE_OUT")
if _OUT:
    try:
        import capture_shapes_probe as P  # same dir as this file, on OVERLAY_PYTHONPATH

        # Only plain-Python callables can be wrapped by a普通 function wrapper. #5 pack_bitmatrix is a
        # triton @jit JITFunction (called as pack_bitmatrix[grid](...)) — a普通 wrapper breaks the
        # [grid] launch syntax (TypeError: 'function' object is not subscriptable). It is only 1.0%
        # GPU (Amdahl-negligible, same tier as the skipped #4), so it is dropped rather than given a
        # triton-specific grid wrapper. Core targets #1/#2 (83% GPU) and #3 are plain functions.
        _DEFAULT_TARGETS = [
            "triton_kernels.matmul_ogs:matmul_ogs",                  # #1/#2 MoE GEMM (83% GPU)
            "aiter.ops.triton.unified_attention:unified_attention",  # #3 attention (5.4% GPU)
        ]
        env_targets = os.environ.get("PROBE_TARGETS", "")
        targets = ([t.strip() for t in env_targets.split(",") if t.strip()]
                   if env_targets else _DEFAULT_TARGETS)

        # Modules may not be importable yet in EVERY process (e.g. APIServer never imports
        # triton_kernels). install() import-errors are caught per-target so one miss doesn't abort
        # the others; the target simply won't be hooked in that process.
        for t in targets:
            try:
                P.install(t, _OUT)
            except Exception as e:
                sys.stderr.write(f"[probe.overlay] skip {t} in pid {os.getpid()}: {e}\n")
    except Exception as e:
        sys.stderr.write(f"[probe.overlay] init failed: {e}\n")
else:
    sys.stderr.write("[probe.overlay] PROBE_OUT not set; probe disabled\n")
