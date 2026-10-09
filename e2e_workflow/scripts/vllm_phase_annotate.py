#!/usr/bin/env python3
"""Emit sglang-dialect `step[...]` phase annotations from a live vLLM v1 server.

WHY THIS EXISTS
---------------
The KernelFusion semantics layer (`semantic_kernel_mapping.py`) attributes every device
event to a serving phase by locating it inside a **step span** on the GPU timeline.  It
recognizes exactly two dialects:

    step[EXTEND bs=8 toks=4096]        <- sglang's profiler annotation
    step[DECODE bs=64]
    execute_context_1024_generation_64 <- vLLM's `detailed_trace_annotation` (legacy)

Current vLLM emits NEITHER: `detailed_trace_annotation` aborts the server on builds whose
`ProfilerConfig` is a strict pydantic schema (see `adapters/vllm.sh`), and vLLM's own
`gpu_model_runner: forward` ranges carry no phase / batch-size / token-count. With no step
spans `_collect_step_spans()` returns empty, `_phase_at()` returns None for every launch,
NO phase-tagged table is built, and the whole fusion chain (candidates -> topk -> unitside
-> apply-back) loses its denominator.

So instead of teaching the parser a third dialect, we make vLLM SPEAK the first one: wrap
`GPUModelRunner.execute_model` in a `torch.profiler.record_function` whose name is
byte-identical to what sglang emits.  Everything downstream — the CPU/GPU dual-window
pairing, layer-boundary resolution, conservation checks — is reused verbatim.

`record_function` produces a `user_annotation` CPU event and, for the kernels launched
inside it, a correlated `gpu_user_annotation` device event.  That is the exact category
pair `_collect_step_spans()` reads.

CONTRACT
--------
- Inert unless `GEAK_VLLM_PHASE_ANNOTATE=1`. An unarmed overlay must not change a serving run.
- Never raises into the serving path. If the phase cannot be determined the original method
  is called with NO annotation — a MISSING span degrades to "unresolved", a WRONG span
  silently corrupts every fusion candidate built on it.
- Idempotent: re-installing on an already-wrapped class is a no-op.
- stderr only. stdout is parsed by the launcher; polluting it breaks the JIT banner scrape.
"""
import os
import sys

_INSTALLED = set()
_WARNED = set()


def _armed():
    return os.environ.get("GEAK_VLLM_PHASE_ANNOTATE", "0") in ("1", "true", "True")


def _warn_once(key, message):
    if key not in _WARNED:
        _WARNED.add(key)
        sys.stderr.write("[GEAK_PHASE_ANNOTATE] %s\n" % message)


def classify_step(runner, scheduler_output):
    """(phase, batch_size, input_tokens) for this forward, or None if undeterminable.

    THE single place vLLM's prefill/decode split is decided. Both consumers read it here:
    the trace annotation below, and the eager shape probe (vllm_semantic_capture). A second
    copy of this rule would be free to drift, and a phase split that disagrees with itself
    across two capture paths is not detectable from either one's output.

    `phase` uses sglang's vocabulary (EXTEND / DECODE) so downstream canonicalization is
    shared too. None is a deliberate outcome, not a failure path: an unlabelled step is
    recorded as phase-unresolved, which is recoverable. A mislabelled step is not.
    """
    per_req = getattr(scheduler_output, "num_scheduled_tokens", None)
    total = getattr(scheduler_output, "total_num_scheduled_tokens", None)
    if not per_req or not total:
        return None                       # dummy / empty batch: nothing to attribute
    num_reqs = len(per_req)
    max_toks = max(per_req.values())

    # `uniform_decode_query_len` is 1 + num_spec_tokens: with speculative decoding a decode
    # step schedules >1 token per request and a `max_toks == 1` test would call it prefill.
    query_len = getattr(runner, "uniform_decode_query_len", 1) or 1

    # Prefer vLLM's own predicate so we stay consistent with how IT classifies the batch
    # (that is also what selects the cudagraph). Fall back to the same arithmetic inline —
    # the helper is a private staticmethod and has moved across releases.
    is_decode = None
    probe = getattr(runner, "_is_uniform_decode", None)
    if probe is not None:
        try:
            is_decode = bool(probe(max_toks, query_len, total, num_reqs))
        except Exception:
            is_decode = None
    if is_decode is None:
        is_decode = (max_toks == query_len) and (total == max_toks * num_reqs)

    return ("DECODE" if is_decode else "EXTEND", num_reqs, total)


def _step_name(runner, scheduler_output):
    """Render classify_step() in sglang's annotation dialect, or None."""
    classified = classify_step(runner, scheduler_output)
    if classified is None:
        return None
    phase, num_reqs, total = classified
    if phase == "DECODE":
        return "step[DECODE bs=%d]" % num_reqs
    return "step[EXTEND bs=%d toks=%d]" % (num_reqs, total)


def install():
    """Wrap GPUModelRunner.execute_model. Called lazily AFTER the module is imported.

    Deliberately does NOT import vllm itself: an eager `import vllm...` from a
    sitecustomize runs before the engine has set up its distributed environment and
    hangs every TP rank. The overlay's post-import hook hands us the already-imported
    module instead.
    """
    if not _armed():
        return False
    try:
        import torch
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as exc:
        _warn_once("import", "not installed (%r) -- traces will have NO step spans" % (exc,))
        return False

    if GPUModelRunner in _INSTALLED:
        return True
    original = GPUModelRunner.execute_model

    def execute_model(self, scheduler_output, *args, **kwargs):
        name = None
        try:
            name = _step_name(self, scheduler_output)
        except Exception as exc:
            _warn_once("classify", "phase classification failed (%r); steps unannotated" % (exc,))
        if name is None:
            return original(self, scheduler_output, *args, **kwargs)
        with torch.profiler.record_function(name):
            return original(self, scheduler_output, *args, **kwargs)

    GPUModelRunner.execute_model = execute_model
    _INSTALLED.add(GPUModelRunner)
    sys.stderr.write(
        "[GEAK_PHASE_ANNOTATE] ENGAGED: GPUModelRunner.execute_model emits "
        "step[EXTEND bs=.. toks=..] / step[DECODE bs=..]\n")
    return True
