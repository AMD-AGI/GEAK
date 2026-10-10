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

SPECULATIVE DECODING (MTP / EAGLE)
----------------------------------
vLLM's own `execute_*` annotation wraps only `execute_model`, the target model's forward.
The draft model's proposals and the rejection sampler run afterwards, in
`GPUModelRunner.sample_tokens` (both the V1 runner, `drafter`, and the V2 runner,
`speculator`, which vLLM picks for MiniMax-M3 + EAGLE3), outside every step span, so the semantics layer could only
count them as out of scope (MiniMax-M3 TP8 + EAGLE3: 415 ms, 5% of the window). With
`GEAK_VLLM_DRAFT_ANNOTATE=1`, and only when the runner has a drafter, `sample_tokens` is
wrapped in `step[DRAFT_EXTEND bs=N]` -- sglang's draft dialect, which `sglang_step_modes`
already maps to the draft phase: kept off the main-layer tables and reported on its own. The
span also holds the sampler, which runs in the same call.

CONTRACT
--------
- Inert unless `GEAK_VLLM_PHASE_ANNOTATE=1` (target steps) or `GEAK_VLLM_DRAFT_ANNOTATE=1`
  (draft steps). An unarmed overlay must not change a serving run.
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


def _draft_armed():
    return os.environ.get("GEAK_VLLM_DRAFT_ANNOTATE", "0") in ("1", "true", "True")


# Runner modules whose GPUModelRunner the hooks wrap: V1, and the V2 runner vLLM selects for
# some configs (MiniMax-M3 + EAGLE3 logs "Using V2 Model Runner"). Each is wrapped only once
# its module is already imported -- the overlay registers a post-import hook on both.
RUNNER_MODULES = ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner")


def draft_step_name(runner):
    """`step[DRAFT_EXTEND bs=N]` for a sample_tokens call that will run a drafter, else None.

    V1 keeps its proposer in `drafter` and the batch in `input_batch`; V2 keeps it in
    `speculator` and the step's batch in `execute_model_state.input_batch`.
    """
    if (getattr(runner, "drafter", None) is None
            and getattr(runner, "speculator", None) is None):
        return None
    batch = getattr(getattr(runner, "execute_model_state", None), "input_batch", None)
    if batch is None:
        batch = getattr(runner, "input_batch", None)
    num_reqs = getattr(batch, "num_reqs", None)
    if not num_reqs:
        return None
    return "step[DRAFT_EXTEND bs=%d]" % int(num_reqs)


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
    """Wrap GPUModelRunner methods. Called lazily AFTER the module is imported.

    Deliberately does NOT import vllm itself: an eager `import vllm...` from a
    sitecustomize runs before the engine has set up its distributed environment and
    hangs every TP rank. The overlay's post-import hook hands us the already-imported
    module instead.
    """
    phase, draft = _armed(), _draft_armed()
    if not (phase or draft):
        return False
    runners = [getattr(sys.modules[name], "GPUModelRunner", None)
               for name in RUNNER_MODULES if name in sys.modules]
    runners = [cls for cls in runners if cls is not None]
    try:
        import torch
    except Exception as exc:
        _warn_once("import", "not installed (%r) -- traces will have NO step spans" % (exc,))
        return False
    if not runners:
        _warn_once("import", "no vLLM GPUModelRunner imported yet -- nothing wrapped")
        return False
    for cls in runners:
        if phase:
            _install_target(torch, cls)
        if draft:
            _install_draft(torch, cls)
    return True


def _install_target(torch, GPUModelRunner):
    if (GPUModelRunner, "execute_model") in _INSTALLED:
        return
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
    _INSTALLED.add((GPUModelRunner, "execute_model"))
    sys.stderr.write(
        "[GEAK_PHASE_ANNOTATE] ENGAGED: %s.GPUModelRunner.execute_model emits "
        "step[EXTEND bs=.. toks=..] / step[DECODE bs=..]\n" % GPUModelRunner.__module__)


def _install_draft(torch, GPUModelRunner):
    if (GPUModelRunner, "sample_tokens") in _INSTALLED:
        return
    original = getattr(GPUModelRunner, "sample_tokens", None)
    if original is None:
        _warn_once("draft", "GPUModelRunner has no sample_tokens; draft steps unannotated")
        return

    def sample_tokens(self, *args, **kwargs):
        name = None
        try:
            name = draft_step_name(self)
        except Exception as exc:
            _warn_once("draft_name", "draft step naming failed (%r); unannotated" % (exc,))
        if name is None:
            return original(self, *args, **kwargs)
        with torch.profiler.record_function(name):
            return original(self, *args, **kwargs)

    GPUModelRunner.sample_tokens = sample_tokens
    _INSTALLED.add((GPUModelRunner, "sample_tokens"))
    sys.stderr.write(
        "[GEAK_PHASE_ANNOTATE] ENGAGED: %s.GPUModelRunner.sample_tokens emits "
        "step[DRAFT_EXTEND bs=..] when a drafter is configured\n" % GPUModelRunner.__module__)
