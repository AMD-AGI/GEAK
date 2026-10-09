#!/usr/bin/env python3
"""Eager shape probe for vLLM: drive the shared GEAK semantics logger from the engine side.

WHY A SEPARATE ENTRY POINT
--------------------------
`semantic_runtime_capture.py` holds the whole capture engine -- the bucketed logger, the
nn.Module marker hooks, the tensor-metadata extraction, the `geak.semantics_runtime.v2`
record format -- and ALL of that is framework-agnostic. Exactly one thing is not: where
the (phase, batch_size, input_tokens) context comes from.

  sglang: the model's own forward receives a `ForwardBatch`, so `install_on_model` wraps
          `Model.forward` and reads `forward_batch.forward_mode`.
  vLLM:   the model forward is `(input_ids, positions, intermediate_tensors, inputs_embeds)`
          -- there is no batch object to read. The batch description lives one level up, in
          the `SchedulerOutput` handed to `GPUModelRunner.execute_model`.

So this module installs the same hooks and drives the same logger, but takes the context
from the runner instead of the model. The phase rule itself is NOT duplicated here: it is
imported from `vllm_phase_annotate.classify_step`, which is also what labels the trace, so
the shape log and the trace annotation cannot disagree about what a decode step is.

WHY IT IS NEEDED AT ALL
-----------------------
Decode replays under a CUDA graph. The whole layer stack becomes one graph launch, the CPU
never walks the module tree, and the clean production trace therefore carries decode's
kernel SEQUENCE but none of its SHAPES (`decode_evidence = sequence_only_shapes_unresolved`,
`decode_requires_graph_capture = true`). Candidate DISCOVERY can live with that; candidate
GENERATION and the unit-side microbench cannot -- they need real dims to build tensors from.
This probe supplies them by running the same workload with cuda graphs off.

WHEN TO USE THIS AT ALL
-----------------------
⚠️ On vLLM >= 0.27 you probably should NOT. Two things were learned by running it:

  1. These hooks are INCOMPATIBLE WITH torch.compile. They sit inside the compiled region,
     and dynamo cannot trace `torch.autograd._profiler_enabled()` through a forward
     pre-hook -- `aot_compile_fullgraph` raises during `profile_run` and the engine never
     starts. So this module only works with `--enforce-eager`.
  2. `--enforce-eager` disables torch.compile as well as cuda graphs, so the captured
     shapes describe the UNFUSED graph. Measured against the production decode trace on
     Qwen3.5-2B: 1113 vs 375 kernels per step, 13 kernels present in only one of them,
     sequence similarity 0.063. Those shapes do not belong to the kernels being optimized.

The supported way to get decode shapes on vLLM is instead
`--compilation-config.cudagraph_mode=NONE` on the fusion capture (see
roles/fusion_trace_collector.md): it stops graph REPLAY while keeping compilation, which
restores the per-layer dispatch ops AND `record_shapes` dims at 0.989 sequence similarity
to production -- no hooks, no shape log, no merge.

This module remains for stacks where that does not apply: a build without per-layer
dispatch ops, or a non-compiled runtime where the eager trace IS the production graph
(sglang's case, which is what it was written for).

CONTRACT
--------
- Installed as a lazy post-import hook (`overlay_setup.py add-hook --module
  vllm.v1.worker.gpu_model_runner --impl-module vllm_semantic_capture`), never by editing
  site-packages. The sglang path `docker cp`s a patched `model_runner.py` into the image
  and keeps a `.geak_semantics_bak`; the overlay is reversible by dropping one PYTHONPATH
  entry, which is what the Director's isolation contract actually asks for.
- Inert unless `GEAK_SEMANTICS_CAPTURE=1` (the shared logger's own switch). An unarmed
  overlay must not perturb a measurement run.
- Never raises into the serving path.
- stderr only; stdout is parsed by the launcher.
"""
import os
import sys

_INSTALLED = set()
_WARNED = set()


def _warn_once(key, message):
    if key not in _WARNED:
        _WARNED.add(key)
        sys.stderr.write("[GEAK_SEMANTICS][vllm] %s\n" % message)


def install():
    """Wrap GPUModelRunner.load_model + execute_model. Called AFTER the module imports."""
    try:
        import semantic_runtime_capture as capture
    except Exception as exc:                       # overlay incomplete
        _warn_once("core", "semantic_runtime_capture not importable (%r)" % (exc,))
        return False
    logger = capture.get_logger()
    if not logger.enabled:
        return False                               # GEAK_SEMANTICS_CAPTURE not set: inert
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        # Bind the MODULE, not the symbol: resolving classify_step per call keeps this
        # probe and the trace annotation on literally the same function object, so the
        # two capture paths cannot end up with different ideas of what a decode step is.
        import vllm_phase_annotate
    except Exception as exc:
        _warn_once("import", "not installed (%r) -- no decode shapes will be captured"
                   % (exc,))
        return False

    if GPUModelRunner in _INSTALLED:
        return True

    original_load = GPUModelRunner.load_model
    original_execute = GPUModelRunner.execute_model

    def load_model(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        try:
            capture.install_hooks_only(self.model)
        except Exception as exc:
            # A capture that cannot hook must say so. Silently serving an unhooked model
            # produces an EMPTY shape log, which downstream reads as "this phase has no
            # shapes" -- indistinguishable from a phase that genuinely has none.
            _warn_once("hook", "FAILED to install module hooks (%r); the shape log will "
                               "be empty and decode shapes will stay unresolved" % (exc,))
        return result

    def execute_model(self, scheduler_output, *args, **kwargs):
        classified = None
        try:
            classified = vllm_phase_annotate.classify_step(self, scheduler_output)
        except Exception as exc:
            _warn_once("classify", "phase classification failed (%r)" % (exc,))
        if classified is not None:
            logger.set_context(*classified)
        result = original_execute(self, scheduler_output, *args, **kwargs)
        if classified is not None:
            # AFTER the forward, mirroring the sglang path: the bucket counter gates
            # `_allowed`, so incrementing it before would refuse the very forward it is
            # supposed to admit.
            logger.mark_forward()
        return result

    GPUModelRunner.load_model = load_model
    GPUModelRunner.execute_model = execute_model
    _INSTALLED.add(GPUModelRunner)
    sys.stderr.write(
        "[GEAK_SEMANTICS][vllm] ENGAGED: hooks on GPUModelRunner.load_model, phase "
        "context from execute_model (layers=%s phases=%s forwards/bucket=%s)\n"
        % (sorted(logger.layers) or "all", sorted(logger.phases) or "all",
           logger.max_forwards))
    return True
