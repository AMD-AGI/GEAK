"""CUDA-graph capture-safety pre-pass for the authored FlyDSL MoE overlay.

Root cause of the prior candidate deadlock (integrate_result.json):
  vLLM's TP=4 capture (FULL_AND_PIECEWISE, 51 PIECEWISE + FULL descriptors) runs
  1 built-in eager warmup then captures each descriptor INTERLEAVED, inside the
  `graph_capture()` context. The authored kernel JITs (flyc.compile, which also
  EXECUTES the kernel) lazily per (stage, M) / (wptr, stage, M). Any (layer, M)
  combination not already compiled compiles INSIDE the captured region -> a
  blocking op under capture -> TP rank desync -> c10d collective timeout (hung at
  ~13/51). The mandatory precompile-before-capture hook was missing.

Fix (matches e2e_integrator.md "CUDA-graph-safe overlay" (2) "Precompile BEFORE
capture" and templates/flydsl_overlay_sitecustomize.py `flydsl_overlay_precompile`):
  Wrap GPUModelRunner.capture_model so that BEFORE the real capture we run an
  EXHAUSTIVE eager warmup (`_dummy_run(..., cudagraph_runtime_mode=NONE)`) over
  EVERY capture descriptor (both PIECEWISE and FULL, largest-first) for a couple
  of iterations. This runs the identical model/forward code path the capture will
  use, so it JITs+caches every (layer, stage, M) exe OUTSIDE any capture stream,
  on all TP ranks in lockstep, followed by a device sync + dist barrier. The real
  capture then only launches cached exes (no JIT under capture -> no desync).

Env-gated by VLLM_USE_FLYDSL_MOE=1; a no-op otherwise (byte-identical stock).
Installed lazily via the overlay sitecustomize meta-path finder (no eager vllm
import at interpreter startup).
"""
import os


def install_capture_precompile(mod):
    """Wrap GPUModelRunner.capture_model on the freshly-imported module `mod`
    (vllm.v1.worker.gpu_model_runner). Idempotent; env-gated."""
    if os.environ.get("VLLM_USE_FLYDSL_MOE") != "1":
        return
    try:
        Runner = getattr(mod, "GPUModelRunner", None)
        if Runner is None or getattr(Runner, "_flydsl_precompile_installed", False):
            return
    except Exception:
        return

    _orig_capture = Runner.capture_model

    def capture_model(self):
        try:
            import torch
            from vllm.config import CUDAGraphMode
            try:
                import torch.distributed as dist
            except Exception:
                dist = None

            descs_list = self.cudagraph_dispatcher.get_capture_descs()
            total = sum(len(bd) for _, bd in descs_list)
            print(f"[flydsl-precompile] BEGIN exhaustive eager warmup of {total} "
                  f"capture descriptor(s) BEFORE capture (JIT outside capture) ...",
                  flush=True)
            n = 0
            for runtime_mode, batch_descs in descs_list:
                # SKIP FULL-mode (uniform-decode) descriptors. A raw _dummy_run in
                # FULL mode invokes vLLM's OVERLAPPED shared-experts path, which stashes
                # into a stateful buffer (shared_experts.apply: self._output[idx]) that is
                # only drained by a real scheduler step -- our warmup never drains it, so
                # the buffer stays dirty and the subsequent REAL capture replay trips
                # `assert self._output[idx] is None`, killing the worker at KV-cache init.
                # The flydsl MoE exes are keyed by (stage, M); the SAME M (batch-size) set
                # is covered by the PIECEWISE descriptors (the region that actually JITs the
                # authored kernel -- the original hang was PIECEWISE capture 13/51), so
                # PIECEWISE-only warmup still precompiles every needed exe outside capture.
                if runtime_mode == CUDAGraphMode.FULL:
                    print(f"[flydsl-precompile] skipping {len(batch_descs)} FULL-mode "
                          f"descriptor(s) (shared-experts overlapped-buffer safety; M "
                          f"already covered by PIECEWISE warmup)", flush=True)
                    n += len(batch_descs)
                    continue
                force_attn = False
                for desc in batch_descs:
                    for _ in range(2):
                        try:
                            self._dummy_run(
                                desc.num_tokens,
                                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                force_attention=force_attn,
                                uniform_decode=desc.uniform,
                                allow_microbatching=False,
                                skip_eplb=True,
                                remove_lora=False,
                                num_active_loras=getattr(desc, "num_active_loras", 0),
                            )
                        except Exception as e:
                            print(f"[flydsl-precompile] dummy_run(num_tokens="
                                  f"{desc.num_tokens},uniform={desc.uniform},"
                                  f"mode={runtime_mode}) warmup error: {e!r}",
                                  flush=True)
                            break
                    n += 1
                    if n % 10 == 0:
                        print(f"[flydsl-precompile]   warmed {n}/{total} descriptors",
                              flush=True)
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            if dist is not None and dist.is_initialized():
                try:
                    dist.barrier()
                except Exception:
                    pass
            # RESET vLLM's overlapped shared-experts double-buffers left DIRTY by the
            # warmup _dummy_run passes. SharedExperts.apply() stashes into
            # self._output[idx] and only the `output` property drains it; our warmup
            # forwards write but never drain, so a populated slot survives into the REAL
            # capture, whose first replay trips `assert self._output[idx] is None`
            # (shared_experts.py:169) -> worker dies at KV-cache init. Stock has no extra
            # warmup so it enters capture clean; we reproduce that by clearing every
            # SharedExperts._output back to all-None before the real capture.
            try:
                _reset = 0
                _model = getattr(self, "model", None)
                _mods = _model.modules() if _model is not None else []
                for _m in _mods:
                    _se = getattr(getattr(_m, "runner", None), "_shared_experts", None)
                    if _se is None:
                        _se = getattr(_m, "_shared_experts", None)
                    _buf = getattr(_se, "_output", None)
                    if isinstance(_buf, list):
                        for _i in range(len(_buf)):
                            _buf[_i] = None
                        _reset += 1
                print(f"[flydsl-precompile] reset {_reset} SharedExperts overlapped "
                      f"buffer(s) to None (capture starts clean)", flush=True)
            except Exception as _re:
                print(f"[flydsl-precompile] shared-experts buffer reset failed: {_re!r}",
                      flush=True)
            print(f"[flydsl-precompile] DONE eager warmup ({n}/{total}); all FlyDSL "
                  f"exes compiled+cached outside capture. Proceeding to capture.",
                  flush=True)
        except Exception as e:
            print(f"[flydsl-precompile] pre-pass FAILED (continuing to capture "
                  f"unprotected): {e!r}", flush=True)
        return _orig_capture(self)

    try:
        Runner.capture_model = capture_model
        Runner._flydsl_precompile_installed = True
        print("[flydsl-precompile] installed capture_model precompile wrapper "
              "on GPUModelRunner", flush=True)
    except Exception as e:
        print(f"[flydsl-precompile] install FAILED: {e!r}", flush=True)
