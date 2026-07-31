---
key: cuda/HIP-graph integration · any gfx · sglang/vllm decode
type: method
confidence: ★★★
effect: the #1 e2e-integration killer — a kernel can win isolated yet never run live (or net ~0 e2e), OR crash the server
confirms: 5
last_seen: 2026-06-27
---
# Make an optimized kernel survive CUDA/HIP-graph capture (or the win vanishes e2e)
- lever: sglang/vLLM capture the decode path into a graph. A kernel that JITs, syncs the host, or
  self-captures inside the captured region FALLS BACK TO EAGER → only static changes survive → ~0 e2e
  even at large isolated speedup (observed: 1.22–2.7× iso → ~0 / no live forwards).
- **harder failure (ANY just-in-time-compiled kernel: a DSL like flydsl, an autotuned/authored Triton
  kernel, a CK-JIT instance, …) — lazy compile in TP>1 multiproc serving can CRASH the server, not just
  fall back to eager.** Any code path that JIT-compiles a *new* config (a tile/shape variant not yet built)
  during the live warmup forward can fail the on-device module load (e.g. ROCm `hipModuleLoadData ->
  hipErrorNoBinaryForGpu`; CUDA equivalent `CUDA_ERROR_NO_BINARY_FOR_GPU`) when no valid code object exists
  for that config in the worker → poisons the device context → worker dies → server never healthy → ZERO
  bench samples → gate=rejected `cuda_graph_capture_unsafe`. The single-process isolated unittest compiles
  & loads FINE, so iso passes (large speedup) and ONLY the e2e serving gate catches it.
- apply: author the STEADY-STATE call (2nd call onward) with ZERO host syncs and ZERO compiles:
  · precompile/register the kernel for **EVERY (shape-bucket × config) the LIVE workload actually hits —
    PREFILL buckets AND decode buckets, every per-bucket config the kernel selects, not decode-only** — at
    WARMUP before capture, via an `*_overlay_precompile(weights, scales, buckets)` hook the integrator
    calls once, pre-capture. This populates the on-disk code-object cache so ALL TP workers load a prebuilt
    binary instead of each racing a lazy compile (the thing that produces NO_BINARY_FOR_GPU under TP>1).
    Rule of thumb: anything the kernel can compile at runtime must be compiled at warmup for the full set
    of live shapes/configs — never lazily on the hot or warmup-forward path.
  · key any weight cache by `weight.data_ptr()` (pure host int, weights persistent) — NEVER a
    `w_scale.sum().item()` fingerprint (a host sync that deadlocks capture).
  · no `.item()/.cpu()/.tolist()/synchronize()`/Python-if-on-GPU-scalar on the hot path.
- verify: the loose-tol unittest oracle will NOT catch a capture hang — only the e2e gate does. Confirm
  the optimized kernel actually launches INSIDE the graph (see [[method-verify-engagement]]), and that
  the candidate fits the SAME mem-fraction as the accepted config (a bf16 weight re-materialization can
  balloon the cache to tens of GB → KV-pool starved → e2e −9% even at +24% GEMM).
- source: exp/e2e_*MiniMax-M3-MXFP8*/ (FULL_AND_PIECEWISE) + exp/e2e_*Qwen3.5-27B-FP8*/ flydsl capture runs
- source: 2026-06-27 grouped-GEMM host-control-flow capture crash (MiniMax-M3-MXFP8, e2e_...T015152Z) —
  distinct sub-mode of the same killer: authored FlyDSL MXFP8 grouped-GEMM is HOST-DRIVEN
  (`num_tokens_post_padded.item()`, `eids.tolist()/sorted_ids.tolist()`, a Python for-loop over experts,
  `w_scale.sum().item()`) → every CAND TP worker crashes at vLLM `capture_model`→`_dummy_run` with
  `hipErrorStreamCaptureUnsupported` (NOT NO_BINARY; this is host control flow inside the captured region).
  Rebind/engagement was correct (4 [flydsl-overlay] ENGAGED banners in eager warmup) and iso was 4.011×
  (director-verified) yet server NEVER healthy, served 0 requests → gate=dead_end `cuda_graph_capture_unsafe`
  over 3 independent launches. Stock `_grouped_gemm_mxfp8` does on-DEVICE dispatch (`tl.load(num_tokens_
  post_padded_ptr)`), so a seam wrapper cannot remove the host control flow — the authored kernel needs a
  kernel-layer rewrite to on-device dispatch to run on the captured decode path. → confirms the on-device-
  dispatch / no-host-sync rule; install never mutated, overlay-only.
