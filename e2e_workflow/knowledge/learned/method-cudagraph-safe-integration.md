---
key: cuda/HIP-graph integration · any gfx · sglang/vllm decode
type: method
confidence: ★★★
effect: the #1 e2e-integration killer — a kernel can win isolated yet never run live (net ~0 e2e from a capture hang, OR a HARD HIP-OOM at engine init → candidate REJECTED)
confirms: 5
last_seen: 2026-06-24
---
# Make an optimized kernel survive CUDA/HIP-graph capture (or the win vanishes e2e)
- lever: sglang/vLLM capture the decode path into a graph. A kernel that JITs, syncs the host, or
  self-captures inside the captured region FALLS BACK TO EAGER → only static changes survive → ~0 e2e
  even at large isolated speedup (observed: 1.22–2.7× iso → ~0 / no live forwards).
- apply: author the STEADY-STATE call (2nd call onward) with ZERO host syncs and ZERO compiles:
  · precompile/register the kernel for ALL decode M-buckets at WARMUP, before capture (an
    `*_overlay_precompile(weight, weight_scale, m_buckets)` hook the integrator calls once, pre-capture).
  · key any weight cache by `weight.data_ptr()` (pure host int, weights persistent) — NEVER a
    `w_scale.sum().item()` fingerprint (a host sync that deadlocks capture) NOR the activation ptr
    `A.data_ptr()` (reallocated every forward → cache misses every call → the conversion re-runs per forward).
  · no `.item()/.cpu()/.tolist()/synchronize()`/Python-if-on-GPU-scalar on the hot path.
  · if the layout conversion is large/destructive, do it ONCE at LOAD time, in place, SAME-BYTE
    (overwrite `layer.w*_weight_packed.data`, chunked over experts) in `process_weights_after_loading`
    — never re-materialize the unpacked `[E,N,K]` int16/bf16 weight per forward (E=384,K=7168 ≈ 5+ GiB).
    Then DROP `--enforce-eager` so vLLM HIP-graph-captures the decode forward and the win surfaces.
- verify: the loose-tol unittest oracle will NOT catch a capture hang — only the e2e gate does. Confirm
  the optimized kernel actually launches INSIDE the graph (see [[method-verify-engagement]]), and that
  the candidate fits the SAME mem-fraction as the accepted config (a per-forward weight
  re-materialization balloons the cache → at worst a HARD HIP-OOM during `determine_available_memory`
  → "Engine core initialization failed" → REJECTED; milder case: KV-pool starved → e2e −9% at +24% GEMM).
- source: exp/e2e_*MiniMax-M3-MXFP8*/ (FULL_AND_PIECEWISE) + exp/e2e_*Qwen3.5-27B-FP8*/ flydsl capture runs
- source: int4-W4A16 FlyDSL apply OOM (exp/e2e_*Kimi-K2.6*155338*/OOM_ROOT_CAUSE_REPORT.md — `A.data_ptr()`
  per-forward re-unpack → HIP OOM → rejected); validated FIX (load-time in-place same-byte conversion + drop
  `--enforce-eager`): FlyDSL +32–34% / AVO +24–30% e2e, GSM8K parity, 0 fallbacks — companion skill apply-flydsl-moe-to-vllm.
