---
key: cuda/HIP-graph integration · any gfx · sglang/vllm decode
type: method
confidence: ★★★
effect: the #1 e2e-integration killer — a kernel can win isolated yet never run live (or net ~0 e2e)
confirms: 4
last_seen: 2026-06-19
---
# Make an optimized kernel survive CUDA/HIP-graph capture (or the win vanishes e2e)
- lever: sglang/vLLM capture the decode path into a graph. A kernel that JITs, syncs the host, or
  self-captures inside the captured region FALLS BACK TO EAGER → only static changes survive → ~0 e2e
  even at large isolated speedup (observed: 1.22–2.7× iso → ~0 / no live forwards).
- apply: author the STEADY-STATE call (2nd call onward) with ZERO host syncs and ZERO compiles:
  · precompile/register the kernel for ALL decode M-buckets at WARMUP, before capture (an
    `*_overlay_precompile(weight, weight_scale, m_buckets)` hook the integrator calls once, pre-capture).
  · key any weight cache by `weight.data_ptr()` (pure host int, weights persistent) — NEVER a
    `w_scale.sum().item()` fingerprint (a host sync that deadlocks capture).
  · no `.item()/.cpu()/.tolist()/synchronize()`/Python-if-on-GPU-scalar on the hot path.
- verify: the loose-tol unittest oracle will NOT catch a capture hang — only the e2e gate does. Confirm
  the optimized kernel actually launches INSIDE the graph (see [[method-verify-engagement]]), and that
  the candidate fits the SAME mem-fraction as the accepted config (a bf16 weight re-materialization can
  balloon the cache to tens of GB → KV-pool starved → e2e −9% even at +24% GEMM).
- source: exp/e2e_*MiniMax-M3-MXFP8*/ (FULL_AND_PIECEWISE) + exp/e2e_*Qwen3.5-27B-FP8*/ flydsl capture runs
