---
key: engagement verification · any gfx · any backend
type: method
confidence: ★★★
effect: turns "did my kernel actually run live?" from a guess into proof
confirms: 5
last_seen: 2026-08-14
---
# Prove the optimized kernel ran on the LIVE serving path (don't infer from an e2e wiggle)
- lever: instrument the candidate kernel with a one-shot stderr banner and grep the server log — this
  PROVES engagement on both the bench and parity legs, instead of inferring it from a throughput delta
  (which can move for unrelated reasons).
- apply: emit `[overlay-mark] <kernel> OPTIMIZED kernel CALLED` (once) from inside the candidate; for an
  overlay rebind also look for `[overlay] injected module <path>` (N hits = N workers) and `[OVERLAY_ENGAGED]`.
- verify: ≥1 banner per worker on the live run = engaged. ZERO banners but server healthy = the seam
  missed (wrong rebind target / a self-capturing wrapper fell back to eager — see
  [[method-cudagraph-safe-integration]]). For cudagraph paths, verify engagement INSIDE the captured
  region, not just at module-injection time.
- lever (rebind that STICKS, and a monotonic counter): prefer a **module-attribute rebind at a seam the
  caller resolves LAZILY** (vLLM op bodies do `from aiter import X` inside the call → the rebind engages
  every call and survives PIECEWISE cuda-graph capture). Emit a BIND line at inject time AND
  `[overlay-cand] CALLS pid=… n=1,2,5,…,10000` at power-of-ten marks (no host sync, no alloc) — a
  monotonic counter through capture + the timed rounds is much stronger proof than a one-shot banner,
  and the reference leg must show ZERO such lines.
- lever (**vLLM `torch.ops.vllm.*` custom ops — patch at REGISTRATION, not after import**): ops declared in
  `vllm/_aiter_ops.py` are registered on that module's LAST line, so any post-import attribute rebind is
  already too late, and the aiter-level rebind is invisible (see the finder caution below). The one seam
  that works is substituting the `op_func` passed to `vllm.utils.torch_utils:direct_register_custom_op`
  (wrap it in `sitecustomize.py` before vLLM imports). Proof shape: one BIND line per registered op plus
  monotonic CALLS counters **in the EngineCore worker** (not just the launcher process); reference leg zero.
- caution (**silent-no-op seam, gfx942/aiter, cost a whole capture round**): `aiter/ops/triton/__init__.py`
  installs a `_BACKWARD_COMPAT_MAP` module `__getattr__` **and** a `sys.meta_path` finder
  (`_backward_compat_find_spec`) that **RE-EXECUTES** the real module on every
  `from aiter.ops.triton.<legacy> import fn` — so each from-import returns a BRAND-NEW function object and
  your patched attribute (on the real module AND on the `sys.modules` alias) is invisible. Observed: 0
  recorded calls despite 157680 live launches. Fix: REMOVE that finder from `sys.meta_path`, point
  `sys.modules['aiter.ops.triton.<legacy>']` at the already-imported real module, THEN set the symbol, and
  re-run the exact from-import as a probe. Non-aliased seams (`act_mul_and_fp8_group_quant`,
  `aiter.ops.quant:per_group_quant_hip`) rebind normally.
- source: exp/e2e_*Qwen3.5-27B*/ FLA overlay runs 2026-06-07 / 06-09;
  /wekafs/test_results/Qwen3_14B_20260813/e2e_Qwen3-14B-FP8_20260813_031549_2866199_26474/
  (BIND+CALLS proof in overlay/cand_kernel_gemm_xdl_cshuffle_v3/integrate_result*.json;
  aiter legacy-alias finder bug in kernels/fused_quant_prologue_task/_capture_overlay/sitecustomize.py;
  2026-08-14 round 2 — the working `direct_register_custom_op` seam + 4 BIND/CALLS pairs in
  overlay/cand_fused_quant_prologue/{overlay_cand,integrate_result.json})
