---
key: moe-grouped-gemm-mxfp4 · gfx950 · decode-dominated (prefill present)
type: routing
confidence: ★★★
confirms: 4
effect: authored whole-file Triton rewrite of the fused seam = +92.5% e2e (gpt-oss-120b TP2, head 8.32% GPU), +26.9% (gpt-oss-120b TP2, head 32.43%, byte-exact after a corrective re-author; finalize/Director-validated_win, full-run 5799→7042 tok/s = 1.285×) and +1.83% (DeepSeek-V4-Flash TP4, head ~15.6%) — all Director/Integrator-verified byte-exact. Isolated grouped-GEMM only shows 1.06–1.57×; it structurally undercounts the live decode win. Higher head share does NOT mean bigger e2e: at 32% the MoE is memory-bound at the HBM wall so the win is the launch-overhead/decode-seam share, not micro-tuning.
last_seen: 2026-08-19
---
# MXFP4 grouped fused-MoE (gpt-oss style) — author a whole-file Triton replacement, not a GEMM swap

- path: (1) identify the seam — it is a FUSED dispatcher (`triton_kernel_moe_forward` in the in-tree
  `gpt_oss_triton_kernels_moe.py` / `UnfusedOAITritonExperts.apply`), NOT a standalone `gemm(...)`;
  both grouped GEMMs (w13 gate/up + w2 down) run inside it with OAI-swiglu + routing/gather/scatter.
  (2) Expect no env/flag win — the aiter mxfp4 fused-MoE is not a drop-in, flydsl dead-ends on the
  swizzle, ckProfiler is absent. (3) Author a whole-file Triton replacement: collapse the
  routing-dispatch cluster and re-tile both GEMMs (w13 block_m 32→16; tune the w2 down tile).
- expected gain: scales with live head share, and the e2e delta legitimately EXCEEDS the naive Amdahl
  ceiling because the seam is launch-overhead-bound at decode. 8.32% head → +92.5%; ~15.6% head →
  +1.83%. Do not size the opportunity from the isolated number.
- apply: whole-file module swap injected via PYTHONPATH shadow (install tree unmutated, reversible).
  Confirm every TP worker loaded it (LOADED banner per proc) plus a first-forward marker.
- verify: immutable op `unittest.py` (mxfp4 tol=3e-2, random-value parity vs the live triton_kernels
  baseline, ≥2-shape CUDA-graph replay), then a same-session e2e A/B with BYTE-EXACT greedy parity
  (temp=0/seed=0/ignore_eos) against a FRESH no-overlay baseline.
- caution: trust the byte-exact e2e gate over the isolated ×, and don't drop the head on a modest
  isolated number. A non-quant tile-shape rewrite (block_m 32→16 + split_k + routing-metadata reuse)
  can SILENTLY BREAK byte-exact greedy parity vs a deterministic baseline (7/12 temp=0 prompts
  diverged, one at the first token) even while e2e is +29% and engagement is proven on every TP
  worker — a faster-but-wrong server is rejected. When it happens, route to a CORRECTIVE re-author
  that preserves accumulation order (avoid split_k / order-changing reduction); it recovered byte
  parity AND kept +26.9% e2e. Also verify: the routing-metadata chain
  (`_topk_forward`/`pack_bitmatrix`/`_bitmatrix_*`/`_sum_bitmatrix_rows`/`_stage2_pow2`) is NOT
  cleanly standalone-extractable on the v3.6.0 SparseMatrix path (split across `topk_fn` +
  `make_routing_data`, returns structured RoutingData/Gather/Scatter, host-launch-bound) — fold it
  INTO the whole-file rewrite rather than scheduling it as its own unittest.
- caution: Do NOT route a fused seam to standalone-gemm-swap or dense-linear-env-overlay —
  there is no call site to bind. flydsl would have to invert two proprietary swizzles (triton_kernels
  `Tensor.storage` + vLLM CDNA4 mxfp4 scale) plus the w13 shuffle: high correctness risk, and
  `aiter.ops.flydsl.moe_kernels` fails to import.
- caution (different sub-variant): under the OCP-MX EMULATION backend
  (`OCP_MXQuantizationEmulationTritonExperts`) an extra `dq_uint8_mxfp4_to_half_kernel` DOMINATES
  (~51% decode GPU) ahead of the ~14% `fused_moe_kernel`, so the lever is a NATIVE-fp4 backend swap
  that kills the dequant round-trip, not a rewrite of the 14% (roofline ceiling there is only
  ~1.37×/+3.7%). Enabling the fast path from inside the emulation `apply()` on runtime predicates
  (`expert_map` on a TP shard, w-scale uint8-vs-E8M0) SILENTLY FELL BACK twice — bit-identical output
  (max_rel_err == 0.0) is the tell; make the engagement assert unconditional.
- source: exp/e2e_*gpt-oss-120b*/ 2026-08-13 (Director +92.5%/1.9252×, byte-exact 12/12, TP2);
  exp/e2e_*DeepSeek-V4-Flash-0731*/ 2026-08-17 (Director validated_win +1.83%, byte-exact 12/12,
  overlay engaged 7/7 workers, TP4); exp/e2e_*DeepSeek-V4-Pro*/ 2026-08-16 (same seam confirmed, TP8,
  bakeoff only — the earlier run's per-head attribution was reversed to dead_end/implausible_speedup
  at review, so it backs the SEAM identity but not a speedup);
  exp/e2e_*Qwen3.5-397B-A17B-MXFP4*/ 2026-08-16 (the emulation sub-variant; fast path never engaged,
  iso 0.972× no-op — no verified win);
  exp/e2e_*gpt-oss-120b*/ 2026-08-19 (gpt-oss-120b TP2 head 32.43%: seam =
  `triton_kernel_moe_forward`/`gpt_oss_triton_kernels_moe.py` editable Triton; op_bench found NO env/flag
  lever (delegated to server-flag path, oracle-only). First author (block_m 32→16 + split_k +
  routing-metadata reuse) engaged 2/2 workers, e2e +29.15% (5267→6802) but REJECTED on byte-parity fail;
  corrective re-author preserving accumulation order = ACCEPTED, byte-parity pass, +26.9% head e2e
  (Integrator A/B 5612→7122, non-overlapping) — CONFIRMED by the finalize/Director gate: full-run
  5799.19→7042.04 tok/s = 1.2848× (+28.48%), validated_win, output parity pass. Post-win the MoE
  grouped GEMM is still #1 ~17% and
  memory-bound at the HBM wall (hbm_util ~0.97–1.01) — only byte-reduction headroom remains).
