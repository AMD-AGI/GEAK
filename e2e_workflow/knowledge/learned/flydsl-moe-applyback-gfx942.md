---
key: int4_w4a16 fused-MoE FlyDSL apply-back (rewrite→live vLLM) · gfx942/MI300X · vLLM
type: method
confidence: ★★★
effect: lands an isolated FlyDSL int4-MoE win e2e WITHOUT OOM and keeps it (4 independent Director-verified confirms). At the FULL FAIR config (mem 0.9, NO max-len cap, 262144) after the scale-leak fix: +77% e2e VERIFIED (257→455 tok/s cc64, TTFT −45%, TPOT −44%, cosine 0.99998, Available KV 20.13 GiB). Re-confirmed 2026-06-26 via a REVERSIBLE OVERLAY (no site-packages mutation): +63.57% e2e (329.6→539.1 tok/s, ISL8192/OSL1024/cc64, non-overlapping, TPOT 168→105ms, KV 19.9 GiB, parity cosine 0.99998 / GSM8K 0.915). Re-confirmed 2026-06-28 (overlay): +66.02% e2e (319.6→530.6 tok/s, non-overlap cand_min 506.4>ref_max 321.8, iso 1.7966×, gsm8k 0.77≥0.70, 240 converts/4 workers, 0 shim errors). Earlier capped result: +32.6% (300.5→398.4 tok/s, --max-model-len 32768 --gpu-mem 0.95). eager-mode is a TRAP (−22…−34%).
last_seen: 2026-06-28
---
# Land a FlyDSL int4-W4A16 fused-MoE rewrite INTO live vLLM (the apply-back, not the rewrite)
- lever: after `flydsl_rewrite_quantized_moe` produces an isolated win, land it via TWO env-gated edits to
  the installed vLLM (off by default ⇒ stock): (1) `compressed_tensors_moe.py` →
  `CompressedTensorsWNA16MoEMethod.process_weights_after_loading` calls `convert_layer_inplace(layer)`;
  (2) top of `fused_moe.py:fused_experts_impl` routes int4-no-zp to `flydsl_fused_experts_impl`. Shim =
  `${FLYDSL_SHIM_DIR}/flydsl_moe_shim.py`.
- apply: convert weights ONCE at LOAD time, IN PLACE, SAME-BYTE (overwrite `w*_weight_packed.data`,
  chunked over experts), key caches by the NEW weight `data_ptr()` ONLY — never `A.data_ptr()` / per-forward
  unpack (that IS the OOM). **MUST also re-home the SCALE param: `layer.w*_weight_scale.data = s*_flat`**,
  not just the weight — else the original `[E,N,G]` bf16 scale stays live on the layer while the converted
  scale is cached → scales DUPLICATED ≈ +246 MiB/layer × N ≈ +14.5 GiB → KV OOM (this was THE binding
  constraint; see caution below). Launch GRAPH mode (NO `--enforce-eager`), env `VLLM_USE_FLYDSL_MOE=1`.
  After the scale-leak fix, NO `--max-model-len` cap is needed: runs at full 262144 + `--gpu-memory-utilization 0.9`.
- verify: startup passes `determine_available_memory` (no HIP-OOM) → `torch.compile` → `Capturing CUDA
  graphs … finished` → healthy; smoke coherent; 0 `shim failed`/`weights not converted`; then same-session
  GRAPH A/B vs Triton baseline + GSM8K parity.
- caution: **pin FlyDSL to ONE checkout** — a stale exported `FLYDSL_ROOT` mixing kernels (code/FlyDSL) with
  bindings from a different build fails compile with `Dynamic int_tuple leaf must be an i32 or i64 value,
  got: <unknown type>` (eager) / `gl$v` (graph). Version mismatch, NOT a torch.compile incompatibility —
  the graph path needs NO custom op. Also verify `python3 -c "import flydsl, kernels.moe_gemm_2stage as k;
  print(flydsl.__file__, k.__file__)"` resolve under the SAME tree.
- caution: **also verify the e2e gate, never trust eager** — eager hides launch latency and reads as
  −22…−34% (graphs required). NOTE (corrected 2026-06-26): the KV shortfall was NOT mainly the A2
  `repeat_interleave` peak (that was only ~0.23 GiB) — it was the scale-duplication leak (~14.5 GiB, see
  apply step + caution below). After re-homing the scale param, KV ≈ Triton (20.13 vs 23.1 GiB).
- caution: **also verify the shim is wired to the NEW kernel AND that it starts at EQUAL mem-parity** — a
  fresh authored FlyDSL 2-stage kernel can be a real isolated win (Director-verified 2.08× iso, 68.7% GPU
  time) yet move e2e 0% if it improves ONLY the GEMM compute and is never bound into the live
  `flydsl_moe_shim.py` (shim mtime predating the kernel = not wired). At the integrator's EQUAL serving
  config (mem 0.9, NO `--max-model-len` cap) the shim failed to start (only 5.73 GiB KV avail vs 17.16
  needed for max seq 262144 → EngineCore fails → reject `mem_footprint_starves_kv`), and the +32.6% above
  was only reachable with `--max-model-len 32768 --gpu-mem 0.95` (UNEQUAL, capped).
- **ROOT CAUSE FOUND + FIXED (2026-06-26)**: the OOM was NOT the A2 `repeat_interleave` (only ~0.23 GiB).
  It was a **scale-duplication leak in `convert_layer_inplace`**: it re-homed `w*_weight_packed.data` but
  left `w*_weight_scale` pointing at the original `[E,N,G]` bf16 storage while ALSO caching the converted
  flat scale → scales duplicated ≈ +14.5 GiB (~246 MiB/layer × 60) → KV 5.96 GiB. **Fix = one line per
  weight: `layer.w*_weight_scale.data = s*_flat`** (re-home scale param to the cached flat buffer; convert
  then truly memory-neutral, alloc delta 0). RESULT at the EQUAL/fair config (mem 0.9, full 262144): engine
  STARTS, Available KV 5.96 → **20.13 GiB** (307,566 tok), e2e **+77%** vs Triton (257→455 tok/s cc64),
  cosine 0.99998 — no context cap needed. Secondary optional follow-ups (not required to start): remove the
  A2 pre-gather via stage-1 `compile_moe_gemm1` compact-input/sorted-row in-kernel gather, and reuse
  scratch + gemm2 `accumulate=True` (stage-2 out `[M,hidden]`) to recover the residual ~3 GiB vs Triton.
- apply (REVERSIBLE OVERLAY variant — preferred, 0 site-packages mutation): instead of editing the installed
  `compressed_tensors_moe.py` / `fused_moe.py` files, deploy the same two edits as an add-module loaded at
  import time that (1) wraps `CompressedTensorsWNA16MoEMethod.process_weights_after_loading` to do the
  in-place same-byte weight+scale convert, and (2) add-rebinds `fused_experts_impl` to the vendored
  `flydsl_moe_shim`. Verified 2026-06-26: live vLLM files keep 0 seam strings; `[overlay] inject+rebind`
  banners fire across all 4 TP workers; identical e2e win as the file-edit path. Same caches/scale-rehome
  rules apply. Fully revertible (drop the env/module ⇒ stock).
- source: in-repo expert skill `perf_knowledge/expert_skills/skills/apply_flydsl_moe_to_vllm/` — the full
  recipe + the vendored validated shim `flydsl_moe_shim.py` (self-contained; no external eval-dir needed).
- source: measured 2026-06-25 (vLLM 0.19.0, Kimi-K2.6 int4-W4A16, MI300X TP4) same-session GRAPH A/B —
  baseline 300.51 → FlyDSL 398.35 tok/s (+32.6%), decode TPOT 168→61ms (2.75x), 0 fallbacks; independently
  reproduced at +32–34% with GSM8K parity 0.915. Companion `[[method-cudagraph-safe-integration]]`.
- source: 2026-06-25 e2e run (eval e2e_..._20260625T111041Z, ROUND 1) — NEW authored FlyDSL 2-stage int4
  kernel iso 2.0786× / 68.71% GPU (Director-verified), but e2e REJECTED 0% at EQUAL config: shim not wired
  to the new kernel + A2 pre-gather starves KV (avail 5.73 GiB < 17.16 needed) → EngineCore won't start.
  Baseline kept 327.0 tok/s. The do-no-harm reject that pins the equal-mem-parity caution above.
- source: 2026-06-26 mem-fix (same eval, gather-fixed base) — instrumented convert with torch.cuda memory
  logging: proved compiled-exe cache is ~free (28 exes = +0.004 GiB; the `_ECACHE`/per-exe-workspace
  hypothesis was REFUTED) and that `convert_layer_inplace` was NOT memory-neutral (+14.5 GiB scale dup,
  +246 MiB/layer). One-line-per-weight scale re-home → convert alloc delta 0, Available KV 5.96→20.13 GiB,
  starts at mem 0.9 + 262144, +77% e2e vs Triton (257→455), cosine 0.99998. Shim-only; no kernel edit.
  Fix is self-contained in the in-repo vendored shim
  `perf_knowledge/expert_skills/skills/apply_flydsl_moe_to_vllm/flydsl_moe_shim.py` (`convert_layer_inplace`).
- source: 2026-06-26 e2e run (eval e2e_moonshotai-Kimi-K2.6_20260626T103907Z, ROUND 1, gfx942 MI300X TP4
  GPU 0,1,2,3, vLLM 0.19.0, Kimi-K2.6 int4-W4A16 g32). Reproduced the skill via a REVERSIBLE OVERLAY (no
  site-packages edits). fused_experts MoE was 70.76% GPU; FlyDSL int4-W4A16 grouped-GEMM rewrite iso geomean
  2.3167× (Director-verified, correctness pass) → e2e **+63.57%** (REF med 329.579 → CAND med 539.106 tok/s,
  ISL8192/OSL1024/cc64), distributions NON-OVERLAPPING (cand_min 513.831 > ref_max 332.868 ≫ 0.5% noise),
  TPOT 168→105ms, Available KV 19.9 GiB (memory-neutral convert, no KV starvation), graph capture healthy
  (no hang), 0 'shim failed'/'weights not converted'. Parity: fresh no-overlay baseline 12/12 byte-exact;
  cand 4/12 byte-exact, 10/12 first-20-char, 11/12 first-token — all divergences are valid int4-rounding
  argmax flips with coherent content (cosine 0.99998 / GSM8K 0.915); byte-exact is the WRONG bar for int4.
  Reprofile: fused_moe_kernel_gptq_awq replaced by FlyDSL moe_gemm2_0 (now #1 at 47.15% eff, MoE share
  70.8%→47.2% eff), next heads = MLA aiter mla_a16w16 #2 (8.1%) + torch_native elementwise overhead (~8.7%,
  fusion/cuda-graph opportunity).
- source: 2026-06-26 FOLLOW-ON (same eval, ROUND 2) — after this apply-back is accepted, re-profile shows
  the FlyDSL `moe_gemm2_0` down-proj leg is the NEW #1 head (47.15% eff GPU, raw 28.9%, skew 1.7×). It is
  itself tunable: a stage-2 prefill tile_m 32→64 wired into the live `_pick_tiles` seam lands a further
  +10.35% e2e (531.7→586.7 tok/s, non-overlap, TPOT 105.9→94.4ms, 12/12 byte-exact). So the apply-back
  OPENS a second lever inside the shim, not a dead end — see companion `[[flydsl-moe-gemm2-tiletune-gfx942]]`.
- source: 2026-06-28 e2e run (eval e2e_moonshotai-Kimi-K2.6_20260628T122336Z, ROUND 1, gfx942 MI300X TP4
  GPU 0,1,2,3, vLLM, Kimi-K2.6 int4-W4A16 g32). 4th independent Director-verified confirm via REVERSIBLE
  OVERLAY. Stock head fused_moe_kernel_gptq_awq 68.45% eff GPU; FlyDSL apply-back iso 1.7966× → e2e
  **+66.02%** (REF med 319.57 → CAND med 530.56 tok/s, non-overlapping cand_min 506.38 > ref_max 321.84
  ≫ 0.5% noise), engagement reproven (convert_layer_inplace 240 calls = 60 layers×4 ranks, runtime forward
  engaged all 4 workers, 0 shim errors, old gptq_awq kernel ABSENT, ref leg 0 flydsl refs). Quant gate:
  gsm8k n=100 cand 0.77 ≥ base 0.70 (the earlier n=40 −7.5pt dip was small-sample noise). KV 19.9 GiB /
  303,998 tok, no OOM (memory-neutral convert). Reprofile (REPROFILE_SHIFT): MoE GEMM now FlyDSL moe_gemm2_0
  43.47% eff (23.66% raw), MoE raw share 240.6→88.5 GB-ns (~2.7× less GPU time) — still #1 but far smaller;
  new #2 editable = aiter MLA decode mla_a16w16 10.36% eff; TP comm collectives big raw (29.86%/17.58%) but
  de-inflate to 2.06%/1.39% (spin-bound, comm-config lever not rewrite); quickreduce_twoshot 4.79% eff
  (transfer-bound skew 0.89) + elementwise fill 3.57% eff (972k calls, Lever-1 fusion) sit just behind.
- source: 2026-06-28 FOLLOW-ON (same eval, ROUND 2) — after this apply-back, the structurally-new GEMM2
  FUSED-XROWS lever (fused-SwiGLU + un-replicated-X, FLYDSL_MOE_FUSED_XROWS=1) lands a further +6.49% e2e
  STACK (provisional, engaged x4, parity-safe, overlapping at 2 reps). It SPLITS the single moe_gemm2_0 head
  into TWO engaged FlyDSL kernels — moe_gemm1_0 (stage1 gate_up w/ in-epilogue SwiGLU) becomes the new #1
  (31.49% eff), moe_gemm2_0 down-proj #2 (14.91% eff), combined editable MoE-GEMM ~46.4% eff. So the
  apply-back keeps opening fresh in-shim levers (now target gemm1 next) — see `[[flydsl-moe-gemm2-tiletune-gfx942]]`.
- NOTE: a separate AUTHORED single-GEMM1 int4 W4A16 candidate (iso 2.0046×, correctness pass) was a gate-2
  REJECT (dead_end) THIS run: it implements only per-expert GEMM1 (A@dequant(w).T) but the live seam
  `fused_experts(hidden_states,w1,w2,topk_weights,topk_ids,...)` is the FULL fused MoE (routing+GEMM1+
  SiLU/mul+GEMM2+topk=8 reduction). Incompatible signatures ⇒ a direct add-rebind would crash; no
  parity-preserving wrapper existed ⇒ no engagement, no distinct candidate leg. LESSON: a sub-op kernel
  (GEMM1 only) cannot be rebound at a fused-op seam without a wrapper — the genuine win is the FULL fused
  shim (apply_flydsl_moe_to_vllm), not an isolated GEMM1. Routing stays POSITIVE (use the validated full
  shim); this is a seam-compatibility caution, not a blocklist.
