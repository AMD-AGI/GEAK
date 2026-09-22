---
name: moe-mxfp4-grouped-authored-gfx950-vllm
description: mxfp4 grouped fused-MoE (gpt-oss style) on gfx950/vLLM — author a whole-file Triton replacement of the fused dispatcher seam; +26.9% to +92.5% e2e, byte-exact, and it REPLAYS from the KB.
keywords: [moe, mxfp4, grouped-gemm, whole-file-overlay, launch-overhead, byte-exact-parity, warm-start-replay, pythonpath-shadow, dispatch-floor, routing-metadata]
kernels: [triton_kernel_moe_forward, matmul_ogs, _fused_sum_bitmatrix_rows_kernel, _topk_forward, pack_bitmatrix]
platforms: [gfx950]
kernel_class: moe_grouped_gemm
regime: both
key: moe-grouped-gemm-mxfp4 · gfx950 · decode-dominated (prefill present) · vLLM
type: routing
confidence: ★★★
confirms: 8
effect: 8th confirm CLOSES THE GEMM LANE ON THIS SEAM — four measured e2e A/Bs on the `_matmul_ogs_*` head (static decode-tile override at two shapes, a corrective re-tile, and an in-kernel gather-div dispatch fold) returned **−20.47% / −12.41% / flat / −6.10%**, all byte-exact and all engagement-proven, confirming the round-1 roofline verdict that both grouped-GEMM decode legs are HBM-pinned with zero micro-tuning headroom; see the `static tile override` and `resolvability` cautions below before funding another GEMM round here. 7th confirm CASHES THE CARD'S OWN PREDICTION — the `pack_bitmatrix` decode lever, recorded here in 2026-08-21 as an untested hypothesis, is now MEASURED e2e on gpt-oss-120b TP2 gfx950/vLLM: **+12.01% e2e, byte-exact, from ONE prologue kernel at only 5.05% GPU** (iso 1.586x serving-wtd; in-trace **4.41x** per decode launch). Also see the WARM-START narrowing in the caution below: the "it REPLAYS" claim holds only when the store carries a bindable overlay ARTIFACT — a record holding just a source diff was classified `winner_kind=authored, cannot be replayed` and produced no replay at all on a later exact-identity match. Original text: authored whole-file Triton rewrite of the fused seam = +92.5% e2e (gpt-oss-120b TP2, head 8.32% GPU), +26.9% (gpt-oss-120b TP2, head 32.43%, byte-exact after a corrective re-author; finalize/Director-validated_win, full-run 1.285×) and +1.83% (DeepSeek-V4-Flash TP4, head ~15.6%) — all Director/Integrator-verified byte-exact. It also REPLAYS: recalling the stored overlay as a warm start on the same deployment identity reproduced +85.9% e2e (byte-exact, non-overlapping) on a fresh run with no re-authoring. Isolated grouped-GEMM only shows 1.06–1.57×; it structurally undercounts the live decode win. Higher head share does NOT mean bigger e2e: at 32% the MoE is memory-bound at the HBM wall so the win is the launch-overhead/decode-seam share, not micro-tuning. 6th confirm (DeepSeek-V4-Pro TP8, 20.83% head, bake-off only): the NO-AUTHOR tuning surface is empty (triton_kernels opt_flags has no env knob and its constraints are GLOBAL: block_m=16 buys decode 1.030x/1.011x but costs prefill 0.638x, serving-wtd 1.008x = +0.16% ceiling), while a device-time profile of the seam exposes a FIXED-COST pack_bitmatrix (13.1% of decode_M64, 30.2% of decode_M1, unchanged at M=1/64/8192 because grid=cdiv(M,512)=1 workgroup serially loops bm_cols) -- the single biggest authorable decode lever, ahead of the GEMMs. NEW ENV LEVER NEVER TRIED BEFORE: on a deepseek_v4 routing method vLLM ranks AITER_MXFP4_BF16 (AiterExperts, W4A16 CK/flydsl mxfp4 MoE) ABOVE TRITON_UNFUSED and only rejects it because is_fused_moe_enabled() is False -- VLLM_ROCM_USE_AITER_MOE=1 makes every static support predicate pass, so probe the fused backend swap before/alongside authoring.
lifecycle: active
last_seen: 2026-08-24
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
  INTO the whole-file rewrite rather than scheduling it as its own unittest. Reason to remember: the win
  there is HOST LAUNCH COUNT (5 extra dispatches × n_layers × every step), and a CUDA-event/device-time
  harness — or a roofline prior — scores that ~1.0× by construction, so an isolated bake-off would
  mis-report it as no-op. Size such a direction ONLY on the e2e serving gate. After the overlay is
  adopted the surviving cluster is still ~5–6% GPU (~11% counting `_fused_sum_bitmatrix_rows`+`_reduce`)
  and is the natural next fold (fuse `_bitmatrix_metadata_compute_stage1`+`_stage2_pow2`, and
  `_ragged_tensor_metadata_memset`+`_compute`, into single launches) — pure launch fusion that preserves
  accumulation order should keep byte parity, but re-gate parity anyway (see the split_k parity failure).
- caution (WHICH routing branch is live — check before folding): `triton_kernel_moe_forward` has TWO
  routing paths. With `expert_map is None` (no EP shard) it calls the FUSED `triton_kernels.routing:routing`;
  otherwise it goes through `topk_fn` + `make_routing_data`. An overlay whose routing patches hang off the
  `make_routing_data` branch silently shadows NOTHING on a deployment that takes the fused branch — the
  metadata cluster stays fully live in the profile even though the overlay is LOADED and ENGAGED. Verify
  the live branch (log/grep the dispatch) before sizing or writing the fold; shadow the branch that runs.
- caution (WHERE the routing-metadata files actually live on ROCm — shadow the wrong path and the overlay
  is a silent no-op): the vLLM patch takes an `is_rocm()` branch and imports the SYSTEM `triton_kernels`
  package, so the live files are `triton_kernels/tensor_details/{bitmatrix.py, ragged_tensor.py,
  bitmatrix_details/sum_bitmatrix_rows.py}` in site-packages — NOT `vllm/third_party/triton_kernels/...`,
  which does not exist on the ROCm image at all. vLLM's own source comments the trap (patching the
  `vllm.third_party` module object "would have no effect" — different module object under the alias).
  Any overlay extension must shadow the three SYSTEM paths and re-run the import-time patch.
- caution (WARM-START replay is artifact-gated, not idea-gated): on a later exact deployment-identity
  match the stored champion did NOT reproduce — the kernel half was classified `winner_kind=authored,
  cannot be replayed from a record` (the store held a source diff, not a bindable overlay bundle) and the
  co-recalled config half (a `--max-model-len` reduction) was applied whole and REJECTED, leaving the run
  at its cold baseline with an empty overlay. So: store the OVERLAY ARTIFACT if you want a replay, and
  when a warm start returns nothing, treat every op in the profile as open and re-author — that run went
  on to bank +12.01% from `pack_bitmatrix` on the very same seam. Also verify a recovered flag was
  honoured in the server log before believing any null from it.
- lever (env, DeepSeek-V4 routing method): `select_deepseek_v4_mxfp4_moe_backend` prefers
  `AITER_MXFP4_BF16` (AiterExperts -> aiter CK/flydsl mxfp4 MoE, bf16 activations so no quant gate) and
  falls back to `TRITON_UNFUSED` ONLY because `AiterExperts.is_supported_config` sees
  `rocm_aiter_ops.is_fused_moe_enabled() == False`. With `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1`
  all of `_supports_current_device / _supports_quant_scheme(kMxfp4Static,None) / _supports_activation(SWIGLUOAI)`
  return True on gfx950. It is NOT isolated-measurable (the expert weights are re-shuffled at load time,
  outside the op oracle) -> hand it to the e2e A/B, and verify with the server-log line
  `Using '<BACKEND>' Mxfp4 MoE backend.`
- lever (author, decode) — **BEST MEASURED e2e-per-effort on this seam; do this before touching a GEMM**:
  `pack_bitmatrix` is a FIXED cost regardless of M, because `grid = cdiv(n_rows, 512)` = ONE workgroup for
  every decode batch, serially looping `bm_cols = cdiv(E,32)` iterations that each materialize a
  [512, top_k, 1] tensor + reduce_or. Two independent sightings: 13.1% of decode_M64 / 30.2% of
  decode_M1 (TP8), and 5.05% GPU per decode launch (gpt-oss-120b TP2). **The fix that gated:**
  re-grid the pack (1 workgroup -> 8 CTAs at decode n_rows=64) AND fold the whole prologue —
  int16 cast, bf16 cast, uint32 zeros-fill, the pack, the `== -1` compare, and the `where` + its 0-dim
  bf16 fill — into a SINGLE `_fused_pack_bitmatrix_prep` launch (6 graph nodes -> 1). Result: **1.57x**
  per decode case isolated, **4.41x** in-trace, total launches 86171 -> 74695, **e2e +12.01%
  byte-exact** (TPOT −8.3%). Pure re-parallelization + node fusion, no reassociation, so byte
  parity survives — it did, 12/12 greedy probes hashing identically to the no-overlay baseline.
- caution (SECOND-ORDER GAIN — size the fold by more than its own %GPU): removing that serialized prologue
  also made the two grouped GEMMs **11–12% faster per decode launch with no change to them at all**
  (gate_up 1.14x, down 1.12x). The per-step arithmetic reconciles the whole win
  (~54% of the modelled per-step saving from the prep kernel + ~46% from the two GEMMs, landing within
  ~11% of the measured TPOT drop),
  which is why the accept survived exceeding its own naive Amdahl ceiling (+1.90% predicted from
  5.05% x 1.586x vs +12.01% measured). This is the mirror image of the recorded de-overlap caution
  (neighbours getting SLOWER after a head speeds up): a serializing prologue HIDES latency in its
  neighbours, so its removal pays twice. **Whenever an accept beats its Amdahl ceiling, reconcile the
  per-step arithmetic (us saved x n_layers vs the measured TPOT delta) before believing it** — the
  arithmetic matching is what separates a real second-order win from output corruption.
- lever (next fold, after `pack_bitmatrix` lands): the surviving routing/norm cluster is still **18.0% GPU
  over ~11 kernels / ~25k launches**, but EVERY member now sits **AT the ROCm dispatch
  floor**. There is no second `pack_bitmatrix` there: no per-kernel rewrite can help a kernel already at
  the floor, so the only remaining lever is FUSION (collapse the ~13 per-layer routing+norm dispatches
  into ~3, ceiling ~14% of GPU time) or fuller cuda-graph capture. Screen by **per-launch time vs the
  dispatch floor**: above the floor = real device work = worth a rewrite; at the floor = launch-bound =
  overlay-fold or cudagraph only.
- caution (no no-author lever exists here): `triton_kernels.matmul_ogs` has ZERO env knobs
  (grep for `environ` in the installed package returns nothing) and its only tuning surface,
  `matmul_ogs_details.opt_flags.update_opt_flags_constraints`, is a GLOBAL dict applied to both
  grouped GEMMs at every M — so a decode-favourable tile is a prefill regression. Measured on
  DeepSeek-V4-Pro TP8 (baselines captured for prefill_M8192 / decode_M64 / decode_M1):
  block_m 16 = 0.638x / 1.030x / 1.011x, block_m 32 = 0.838x / 1.009x / 0.998x, block_m 64 (~default
  prefill) = 0.986x / 0.809x / 0.869x, block_m 128 = 0.560x / 0.180x / 0.410x, epilogue_subtile=1
  neutral; `num_warps` / `num_stages` are not enforceable constraints (AssertionError) and `split_k`
  raises `InapplicableConstraint`. Conclusion: M-conditional tiles are an AUTHOR-lane change, not a
  config lever.
- caution (a STATIC decode-tile override is now e2e-MEASURED as a do-no-harm violation, three times —
  also verify against the LIVE routed shape distribution, not the captured bucket): an authored
  `if M <= 512: block_m=16, group_m=1, xcd_swizzle=1, waves_per_eu=3, epilogue_subtile=2` override on
  `_matmul_ogs_*` engaged perfectly (70–236 markers on both TP workers, byte-exact 12/12) and still gated
  **−20.47%** (128x128 swiglu, iso 1.148x) and **−12.41%** (32x128 swiglu + `matrix_instr_nonkdim=0`,
  iso 1.030x), with median TPOT +36% / +28% — the regression grows as the running batch fills. Reason:
  under continuous batching at conc=64 the routed per-expert M is a DISTRIBUTION, and one static tile
  chosen off one captured bucket mis-serves most of it. Also verify the bake-off's own weighting: the
  1.156x that sized the first round was measured on the PREFILL shape (36 of the family's 2412 in-window
  launches) while 2124 launches were the M=64 decode shape — weight a grouped-MoE bake-off by LIVE LAUNCH
  COUNT per M bucket from the trace, never by the profiled row's tile name.
- caution (also verify the ceiling is RESOLVABLE before spending the A/B): every candidate in that round
  had an Amdahl ceiling of +1.1–1.8% (8.44–14.14% GPU x iso 1.03–1.15x) while the box's within-session
  throughput spread under foreign load was 6–11%. A gate that cannot resolve the candidate in either
  direction buys noise: the corrective re-tile read **+5.90%** at the integrate gate on 2 OVERLAPPING
  repeats, was gated `stack` not `accepted`, and the next reprofile falsified it (0.995x,
  flat; retiled swiglu leg only −1.24% per decode launch while the untouched down-proj leg went +9.2%).
  A dispatch-DELETION fold (gather `src_indx` trunc-division moved into the GEMM) likewise read −6.10%
  on three fully overlapping repeats — unresolved, not refuted. Treat a `stack` verdict as a HYPOTHESIS
  to re-verify against in-trace per-launch us, never as a small banked win.
- caution (the swizzle dead-end is narrower than previously recorded): on this build the mxfp4 expert
  weights are a plain `StridedLayout` triton_kernels Tensor (`w1.storage.data` [E, K/2, N] uint8,
  scale [E, K/32, N]) — NOT a CDNA4-swizzled blob — even though `quant_config.is_scale_swizzled` is
  True. What still blocks a direct aiter/flydsl bind is (a) the [E,K/2,N] vs [E,N,K/2] preshuffle and
  (b) aiter's fused mxfp4 MoE requiring `q_dtype_a` in {fp4x2, fp8} for most kernels plus an OAI
  swiglu with `gemm1_clamp_limit` it does not implement. Reach the aiter family through the vLLM
  backend selector (env lever above), not by hand-converting the weights.
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
  1.2848× (+28.48%), validated_win, output parity pass. Post-win the MoE
  grouped GEMM is still #1 ~17% and
  memory-bound at the HBM wall (hbm_util ~0.97–1.01) — only byte-reduction headroom remains);
  exp/e2e_*gpt-oss-120b*/ 2026-08-20 (WARM-START REPLAY on an exact deployment-identity match, TP2
  isl/osl 1024 conc 64: the stored whole-file overlay was re-applied with no re-authoring and passed the
  same gate a fresh idea faces — Integrator A/B +85.93%, non-overlapping,
  byte-exact parity, overlay LOADED on every worker + first-forward marker. The co-recalled CONFIG record
  did NOT reproduce — its payload was byte-identical to the already-accepted config, i.e. kernel overlays
  replay far more reliably across boxes than config records do; a later round of the same run confirmed by
  source inspection of the adopted overlay that its three routing patches all hang off the
  `make_routing_data` branch while this deployment takes the fused `routing()` branch, which is why the
  ~11% metadata cluster survived the adopted overlay — structural finding, no e2e measured for it);
  exp/e2e_*DeepSeek-V4-Pro*/ 2026-08-21 (TP8, head 20.83%, op bake-off
  only: seam re-confirmed as `UnfusedOAITritonExperts.apply` in `gpt_oss_triton_kernels_moe.py` and it
  DOES take the `make_routing_data`/SparseMatrix branch — unlike the fused-`routing()` gpt-oss case, so
  routing-metadata patches DO bind here; opt_flags sweep + per-kernel device profile as above; no
  direct_light isolated win, routed to author + the AITER_MXFP4_BF16 env probe);
  exp/e2e_*gpt-oss-120b*/ 2026-08-23 (7th confirm, TP2 gfx950 vLLM 0.26 mxfp4, isl/osl 1024 conc 64,
  cold baseline re-established after the warm start reproduced nothing: `pack_bitmatrix` re-grid +
  6-node prologue fold ACCEPTED — iso 1.586x, Integrator A/B pooled REF vs CAND med =
  **+12.01%**, non-overlapping (cand_min > ref_max), byte-exact 12/12 greedy probes,
  engagement 2/2 TP workers. Post-win reprofile: the head `_matmul_ogs_*` family rises to **38.5% GPU**
  (from 30.98%) purely as the denominator shrank, and its byte model is now INFEASIBLE at 105.8–116.6%
  of the HBM roof ⇒ genuinely bandwidth-pinned, no micro-tuning headroom left; the only lever there is
  BYTE REDUCTION (fewer distinct experts resident per rank, i.e. EP-style placement). Comm rose to
  17.02% relatively (skew<3 on both collectives ⇒ genuine work, a comm-CONFIG lever, not a kernel one));
  exp/e2e_*gpt-oss-120b*/ 2026-08-24 (8th confirm, same TP2 gfx950 vLLM deployment, ROUND 2 = the GEMM
  lane on this head, four engagement-proven byte-exact A/Bs: static decode-tile override at the 128x128
  swiglu shape −20.47%, corrective re-tile `stack`/+5.90% on overlapping repeats then falsified flat by
  the reprofile, the same override at the 32x128 swiglu shape −12.41%, and an in-kernel gather-div
  dispatch fold −6.10% on overlapping repeats. Post-round rollups: MoE grouped GEMM 38.5 → 33.85%, MoE
  routing 18.0 → 18.07%, comm 17.02 → 15.68%, dense GEMM 9.16 → 10.26%, attention 4.23 → 6.12% — but the
  in-window total moved −4.8% mostly from PREFILL-CHUNK COMPOSITION (both windows 6 prefill +
  58 decode steps, r1 prefill 128x128 x108 calls vs r2 64x512 x144 calls), so prefill-tile deltas between
  two reprofiles of this seam are NOT an overlay effect; compare decode per-launch us instead).
