---
# --- discovery header ---
name: editable-triton-cluster-amdahl
description: Many-tiny-kernel editable clusters (FLA/mamba, MoE routing/metadata) are Amdahl- or launch-bound → stack them into the owning seam's whole-file overlay and gate on e2e, never as solo extractions.
keywords: [amdahl, launch-bound, host-dispatch, cluster, whole-file-overlay, extraction-seam, moe-routing, linear-attention, carry-forward, noise-band]
kernels: [chunk_gated_delta_rule_fwd_h, chunk_fwd_kernel_o, causal_conv1d, recompute_w_u_fwd, moe_align_block_size, moe_sum_vec, act_and_mul, topkGating]
platforms: [gfx942, gfx950]
kernel_class: method
regime: both
# --- classification + evidence ---
key: editable small-kernel clusters (FLA/mamba linear-attn; MoE routing/metadata launch chains) · gfx942/gfx950 · vLLM + sglang
type: routing
confidence: ★★★
effect: per-kernel iso 1.10–1.18× real, but each ~1–3% GPU → solo e2e below the 0.5% noise band. The overlay-extension route is now MEASURED, not just recommended: folding a bf16 MoE launch chain into the owning module's whole-file overlay landed +6.15% e2e byte-exact while the head GEMM's per-launch time did not move at all. 8th confirm sharpens the screen from %GPU to PER-LAUNCH time vs the ROCm dispatch floor: at the floor ⇒ fold-only (a device-time harness scores a launch-collapse ~1.0× by construction); well above the floor ⇒ a solo extraction is justified even at 5% GPU, and one such prologue gated at +12.01% e2e byte-exact.
9th confirm adds the cheapest screen of all: before funding an overlay on a peripheral chain, check whether a backend swap already in the config lane DELETES the chain — one did, for free, and the declined cluster's six kernels lost **77% of their GPU time** with no kernel work at all.
10th confirm gives the lane's STOP SIGNAL a concrete shape: after two accepted rounds the top of the profile was 26.9% fused-MoE at 95.7% of the HBM roof + 24.8% TP all-reduce pinned at the xGMI interconnect roof — both non-editable/config-only — and the largest EDITABLE head had fallen to 3.56%, collapsing the kernel-rewrite ceiling from ~+4% to ~+1.5%. When no editable kernel clears the 5% bar and the two heads that do are at a hardware roof, the honest move is to STOP the kernel lane and spend the round on config (comm algorithm/threshold, AR quantization, comm-compute overlap) or byte reduction — not to nominate the biggest sub-bar kernel to fill a task floor.
confirms: 10
last_seen: 2026-08-23
---
# Editable Triton cluster → STACK-and-compound, don't expect a solo e2e pass
- lever: the gated-delta / FLA / mamba Triton kernels (chunk_gated_delta_rule_fwd_h, chunk_fwd_kernel_o,
  causal_conv1d, recompute_w_u_fwd) are the best EDITABLE targets on hybrid gfx942, with large *isolated*
  wins — but each sits at ~1–3% GPU, so by Amdahl no single one moves e2e past noise in the
  prefill regime (where ~80% GPU is dense GEMM). Optimize them as a COMBINED cluster and let the
  Director's final stacked gate decide.
- apply: spend the head/config budget on dense GEMM FIRST; route the whole cluster as carry-forward and
  measure the SUM. Extract seam = modules under `sglang.srt.layers.attention.{fla,mamba}`; for MoE
  launch chains it is the fused-experts module that already imports/calls them.
- verify: **pre-dispatch screen** — if `pct_gpu × (1 − 1/plausible_iso) < NOISE_BAND_PCT (~0.5%)`, mark
  carry-forward-only (don't expect a solo gate to pass). Engagement via the overlay banner — see
  [[method-verify-engagement]]. e2e A/B via [[method-e2e-ab-harness]].
- verify (cheap, no server round): decide extractability ON DISK before spending a capture. Grep each
  cluster member's Python wrapper: if it dispatches to a compiled csrc op with an OUT-PARAM signature
  (`torch.ops._C.<op>(out, in, …)`) or is a pure indices/metadata producer, it can NEVER satisfy the
  `call(args) -> fresh tensor` seam contract — `editable:false` is knowable in minutes. Library/runtime
  symbols (gating, `__amd_rocclr_copyBuffer`) have no Python seam at all. **Three more on-disk
  disqualifiers, each individually sufficient and each readable from source in minutes** (all three fired
  on one MoE routing-metadata chain; declining it cost 0 server rounds):
  (a) **the seam returns a DATACLASS, not a tensor** (`RoutingData`/`BitmatrixMetadata`/
  `RaggedTensorMetadata`) — the shape-capture cannot serialize it and the correctness checker cannot
  compare it;
  (b) **the baseline itself fails the fresh-output gate** — production metadata builders deliberately
  CARVE several outputs as slices of ONE allocation, so the harness's distinct-`data_ptr` /
  no-cross-output-mutation anti-exploit rule flags the UNMODIFIED library code as the shared-buffer
  cheat. There is no tolerance knob for that; grep for a `combined`/`_all` buffer that is then sliced;
  (c) **no importable `module:attr`** — the hot launcher is a LOCAL CLOSURE installed by import-time
  mutation of `__globals__`, so there is no dotted name for `candidate_bind` and the legs-differ
  assertion has nothing to resolve;
  (d) **the "kernel" is an AGGREGATE with no common callable** — a profile row that is really N distinct
  ops summed together (one fp8 MoE routing/quant row = 6 ops, 8.69% GPU, ~30k launches) has no single
  `module:attr` whose rebind reaches all N, so `target_callable` / `candidate_bind` cannot even be
  written and `deepest_verified` is unobtainable. Check whether the row is one op before routing it;
  (e) **a compiled csrc member cannot be shadowed by the overlay mechanism at all** — a single-file
  PYTHONPATH overlay cannot replace `torch.ops._C.<op>`; only a full csrc rebuild would, which is out of
  scope for a python-editable task. The escape hatch is that a whole-file overlay does not need to
  rewrite the csrc op — it REPLACES the call with Triton — but that only works if the call site is inside
  a python file you can shadow.
- caution: the %GPU screen measures DEVICE time, so it under-values a cluster of many tiny same-seam
  kernels whose real cost is HOST DISPATCH COUNT (e.g. a MoE routing/metadata chain: 5–7 launches ×
  every layer × every step, each at or barely above the dispatch floor). For those, also verify by launch count, not just %GPU — and
  do NOT schedule them as standalone extractions/unittests: a CUDA-event harness scores a launch-collapse
  ~1.0× by construction. Fold them into the whole-file overlay of the seam that already owns them and let
  the e2e gate judge.
- caution (issue it as the RIGHT KIND of task): a launch-bound cluster nominated as a `kernel_candidate`
  goes to the extraction lane, which correctly returns `editable:false / dead_end` because no
  `module:attr` honors a `call(args) -> tensor` contract — and a whole round is spent measuring nothing.
  Seen three times across two models. Nominate it instead as an EXTENSION of the existing
  whole-file overlay task for the seam that owns the launches (same `candidate_bind`, kind=module),
  with the e2e serving A/B as the only gate; and read a prior `dead_end` on such a direction as
  "never measured", not as "no opportunity". Concrete overlay directions for a MoE chain: fuse
  SiLU-and-mul into the gate_up grouped-GEMM epilogue, fold the top-k weighted sum into the down-GEMM
  epilogue, collapse align/gating metadata launches into the dispatcher, keep permuted activations in
  place to kill staging copies.
- verify (THE SCREEN THAT DECIDES WHICH LANE — per-launch us vs the DISPATCH FLOOR, not %GPU): measure the
  ROCm dispatch floor directly from your own trace — take a dozen structurally unrelated small decode
  kernels (fills, copies, reduces, topk, reshape_and_cache, both rms_norms) and read their per-call time.
  They pin at the SAME **per-launch time regardless of size** (4 KB and 0.4 MB alike — three orders of
  magnitude apart in achieved bandwidth), and that number IS the floor. Then:
  · a member **at** the floor is pure `launches x fixed dispatch cost` — a rewrite buys ~1.00x and a
    device-time harness is blind to the only real lever ⇒ overlay-fold / cudagraph coverage ONLY, never a
    solo extraction. Its %GPU is a launch-count budget, not device work.
  · a member **well above** the floor is genuine device work and IS worth a solo extraction even at a small
    %GPU. Worked counter-example on the same MoE seam: a single-workgroup `pack_bitmatrix` prologue at
    **4x the dispatch floor per launch** (fixed cost at every M because `grid=cdiv(M,512)`=1 CTA) was extracted,
    re-gridded + node-fused, and gated at **+12.01% e2e byte-exact from only 5.05% GPU** — see
    [[moe-mxfp4-grouped-authored-gfx950-vllm]]. So "it's in the small-kernel cluster" is NOT by itself a
    reason to decline; "it is already at the dispatch floor" is.
  · after such a win, re-run the screen: the survivors in that cluster were all AT the floor (18% GPU
    over ~11 kernels / ~25k launches) ⇒ correctly re-routed to fusion, with no second extraction funded.
- caution (a compile-graph-internal kernel is never bindable): under `@support_torch_compile`, Inductor-
  generated kernels (`triton_poi_fused_*`) have no source file in the install and are re-emitted under a new
  name on every compile ⇒ `editable:false` by inspection of the decorator alone. Likewise an ATen
  TensorIterator internal (`elementwise_kernel_manual_unroll`, `vectorized_elementwise_kernel`) launched by
  bare `Tensor.copy_`/`fill_` from dozens of unrelated call sites has no `module:attr` that lands on exactly
  that device kernel — one such "cluster" spanned 16 distinct shape cases across attention metadata, MoE
  routing indices and bitmatrix scratch, i.e. it violates op-identity at the root.
- caution (dtype-independent): the pattern is NOT quant-format-specific — it reproduced identically on a
  bf16 fused-MoE (Mixtral-class, TP8) where 5 peripheral kernels held 11.5% GPU across ~1.8–2.0k launches
  each, and on an mxfp4 grouped MoE at 5.7% GPU. Expect it on any fused-MoE deployment.
- caution (a FULL-cudagraph-captured member is a different, harder case — budget for it or skip it): when
  the linear-attention/gated-delta wrapper is captured into a FULL cuda graph, (a) its *reweighted* %GPU
  can overstate the raw trace share ~3× (6.23% reweighted vs 1.98% attributed) — cross-check both before
  funding a round; and (b) the Python wrapper is only entered during the engine's EAGER dummy/warmup runs,
  where every state-slot index is the NULL padding id, so a live I/O capture records an ALL-ZEROS golden
  that `return zeros` would pass. Use the live capture ONLY to pin geometry/dtype/scale/flags, then
  SYNTHESIZE in-regime inputs (real per-layer params from the checkpoint, a warm scattered state pool, a
  realistically NULL-padded batch), take the golden from the installed kernel, and CROSS-CHECK it against
  an independent fp32 reference implementation so a broken baseline cannot become the oracle. Gate the
  in-place state WRITE with a multi-step sequence case on one shared buffer (otherwise a candidate that
  skips the store — about half the traffic — still passes), and under `torch_compile` expose the launcher
  as a `torch.library.custom_op` with a `register_fake`: dynamo cannot fullgraph-trace a raw Triton
  launcher or the importlib leg selection, and `disable`/`suppress_errors` do not help. This mirrors
  deployment, where the engine already splits the inductor graph around the same op.
- caution (capture cost): never snapshot a paged/recurrent STATE POOL wholesale. Dumping the full pool per
  call across TP workers wrote a 64 GB then 127 GB reference blob to shared storage and stalled the server
  on its shm broadcast. Slice the pool to the touched slots first.
- PAYOFF CONFIRMED (the route works, and it is where the e2e came from): on bf16 Mixtral-class MoE
  (gfx950, vLLM 0.26, TP8) the whole-file overlay of the fused-MoE module was accepted at +6.15% e2e,
  and the post-hoc reprofile attributes it to the CHAIN, not the head: the C++
  `moe_align_block_size_small_batch_expert` (3.23% GPU) was replaced by a Triton
  parallel align kernel (1.96% GPU) — the cluster shrank 11.48% → ~10.4% and total GPU time in the
  window fell 4.0% — while the head `fused_moe_kernel` per-launch time was unchanged (identical both
  legs, 0.64 of the memory roof). Corollary worth remembering: after such a win a head's %GPU can RISE
  (35.0% → 37.9%) purely because the DENOMINATOR shrank — read per-launch us, not share, to decide
  whether a kernel actually got faster.
- 🆕 verify (THE CHEAPEST SCREEN — ask whether a CONFIG lever already deletes the chain, before funding
  any overlay round): a peripheral MoE/routing/quant chain exists only because a particular *backend* is
  live. If the config lane has (or could have) a whole-backend swap for the owning op, that swap may
  remove the entire chain at zero kernel cost, and any overlay you author for it becomes dead code.
  MEASURED: on a hybrid linear-attn fp8 MoE (gfx950, vLLM 0.26, TP2) a 6-op routing/quant cluster at
  8.69% GPU / ~30k launches was DECLINED as an extraction on disqualifiers (d)+(e)+out-param+device-time-
  blindness, at a cost of 0 servers / 0 captures / 0 task dirs — and the same round's accepted
  `VLLM_ROCM_USE_AITER=1` config then deleted all six kernels outright (**−77% GPU time over the six**; a full-trace name
  scan confirmed absence, not just a drop out of the Top-25). So the ordering rule is: **config-lane
  backend swaps FIRST, re-profile, and only then decide whether the surviving chain is worth an overlay.**
  Corollary for the same run: the fused replacement's GEMM was NOT faster (identical us/layer, 97.3% of
  the HBM roof on both legs) — every gram of the win was the chain, which is exactly this card's thesis
  arriving through the config lane instead of the kernel lane.
- caution (device-time timers, restated precisely for a host-dispatch cluster): 2.84% GPU over 15360
  launches is a sub-floor slice of DEVICE time per launch against a dominant per-launch HOST dispatch cost. The
  mandated `time_op` is a CUDA-event device-time timer that EXCLUDES host dispatch by design (an
  anti-exploit rule), so a correctly-built isolated harness will score such a cluster ~1.0× no matter how
  good the candidate is. That is the instrument measuring the wrong axis, not a no-win — the win is
  launch COLLAPSE and it is only visible in an e2e serving A/B.
- source: exp/e2e_*Qwen3.5-27B*/ 2026-06-05 / 06-07 / 06-09;
  exp/e2e_*gpt-oss-120b*/ 2026-08-20 (the launch-count caution + the mis-routed-task-kind caution, both
  rounds; see [[moe-mxfp4-grouped-authored-gfx950-vllm]]);
  exp/e2e_*Mixtral-8x7B*/ 2026-08-21 (bf16 MoE gfx950 vLLM TP8: architect declined the extraction
  up-front on the on-disk csrc/out-param evidence, costing 0 server rounds; the same round's
  overlay-extension then banked the +6.15% e2e above);
  exp/e2e_*Qwen3.5-27B-FP8*/ 2026-08-22 (7th confirm — gated-delta packed DECODE step on a hybrid
  linear-attn fp8 model, TP4 gfx950 vLLM: the FULL-cudagraph cautions above; the seam was reachable but
  the round ended without a measured e2e number);
  exp/e2e_*gpt-oss-120b*/ 2026-08-23 (8th confirm, mxfp4 MoE TP2 gfx950 vLLM: the dispatch-floor screen and
  its POSITIVE side — `pack_bitmatrix` at 4x the floor extracted solo and accepted at +12.01% e2e; two
  same-round clusters declined `editable:false` purely on source inspection at a cost of 0 server rounds
  and 0 captures — a routing-metadata chain on the dataclass / carved-allocation / closure-launcher
  disqualifiers, and an elementwise+inductor-pad cluster on op-identity + compile-graph-internal);
  exp/e2e_*Qwen3.5-122B-A10B-FP8*/ 2026-08-23 (9th confirm, fp8-blockscale MoE TP2 gfx950 vLLM: the
  aggregate-row (d) and compiled-csrc (e) disqualifiers; declined at 0 server rounds / 0 captures; the
  config-lane aiter swap then deleted the whole 6-kernel chain (**−77% GPU time**), while the fused GEMM
  that replaced the two Triton launches was exactly as fast, both legs at 97.3% of the HBM roof);
  same eval dir, round 2 (10th confirm, the STOP SIGNAL: post-accept the only >5% heads are an
  HBM-pinned fused MoE and an interconnect-pinned TP all-reduce, largest editable head 3.56%,
  kernel-lane ceiling ~+1.5% against a 0.5% noise band)
