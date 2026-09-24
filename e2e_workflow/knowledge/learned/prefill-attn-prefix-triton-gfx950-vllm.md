---
key: dense GQA chunked-prefill attention (prefix_prefill) · gfx950 · vLLM
type: routing
confidence: ★★★
confirms: 8
effect: no op-level env/flag lever exists — the live path already IS the editable in-tree Triton kernel, so Tier-C rewrite is the only route. Head 8.6–25% GPU across 6 models. FIRST e2e TRANSFER MEASURED (Mixtral-8x7B gfx950 TP8, 8.6% head): iso 1.45× and 1.35× IN SITU, yet e2e only +0.5% (marginal) — the head is prefill-only and the ISL=OSL=1024/conc=64 run is decode-dominated; and the split-KV rewrite that produced the speedup BROKE greedy byte-parity (7/12 prompts) and was rejected. 7th confirm (Qwen3-14B-FP8 TP1, 6.01% head) found the REAL, PARITY-SAFE lever: an XCD grid collision (grid dim0=batch varies fastest, but only prefill seqs work -> gcd(batch,num_xcd) XCDs; S=16/32/64 are 4-8x slower than S=15/17 at identical shape) -> remap the grid to active (seq,tile) pairs. Projected ~3.8x serving-wtd / +4.4% ceiling; MEASURED (e2e gate now CLOSED) iso 1.874x -> e2e +1.543% at the integrate gate, i.e. the projection was ~2x optimistic because the collided buckets only partly recover, but the lever is REAL and clears a 1% noise band at a 6% head. Accepted as STACK (cand_med > ref_med, ranges overlap at 2 repeats) into a stack that the Director validated same-session at 1.7706x / validated_win, parity pass. 8th confirm (Qwen3.5-122B-A10B-FP8 TP2, 6.21% head, NON-pow2 KV page 2096) is the SECOND e2e-accepted instance and it came from DELETING the `is_pow2` tile branch rather than from a new kernel: identical patch scored -0.222% (rejected) with the branch in place and +1.2249% (accepted, disjoint) with it removed, iso 2.579x at the LIVE page size vs 1.194x at the harness's pow2 default, inside a +3.95% ceiling, riding a Director-validated 1.2081x / validated_win.
last_seen: 2026-08-24
8th confirm ADDS THE PRE-FLIGHT EVERY TILE TUNE ON THIS OP NEEDS, **AND THE FIX**: `context_attention_fwd` picks its tiles from a HARD-CODED two-way branch on whether the KV-cache `block_size` (`v_cache.shape[3]`) is a power of two — pow2 -> BLOCK_M=128/BLOCK_N=64, non-pow2 (hybrid/Qwen3-next-class, e.g. 544 or 2096) -> BLOCK_M=32/BLOCK_N=32. A rewrite that ships tuned `BLOCK_M/BLOCK_N` constants is DEAD CODE on the non-pow2 branch (measured: e2e -0.22%, rejected). That branch is PERFORMANCE-ONLY — the paged loop always walks the cache in `TRITON_BLOCK_SIZE=32` token tiles and resolves each token by `//`/`%`, so the tile shape is correctness-independent of the page size — so **DELETE the branch and apply the tuned tile unconditionally**: same patch, same run, re-gated at **+1.2249% e2e (disjoint, ACCEPTED into a Director-validated_win 1.2081x stack)**. Read the live tile values off the in-worker banner before AND after.
---
# vLLM V1 ROCM_ATTN chunked-prefill — the live path IS the editable Triton `_fwd_kernel`

- path: (1) the head is `vllm.v1.attention.ops.prefix_prefill:context_attention_fwd` (the `_fwd_kernel`
  flash-attn), routed via `chunked_prefill_paged_decode` when max_query_len>1. (2) Do not expect an
  op-level env winner: the aiter/CK cross-backend swap is a SERVER flag (`--attention-backend`), i.e.
  the Config Tuner's job, so `op_bench.py:bench_attn` only validates the oracle. `current correct
  rel=0`, `winner=none`, `harness_suspect=false`, smoke speedup ~1.0 is the EXPECTED reading here, not
  a fault — target==baseline because both are the same live seam. (3) Go straight to Tier-C Triton
  `route=rewrite` (`mode=optimize`, an editable impl exists): autotune BLOCK_M/BLOCK_N, num_warps,
  num_stages, waves_per_eu, matrix_instr_nonkdim for GQA 32q/8kv, head_dim=128, causal.
- expected gain: purely Amdahl-scaled off the head share, so screen before spending a round —
  at 8.58% head a 1.2×/1.5× buys +1.5%/+2.9% e2e; at 16.94% → +2.8%/+5.6%; at 25.11% → +4.4%/+9.1%.
- apply: in the served regime **context_len=0** (fresh chunks) → the paged K/V cache is NEVER
  dereferenced (the context loop is `range(0, ctx=0)`). Optimize the current-chunk causal QKᵀ·softmax·V
  path, not the prefix-cache loop. Oracle cases are many SHORT varlen prefill chunks (M∈{16..336}), not
  one long sequence — the win comes from small-tile efficiency.
- verify: judge against the immutable `unittest.py` (bf16, 3 random draws) with
  baseline_callable==target_callable==`context_attention_fwd` — never a naive scaffold. Prefill-ONLY
  head (decode is served by a separate paged_attention kernel) → `served_regimes=['prefill']`, no decode
  bucket. Must stay CUDA-graph capture-safe (no host sync).
- caution (weight the head by PHASE, not just by %GPU): this kernel is PREFILL-ONLY, so its profile
  share is spent in the few prefill steps of a decode-dominated serving mix (e.g. 52 decode vs 12
  prefill steps at ISL/OSL 1024/1024, conc 64). A clean 1.35× in situ there moved e2e ~+0.5%, i.e.
  ~stack-grade at best. Before spending a round, multiply the Amdahl ceiling by the prefill fraction of
  the served steps — %GPU alone over-ranks it.
- caution (split-KV / flash-decoding reassociation is a PARITY risk on a NON-quant kernel): splitting
  the KV loop and re-associating the softmax accumulation is the obvious source of the 1.4–1.5× iso
  win, but it flips argmax on greedy decoding (max abs err 5e-4..2e-3 — passes a loose 2e-2 unittest
  tol yet diverges on 7/12 greedy prompts, first divergence ~100 chars in, reproducibly). Outputs stay
  fluent, so ONLY a byte-parity probe against a fresh no-overlay baseline catches it. Also verify the
  baseline's own determinism first: vLLM is not self-deterministic across repeated probes on one server
  (batch-shape nondeterminism), so compare candidate vs the SAME probe slot (first-probe vs first-probe),
  which IS reproducible across independently launched servers. If a rewrite needs the reassociation,
  route it as a corrective re-author that preserves accumulation order (or gate it on accuracy).
- caution: flydsl is a GEMM DSL, not an attention author target; ck/hip are absent-gated on this image
  (no ckProfiler). e2e rebind seam = `context_attention_fwd`.
- caution (8th sighting — do NOT read a post-swap %GPU RISE as new opportunity): after a broad
  `VLLM_ROCM_USE_AITER=1` swap shrank total window GPU time 16.3% on a hybrid linear-attn fp8 MoE
  (gfx950 vLLM 0.26 TP2), this kernel's share rose 5.17% → 6.21% at an **unchanged absolute GPU
  time (1.005×)** — pure denominator illusion, no new headroom. It became the largest EDITABLE head only
  because the two ops above it are now non-editable/config-bound (aiter asm fmoe at the HBM wall, aiter
  TP all-reduce at the interconnect roof). Apply the card's own screen to the ABSOLUTE GPU time and the
  prefill step fraction: 6.2% head, roofline_pct 0.128, latency/occupancy-bound ⇒ ~+4% e2e ceiling,
  i.e. still under the ">=15% head" bar this card sets for funding a full round.
- source: 2026-08-13, gfx950 / vLLM 0.26.0 — Llama-3.1-8B TP1 (head 14.98% and 16.94%), Qwen3-0.6B TP1
  (25.11%), Qwen3-8B TP1 (16.59%, GQA 32q/8kv hd128), Mixtral-8x7B TP8 (8.58% prefill, per-rank GQA
  4q/1kv). All five: op_bench current correct, winner=none, harness_suspect=false — as predicted.
- source (6th confirm, first e2e transfer): exp/e2e_*Mixtral-8x7B*/ 2026-08-21, gfx950 / vLLM 0.26.0
  TP8, head 8.60% GPU. Tier-C split-KV Triton rewrite bound as a whole-file overlay of the
  prefix_prefill module, stacked on an accepted fused-MoE overlay; engagement proven in all 8 TP
  workers. iso 1.453×; reprofile shows **1.347× IN SITU** per launch (the in-situ number
  corroborated the isolated one); e2e +0.52..0.55% steady-state (ceiling +5.7%); REJECTED on byte-parity,
  not on throughput. Head share fell 8.60% → 7.26% after the change.
- **NEW LEVER (7th confirm, and the biggest one found so far) — XCD GRID COLLISION, and it is
  BYTE-PARITY-SAFE.** `_fwd_kernel`'s grid is `(batch, head, cdiv(max_input_len, BLOCK_M))` with
  `pid0 = batch` varying FASTEST, but in a chunked-prefill batch only the PREFILL sequences' `pid0`
  values do work (`SKIP_DECODE` early-returns the rest). The working workgroups are therefore STRIDED
  BY `batch` in the linear workgroup id, so on a multi-XCD MI3xx they land on only `gcd(batch, num_xcd)`
  XCDs — with ONE prefill seq in a batch of 8/16/32/64 the entire kernel runs on a SINGLE XCD.
  Measured (gfx950, same shape chunk=306 ctx=512 1 prefill seq, eager ms by batch):
  S=1 0.105 · S=2 0.104 · S=4 0.153 · S=8 0.268 · S=12 0.154 · S=15 0.103 · S=16 0.465 · S=17 0.104 ·
  S=32 0.845 · S=64 0.842. A 4–8× swing driven purely by `gcd(batch,8)`, reproducible within and across
  processes. It shows up directly in the oracle buckets: S=16/M=337 and S=64/M=417 cost ~7× and ~8× MORE
  than the *larger* S=1/M=1024 and S=15/M=306 cases respectively.
  FIX: flatten to a 1-D grid over ACTIVE `(prefill-seq, tile)` pairs, or swizzle `program_id(0)`, or
  simply reorder the grid so `head` (=40) is the fastest-varying dim. This is a pure SCHEDULING change —
  it does not touch the softmax accumulation order — so unlike the split-KV route above it should stay
  BITWISE identical. **Prefer it over split-KV for this kernel: same class of win, none of the parity
  risk.** Keep it CUDA-graph-capture-safe: the grid must remain a pure function of the host-side args
  (compute the active-seq remap host-side from `b_start_loc`/`b_seq_len`, never a device-side scan).
  Projected serving-weighted iso from the frozen oracle weights if the collided buckets recover to their
  non-collided twins: ~3.8× ⇒ +4.4% e2e ceiling at a 6.01% head.
- correction to the `apply` bullet above: **`context_len=0` is NOT universal.** On this deployment the
  served buckets carry real prefix (`ctx` 224–1008 on 6 of 7 cases), so the paged-KV gather loop IS
  entered and dominates the small-M buckets (M=16/ctx=1008 costs MORE than M=1024/ctx=0).
  Check the extracted cases' `max_ctx` before assuming the prefix loop is dead code.
- kv-fp8 (`--kv-cache-dtype fp8_e4m3`) is NOT a lever for THIS kernel: measured 0.990–1.008× (it only
  halves the cached-prefix read of a latency-bound kernel) while breaking bf16 tolerance. Contrast with
  the decode paged-attn sibling where kv-fp8 is worth 1.5–1.6×.
- source (7th confirm): Qwen3-14B-FP8 gfx950 / vLLM 0.26.0 TP1, head **6.01%** GPU, GQA 40q/8kv hd128,
  block_size=16, ROCM_ATTN, bf16 KV. op_bench `measured=false` / `winner=none` / `harness_suspect=false`
  again (attn bake-off is a server-flag delegation, as predicted). No env/flag winner; routed to Tier-C
  Triton `route=rewrite` on the XCD-remap lever. **e2e RESOLVED (same run, final gate): the XCD-remap
  rewrite measured iso 1.874x and gated at +1.543% e2e** (2 repeats,
  ranges OVERLAP -> STACK not standalone-accept), inside the +2.88% Amdahl ceiling for a 6.01% head at
  1.874x; it rode into a Director-validated_win stack (1.7706x overall, parity pass). So the XCD-collision
  lever is confirmed real and parity-safe enough to carry, but at a single-digit head it buys stack-grade
  points — spend the round only when this kernel is >=15% GPU, and expect ~half the projected recovery
  because the non-collided twin timing is an upper bound, not an attainable target.
- **NEW LEVER (8th confirm) — DELETE the `is_pow2` tile branch; it is a performance-only gate that
  starves every hybrid-model deployment.** `context_attention_fwd` selects tiles from a hard-coded branch
  on `is_pow2(v_cache.shape[3])` (the KV block size): pow2 -> `BLOCK_M=128, BLOCK_N=64`; non-pow2
  (hybrid / Qwen3-next-class page size, e.g. 544 or 2096) -> `BLOCK_M=32, BLOCK_N=32`. `BLOCK_M` tiles the
  QUERY rows and `BLOCK_N` the NEW unpaged K/V of the chunk; NEITHER indexes the paged cache, which is
  always walked in `TRITON_BLOCK_SIZE=32` token tiles with per-token `//`/`%` resolution. So the branch
  never protected correctness — it only forced a 32x32 tile that quadruples the grid's z-dim, with every
  extra query tile still paying the FULL context loop. Applying the tuned tile UNCONDITIONALLY is the
  accepted fix: live banner `BLOCK_M=128 BLOCK_N=64 ... real_block_size=2096`, **iso 2.579x at the live
  page size, e2e +1.2249% disjoint, TTFT −5.5%, parity pass** — versus **-0.222% (rejected)** for the
  identical patch with the branch left in place. Cheap pre-flight either way: print the chosen
  `BLOCK_M/BLOCK_N` AND `real_block_size` in the ENGAGED banner and read them from the live worker log.
- **caution (8th confirm — pin the ISOLATED HARNESS to the live branch or its number is fiction).** The
  same kernel measured **1.194x** at the harness's default pow2 `bs=16` and **2.579x / 2.578x** at the live
  `block_size=2096 / 544` (correctness PASS, max_rel 4.9e-3 vs tol 0.02). The 1.21x that was reported —
  and that sized the whole round at a +1.09% ceiling — was a measurement of a code path the server never
  executes; the true ceiling was +3.95%. Whenever a kernel has a geometry-dependent branch, the harness's
  `v_cache` block dim (or equivalent) must be set from the live capture, not left at a convenient default.
- **caution (8th confirm — a REJECT whose banner shows the tuned values did not bind is a CORRECTIVE
  candidate, not a dead end).** Round-1 integrate was clean in every respect (provenance verified,
  engagement 2/2 banners vs 0/0, both legs on the same accepted stack) and still returned -0.222%. The
  banner said why in one line (`BLOCK_M=32 BLOCK_N=32`). One corrective A/B on the same patch converted it
  into an accept. Do not close a direction on a null delta until the banner has confirmed the tuned VALUES
  — not merely the module — were live.
- **caution (8th confirm — on a PREFILL-ONLY head, size and judge by TTFT; the e2e throughput number is
  small even when the kernel gets hugely faster).** Same run, four instruments and they spanned 2x in both
  directions: isolated-at-wrong-branch 1.21x, isolated-at-live-branch **2.579x**, in-trace per-launch
  **2.55x** (identical 108 launches and identical M distribution = pure per-call speedup),
  and e2e **+1.22%** once bound (vs **-0.22%** unbound). +1.22% against a +3.95% Amdahl ceiling at a 6.21%
  head is what a prefill-only kernel yields in a decode-dominated mix: the win lands on TTFT
  (-5.4% integrate; -7.1% in the window bench) with TPOT essentially flat.
  Instrument trust order that held: **Amdahl ceiling > e2e gate > in-trace us > isolated harness** — and
  the isolated harness is only usable at all once pinned to the live branch. Funding bar stays: a full
  round is worth it at >=15% head; below that expect stack-grade points, and decide on TTFT.
- source (8th confirm): exp/e2e_*Qwen3.5-122B-A10B-FP8*/ 2026-08-23..08-24, gfx950 / vLLM 0.26 TP2, hybrid
  linear-attn fp8 MoE, ISL/OSL 1024/1024 conc 64, head 6.21% GPU, non-pow2 KV block (`block_size=2096`),
  stacked on an accepted aiter fp8-blockscale env stack. TWO A/Bs on the SAME patch: round 1 (branch left
  in place, banner 32x32) ref 2949.60 -> cand 2943.05 = **-0.222%, rejected**; corrective round 2 (`is_pow2`
  branch removed, banner 128x64 @ real_block_size=2096) ref 2929.40 -> cand 2965.28 = **+1.2249%,
  ACCEPTED**, distributions disjoint (cand_min 2964.31 > ref_max 2943.55), engagement 2/2 banners vs 0/0,
  provenance verified (reference_io sha256 match, site-packages unmodified, patch applies clean to pristine
  `kernel_src`). Parity gated as ACCURACY, not byte-exact, because the REFERENCE is not self-reproducible
  under continuous batching (ref-vs-ref 2/12 byte-identical, mean seqmatch 0.72) and the candidate lands
  INSIDE that envelope (matches the ref's 2nd probe on ~5/12, more than the ref's own two probes match each
  other); content spot-checks correct, no degeneration. Rode into a Director same-session
  `validated_win`: **1.2081x**, disjoint, TTFT -19.83%, TPOT -17.06%, parity
  pass. Post-round-1 the head had read 2.55% GPU (denominator + the real per-launch win), i.e. below the
  5% bar — do not re-nominate it on this deployment.
- caution: also verify the rebind seam is the CALLER module, not just the defining one. The head is
  defined in `...ops.prefix_prefill:context_attention_fwd` but the live dispatch imports it into
  `...ops.chunked_prefill_paged_decode`; rebinding only the defining module leaves the engaging call
  site pointing at the stock function. Rebind BOTH and require the in-worker ENGAGED banner before the
  timed leg.
