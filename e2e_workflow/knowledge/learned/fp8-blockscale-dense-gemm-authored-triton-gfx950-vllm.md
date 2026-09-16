---
name: fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm
description: Tier-C authored Triton replacement of the aiter/CK fp8 128x128 block-scale dense GEMM, rebound by a sitecustomize overlay — iso 2.69x, Director-validated +15.7% e2e where the env CK tune of the SAME op returned +0.18%.
keywords: [authored-triton, tier-c-author, fp8-blockscale, dense-gemm, sitecustomize-overlay, rebind-seam, accuracy-gate, amdahl, cudagraph-safe, split-k, lazy-import, launch-bound]
kernels: [gemm_a8w8_blockscale, kernel_gemm_xdl_cshuffle_v3, _w8a8_triton_block_scaled_mm, _gemm_a8w8_blockscale_kernel, _gemm_blockscale_splitk_kernel, _splitk_reduce_kernel]
platforms: [gfx950]
kernel_class: dense_gemm
regime: both
key: fp8 a8w8 128x128 block-scale DENSE GEMM, Tier-C AUTHORED replacement of the library CK kernel (not the env tune DB) · gfx950 · vLLM
type: lever
confidence: ★★★
effect: iso 2.687x serving-weighted (decode guard 1.23, correctness PASS); e2e +16.77% at the integrate gate and +15.72% (1.1572x, non-overlapping, TPOT -15.4%) Director-validated same-session, parity PASS under the accuracy gate. Decisive datum: on the SAME op in the SAME run the env CK per-shape tune measured a comparable iso 2.577x yet moved e2e only +0.18% (rejected) — isolated rank did NOT order e2e, so budget the author lane, do not stop at the tune. 2nd CONFIRM on a different model/box (hybrid linear-attn 27B-FP8, TP4, ISL/OSL 1024/1024 conc 64, 41.22% head): iso 1.899x serving-wtd -> e2e +15.81% at the integrate gate (non-overlapping, TTFT -24%, TPOT -12.9%), vs the same run's env CK tune at iso 1.523x/+6.46% and a flydsl author at iso 1.735x/-51.9% (rejected) — the author lane won on both boxes and the backend ORDER (triton > ck-env > flydsl) was identical. FINAL-GATE RECONCILIATION for that 2nd confirm: the run closed `flagged`, NOT validated_win — the mechanism and the correctness gate were independently reproduced by the Director (rebind banner + live ENGAGED calls in all 4 TP workers under its own launch; same-session accuracy probe 24/30 base vs 26/30 final, output_parity pass, kind=accuracy; byte-parity unusable on this async-scheduled fp8 stack), but ZERO independent timed samples were obtained, so no run-level speedup is certified. The quoted whole-run headline (1.0719x, TTFT -32.6%, TPOT -6.0%) is CROSS-session against a 21-h-old baseline, and a same-day no-overlay leg at the identical config read 7.4% BELOW that baseline, so the cross-session ratio is not trustworthy in either direction; only the same-session integrate pair carries evidential weight on this box. 3rd CONFIRM, same model class but a DECODE-heavy workload and a bigger head (dense 14B-FP8, TP1, ISL/OSL 1024/1024 conc 64, head 67.96%, stock path = UNTUNED in-tree Triton `_w8a8_triton_block_scaled_mm` with no shipped device config): iso 2.368x serving-wtd (decode M64 2.53x, M1 3.36x) -> integrate gate +70.62% (strictly non-overlapping, TPOT -45%), and the run it anchored was Director-validated same-session at 1.7706x (+77.1%, non-overlapping, TTFT -36%, TPOT -43.7%), validation_status validated_win, parity pass under the accuracy gate (gsm8k 5-shot n=400: 0.9025 base vs 0.8950 cand, inside binomial noise). Backend ORDER reproduced a THIRD time and is now the stable prior: authored triton (2.368x / +70.6%) > env aiter Triton->CK swap (1.792x / +50.3%) > authored flydsl (1.551x / -18.0%, rejected). Here the realized e2e (+70.6%) EXCEEDED the nominal Amdahl ceiling (+64.7%) because the stock baseline was an untuned Triton path (no per-device config) and the iso number was measured under co-tenancy — when that happens, re-check accuracy on a larger sample against a FRESH no-overlay baseline before believing it.
confirms: 3
lifecycle: active
last_seen: 2026-08-22
---
# gfx950 vLLM fp8 block-scale dense GEMM — the AUTHORED Triton lever, not just the tune DB

- lever: `edit=N` (vendor CK library) is not "skip". When the fp8 block-scale GEMM is the profile head,
  author a full Triton kernel for it and REBIND `aiter:gemm_a8w8_blockscale` — the attribute vLLM
  re-imports on every call, so a module-attr rebind reaches the live seam with no source patch.
- apply: overlay-only `sitecustomize.py` on `PYTHONPATH` (site-packages untouched, fully reversible).
  Winning kernel shape, for seeding: rank-1 collapse + hoist of the block-scale dequant out of the K
  loop (`EVEN_M/N/K` specialization) to reach native fp8 MFMA, `.cg` on the B/weight global loads
  (decode M=64 -17%, M=1 -35%), a per-(M-bucket,N,K) static config table, bf16 split-K partials, and a
  coalesced `x_scale` transpose done in the WRAPPER. Technique detail lives in the kernel sink
  (`kernel_workflow/knowledge/learned/`, quantized-gemm · gfx950 cards) — this card records the e2e side.
- expected gain: pure Amdahl — 51.95% head x 2.687x iso = +48.4% ceiling, realized +15.7%. Realization
  is ~1/3 of the ceiling because the iso win is prefill-heavy while the mix is decode-weighted; size a
  round by the ceiling but expect the regime discount.
- verify: (a) engagement — rebind banner in the server log plus a nonzero ENGAGED call counter (32768
  here) BEFORE spending an A/B; (b) PIECEWISE cuda-graph capture must complete (no host sync on the
  hot path); (c) gate parity by ACCURACY, not bytes — a re-implemented fp8 block-scaled GEMM changes
  tiling/accumulation order, so byte-greedy read 4/12 against a deterministic baseline while gsm8k
  5-shot on a seeded n=200 subset was 0.810 -> 0.790, McNemar p~0.57 (inside 1 sigma).
- caution: also verify the overlay imports LAZILY. An eager `import aiter` at interpreter startup inside
  `sitecustomize.py` makes every python3 on that PYTHONPATH re-enter the overlay — aiter shells out to
  `rocm_agent_enumerator`, itself a python3 script, which re-runs sitecustomize: a self-replicating
  process storm (~3.5k procs, load ~200) that depressed the whole box ~13%. Pin `GPU_ARCHS` or early-return
  for non-worker interpreters, and add ~4-5 min per server launch to your wall-clock budget either way.
- follow-on lever (BUDGET THE NEXT ROUND FOR IT): after the authored kernel lands, reprofile — the win is
  regime-split and it MOVES the head into your own code. Measured on the 2nd box: prefill kernel −67%,
  decode split-K path only −11%, so the new #1 is the SAME logical op, now GEAK-owned and editable. Its
  decode half elects a split-K variant plus a separate `_splitk_reduce_kernel` costing 256 launches/step at
  the dispatch floor = **9.8% GPU of the decode step** of pure bookkeeping. Killing the
  reduction (fp32 atomics / fused epilogue, or a lower SPLIT_K via the kernel's own tile knobs) is worth up
  to ~+9.8% ww at zero accuracy risk and needs no new kernel.
- caution: also verify what the speedup DE-OVERLAPS. On the 2nd box the aiter collective (+21% us/launch)
  and the linear-attention decode kernel (+17%) each got slower per launch with NO config change once the
  GEMM stopped hiding them — ~25% of the GEMM win was clawed back. Read per-launch us on the neighbours,
  not just the head, when reconciling ceiling vs realized.
- caution: also verify the overlay's rebind actually costs nothing at STARTUP under TP. Even lazily, an
  `import aiter` inside the worker took ~10 min on one image and blew vLLM's 600 s TP rendezvous
  (`DistStoreError`, 2/4 clients joined) — a meta-path finder that applies the rebind at real-import time
  fixed it (healthy cand server in 120 s). Prove engagement before the timed leg: the rebind banner plus a
  live ENGAGED counter in EVERY TP worker (>100k calls here, prefill and decode shapes both).
- caution: also verify whether byte-parity is even AVAILABLE before choosing the gate. On this stack the
  TRUE no-overlay baseline was NOT deterministic — two identical greedy/temp-0/seeded probes on the SAME
  reference server diverged on 8/12 prompts. Byte-greedy comparison is then meaningless in both directions;
  fall back to a task-accuracy probe with an explicit sigma (here 21/30 vs 23/30, inside 1 sigma of a
  30-item binomial) plus an isolated numeric cross-check against the stock op over the live (N,K)×M grid.
- caution: also verify the DENOMINATOR is same-session. This win first read as +0.86% because the
  finalize bench divided by a 22-h-old baseline on a box that had since drifted 13% slower.
- source: exp/e2e_*Qwen3-14B-FP8-vllm*/e2e_cycle0 2026-08-21..08-22 (TP1, gfx950/MI355, vLLM 0.26,
  ISL/OSL 8192/1024 conc 64, head 51.95% GPU, Director validated_win).
- caution: also verify what the STOCK path really is before sizing the ceiling. When the live seam is the
  in-tree Triton block-scale GEMM and the server logs `Using default W8A8 Block FP8 kernel config` (no
  tuned config for this device), the profile's Amdahl ceiling UNDER-states the win — the 3rd confirm
  realized +70.6% against a +64.7% nominal ceiling. Treat an over-ceiling result as a corruption ALARM,
  not a bonus: re-run the task-accuracy probe at a larger n against a fresh no-overlay baseline and
  confirm both legs emitted the identical fixed token volume (`--ignore-eos`, equal completed requests),
  so speed cannot be coming from truncated/degenerate generation. Both checks passed there.
- source: exp/e2e_*Qwen3-14B-FP8*/ 2026-08-22 (TP1, gfx950, vLLM 0.26, ISL/OSL 1024/1024 conc 64, head
  67.96% GPU, stock = untuned in-tree Triton) — 3rd confirm, iso 2.368x -> +70.62% integrate, run
  Director-validated 1.7706x validated_win; source of the over-ceiling caution and the stable backend
  order triton > aiter-env > flydsl.
- source: exp/e2e_*Qwen3.5-27B-FP8*/ 2026-08-21..08-22 (TP4 gfx950, vLLM 0.26, hybrid linear-attn fp8
  block-scale, ISL/OSL 1024/1024 conc 64, head 41.22% GPU) — 2nd confirm +15.81% integrate; run closed
  `flagged` (engagement + accuracy reproduced by the Director, no independent timed samples); source of the
  split-K-reduce follow-on, the de-overlap, the TP-rendezvous and the nondeterministic-baseline cautions.
- caution: also verify a same-session BASE leg actually got measured before quoting a run headline. A
  kernel win can be real at the integrate gate and still leave the run unvalidated: budget the Director's
  A/B as a first-class step (its own two server launches), keep the driver's port inside whatever window
  the bench harness enforces, and treat a stored baseline older than a few hours as unusable — a 21-h gap
  was worth 7.4% of box drift here, more than half the claimed run-level delta.
