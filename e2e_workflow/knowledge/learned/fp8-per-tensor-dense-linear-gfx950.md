---
key: dense_linear_fp8_per_tensor · gfx950 (MI350X) · sglang torch._scaled_mm live seam
type: routing
confidence: ★★★
confirms: 5 (mlp gate_up N=34816; qkv_proj N=7168 + o_proj N=5120 re-confirmed on independent runs 2026-08-13 and 2026-08-17; down_proj N=5120 K=17408 2026-08-14)
effect: every alternative backend LOSES to the stock hipBLASLt/Tensile seam (CK/Triton/FlyDSL 0.29–0.93x across three heads); TunableOp is the only env lever and it nets 0.98–1.05x weighted, ≤1.024x even pruned → below the identity noise band. The REAL headroom is Tier-C, and it is head-dependent: check bytes/µs AND whether the library already split K before believing either a "no headroom" or a "just add split-K" claim. The Tier-C AUTHORED Triton decode-tile overlay is now the one thing that DOES beat the seam and it REPRODUCES across runs: 1.052–1.064x serving-weighted / 1.089–1.105x device on the small-N qkv+o_proj head, bit-identical output.
last_seen: 2026-08-17
---
# per-TENSOR fp8 W8A8 dense GEMM is NOT the block-scale story — read the quant scheme first
- routing: `--quantization fp8` on a dense model (Qwen3-14B) gives **per-tensor** w8a8, and sglang's
  `fp8_utils.apply_fp8_linear` takes the `per_tensor_weights and per_tensor_activations` branch →
  **`torch._scaled_mm` (hipBLASLt/Tensile)**, *not* `aiter.ops.triton.gemm_a8w8_blockscale`. The
  fp8-blockscale CK skill's premise ("baseline is the untuned Triton default") therefore does **not**
  hold here — check `meta.quant_scheme` / `weight_block_size` before choosing the lever.
- lever worth trying: **PyTorch TunableOp** (`ScaledGemmTunableOp`) — it binds here precisely because the
  live seam IS the torch dispatch (same routing rule as the sglang-diffusion card). Tune offline, then
  **PRUNE**: only the M≈1k–2k rows won under cold-flush timing; the decode rows regressed 0.53–0.56x
  (TunableOp times HOT and 178 MB of weights fits the 256 MB Infinity Cache) and at M≥8192 TunableOp
  itself picks `Default`.
- apply: `PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME=<x>.csv`
  (torch appends the device ordinal). In-process A/B trick: `torch.cuda.tunable.enable(True/False)`
  around the candidate call, so the harness's baseline leg keeps the stock heuristic instead of scoring
  a fake 1.00x identity.
- verify: TunableOp keys on the EXACT M (`tn_N_M_K_ld_..._bias_None`), so check the live keys with
  `PYTORCH_TUNABLEOP_RECORD_UNTUNED=1` — with `--chunked-prefill-size 16384` the server issues M≈16.4k
  prefills that never match an M=1024 row (the isolated win then transfers to nothing).
- also-verified-null (all correct, all slower than the live `torch._scaled_mm`, cold-flushed device time,
  M=1/64/1024): aiter CK `gemm_a8w8` 0.57/0.49/0.89x, CK `bpreshuffle` 0.54/0.56/1.06x, aiter Triton
  `gemm_a8w8` 0.61/0.50/0.83x, FlyDSL `flydsl_preshuffle_gemm_a8` after a 72-point tile sweep
  0.56/0.61/0.96x. aiter's CK a8w8 tuner runs (101 s for 3 shapes) and its CSV engages, but the
  production op does not speed up (the kernelId is baked in at JIT codegen; only `splitK` is read from
  the CSV) and its own hot numbers still lose to hipBLASLt's. `gemm_a8w8_ASM` is int8-only, `wvSplitKQ`
  rejects N=34816, hipBLASLt bpreshuffle has 0 fp8 solutions on gfx950.
- caution: **check the roofline PER HEAD before funding OR refusing a Tier-C author lane — the answer
  flips between heads of the same op class.** Measured achievable HBM copy bandwidth on MI350X is
  4.73 TB/s. On the BIG-N head (gate_up, N=34816, 178 MB) the decode buckets already run ~3.6 TB/s
  (~76% of achievable) → ≤1.3x headroom, author lane not worth funding. On the SMALL-N heads
  (qkv N=7168 = 36.7 MB, o_proj N=5120 = 26.2 MB) the SAME kernel class runs at only 1.4 / 1.1 TB/s
  cold-flushed (2.5 / 1.8 TB/s against the live per-call profile time) = 23–54% of achievable → 1.8–3.3x
  headroom. Mechanism: hipBLASLt picks `MT64x32` with NO GlobalSplitU, so a 64xN decode output is
  224 / 160 workgroups on 256 CUs — one badly-underfilled wave (0.283 of roofline vs 0.509 for the
  same-class down_proj which does get GSU=8). K=5120 is shared by both families, so a **K-split /
  GSU decode kernel is the only way to fill the machine** — that is the Tier-C target, and it matches
  the verified gfx950 `mxfp8-linear-decode-rewrite` split-K/decode-tile Triton precedent (+21.8% e2e).
- caution: TunableOp's tuning loop times with a **4 MiB rotating buffer** (`Rotating buffer 4 MiB.
  Needed Size: 71 MiB`), i.e. HOT, which is why its picks regress under the cold-flushed harness
  (0.87x weighted). Setting `PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=1024` makes it pick different
  solutions and lifts the result to 0.96x, and PRUNING to only the winning keys reaches 1.012x
  weighted / 1.015x device — still inside the ~1.006x identity noise band, and its gain sits almost
  entirely in the M=1024 prefill rows, which a `--chunked-prefill-size 16384` server never issues.
  Treat TunableOp on this op as a measured dead end, not an unexplored lever.
- also-verified-null on the qkv/o_proj head (all cold-flushed device time, M=1/64/1024, correct unless
  noted): `preferred_blas_library('cublas')` is a NO-OP for `_scaled_mm` (1.001x, byte-identical);
  `torch._scaled_mm(..., use_fast_accum=True)` is likewise a no-op on gfx950 hipBLASLt fp8 —
  bit-identical output, 1.006–1.066x = noise; aiter CK `gemm_a8w8` 0.44–0.73x; aiter Triton
  `gemm_a8w8` 0.78–0.93x; aiter CK `bpreshuffle` 0.64–0.87x AND rel_err 1.46 (WRONG) at M=1024/N=5120;
  FlyDSL `flydsl_preshuffle_gemm_a8` over a 12-point tile sweep 0.60–0.61x at decode.
- caution: a FlyDSL tile sweep MUST gate on correctness with a NaN-safe comparison — tile 32x32x128
  is the fastest config at M=1024 (1.02–1.07x) and returns **NaN** in ~0.2% of elements; a naive
  `if rel_err > tol: skip` passes it through because `nan > tol` is False. Also note FlyDSL reports
  `mi_input_type=BFloat8Float8_fnuz` on gfx950 (an OCP-e4m3 box) and warns its latency table is
  missing, so its own tile heuristic is flying blind here.
- re-confirmed 2026-08-13 on the SAME qkv/o_proj head (9.36% GPU, Qwen3-14B fp8, gfx950), measured with the
  task's IMMUTABLE `unittest.py` as the bake-off harness (it is backend-parameterized by
  `GEAK_CANDIDATE_GEMM`, so no separate driver is needed -- the shared dense `op_bench.py` cannot express a
  per-tensor fp8 w8a8 GEMM and returns `harness_suspect` / `addmm_cuda not implemented for Float8_e4m3fn`;
  that is a known driver limitation, NOT a broken task): aiter `gemm_a8w8` 0.638 weighted (0.53 device),
  `rocblas` (preferred_blas_library cublas) 1.0026 = identity no-op, TunableOp tuned-all 1.0004,
  TunableOp PRUNED to the winning M=1/M=1024 keys 1.0088 weighted / 1.0253 device -> Amdahl ceiling 0.081%
  e2e, i.e. inside noise. Its whole gain is the M=1024 prefill rows, which a `--chunked-prefill-size 16384`
  server never issues. **Tier A/B on this seam is CLOSED; route the head straight to Tier-C.**
- caution: **read the Tensile solution NAME before picking the Tier-C mechanism — heads of this same op
  class differ in whether the K-split is already taken.** On the TALL-K head (down_proj N=5120
  **K=17408**, 89.13 MB weight, 9.13% GPU) hipBLASLt selects a **StreamK** solution
  (`..._MT128x64x256_..._SK3_SKFTR0_SKXCCM8`) and launches a SEPARATE `PostGSU8_VW4` reduction kernel
  **1:1** that pulls the fp32 partials back out of workspace and applies ScaleAB/ScaleAlphaVec. So
  "add split-K" — the winning idea on the K=5120 heads (MT64x32, GSU0) — only re-derives what the
  library already does here. The exploitable slack is instead the **HBM round trip of the partials**:
  the epilogue is 4.972 of the 27.34 µs unit (**18.2%**), and fusing it on-chip (or one-pass atomics)
  is the funded Tier-C target. Corollary for the Extractor: `pct_gpu_time` for such a head must sum the
  GEMM row AND its reduction row (both carry identical launch counts) — they are one seam, not two ops.
- caution: this tall-K head is at **71.3% of achievable HBM bandwidth** cold-flushed (4.631 TB/s
  measured copy roof; M=64 traffic floor 19.6 µs vs 27.5 µs deployed) ⇒ **hard ceiling 1.40x**, realistic
  1.15–1.25x, i.e. 1.5–2.6% e2e at 9.13%. It sits BETWEEN the two regimes the previous confirms found
  (big-N gate_up ~76% = no headroom; small-N qkv/o_proj 23–54% = 1.8–3.3x). Rule of thumb that now holds
  3/3: **measure bytes/µs per head; do not inherit a sibling head's verdict, and do not trust the
  Architect's roofline card** (it claimed 1.77x here; the measured answer is 1.40x).
- re-confirmed 2026-08-14 on the down_proj tall-K head, again with the task's own `unittest.py`
  (`GEAK_CANDIDATE_GEMM`) because the shared dense `op_bench.py` still returns `harness_suspect` /
  `addmm_cuda not implemented for Float8_e4m3fn` — it builds `F.linear(A,B)` and structurally cannot
  express a per-tensor fp8 w8a8 GEMM. Numbers (weighted / device): identity 1.0057 / 1.0049 (the noise
  floor), aiter CK `gemm_a8w8` **0.3693 / 0.3027** (0.29–0.62x per bucket — it loses HARDER at K=17408
  than at K=5120), `preferred_blas_library('cublas')` 1.0033 / **1.000 byte-identical** (3rd confirm it
  is a no-op for `_scaled_mm`), TunableOp tuned at M=1/64/1024 with
  `PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=1024` **0.981 / 0.9784 — a net LOSS**. Per bucket TunableOp is
  M=1 1.085x, **M=64 0.976x (the steady state, and it REGRESSES)**, M=1024 1.236x, M=16384 0.995x:
  the identical hot-tuning/cold-serving signature as the other two heads, with the whole gain again
  parked in the M=1024 prefill rows a `--chunked-prefill-size 16384` server never issues. Pruned to the
  M=1 key alone it recomputes to 1.024 weighted = still inside the band. **Tier A/B is CLOSED on this
  seam for a third distinct head — stop re-benching backends and fund Tier-C.**
- **KNOW THE TRITON FLOOR BEFORE FUNDING A TRITON AUTHOR LANE, and seed from it.** aiter's PRODUCTION
  editable Triton a8w8 (`/sgl-workspace/aiter/aiter/ops/triton/gemm/basic/gemm_a8w8.py`, split-K with a
  separate `_gemm_a8w8_reduce_kernel`) measures **0.7785 weighted / 0.7218 device** on the tall-K head:
  0.914x at M=1, **0.743x at the M=64 steady state**, 0.653x at M=1024, 0.557x at M=16384. So a Tier-C
  Triton lane does not start at parity — it starts ~26% BEHIND at the bucket that carries the weight, and
  must recover that before the fused-reduction idea buys anything. Fund the lane (the mechanism is real
  and the M=1 gap is only 9%), but (a) seed the optimizer from this production file, NEVER from a naive
  from-scratch GEMM, and (b) require the authored kernel to keep a **bit-identical fall-through to
  `torch._scaled_mm` for M≥128**, where the library scales to 2.0–2.6 PFLOP/s and Triton is 0.56–0.65x.
- note: the **fp8 CK-blockscale skill does NOT apply to any of these heads** and skipping it is not a
  defection. It targets `gemm_a8w8_blockscale` (weight_block_size set, live seam =
  `aiter.ops.triton.gemm_a8w8_blockscale`, baseline = the untuned Triton default). Per-tensor w8a8 has
  `weight_block_size=null` and never reaches that seam, so `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` would
  bind to nothing. Check `meta.quant_scheme` first, every time.
- **re-confirmed 2026-08-17, and this time the AUTHORED lane is the confirm.** Same qkv/o_proj head
  (10.17% GPU, Qwen3-14B fp8 per-tensor, gfx950/MI350X, sglang 0.5.12), same immutable `unittest.py`
  bake-off harness. The r1_d1 decode-tile Triton overlay authored on the 2026-08-13 run **replays on a
  fresh run's oracle unchanged**: 3 scored runs gave **1.0636 / 1.0521 / 1.0560 serving-weighted (wall
  basis) and 1.1046 / 1.0891 / 1.0999 device**, `CORRECTNESS PASS` with `max_rel_err = 0.0` on every
  random draw and every graph-replay case, and the `M>=128` prefill fall-through measured 0.9985-1.0012x
  (bit-identical, as designed). The identity leg (`GEAK_CANDIDATE_GEMM=default`) scored 0.995 weighted /
  1.0006 device on the same day, so the win clears the identity band by ~6x on the wall basis. **An
  authored overlay is a REUSABLE ARTIFACT: check `overlay/cand_<op>/` of prior runs on the same
  (model, gfx, quant scheme) before funding an author lane from zero.**
- **the harness's own run-to-run spread on the wall-basis primary metric is +/-0.6%** (1.0521 vs 1.0636
  for the SAME bytes, 3 alternating min-of-rounds each). Anything under ~1% weighted is not a tune, it is
  noise. Corollary: screen tile configs on DEVICE time (spread ~0.2%), and only promote a config to the
  scored harness once it wins by more than that.
- **the N-partition axis is MEASURED-EXHAUSTED at decode -- split-K is the only remaining way to add
  CTAs.** A 24-config sweep of the authored decode tile (2 families x M in {1,64}, cold-flushed device
  time under graph replay; driver `opbench_h0/sweep_decode.py`) found the shipped
  `BLOCK_M=64,BLOCK_N=32,BLOCK_K=512,warps=8,stages=3,mfma=16,B_CM=".cg"` is already at/near the local
  optimum at M=64: **`BLOCK_N=16` regresses to 0.81-0.84x** (the weight burst narrows to 16 B/row and
  coalescing collapses) and **`BLOCK_N>=64` / `BLOCK_K=1024` / `num_stages=4` do not fit** (LDS 196-295 KB
  vs the 160 KB gfx950 limit). So you cannot buy more workgroups by slicing N, and o_proj (N=5120,
  BLOCK_N=32 -> **160 CTAs on 256 CUs**) stays a 62%-filled single wave. That is direct measured support
  for the split-K claim rather than a roofline inference. Other sweep results, all losers at M=64:
  `BLOCK_K=256` 1.03-1.05x, `warps=4` 1.09-1.10x, `stages=2` 0.96-0.98x, `stages=1` 0.86-0.90x,
  `XCD=8` neutral, `mfma=32` 0.99-1.02x, dropping the `.cg` weight cache modifier 0.86-0.89x (the single
  biggest single-knob loss -- keep it).
- caution: `B_CM=".cs"` does not compile on this triton (3.6.0) -- `tl.load(..., cache_modifier=".cs")`
  raises a `CompilationError`; only `""`, `".cg"` and `".cv"` are usable. And **`B_EVICT` in the shipped
  config is `"evict_last"`, which contradicts that module's own docstring** (B is 95%+ of the traffic and
  is read exactly once). Flipping it to `"evict_first"` won every one of the four decode cases on a
  single-round screen (qkv M=64 1.1334->1.1414, o_proj M=64 1.0941->1.0980, o_proj M=1 1.1932->1.2060)
  but scored **1.0560 weighted -- inside the +/-0.6% band above**, i.e. real direction, unbankable size.
  Carry it as the author lane's starting config, not as a shippable delta.
- **post-authored roofline (the number that sizes the REMAINING Tier-C target).** Against the 4.73 TB/s
  measured MI350X copy roof, at the M=64 steady state: qkv (37.95 MB/call) went 1.475 -> **1.653 TB/s**
  (31.2% -> 34.9% of roof, floor 8.02 us vs 22.96 us deployed = **2.86x left**); o_proj (27.20 MB/call)
  went 1.129 -> **1.234 TB/s** (23.9% -> 26.1%, floor 5.75 us vs 22.04 us = **3.83x left**). The authored
  kernel banked only ~11% of a 2.9-3.8x envelope, so this head is NOT closed -- a 2x device win would be
  a 5.1% e2e ceiling at 10.17%.
- caution: **the wall/device split is where this head's e2e story is decided, so report BOTH.** The
  scored decode buckets use launch-INCLUSIVE wall under single-op graph replay (~15 us of host replay on
  top of a ~23 us kernel), which the live server pays ONCE per whole-model graph, not per op. That drags
  1.10x device down to 1.056x weighted and the Amdahl ceiling from 0.92% to 0.54% e2e. The 2026-08-13
  run measured **+1.659% e2e** for this same overlay and the Director REJECTED it as implausible against
  a +0.346% ceiling -- but the ceiling it was compared against was the *wall*-basis one. Hand the
  Integrator the device-basis ceiling as the upper bound and the wall-basis one as the lower, and flag
  the head as ceiling-ambiguous instead of letting a single conservative denominator auto-reject a
  correctness-clean, engagement-verified kernel.
- source: exp/e2e_*Qwen3-14B*/ 2026-08-11 (h1
  mlp_gate_up_proj_fp8_gemm, 14.44% GPU time; kernels/.../opbench_result.json, opbench_h1/);
  exp/e2e_*Qwen3-14B*/ 2026-08-12 (h0
  Cijk_Alik_Bljk_F8BS_..._MT64x32x512 = qkv_proj+o_proj, 10.3% GPU time; opbench_h0/);
  exp/e2e_*Qwen3-14B*/ 2026-08-13 (h0 again,
  9.36% GPU time; opbench_h0/rung_*.log + tunableop_pruned.csv);
  exp/e2e_*Qwen3-14B*/ 2026-08-14 (h1
  down_proj_fp8_gemm_MT128x64x256, 9.13% GPU time incl. its PostGSU8 reduction row;
  opbench_h1/bakeoff_summary.json + rung_*.log);
  exp/e2e_*Qwen3-14B*/ 2026-08-17 (h0
  qkv_proj+o_proj again, 10.17% GPU time; opbench_h0/rung_identity.log, rung_c0_triton_seed*.log,
  rung_c1_triton_tuned.log, sweep_decode.json/.log; reusable overlay staged at
  overlay/cand_fp8_pertensor_qkv_o_proj_gemm/overlay_c0_triton)
