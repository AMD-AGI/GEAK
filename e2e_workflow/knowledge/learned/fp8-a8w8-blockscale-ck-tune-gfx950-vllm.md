---
key: fp8 a8w8 blockscale GEMM · gfx950 · vLLM prefill+decode
type: lever
confidence: ★★★
effect: e2e +16.1% / +65.69% / +18.86% (tuned) and +53.72% (swap-only untuned-CK, Director-validated_win on Qwen3-14B-FP8) across four models; iso 1.86–3.17× serving-weighted (1.513× swap-only; 2.87× vs untuned CK on Qwen3.5-27B TP4). Gain is gated by ALREADY-BOUND coverage — ~1.03× when the head shapes already bind tuned (saturated shipped table, or a second pass over your own table). Probe `is tuned` vs `use default` before budgeting. Tunable even when aiter ships no csrc/ckProfiler (race the prebuilt CK instances by kernelName). **WARM-START REPLAY works**: an earlier run's merged CSV for the SAME (model, gfx, cu_num) family set re-binds 16/16 with no re-tune and reproduces iso ~1.79x serving-wtd from an AITER-OFF Triton baseline (+42.9% ceiling at a 67.96% head). ZERO-coverage case measured at iso 2.577x serving-wtd / +46.6% ceiling (Qwen3-14B-FP8 TP1, 51.95% head) -- prefill-only (3.6-4.3x at M=8192, decode ~1.0x); keep rows M>=128 or the cold-cache decode guard tanks. Re-racing from scratch when NO warm-start CSV exists costs 15 s / 32 rows (iso 1.54-1.70x at a 67.72% AITER-OFF head) -- never gate the lever on finding a prior CSV. Reproduced a SECOND zero-coverage case on Qwen3.5-27B-FP8 TP4 (aiter-linear already ON, 0/30 bound): iso 1.523x serving-wtd / +16.5% ceiling in ~5 min of racing, decode neutral. THIRD zero-coverage repro on a 671B-class MoE (DeepSeek-V4-Pro TP8, aiter ON, 0/36 bound): iso 1.536x serving-wtd / +5.5% ceiling; the dense-linear head on a MoE model is small (15.8% GPU) but the lever still clears a 1.0% noise band. A non-128-multiple N (lm_head 16160) tunes fine and is the biggest per-shape win (5.95x). FOURTH zero-coverage repro on a hybrid linear-attn MoE (Qwen3.5-122B-A10B-FP8 TP2, AITER OFF, 0/45 bound, 25.43% head): iso 1.83x serving-wtd / +13.0% ceiling, 55 rows raced in 11 s; here the COLD-timed racer INVERTED the low-M rule -- adding M<128 rows took weighted 1.14 -> 1.83, so restrict to M>=128 only for HOT-timed tuners. aiter-TRITON blockscale measured 0.84x (regression) -- the CK/cktile route is the only winner. PARTIAL shipped coverage (2/5 families) measured on DeepSeek-V4-Pro TP8 (20.18% head): iso 2.5703x serving-wtd / +14.06% ceiling -- and there the `is_triton_gemm_w8a8_tuned` allow-list routes a served family to an aiter-Triton kernel that is NUMERICALLY BROKEN for M>=1024 (max_rel 0.68-0.90), so the swap needs a reversible sitecustomize overlay forcing that predicate False. Probe allow-listed families for CORRECTNESS, not just coverage. ⚠ E2E-TRANSFER CEILING MEASURED: the ZERO-coverage Qwen3-14B-FP8 TP1 case above (iso 2.577x, +46.6% projected ceiling at a 51.95% head) went through a full e2e gate and returned only **+0.18% (rejected)**, while a Tier-C AUTHORED Triton kernel on the SAME seam in the SAME run at a near-identical iso 2.687x returned **+15.72% Director-validated** — so treat this lever's iso number as a ceiling that may not transfer, and always keep the author lane funded alongside it (see fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm.md). SECOND head-to-head on the SAME seam (Qwen3-14B-FP8 TP1, decode-heavy 1024/1024 mix, 67.96% AITER-OFF head): here the env swap+warm-start CSV DID transfer strongly -- iso 1.792x -> e2e +50.29% at the candidate gate -- but the Tier-C authored Triton kernel still beat it at iso 2.368x / +70.62% and took the stack. So the tune is a genuine, cheap, ~minutes-scale win worth taking when the author lane is unaffordable, and a reliable FLOOR when it is; it has now been out-scored by the author lane on 3/3 boxes where both ran. WARM-START REPLAY confirmed a 2nd time (Qwen3.5-122B-A10B-FP8 TP2, 2026-08-23, 20.0% head, AITER OFF): a prior eval dir's merged CSV re-bound 0/50 -> 50/50 in seconds, iso 1.9029x serving-wtd / +9.5% ceiling, and against an untuned-TRITON baseline the win is NOT prefill-only (decode M1/M64 1.87/1.91x = 93% of the serving weight). PROBE A PRIOR EVAL DIR FOR THE SAME (model,gfx,cu_num,TP) FIRST. That warm-start replay then E2E-TRANSFERRED at **+19.45%** (integrate, non-overlapping) / +21.3% window bench — **2.05x its own +9.5% single-head Amdahl ceiling** — and a post-hoc trace decomposition says why: the swap is a WHOLE-STACK re-route, not a GEMM swap. Budget it as such.
last_seen: 2026-08-24
---
# gfx950 vLLM fp8 a8w8 blockscale — per-shape CK tune DB (no overlay needed)

- path: (1) probe the live seam + coverage — `AITER_LOG_TUNED_CONFIG=1`, count `use default` vs
  `is tuned`; (2) if AITER is OFF the live baseline is UNTUNED Triton `_w8a8_triton_block_scaled_mm`,
  so add the swap `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_LINEAR=1` (Triton→CK) — this is the
  dominant term, ~1.5× on its own; (3) tune the `use default` head shapes and deploy the DB — this
  recovers the prefill regression that default-CK introduces. No fp8_utils overlay: on vLLM the tuned
  CSV alone binds (unlike the sglang seam). ⚠ ONE code_patch exception — if any served (N,K) hits the
  `is_triton_gemm_w8a8_tuned` allow-list you may need a tiny sitecustomize overlay forcing that
  predicate False (see the 2026-08-21 DeepSeek-V4-Pro TP8 confirm: the aiter-Triton branch it selects
  is numerically BROKEN there, so the swap silently corrupts prefill without the overlay).
- expected gain: read it off step 1. All shapes `is tuned` → ~1.0×, skip the lever. Zero coverage →
  the full 1.86–3.17× serving-weighted, e2e-transferring at +16% to +65% depending on head share
  (67.96% GPU → +65.69%; 19.78% → +18.86%). Partial coverage → scale by the uncovered fraction.
  A broad aiter-linear swap can EXCEED the single-head Amdahl ceiling, since it re-routes every fp8
  blockscale linear rather than one kernel.
- apply: `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py --libtype both --mp <all GPUs>`
  over the captured head (M,N,K) → tuned CSV, deployed as
  `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<csv>`. Pre-merge your rows OVER the shipped table into one
  file — a bare path REPLACES aiter's merge and drops shipped coverage. ZERO extra HBM.
- verify: `is tuned on cu_num` for every head shape, errRatio 0. Parity rel_err ~2e-4 vs the fp32
  dequant oracle (<< TOL 0.05).
- caution: gains are prefill-driven (M=4096 ~3.3×; decode M1/M64 ~1.14×), so a decode-bound mix banks
  less than the serving-weighted figure — check the e2e gate, and extend the tune to small-M decode.
  Also verify parity on EVERY deploy: a re-run of the +18.86% lever read +16.79% but failed the
  integrate gate on output_corruption. Quote the +65.69% as a free-GPU number — a contended re-run of
  the same lever landed 1.067×. The +16.1% (Qwen3.5-27B) is the weakest of the three: parity n/a under
  the fp8 accuracy gate, tuned-shape binds unreproduced, and a server-flag bundle confounds it.
  If ckProfiler is absent the CK *author* lane is unavailable, but the tune DB still applies.
- confirm (2026-08-19, Qwen3-14B-FP8 TP1, gfx950/MI355, head 67.3% GPU): swap-only aiter-linear
  Triton->CK (VLLM_ROCM_USE_AITER[_LINEAR]=1), UNTUNED CK, measured on the immutable unittest
  (`aiter.gemm_a8w8_blockscale`, non-transposed scale, parity rel~7e-3 << TOL 0.05): serving-weighted
  1.513x, geomean 1.28x. Regime split as the card warns — decode M1/M64 1.8-2.3x, prefill M571 REGRESSES
  0.66-0.84x (untuned CK). The per-shape CK tune (recovers prefill -> card's 1.86-3.17x) was NOT runnable
  here: the offline tuner `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py` + ckProfiler are
  ABSENT from this image's aiter wheel (no csrc dir); foreign-checkout tuners exist on NFS but version-
  mismatch the installed aiter -> unsafe. So shipped swap-only; e2e prefill-regression risk left to the
  Integrator gate + operator-provisioned tuner. E2E OUTCOME (2026-08-19, same run): the swap-only,
  UNTUNED-CK config PASSED the e2e gate at +54.24% (cand_med vs ref_med,
  non-overlapping; accuracy gate gsm8k 0.725 >= 0.675) -> the isolated prefill regression did NOT sink
  e2e in this decode-dominated mix, and the +54% EXCEEDS the single-GEMM Amdahl ceiling because
  VLLM_ROCM_USE_AITER also routes aiter rmsnorm/quant (ck_tile Rmsnorm2dFwd, _fused_rms_fp8_group_quant,
  dynamic_per_group_scaled_quant) alongside the linear swap. Post-swap the bottleneck MOVED to the
  aiter-dispatched CK `kernel_gemm_xdl_cshuffle_v3` (decode 46.1% + prefill 10.7% = 56.8% combined,
  NON-editable library CK); server.log shows every head shape `will use default config` -> the per-shape
  CK tune (step 3) is the only remaining GEMM lever and its headroom is UNREALIZED here (tuner absent).
  DIRECTOR-VALIDATED FINALIZE (same-session A/B, authoritative): base -> final =
  **1.5372x (+53.72%)**, ranges NON-overlapping (final_min > base_max), spreads tight
  (base 3.52%, final 0.09%) -> validated_win. Latency also improved: TTFT −11.3%, TPOT
  −34.0%. Parity PASS (accuracy gate, quant swap: aiter gsm8k 0.700 >= Triton 0.675, not
  degenerate; engagement AiterFp8BlockScaledMMKernel vs base TritonFp8BlockScaledMMKernel). Win is
  ENV-ONLY (no code overlay, site-packages untouched) -> confirms the swap-only untuned-CK lever
  transfers e2e on a decode-dominated fp8 mix even without the per-shape CK tune.
- caution (post-swap remaining editable head, decode-dominated): once the aiter-linear swap dominates,
  the top EDITABLE op is prefill flash-attn `_fwd_kernel` (~7-9% GPU), but swapping it to the aiter FA-2
  asm/CK fast path net-REGRESSED e2e -1.48% (engagement PROVEN at M=1024 Hq=40/Hkv=8/D=128, yet cand
  range entirely below ref) — a 7.19% prefill head at iso 1.43x has only a +2.21% Amdahl ceiling and the
  batch==1 M=1024 aiter dispatch net-costs vs the retuned Triton stack. In a decode-dominated fp8 run,
  do not budget a full A/B on prefill-attn after this swap; prioritise the CK GEMM tune instead.
- 🔧 **TUNER RECOVERY (2026-08-20, gfx950/MI355, aiter wheel with NO `csrc/`): you can tune WITHOUT the
  offline tuner or ckProfiler.** The prebuilt `module_gemm_a8w8_blockscale.so` (19 CK instances) and
  `..._cktile.so` (22) expose per-instance entry points
  `aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale_ck(XQ,WQ,xs,ws,Out,splitK,kernelName)`; harvest the instance
  names with `strings <so> | grep '^a8w8_blockscale_'` and race them per (M,N,K) against an fp32-dequant
  oracle. Emits the exact `get_CKGEMM_config` CSV schema. So "csrc/ckProfiler absent" no longer blocks the
  tune DB — only the CK *author* lane.
- 🎯 **COVERAGE, cheaply: tune the POWER-OF-TWO M ladder, not the observed M list.** `get_CKGEMM_config`
  probes exact M, then `get_padded_m(...,gl=0)`, then `gl=1` — and **gl=1 is a ceil-to-power-of-2**. So
  rows at M∈{1,2,4,…,4096} give COMPLETE coverage of every live M ≤ 4096 (65 rows for a 5-family dense
  model, ~25 min on one GPU), instead of chasing the 50+ distinct live paddings (112/1088/3008/…).
- ⚠ **A BARE CSV path replaces aiter's whole configs/+model_configs/ colon-merge** — pre-merge shipped rows
  into ONE file. Do NOT colon-join your own tables: `update_config_files()` RAISES on any duplicate
  (gfx,cu_num,M,N,K) across files **and rewrites the source csvs** (it will mutate site-packages).
  Also check the shipped `model_configs/*_<model>.csv` gfx/cu_num before assuming it covers you — the
  `qwen36_27b` table matching our exact (N,K) is `gfx942/cu304` and never binds on gfx950/cu256.
- confirm (2026-08-20, Qwen3.5-27B-FP8 TP4 gfx950 vLLM 0.26, head 29.83%, ON TOP of an already-deployed
  15-row head CSV): the RESIDUAL coverage gap is worth only **iso 1.032× serving-weighted** (4v4 interleaved,
  −3.1% weighted) = **+0.93% Amdahl ceiling, UNDER this box's 2.0% noise band** → stack-only, do
  not A/B it alone. Same tune vs the UNTUNED CK default is 2.87× (+24.1% ceiling) — i.e. this lever's value
  is ~all in the FIRST deployment; a second pass only buys the uncovered M. Per-shape it is dramatic where
  coverage was missing (M=1024: 1.91–3.01×) and ~1.0–1.2× where it already bound. Engagement verified
  in-process: 20/20 `is tuned on cu_num`, 0 `use default` (was 15/5). Zero extra HBM. Also measured here:
  `gemm_a8w8_blockscale_bpreshuffle` is 2.7× the untuned default at M=4096 but LOSES to the tuned CK
  (1.32× SLOWER) and needs a preshuffled-B weight layout → dead end on the vLLM plain-B seam.
- confirm (2026-08-20, Qwen3-14B-FP8 TP1 gfx950/MI355 vLLM 0.26, head 14.22% GPU, ON TOP of the
  already-live warm-start cfg0 stack = aiter-linear swap + a partial tuned CSV): the in-image
  instance-race tuner reproduces cleanly on a SECOND model. Baseline for the A/B is the LIVE cfg0
  table (not untuned CK), and the residual gap is still large here because cfg0's table has ZERO rows
  for our four (N,K) at prefill M: measured on the immutable unittest **serving-weighted 4.345×,
  geomean 3.079×, decode guard 0.993 (neutral), parity PASS** → Amdahl ceiling +12.29% e2e. Dominant
  case M7233×N34816×K5120 **4.38×**; per-shape range 1.38×–4.91×. Winners split:
  cktile `192x256x128_4x2x1_16x16x128_intrawave` takes large M, CK `256x128x128x128..._v3` /
  `256x64x128x128` take mid M. ⚠ **Restrict your rows to M>=128 when a prior tuned table already owns
  decode**: a first pass that also wrote M∈{1..64} rows REGRESSED the decode guard to 0.706 — the
  tuner times HOT while the guard times COLD-cache, so hot-optimal small-M picks lose. Dropping the
  low-M rows (and ordering the pre-merge so the incumbent decode rows win) restored 0.993 at
  unchanged prefill (4.310 → 4.345). Binding re-verified through aiter's own `get_CKGEMM_config()`
  against the merged CSV: 64/68 live (M,N,K) hit, the 4 misses all M=16 decode.
- confirm (2026-08-21, Qwen3-14B-FP8 TP1 gfx950/MI355 vLLM 0.26, head 51.95%, baseline already has the
  AITER-linear swap ON so the live seam is UNTUNED CK `kernel_gemm_xdl_cshuffle_v3`): the ZERO-coverage
  case, and it is the biggest single-lever number this card has recorded from a bake-off. In-process
  `get_CKGEMM_config` probe: **0/20 bind with the env unset** (shipped `configs/` + all 8
  `model_configs/*a8w8_blockscale_tuned_gemm*.csv` cover none of the 4 (N,K) families — 34816×5120,
  5120×17408, 7168×5120, 5120×5120), i.e. every live GEMM runs `will use default config`. The in-image
  instance-race tuner (19 CK + 22 cktile prebuilt instances, no csrc/ckProfiler on this wheel) over the
  pow2 M ladder took **<4 min on ONE GPU for 56 rows** (14 M × 4 families). Measured on the immutable
  unittest: **serving-weighted 2.577×, geomean 1.599×, decode guard 1.0011, parity PASS** (max_rel 0.0077
  vs TOL, random-parity byte-exact 0.0, graph_replay + compile_parity ok) → **+46.6% Amdahl ceiling** at
  the 51.95% head. Gain is entirely prefill: M=8192 **3.58×–4.33×** (gate_up 4.33×), decode
  M=1/M=64 ~1.00–1.04× (untuned CK default is already near-optimal at low M here). Winners split
  cktile `192x256x128_4x2x1_16x16x128_intrawave` for M>=2048 / CK `256x128x128x128..._v3` and
  `256x64x128x128` for mid M. ⚠ **The low-M decode-guard trap re-fires even with ZERO incumbent
  coverage** — it is NOT conditional on a prior tuned table owning decode as this card previously said:
  the full 1..8192 ladder scored a *higher* weighted 2.609× but tanked **decode guard to 0.807**, because
  the tuner times HOT while the guard times COLD-cache. Dropping the 28 rows at M<128 cost only
  2.609→2.577 weighted and restored the guard to 1.0011. **Default to M>=128 rows for this lever.**
- confirm (2026-08-21, Qwen3-14B-FP8 TP1 gfx950/MI355 vLLM 0.26, head 67.96%, baseline AITER **OFF** =
  untuned Triton `_w8a8_triton_block_scaled_mm`, short ISL/OSL 1024/1024 conc 64): the **warm-start REPLAY**
  case. In-process `get_CKGEMM_config` probe: 0/16 bind with the env unset (shipped `configs/` +
  `model_configs/` have no gfx950/cu256 row for 34816x5120, 5120x17408, 7168x5120, 5120x5120). Re-deploying a
  PRIOR run's merged CSV for the same model/box (no re-tune, seconds of work) bound 16/16 and measured on the
  immutable unittest **serving-weighted 1.788x, geomean 1.707x, parity PASS**. Decomposition on the same box:
  swap-only untuned CK = **1.435x weighted** (decode M1/M64 1.63/1.73x, prefill M337/M1024 **REGRESS**
  0.85/0.71x -- the card's prefill-regression warning reproduces exactly), and the tuned CSV is what converts
  prefill to 1.60-1.79x. So on a mixed prefill+decode workload the CSV is not optional polish, it is ~55% of
  the lever. A small residual re-race (in-image instance racer, 2 M x 4 (N,K), COLD-cache timed, head-to-head
  vs the incumbent pick) added only **2 rows** at the live **exact M=337** (o_proj 1.339x, down_proj 1.266x
  vs the incumbent's padded-to-512 row) -> same-window A/B 1.7876 -> **1.7915** weighted (prefill_M337
  1.490 -> 1.600). Lesson: a warm-start table tuned on a pow2 ladder leaves per-shape headroom at **live
  non-pow2 M** that `get_padded_m` rounds UP to a bigger tile; racing just those exact M is a ~5-minute
  top-up, but it is worth only ~0.2% e2e -- do not re-tune the whole ladder. ⚠ MEASUREMENT: on a co-tenanted
  box the weighted number drifted 1.94 -> 1.79 over ~40 min with the SAME csv; only compare artifacts in the
  SAME time window (3 interleaved runs settled it). Engagement PROVEN on a warm server: `Selected
  AiterFp8BlockScaledMMKernel`, **220 `is tuned on cu_num` / 0 head-family `use default`** (the 7 defaults are
  all lm_head N=151936, a different table). ⚠ NEW TRAP on vLLM: `AiterFp8BlockScaledMMKernel.__init__` sets
  `use_triton = not fp8_fnuz and rocm_aiter_ops.is_triton_gemm_w8a8_tuned(n,k)` -- if a *Triton* w8a8 tuned
  config exists for an (N,K), the aiter kernel dispatches aiter-TRITON and your CK CSV binds to NOTHING.
  Probe `is_triton_gemm_w8a8_tuned(n,k)` per family (all False here) before budgeting the CK tune.
- confirm (2026-08-20, DeepSeek-V4-Pro TP8 gfx950 vLLM 0.26, head 7.53%, ON TOP of a warm-start CSV):
  the gap-fill re-fires but is **serving-weight-gated, not coverage-gated** — read the WEIGHTS, not the
  row count. Live table had rows only at M∈{1,64,8192} for 3 of the 4 (N,K) families, so every live
  padded M in (64,4096] missed (`will use default`), while M=1/64 bind exactly and M=5910 binds via the
  ceil-pow2 fallback → M=8192. Racing the mid-M pow2 ladder {128..4096} × those 3 (N,K) (18 rows, ~4 min
  on one GPU with the in-image instance racer) gave big PER-SHAPE wins exactly where coverage was missing
  (M=1024: 1.73× / 1.50× / 1.25×) — but those buckets carry only ~16% of the served weight, so
  **serving-weighted 1.0148, geomean 1.088 → +0.11% Amdahl ceiling, far under the 0.5% noise band**:
  ship it as a zero-HBM stack/compounding item, never A/B it alone. Parity byte-exact (err 0.0 on every
  case incl. the re-dispatched ones). Corollary: a strategy note claiming "ZERO `is tuned` hits" from a
  server-log grep can be wrong — the in-process `get_CKGEMM_config` probe showed coverage was PARTIAL
  (only 3 shapes uncovered); probe in-process before budgeting. Also seen again: `bpreshuffle` FAILS
  correctness on the plain-B vLLM seam (max_rel 37) — same dead end as the 27B run.
- confirm (2026-08-21, Qwen3.5-27B-FP8 TP4 gfx950 vLLM 0.26, head 41.22% GPU, baseline ALREADY has the
  aiter-linear swap ON so the live seam is UNTUNED CK `kernel_gemm_xdl_cshuffle_v3`, ISL/OSL 1024/1024
  conc 64): the ZERO-coverage case again, on a 5-family hybrid linear-attention model (N,K =
  8704x5120 gate_up, 5120x4352 down, 4096x5120 linattn_in_proj, 5120x1536 o_proj, 3584x5120 qkv).
  In-process `get_CKGEMM_config` probe: **0/30 bind** with the env unset; baseline server.log has 2072
  `will use default config` / 0 `is tuned`. In-image instance-race tuner (19 CK + 22 cktile prebuilt
  instances harvested by `strings`, cold-cache timed, splitK {0,2} below M=512) over the pow2 ladder
  M∈{128..4096} x 5 families = **30 rows in 5 SECONDS on one GPU**. Measured on the immutable unittest
  (candidate = a shim that forces the tuned CSV while the in-process baseline keeps the untuned default,
  so the A/B is honest inside ONE process): **serving-weighted 1.523x, geomean 1.504x, PASS** (oracle
  max_rel ~0.008 << TOL 0.06, random-parity vs the live baseline byte-exact 0.0, graph_replay ok)
  → **+16.48% Amdahl ceiling**. Gain is entirely prefill: M=4096 **2.57x–3.95x** (gate_up 3.95x,
  down 3.39x, linattn 3.14x, qkv 2.89x, o_proj 2.57x); decode M=1/M=64 **0.99–1.00x** (untuned CK default
  is already optimal at low M) — so the M>=128-only row set costs nothing and keeps the decode guard
  neutral. Per-row race vs the untuned default: 1.02x at M=128 rising monotonically to 4.06x at M=4096;
  winners split cktile `192x256x128_4x2x1_16x16x128_intrawave` for the big-M/large-N cells and CK
  `256x128x128x128..._v3` / `256x64x64x256` elsewhere. Engagement PROVEN on a warm TP4 server:
  **980 `is tuned on cu_num` (196 per head family) / 0 head-family `use default` above M=64**; the
  remaining defaults are M<=64 (deliberate) plus the vision-tower N=24 shapes. Also re-confirmed here:
  `gemm_a8w8_blockscale_bpreshuffle` FAILS correctness on the plain-B vLLM seam (max_rel 38) — third
  independent reproduction of that dead end. Note this model's head is only 41.22% GPU yet the lever is
  worth +16.5% ceiling, i.e. worth a standalone A/B.
- confirm (2026-08-21, Qwen3-14B-FP8 TP1 gfx950/MI355 vLLM 0.26, head 67.72% GPU, AITER **OFF** baseline =
  untuned Triton `_w8a8_triton_block_scaled_mm`, ISL/OSL 1024/1024 conc 64): the same box/model as the
  warm-start-replay confirm above but with **NO prior CSV on disk** — so this is the "replay unavailable,
  re-race from scratch" cost. In-process probe: **0/16 bind** unset; `is_triton_gemm_w8a8_tuned` False for
  all 4 families (dispatch trap clear). In-image instance racer (19 CK + 22 cktile), pow2 ladder
  M∈{128..8192} **plus the live non-pow2 M=464**, 4 families = **32 rows in 15 s on ONE GPU** — i.e.
  re-racing is cheap enough that a missing warm-start CSV costs nothing; do not gate the lever on finding one.
  Merged over the 8 shipped tables (0 key collisions) → 8/16 live shapes bind (all prefill; decode M1/M64
  deliberately left on the CK default). Immutable unittest: **serving-weighted 1.6996 (clean window) /
  1.5427, 1.5351 (co-tenanted re-runs), geomean 1.69, decode guard 1.68, parity PASS** (max_rel ~0.0077,
  random-parity PASS, graph_replay + compile_parity ok) → **+31% to +39% Amdahl ceiling**. Decomposition in
  the same window: swap-only untuned CK = **1.386-1.395** (decode M1/M64 1.8-2.3x, prefill M464/M1024
  **REGRESS 0.62-0.99x**) — the CSV converts prefill to 1.55-1.98x and is again ~half the lever.
- ⚠ **Do NOT add M<128 rows on this stack — and now we know WHY it is not just a hot/cold timing artifact.**
  Racing the low-M ladder {1..64} COLD-CACHE (same timer the guard uses) still produced picks SLOWER than
  aiter's untuned CK default (gate_up M=1: best named instance is 1.20× SLOWER than the default; M=64: 1.13×
  slower). The default's `compute_gemm_SplitK` heuristic beats every explicitly-named prebuilt instance at
  low M, so the M>=128 rule holds even with a cold-cache tuner. Skip the low-M race entirely.
- ⚠ **Shipped-table landmine: `get_CKGEMM_config(M,N,K)` with its DEFAULT `tuned_file`** (the *a8w8*, not
  *a8w8_blockscale*, table) raises `ValueError: DataFrame index must be unique for orient='index'` on this
  wheel — duplicate (gfx,cu_num,M,N,K) in the merged a8w8 table. Any coverage probe MUST pass
  `AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE` explicitly, or you will misread a shipped-table bug
  as "the blockscale probe is broken".
- confirm (2026-08-21, DeepSeek-V4-Pro TP8 gfx950/MI355 vLLM 0.26, head 15.84% GPU, baseline already has
  `VLLM_ROCM_USE_AITER=1` so the live seam is UNTUNED CK `kernel_gemm_xdl_cshuffle_v3`, ISL/OSL 1024/1024
  conc 64): the ZERO-coverage case on a large MoE model whose *dense* linear head is only 4 families
  (qa_kva_fused 2048x7168, shared_gate_up 768x7168, shared_down 7168x384, lm_head 16160x7168 — the MoE
  expert GEMMs are a separate op). In-process `get_CKGEMM_config` probe: **0/36 bind** unset;
  `is_triton_gemm_w8a8_tuned` False for all 4 (dispatch trap clear); baseline server.log 1664
  `not found tuned config`. In-image instance racer (19 CK + 22 cktile) over the pow2 ladder
  M∈{128..8192} x 4 families = **28 rows in ~8 min on ONE GPU** (cold-cache timed with the task's own
  `harness_lib.time_op`, and each row only kept when it BEAT aiter's default heuristic head-to-head —
  28/28 did). Immutable unittest: **serving-weighted 1.536x, geomean 1.470x, decode guard 1.017,
  parity PASS** (oracle max_rel 0.0077 << TOL 0.05, random-parity byte-exact 0.0, graph_replay ok)
  → **+5.53% Amdahl ceiling** vs a 1.0% noise band. Gain is again entirely prefill: M=8192 1.66x-5.95x,
  decode M1/M64 1.02-1.05x. Winners split cktile `192x256x128_4x2x1_16x16x128_intrawave` for the big-N
  lm_head at M>=1024, CK `256x128x128x128..._v3` / `256x64x64x256_..._v1` / `256x16x64x256_..._v1`
  elsewhere. 🆕 **A non-multiple-of-128 N tunes fine**: lm_head N=16160 (ceil to 127 scale blocks) is the
  single biggest win, **5.95x** — build the tuner's case with the unittest's
  pad-aware ceil blocking (`sN=ceil(N/128)`, zero-pad, slice back) and the reshape-by-128 assumption that
  breaks naive tuners disappears. Zero collisions merging 28 rows over the 8 shipped tables (37431 rows).
  Also re-confirmed: `bpreshuffle` FAILS correctness on the plain-B vLLM seam (max_rel 34.6) — 4th repro.
- confirm (2026-08-21, Qwen3.5-122B-A10B-FP8 TP2 gfx950/MI355 vLLM 0.26, head 25.43% GPU, baseline AITER
  **OFF** = untuned Triton `_w8a8_triton_block_scaled_mm`, ISL/OSL 1024/1024 conc 64, hybrid linear-attn
  MoE, 5 dense families N,K = 8704x3072 qkv / 10240x3072 linattn_in_proj / 3072x4096 o_proj /
  1024x3072 shared_gate_up / 3072x512 shared_down): ZERO-coverage again — in-process
  `get_CKGEMM_config` 0/45 with the env unset (shipped `configs/` + all `model_configs/` have no
  gfx950/cu256 row for any family). In-image instance-race tuner (19 CK + 22 cktile prebuilt instances,
  cold-cache timed, splitK {0,2} below M=512): **35 rows (7 pow2 M x 5 families) in 7.6 s** on one GPU,
  plus 20 low-M rows in 3.3 s. Immutable unittest: **serving-weighted 1.83x, geomean 1.69x, decode guard
  1.85x, parity PASS** (max_rel 0.023 << TOL 0.05, compile_parity ok) → **+13.0% Amdahl ceiling**.
  ⚠ **The low-M decode-guard trap did NOT fire here — it INVERTED.** With M>=128 rows only, weighted was
  1.14 and the decode guard 1.13; ADDING cold-raced M∈{1,16,32,64} rows lifted it to 1.83 / guard 1.85.
  Reconciliation: the trap is about HOT-timed tuners; this racer times COLD (512 MB flush before every
  sample), so its low-M picks match the guard's cost model. **If your racer is cold-timed, tune the FULL
  ladder from M=1; keep the M>=128 restriction only for hot-timed tuners.**
  ⚠ Decomposition in ONE window (4-way min-of-25 cold table, live Triton = 1.0): untuned-CK swap-only
  **2.42x** weighted, tuned CSV **2.55x**, aiter-TRITON blockscale **0.84x (a REGRESSION — never route
  here)**, bpreshuffle fails correctness (max_rel 1478; 4th independent reproduction of that dead end).
  So on a decode-dominated mix the swap carries ~95% of the weighted win (decode weight outnumbers
  prefill 16:1 and dilutes the untuned-CK prefill regression), but the CSV is still what removes it:
  qkv M=1024 and M=8192 (2.3-3.6x), i.e. the CSV is TTFT insurance, not polish.
  ⚠ MEASUREMENT on a co-tenanted box: the unittest read **bimodally** — 1.82-1.83 in clean windows and
  1.12-1.16 whenever an additive per-launch floor appeared on BOTH legs (which compresses the
  ratio toward 1). `op_bench.py` on the same task swung 1.06 / 1.14 / 1.37 / 5.49 across four back-to-back
  runs. Use min-of-N interleaved sampling (or discard floored windows) before believing any single number.
- confirm (2026-08-21, DeepSeek-V4-Pro TP8 gfx950/MI355 vLLM 0.26, head **20.18%** GPU, baseline
  AITER-LINEAR **OFF** = untuned Triton `w8a8_triton_block_scaled_mm`, ISL/OSL 1024/1024 conc 64,
  5 dense families N,K = 8192x1536 / 2048x7168 / 768x7168 / 7168x2048 / 7168x384): the first **PARTIAL
  shipped-coverage** case — the shipped merged table already covers 2/5 families (8192x1536, 7168x2048)
  at ALL M, and MISSES the other 3. Probe per-family, not per-model: a global "0/N bound" heuristic
  would have been wrong here, and so would "already covered → skip".
  🆕🆕 **HARD FINDING — the `is_triton_gemm_w8a8_tuned` dispatch trap is WORSE than "CK CSV binds to
  nothing": on this wheel the branch it selects returns GARBAGE.** `AiterFp8BlockScaledMMKernel.__init__`
  does `use_triton = (not fp8_fnuz) and rocm_aiter_ops.is_triton_gemm_w8a8_tuned(n, k)`, whose hardcoded
  allow-list contains **(7168, 2048)** — a served family here. Measured against an fp32 dequant oracle,
  `aiter.ops.triton.gemm_a8w8_blockscale` at N=7168 K=2048 is correct for M<=512 (2e-3 .. 3e-3) and
  **BROKEN for M>=1024: max_rel 0.90 / 0.68 / 0.70 / 0.76 at M=1024/2048/4096/8192**. CK is correct at
  every M (~3e-3) and faster. So a plain `VLLM_ROCM_USE_AITER_LINEAR=1` swap would have **silently
  corrupted prefill activations** on 1 of 5 families with no crash and no correctness signal at the
  server level. Fix = reversible PYTHONPATH `sitecustomize.py` (site-packages untouched) that wraps the
  `vllm._aiter_ops` loader and sets `ops.is_triton_gemm_w8a8_tuned = staticmethod(lambda n, k: False)`
  after exec, forcing the CK branch for every (n,k). ⚠ Do NOT implement that finder by calling
  `importlib.import_module` re-entrantly inside `find_spec` — the double exec re-registers vLLM's
  custom ops and hard-crashes with `Tried to register an operator ... multiple times`; delegate to the
  downstream meta_path finders and wrap `spec.loader.exec_module`. **Lesson: probe every allow-listed
  family for CORRECTNESS, not just for coverage.**
  In-image instance racer (19 CK + 22 cktile prebuilt instances, cold-cache timed): 28 rows over the
  pow2 ladder M∈{128..8192} + 12 low-M rows M∈{1,16,32,64}; **all 28 high-M winners were CK, zero
  cktile** (contrast the 16160-N lm_head case where cktile won) — always race both libtypes, the
  winner split is shape-dependent. Merged over the shipped 37403-row table → **37435 rows, 0 key
  collisions**. Immutable unittest ladder: swap-only untuned CK **1.920x** weighted → overlay + 21 rows
  (M>=128) **1.995x** → overlay + **32 rows incl. low-M 2.5703x weighted / 1.7782 geomean → +14.06%
  Amdahl ceiling**, CORRECTNESS PASS. ⚠ **Low-M rows were decisive (1.995 → 2.570), re-confirming the
  cold-timed inversion**: `decode_M64_N2048_K7168` alone carries **81% of the serving weight** and went
  2.098 → 2.978x. Drop any low-M row that loses its head-to-head vs aiter's default (1 of 13 did:
  M=16 N=7168 K=384 at 0.85x). Per-case final: prefill M8192 1.24-1.87x; decode M1/M64 up to 3.42x on
  the two shared-expert families, ~1.0x on 7168x384.
  Parity nuance: random value-parity vs the LIVE Triton baseline was byte-exact (0.0) for the swap-only
  leg but reads **0.087-0.383 max_rel on the 12 decode draws of the two families that got new low-M CK
  rows** — still PASS, because `harness_lib.correct` gates on `|d| <= tol*RMS(ref) + tol*|ref|` while the
  REPORTED `max_rel_err` is `|d|/(|ref|+atol)`, which saturates toward 1.0 on near-zero elements. vs the
  fp32 dequant oracle the candidate is 0.0077 << TOL 0.02. Read the gate, not the printed err.
  Engagement on a warm TP8 server (`REPEATS=1 NUM_PROMPTS=64`): overlay banner on all 10 ranks,
  `AiterFp8BlockScaledMMKernel` selected, **2816 `is tuned on cu_num` vs 272 `will use default`** (all
  272 = the deliberately-unraced M<128 rows), single-run probe, clean group teardown.
- caution (also verify the E2E TRANSFER, not just the iso ceiling — 2026-08-22, Qwen3-14B-FP8 TP1
  gfx950 vLLM 0.26, ISL/OSL 8192/1024 conc 64, the 51.95%-head ZERO-coverage confirm above): the tuned
  CSV measured iso 2.577x serving-wtd (+46.6% projected) but its e2e A/B came back **+0.18% — inside the
  0.5% noise band, rejected**, whereas an authored Triton kernel bound at the same
  `aiter:gemm_a8w8_blockscale` seam at iso 2.687x delivered +16.77% (integrate) / +15.72%
  (Director same-session, non-overlapping). Two candidates with the SAME isolated
  score can differ by 15 points e2e, so the isolated bake-off cannot rank them — run the author lane in
  parallel with the tune and let the e2e gate decide. Cross-ref:
  `fp8-blockscale-dense-gemm-authored-triton-gfx950-vllm.md`.
- caution (single-pass race mis-picks): one sweep pass can pick a ~5% worse instance on a contended node.
  Re-time the new pick head-to-head against the previously deployed row and keep the winner, so the new
  table is a strict superset-and-improvement (8/15 overlapping shapes flipped to the new pick, 1 kept old).
- confirm (2026-08-23, Qwen3.5-122B-A10B-FP8 TP2 gfx950/MI355 vLLM 0.26, head 20.0%, baseline AITER
  **OFF** = untuned Triton `_w8a8_triton_block_scaled_mm`, ISL/OSL 1024/1024 conc 64): **WARM-START
  REPLAY, 2nd instance and now on the SAME model/box as the 2026-08-21 122B confirm** — re-deploying
  that run's `config/ck_tune/a8w8_blockscale_tuned_gemm_merged_full.csv` (shipped rows + 35 pow2 +
  20 low-M rows) took SECONDS and needed no re-tune: in-process `get_CKGEMM_config` **0/50 bound with
  the env unset → 50/50 bound with it** (the 10 M probed include the live non-pow2 M=7393, which binds
  through the ceil-pow2 fallback), `is_triton_gemm_w8a8_tuned` False on all 5 families (dispatch trap
  clear, so NO sitecustomize overlay needed here). IMMUTABLE unittest, candidate = the CK seam behind a
  signature-only shim: **serving-weighted 1.9029x, geomean 1.8150x, PASS** (eager-vs-oracle max_rel
  0.0078, random value-parity vs the live Triton baseline 0.0074, graph_replay 0.0091, compile_parity
  0.0 — all << TOL 0.02) → **+9.5% Amdahl ceiling** at the 20.0% head. Unlike every earlier confirm the
  win is NOT prefill-only: decode M1/M64 **1.87x/1.91x** and prefill M1024/M7393/M8192 1.63x/1.84x/1.84x,
  because the AITER-OFF baseline is untuned *Triton* (not untuned CK) — decode M=64 carries 93% of the
  serving weight, so this is the case where the lever pays on a decode-dominated mix. Same-window 4-way
  min-of-25 cold table reproduced 2026-08-21 almost exactly: live Triton 1.0 / untuned-CK swap **2.393x**
  / tuned CSV **2.580x** / aiter-TRITON blockscale **0.810x (regression, never route here)** /
  bpreshuffle correctness FAIL max_rel 1478 (**5th** repro of that dead end). ENGAGEMENT verified on a
  warm TP2 server (REPEATS=1 CONC=16): `AiterFp8BlockScaledMMKernel` selected, **580 `is tuned on
  cu_num`, ZERO `use default` for any a8w8_blockscale shape** (the 254 defaults are bf16 vision/ViT
  GEMMs against a different table). Practical rule this adds: when a prior eval dir for the same
  (model, gfx, cu_num, TP) exists, **probe it FIRST** — the whole lever collapsed to a file copy plus a
  50-shape bind probe (minutes), leaving the round's budget for the Tier-C author lane.
  🆕 **E2E OUTCOME of that replay + the trace decomposition of WHERE the win comes from** (same run,
  same TP2/GPU-set, 1024/1024 conc 64): integrate A/B ref → cand = **+19.45%**,
  non-overlapping (cand_min > ref_max), spreads 0.26–0.42%; the round-1 window bench read 2423.7 → 2939.4
  (+21.3%) with total GPU time in the window −16.3%. That is **2.05× the +9.5%
  single-head Amdahl ceiling**, and a full-trace name scan attributes it to FOUR places, only one of which
  is the head you tuned:
  · dense fp8 blockscale GEMM Triton → **−54.8% GPU time** across ck_tile QuantGemm + CK
    `kernel_gemm_xdl_cshuffle_v3` + a little hipBLASLt — the tuned head, ~half the win;
  · the MoE **pre/post chain** over six Triton/csrc kernels (per-token-group quant, silu+mul
    per-block quant, moe_align, count_and_sort, moe_sum, topkGating) → **−77% GPU time** in two aiter kernels — a chain you never touched, deleted for free by the same env switch;
  · the **MoE GEMM itself did NOT get faster**: aiter's fused asm fmoe does a decode layer in ONE
    launch for what the two Triton launches cost together — identical, and 97.3% of the HBM roofline in BOTH legs;
  · **comm swapped implementation at flat cost**: RCCL `ncclDevKernel_Generic` → aiter
    `reduce_scatter_cross_device_store`, both flat at the interconnect roof (total comm −3.2%).
  Two routing rules this hands the next run: (a) **size a broad AITER swap by the whole Triton chain it
  deletes, not by the head GEMM's own iso number** — the peripheral quant/routing chain was 39% of the
  saving; (b) **after the swap, re-derive the head list from scratch**: both new heads are non-editable
  and config-only (aiter asm fmoe 25.5% at the HBM wall, aiter TP all-reduce 23.9% at the interconnect
  roof), the largest editable head with modelled headroom collapses to the Triton prefix-prefill
  `_fwd_kernel` at 6.2% (~+4% ceiling), and shares RISE on kernels that did not change (`_fwd_kernel`
  5.17% → 6.21% at a flat absolute cost) purely because the denominator shrank 16%.
  🏁 **RUN CLOSE-OUT (Director same-session, 2 legs x 2 timed repeats, TP2 GPU 0,1, identical flags):
  1.2081x (+20.81%)**, distributions disjoint (final_min > base_max by 41x the 0.5%
  band), TTFT −19.83%, TPOT −17.06%, parity
  pass at `parity_kind=accuracy`, `validated_win`. Engagement on the official final leg:
  `AiterFp8BlockScaledMMKernel` vs `TritonFp8BlockScaledMMKernel` on base. This warm-start replay is
  ~94% of the run's total gain (the stacked prefill-attn kernel added +1.22%). ⚠ the 33-h-old stored
  baseline (2376.3) would have reported 1.2492x — 3.4% of pure box drift; quote the same-session ratio.
  ⚠ op_bench.py is BLIND to this lever on this task: its blockscale path benched only
  `aiter_blockscale`(= the live vLLM Triton seam) and `bpreshuffle`, reported winner=live/1.0x, and
  never constructed the CK candidate — a `no_win` read off op_bench alone would have buried a 1.90x.
- source: exp/e2e_*Qwen3.5-27B-FP8*/ 2026-08-12; exp/e2e_*Qwen3.5-27B-FP8*/ 2026-08-20 (TP4, residual-gap
  confirm + in-image instance-race tuner); exp/e2e_*Qwen3-14B-FP8*/ 2026-08-13
  (Director-validated_win TP1, head 67.96% GPU); exp/e2e_*Qwen3-14B-FP8*/ 2026-08-19..08-20
  (swap-only untuned-CK, Director-validated_win TP1 head 67.3% GPU: 1.5372x
  (+53.72%), parity pass; post-swap prefill-attn `_fwd_kernel` aiter FA-2 swap = dead_end -1.48%);
  exp/e2e_*Qwen3.5-122B-A10B-FP8*/ 2026-08-13
  (Director-validated_win TP2, head 19.78% GPU); exp/e2e_*DeepSeek-V4-Flash-0731*/ 2026-08-13
  (saturated shipped tables → iso 1.03×, the zero-headroom case); exp/e2e_*DeepSeek-V4-Pro*/ 2026-08-16
  (run-level 1.26× Director-validated, but both attributed heads reversed to dead_end
  (implausible_speedup) at review — not counted as a confirm here).
  exp/e2e_*Qwen3-14B-FP8*/ 2026-08-20 (TP1, instance-race tuner on top of warm-start cfg0:
  iso 4.345× serving-wtd, +12.29% ceiling; low-M decode-guard trap).
  exp/e2e_*Qwen3-14B-FP8*/ 2026-08-21 (TP1, AITER-OFF baseline, head 67.96%:
  warm-start CSV replay iso 1.788x -> exact-M=337 top-up 1.7915x, +42.9% ceiling; swap-only decomposition
  1.435x; is_triton_gemm_w8a8_tuned dispatch trap).
  exp/e2e_*DeepSeek-V4-Pro*/ 2026-08-20 (TP8 warm-start gap-fill: per-shape 1.25-1.73x at the uncovered
  mid-M, serving-weighted only 1.015 at a 7.53% head -> stack-only).
  exp/e2e_*Qwen3-14B-FP8-vllm*/e2e_cycle0 2026-08-21 (TP1, ZERO-coverage instance-race tune: iso 2.577x
  serving-wtd, +46.6% ceiling at a 51.95% head; low-M decode-guard trap reproduced with no incumbent table).
  exp/e2e_*Qwen3.5-27B-FP8*/ 2026-08-21 (TP4, aiter-linear already ON,
  ZERO-coverage 5-family instance-race tune: 30 rows in 5 s, iso 1.523x serving-wtd, +16.5% ceiling at a
  41.22% head; engagement 980 `is tuned` on a warm TP4 server).
  exp/e2e_*Qwen3-14B-FP8*/ 2026-08-21 (TP1, AITER-OFF baseline, head 67.72%,
  NO warm-start CSV available: fresh 32-row pow2+exact-M race in 15 s -> iso 1.54-1.70x serving-wtd,
  +31..39% ceiling; low-M race measured WORSE than the untuned CK default; shipped a8w8 (non-blockscale)
  table raises in get_CKGEMM_config).
  exp/e2e_*DeepSeek-V4-Pro*/e2e_cycle1 2026-08-21 (TP8, MoE model dense head 15.84%, ZERO-coverage
  28-row pow2 race: iso 1.536x serving-wtd, +5.53% ceiling; non-pow2 N=16160 lm_head 5.95x).
  exp/e2e_*Qwen3.5-122B-A10B-FP8*/ 2026-08-21 (TP2, AITER-OFF, 5-family zero-coverage
  instance-race: 55 rows in 11 s, iso 1.83x serving-wtd, +13.0% ceiling; cold-timed racer inverts the low-M rule;
  swap-only 2.42x vs tuned 2.55x in a 4-way min-stat table; aiter-triton 0.84x; bimodal contention floor).
  exp/e2e_*DeepSeek-V4-Pro*/ 2026-08-21 (TP8, AITER-LINEAR OFF, head 20.18%,
  PARTIAL shipped coverage 2/5 families: swap-only 1.920x -> +32 raced rows + sitecustomize overlay
  iso 2.5703x serving-wtd, +14.06% ceiling; aiter-TRITON at N=7168,K=2048 numerically BROKEN for M>=1024
  (max_rel 0.68-0.90) so the dispatch trap needs a code_patch, not just a coverage probe; low-M cold-raced
  rows 1.995 -> 2.570; 2816 `is tuned` / 272 deliberate defaults on a warm TP8 server).
  exp/e2e_*Qwen3-14B-FP8-vllm*/e2e_cycle0 2026-08-22 (TP1, the e2e gate on the 51.95%-head zero-coverage
  tune: iso 2.577x -> e2e +0.18% REJECTED, while the authored-Triton lane on the same seam was
  Director-validated at +15.72%).
  exp/e2e_*Qwen3.5-122B-A10B-FP8*/ 2026-08-23..08-24 (TP2, AITER-OFF, head 20.0%,
  WARM-START REPLAY of the 08-21 122B CSV: 0/50 -> 50/50 bind, iso 1.9029x serving-wtd / +9.5% ceiling,
  decode 1.87-1.91x AND prefill 1.63-1.84x, 580 `is tuned` on a warm TP2 server, aiter-triton 0.81x,
  bpreshuffle 5th correctness dead-end; op_bench never built the CK candidate; e2e +19.45% integrate /
  +21.3% window bench = 2.05x the +9.5% ceiling, with the trace decomposition dense-GEMM −54.8% +
  MoE-chain −77% + fmoe/comm flat; **Director-validated_win 1.2081x same-session, TTFT -19.8%,
  TPOT -17.1%, parity pass**).
  Recipe: `gemm_tuning/fp8_gemm_tuning_sglang_aiter.md`; tuned CSVs under `config/ck_tune/`.
