---
key: fp8_a8w8_blockscale dense GEMM · gfx942 · sglang Triton live path
type: lever
confidence: ★★★
effect: iso ~1.06–1.16× prefill (Triton-overlay, DEPRECATED); CK-tuned KERNEL ~1.78× vs untuned Triton on the M=13645 head (kernel-level). down_proj WITH production-module rebuild: decode 4.2–4.6×, prefill M=1024 1.43× (CK wins ALL M — unconditional overlay)
confirms: 43
last_seen: 2026-08-02
status: DEPRECATED-FOR-THIS-EVAL
caution: the tuned CSV lookup logging "is tuned on cu_num" does NOT guarantee the tuned kernel RUNS — a stale prebuilt production module_gemm_a8w8_blockscale.so (shipped in image) silently falls back to the untuned default (prefill 0.73ms → looks like 0.86× loss). MUST rebuild the production module with AITER_CONFIG_GEMM_A8W8_BLOCKSCALE set (rm the prebuilt .so + blob, call the op once, gen_instances --tune_file bakes the tuned instance in) BEFORE measuring/deploying; then prefill M=1024 wins 1.43×.
---
> ✅ **43rd confirm (2026-08-02, Qwen3_14B_20260802 / Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304,
> conc=64, isl=osl=1024, qkv_proj head 11.52% GPU — N=7168/K=5120, M={1,64,1024}; reference_io.pt oracle
> sha 8adbfdf..., sibling of the o_proj/gate_up 41st/42nd this eval; sglang 0.5.12, aiter HEAD a6bb49937 =
> descendant of skill 303a583c).** CK skill end to end (bakeoff). op_bench raised the torch._dynamo
> circular-import (aiter before torch) → harness_suspect; self-repaired with a $EVAL_DIR driver
> (opbench_fp8_blockscale_driver.py: pre-import torch._dynamo, load the frozen reference_io.pt cases,
> harness_lib.time_op graph+flush). UNTUNED CK already 1.37× at M=1024 vs untuned Triton default, correct
> (rel_err 7e-5 at M=64 AND M=1024 → §9.1 non-transposed layout confirmed, matches live gfx942
> use_triton=True seam → drop-in); bpreshuffle 0.13ms but WRONG rel_err 1.44 (discarded). aiter CK tuner
> `--libtype both --mp 1` (18.8s, 3 shapes): ALL winners `libtype=ck`, errRatio 0.0, 0 failed. Tuned CSV
> engaged 3/3 "is tuned on cu_num=304" (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE, NO module rebuild this run);
> re-timed tuned CK vs Triton at M=1024: 0.134 vs 0.187ms = **1.40×** (prefill WINS here, unlike the 34th/39th
> qkv confirms that saw prefill loss — cross-run variance; CK-default≈CK-tuned so the win is the Triton→CK
> switch). amdahl @11.52% GPU, 1.40× dominant ≈ +3.4% e2e ceiling (combined CSV across the ~75% GPU fp8
> families → far larger real headroom). Shipped `winner_kind=env`
> (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=config/qkv_fp8_blockscale_tuned.csv) + reversible fp8_utils Triton→CK
> overlay (config/fp8_utils_ck_switch.overlay.diff; imports CK gemm_a8w8_blockscale + swaps the use_triton
> branch, transpose_scale unchanged=False). ckProfiler ABSENT (env_report) but IRRELEVANT — aiter CK GEMM
> tuner needs no ckProfiler. Integrator: heed the caution (rebuild the production module if a stale prebuilt
> .so ignores the tuned instance), §10.1 greedy parity M≥2 on the engaged CK server, merge the other fp8
> families into the combined CSV. flydsl author + Triton-JSON overlay + aiter-bf16 all FORBIDDEN this eval.
> e2e gate pending.

> ✅ **42nd confirm (2026-08-02, Qwen3_14B_20260802 / Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304,
> conc=64, isl=osl=1024, o_proj head 11.54% GPU — square N=K=5120, M={1,64,1024}; sglang 0.5.12,
> aiter at /sgl-workspace/aiter).** CK skill end to end (bakeoff). op_bench blockscale path benched
> Triton baseline (0.1655ms M=1024) + bpreshuffle (rel_err 41.45 WRONG, §9.1 transposed trap) → shared
> bake-off 1.0×, NOT no-win. §9.1 probe: CK correct at `transpose_scale=False` (rel_err 2e-5 M=64/1024;
> True → 0.196 garbage) → matches live gfx942 use_triton=True non-transposed seam → drop-in. aiter CK
> tuner `--libtype both --mp 1` (1 GPU visible, 3 shapes): ALL winners `libtype=ck`, errRatio 0.0.
> Engagement CONFIRMED via the REAL seam: applied the env-gated fp8_utils overlay + set
> SGLANG_USE_AITER_CK_BLOCKSCALE=1 + AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<csv>, called
> `aiter_w8a8_block_fp8_linear` → log "is tuned on cu_num=304 in .../o_proj...tuned_gemm.csv", out rel_err
> 1.7e-5 vs Triton (NO module rebuild needed this run — plain CK runtime lookup picked up the CSV). Cold
> CUDA-event A/B (harness_lib.time_op, flush_cache): **decode M=1 4.33×, M=64 4.12×; prefill M=1024 1.91×
> — CK WINS ALL M** (square K=5120 family strong at prefill too, matching 29th/32nd o_proj). amdahl
> @11.54% GPU, 1.91× dominant ≈ +5.5% e2e ceiling (decode ~4× → far larger real headroom across the
> ~75% GPU the 4-family combined CSV covers). Shipped `winner_kind=env` (SGLANG_USE_AITER=1
> SGLANG_USE_AITER_CK_BLOCKSCALE=1 AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=config/o_proj_fp8_a8w8_blockscale_tuned_gemm.csv)
> + reversible env-gated fp8_utils Triton→CK overlay (config/o_proj_fp8_ck_switch.fp8_utils.diff, DEDICATED
> CK branch keeps transpose_scale=False — env off preserves original behavior). ckProfiler ABSENT in
> env_report (CK not in available_backends) but IRRELEVANT — the aiter CK GEMM tuner needs no ckProfiler
> (that only gates the CK attention sweep); CK GEMM lever fully feasible. Integrator: §10.1 greedy parity
> M≥2 on the engaged CK server + HIP-graph decode capture; merge other fp8 families into combined CSV.
> flydsl author + Triton-JSON overlay + aiter-bf16 all FORBIDDEN for this op this eval. e2e gate pending.

> ✅ **41st confirm (2026-08-02, Qwen3_14B_20260802 / Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304,
> gate_up_proj head 15.4% GPU — N=34816/K=5120, M={1,64,1024}; sglang 0.5.12, aiter at /sgl-workspace/aiter).**
> CK skill end to end (bakeoff). op_bench raised the torch._dynamo circular-import (aiter before torch) →
> harness_suspect; self-repaired via a driver in $EVAL_DIR (import torch first, mirror unittest._synth).
> Untuned CK already 2.17×/2.15×/1.79× (M=1/64/1024), serving-weighted 2.04× vs untuned Triton default,
> ck_vs_triton rel_err 0.0000 (§9.1 non-transposed correct at M≥2). aiter CK tuner `--libtype both --mp 1`
> (1 GPU visible) → all 3 target shapes `libtype=ck`, errRatio 0.0. Tuned CSV engaged (3/3 "is tuned on
> cu_num=304"); re-timed tuned CK vs Triton: decode M=1 1.99×, M=64 1.98×; prefill M=1024 1.69× — CK WINS
> ALL M, serving-weighted 1.89×; amdahl @15.4% GPU ≈ +7.8% e2e ceiling. Shipped `winner_kind=env`
> (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=config/a8w8_blockscale_tuned_gemm.csv + SGLANG_USE_AITER_CK_BLOCKSCALE=1)
> + fp8_utils Triton→CK overlay (config/fp8_utils_ck_switch.overlay.diff; DEDICATED CK branch keeps
> transpose_scale=False — a naive use_triton flip would route bpreshuffle+transposed = the §9.1 garbage trap).
> Integrator: heed the caution (rebuild the production module with env set if the prebuilt .so ignores the
> tuned instance), run §10.1 greedy parity M≥2 on the engaged CK server. e2e gate pending.

> ✅ **40th confirm (2026-08-02, Qwen3_14B_20260802 / Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304,
> conc=64, isl=osl=1024, down_proj head 36.95% GPU (live_pct 75.41% across 4 fp8 families) —
> N=5120/K=17408, M={1,64,1024}; sglang 0.5.12, aiter HEAD a6bb49937 = descendant of skill 303a583c).**
> CK skill end to end (bakeoff). op_bench dominant M=1024 maps "ck"→bpreshuffle (rel_err 41.68 WRONG,
> §9.1 transposed trap) → shows 1.0×, NOT no-win. aiter CK tuner `--libtype both --mp 1` (1 GPU visible;
> 12 shapes = 4 families qkv/o/gate_up/down × M{1,64,1024}): ALL winners `libtype=ck`, errRatio 0.0, 0
> failed (22s head, ~90s full). §9.1 (down M=16): CK correct `transpose_scale=False` rel_err 3e-5,
> transposed → 0.193 garbage → matches live gfx942 use_triton=True non-transposed seam → drop-in.
> **HIT THE CAUTION'S STALE-PREBUILT BUG:** env+CSV set, all shapes logged "is tuned on cu_num=304",
> yet cold A/B showed down M=1024 = 0.724ms = 0.73× LOSS (the prebuilt module_gemm_a8w8_blockscale.so
> ignored the tuned instance). Forcing production rebuild via tuner `--run_config <tuned.csv>` (JIT
> rebuild) baked the tuned instances in → re-bench CK WINS ALL M ALL families: down 4.61×/4.11×/1.43×,
> qkv 4.15×/3.90×/1.64×, o 4.26×/3.83×/1.45×, gate_up 3.91×/3.27×/1.92× (rel_err ~2e-5 all). down_proj
> GEAK serving-weighted **2.66×**; amdahl @36.95% ≈ +30% e2e (4-family overlay @75.41% ≈ +89% ceiling).
> Shipped `winner_kind=env` (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=config/a8w8_blockscale_tuned_gemm.csv,
> 12-shape combined) + fp8_utils Triton→CK overlay (config/fp8_utils_ck_switch.diff, UNCONDITIONAL CK,
> transpose_scale unchanged). Integrator MUST rebuild the production module with the env set (`--run_config`)
> before the e2e A/B (else silent prefill-loss fallback), run §10.1 greedy parity M≥2 on the engaged CK
> server. flydsl author + Triton-JSON overlay + aiter-bf16 all FORBIDDEN for this op this eval. e2e gate pending.

> ✅ **39th confirm (2026-08-01, Qwen3_14B_20260801 / Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304,
> conc=64, isl=osl=1024, qkv_proj head 11.61% GPU — N=7168/K=5120, M={1,64,1024}; THIRD sibling of the
> down_proj(37th) + gate_up(38th) heads, same eval/seam).** CK skill end to end (bakeoff). op_bench
> hit the torch._dynamo circular-import (aiter imported before torch) → harness_suspect; self-repaired
> with a driver in $EVAL_DIR (pre-import torch._dynamo, mirror unittest._synth). aiter CK tuner
> `--libtype both --mp 1` (1 GPU visible, fast, 12 shapes across all 4 fp8 families qkv/o/gate_up/down):
> ALL winners `libtype=ck`, errRatio 0.0, 0 failed. §9.1 (M=64,qkv): CK correct at
> `transpose_scale=False` rel_err 1.6e-5, transposed → 0.199 garbage → matches live gfx942 use_triton=True
> non-transposed seam → drop-in. Engagement CONFIRMED 3/3 "is tuned on cu_num=304". Cold CUDA-event A/B
> vs untuned Triton live default (rel_err ~2e-5 all M): **decode M=1 4.81×, M=64 4.74×; prefill M=1024
> 1.06× — CK WINS ALL M** (weaker prefill than down/gate_up siblings — qkv N=7168/K=5120 is a squarer
> shape where Triton prefill is already decent; CK-default≈CK-tuned here so the win is the Triton→CK
> switch, not the per-shape tune). GEAK serving-weighted ≈2.32×; amdahl @11.61% GPU ≈+7.1% e2e ceiling
> (the combined CSV covers ~75% GPU across all 3 sibling heads → far larger real headroom). Shipped
> `winner_kind=env` (AITER_USE_CK_BLOCKSCALE=1 + AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=config/
> a8w8_blockscale_tuned_gemm.csv) + fp8_utils Triton→CK overlay (config/fp8_utils_ck_switch.diff).
> Integrator MUST heed the caution: rebuild the production module with the env set before the e2e A/B
> (stale prebuilt .so can silently ignore the tuned instance), run §10.1 greedy parity M≥2 on the
> engaged CK server, and use the ONE combined 12-shape CSV for all fp8 families. e2e gate pending.

> ✅ **38th confirm (2026-08-01, Qwen3_14B_20260801 / Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304,
> conc=64, isl=osl=1024, gate_up_proj head 15.46% GPU — N=34816/K=5120, M={1,64,1024}; sibling of the
> down_proj head in the 37th confirm, same eval).** CK skill end to end (bakeoff). op_bench dominant
> M=1024 maps "ck"→bpreshuffle (rel_err 46.48 WRONG) → 1.0× (NOT no-win). aiter CK tuner `--libtype
> both --mp 1` (3 shapes): all winners `libtype=ck`, errRatio 0.0, 0 failed. §9.1 (M=16): CK correct at
> `transpose_scale=False` rel_err 1.4e-5, transposed → 0.196 garbage. **Hit the caution's stale-prebuilt
> bug directly:** with only the env set + AITER_LOG_TUNED_CONFIG, all 3 shapes logged "is tuned on
> cu_num=304" yet M=1024 still ran 1.4556ms (0.61× LOSS) — the prebuilt module_gemm_a8w8_blockscale.so
> ignored the tuned instance. Forcing a production rebuild via the tuner's `--run_config <tuned.csv>`
> (JIT rebuild 142s) gave the true tuned kernel: M=1→46.2µs, M=64→68.2µs, M=1024→476.3µs (checkAllclose
> atol/rtol 0.01 passed). Cold CUDA-event A/B vs untuned Triton (live default): **decode M=1 3.06×,
> M=64 2.00×; prefill M=1024 1.85× — CK WINS ALL M.** GEAK serving-weighted ≈2.0×; amdahl @15.46% GPU
> ≈7.8% e2e ceiling. Shipped `winner_kind=env` (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=config/ck_tune/
> a8w8_blockscale_tuned_gemm.csv) + fp8_utils Triton→CK overlay (overlay/ck_fp8_switch/
> fp8_utils.ck_switch.diff). Integrator MUST rebuild the production module with the env set before the
> e2e A/B (else silent fallback), run §10.1 greedy parity M≥2 on the engaged CK server, and fold the
> other fp8 families' shapes into the combined CSV. e2e gate pending.

> ✅ **37th confirm (2026-08-01, Qwen3_14B_20260801 / Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304,
> conc=64, isl=osl=1024, down_proj head 36.71% GPU — N=5120/K=17408, M={1,64,1024}; sglang 0.5.12,
> aiter HEAD a6bb49937 = descendant of skill's 303a583c, fresh immutable-unittest eval, synthesized
> oracle, no reference_io.pt).** CK skill end to end (bakeoff phase). op_bench dominant-bucket M=1024
> maps "ck"→bpreshuffle (WRONG rel_err 41.68) so shared bake-off shows 1.0× — NOT a no-win. aiter CK
> tuner `--libtype both --mp 1` (24.3s, 3 shapes): ALL winners `libtype=ck`, errRatio 0.0, 0 failed;
> M=1024 winner = kernelId 0 (256x128x128x128 intrawave_v3). Engagement CONFIRMED 3/3 "is tuned on
> cu_num=304" from tuned CSV (AITER_LOG_TUNED_CONFIG=1, no module rebuild needed this run). §9.1 probe
> (M=16): CK correct at `transpose_scale=False` rel_err 1e-5, transposed → 0.188 garbage → live gfx942
> use_triton=True seam already produces non-transposed → drop-in. Cold CUDA-event A/B (tuned CSV engaged,
> transpose_scale=False, rel_err ~1e-5 all M): **decode M=1 4.17×, M=64 4.14×; prefill M=1024 1.70× — CK
> WINS at ALL M** (stronger prefill than the 33rd's 1.43×; no production-module rebuild required here for
> the prefill win, unlike the 33rd — cross-run variance, self-verify at integrate). GEAK serving-weighted
> ≈2.89×; amdahl_ceiling @36.71% GPU ≈ 24% e2e — huge headroom, far above noise. Shipped `winner_kind=env`
> (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=/wekafs/test_results/Qwen3_14B_20260801/config/a8w8_blockscale_down_tuned.csv)
> + fp8_utils Triton→CK switch overlay (config/fp8_utils_ck_switch.overlay.diff, unconditional CK, env
> M-routable via SGLANG_CK_BLOCKSCALE_MAX_M, applies clean via `git apply`). ckProfiler ABSENT in
> env_report but IRRELEVANT — aiter CK tuner needs no ckProfiler; CK env lever fully feasible. e2e gate
> pending (Integrator; §10.1 greedy parity on the engaged CK server + HIP-graph decode capture; CSV
> covers only down_proj shapes → Integrator must tune the other fp8 families into the combined CSV).
> ✅ **36th confirm (2026-07-31, Qwen3_14B_20260731 / Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304,
> conc=64, isl=osl=1024, FULL 4-family fp8 blockscale head 75.04% GPU — down N=5120/K=17408, gate_up
> N=34816/K=5120, qkv N=7168/K=5120, o N=5120/K=5120, M={1,64,1024}; sglang 0.5.12, aiter HEAD at
> /sgl-workspace/aiter, fresh immutable-unittest eval, synthesized oracle, no reference_io.pt).** CK skill
> end to end. HARNESS SELF-REPAIR: op_bench raised `torch._dynamo` circular-import AttributeError because
> running with cwd=task_dir shadows stdlib `unittest` with the task's immutable `unittest.py`; fixed by
> invoking op_bench from a neutral cwd with an absolute --task path (no edit to shared op_bench.py or the
> immutable UT). op_bench dominant-bucket M=1024: Triton baseline 0.5265ms, CK target 0.3754ms = **1.40×**
> correct (rel_err 0.0073); bpreshuffle 0.3195ms but WRONG rel_err 41.68 (§9.1 transposed-scale trap) →
> discarded. §9.1 probe (M=16,64 × 2 NK): CK correct at `transpose_scale=False` (rel_err ~0), transposed →
> ~0.19 garbage — live gfx942 use_triton=True seam already produces non-transposed → drop-in swap. aiter CK
> tuner `--libtype both --mp 1` (~5min, 28 shapes = 4 families × M{1,16,64,128,256,512,1024}): ALL winners
> `libtype=ck`, errRatio 0.0, 0 failed. Engagement CONFIRMED "is tuned on cu_num=304" from the tuned CSV
> (AITER_LOG_TUNED_CONFIG=1, no module rebuild needed this aiter build). op_bench re-run with tuned CSV
> engaged: **1.4247×** at M=1024 (CK 0.3704 vs Triton 0.5277ms). amdahl_ceiling @75.04% GPU ≈ 28.8% e2e —
> huge headroom, far above noise. Shipped `winner_kind=env`
> (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=/wekafs/test_results/Qwen3_14B_20260731/config/a8w8_blockscale_tuned_qwen3_14b.csv)
> + fp8_utils Triton→CK switch overlay (config/fp8_utils_ck_switch.overlay.diff, import CK
> `gemm_a8w8_blockscale` + swap op, scale layout unchanged). e2e gate pending (Integrator; §10.1 greedy
> parity on the engaged CK server + HIP-graph decode capture).
> ✅ **35th confirm (2026-07-29, Qwen3_14B_20260730 / Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, conc=64,
> isl=osl=1024, o_proj head 11.2% GPU — square N=K=5120, M={1,64,1024}, fresh immutable-unittest eval,
> synthesized oracle sha="", aiter CK kernel `aiter.gemm_a8w8_blockscale` present + CK tuner runnable).**
> CK skill end to end. op_bench single-bucket M=1024 probe: Triton baseline 0.1679ms, untuned CK target
> 0.2341ms (0.72×, prefill-only), bpreshuffle WRONG rel_err 41.45 → shared bake-off shows 1.0× — NOT a
> no-win. aiter CK tuner `--libtype both --mp 1` (23s, 3 shapes; also ran a 12-shape full-model CSV covering
> qkv/o/gate_up/down for the model-wide overlay, all winners `libtype=ck`, errRatio 0.0). Runtime CSV
> engagement CONFIRMED 3/3 "is tuned on cu_num=304" (AITER_LOG_TUNED_CONFIG=1, NO module rebuild needed on
> this aiter build). §9.1: CK correct at `transpose_scale=False` (immutable-UT rel_err ~0.007 all M incl
> 3-draw random-parity + graph-replay PASS; live gfx942 use_triton=True seam already produces non-transposed
> → drop-in). Immutable-UT serving-weighted A/B (tuned CSV engaged): **decode M=1 2.28×, M=64 2.43-2.69×;
> prefill M=1024 0.70-0.72× (SLOWER) → GEAK_WEIGHTED 1.99-2.15× (geomean 1.61)**. ⚠️ CONTRAST the 32nd/29th
> o_proj confirms which saw prefill M=1024 CK WIN 1.31-1.35× — THIS run's tuned M=1024 kernel = kernelId 0
> (=CK default intrawave_v3, 101us tuner-hot vs 236us UT-cold) LOSES to Triton at prefill → cross-run
> variance, so shipped **M-ROUTED** overlay (M≤SGLANG_CK_BLOCKSCALE_MAX_M=64→CK, else Triton; clean
> `patch -p1`) as the SAFE ship, not unconditional. `winner_kind=env`
> (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<full 4-family tuned csv>). amdahl_ceiling @11.2% GPU, ~2.0× iso ≈
> +5.6% e2e. ⚠️ overlay is model-wide → tuned all 4 fp8 families into the CSV so non-o_proj shapes don't
> run untuned-CK default; Integrator: §10.1 greedy parity on the engaged CK server + HIP-graph decode
> capture before the e2e gate. e2e gate pending (Integrator).
> ✅ **34th confirm (2026-07-29, Qwen3_14B_20260730 / Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, conc=64,
> isl=osl=1024, qkv_proj head 11.18% GPU — N=7168,K=5120, M={1,64,1024}, fresh immutable-unittest eval,
> synthesized oracle sha="", aiter HEAD a6bb49937 = descendant of skill's 303a583c).** CK skill end to end.
> op_bench single-bucket M=1024 probe: Triton baseline 0.184ms, untuned CK target 0.316ms (0.60×, prefill-only),
> bpreshuffle WRONG rel_err 41.5 → shared bake-off shows 1.0× — NOT a no-win. aiter CK tuner `--libtype both
> --mp 1` (23s, 3 shapes): all winners `libtype=ck`, errRatio 0.0; RUNTIME CSV engagement CONFIRMED 3/3
> "is tuned on cu_num=304" (AITER_LOG_TUNED_CONFIG=1, no module rebuild needed this run). §9.1: CK correct at
> `transpose_scale=False` (immutable-UT rel_err ~0.007 all M incl 3-draw random-parity + graph-replay PASS; the
> live gfx942 use_triton=True seam already produces non-transposed → drop-in). Immutable-UT serving-weighted A/B
> (tuned CSV engaged): UNCONDITIONAL CK **decode M=1 2.17×, M=64 2.36×; prefill M=1024 0.60× → GEAK_WEIGHTED 1.81×
> (geomean 1.45)**. M-ROUTED (M≤64→CK, else Triton) reconfirmed STRICTLY BETTER: decode M=1 2.07×, M=64 2.51×,
> prefill 0.99× (parity) → **GEAK_WEIGHTED 2.18× (geomean 1.72)**. qkv prefill LOSES here (0.60×) unlike the 23rd
> confirm's qkv prefill 1.08× WIN — cross-run variance, so M-routing (not unconditional) is the safe ship for qkv
> too. Shipped M-routed fp8_utils Triton→CK overlay (clean `patch -p1`, `SGLANG_CK_BLOCKSCALE_MAX_M` default 64)
> + `winner_kind=env` (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<qkv tuned csv>). amdahl_ceiling @11.18% GPU, 2.18× iso
> ≈ +6.1% e2e. ⚠️ CSV covers ONLY qkv shapes; Integrator must merge the other fp8 families' tuned rows +§10.1
> greedy parity on the engaged CK server + HIP-graph decode capture. e2e gate pending (Integrator).
> ✅ **33rd confirm (2026-07-29, Qwen3_14B_20260730 / Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, conc=64,
> isl=osl=1024, down_proj head 36.12% GPU — N=5120,K=17408, M={1,64,1024}, fresh immutable-unittest eval,
> synthesized oracle sha="").** Re-confirm of the 19th/22nd/27th/30th down_proj, but RESOLVES the prefill-loss
> ambiguity. aiter CK tuner `--libtype both --mp 1` (726s incl. tune-module JIT build; 3 shapes): all winners
> `libtype=ck`, errRatio 0.0; tuned CSV M=1024 kernel = ...256x128x128x128...intrawave_v3, 325us. Engagement
> "is tuned on cu_num=304" 3/3. **KEY: rebuilt the stale prebuilt production module_gemm_a8w8_blockscale.so
> (May-16 image build) WITH the CSV** (rm .so+blob → call op once → gen_instances --tune_file, 174s). AFTER
> rebuild, real warm A/B tuned-CK vs frozen Triton (relerr 0/1e-5/2e-5, §9.1 transpose_scale=False): **decode
> M=1 4.60×, M=64 4.21×; prefill M=1024 1.43× — CK WINS at ALL M**, unlike the 30th confirm's prefill 0.86×
> LOSS which was the un-rebuilt production module silently running the untuned default (the "is tuned" log line
> alone is NOT proof the tuned kernel executes). op_bench cold M=1024 after rebuild: Triton 0.5263 → tuned CK
> 0.3782 = **1.39×**, amdahl_ceiling @36.12% GPU = **11.3%**. Shipped UNCONDITIONAL CK overlay (clean `patch
> -p1`, all M win so no M-routing needed) + `winner_kind=env` (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<tuned.csv>).
> ⚠️ CSV covers ONLY the down_proj shapes; the overlay routes the WHOLE fp8 blockscale seam (qkv/gate_up/o/lm_head)
> to CK — Integrator MUST capture the full live (M,N,K) set (SGLANG_DUMP_AITER_FP8_GEMM_SHAPES) + tune all
> families into the combined CSV, rebuild the production module, then §10.1 greedy parity on the engaged CK
> server before the e2e gate, else untuned-CK families may regress. e2e gate pending (Integrator).
> ✅ **32nd confirm (2026-07-25, Qwen3_14B_20260729 / Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, conc=64,
> isl=osl=1024, o_proj head 11.44% GPU — square N=K=5120, M={1,64,1024}, fresh immutable-unittest eval,
> synthesized oracle sha="").** Re-confirm of the 21st/23rd/29th o_proj. aiter CK tuner `--libtype both --mp 1`
> (23.4s, 3 shapes): all winners `libtype=ck`, errRatio 0.0. Engagement CONFIRMED 3/3 "is tuned on cu_num=304"
> (AITER_LOG_TUNED_CONFIG=1, tuned CSV loaded at runtime — no rebuild needed this run). Cold CUDA-event A/B
> (tuned CK vs frozen untuned Triton, §9.1 transpose_scale=False, relerr 0.0/2e-5/2e-5 at M=1/64/1024):
> **decode M=1 4.37×, M=64 4.46×; prefill M=1024 1.31× — CK WINS at ALL M** (square K=5120 family; can route
> ALL M→CK, unlike gate_up/down prefill loss). Weighted iso ≈2.58× (M-routed decode-only ≈2.19× under shared
> MAX_M=64 overlay). op_bench single-bucket M=1024 probe reports untuned-CK 0.2323 vs Triton 0.1661 =1.0× (untuned
> CK prefill sample; bpreshuffle WRONG rel_err 41.5) — NOT a no-win. Shipped `winner_kind=env`
> (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<o_proj tuned csv>) + shared fp8_utils Triton→CK M-routed overlay;
> recommend Integrator merge o_proj rows into combined CSV + raise MAX_M for this family to bank prefill 1.31×.
> e2e gate pending (Integrator; §10.1 greedy parity on engaged CK server + HIP-graph decode capture).
> ✅ **31st confirm (2026-07-25, Qwen3_14B_20260729 / Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, conc=64,
> isl=osl=1024, gate_up_proj head 15.23% GPU — N=34816,K=5120, M={1,64,1024}, fresh immutable-unittest eval,
> synthesized oracle sha="", aiter HEAD a6bb49937).** CK skill end to end. op_bench: Triton baseline 0.805ms,
> untuned CK target 0.504ms (**1.60×**), bpreshuffle WRONG rel_err 46.5. aiter CK tuner `--libtype both --mp 1`
> (26s): all 3 winners `libtype=ck`, errRatio 0.0. **RUNTIME CSV engagement CONFIRMED on aiter HEAD
> a6bb49937** — `AITER_LOG_TUNED_CONFIG=1` printed "is tuned on cu_num=304" 3/3 with the tuned kernelName
> loaded from the CSV; **NO module rebuild was needed here** (contrast the 28th confirm's rebuild-gotcha on
> the same HEAD — the plain CK `gemm_a8w8_blockscale` runtime lookup DID pick up the CSV this run; still
> self-verify engagement at integrate). Real warm A/B (tuned CSV engaged, relerr 0.0000 vs Triton at all M →
> §9.1 `transpose_scale=False` drop-in reconfirmed): **decode M=1 3.38×, M=64 2.67×; prefill M=1024 1.87× —
> CK WINS at ALL M including prefill** (matches the 28th's tuned-lookup finding; unlike the 20th/24th UNTUNED
> gate_up prefill 0.60-0.65× loss). ⇒ shipped **UNCONDITIONAL CK overlay** (SGLANG_USE_CK_BLOCKSCALE=1, reversible,
> not M-routed — CK wins every bucket here) + `winner_kind=env` (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<tuned csv>).
> amdahl_ceiling @15.23% GPU = 6.04% (op_bench 1.60×; per-bucket decode wins push higher). e2e gate pending
> (Integrator; §10.1 greedy parity on engaged CK server + HIP-graph decode capture).
> ✅ **30th confirm (2026-07-25, Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, conc=64, isl=osl=1024,
> down_proj head 36.27% GPU — N=5120,K=17408, M={1,64,1024}, fresh immutable-unittest eval, synthesized
> oracle sha="", aiter HEAD a6bb49937).** Re-confirm of the 19th/22nd/27th down_proj. op_bench blockscale
> path: Triton baseline 0.528ms, untuned CK target 0.752ms (0.70×), bpreshuffle WRONG rel_err 41.68 — the
> single-bucket M=1024 probe = 1.0× is prefill-only, NOT a no-win. Real per-bucket cold CUDA-event A/B
> (CK `aiter.gemm_a8w8_blockscale` vs frozen Triton, relerr 0.0/5e-5/7e-5 at M=1/64/1024 → §9.1
> `transpose_scale=False` drop-in reconfirmed): **decode M=1 4.16×, M=64 4.36× FASTER; prefill M=1024 0.86×
> SLOWER → routed to Triton**. aiter CK tuner `--libtype both --mp 1` (single GPU on box): all 3 winners
> `libtype=ck`, errRatio 0.0; `get_CKGEMM_config` resolves all 3 shapes from the tuned CSV (config-lookup
> engagement confirmed). Tuned CSV ≈ CK default for the decode buckets (default already optimal at small M),
> so the M-routed overlay banks the decode win without needing the 28th-confirm module rebuild (rebuild only
> matters if you route prefill→CK). Shipped M-routed fp8_utils overlay (M≤SGLANG_CK_BLOCKSCALE_MAX_M=64→CK,
> else Triton, clean `git apply`) + `winner_kind=env` (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<tuned.csv>). e2e
> gate pending (Integrator; §10.1 greedy parity on engaged CK server + HIP-graph decode capture).
> ✅ **29th confirm (2026-07-25, Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, conc=64, isl=osl=1024,
> o_proj head 12.17% GPU — square N=K=5120, M={1,64,1024}; aiter HEAD a6bb49937).** CK skill on o_proj,
> aiter CK tuner `--libtype both --mp 1` (35.9s, 7 shapes M={1,16,32,64,128,256,1024}): ALL winners
> `libtype=ck`, errRatio 0.0. Runtime CSV engagement CONFIRMED (contra the 28th gotcha on this same aiter
> build): `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<tuned_o_proj.csv>` + AITER_LOG_TUNED_CONFIG=1 printed
> "is tuned on cu_num=304" 3/3 AND real cold CUDA-event A/B vs untuned Triton: **decode M=1 5.09×, M=64
> 5.14×; prefill M=1024 1.32× — CK WINS at ALL M** (square K=5120 family, matching the 23rd confirm's o_proj
> prefill 1.35×; unlike down_proj/gate_up prefill 0.6–0.8× loss). §9.1: CK correct at `transpose_scale=False`
> (relerr 0.0/1e-5/2e-5 at M=1/64/1024 vs Triton — the live gfx942 use_triton=True seam already produces
> non-transposed → drop-in). Merged o_proj rows into `ck_tune/combined_blockscale_tuned_gemm.csv` (now 3 NK
> families: down N=5120/K=17408, gate_up N=34816/K=5120, o N=K=5120). Reused the shared M-routed fp8_utils
> overlay; SINCE o_proj wins prefill too, recommend the Integrator raise SGLANG_FP8_CK_M_MAX (or per-(N,K)
> route) to also bank o_proj prefill 1.32×. Shipped `winner_kind=env` (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=
> combined csv) + fp8_utils Triton→CK overlay. op_bench "ck"→bpreshuffle still WRONG (rel_err 41.5) → shared
> bake-off shows 1.0×, NOT a no-win. e2e gate pending (Integrator; verify HIP-graph decode capture + §10.1
> greedy parity on the engaged CK server).
> 🔧 **28th confirm + NEW GOTCHA (2026-07-25, Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, conc=64,
> isl=osl=1024, gate_up_proj head 16.44% GPU — N=34816,K=5120, M={1,64,1024}; aiter HEAD a6bb49937,
> a DESCENDANT of the skill's pinned 303a583c).** ⚠️ **On this aiter build the CK blockscale instance is
> selected by a COMPILE-TIME lookup table (`gemm_a8w8_blockscale.cu` → `GENERATE_LOOKUP_TABLE` from
> `gemm_a8w8_blockscale_lookup.h`), NOT read from the tuned CSV at runtime.** The Python wrapper's
> `get_CKGEMM_config` only picks libtype (ck vs cktile); for libtype=ck it calls the SAME
> `gemm_a8w8_blockscale_ck` whether tuned or not. Consequence: `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<csv>`
> ALONE = **0 engagement** (verified: tuned==untuned==1.49ms @ M=1024; the lookup is empty so it falls to
> the slow legacy `256x16x128x256_v1` default → CK is a REGRESSION vs Triton, 0.59×). **To engage you MUST
> rebuild the module with the CSV baked into the lookup:** the build recipe
> (`optCompilerConfig.json:module_gemm_a8w8_blockscale`) runs `gen_instances.py --tune_file
> {AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE}`, so `rm aiter/jit/module_gemm_a8w8_blockscale.so` with the env
> set → next `import aiter` rebuilds (166s, compiles only the tuned instances) with the tuned lookup. AFTER
> rebuild, engagement confirmed and immutable UT **PASSES: decode M=1 1.10×, M=64 1.16×, prefill M=1024
> **1.73×** FASTER → GEAK_WEIGHTED_SPEEDUP=1.28× (geomean 1.30), rel-err <1e-4 all M, graph-replay + 3-draw
> random-parity PASS. NOTE: unlike the 20th/24th confirms (which saw gate_up prefill 0.60-0.65× SLOWER on
> the UNTUNED default and M-routed prefill→Triton), with the tuned lookup BAKED IN, CK wins at ALL M
> including prefill — so the overlay should route ALL M to CK here, not just M≤64. §9.1: CK correct at
> `transpose_scale=False` (the live gfx942 `use_triton=True` seam already produces non-transposed → drop-in
> swap, no scale change). op_bench blockscale path still maps "ck"→bpreshuffle (WRONG rel_err 46.5) so the
> shared bake-off shows 1.0× — measure real CK via `aiter:gemm_a8w8_blockscale` + the rebuilt lookup.
> Shipped `winner_kind=env` (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<merged csv>) + fp8_utils Triton→CK overlay
> + **mandatory module-rebuild prerequisite**; tuning_artifact = merged tuned csv. e2e gate pending.
> ✅ **27th confirm (2026-07-25, Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, conc=64, isl=osl=1024,
> down_proj head 40.32% GPU — N=5120,K=17408, M={1,64,1024}, fresh immutable-unittest eval, synthesized
> oracle sha="").** op_bench mapped "ck"→bpreshuffle (WRONG, rel_err 41.68 — §9.1 transposed-scale bug) so
> the shared bake-off shows 1.0×; the REAL CK is plain `aiter.gemm_a8w8_blockscale`. Ran it via the immutable
> UT (CURRENT_GEMM_CALLABLE): PASS (eager+3-draw random-parity+graph-replay), **decode M=1 2.56×, M=64 2.90×
> FASTER; prefill M=1024 0.71× SLOWER → GEAK_WEIGHTED_SPEEDUP=2.23× (geomean 1.74)**. aiter CK tuner
> `--libtype both --mp 1` (38s, 7 shapes M={1,16,32,64,128,256,1024}): ALL winners `libtype=ck`, errRatio 0.0;
> engagement verified 3/3 "is tuned on cu_num=304" (AITER_LOG_TUNED_CONFIG=1). Tuned CSV ≈ CK default in the
> UT (prefill still 0.71× cold/graph — tuner's 324us/M1024 is hot-cache; trust the UT graph A/B). §9.1: plain
> CK correct at NON-transposed scale (use_triton=True branch already produces it). Shipped M-routed fp8_utils
> overlay (M≤SGLANG_FP8_CK_M_MAX=64→CK, else Triton) + `winner_kind=env`
> (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<tuned.csv>). ckProfiler ABSENT in env_report but IRRELEVANT — the aiter
> CK tuner needs no ckProfiler; CK lever fully feasible. e2e gate pending (Integrator).
> ✅ **26th confirm (2026-07-25, Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, FULL 4-family head 73.18%
> GPU — down/gate_up/qkv/o × M={1,64,1024}, fresh immutable-unittest eval, synthesized oracle sha="").**
> CK skill end to end: aiter CK tuner `--libtype both --mp 1` (52s), ALL 12 winners `libtype=ck`,
> errRatio 0.0; engagement verified 12/12 "is tuned on cu_num=304" (AITER_LOG_TUNED_CONFIG=1). §9.1: CK
> correct at `transpose_scale=False` (relerr 4e-5 at M=16; transposed → 0.193 garbage). Warm CUDA-event A/B
> (tuned CSV engaged, parity relerr ~0 all buckets): **decode M=1/M=64 — down 4.30/4.53×, gate_up 3.80/2.24×,
> qkv 4.61/4.95×, o 4.69/4.95× FASTER; prefill M=1024 — down 0.94×, gate_up 0.65× (SLOWER→Triton), qkv 1.07×,
> o 1.31×** → REGIME-SPLIT reconfirmed (decode geomean ≈4.15×). op_bench single-bucket M=1024 probe = 1.0×
> (untuned CK target 0.759 vs Triton 0.528ms, prefill-only) — NOT a no-win. Shipped M-routed fp8_utils CK
> overlay (M≤SGLANG_CK_BLOCKSCALE_MAX_M=64→CK, else Triton) + `winner_kind=env`
> (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<tuned.csv>). e2e gate pending (Integrator).
> ✅ **25th confirm (2026-07-25, Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, conc=64, isl=osl=1024,
> FULL 4-family head 76.17% GPU — down/gate_up/qkv/o × M={1,64,1024}, fresh immutable-unittest eval).**
> CK skill end to end: aiter CK tuner `--libtype both --mp 1` (52s), ALL 12 winners `libtype=ck`,
> errRatio 0.0; engagement verified 12/12 "is tuned on cu_num=304" (AITER_LOG_TUNED_CONFIG=1). §9.1: CK
> correct at `transpose_scale=False` (relerr 1.4–1.9e-5 at M=64 all 4 families; transposed → 0.19 garbage).
> Per-bucket cold CUDA-event A/B (tuned CSV engaged): **decode M=1/M=64 — down 4.45/4.64×, gate_up
> 3.95/2.27×, qkv 4.99/5.08×, o 4.80/5.05× FASTER; prefill M=1024 — down 0.93×, gate_up 0.65× (SLOWER,
> routed to Triton), qkv 1.05×/o 1.30×** → REGIME-SPLIT reconfirmed; share-weighted decode ≈4.4×.
> op_bench single-bucket M=1024 probe = 1.0× (CK target 0.757 vs Triton 0.531ms, prefill-only) — NOT a
> no-win. Shipped M-routed fp8_utils CK overlay (M≤SGLANG_CK_BLOCKSCALE_MAX_M=64→CK, else Triton) +
> `winner_kind=env` (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<tuned.csv>). e2e gate pending (Integrator).
> ✅ **24th confirm (2026-07-25, Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, conc=64, isl=osl=1024,
> gate_up_proj head 15.41% GPU — N=34816,K=5120, RE-CONFIRM of the 20th, fresh eval, immutable-unittest task).**
> CK skill on gate_up (M={1,64,1024}), aiter CK tuner `--libtype both --mp 1` (24s): all 3 winners
> `libtype=ck`, errRatio 0.0; engagement verified ("is tuned on cu_num=304" M=1/64/1024 via
> AITER_LOG_TUNED_CONFIG=1). §9.1: CK correct at `transpose_scale=False` (relerr 1.8e-5 at M=16; transposed →
> 0.18 garbage; bpreshuffle drop-in WRONG relerr 46.5). Per-bucket cold CUDA-event A/B (tuned CSV engaged,
> parity relerr ~1e-5 all M): **decode M=1 4.01×, M=64 2.27× FASTER; prefill M=1024 0.64× (SLOWER→routed to
> Triton)** → REGIME-SPLIT reconfirmed; NOTE gate_up M=64 came in at 2.27× this eval (vs the 20th's weak
> 1.38×) — still the weakest-decode of the four families but healthier here. Reused the global M-routed
> fp8_utils overlay (M≤64→CK) + `winner_kind=env` (`AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<combined tuned csv>`;
> gate_up rows merged in, now covering all 4 NK families). op_bench single-bucket M=1024 probe = 1.0× (untuned
> CK target 1.50ms vs Triton 0.81ms, prefill-only sample) — NOT a no-win. e2e gate pending (Integrator).
> ✅ **23rd confirm (2026-07-25, Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, conc=64, isl=osl=1024,
> MERGED qkv+o_proj head 22.9% GPU — qkv N=7168,K=5120 + o_proj N=5120,K=5120, M={1,64,1024}).** CK skill,
> aiter CK tuner `--libtype both --mp 1` (30s): all 6 winners `libtype=ck`, errRatio 0.0; engagement
> verified ("is tuned on cu_num=304" both families via AITER_LOG_TUNED_CONFIG=1). §9.1: CK correct at
> `transpose_scale=False` (relerr 2e-5 at M=64; transposed → 0.19/0.20 garbage). Per-bucket cold CUDA-event
> A/B (tuned CSV engaged): **decode M=1 qkv 5.09× / o_proj 4.88×; M=64 qkv 5.19× / o_proj 5.17×; prefill
> M=1024 qkv 1.08× / o_proj 1.35× — CK WINS at prefill too here** (unlike down_proj's 0.70× prefill loss;
> these square-ish/wide-N K=5120 families are strong across ALL M). Serving-weighted (M-routed overlay,
> decode→CK, prefill→Triton to protect down_proj) ≈2.29×. op_bench single-bucket M=1024 probe = 1.0×
> (prefill sample routed to Triton) — NOT a no-win. Reused the M-routed fp8_utils overlay (global, M≤64→CK)
> + `winner_kind=env` (`AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<combined tuned csv>` covering all 3 NK families).
> e2e gate pending (Integrator). Amdahl ceiling @22.9% GPU (live_pct 75.4%): decode-weighted ~5× → large.
> ✅ **22nd confirm (2026-07-25, Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, conc=64, isl=osl=1024,
> down_proj head 37.07% GPU — RE-CONFIRM of the 19th, fresh eval).** Immutable-unittest serving-weighted
> A/B (CUDA-graph replay), CK target vs frozen Triton baseline: **decode M=1 2.60×, M=64 2.92× FASTER;
> prefill M=1024 0.70× SLOWER → GEAK_WEIGHTED_SPEEDUP=2.23× (geomean 1.74)**, CORRECTNESS PASS (eager +
> 3-draw random-parity + graph-replay; relerr 2e-4..8e-3). aiter CK tuner `--libtype both --mp 1` (24s):
> all winners `libtype=ck`, errRatio 0.0; tune ENGAGES cleanly ("is tuned on cu_num=304" for M=1/64/1024
> via AITER_LOG_TUNED_CONFIG=1) but is ≈CK-default (no extra gain — CK default heuristic already optimal).
> §9.1 `transpose_scale=False` correct (bpreshuffle drop-in WRONG relerr 41.7). Shipped **M-routed overlay**
> (`input_2d.shape[0] ≤ SGLANG_CK_BLOCKSCALE_MAX_M`, default 64) to bank decode win + dodge the prefill
> regression, + `winner_kind=env` (`AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<tuned csv>`). op_bench single-bucket
> M=1024 probe = 1.0×/0.70× (prefill-only sample) — NOT a no-win; the decode-weighted metric is the truth.
> e2e gate pending (Integrator). Amdahl ceiling @37.07% GPU, 2.23× iso ≈ +25.7% e2e (live_pct 75.4% → more).
> ✅ **21st confirm (2026-07-25, Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, conc=64, isl=osl=1024,
> o_proj head 11.31% GPU — square N=K=5120, sibling of the down_proj/gate_up_proj confirms).** CK skill on
> o_proj (N=5120,K=5120 × M={1..1024}), aiter CK tuner `--libtype both --mp 1`: all winners `libtype=ck`,
> errRatio 0.0, engagement verified ("is tuned on cu_num=304" for every M via `AITER_LOG_TUNED_CONFIG=1`).
> ckProfiler ABSENT (env_report) but the aiter CK tuner (`csrc/ck_gemm_a8w8_blockscale/`) does NOT need it —
> CK env+overlay lever fully feasible; ckProfiler only gates the CK *author* lane. §9.1: CK correct at
> `transpose_scale=False` (relerr 3e-3..7e-3 at M≥2; bpreshuffle drop-in WRONG relerr 41.5). Per-bucket cold
> CUDA-event A/B (tuned CSV engaged): **decode M=1..64 3.6–3.9× FASTER; M=128 2.66×, M=256 2.01×; prefill
> M=1024 1.04× (≈tie, routed to Triton)** → REGIME-SPLIT reconfirmed; square o_proj is a STRONG-decode family
> (~3.9× at M64, vs gate_up_proj's weak 1.38×). Tuned CSV ≈ CK default (default already optimal). Shipped
> M-ROUTED overlay (`input_2d.shape[0] ≤ SGLANG_CK_BLOCKSCALE_MAX_M`, default 64; patch applies clean to live
> sglang tree) + `winner_kind=env` (`AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<tuned combined csv>`). op_bench
> single-bucket M=1024 probe = 1.0× (prefill-only sample) — NOT a no-win. e2e gate pending (Integrator).
> ✅ **20th confirm (2026-07-25, Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, conc=64, isl=osl=1024,
> gate_up_proj head 15.22% GPU — sibling of the down_proj 19th confirm).** CK skill on gate_up_proj
> (N=34816,K=5120 × M={1,16,32,48,64}), aiter CK tuner `--libtype both --mp 1`: all winners `libtype=ck`,
> errRatio 0.0, engagement verified ("is tuned on cu_num=304" for M=1 & M=64; M=1024 → "use default" =
> routes to Triton). §9.1: CK correct at `transpose_scale=False` (relerr ~2e-5 at M≥2; transposed → 0.19
> garbage; bpreshuffle drop-in WRONG relerr 46.5). Per-bucket cold CUDA-event A/B: **decode M=1 3.41×,
> M=64 1.38× FASTER; prefill M=1024 0.60× (SLOWER)** → REGIME-SPLIT reconfirmed (gate_up_proj is the
> weakest-decode family, matching meta's ~1.37×/M64 prior). Tuned CSV ≈ CK default (default already
> optimal for these shapes). Shipped M-ROUTED overlay (`input_2d.shape[0] ≤ _CK_M_MAX`, default 64) +
> `winner_kind=env` (`AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<tuned.csv>`) + `code_patch=<fp8_utils M-routed CK
> overlay>`. op_bench single-bucket M=1024 probe = 1.0× (prefill-only sample) — NOT a no-win. e2e gate pending.
> ✅ **19th confirm (2026-07-25, Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, conc=64, isl=osl=1024,
> down_proj head 36.7% GPU).** CK skill on down_proj (N=5120,K=17408 × M={1,64,1024}), aiter CK tuner
> `--libtype both --mp 1`: all winners `libtype=ck`, errRatio 0.0, engagement verified ("is tuned on
> cu_num=304"). ckProfiler ABSENT but the aiter CK tuner (`csrc/ck_gemm_a8w8_blockscale/`) does NOT need
> it — the CK env+overlay lever is fully feasible (ckProfiler only gates the CK *author* lane). §9.1: CK
> correct at `transpose_scale=False` (relerr 4e-4..6e-3 at M≥2; bpreshuffle drop-in WRONG relerr 41.7).
> Per-bucket production A/B (op_bench cold-cache CUDA events): **decode M=1 3.19×, M=64 3.49× FASTER;
> prefill M=1024 0.82× (SLOWER)** → REGIME-SPLIT reconfirmed. Tuned CSV ≈ CK default here (default already
> optimal). Shipped M-ROUTED overlay (`input_2d.shape[0] ≤ SGLANG_CK_BLOCKSCALE_MAX_M`, default 64) +
> `winner_kind=env` (`AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<tuned.csv>`). op_bench single-bucket M=1024 probe
> shows 1.0× (prefill-only sample) — NOT a no-win. e2e gate pending (Integrator).
> ✅ **18th confirm (2026-07-24, Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, conc=64, isl=osl=1024).**
> CK skill end to end on the 12 live (M,N,K) (4 NK families × {1,64,1024}), `--libtype ck --mp 1` (single
> GPU on box): ALL 12 winners `libtype=ck`, errRatio 0.0, engagement verified ("is tuned on cu_num=304" ×12).
> §9.1 layout reconfirmed: CK wants `transpose_scale=False` (CK-vs-Triton rel-err ~1e-5 at M≥2). Production-op
> A/B (deployed CSV, CUDA-event device time): **decode M=1 3.3–5.2×, M=64 1.42–5.3× FASTER; prefill M=1024
> 0.60–0.73× (SLOWER on every family)** → REGIME-SPLIT reconfirmed. Share-weighted decode speedup ~3.9× (M=1) /
> ~3.3× (M=64). Shipped M-ROUTED overlay (this box: `input_2d.shape[0] ≤ AITER_CK_BLOCKSCALE_MAX_M`, default
> **64** = the decode ceiling at conc=64; larger M→stock Triton) + `winner_kind=env`
> (`AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<tuned.csv>`). op_bench's single-bucket M=1024 probe reports 1.0× (it
> only sees the prefill regime) — do NOT read that as "no win"; the win is the decode regime it doesn't sample.
> e2e gate pending (Integrator; verify HIP-graph decode capture keeps the small-M CK path live).
> ✅ **17th confirm (2026-07-06, Qwen3-14B-FP8 TP=1, gfx942/MI300X cu_num=304, e2e_cycle0).**
> CK skill end to end on the 12 live (M,N,K) (4 NK families × {1,64,16384}), `--libtype both --mp 1`:
> ALL 12 winners `libtype=ck`, errRatio 0.0. §9.1 scale-layout check (M=64, all 4 families): CK wants
> `transpose_scale=False` (rel-err ~1e-5; True→~0.19 = the catastrophic layout bug). In-process CK-vs-untuned-Triton
> A/B (correct non-transposed scale): **decode M=1 4.7–6.2×, M=64 1.37–6.0× FASTER** (up_gate weakest at
> 1.37×/M64; qkv/o ~6×); **prefill M=16384 0.56× (SLOWER on every family)** → M-routed overlay (M≤256→CK)
> is mandatory, reconfirmed. Share-weighted decode speedup ~1.87× (M64) / ~4.1× (M1). CK also FIXES the
> known untuned-Triton small-M/large-K down-proj defect (CK rel_err ~true-math; Triton wrong at M=1,K=17408).
> Engagement verified ("is tuned on cu_num = 304"). Shipped `winner_kind=env` (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE)
> + M-routed fp8_utils CK overlay. e2e gate pending (Integrator).
> ✅ **16th confirm (2026-06-25, Qwen3-14B-FP8 TP=2, gfx942/MI300X cu_num=304, e2e_qwen3_14b_fp8_..._3515_8954).**
> Re-ran the CK skill end to end on the exact 16 live (M,N,K) (4 NK families × {1,16,2048,13645}), `--libtype
> both --mp 2`: ALL 16 winners `libtype=ck`, errRatio 0.0. Dominant head M=13645,N=5120,K=8704 CK
> kernelId=0 = 1.469 ms (matches the 1.456 ms / ~1.78× kernel prior). Engagement reconfirmed
> ("is tuned on cu_num = 304"); faithful block-scale dequant correctness rel_err 0.0017 (tol 0.06). Op_bench
> bake-off reconfirmed bpreshuffle drop-in is WRONG (rel_err 42.4). Shipped the M-ROUTED overlay (M≤256→CK,
> else Triton) per the banner below. NOTE: one stale JIT baton lock from a prematurely-killed tuner run
> stalled the build for ~25 min with 0 compiler procs — clearing `aiter/jit/build/lock_module_*` +
> `build/module_*_tune` and restarting (per skill §2/pitfall) fixed it; watch for this.
> 🔑 **REGIME SPLIT is the deployable refinement (2026-06-25, Qwen3-14B-FP8 TP=2, gfx942, conc=16).**
> Re-ran the CK skill end to end (tuned 16 live (M,N,K) = 4 NK families × {1,16,2048,13645}, `--libtype both
> --mp 2`, all winners `libtype=ck` errRatio 0.0, engagement confirmed via "is tuned on cu_num=304"). The
> PRODUCTION custom-op `aiter.gemm_a8w8_blockscale` (what fp8_utils calls post-switch), measured eager with
> CUDA events on this box, is **regime-split, not uniformly faster**:
>   · decode/skinny-M (M≤256): CK ~**3.4–4.0× FASTER** than the untuned Triton block-scale default
>     (Triton is ~0.11–0.14 ms flat for any small M; CK ~0.03 ms). Crossover at M≈256↔512.
>   · prefill/large-M (M≥512): CK ~**0.5–0.66× (1.5–2× SLOWER)** than Triton on EVERY family (down/gate_up/
>     qkv/o), measured both device-only (CUDA events) and wall, plain & transposed x_scale layouts identical.
>     This contradicts the tuner's own ~1.46–1.49 ms "kernel" `us` column — the production
>     `gemm_a8w8_blockscale_ck` lookup path runs ~4.85 ms eager on M=13645,N=5120,K=8704. Trust the
>     production-op A/B, not the tuner CSV `us`, for the prefill verdict.
> ✅ **Deployable winner = M-ROUTED overlay (NOT a wholesale Triton→CK import swap).** A bare import swap
> routes prefill to CK too → regresses the GPU-time-heavy prefill. Instead the fp8_utils overlay rebinds
> `triton_gemm_a8w8_blockscale` to a dispatcher: `M≤256 → CK (tuned), else → stock Triton`
> (env `SGLANG_CK_BLOCKSCALE_M_MAX`, default 256). Verified correct (err~0.003, tol 0.06): decode hits CK
> 0.0355 ms while prefill passes through to Triton at parity. Captures the e2e-critical decode/TPOT win
> (steady-state M≈conc=16) with ZERO prefill regression. `winner_kind=env` (apply_env
> `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<tuned.csv>`) + `code_patch=<fp8_utils M-routed CK overlay>` +
> `tuning_artifact=<tuned.csv>`. Still gate at e2e (HIP-graph capture must keep the small-M CK path live).
---
> ✅ **CK-skill result (this eval, Qwen3-14B-FP8 TP=2, gfx942/MI300X):** ran the MANDATED CK playbook end
> to end. Tuned all 20 live (M,N,K) (4 NK families × {1,16,2048,13645,16385}) with `gemm_a8w8_blockscale_tune.py
> --libtype both --mp 2`; ALL winners are `libtype=ck`, errRatio 0.0. Dominant head (M=13645,N=8704,K=5120)
> CK kernelId=0 = **1.456 ms vs untuned Triton 2.60 ms = 1.78× at the kernel level** (tuner & raw `gemm_a8w8_blockscale_tune`
> agree exactly). Engagement verified: `AITER_LOG_TUNED_CONFIG=1` prints "is tuned on cu_num=304".
> ⚠️ **CONDITIONED caution for the Integrator e2e gate:** the *production* custom-op `aiter.gemm_a8w8_blockscale`
> (what fp8_utils calls after the Triton→CK switch) measured **4.85 ms eager** on the same shape — ~3.4 ms
> per-call Python/dispatch wrapper overhead ON TOP of the 1.46 ms kernel (steady-state over 500 iters, NOT a
> one-time cost). So in EAGER it is ~1.9× SLOWER than Triton despite the faster kernel. This overhead is the
> kind HIP-graph / torch.compile capture amortizes (the live serving path captures decode & wraps prefill),
> which is exactly why the e2e gate — not an isolated eager bench — is the judge. The win is REAL at the
> kernel; verify the wrapper overhead is absorbed under graph capture at e2e before crediting it.
---
> ⛔ **DEPRECATED for the current eval.** The Triton config-JSON overlay below keeps the slow Triton
> blockscale seam live and **bypasses the user's CK skill**. For an sglang fp8 a8w8 blockscale head the
> MANDATED lever is the CK playbook `gemm_tuning/fp8_gemm_tuning_sglang_aiter.md`
> (capture live (M,N,K) → aiter CK tuner `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py` →
> fp8_utils Triton→CK switch overlay + `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<tuned.csv>`). The correct
> baseline for the speedup is the **UNTUNED Triton default** the stack runs by default — NOT a CK-default
> heuristic. The "CK env yields ~1.00×" caution below was measured CK-default-vs-CK-tuned on a vLLM CK
> live path; it does NOT describe CK-tuned-vs-Triton-untuned on this sglang rig. Do not let this card
> steer the bake-off to the Triton overlay / flydsl / aiter-bf16 levers — follow the CK skill.

# fp8 a8w8 blockscale GEMM → per-(N,K) M-bucketed Triton config-JSON overlay  (DEPRECATED — see banner)
- lever: the live seam is `aiter.ops.triton.gemm_a8w8_blockscale`. Stock has NO per-(N,K) config for
  the model's large shapes → it uses a generic tile = headroom. Win = drop a tuned
  `gfx942-GEMM-A8W8_BLOCKSCALE-N=<N>-K=<K>.json` into `aiter/ops/triton/configs/gemm/`
  (winner_kind=**patch**; `AITER_TRITON_CONFIGS_PATH` is `__file__`-fixed, not env-overridable).
- apply: **M-bucket the config** — tile shape depends on (N,K):
  · wide-N up/gate (N=34816,K=5120): prefill BM=256/BN=128/GROUP_M=4/nw=8.
  · K-heavy/narrow-N down (N=5120,K=17408): prefill **BM=128/BN=256**/GM=4/nw=8 (widen BN, keep BM).
  · square-ish qkv/o (N=5120,K=6144): prefill BM=256/BN=128/GM=4/nw=8 (small clean win).
  · decode (M≤1024) MUST stay generic BM=128. Integrator must rebind BOTH `sglang...fp8_utils` globals
    `triton_gemm_a8w8_blockscale` + `gemm_a8w8_blockscale_bpreshuffle`.
- verify: honest in-process `config=` kwarg A/B, same synth fp8 operands held fixed, interleaved
  min-of-N; confirm engagement via live `_get_config(M,N,K)` (returns a (dict,use_persistent) tuple → [0]).
- caution: a FLAT overlay (BM=256 for all M) tanks decode 0.6–0.7× — decode MUST stay generic.
- caution: BN=256 + BM=256 together = LDS spill (0.29×) — widen only one dim.
- caution: on the **vLLM CK live path** (not Triton) this overlay does NOT apply — live is CK
  xdl-cshuffle; the lever there is env `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<csv>`, but it yields ~1.00×
  (CK default heuristic already picks the optimal `256x128x128 intrawave_v3`). ALWAYS check which live
  path (CK vs Triton) is engaged BEFORE choosing the lever.
- caution: **backend availability is a PROVISIONING gate, not a no-win.** The mandated CK lever needs the
  aiter CK tuner (`csrc/ck_gemm_a8w8_blockscale/`); the FlyDSL alternative needs aiter's `aiter/ops/flydsl/`
  wrapper AND the top-level `flydsl` pip pkg. If `env_report.absent_backends` lists either, record the
  two-part remedy (flydsl: `pip install 'flydsl>=0.1.5'` AND a flydsl-enabled `amd_aiter` build that ships
  `aiter/ops/flydsl/` — pip flydsl ALONE is insufficient; `aiter.ops.flydsl` stays ModuleNotFoundError) and
  fall back to an available lever — never silently drop the head. See `gemm_tuning/fp8_gemm_tuning_sglang_aiter.md`.
- caution: the aiter `gemm_a8w8_blockscale_bpreshuffle` path benches ~1.5× faster than the plain
  Triton blockscale kernel BUT is WRONG as a naive drop-in (op_bench Qwen3-14B-FP8: rel_err 43.6 vs the
  blockscale baseline's 0.0075) — it needs weights preshuffled first. It is the large-M prefill lever,
  not a free swap; only use via the preshuffle seam (`aiter:gemm_a8w8_blockscale_bpreshuffle` + a once
  `shuffle_weight`), never by rebinding the live blockscale call to it directly.
- source: exp/e2e_*Qwen3.5-27B-FP8*/ runs 06-08 … 06-15 (11 re-confirms); + exp/e2e_qwen3_14b_fp8_20260624
  (13th: Qwen3-14B-FP8 TP=2, 4 per-GPU families. In-process config= A/B, fixed synth fp8 operands, min-of-50.
  Prefill M={13645,16385}: up/gate N=17408,K=5120 BM256/BN128/GM4/nw8 = 1.16×; qkv N=3584,K=5120 same = 1.15×;
  down N=5120,K=8704 BM256/BN128/GM1/nw8 = 1.09×; o N=5120,K=2560 BM256/BN128/GM4/nw8 = 1.09×; prefill geomean 1.11×.
  All correct (fp8 tol 0.06). BM128_BN256 LOST on every family here (default-class), reconfirming "widen BM not BN"
  for these K=5120-ish shapes. Decode kept generic via M_LEQ_1024 key. Overlay engagement re-verified live via
  get_gemm_config (BM 128→256 on prefill). aiter bpreshuffle benched 4.63 vs 6.48 ms (1.4×) but rel_err 41.8 = NOT a drop-in.)
  (12th: re-confirmed live seam + "not found tuned config, will use default config" headroom on up/gate
  N=17408,K=5120; aiter bf16 DB tune is the WRONG lever here — fp8 path is the Triton seam, not aiter.tuned_gemm)
