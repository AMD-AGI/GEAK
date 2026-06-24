---
title: GEMM tuning workflow — the aiter per-shape DB recipe (capture → tune → deploy → gate)
kind: workflow
operator: dense_gemm
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - GEAK/e2e_workflow/knowledge/gemm_tuning/aiter_gemm_tuning.md
  - GEAK/perf_knowledge/backends/aiter/tuned_gemm.md
  - ROCm/aiter@a6bb499:aiter/tuned_gemm.py
  - ROCm/aiter@a6bb499:gradlib/gradlib/gemm_tuner.py
---

# GEMM tuning workflow (aiter per-shape DB)

## TL;DR
On sglang/vLLM ROCm the dense bf16/fp16 GEMM mass (~78% of GPU time on Qwen3.5-27B) is
dispatched by **`aiter.tuned_gemm.gemm_a16w16`**, which picks the fastest of
hipBLASLt/asm/triton/skinny/flydsl per shape from **aiter's own per-shape DB**. The only
GEMM lever that engages this live path is **tuning that DB**: live `AITER_TUNE_GEMM=1`
capture → `gradlib gemm_tuner.py` → deploy via `AITER_CONFIG_GEMM_BF16` → verify with
`AITER_LOG_TUNED_CONFIG=1`. A bias-correct, full-coverage tune banked **+2.23% e2e on
Qwen3.5-27B** (stacked on `--attention-backend triton`). See
[`../backends/aiter/tuned_gemm.md`](../backends/aiter/tuned_gemm.md) and
[`../operators/dense_gemm/tuning.md`](../operators/dense_gemm/tuning.md).

## The two traps that make a tune worthless (read first)
1. **TunableOp / `HIPBLASLT_TUNING_FILE` bypass.** PyTorch TunableOp hooks
   `torch.addmm/matmul/F.linear`; `HIPBLASLT_TUNING_FILE` sits under the hipBLASLt C path.
   The live sglang path calls **aiter directly**, bypassing both → **zero engagement**
   (the prior attempt measured −0.11%/−0.30%). **WRONG LEVER.** The aiter DB is the only
   right one.
2. **Bias mismatch.** The DB lookup key is the 9-tuple
   `(cu_num, padded_M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle)`. sglang issues
   most dense GEMMs with **`bias=False`** (bias added separately). If you synthesize the
   untuned set from the profile and guess `bias=True`, **every** tuned row mismatches the
   live `bias=False` call → 0 engagement, a tune that looks deployed but does nothing.
   **Always get the untuned set from the live capture**, never from a guessed schema.

## Preconditions
- aiter on the live path (sglang ROCm default; vLLM needs `VLLM_ROCM_USE_AITER=1`).
- A warm server you can bench at the target `ISL/OSL/conc`, TP=1.
- The current accepted config is fixed (gate the GEMM tune **stacked** on it, never alone).

## Step 0 — Back up the package CSVs you touch
```bash
CFG=$(python -c "import aiter,os;print(os.path.dirname(aiter.__file__)+'/configs')")
cp -a $CFG/bf16_untuned_gemm.csv /tmp/bf16_untuned.orig   # capture writes here (hardcoded path)
cp -a $CFG/bf16_tuned_gemm.csv   /tmp/bf16_tuned.orig     # we deploy via env, not by editing this
```

## Step 1 — Capture REAL shapes (AITER_TUNE_GEMM=1)
`AITER_TUNE_GEMM=1` makes `gemm_a16w16::save_shapes()` append every live GEMM shape (with
the **true `bias`** and the full family set) to `$CFG/bf16_untuned_gemm.csv`. Reset first,
run ONE warm bench at the target workload, snapshot, restore the package file:
```bash
printf 'M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle\n' > $CFG/bf16_untuned_gemm.csv
# run one warm bench at target ISL/OSL/conc with EXTRA_ENV="AITER_TUNE_GEMM=1"
cp $CFG/bf16_untuned_gemm.csv $EVAL/captured_untuned_gemm.csv
cp -a /tmp/bf16_untuned.orig  $CFG/bf16_untuned_gemm.csv
```
This captures all GEMM families: up/gate/gate_up, down-proj (K=intermediate), qkv, lm_head,
and the decode M-buckets. **Coverage matters** — uncovered shapes fall back to default and
never count.

## Step 2 — Bucket-reduce (optional, bounds tuning time)
Racing ~1365 hipBLASLt + asm/triton/skinny solutions on a big prefill shape is
minutes/shape. Collapse captured M to runtime buckets (`get_padded_m(M,N,K,0)` — runtime
pads M, so one representative M per bucket suffices) and keep the GPU-time-dominant
families (from the Profiler Top-N). On Qwen3.5-27B this took ~234 shapes → ~17 hot shapes
(the K=5120 up/gate trio) for a first partial tune.

## Step 3 — Tune with gradlib across ALL GPUs (err_ratio < 0.05)
```bash
cd /sgl-workspace/aiter/gradlib/gradlib
python gemm_tuner.py --input_file $EVAL/hot_untuned_gemm.csv \
  --tuned_file $EVAL/<model>_bf16_tuned_gemm.csv --indtype bf16 --mp <num_gpus>
```
- `--mp` defaults to `torch.cuda.device_count()`. **Do NOT set `HIP_VISIBLE_DEVICES=0`** —
  that cripples it to 1 GPU (~8× slower). Leave all GPUs visible.
- The tuner **races hipBLASLt/asm/triton/skinny/flydsl per shape** and gates each candidate
  on **`err_ratio < 0.05`** (numerically validated). One aiter tune therefore covers
  per-backend GEMM tuning — including FlyDSL (`libtype=flydsl`) where it wins. See
  [`../backends/aiter/flydsl_path.md`](../backends/aiter/flydsl_path.md).
- Writes **incrementally** (per shape) → stop early and deploy a partial DB; uncovered
  shapes fall back to default, so a partial tune **never regresses**.
- `--compare --update_improved --min_improvement_pct N` keeps a shape only if its tuned
  kernel beats default by ≥N% (a built-in isolated gate — drops no-op shapes).
- Output (18 cols): `gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,
  solidx,splitK,us,kernelName,err_ratio,tflops,bw`. First 10 = the lookup key.

## Step 4 — Deploy by ENV (reversible, no site-packages edit)
```bash
EXTRA_ENV="AITER_CONFIG_GEMM_BF16=$EVAL/<model>_bf16_tuned_gemm.csv AITER_LOG_TUNED_CONFIG=1"
```
`AITER_CONFIG_GEMM_BF16` takes one path or a `:`-separated merge list
(`<default>:<model_configs/*>:<mine>`). `AITER_LOG_TUNED_CONFIG=1` logs
`... is tuned on cu_num = ... libtype is ...` per hit — this is your **engagement proof**.

## Step 5 — Gate: same-session A/B + engagement + parity
- **Same-session A/B**, env unset (ref) vs env set (cand). gfx942 boxes drift several %
  across hours → only a same-session, drift-cancelled comparison is trustworthy. Accept on
  `delta% > 0.5% AND cand_min > ref_max` (see [`optimize_e2e_model.md`](optimize_e2e_model.md)).
- **Engagement:** `grep -c 'is tuned on cu_num' server.log` must be > 0 (the validated run
  saw 246), and "not found tuned config" must drop for tuned shapes. **No engagement →
  reject** (the TunableOp lesson).
- **Parity:** tuned entries passed the tuner's `err_ratio<0.05`, so they're numerically
  equivalent bf16 GEMM swaps; still run a greedy/temp=0 fixed-seed probe (≥10 prompts),
  especially for any shape that fell to a non-hipBLASLt `libtype` (triton/asm/flydsl).
- **Stack, don't isolate:** the GEMM tune's contribution depends on the surrounding timing
  regime (e.g. its effect on default attention differs from its effect on
  `--attention-backend triton`). Always gate it stacked on the current accepted config.

## The validated win (no invented numbers)
> **Qwen3.5-27B @ MI300X gfx942, sglang 0.5.11 / aiter, torch 2.9.1+rocm7.2,
> ISL/OSL=1024 conc=64 (2026-06-05):**
> - Partial tune (17 of ~234 shapes): **+1.22% e2e** (1462.8 → 1480.6 tok/s), 81
>   `is tuned on cu_num` hits, on default attention.
> - Bias-correct full-coverage tune stacked on `--attention-backend triton`: **+2.23% e2e**
>   (1548.9 → 1583.5 tok/s, non-overlapping 5-repeat A/B), **246** engagement hits.
> - Combined with the attention-backend flag: ~+6% over the stack-default baseline.
> The two things that made it work (and that the earlier ~0/−0.59% attempt got wrong):
> (1) tune input from the LIVE capture so `bias=False` + full shape set match the runtime
> keys; (2) gated STACKED on the current accepted config with the tight A/B.

`solidx`/`kernelName` are **ROCm/hipBLASLt/aiter-build-specific** — never ship a hand-copied
CSV; regenerate with gradlib on the target stack. The CSV is a per-run eval-dir artifact.

## Gotchas (all hit on the validated run)
- **Port flakiness:** sglang sets `grpc_port = port + 10000`; a random high port pushes grpc
  >65535 → `ValueError`. Pin a low `PORT` (e.g. 31237) for capture/tuning benches.
- **Don't pin the tuner to 1 GPU** (step 3).
- **Big prefill GEMMs are slow** to race (~1365 solutions) → bucket-reduce; partial is usable.
- **Version-locked:** re-tune on any ROCm/hipBLASLt/aiter upgrade (aiter validates `cu_num`).
- **`--model_dir` analytic path fails on wrapped models:** Qwen3.5 is
  `...ForConditionalGeneration` and nests hidden_size/intermediate_size → read as None. Use
  the live `AITER_TUNE_GEMM=1` capture (more accurate anyway).
- **Fork-storm caveat:** nesting head GEMM author + parallel tunes spawns hundreds of
  `rocm_agent_enumerator` processes → CPU thrash + corrupted timing. Serialize heavy tunes;
  never measure e2e A/B during a process storm.

## fp8 / CDNA4 note
This recipe is the bf16/fp16 path (`AITER_CONFIG_GEMM_BF16`). For fp8 (a8w8) GEMM reach
flydsl-fp8 via the DB tune (`libtype=flydsl`) and the author route; on gfx950 prefer
block-scaled MXFP8/MXFP6/MXFP4 (FP6 at FP4 rate). See [`../quantization/`](../quantization/),
[`../operators/scaled_quant_gemm/`](../operators/scaled_quant_gemm/), and
[`../hardware/cdna4_mi350/`](../hardware/cdna4_mi350/).

## Cross-links
- Dispatch internals: [`../backends/aiter/tuned_gemm.md`](../backends/aiter/tuned_gemm.md), [`../backends/aiter/configs_db.md`](../backends/aiter/configs_db.md)
- GEMM operator: [`../operators/dense_gemm/overview.md`](../operators/dense_gemm/overview.md), [`../operators/dense_gemm/tuning.md`](../operators/dense_gemm/tuning.md)
- FlyDSL path: [`../backends/aiter/flydsl_path.md`](../backends/aiter/flydsl_path.md), [`../languages/flydsl/`](../languages/flydsl/)
- e2e flow: [`optimize_e2e_model.md`](optimize_e2e_model.md) · Wire-in: [`integrating_a_new_kernel.md`](integrating_a_new_kernel.md)

## Sources
- Full recipe, both traps, the +1.22%/+2.23% Qwen3.5-27B wins, all gotchas: `GEAK/e2e_workflow/knowledge/gemm_tuning/aiter_gemm_tuning.md`.
- Dispatch + key construction: `GEAK/perf_knowledge/backends/aiter/tuned_gemm.md`; ROCm/aiter@a6bb499 `aiter/tuned_gemm.py`, `gradlib/gradlib/gemm_tuner.py`.
- aiter as default ROCm backend: https://github.com/ROCm/aiter
