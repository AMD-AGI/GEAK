# aiter GEMM tuning — the CORRECT GEMM lever for sglang on MI300X (gfx942)

> **Verified 2026-06-05 on Qwen-Qwen3.5-27B** (hybrid_linear_attention_dense, bf16, sglang 0.5.11,
> torch 2.9.1+rocm7.2, gfx942). A partial tune (17 of ~234 shapes, just the K=5120 up/gate GEMM
> trio) gave **+1.22% e2e** (1462.8 → 1480.6 tok/s, same-session, default attention, ISL/OSL=1024
> conc=64), with **81 live "is tuned on cu_num" hits** in the server log. This is **orthogonal** to
> the `--attention-backend triton` win (+4.96%) and is expected to stack.

## Why this file exists (the lesson that cost a whole run)
On sglang/gfx942 the dense-GEMM mass (~78% of GPU time) is dispatched by **aiter's
`tuned_gemm.py` → `gemm_a16w16`**, which executes hipBLASLt/Tensile `Cijk_*` kernels (or asm/triton/
skinny) chosen from aiter's OWN per-shape DB. Therefore:

- ❌ **PyTorch TunableOp** (`PYTORCH_TUNABLEOP_ENABLED=1`) hooks `torch.addmm`/`matmul`/`F.linear`
  dispatch. The live sglang path calls aiter directly, so TunableOp **never engages** (0 activity in
  the server log; the prior run measured −0.11%/−0.30%). WRONG LEVER.
- ❌ **`HIPBLASLT_TUNING_FILE` / `HIPBLASLT_TUNING_OVERRIDE_FILE`** also sit under the PyTorch/hipBLASLt
  C path that aiter bypasses for its tuned shapes. WRONG LEVER on this stack.
- ✅ **aiter's `bf16_tuned_gemm.csv`** IS the live path. The "`[aiter] ... not found tuned config ...
  will use default config`" log lines are aiter telling you exactly which shapes are UNtuned — that
  is the target list.

## How the live lookup works (so you know what to tune)
`aiter/tuned_gemm.py::get_GEMM_A16W16_config(M,N,K,bias,dtype,otype,scaleAB,bpreshuffle)`:
1. tries the **exact M**, then two padded buckets `get_padded_m(M,N,K,gl)` for `gl in {0,1}`.
2. key = `(cu_num, padded_M, N, K, bias, str(dtype), str(otype), scaleAB, bpreshuffle)`.
3. hit → use `libtype`(hipblaslt|asm|triton|skinny|torch) + `solidx` from the CSV. miss → default
   (`torch` for large dense, `skinny` for decode-skinny) + the "not found tuned config" log line.
Because runtime pads M to buckets, you only need to tune **one representative M per bucket** (see the
bucket trick below), not every M the server issues.

## The recipe (capture → tune → deploy → gate), no package edits

### 0. Back up the two package CSVs you touch
```bash
CFG=$(python -c "import aiter,os;print(os.path.dirname(aiter.__file__)+'/configs')")
cp -a $CFG/bf16_untuned_gemm.csv  /tmp/bf16_untuned.orig   # capture writes here (hardcoded path)
cp -a $CFG/bf16_tuned_gemm.csv    /tmp/bf16_tuned.orig     # we DON'T edit this (env override instead)
```

> ⚠️ **The lookup key includes the `bias` flag — capture it, don't guess it.** sglang issues most dense
> GEMMs with **`bias=False`** (bias added separately). If you synthesize the untuned set from the profile
> and assume `bias=True`, every tuned row mismatches the live `bias=False` calls and you get **0
> engagement** (a worthless tune that looks deployed). ALWAYS get the untuned set from the live
> `AITER_TUNE_GEMM=1` capture below (it records the true bias + the full shape set), and self-verify
> `is tuned on cu_num` hits >0 before trusting it.

### 1. Capture the REAL shapes from a warm server
`AITER_TUNE_GEMM=1` makes `gemm_a16w16::save_shapes()` append every live GEMM shape to
`$CFG/bf16_untuned_gemm.csv`. Reset it first so you capture only this workload:
```bash
printf 'M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle\n' > $CFG/bf16_untuned_gemm.csv
# run ONE warm bench at the target ISL/OSL/conc with EXTRA_ENV="AITER_TUNE_GEMM=1" via bench_e2e.sh
cp $CFG/bf16_untuned_gemm.csv  $EVAL/captured_untuned_gemm.csv     # snapshot
cp -a /tmp/bf16_untuned.orig   $CFG/bf16_untuned_gemm.csv          # restore package
```

### 2. (optional but recommended) bucket-reduce to cut tuning time
Tuning a large prefill shape races ~1365 hipBLASLt + asm/triton/skinny solutions → minutes/shape.
Collapse captured M to runtime buckets (`get_padded_m(M,N,K,0)`) and keep the GPU-time-dominant
families (from the Profiler Top-N: e.g. K=5120 → N∈{up,gate,gate_up}, and down/qkv N=hidden). This
took ~234 shapes → ~17 hot shapes for the dominant up/gate trio. Tune more later for full coverage.

### 3. Tune with gradlib across ALL GPUs
```bash
cd /sgl-workspace/aiter/gradlib/gradlib
python gemm_tuner.py --input_file $EVAL/hot_untuned_gemm.csv \
  --tuned_file $EVAL/<model>_bf16_tuned_gemm.csv --indtype bf16 --mp <num_gpus>
```
- `--mp` defaults to `torch.cuda.device_count()`. **Do NOT set `HIP_VISIBLE_DEVICES=0`** — that
  cripples it to 1 GPU (~8× slower). Leave all GPUs visible.
- Writes the tuned CSV **incrementally** (per shape) — you can stop early and deploy a partial DB;
  uncovered shapes simply fall back to default (same as baseline), so a partial tune never regresses.
- Output columns: `gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw`. The tuner gates each solution on `err_ratio < 0.05` → numerically validated.
- `--compare --update_improved --min_improvement_pct N` exists: keep a shape only if its tuned kernel
  beats the default by ≥N% (a built-in isolated gate; useful to drop no-op shapes).

### 4. Deploy by ENV (reversible, no site-packages edit)
```bash
EXTRA_ENV="AITER_CONFIG_GEMM_BF16=$EVAL/<model>_bf16_tuned_gemm.csv AITER_LOG_TUNED_CONFIG=1"
```
`AITER_CONFIG_GEMM_BF16` accepts a single path or a `:`-separated merge list (e.g.
`<default>:<model_configs/*>:<mine>`). A single path is used as-is. `AITER_LOG_TUNED_CONFIG=1` makes
the server log "`... is tuned on cu_num = ... libtype is ...`" per hit — **use it to PROVE engagement**
(this is the check the TunableOp attempt failed). This env composes into the cuda-graph at startup, so
it is captured and deployable.

### 5. Gate: same-session A/B + engagement + parity
- Bench default (env unset) vs tuned (env set) **in the same session** (gfx942 boxes drift several %
  across hours — only same-session A/B is trustworthy).
- Confirm engagement: `grep -c 'is tuned on cu_num' server.log` should be >0 (we saw 81), and the
  "not found tuned config" count should drop for tuned shapes.
- Parity: tuned entries are hipBLASLt/asm GEMM algorithm swaps that passed the tuner's `err_ratio<0.05`
  numerical check → numerically equivalent bf16 GEMM (same class as an attention-backend swap). A
  greedy temp=0 fixed-seed probe is expected to pass; do it for any shape that fell to a non-hipBLASLt
  libtype (triton/asm) to be safe.

## Gotchas (all hit during the 2026-06-05 run)
- **Port flakiness:** sglang sets `grpc_port = port + 10000`; bench_e2e auto-allocates a random free
  port that often pushes grpc >65535 → `ValueError: SGLANG_GRPC_PORT must be between 1 and 65535`. Pin
  a low `PORT` (e.g. 31237) for tuning/capture benches.
- **Don't pin to 1 GPU** for the tuner (see step 3).
- **Slow for big prefill GEMMs:** racing ~1365 solutions on a 16384×34816×5120 GEMM is minutes each;
  bucket-reduce and tune the dominant families first; it writes incrementally so partial is usable.
- **Version-locked:** aiter solution indices are tied to the ROCm/hipBLASLt/aiter build. Re-tune on any
  upgrade (the tuned CSV records `gfx`/`cu_num`; aiter validates `cu_num` on lookup).
- **Config-generation analytic path won't work for wrapped models:** `gemm_tuner.py --model_dir` reads
  flat `config.json` fields (hidden_size, intermediate_size). Qwen3.5 is `...ForConditionalGeneration`
  and nests them → fields read as None. Use the `AITER_TUNE_GEMM=1` capture path instead (more accurate
  anyway — it records the exact shapes the server issues).

## Confirmed result (Qwen3.5-27B / gfx942 / sglang, ISL/OSL=1024 conc=64)
A bias-correct, full-coverage aiter GEMM tune (live `AITER_TUNE_GEMM=1` capture → gradlib →
`AITER_CONFIG_GEMM_BF16`) **wins +2.23% e2e** stacked on `--attention-backend triton`
(ref 1548.9 → cand 1583.5 tok/s, non-overlapping 5-repeat A/B, **246 `is tuned on cu_num` hits**).
Combined with the attention-backend flag that's ~+6% over the stack-default baseline. The two things
that make it work (and that an earlier ~0/−0.59% attempt got wrong): (1) the tune input comes from the
LIVE capture so `bias=False` and the full shape set match the runtime keys (engagement >0), and (2) it's
gated STACKED on the current accepted config with the tight A/B. So: **GEMM tuning IS a real lever here**
— do it (this supersedes any earlier "GEMM tune nets ~0" note).

## ⚠️ Cost / fork-storm caveat (finding #8)
The head GEMM optimization is the dominant runtime cost, and nesting it (head Triton author + milestone
parallel recursions, each spawning ROCm/aiter init) can fork-storm the host with hundreds of
`rocm_agent_enumerator` processes → CPU thrash → near-stall AND corrupted e2e timing. For bounded/
validation runs: cap the tune shape count (bucket-reduce hard), keep `head_author_max` low, lower
`kernel_budget`, and avoid running many heavy nested tunes concurrently (serialize them). Never run an
e2e A/B measurement while a process storm is active — pin it to a quiet window.

## Measurement methodology (so the result is trustworthy)
- **Always gate a GEMM tune STACKED on the current accepted config, never in isolation.** The GEMM
  tune's contribution depends on the surrounding timing regime — e.g. its standalone effect on default
  (aiter) attention differs from its effect stacked on `--attention-backend triton`. Measure it on top
  of whatever is already accepted.
- **Verify engagement** (`AITER_LOG_TUNED_CONFIG=1` → `is tuned on cu_num` hits >0) so you know the
  tuned solutions are actually executing on the live path before trusting any throughput delta.
- **Use the tight interleaved A/B** (E2E_REPEATS per leg, ref vs cand alternating, accept on
  `delta > 0.5% AND cand_min > ref_max`) — gfx942 boxes drift several % across hours, so only a
  same-session, drift-cancelled, non-overlapping comparison is decisive at the 0.5% band.
- **Coverage matters**: capture the full real shape set (down-proj K=intermediate, qkv K, lm_head, and
  the decode M-buckets), not just the up/gate trio — uncovered shapes fall back to default and never
  count. Bucket-reduce via `get_padded_m` to bound tuning time.

## Tuned-CSV format (what gradlib writes / what aiter reads)
Header (18 cols):
`gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw`
- The first 10 cols are the **lookup key** (must match `get_GEMM_A16W16_config`'s key exactly).
- `libtype` ∈ `hipblaslt|asm|triton|skinny|torch`; `solidx` = the chosen solution index (for
  `hipblaslt`/`asm`; 0 for `triton`/`torch`). `us`/`tflops`/`bw`/`err_ratio` are measured stats.
- Example rows observed on Qwen3.5-27B / gfx942 (up/gate, K=5120), illustrating that the winner
  varies by M — exact numbers are build-specific, don't copy them, re-tune per stack:
  - decode `M=64,  N=14336` → `libtype=hipblaslt, solidx≈196827`
  - prefill `M=8192, N=14336` → `libtype=triton, solidx=0`
  - prefill `M=15360,N=14336` → `libtype=hipblaslt, solidx≈204387, ~641 TFLOPS`
  - `M=1024, N=14336` → `libtype=torch` (i.e. tuning found nothing better than default for that bucket)
- ⚠️ `solidx`/`kernelName` are **ROCm/hipBLASLt/aiter-build-specific**. Never ship a hand-copied CSV;
  always regenerate with gradlib on the target stack. The CSV is a per-run artifact that belongs in
  the eval dir (e.g. `$EVAL_DIR/config/<model>_bf16_tuned_gemm.csv`), not in this repo.

## At runtime: keep temp scripts in the eval dir, not in `scripts/`
The shared `scripts/` dir holds only generic infra (bench_e2e.sh, parse_profile.py, …). The capture/
tune driver for THIS recipe is experiment-specific (model, workload, shape subset), so the Config Tuner
should **write its own small driver into `$EVAL_DIR/config/` at runtime** (or run the steps inline) by
following the commands above. Discover the gradlib tuner path generically rather than hardcoding it:
```bash
GRADLIB=$(python3 -c "import aiter,os;print(os.path.dirname(os.path.dirname(aiter.__file__)))")/gradlib/gradlib/gemm_tuner.py
[ -f "$GRADLIB" ] || GRADLIB=$(find / -name gemm_tuner.py -path '*gradlib*' 2>/dev/null | head -1)
```

### Example driver (write to `$EVAL_DIR/config/tune_gemm.sh`, then run it)
A complete, parameterized, fault-tolerant example. It is **not shipped in `scripts/`** — the agent
writes it (adapting as needed) into the eval dir at runtime. All inputs are env vars (no hardcode); it
backs up + restores the package CSV, pins a low port, discovers gradlib, tunes across all GPUs, and
prints the deploy env to use next.
```bash
#!/usr/bin/env bash
# $EVAL_DIR/config/tune_gemm.sh — capture → tune aiter's dense-GEMM DB. Generic; env-driven.
set -uo pipefail
: "${MODEL:?set MODEL}"; : "${OUT_DIR:?set OUT_DIR (e.g. \$EVAL_DIR/config)}"
: "${WF_SCRIPTS:?set WF_SCRIPTS (dir with bench_e2e.sh)}"
GPU=${GPU:-0}; ISL=${ISL:-1024}; OSL=${OSL:-1024}; CONC=${CONC:-64}
PORT=${PORT:-31237}                                  # grpc_port=port+10000 must stay <65535
NGPUS=${NGPUS:-$(python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 1)}
TAG=$(basename "${MODEL%/}"); mkdir -p "$OUT_DIR"

CFG=$(python3 -c "import aiter,os;print(os.path.dirname(aiter.__file__)+'/configs')") || { echo "aiter import failed"; exit 2; }
UNTUNED="$CFG/bf16_untuned_gemm.csv"
GRADLIB=$(python3 -c "import aiter,os;print(os.path.dirname(os.path.dirname(aiter.__file__)))")/gradlib/gradlib/gemm_tuner.py
[ -f "$GRADLIB" ] || GRADLIB=$(find / -name gemm_tuner.py -path '*gradlib*' 2>/dev/null | head -1)
[ -f "$GRADLIB" ] || { echo "gemm_tuner.py not found"; exit 2; }

cp -a "$UNTUNED" "$OUT_DIR/bf16_untuned.orig"                          # backup (restored below)
trap 'cp -a "$OUT_DIR/bf16_untuned.orig" "$UNTUNED" 2>/dev/null' EXIT  # always restore the package CSV

# [1] capture real shapes
printf 'M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle\n' > "$UNTUNED"
env BACKEND=sglang MODEL="$MODEL" GPU="$GPU" PORT="$PORT" ISL="$ISL" OSL="$OSL" CONC="$CONC" \
    REPEATS=1 PROFILE=0 OUT_DIR="$OUT_DIR/capture" \
    EXTRA_ENV="AITER_TUNE_GEMM=1 SGLANG_GRPC_PORT=$((PORT+10000))" \
    bash "$WF_SCRIPTS/bench_e2e.sh" > "$OUT_DIR/capture.log" 2>&1 || { echo "capture bench failed; see capture.log"; exit 3; }
cp "$UNTUNED" "$OUT_DIR/captured_untuned_gemm.csv"
echo "captured $(($(wc -l < "$OUT_DIR/captured_untuned_gemm.csv")-1)) shapes"

# [2] (optional) bucket-reduce to the profiler's dominant (N,K) families to cut tuning time; here use all.
INPUT="${INPUT_CSV:-$OUT_DIR/captured_untuned_gemm.csv}"

# [3] tune across all GPUs (writes incrementally; safe to stop early -> partial DB still valid)
TUNED="$OUT_DIR/${TAG}_bf16_tuned_gemm.csv"; rm -f "$TUNED"
( cd "$(dirname "$GRADLIB")" && python "$GRADLIB" --input_file "$INPUT" --tuned_file "$TUNED" \
    --indtype bf16 --mp "$NGPUS" ) > "$OUT_DIR/tune.log" 2>&1 || echo "tuner exited nonzero (partial DB may still be usable); see tune.log"
echo "tuned $(($(wc -l < "$TUNED" 2>/dev/null)-1)) shapes -> $TUNED"

echo "DEPLOY:  EXTRA_ENV=\"AITER_CONFIG_GEMM_BF16=$TUNED AITER_LOG_TUNED_CONFIG=1\""
echo "VERIFY:  grep -c 'is tuned on cu_num' <tuned-run server.log>  (must be >0)"
echo "GATE:    same-session A/B (env unset vs set); keep only if > noise band; parity-probe non-hipblaslt picks"
```
Invoke it (example):
```bash
MODEL=/path/to/model OUT_DIR="$EVAL_DIR/config" \
  WF_SCRIPTS="$SKILL_DIR/scripts" GPU=0 ISL=1024 OSL=1024 CONC=64 \
  bash "$EVAL_DIR/config/tune_gemm.sh"
```
Then deploy the printed `AITER_CONFIG_GEMM_BF16=...` env on the next `bench_e2e.sh` launch and gate it.
