# Qwen3.8-2.4T-A95B-Quark-MXFP4 on MI355X ×8 — SGLang + aiter, MXFP4 experts + FP8 KV — **NO WIN**

**This entry records no win. Nothing shipped, there is no deployable artifact, and there is no
number to reproduce.** It is here for three things and nothing else: its **negatives**, its
**shape and target inventory**, and its **cost model** — because this is the largest model in the
campaign, every server instance costs about twenty minutes to create, and the next reader's most
expensive mistake would be to spend those instances re-walking lanes that are already measured
and closed.

The run's own verdict, verbatim from `EXPERIMENT_COMPLETE`:

> Negative result: no throughput win over the locally measured 1266.119 tok/s baseline (best arm
> +0.17%, below the 1.03% restart-to-restart noise floor) — decode's MoE GEMMs are already the
> fastest of 209 compiled FlyDSL variants and run at 88% of this GPU's measured 6.61 TB/s
> streaming-read ceiling, the decode collective is already the fastest of three implementations,
> and decode is at 99.9% GPU utilisation; two real kernel-level wins (1.15-1.30x dense bf16 GEMMs,
> 1.019x prefill MoE stage 2) were built, installed, engagement-verified live and gsm8k-gated but
> provably lack the wall-time leverage to reach the benchmark, and are exported to patches/rejected/
> with their measurements.

Measured 2026-08-20 over a single day, on one machine, across at least five server instances.

> The directory rule is that only reproduced *wins* go in, because an unreproduced claim costs the
> next reader more than it saves. This entry makes no claim to reproduce, so it does not violate
> that rule — but do not read it as a soft positive. **Two arms were installed and verified live
> and both are in `rejected/`.** The one real, defensible non-throughput movement is a −2.97% mean
> TTFT, and even that is stated below as what it is: a latency finding on a benchmark whose key
> metric is output throughput.
>
> One further honesty note that shapes how the rest reads: "no win" here means **no win was found
> among the four lanes opened within the instances available**, with two of those lanes closed
> against measured hardware ceilings rather than merely abandoned. See
> [What "no win" does and does not mean](#what-no-win-does-and-does-not-mean) — the distinction is
> load-bearing and the bundle only supports one of the two readings.

## Environment fingerprint

Marked per field. The load-bearing/descriptive split still matters on a null entry: it tells you
whether the *closures* below apply to your stack, since a closure keyed on `gfx950,256` is as
inapplicable off that key as a win would be.

| field | value | load-bearing? |
| --- | --- | --- |
| GPU | 8× AMD Instinct MI355X, `gfx950`, device `0x75a3`, **256 CU** each | **yes** — every aiter lookup key here literally carries `gfx950,256`; `_INDEX_COLS` in `fused_moe.py` leads with `gfx`. Both closures (MoE selection, allreduce choice) were measured on this part and transfer nowhere else. |
| host | `crsuse2-m2m-272`, Ubuntu 22.04.5 | descriptive — but see the noise-floor section: the floor is a property of the node. |
| container | **image digest not recorded.** `scripts/start_container.sh` names the tag `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix`, which the bundle author matched to the framework version the source session reported; `docker_image` is null in every config the session wrote. | descriptive, but **a real gap** — the image's `ENV` block is what makes the biggest single lane in this run exist at all (below), so "which image" is closer to load-bearing than it looks. |
| SGLang | 0.5.17, source tree at `/sgl-workspace/sglang`, HEAD **`29481685462732237d80d86076d6563e1f658102`** | descriptive — no sglang source file was modified in this run. |
| aiter | commit **`d9e5ef7ce08ee7045d583aed768cff41aa9210fe`** | **yes** — every attempt here is an aiter config-table row or an aiter kernel-registry selection. The FlyDSL registry that was raced (272 stage-1 / 257 stage-2 entries) is a property of this commit; the counts and the kernel name strings move with it. |
| torch / HIP / ROCm | `2.9.1+rocm7.2.0.git7e1940d4` / `7.2.26015-fc0010cf6a` / 7.2.0 | descriptive |
| model | Qwen3.8-2.4T-A95B-Quark-MXFP4, **TP=8**, **EP=8** (`--expert-parallel-size 8`) | **yes** — TP=8 is what puts two different projections on the same N=4608, and EP=8 is what gives 64 local experts of 512. Both are in the shape list below. |
| weights | **MXFP4** via Quark (`quant=quark`; `--quantization` is *not* passed and the server reports `quantization=None`) | **yes** |
| KV cache | `fp8_e4m3` | **yes** — one of the three baseline flags, worth +5.26% in the source session. |
| MoE op | aiter `fused_moe` → **FlyDSL** a4w4 path (`use_mxfp4_flydsl`), `QuantType.per_1x32`, `use_g1u1=1` | **yes** — this is *which registry gets selected from*, and it is the single most misread fact in the stack. See attempt 3. |
| attention backend | `aiter`; mamba/linear-attention backend `triton` | descriptive (frozen flags) |
| library shas below aiter | **not recorded.** `preflight.sh` prints triton and python versions as notes but nothing captured them into the bundle; there is no CK submodule sha, no hipBLASLt version. | **a gap.** The hipBLASLt solution indices in the rejected GEMM table (438275…440237) are only meaningful against a specific hipBLASLt build, and that build is not written down. Anyone re-applying `rejected/0001` should re-run `analysis/validate_config.py` rather than trust the indices. |
| process environment | **not captured into the bundle as a file.** `FINDINGS.md` transcribes ten variables read from the live process; there is no `/proc/<pid>/environ` dump. | **load-bearing and a gap** — see immediately below. |

### The launch script sets no environment variables. The image sets ten, and one of them created a whole lane

`BASELINE.md` says this "is the only bundle in the batch whose baseline does not even set
`SGLANG_USE_AITER`". That is wrong about the process. The container image exports:

```
SGLANG_USE_AITER=1                   AITER_USE_SYSTEM_TRITON=1
SGLANG_MOE_PADDING=1                 ROCM_QUICK_REDUCE_QUANTIZATION=INT8
SGLANG_ROCM_FUSED_DECODE_MLA=1       HIP_FORCE_DEV_KERNARG=1
SGLANG_USE_ROCM700A=1                HSA_NO_SCRATCH_RECLAIM=1
SGLANG_SET_CPU_AFFINITY=1            TORCHINDUCTOR_MAX_AUTOTUNE=1
```

This is the most consequential fact about where the time goes on this stack, in both directions:

- `SGLANG_USE_AITER=1` is why `UnquantizedLinearMethod.apply` routes all 529 dense bf16 linears
  per forward through aiter's `tuned_gemm` table instead of `F.linear` — and that table has no rows
  for this model, so every one of them resolves to `libtype: torch`. **Attempt 1 exists entirely
  because of an environment variable the baseline document says is not set.**
- `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` is why the prefill allreduce is quickreduce-INT8 at all,
  which is what made "give decode the same algorithm" look like a 13.4%-of-decode lever. Attempt 4
  measured it and it is a regression, not a lever.
- `ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16` defaults on, so the quickreduce size threshold that governs
  a **bf16** tensor is the **fp16** row of `_QR_MIN_SIZE`. Reading the bf16 row gives 2 MB and sends
  you hunting a gain that is not there; the real INT8 floor is 4 MB.

The variables were part of the frozen configuration and were not changed. **Read
`/proc/<server-pid>/environ` before reasoning about which paths are live** — and note that this
bundle records the transcription but not the dump, so the ten above are as good as the run's
transcription and no better.

## Launch configuration

Reproduce verbatim. `scripts/launch_server.sh` (copied to `artifacts/scripts/launch_server.sh`)
builds exactly this argument vector and then verifies it against `/get_server_info`:

```bash
python3 -m sglang.launch_server \
    --model-path /shared_nfs/hyperloom/models/Qwen3.8-2.4T-A95B-Quark-MXFP4 \
    --host 0.0.0.0 --port 43114 \
    --tp-size 8 \
    --context-length 11264 \
    --watchdog-timeout 1200 \
    --attention-backend aiter \
    --page-size 1 \
    --chunked-prefill-size 16384 \
    --mem-fraction-static 0.9 \
    --model-loader-extra-config '{"enable_multithread_load":true}' \
    --trust-remote-code \
    --disable-radix-cache \
    --kv-cache-dtype fp8_e4m3 \
    --mamba-ssm-dtype bfloat16 \
    --expert-parallel-size 8
```

The script asserts nine fields on the live server — `context_length`, `tp_size`, `page_size`,
`attention_backend`, `chunked_prefill_size`, `kv_cache_dtype`, `ep_size`, `mamba_ssm_dtype`,
`disable_radix_cache` — and prints `config verified`. Three of those (`kv_cache_dtype: fp8_e4m3`,
`ep_size: 8`, `mamba_ssm_dtype: bfloat16`) *are* the baseline. The source session measured them as
successive increments over the untouched 1152.142 tok/s — `--mamba-ssm-dtype bfloat16` +2.32% to
1178.821, then `--expert-parallel-size 8` +3.15% to 1215.951, then `--kv-cache-dtype fp8_e4m3`
+5.26% to 1279.949, **+11.09% in total** — so a silently dropped one costs several percent. The run
reports this verification passing on every start.

Resolved values that are not visible in the invocation, from the server log rather than assumed:

- `mem_fraction_static` resolves to **0.765**, not the 0.9 requested — SGLang rescales by 0.85 on
  this build. The launch script accepts either value for that reason.
- `max_prefill_tokens=16384` (follows `--chunked-prefill-size`), `max_total_num_tokens=2,512,884`,
  `max_running_requests=714`, `quantization=None` despite MXFP4 weights (`quant=quark` is reported
  by the loader instead), `page_size=1`.
- KV cache **13.78 GB for K and 13.78 GB for V per rank** in `torch.float8_e4m3fn`.
- Decode CUDA graph: `backend=full`, 52 batch sizes **1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64,
  72 … 512**. **Prefill graph is disabled.**
- `--mamba-ssm-dtype bfloat16` overrides a checkpoint that asks for `float32` SSM state on 69
  linear-attention layers. It is recorded in the source session's journal as an `integrate_patch`
  with `scope: source_patch`, which reads like a landed patch. **It is a server flag and nothing
  else** — the run that produced it names no overlay, no setup step and no patch, and the session's
  `patches/` directory is empty.

**A launch is fragile and slow.** The script allows **7200 s** for `/health` and warns that a first
start also JIT-compiles aiter kernels. `--watchdog-timeout 1200` is the other side of that: it is
what killed a server when the profiler hung (below). The source session recorded **five server boot
failures** and a concurrency sweep that failed at **all eight** points it attempted, closing on
`conc_sweep_failed`. Plan for a failed start.

## Workload

Frozen, and identical to the rest of this directory:

```
ISL 8192   OSL 1024   concurrency 64   num_prompts 192   num_warmups 8
random dataset, random_range_ratio 1.0, --ignore-eos, seed 0
InferenceX benchmark_serving fork, --backend vllm against /v1/completions
--percentile-metrics ttft,tpot,itl,e2el  --metric-percentiles 90,99,99.9
```

Which parameters set the shapes in the target inventory:

- **concurrency 64** → the decode token bucket **64**, and the 1 MiB decode allreduce message
  (64 × 8192 × bf16).
- **`--chunked-prefill-size 16384`** → the prefill tier **16384**, because `get_padded_M` is
  `nextPow2` and so every chunk of 8193–16384 tokens keys on 16384. This is why a race at token
  8192 is worthless here: **8192 never occurs.**
- **TP=8 and EP=8** → `N=4608` for two different projections, `inter_dim=2048`, 64 local experts
  of 512.
- **ISL 8192 / OSL 1024** → the run is decode-bound: one ~11 s prefill against 1024 decode steps at
  ~40 ms. This single ratio is why both kernel wins in this run are unreachable, and it is the most
  transferable fact in the entry.

## The cost model — read this first

This is the largest model in the campaign and this section is the reason the entry exists. Every
figure below is traced to a file; where the bundle records two figures that disagree, both are
given.

### One server instance

| item | figure | source |
| --- | --- | --- |
| checkpoint | **1.25 TiB over 213 safetensors shards** | `BASELINE.md`; `preflight.sh` asserts `EXPECT_SHARDS=213` |
| torch distributed init | 13.78–14.68 s per rank | reference server log |
| **weight load** | **676.29–795.77 s per rank** (slowest rank TP7), with `enable_multithread_load` on | reference server log |
| KV cache alloc + memory pool | within the same second as the last rank's load end | reference server log |
| decode CUDA graph capture | **27.56–27.87 s** for all 52 batch sizes | reference server log |
| **process start → `The server is fired up and ready to roll!`** | **862 s = 14 min 22 s** (04:39:01 → 04:53:23) | reference server log, `reference/results/baseline_warmup/server.log.gz` |
| the run's own figure for *this* machine | **"~20 minutes per server start"** | `FINDINGS.md`, skillset assessment |
| health-wait allowance in the launch script | 7200 s | `scripts/launch_server.sh` |

**Which to plan against: twenty minutes.** The 14 min 22 s breakdown is the only complete startup
log in the bundle and it comes from the *source session's* machine on 2026-08-15, not from the local
run. **No local server log is in the bundle** — that is a recording gap, and it is why the
authoritative local figure is a sentence of prose rather than a timestamp. The two are consistent
in order of magnitude and the prose is the conservative one.

### One benchmark round

| item | figure |
| --- | --- |
| reported benchmark duration, all **15** non-profiled rounds | **154.27 – 156.16 s** |
| prompt generation | 2.1 – 2.4 s |
| 8 warmups | **20 – 22 s**, except **38.8 s** for the very first round against a fresh server |
| dead time between consecutive rounds in one process (`benchmark_end` → next `benchmark_start`) | 27.6 – 51.4 s |
| **wall clock per round**, launch → benchmark end | **182 – 200 s** (200 s is the first round on a fresh server) |
| back-to-back cadence, consecutive round launches | 182 – 214 s |

So **roughly 3.1 minutes per round**, and about **19 rounds per hour** *if* you already have a live
server and change nothing. `benchmark_start_time_unix` and `benchmark_end_time_unix` are recorded in
every `inferencex_result.json`, so this is measured rather than inferred from file mtimes.

### The other two fixed costs

- **gsm8k gate: 490.59 / 516.89 / 595.96 s** — 8.2 to 9.9 minutes, from
  `total_evaluation_time_seconds` in the three `results_*.json`. 1319 problems, 5-shot,
  `max_tokens=9216`, `num_concurrent=64`.
- **Profiling is expensive and dangerous.** The two profiled rounds measured 159.81 s and 169.41 s
  duration at 1230.29 and 1160.51 tok/s — that is 2.8% and 8.3% below the baseline mean, so **a
  profiled round is not comparable to an unprofiled one** and cannot be pooled with it. Worse,
  `/stop_profile` hung every rank inside torch's `ProfilerResult` assembly and the 1200 s watchdog
  killed the server; `analysis/prof_gemm/` holds the partial capture (EXTEND traces only, no
  DECODE) and the one salvaged trace was discarded as warmup-skewed rather than over-read.
  **Profile on a server you are willing to lose, and take the trace before the arm you care
  about, not after.**

### Restart turnaround, as actually achieved

The tightest observed turnaround — from one instance's last benchmark ending to the next instance's
first benchmark round being launched, which includes stop, VRAM reap, installing a CSV, the full
start and the health wait — is **10 min 30 s** (13:00:23 → 13:10:53, the baseline-restart instance
handing off to the tuned-GEMM instance; that round's own benchmark then began at 13:11:21). Treat
that as a floor observed once, not as a typical figure: the other two transitions took 61 min and
118 min from benchmark end to benchmark start, and each of those absorbed a gsm8k gate, analysis
work, and in the second case the profiler killing the server.

Over the whole measured day, **11:22:45 → 15:40:37 (4 h 18 min) produced at least five server
instances**: two baseline instances (5 rounds and 3 rounds), one tuned-GEMM instance (3 rounds),
one tuned-fmoe instance (4 rounds), and at least one more, because the profiler killed the
tuned-GEMM server before its accuracy gate ran at 14:05:43. Four are directly attested by
`FINDINGS.md` and the `results/` grouping; the fifth is inferred from the profiler kill and is
labelled as an inference.

### Therefore: the smallest effect worth chasing

Arithmetic on the two measured figures above, marked as arithmetic:

- **An instance carrying three rounds costs about 20 minutes** at the best turnaround observed
  (10.5 min + 3 × 3.1 min), so **three instances per hour is the arithmetic ceiling** and the run itself
  achieved about **1.2 per hour** once analysis, evals and one lost server are counted.
- The restart floor is **1.03%**, which at the 1266.119 tok/s baseline is **13.0 tok/s**.
- A Mixtral-style design — 3 baseline instances against 3 patched instances, interleaved — is
  **six instances ≈ 2 hours of pure measurement**, and it resolves an effect of roughly the floor,
  i.e. **about 1%**. Anything you want to claim at a fraction of a percent on this model needs
  either many more instances than a day affords, or a quieter metric than throughput.
- The companion rule, which this run says is missing from the skillset and which is the single most
  useful sentence to carry away: **a kernel win of `x` on a kernel that is `f` of wall time is
  worth `f·(1 − 1/x)`. If that is under your floor you are measuring noise, no matter how clean the
  microbenchmark is.** Put the 1.03% floor into it and you get the affordability test for this model
  directly: a surface at 10% of wall needs a **1.11×** kernel win to clear the floor, one at 5% of
  wall needs **1.26×**, and one at 2% of wall needs **2.06×**. Compute it *before* you spend an
  instance; this run did, twice, and was right both times.

## Baseline and noise floor

### Which baseline to quote: **1266.119 tok/s**, the local one

| | tok/s | note |
| --- | --: | --- |
| documented (source session `baseline_measure`) | 1279.949 | three reference rounds 1280.994 / 1279.949 / 1272.569, **spread 0.66%**; the session's code-level lane independently re-measured the same configuration at 1279.199, agreeing to 0.06% |
| **local, pooled, n=8 across 2 instances** | **1266.119** | **−1.08% from the documented figure, outside its own 0.66% reference spread** |
| untouched configuration, for scale | 1152.142 | the documented baseline is the source session's *configuration ceiling*, +11.09% over this |

`BASELINE.md`'s own rule decides it: a local mean outside the reference spread becomes the
baseline. **Every comparison in this entry is against 1266.119**, and the run states the same. The
shortfall is most likely machine-to-machine — nothing suggests a dropped flag, since the launch
script verified `kv_cache_dtype`, `ep_size` and `mamba_ssm_dtype` against the live server on every
start. Where a comparison is marginal the adjacent instance mean (1263.086) is also given, because
that instance shares the day and the thermal state with the patched arms.

### The eight baseline rounds

| run | tok/s | mean TTFT | mean TPOT | instance |
| --- | --: | --: | --: | --- |
| `base_r1` | 1269.874 | 10961.8 ms | 39.72 ms | 1 |
| `base_r2` | 1272.040 | 10960.2 ms | 39.64 ms | 1 |
| `base_r3` | 1259.052 | 10968.8 ms | 40.15 ms | 1 |
| `base_r4` | 1269.455 | 10987.6 ms | 39.71 ms | 1 |
| `base_r5` | 1269.271 | 10992.3 ms | 39.72 ms | 1 |
| `rst_r1` | 1261.221 | 10953.2 ms | 40.08 ms | 2, after a full restart |
| `rst_r2` | 1268.353 | 10952.7 ms | 39.79 ms | 2 |
| `rst_r3` | 1259.684 | 10962.6 ms | 40.13 ms | 2 |

### The floor, both terms separately

| noise floor | spread |
| --- | --- |
| repeating the benchmark within one process, instance 1 (n=5) | mean 1267.938, **sd 0.40%**, spread **1.02%** |
| repeating the benchmark within one process, instance 2 (n=3) | mean 1263.086, **sd 0.37%**, spread **0.69%** |
| across a restart | the two instance means differ by **−0.38%** |
| pooled across both, n=8 | mean 1266.119, **sd 0.41%**, **total spread 1.03%** |

**Use the restart floor: 1.03%.** Both candidate mechanisms in this run are aiter config-table
lookups, which are `functools.lru_cache`d per process and whose choice is frozen into the decode
CUDA graphs at capture. A config cannot take effect without a restart, so a within-process A/B is
not available even in principle (`../../tuning-core/measurement.md` Rule 3b).

Two things about that floor that a reader should not skip:

1. **A restart on this stack costs about what the within-instance scatter costs, and does not add a
   larger separate term.** That is unusual — contrast DeepSeek's 5.5–6.4% restart floor — and it is
   what makes the 1.03% figure usable at all.
2. **The restart term rests on two instances.** Two instance means give you a difference, not a
   standard deviation. The run's conclusion that a restart "does not introduce a larger separate
   term" is the right reading of the evidence available and is also the weakest link in the floor.
   If you intend to claim something near 1% here, measure a third and fourth baseline instance
   first — that is one hour, per the cost model, and it is the highest-leverage hour available.

### TTFT and TPOT are the quieter instruments, and it mattered

| metric | within-instance spread | what it cost or bought |
| --- | --- | --- |
| output throughput | 0.69% (instance 2) – 1.02% (instance 1) | could not resolve anything this run produced |
| **mean TTFT** | **0.09%** (instance 2) – **0.29%** (instance 1) | resolved a −2.97% prefill effect unambiguously |
| mean TPOT | 0.84% (instance 2) – 1.28% (instance 1) | resolved nothing; +0.70% and −0.23% both sit inside it |

`FINDINGS.md` quotes the TTFT spread as both "0.09–0.29%" and "0.03–0.29%" in different places and
both are correct under the right reading: **0.09% and 0.29% are the two baseline instances**, while
**0.03% is the tuned-GEMM arm's own three rounds** (10640.49 / 10643.20 / 10642.28 ms). Anyone
continuing on this model should report all three metrics, and should know that **at ISL 8192 /
OSL 1024 a prefill-side change is close to unmeasurable in `output_throughput`** — decode is 78.8%
of the per-request budget (prefill is 10 967 ms of 51 754 ms = 21.2%).

## There is no artifact. What would have to be true for there to be one

The template's artifact and deploy sections are dropped, because nothing was kept: there is no
`artifacts/` file here that you should install. `artifacts/analysis/qwen3_8_2_4t_a95b_bf16_tuned_gemm.csv`
and `artifacts/analysis/qwen3_8_2_4t_a95b_mxfp4_tuned_fmoe.csv` are present as **evidence, not as a
deploy target** — they are the two tables the run built, measured, gated and then removed. The
container was left stock with the server down and both CSVs deleted from
`/sgl-workspace/aiter/aiter/configs/model_configs/`, so a fresh `launch_server.sh` reproduces the
baseline and not a rejected arm. No sglang or aiter source file was ever modified; every change in
this run was an added config row, which is why every patch is a pure addition against
`aiter@d9e5ef7ce08ee7045d583aed768cff41aa9210fe` and needed no pristine copy to diff against.

For either of the two rejected tables to become shippable on this workload, one of these would have
to be true — and each is a concrete, checkable condition rather than a hope:

- **For the dense bf16 GEMM table (`rejected/0001`):** the operating point moves so that prefill
  carries real weight. At OSL 1024 prefill is 21.2% of the per-request budget and the table's
  prefill rows (up to 1.301× on M=16384) buy +0.08% of throughput. **Shorten OSL, or raise ISL, and
  the same eight rows become interesting** — a workload whose prefill share is 3–4× larger would put
  the measured kernel wins above the 1.03% floor by arithmetic. The table is also already a real
  win on the metric it does move: **mean TTFT −2.97%, ten times that metric's own spread.** If TTFT
  is what you are being asked for, this is not a negative result at all, and it is filed here only
  because output throughput was the key metric.
- **For the prefill MoE stage-2 row (`rejected/0003`):** nothing about the operating point saves
  this one. 1.019× on 13.0% of prefill on 21.2% of wall is 0.05%, and the arithmetic was done before
  the server was restarted and then confirmed by measurement. It would need the kernel-level win to
  be roughly twenty times larger.
- **For the MoE at the decode tier, the biggest surface in the model:** it would need a kernel that
  does not exist. 209 compiled alternatives were raced and the heuristic's pick is the fastest of
  them; `moe1` runs at 88% of this GPU's measured 6.61 TB/s streaming-read ceiling, so **a perfect
  replacement kernel wins at most 13% of `moe1` — and `moe1` is ~20% of decode, which comes to about
  2.1% end to end.** That is the number that closes the lane, and it also tells you the whole lane's
  remaining prize is worth about two noise floors.

## Engagement checks — both passed, and both things still failed to ship

Worth keeping precisely because they decouple "is it live" from "does it help". Prefer these to a
log line where you can, but on this stack the log lines are the practical instrument.

**The tuned fmoe row (`rejected/0003`).** With the row installed, on **all 8 ranks**, the server log
flips from the unconditional *miss* line

```
[fused_moe] no tuned FlyDSL config for ('gfx950', 256, 16384, 8192, 2048, 64, 9, ...)
```

to the *hit* line

```
[fused_moe] using 2stage (kernelName1='flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w3_bnt0',
            kernelName2='flydsl_moe2_afp4_wfp4_bf16_t64x256x256_atomic_persist')
            for ('gfx950', 256, 16384, 8192, 2048, 64, 9, ...)
```

Counted with `grep -c "using 2stage (kernelName1=.*for ('gfx950', 256, 16384," <server log>`;
**8 hits = all ranks engaged.** The asymmetry is the whole trick: **the miss line prints
unconditionally, the hit line is gated behind `AITER_LOG_TUNED_CONFIG=1`** (which this image
happens to set). Grepping only for the hit line against a working deploy on an image that does not
set it returns zero. The miss line is also how the untuned MoE was discovered in the first place.

**The tuned dense GEMM table (`rejected/0001`).** Verified by `analysis/validate_config.py`, which
re-times every installed row tuned-versus-torch **interleaved**, medians over 5 rounds, through the
real `tgemm.mm` entry point that sglang's `UnquantizedLinearMethod` calls — and exits nonzero on any
row that fails. It reported `0 failing rows` on an independent second run. This is a better check
than a log line because it exercises the same dispatch the server does; the flag-free negative
control is the `[aiter] … not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv, will use
default config! using torch solution:0` line, which the stock server emits during graph capture.

**Both arms passed. Both are in `rejected/`.** That is the point of separating engagement from
result, and it is why "verified live" is never by itself a reason to claim something.

## Accuracy gate

gsm8k 5-shot, 1319 problems, lm-eval `0.4.12` with the `[api]` extra in a venv of its own, task
from `eval/gsm8k.yaml`, `max_tokens=9216`, `temperature=0`, `top_p=1`, `--seed 0,1234,1234,1234`,
`num_concurrent=64`. Harness copied to `artifacts/scripts/run_eval.sh`.

| configuration | `exact_match,strict-match` | flexible-extract | source |
| --- | --- | --- | --- |
| source-session reference, **stock** configuration | 0.978014 ± 0.004039 | identical | `reference/eval/stock_gsm8k/results.json` |
| **local baseline — the gate** | **0.978014 ± 0.004039** | identical | `eval_results/base_gate_20260820_115331/` |
| tuned dense bf16 GEMM table | 0.981046 ± 0.003756 | identical | `eval_results/gemm_gate_20260820_140543/` |
| tuned prefill fmoe row | 0.979530 ± 0.003900 | identical | `eval_results/qwen38_24t_20260820_153225/` |

Three things about this table are worth more than the numbers:

1. **It closes the question `BASELINE.md` left open.** The local baseline — FP8 KV cache included —
   scores **bit-identical to the stock-configuration reference**. The FP8 KV cache is the largest of
   the three baseline gains and the only one with a mechanism for changing answers, and it was never
   evaluated in the source session. **It costs nothing on gsm8k.**
2. Both candidates pass. +0.0030 and +0.0015 are inside the combined standard error; **no accuracy
   improvement is claimed.** The GEMM gate had to be run regardless of the throughput verdict,
   because asm split-K reduction, triton and hipBLASLt all round differently from torch.
3. **The fmoe gate is weaker than it looks and the run says so.** The changed kernel only fires at
   padded token 16384, and gsm8k's short prompts reach that tier only when chunked prefill happens
   to batch enough of them together. It passes; it is not a tight test of that kernel's numerics.

Note also, if you ever get a score near 0.03 on this model: lm-eval's default 256-token generation
budget truncates the reasoning so the answer never arrives, and it measured 0.0318 strict-match on
this stack. That is a broken measurement, not a broken model.

## The target inventory — the part that survives

Nothing here depends on anything having shipped, and this is what a future run should start from
instead of re-deriving it. All of it is from torch traces at the real workload (concurrency 64, all
8 ranks), not from an analytic model.

### Decode: 31.35 ms/step, 2472 kernel launches/step, 92 layers ≈ 341 µs/layer

| item | µs/layer | % of decode | ≈ % of wall (derived: × 78.8%) |
| --- | --: | --: | --: |
| `moe1` up/gate GEMM (mxfp4) | 73.6 | 21.6 | 17.0 |
| `cross_device_reduce_2stage` ×2 | 45.3 | 13.4 | 10.6 |
| `moe2` down GEMM (mxfp4) | 39.3 | 11.5 | 9.1 |
| `MT16x16x1024` ×253/step — router, shared `gate_up`, GDN `in_proj_ba` | 32.4 | 9.5 | 7.5 |
| `MT192x64x128` + `PostGSU8` — attention QKV / GDN `in_proj_qkvz` | 23.4 | 6.9 | 5.4 |
| `paged_attention` (23/step) | — | 5.3 | 4.2 |
| `fused_recurrent_gated_delta_rule` (69/step) | — | 4.5 | 3.5 |
| `topkGatingSoftmax` | 10.4 | 3.1 | 2.4 |
| `fused_mx_quant_moe_sort` ×2 | 10.5 | 3.1 | 2.4 |
| `gemm_a16_w16` M64/N32/K256 — shared `down_proj` | 10.2 | 3.0 | 2.4 |
| `elementwise_manual_unroll` (212/step) | — | 2.9 | 2.3 |
| `moe_sorting` ×2 | 9.2 | 2.7 | 2.1 |
| `gemma_fused_add_rmsnorm` ×2 | — | 2.6 | 2.0 |
| `MT64x32x128` — shared `down_proj` | 6.5 | 1.9 | 1.5 |

The "% of wall" column is arithmetic, not a measurement: decode is 78.8% of the 51 754 ms
per-request budget because prefill is the other 21.2%. It is included because the Amdahl share
against **wall**, not against decode, is what decides whether a lane is affordable. It also
reconciles the run's statement that the MoE GEMMs are "26% of wall time": 33.1% of decode × 78.8%.

### Ranking of surfaces by Amdahl share, with each one's status

| rank | surface | share | status after this run |
| --: | --- | --: | --- |
| 1 | MoE GEMMs `moe1`+`moe2` | 33.1% of decode ≈ **26% of wall** | **Closed.** Fastest of 209 compiled FlyDSL variants at the decode tier; `moe1` at 88% of the measured 6.61 TB/s ceiling. A perfect kernel is worth ≤13% of `moe1`, i.e. ~2.1% end to end. |
| 2 | dense bf16 GEMMs (5 shapes, 529 launches/forward) | ~24.5% of decode | **Measured, ~1.2× closer to roofline, unshippable here.** Was ~3.1 TB/s effective, ~2.5× off. Kernel wins 1.04–1.30×; end to end +0.08%. |
| 3 | decode collective `cross_device_reduce_2stage` | 13.4% of decode ≈ 10.6% of wall | **Closed.** Fastest of three implementations at 1 MiB. Only surface left is the custom AR's launch geometry, which needs aiter's JIT module rebuilt. |
| 4 | MoE routing and sorting | **8.8% of decode** — `topkGatingSoftmax` 3.1% + two `moe_sorting` 2.7% + two `fused_mx_quant_moe_sort` 3.1% | **Open and never benchmarked.** 10.4 µs for a [64, 512] softmax-and-top-10 to route 64 tokens to 10 of 512 experts is slow enough to deserve a second implementation; aiter ships its own `topk_softmax`. **Start here.** |
| 5 | attention paths | **9.8% of decode** — `paged_attention` 5.3% + `fused_recurrent_gated_delta_rule` 4.5% | Untouched. Note the analytic roofline snapshot attributes **78.6% of decode** to attention, and the source session's whole-GPU profile put the two paths at 13.5% combined (8.55% + 4.93%). This run's decode-only trace says 9.8%. **Where they disagree, prefer the profile — and check which profile you are reading.** |
| 6 | `elementwise_manual_unroll`, 212 launches/step | 2.9% of decode | Open, never attributed to a source line. Small, but 212 launches is a lot of launches. |

### The five dense bf16 shapes, and why there are only five

| shape (M, N, K) | what it is | launches/forward |
| --- | --- | --: |
| (M, 4608, 8192) | attention QKV / GDN `in_proj_qkvz` | 92 |
| (M, 8192, 2048) | attention `o_proj` / GDN `out_proj` | 92 |
| (M, 512, 8192) | router `gate` + shared-expert `gate_up` | 184 |
| (M, 8192, 256) | shared-expert `down_proj` | 92 |
| (M, 32, 8192) | GDN `in_proj_ba` | 69 |

529 launches per forward, every one on aiter's untuned `libtype: torch` default. **The two 4608s
coincide for a reason worth knowing:** at TP=8 the attention QKV projection (64 Q heads × 256,
doubled for `attn_output_gate`, plus KV) and the GDN `in_proj_qkvz` (2048 q + 2048 k + 16384 v +
16384 z, /8) both land on N=4608. Change TP and they separate.

**M values that actually occur:** **64** (decode, from concurrency 64; by aiter's `get_padded_m`
rules an M=64 row serves M ∈ [48, 64]) and **8192 and 16384** (the two prefill batch sizes this
workload produces). Decode CUDA graphs are also captured at 8, 72, 80, 88, 96 … 512 and those all
fall through to `libtype: torch` — irrelevant at concurrency 64, relevant the moment the workload
changes.

### The MoE geometry and its prefill tiers

`('gfx950', 256, <token>, 8192, 2048, 64, 9, ActivationType.Silu, torch.bfloat16,
torch.float4_e2m1fn_x2, torch.float4_e2m1fn_x2, QuantType.per_1x32, True, False)` — hidden 8192,
inter 2048, **64 local experts of 512 at EP=8**, 9 routed slots plus the masked EP slot, per_1x32
mxfp4 with the (16,16) weight shuffle.

From the EXTEND trace, which is the only thing that can tell you which tier costs anything:

| tier | kernels | launches | GPU time | share of the 6166.6 ms extend span |
| --- | --- | --: | --: | --: |
| `tile_m=64` (padded token 16384) | `mfma_moe{1,2}_…_t64x128x256_…` | 818 | **802.6 ms** | **13.0%** |
| `tile_m=128` (padded tokens 4096–8192) | `…t128x128x256` | 92 | 61.1 ms | 1.0% |

The `no tuned FlyDSL config` line fires once per padded token tier per rank, so the log enumerates
every tier the server ever touches — 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
16384 — and says **nothing** about which of them costs anything. **Racing a tier you have not seen
in a trace is racing a branch the server may never take.**

### Three corrections to carry forward, all load-bearing

- **`BASELINE.md`'s profile table has one bad row.** It reports `greedy_sample_kernel` at 5.86% of
  GPU time, the sixth-largest entry. In the decode traces taken here **sampling is 0.06%** — about a
  hundredth of that. The rest of that table reproduces well, so this is one bad row rather than a
  different workload. **Drop the sampling line before planning against it.**
- **The "8.4% of roofline" figure is an artifact.** The analytic ceiling of 15301.9 tok/s was
  computed with `quantization: none`, i.e. 2 bytes per weight on a 4-bit model, so it over-counts
  weight traffic and under-states the ceiling. The true picture is not "92% of the performance is
  missing" but: a third of decode is within 12% of the memory system, an eighth is in a collective
  that is already the fastest of three implementations, and a quarter is in dense GEMMs that were
  ~2.5× off the roofline and are now ~1.2× closer.
- **There is no idle time to reclaim.** Union of busy intervals on the compute stream per rank gives
  **99.9% GPU utilization** — rank 0 spends 298.10 ms busy in a 298.38 ms span, **0.28 ms of total
  gap across 23 026 launches**, and rank 3 is the same. Decode is entirely graph-captured and
  entirely GPU-bound, so nothing here is winnable by removing launch overhead, overlapping host work
  or scheduling differently — only by making kernels faster. Worth stating because "reduce CPU
  overhead" is the usual first guess at 2472 launches/step.

## What was tried and did not work

The body of the entry. Every row has its measured kernel-level number *and* its measured end-to-end
number, because the gap between the two is the finding.

| attempt | kernel / op level | end to end | verdict |
| --- | --- | --- | --- |
| **1. aiter tuned dense bf16 GEMM table**, 8 rows, `model_configs/qwen3_8_2_4t_a95b_bf16_tuned_gemm.csv` | **8 rows survived at 1.044–1.301×**, out of 13 shapes searched, each independently re-validated interleaved: M=64 N=4608 K=8192 **1.152×** (asm splitK=2), M=64 N=8192 K=256 **1.260×** (triton), M=64 N=32 K=8192 **1.145×**, M=8192 N=512 K=8192 **1.187×**, M=8192 N=8192 K=256 **1.044×**, M=16384 N=4608 K=8192 **1.301×**, M=16384 N=512 K=8192 **1.184×**, M=16384 N=8192 K=256 **1.062×** | throughput **1266.119 → 1267.102 tok/s = +0.08%** (n=3; +0.32% against the adjacent instance) — **inside the 1.03% floor**. Mean TTFT **10967.4 → 10642.0 ms = −2.97%**, against that metric's 0.09–0.29% spread, reproduced on all three rounds (own spread 0.03%). Mean TPOT +0.70%, inside its 0.84–1.28% spread. gsm8k 0.981046 ± 0.003756 vs gate 0.978014 ± 0.004039 | **Rejected on the key metric; kept as a documented latency finding.** The reason is structural and was computable in advance: at ISL 8192 / OSL 1024 the run is decode-bound, so prefill rows have ~a tenth of the leverage their size suggests, and the decode-side rows land on 24.5% of decode at only 1.15–1.26×. The arithmetic never got above ~2%. `artifacts/patches/rejected/0001-…patch` |
| **1a. the four rows the search liked and validation killed** | single-shot search said 1.078× / 1.234× / 1.053× / 1.129×; **interleaved re-timing said 0.976× / 1.017× / 0.997× / 0.965×** | not run | **Dropped — the search had been measuring clock drift, not a kernel.** The search times each candidate once in its own process, which is fine for ranking ~2090 candidates and far too loose to accept a 5% claim. **This row is the most reusable thing in the table:** a third of the winners did not survive being re-timed properly. |
| **1b. two shapes deliberately not claimed**, (8192, 8192, 2048) and (16384, 8192, 2048) | tuned at 1.056× and 1.040× | not run | **Left out on purpose.** `llama70B_bf16_tuned_gemm.csv` already owns those keys and picks torch. aiter merges every `model_configs/*bf16_tuned_gemm*.csv` into one table and **raises on a duplicate shape key after rewriting the source CSVs** — so claiming them for a gain inside the noise floor would break the launch *and* corrupt another model's config. |
| **2. Fuse the router `gate` and shared-expert `gate_up` into one N=1024 GEMM** (184 launches/forward, 9.5% of decode) | the GEMM behaves exactly as hoped — at M=64, **2× torch N=512 19.48 µs → 1× tuned N=1024 10.65 µs**. Then the two output column-slices are non-contiguous and both consumers (`topk_softmax`, `silu_and_mul`) need contiguous input: **+ the two `.contiguous()` = 18.82 µs, net 1.035×.** Prefill: 1.195× at M=8192, 1.311× at M=16384 | **never landed** — 1.035× on 9.5% of decode is **0.3%**, under a third of the floor | **Rejected on price, before landing.** The fused GEMM saves 8.8 µs and the two 64 KB copies give back 8.2 µs; that time is pure launch overhead, not bandwidth, so a better copy kernel does not shrink it, and a row-major GEMM cannot write two separately-contiguous outputs. **Recorded because two-GEMMs-on-one-input looks like free money in a profile and is not.** `artifacts/patches/rejected/0002-…NOT-LANDED.md` |
| **3. The MoE GEMMs at the decode tier**, 33.1% of decode, the largest surface in the model | **209 compiled variants raced through the real dispatch** — 80 stage-1 and 129 stage-2, in two phases (stage 1 against a fixed stage 2, then stage 2 against the winning stage 1) because the two kernels are sequential and share only `block_m` — interleaved, 7 rounds × 30 iterations. The heuristic pair wins: `_w4` "leads" the heuristic `_w2` by **0.17% against its own 3.25% round-to-round spread**, and in the stage-2 phase the heuristic came first outright at 246.53 µs. **Final: 1.0000×.** The race did discriminate — `_xcd4` variants are consistently **1.6× slower** (385–396 µs) and the `reduce` stage-2 family loses to `atomic` by 8–10% | not run — there was nothing to install | **Closed as a measured negative.** And bounded, which is the valuable part: `moe1` moves 42 × 17.83 MB in 128.5 µs = **5.83 TB/s** against a **measured** 6.61 TB/s pure-streaming-read ceiling (`bench_hbm_ceiling.py`; `torch.sum` 3.75, `copy_` 4.83 TB/s counting read+write) = **88% of achievable**. A perfect replacement wins ≤13% of `moe1`, and `moe1` is ~20% of decode, so ≈ **2.1% end to end — for a kernel nobody has written.** |
| **3a. the CK 2-stage knobs — `block_m`, `ksplit`, `use_nt`** | **all 24 combinations within 0.3%** of each other and of the default | not run | **A no-op, and the first wasted half-day.** For a4w4 mxfp4, `fused_moe` never takes the CK 2-stage path; it routes to FlyDSL where those three parameters are **dead**. The `[fused_moe] using 2stage default for (…)` log line names a lookup with no effect on this model — **the line that matters is the FlyDSL one underneath it.** Anyone reading the first line and hand-writing a `*_tuned_fmoe.csv` would be tuning nothing. |
| **3b. a race at token 8192** | reported 1.0845× | never installed | **Worthless, and instructively so.** `get_padded_M` is `nextPow2`, so **8192 never occurs** — any 8193–16384 chunk keys on 16384. What it "found" was the `tile_m=64` kernel the ladder already selects one tier up. |
| **3c. the prefill MoE stage-2 row at token 16384** (`rejected/0003`) | **1130.16 → 1109.95 µs = 1.0182×**, interleaved 9 rounds × 30 iterations at the production `ksplit=0`. Reproduced across four independent races: **1.0211×** (80 × 129 candidates), **1.0195×** (the full 272 × 257 space), **1.0207× / 1.0201×** (head-to-head at `ksplit=-1`) and **1.0194× / 1.0182×** (head-to-head at `ksplit=0`); the two winning stage-2 kernels are tied with each other, 0.07% and 0.11% apart in the two direct comparisons, one win each. **All of it is stage 2** — stage-1 variants `_w3_bnt0`/`_w4_bnt0`/`_w2_bnt0` span 0.15% (1140.51 / 1142.05 / 1142.16 µs) | **installed, engaged on all 8 ranks, measured: 1266.119 → 1266.767 tok/s = +0.05%** (n=4, own spread 0.59%); **+0.17%** dropping the first run against a fresh server (n=3, spread 0.19%). TTFT −0.01%, TPOT −0.23%. gsm8k 0.979530 ± 0.003900 — passes | **Rejected: a real, live, reproducible kernel win with nowhere near the leverage to be a throughput win.** The null was **predicted before the server was restarted** — 802.6 ms of a 6166.6 ms extend span = 13.0% of prefill; prefill is 21.2% of wall; 1.9% of that = **≈0.05%, nineteen times below the floor**. Predicted +0.05%, measured +0.05%. It was installed anyway, because a prediction never checked against a live server is exactly the failure mode worth avoiding. |
| **4. Give decode the prefill allreduce (lower the quickreduce threshold)** — 13.4% of decode, 184 allreduces/step, ~22.6 µs each on 1 MiB across 8 ranks ≈ 46 GB/s, latency-bound | 8-rank interleaved race (7 rounds × 200 calls, median per rank then **max across ranks**, because a collective is only as fast as its slowest participant). At the **decode size, 1 048 576 B: quickreduce 32.66 µs, aiter custom AR 18.82 µs, RCCL 33.46 µs.** Quickreduce is **1.74× slower**, and slower at every size where both are available (2 MiB: 35.14 / 18.16 / 38.86; 4 MiB: 43.66 / 26.45 / 55.02; 8 MiB: 76.45 / 44.62 / 77.96). At the prefill size, 268 435 456 B, the custom AR **declines outright** (>64 MiB cap) and quickreduce beats RCCL 1030.77 vs 1191.54 µs | not run — there was nothing to install | **Closed as a measured negative, and it was the lane the earlier draft called most promising.** The 4 MB INT8 floor is **protecting** the stack; lowering it would be a regression on 13.4% of decode. RCCL is slowest of the three everywhere. **The 256 MiB custom-AR cell is "declined", not "fast"** — `custom_all_reduce` returns `None` above its cap, so there is no kernel to time; the raw run printed 0.27 µs and recording that as a measurement would have been a fabrication. |
| **4a. the remaining allreduce knobs**, all read in aiter's `custom_all_reduce.cuh` / `.cu` rather than guessed | *1-stage vs 2-stage:* at world_size 8 the 1-stage path is taken only below 80 KB; the decode tensor is 1 MiB, 13× over, and 1-stage at 8 ranks is 8× the cross-device traffic. *`open_fp8_quant`:* exposed in the Python API but applied **only inside the fp16 branch**; the bf16 path ignores the argument, and the kernel behind it carries the upstream comment *"bf16 quant fp8 kernel function / too slow need to be optimized"*. *Write mode:* gated on `arch.find("gfx942")` **and** `bytes > 4 MiB` — wrong architecture, and the interesting sizes are past the 64 MiB cap anyway. *Launch geometry* (`kMaxBlocks = 80`, `threads = 512`, ROCm `block_limit = 16`): the only surface left, needs aiter's JIT module rebuilt | not run | **All dead ends, three of them provably.** Enabling `open_fp8_quant` on this model is not a flag flip, it is writing the kernel that comment says does not exist. **Read the dispatch order and the guards before tuning a constant.** |
| **5. Shared-expert fusion into the MoE kernel** | not measured — unavailable | not run | **Unavailable by configuration.** The `enable_shared_expert_fusion` path is gated on this configuration's sense of `moe_ep_size > 1` being false, and `--expert-parallel-size 8` is one of the three frozen baseline flags. The N=1 `shared_expert_gate` is meanwhile already folded into `_fused_gate_sigmoid_mul_add_kernel`, so there is no free launch to reclaim there either. Nothing to do without changing a frozen flag. |

### The two rows to read twice

**A real kernel win is not a result.** This run produced two — 1.15–1.30× on the dense GEMMs and
1.019× on the prefill MoE stage 2 — and both are live, engagement-verified, accuracy-gated and
worthless on the key metric, for reasons that were arithmetic on the profile before either was
installed. That matches every other entry in this directory. What is *new* here is that the
arithmetic was done first and then confirmed, rather than invented afterwards to explain a
disappointing number.

**A search can lie in both directions.** Four of twelve GEMM winners evaporated on interleaved
re-timing, and a FlyDSL variant "beat" the heuristic by 0.17% against its own 3.25% spread. Run
either sequentially and both would have been reported as wins. `measurement.md` Rule 6b —
interleave, never A/B back to back — is the single most load-bearing item in the skillset for this
model; on gfx950 a back-to-back comparison drifts 20–67%.

## What "no win" does and does not mean

The bundle supports exactly one of the two readings, and the difference matters more than anything
else in this entry.

**Supported: no win was found among the four lanes opened, within roughly five server instances on
one machine on one day.** Four lanes were opened and one was unavailable by configuration. Surfaces
4, 5 and 6 in the ranking table — MoE routing and sorting at 8.8% of decode, the attention paths,
`elementwise_manual_unroll` — were never benchmarked at all. A fifth lane, `--moe-runner-backend
triton`, was rejected by the source session as unstable (+1.13% once, then +0.59%, both inside the
floor established here) and was never profiled. **Any of these could still hold something.**

**Not supported: "there is no win on this model."** Nobody measured that and this entry does not
claim it.

**Stronger than either, for two specific surfaces:** the MoE GEMMs at the decode tier and the decode
collective are closed against *measured hardware and implementation ceilings*, not merely
abandoned. `moe1` at 88% of a measured 6.61 TB/s read ceiling caps the entire MoE lane at ~2.1% end
to end even with a perfect kernel; the custom allreduce at 18.82 µs is the fastest of three
implementations at the size that occurs. Those two together are **46.5% of decode** (33.1% + 13.4%)
and they are genuinely closed to *selection*. **If you have one day on this model, do not spend it
there.**

One more thing that deserves saying plainly. The source session's code-level lane returned
`no_gain` at exactly 1.0× after 8h15m in which it profiled the stack, found 47 kernels of which 25
were hot, marked **every one** `selected_for_optimization: false`, and attempted nothing. Its
conclusion happened to be right and its process was not evidence of anything. **This run reached
the same 1.0× having actually opened four lanes, and that is a different claim with the same
number.** Do not let the matching headline collapse the two.

## What the run would do differently

In its own priority order, all of it traceable to something that cost it time:

1. **Measure the achievable ceiling on the part before searching.** The two measurements that
   actually closed this run were not searches: `moe1` at 5.83 TB/s against a *measured* 6.61 TB/s
   streaming-read ceiling, and 99.9% GPU utilisation across 23 026 launches with 0.28 ms of gap.
   `bench_hbm_ceiling.py` is twenty minutes and it would have saved most of a day.
2. **Do the leverage arithmetic before chasing the kernel, not after.** `f·(1 − 1/x)` against your
   floor. Every kernel win in this run was real and none of them cleared the floor, and all of it
   was predictable from the profile in advance.
3. **Read which dispatch path the operator actually takes before tuning its knobs.** The half-day
   lost to `block_m`/`ksplit`/`use_nt` was lost because the skillset classifies FlyDSL as
   "author a search space" when for this operator it is **"select from a pre-compiled set"** — a
   registry of 272 stage-1 and 257 stage-2 **name-addressed** precompiled kernels in
   `aiter.ops.flydsl.moe_kernels._KERNEL_PARAMS`, with no `Config`, no `key` and no autotuner in the
   path. The live knobs are `kernelName1`/`kernelName2`; the deploy path is a `*_tuned_fmoe.csv` row.
   Beware also `docs/coverage_gfx950.md` §12, whose `moe_mxfp4` case tunes `BLOCK_SIZE_M` against
   the **Triton** `fused_moe` — a different operator with the same name, and not the one an SGLang
   MXFP4 model dispatches to.
4. **Take the trace before the arm you care about.** `/stop_profile` hung every rank and the
   watchdog took the server with it.
5. **Report all three metrics.** TTFT resolved a real effect that throughput could not see at all.
6. **Measure a third and fourth baseline instance.** The restart term rests on two.

### Leads left genuinely open

- **MoE routing and sorting, 8.8% of decode.** Never benchmarked. 10.4 µs for a [64, 512]
  softmax-and-top-10; aiter ships its own `topk_softmax`. The highest-share untouched surface.
- **`--moe-runner-backend triton`, from the profile side.** The source session's rejection was
  correct on the evidence — both its figures are inside the floor established here — but it was
  never profiled, and the routing/sort overhead above is exactly what a different runner backend
  would change.
- **`elementwise_manual_unroll`**, 212 launches/step for 2.9%, never attributed to a source line.
- **The custom allreduce's launch geometry** (`kMaxBlocks`, `threads`, `block_limit`), which needs
  aiter's JIT module rebuilt. Note the headroom being sought is against the hardware, not against a
  bad choice: 18.82 µs already beats two independent implementations.
- **Decode batch sizes other than 64.** Graphs are captured at 8, 72, 80, 88, 96 … and all fall
  through to `libtype: torch`. Irrelevant at concurrency 64, relevant the moment the workload moves.
- **The dense GEMM table at a prefill-heavier operating point.** Same eight rows, different
  arithmetic.

### Leads closed

- **MoE GEMM selection at the decode tier** — fastest of 209 at 88% of the measured ceiling.
- **The decode collective** — fastest of three implementations at 1 MiB; the threshold keeping
  quickreduce out is doing its job.
- **The CK 2-stage knobs on this model** — dead parameters on the path that actually runs.
- **The router/`gate_up` fusion** — priced out by two 64 KB copies.
- **Shared-expert fusion** — unavailable under `--expert-parallel-size 8`.
- **The `tile_m=128` prefill tier** (padded tokens 4096–8192) — 92 launches / 61.1 ms = 1.0% of
  extend, so even a 10% win is 0.02% of wall. **Skipped deliberately, not overlooked.**
- **CPU/launch overhead anywhere in decode** — 99.9% GPU utilisation, 0.28 ms of gap in 23 026
  launches.

## When this entry stops applying

Silently, in every case — and for a null entry "stops applying" means the *closures* stop being
trustworthy, which is the more dangerous failure because it reads as permission:

- **arch ≠ gfx950 or `cu_num` ≠ 256** — every lookup key here carries them. Both closures were
  measured on this part and on no other.
- **A different aiter commit** — the FlyDSL registry that was raced (272 + 257 name-addressed
  kernels) belongs to `d9e5ef7c`. A new commit can add the kernel that would have won.
- **TP ≠ 8 or EP ≠ 8** — the two N=4608 projections separate, `inter_dim` moves, the 64-local-expert
  geometry changes, and the whole shape list is off.
- **`--chunked-prefill-size` ≠ 16384** — moves the prefill tier off 16384 and invalidates the
  entire prefill half of the target inventory.
- **Concurrency ≠ 64** — moves the decode bucket off M=64 *and* changes the allreduce message size,
  which is the axis the collective closure was measured along.
- **OSL ≫ or ≪ 1024** — does not invalidate any measurement, but **inverts the conclusions**. The
  rejected dense GEMM table becomes interesting as prefill's share rises, and the −2.97% TTFT
  becomes the headline rather than a footnote.
- **bf16 or FP8 weights instead of MXFP4** — different operator entirely; the a4w4 FlyDSL path is
  not taken and the CK 2-stage knobs stop being dead.
- **An image that does not set `SGLANG_USE_AITER=1`** — the 529 dense linears go back to `F.linear`
  and attempt 1 does not exist. An image that does not set `AITER_LOG_TUNED_CONFIG=1` breaks the
  engagement check's hit line while leaving the miss line intact.

**Still reusable when all of that changes:** the cost model's structure (weight-load-dominated
start, ~3 min rounds, restart-mandatory config changes), the method — leverage arithmetic before
the search, measured ceiling before the roofline claim, interleaved re-timing before accepting a
search winner — and every row of the negatives table as a record of what turned out not to matter.

## What the bundle does not record that the template asks for

Stated as gaps rather than filled in, because a wrong number is worse than a missing one:

| field | status | why it is a gap |
| --- | --- | --- |
| container image **digest** | **not recorded.** `docker_image` is null in every config the session wrote; the tag in `start_container.sh` was matched to the reported framework version by the bundle author, not read off a running container. | The image's `ENV` block is what creates the largest lane in this run. "Which image" is nearly load-bearing here. |
| **process environment dump** | **not recorded as a file.** Ten variables are transcribed in `FINDINGS.md`; there is no `/proc/<pid>/environ` capture. | The run's own headline lesson is that the launch script's silence about env vars is misleading. The evidence for the correction is a transcription. |
| triton / hipBLASLt / CK versions | **not recorded.** `preflight.sh` prints triton and python as notes; nothing captured them. | The eight rejected GEMM rows name hipBLASLt **solution indices** (438275…440237), which are only meaningful against a specific build. Re-run `validate_config.py`; do not trust the indices. |
| **local server logs** | **not in the bundle.** Only the source session's `reference/results/baseline_warmup/server.log.gz` is present. | The local startup cost — the single most important number in the cost model — is therefore prose ("~20 minutes") rather than a timestamp, and the 14 min 22 s breakdown is from a different machine six days earlier. |
| per-run raw results | **not a gap — noted for contrast.** All 22 `inferencex_result.json` files carry full-precision metrics plus `benchmark_start_time_unix` / `benchmark_end_time_unix`. | The cost model's per-round and cadence figures are measured from these rather than inferred from file mtimes, which is why they are quoted to the tenth of a second. |
| a working reference profile | **does not exist.** TraceLens failed both attempts with `steady_state_chunk_empty`; `reference/tracelens/` holds the logs plus two graph-capture traces at **batch 1, concurrency 1**, which say nothing about the concurrency-64 workload. | The entire target inventory here comes from this run's own torch traces. There is no independent profile to check it against. |
| lm-eval **commit sha** | **not recorded.** Version is pinned in prose to `0.4.12` with the `[api]` extra; `git_hash` is null in all three results files. | Other entries in this directory pin a sha. This one pins a release number. |
| a third baseline instance | **not run** | The restart floor rests on two instance means. |
| reproduction by an independent party | **not applicable** — there is nothing to reproduce | Stated for completeness: the two rejected patches were both re-verified with `git -C /sgl-workspace/aiter apply --check` against a restored stock tree and both return 0, which is a statement about the patches applying, not about their numbers. |

## Provenance and artifacts

Task bundle: `tuning_workspace/experiment_standalone/qwen38_24t_a95b_mxfp4_tuning/`.

- `EXPERIMENT_COMPLETE` — the one-line verdict, quoted in full at the top of this entry.
- `FINDINGS.md` (661 lines) — the run report and the main source for everything here. "Local
  baseline" has the eight rounds and both floor terms; "Where the time actually goes" is the target
  inventory; "What was tried" §1–§5 are the attempts; "Open threads" is the open/closed split;
  "Assessment of `tuning_skillset/`" is where the FlyDSL misclassification and the missing
  leverage-arithmetic rule are argued.
- `BASELINE.md` — the documented 1279.949, its three reference rounds, the three baseline flags and
  their individual gains, the source session's timeline, and the two figures this run corrected
  (the 5.86% sampling row and the 8.4%-of-roofline artifact).
- `results/` — 17 result directories: the **15** clean rounds in this entry's tables plus the two
  profiled rounds, which are not comparable to them. Each holds a `bench_stdout.log` with the full
  serving-benchmark block and an `inferencex_result.json` with full-precision metrics and benchmark
  start/end unix timestamps.
- `eval_results/` — the three gsm8k runs, with `total_evaluation_time_seconds` in each
  `results_*.json` (the cost-model eval figures).
- `reference/` — the source session, copied not re-derived. `results/baseline_warmup/server.log.gz`
  is the startup timeline; `codelane/` is the 8h15m lane that attempted nothing;
  `tracelens/` is two failed profile attempts. Note the directory's own warning: two upstream tool
  names were redacted, so **paths inside those files no longer resolve** — treat them as a record of
  where something was.
- `analysis/prof_base/`, `analysis/prof_shapes/`, `analysis/prof_gemm/` — the torch traces behind
  the target inventory, 8 ranks each, DECODE and EXTEND. **Not copied here** (~260 MB); read them
  from the bundle with `artifacts/analysis/trace_summary.py` and `trace_shapes.py`.
  `prof_gemm/` is the partial capture from the profiler hang — EXTEND only, and warmup-skewed.

### Copied into `artifacts/`

The reusable part of a null run is its tooling, so all of it is here.

**The rejected changes, headers intact** — each header carries its base commit, its apply command,
its kernel-level measurement, its end-to-end measurement and the reason for rejection:

| file | what it is |
| --- | --- |
| `artifacts/patches/rejected/0001-aiter-tuned-gemm-qwen3_8_2_4t_a95b.patch` | 8-row dense bf16 GEMM table. +0.08% throughput, −2.97% TTFT. |
| `artifacts/patches/rejected/0002-router-shared-gate_up-fusion.NOT-LANDED.md` | the fusion that was priced and never landed. No diff — the file exists so the measurement is not lost. |
| `artifacts/patches/rejected/0003-aiter-tuned-fmoe-prefill-16384.patch` | the one-row prefill MoE stage-2 table. 1.0182× at the kernel, +0.05% end to end, predicted before measurement. |
| `artifacts/patches/README.md` | the bundle's own rules for what a patch header must carry. |

**The harness** — reproduce the operating point exactly, or you are not comparing to anything:
`artifacts/scripts/launch_server.sh` (the verbatim launch plus the nine-field live-config
assertion), `run_bench.sh` (the workload contract), `run_eval.sh` (the gsm8k gate, including the
`max_tokens=9216` and `reasoning_content` fixes without which this model scores 0.03),
`preflight.sh` (stack assertions: sglang 0.5.17, ROCm 7.2.0, 8 GPUs, gfx950, 213 shards),
`start_container.sh` (including the non-contiguous `/dev/dri/renderD*` trap on these nodes).

**The analysis and kernel-benchmark tooling** — this is what a future run on this model should
actually reuse:

| file | what it does |
| --- | --- |
| `artifacts/analysis/bench_hbm_ceiling.py` | **measured** streaming-read ceiling on this part — the roofline denominator, and the twenty minutes that closed the MoE lane |
| `artifacts/analysis/tune_flydsl_moe.py` | races the 272 + 257 compiled FlyDSL MoE kernels through the real dispatch, by planting a synthetic row into `fused_moe.cfg_2stages` and clearing the memoized lookup |
| `artifacts/analysis/bench_moe.py` | rebuilds this model's exact MoE shape standalone (64 local experts of 512 at EP=8, hidden 8192, inter 2048, 9 routed slots, per_1x32 mxfp4 with the (16,16) shuffle); prices the CK 2-stage knobs that turn out to be dead |
| `artifacts/analysis/bench_allreduce.py` | 8-rank interleaved race, quickreduce vs aiter custom AR vs RCCL, at the sizes this workload produces; its header documents the `_QR_MIN_SIZE` fp16-row trap |
| `artifacts/analysis/tune_one.py` + `tune_driver.sh` | crash-resumable search over ~2090 GEMM candidates for one shape (hipBLASLt `findallsols` + triton + torch + asm at splitK ∈ {0,1,2,4,8}); the driver restarts past candidates that hard-fault the GPU |
| `artifacts/analysis/tune_gemm.py` | the first, all-shapes-in-one-process version of that search. Superseded by `tune_one.py` + the driver **because it dies with the process** on a faulting candidate. Kept because it is the shorter thing to read first. |
| `artifacts/analysis/summarize_tune.py`, `gen_config.py` | best-N per shape out of the search JSONL, then search output → aiter CSV, carrying the skip rules and their reasons |
| `artifacts/analysis/validate_config.py` | **the accept/reject gate.** Interleaved tuned-vs-torch re-timing through the real `tgemm.mm`, medians over 5 rounds, nonzero exit on any failing row. This is what killed four of twelve winners. |
| `artifacts/analysis/check_fuse.py` | prices the router + shared-`gate_up` fusion with the two `.contiguous()` calls included |
| `artifacts/analysis/trace_shapes.py`, `trace_summary.py` | correlate GPU kernels back to the launching aten op and its shapes; per-kernel and per-bucket GPU time out of a trace |
| `artifacts/analysis/asm_name.py`, `micro_*.py` | small single-purpose probes: recovering asm kernel symbols, graph-replay timing harnesses, an N-padding sweep, per-kernel profiling of `tgemm.mm` |
| `artifacts/analysis/gemm_tune/*.jsonl` | **the raw search output**, 18 files = 6 shape families × M ∈ {64, 8192, 16384} — the five real dense shapes plus `fused1024`, the N=1024 fusion candidate from attempt 2. Each begins with what the stack does today (`kind: current`) and the candidate count (`ncands`, 2091 at M=64 N=4608 K=8192), then one record per candidate. This is the ranked search space itself — the evidence that the space was covered, and the thing you would otherwise spend hours regenerating. |
| `artifacts/analysis/qwen3_8_2_4t_a95b_bf16_tuned_gemm.csv`, `qwen3_8_2_4t_a95b_mxfp4_tuned_fmoe.csv` | the two tables that were built, installed, verified live, gated and **removed**. Present as evidence. **Not a deploy target.** |
