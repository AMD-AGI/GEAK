# Kimi-K3 on 8× MI355X — SGLang 0.5.15.post1 (K3 build), TP=8, five stacked code-lane patches

**Measured win: +18.19% output throughput** (803.786 → 949.964 tok/s), gsm8k 5-shot strict-match
0.9765 ± 0.0042 → 0.9788 ± 0.0040. The win is carried by **five patch files applied as one stack** —
four source changes to SGLang plus one new tuned-GEMM CSV inside aiter — with **no change to the
server flags, the environment or the workload.**

Found 2026-08-20 over a single day's run: nine server instances, **28 counted benchmark rounds**
(6 + 5 + 10 + 7 across the four arms), one round excluded with cause, and one live confirmation after
the set closed.

> **Reproduction status: not yet reproduced on a clean instance from the exported artifact alone.**
> This does not meet the house bar in [`../README.md`](../README.md). Read the next section before
> you read anything else.

## What the confirmation run does and does not establish

After the measurement set was closed, one more benchmark was run against the still-live arm-D server
(server 9, up for over an hour): **950.495 tok/s**, 0.06% from the reported 949.964
(`results/candD_s9_r5_confirm_20260820_174228`, TTFT 16479 ms, TPOT 51.28 ms). It is deliberately
*not* folded into the headline, which stays the pre-registered n=7.

| it establishes | it does not establish |
| --- | --- |
| The gain is live and stable in a long-lived process — no thermal or memory-fragmentation decay after an hour of serving | That a **fresh apply** of the five patches reproduces it. Nothing was re-applied; the tree was already patched |
| The reported 949.964 was not an artifact of a narrow measurement window | That the **exported diffs** are sufficient. They were checked with `git apply --check` forward and in reverse against the assembled tree, but never applied to a clean checkout and measured |
| Arm D's own restart spread is tight — two servers agreeing to 0.003% on their means | Anything about a **different node**. All 9 instances ran on one machine, and the node identity is **not recorded** in the run report |

**What is missing for promotion** is one clean-instance run: check out the base commits, apply
`artifacts/01`…`artifacts/05`, delete `/tmp/aiter_configs/bf16_tuned_gemm.csv`, launch, and measure
its own three-round-plus-restart spread. That is roughly two hours, most of it weight loading. Until
someone does it, treat the number as a strong single-machine result rather than a portable one.

## What the +18.19% is the sum of

This matters more here than in any other entry in this directory, because five patches were measured
in **three** arms, not five. Nothing below is a per-patch measurement unless it says so.

| arm | patches applied | tok/s | n (servers) | vs baseline | vs previous arm |
| --- | --- | --: | --- | --: | --: |
| **A** | none — frozen baseline | **803.786** | 6 (2) | — | — |
| **B** | `01` | **889.683** | 5 (2) | +10.69% | +10.69% |
| **C** | `01` + `02` | **936.952** | 10 (3) | +16.57% | +5.31% |
| **D** | `01` + `02` + `03` + `04` + `05` | **949.964** | 7 (2) | **+18.19%** | +1.39% |

So the attribution that the data supports is:

| patch | measured how | credited |
| --- | --- | --: |
| `01-codelane-mla-decode.diff` | **in isolation, end to end** (arm B over arm A) | **+10.69%** |
| `02-attn-residual-triton.diff` | **in isolation, end to end** (arm C over arm B, stacked on 01) | **+5.31%** |
| `03-kda-packed-decode-warps16.diff` | **bundled**, never measured alone end to end | jointly +1.39% |
| `04-fused-front-gemm-via-aiter.diff` | **bundled** — and inert without `05` | jointly +1.39% |
| `05-aiter-kimik3-fused-tuned-gemm.diff` | **bundled** — and inert without `04` | jointly +1.39% |

**Why 03/04/05 were bundled, and why that was the honest choice.** In isolation patch 03 moves
0.23 ms of a 36 ms decode step, about 0.6%, which is *below* this machine's 0.713% restart-to-restart
noise floor. Measuring it alone would have produced a number that could not be defended. It was
therefore bundled with 04+05 into one arm that clears the floor. The arm-D decode profile
*attributes* 0.73 ms/step to the GEMM change and 0.23 ms/step to the warp-count change, but the run
was explicit that **this is an attribution, not a measurement**, and refused to split the +1.39%
between them on that basis.

**The floor ratio the run chose to quote.** Arm D over arm C is +1.39%, which is **1.95× the 0.713%
restart floor** — and that is the ratio the run put in its own headline, rather than the far more
flattering ratio available on the +18.19% total. Preserve that when you cite this entry. The two
larger increments (+10.69% and +5.31%) are well clear of the floor; the last one is not comfortable,
it is merely defensible, and it is defensible only because of the disjointness and the t-statistic
recorded below.

**Against the reference document rather than the local baseline.** The bundle's reference figure is
804.190 tok/s and this machine's own baseline came out 0.05% under it, so quoted against the
document the same arms read +10.63%, +16.51% and **+18.13%**. Every other number in this entry is
against the local 803.786.

**One open question this closed.** The source session had separately measured a +14.18%
configuration win (`--max-running-requests 64`, which *is* this baseline) and a +12.10% code win on
the *untuned* configuration, and never combined them. They compose: patch `01` is that code win, and
it reproduced at **+10.69% on top of the tuned configuration** — about seven eighths of its
standalone figure, the rest given up to overlap. Do not expect 1.1418 × 1.1210.

## Environment fingerprint

Diff this table before deploying. Every "yes" below is either a literal column in a lookup key or a
determinant of the shapes the patches were tuned for; a mismatch there means the artifact silently
does nothing or does something else.

| field | value | load-bearing? | why |
| --- | --- | --- | --- |
| GPU | 8× MI355X, `gfx950`, **256 CU**, CDNA4 | **yes** | `artifacts/kimik3fused_bf16_tuned_gemm.csv` rows are keyed literally `gfx950,256,M,N,K`. Separately, both launch-geometry wins (`02`, `03`) are wave64 reasoning — they are *wrong* on a 32-lane-warp part |
| container image | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang-rocm-k3:rocm720-mi35x-k3-20260727-tl312-08011830` | **yes**, unusually so | This is the only image on the cluster that can serve Kimi-K3 at all (see below). Pinned in `artifacts/scripts/start_container.sh` |
| container **digest** | **not recorded** | — | Only the tag was ever captured; the source session's own config has `docker_image: null`. A tag can be re-pushed, so anyone matching this environment cannot prove they have the same bytes. This is the largest fingerprint gap in the entry |
| SGLang | `0.5.15.post1.dev20260723+g6c9fd0adc5`, git **`0e756912eb3cd8f531f95c17299485b9312b2534`**, editable checkout at `/sgl-workspace/sglang` | **yes** | Patches `01`–`04` are source diffs against that commit. They were verified to apply cleanly, forward and reverse, against exactly it |
| aiter | git **`68e42f5f461556596ae294200f1a3f13378c8582`**, source checkout at `/sgl-workspace/aiter` | **yes** | Patch `05` adds a file under `aiter/configs/model_configs/`, and both the 18-column CSV schema and the `/tmp` merge behaviour are properties of this tree. The aiter *version string* is **not recorded** — only the commit |
| torch / Triton / ROCm / driver | `2.9.1+rocm7.2.0.git7e1940d4` / `3.6.0` (image reports `3.6.0+git42270451`) / `7.2.0` / `6.14.14` | descriptive | Nothing here keys on them, but a Triton bump is exactly what would change patch `02`'s codegen. A sibling Gemma run lost 4.4% of its baseline to an unnoticed Triton bump, so record yours |
| tilelang | `0.1.7.post3` | descriptive | Present in the image; nothing in this result touches it |
| model | Kimi-K3, `KimiK3ForConditionalGeneration`, **93 layers — 69 KDA + 24 MLA**, **TP=8** | **yes** | TP=8 sets every N and K, including the `6016 = 1536 + 896 + 3584` merged front width that patch `05` tunes. The 69/24 layer split is why the profile's call counts are 69, 24 and 186 (= 2 × 93) |
| quantization | resolved `quant=compressed-tensors`; routed experts **MXFP4**, dense linears **bf16**; `dtype='bfloat16'` | **yes** | Patch `05` is a *bf16* GEMM table (`bf16_tuned_gemm`). On an fp8 or fp4 dense path it is unreachable |
| KV cache | `kv_cache_dtype='auto'` → resolved **`torch.bfloat16`**, pool `#tokens: 922585`, 23.76 GB/rank | **yes** | The pool size is part of the frozen configuration's fingerprint and a smaller pool queues at concurrency 64 |
| backends | attention `triton`, linear-attn `triton`, mamba `triton`, MoE runner `aiter`, sampling `pytorch`, decode CUDA graph `full` @ max_bs 256, prefill graph `disabled` | **yes** | `--attention-backend triton` is what puts patch `01`'s kernels on the decode path at all. The decode graph is why every change needs a restart |
| process environment | 8 exported variables, listed below | **yes** | One of them, `SGLANG_AITER_K3_OPT`, is worth 54.9 GB/rank of weight footprint *and* selects a code path |

**Where the resolved values came from.** `dtype`, `kv_cache_dtype`, `quantization`,
`attention_backend`, `moe_runner_backend`, the CUDA-graph configuration and the 194.38 GB/rank
footprint were all read out of the archived arm-D server log
(`analysis/serverlogs/candD_s9.log.gz`), not assumed from the flags.

**One label that disagrees with reality, and it is a trap.** The version string
`0.5.15.post1.dev20260723+g6c9fd0adc5` shares its prefix with the *released* SGLang 0.5.15.post1,
which **cannot serve this model at all**: no `sglang.srt.configs.kimi_k3`, no
`KimiK3ForConditionalGeneration` in the registry, and no counterpart for 20 of the checkpoint's
weight groups. A version check alone passes on the wrong tree. `artifacts/scripts/preflight.sh`
asserts the registry entry directly, which turns a 25-minute failed weight load into a two-second
failure. All four `0.5.15.post1` tags cached on that cluster were checked; none can serve the model.
Moving to 0.5.17 is not a fix either — it serves the model but is a different framework version from
the one 804.190 tok/s was measured on, so nothing measured there is quotable against it.

**Environment fields the run did not capture, and why each is a gap:**

- **The container digest** (above) — a tag is not a build.
- **The measured servers' actual `/proc/<pid>/environ`.** The launch script *exports* eight
  variables and the startup fingerprint proves that `SGLANG_AITER_K3_OPT` took effect (194.38 GB/rank
  rather than 249.29), but no environment dump was taken, so nothing rules out an extra ambient
  variable inherited from the session's shell. This directory's cross-cutting lesson is that
  `cat /proc/<pid>/environ | tr '\0' '\n'` is the only ground truth; do that on your own server and
  diff it against the eight below.
- **`AITER_SITUV2_A8W4`'s effect is unverified.** It is worth 0.09 GB of footprint, so the
  fingerprint check cannot prove it took effect. If you go looking at MoE kernels, mind that.
- **The node.** The run reports "8× AMD Instinct MI355X" but not a hostname, and the noise floor is a
  property of the node.

## Launch configuration

Verbatim from `artifacts/scripts/launch_server.sh`, which is the script every one of the nine servers
was started by. No server was launched by hand.

```bash
export SGLANG_USE_AITER=1
export SGLANG_AITER_K3_OPT=1
export AITER_SITUV2_A8W4=1
export SGLANG_MOE_PADDING=1
export SGLANG_ROCM_FUSED_DECODE_MLA=1
export SGLANG_AITER_MLA_PERSIST=1
export AITER_FLYDSL_FORCE=1
export HSA_NO_SCRATCH_RECLAIM=1

python3 -m sglang.launch_server \
    --model-path /shared_nfs/hyperloom/models/Kimi-K3 \
    --host 0.0.0.0 --port 43113 \
    --tp-size 8 \
    --context-length 11264 \
    --watchdog-timeout 1800 \
    --attention-backend triton \
    --dtype bfloat16 \
    --cuda-graph-max-bs 256 \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    --trust-remote-code \
    --moe-runner-backend aiter \
    --mem-fraction-static 0.8 \
    --chunked-prefill-size 16384 \
    --disable-radix-cache \
    --max-running-requests 64
```

**"No env recipe" is emphatically not the case here, and assuming otherwise cost the source session
most of a working day.** The reference round's own `config.yaml` records exactly one variable,
`SGLANG_USE_AITER=1`. That record is incomplete. A lane whose subprocesses inherited none of the
ambient environment produced servers that OOMed at the sealed `--mem-fraction-static 0.8` with
`max_mamba_cache_size=-225`, reported it as a host fault ("weight footprint drifted 194.38 → 249.29
GB/rank"), and had every throughput number it produced discarded. The mechanism is exact:
`mxfp4.py:148,440` use `SGLANG_AITER_K3_OPT` to pick a 128-byte routed-expert intermediate alignment
instead of 256, and at TP=8 this model's `intermediate_size_per_partition` is 3072/8 = 384, which
256-alignment rounds up to 512 — +33% routed-expert weight bytes, which over 90 MoE layers and 896
experts is **+54.93 GiB/rank**, closing the observed 54.91 GB discrepancy to 0.04%. `kimi_k3.py:121`
branches on the same variable, so it is a code-path selection and not only a footprint.

### Resolved values not visible in the invocation

Read out of the archived server log, not assumed: `page_size=1`, `max_prefill_tokens=16384`,
`mem_fraction_static=0.8`, decode graph batch-size list `[1, 2, 4, 8, 12, 16, …, 248, 256]`, prefill
graph `disabled`, `speculative_algorithm=None`, `sampling_backend='pytorch'`,
`disable_custom_all_reduce=False`. Weights land at **194.38 GB/rank**; the archived arm-D start shows
weight load taking 136–148 s per rank with a warm page cache, and the pre-handover validation run
reached `/health` 274 s after launch. The run report advises allowing ~19 minutes; the launch script
allows 5400 s because a first start also JIT-compiles aiter kernels.

### The startup fingerprint, and why each of its three checks earns its place

`launch_server.sh` greps the server log for three values that the reference's eight starts agree on
to the byte, and refuses to bless a server that misses any of them. All nine servers in this run
printed `[server] fingerprint matches the reference (194.38 GB/rank, mamba 64, pool 922585)`
followed by `config verified`.

| check | value | what a miss means |
| --- | --- | --- |
| `mem usage=` at Load weight end | **194.38 GB**/rank | 249.29 GB means the environment above was lost. The server still starts and still serves correctly — at a different footprint, on a different code path, with a smaller KV pool |
| `max_mamba_cache_size` | **64** | The explicit `--max-running-requests` branch was taken. The auto-fit lands on 244 and spends ~9.5 GB/rank of hybrid state cache on slots this workload never uses |
| KV pool `#tokens` | **922585** (23.76 GB) | A pool near 556,885 is the pre-baseline sizing and will queue at concurrency 64, "regressing" for reasons unrelated to whatever you are testing |

It then re-reads the live configuration from `/get_server_info` and asserts `context_length=11264`,
`tp_size=8`, `attention_backend='triton'`, `max_running_requests=64`, `cuda_graph_max_bs=256`,
`page_size=1`, `chunked_prefill_size=16384`, `disable_radix_cache=True`, and `mem_fraction_static` of
either 0.8 or 0.68 — SGLang rescales it by 0.85 on builds that combine aiter with a context length
above 8192, so both values are legitimate and a check that accepted only one would fail on a healthy
server.

**`--disable-radix-cache` is not a detail.** With `random_range_ratio 1.0` every prompt is unique, so
prefix caching buys nothing and its bookkeeping is pure overhead. If you turn it on, that is a change
to measure, not a correction to make.

## Workload

Verbatim from `artifacts/scripts/run_bench.sh`, unchanged for every one of the 28 counted rounds:

```
ISL 8192   OSL 1024   concurrency 64   num_prompts 192   num_warmups 8
random dataset, random_range_ratio 1.0, random_prefix_len 0, ignore_eos, seed 0
InferenceX benchmark_serving fork, --backend vllm against /v1/completions
```

Warmups are **8** here, matching every entry in this directory except DeepSeek. A round takes about
four minutes at baseline and about 3.5 at arm D.

**Which workload parameters set the shapes that were tuned:**

- **Concurrency 64 gives decode batch size 63.** That single number drives three of the five patches.
  It is the M the front GEMM is actually called at, and the tuned row that ends up serving it is the
  `M=64` bucket — verified rather than assumed, since the profile at batch 63 shows `MT48x64x256`,
  which is the tuner's `M=64` winner. It is the `B=63` in the KDA grid `B × HV = 63 × 12 = 756` CTAs
  over 256 CUs that motivated the warp-count change. And it is why the two purpose-built tiny-GEMM
  kernels are *never* reached, since their dispatch limits are 16 and 12 tokens. Change concurrency and
  you change which of these fires.
- **ISL 8192 with `--chunked-prefill-size 16384`** sets the prefill tile sizes at which patch `02`
  was swept (T = 2048 and 16384 were both measured, at 1.49–2.03×).
- **`--max-running-requests 64` matching the client's concurrency** is the entire configuration
  baseline: it is worth +14.18% over stock, all of it in TTFT (21101.9 → 17882.2 ms), with per-token
  decode 1.5% *worse*. That is already banked in the 803.786 and is not rediscoverable.
- **`--ignore-eos` with OSL 1024** makes every request generate exactly 1024 tokens, so the decode
  batch stays pinned at the top rather than draining.

## Baseline and noise floor

### The arms, every counted run

| | value |
| --- | --- |
| stock, this stack (6 rounds, 2 servers) | **803.786 tok/s** |
| with the five-patch stack (7 settled rounds, 2 servers) | **949.964 tok/s** |
| delta | **+18.19%** |

Per-run output throughput, from `results/*/inferencex_result.json`:

| arm | server | rounds (tok/s) | server mean |
| --- | --- | --- | --: |
| A | s1 | 802.252, 803.407, 800.996 | 802.218 |
| A | s4 | 803.682, 806.730, 805.651 | 805.354 |
| B | s2 | 886.642, 890.572, 891.963 | — |
| B | s5 | 890.300, 888.936 | — |
| C | s3 | 938.305, 937.082, 938.301 | 937.896 |
| C | s6 | 936.663, 935.030, 934.774 | 935.488 |
| C | s8 | 936.970, 937.693, 936.223, 938.491 | 937.343 |
| D | s7 | *(892.556 — cold, excluded)*, 951.931, 951.560, 946.349 | 949.947 |
| D | s9 | 952.302, 946.354, 951.589, 949.667 | 949.978 |

Arm B's per-server means were not reported in the run document, so they are left blank rather than
recomputed here; its two servers' rounds interleave completely, which is the relevant property.

A separate pre-existing baseline round, `coordinator_validate_20260820_115227`, measured 804.043 —
that is the pre-handover validation of the bundle against the reference 804.190, 0.018% apart, and it
is not one of the six counted baseline rounds.

Two directories in `results/` hold aborted rounds with no result file
(`cand_s2_r3_20260820_124850`, `candC_s8_r3_20260820_165416`). Both were killed by the driving
shell's own ten-minute wall clock partway through the 192-prompt phase, not by anything on the
server, and both were re-run immediately on the same unrestarted server. They are kept so the run
sequence reads honestly; they are not counted.

### The floor

| noise floor | spread |
| --- | --- |
| repeating the benchmark within one process | **0.300%** (server 1, 3 rounds) and **0.379%** (server 4, 3 rounds) |
| **across restarts, unmodified baselines** | **0.713%** (6 rounds, 2 servers; server means 802.218 and 805.354, 0.39% apart) |

**The restart floor is the one that applies, to every claim in this entry.** Every one of these five
patches changes code or a config table consumed at startup, and the decode path is HIP-graph
captured, so nothing can be dropped in live —
[`../../tuning-core/measurement.md`](../../tuning-core/measurement.md) Rule 3b. The within-process
0.300% figure is only useful for judging whether a single server has settled.

**Are the arms disjoint? Yes, completely, on settled runs.** Ordering all counted rounds: baseline
tops out at 806.730, arm B spans 886.642–891.963, arm C spans 934.774–938.491, and settled arm D
spans 946.349–952.302. No two arms overlap anywhere, and the tightest gap — the one that matters — is
that the **lowest settled arm-D round (946.349) sits 0.84% above the highest of the ten arm-C rounds
(938.491)**. Welch's t on arm C (n=10, sd 1.318) against settled arm D (n=7, sd 2.606) is **t = 12.2**
on a 13.01 tok/s difference. The effect is small; the separation is not marginal.

**Arms were interleaved across restarts**, not run in blocks
([`../../tuning-core/measurement.md`](../../tuning-core/measurement.md) Rule 6b). The order was
A(s1), B(s2), C(s3), A(s4), B(s5), C(s6), **D(s7), C(s8), D(s9)** — note that the arm-C control on
server 8 was deliberately run *between* the two arm-D servers in time. That is what makes the next
section a statement about clocks rather than a confound.

### The cold first run, and the decision rule around it

`candD_s7_r1` came in at **892.556 tok/s**, 6.0% below arm D's own mean, and it is the only round in
the whole experiment that sits outside its arm's spread. It is excluded. The case for excluding it:

- **It is not a property of arm D.** The same code on server 9 opened at 952.302, and the two arm-D
  server means agree to **0.003%**.
- **The whole latency distribution shifted, rather than the tail stalling** — p99 ITL 42.11 ms in that
  round against arm D's steady 38.00 ms, about 11%.
- **There is an independent physical cause.** Server 7 is the only server in the experiment whose
  first benchmark followed a long idle: the aiter tuner finished at 15:29, a server start failed on a
  port race, and the next compute the GPUs saw was this round at 16:10 — 40 minutes later, with only
  weight loading in between. Every other `r1` followed another arm's benchmarks by minutes. This is
  the clock ramp that [`../../tuning-core/clocks_and_power.md`](../../tuning-core/clocks_and_power.md)
  documents: ~13% slow at session start, flattening around round 7, **not** removed by warmup, and
  `rocm-smi --setperfdeterminism` is inert in this container (exit 0, perf level still `auto`).

**Both numbers are stated so the choice is visible: including the cold round, arm D is 942.788, a
+0.62% over arm C that would be below the floor and therefore unclaimable.** Half of the decision rule
was fixed in advance — a delta under 0.713% is not claimed. The other half, the exclusion criterion,
was written down *after* seeing the cold round. What makes it defensible is that both of its
conditions were met by evidence collected independently of the throughput number: the within-round
ITL shift, and server 9 agreeing with server 7 to 0.003%.

**The practical rule: do not benchmark a server whose GPUs have idled for tens of minutes without a
throwaway round first.** On this stack that is worth 6%, which is larger than most wins anyone is
looking for.

## The artifacts

Everything needed to deploy is under `artifacts/`. Each patch file carries its own header block with
its base commit, its apply command and its measurement, so the patches survive this document.

| file | target repo | touches | kernel-level result |
| --- | --- | --- | --- |
| `artifacts/01-codelane-mla-decode.diff` | sglang @ `0e756912e` | `kernels/ops/attention/decode_attention.py` | **2.6876×** on the kernel, as measured by the source lane on its own stack |
| `artifacts/02-attn-residual-triton.diff` | sglang @ `0e756912e` | `srt/layers/attn_residual.py` | **1.83×** on the pair at the decode shape; 1.49–2.03× at prefill tiles |
| `artifacts/03-kda-packed-decode-warps16.diff` | sglang @ `0e756912e` | `kernels/ops/attention/kda_packed_decode.py` | 31.0 → 28.3 µs, **1.095×**, output and state bitwise identical |
| `artifacts/04-fused-front-gemm-via-aiter.diff` | sglang @ `0e756912e` | `srt/models/kimi_k3.py` | routes one GEMM; the win is in `05` |
| `artifacts/05-aiter-kimik3-fused-tuned-gemm.diff` | **aiter** @ `68e42f5f4` | new `aiter/configs/model_configs/kimik3fused_bf16_tuned_gemm.csv` | front GEMM 41.52 → 30.04 µs, **1.382×** |

`artifacts/kimik3fused_bf16_tuned_gemm.csv` is the same 22-row table as a bare file, if you would
rather drop it in than apply a patch. `artifacts/gemm_tune/` holds the tuner's **input**
(`untuned_k3_fused.csv`) as well as its output, so the table is regenerable rather than a mystery
artifact.

The rest of the inventory, all byte-identical to the bundle:

| path | what it is |
| --- | --- |
| `artifacts/kimik3fused_bf16_tuned_gemm.csv` | the tuned table as a bare drop-in; identical to `gemm_tune/tuned_k3_fused.csv` and to what patch `05` adds |
| `artifacts/gemm_tune/untuned_k3_fused.csv` | the tuner **input** — 22 rows of `M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle`, harvested from the profile |
| `artifacts/gemm_tune/bench_tiny_gemm.txt` | raw output of the tiny-GEMM crossover sweep |
| `artifacts/rejected/attn-residual-fixed-block-r.diff` | the rejected `BLOCK_R = 8` variant, which applies on top of `02` and puts the constant back |
| `artifacts/scripts/launch_server.sh` | the frozen launch, with the environment exports and the startup fingerprint check |
| `artifacts/scripts/run_bench.sh` | the frozen workload |
| `artifacts/scripts/run_eval.sh` | the gsm8k gate, including the two fixes without which it scores 0.03 |
| `artifacts/scripts/preflight.sh` | asserts the stack *and* that `KimiK3ForConditionalGeneration` is in the model registry, which is what catches a wrong image in seconds |
| `artifacts/scripts/start_container.sh` | the exact image tag |
| `artifacts/analysis/profile_server.py` | makes the decode profile; no profile of this model existed before |
| `artifacts/analysis/bench_attn_res.py`, `attn_res_cand.py`, `check_attn_res.py` | patch `02`'s A/B sweep, its standalone candidate kernels, and its fp64-oracle correctness check |
| `artifacts/analysis/bench_kda.py` | builds one JIT KDA module per `kWarps` and races them, checking output *and* state against `kWarps=8` |
| `artifacts/analysis/bench_k3_gemm.py` | times each `_k3_bf16_gemm` call site three ways in one process so the clocks are shared |
| `artifacts/analysis/bench_tiny_gemm.py` | sweeps the `max_m` template parameter to locate the dispatch crossover |
| `artifacts/analysis/trace_gaps.py`, `summarize.py` | ranks idle gaps between kernels; tabulates every result file with n / mean / spread |
| `artifacts/analysis/triton_amd_ptr_if_bug.md` | the `CanonicalizePointers` assert, its trigger and its workarounds |
| `artifacts/profiles/*_kernel_table.txt` | the three arms' ranked kernel tables — the engagement reference |

### `01` — the MLA split-K decode rewrite (+10.69%)

Not this campaign's work: it is `reference/codelane/final_patch.diff` carried forward verbatim to
answer whether the code lane composes with the tuned configuration. It rewrites the Triton
grouped-MLA split-K decode pair — bf16 `mid_o` instead of fp32, split target floored to whole waves,
retuned stage 2 (`BLOCK_S=16`, 1 stage, 1 warp), natural-K gather with `.cg`/`evict_first` hints, and
manual one-tile-ahead pipelining. `git apply` was clean on this tree, so the "may not apply cleanly"
hazard flagged in the bundle did not materialise.

Almost all of its gain is decode: mean TPOT 62.22 → 54.56 ms (−12.3%) with mean TTFT essentially flat
(17874 → 17837 ms).

**On the kernel-level number, read the framing carefully.** The source lane measured **2.6876× on the
kernel alone** and recorded the pair at 15.35% of GPU time on *its* configuration. In this run's arm-B
profile the pair sits at 6.57% + 0.33% of decode GPU time. Those two figures are from different
configurations and different profiling windows, so **6.90% against 15.35% is not a before/after** —
treat the 15.35% as the source lane's estimate of the opportunity and the 6.90% as where the pair
landed here.

**Engagement on this patch is the textbook false negative.** Its own `[overlay]` stderr marker fires on
all 8 ranks at startup and then stops incrementing, because the Python wrapper does not execute during
graph replay — exactly what
[`../../tuning-core/engagement_verification.md`](../../tuning-core/engagement_verification.md) warns
about. The evidence that counts is kernel identity in a profile.

### `02` — the attention-residual Triton pair (+5.31%)

This is the transferable one; the mechanism gets its own section below. `_score_kernel` and
`_combine_kernel` were **6.48% of arm-B decode GPU time** across 186 calls each — two aggregation
points on each of 93 layers — and `_score_kernel` was moving 0.9 MB in 8.3 µs, about **110 GB/s on a
part that does several TB/s**. That is not a bandwidth number, which is what said something structural
was wrong rather than something tunable.

Three changes:

- **`_score_kernel`: hoist the two cross-lane reductions out of the H loop.** The stock body did
  `sumsq += tl.sum(v*v); dotv += tl.sum(v*cw)` per H block — 14 block-wide reductions per CTA at
  H=7168, each a barrier plus an LDS round trip — where two per-lane accumulators and one reduce each
  at the end suffice.
- **`_combine_kernel`: mix the bank rows as one `[BLOCK_R, BLOCK_H]` tile.** The stock loop paid a
  cross-lane reduce per row purely to turn `p[j]` into a scalar; as a tile the weight is a broadcast
  and the mix is one `tl.sum`. `BLOCK_R` is sized to `nvb` by the caller — see the rejected variant
  below for why that matters.
- **Launch configuration, ROCm only: one wave64 per CTA for both kernels** (was 8 and 4 warps), and
  combine tile 512 (was 1024). In the isolated sweep this is *the larger half* of the 1.83×.

Isolated, graph-captured with a `dirty`/`verify` guard and `mode == "cudagraph"` asserted on every
result: **1.83× on the pair at the decode shape** (T=64, summed over nvb ∈ {1,4,7}: 29.5 → 16.1 µs)
and 1.49–2.03× at the prefill tiles. The 1-warp optimum held at every shape swept.

In-server, over the same 186-call window: `_score_kernel` 1.54 → 0.78 ms, the pair 6.48% → 4.42% of
decode GPU time, total decode-step GPU time 36.86 → 36.18 ms. Both phases improve because the pair
runs in prefill too, and the win is larger on the big tiles: TTFT 17837 → 16480 ms (−7.6%), TPOT
54.56 → 52.24 ms (−4.3%).

Correctness: `artifacts/analysis/check_attn_res.py` checks every nvb 0…8 × T ∈ {1, 7, 64, 129, 2048,
16384} against an fp64 oracle. Worst error **1.561e-2 — which is the same value the stock kernels
produce on the same shapes**, checked by loading the pristine file from git alongside. About 2 bf16
ulp at the output scale. The rewrite reassociates the sums without losing accuracy.

### `03` — KDA packed decode, 8 warps → 16 (bundled)

One constant. `kda_packed_decode_kernel<8,false>` was 5.93% of arm-C decode (69 calls, one per KDA
layer, ~31 µs each) moving 99 MB of fp32 state per call — about 3.2 TB/s on a part whose HBM does
~8. The kernel's own header describes it as a row-streaming design that should hit the read+write
bandwidth of the part, so the gap is latency hiding, not bandwidth: at `kWarps=8` the block is 256
threads = 4 wave64s and the grid is only B × HV = 63 × 12 = 756 CTAs over 256 CUs, roughly 3 CTAs per
CU, which is thin for a pure streaming loop.

Isolated: 31.0 → 28.3 µs, **1.095×**, with output and updated state **bitwise identical** to
`kWarps=8`. 2 and 4 warps are worse; 32 regresses. `_WARPS` is a JIT template argument, so the
compiled module name changes with it and a stale build cannot be silently reused — a rare case where
the deploy cannot no-op on you.

### `04` + `05` — the GEMM shapes SGLang synthesises (bundled)

Six `Cijk_*` (hipBLASLt/Tensile) kernels were **14.4% of arm-C decode**, and nothing had ever tuned
them. Every *other* dense bf16 linear on this stack goes through `aiter.tuned_gemm`, because
`SGLANG_USE_AITER=1` makes `UnquantizedLinearMethod` dispatch to `tgemm.mm`. These do not, for a
reason that is in the tree rather than in the configuration:

- `models/kimi_k3.py::_k3_bf16_gemm` calls `F.linear`/`torch.mm` **directly, by design** — the fused
  MoE front and the deferred shared-down GEMM run on raw merged weight *views*, not on `Linear`
  modules, so there is no quant method to dispatch through.
- `kernels/ops/kimi_k3/__init__.py::kimi_k3_tiny_gemm` has purpose-built kernels for the two skinny
  decode GEMVs but gates them on a hard token limit and falls through above it. **The frozen workload
  decodes at batch 63**, so both fall through on every layer of every step.

The identity was *confirmed, not inferred*: `torch.nn.functional.linear` on a
`[63, 7168] × [6016, 7168]ᵀ` bf16 pair emits `Cijk_Alik_Bljk_..._MT160x64x128_..._MIWT5_2_...`,
character-for-character the 92-call, 7.86% kernel in the arm-C profile. And N = 6016 is
`shared gate_up (1536) + router gate (896) + latent down (3584)` — the three-way merge
`_merge_front_weights()` builds at load time.

**Why they were untuned.** The image *does* ship `aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv`
(100 rows, gfx950/cu 256, M 1…1024) covering every K3 *module* linear: 1536/7168, 2112/7168,
2304/1536, 3072/512, 6144/7168, 7168/{768,1536,3584,4224}, 8448/7168. It does not contain 6016×7168 or
144×7168, **because those shapes do not exist in the checkpoint** — SGLang synthesises them at load
time. AMD's tuning run never saw them, and `get_GEMM_A16W16_config` falls back to `libtype: torch` on
a miss.

Measured headroom before the tune, at M=63, graph-captured and clock-warmed
(`artifacts/analysis/bench_k3_gemm.py`):

| call site | N × K | torch (today) | aiter tuned pick | Triton |
| --- | --- | --: | --: | --: |
| fused MoE front | 6016 × 7168 | 41.96 µs | — (miss → torch) | **33.90 µs, 1.24×** |
| shared down | 7168 × 768 | 15.30 µs | 18.72 µs (flydsl) **0.82×** | 14.80 µs |
| merged `[f_a\|b]` | 144 × 7168 | 17.66 µs | — (miss → torch) | 28.52 µs |

Re-run with the tuned CSV in place, in one process so the clocks are shared:

| call site | torch | `tgemm.mm` | ratio |
| --- | --: | --: | --: |
| fused MoE front, 6016 × 7168 | 41.52 µs | **30.04 µs** | **1.382×** |
| shared down, 7168 × 768 | 15.08 µs | 18.26 µs | 0.826× |
| merged `[f_a\|b]`, 144 × 7168 | 17.36 µs | 16.52 µs | 1.051× |

**1.382× beats the 1.24× Triton offered, which is the argument for tuning rather than just switching
backend.** The tune itself used aiter's own multi-backend tuner, unmodified, over both missing shapes
at M ∈ {1,2,4,8,16,32,64,128,256,512,1024} — 22 rows, ~25 minutes on 8 idle MI355X, searching
asm/opus/flydsl/triton/skinny/torch plus roughly 2000 hipBLASLt solutions per shape:

```bash
cd /sgl-workspace/aiter && python3 csrc/gemm_a16w16/gemm_a16w16_tune.py \
  -i artifacts/gemm_tune/untuned_k3_fused.csv \
  -o artifacts/gemm_tune/tuned_k3_fused.csv --libtype all --with-hipblaslt --all
```

At the decode M the winners are hipBLASLt solutions the heuristic does not pick: 6016×7168 → solidx
**438691**, `MT48x64x256`, 21.384 µs; 144×7168 → solidx **440197**, `MT16x16x512`, 8.853 µs. The
6016 shape resolves to **flydsl** at M ≤ 16 and to hipBLASLt from M = 32 up, and the 144 shape picks
`skinny` at M ≤ 4 and plain `torch` at M = 256 — the table is genuinely heterogeneous, which is worth
knowing before assuming one backend wins a shape everywhere.

The shape list came from the profile, not from the config file, per
[`../../tuning-aiter/SKILL.md`](../../tuning-aiter/SKILL.md)'s rule: *the shape list is the tune —
harvest it, never guess it.*

**The code change, `04`, is a gate rather than a reroute.** Inside `_k3_bf16_gemm`, the `out is None`
path calls `tgemm.mm` **only when `_aiter_has_tuned_gemm(m, n, k)` confirms aiter holds a real
(non-`torch`) entry for that exact shape**. A shape nobody tuned keeps its present callee, so the
patch can only act where there is evidence. The `out is not None` path — the deferred shared-down
GEMM — is deliberately left on `torch.mm(out=)`: `tgemm.mm` has no `out=` parameter, and aiter's own
tuned pick for 7168×768 measures **0.826×**, i.e. slower than what is already there. Keeping a slower
"optimised" path would have been the easy mistake.

Effect on the decode profile: the 92 front-GEMM calls move from `MT160x64x128` at 2.84 ms (7.86%) to
`MT48x64x256` at 2.11 ms (6.03%), and total decode-step GPU time goes 36.18 → 35.07 ms (−3.1%).

**A loose end stated rather than hidden: the 11 tuned N=144 rows are correct but unused.** That GEMV
reaches `kimi_k3_tiny_gemm`, not `_k3_bf16_gemm`, and routing its fallback to aiter measures 1.051×,
which is 0.17% of decode and under the floor — so no code change was made for it. The rows stay so
the shape is covered if a future call site does reach it.

## Deploy

**A restart is mandatory.** The decode path is HIP-graph captured; editing a kernel or dropping a CSV
under a live server changes nothing at all, and the resulting benchmark is clean, plausible and wrong.

```bash
# 0. Pristine trees. Both frameworks are editable git checkouts here, so this is exact.
cd /sgl-workspace/sglang
git checkout 0e756912eb3cd8f531f95c17299485b9312b2534 -- python/
cd /sgl-workspace/aiter && git status --porcelain          # expect clean at 68e42f5f4

# 1. The four SGLang patches. Order is free (disjoint files); this order matches the arms.
cd /sgl-workspace/sglang
git apply <kb>/artifacts/01-codelane-mla-decode.diff        # arm B
git apply <kb>/artifacts/02-attn-residual-triton.diff       # arm C
git apply <kb>/artifacts/03-kda-packed-decode-warps16.diff  # arm D
git apply <kb>/artifacts/04-fused-front-gemm-via-aiter.diff # arm D

# 2. The aiter table. MUST be paired with 04.
cd /sgl-workspace/aiter
git apply <kb>/artifacts/05-aiter-kimik3-fused-tuned-gemm.diff
#   equivalently, without git:
#   cp <kb>/artifacts/kimik3fused_bf16_tuned_gemm.csv aiter/configs/model_configs/

# 3. Cache invalidation.
rm -f /tmp/aiter_configs/bf16_tuned_gemm.csv                # MANDATORY - see below
rm -rf ~/.triton/cache                                      # prophylactic
find /sgl-workspace/sglang -name __pycache__ -prune -exec rm -rf {} +   # prophylactic

# 4. Restart, verify, measure.
<kb>/artifacts/scripts/launch_server.sh    # must print the fingerprint line AND 'config verified'
<kb>/artifacts/scripts/run_bench.sh
```

Stop after `02` for arm C and after `01` for arm B. All five were checked with `git apply --check`
both forward from the base commits (individually, against the pristine index) and in reverse against
the assembled arm-D tree, so the stack on disk is exactly base + patches.

### Order and interdependencies, for a five-patch stack

- **Order does not matter.** The five touch disjoint files:
  `kernels/ops/attention/decode_attention.py`, `srt/layers/attn_residual.py`,
  `kernels/ops/attention/kda_packed_decode.py`, `srt/models/kimi_k3.py`, and a new file in aiter.
- **`04` and `05` are a pair, in both directions.** `04` without `05` is a *deliberate* no-op: the
  shape gate finds no tuned entry and leaves the callee alone. `05` without `04` is also a no-op:
  nothing consults the table for those shapes. Deploy both or neither.
- **`02` is independent of `01`** but was only ever measured stacked on it. If you deploy `02` alone
  you are outside this entry's evidence.
- **`03` was never measured alone end to end** and is worth ~0.6%, under the floor. It is safe (bitwise
  identical output) but do not expect to see it in a throughput number by itself.

### On the three cache removals

Only the first is established by this run. `rm -f /tmp/aiter_configs/bf16_tuned_gemm.csv` is
**mandatory and was the run's own biggest near-miss**: aiter merges `configs/bf16_tuned_gemm.csv` with
every `configs/model_configs/*bf16_tuned_gemm*.csv` into that cached path at import time, behind an
`lru_cache` on `get_config_file` (`aiter/jit/core.py`). Adding a CSV without removing the merged file
changes nothing, and **the symptom is indistinguishable from "the tune found nothing"** — you conclude
your 25-minute tuner run was worthless. Patch `05`'s own Apply line carries the `rm` for this reason.

The Triton cache and `__pycache__` removals are carried from this directory's cross-cutting lessons
rather than from anything this run hit; neither was reported as necessary here. They cost seconds and
they each independently serve you the old kernel when they do bite, so do them anyway.

### Every way this deploy silently does nothing

| failure | symptom | how to catch it |
| --- | --- | --- |
| **No restart** | Benchmark runs perfectly at the old speed. Decode graphs were captured at startup | There is no in-process check. Restart, always |
| **Stale `/tmp/aiter_configs/bf16_tuned_gemm.csv`** | `05` is inert; reads exactly like "the tuner found nothing" | The merged file must contain the new rows — see the engagement check |
| **`04` without `05`** | Nothing changes, by design | `_aiter_has_tuned_gemm` returns False; profile still shows `MT160x64x128` |
| **`05` without `04`** | Nothing changes; no call site reaches the table | Same profile check |
| **A different SGLang image with the same version prefix** | The server never starts — but the *first* error you hit is `--reasoning-parser kimi_k3` being rejected, which reads like an optional flag problem rather than a wrong image | `artifacts/scripts/preflight.sh` asserts `KimiK3ForConditionalGeneration` is in the model registry |
| **Any of the eight environment variables missing** | Server starts and serves correctly at 249.29 GB/rank on a different MoE code path, or OOMs outright at `--mem-fraction-static 0.8` | The `mem usage=194.38 GB` fingerprint line |
| **`--max-running-requests 64` dropped** | Mamba cache auto-fits to 244, the token pool shrinks, and the *baseline itself* moves by 14% | `max_mamba_cache_size: 64` and `#tokens: 922585` |
| **Concurrency ≠ 64** | Decode batch is no longer 63, so the `M=64` CSV row, the KDA grid and the tiny-GEMM dispatch behaviour all change together | Nothing warns. Do not change the workload |
| **Patches applied to a path nothing imports** | Silent. Here both frameworks are editable checkouts under `/sgl-workspace`; if yours is a pip install, the diff lands somewhere unused | `preflight.sh` prints `sglang at =` and the two git shas |
| **Benchmarking a server whose GPUs idled tens of minutes** | Not a no-op, a false *negative*: reads ~6% low and looks like a regression | Throwaway round first; check p99 ITL against 38.00 ms |
| **Trusting patch `01`'s `[overlay]` marker as a live check** | The marker fires at startup on all 8 ranks and then stops incrementing, because the Python wrapper does not execute during graph replay | Use the profile, not the log |

## Engagement check

**Prefer kernel identity from a profile.** Three of the five patches put their own identity into a
kernel symbol, which makes this the strongest available form
([`../../tuning-core/engagement_verification.md`](../../tuning-core/engagement_verification.md) form
4) and the only one that is safe given the `/tmp` cache hazard above.

```bash
python3 artifacts/analysis/profile_server.py     # drives 64 concurrent ISL-8192 requests,
                                                 # settles into decode, POSTs /start_profile,
                                                 # ranks rank-0 kernels -> kernel_table.txt
grep -E 'kda_packed_decode_kernel<|MT48x64x256|MT160x64x128|_score_kernel' kernel_table.txt
```

| | engaged (arm D) | not engaged (arm C or stock) |
| --- | --- | --- |
| `03` | `kda_packed_decode_kernel<16, false>` — 1.91 ms, 69 calls, 5.44% | `kda_packed_decode_kernel<8, false>` — 2.14 ms, 69 calls, 5.93% |
| `04`+`05` | `Cijk_..._MT48x64x256_...` — 2.11 ms, 92 calls, 6.03% | `Cijk_..._MT160x64x128_...` — 2.84 ms, 92 calls, 7.86% |
| `02` | `_score_kernel` — 0.76–0.78 ms, 186 calls | `_score_kernel` — **1.54 ms**, 186 calls |
| total decode step | 35.07 ms | 36.18 ms (arm C), 36.86 ms (arm B) |

The macro tile in the hipBLASLt symbol *is* the proof for `04`: `MT48x64x256` is the solution the
aiter tuner selected (solidx 438691), and `MT160x64x128` is what the default heuristic picks. Likewise
`<16, false>` is a template argument, so it appears in the mangled name and cannot be faked by a stale
build.

**The negative control that must survive.** Exactly one GEMM call site was rerouted, so the other five
`Cijk_*` symbols must be unchanged — in particular `MT16x16x1024` at 69 calls, 0.82 ms in arm C and
0.81 ms in arm D. Likewise `_combine_kernel` stays at 186 calls and 0.82–0.83 ms across arms C and D.
If those also move, something other than your deploy changed. Reference tables for all three arms are
checked in at `artifacts/profiles/{cand_s2,candC_s3,candD_s7}_kernel_table.txt`.

**A cheap pre-flight check for `05` that needs no profile,** worth running before you spend a server
start:

```bash
grep -c ',6016,7168,' /tmp/aiter_configs/bf16_tuned_gemm.csv
```

Expect **11** (the eleven M buckets for that shape) once the merge has picked the file up, and **0**
if the cache is stale or the CSV is not installed. Note that this follows from the merge behaviour
recorded in the run report rather than from a check the run itself performed — the run verified
engagement by kernel identity. Two caveats on log-based alternatives: aiter's *hit* line needs
`AITER_LOG_TUNED_CONFIG=1`, which is an environment change and therefore outside the frozen
configuration; and patch `01`'s `[overlay]` marker is a startup-only signal, as noted above.

**Two things that had to be right for the profile itself to be meaningful.** No profile of this model
existed before this run. `activities` must include `CPU` — with `["GPU"]` alone kineto keeps only the
ungraphed sampler and reports a 0.25 ms "decode step" — and the load must be settled into decode
before the window opens. `artifacts/analysis/profile_server.py` does both.

## Accuracy gate

gsm8k 5-shot, 1319 problems, **lm-eval 0.4.12** (`lm-eval[api]`, gsm8k task version 3.0),
`--apply_chat_template`, `--num_fewshot 5`, `--seed 0,1234,1234,1234`,
`max_tokens=9216, temperature=0, top_p=1`, via `local-chat-completions` against the running server.
Exact invocation in `artifacts/scripts/run_eval.sh`.

| config | `exact_match,strict-match` | flexible-extract | source |
| --- | --- | --- | --- |
| **gate (this run's baseline, arm A)** | **0.976497 ± 0.004173** | 0.975739 ± 0.004238 | `eval_results/base_s1_20260820_122534` |
| arm B (`01`) | 0.976497 ± 0.004173 | 0.976497 ± 0.004173 | `eval_results/cand_s2_20260820_125449` |
| arm C (`01`+`02`) | 0.978014 ± 0.004039 | 0.977255 ± 0.004107 | `eval_results/candC_s3_20260820_133546` |
| **arm D (all five)** | **0.978772 ± 0.003970** | 0.978014 ± 0.004039 | `eval_results/candD_s7_20260820_163121` |
| source session, stock configuration | 0.978772 ± 0.003970 | 0.978772 ± 0.003970 | `reference/eval/stock_gsm8k/` |

**Threshold: a candidate must land within about one stderr of 0.9765 strict-match.** Arm D is 0.5
stderr *above* it, arm C 0.4 stderr above, arm B identical to six decimal places. **Pass.** Read the
increases as ties, not improvements — nothing in these patches has a path to the answers, and arm B
being bitwise-equal to the gate on strict-match is the clearest sign the harness is stable.

Two honest caveats. The baseline gate was established **by this run** — the source session evaluated
only its stock configuration, and its 0.978772 is what the baseline landed about half a stderr under,
i.e. consistent with it. And **each arm was evaluated exactly once**, so the ± figures are lm-eval's
binomial stderr on 1319 problems, not a measured run-to-run spread. `lm_eval` is not installed in the
image; the venv route in `run_eval.sh` works (`python3 -m venv /tmp/lmeval_venv`, then
`pip install 'lm-eval[api]==0.4.12'`).

**If you get ~0.03 strict-match, your gate is broken, not the model.** lm-eval's default 256-token
generation budget truncates the reasoning so the answer never arrives; on this stack that scores
0.0318 and reads exactly like a destroyed model. `max_tokens=9216` is the served context less room for
the 5-shot prompt. The second required fix is the `sitecustomize` patch in `run_eval.sh`: this server
puts text in `reasoning_content` and leaves `content` empty, and lm-eval 0.4.12 substitutes a
placeholder and merely warns.

## The transferable mechanism: tuned constants stranded behind an architecture predicate

**This is the part of the entry most likely to be useful on a model that is not Kimi-K3.** Patch `02`
is the second-largest single win in the stack (+5.31%) and it was found by reading a predicate, not by
profiling harder.

**The shape of the defect.** A kernel ships two things: a fast path with launch constants somebody
measured, and a predicate deciding who gets it. The predicate is an *architecture* test. On the
architecture that fails the test, the "fallback" is not a fallback — it is the production path, the
only code that ever runs, and nobody has ever profiled it on the hardware that runs it. Nothing logs
anything. The server is fast, correct, and mis-shaped.

Here it is `attn_residual.py::_use_fast()`, which requires
`torch.cuda.get_device_capability()[0] >= 10` — i.e. NVIDIA SM100+. **On CDNA the Triton pair is the
only path that ever executes**, and its launch configuration had plainly been chosen on hardware that
never runs it: 8 and 4 warps per CTA, in kernels where **one wave64 per CTA won at every shape swept
from T=64 to T=16384**. That was worth 1.83× on a pair that was consuming 6.48% of decode GPU time, in
a file whose header documents the fast path in loving detail.

**Why one wave64 is often the answer on CDNA.** Reduction-heavy Triton kernels ported from NVIDIA carry
warp counts that assume 32-lane warps and cheap intra-CTA reductions. On CDNA a single wave reduces
in-register with no barriers and no LDS traffic, and loads 32 B per lane instead of 4.

**But do not turn that into a rule — sweep it.** This run found two wrong launch geometries and they
moved in *opposite* directions: patch `02` went 8/4 warps → 1, and patch `03` went 8 warps → 16,
because there the problem was the reverse — a 756-CTA grid over 256 CUs was too thin to hide latency in
a pure streaming loop. The shared root cause is not "fewer waves"; it is **a launch geometry reasoned
from a different part's warp width and CU count**, which is exactly the class
[`../../tuning-core/arch_migration.md`](../../tuning-core/arch_migration.md) §4 predicts. Which
direction it needs is an empirical question every time.

**Four independent sightings of this class, three of them elsewhere in this directory.** Go looking for
it; do not wait to trip over it.

- [`gpt-oss-120b/`](../gpt-oss-120b/) found `_get_block_sizes_for_extend_attention` carrying a real
  gfx950 tuning gated on `128 < Lq <= 256`, with head_dim 64 falling through to a generic default.
  Worth **2.088× on a prefill pass and +6.30% end to end**, and that entry's own section on the pattern
  is worth reading in full.
- [`gemma-4-26b-a4b-it/`](../gemma-4-26b-a4b-it/) diagnosed the *same predicate in the same function*
  independently, on a different model at head dims 256 and 512 — **+7.20%** there.
- The Qwen3.5-397B-A17B run hit the class from the other side: AMD had tuned that model's dense GEMMs
  for TP=2 while the frozen configuration is TP=4. Same failure mode with a **topology** key instead
  of an architecture key.
- Kimi-K3, here, twice.

**The cheap sweep this justifies, in priority order, on any new model on AMD:**

1. `grep -rn get_device_capability` over the framework's kernel and layer directories, and check what
   each predicate selects on `gfx9xx`. Anything gated on SM90/SM100 is a path you are *not* taking,
   and whatever you *are* taking has probably never been profiled.
2. Grep the hot kernels for `num_warps`, `_WARPS`, `BLOCK_*` and any constant reasoned from CU count,
   register-file size or waves-per-EU — [`../../tuning-core/arch_migration.md`](../../tuning-core/arch_migration.md)
   §4 predicts exactly this class, and it predicted both of this run's constant wins before they were
   found.
3. Sweep those constants *before* attempting any rewrite. It costs a couple of hours. In this run the
   launch-constant half of patch `02` was the larger half of its 1.83×, and patch `03` is a single
   token — **launch constants carried more of this stack's win than the rewrites did.**

**A second, related pattern this run establishes, which is not in the skillset anywhere.** Anything
the framework **fuses or merges at load time** is untuned by construction. The three biggest `Cijk_*`
kernels in the decode profile existed because SGLang synthesises their shapes by concatenating
checkpoint weights: those shapes are in no config file, so the vendor's tuning run never saw them, and
they are not `Linear` modules, so aiter's dispatch never saw them either. Two misses, one root cause.
The sweep is: **grep `load_weights` for `torch.cat`/`narrow` over weights, profile for
`Cijk_*`/`hipblaslt` symbols, harvest those shapes into the tuner.**

## What was tried and did not work

| attempt | kernel-level result | end to end | verdict |
| --- | --- | --- | --- |
| **Fixed `BLOCK_R = 8` in `_combine_kernel`** as a module constant instead of sizing per call (`artifacts/rejected/attn-residual-fixed-block-r.diff`) | **Regression at low nvb: T=2048, nvb=1 55.1 µs stock → 81.3 µs (0.678×); T=16384 393 → 569 µs (0.691×).** The chosen `_pow2_ge(max(nvb,1))` gets 31.5 µs and 225 µs (1.75×) on the same shapes. Wins at nvb ≥ 4 either way | **Never run.** Rejected on the isolated regression before spending a server instance | Rejected. A `[8, BLOCK_H]` fp32 tile is 32 KB of registers per CTA whether or not the rows are masked, and occupancy pays for it. `nvb` is a host value, so specialising per call is free — and after the fix **nvb=1 became the best case at 1.75×** |
| **Hoist `_score_kernel`'s uniform row select into a base pointer above the H loop** | **Does not compile.** `TritonAMDGPUCanonicalizePointers` requires both arms of an `scf.if` yielding a pointer to agree on the `canNarrow` bit, and `prefix [T, H]` and `bank [T, NB, H]` do not. Surfaces in Python as a bare `RuntimeError: PassManager::run failed`, naming neither the construct nor the line | n/a | Rejected, and it costs nothing — the branch stays inside the loop, where the whole win was anyway. Written up in `artifacts/analysis/triton_amd_ptr_if_bug.md` |
| **Raise the tiny-GEMM dispatch limits** so batch 63 reaches the purpose-built kernels instead of `F.linear` (`_K3_N_GEMM_DISPATCH_MAP = {(144,7168): 16, (896,7168): 8}`, `_K3_K_GEMM_DISPATCH_MAP = {(1536,128): 12}`) | **The shipped limits are already correct.** Sweeping `max_m` against torch: `tiny_n` 144×7168 is 1.056× at m=8, **1.112× at m=16**, 1.018× at 24, 0.897× at 32, **0.638× at 63**, 0.345× at 128; `tiny_k` 1536×128 is 1.021× at m=8, 0.997× at 16, 0.788× at 63, 0.569× at 128. Both degrade monotonically in m | **Not run.** Rejected on the sweep | Rejected. "This constant excludes my workload" read like an oversight and was a correctly tuned bound landing within one sweep point of the crossover. Raising the limits to reach batch 63 would cost **36% and 21%**. The residual is real — 3.4% of decode in two GEMVs at a fraction of a TB/s — but the fix is a different kernel, not a different threshold |
| **Route the shared-down GEMM (7168×768) through `tgemm.mm`** | aiter's own tuned pick measures **18.26 µs against torch's 15.08 µs — 0.826×** | not attempted | Rejected. Also blocked mechanically: `tgemm.mm` has no `out=` and this call site stores into a provided buffer. The temptation to reroute everything to the "optimised" backend is exactly how you ship a slowdown |
| **Route the N=144 GEMV fallback through aiter** | **1.051×** (17.36 → 16.52 µs) | 0.17% of decode — under the 0.713% floor | No code change. The 11 tuned rows ship anyway so the shape is covered if a future call site reaches it |
| **`rocm-smi --setperfdeterminism`** to fix the cold-clock ramp | Inert in this container: exit 0, no error, perf level still `auto` | n/a | Dead end, and pre-recorded as one in [`../../tuning-core/clocks_and_power.md`](../../tuning-core/clocks_and_power.md). Knowing that saved an afternoon. Scheduling is the only defence |
| **Tuning hipBLASLt directly** for the `Cijk_*` kernels | not measured | not measured | Read and *correctly not followed*. aiter already owns dispatch for every other bf16 linear on this stack, so a hipBLASLt-level win would live outside the mechanism the rest of the model uses. Going through aiter's tuner got the hipBLASLt solution anyway (`--with-hipblaslt` searches ~2000 of them and picked one) while keeping the result in a table the framework consults |
| **Two enabling patches for SGLang 0.5.17** (`patches/superseded/`) | n/a | n/a | Dropped when the bundle was pinned to the K3 image, where neither is needed. Kept in the bundle because the reasoning matters if that image is ever unavailable |

### The methodology error worth copying down

**The first microbenchmark measured its own harness, and nearly discarded the second-largest win in
the stack.** With one graph-captured invocation per replay, the stock attention-residual pair reported
17.9 µs at nvb=1 and 20.2 µs at nvb=7 — essentially flat against an **8× change in bytes moved**,
because a two-kernel graph replay costs about as much as the kernels do. At that resolution the
reduction hoist looked worthless (0.98× at the decode shape). Capturing **32 back-to-back invocations
per graph** and dividing resolved it: the same change is 1.80× at nvb=1.

This is [`../../tuning-core/graph_captured_benchmarking.md`](../../tuning-core/graph_captured_benchmarking.md)'s
"replay time barely moves with problem size" symptom, one step short of the empty-graph case. **The fix
is not "use a graph", it is "put enough work in the graph"** — in the server these kernels sit inside a
decode graph of several hundred, where the fixed cost is pipelined away. Every benchmark in
`artifacts/analysis/` asserts `mode == "cudagraph"` and has a `warm_clocks()` spin.

## What is left on the table

Ranked by the arm-D decode profile (`artifacts/profiles/candD_s7_kernel_table.txt`, 35.07 ms of device
time in one decode step), so the next run does not have to re-derive the target list:

| target | share of arm-D decode | note |
| --- | --: | --- |
| `mfma_moe1_...` + `mfma_moe2_...`, 92 calls each | **12.83% + 8.77%** | One fifth of decode and by a distance the largest thing on the board. aiter MXFP4 kernels, so the work is in `/sgl-workspace/aiter`; the tile shape is baked into the name (`t32x128x256`, `t32x256x128`). A variant shaped for decode batch 64 is a kernel-authoring job, not a table entry |
| `cross_device_reduce_2stage`, 187 calls | **9.39%** | Two allreduces per layer is a lot for TP=8 in one node. Custom-allreduce selection is partly environment, so care is needed to stay inside the frozen configuration |
| `_fwd_grouped_kernel_stage1`, 24 calls | **6.87%** | What patch `01` already rewrote, on the 24 non-KDA layers. 100 µs per call is the largest per-call cost in the profile; a second pass is plausible now that it is third rather than first |
| `kda_packed_decode_kernel<16, false>` | **5.44%** | Still 28 µs for 99 MB — 3.5 TB/s against a ~9.6 TB/s in-place probe. Warp count was not the whole story; the row loop's access pattern is next |
| the attention-residual pair after `02` | **4.42%** | Score and combine read the same rows twice, so fusing them halves the traffic — but at the decode shape the working set is ~8 MB and fits in Infinity Cache, so the payoff is probably prefill-only |
| the two tiny GEMVs | **3.4% combined** | Torch wins at batch 63, the shipped kernels win at ≤ 16, and *both* run at a fraction of a TB/s. A GEMV kernel designed for tens of tokens is the missing piece |

One loose thread worth a note: `_score_kernel` improved 2.0× in-server (1.54 → 0.78 ms) but
`_combine_kernel` only 1.04× (0.85 → 0.82 ms), where the isolated benchmark predicted more from
combine. The likely cause is the nvb distribution across the 186 call sites, which the microbenchmark
approximated as {1, 4, 7}. A per-call-site nvb histogram would settle it and would sharpen any further
tuning of that kernel.

## What would promote this entry to a verified win

1. **Apply `artifacts/01`…`05` to a clean checkout on a fresh instance and measure** — three rounds,
   then a restart and three more, recording that instance's own spread. This is the only missing
   requirement.
2. **A second node**, since the 0.713% floor and the cold-clock behaviour are properties of the
   machine, not the model.
3. **Optionally, splitting arm D.** With a quieter node or more instances, `03` and `04`+`05` could be
   separated, replacing the current joint +1.39% with two measurements. Do not do this by dividing the
   profile shares — that is the attribution the run explicitly declined to publish as a measurement.

## When this entry stops applying

Silently, in every case:

- **arch ≠ gfx950 or CU count ≠ 256** — literal columns in the CSV key, so patch `05`'s rows are
  unreachable; and `02`/`03`'s wave64 reasoning is simply wrong off CDNA.
- **TP ≠ 8** — N and K shard differently, so `6016 = 1536 + 896 + 3584` is a different number and the
  tuned rows miss.
- **A different SGLang commit** — `01`–`04` are source diffs; they will fail to apply loudly, which is
  the good case, or apply with fuzz into changed code, which is not.
- **A different aiter commit** — the CSV schema and the `/tmp` merge path are properties of that tree.
- **A non-bf16 dense path** — patch `05` is a `bf16_tuned_gemm` table.
- **`--attention-backend` ≠ `triton`** — patch `01`'s kernels leave the decode path entirely.
- **Concurrency, ISL or OSL changed** — decode batch 63 is what makes three of the five patches fire.
- **`--max-running-requests` ≠ 64** — moves the baseline itself by 14%, and changes the mamba cache and
  KV pool sizing with it.
- **A stale `/tmp/aiter_configs`, or no restart.**

**Still reusable when inert:** the two sweep recipes in the transferable-mechanism section (grep
`get_device_capability`, grep `load_weights` for weight merges) — those are the most portable things
here; the shape list (`6016 × 7168` and `144 × 7168`, M 1…1024) and `artifacts/gemm_tune/untuned_k3_fused.csv`
as the tuner input format; the target ranking above; the benchmark harnesses in `artifacts/analysis/`,
which are graph-captured, clock-warmed and correctness-gated and can be re-pointed at other shapes;
and the cold-clock scheduling rule, which cost this run one server instance to learn.

## Provenance

Task bundle: `tuning_workspace/experiment_standalone/kimi_k3_tuning/`.

| what | where |
| --- | --- |
| one-line outcome | `EXPERIMENT_COMPLETE` |
| full run report — arm table, all four negatives, the cold-run decision, the skillset assessment | `FINDINGS.md` (§0 the image precondition, §1 patch 01, §2 patch 02, §3–§5 negatives, §6 patches 04+05, §7 patch 03, §8 the tiny-GEMM negative) |
| what the baseline is, where 804.190 came from, the environment trap, the roofline and concurrency-sweep caveats | `BASELINE.md` |
| the source session's own material, including the +12.10% code-lane cycle that became patch `01` | `reference/`, and `reference/codelane/final_patch.diff` specifically |
| per-patch manifests: base commit, apply command, isolated and end-to-end measurement | the header block of each file in `patches/` |
| raw per-round throughput for all 28 counted rounds, the excluded cold round and the live confirmation | `results/*/inferencex_result.json` |
| accuracy runs with per-sample logs | `eval_results/` |
| the three decode profiles the arms are compared through | `analysis/profiles/{cand_s2,candC_s3,candD_s7}/kernel_table.txt` |
| gzipped launch logs carrying the fingerprint lines and resolved `ServerArgs` | `analysis/serverlogs/{cand_s2,candD_s9}.log.gz` |
| why the released 0.5.15.post1 cannot serve this model | `analysis/stock_0515_cannot_serve_kimi_k3.txt` |

**Not copied into `artifacts/`, deliberately:** the gzipped server logs and profile traces (large,
and the ranked kernel tables carry what a reader needs), `analysis/moe_mem_probe.py` and
`analysis/diaghook/` (diagnostics from the image-precondition work, not part of the deploy), and
`patches/superseded/` (patches for a framework version this entry does not use).
