# Qwen3.5-397B-A17B-MXFP4 on 4× MI355X — SGLang 0.5.17, TP=4, four patches across aiter and SGLang

**Measured win: +3.93% output throughput** (2505.96 → 2604.42 tok/s against the baseline measured on
this machine; **+4.58%** against the 2490.308 tok/s recorded by the source session on another
machine). gsm8k 5-shot strict-match 0.9773 → **0.9727 ± 0.0045** against a gate of 0.9691 — a pass,
by **the thinnest margin of any arm in this run**, and the only score here that moves further than
the harness's own ±4-answer reproducibility band. TTFT 5039.9 → 4979.2 ms and TPOT 20.63 → 19.72 ms
both improve, so the throughput is not bought with latency.

Four changes carry it, two data and two code, none of them touching the frozen launch configuration:
a TP=4 bf16 dense-GEMM tuned table, a load-width dispatch in the topk-softmax routing kernel, an
analytic launch rule for the Gated-DeltaNet state-update kernel, and two re-raced fused-MoE rows.

Measured 2026-08-20 over a single day on host `crsuse2-m2m-287`.

## Reproduction status — read this before the result

**Not reproduced from the artifact alone on a clean instance.** That is the house bar
(`../README.md`, "Adding an entry") and this entry does not meet it. What the bundle does establish
is weaker but not weak:

| what was verified | how |
| --- | --- |
| the baseline reproduces the source session's number | 2491.854 warm mean against 2490.308, **+0.06%**, on a different machine from the one that produced the reference |
| each reported arm survives a restart | the final arm ran on **two independent server instances**, warm means 2605.87 and 2602.96 — a **0.11%** spread, tighter than either instance's internal spread |
| the arms are separated, not just different on average | the four arms' warm run ranges are **mutually disjoint and monotonically ordered** (table below) |
| the exported patches are the trees that were measured | the three aiter patches were applied to a pristine `git archive` of `d9e5ef7c` in every order and the resulting tree is byte-identical to the measured tree |
| the SGLang patch is *almost* that tree | one recorded byte-level difference, below |

What is missing is the step that matters most for reuse: nobody has taken
`artifacts/patches/*.patch`, applied them to a fresh container, and re-measured 2604.42. Every arm
here was measured by editing the live tree and restarting, then exporting the diff afterwards.

**The one recorded discrepancy between artifact and measured build.** The server that produced the
intermediate arm was started before an unused `autotune_cache_kwargs` import was dropped from
`chunk_delta_h.py`, so the benchmarked build still carried that binding while
`artifacts/patches/gdn_chunk_h_launch_config.patch` imports only `is_nvidia_hopper`. Removing
`@triton.autotune` is what made the name unused; it is referenced nowhere afterwards, the same
module is imported either way, and no kernel-path code reads it. Every other byte of the file
matches, and the patch was regenerated against a fresh `git archive` and compared byte-for-byte
against the live tree. The three aiter patches have no such gap.

**The bundle was on hold, and the reason does not affect the numbers.** Its directory keeps a
`hold_` prefix and `RESUMED.md` explains it: the bundle was parked solely because a decision was
outstanding about *which model it should cover*. That decision was made in favour of
Qwen3.5-397B-A17B-MXFP4 as built, nothing had to be redone, and the prefix is historical rather
than a status. The one practical consequence is that every path in `patches/`, `results/` and
`eval_results/` contains `hold_qwen35_397b_a17b_mxfp4_tuning`, so the prefix is load-bearing for
finding the evidence.

## Flagged prominently: this is the one entry where you should re-run the accuracy gate

The final arm — all four patches, the one the +3.93% is claimed for — passes gsm8k, and it passes by
less than one standard error. (A terminology warning for anyone reading the bundle alongside this
entry: `FINDINGS.md` calls the *intermediate* two-patch arm "the reported arm" and this one "the
final arm" or "the second arm". Everything below is about the four-patch arm, `gdnmoe` in
`eval_results/`.)

| quantity | value | in answers of 1319 |
| --- | --- | --- |
| gate (requirement) | **0.9691** | — |
| final arm (all four patches), strict-match | **0.9727066 ± 0.0044881** (1283 correct) | — |
| **margin above the gate** | **+0.0036066** | **4.76 answers** |
| margin expressed in σ | **0.80σ** of the arm's own ±0.0044881 (0.88σ of the baseline's ±0.0041066) | — |
| drop from the 0.9773 baseline (1289) | −0.0045489 | 6 answers, 1.01σ |
| drop from the intermediate arm's 0.9765 (1288) | −0.0037907 | 5 answers, 0.84σ (the bundle records this as 0.8σ) |
| one gsm8k answer, for scale | 0.00075815 | 1 |

**The σ figure by itself is not what makes this the re-run case, and it is worth being precise about
why, because other entries here sit closer to their gates and are fine.** GLM-5.2-MXFP4 ships *two
problems below* its gate; Gemma-4-26B ships at −0.35σ against its bundle contract. Both are sound,
and for the same reason: each entry measured its own eval resolution or its own cross-machine
envelope and found the movement smaller than it. GLM-5.2 measured the resolution directly — two
bit-identical arms scored 6 problems apart, so a 2-problem shortfall is inside its own noise — and
Gemma's whole spread sits inside a baseline envelope its entry establishes across two machines.

**Here the movement is larger than the measured resolution, not smaller.** Six answers against a
±4-answer band established on this exact stack by running the identical build twice, in the one arm
that contains a change to the arithmetic, with **4.76 answers of remaining slack — about the width of
the band itself.** One run cannot distinguish that from noise, and equally cannot show it is nothing.
That is not a reason to disbelieve the result; it is a reason not to inherit it.

**Recommendation: re-run `artifacts/harness/run_eval.sh` on your own deployment of the final arm
before shipping it, and do not inherit the 0.9727.** Two reasons, both measured rather than
inferred:

1. **A build's answer set here is not reproducible to better than ±4 questions.** Two gsm8k runs of
   the *identical* intermediate build both scored 1288 — but not on the same 1288 questions; four
   flipped each way between them. Batch composition varies with arrival timing and that changes
   reduction order. A second draw from the same band could plausibly land below 0.9691.
2. **One of the four patches genuinely changes the arithmetic.**

**The numerical risk is `artifacts/patches/fmoe_tuned_rows_tp4.patch`, and specifically its
token=64 row.** The stage-1 kernel it selects,
`flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w3_fp4`, keeps the stage-1 expert-GEMM intermediate in **fp4
instead of bf16**. The tuner's own error column goes 0.0% → 0.9%, and aiter's `--run_config`
accuracy check against its reference is blunter still:

| token=64 candidate | max abs delta | elements differing |
| --- | --- | --- |
| shipped row | 0.09375 | 0.2% (443 of 262144) |
| error-preserving alternative | 0.0625 | 0.2% (431 of 262144) |
| **the row this patch deploys** | **0.5234375** | **63.1% (165422 of 262144)** |

That path runs once per layer per token step. Nothing else in the arm can move an answer:
`gdn_chunk_h_launch_config.patch` is bit-identical by construction — `BV` tiles `V`, which the
kernel never reduces over — and was verified so on eight shapes with `torch.equal` on `h`, `v_new`
and the in-place-updated recurrent state. The topk patch is bit-identical across a 30-case
shape × dtype grid. The GEMM table changes which kernel computes a dense bf16 matmul and therefore
its reduction order, which is a real but far smaller numerical surface — and the arm containing it
without the fp4 row scored 0.9765 twice.

**If you want the throughput without the numerics change**, swap the token=64 row for the
error-preserving pair in `artifacts/tables/fmoe_cand_new_err0.csv`: production-operator time
125.55 µs against the shipped 127.85 µs, i.e. −1.80% instead of −4.92%, tuner error 0.0%, aiter
delta 0.0625 on 0.2% of elements. The end-to-end cost of that swap was **not measured**; the
arithmetic in the bundle puts the fmoe row's whole contribution at about 1%, so giving back 3 µs of
its 6 µs is roughly half of that — an estimate, not a measurement.

## Which baseline number to quote

Four figures for "the baseline" are in circulation in this bundle and three of them are wrong for
most purposes. This matters more here than usual because the win is 3.93% and the figures span
0.63%.

| figure | what it is | use it for |
| --- | --- | --- |
| 2490.308 tok/s | the **source session's** measurement, on another machine, 2026-08-17 | quoting against the documented number: the win is **+4.58%** on this basis |
| 2491.854 | the local alignment run's warm mean over three runs — but it **includes** a first-run-after-restart (2470.005) that the project's own discard rule says to drop | nothing; it is superseded |
| 2505.50 | instance B alone, three warm runs | it is what §4's per-attempt tables in `FINDINGS.md` were written against; differs from the pooled figure by 0.02% and no verdict turns on it |
| **2505.96** | **the mean of all 8 warm runs across 3 server instances, cold runs discarded** (TTFT 5039.9 ms, TPOT 20.63 ms) | **quote this.** The headline +3.93% is against it |

This machine's frozen-config baseline runs **0.63% above** the documented one. Both numbers are
stated whenever a delta is claimed in the bundle and both are stated here. The reference session's
two rounds agreed to 0.015% (2490.308 and 2489.937) — that is a property of those two rounds, not a
noise floor, and `BASELINE.md` says so explicitly. Do not use it to license a small delta.

## Environment fingerprint

| field | value | load-bearing? |
| --- | --- | --- |
| GPU | **4× MI355X, `gfx950`, 256 CU** each, of eight on the node | **yes** — `gfx` and `cu_num` are literal leading columns of both the bf16 GEMM key and the fused-MoE key. A different arch or CU count makes every row unreachable |
| container | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix` — **digest not recorded** | descriptive, but see the gap note: the tag is what pins the stack and the digest is missing |
| framework | SGLang **0.5.17**, git checkout at `/sgl-workspace/sglang` @ `29481685462732237d80d86076d6563e1f658102` | **yes** for the GDN patch — it is a source diff against that commit |
| aiter | vendored git checkout at `/sgl-workspace/aiter` @ **`d9e5ef7ce08ee7045d583aed768cff41aa9210fe`** | **yes** — all three aiter patches are diffs against it, and the CSV schema and kernel names are its |
| torch / ROCm / Triton / python | 2.9.1+rocm7.2.0.git7e1940d4 / 7.2.0 / 3.6.0 / 3.10.12 | descriptive, but a Triton bump is exactly what cost the sibling Gemma run 4.4% of its baseline |
| model | Qwen3.5-397B-A17B-MXFP4, **TP=4** | **yes** — TP sets every N and K, and TP=4 versus the shipped TP=2 tuning *is* this entry's main finding |
| architecture | 60 layers = **45 Gated-DeltaNet linear-attention + 15 full-attention** (`full_attention_interval: 4`), 512 experts top-10, `moe_intermediate_size` 1024, shared expert in all 60 layers, `head_dim` 256 with 32 q : 2 kv heads | **yes** — the GDN patch only exists because of the 45 GDN layers; the fmoe key encodes `expert 512, topk 10, inter_dim 256` |
| precision, weights | MXFP4 experts via **quark** (`--quantization quark`), with a large `quantization_config.exclude` set left in **bf16**: all GDN projections, `mlp.gate`, the whole shared expert, the full-attention projections and `lm_head` | **yes** — the exclude list is why an 18.41% dense bf16 GEMM bucket exists at all, and it is what the GEMM table addresses |
| precision, KV cache | **no `--kv-cache-dtype` flag at all** — unlike the vLLM entries here, nothing selects fp8. The mamba/SSM state pool is **float32** (50.10 GB for 1139 slots, 4 B/element) | **yes** — the float32 state pool is what makes aiter's fast bf16-state GDN decode row inapplicable |
| attention backend | `--attention-backend aiter`; full attention at decode is `paged_attention_ll4mi_QKV_mfma16_kernel<… BLOCK_SIZE 1, HEAD_SIZE 256, NUM_THREADS 256, GQA_RATIO 8 …>`; prefill is ck_tile FMHA; **GDN is Triton-only** by dispatcher restriction | **yes** — a different backend changes which kernels are hot and strands the GDN patch entirely |
| host | `crsuse2-m2m-287`, devices `renderD128/136/144/152` | descriptive — but the non-contiguous numbering is a real trap, below |
| process environment | **not recorded.** `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` is stated in `FINDINGS.md` §5 as set in the frozen environment; no `/proc/<pid>/environ` dump exists in the bundle | **treat as load-bearing and verify yourself** — see the gap note |

**Where a config label disagrees with what ran.** The harness config
(`reference/workload_config.yaml`) exports `MAX_MODEL_LEN=13312`, while its own
`EXTRA_SGLANG_ARGS` passes `--context-length 11264` and the live server reports 11264. **11264 is
what ran**, verified by `launch_server.sh` against `/get_server_info`; 13312 is a harness variable
that this model's explicit flag overrides. Quote 11264.

**The device-numbering trap, because it fails as a wrong answer rather than an error.** The render
nodes on these hosts step by eight — `renderD128, 136, 144, 152` — not contiguously from 128. Naming
`renderD129` does not fail: docker creates a node that maps to nothing and the container comes up
seeing one GPU where TP=4 needs four, which surfaces much later as an unexplained server start
failure. `artifacts/harness/start_container.sh` discovers the first four that exist and refuses
otherwise; `artifacts/harness/preflight.sh` asserts the count.

## Launch configuration

Reproduce verbatim. Every flag came from the reference measurement's own `ServerArgs` dump, including
the ones that do not look like tuning knobs, because they determine the shapes.

```bash
python3 -m sglang.launch_server \
    --model-path /shared_nfs/hyperloom/models/Qwen3.5-397B-A17B-MXFP4 \
    --host 0.0.0.0 --port 43103 \
    --tp-size 4 \
    --context-length 11264 \
    --watchdog-timeout 1800 \
    --mem-fraction-static 0.68 \
    --chunked-prefill-size 16384 \
    --page-size 1 \
    --disable-radix-cache \
    --attention-backend aiter \
    --trust-remote-code \
    --quantization quark
```

`artifacts/harness/launch_server.sh` is that invocation plus a health wait and a config check, and
the check earns its place: it refuses to let you benchmark unless the live server reports
`context_length=11264`, `tp_size=4`, `attention_backend=aiter`, `chunked_prefill_size=16384`,
`disable_radix_cache=True`, `page_size=1`, `quantization=quark`. Two details in it are worth
knowing. It accepts `mem_fraction_static` of either 0.68 **or** 0.578, because SGLang rescales that
value by 0.85 on builds combining aiter with a context length above 8192 — a mismatch there is
expected, not a fault. And it refuses to attach to a server already on port 43103, because
attaching silently would measure *that* configuration instead of this one.

**Startup cost, which shapes what experiments are affordable:** first start **1068 s** (94
safetensors shards over NFS), **95 s** on a restart with the page cache warm. The health wait is set
to 2400 s. That is cheap compared to the DeepSeek entry's half hour, which is why this run could
afford two server instances per arm.

**Environment variables: the bundle does not record the process environment, and this is a gap.**
`launch_server.sh` sets none. Per this directory's own cross-cutting lesson, that does not mean the
process has none — Mixtral's image exported twelve, and Gemma's exported the
`ROCM_QUICK_REDUCE_QUANTIZATION=INT8` its all-reduce actually ran. `FINDINGS.md` §5 states that the
same variable is set in **this** frozen environment too, and uses it to rule out INT4 as a lever,
but no `cat /proc/<pid>/environ | tr '\0' '\n'` dump was captured. **Run that yourself before
attributing anything to the environment**, and record it, because on this stack the quickreduce
codec quantization level is worth roughly 5.8% and is decided entirely by that variable.

## Workload

ISL 8192, OSL 1024, concurrency 64, 192 prompts, **8 warmups**, `random_range_ratio 1.0`,
`--random-prefix-len 0`, `--ignore-eos`, seed 0, InferenceX `benchmark_serving` fork
(`artifacts/harness/run_bench.sh`). Every run in this entry moved exactly 1,572,864 input and
196,608 output tokens; a run that does not is not comparable.

Which parameters set the tuned shapes — all four patches are keyed on these, so changing any one of
them makes most of this entry inert:

- **concurrency 64** → decode `M=64`, the batch every token step runs at. It is the M of five of the
  thirteen GEMM rows and of the fmoe token=64 row.
- **`--chunked-prefill-size 16384` over ISL 8192** → prefill `M ∈ {8192, 16384}`, the M of the other
  eight GEMM rows and of the fmoe token=16384 row. It is *also* what makes N (sequences per chunk) 1
  or 2 in the GDN kernel, which is the entire basis of the GDN patch.
- **TP=4** → halves every N and every K relative to the shipped TP=2 table.
- **512 experts, top-10** → the `EXPERTS < 512 ? 32 : 64` branch in the topk-softmax launcher, which
  is what the topk patch dispatches around.

Warmups are 8, as in the frozen workload the other bundles in this campaign share; DeepSeek's 128 is
the documented exception (`../README.md`).

## Where the time actually goes — the bundle's documented breakdown is wrong

Read this before choosing a target, because the number in the task documents would send you to the
wrong place. `README.md` and `reference/tracelens/` attribute **86.3% of decode time to attention**.
That is an analytic estimate built on a dense-transformer model of the network. Measured on the
trace, over 4 ranks × 8 decode steps, 563,619 µs of device time:

| bucket | % device | µs | calls |
| --- | --: | --: | --: |
| MoE gemm1 (mxfp4) | 18.92% | 106639 | 1920 |
| **Dense bf16 GEMM (projections)** | **18.41%** | 103755 | 11072 |
| Full attention (paged) | 13.32% | 75058 | 960 |
| MoE gemm2 (mxfp4) | 11.28% | 63602 | 1920 |
| TP all-reduce | 8.17% | 46056 | 3904 |
| GDN linear-attention core | 7.38% | 41584 | 1440 |
| MoE quant + sort | 7.24% | 40815 | 7680 |
| GDN aux (conv1d, norm, split, gate) | 5.20% | 29318 | 6720 |
| MoE routing softmax | 3.52% | 19836 | 1920 |
| RMSNorm | 3.32% | 18686 | 4352 |
| Activation (shared expert) | 1.46% | 8242 | 1920 |
| RoPE / KV-cache store / other | 1.78% | 10028 | |

**MoE totals 40.97%; attention of both kinds totals 25.90%. Not 86.3%.** A decode step is
563619 / (4 × 8) = **17613 µs** of summed device time per rank.

Two method notes that made the difference, both reusable: rank kernels by **summed device duration**
over `ph=="X"` events with `cat ∈ {kernel, gpu_memcpy, gpu_memset}`, never by a percentage column —
on these traces that column sums `hipGraphLaunch` wrappers and is meaningless. And capture through
the live server (`artifacts/tuning/profile_decode.py` drives 64 concurrent `/v1/completions` streams
at the frozen shape, warms 25 s, then captures 8 steps on all four ranks via `/start_profile`).

**And a third of the benchmark is not decode at all.** The frozen workload sends 1,572,864 prefill
tokens against 196,608 decode tokens, an 8:1 ratio. Three independent accountings of a 78.33 s warm
run agree: decode is 3 waves × 1024 tokens × 16.24 ms median ITL = 49.9 s, leaving **28.4 s (36%)
for prefill** as the residual; counted forward instead, 192 × 8192 / 16384 = 96 chunks ×
279 ms/chunk = 26.8 s. **Those two closing to within 6% is the check that the phase split is real**,
and mean TTFT 5.03 s closes it a third way — the first 64 requests are 32 chunks, and half of
32 × 0.29 s is 4.6 s. The 279 ms/chunk is measured, not assumed: it is the summed per-rank device
time of a captured prefill pass. **The sensitivity that follows: 1% off prefill time ≈ +0.35% output
throughput**, so a 20% prefill win would be worth +8%. Every profile in this bundle before this run
was decode-only and therefore blind to a third of the problem.

The prefill hot list is a *different list*, which is the most transferable fact in this entry:

| bucket | % of prefill | µs |
| --- | --: | --: |
| **TP all-reduce** | **33.42%** | 2987669 |
| Dense bf16 GEMM | 17.50% | 1564723 |
| GDN linear attention | 15.23% | 1361115 |
| MoE gemm (mxfp4) | 13.30% | 1189244 |
| MoE quant / sort / reduce | 7.73% | 690826 |
| Full attention (ck_tile FMHA) | 5.41% | 483430 |
| RMSNorm | 4.41% | 394135 |
| Activation / gate, other | 3.01% | 268753 |

8,939,895 µs over 4 ranks × 8 passes = **279.4 ms per rank per pass**. Combining phases, the
all-reduce is 33.42% of a 26.8 s prefill plus 8.17% of a 49.9 s decode = **13.0 s, 16.6% of the whole
benchmark** — the single largest line item, and the one this run establishes you should *not* work on
(see "what did not work").

## The four patches

### 1. `artifacts/patches/gemm_tuned_table_tp4.patch` — the +1.60%, and the finding

**AMD tuned this model's dense bf16 GEMMs for TP=2, and the frozen configuration is TP=4.** aiter
ships `model_configs/qwen3_5_397b_bf16_tuned_gemm.csv` whose `(N, K)` set is
`10240×4096, 8704×4096, 4096×4096, 4096×8192, 4096×512, 1024×4096, 512×4096, 256×4096, 64×4096`
(`FINDINGS.md` §4; the patch header renders the same table as the looser cross-product
N ∈ {10240, 8704, 8192, 4096} × K ∈ {8192, 4096, 512} — quote the pair list, it is the one that
matches the file).
Work those back through the checkpoint shapes — `in_proj_qkvz` 20480, attention `qkv` 17408,
`out_proj`/`o_proj` K=8192, shared-expert `gate_up` 2048 and `down` K=1024, `in_proj_ba` 128 — and
every one is the tensor divided by **two**. At TP=4 every N and K halves again, the lookup key
`(gfx, cu_num, M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle)` is exact, and most of the
table is unreachable. The miss is logged unconditionally by `aiter/tuned_gemm.py`, once per unique
shape per process because the lookup is `lru_cache`d: **the baseline server log carries 956 such
lines.**

The seven TP=4 decode shapes, resolved, with isolated per-call cost at M=64 measured through the
same `gemm_a16w16` entry point the server uses (`artifacts/tuning/bf16_gemm_probe.py`) and
cross-checked against the trace — every kernel identity and per-call time matches:

| N | K | what it is | calls/step/rank | resolves to | µs/call | weight GB/s | µs/step/rank |
| --: | --: | --- | --: | --- | --: | --: | --: |
| 5120 | 4096 | GDN `in_proj_qkvz` | 45 | **miss → torch** | 16.92 | 2479 | 761.3 |
| 4608 | 4096 | full-attn `qkv_proj` (KV heads replicated 2→4) | 15 | **miss → torch** | 16.65 | 2267 | 249.8 |
| 4096 | 2048 | `out_proj` / `o_proj` | 60 | flydsl, tuned | 7.49 | 2241 | 449.2 |
| 4096 | 256 | shared-expert `down_proj` | 60 | **miss → torch** | 4.30 | **488** | 258.0 |
| 512 | 4096 | router `gate` + shared-expert `gate_up` | 120 | flydsl, tuned | 5.49 | **764** | 658.5 |
| 32 | 4096 | GDN `in_proj_ba` | 45 | **miss → torch** | 6.55 | **40** | 294.9 |
| 1 | 4096 | `shared_expert_gate` | 60 | **miss → torch** | 6.39 | **1** | 383.3 |
| | | | | | | | **3055.0** |

3055 µs against a 17613 µs step is 17.3%, consistent with the 18.41% trace bucket. **The bandwidth
column is the point.** MI355X HBM runs at roughly 8 TB/s. `shared_expert_gate` moves 8 KB of weights
and takes 6.39 µs, sixty times per step. Even the two shapes that *do* hit tuned rows are
unimpressive — 764 GB/s on the 120-call `512×4096`. These are small memory-bound kernels dispatched
to configurations chosen for a different tensor-parallel degree, or to none at all.

The artifact is data only, generated by aiter's own tuner:

```bash
python3 /sgl-workspace/aiter/csrc/gemm_a16w16/gemm_a16w16_tune.py \
  --input_file <shapes>.csv --tuned_file <out>.csv --libtype all \
  --with-hipblaslt --splitK --compare --update_improved --min_improvement_pct 3
```

It ships **13 rows** in a new `model_configs/qwen3_5_397b_bf16_tuned_gemm_tp4.csv`
(`artifacts/tables/qwen3_5_397b_bf16_tuned_gemm_tp4.csv`, extracted from the patch as a drop-in):
five at M=64, four at M=8192, four at M=16384; five `flydsl` and eight `hipblaslt`. Note what is
*not* there. The tuner produced candidates for all seven decode shapes
(`artifacts/tables/tuned_tp4_decode.csv`), but the two shapes that **already** hit tuned rows came
back **worse than the incumbent**, so they were dropped: the tuner's best candidate for `512×4096`
was 6.83 µs and for `4096×2048` was 9.09 µs, while the probe measures the rows already installed for
those two shapes at 5.49 and 7.49 µs in-server. Those are two different harnesses — the tuner's own
timing loop and `bf16_gemm_probe.py` — so take from this only the verdict `FINDINGS.md` records,
that the candidate lost, and not a like-for-like ratio. At prefill, 8 of 12 shapes cleared the 3% update
threshold, and there the tuner's `--compare` pass measured both sides itself:

| M | N | K | before | after | |
| --: | --: | --: | --: | --: | --: |
| 8192 | 32 | 4096 | 20.99 µs | 15.44 µs | **1.36×** |
| 8192 | 512 | 4096 | 52.45 | 44.38 | 1.18× |
| 16384 | 4608 | 4096 | 519.65 | 443.71 | 1.17× |
| 16384 | 512 | 4096 | 70.76 | 65.91 | 1.07× |
| 8192 | 5120 | 4096 | 274.98 | 260.07 | 1.06× |
| 8192 | 4096 | 256 | 28.21 | 27.13 | 1.04× |
| 16384 | 4096 | 256 | 57.07 | 54.94 | 1.04× |
| 16384 | 4096 | 2048 | 222.52 | 214.94 | 1.03× |
| 16384 | 5120 | 4096 | 465.89 | 464.64 | not taken |
| 16384 | 32 | 4096 | 27.36 | 26.91 | not taken |
| 8192 | 4096 | 2048 | 112.01 | 115.74 | worse |
| 8192 | 4608 | 4096 | 250.46 | 254.57 | worse |

**The patch also deletes three lines from two other models' tables, and that is not optional.** aiter
merges every file under `model_configs/` into one table and **hard-errors at import** if two files
key the same shape. `dsv4_bf16_tuned_gemm.csv` and `glm5_bf16_tuned_gemm.csv` had already registered
three shapes that TP=4 Qwen3.5-397B also hits — meaning those shapes were never "missing" from the
merged table, they were resolving to *another model's* tuning, one of them to plain `torch`. The new
rows are faster at all three: 45.76→39.36, 68.36→64.19, 212.35→210.23 µs. aiter's auto-resolver
keeps the faster row but rewrites both files wholesale; deleting the three lines by hand gets the
same table with a three-line diff. **This is the one failure mode in the whole deploy that is loud
rather than silent** — skip the deletions and the server refuses to import.

### 2. `artifacts/patches/topk_softmax_ldg_width.patch` — +0.91% by difference, bit-identical

Worth reading for its method more than its size, because the first hypothesis was wrong in an
instructive way. `topkGatingSoftmaxLauncherHelper` sizes its per-thread load as
`MAX_BYTES_PER_LDG = EXPERTS < 512 ? 32 : 64`. At 512 bf16 experts it takes the 64-byte branch:
VPT=32, THREADS_PER_ROW=16, ROWS_PER_CTA=8 — so a 64-row decode batch launches **8 workgroups on a
256-CU GPU**. That looks exactly like an occupancy bug, and the run originally filed it as one.

The measurement said otherwise (`artifacts/tuning/topk_prof_topk.py`, device time from
`torch.profiler`; note that a cuda-event loop around the python wrapper reports a flat ~19 µs at
every M because it times dispatch — `artifacts/tuning/topk_bench_topk_WRONG_METHOD.py` is kept as
the record of that mistake):

| rows | workgroups | stock | patched |
| --: | --: | --: | --: |
| 64 | 8 | 10.57 µs | **8.40 µs (−20.5%)** |
| 128 | 16 | 10.76 | 8.35 |
| 256 | 32 | 10.96 | 8.51 |
| 512 | 64 | 10.67 | 8.61 |
| 8192 | 1024 | 13.13 | 13.21 |
| 16384 | | 20.22 | 20.13 |

**Stock being flat from 8 workgroups to 64 is the evidence: occupancy was never the constraint.**
What is constant per row is the top-k loop — it rescans all VPT=32 registers on each of its k−1
iterations, 288 serialised compares at k=10, with 16 lanes sharing the row's reduction. For
calibration, a trivial elementwise kernel at this size costs 4.40 µs on the same device
(`artifacts/tuning/topk_calib_elementwise.py`), so roughly 6 µs of the 10.6 is that chain. The patch
templates the load width and picks it from `num_rows`: narrow enough that THREADS_PER_ROW reaches
the full 64-lane wavefront (VPT=8 for bf16) at ≤2048 rows, the stock wide form above. **The switch
is not optional** — the narrow form is 20.1 µs at M=8192 against the wide form's 13.1 — and the
threshold sits in the empty span between decode batches of tens of rows and prefill chunks of
thousands.

Output is **bit-identical** to stock, not merely close: `artifacts/tuning/topk_check_topk.py` and
`topk_dump_topk.py` compare weights, indices and `token_expert_indices` for
M ∈ {1,3,64,65,128,255,256,512,8192,16384} × {bf16, fp16, fp32} and all 30 cases pass
`torch.equal`. (All three dtypes also disagree with `torch.topk` on *index* for tied logits,
identically before and after — bf16 over 512 random logits produces many exact ties and the two
tie-break differently. Weights match to 6e-8, so it is tie-breaking, and it is pre-existing.)

This is the only patch requiring a **rebuild**: it edits `csrc/kernels/topk_softmax_kernels.cu`, so
`module_moe_asm.so` must be regenerated (`artifacts/tuning/build_moe_asm.py`, ~2 min).

### 3. `artifacts/patches/gdn_chunk_h_launch_config.patch` — the disabled-autotuner case

`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` is second in the prefill profile at **7.70% of
prefill device time** (477.8 µs × 1440 calls). It is a Triton kernel in SGLang's own tree carrying a
`@triton.autotune` with **exactly one config** — `BV=32, num_warps=4, num_stages=2` — and the
in-tree comment's reason is real, not laziness: the kernel writes the final recurrent state back
into `initial_state` **in place**, so a multi-config autotune would run candidates over live state
during its benchmark phase and corrupt the pool, and `restore_value=["initial_state"]` OOMs at
production scale. **The config is pinned by necessity, not by measurement.**

The grid is `(cdiv(V, BV), N * H)`. `BV` is pinned; the grid is not. At this model's TP=4 shard —
V=128, H=16 value heads — with `--chunked-prefill-size 16384` over ISL 8192, **N is 1 or 2**, so the
launch is 64 or 128 workgroups on a 256-CU part. The pinned tile is sized for a regime this workload
never enters. `artifacts/tuning/gdn_bench_chunk_h.py` sweeps `(BV, num_warps, num_stages)` at the
real shard shape with synthetic varlen inputs, forcing one `triton.Config` at a time so the state
pool is never benchmarked over:

| N seqs | grid at BV=16 | shipped 32/4/2 | best found | speedup |
| --: | --: | --: | --- | --: |
| 1 | 128 | 369.45 µs | 272.78 (16/4/3) | **1.354×** |
| 2 | 256 | 506.91 | 421.62 (16/4/3) | 1.202× |
| 3 | 384 | 540.21 | 481.05 (16/4/2) | 1.123× |
| 4 | 512 | 559.85 | 504.72 (16/4/2) | 1.109× |
| 8 | 1024 | 719.04 | 718.71 (shipped) | 1.000× |

The shape of that table is the finding: halving `BV` doubles the grid and it pays exactly until the
device is oversubscribed, after which each `i_v` workgroup re-reading all of `k` costs more than the
extra parallelism is worth. The patch replaces the pinned config with an analytic rule computing the
grid the caller is about to launch — small tile only while it fits — and SGLang's env overrides are
preserved and take precedence. **Note that the autotune key (`H, K, V, BT, USE_GK, NT_BUCKET`) could
not have expressed this even with more configs, because it does not contain `N`** — the thing that
decides the grid.

Bit-identical on `h`, `v_new` and the in-place-updated state across 8 shapes
(`artifacts/tuning/gdn_check_chunk_h.py`). `BV` tiles `V`, which the kernel never reduces over, so
that is the expected result rather than a lucky one — but it is checked, because "should be
identical" is how the state-pool bug above would have been missed too.

Size: 1.20–1.35× on 7.70% of prefill, and prefill is 36% of wall clock ⇒ **~0.5% end to end**,
which is *below* this machine's restart floor. Not claimed alone; measured stacked.

### 4. `artifacts/patches/fmoe_tuned_rows_tp4.patch` — two rows, and the numerics risk

MoE is 41.0% of decode, and here the attempt-1 story does **not** repeat: aiter's
`qwen3_5_397b_fp4_tuned_fmoe.csv` already carries TP=4 rows (`inter_dim` 256 =
`moe_intermediate_size` 1024 / 4), the server log shows tuned hits at every M tier, and an early
grep of the *default* `tuned_fmoe.csv` showing zero fp4/gfx950 rows was simply looking at the wrong
file. What is off is subtler: **the shipped row for token=64 records 114.61 µs for a pair that
measures 111.61 µs on this build.** The ranking was decided against different kernel timings than
the ones now installed.

Method: re-race all **1688** stage1 × stage2 candidates at the two token tiers this workload
actually runs — 64 (decode batch = concurrency) and 16384 (`--chunked-prefill-size`) — **with the
server stopped**, then let aiter's own production-operator benchmark decide (`--run_config`, which
re-measures rather than echoing the CSV):

| token | | kernel µs | **E2E µs** | vs shipped |
| --: | --- | --: | --: | --: |
| 64 | shipped | 114.61 | **127.85** | — |
| 64 | best error-preserving | 111.61 | **125.55** | −1.80% |
| 64 | **best overall, deployed** | 108.74 | **121.56** | **−4.92%** |
| 16384 | shipped | 986.82 | **991.99** | — |
| 16384 | **best error-preserving, deployed** | 976.53 | **973.21** | **−1.89%** |
| 16384 | best kernel time (fp4 stage1) | 935.61 | **994.08** | **+0.21%, a regression** |

**The two tiers disagree, and that is the interesting part.** The fp4-intermediate stage-1 kernel is
5% faster *as a kernel* at token=16384 and slower end to end: quantizing a 16384-token intermediate
costs more than it saves. So the deployed row set is mixed — fp4 stage1 at decode, error-preserving
at prefill — which no single global preference would have produced. It is also the reason the E2E
column and not the kernel column decides: at token=16384 they point opposite ways.

The near-miss worth recording: the first attempt used `--compare --update_improved
--min_improvement_pct 3`, which raises `KeyError: "['gfx'] not in index"` inside
`base_tuner._merge_compare_filtered_results` — the shipped `model_configs` CSVs begin at `cu_num` and
have no `gfx` column — **and then writes the old rows to `-o` anyway.** The output file looks exactly
like a tuning result and is the pre-tuning selection; its improvement would have been zero and
nothing would have said why. That run had also raced only the `--mxfp4-flydsl` subset, 14 candidates
against 1688. Reproduced in `artifacts/evidence/fmoe_tune_first_attempt.log`. **Do not pass
`--compare --update_improved` to `gemm_moe_tune.py` on these files.**

### Per-patch attribution — what was measured together, and against what

Precision here matters because two of the four are individually below the noise floor and the
bundle is explicit about not claiming them.

| arm | patches deployed | measured against | warm runs / instances | tok/s | delta | what is claimed |
| --- | --- | --- | --: | --: | --: | --- |
| baseline | none | — | 8 / **3** | **2505.96** | — | the reference for everything below |
| A1 | `gemm_tuned_table_tp4` **alone** | baseline | 5 / **1** | 2546.04 | **+1.60%** | **a result** — disjoint from the baseline range, but see the caveat |
| A2 | A1 **+** `topk_softmax_ldg_width` | A1, by difference | 8 / **2** | 2569.26 | +2.53% cumulative, **+0.91% increment** | the increment is an **attribution by difference**, not an independent result: it is under the floor. Supported by the kernel's −20.5%, by bit-identical output, and by TPOT moving 20.25 → 20.04 ms |
| A3 | A2 **+** `gdn_chunk_h_launch_config` **+** `fmoe_tuned_rows_tp4`, **together** | A2 | 8 / **2** | **2604.42** | **+3.93% cumulative, +1.37% increment** | the **pair** is a result: two instances each side, run ranges disjoint. The split between the two patches is arithmetic only |
| — of which GDN alone | | | | | ~0.5% | arithmetic: kernel 1.20× × 7.70% of prefill × 36% of wall clock |
| — of which fmoe alone | | | | | ~1% | arithmetic: operator −4.92% at decode × ~32% of the step |

Three things to take from that table:

- **No patch was measured in isolation except the GEMM table.** The topk patch applies cleanly on
  its own but was only ever run on top of the GEMM table; the GDN and fmoe patches were only ever
  run together, on top of both. Separating the last pair would take four more server instances to
  buy a split smaller than the floor in either direction.
- **The +1.60% step rests on one tuned server instance against three baseline instances.**
  `FINDINGS.md` §6 asserts in prose that every arm rests on at least two instances; that is true of
  the baseline (3), A2 (2) and A3 (2), but its own table records A1 as 5 warm runs from a single
  start. The claim survives on disjointness of ranges, but it is a restart-to-restart delta on only
  one side.
- **The GEMM-table-only arm was never accuracy-gated.** It is a strict subset of the gated A2 arm
  and the half it drops is the bit-identical one, so the gated result bounds it.

## Baseline and noise floor

| | value |
| --- | --- |
| stock, this stack (8 warm runs, 3 instances: A, B, C) | **2505.96 tok/s**, TTFT 5039.9 ms, TPOT 20.63 ms |
| with all four patches (8 warm runs, 2 instances) | **2604.42 tok/s**, TTFT 4979.2 ms, TPOT 19.72 ms |
| delta | **+3.93%** |

| noise floor | spread |
| --- | --- |
| repeating the benchmark within one process | **0.25%** (A, 2 warm), **0.47%** (B, 3 warm), **0.65%** (C, 3 warm) |
| **across restarts, warm runs pooled, cold discarded** | **0.65%** (2497.737 … 2514.111 over the 8 pooled warm runs) |
| across restarts, keeping instance A′'s undiscarded first run (2470.005) | **1.60%** as recorded in `FINDINGS.md` §1 |

**The restart floor is the one that applies.** Both CSVs are read by aiter at import, the topk change
lives in a compiled `.so`, the GDN change is python source read at import, and the decode path is
HIP-graph captured — so no candidate can be swapped into a live process and every A/B here is
restart-to-restart (`../../tuning-core/measurement.md` Rule 3b,
`../../tuning-core/graph_captured_benchmarking.md`).

The honest reading is that the floor is between **0.65% and 1.60%**, and the only thing that widens
it is a run the project's own rule says to discard. `FINDINGS.md` keeps it in the table so the spread
is not understated, and states plainly that **a delta under ~1.6% is not claimed as a result on its
own** — which is precisely why the topk and GDN increments are reported as attributions. (One
arithmetic note for a reader checking the number: recomputed on the same `(max−min)/min` convention
the arm table's "spread" column uses, the pooled-warm range extended with 2470.005 comes to 1.79%
rather than 1.60%. Nothing turns on it — either figure sits above every increment that is not
claimed as a result, and the conclusion is unchanged.)

**Are the arms disjoint? Yes — all four, and monotonically.** This is the strongest evidence in the
entry, stronger than the difference of means:

| arm | instances | cold runs (discarded) | warm runs | range | mean |
| --- | --- | --: | --- | --- | --: |
| baseline | A | 2258.91 | 2505.92, 2499.64 | | |
| baseline | B | 2465.31 | 2498.05, 2509.93, 2508.52 | | |
| baseline | C | 2498.80 | 2497.74, 2514.11, 2513.79 | **[2497.74, 2514.11]** | **2505.96** |
| A1 GEMM table | 1 | 2534.25 | 2531.36, 2545.87, 2561.08, 2553.76, 2538.15 | **[2531.36, 2561.08]** | **2546.04** |
| A2 + topk | 1 | 2569.78 | 2570.65, 2574.41, 2566.18 | | |
| A2 + topk | 2 | 2537.58 | 2569.76, 2566.08, 2565.87, 2566.80, 2574.29 | **[2565.87, 2574.41]** | **2569.26** |
| A3 + GDN + fmoe | 1 | 2602.54 | 2603.02, 2603.28, 2611.06, 2606.13 | | |
| A3 + GDN + fmoe | 2 | 2570.71 | 2601.62, 2605.31, 2597.63, 2607.29 | **[2597.63, 2611.06]** | **2604.42** |

No two ranges overlap and no range even touches its neighbour. (Two bookkeeping notes for a reader
recomputing from `results/`: the A2 range is stated as `[2565.87, 2574.29]` in `FINDINGS.md`, which
takes its maximum from instance 2's 2574.286 and overlooks instance 1's 2574.413 — the pooled
maximum is 2574.41, and disjointness is unaffected either way. And instance C's discarded "cold" run
of 2498.80 sits inside the warm range, which is why it is excluded by rule rather than by looking
like an outlier.) The final arm's two instances are the
cleanest restart pair in the bundle: warm means **2605.87 and 2602.96, a 0.11% restart-to-restart
spread** — smaller than the within-instance spread of either one, which is what makes the +1.37%
increment claimable at all.

**Discard the first benchmark after every server start; the cold penalty is a property of the
machine's cache state, not of the model.** Instance A's first run came in at 2258.91, **9.29% low**,
with TTFT 6831.2 ms against a warm 5039.9. But instance B's first run was only 1.6% low (2465.31,
TTFT 5125.8 ms) because B started with the aiter JIT cache and the model's page cache already warm
from A — and instance C's "cold" run, 2498.80, landed *inside* the warm range. **So a small first-run
gap is not evidence that the server is warm.** The rule earns its keep on the final arm: instance
2's discarded first run was 2570.71, which sits inside the *previous* arm's range. A single
undiscarded first run against a fresh server would have erased this entire result.

Eight warmup prompts are not enough to warm a 397B MoE: expert routing has to touch enough experts
and the aiter kernels have to be resident.

## Deploy

Inside the container, with the server stopped. Order matters only in that the caches must be cleared
after the edits and before the restart.

```bash
# 1. the three aiter changes (all against d9e5ef7ce08ee7045d583aed768cff41aa9210fe)
cd /sgl-workspace/aiter
git apply /path/to/artifacts/patches/gemm_tuned_table_tp4.patch
git apply /path/to/artifacts/patches/topk_softmax_ldg_width.patch
git apply /path/to/artifacts/patches/fmoe_tuned_rows_tp4.patch

# 2. the SGLang change (against 29481685462732237d80d86076d6563e1f658102)
cd /sgl-workspace/sglang
git apply /path/to/artifacts/patches/gdn_chunk_h_launch_config.patch

# 3. MANDATORY cache invalidation: aiter caches the merged model_configs table here and will
#    otherwise serve the pre-edit selection, silently, for both CSV changes.
rm -rf /tmp/aiter_configs

# 4. MANDATORY rebuild, for the topk patch and only for it (it edits a .cu). ~2 min.
python3 /path/to/artifacts/tuning/build_moe_asm.py

# 5. restart — mandatory. Not optional for any of the four.
/path/to/artifacts/harness/launch_server.sh          # must print "config verified"

# 6. measure. The first benchmark after a start is cold.
TAG=verify /path/to/artifacts/harness/run_bench.sh   # discard this one
TAG=verify /path/to/artifacts/harness/run_bench.sh   # and this is the number
```

The three aiter patches touch disjoint files and were verified to apply in any order. Reverting is
`git apply -R` plus `rm -rf /tmp/aiter_configs` again for either CSV.

Equivalently, if you would rather drop the table in than patch: the GEMM change is exactly
`cp artifacts/tables/qwen3_5_397b_bf16_tuned_gemm_tp4.csv /sgl-workspace/aiter/aiter/configs/model_configs/`
**plus** the three deletions in `dsv4_bf16_tuned_gemm.csv` and `glm5_bf16_tuned_gemm.csv` that the
patch also makes. Without those the merged table has duplicate keys and aiter raises at import.

### Every way this deploy silently does nothing

Ordered by how likely it is to catch you. All of these produce a clean, plausible, wrong number
except the last, which is the only loud failure in the set.

1. **A stale `/tmp/aiter_configs`.** aiter's merged model-configs table is derived and is *not*
   regenerated if it already exists. Edit a tuned CSV, restart, and the server happily reads the
   pre-edit merge — both the GEMM table and the fmoe rows go dark together. This is the single most
   likely no-op and the reason `rm -rf` is in every recipe in the bundle.
2. **No restart.** aiter reads its tables once, at import; the GDN change is python source read at
   import; and the decode path is HIP-graph captured, so a live drop-in benchmarks perfectly and
   changes nothing.
3. **Skipping the `module_moe_asm` rebuild.** The topk patch edits `topk_softmax_kernels.cu`. Without
   the rebuild the old `.so` is imported and that patch contributes exactly zero — while the other
   three still work, so the arm comes in around 2569 instead of 2604 and looks like a measurement
   problem rather than a deploy problem.
4. **`SGLANG_GDN_CHUNK_H_BV`, `SGLANG_GDN_CHUNK_H_NUM_WARPS` or `SGLANG_GDN_CHUNK_H_NUM_STAGES` set
   in the environment.** The patch deliberately lets an explicit override win over its heuristic, so
   any of these being set — including to the stock `32` — bypasses the entire rule and returns the
   shipped tile. This is a no-op path the patch itself introduces, and it will not warn you.
5. **Stale `__pycache__` for `chunk_delta_h.py`.** Python source read at import is only re-read if
   the bytecode cache is invalidated; the mtime check normally handles it, but a `git apply` that
   preserves mtimes, or a read-only tree, will not. Not specifically recorded as having bitten this
   run — listed because `../README.md` records it biting others.
6. **A stale `~/.triton/cache`.** The GDN change alters `BV`, `num_warps` and `num_stages`, which
   are part of Triton's compilation key, so a stale cache *should* miss rather than serve the old
   kernel. Also not recorded as a problem here. Clear it if a bit-identity check disagrees with the
   one in this entry.
7. **Any workload or topology change** — concurrency, `--chunked-prefill-size`, ISL, TP. The GEMM
   rows and the fmoe rows are keyed on M and token exactly; they miss and fall back to torch, with
   no warning beyond the miss lines nobody reads. See "when this entry stops applying".
8. **Omitting the three duplicate-key deletions** — which fails *loudly*, at import, and is the only
   member of this list you cannot ship past.

## Engagement check

Three checks, one per surface that can silently miss. None of these patches fails loudly.

**1. The fused-MoE rows are live.** This is the one to run first, because it is also the check that
catches a stale `/tmp/aiter_configs`.

```bash
grep "using 2stage" /tmp/sglang_server_qwen3_5_397b_a17b_mxfp4.log | grep "256, 64, 4096, 256"
```

- **Engaged:** `kernelName1='flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w3_fp4'` and
  `kernelName2='flydsl_moe2_afp4_wfp4_bf16_t16x128x256_atomic_persist_sbm32'`.
- **Not engaged:** `kernelName1='flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w3'` (no `_fp4` suffix) and
  `kernelName2='flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic_bnt2'`. If you see this, the merge
  cache was not cleared and **nothing in that patch is live** — including the numerics change, which
  is a reason a gsm8k re-run can come back looking suspiciously like the baseline.

The `_fp4` suffix is doing double duty here: it is both the engagement marker and the name of the
numerical risk. That is convenient and it is worth being conscious of, because "engaged" and "the
arithmetic changed" are the same observation for this patch.

**2. The GDN launch rule is live.**

```bash
python3 -c "import sys; sys.path.insert(0,'/sgl-workspace/sglang/python');
from sglang.kernels.ops.attention.fla.chunk_delta_h import gdn_chunk_h_launch_config as f;
print(f(128, 2, 16))"
```

- **Engaged:** `(16, 4, 3)` — V=128, N=2, H=16 gives a small-tile grid of `cdiv(128,16) × 2 × 16 =
  256`, which is ≤ 256 CUs, so the deepest pipelining is selected.
- **Not engaged, patch not applied:** `ImportError` — the symbol does not exist in the stock file at
  all, which makes this a genuinely two-sided check rather than a value comparison.
- **Not engaged, patch applied but bypassed:** `(32, 4, 2)`, which means one of the three
  `SGLANG_GDN_CHUNK_H_*` variables is set. Check the process environment, not just the shell.
- Sanity-check the rule's shape while you are there: `f(128, 1, 16)` → `(16, 4, 3)`,
  `f(128, 4, 16)` → `(16, 4, 2)`, `f(128, 8, 16)` → `(32, 4, 2)`. If the last one does not fall back
  to the shipped tile, `_num_cus()` is not reporting 256 and you are not on this part.

**3. The bf16 GEMM table is live.** The miss line is unconditional — `aiter/tuned_gemm.py` logs
`not found tuned config` at `logger.info` on every lookup failure, once per unique shape per process
because the lookup is `lru_cache`d — so the check is that specific misses disappear:

```bash
grep -c 'not found tuned config' /tmp/sglang_server_qwen3_5_397b_a17b_mxfp4.log
grep 'not found tuned config' /tmp/sglang_server_qwen3_5_397b_a17b_mxfp4.log \
  | grep -oE 'N:[0-9]+, K:[0-9]+' | sort | uniq -c
```

- **Not engaged:** **956** miss lines on a baseline instance, including the five patched decode
  shapes (`N:5120 K:4096`, `N:4608 K:4096`, `N:4096 K:256`, `N:32 K:4096`, `N:1 K:4096`).
- **Engaged:** the count drops and **no miss line names any of those five shapes at M=64**, nor the
  eight patched prefill shapes.
- **The negative control that must survive:** the four shapes the tuner deliberately did *not* take
  — `M=16384 N=5120 K=4096`, `M=16384 N=32 K=4096`, `M=8192 N=4096 K=2048`, `M=8192 N=4608 K=4096` —
  should still miss. If the log goes completely quiet, something other than this patch changed the
  merged table.

**The exact post-patch miss count is not recorded in the bundle**, so assert the direction and the
per-shape identities rather than a number. 956 is the only figure recorded, and it is the
not-engaged side.

**No engagement check is recorded for the topk patch, and that is a gap.** The bundle verifies the
rebuild happened and relies on the arm number. The best available substitute, constructed from
recorded measurements rather than run against the server by this run: re-run
`artifacts/tuning/topk_prof_topk.py` in the container after the rebuild. **Engaged: ~8.4 µs at 64
rows. Not engaged: ~10.6 µs.** That is a 26% separation on a directly measured kernel, which is far
outside any timing ambiguity, and it is the check to prefer over the arm number.

## Accuracy gate

gsm8k 5-shot, 1319 problems, greedy, **lm-eval 0.4.12** with `--apply_chat_template` and the
harness's default `fewshot_as_multiturn=True`, task definition in `artifacts/harness/gsm8k.yaml`,
dataset shipped as parquet rather than downloaded. `artifacts/harness/run_eval.sh` is the exact
invocation.

| arm | instance | strict-match | flexible-extract | correct/1319 | vs gate 0.9691 |
| --- | --- | --: | --: | --: | --- |
| requirement | — | **0.9691** | — | — | — |
| baseline | earlier instance | **0.9772555 ± 0.0041066** | 0.9772555 ± 0.0041066 | 1289 | reference |
| A2 GEMM + topk | 1 | **0.9764973 ± 0.0041729** | same | 1288 | pass, +0.0074 |
| A2 GEMM + topk | 2 | **0.9764973 ± 0.0041729** | same | 1288 | pass, +0.0074 |
| **A3 + GDN + fmoe** | 1 | **0.9727066 ± 0.0044881** | same | 1283 | **pass, +0.0036 — 0.80σ** |

The gate is the baseline score less two standard errors. `FINDINGS.md` states it as **0.9691**;
0.9772555 − 2 × 0.0041066 computes to 0.96904, so 0.9691 is a conservative rounding, and since one
answer is worth 0.00076 nothing turns on the difference. **Quote 0.9691, as the bundle does.**

Three points a reader needs, in the order they matter:

**First, this harness resolves to about ±4 questions, and that number is measured, not assumed.** The
two gsm8k runs of the **identical** A2 build both scored 1288 — but *not on the same 1288 questions*,
with four answers flipping each way. Batch composition varies with arrival timing, that changes
reduction order, and the answer set moves with it. So A2's one-answer deficit against baseline is
nothing at all.

**Second, A3 is the only score here that moves further than that band.** It is 5 below A2 and 6 below
baseline, in the one arm that contains a change to the arithmetic, and the mechanism is named: the
fp4 stage-1 intermediate. It passes. It passes with **0.0036 of slack — barely more than the
±4-question band is wide** (4 answers is 0.0030; 0.0036 is 4.7 answers). That combination — a score
that moved past the harness's own resolution, in the arm where a named change reduces precision, with
slack of the same order as the resolution — is why the recommendation at the top of this entry is to
re-run the gate rather than to trust this row.

Which half of which arm can move an answer at all is worth being exact about, because it is what makes
the reading above more than a guess. The topk patch and the GDN patch are **bit-identical by direct
check** (30 and 8 shapes respectively, every tensor `torch.equal`), so neither can contribute. The
GEMM table *can*: it changes which kernel computes a dense bf16 matmul and therefore its reduction
order — and the arm containing it landed inside the ±4 band, twice. The fmoe row is the only change
in the stack that both alters precision and sits in an arm whose score moved.

**Third, the gate was run per arm, not once, and the duplicate run is what earned the band.** A2 was
gated on both of its server instances. The second run of an identical build looks redundant and is
not: it is the only way to learn your gate's own resolution, and without it A3's 6-answer drop would
have been unreadable. Copy the protocol.

For calibration against the nearest comparable case in this directory: Gemma-4-26B ships at **−0.35σ
against its requirement**, further out than this arm, and is argued as a pass on cross-machine grounds
— its local pristine base read +0.34σ *high*, so the whole spread sits inside the baseline envelope.
That argument is available there and is not available here: every gsm8k run in this entry — baseline
and both candidate arms — was scored on the same machine and the same stack, so the 6-answer drop
cannot be charged to a cross-machine baseline difference.

At 0.9773 there are 30 wrong answers in 1319. The gate is tight in both directions — a handful of
flipped answers trips it, and a change that breaks a narrow class of inputs has very little room to
hide.

## What was tried and did not work

| attempt | kernel / op-level result | end to end | verdict |
| --- | --- | --- | --- |
| **Raise the custom all-reduce one-shot/two-shot crossover to 1 MiB** (`patches/rejected/allreduce_1stage_crossover_1mib.patch`) | n/a — deployed directly | **−3.75%** (warm mean 2411.67 vs 2505.50), ~6× the 0.65% floor. TPOT 20.64 → 21.63 ms, median ITL 16.25 → 17.28 ms, TTFT unchanged | **Refuted, informatively.** The hypothesis was that the 512 KiB decode all-reduce at ~71 GB/s is latency-bound so the two-shot path's second barrier should cost more than the traffic it saves — and that aiter's 160 KiB threshold for `world_size ≤ 4` was inherited, since the legacy vllm path a hundred lines down the same file uses 512 KiB. On this node one-shot is *worse*: AMD's 160 KiB crossover is well chosen and the 512 KiB constant is the stale one. No gsm8k run; a regression does not need a correctness gate |
| **bf16 GEMM tuning for the TP=4 *decode* shapes only** | best case ≈204 µs per decode step per rank against a 17613 µs step = **1.16% of decode device time**, ≈0.7% of wall clock. Two of the seven shapes — the two that already hit tuned rows — produced candidates that lost to the incumbent | not measured alone — under the floor by construction | **Held, not discarded.** Folded into the prefill tuning where the same table miss costs more, and shipped as part of A1. This is why the shipped CSV has 13 rows and not 7 |
| **The whole TP all-reduce as a target** (16.6% of wall clock, the largest single line item) | measured p2p: each directed xGMI link carries **60.5 GB/s**, so ~181 GB/s egress per GPU. RCCL `all_reduce` on 134.2 MB: **1181.4 µs = 170.4 GB/s bus, 94% of the link ceiling**, flat from 33 MB up. quickreduce two-shot CodecQ8: **801.5 µs**, 1.47× RCCL, at ~73% of the ceiling because it compresses | headroom in the collective is ~210 µs per call ≈ 7.9% of prefill ≈ **2.9% of wall clock** | **Closed as a source of headroom.** `rocm-smi --showtopo` says full mesh, 1 hop, weight 15 — and the fabric measurement is what says otherwise. Capturing what is left means making the Q8 codec kernel faster, not choosing a different collective. INT4 would be worth ~5.8% but `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` is frozen and overriding it in code changes the arithmetic |
| **Beating the MoE expert GEMMs by finding untuned shapes** (30.2% of decode) | none — the shapes are already tuned. The lookup in `fused_moe.py:get_2stage_cfgs()` merges `qwen3_5_397b_fp4_tuned_fmoe.csv` over the default and the log shows exact tuned hits at every M tier | n/a | **Ruled out before spending an instance.** An early grep of the *default* `tuned_fmoe.csv` showed zero fp4/gfx950 rows and suggested headroom; that was the wrong file. Headroom is in the *ranking*, not the coverage — which is what attempt 4 exploited for ~1% |
| **"86.3% of decode is attention"** (the documented breakdown) | measured: attention of both kinds is **25.90%**, MoE is **40.97%** | n/a | **Refuted.** An analytic estimate on a dense-transformer model of a hybrid MoE. Anyone building on `reference/tracelens/` should re-derive from the trace first |
| **The 25774.2 tok/s roofline ceiling** (baseline = 9.7% of it) | not re-derived | n/a | **Treat as an upper bound of unknown tightness, not a target.** It is computed from aggregate HBM bandwidth across all four devices against ~17B active parameters, which is exactly the assumption that flatters a sparse MoE at TP=4 |

Two more surfaces were measured and left open rather than tried, and both are worth knowing before
you spend an instance on them:

- **aiter's FlyDSL Gated-DeltaNet decode kernel has a tuned row for exactly this shard**
  (`gfx950, b=64, sq=1, k_heads 4, v_heads 16, dims 128/128`) — but only the **bfloat16-state** row
  is fast: 17.28 µs against the Triton kernel's 28.88 µs. This server's mamba pool is **float32**
  state, and the tuned float32 row is 27.04 µs, a 6% edge that does not survive needing an unfused
  q/k/v split first. Halving the state to bf16 is `--mamba-ssm-dtype`, a frozen server flag, not a
  code axis.
- **Full attention at decode**, 12.89% of decode:
  `paged_attention_ll4mi_QKV_mfma16_kernel<… BLOCK_SIZE 1, HEAD_SIZE 256, NUM_THREADS 256,
  GQA_RATIO 8 …>` at 151.4 µs × 15 layers per step. With 2 KV heads replicated across TP=4 it reads
  64 seqs × ~8704 tokens × 256 dims × 2 B × 2 = 570 MB per layer per rank, i.e. **3.77 TB/s** against
  a measured achievable ~5.2 TB/s (a 2 GB `Tensor.copy_` sustains 5198 GB/s r+w,
  `artifacts/tuning/hw_hbm_bandwidth.py`). 72% of what the memory system will give; closing it is
  worth ~2.2% of wall clock and means reworking the gather pattern of a `page_size=1` KV cache. The
  largest remaining single-kernel target, and the hardest.

And the floor nobody can tune around: a decode step runs ~1400 kernels per rank and **nothing costs
less than ~4–5 µs however little it touches**. An 8 KB GEMV (`shared_expert_gate`, N=1, K=4096) costs
6.39 µs; a 512 KB rmsnorm costs 4.26 µs. About **15.8% of decode device time sits in kernels that
each move under 2 MB.** Fusion, not tuning, is what would move that.

## The transferable pattern: tuned constants stranded behind a predicate

`gpt-oss-120b/README.md` names this pattern and already cites this run as one of its independent
sightings. Stated once, from that entry: *a kernel ships with hand-tuned launch constants and a
predicate deciding who gets them; the constants are correct, the predicate is narrower than the
argument that justified them, and everything outside it silently takes a generic default that nobody
tuned.* Nothing logs anything. The server is fast, correct, and leaving a factor on the floor.

**Be precise about what this bundle actually contains, because the canonical form of the defect is
not here.** The literal "published launch config reachable only behind an SM100+/CUDA test, so the
untuned fallback is the only path that runs on CDNA" is documented in `gpt-oss-120b/` and
`gemma-4-26b-a4b-it/` (`_get_block_sizes_for_extend_attention`, gated on `128 < Lq <= 256`) and in
Kimi-K3's bundle. **No SM100 or CUDA-capability predicate appears anywhere in this bundle**; the
closest thing to one is `NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8, 16]` in the very file
the GDN patch edits, and nothing measured here turns on it. What this run found is three sightings of
the same *class*, with the arch key replaced by something else:

| instance here | the key that misses | consequence | worth |
| --- | --- | --- | --- |
| **A topology key instead of an arch key.** `qwen3_5_397b_bf16_tuned_gemm.csv` is a TP=2 tuning; the config is TP=4 | `(gfx, cu_num, M, N, K, …)` is exact, and TP halves every N and K | five of seven decode shapes and most prefill shapes fall through to a generic torch/hipblaslt call. 956 miss lines nobody was reading | **+1.60%** |
| **A pinned constant with an unpinnable key.** SGLang pins GDN `BV=32` through a single-config `@triton.autotune` | the autotune key `(H, K, V, BT, USE_GK, NT_BUCKET)` **does not contain `N`**, and `N` is what sets the grid | 64–128 workgroups on a 256-CU part at this shard; the pinned tile is sized for a regime this workload never enters | 1.20–1.35× on the kernel, ~0.5% end to end |
| **A platform predicate walling off a tuned table entirely.** aiter ships a HIP/FlyDSL implementation of the whole GDN op with its own `chunk_gdn_h_tuned.csv`, containing a tuned row for exactly this shard | SGLang's `GDNKernelDispatcher` **permits only Triton on ROCm** | the tuned rows are unreachable from this framework at any shape. This is the closest analogue here to the SM100+/CUDA form: a published tuned config that the deployment cannot reach because of a platform test | **not attempted** — the larger prize and an open thread |

The third row is the one to carry forward, because it is the form the pattern takes when the
framework rather than the kernel holds the predicate, and it is invisible from inside the kernel
code. **A skill whose job is "where are the tuning surfaces in this framework" should list the
reachable ones and the unreachable ones, because the second list is where a run loses a day.**

**The detection method, which is the reusable part.** All three were found by the same move: profile
the phase, then ask of every hot kernel *what is this launch's actual grid, and what does the shipped
config assume it is?* The GEMM one came out of resolving each shape's dispatch by hand against the
key; the GDN one came out of noticing that a `@triton.autotune` with one config is a hard-coded
constant wearing a costume; the dispatcher wall came out of reading what aiter ships and asking why
none of it appears in the trace. None of the three required a profiler feature — they required
reading a lookup key next to a shard shape.

*(The +1.60% and 1.20–1.35× figures above are from this bundle and checkable in it. The gpt-oss,
Gemma and Kimi-K3 figures are quoted from those entries and bundles and were not re-verified here.)*

## When this entry stops applying

Silently, in every case except the last:

| change | what happens | still reusable |
| --- | --- | --- |
| **arch ≠ gfx950 or CU count ≠ 256** | `gfx` and `cu_num` are literal columns in both keys; every GEMM and fmoe row is unreachable. The GDN rule also reads `multi_processor_count`, so its thresholds move | the shape list, the target ranking, the method |
| **TP ≠ 4** | every N and K shards differently; all 13 GEMM rows and both fmoe rows miss. This is the same defect the entry is about, one degree over | the *finding* transfers directly: check what TP the shipped table was tuned at |
| **concurrency ≠ 64** | decode M ≠ 64; the five decode GEMM rows and the fmoe token=64 row — the one carrying most of the win — go unreachable | the tier-derivation rule: decode M = concurrency |
| **`--chunked-prefill-size` ≠ 16384, or ISL ≠ 8192** | prefill M ≠ 8192/16384; the eight prefill GEMM rows and the fmoe token=16384 row miss. **And the GDN patch degrades**: `N` per chunk changes, so the analytic rule may select the shipped tile — correctly, but with no gain | the rule itself is analytic and will do the right thing at any grid; it just may have nothing to give |
| **`--attention-backend` ≠ aiter** | different kernels are hot; the profile in this entry does not describe your server | the phase split (36% prefill) probably survives; the kernel list does not |
| **a different aiter or SGLang commit** | the patches may not apply, and the kernel names in the engagement checks may not exist | everything except the diffs |
| **quantization ≠ quark/MXFP4, or a checkpoint with a different `quantization_config.exclude`** | the exclude list is what creates the dense bf16 bucket; a different one changes which shapes exist | the method for deriving the shape list from the exclude list |
| **stale `/tmp/aiter_configs`, no restart, or no `module_moe_asm` rebuild** | partial or total no-op, silently | — |
| **omitting the three duplicate-key deletions** | aiter raises at import. **Loud** | — |

## What the bundle does not record

Listed because `ENTRY_TEMPLATE.md` asks for each of them and a missing number is better than a
wrong one.

- **The container image digest.** Only the tag,
  `sglang:v0.5.17-rocm720-mi35x-profilerfix`. `reference/README.md` says the image is not recorded
  anywhere in the source session at all — `start_container.sh` calls its own tag an inference, and
  `FINDINGS.md` records the tag as what was confirmed inside the container. So the stack is pinned
  by a mutable tag. **This is the largest gap in the fingerprint**: two containers from that tag are
  not guaranteed to be the same stack, and the sibling Gemma run lost 4.4% of its baseline to an
  unnoticed Triton bump inside one.
- **The process environment.** No `/proc/<pid>/environ` dump. `ROCM_QUICK_REDUCE_QUANTIZATION=INT8`
  is asserted in prose and is load-bearing for the all-reduce conclusion; nothing else is
  enumerated. Capture it before you attribute anything.
- **A gsm8k reference from the source session.** None exists — the session recorded no accuracy
  figure for this model, so this run's own first clean measurement (0.9773) *is* the reference and
  the gate is derived from it. There is no independent confirmation that 0.9773 is this model's
  correct score on this harness.
- **The post-patch `not found tuned config` count**, so engagement check 3 asserts identities rather
  than a number.
- **Any engagement check for the topk patch.** The substitute above is constructed, not run.
- **A from-artifact reproduction**, which is the gap that keeps this entry off the "verified" bar.
- **An end-to-end measurement of the error-preserving fmoe alternative**, which is the swap the
  accuracy recommendation points at. Its operator-level number is measured; its throughput cost is
  arithmetic.
- **An isolated measurement of the topk, GDN or fmoe patches.** Only the GEMM table was ever run
  alone.

## Provenance

Task bundle: `tuning_workspace/experiment_standalone/hold_qwen35_397b_a17b_mxfp4_tuning/`.

| claim in this entry | where it comes from |
| --- | --- |
| headline, arms, per-run numbers, noise floor | `FINDINGS.md` §1 and §6; raw per-run JSON in `results/` (`align_*` = instance A and A′, `base_*` = B, `baseC_*` = C, `gemm_only_*`, `gemm_topk_*`, `final_*` = A2's two instances, `gdnmoe_*`/`gdnmoeB_*` = A3's two) |
| the one-line outcome | `EXPERIMENT_COMPLETE` |
| baseline provenance, the four baseline figures, the cold-run rule, the device-numbering trap | `BASELINE.md`; source session `/shared_nfs/hyperloom-claw/Qwen3.5-397B-A17B-MXFP4/20260817T023510Z`, whose rounds are in `reference/results/baseline_measure/` (2490.308) and `baseline_warmup/` (2489.937, plus the `ServerArgs` dump that every launch flag was taken from) |
| why the bundle was held, and that nothing needed redoing | `RESUMED.md` |
| the decode and prefill breakdowns | `FINDINGS.md` §3 and §5; traces and rankings in `analysis/prof_baseline/` and `analysis/prof_prefill/`, copied here as `artifacts/evidence/kernel_rank_*.json` |
| the TP=2 table finding and the shape probe | `FINDINGS.md` §4; `analysis/bf16_gemm/` |
| per-patch bases, apply commands, measurements, engagement checks | the header block of each file in `patches/`, reproduced verbatim in `artifacts/patches/` |
| the fmoe head-to-head and the numerics deltas | `analysis/fmoe/runcfg_{cand_shipped,cand_new_err0,tuned_full}.log`, copied to `artifacts/evidence/` |
| accuracy | `eval_results/{baseline_gsm8k_20260820_060854, gemm_topk_gsm8k_20260820_133923, final_gsm8k_20260820_142652, gdnmoe_gsm8k_20260820_152553}` — the four `results_*.json` files are the authority for every score and stderr quoted above |
| the rejected all-reduce arm | `patches/rejected/allreduce_1stage_crossover_1mib.patch`; `results/ar_1stage_512k/` |
| the skillset self-check | `analysis/skillset/claims_report_here.json`, copied here: **28 PASS, 1 FAIL, 8 N/A** of 37 claims. The one FAIL is a real correction — `tuning-aiter/SKILL.md` §7's "the shipped MX surface is FP4-only, MXFP8 has no aiter operator" is contradicted in this image, which ships both `dynamic_mxfp8_quant` and `gemm_afp8wfp8` |

Two multi-megabyte evidence files were left in the bundle rather than copied here:
`analysis/bf16_gemm/tune_prefill.log` (3.4 MB, the source of the 12-shape prefill compare table
above) and `analysis/bf16_gemm/profile_tp4_decode.csv` (4.2 MB, the raw tuner profile). Neither is
needed to deploy.

### Artifact index

| path | what it is |
| --- | --- |
| `artifacts/patches/gemm_tuned_table_tp4.patch` | the +1.60%: new TP=4 tuned-GEMM CSV plus the three-line duplicate-key deletion |
| `artifacts/patches/topk_softmax_ldg_width.patch` | topk-softmax load-width dispatch (aiter `topk_softmax_kernels.cu`) — needs the `.so` rebuild |
| `artifacts/patches/gdn_chunk_h_launch_config.patch` | GDN state-update launch geometry (sglang `chunk_delta_h.py`) — half of A3 |
| `artifacts/patches/fmoe_tuned_rows_tp4.patch` | re-raced fused-MoE rows for token 64 and 16384 — the other half, and the numerics risk |
| `artifacts/patches/rejected/allreduce_1stage_crossover_1mib.patch` | the −3.75% crossover change, with its four runs in the header |
| `artifacts/patches/BUNDLE_PATCHES_README.md` | the bundle's own `patches/README.md`, kept because it states the header convention every patch here follows: base commit, apply command, measurement, engagement check |
| `artifacts/tables/qwen3_5_397b_bf16_tuned_gemm_tp4.csv` | the 13 deployable rows, extracted from the patch as a drop-in file |
| `artifacts/tables/tuned_tp4_{decode,prefill}.csv` | full tuner output, including the two decode rows that were **not** shipped because they lost to the incumbent |
| `artifacts/tables/untuned_tp4_{decode,prefill}.csv` | the shape lists fed to the tuner |
| `artifacts/tables/fmoe_replacement_rows_tp4.csv` | header plus the two rows the fmoe patch installs. **Not a drop-in file** — these replace two rows inside the shipped `qwen3_5_397b_fp4_tuned_fmoe.csv` |
| `artifacts/tables/fmoe_cand_new_err0.csv` | **the error-preserving swap** the accuracy recommendation points at |
| `artifacts/tables/fmoe_cand_shipped.csv` | what the shipped rows measure on this build — the baseline for the head-to-head |
| `artifacts/tables/fmoe_tuned_full.csv` | the fp4-at-both-tiers result, of which only the token=64 row was deployed |
| `artifacts/tables/fmoe_{tuned_tp4_shipped_rows,untuned_tp4_shapes,profile_full}.csv` | the shipped rows as found, the two-tier shape list, and the candidate profile |
| `artifacts/harness/{start_container,preflight,launch_server,run_bench,run_eval}.sh`, `gsm8k.yaml` | the measurement contract, verbatim. `launch_server.sh` must print `config verified` before any number counts |
| `artifacts/tuning/bf16_gemm_probe.py` | per-shape probe through the production `gemm_a16w16` entry point — the µs/call and GB/s table above |
| `artifacts/tuning/gdn_bench_chunk_h.py`, `gdn_check_chunk_h.py` | the `(BV, warps, stages)` sweep at the TP=4 shard, and the 8-shape bit-identity check |
| `artifacts/tuning/topk_prof_topk.py` | the device-time measurement that decided the topk patch — and the substitute engagement check |
| `artifacts/tuning/topk_{check,dump}_topk.py`, `topk_calib_elementwise.py` | reference comparison, the 30-case bit-identity dump, and the 4.40 µs elementwise floor the 10.6 µs kernel was judged against |
| `artifacts/tuning/topk_bench_topk_WRONG_METHOD.py` | kept deliberately: a cuda-event loop around the python wrapper reports a flat ~19 µs at every M because it times dispatch |
| `artifacts/tuning/build_moe_asm.py` | the aiter JIT rebuild, required for the topk patch |
| `artifacts/tuning/{allreduce_bench_ar,allreduce_bench_rccl,hw_hbm_bandwidth,hw_p2p_bandwidth}.py` | the 60.5 GB/s-per-link, 170.4 GB/s-RCCL and ~5.2 TB/s rooflines the open threads are scored against |
| `artifacts/tuning/{profile_decode,profile_prefill,group_rank,show_results}.py` | live-server capture on all four ranks, kernel ranking by summed device duration, arm tables |
| `artifacts/evidence/runcfg_*.log` | aiter `--run_config` head-to-heads: the E2E column and the numerics deltas quoted above |
| `artifacts/evidence/fmoe_tune_first_attempt.log` | the `KeyError: "['gfx'] not in index"` that still writes the old rows to `-o` |
| `artifacts/evidence/{fmoe_tune_full,bf16_gemm_tune_decode}.log`, `bf16_gemm_probe_baseline.json`, `kernel_rank_{decode,prefill}.json` | tuner and profile evidence for the tables above |
| `artifacts/evidence/claims_report_here.json` | the skillset self-check run in this image |
