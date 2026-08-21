# Qwen3-14B-FP8 on MI355X — SGLang, TP=1, three source patches to aiter and sglang

**Measured win: +29.05% output throughput** (1538.009 → 1985.566 tok/s), gsm8k 5-shot 0.9454 → 0.9454,
**identical on all 1319 problems**. Three patches carry it: a tuned CK block-scale / B-preshuffle
GEMM table for this model's four linear shapes (+23.18%), selecting aiter's EXPERIMENTAL
`pa_ragged` decode-attention kernel by default (+3.07%), and taking the B-preshuffle FP8 scale
layout straight out of the quant kernel instead of a transposing copy (+1.42%). No flag, no
environment variable and no workload change is involved in any of them.

Found 2026-08-20 over a single-day run on `crsuse2-m2m-115`.

## Reproduction status — read this before the number

**This is the strongest-evidence entry in the directory on measurement discipline, and it still has
not cleared the house bar of "re-deployed from the artifact on a clean instance."** What was
actually done:

- The headline was measured **last**, after all three patches were frozen, as an interleaved
  across-restart A/B: A B A B, four fresh server instances, three benchmark runs each, compared
  position-matched. **Six of six position-matched pairs positive, +28.91..+29.19%, and the two arms'
  distributions are disjoint** — arm A spans [1533.060, 1540.694] and arm B spans
  [1978.323, 1987.298].
- Arm B was installed by copying the exported files (`artifacts/ab/stack_B.sh`), so **the win was
  reproduced twice from exported artifacts across a restart** — but on the same machine, the same
  container and within the same session, not on a clean instance.
- Arm A restores the pristine stack file-by-file and lands at 1538.009 median, **0.09% from the n=8
  baseline mean measured hours earlier the same day** (1536.610); its first run, 1540.694, is 0.006%
  from the untouched stack's own first run (1540.594). The restore is exact and the machine returns
  to the same place, which is most of why the A/B is trustworthy.
- The run states it re-verified all three patches, after their final header edits, to apply cleanly
  to pristine copies of their bases and to reproduce the measured working trees byte-for-byte.
  **Re-checked while writing this entry**, against the artifacts in this directory: patches 2 and 3
  apply cleanly to `artifacts/patches/base/` and produce files byte-identical to the arm-B snapshots
  `artifacts/ab/pa_ragged.B.py` and `artifacts/ab/fp8_utils.B.py`; patch 1's payload is
  byte-identical to `artifacts/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen3_14b.csv`; and the arm-A
  snapshots are byte-identical to the archived pristine bases. **The artifact-to-measured-code path
  is closed. The artifact-to-clean-instance path is not.**

So: **reproduced twice across restarts within the run from the exported files, never re-deployed
from the patch files on a clean instance.** The magnitude makes that gap much less alarming than it
would be elsewhere in this directory — +29.05% is 138× the 0.21% position-matched restart floor, and
16× the worst ambient drift the run ever observed — but the check is still owed. See
[What would promote this entry](#what-would-promote-this-entry) for what closing it takes.

## Environment fingerprint

| field | value | load-bearing? |
| --- | --- | --- |
| GPU | 1× MI355X, `gfx950`, **256 CU**, one of eight devices on the node (`renderD128`) | **yes** — the CSV rows are keyed `(gfx, cu_num, M, N, K)`, and `cu_num=256` is a literal column |
| container | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix` — **digest not recorded** | descriptive, pins the stack |
| SGLang | 0.5.17, git `29481685462732237d80d86076d6563e1f658102` | **yes** for patch 3 — that sha is the diff base |
| aiter | git **`d9e5ef7ce08ee7045d583aed768cff41aa9210fe`** at `/sgl-workspace/aiter`, source checkout with a working tuner | **yes** for patches 1 and 2 — schema, merge path and the `pa_ragged.py` base |
| ROCm / torch / Triton / python | 7.2.0 / 2.9.1+rocm7.2.0.git7e1940d4 / 3.6.0 / 3.10.12 | descriptive |
| model | Qwen3-14B-FP8, **TP=1** — 40 layers, 40 q heads / 8 kv heads, head_dim 128, 4 safetensors shards | **yes** — TP=1 is what makes N and K unsharded, so the four tuned `(N,K)` pairs are what they are |
| weights | FP8 checkpoint; SGLang resolves the quantization itself (`quantization=None` in the reference server args, no `--quantization` flag) | **yes** — selects the `a8w8_blockscale_bpreshuffle` table |
| KV cache | **bf16** (no `--kv-cache-dtype` flag; the default) | **yes** — patch 2's guard requires `kv_dtype == "__hip_bfloat16"`; on an FP8 KV cache it silently stays on GOLDEN |
| attention backend | `aiter`, `--page-size 1`, radix cache disabled | **yes** — `--attention-backend aiter` is what routes decode through `pa_ragged` at all |
| host | `crsuse2-m2m-115` | descriptive |

**Where the fingerprint is thin, and why it matters.** The **container digest is not recorded** —
only the tag, which is mutable. `scripts/start_container.sh` says outright that the image is not
recorded anywhere in the source session's state, so the tag is the run's own choice, not something
inherited. Anyone matching this environment can pull the same tag and get different bytes, and the
sibling Gemma experiment in this directory lost 4.4% of its baseline to exactly that kind of
unremarked bump. The two framework commits, by contrast, **are** recorded, which is a step up on
DeepSeek's wheel-only aiter.

**The environment variables were never dumped, and that is the second gap.** `launch_server.sh` sets
none and `BASELINE.md` says "no environment variables beyond the harness defaults", but the run did
not record `cat /proc/<pid>/environ | tr '\0' '\n'` for the live server, so what the *image* exports
is unknown. This matters concretely rather than pedantically: **patch 2 reads
`os.getenv("QKV_VERSION")` and lets it override the new default**, so an image that exports
`QKV_VERSION=GOLDEN` makes patch 2 inert with no warning. Dump the environment before deploying.

## Launch configuration

Exactly this, inside the container, from `artifacts/harness/launch_server.sh`:

```bash
python3 -m sglang.launch_server \
    --model-path /shared_nfs/hyperloom/models/Qwen3-14B-FP8 \
    --host 0.0.0.0 --port 43102 \
    --tp-size 1 \
    --context-length 11264 \
    --watchdog-timeout 1800 \
    --mem-fraction-static 0.68 \
    --chunked-prefill-size 16384 \
    --page-size 1 \
    --disable-radix-cache \
    --attention-backend aiter
```

**No environment variables are set by the launch script.** Every flag comes from the ServerArgs of
the reference measurement's own server log, including the ones that do not look like tuning knobs.

The script verifies the live server against `/get_server_info` and refuses to let you benchmark on a
mismatch: `context_length=11264`, `tp_size=1`, `attention_backend=aiter`,
`chunked_prefill_size=16384`, `disable_radix_cache=true`, `page_size=1`. One resolved value is not
what you asked for and the check accommodates it deliberately: **SGLang rescales
`mem_fraction_static` by 0.85 on builds that combine aiter with a context length above 8192**, so
both 0.68 and 0.578 are legitimate readings and only some other value is a fault.

**Why the config check earns its place here:** `--chunked-prefill-size 16384` and `--page-size 1`
are not incidental. The first sets the prefill M values the GEMM table is tuned at (8192 and 16384);
the second is what makes decode attention a page_size-1 ragged paged read, which is the shape patch
2's kernel is written for. A server that came up with the source session's `--page-size 1024`
instead would pass a naive "is it serving?" test and reach neither win.

## Workload

ISL 8192, OSL 1024, concurrency 64, 192 prompts, **8 warmups**, seed 0, `random` dataset with
`random_range_ratio 1.0`, `--random-prefix-len 0`, `--ignore-eos`, InferenceX `benchmark_serving`
against `/v1/completions`. Frozen; `artifacts/harness/run_bench.sh` is the contract.

What sets the tuned shapes:

| workload parameter | shape it produces |
| --- | --- |
| concurrency 64 | decode **M=64**, hit on every token step — all four decode rows |
| `--chunked-prefill-size 16384` + ISL 8192 | prefill **M=8192** and **M=16384** — all eight prefill rows |
| TP=1, Qwen3-14B geometry | the four `(N,K)` pairs: 7168/5120 (qkv), 5120/5120 (o), 34816/5120 (gate_up), 5120/17408 (down) |

Change concurrency and the decode rows go unreachable; change `--chunked-prefill-size` and the
prefill rows do. Both fail silently.

The run is **not decode-only**: at 192 prompts and concurrency 64 it is three decode waves (≈78 s)
plus ≈50 s of prefill, so prefill is ~39% of the benchmark at baseline and a prefill-only change
moves the headline. That is why patch 1, whose largest rows are prefill `cktile` rows, shows up as a
34.90% TTFT improvement as well as a throughput one.

## Baseline and noise floor

### The arms

| | value |
| --- | --- |
| stock, this stack (n=8 across two instances and two restarts) | **1536.610** tok/s mean, stdev 3.917 = 0.255% |
| pristine arm of the final interleaved A/B (n=6, two fresh instances) | **1538.009** median |
| with all three patches (n=6, two fresh instances) | **1985.566** median |
| **delta** | **+29.05%** |

The +29.05% is the **median of the six position-matched pair deltas**, not the ratio of the two
medians. The full table:

| pair | A (pristine) | B (3 patches) | delta | median ITL |
| --- | --: | --: | --: | --: |
| A1/B1 run 1 | 1540.694 | 1986.147 | +28.91% | −14.17% |
| A1/B1 run 2 | 1537.989 | 1986.961 | +29.19% | −14.36% |
| A1/B1 run 3 | 1533.060 | 1980.168 | +29.16% | −14.41% |
| A2/B2 run 1 | 1540.412 | 1987.298 | +29.01% | −14.14% |
| A2/B2 run 2 | 1538.029 | 1984.984 | +29.06% | −14.40% |
| A2/B2 run 3 | 1533.195 | 1978.323 | +29.03% | −14.30% |

Secondary metrics, medians of n=6 each: mean TTFT 8586.6 → 5572.5 ms (−35.10%), mean TPOT
33.249 → 26.804 ms (−19.38%), median ITL 25.431 → 21.780 ms (−14.33%), mean E2E latency
42599.9 → 32995.0 ms (−22.55%).

### The floor — three spreads, and the one that bites is not in the template

| noise floor | how measured | spread |
| --- | --- | --: |
| repeating the benchmark within one process | runs 1→3 of instance A, and of instance B | **0.50%** and **0.51%** |
| **restart-to-restart, position-matched** | run 1 of each of four fresh instances | **0.21%** |
| pooled / unmatched | n=8 min–max, 1529.512 … 1540.594 | 0.72% |
| **ambient drift across hours** | byte-identical code on two fresh instances about an hour apart | **1.8%** |

**Which floor applies: the 0.21% position-matched restart spread**, because all three changes need a
server restart (`../../tuning-core/measurement.md` Rule 3b) — and it applies *only* to
position-matched comparisons. Every arm in this campaign was run as positions 1–3 of a freshly
started server, so every comparison is like-for-like.

**The within-instance spread is not scatter, it is monotone decline, and it runs the opposite way to
the guidance.** Instance A: 1540.594 → 1537.683 → 1532.946. Instance B: 1537.399 → 1534.810 →
1529.512. Six runs, two instances, **zero inversions**, ~0.25% lost per successive run on the same
server; the same decline is visible in all four instances of the headline A/B and it **cancels
exactly under position matching**. `rocm-smi --setperfdeterminism` exits 0, prints nothing and leaves
`--showperflevel` reading `auto` in this container, so the clock cannot be pinned — it was sampled
instead, and SCLK free-ran between **1901 and 2398 MHz** under sustained load. Every other entry in
this directory discards run 1 as cold. **Here run 1 is the fastest run you will get.** Do not apply
the ramp-up model to this stack; compare position-matched or not at all. Run 1 vs run 1 resolves
0.21%; run 1 vs run 3 needs more than 0.72% and gets the sign wrong for free.

**The spread that actually ruined a measurement is the hour-scale one, and it is episodic.**
Byte-identical code read 1974.365 on one fresh instance (`results/gemm_pa_r2_*`) and 1939.687 on a
later one (`results/scaleab_A1_r1_*`) — a **1.8% loss with no code change**, 8.6× the restart floor
and larger than patches 2 and 3 combined. It is not secular: the pristine arm of the final A/B, run
later still, came back to 1540.694 against the untouched stack's earlier 1540.594 — within 0.006%.
So neither a linear drift correction nor "measure the baseline again at the end" works, and
**interleaving is the only defence**. It earned its keep twice here — see
[What was tried](#what-was-tried-and-did-not-work).

*(A provenance wrinkle worth knowing if you audit this: `FINDINGS.md` §1 annotates these runs with
wall-clock times — "09:20", "10:50", "08:10" — that do not line up with the timestamps on the
corresponding `results/` directories. The throughput figures and their ordering are consistent
everywhere; only the clock annotations disagree, so this entry cites runs rather than times.)*

**Are the arms disjoint? Yes, and not marginally.** A ∈ [1533.060, 1540.694], B ∈ [1978.323,
1987.298]; the gap between arm A's best and arm B's worst is 437.6 tok/s, against a within-arm
restart spread of a few tok/s. Both marginal patches are disjoint too, and those are the ones where
it matters: patch 2's arms are A max 1928.242 < B min 1973.383, and patch 3's are
A ∈ [1931.262, 1939.737] against B ∈ [1957.759, 1967.707].

Gain ÷ floor for this entry is **138×**, the highest in this directory — ahead of Gemma's 100×, and
more than an order of magnitude clear of every other entry.

## What carries the win

Three patches, in `artifacts/patches/`. They are independent in code but were measured as a stack;
apply 1, then 2, then 3.

Everything the run produced that is deployable, auditable or re-runnable is checked in here:

| path | what it is |
| --- | --- |
| `artifacts/patches/000{1,2,3}-*.patch` | the three patches; each header carries its base commit, apply command, its own measurement and its engagement check |
| `artifacts/patches/PATCHES.md` | the run's own patch manifest — order, bases, and why there is no `rejected/` |
| `artifacts/patches/base/aiter-d9e5ef7c-pa_ragged.py`, `artifacts/patches/base/sglang-29481685-fp8_utils.py` | the pristine diff bases, archived because **neither working tree was a clean checkout** — both carry unrelated local modifications, so `git diff` against the tree is not meaningful |
| `artifacts/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen3_14b.csv` | the 12 tuned rows, i.e. the whole of patch 1, as a droppable file |
| `artifacts/ab/stack_{A,B}.sh` | install the pristine and the patched stack; `stack_B.sh` is how the headline was actually installed |
| `artifacts/ab/{pa_ragged,fp8_utils}.{A,B}.py` | the exact file snapshots those scripts copy |
| `artifacts/ab/run_ab.sh`, `artifacts/ab/run_ab_stack.sh`, `artifacts/ab/report_ab.py` | the interleaved A/B driver and its position-matched reporter |
| `artifacts/ab/sample_clocks.sh` | samples junction temp / memory temp / SCLK once a second, because the clock cannot be pinned in this container |
| `artifacts/analysis/bench_pa.py`, `bench_pa_partition.py`, `bench_scale_layout.py` | the three isolated microbenchmarks — decode attention, the partition-size sweep, the scale layout |
| `artifacts/analysis/profile_decode.py`, `profile_prefill.py`, `summarize_trace.py` | the profiler drivers and the trace summariser that produced the breakdowns below |
| `artifacts/gemm_tune/untuned_{decode,prefill}.csv` | the tuner **inputs** — re-tune from these at a different operating point |
| `artifacts/gemm_tune/tuned_{decode,prefill}.csv`, `tuned_decode_splitk.csv` | tuner outputs, including the split-K run that was rejected |
| `artifacts/gemm_tune/profile_{decode,prefill,decode_splitk}.csv` | every candidate timing — 228 decode and 456 prefill candidates, plus 792 rows from the 396-candidate split-K sweep (two timing passes each) |
| `artifacts/harness/{launch_server,run_bench,run_eval,preflight}.sh` | the launch, benchmark, gate and stack-verification scripts, unmodified from the bundle — they are the measurement contract |

### Patch 1 — tuned CK block-scale / B-preshuffle GEMM table, +23.18%

`artifacts/patches/0001-aiter-tuned-gemm-table-qwen3-14b-gfx950.patch`, whose entire payload is
`artifacts/a8w8_blockscale_bpreshuffle_tuned_gemm_qwen3_14b.csv` — **12 rows, one new file** under
`aiter/configs/model_configs/`, picked up by the existing glob in `aiter/jit/core.py`.

aiter ships tuned tables for dsv3, qwen3_235b, glm5.2 and mm2.5 and **none for this model**, so every
one of its linear shapes logged `not found tuned config in
/tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv, will use default config!`. The run
used aiter's own tuner (`csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py --preshuffle`)
over the four shapes at the three M values the frozen workload produces, timing **228 decode and 456
prefill candidates**; all candidate timings are kept in `artifacts/gemm_tune/profile_{decode,prefill}.csv`.

| rows | winner | per-call µs (tuner) |
| --- | --- | --: |
| M=64, 5120×5120 / 5120×17408 / 7168×5120 | `ck` kernel 12 | 13.9587 / 39.176 / 14.6796 |
| M=64, 34816×5120 | `ck` kernel 17 | 47.6302 |
| M=8192 and M=16384, all four shapes | `cktile` kernels 11, 27, 28, 29 | 224.7 … 2634.4 |

**All twelve rows tuned at `errRatio 0.0`** — pure kernel selection, no numerics change, which is
why accuracy is unchanged to the digit.

Measured alone: 1536.610 → **1892.866** tok/s (n=3: 1896.075 / 1893.410 / 1889.113), **+23.18%**
(the patch header rounds the same ratio to +23.19%). TTFT 8570.5 → 5579.8 ms (−34.90%), median ITL
25.514 → 23.337 ms (−8.53%). Re-profiling in situ shows decode GEMM **6.876 → 4.613 ms/step
(−32.9%)**, against the tuner's predicted 171.9 → 115.4 µs/layer = 4.62 ms/step — **prediction and
in-situ measurement agree to 0.2%**.

That agreement is worth dwelling on, because the neighbouring `qwen3-8b/` entry warns that the
tuner's warm-cache µs can be a lie at large M. It was not one here, at M=64. The honest boundary is
narrower than "distrust the tuner": distrust it **at large M**, where one weight pass dominates and
the cache state is the whole question.

**This is the one figure not measured interleaved**, and it does not need to be: at 110× the restart
floor and 13× the worst observed drift, no drift story reaches it. It is also corroborated from the
other end — dividing the two interleaved marginals out of the headline implies **+23.45%** for this
patch, 0.2% from the directly measured figure.

### Patch 2 — select the EXPERIMENTAL `pa_ragged` decode kernel by default, +3.07%

`artifacts/patches/0002-aiter-default-experimental-pa-ragged-kernel.patch`, ~20 lines of Python in
`csrc/cpp_itfs/pa/pa_ragged.py`.

aiter ships **two** implementations of `paged_attention_ll4mi_QKV_mfma16_kernel` behind a
compile-time `VERSION_ID`. `VERSION_ID=1` ("EXPERIMENTAL") replaces GOLDEN's K/V fetch — 16 lanes
each striding to a different token 2048 B apart — with 16 B-per-lane non-temporal loads staged
through a double-buffered LDS tile. It was reachable only through an **undocumented `QKV_VERSION`
environment variable**, so it is dark in every default deployment, and its shape guard (head_size
128, bf16 KV) is exactly this model's decode shape. The patch selects it by default where its
preconditions hold, with a guard deliberately **narrower** than the kernel's stated support — alibi
and logits-soft-cap take separate, unvalidated paths inside it and stay on GOLDEN. It also fixes a
real bug: the old guard printed "Fallback to original kernel" on an unsupported shape and then never
fell back.

Isolated first (`artifacts/analysis/bench_pa.py`, CUDA-graphed, bs=64 / ctx=8704 / 40 q heads / 8 kv
heads / page_size 1), over three page-table layouts so a win cannot be an artifact of a tidy page
table:

| page-table layout | GOLDEN | EXPERIMENTAL | delta |
| --- | --: | --: | --: |
| contiguous | 404.7 µs | 377.1 µs | −6.8% |
| interleaved (what a page_size-1 free list actually produces) | 410.2 µs | 384.5 µs | −6.3% |
| shuffled | 420.8 µs | 391.2 µs | −7.0% |

~5.6 → ~6.0 TB/s. The two versions JIT to different build hashes, so they cannot alias each other.

End to end, interleaved with patches 1 and 3 held identical in both arms: **median +3.07%, range
+2.84..+3.25%, 6/6 pairs positive, arms disjoint** — 15× the floor. **TTFT is flat** while median ITL
falls 4.26%, which is precisely the signature a decode-attention change should have and is most of
why it is believable. Cross-check: −4.26% of a ~22.5 ms step is ~0.96 ms/step saved, against the
microbenchmark's predicted 40 layers × 27.6 µs = 1.10 ms/step — same size, slightly under, which is
the direction that does not need explaining away.

**The caveat the run put on the record, and this entry keeps:** the identical kernel selection is
reachable at runtime with `QKV_VERSION=EXPERIMENTAL`. It was landed as a source change because the
frozen-configuration rule forbids touching the environment, and the measured run's environment is
untouched — but a reviewer who considers that too close to an env-var change can **drop patch 2 and
keep patch 1's +23.18% at bit-identical accuracy**. The two are independent.

### Patch 3 — B-preshuffle scale layout straight from the quant kernel, +1.42%

`artifacts/patches/0003-sglang-bpreshuffle-scale-from-quant-kernel.patch`, ~6 lines in
`python/sglang/srt/layers/quantization/fp8_utils.py`.

On gfx95 with B-preshuffle, `aiter_w8a8_block_fp8_linear` quantised the activation to a row-major
`[M, G]` scale and then rearranged it with `scale.t().contiguous().t()` — a real transposing copy
kernel, launched **once per FP8 GEMM, four per layer, 40 layers**: 163 calls and **648.5 µs per
decode step**, 2.9% of the step. `per_group_quant_hip` can emit those bytes directly through its
`transpose_scale` argument, leaving only a stride reinterpretation.

**Nobody had done it because a comment said it was impossible.** aiter's own
`aiter/dist/device_communicators/communicator_cuda.py:541` asserts that with `transpose_scale=True`
the kernel "returns a contiguous (M, num_groups) buffer with SHUFFLED bytes — a different physical
arrangement". That comment is **wrong for this shape family**, and checking it was the whole job: for
(64, 5120) and (64, 17408) the quantised tensor is `torch.equal` either way, and
`torch.as_strided(s, s.shape, (1, M))` on the `transpose_scale=True` scale is `torch.equal` to
`s.t().contiguous().t()` with matching raw storage bytes. Identical values behind identical strides —
the GEMM cannot tell. Isolated (`artifacts/analysis/bench_scale_layout.py`, interleaved, 9×50):
quant+layout 4.43 → 2.49 µs at (64, 5120) and 4.75 → 2.68 µs at (64, 17408), −44%, output equal.

End to end, interleaved: **median +1.42%** (+1.19, +1.44, +1.37, +1.53, +1.43, +1.42), median ITL
−1.96%, arms disjoint, 6.8× the floor. Confirmed in the trace rather than from a log line:
`direct_copy_kernel_cuda<float>` goes 163 calls/step → ~0, `dynamic_per_group_scaled_quant` goes
660.7 → 675.9 µs/step (the shuffled write costs 15.2 µs/step and is worth it), net −620 µs/step =
−2.75% predicted against −1.96% measured.

**Scope:** the `input_scale is not None` branch is untouched, because there the caller already owns
the scale and the layout is not ours to choose.

## Deploy

Inside the container, from a checkout of this entry's `artifacts/`:

```bash
# 1. aiter — patches 1 and 2
cd /sgl-workspace/aiter
git apply -p1 <artifacts>/patches/0001-aiter-tuned-gemm-table-qwen3-14b-gfx950.patch
git apply -p1 <artifacts>/patches/0002-aiter-default-experimental-pa-ragged-kernel.patch

# 2. sglang — patch 3
cd /sgl-workspace/sglang
git apply -p1 <artifacts>/patches/0003-sglang-bpreshuffle-scale-from-quant-kernel.patch

# 3. CACHE INVALIDATION — mandatory, and the deploy is a silent no-op without it
rm -rf /tmp/aiter_configs
find /sgl-workspace/aiter/csrc/cpp_itfs/pa /sgl-workspace/sglang/python/sglang/srt/layers/quantization \
     -name '__pycache__' -prune -exec rm -rf {} +

# 4. RESTART — mandatory. The first start after patch 2 JIT-builds the new pa_ragged module.
<workdir>/scripts/launch_server.sh --stop
<workdir>/scripts/launch_server.sh
```

Then run the [engagement check](#engagement-check) before believing any number.

`artifacts/ab/stack_B.sh` is the alternative install path, and it is the one the headline was
actually measured through: it copies the tuned CSV and the two patched source files into place and
does the `rm -rf /tmp/aiter_configs`. `artifacts/ab/stack_A.sh` is its exact inverse and is what you
want for a control arm.

**A restart is required, and there is no live drop-in.** aiter merges its config tables at import,
SGLang captures decode CUDA graphs at startup, and `pa_ragged`'s JIT module is resolved at import.
A live drop-in benchmarks perfectly and measures nothing.

**On Triton's cache:** nothing in this stack's win path is a Triton kernel — the GEMMs are `ck` and
`cktile`, the attention is a JIT-compiled HIP kernel, and patch 3 removes a PyTorch copy. So
`~/.triton/cache` is not a hazard *for these three patches*. It is still worth clearing if you have
been experimenting on the Triton GEMM path, since patch 3's branch is gated on `not use_triton`.

### Every way this deploy silently does nothing

Each of these produces a clean, plausible, wrong number rather than an error.

| failure | what you see | which patch dies |
| --- | --- | --- |
| **stale `/tmp/aiter_configs`** | the merged table is derived and is *not* regenerated if it already exists; the new rows never appear. In the other direction, removing the CSV without dropping the cache leaves a "pristine" arm still using the tuned table | 1 |
| **no server restart** | graphs captured, tables merged and modules imported at startup | all three |
| **stale `__pycache__`** for `pa_ragged.py` or `fp8_utils.py` | the old module is served | 2, 3 |
| **`QKV_VERSION` exported by the image or the shell** | patch 2 explicitly lets the env var override its new default; `QKV_VERSION=GOLDEN` reverts it | 2 |
| **KV cache not bf16** (e.g. someone adds `--kv-cache-dtype fp8`) | the guard is false, GOLDEN is selected, no message | 2 |
| **head_size ≠ 128**, alibi, or logits soft cap | same — the guard is deliberately narrow | 2 |
| **arch ≠ `gfx950`, or CU count ≠ 256** | both are literal columns in the CSV key; every lookup misses and falls back to the default config | 1 |
| **concurrency ≠ 64** | decode M ≠ 64, all four decode rows unreachable | 1 |
| **`--chunked-prefill-size` ≠ 16384 or ISL ≠ 8192** | prefill M ≠ 8192/16384, all eight prefill rows unreachable | 1 |
| **TP ≠ 1** | N and K shard, so none of the four `(N,K)` pairs exist | 1 |
| **`--attention-backend` ≠ `aiter`** | decode never enters `pa_ragged` | 2 |
| **`--page-size` ≠ 1** | a different decode attention path and a different KV layout; note the source session's own final config used `--page-size 1024` | 2 |
| **sglang installed non-editable** | editing `/sgl-workspace/sglang` has no effect on the imported package | 3 |
| **`_use_aiter_bpreshuffle_gfx95` false, or `use_triton` true** | patch 3's branch is not taken | 3 |

## Engagement check

Three checks, one per patch, each with both directions. All three were run on the live server in
both arms of the headline A/B rather than assumed from the install.

**Patch 1 — the merged table grew by exactly 12 rows, and the misses stopped:**

```bash
wc -l /tmp/aiter_configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv
grep -c "M:64, N:" /tmp/sglang_server_qwen3_14b_fp8.log
```

- **Not engaged:** `3456` rows; `4` miss lines, one per decode shape.
- **Engaged:** `3468` rows — exactly +12, the cheapest possible confirmation the merge took — and
  `0` miss lines.

Note the miss line is *unconditional*, so this half needs no logging flag. Do not treat a
non-zero count as failure without reading it: **`M:512` and the rest of the 1..512 CUDA-graph
capture ladder still log misses by design** (see below), so grep for `M:64` specifically.

**Patch 2 — kernel identity off the loaded object of the live measured process:**

```bash
PID=$(pgrep -f "sglang::scheduler" | head -1)
grep -o "pa_ragged_[0-9a-f]*" /proc/$PID/maps | sort -u
```

- **Not engaged (GOLDEN):** `pa_ragged_1eda14407ca86673242536de3e4e5472`
- **Engaged (EXPERIMENTAL):** `pa_ragged_718e0ba0dd41ab1acefe187f6a0c6fac`

**This is the best check in this directory and it is not one of the four forms in
`../../tuning-core/engagement_verification.md`.** aiter hashes its build inputs into the module name,
so the build hash *is* kernel identity — read off the live measured process, with no profiler, no
logging flag, no side run and no perturbation of the thing being measured. For any library that
hashes its build inputs into the module name, it is strictly better than form 4 (profiler evidence),
and it belongs in that page as form 5. Use `pgrep -f "sglang::scheduler"` rather than the launcher
PID: the scheduler is the worker that holds the KV cache and loads the kernel.

**Patch 3 — from a decode trace** (`artifacts/analysis/profile_decode.py` +
`artifacts/analysis/summarize_trace.py`), because there is no log line to read:

- **Not engaged:** `at::native::direct_copy_kernel_cuda<float>` (unroll<128,4>) at **163 calls/step,
  648.5 µs/step**, and `dynamic_per_group_scaled_quant` at 660.7 µs/step.
- **Engaged:** `direct_copy_kernel_cuda<float>` at **~0 calls/step**, and
  `dynamic_per_group_scaled_quant` up to 675.9 µs/step.

The rise in the quant kernel is the two-sided half: it is the shuffled write, it costs 15.2 µs/step,
and if it does *not* rise while the copies disappear, something other than this patch changed.

## Accuracy gate

gsm8k 5-shot, **lm-eval 0.4.12** (`lm-eval[api]==0.4.12` in a venv of its own), task version 3.0,
1319 problems, chat template applied, `max_tokens=9216, temperature=0, top_p=1`, seeds
`0,1234,1234,1234`, `num_concurrent=64`, `max_length=11264`, served over
`/v1/chat/completions`. `artifacts/harness/run_eval.sh` is the exact invocation.

| arm | strict-match | flexible-extract | problems differing from baseline |
| --- | --- | --- | --: |
| pristine baseline | **0.9454 ± 0.0063** | 0.9454 ± 0.0063 | — |
| patch 1 only | 0.9454 ± 0.0063 | 0.9454 ± 0.0063 | 0 |
| patches 1+2 | 0.9447 ± 0.0063 | 0.9447 ± 0.0063 | 1 (doc 494) |
| **patches 1+2+3 (reported arm)** | **0.9454 ± 0.0063** | **0.9454 ± 0.0063** | **0 of 1319** |

**Threshold: not below 0.9328**, two standard errors down from the baseline. Every arm passes.

Three things about this gate are worth copying, not just its number.

**No accuracy figure existed for this model** — the source session recorded none — so the run's own
first clean measurement defines the reference. `max_tokens=9216` is load-bearing: lm-eval's default
256-token budget truncates this model's reasoning before the answer arrives and scores **0.0318**
strict-match, which reads as a broken model rather than a broken measurement.

**Strict and flexible agree exactly on every arm.** That says the model is reliably ending in the
`#### number` form the strict filter wants, so the flexible filter never has to rescue anything — a
useful property, because it means a formatting regression would show up rather than being papered
over.

**The gate's own reproducibility was measured, and it is what licenses the per-problem comparisons.**
Two full gsm8k runs against one untouched, unrestarted server agree on **1319 of 1319** problems.
Greedy decoding here is deterministic run-to-run, so a per-problem difference between arms is real
signal about that server instance and not eval noise. That is what makes the one-problem wobble at
patches 1+2 readable: the later arm that also contains patch 2 scores 0.9454 with doc 494 *correct*,
so the difference tracks the **instance**, not the kernel — and the final re-gate of the reported
stack differs from the pristine baseline on zero problems, which settles it.

## Where the time actually goes

Captured before anything was changed — 12 steady-state decode steps at the workload's own operating
point, bs=64 and average context ≈8760 (`artifacts/analysis/profile_decode.py`):

```
per step: GPU busy 25.738 ms   wall 25.51 ms (median ITL)

  attention (paged decode)      15.510 ms/step   60.3%
  GEMM (fp8 blockscale)          6.876 ms/step   26.7%
  norm / quant                   1.358 ms/step    5.3%
  activation (silu/mul)          0.257 ms/step    1.0%
  other                          1.565 ms/step    6.1%
```

**This contradicts the bundle's own reference material, and the contradiction is the whole result.**
`BASELINE.md` states that 86.3% of decode time is attention and does not list GEMM as a decode cost
at all. Measured: attention 60.3%, GEMM **26.7%**. Acting on the 86.3% figure would have sent every
hour at the one component already running at ~75% of HBM peak, and missed the one running at ~24% of
its achievable bandwidth.

The two derived headrooms are what chose the targets:

| component | traffic per decode step | achieved | headroom |
| --- | --- | --- | --- |
| GEMM | 13.2 GB of FP8 weights | 1.9 TB/s | ~24% of achievable — roughly 4× |
| attention | 91.8 GB of KV | ~5.9 TB/s | ~75% of HBM peak — near the floor, worth only a bounded attempt |

Both predictions held: the GEMM target returned +23.18%, and the bounded attention attempt returned
+3.07% and stopped there.

**Prefill was profiled separately and is closed, not open** (`artifacts/analysis/profile_prefill.py`,
whose driver arms the profiler *before* firing the requests so the captured steps are extend batches
rather than decode steps). After patch 1 prefill is ~32% of wall clock;
its GEMMs are 58.4% of it and run at **~2145 TFLOPS against ~2.3 PFLOPS FP8 dense peak — 93% of
peak**. That is the Rule-1 sanity check passing in the direction that *ends* an investigation: there
is no factor left there, and any "win" later measured on that path would have to be an accuracy
change or a measurement error.

## What was tried and did not work

The most valuable section, and on this run most of it was killed by a profile or a source read
**before** it cost a server instance.

| attempt | kernel-level result | end-to-end | verdict |
| --- | --- | --- | --- |
| **Split-K on the decode GEMMs** — re-tuned all four decode shapes with `--splitK`, 396 candidates against 228 (`artifacts/gemm_tune/tuned_decode_splitk.csv`) | Only the `asm` backend accepts split-K; every `ck`/`cktile` row returns `splitK=0`, so **three of four shapes are unchanged**. The one change at `errRatio 0.0` is down_proj (5120, 17408) → `asm` kernel 1 / splitK 2 at **37.93 µs vs `ck`'s 39.18 µs, −3.2%** | **not measured** — 3.2% of one of four GEMMs is **~0.2% of the decode step**, inside the 0.21% floor | **Dropped.** A patch whose effect cannot be distinguished from restart noise is not a win. And the tempting rows are wrong, not fast: the fastest `asm` splitK=6 candidates (**23.68 µs, −40%**) all carry `errRatio` 0.0134–0.0137, because K/128 = 136 k-tiles is not divisible by 6 and the split misaligns the reduction |
| **`partition_size` for decode paged attention** — `_AITER_PARTITION_SIZE_ROCM = 256` is a plain module constant with no env var and no recorded justification, so it looked like free source-level headroom. Interleaved sweep, 9 rounds × 30 iters (`artifacts/analysis/bench_pa_partition.py`) | P=128: 410.7 µs (+1.55%, 2.97% spread). **P=256: 404.4 µs, 5.642 TB/s, 0.50% spread.** P=512: 264.6 µs (**−34.57%**), max\|Δout\| 0.0874. P=1024: 139.5 µs (**−65.51%**), **16.357 TB/s**, max\|Δout\| 0.1631 | **not measured** | **Dropped, and it was never real.** 16.4 TB/s is **2× the 8 TB/s HBM peak**, so the kernel cannot have read the KV it was asked to read — and the output moves. Root cause found in the source: `csrc/cpp_itfs/pa/pa_ragged.cuh:72` hard-codes `constexpr int T_PAR_SIZE = 256` and ignores the `PARTITION_SIZE` template parameter it was given, while the reduce kernel (`pa_kernels.cuh:915`) honours it. At P>256 the two disagree about tokens per partition. **256 is the only self-consistent value; 128 is self-consistent but slower and the noisiest arm in the sweep.** A latent upstream aiter bug found on the way out |
| **Tuning the CUDA-graph capture ladder** — `M:512` and the 1..512 buckets still log `not found tuned config` after patch 1 | not attempted (~208 further shapes) | not attempted | **Dropped.** The frozen workload runs at bs=64 in steady state; those M values are only reached during the brief ramp-down at the end of each decode wave. Real value at a workload whose steady-state batch is *not* a tuned M |
| **Fused QK-norm + rope + KV-write** — SGLang's `forward_prepare_aiter_fused_mrope` would collapse four decode kernels (2× `add_rmsnorm_quant`, `rotary_embedding`, `store_kvcache`) into one | those four are together **0.72 ms/step** | not attempted | **Dropped as inapplicable.** Gated on `isinstance(self.rotary_emb, MRotaryEmbedding)` with an `mrope_section`; Qwen3-14B is text-only 1D rope, so the gate is correctly false. Reaching it means writing a new fused kernel, not flipping a switch |
| **Strided QK-RMSNorm** — `direct_copy_kernel_cuda<c10::BFloat16>`, 2 launches per layer materialising the non-contiguous `qkv.split()` views | **81.6 calls/step, 354.2 µs/step ≈ 1.6% of the step** (7.6× the floor, so it *is* measurable) | not attempted | **Scoped and costed, not attempted.** `models/utils.py:433` takes the copy deliberately — ROCm's RMSNorm kernels fault on strided input. SGLang's `fused_qk_gemma_rmsnorm` has the right shape of solution but is Gemma-flavoured (`w_q + 1.0`) and, contradicting its own docstring, still reshapes internally, so it would take the same copy. The best-scoped remaining candidate on this model |
| **Paged prefill attention** — `FmhaBatchPrefillWithPagedKVCache`, the one prefill component not near its roof | **~683 TFLOPS**, 23.3% of prefill | a 20% improvement is **~1.5% end to end** | **Recorded as an open thread.** Improving it means writing FMHA, not choosing a kernel |
| **`--page-size 1024` + `SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d`** — the source session's own biggest configuration levers | — | **not measured** | **Out of scope by rule.** Both are frozen configuration, so neither could be claimed as a win here regardless of outcome. Their standalone value on this model remains unmeasured |

Two more corrections the run made to its own earlier work, recorded because each nearly became a
published wrong number:

**The "7.3% in-graph GPU idle" was an artifact.** Total inter-kernel gap >1 µs is 20.94 ms across 39
gaps in the decode trace, but **20.5 ms of that is a single stall occurring 0.1× per step**.
Recurring idle is ~33 µs/step — 17.4 µs between two `index_elementwise` kernels, 8.2 µs before
`greedy_sample`, 5.4 µs after `create_flashinfer_kv_indices_triton` — which is **0.14% of the step,
not 7.3%**. Taking a mean over a short capture put one rare stall on every step. Without the
correction a future session would go looking for launch overhead that does not exist.

**Trace-based attribution of patch 2 was misleading, and median ITL is the right instrument.**
Comparing before/after traces suggested attention had improved only ~2%, which conflicted with
everything else. The cause is a confound in the capture: the profiler settles for a fixed wall-clock
time, so a *faster* server has generated more tokens by then and is profiling a **longer context**
(8760 → ~8860), which inflates the attention work it measures. The frozen benchmark's median ITL has
no such confound.

## What the measurement method changed, in numbers

Worth its own section because on this machine it changed conclusions rather than tidying them.

| figure | non-interleaved | interleaved | what happened |
| --- | --: | --: | --- |
| patch 3 | **−0.60%** — a regression, consistent across all three positions, with median ITL apparently up | **+1.42%** | The sign reversed. The candidate had been compared against numbers recorded earlier the same morning; re-running the *unmodified* code as arm A produced 1939.7 where the same code had produced 1974.4 about an hour earlier. **The machine had slowed 1.8%.** Without interleaving, a real 1.4% would have been written up as a negative result with a plausible mechanism story attached |
| patch 2 | +3.95% | **+3.07%** | A third of the claim was drift |

The rule this establishes for this stack: **a single arm measured now against a number recorded an
hour ago is not a comparison.** Use `artifacts/ab/run_ab_stack.sh` with `stack_A.sh` / `stack_B.sh`,
and read it with `artifacts/ab/report_ab.py`, which prints each instance's runs in order (so the
within-instance decline stays visible), compares position-matched, and reports disjointness rather
than only a difference of means.

## When this entry stops applying

Silently, in every case — see the [deploy failure table](#every-way-this-deploy-silently-does-nothing)
for the full list. The load-bearing ones: **arch ≠ `gfx950` or CU count ≠ 256** (literal columns in
the CSV key); **TP ≠ 1** (N and K shard, so the four tuned pairs cease to exist); **concurrency,
ISL or `--chunked-prefill-size` changed** (M moves off the tuned buckets); **KV cache not bf16 or
head_size ≠ 128** (patch 2's guard goes false); **`--attention-backend` ≠ `aiter`** or
**`--page-size` ≠ 1** (decode leaves `pa_ragged`); **a different aiter or sglang commit** (patches 2
and 3 are source diffs and will fail to apply, which is at least loud); **stale
`/tmp/aiter_configs`, stale `__pycache__`, or no restart**.

Still reusable when the artifact is inert:

- **The four-shape target list** — 7168/5120, 5120/5120, 34816/5120, 5120/17408 — and the three M
  values. Re-tune those shapes at your own M rather than re-deriving the target list.
- **`artifacts/gemm_tune/untuned_{decode,prefill}.csv`** are the tuner inputs; regenerating the table
  at a new operating point is a few minutes of tuner time (249.7 s decode, 221.1 s prefill as run).
- **The EXPERIMENTAL `pa_ragged` kernel itself**, which is dark by default in *every* aiter
  deployment and whose preconditions (head_size 128, bf16 KV, no alibi, no soft cap) many models
  meet. `artifacts/analysis/bench_pa.py` will re-time it on your shape in minutes.
- **Patch 3's finding that the aiter comment is wrong**, which applies to any FP8 B-preshuffle linear
  on gfx95, not just this model.
- **The measurement protocol** — position-matched comparison, interleaved arms, `report_ab.py` — is
  the most portable thing in this entry, and the `/proc/<pid>/maps` build-hash engagement check is
  the second.

## What would promote this entry

1. **Deploy from `artifacts/patches/` on a clean instance** — fresh container from the recorded tag,
   `git apply` the three patches rather than copying the arm-B snapshots, and re-measure. The
   deploy path exercised so far is `cp`, and the two are only known-equivalent by file comparison.
2. **Record the container digest and a `/proc/<pid>/environ` dump** on that instance, closing the two
   gaps in the fingerprint above.
3. **Re-measure the restart floor on that instance before trusting the delta**, since the floor is a
   property of the node, and this node's hour-scale drift is 1.8%.

## Provenance

Task bundle: `tuning_workspace/experiment_standalone/qwen3_14b_fp8_tuning/`.

| claim | where |
| --- | --- |
| headline, per-position A/B table, engagement per arm | `FINDINGS.md` §5, `results/stackab_{A,B}{1,2}_r{1,2,3}_*` |
| baseline n=8, the three spreads, the hour-scale drift table | `FINDINGS.md` §1, `results/align_*` and `results/mybase_*` |
| accuracy gate and its reproducibility | `FINDINGS.md` §2, `eval_results/` (7 runs) |
| decode and prefill profiles | `FINDINGS.md` §3 and §3.1, `analysis/traces*/` |
| patch 1, candidate timings, tuner logs | `FINDINGS.md` §4.1, `analysis/gemm_tune/` |
| patch 2, microbenchmark, interleaved A/B | `FINDINGS.md` §4.2, `results/paab_*` |
| patch 3, the disproved comment, interleaved A/B | `FINDINGS.md` §4.4, `results/scaleab_*` |
| every negative result | `FINDINGS.md` §4.3 and §7 |
| per-patch measurement, apply command and base commit | `artifacts/patches/*.patch` headers, `artifacts/patches/PATCHES.md` |
| baseline provenance and why the session's 2278.008 is not a baseline | `BASELINE.md`, `reference/README.md` |

One number in the bundle that is **not** a measurement and should not be quoted as one: the final
confirming run on the delivered live tree reads **1956.943 tok/s** (`results/final_confirm_*`), below
the 1985.566 headline because it is the fifth workload that server instance had served — three
benchmark runs plus a full 1319-problem gsm8k — and the within-instance decline applies. It is
offered as proof the delivered tree serves at the improved rate, not as the result.
