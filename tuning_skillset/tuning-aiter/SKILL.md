---
name: tuning-aiter
description: Tune GEMM and fused ops through aiter on AMD Instinct — run the gradlib tuner across hipblaslt/triton/flydsl/asm/torch backends, write tuned config CSVs to the directory the runtime actually reads, and prove engagement with AITER_LOG_TUNED_CONFIG. Use when tuning ops dispatched through aiter in vLLM or SGLang.
---

# Tuning through aiter

Read `../tuning-core/SKILL.md` first.

aiter is where most serving-path GEMMs actually land, and it is the one backend that tunes
*across* languages: a single gradlib run races hipBLASLt solutions, Triton kernels, FlyDSL
kernels, hand-written assembly and torch against each other, then records the winner per
shape. That makes it the highest-leverage tuning target and the one with the most ways to
produce an artifact that never gets read.

Everything below was measured on gfx942 / MI300X, aiter 0.1.13, in the vllm image.

## 1. The artifact is a CSV row, keyed on your hardware

Tuned configs are CSV files under `aiter/configs/`. The bf16 GEMM schema:

```
gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw
```

The first two columns are the whole story of why tuned artifacts do not travel. A lookup
matches on `gfx` **and** `cu_num`. Check what you are:

```python
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
get_gfx(), get_cu_num()      # -> ('gfx942', 304) on MI300X
```

Then check what shipped. In the vllm image, measured:

| config file | rows | `gfx/cu_num` present |
| --- | --- | --- |
| `bf16_tuned_gemm.csv` | 780 | `gfx950/256` only |
| `a4w4_blockscale_tuned_gemm.csv` | 1500 | `gfx950/256` only |
| `a8w8_blockscale_bpreshuffle_tuned_gemm.csv` | 1498 | `gfx950/256` only |
| `a8w8_blockscale_tuned_gemm.csv` | 15611 | `gfx942/304`, `gfx950/256` |
| `a8w8_bpreshuffle_tuned_gemm.csv` | 2084 | `gfx942/304`, `gfx942/80`, `gfx950/256` |

**On an MI300X box: 780 tuned bf16 rows, none of them usable.** Not a bug — the image ships
for several targets and the bf16 table happens to be populated for gfx950 only. But anyone
assuming "aiter ships tuned configs, so bf16 GEMM is tuned" is wrong on that box, and nothing
says so at runtime unless you ask. `cu_num=80` rows in the a8w8 file are from a partitioned
MI300 mode — same `gfx`, different CU count, still no match at 304.

Run the same image on an MI355X and every one of those rows hits. That is the useful thing
about this table: it is the same artifact, and which half of it is dead depends entirely on
what you plugged in. Two consequences that showed up when the corpus was swept on both parts:

- **Tuning headroom is a property of the box, not of the library.** The same sweep over the
  same cases returned a dispatch-weighted 1.96x on gfx942 and 1.40x on gfx950, and the gfx950
  baselines were already 2-3.6x faster in absolute terms. Less headroom because more of the
  shipped tables applied, not because the tuner got worse. Reporting an uplift ratio without
  saying which tables were live is reporting an artifact of the image.
- **A well-populated table can still ship an unsound row.** On gfx950, the shipped
  `gemm_a16w8_blockscale` config selects `NUM_KSPLIT=8` for M≤16 at K=2880 — but K/group_k is
  22.5 groups, so the split boundaries cut quantization groups and the baseline is simply
  wrong, by ~40%, before any tuning starts. A harness that trusts the shipped config as its
  reference measures every candidate against a wrong answer. Check the baseline's correctness
  first; see `../tuning-core/correctness_gates.md`.

This is the concrete form of the general rule: **a tuned artifact is never portable across
architectures.** Re-tune per target — and verify the target's own rows before trusting them.

## 2. Turn on the engagement signal before you change anything

```bash
export AITER_LOG_TUNED_CONFIG=1
```

aiter then prints, per call, whether the lookup hit. Untuned:

```
[aiter] shape is M:4096, N:4096, K:4096 dtype='torch.bfloat16' ... not found tuned config
        in /tmp/aiter_configs/bf16_tuned_gemm.csv, will use default config! using torch solution:0
```

Tuned:

```
[aiter] shape is M:4096, N:4096, K:4096 ... found padded_M: 4096, N:4096, K:4096
        is tuned on cu_num = 304 in /tmp/aiter_configs/bf16_tuned_gemm.csv, libtype is hipblaslt
```

Take the baseline reading *first*. If the shape already says `is tuned on cu_num = <yours>`,
you are tuning something already tuned and your comparison needs a different baseline. If it
says `will use default config`, you have found real headroom.

Two details in that log line worth reading carefully:

- **`padded_M`** — where this field appears, aiter bucketed M before the lookup, so the M you
  tuned is not necessarily the M that hit. **Do not generalize that into coverage.** Padding is a
  property of the entry point, not of aiter: `a8w8_blockscale_bpreshuffle` matches exactly on
  (M,N,K) with no interpolation, so there a row tuned at M=64 serves M=64 and nothing else.
  Earlier versions of this document claimed a sparse grid is safe because "a row tuned at one M
  serves a range of nearby M" — true for the generic `bf16_tuned_gemm` path shown above, false for
  the blockscale paths, and it cost a run. See the reproduced case below.
- **`libtype`** — which backend won. This tells you whether your Triton work is even in play
  for that shape, or whether hipBLASLt is beating it.

### The shape list *is* the tune — harvest it, never guess it

The same log that reports engagement also tells you, for free, exactly which shapes the workload
dispatches: every miss names one. Take that reading on the untuned baseline you are already
running, and let it write your tuner input.

```bash
AITER_LOG_TUNED_CONFIG=1 <your serve command> 2>&1 | tee /tmp/aiter_dispatch.log
# drive the REAL benchmark workload, not a smoke test — the M distribution is the thing you want
grep -o 'shape is M:[0-9]*, N:[0-9]*, K:[0-9]*' /tmp/aiter_dispatch.log | sort -u
```

The served M values are chosen by the engine, not by you: decode M follows the CUDA-graph capture
buckets and prefill M follows the chunk size. A grid that looks sensible on paper — a few powers of
two, one decode point, one prefill point — can miss every single one of them.

**Reproduced (gfx950, sglang + aiter, fp8 a8w8 blockscale bpreshuffle, 85.1% of GPU time).** A
per-shape tune was run over M ∈ {1, 64, 1024, 16384} across the four served (N,K) families. The
engine dispatched decode M ∈ {2..512} and prefill chunks M ∈ {~7168..15360} — no intersection.
Result: **0 `is tuned on cu_num` hits against 272 `not found tuned config`.** The tuned artifact
was never the code that ran. The tuner had genuinely found a 1.28x and it was unreachable *as
tuned*; the failure was in binding, not in search. The e2e A/B then measured +1.18%, which was box
drift, and anyone not counting hits would have banked it as a win.

**The shape list is the *engine's* property, not the model's.** Same model, same workload, same node,
two sglang versions: 0.5.12 dispatched **348** distinct shapes over **5** `(N,K)` families; 0.5.15
dispatched **444** over **12**, including prefill M values (7171, 10260, 11945, 13096, 15128, 15176,
15284, 15380, 16276) absent from the first list entirely.

The cause is visible in the engine's own knobs: 0.5.12 has a single `cuda_graph_bs`, while 0.5.15
splits it into `cuda_graph_bs_decode` and `cuda_graph_bs_prefill`, so prefill is now captured at its
own batch sizes. Together with `chunked_prefill_size`, those knobs — not the model, and not aiter —
are what fixes the M distribution you must cover. Carrying a harvested list across an engine upgrade
silently reintroduces the miss it was built to prevent, and changing either knob invalidates it too.
Re-harvest.

So treat the shape list as a derived artifact, and gate on binding before spending real time:

1. Harvest the dispatched (M,N,K) set from the baseline log, as above.
2. Tune those rows.
3. **Re-run the server briefly and require `is tuned on cu_num` > 0, with misses going to 0 on the
   shapes you tuned.** Two minutes here rejects an unreachable tune before an e2e A/B spends an
   hour measuring nothing.
4. Only then measure.

Step 3 is not bookkeeping. A tune that fails to bind returns a *plausible* number, because the
drift it measures is the same size as the win you were hoping for.

### 2b. Engagement is not selection — the hit can bind to the wrong kernel

A hit count proves the *row* was found. It does not prove the *kernel* in that row ran. On some
aiter builds the serving wrapper reads only `libtype` from the tuned row and drops `kernelName`,
so a row that says "cktile instance #11" deploys as "cktile, whatever cktile picks by default".
The log still prints `is tuned on cu_num`. Every gate in §2 passes. And the library's internal
default can be far worse than the default you displaced.

**Measured (gfx950, `a8w8_blockscale_bpreshuffle`, M=15104 N=34816 K=5120, 43.8% of the dominant
head), on an image whose wrapper drops `kernelName`:**

| path | time | TFLOPS |
| --- | --- | --- |
| production wrapper, no tuned config — what the server did before | 4165.0 µs | 1293 |
| production wrapper, tuned row installed (`libtype=cktile`) | 7733.7 µs | 696 |
| tune entry point, `cktile kernelId=11` — what the tuner measured | 2646.5 µs | 2035 |

The tuner was right: that kernel is **36% faster** than the default, reproducibly, and its recorded
time matches to 0.2%. Deploying it made the op **85.7% slower than doing nothing**, because
selecting a library without an instance is not selecting the instance you tuned. Applied across the
shapes carrying ~90% of the head: 290 engagement hits, and **e2e throughput −6.48%**.

Then the same CSV, same node, same tuner, on a build whose wrapper passes `kernelName`:
production wrapper **2655.8 µs / 2028 TFLOPS**, matching the tuner's recorded 2655.3 µs to **0.02%**,
and **e2e +23.88%** (TTFT −30.8%, TPOT −18.0%). Nothing about the tuning changed. The only
difference was whether the runtime could honour a kernel selection.

So add one gate between "it binds" and "measure e2e" — **does the production path reproduce the
tuned number?**

```python
# Time the PRODUCTION entry point, the one the server calls, against the tuner's recorded `us`
# for that same row. Not the *_tune entry point, and not --run_config.
op.gemm_a8w8_blockscale_bpreshuffle(x, w_shuffled, x_scale, w_scale)
```

If it does not land within noise of the CSV's `us`, the selection is not binding, whatever the hit
count says. This is the whole gate: **one shape, two numbers, before you spend an hour on an A/B.**

Do not substitute the tuner's own `--run_config` for this. Measured on the broken build,
`--run_config` reported 2660 µs — the *tuned* number, not the 7733 µs the server gets — because it
does not go through the production wrapper. A vendor verification tool that agrees with the vendor
tuner tells you nothing about the vendor runtime.

The static form of the same check, useful before you tune at all:

```bash
A=/sgl-workspace/aiter/aiter/ops/gemm_op_a8w8.py
# 1. does the production op even accept a kernel name?  (0 = it cannot be told what to run)
grep -A12 'def gemm_a8w8_blockscale_bpreshuffle_cktile(' "$A" | grep -c kernelName
# 2. read the wrapper's branches yourself -- see the trap below before automating this
sed -n '/^def gemm_a8w8_blockscale_bpreshuffle(/,/^def [a-z]/p' "$A" | grep -nE 'libtype|kernelName'
```

**Trap: "does `kernelName` appear in the wrapper" is not the test, and it returns a false OK on the
broken build.** The broken wrapper *does* contain `kernelName=kernelName` — in its `asm` branch —
while the `ck` and `cktile` branches call with positional args only:

```python
if libtype == "cktile":
    return gemm_a8w8_blockscale_bpreshuffle_cktile(XQ, WQ, x_scale, w_scale, Y)     # no kernel
elif libtype == "ck":
    return gemm_a8w8_blockscale_bpreshuffle_ck(XQ, WQ, x_scale, w_scale, Y)         # no kernel
elif libtype == "asm":
    kernelName = config["kernelName"]                                                # only here
```

Test the **signature of the op your row names**, and the forwarding in **that specific branch**.
`env-setup/audit_tools.sh` does this and is validated against both images — BROKEN on
`v0.5.12`/`a6bb49937`, OK on `v0.5.15.post1`/`9127c94a1`. Run it before you tune:

```
BROKEN   sig-cktile        gemm_a8w8_blockscale_bpreshuffle_cktile takes NO kernelName …
ACTION   update-aiter      predates 7136b240e (#3075); need >= v0.1.15 …
```

Read the source, not `inspect.signature`: these ops are JIT-wrapped and report `(*args, **kwargs)`
until the module is built. After the first call aiter prints its real signature on a
`type hints mismatch, override to -->` line, which is the other place to look.

### The version boundary — know it by name

The fix is a single identifiable commit:

```
7136b240e102e3b54bc8e960abf59e59f953f5c8   2026-05-21
blockscale gemm: dispatch by kernelName, strict tuned-CSV validation (#3075)
```

**First release containing it: `v0.1.15` (2026-05-29). `v0.1.14` (2026-05-15) does not.** So the
floor is **aiter ≥ v0.1.15**, and below it a per-shape blockscale GEMM tune is not deployable.

**This gate is aiter's alone — the engine version is irrelevant to it.** Grepping the whole sglang
tree for `kernelName`, `libtype` and `AITER_CONFIG_GEMM` returns **0 matches, in both v0.5.12 and
v0.5.15.post1**. sglang calls `gemm_a8w8_blockscale_bpreshuffle(XQ, WQ, x_scale, w_scale)` from
`srt/layers/quantization/fp8_utils.py` and takes no part in choosing a kernel; the wrapper that
drops `kernelName` lives in `aiter/ops/gemm_op_a8w8.py`. So do not reason about deployability from
an image tag or an engine version. Ask aiter.

Builds measured (the image tags are only *where* these aiter commits were found):

| aiter commit | `git describe` | reported version | has `7136b240e` | found in |
| --- | --- | --- | --- | --- |
| `a6bb49937` (2026-04-29) | `v0.1.12.post1-150-g…` | `0.1.12.post2.dev150+…` | **NO** — 169 commits short | sglang `v0.5.12-rocm720-mi35x`, **GEAK CI default** |
| `9127c94a1` (2026-06-25) | `v0.1.16-110-g…` | `0.1.17.dev110+…` | yes | sglang `v0.5.15.post1-rocm720-mi35x` |
| `0ba802e2d` (2026-07-27) | `v0.1.18-46-g…` | — | yes | `clone_libs.sh` pin |

The two are bundled in practice, which is why they are easy to conflate, but they vary
independently — and the tag is not even a reliable proxy for the aiter *date*: the `v0.5.12` image's
aiter reports `…+ga6bb49937.d20260516`, i.e. built 2026-05-16 from a 2026-04-29 commit.

Check it, cheapest first:

```bash
# 1. version screen. Anything whose release component is below 0.1.15 is certainly broken.
python3 -c "import aiter._version as v; print(v.__version__)"     # e.g. 0.1.17.dev110+g9127c94a1

# 2. authoritative, and sglang images ship the git tree:
git -C /sgl-workspace/aiter merge-base --is-ancestor 7136b240e HEAD \
  && echo "kernelName dispatch PRESENT" \
  || echo "BROKEN: aiter predates #3075 -- update to >= v0.1.15 before tuning blockscale GEMM"
```

Two traps in reading the version string. It is a PEP 440 dev version — `0.1.17.dev110+g9127c94a1`
means *110 commits past `v0.1.16`, heading for 0.1.17*, so it is **newer** than `v0.1.16`, not older
than `v0.1.17`. And there is a genuinely ambiguous band: commits between the fix (2026-05-21) and the
`v0.1.15` tag (2026-05-29) contain it while still describing as `v0.1.14-rc0-NN`. In that window the
version cannot answer the question — use check 2, or the behavioural check above.

**The floor is necessary, not sufficient.** The commit title says *blockscale*, and it means it. Even
at `v0.1.16+`, only **19** production ops accept `kernelName`: the a8w8 blockscale family
(`_ck`, `_cktile`, `_bpreshuffle_ck`, `_bpreshuffle_cktile`, `_bpreshuffle_asm`), a4w4 blockscale,
the a8w8/a16w16 `asm` entry points, and the MoE stage ops. Roughly 53 others do not — including the
plain, non-blockscale `a8w8` and `a8w8_bpreshuffle` paths. Passing the version check does not license
skipping the per-op check.

On a build below the floor, `asm` is the only family whose tuning is deployable. A tuned table that
is mostly `ck`/`cktile` is then not a small loss but an **active regression**, and the right move is
to raise the runtime, not to keep tuning. If you cannot change the image, restrict the deployed table
to `libtype=asm` rows and expect a correspondingly small ceiling.

One side effect of the same commit: it also added **strict tuned-CSV validation**, so at ≥ `v0.1.15` a
malformed or schema-mismatched tuned CSV tends to be rejected outright rather than silently ignored.
That is an improvement, and it softens the §3 warning below — on these builds a tuner/runtime
generation mismatch is more likely to raise than to become a silent miss.

**Corollary for reading a no-win.** "The tuner found nothing, so the default is already good" and
"the tuner found something the runtime cannot execute" produce the same e2e number and opposite
conclusions. The first says stop; the second says fix the runtime and a large win is waiting. Only
the production-path timing above distinguishes them, and the original 27B run — which concluded
"default CK heuristic already near-optimal in the live serving context" — had it backwards by 30
percentage points.

## 3. Get the tuner (vllm image only)

The images differ in a way that changes the workflow:

- **sglang** ships aiter as a git checkout at `/sgl-workspace/aiter` — tuners present, tune
  in place.
- **vllm** ships aiter as a wheel — `configs/*.csv` present, but no `gradlib/`, no `csrc/`.
  Wheels carry the runtime, not the build tree.

For vllm, supply source matching the installed wheel:

```bash
python3 -c "import aiter._version as v; print(v.__version__)"     # -> 0.1.13
git clone https://github.com/ROCm/aiter.git /work_aiter
git -C /work_aiter checkout $(git -C /work_aiter rev-list -n1 v0.1.13)
```

Resolve the tag from the installed version rather than copying a hash. Do **not**
`pip install` the checkout over the wheel — run the tuner from the source tree and let it
write into the installed package. A tuner from a different aiter generation can emit columns
the installed runtime does not read, which fails as a silent miss rather than an error.

## 4. Run the tuner

**First, confirm which tuner your op uses. This section documents `gradlib`'s
`gemm_tuner.py`, which serves dense bf16 only.** Every quantized GEMM goes through a
*different* per-op tuner with a different entry point, different flags, a different schema, and
a different cost — none of which is what you read below. Getting this wrong is not a small
detour: you will run a tuner that never touches the op you care about.

| your op | tuner | this section applies? |
| --- | --- | --- |
| dense bf16 | `gradlib/gradlib/gemm_tuner.py` | yes |
| fp8 / int8 / fp4 / batched / MoE | `csrc/ck_gemm_*/..._tune.py`, or `aiter/utility/pretune.py` | **no** — see `../tuning-ck/` §3 and §4b below |

### 4b. If it is a per-op quantized tuner, these are the differences that bite

Verified on aiter `d9e5ef7c` / gfx950 against
`csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`, the FP8 pre-shuffled path an
FP8 SGLang deploy actually dispatches:

| | gradlib (below) | per-op quantized tuner |
| --- | --- | --- |
| flags | `--input_file` / `--tuned_file` | **`-i` / `-o`** |
| `--libtype` | no such flag | **exists**, `default=["all"]`, choices `all, asm, ck, cktile, flydsl` |
| `torch` in the race | yes, and it is your floor | **not selectable** — there is no torch candidate |
| result schema | `…,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw` | **`…,kernelId,splitK,us,kernelName,tflops,bw,errRatio`** (`base_tuner.py:1546`) |
| split-K | — | **off by default**; `-k/--splitK` is `store_true` (`base_tuner.py:130`) |
| cost, 2 shapes | 120 s (gfx950) | **≈50 min** across 4 libtypes — plan for ~14× the gradlib figure |

Three of those change what you do:

- **There is no `torch` floor, so build one.** The checklist item below tells you to require a
  `torch` row as the baseline the winner must beat. On this tuner that row cannot exist. The
  real floor is the **CK `kernelId=0` fallback** — what the op dispatches when your config is
  absent — and you have to measure it yourself, on a harness you control. Do not substitute the
  tuner's own before/after for it; see Rule 3b/5 in `../tuning-core/measurement.md` and the
  cold-timing warning in `../tuning-ck/` §3c.
- **Split-K is off unless you ask.** The schema has a `splitK` column, which makes the search
  look fully explored while it holds a zero in every row. With the flag unset the tuner never
  races a single split-K candidate, for either the `ck` or `cktile` path
  (`useSplitK = args.splitK`, `gemm_a8w8_bpreshuffle_tune.py:631`). Pass `-k` for
  tall-skinny-M or long-K shapes; budget roughly 4× the runtime.
- **Scope the sweep from the right number.** Anyone planning a 23-shape run from the 120 s /
  214 s figures below will discover a multi-hour job. Time two shapes first, then extrapolate.

### 4a. gradlib's `gemm_tuner.py` (dense bf16)

Input is a CSV of shapes with the same key columns minus the results:

```csv
M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle
4096,4096,4096,False,torch.bfloat16,torch.bfloat16,False,False
1,4096,4096,False,torch.bfloat16,torch.bfloat16,False,False
```

```bash
export HIP_VISIBLE_DEVICES=4          # pin an idle GPU
python3 gradlib/gradlib/gemm_tuner.py \
    --input_file  /tmp/untuned.csv \
    --tuned_file  /tmp/tuned.csv
```

**Do not pass `--indtype`, and do not look for `--libtype`.** Both appear in every write-up
of this tuner, including earlier versions of this document, and neither works at the aiter
commits currently in play (`0ba802e2`, 2026-07-27, which `clone_libs.sh` pins; and
`a6bb4993`, 2026-04-29, which the sglang image ships). Verified on both.

`--indtype` raises. `gemm_tuner.py:121` rewrites the argument in place from the CLI string
to a torch dtype, and `GemmTuner.pre_process` then looks the string up again:

```python
args.indtype = get_dtype(args.indtype)              # "bf16" -> torch.bfloat16
...
self.untunedf["dtype"] = f"dtypes.{_cli_to_dtypes[args.indtype]}"
# KeyError: torch.bfloat16
```

The two files disagree about the type of one attribute, so this is aiter's bug rather than
yours — but there is no invocation with `--indtype` that runs. **Put the dtype in the CSV's
`dtype` column instead** (the schema above already has one) and omit the flag: `args.indtype`
stays `None`, the conversion is skipped, and the column is used.

`--libtype` does not exist as a flag at all. `libtype` is a **column in the tuned CSV** and a
filter argument to `chip_info.build_tune_dict`, not a command-line option. The cross-backend
race still happens and the winning backend is still reported per row — you just cannot
restrict it from the command line. So this remains the reason aiter is the integration point
for the per-language skills rather than a competitor to them, and the table below is how to
read the `libtype` column of the output, not a menu:

| libtype | what won | authoring-time tuning lives in |
| --- | --- | --- |
| `hipblaslt` | a hipBLASLt solution index | `../tuning-hipblaslt/` |
| `triton` | one of aiter's Triton kernels | `../tuning-triton/` |
| `flydsl` | a FlyDSL kernel | `../tuning-flydsl/` |
| `ck` / `cktile` | a Composable Kernel instance | `../tuning-ck/` |
| `asm` | a hand-written assembly kernel | — |
| `skinny` | a narrow-GEMM specialization | — |
| `torch` | the torch fallback — your floor | — |

A `torch` row is the baseline the winner had to beat, measured by the same harness, and it is
what aiter falls back to on a miss. **This holds for gradlib only** — the quantized per-op
tuners have no torch candidate, and their floor is the CK `kernelId=0` fallback, which you must
measure yourself (§4b).

Measured on gfx950, 2 shapes, all backends: **120.2 s** (gfx942, 2 shapes: 214 s). Not
transferable to the per-op quantized tuners, which ran ~50 min for the same two shapes. Cost
scales with the shape count, so scope the shape list — but scope it by harvesting what the
workload actually dispatches (§2), never by thinning a generated sweep. Capturing from the live
workload is a correctness requirement, not a cost optimization: a shape you did not harvest is a
shape that will fall back. Per-engine capture details are in `../tuning-in-vllm/` and
`../tuning-in-sglang/`.

Useful flags: `--compare` benchmarks before and after and prints the delta;
`--update_improved --min_improvement_pct N` only writes rows that improved by at least N%,
which is the built-in defence against enshrining a within-noise "win".

**That defence has a hole, and it is not theoretical.** Shapes not already present in the tuned
CSV are classified `NEW` and written regardless of the measured comparison — including when the
comparison shows a slowdown. Verified on gfx950 for fp8 `M=1, N=7168, K=8192`: measured 0.82x,
printed `OK`, improvement reported as `N/A`, row written. Read the Speedup column yourself.
Worked through in `../tuning-ck/` §3b, which is also where the per-op CK tuners live.

Output carries your keys and a correctness column:

```
gfx942,304,4096,4096,4096,False,torch.bfloat16,...,hipblaslt,198969,0,243.7286,Cijk_...MT256x224x64...ISA942...,0.0,563.9,413.01
gfx942,304,1,4096,4096,False,torch.bfloat16,...,hipblaslt,198198,0,13.7208,Cijk_...MT16x16x512...ISA942...,0.0,2.45,2446.71
```

The same two shapes on gfx950, from the run above:

```
gfx950,256,4096,4096,4096,False,torch.bfloat16,...,hipblaslt,440518,0,103.5316,Cijk_...MT256x256x64...ISA950...,0.0,1327.51,972.3
gfx950,256,1,4096,4096,False,torch.bfloat16,...,hipblaslt,439841,0,  9.4608,Cijk_...MT16x16x512...ISA950...,0.0,   3.55,3548.41
```

Worth comparing column by column, because which parts moved is the useful part:

| | gfx942 | gfx950 |
| --- | --- | --- |
| `gfx` / `cu_num` — the lookup key | `gfx942` / 304 | `gfx950` / 256 |
| 4096³ solution index | 198969 | 440518 |
| 4096³ macro-tile | `MT256x224x64` | `MT256x256x64` |
| 4096³ throughput | 563.9 TFLOPS | 1327.5 TFLOPS |
| M=1 macro-tile | `MT16x16x512` | `MT16x16x512` — unchanged |
| M=1 bandwidth | 2446.7 GB/s | 3548.4 GB/s |

The solution index is a different number on a different part, and the `ISA942`/`ISA950` field
in the kernel name says why: these are separately compiled kernels, so an index copied across
parts names something unrelated or nothing. The macro-tile, by contrast, transfers as a
*family* — M=1 wants `16x16x512` on both, and the square shape wants a 256-wide tile on both,
merely 256 deep rather than 224. Read the tile to learn what the problem wants; re-derive the
index per part.

Read the two rows together: the 4096³ winner uses a `256x224x64` macro-tile at 563.9 TFLOPS;
the M=1 winner uses `16x16x512` at 2.45 TFLOPS but **2447 GB/s**. Opposite tile shapes,
opposite figures of merit. Judge the decode row on bandwidth, not FLOPS
(`../tuning-core/measurement.md`). Check `err_ratio` on every row before deploying — the
tuner records it precisely so you can reject a fast-but-wrong solution.

## 5. Deploy to the directory the runtime reads — not the one it names

This is the step most likely to silently do nothing.

The log line points at `/tmp/aiter_configs/bf16_tuned_gemm.csv`, and
`AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE` returns that same path. Both are true and both
are traps: **`/tmp/aiter_configs/` is a derived cache, regenerated on import** by merging
`aiter/configs/*.csv` with `aiter/configs/model_configs/*.csv`.

Measured — appending correct `gfx942/304` rows directly to the `/tmp` file:

```
rows before import: 0      # (after writing 2 rows, then importing aiter)
rows after import:  0
```

The edit is erased and the lookup still reports `not found tuned config`. Writing to the
path the runtime *names* accomplishes nothing.

Deploy into the **installed package's** config directory instead. There are two ways, and
**prefer the drop-in**:

```bash
# PREFERRED — a new file in model_configs/, which the import step merges in
C=/usr/local/lib/python3.12/dist-packages/aiter/configs/model_configs
cp /tmp/tuned.csv "$C/a8w8_bpreshuffle_tuned_gemm_mymodel.csv"
```

This is what upstream itself does — aiter already ships
`a8w8_bpreshuffle_tuned_gemm_{glm47,glm5.2,minimax_m3,…}.csv` there. It produces a
few-line reviewable diff, exports cleanly as a patch, and A/B is `mv` the file away. Nothing
shipped is modified.

```bash
# WORKS, but mutates a shipped file — use only if the op has no model_configs/ path
P=/usr/local/lib/python3.12/dist-packages/aiter/configs/bf16_tuned_gemm.csv
cp "$P" "$P.bak"                       # always, so you can prove a clean A/B
tail -n +2 /tmp/tuned.csv >> "$P"      # skip the header row
```

Appending rewrites a 598-row vendored file, so your diff is indistinguishable from a vendor
change, and reverting depends on a `.bak` that lives outside version control.

Then the same lookup reports:

```
found padded_M: 4096, N:4096, K:4096 is tuned on cu_num = 304 ... libtype is hipblaslt
```

That transition is the cheap engagement check. **Two corrections to how it is usually stated,
both of which have cost real debugging time:**

The transition is only `not found tuned config` → `is tuned on cu_num = 304` **when
`AITER_LOG_TUNED_CONFIG=1` is set.** The miss line prints unconditionally
(`gemm_op_a8w8.py:456`, `:519`); the hit line sits behind `if AITER_LOG_TUNED_CONFIG:`
(`:449`, `:509`). Without the flag the transition is `not found tuned config` → **silence**,
so `grep -c "is tuned on cu_num"` returns 0 on a deploy that is working perfectly. If you
cannot set the flag — because the environment is the measurement contract — then use the
disappearance of the *miss* line, which needs no flag, and keep an untuned shape as a negative
control.

And a config hit is **not** the strongest available proof, despite how this is usually
phrased. It shows a *lookup* succeeded, not that the tuned kernel *executed* — and on some
builds those differ, which is exactly the §2b failure where the wrapper reads `libtype` and
discards `kernelName`. Kernel identity in a profile is strictly stronger and does not go stale
when the library changes its logging. Use the log line as the fast loop; confirm anything you
ship with a trace (`../tuning-core/engagement_verification.md`, form 4).

Generalize the rule: **find the deploy target empirically.** Write a row, import fresh, and
check whether it survived and whether the lookup hits. Do not infer the target from a config
constant or a log path.

### The derived cache has a more dangerous sibling: the merge writes *back*

The trap above costs you a change that does not take effect, which is loud and cheap. The same
merge step can also **edit the source file you are about to export as a patch**, which is
neither.

If a shape appears in more than one config source — say your drop-in duplicates a row aiter
already ships — the merge does not error and does not prefer yours. It **sorts by the `us`
column and keeps the lowest** (`aiter/utility/base_tuner.py:300-303`), rewrites
`aiter/configs/model_configs/*.csv` in place, and on the version in the SGLang image raises a
`RuntimeError` asking you to re-run. Re-running then succeeds against the file it just
modified.

Two things make this the worst failure mode in this document:

1. **The arbiter is the one number you should not trust.** `us` comes from the tuner's own
   timing, which is warm-cache and was measured 23% optimistic on one shape in this workload
   (`../tuning-ck/` §3c). A row that loses on cold-cache reality can win the merge.
2. **It corrupts the artifact, not the run.** Every other trap here produces a null result.
   This one produces a *wrong patch* that reproduces — because the file you export already
   contains the substitution.

Defend against it: `git diff` (or `md5sum`) your config files after the first import that
follows a deploy, not just after your edit. Keep your rows in a **uniquely named** drop-in file
so duplicates cannot arise, and if a shape genuinely overlaps a shipped row, decide the winner
yourself with a cold measurement and delete the loser rather than letting `us` arbitrate.

## 6. Verify, then re-measure

```bash
AITER_LOG_TUNED_CONFIG=1 <your workload> 2>&1 | grep -c "is tuned on cu_num"
```

**The flag is mandatory, not decorative** — without it the hit line is never emitted and this
command returns 0 no matter how well the deploy works (§5).

Expect a non-zero count. Do **not** expect the count to track how often the shape runs: the
lookup is wrapped in `functools.lru_cache(maxsize=1024)` keyed on **raw M**
(`gemm_op_a8w8.py:360`, `:401`, `:416`, `:465`), so each distinct M logs once and then goes
quiet, with sporadic re-logging as eviction thrashes. **Log-line counts measure the diversity
of M, not call frequency.** Measured on Qwen3-8B decode: 617 distinct M values, and the single
most valuable shape (M=64, executing ~110k times) logged **twice** — while a diffuse prefill
band logged hundreds of times for far less wall clock. Ranking targets by miss count inverts
the priority order. Rank by measured time.

Zero means either the flag is unset or the deploy did not take, in that order of likelihood.
Distinguish them with the miss line: if it still names your shape, the deploy really did not
take; if it has gone quiet, the lookup is fine and the logging is what is missing.

Only after engagement is established does an end-to-end timing mean anything. Restore from
`.bak` and re-measure to confirm the delta is attributable to your rows and not to drift — and
note that any config change requires a process restart, so your noise floor is the
restart-to-restart spread, not the repeat-the-benchmark spread. These differ by ~26× on this
workload; see `../tuning-core/measurement.md` Rule 3b.

## 7. The MX (MXFP4/MXFP8) ops on gfx950: a different table, and four traps

The microscaled ops are the gfx950-only surface and they do **not** go through
the CSV mechanism the rest of this skill describes. They are Triton kernels
configured from JSON under `aiter/ops/triton/configs/`, keyed on arch and shape
rather than on `cu_num`, and read through a per-op `_get_config(M, N, K)` rather
than through the tuner. §1–§6 do not apply to them.

**MXFP8 is not here.** On both shipped images the MX surface is FP4-only:
`gemm_afp4wfp4`, `gemm_a16wfp4`, `gemm_a8wfp4`, `batched_gemm_afp4wfp4`,
`fused_moe_mxfp4`, `dynamic_mxfp4_quant`. There is no `gemm_afp8wfp8` and no
`dynamic_mxfp8_quant`. Newer aiter source has them, but mostly bound to
**gfx1250** code objects, which do not load on MI355. For an MXFP8 GEMM on this
part, use CK — see `../tuning-ck/SKILL.md` §2c. Where FP8 appears on the gfx950
MX path it is as the activation side against FP4 weights (`gemm_a8wfp4`).

Four faults worth knowing before writing anything against these ops. Each was
hit while building corpus cases, and none of them reports the actual problem:

**`_get_config` does not take the same K in every op.** The batched op derives
its shapes from the packed tensors it is handed, so it works in `K/2`; the
unbatched op takes logical `K`.

```python
gemm_afp4wfp4._get_config(M, N, K)              # logical K
batched_gemm_afp4wfp4._get_config(M, N, K // 2) # packed K
```

The knob that overruns is `SPLITK_BLOCK_SIZE`: the op sets it to `2*K` and then
strides by `SPLITK_BLOCK_SIZE // 2` over a buffer of `K/2` packed bytes, so the
logical-K call strides twice the buffer. `BLOCK_SIZE_K` comes back at 256 either
way on most shapes, which is why this is hard to spot by diffing configs.

Since it is an out-of-bounds read, the symptom is undefined and not stable
enough to debug by observation: the same mistake gives `Memory access fault by
GPU node-5 ... Reason: Unknown` (no traceback, process dies) at K=4096, a silent
`nan` at K=256, and a plausible in-tolerance answer on a rerun of that same
call. A clean run does not mean the convention was right. Nothing in any of the
three outcomes mentions K.

**Ops with an output parameter return `None`.** `gemm_a8wfp4`,
`batched_gemm_afp4wfp4` and `fused_moe_mxfp4` all write into a buffer you pass
and all have docstrings promising `Returns: torch.Tensor`. Read the buffer.

**`batched_gemm_afp4wfp4(y=None)` raises.** `y` is typed
`Optional[torch.Tensor] = None`; the body does `By, _, _ = y.shape` before any
check. It is required.

**In `fused_moe_mxfp4`, `BLOCK_SIZE_M` changes the input.** The token-to-block
map from `moe_align_block_size_triton` is built *from* `BLOCK_SIZE_M`. Sweeping
that knob against a map computed once from the default config feeds candidates a
token layout built for a different block size, and nothing range-checks it.
Rebuild the alignment per config.

The payoff for going through all that is that these tables are thin —
`gfx950-MOE-MX_FP4.json` is 689 bytes and holds one config for every shape — so
there is a lot on the table. The corpus cases find 34.8% on `gemm_a8wfp4` and
73.1% on `gemm_mxfp4_batched`, the largest uplift in the corpus.

## Checklist

- [ ] `get_gfx()` / `get_cu_num()` recorded; shipped rows checked for *your* key
- [ ] **aiter ≥ v0.1.15** for blockscale GEMM tuning — i.e. contains `7136b240e` (§2b). Below that
      floor, stop and raise the runtime; only `asm` rows are deployable
- [ ] **runtime can select a kernel, not just a library** — *your* production op takes `kernelName`
      and the wrapper passes it (§2b). The version floor covers blockscale only; 53 ops still do not
- [ ] `AITER_LOG_TUNED_CONFIG=1` baseline taken before any change — and if the environment is
      frozen so you cannot set it, the flag-free substitute is in place instead: miss-line
      disappearance plus an untuned negative control (§5)
- [ ] shape list **harvested** from that baseline log — not authored, not a generated grid, and
      re-harvested for **this** build, not carried over from another image (§2). Targets ranked
      by **measured time**, never by log-line count, which measures diversity of M (§6)
- [ ] **correct tuner identified for the op** — gradlib for dense bf16, a per-op
      `csrc/ck_gemm_*/..._tune.py` for anything quantized (§4)
- [ ] tuner source version matches the installed `aiter.__version__`
- [ ] floor established: a `torch` row for gradlib, or — for per-op quantized tuners, which have
      no torch candidate — the CK `kernelId=0` fallback **measured on your own harness** (§4b)
- [ ] `-k/--splitK` decision made deliberately; it is off by default and the zero-filled
      `splitK` column does not mean the space was searched (§4b)
- [ ] `err_ratio` / `errRatio` inspected per row before deploying
- [ ] **every winner re-timed cold on a harness you control** before it ships — the tuner's `us`
      is warm-cache and was 23% optimistic on one shape here (`../tuning-ck/` §3c)
- [ ] rows deployed as a uniquely named drop-in under the **package** `configs/model_configs/`,
      not appended to a shipped file and not written to `/tmp/aiter_configs/` (§5)
- [ ] config files **verified unmodified after the first post-deploy import** — the merge
      silently resolves duplicate shapes by `us` and rewrites your source file (§5)
- [ ] `.bak` kept; lookup transition observed
- [ ] engagement pre-gate passed **before** any e2e A/B — `is tuned on cu_num` > 0 **with
      `AITER_LOG_TUNED_CONFIG=1` set**, misses → 0, and kernel identity confirmed in a trace for
      anything you intend to publish
- [ ] **selection-fidelity pre-gate passed** — the *production* entry point reproduces the tuned
      row's recorded `us` (§2b). Not `*_tune(kernelId=…)`, not `--run_config`
- [ ] incumbent floor measured through the production path, so a "null" can be told apart from an
      unexecutable win (§2b corollary)
- [ ] A/B run with the **reference leg repeated** (ref → cand → ref) so drift is measured rather
      than assumed. A config or CSV change cannot take effect without a process restart, so
      interleave **across restarts** and compare distributions — the same-session spread
      understates the real floor by ~26× here (`../tuning-core/measurement.md` Rule 3b)
- [ ] correctness gated against a **same-config control** (ref vs ref), requests issued one at a
      time — see `../tuning-core/correctness_gates.md`
