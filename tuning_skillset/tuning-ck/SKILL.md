---
name: tuning-ck
description: Tune GEMMs on the Composable Kernel path — race CK's compiled instances with ckProfiler, read the instance name to understand why one won, and route the result into a framework through aiter's per-op CK tuners. Use when the op dispatches through CK or when establishing what a CK path could reach.
---

# Tuning Composable Kernel

Read `../tuning-core/SKILL.md` first. Install the profiler per
`../env-setup/ckprofiler_install.md` — it is an apt package, about a minute, not a source
build.

Like hipBLASLt, CK is a **selection** problem: CK ships pre-compiled instances and tuning
means finding which one wins for your shape. Unlike hipBLASLt, the instance name is fully
descriptive, so a CK sweep also tells you *what shape of kernel* your problem wants — useful
even when you end up deploying through a different backend.

**Scope, stated up front.** This skill covers selection: `ckProfiler` races the 71 ops the
profiler exposes (39 of them GEMM-family), and aiter's seven per-op CK tuners deploy the
result. It does **not** cover CK's other tuning surface. `tile_engine/` in the CK source is a
JSON-declared config space (`tile_m/n/k` as `{min,max,step}`, `warp_*` as value lists) that
*generates and compiles instances that do not yet exist* — the Triton model, in CK C++. If
your shape's best kernel is not in the shipped library, selection cannot reach it and
`tile_engine` is where you go. Nothing here describes that path; `../docs/coverage_gfx950.md`
§11 records it as an open gap rather than pretending it is covered.

Measured on gfx942 / MI300X **and** gfx950 / MI355X, ROCm 7.2.2, bf16. Numbers below are
labelled per part; where only one is given it is gfx942. The apt package ships gfx950
instances, so nothing about the install changes between the two.

## 1. The interface is positional

`ckProfiler` takes positional arguments, not flags. Run `ckProfiler <op>` with nothing else
and it prints its own legend — do that for any op you have not used lately, since the legend
is authoritative for your build:

```
arg1: tensor operation (gemm: GEMM)
arg2: data type (0: fp32; 1: fp16; 2: bf16; 3: int8; 4: fp8)
arg3: matrix layout (0: A[m,k]*B[k,n]; 1: A[m,k]*B[n,k]; 2: A[k,m]*B[k,n]; 3: A[k,m]*B[n,k])
arg4: verification (0: no; 1: yes)
arg5: initialization (0: no init; 1: integer; 2: decimal)
arg6: print tensor value (0: no; 1: yes)
arg7: time kernel (0: no, 1: yes)
arg8 to 13: M, N, K, StrideA, StrideB, StrideC
arg14/15: warm-up cycles (default 1), iterations (default 10)
```

`ckProfiler` with no arguments at all lists every op — `gemm`, `batched_gemm`,
`grouped_gemm`, fused variants like `gemm_add_relu`, conv, and more.

A 4096³ bf16 run with verification on, 5 warm-up and 20 iterations:

```bash
ckProfiler gemm 2 1 1 1 0 1 4096 4096 4096 4096 4096 4096 5 20
```

Two arguments are not optional in practice:

- **arg4 verification = 1.** An instance that is fast because it computes the wrong thing is
  a real failure mode, and this is the cheapest place to catch it. Turn it off only for
  timing-sensitive reruns after correctness is established.
- **arg7 time kernel = 1.** Otherwise you get no timings.

The defaults for warm-up (1) and iterations (10) are too few to be stable — raise them, and
read `../tuning-core/measurement.md` on why one run is not a measurement.

**arg3 layout is where results silently become incomparable.** CK's layout codes describe
which operands are transposed; a number from layout 1 cannot be compared against a number
from layout 0. Match the layout to what your workload actually issues.

## 2. Read the instance name, not just the time

Output is one `Perf:` line per instance, then a best line:

```
Perf: 0.381103 ms, 360.634 TFlops, ... DeviceGemm_Xdl_CShuffle<MNKPadding, 256, 128, 128, 32, 8, 8, 32, 32, 2, 2, 8, 8, 1, 1> ... PipelineVersion: v2
Perf: 0.983496 ms, 139.745 TFlops, ... DeviceGemm_Xdl_CShuffle<MNKPadding, 128,  32, 128, 32, 8, 8, 32, 32, 1, 2, 8, 8, 1, 1> ... PipelineVersion: v2

Best Perf ... 0.251566 ms, 546.334 TFlops, 400.147 GB/s,
  DeviceGemm_Xdl_CShuffleV2<Default, 256, 256, 256, 32, 8, 8, 32, 32, 4, 4, 8, 8, 1, 1>
  LoopScheduler: Default, PipelineVersion: v1
```

**7× spread between the best and worst instance on one shape** (546 vs 140 TFLOPS). That
range is the value of racing: picking an instance by intuition has a wide floor.

The same command on gfx950 / MI355X, 126 instances timed:

```
Best Perf ... 0.110352 ms, 1245.46 TFlops, 912.204 GB/s,
  DeviceGemm_Xdl_CShuffleV2<Default, 256, 256, 256, 32, 8, 8, 32, 32, 4, 4, 8, 8, 1, 1>
  LoopScheduler: Default, PipelineVersion: v1
```

| | gfx942 | gfx950 |
| --- | --- | --- |
| best | 546.3 TFLOPS | **1245.5** |
| worst | 139.7 | 186.9 |
| best/worst spread | 7× | 6.4× |
| winning instance | `CShuffleV2<Default, 256,256,256,…>` v1 | **byte-identical** |

The spread is the same order on both parts, so the argument for racing holds unchanged. The
striking part is the last row: the winner is the *same instance string*, including padding
strategy, tile, MFMA shape, pipeline and loop scheduler. Two parts with different CU counts
and 2.5× the LDS agreed on which of 126 instances to pick.

Do not over-read that. It is one shape, and the hipBLASLt tuner's winner on the same shape
*did* move between parts (a `224`-deep tile to a `256`-deep one). What it supports is the
weaker and more useful claim in the next paragraph — the tile is evidence about the problem —
rather than "CK winners are portable". Re-race per part; expect the answer to be recognisable.

The template parameters are the tuning knobs, made visible:

| position | meaning |
| --- | --- |
| 1st field | padding strategy (`Default`, `MNKPadding`) |
| next three | block tile M, N, K (`256, 256, 256`) |
| following | K-per-block, scalar-per-vector widths |
| `32, 32` | MFMA instruction M/N |
| next pair | waves per block in M and N |
| tail | C-shuffle and vector-write configuration |

Plus `PipelineVersion` and `LoopScheduler`, which control software pipelining. Note the
winner here is `CShuffleV2 ... PipelineVersion: v1` — a *different device op and pipeline*
from the v2 instances just below it, not merely a different tile. When you compare winners
across shapes, compare all three: op variant, tile, pipeline.

This is why a CK sweep is worth running even if you deploy elsewhere: `256×256×256` winning
by 1.5× over `256×128×128` on this shape is transferable evidence about what the problem
wants, and it corroborates the macro-tiles chosen independently by the hipBLASLt and aiter
tuners on the same shape.

### 2b. The candidate pool is not the same on both parts, and the gate is a string

"126 instances timed" is not a property of the op. CK's instance registration contains
**runtime device-name checks** that add or remove instances before you ever see them. In
`library/src/tensor_operation_instance/gpu/`:

```cpp
add_device_operation_instances(instances, ..._comp_instances<GemmDefault>{});

if(ck::get_device_name() != "gfx950")
{
    add_device_operation_instances(instances, ..._comp_instances_part2<GemmDefault>{});
}
```

The `part2` list is introduced in its header by the comment `// instances not working on
gfx950`. Across the instance tree there are **166 such gates: 87 excluding on gfx950 and 79
adding only on gfx950**. They are concentrated exactly where LLM serving lives —
`gemm_universal` alone has 45 exclusion sites and 6 gfx950-only ones, plus
`gemm_universal_streamk` (10), `gemm_universal_reduce` (8 / 4), `batched_gemm` (4 gfx950-only),
`grouped_gemm_tile_loop` (4), and the three `gemm_*fastgelu` fusions (4 each).

Three consequences:

- **An instance count is not comparable across parts.** Fewer instances on gfx950 for a given
  op may mean the deny-list fired, not that codegen produced less.
- **A tile that won on gfx942 may not be in the pool on gfx950 at all.** The excluded bf16
  `part2` lists contain the deep `256×256×224`-class tiles; if you carry a gfx942 winner over
  and ckProfiler never reports it, check the deny-list before concluding the instance is slow.
- **It is an exact string compare on `get_device_name()`**, the same failure class as the
  framework device-name split in `../tuning-in-vllm/` §1. Anything reporting a device name that
  is not literally `gfx950` takes the other branch.

None of this is logged. The only way to see it is to read the registration file for your op.

## 2c. `gemm_mx` — and why it is the only route to MXFP8 on MI355

The microscaled formats are CDNA4-only and are the reason to care about gfx950
at all, so it matters that **CK is where they are reachable**. On the shipped
images, MXFP8 has no aiter operator (`../docs/coverage_gfx950.md` §12): the
MXFP8 Python surface exists in newer aiter source but not in either installed
build. CK's `gemm_mx` is present and works.

It is a separate profiler op with its own positional list — the `arg2` slot is
the *format*, not a layout:

```bash
# MXFP4  (f4 -> f16), layout 1 = A[m,k] * B[n,k], verify on, 4096x4096x8192
ckProfiler gemm_mx 0 1 1 1 0 1 4096 4096 8192 -1 -1 -1 1 5 20 0
# MXFP8  (f8 -> f16)
ckProfiler gemm_mx 1 1 1 1 0 1 4096 4096 8192 -1 -1 -1 1 5 20 0
# MXFP8  (f8 -> bf16)
ckProfiler gemm_mx 2 1 1 1 0 1 4096 4096 8192 -1 -1 -1 1 5 20 0
```

`arg2` is `0: f4->f16`, `1: fp8->f16`, `2: fp8->bf16`. `arg3` is the layout
(`0` MK_KN, `1` MK_NK, `2` MK_MFMA / B-preshuffled). Everything after that
matches the ordinary `gemm` op.

Measured on gfx950, 4096·4096·8192, verification on:

| format | instances | best | worst | spread |
| --- | --- | --- | --- | --- |
| MXFP4 `f4->f16` | 11 | **3844 TFLOPS** | 1375 | 2.8× |
| MXFP8 `f8->f16` | 5 | 1765 | 778 | 2.3× |
| MXFP8 `f8->bf16` | 5 | 1758 | 781 | 2.3× |

Three things worth taking from that table.

**The spread justifies racing.** 2.3–2.8× between best and worst instance is the
same order as the bf16 case in §2, so the argument for measuring rather than
guessing survives into the MX formats.

**The candidate pool is small and asymmetric.** Eleven FP4 instances against
five FP8. This is the apt-packaged profiler; the CK source tree carries 67 MX
instances across six type combinations (`f4/f8/f6/bf6/bf8`), so what you can
race is a fraction of what CK can build, and `gemm_mx` is gated to `gfx95` at
build time. If you need an instance that is not in the shipped set, that is a
build, not a flag.

**The instance names confirm the format, and are worth reading for that alone.**
The winner prints as `ck::f4x2_pk_t` (two FP4 packed per byte) or `ck::f8_ocp_t`
— the OCP dialect, not FNUZ, which is the gfx950 side of the inversion in
`../tuning-core/measurement.md`. Every MX instance also prints
`ScaleBlockSize: 32`, which is the one number the whole format hangs on: one
E8M0 exponent per 32 elements along K. A split-K or block-K choice that does not
divide 32 cuts a scale group, and the result is finite, plausible and wrong —
the same trap the FP4 corpus cases guard against with an explicit
`NUM_KSPLIT` constraint.

These are race numbers, so read them under `../tuning-core/measurement.md`
Rule 6b: on gfx950 a back-to-back race inflates. The 2.3–2.8× *ranking* is far
outside that error, but the absolute TFLOPS are not deploy-grade until replayed.

## 3. ckProfiler does not change what your framework runs

A ckProfiler winner is a fact about CK, not a change to your workload. Nothing in your
serving path consults ckProfiler output.

Use it for:

- **Establishing a ceiling** — what could a CK path reach for this shape, before investing in
  wiring one up.
- **Backend triage** — is CK even competitive here versus hipBLASLt or Triton? Run the same
  shape through each and compare. On 4096³ bf16 CK's best was 546 TFLOPS against hipBLASLt's
  633 on gfx942, and 1245 against 1241 on gfx950 — worth knowing before spending a day on the
  CK route. Note that the gap closed to nothing on the newer part, so this is a question to
  re-ask per part rather than a settled ranking.
- **Sanity-checking** a suspicious result from a higher-level tuner.

To actually change dispatch, go through aiter's per-op CK tuners, which write config CSVs the
runtime reads. **There are seven**, and which one you want is set by the dtype pair:

```
csrc/ck_gemm_a8w8/gemm_a8w8_tune.py                         fp8/int8 per-tensor
csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py   fp8 block-scaled
csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py fp8, pre-shuffled B
csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py   FP4 -- gfx950 only
csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py         batched fp8
csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py         batched bf16
csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py           fused-MoE experts
```

Earlier versions of this file also listed `csrc/gemm_a16w16/` and `csrc/opus_gemm/`. Neither
exists in the aiter shipped in these images; there is no dense bf16 CK tuner. bf16 goes through
gradlib instead (`../tuning-aiter/` §4).

Prefer the umbrella entry point over calling the scripts directly:

```bash
python3 aiter/utility/pretune.py --list      # 8 tune modules
python3 aiter/utility/pretune.py all
```

It auto-detects arch and `cu_num`, tags results with them, and rebuilds the inference `.so` with
the winners afterwards — which the individual scripts do not do. It lists eight modules against
seven scripts because the `cktile` variants of `a8w8_blockscale` and `a8w8_bpreshuffle` share
their parent's tuner.

**Do not use the names `--list` prints as environment variables.** It prints
`AITER_CONFIG_BF16_BATCHED_GEMM_FILE`, which is the resolver property on
`aiter.jit.core.AITER_CONFIGS`. The variable the runtime reads is the same name **without
`_FILE`**. Verified on gfx950:

| exported | resolves to |
| --- | --- |
| nothing | `…/aiter/configs/bf16_tuned_batched_gemm.csv` |
| `AITER_CONFIG_BF16_BATCHED_GEMM=/work/mine.csv` | `/work/mine.csv` |
| `AITER_CONFIG_BF16_BATCHED_GEMM_FILE=/work/mine.csv` | `…/aiter/configs/bf16_tuned_batched_gemm.csv` |

The `_FILE` form is accepted by the shell, changes nothing, and warns about nothing.

These live in the aiter source tree — present in the sglang image, absent from the vllm wheel
(`../env-setup/image_tool_matrix.md`). The deploy target and its trap are in
`../tuning-aiter/`.

### Verified on gfx950

All seven were run end to end on MI355, two shapes each, writing rows tagged
`gfx950, cu_num=256`:

| tuner | wall | outcome |
| --- | --- | --- |
| `batched_gemm_bf16` | 98 s | null — 0.05%, 2.11% |
| `gemm_a4w4_blockscale` | 120 s | null — −0.13%, 0.34%; 4993 TFLOPS FP4 |
| `gemm_a8w8_blockscale` | — | null — 0.45%, −0.15% |
| `gemm_a8w8_bpreshuffle` | 343 s | **4.53% win** at M=16 |
| `batched_gemm_a8w8` | — | **6.78% win** at B=32 M=1024 |
| `gemm_a8w8` | 175 s | **gate broken** — writes a 0.82x regression, see §3b |
| `gemm_moe_2stages` | 76 s | **crashes on the shipped input**, see below; then 4.23% bf16 and **1.73x** fp8 |

Wall time is modest but CPU cost is not — the a8w8 run spent 205 CPU-minutes in 3 wall-minutes
compiling candidates. Budget cores, not just time. There is no `--indtype` flag on any of these
(that bug is specific to gradlib's `gemm_tuner.py`), so the CSV-column workaround does not
apply here.

**Two shapes is not a cost model.** A 23-shape FP8 preshuffle run across all four libtypes took
~50 min, ~14× what the two-shape figures above extrapolate to. Time your own two shapes on your
own libtype selection before scoping a sweep.

**`-k/--splitK` is off by default and is not in any of the numbers above.** It is
`action="store_true"` (`aiter/utility/base_tuner.py:130`), and with it unset the tuner races no
split-K candidate at all for either the `ck` or `cktile` path — `useSplitK = args.splitK` gates
the sweep. The result schema carries a `splitK` column regardless, so an unsearched space looks
like a searched one that always chose zero. Pass it for tall-skinny-M or long-K shapes and budget
~4× the runtime.

**`gemm_moe_tune.py` cannot be run with its own default input on gfx950.** Eleven of the
thirteen rows in `aiter/configs/untuned_fmoe.csv` specify `torch.float8_e4m3fnuz`, and FNUZ is
the gfx942 FP8 dialect. The first such row raises

```
KeyError: torch.float8_e4m3fnuz
```

inside the dtype-name lookup, and the exception takes down the whole run — including shapes that
had already enumerated their candidates. The summary then reads `tune 0 shapes` with an error,
so the failure is at least loud. Filter the input first:

```bash
grep -v fnuz aiter/configs/untuned_fmoe.csv > /tmp/untuned_fmoe_950.csv
```

With that, both remaining shapes tune, and this is the largest win of the seven: bf16 MoE 390.04
→ 373.53 µs (4.23%), and OCP-fp8 per-token MoE 377.33 → **217.74 µs, a 1.73x speedup** at 818
TFLOPS. The default shape list is a gfx942 artifact, so re-deriving the shape list for this part
is not a formality — it is where the result came from.

Two things to watch in that run. The bf16 path reports
`ASM kernel list file not exist: hsa/gfx950/fmoe_2stages/fmoe_stage1_bf16_pertoken_g1u1.csv`,
so bf16 MoE races 15 CK candidates and **zero** ASM ones, while the fp8 path gets 18 ASM plus 25
CK. And the tuner's correctness check passed a candidate at
`max abs delta 0.625, 4.4% of elements` — under the default `--errRatio 0.05` that is a pass.
Tighten `--errRatio` if 4% of elements moving matters for your model.

## 3b. The `--compare` gate, and the hole in it

These tuners have something the rest of the ecosystem does not: a built-in before/after check.

```bash
python3 csrc/ck_gemm_a8w8/gemm_a8w8_tune.py -i shapes.csv -o tuned.csv --compare
```

It benchmarks the production op with default dispatch, tunes, benchmarks again, and **refuses to
write a config that wins by less than 3%** (`--min_improvement_pct`). That is the noise-floor
discipline of `../tuning-core/measurement.md` enforced by the vendor's own tool, and on gfx950 it
worked as advertised in six of the seven tuners: three returned pure nulls, bpreshuffle skipped a
−0.52% and passed a 4.53%, batched a8w8 skipped a 0.35% and passed a 6.78%, MoE passed a 4.23%.

Those nulls came from square, cache-friendly probe shapes near 4096³. **Do not generalize them
into "there is no headroom on large shapes"** — an earlier version of this document said aiter's
default CK dispatch on gfx950 "is already at or near the tuned optimum" on large dense shapes, and
a serving shape falsified it outright:

| `a8w8_blockscale_bpreshuffle`, M=15104 N=34816 K=5120, gfx950 | time | TFLOPS |
| --- | --- | --- |
| default dispatch | 4153.1 µs | 1297 |
| tuned instance, through the production wrapper | 2655.8 µs | 2028 |

**−36% on the single largest GEMM in a live 27B serving mix, and +23.88% e2e throughput** once the
whole harvested shape list was tuned. What distinguishes it from the 4096³ probes is not size but
*shape*: a tall, thin, non-square problem at a serving M, which is not what the shipped tables are
populated for. So the real rule is the coverage one, and it is stronger than "look for untabulated
ops": point these tuners at the shapes **your workload actually dispatches** (`../tuning-aiter/`
§2), because that is where the shipped tables stop being representative. Batched GEMM (zero
`cu_num=256` rows in either image) and fused MoE (fp8 path, 1.73x) are instances of that rule, not
the whole of it.

One caution on reading a null from `--compare` specifically: it benchmarks the *production* op
after tuning, so on a build where the serving wrapper cannot select a tuned instance
(`../tuning-aiter/` §2b — measured to cost −6.48% e2e on one such build) its post-tune leg can be
measuring the library's internal default rather than your winner. Before trusting a null, confirm
the production path can reproduce a tuned row's recorded `us` at all.

**`gemm_a8w8_tune.py` does not gate correctly, and installs a regression.** On
`M=1, N=7168, K=8192` fp8:

```
(1, 7168, 8192, torch.float8_e4m3fn)  |  14.78 |  17.97 |  0.82x |  OK
Total shapes: 2 | Updated: 2 (improved: 0, new: 2) | Skipped: 0
--- Updated (2 shapes) ---
(1, 7168, 8192, torch.float8_e4m3fn)  |  14.78 |  17.97 |   N/A  |  NEW
```

The tool measured a 22% slowdown, printed it, labelled it `OK`, reported the improvement as
`N/A`, classified the shape as `NEW`, and with `--update_improved` wrote the row. `NEW` is
supposed to mean "no baseline to compare against", which is the one case the threshold is allowed
to skip — but there plainly was a baseline, it is printed on the same line, and the shape already
exists in `a8w8_tuned_gemm.csv`. The gate's pre-run and post-run failed to line up and both were
treated as baseline-less.

It is specific to this tuner, not to the shared `base_tuner.py`: `gemm_a8w8_bpreshuffle_tune.py`
has the same key columns including the string `q_dtype_w`, ran the same two shapes, and produced
correct percentages.

The slowdown itself is real and repeatable — four runs gave 0.79x, 0.82x, 0.85x, 0.86x. The best
CK instance for that shape is genuinely slower than what aiter dispatches by default, so tuning
the CK table *pins* the op to CK and loses. A tuner can only choose among the candidates it owns;
it cannot decline in favour of another backend. That possibility is worth holding onto generally:
**a tuner finding its own best answer is not the same as that answer being the best available.**

So: **read the Speedup column yourself, and do not treat `NEW` as a pass.** Delete sub-1.00x rows
before deploying. Anything below 1.00x is a config that makes the op slower than shipping
nothing.

One more artifact worth knowing: the first `--compare` run after a JIT module build reported
27.63 µs for a shape that measured 16.98, 17.50 and 17.43 µs on the three following runs — 59%
high. Discard the first run after a build. Note also that pre and post are run as two blocks, all
pre then all post, which is exactly the pattern `../tuning-core/measurement.md` Rule 6b warns
against on this part.

## 3c. The `us` column is warm-cache, and these tuners have no invalidation knob

`../tuning-core/measurement.md` Rule 5 says to invalidate caches when the benchmark is not the
real workload, and notes that aiter's tuner has `CACHE_INVALIDATE_BUFFERS` for exactly this.
**That variable does not exist on these tuners.** It lives only in
`gradlib/gradlib/GemmTuner.py:135` (default 37 buffers), i.e. the bf16 tuner. The per-op
quantized tuners time through `run_perftest` on a **single operand set** and never reference it.

So their `us` is partly cache-served — on a part with a 256 MB MALL, a 100 MB weight read can sit
entirely in cache — and the error is not small or one-directional. Five winners from
`gemm_a8w8_bpreshuffle_tune.py` on gfx950, tuner `us` against an independent cold harness:

| shape | tuner `us` | cold `us` | tuner error |
| --- | --- | --- | --- |
| gate_up M=64 | 22.11 | 23.93 | optimistic 8% |
| gate_up M=16384 | 1321.46 | 1319.08 | accurate |
| o_proj M=16384 | 236.52 | 216.99 | **pessimistic 8%** |
| **down_proj M=16384** | **591.38** | **727.69** | **optimistic 23%** |

The `down_proj` row is a shipping decision that would have gone the wrong way. At face value it
is a 19% win over the CK default; cold it is a **dead tie** (727.69 vs 727.61) and slower than
the default in an eager profile. Taken on trust it would have shipped a row that buys nothing,
added a dependency, and published a kernel table claiming a win that does not exist.

**Re-time every winner on a harness you control before it ships**, cold, with rotated buffers,
against the incumbent — which for these tuners is the CK `kernelId=0` fallback, since there is no
`torch` candidate to serve as a floor. `--compare` is not a substitute: it shares the tuner's
timing path, so it inherits the same warm cache.

This also compounds with the duplicate-resolution behaviour in `../tuning-aiter/` §5 — the merge
step arbitrates duplicate shapes by the `us` column, so an optimistic tuner number does not just
mislead you, it can win an argument against a correct row and rewrite your config file.

## 4. Cross-check before believing

ckProfiler reports its own timings from its own harness. Before deploying anything based on
them, re-measure the shape independently:

```bash
python3 ../benchmark/run_case.py --M 4096 --N 4096 --K 4096
```

Agreement within the noise floor means the number is real. Disagreement beyond it means one
harness is measuring something you did not intend — check the layout code (arg3) and the
strides (args 11–13) first, then input initialization. Do not average two disagreeing
harnesses.

## Checklist

- [ ] legend re-read via `ckProfiler <op>` for the op in use
- [ ] verification (arg4) on for the first run of every new shape
- [ ] layout (arg3) matches what the workload issues
- [ ] warm-up and iteration counts raised above the defaults
- [ ] winning instance recorded by full name, not just its time
- [ ] on gfx950, instance registration for the op checked for a `get_device_name()` gate before
      comparing candidate counts against another part (§2b)
- [ ] result cross-checked against an independent harness
- [ ] if using an aiter CK tuner: input shape list re-derived for this part, not the
      shipped one — `untuned_fmoe.csv` is 11/13 FNUZ and aborts on gfx950
- [ ] `--compare` Speedup column read per row; anything below 1.00x deleted, and `NEW`
      not treated as a pass
- [ ] `-k/--splitK` decision made deliberately — it is `store_true` and **off by default**, so a
      zero-filled `splitK` column means the space was never searched, not that it lost. Worth
      passing for tall-skinny-M or long-K shapes, at roughly 4× the runtime
- [ ] every winner **re-timed cold on an independent harness** before deploying; these tuners
      have no cache-invalidation knob and were 23% optimistic on one shape (§3c)
- [ ] floor measured against the CK `kernelId=0` fallback — there is no `torch` candidate here,
      so the gradlib "torch row is your floor" rule does not apply
- [ ] override exported as `AITER_CONFIG_<OP>`, not the `_FILE` name `--list` prints
- [ ] if deploying: routed through an aiter CK tuner, and engagement proven
