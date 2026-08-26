---
name: tuning-triton
description: Tune Triton and Gluon GPU kernels on AMD Instinct — build a config space including the AMD-only knobs (waves_per_eu, matrix_instr_nonkdim, kpack), prune it before racing, set the autotune key correctly, and confirm the win exceeds the noise floor. Use when the kernel you are tuning is written in Triton or Gluon.
---

# Tuning Triton kernels

Read `../tuning-core/SKILL.md` first — this skill is the Triton specialization of that loop.

Triton is the one backend where you tune by **authoring the search space**, not by invoking a
tuner. `@triton.autotune` will faithfully race whatever list you hand it, so the quality of
your result is capped by the quality of that list. Three things decide the outcome:

1. whether the AMD-specific knobs are in the space at all (they are easy to omit silently);
2. whether the `key=` is right (getting it wrong costs multiples, silently);
3. whether you pruned before racing (an unpruned space wastes most of the compile budget).

Measurements below are gfx942 / MI300X, Triton 3.6.0, bf16.

## 1. Put the AMD knobs in the space

This is the trap that costs the most performance, because it fails as a `TypeError` you can
"fix" by deleting the knob.

`triton.Config.__init__` accepts only `num_warps`, `num_stages`, `num_ctas`, `maxnreg`,
`pre_hook`, `ir_override`. Passing an AMD knob as a keyword raises:

```python
triton.Config({'BLOCK_M': 64}, waves_per_eu=2)
# TypeError: Config.__init__() got an unexpected keyword argument 'waves_per_eu'
```

The knobs are real, but they travel in the **first positional dict**, alongside your
`tl.constexpr` block sizes:

```python
triton.Config({'BLOCK_M': 64, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 2},
              num_warps=8, num_stages=2)   # -> accepted; all four reach the backend
```

Verify rather than trust: `cfg.all_kwargs()` should list them. They are consumed by the HIP
backend, whose full option set you can read on any box with

```python
import dataclasses
from triton.backends.amd.compiler import HIPOptions
[f.name for f in dataclasses.fields(HIPOptions)]
```

The three worth searching, and what they trade:

| knob | default | what it controls | when it matters |
| --- | --- | --- | --- |
| `matrix_instr_nonkdim` | 0 (auto) | MFMA instruction shape (e.g. 16 → 16×16, 32 → 32×32) | most impactful; small tiles and skinny shapes often want 16 |
| `kpack` | 1 | K-elements packed per load into the MFMA operand | second-order; interacts with `BLOCK_K` |
| `waves_per_eu` | 0 (auto) | occupancy hint — caps registers to fit more waves per EU | latency-bound kernels; can hurt register-heavy ones |

Measured on 4096³ bf16, tile `128×128×64`, `num_warps=8`, `num_stages=2`, on both parts
(`tuning_benchmark/tools/triton_tables.py` reproduces this):

| variant | gfx942 TFLOPS | spread | vs base | gfx950 TFLOPS | spread | vs base |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (all auto) | 356.8 | 5.4% | — | 760.5 | 5.1% | — |
| `matrix_instr_nonkdim=16` | 384.3 | 0.8% | **+7.7%** | 791.1 | 1.6% | **+4.0%** |
| `matrix_instr_nonkdim=32` | 372.5 | 7.7% | +4.4% | 757.9 | 2.8% | −0.3% |

Read that table the way the next section says to, not the way it first looks. The conclusion
survives the move — `nonkdim=16` is the real win on both parts, `nonkdim=32` is not
distinguishable from nothing on either — but the magnitude halves and `nonkdim=32` crosses
from marginally positive to marginally negative. That is the general pattern for this whole
skill on gfx950: the knob that matters still matters, and by less.

The gfx950 column was itself only measurable after interleaving the three variants. Timed
back-to-back on that part they report 13.8–20.7% spreads, which swallows the entire result and
would have made this table read as three indistinguishable numbers. See
`../tuning-core/measurement.md` Rule 6b before reproducing any of this on MI355.

## 2. Compare the win to the noise floor, not to zero

Both variants above are "faster than baseline". Only one of them is a result.

- `nonkdim=16`: +7.7%, and its own spread is 0.8%. The effect is an order of magnitude
  larger than the measurement's own variability. **Real.**
- `nonkdim=32`: +4.4%, spread 7.7%. The claimed gain is smaller than the noise in the
  measurement that produced it. **Not distinguishable from nothing.**

A single `do_bench` call reports neither of these facts. It returns one number for each and
they both look like wins. The discipline — take independent repeats, report the median and
the spread, and refuse to believe a delta smaller than the spread — is what separates them.
Full rules in `../tuning-core/measurement.md`.

Practical consequence for autotuning: `@triton.autotune` picks the single fastest config from
one timing pass. When several configs sit inside each other's noise band, which one "wins" is
partly chance. That is acceptable — you get *one of* the good configs — but it means you must
not report the autotuner's margin as a speedup without re-measuring the winner properly.

## 3. Set `key=` to every variable that changes the answer

`key` is the list of kernel arguments whose values invalidate the cached choice. Autotuning
runs once per distinct key tuple; every later call with a matching tuple reuses that config.

Omit a dimension and you silently serve a config tuned for a different shape. Same kernel,
same config list, only `key` differs:

| shape | `key=['M','N','K']` | `key=['N','K']` |
| --- | --- | --- |
| M=4096 N=4096 K=4096 | 348.9 µs — picks `BM=256,BN=128` | 350.2 µs — picks `BM=256,BN=128` |
| M=1 N=4096 K=4096 | **43.4 µs** — picks `BM=16,BN=64` | **162.9 µs** — reuses `BM=256,BN=128` |

**3.75× slower on the decode shape, no error, no warning.** The big-M config was cached
first; M is not in the key, so the M=1 call is treated as the same problem. A `BM=256` tile
on a 1-row GEMM computes 255 rows of padding.

On gfx950 the same experiment gives **2.45×** (M=1: 54.5 µs with the decode tile, 22.2 µs with
the right one — and 180.6 µs vs 479.7 µs at M=4096 in the other direction). Smaller, still
multiples, still silent. The penalty shrinks because a 256-row tile wastes proportionally less
on a part with more bandwidth per CU, not because the mistake became safe.

Rules that follow:
- include every dimension the kernel branches or tiles on — for a GEMM that is `M`, `N`, `K`;
- include dtype-selecting or feature-flag arguments if they change which code path runs;
- if a value is genuinely constant, make it a `tl.constexpr` rather than trusting yourself to
  remember it is constant.

Cost of being right: one autotune pass per distinct shape. In serving, bound that by
bucketing M (see `../tuning-core/search_strategy.md`) rather than by shortening the key.

## 4. Prune before you race

Every config in the list costs a compile plus a timing pass, and the space is multiplicative:
block sizes × warps × stages × three AMD knobs reaches thousands of entries. Most are
invalid — they exceed LDS, or give threads no work — and Triton finds that out by compiling
them.

Filter analytically first. The pruning predicate and its measured reduction (1600 → 912
configs, 57%) are in `../tuning-core/search_strategy.md`; `example_autotune.py` here shows it
wired into a real kernel.

Then narrow by regime — this is worth more than it sounds. Same kernel, same autotuner, same
M=1 N=4096 K=4096 shape, only the *space* differs:

| space raced | winner | gfx942 | gfx950 |
| --- | --- | --- | --- |
| general (`BM ∈ 64,128,256`) | `BM=64, BN=64, BK=64, nonkdim=16` | 0.041 ms | 0.041 ms |
| decode (`BM ∈ 16,32`, deep `BK`) | `BM=16, BN=32, BK=256, nonkdim=0, warps=2` | **0.027 ms** | **0.023 ms** |

**34% faster from changing the candidate list alone on gfx942, and 44% on gfx950.** The general
space simply did not contain the right answer, and an autotuner cannot select what you did not
offer it. Note the winner also chose `nonkdim=0` — the auto MFMA shape — where the square case
wanted 16; another reason a single fused list serves neither regime well.

This is the one measurement in this skill that got *larger* on the newer part. Splitting the
space by regime is the highest-value item here on gfx950, above any individual knob.

One caveat the example makes explicit: `@triton.autotune` is evaluated at import time, so the
config list is fixed before any argument is seen. You cannot choose the space from the shape
inside the decorator. Either accept one larger union list, or define separate kernels per
regime and dispatch between them yourself — a real cost of regime-splitting worth pricing in.

You can also use Triton's own hook, `prune_configs_by=`, which additionally lets you drop
configs using the runtime argument values (`early_config_prune`, `perf_model`, `top_k`).

## 5. Control the cache while you tune

Autotune results and compiled binaries are cached under `/root/.triton/cache`
(`knobs.cache.dir`). During a tuning session this is a hazard: you change the config list,
rerun, and get the previous answer.

```bash
export TRITON_PRINT_AUTOTUNING=1     # print the winning config — do this always
rm -rf /root/.triton/cache           # force a real re-race after editing the space
```

`TRITON_PRINT_AUTOTUNING=1` is the cheapest engagement signal Triton offers: it tells you
autotuning actually ran and which config won. If it prints nothing, you are on a cached
result and your edits did not take effect. Also readable in-process as
`kernel.best_config` after the first call — that is how the `key=` table above was produced.

## 6. Gluon

Gluon (`triton.experimental.gluon`) is the lower-level dialect: you place data layouts by
hand instead of letting the compiler infer them. Everything above still applies — same
`Config`, same `key`, same noise discipline — but the search space gains layout choices
(`BlockedLayout`, `SwizzledSharedLayout`, `PaddedSharedLayout`, `DotOperandLayout`,
`AMDMFMALayout`).

The arch split is explicit in the API, which makes it a good place to reason about gfx942 vs
gfx950:

```python
from triton.experimental.gluon.language.amd import cdna3, cdna4
```

- `cdna3` (gfx942): `mfma`, `buffer_load`/`buffer_store`, `buffer_atomic_*`
- `cdna4` (gfx950): all of the above **plus** `mfma_scaled`, `get_mfma_scale_layout`,
  `async_copy`

So a Gluon kernel written against `cdna3` runs on both; one written against `cdna4` does not
run on gfx942. Tune on the arch you will deploy on — and note that `mfma_scaled` is the entry
point for the microscaled dtypes that only gfx950 has, which is why those shapes have no
gfx942 equivalent to carry a tuned config over from.

Reach for Gluon only after the Triton-level space is exhausted: the extra layout freedom is
also extra ways to be slow, and the compiler's inferred layouts are usually good.

### Tuning someone else's Gluon kernel: the space is smaller than the signature

A hand-placed layout can encode the tile shape, and then the tile shape stops being tunable.
aiter's `gluon.gemm_afp4wfp4` accepts the usual ten-key GEMM config and writes its layouts as
literal basis vectors:

```python
linear_mn = gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [0, 4], [0, 16], [0, 128], [64, 0], [128, 0]],
    warp_bases=[[0, 32], [0, 64], [32, 0]],
    shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
)
```

A basis of 128 needs a dimension above 128, and three warp bases mean eight warps. The tile
is pinned at 256×256×256 on `num_warps=8` and nothing in the signature says so. Measured on
gfx950, one knob moved at a time:

| knob | result |
| --- | --- |
| `BLOCK_SIZE_M/N/K`, `num_warps` | `LLVM ERROR ... abort()` — the process dies |
| `num_stages`, `matrix_instr_nonkdim` | accepted, declared, never read; median moves <2% |
| `GROUP_SIZE_M`, `waves_per_eu`, `NUM_KSPLIT`, `cache_modifier` | real |

Three consequences worth carrying to any hand-written-layout kernel:

**Probe the space in subprocesses.** The tile-shape failure is `abort()`, not an exception —
no `try` will catch it, and an in-process sweep loses every result it had gathered. One
subprocess per candidate, judged by exit code, is the only safe way to map the boundary. Do
it once and turn the answer into a constraint.

**A thin config file is evidence about degrees of freedom, not effort.** The shipped Gluon
config is a single `"any"` entry, against M-binned entries across forty-odd N/K-specialised
files for the Triton path to the same math. That looks like an untuned kernel and is not: one
entry covers a four-knob surface. Sweeping it found exactly one win, `NUM_KSPLIT=2` at
M=16 (11.2%), which is the axis you would predict matters at decode shapes and the one thing
a single shape-independent entry cannot get right.

**Declared is not read.** `num_stages` and `matrix_instr_nonkdim` are in the kernel signature
and absent from the body — Gluon pipelines by hand and the MFMA shape is hardcoded. Leaving
them in the tunable list buys recompiles of identical machine code, and their apparent 1-2%
"wins" are noise that a budget-limited sweep will happily report.

## 7. Then verify engagement

A tuned Triton config is only useful if the framework calls *your* kernel. In vLLM and SGLang
many Triton kernels are selected by env var or by a config JSON keyed on shape and device
name — and the device-name lookup has its own failure mode (see `../tuning-in-vllm/`). Finish
with `../tuning-core/engagement_verification.md`.

## Files

- `example_autotune.py` — a GEMM with pruning, AMD knobs, correct `key`, and a noise-aware
  comparison. Runnable; written to be read and adapted, not imported.

## Checklist

- [ ] AMD knobs are inside the positional dict, and `all_kwargs()` confirms it
- [ ] `key=` names every dimension that changes the answer
- [ ] space pruned analytically before racing; regimes tuned separately
- [ ] cache cleared after editing the space; `TRITON_PRINT_AUTOTUNING=1` set
- [ ] winner re-measured with repeats; reported gain exceeds the spread
- [ ] correctness gated on a relative metric (`../tuning-core/correctness_gates.md`)
- [ ] engagement proven in the real workload
