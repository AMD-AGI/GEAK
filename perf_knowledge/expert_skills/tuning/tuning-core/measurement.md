# Measurement

If your timing method is wrong, every decision downstream of it is wrong. This page is
short because there are only a few rules, but each is load-bearing.

All numbers below were measured on one MI300X (gfx942), bf16 `torch.mm` at 4096³.

## Rule 1 — GPU work is asynchronous. Synchronize or you are timing nothing.

A kernel launch returns immediately. Timing a loop of launches without a sync measures how
fast you can enqueue, not how fast the GPU computes.

```python
t0 = time.perf_counter()
for _ in range(50): torch.mm(a, b)
t1 = time.perf_counter()                    # NO sync
```

| method | reported |
| --- | --- |
| wall clock, no sync | **4363 TFLOPS** |
| wall clock, `torch.cuda.synchronize()` before stopping | 310 TFLOPS |

MI300X peak bf16 is roughly 1.3 PFLOPS. **4363 TFLOPS is above the hardware limit** — that
is the tell. Always sanity-check a result against peak; a physically impossible number
means a broken harness, not a breakthrough.

If a measurement looks too good, it is a bug until proven otherwise.

## Rule 2 — Warm up.

The first call pays for JIT compilation, autotune cache population, memory allocation, and
clock ramp. Including it drags the average down.

| method | reported |
| --- | --- |
| `do_bench(warmup=0, rep=1)` | 635 TFLOPS |
| `do_bench(warmup=25, rep=100)` | 646 TFLOPS |

Use a purpose-built harness rather than hand-rolling. `triton.testing.do_bench` handles
sync, warmup, and repetition:

```python
ms = triton.testing.do_bench(lambda: my_kernel(...), warmup=25, rep=100)
```

FlyDSL ships an equivalent (`flydsl.autotune.do_bench`, defaults `warmup=5, rep=25`).

## Rule 3 — Report a median, and know your noise floor.

Seven identical `do_bench(warmup=25, rep=100)` calls, same process, same GPU:

```
[581.7, 638.4, 602.3, 600.2, 638.6, 602.9, 639.1] TFLOPS
spread = 9.3% of median
```

**A 9% spread on an unchanged kernel.** Any "improvement" under ~10% measured as a single
sample is indistinguishable from noise on this setup. This is why:

- Take a **median of several repeats**, not one number, and not a mean (means chase outliers).
- Use quantiles when you want the distribution: `do_bench(..., quantiles=[0.5, 0.2, 0.8])`
  returned median 650.5 / p20 662.3 / p80 638.6 TFLOPS.
- Establish *your* noise floor on *your* box before believing any small win. Run the
  baseline against itself seven times; whatever spread you see is the threshold below
  which you cannot make claims.

Sources of run-to-run variance: clock/power management (DVFS), other tenants on the GPU,
XCD scheduling, cache state from prior iterations.

## Rule 3b — If the change needs a restart, your noise floor is the *restart* spread.

Rule 3 says establish your noise floor. **The obvious way to do that gives the wrong answer
whenever the thing you changed only takes effect at process start** — which covers every
environment variable, every server flag, every tuned-config CSV, and anything captured into a
HIP graph.

Measured on two SGLang serving benchmarks, same host, same GPUs, nothing changed:

| what was repeated | Qwen3-8B, FP8, TP=1 | Gemma-4-26B, bf16, TP=2 |
| --- | --- | --- |
| the benchmark, within one server process | 0.014% | 0.15% |
| the benchmark, restarting the server between runs | **0.36%** | **0.16%** |
| ratio | **26×** | **1.1×** |

On the first stack, repeating the benchmark tells you that you can resolve a 0.05% effect. You
cannot: a candidate that requires a restart is compared across restarts, so 0.36% is the floor and
a 0.5% "win" is barely outside it. Publishing from the within-process floor is how drift gets
shipped as an improvement.

**But note the second column: the amplification is not a property of the technique, it is a
property of the stack.** On Gemma the two floors are the same to within a rounding error, so the
restart spread is not always the larger one and a 26× rule of thumb would badly misestimate it in
both directions. What survives is the procedure, not the factor: **the restart spread is the one
your claim rests on, so measure that one directly.** You cannot predict the ratio from the model,
the precision, or the topology, and you do not need to — measuring the right floor costs the same
as measuring the wrong one.

Restart variance has its own causes — allocator and KV-pool layout, memory fragmentation, which
XCDs the process lands on, graph-capture decisions made once at startup — and none of them are
visible from inside a single process, which is exactly why repeating the benchmark cannot see
them.

What to do:

- **Interleave across restarts**, not within a process: `A A′ B B′ A″ B″`, each leg its own
  server lifetime. Rule 6b's argument applies with more force here, because a restart is a
  bigger perturbation than a callable swap.
- **Compare distributions, not means.** The claim you want is "the candidate's range is
  disjoint from the baseline's range across N restarts", which survives review. A difference of
  means with n=1 per arm does not, and cannot be rescued by repeating the benchmark inside
  each arm — that only shrinks the error bar on the wrong quantity.
- **Report both numbers.** Stating the within-process and across-restart spreads separately
  tells a reader which effects you were able to resolve at all.
- If a candidate's effect is genuinely smaller than the restart floor, that is a **result**:
  it is not shippable on this evidence, and no amount of extra sampling inside one process will
  change that. Say so rather than reaching for a tighter-looking number.

## Rule 4 — Pin idle GPUs.

On a shared box, another tenant's workload lands directly in your numbers.

```bash
rocm-smi --showuse --showmemuse       # find genuinely idle GPUs
export HIP_VISIBLE_DEVICES=4          # pin. every run.
```

Note `HIP_VISIBLE_DEVICES` composes: if a container was started with `4,5,6`, then setting
`HIP_VISIBLE_DEVICES=0` *inside* it selects the first of those (physical GPU 4), not
physical GPU 0. Indices are relative to what the container can already see.

## Rule 5 — Invalidate caches when the benchmark is not the real workload.

A microbenchmark calling the same kernel on the same buffers leaves operands hot in cache
across iterations. Real serving does not. This inflates memory-bound results and is
misleading for small/skinny shapes; it matters less for large compute-bound GEMMs.

Tools expose knobs for this — `hipblaslt-bench` has `--rotating_buffer` and `--flush`,
torch's TunableOp has `set_rotating_buffer_size`. Use them when the op is memory-bound. If you
hit OOM from rotating buffers, reduce the buffer count rather than disabling the feature.

**Do not assume the tuner in front of you has such a knob.** aiter's
`CACHE_INVALIDATE_BUFFERS` exists only in `gradlib/gradlib/GemmTuner.py:135` (default 37
buffers) — the bf16/gradlib tuner. The per-op quantized tuners, e.g.
`csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`, **do not reference it at all**;
they time through `run_perftest` on a single operand set. Their reported `us` is therefore
partly cache-served, and on a part with a 256 MB MALL a 100 MB weight read can sit entirely in
cache. Knowing the ecosystem "has a knob for this" is worse than knowing nothing if you do not
check whether *your* tuner has it.

### Re-time every tuner winner on a harness you control, before it ships.

This is not belt-and-braces. Measured on the FP8 preshuffle tuner, gfx950, five winners — its
`us` column was wrong in **both** directions:

| shape | tuner `us` | cold `us` | tuner error |
| --- | --- | --- | --- |
| gate_up M=64 | 22.11 | 23.93 | optimistic 8% |
| gate_up M=16384 | 1321.46 | 1319.08 | accurate |
| o_proj M=16384 | 236.52 | 216.99 | **pessimistic 8%** |
| **down_proj M=16384** | **591.38** | **727.69** | **optimistic 23%** |

The last row decided a shipping question. At face value it is a 19% win over the CK default;
cold it is a **dead tie** (727.69 vs 727.61) and slower than the default in an eager profile.
Taking the tuner at its word would have shipped a row that buys nothing, added a dependency,
and published a kernel table claiming a win that does not exist. The pessimistic direction
matters too — it discards real wins.

An independent cold harness is the only thing that catches this, and it is cheap next to the
tuning run that produced the candidate.

## Rule 6 — Compare like with like.

The single easiest way to produce a meaningless backend comparison is to change the
problem while changing the tool. Before comparing two numbers, confirm they share:

- **operand layout** — row-major vs column-major B is a *different problem*, not a
  different implementation. (`transA`/`transB` in hipBLASLt, the layout arg in ckProfiler,
  strides in a Triton kernel.)
- dtype **and** compute/accumulate type (bf16 in / fp32 accumulate is not bf16 throughout)
- exact M, N, K, batch, and strides
- whether an epilogue (bias, activation) is included
- whether the timing is GPU-timer or wall-clock based

Worked example of getting this wrong: at 4096³ bf16 this box reports 611 TFLOPS from
`hipblaslt-bench` (N/T), 549 from `ckProfiler` (layout 1 = `A[m,k]·B[n,k]`), and 426 from
a hand-written row-major Triton kernel. Those three numbers do **not** rank the three
backends — the layouts differ. Fix the layout across all three before drawing a conclusion.

## Rule 6b — Interleave an A/B. Do not run A to completion and then B.

Rule 6 is about comparing the same *problem*. This one is about comparing at the same
*time*, and on gfx950 it matters more than everything above it combined.

The obvious structure for "is this config faster" is: time the baseline, time the candidate,
subtract. That structure assumes the machine is the same machine for both measurements, and
on MI355 it is not. Clocks drift over a run, and whichever callable happens to execute while
the part is fast absorbs the entire difference. The error is therefore **systematic, not
random** — it tracks position in the run, so a sweep finds "wins" that are really the order
the candidates were tried in, and no amount of averaging inside each block removes it.

Measured on gfx950, same kernels, same shapes, same iteration counts, only the ordering
changed:

| comparison | timed back-to-back | timed interleaved |
| --- | --- | --- |
| three `matrix_instr_nonkdim` variants, 4096³ | 13.8–20.7% spread | 1.6–5.1% |
| six backends, 4096³ | 31–48% | 1.0–1.5% |
| one MoE config against itself, M=1 | 67.0% | 1.0% |

The fix is one loop inversion: rotate through the candidates once per round and take the
median across rounds, rather than finishing each candidate before starting the next.
`tuning_benchmark/common/bench.py::time_pair` does this, and `sweep.py` re-times the baseline
alongside every candidate instead of once at the top.

**Why not just pin the clocks instead.** Because you almost certainly cannot, and the tool
that is supposed to do it will tell you that you did. `rocm-smi --setperfdeterminism` exits 0
and prints no error inside a container even as root, while changing nothing, because sysfs is
mounted read-only — the performance level stays `auto` and the drift is unaffected.
Believing you pinned clocks is strictly worse than knowing you cannot, since it is the one
belief that would justify abandoning this rule. `clocks_and_power.md` has the reproduction,
the isolated 13-17% ramp underneath all of this, and the one-line check that actually tells
you the truth (`--showperflevel` must not read `auto`).

### What it changed across the whole corpus

The corpus was then re-run end to end on gfx950 — 22 cases, 56 shapes, same kernels, same
search, same 40-candidate budget, only the timing loop inverted:

| | back-to-back | interleaved |
| --- | --- | --- |
| REAL wins | 21 over 19 cases | **27 over 14 of 22 cases** |
| median win | 14.9% | 11.0% |
| largest win | 51.7% | 71.4% |

The headline counts understate it, because the two runs disagree in *both* directions and the
disagreements cancel in the totals. Shape by shape:

- **Three wins were manufactured by back-to-back timing.** `gemm_batched_bf16`
  `B4xM1024xN4096xK4096` scored a 32.5% REAL win; interleaved, the best of the same 40
  candidates comes in at **+0.1%, inside noise**. `gemm_mxfp4_triton` `M512xN4096xK4096` (13.1%)
  and `gemm_fused_mul_add` `M2048xN2624xK6144` (8.6%) collapse to null the same way. Each of
  those was a config that would have been deployed.
- **Seven wins were hidden by it**, including `gemm_mxfp4_gluon` `M8192xN5120xK2880` at 37.8%
  and `gemm_grouped_gmm` `G8xM3000xN1408xK2048` at 11.0%.
- **Of the 15 shapes that won under both, 12 shrank and 3 grew.** The shrinkage is often most
  of the result: `gemm_a8w8_blockscale` `M2048xN2624xK6144/g128` goes 14.9% → **1.4%**,
  `ff_a16w16_gated` `M512xN5632xK2048` 19.5% → **4.4%**, `gemm_batched_bf16`
  `B32xM128xN2048xK2048` 26.1% → **5.5%**. In the other direction `gemm_mxfp4_gluon`
  `M16xN5120xK2880` goes 11.2% → **31.8%**.

So this is not a bias you can correct for with a haircut. Back-to-back timing on this part does
not over-report or under-report; it reports a different, largely unrelated number, and the only
way to know which of your wins are real is to re-measure them interleaved.

Two other consequences worth naming:

- One shape had been *discarded as unmeasurable* at 67% baseline spread. Interleaved it is a
  6.9% win at 1.0% noise. Another read "no candidate beat the noise floor (33.2%)" and is a
  20.7% win at 0.9%.
- `hipblaslt-bench --algo_method all` is itself a back-to-back race, and it inflates: its
  4096³ winner reports 1553 TFLOPS during the race and **1241 when replayed alone by index**,
  a 20.1% drop. Racing 2085 solutions holds the clocks up. The replayed number is the one to
  deploy against. On gfx942 the same comparison is 5.5% and can be waved off as spread; at 20%
  it cannot. Interleaving the replayed winner against the library's own default shrinks an
  apparent 19–40% uplift to −2%…+14% (`../tuning-hipblaslt/` §3b).

On gfx942 none of this was visible, which is exactly why it is easy to inherit a harness that
has the bug. If you are moving a tuning harness to a new part, time one config against itself
in two separate blocks before trusting anything else it tells you: the answer should be zero.

## Rule 7 — Measure the thing you intend to improve.

An isolated kernel benchmark answers "how fast is this kernel". It does not answer "will
my application get faster". Before investing in an op, check its share of total runtime;
after tuning, re-measure end to end. A 3× win on 2% of runtime is a 1.3% win.

## Checklist

- [ ] Synchronized (or using `do_bench`)
- [ ] Warmed up
- [ ] Median of repeats, with a known noise floor
- [ ] Noise floor measured **at the granularity the change requires** — across restarts if the
      change only takes effect at process start (Rule 3b), not just across repeats
- [ ] A/B measurements **interleaved**, not run back-to-back (Rule 6b), and interleaved across
      restarts when applicable
- [ ] Claim stated as **disjoint distributions**, not a difference of two means
- [ ] Idle GPU, pinned
- [ ] Result is below hardware peak — sanity-checked
- [ ] Layout/dtype/shape identical across everything being compared
- [ ] Cache behavior appropriate to the regime, and **checked on the specific tool in use**
      rather than assumed from the ecosystem (Rule 5)
- [ ] Any tuner-reported winner **re-timed cold on your own harness** before shipping (Rule 5)
- [ ] If the decode path is graph-captured, harness matches that reality —
      `graph_captured_benchmarking.md`
