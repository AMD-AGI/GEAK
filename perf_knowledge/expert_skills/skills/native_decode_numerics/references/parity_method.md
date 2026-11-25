# Parity method: bounds, tolerance calibration, flip test, negative controls

How to decide whether a candidate kernel agrees with the native backend, how to find the one numerics switch
that is wrong, and how to know when the remaining gap is the floor. Use it to localise a parity failure and to
write your own per-step self-checks. **It never authorises changing the oracle or its tolerance**: the unit
test of a lane task is immutable.

Origin of the figures: an internal mono-kernel generation workflow (not public), measured on gfx942 against
captured decode steps of `MiMo-V2.6-Flash` and of a sparse-latent-attention model of the generation before
`DeepSeek-V4.1`, and against real-op probes. All figures are relative (multiples of a tolerance, relative
errors, shares of elements).

## Three levels of agreement

| Level | What must agree | Demand it for |
|---|---|---|
| Byte-exact | every stored byte | anything written through a native rounding path: the KV-cache slot of the current token, fp8 codes and fp32 scales of a norm + quant, top-k expert ids away from ties, the whole cache outside the written slot |
| Inside a derived bound | max error over the compared elements | GEMM, attention and MoE outputs, where only the fp32 summation order differs from the native library |
| Tie-tolerant | either of two outcomes | a router token whose k-th and (k+1)-th scores are closer than the tie band (see "Calibrating a tolerance") |

A candidate that is *more accurate* than the native backend is not at the first two levels: an exact-softmax
decode attention, or a GEMM that multiplies the bf16 activation by the fp8 weight without the native
activation quant, misses a tolerance calibrated against native (the second by about 15x).

## Deriving a bound from the arithmetic

A bound is the sum of the terms below, written next to the number it produces. It is never fitted to a
measurement and never raised above the derivation.

| Term | When it counts | Size |
|---|---|---|
| Accumulation | an fp32 dot or sum of K terms whose order differs between the two sides | grows with K; the only term that is allowed to differ freely |
| Store rounding | each store whose two inputs differ (fp32 on one side, float64 on the other) | half a unit of the stored format when rounding to nearest; a **whole** unit when the store truncates, and biased toward zero |
| A rounding both sides perform on identical inputs | never | 0: the first quant of a step reads the captured input and repeats the backend's fp32 steps, so it adds nothing |
| Re-rounding of a computed intermediate (an fp8 activation quant fed by a value the step computed; a bf16 or fp8 softmax numerator before the PV dot) | when the upstream values differ by `delta` relative | at most `sqrt(min(1, delta / u)) * 2u`, `u` = unit roundoff of the coarse format: the chance of crossing a grid point at the finest spacing times the step at the coarsest |
| Dequant scale multiply, rsqrt, exp, hardware reciprocal | once per use | a few fp32 units each |
| TP boundary | every rank stores its partial in the output dtype **before** the reduce | one bf16 store per rank, then the reduce's own store; a bound that treats the all-reduce as "the exact sum" came out half of what native needed |

Rules that go with the table:

- The check is a maximum over elements: widen a per-element bound to the maximum over the number of compared
  elements before using it as a tolerance.
- Chained re-roundings (attention over a quantised cache, then a quantised projection) give a derived value
  that is an RMS dominated by rare cascades. It is an upper bound, not a target: on the sparse latent attention
  of the second model above the derived value was 0.04 to 0.07 (RMS over runs) and 0.17 to 0.28 (max) against
  0.005 to 0.010 measured. Say in the derivation that the check is loose there.
- One bf16 rounding of a factor in (0, 1), such as a rounded sigmoid gate or an activated gate before the
  multiply, moves the output by at most half a unit, below the output's own store term. No bound can separate
  it; take it from the source (see "Switches a passing test does not confirm").

## Calibrating a tolerance

When you set a tolerance for a self-check of a whole layer or a fused step:

`tol = min(derived, max(4 x measured, 4 x noise))`

- `measured`: max error of the reference (built from the per-step references) against the native output, over
  every captured step and every rank, in the same metric the check uses.
- `noise`: the native backend against itself (run to run, or eager against graph replay).
- `derived`: composed from the per-step bounds only. Never move it toward `measured`.

Why not the derived envelope alone: it is a worst case and usually too loose to catch a real bug. Seen on one
layer: dropping the TP all-reduce altogether gave an error of 0.50 where the derived envelope was 0.45.

Metric rules that changed verdicts:

| Rule | Reason | Evidence |
|---|---|---|
| Judge the **increment** the layer adds to the residual stream, not the whole output | the attention branch is 1% to 3% of the residual stream, so a whole-output metric dilutes a single-step bug below tolerance | under the whole-output metric, at the calibrated tolerance, two planted bugs (scaled cos / sin table, scaled norm weight) were caught only by the multi-step sequence check |
| The absolute floor is **one full** unit of the output format, not half | two sides that each round to bf16 can differ by a whole unit even when their exact values almost agree | with a half-unit floor, rounding flips on sliding-window layers read as errors of 0.19 to 0.56; with a full unit the same reference measured 0.035 to 0.086 |
| A near-tie router token is right with either routing | one bf16 unit on the router input reorders experts whose scores almost tie | 5 of 14 captured steps of a 256-expert, top-8 layer had a token whose 8th and 9th scores differed by less than 1e-3 relative |
| Tie band = 2 x the largest per-step 99.9th percentile of the score-gap change under one-unit noise on the router input | makes the band a property of the layer, not a guess | 0.0082 on full-attention MoE layers and 0.0008 on sliding-window MoE layers of `MiMo-V2.6-Flash` |
| An fp8 dynamic quant downstream amplifies one bf16 unit | one unit after the norm can move an fp8 code of the per-128 group quant at the MLP entry | a reference that differed from native by one bf16 unit after the norm measured 1.5x a layer tolerance that had been set without this effect |

`measured` covers captured decode steps only; random inputs are not in it. The factor 4 and the
negative-control margin below are what cover that gap.

## The flip test (confirming a `source only` fact)

A fact read from source is a hypothesis. To test it against the oracle's data:

1. Build a reference of the step in float64 with the native rounding points you believe in (store formats,
   quant maximum and scale operation, tile, segments, window and sink conventions).
2. Feed it the captured inputs and compare with the captured native output: error `a`.
3. Flip exactly one switch and compare again: error `b`.
4. Read the result:
   - `b` differs from `a` by percents (several times the tolerance): the switch matters here, and the setting
     with the lower error is the native one.
   - `b` differs from `a` by about one unit of the output format: this comparison cannot see the switch. Do not
     conclude anything; keep the source value and mark it "not separable at this boundary".
5. To separate an invisible switch, probe the real native op on inputs built for it: values that sit exactly on
   a rounding midpoint (for a divide against a reciprocal multiply: scales of the form `7 * 2^-n`), a
   saturating value, an all-zero group, duplicated gate rows for tie rules, a context shorter than one
   segment and one much longer. Compare bytes, not a tolerance.

Reading aids:

- One layer kind or one shape measuring about 10x the others is a **grid mismatch** before it is a precision
  problem: wrong tile origin, wrong segment count, window tiled from its own index 0. Do not widen a bound for it.
- A fix that improves the error by one unit is not a fix.
- These roundings matter most where nothing averages them out: a short context, a freshly filled cache. A
  check that only uses long captured steps does not cover them.

## Switches a passing test does not confirm

Flipping these stays **inside** the tolerance on every input tried, so a kernel with the wrong value passes
its unit test and drifts later (fp8 codes flip downstream, sequences diverge). Set them from the reference
tables, never from a green test.

| Switch | Flip measured at |
|---|---|
| bf16 numerator format, tile length and split rule of MLA decode under a bf16 numerator | 0.1 to 0.44 of the tolerance |
| Segment split and per-segment bf16 partial of block-list sparse decode | 0.2 to 0.46 |
| Divide against reciprocal multiply in the KV-cache write scale (while nothing saturates) | 0.39, on inputs built to hit midpoints; invisible otherwise |
| Rounded against unrounded sigmoid gate on an attention output | 0.02 to 0.2 |
| Bias dtype of a linear | 0.2 to 0.7 |
| Activated gate rounded to bf16 before the multiply (C++ op) or not (compiled graph) | 0.08 to 0.44, although 27% of the outputs differ by one unit |
| Quotient formed by divide or by reciprocal multiply in the activation quant | 0.3 to 0.4 |
| RMSNorm: residual sum rounded first, or normalised value rounded before the weight (bf16 output) | about 0.6 and 0.2 |
| Where a q / k pre-norm rounds before RoPE | inside the tolerance, yet hundreds of output bytes move |
| fp32 cos / sin table read unrounded instead of through its bf16 cast | inside the tolerance |
| Stream-mixing: variance from the unrounded or the stored collapse; one extra bf16 rounding of the normed row | at most 0.22 |

With an **fp8 output** the picture reverses: the fp32 scale port has a tight tolerance and sees every one of
the norm's rounding switches (10x and far more).

## Negative controls

A check is worth what it rejects. Keep, next to every self-check, mutants of the same kernel with one switch
flipped, and require the check to fail on each.

- **Margin.** Every control must still be caught by at least one check at 3x the tolerance. A control the
  checks cannot see is a failure of the checks, not a pass.
- **Strengthen, never swap.** If a control slips through, enlarge the same bug (a bigger scale factor, more
  corrupted elements). Do not replace it with another bug, delete it, or point it at another check. If no
  meaningful larger size exists, report the numbers.
- **Single step against sequence.** A KV-write bug touches one token of the context per step; often only a
  multi-step sequence check, where the error accumulates through the cache, catches it. Name the controls that
  only the sequence check catches.
- **Compare the whole cache.** Reset the cache to random contents before the call and compare all of it after:
  a kernel that writes nothing, writes `slot + 1` or touches a neighbour then fails by an error of order 1.
- **Poison what must not be read.** Fill block-table entries past a row's own blocks with an invalid value so
  that an unclamped address faults instead of reading a plausible page.

Typical sizes of a real mistake, for judging whether a check is sharp enough:

| Mistake | Error |
|---|---|
| Sink dropped, or read with the wrong dtype | 2.2x to 38x tolerance; max relative error 0.92 on windowed layers |
| Current token attended from stale cache bytes instead of its rounded registers | 2x to 170x (paged), 28x to 309x (MLA) |
| KV scale not applied on a cache with a non-unit scale | 11x to 57x (paged), 50x to 70x (MLA) |
| Quant maximum 224 against 240 | 7x to 33x |
| No activation quant before an fp8 block-scale GEMM | about 15x; twice the error on a captured MLP |
| Gate / up halves read as concat instead of interleaved | 100x and more |
| Window tiled from its own first position | about twice the error; one layer kind about 10x the others |
| Block-list selection off by one, or a padding entry read as block 0 | 60x to 205x |
| Packed q / k / v split with the wrong offsets | 100x to 400x |

## The floor

What is left when every row of the reference tables is matched:

- MFMA and the native library accumulate in different orders: about one unit, now and then.
- Greedy outputs then fork only at near-ties, and native forks with itself at the same places between batch
  sizes and between eager and graph mode. Judge the candidate's agreement with native against native's
  agreement with itself.
- For a norm + quant step, count fp8 code flips of the output against the native op on captured inputs: the
  count should be 0.
- The end-to-end criterion is an accuracy benchmark, not bit identity.
- An oracle captured in eager mode holds the all-reduce and the following add + RMSNorm as separate ops; a
  compiled aiter deployment at TP > 1 fuses them with other rounding (norm on the fp32 sum, one truncating
  store). Match the oracle you were given, and say which of the two it is.
