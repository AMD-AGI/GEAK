---
name: tuning-hipblaslt
description: Tune GEMMs on the hipBLASLt path — capture the exact problem from a live workload, race all solutions with hipblaslt-bench, replay the winner by solution index, and avoid the bracket-number and hot-cache traps. Use when the op dispatches through hipBLASLt (torch matmul, aiter libtype=hipblaslt).
---

# Tuning hipBLASLt

Read `../tuning-core/SKILL.md` first. Build the client per
`../env-setup/hipblaslt_bench_build.md` — no package ships it.

hipBLASLt is the default GEMM backend under torch on ROCm, so it is usually what you are
implicitly benchmarking *against*. Unlike Triton, you do not author a config space: hipBLASLt
ships thousands of pre-compiled solutions and tuning means **selecting** among them. The
whole job is (1) state the problem exactly as the workload states it, (2) race, (3) carry the
winner's identity forward correctly.

**Scope.** Selection is all of this skill. The layer beneath it — `tensilelite/` in the
hipBLASLt source, which generates and assembles the solutions in the first place — is not
covered. That ordering is deliberate: selection is a day's work with immediate payoff, adding
a Tensile kernel is a different kind of project. But on MI355 it is worth knowing the deeper
layer exists, because §7 shows the shipped selection logic is now mostly *analytical*, and §3b
shows how little racing buys on top of it.

Measured on gfx942 / MI300X **and** gfx950 / MI355X, ROCm 7.2.2, bf16. Numbers are labelled
per part; where only one is given it is gfx942. Build with `-a gfx950` for MI355
(`../env-setup/hipblaslt_bench_build.md`) — ~10 min, and the resulting binary races 2085
solutions rather than gfx942's 1231.

## 1. Capture the problem, do not reconstruct it

A hipBLASLt "problem" is more than M/N/K: it includes transposes, leading dimensions,
strides, dtypes for A/B/C/D, compute type, bias, and epilogue. Tune the wrong one and you
have tuned nothing.

Get it from the workload rather than writing it by hand:

```bash
export HIPBLASLT_LOG_MASK=32          # log every matmul as a runnable bench command
python3 your_workload.py 2> hipblaslt.log
grep -o '\-\-api_method.*' hipblaslt.log | sort | uniq -c | sort -rn | head
```

Each line is a complete `hipblaslt-bench` invocation. **Replay it verbatim.**

The reason for the emphasis: a torch matmul of `(512,1024) × (1024,2048)` dumps
`-m 2048 -n 512 -k 1024`. M and N are swapped relative to the torch call, because the library
sees the column-major transposed form. Hand-reconstructing "M=512, N=2048" tunes a different
problem that happens to have the same FLOP count. Copy the captured command.

The `sort | uniq -c` matters as much as the capture: it ranks problems by call count, which is
your tuning priority order. Tuning the shape that appears 40,000 times beats tuning the one
that appears twice, whatever their individual timings.

## 2. Race all solutions

```bash
hipblaslt-bench -m 4096 -n 4096 -k 4096 -r bf16_r --transA T --transB N \
    --algo_method all --print_kernel_info -i 20 -j 5
```

| flag | why |
| --- | --- |
| `--algo_method all` | try every supported solution. `heuristic` (default) returns only the library's guess — that *is* your baseline, not your search |
| `--print_kernel_info` | print `--Solution index:` — **required**, see §3 |
| `-i` / `-j` | timing iterations / cold (warmup) iterations. Defaults 10/2 are too few to be stable |
| `--rotating <MB>` | defeat cache reuse — see §4 |
| `--flush` | flush icache between iterations |
| `--initialization` | `hpl`, `norm_dist`, `trig_float`, … — see `../tuning-core/correctness_gates.md` on why input distribution matters |

Output ends with a `Winner:` block: a CSV row (`Gflops`, `GB/s`, `us`), a `--Solution index:`,
and the solution/kernel names. On the 4096³ bf16 case above the winner was index **205610**,
`MT256x224x64`, 633505 Gflops (633 TFLOPS).

The solution name is readable and worth reading. `MT256x224x64` is the macro-tile;
`MI16x16x1` the MFMA instruction shape; `ISA942` the target arch; `GSU`/`SK` fields describe
split-K. Comparing the winners across regimes tells you *why* one won — the 4096³ winner takes
a 256×224 tile, while decode-shaped problems win with tiles like `MT16x16x512`.

The `ISA` field is also the reason a solution index never crosses parts. The same command on
gfx950 / MI355X:

| | gfx942 | gfx950 |
| --- | --- | --- |
| solutions raced | 1231 | **2085** |
| winning index | 205610 | 441281 |
| winning macro-tile | `MT256x224x64` | `MT256x256x64` |
| arch field in the name | `ISA942` | `ISA950` |
| race throughput | 633505 Gflops | 1552980 Gflops |

Two changes worth separating. The tile got deeper (224 → 256), which is a statement about the
part. The index changed entirely, which is not — these are separately compiled kernel
libraries, and 205610 on gfx950 either names something unrelated or nothing at all. Carry the
tile across parts as a hint; re-derive the index.

## 3. The bracket number is not the solution index

The most expensive small trap here.

Every result line is prefixed with a bracket number — `[138]`, `[1357]`. That is the
**enumeration position in this run's output**, not an identifier. Feeding it back fails:

```bash
--algo_method index --solution_index 138
# error: NO solution found! ... testing_matmul.hpp:2815
```

The real identifier comes only from `--print_kernel_info`:

```
--Solution index: 205610
```

which replays correctly:

```bash
hipblaslt-bench -m 4096 -n 4096 -k 4096 -r bf16_r --transA T --transB N \
    --algo_method index --solution_index 205610 -i 50 -j 10
# Is supported 1 / Total solutions: 1   -> 598810 Gflops
```

Worse than failing loudly: the bracket number **drifts between identical runs** (the same
winner appeared as `[99]` and `[66]` across two runs of one command). So a bracket number
recorded from one run can silently name a different solution later. If you did not run with
`--print_kernel_info`, you do not have a result you can deploy — rerun.

Confirmed on gfx950: the `Winner:` block is prefixed `[1910]` while its `--Solution index:` is
**441281**. The two are not related, and note that `--Solution index:` is printed *after* the
CSV row it belongs to, so a parser that reads forward from the bracket line pairs every row
with the previous row's index. (One place this bites: `../env-setup/hipblaslt_bench_build.md`
used to state the bracket *is* the solution index. It is not; that has been corrected.)

### Replay the index standalone. On gfx950 this is not a formality.

Always close the loop: replay the winner by index and confirm it reproduces the winning time.

| | race (`--algo_method all`) | replay (`--algo_method index`) | drop |
| --- | --- | --- | --- |
| gfx942 | 633505 Gflops | 598810 | 5.5% |
| gfx950 | 1552980 Gflops | 1241320 | **20.1%** |

On gfx942, 5.5% is run-to-run spread and the race number is usable. On gfx950 a fifth of the
reported throughput is not spread — it is the race inflating itself. Racing 2085 solutions
back-to-back keeps the clocks boosted and the caches warm for the whole run; a single solution
replayed alone starts from a colder state, which is also the state your workload starts from.

**Deploy against the replayed number.** If you size a decision on the raced figure on MI355 you
will over-predict by ~20%, and if you compare a raced hipBLASLt number against a
non-raced number from another backend you will conclude hipBLASLt wins by more than it does.
This is the same effect as `../tuning-core/measurement.md` Rule 6b, arriving through a vendor
tool rather than your own loop.

### 3b. What the race is actually worth on MI355

The number you care about is not the race figure. It is *how much the raced winner beats the
solution the library would have chosen on its own* — `--algo_method heuristic`, the default,
which is what your workload gets if you tune nothing. Both sides have to be replayed, and on
gfx950 they have to be interleaved. Four bf16 shapes, `-i 50 -j 10`, five alternating rounds
each, medians:

| shape (m·n·k) | default | raced winner | what the race *claimed* | honest gain |
| --- | --- | --- | --- | --- |
| 4096·4096·4096 | 1174 | 1238 | +33% | **+5.4%** |
| 8192·1024·8192 | 956 | 1057 | +40% | **+10.6%** |
| 2048·8192·2048 | 1212 | 1185 | +19% | **−2.3%** |
| 1024·4096·4096 | 746 | 848 | +35% | **+13.7%** |

(TFLOPS. "claimed" = the race's own Gflops against a separately-run default, i.e. what you
would have written down had you not replayed.)

Two things to take from this. The apparent 19–40% uplift is almost entirely measurement
artefact; the real range is −2% to +14%. And on one of four shapes **the raced winner is not
better than the default** — the race dutifully reported a winner, because a race always has a
winner, and adopting it would have cost 2%. `--algo_method all` does not tell you whether
tuning was worth doing. Only the interleaved A/B against the heuristic does.

That the shipped default holds up this well on MI355 is not luck, and §7 explains where it
comes from.

## 4. Rotating buffers, or you will tune the cache

A benchmark loop reuses the same A and B buffers every iteration. After the first, they are
resident in cache — and the timing measures a memory hierarchy your real workload never sees.

Same solution, same shape, `1024³` bf16 (small enough to fit in cache):

| part | `--rotating` | Gflops | GB/s | µs | inflation |
| --- | --- | --- | --- | --- | --- |
| gfx942 | 0 (default) | 57899 | 158.0 | 37.09 | **43%** |
| gfx942 | 512 MB | 40610 | 110.8 | 52.88 | — |
| gfx950 | 0 (default) | 65036 | 177.4 | 33.02 | **6.7%** |
| gfx950 | 512 MB | 60973 | 166.4 | 35.22 | — |

Inflation from cache reuse alone. Not measurement noise — a systematically wrong answer that
flatters every small shape and can reorder which solution appears to win.

The magnitude is very different between parts, and the direction of the surprise is worth
noting: on gfx942 the cache is the dominant measurement error at this shape and on gfx950 it
is nearly negligible, while the *clock* error (§3, 20%) is far larger there. So keep using
`--rotating` on both — it costs nothing and the effect is real on both — but do not assume the
error budget you learned on one part describes the other. Re-measure which artefact dominates
before deciding what to control for.

Use `--rotating` with a working set larger than last-level cache for any shape whose operands
fit in cache. Large shapes are less affected but the flag costs nothing. This is the concrete
case of the cache-invalidation rule in `../tuning-core/measurement.md`.

## 5. Deploying the winner

An index in your notebook changes nothing. Three ways to make it real:

- **Through aiter** — run the aiter gradlib tuner, which races hipBLASLt against the other
  backends and, when hipBLASLt wins, records `libtype=hipblaslt` plus the solution index in a
  tuned CSV that the runtime reads. This is the path that matters for serving. There is no flag
  to force the backend — `--libtype` is an output column, not an input, and documenting it as an
  option here was wrong. See `../tuning-aiter/` §4, including where the CSV must be written.
- **Through torch TunableOp** — `torch.cuda.tunable` tunes and persists selections for torch's
  own matmuls, writing a CSV whose metadata includes `GCN_ARCH_NAME`.
- **Directly in application code** — via the hipBLASLt extension API, when you own the call
  site.

In all three, finish with `../tuning-core/engagement_verification.md`. A solution index that
is fast in `hipblaslt-bench` proves the solution exists and is fast; it says nothing about
whether your workload selects it.

## 6. Sanity-check across tools

Independent agreement is the cheapest validation available. On 4096³ bf16 the aiter tuner
independently selected an `MT256x224x64` hipBLASLt solution — the same macro-tile family this
bench chose. Two tools with separate search procedures landing on the same tile shape is
strong evidence the result is real rather than an artifact of one harness.

When tools *disagree* by more than the noise floor, do not average them. One is measuring
something you did not intend — check layout and transpose first
(`../tuning-core/measurement.md`, rule 6).

## 6b. MX (MXFP4 / MXFP8): everything in §2–§3 stops working

The microscaled formats are the gfx950-only surface, so this is the first place
people take a new MI355. The whole workflow this skill is built on — race all
solutions, read the index, replay it — is **unavailable for them**, and it fails
in three different ways that have to be learned separately.

Running an MX GEMM at all needs the block-scale mode, not just an MX element
type. `--scaleA 3` / `--scaleB 3` mean "block", which the client maps to
`HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0` when the block sizes are the MX
default (A 32×1, B 1×32). Block scaling also requires `f32_r` compute:

```bash
hipblaslt-bench -m 4096 -n 4096 -k 8192 --transA T --transB N \
    --a_type f8_r --b_type f8_r --c_type f16_r --d_type f16_r \
    --compute_type f32_r --scaleA 3 --scaleB 3 -i 20 -j 5
```

`f4_r` is MXFP4, `f8_r` is MXFP8 E4M3, `bf8_r` is MXFP8 E5M2. Omit `--scaleA/B`
and you are not benchmarking MX.

**Trap 1 — the race is refused.** `--algo_method all` does not fall back or
degrade; it prints

```
MX data types do not support algorithm "all"
```

and exits. §2 does not apply to any MX shape.

**Trap 2 — enumeration exists, under a different flag.** The substitute is
`--algo_method heuristic --requested_solution -1`, which returns every solution
the heuristic will admit. It is worth doing once, because the spread is large:

| format | solutions | best | worst | spread | rank of the heuristic's own first pick |
| --- | --- | --- | --- | --- | --- |
| MXFP8 `f8_r` | 12 | 1305 TFLOPS | 263 | **5.0×** | **1st** |
| MXFP4 `f4_r` | 27 | 2567 | 242 | **10.6×** | **1st** |

**Trap 3 — the winner has no address.** `--print_kernel_info` emits **zero**
`--Solution index:` lines for MX. The accepted solutions print only a bracket
number, and §3 already establishes that the bracket is not an identity. So there
is nothing to pass to `--algo_method index`, and the deploy path in §5 has no
input.

The reason all three behave this way is visible in the rejected candidates,
which *do* print names:

```
[rr/error] Predicate mismatch for RR_GEMM_TN_FP8_FP8_Half_Half_Float_SA_B_SB_B_WGT_192x64x128_UR_2_WGM_:
           M must be a multiple of workgroupTile.m=192
```

`RR_` is **RocRoller**, not Tensile. MX GEMMs route through a different kernel
provider whose solutions are not in the Tensile index space at all — hence no
index, hence no `all`. (`WGT_192x64x128` is the workgroup tile and `UR_2` the
unroll, so the *rejected* ones are legible even though the accepted ones are
not.)

**What to actually do.** The last column of the table is the useful one: for
both formats the heuristic's first pick was the fastest of everything on offer,
despite 5–10.6× between best and worst. Combined with the fact that you could
not deploy an alternative if you found one, **hipBLASLt MX tuning is a no-op on
MI355 today** — run the enumeration once to confirm the ranking on your shape,
then spend the time on CK (`../tuning-ck/` §2c, the only route to MXFP8 here) or
on the aiter Triton FP4 kernels, where the config space is real and the corpus
finds 32–72% wins. This is the same conclusion §7 reaches for bf16, arrived at
from a different direction.

§3b showed the shipped heuristic landing within a few percent of the best solution on three of
four shapes and winning outright on the fourth. That is not how gfx942 behaves, and the reason
is visible in the source tree rather than in any benchmark.

hipBLASLt's solution choice is driven by "logic" YAML under
`library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/`, and the two parts organise it
on different axes entirely:

| | gfx942 (`aquavanjaram/`) | gfx950 (`gfx950/`) |
| --- | --- | --- |
| logic files | 816 | 566 |
| partitioned by | **CU count** — `gfx942`, `_20cu`, `_38cu`, `_64cu`, `_80cu`, `_152cu`, `_228cu` | **selection strategy** — `Equality`, `GridBased`, `Origami`, `Range` |
| strategies present | Equality, Experimental, FreeSize, GridBased, StreamK | Equality (25), GridBased (67), **Origami (471)**, Range (3) |

**Origami is 83% of the gfx950 logic tree and does not appear in the gfx942 tree at all.** It is
a separate project in the same monorepo (`shared/origami/`) described by its own README as
"Analytical GEMM Solution Selection": it models compute and memory latency across candidate
tile sizes, estimates occupancy and L2 hit rate, and picks a tile analytically instead of
reading off an exhaustively benchmarked table.

What that changes for you:

- **The MI355 default is a model's prediction, not a lookup of a measured winner.** It
  generalises to shapes nobody benchmarked, which is why it stays within ~14% across the four
  shapes above rather than falling off a cliff between table entries.
- **The gfx942 intuition "the shipped table is sparse, so racing usually wins big" does not
  transfer.** Budget tuning time on MI355 against the measured −2%…+14%, not against the
  race's advertised 19–40%.
- **The per-CU-count split is gone.** On gfx942, a harvested part with a different CU count
  reads a different logic directory; on gfx950 there is one tree and the CU count enters
  through the analytical model instead. Do not go looking for a `gfx950_256cu` directory — the
  absence is by design, not an omission.

This is a structural read of the source, not a measurement, and it is offered as the
explanation for §3b rather than as an independent result. The load-bearing claim is §3b's
table, which is measured.

## Checklist

- [ ] problem captured with `HIPBLASLT_LOG_MASK=32` and replayed verbatim, not retyped
- [ ] shapes prioritized by call count
- [ ] `--algo_method all`, not the default heuristic
- [ ] `--print_kernel_info` on; bracket numbers never recorded as identities
- [ ] `--rotating` set for cache-resident shapes
- [ ] winner replayed by index; time reproduces within the spread
- [ ] winner A/B'd against `--algo_method heuristic`, both replayed and interleaved — a race
      always produces a winner, and on MI355 one shape in four does not beat the default
- [ ] deployed through a path the runtime reads, and engagement proven
