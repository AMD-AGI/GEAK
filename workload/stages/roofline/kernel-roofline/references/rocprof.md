# Mode: rocprof

Measured roofline efficiency from rocprof-compute hardware counters. Every term
here is measured on a real GPU running a real command — unlike `tl-only`, which
predicts from a shape. Record `mode: rocprof` and keep outputs under
`results/rocprof/`.

Collapse the roofline picture into **one number per kernel**:

```
attainable   = min(Peak_Compute[dtype], AI_HBM × Peak_BW)
Roofline Eff = Achieved_FLOPs / attainable
```

so kernels can be ranked: "this one is at 0.31 of its ceiling, that one is at
0.93, stop tuning the second."

## The recipe

```bash
python3 scripts/run_roofline.py \
    --workdir /path/to/kernel --cmd "python bench.py" \
    --name mykernel --compute-peak auto --peaks both \
    --output results/rocprof/
```

Two `rocprof-compute` passes over the same command: `profile --roof-only` for the
empirical ceilings (`roofline.csv`), and `profile` + `analyze -p … -b 4` for the
kernel's achieved FLOP/s, HBM bytes and AI. The efficiency is then arithmetic —
doable by hand from the `analyze -b 4` table if the script is unavailable.

`run_roofline.py` is the vendored, **patched** driver. Upstream GEAK's copy does
not carry the three fixes below (bf16 empirical peak handling, the HBM read-byte
backport, microbench freezing). Use this one.

**Which compute peak.** `attainable` only means something against the peak for the
dtype the kernel's MFMA instructions actually use. An FP8 GEMM scored against the
FP32 roof reads ~3% efficient and means nothing. Select by the kernel's **MFMA
dtype**, not the tensor dtype of its inputs, and record which peak was used next to
the number. gfx950 F6F4 has no separate microbench entry below 3.6.0; `2 ×
FP8_empirical` is a defensible stand-in, but label it an estimate.

**Report both denominators, always** (`--peaks both`). Empirical and datasheet
peaks disagree, sometimes by 4×, and neither is authoritative alone:

| | what it is | fails when |
|---|---|---|
| empirical | what the microbench got on *this* box today | microbench bug, or it never saturates the unit |
| datasheet | the spec ceiling | no real kernel reaches it; ignores your clocks and power cap |

---

## 1. Version gate: rocprof-compute ≥ 3.6.0 on gfx95x

Below 3.6.0 the microbench emits CDNA3 `v_mfma_f32_32x32x8_{f16,bf16}` on a CDNA4
part and scores it at 16384 FLOP/iter, where the part's `v_mfma_f32_32x32x16_*`
does 32768. The BF16/FP16/INT8 empirical peaks therefore read 2–4× low, and every
efficiency computed against them reads correspondingly high (196% against a bad
ceiling is the case that surfaced this). 3.6.0 (ROCm 7.13.0) fixes it: *"Fixed
roofline benchmark MFMA FP16/BF16/INT8 peaks for MI 350"*. Measured on gfx950,
GFLOP/s:

| dtype | < 3.6.0 | 3.6.0 | ratio | vs. native MFMA | fixed? |
|---|---|---|---|---|---|
| bf16 | 614 663 | 2 435 517 | 3.96× | 2 453 780 | yes, 0.7% low |
| int8 | 1 222 436 | 4 854 023 | 3.97× | 4 893 977 | yes, 0.8% low |
| fp16 | 1 227 048 | 2 162 758 | 1.76× | 2 449 443 | **partly**, 12% low |
| **fp8** | 2 384 603 | 2 436 225 | **1.02×** | **4 949 468** | **no, 2.03× low** |
| fp4 / F6F4 | *absent* | 9 612 518 | — | 9 748 296 | n/a, 1.4% low |
| fp32 | 151 210 | 153 731 | 1.02× | 154 930 | n/a, correct |
| fp64 | 75 808 | 77 415 | 1.02× | 77 622 | n/a, correct |

gfx942 is CDNA3 and was never affected.

**The gate is necessary, not sufficient.** The "vs. native MFMA" column is an
independent measurement — `v_mfma_*` issued by hand on this part, FLOP/instr =
2·M·N·K — and it says 3.6.0 fixed bf16 and int8, half-fixed fp16, and **did not
touch fp8 at all**. A dtype that *did not move* between 3.4.0 and 3.6.0 is not
thereby a control proving the change was a fix; for fp8 it only means 3.6.0 left the
bug in place. (An earlier version of this section read that column the wrong way
round. fp32/fp64 are genuine controls because they match native; fp8 does not.)

So after upgrading, **still** verify against the hardware rate table rather than
against the previous release. Two cheap tells, no microbench needed:

- **`MFMABF16Flops == MFMAF16Flops`** — right in hardware, but never what a
  microbench produces, since its fp16 path does not saturate the unit. Equality
  means somebody pinned the value instead of measuring it.
- **`MFMAF8Flops == MFMABF16Flops`** — the fp8 failure, and 3.6.0 reads
  2 436 225 vs 2 435 517. CDNA4 runs fp8 MFMA at **2× the bf16 rate**
  (`v_mfma_f32_32x32x32_fp8` vs `..._32x32x16_bf16`); a bench issuing the bf16-K
  shape for both lands on exactly one rate for both. The same doubling holds for
  int8, which is why native fp8 ≈ native int8 ≈ 2× native bf16.

**Consequence for fp8 kernels:** score them on a natively measured fp8 peak or on
the datasheet 5 PF column. Using 3.6.0's `MFMAF8Flops` halves the denominator and
therefore **doubles** every fp8 compute efficiency — the direction that manufactures
"already saturated, no headroom" verdicts. Do not score an fp8 seam on the 3.6.0
microbench value even once the version gate passes.

**If you cannot upgrade, score on the datasheet column** (gfx950: HBM 8.0 TB/s,
BF16/FP16 2.5 PF, FP8 5 PF, F6F4 10 PF). The equal-rate-sibling guard in
`run_roofline.py:sanity_check_peak` is a **backstop floor, not a cure**: it flags a
dtype whose empirical peak is not within ~10% of its sibling and refuses to rank on
it. Passing the version gate and *still* reading >100% means the ceiling is still
wrong — keep investigating, do not accept the number.

Note the asymmetry in the check: a one-sided "empirical must not exceed datasheet"
test **passes** the 614663-vs-2.5e15 bug. The empirical roof must be both below the
datasheet peak and within a sane fraction of it.

## 2. Freeze the denominator across every comparison

The microbench re-runs per invocation and **drifts ~17%** on the pre-3.6.0 HBM roof
— the same GPU on the same day read 5326 → 6181.18 → 6246.09 GB/s. Two arms
profiled separately are scored against two different ceilings, and the difference
between them is partly noise in the denominator. That invalidates the whole
comparison table, not just one row.

Measure the peaks once and reuse that `roofline.csv` for every arm:

```bash
export ROOFLINE_MIBENCH_CACHE=/path/to/frozen/roofline.csv
```

`run_roofline.py` seeds the cached CSV into `--path` before profiling, so
`rocprof_compute_soc/soc_base.py`'s `if not (path/"roofline.csv").is_file():
mibench(...)` skips the benchmark entirely. Verify by its absence from the log.
This also buys back 3–5 minutes per case. The 3.6.0 bench is much steadier (1.86%
worst case, 0.02% on BF16) but pinning is still the right default.

Generate a fresh frozen CSV with `scripts/new_mibench.py --bench-src <rocprof-compute
≥3.6.0 src/>`.

**The frozen CSV is also how you get off an un-upgradable binary.** §1's gate is about
where the ceilings come from, not about which executable reads the counters: the
counters are fine on 3.4.0, only its microbench is wrong. Seeding a `roofline.csv`
built from *measured* ceilings substitutes the denominator and leaves the measurement
alone — a 3.4.0 run scored that way is sound where the same binary left to its own
microbench is not. Two cheap checks that a CSV really carries corrected ceilings
rather than a CDNA3-encoded bench's output:

- `MFMABF16Flops != MFMAF16Flops`. Equality is the tell that bf16 was inherited from
  fp16 rather than measured (§1).
- `MFMAF8Flops` near the datasheet fp8 number, not near half of it. On gfx950,
  4,949,468 is the measured value and ~2,455,000 is the CDNA3-instruction artefact.

Record in the report which route the ceilings took. "rocprof-compute 3.4.0" alone is
not provenance; "3.4.0 binary, ceilings from `<file>`, bf16 measured not inherited" is.

## 3. HBM read bytes read 2× low on gfx950

Independent of the peak bug, and it moves the *other* term. `FETCH_SIZE` was defined
using `TCC_BUBBLE` as the 128B-read counter, but that counter is **identically 0 on
gfx950**, so 128B reads are billed at 64B and reads come out exactly half. Fixed
upstream in `ROCm/rocm-systems@aa5dfb98` (rocprof-compute ≥ 3.6.0), which switches
gfx950 to `TCC_EA0_RDREQ_128B`. The correct form is:

```
read_bytes = 128*TCC_EA0_RDREQ_128B + 64*TCC_EA0_RDREQ_64B + 32*TCC_EA0_RDREQ_32B
```

**Self-check without running a memcopy:** within *one* profile, the SoL panel
(2.1.24) and the L2 panel (17.1.3) report HBM read traffic differing by ~2×. That
discrepancy is the bug; no reference workload needed.

**The correction factor is not a constant 2.** Closed form:

```
k = (32·r32 + 64·r64 + 128·r128) / (32·r32 + 64·(r64 + r128)),   k ∈ [1, 2]
```

Streaming kernels land at exactly 2.000; scattered reads at 1.00–1.15. Multiplying
a mixed workload by 2 overshoots.

**Three independent copies of the bad formula exist**: rocprof-compute's
`analysis_configs`, its bundled `counter_defs.yaml`, and `rocprofiler-sdk`'s own.
Patching one does not patch the others. `scripts/rocprof_hbm_patch.py` backports the
fix into a 3.4.0 tree: it copies the tree to `rocprof_compute_hbmfix/`, patches all
four sites (including splitting gfx950 out of the shared `counter_defs.yaml`
`FETCH_SIZE` definition), and drops a PATH shim in `rocprof_hbmfix_bin/` to prepend.

## 4. `rocprofv3` is not `rocprof-compute`

Different tool, different output contract. `rocprofv3 --pmc FETCH_SIZE` /
`WRITE_SIZE` are in **KiB**, not bytes. The `rocprofiler-sdk` copy of the broken
`FETCH_SIZE` definition is only fixed in ROCm 10.0.0, so a container can carry a
patched `rocprof-compute` and an unpatched `rocprofv3` simultaneously.

Diagnostic rule: **a ratio that stays constant across orders of magnitude of input
size is a unit bug, not cache residency.** Real cache effects vary with the working
set. Validate by hand-summing the `TCC_EA0_RDREQ_{32B,64B,128B}` request histogram
and comparing.

## 5. At most 4 TCC counters per pass

A fifth returns `Request exceeds the capabilities of the hardware`. Split the
counter set across passes, and remember each pass is another replay of the workload.

## 6. One shape per profiled command

rocprof aggregates per device kernel over the **whole run**. A command that sweeps
shapes yields one blended block per kernel and no per-(kernel, shape) roofline at
all. To get per-shape numbers the command must launch exactly one shape — see
`scripts/case_runner_template.py`, which also carries the per-arm guard.

That gives you one roofline per (shape, arm), each one a *per-kernel* block rather
than a per-shape number. Collapsing those back into one figure per kernel/regime is
[folding.md](folding.md)'s job, and the arithmetic there is identical to the one the
other two modes use — which is the only reason a rocprof column and a TL column can
be laid side by side at all. Note the cost asymmetry while you are choosing cases:
each (shape, arm) is a full profile (0927: 18 runs, median 46 s, max 427 s), whereas a
`tl-only` leg covers every shape for free. Covering the dominant case per seam is the
normal choice; what is not optional is saying so, since the two legs then no longer
span the same case set.

Two further choices inside that runner materially change the AI rocprof reports:

- **`flush_cache` every iteration.** rocprof's `AI (HBM)` counts traffic that
  actually reached HBM, so a cache-warm loop reports *fewer* bytes and therefore a
  *higher* AI — against a TraceLens byte model that assumes a cold read. Flushing
  makes the two byte counts talk about the same thing. The flush kernel lands in its
  own roofline block under its own name, is read separately, and is **never summed
  into the seam's**.
- **Few iterations.** rocprof replays the kernel once per counter set; 30 iterations
  buys no accuracy and costs wall clock linearly.

The **arm guard** belongs in the runner too. Either direction is a wrong number that
looks exactly like a right one: a candidate sneaking into the reference leg, or a
baseline quietly standing in for the candidate and reporting "no regression". Resolve
the callable's `__code__.co_filename` — a `getattr`'d function has an empty
`__file__`, so checking `__file__` passes everything — assert it sits under the
*selected* arm's tree, and verify the guard **discriminates** by pointing it at the
other arm and confirming rejection.

The runner takes `--arm {baseline,candidate}` because the two-implementation natural
experiment of [combination.md §6](combination.md#6-the-only-experiment-that-adjudicates)
is the only construction that settles which engine is wrong, and it needs both arms
through this same path. `--arm candidate` selects which arm the guard asserts; it is
not a way to switch the guard off.

One shape per command means a **batch loop** around it, and that loop has one trap
worth stating outright: the profiled child reads stdin. A driver written the obvious
way —

```bash
while read -r task case; do run "$task" "$case"; done < cases.txt   # WRONG
```

— hands `cases.txt` to the first profiled process, which drains it. The loop then
sees EOF and prints its own completion banner after **one** case. Nothing fails, no
exit code moves, and the output directory looks like a short case list rather than a
truncated run. Redirect the child's stdin and read the list on a private descriptor:

```bash
run() { ... timeout 2400 python3 run_roofline.py ... </dev/null; }
while read -r task case <&3; do run "$task" "$case"; done 3< cases.txt
```

Then check the count, not the banner: the number of `*_roofline.txt` files must equal
the number of cases times the number of arms before any table is folded from them.

`scripts/rocprof_leg_template.sh` is that loop with both halves of the fix, the frozen
denominator, the resume-by-skip, and the closing file count already in it. Copy it
next to your `case_runner.py` and edit its CONFIG block rather than rewriting the loop.

## 7. Judge contention on `achieved`, not on AI

Counters are stable; clocks are not. The same case profiled twice gave achieved
84785 vs 183164 GFLOP/s (2.17×) while its AI was bit-identical at 252.6 FLOP/byte.
So AI is safe to compare across runs, `achieved` is not — a moved `achieved` with an
unmoved AI is a neighbour on the node, not a code change.

## 8. Eff > 100% is a disproof of the denominator

Mark the row suspect, publish **no** speedup from it, and do not clamp. A clamped
100% reads as "saturated", which is the opposite of what the row is saying. On
gfx950, every hit observed was a bf16 MFMA GEMM scored against a pre-3.6.0 empirical
`mfma_bf16` peak — see §1. Use the datasheet column for those.

## 9. VALU kernels have no datasheet peak

`DATASHEET_PEAKS["gfx950"]` carries only `mfma_*` entries. A VALU kernel (e.g. a
`_sparse_mla_reduce_kernel` on `valu_f32`) therefore has no datasheet peak at all.
Drop it from the **datasheet denominator** and report reduced ds-coverage
separately; scoring it 0 silently deflates the whole seam.

## 10. Changing the peak policy does not require touching the GPU

`run_roofline.py` preserves `<tag>_roofline_raw.txt`. The efficiency block is a pure
function of (parsed rocprof output, peak policy) — `Value` and `Peak` are two columns
of the same `analyze` table, and only the second moves. So:

- `scripts/rerender.py --src <leg> --dst <new>` re-applies a changed `--compute-peak`
  / `--hbm-peak-const` / sibling-guard policy offline.
- `scripts/repeak.py --in-dir … --out-dir … --peaks-csv <new roofline.csv>
  --scope {compute,all}` substitutes a whole new microbench, with `--scope compute`
  isolating the 3.6.0 fix from the HBM roof's own drift.

Both refuse to write in place: the comparison is the point. Re-profiling instead
would mix the denominator fix with a fresh draw of timing noise, and the two are
inseparable in the diff. This is already verified — patched-3.4.0 vs 3.6.0 gave
bit-identical AI and 0.63% on achieved.

Keep the settings identical except the single one being changed, or the delta moves
two variables and means nothing.

## 11. `--roof-only` and `-b 4` are not the same thing — do not delete only one

This comes up because "delete the `-b` line to get the fullest result" is common
advice, and it is a trap as stated.

- **`--roof-only` is on the profile side.** It decides which counter groups get
  collected and how many replays run. It decides **whether the data exists**.
- **`-b 4` is on the analyze side.** It decides which blocks get printed. It decides
  **whether the data is displayed**.

The default keeps **both** (`profile --roof-only` + `analyze -p … -b 4`). That is
the path that yields AI / achieved / Roofline Eff. in one replay, with wall clock
that stays bounded across a batch of dozens of cases.

Dropping `-b` while keeping `--roof-only` is the trap: the extra blocks (2 SoL,
7 Wavefront, 10–11 CU, 16 Vector L1, 17 L2) depend on counters that were never
collected, so they print empty or unavailable. It looks like "more complete" and
carries no information.

To actually get those blocks, use `--mode full` — profile **without** `--roof-only`
and analyze **without** `-b`. The cost is an order-of-magnitude increase in replay
passes.

**The escalation path, not the default:** run the roofline mode to classify bound
type across all cases, then open `--mode full` on the *single* kernel that matters —
blocks 16/17 if it came out memory-bound, blocks 2/7 if compute-bound. Do not collect
everything up front.

## 12. A finished profile can still exit non-zero

`profile --roof-only` ends by rendering the empirical roofline to a PDF through plotly.
That needs Kaleido, and when it is missing the `RuntimeError` escapes `post_profiling()`
and fails the **whole** run — after all three counter passes have completed and
`pmc_perf.csv` is already on disk. Observed: a seven-minute profile discarded for a
decorative plot.

Do **not** fix it with `pip install "kaleido>=1"`. Two reasons: 1.x rasterises through
headless Chrome, which ROCm images do not carry, and rocprof-compute's launcher checks its
own `requirements.txt`, which pins `kaleido==0.2.1` — after the upgrade `rocprof-compute
--version` itself refuses to run. `rocprof_hbm_patch.py` routes both `write_image` calls
through a `_safe_write_image` that warns and continues; the numbers come from `analyze`.

The reason this cost a whole run rather than a minute is the second half of the trap: the
saved `*_roofline_error.txt` ended at an innocuous `Checking for roofline.csv` line and
said nothing about why. `error = r.stdout.strip() or r.stderr.strip()` looks like a sensible
fallback and is not — rocprof-compute narrates several hundred INFO lines to stdout on every
run, so stdout is never empty, so the `or` discards stderr unconditionally, and stderr is
the only place the traceback appears. `run_roofline.py` now saves both streams, labelled.
Any wrapper that captures a subprocess for a report needs the same treatment.

## Verify

- The profiled command's own exit status, separately from the wrapper's: a non-zero
  wrapper with a complete `pmc_perf.csv` means the post-processing failed, not the run.
- `roofline.csv` exists, and its BF16 and FP16 compute peaks agree within ~10%.
- The empirical compute roof is **both** below the datasheet peak and within a sane
  fraction of it (the one-sided check alone passes the 614663 bug).
- Every reported efficiency is in `(0, 1]`; anything else is §8.
- The dtype of the peak used is recorded alongside each number.
- On gfx950, the SoL and L2 HBM read figures in the same profile agree (§3).
- The frozen `roofline.csv` was used: no microbench line in the log, and peaks are
  bit-identical across cases.

## Pitfalls

- Scoring against the empirical peak without checking the tool version — the failure
  is silent and the number looks plausible.
- Reporting >100% as a result rather than as a broken denominator.
- Comparing two arms scored against two separately-measured microbench runs.
- Taking a single gfx950 `hbm_util` at face value without the `FETCH_SIZE` check.
- Sweeping shapes inside one profiled command and expecting per-shape blocks.
- Blaming a code change for a moved `achieved` when the AI did not move.

## Sources

- `--roof-only`, `roofline.csv`, `analyze` peak columns: ROCm Compute Profiler
  profile-mode docs.
- MFMA FP16/BF16/INT8 peak fix for MI350: rocprof-compute 3.6.0 changelog (ROCm 7.13.0).
- `FETCH_SIZE` / `TCC_BUBBLE` gfx950 read-byte fix: `ROCm/rocm-systems@aa5dfb98`;
  the `rocprofiler-sdk` copy of the same definition, fixed in ROCm 10.0.0.
- Measured gfx950 microbench numbers: in-house A/B of rocprof-compute 3.4.0 vs 3.6.0
  on MI355X.
- Upstream workflow reference: `AMD-AGI/GEAK@GEAK_v4_kernel_kd:perf_knowledge/kernel_roofline/SKILL.md`
  — note it carries none of the three fixes above.
