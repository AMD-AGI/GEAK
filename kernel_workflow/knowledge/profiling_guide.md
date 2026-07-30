# Profiling Analysis Guide

## Reading the raw profiler dump (START HERE — the script does NOT parse for you)

`scripts/profile_kernel.sh` is intentionally thin: it warms up, picks the best available profiler, runs
it, and dumps the **raw, unparsed** output. YOU (the profile engineer) extract the metrics and classify
the bottleneck. The script never greps for version-specific section names, so it stays portable — which
means the parsing responsibility is yours, and you must adapt to whichever profiler actually ran.

1. **Entry point**: read `<profile_output_dir>/profile_report.txt`. Its tail prints `Profiler used: <name>`
   and an `Artifacts:` list. Branch your parsing on which profiler produced it (the four cases below).
   Native artifacts (e.g. rocprofv3 CSVs, the `*_profile_raw.log`) are also left in the dir for deeper
   parsing if `profile_report.txt` is not enough.
2. **Always extract the dispatch count** (kernels launched per call) regardless of profiler — it is the
   key geomean/overhead signal (see `geomean_levers.md`). How to find it differs per profiler (below).
3. **Degrade gracefully**: if a metric/field is absent in the available profiler, say so explicitly in
   your summary and classify from whatever IS present (at minimum: per-case latency + dispatch count).
   Never block on a field that this toolchain doesn't emit.

## Profiler failed? — fault-tolerance ladder

`profile_kernel.sh` no longer degrades **silently**. If a profiler errors out (almost always because a
flag was renamed or removed across ROCm/profiler versions), the report contains a block like:

```
!!! PROFILER FAILED: rocprofv3 exited 2 — its output may be unusable; degrading.
>>> Self-heal: run `rocprofv3 --help` to find the current flag, then re-run this script with
>>>   an override, e.g.   RPV3_TRACE_ARGS="<corrected args>" bash profile_kernel.sh <gpu> <cmd> <out>
>>> Recipe: knowledge/profiling_guide.md → "Profiler failed? — fault-tolerance ladder" → rocprofv3
>>> Last error lines from rocprofv3_run.log: ...
```

When you see that block, **do not just accept the degraded result** — work this ladder:

1. **Read the error** (the `Last error lines` in the block, or the named raw log in the output dir).
2. **Discover the correct flag**: run `<tool> --help` (or `<tool> <subcmd> --help`). Map the rejected
   option to its current equivalent (see the per-tool notes below).
3. **Re-run once with an env override** — the script takes every profiler's args from an env var, so you
   never edit the script: prefix the same `profile_kernel.sh` invocation with the corrected var.
4. **Still failing → degrade deliberately**, one rung down the priority list, and **say so**: set
   `profiler_used` to what actually ran and add one line to your `profiling_summary.md` naming the failed
   tool and why (e.g. "rocprofv3 rejected `--output-format`; fell back to rocprof --stats"). Never let a
   degrade pass unrecorded.

Priority / degrade order: `rocprof-compute → omniperf → rocprofv3 → rocprof → benchmark-only`.
Override env vars (defaults in `profile_kernel.sh`): `PROFILER_PRIORITY`, `WARMUP_RUNS`,
`RPC_PROFILE_ARGS` (rocprof-compute/omniperf `profile`), `RPV3_TRACE_ARGS` (rocprofv3), `RPROF_ARGS`
(legacy rocprof).

### Per-tool "if it fails"

- **rocprof-compute / omniperf** — override `RPC_PROFILE_ARGS`.
  - `--no-roof` rejected → newer builds may drop it (roofline already off by default); retry with
    `RPC_PROFILE_ARGS=""`.
  - `profile`/`analyze` subcommand missing → check `rocprof-compute --help`; on some installs the entry
    point is `omniperf` (or vice-versa) — set `PROFILER_PRIORITY="omniperf rocprofv3 rocprof"`.
  - workload dir empty / analyze finds nothing → counters likely need permissions (see rocprofv3 note);
    drop to rocprofv3.
- **rocprofv3** — override `RPV3_TRACE_ARGS`.
  - `unrecognized argument --output-format` → older/newer builds spell it differently; check
    `rocprofv3 --help | grep -i output` (e.g. `-f csv`, or CSV is the default and the flag is dropped).
  - counter collection needs elevated perf access → fall back to trace only:
    `RPV3_TRACE_ARGS="--kernel-trace"` (you lose SoL/cache, keep durations + dispatch counts).
  - still nothing → degrade to `rocprof`.
- **rocprof (legacy)** — override `RPROF_ARGS`.
  - `--stats` rejected or empty → try `RPROF_ARGS="--hip-trace --stats"`; if the binary is absent
    entirely, you are at the bottom rung → **benchmark-only**.
- **benchmark-only** (no profiler usable) — not a failure to fix, it is the floor. Classify from the
  per-case latency table + dispatch shape per the `benchmark-only` bullet above, and state plainly in
  your summary that no profiler was available on this box.

### Per-profiler extraction

- **`rocprof-compute` / `omniperf`** (richest): `profile_report.txt` holds the full `analyze` text —
  Speed-of-Light, Wavefront, Compute Pipeline, cache hierarchy. Parse it with the section tables further
  down this guide. Dispatch count = number of distinct kernel rows in the kernel/dispatch breakdown.
- **`rocprofv3`** (modern, trace-based): the report embeds the run log + every CSV/JSON artifact. Use the
  **kernel-stats CSV** (a `*kernel*stats*`-style file, but do NOT rely on the exact name — scan the
  artifact list): each row is a kernel with a call/dispatch **count** and total/avg **duration**.
  Dispatch count = sum of per-kernel counts per call (or distinct kernels × calls); top kernels =
  highest total-duration rows. No SoL/cache fields here — classify from durations + dispatch shape +
  the per-case latency table.
- **`rocprof`** (legacy `--stats`): a stats CSV/table of kernels with counts + durations. Same approach
  as rocprofv3 (counts → dispatch, durations → top kernels); no SoL/cache fields.
- **`benchmark-only`** (no profiler on the box): only the benchmark stdout. Classify from the per-case
  latency table + `geomean_levers.md` heuristics: cases of very different sizes at near-equal latency ⇒
  **overhead-bound** (floor); a large-N case far above the floor ⇒ likely **compute-bound**. State that
  no profiler was available.

## rocprof-compute (formerly omniperf) Output Interpretation

### Section 2: System Speed-of-Light (SoL)

The most important section. Shows overall utilization as percentage of peak.

| Metric | What it means | Threshold |
|--------|--------------|-----------|
| VALU Utilization | Vector ALU usage | > 60% = compute-bound |
| MFMA Utilization | Matrix unit usage | > 40% = MFMA-active workload |
| VMEM Utilization | Vector memory pipe | > 60% = memory-bound |
| LDS Utilization | Local data share | > 50% = LDS-heavy |
| Bandwidth (GB/s) | Effective HBM BW | Compare to this card's HBM peak (≈5300 GB/s MI300X/300A, ~6000 MI325X, ~8000 MI350/355 — see `amd_instinct.md`) |

**Classification from SoL:**
- VALU > 60% AND VMEM < 40% → **compute-bound**
- VMEM > 60% AND VALU < 40% → **memory-bound**
- Both < 40% → **latency-bound**
- LDS > 50% → **lds-bound** (check bank conflicts)
- Both 40-60% → **balanced**

### Section 7.2: Wavefront Runtime Stats

Shows how wavefronts spend their time.

| Metric | What it means |
|--------|--------------|
| Active Cycles | Cycles actually computing |
| Dependency Wait | Stalled waiting for data |
| Issue Wait | Stalled on instruction issue |
| Total Wave Cycles | Total cycles alive |

**Key ratios** (each row is an independent accumulator over the SAME `Total Wave Cycles` — divide each
by Total, **never subtract them from one another**; subtracting produces negative "active" values,
which is a mis-read, not a finding):
- `Active / Total` = Kernel efficiency (< 20% = CRITICAL inefficiency)
- `Dependency Wait / Total` = fraction stalled on a data dependency feeding the math units
- `Issue Wait / Total` = fraction stalled because too few waves are resident to issue from

These are **averages over every wave in the dispatch** (and every dispatch, if you aggregated). A
kernel whose waves diverge — an MoE expert with uneven token counts, attention with ragged sequence
lengths — can average into a near-even split that describes none of its waves. If the split comes out
ambiguous, re-profile a single representative launch before trusting it.

**Diagnosis — and do NOT collapse dependency-wait into "memory-bound".** A high dependency wait means
the math units are waiting on a *serial data-dependency chain*; the fix is to shorten that chain, which
is not the same as a bandwidth problem. See "Splitting latency-bound" below — the two latency sub-cases
have **opposite** fixes, so resolve the sub-case before touching code.
- High Dependency Wait → serial dependency chain (latency-bound **C1**) — shorten the chain
- High Issue Wait → not enough resident waves (latency-bound **C2**) — raise occupancy / fill the GPU
- Low Active + Low Wait → occupancy too low, or the launch is dominated by dispatch overhead

### Splitting latency-bound before choosing a fix (the two remedies pull opposite directions)

"Latency-bound" on its own does not select a fix. It covers two situations whose remedies conflict:
C1 wants **shorter dependency chains** (often more registers per wave); C2 wants **more resident
waves** (fewer registers per wave). Guess wrong and the kernel gets slower.

**The split is a property of the configuration, not of the source.** Tile size in particular moves a
kernel between the two, in opposite directions:
- **Small tiles** → little math per wave, so a serial preamble (dequant, scale, address math) dominates
  the wave lifetime → reads as **dependency wait (C1)**.
- **Large tiles** → bigger accumulator → more registers → fewer waves fit → too few to cover latency →
  reads as **issue wait (C2)**.

We measured one kernel cross this line under nothing but a tile-size change: 72% dependency wait at the
small tile, 42% issue wait at the large one — same source, opposite fix. Two consequences: **re-read
the split after every change to tile size / `num_stages` / `num_warps`** instead of carrying the prior
diagnosis forward; and recognize that an autotuner sweeping tiles is implicitly sweeping both branches
— the winning config usually *balances* the two stalls rather than minimizing either alone.

### Section 11: Compute Pipeline

| Metric | What it means |
|--------|--------------|
| VALU Active Threads | Average active threads per VALU instruction |
| VALU Utilization % | How much of peak VALU is used |
| Branch Divergence | Fraction of divergent branches |

**Key checks:**
- Active Threads < 64 → wavefront divergence (threads disabled by branches)
- Branch Divergence > 10% → significant divergence penalty
- VALU Util close to SoL → compute is the bottleneck

### Sections 13-16: Cache Hierarchy

#### Section 13: L1 Cache (vL1D)
| Metric | What it means | Threshold |
|--------|--------------|-----------|
| Hit Rate | L1 cache hit % | < 60% = likely memory-bound |
| Bandwidth | L1 effective BW | Compare to peak |
| Coalescing | Memory coalescing efficiency | < 50% = fix access patterns |

#### Section 14: L2 Cache
| Metric | What it means | Threshold |
|--------|--------------|-----------|
| Hit Rate | L2 cache hit % | < 50% = heavy HBM traffic |
| Read/Write BW | L2 bandwidth used | |

#### Section 16: HBM
| Metric | What it means |
|--------|--------------|
| Read BW | HBM read bandwidth achieved |
| Write BW | HBM write bandwidth achieved |
| Total BW | Should be < this card's HBM peak (≈5300 GB/s MI300X; higher on MI325X/MI350/MI355 — `amd_instinct.md`) |

## Bottleneck Classification Decision Tree

```
1. Check SoL VALU vs VMEM utilization
   ├─ VALU > 60%, VMEM < 40% → COMPUTE-BOUND
   ├─ VMEM > 60%, VALU < 40% → MEMORY-BOUND
   ├─ Both > 50% → BALANCED
   ├─ Both < 40% → go to step 2
   └─ LDS > 50% → LDS-BOUND

2. Check Wavefront stats (Active / Total ratio)
   ├─ < 20% → LATENCY-BOUND (critical inefficiency) → split C1/C2 below
   ├─ 20-50% → check Dependency vs Issue wait (both are latency sub-cases, NOT memory-bound)
   │   ├─ Dependency dominant → LATENCY-BOUND C1 (serial dep chain) → shorten the chain
   │   └─ Issue dominant      → LATENCY-BOUND C2 (too few waves) → raise occupancy / GPU fill
   └─ > 50% → check cache hit rates
       ├─ L1 < 60% → MEMORY-BOUND (poor locality)
       └─ L1 > 60% → BALANCED (likely small kernel, launch overhead)

   Note: a low VMEM/VALU SoL with a dependency-wait-dominant wavefront is latency-bound (C1), not
   memory-bandwidth-bound. Only classify MEMORY-BOUND when HBM/VMEM utilization is actually high
   (≥60% SoL, or L1 locality is the demonstrated problem) — a small arithmetic intensity alone does
   not make a kernel memory-bound.
```

## Cheap checks to run from the raw counters BEFORE trusting a label

The report does not surface these, but each is a few counters already collected, and each has caught a
real mislabel. Run them before forming a hypothesis.

**Validate the peaks first (a wrong denominator invents or hides a bottleneck).**
- **BF16 compute peak reads ~2× low.** BF16 and FP16 MFMA run at the same rate on these parts, so the
  two reported peaks must be equal — the empirical BF16 peak often is not, which makes BF16 compute
  efficiency read ~2× high (we saw 185%, 396%). Any roofline efficiency **> 100%** is a mis-calibrated
  peak (or SFU ops folded into the perf counter on rmsnorm/rope), not a record. Prefer HBM% and F32
  MFMA%; pass `--roofline-data-type` if the tool supports it.
- **HBM can under-report on multi-XCD.** If SoL HBM% looks implausibly low, cross-check with
  `TCC_EA*_RDREQ_DRAM × 64B`, or bytes/time by hand.
- **MoE padding inflates AI.** Recompute arithmetic intensity from *effective* FLOPs (not padded rows)
  and confirmed bytes before believing an "AI far right of ridge → compute-bound" call.

**Check the GPU is actually filled (this was the real limiter in a large fraction of kernels).**
- **Fill:** `CTAs = Grid_Size / Workgroup_Size`. If `CTAs < CU count`, the kernel physically cannot
  occupy the GPU — no tile or register tuning helps; you must partition more (split-K, finer tiles,
  more blocks). This is separate from occupancy: a kernel can hit its per-wave occupancy ceiling and
  still leave most of the GPU idle because it never launched enough work.
- **Occupancy ceiling:** `waves/SIMD ≈ min(8, 512 / (Arch_VGPR + Accum_VGPR))`; 1–2 is register-starved.
- **Spill:** any nonzero `Scratch_Per_Workitem` comes first, before other register work.
- **LDS bank conflict:** `SQ_LDS_BANK_CONFLICT / SQ_LDS_IDX_ACTIVE > 20%` → pad the row stride / swizzle.
- **Coalescing:** `TD_COALESCABLE_WAVEFRONT_sum / TD_LOAD_WAVEFRONT_sum < 50%` → fix access pattern.

Only tune registers/occupancy when achieved occupancy actually sits at the register ceiling; if it is
far below the ceiling, the register footprint is not what constrains you and cutting VGPRs does nothing.

## Bottleneck Shift Analysis (for re-profiling after optimization)

After each optimization round, compare before/after metrics:

1. **What changed**: Which metrics improved/degraded?
2. **New bottleneck**: Did the bottleneck shift? (e.g., compute-bound → memory-bound)
3. **Why**: What optimization caused the shift? (e.g., "Template params freed registers, now memory latency is exposed")
4. **Next action**: What strategy should target the new bottleneck?

Format the analysis as:
```
BEFORE: [bottleneck type] - [key metric value]
AFTER:  [bottleneck type] - [key metric value]
SHIFT:  [old] → [new] because [reason]
NEXT:   Target [new bottleneck] with [strategy]
```
