---
title: profiling — rocprofv3 hardware counters on MI300/MI350
kind: technique
gens: [gfx90a, gfx942, gfx950]
updated: 2026-10-07
sources:
  - https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-mi200-performance-counters.html
  - https://rocm.blogs.amd.com/software-tools-optimization/profilers/README.html
---

# rocprofv3 hardware counters (MI350/MI355X gfx950 first; MI300/MI325X gfx942)

## TL;DR
Counters are grouped by HW block: **SQ** (compute unit: instruction mix, MFMA, waves, LDS), **TA/TD/TCP**
(vector L1), **TCC** (L2 / channel cache), **command processor**, and HBM. The bottleneck-relevant
handful: **`SQ_VALU_MFMA_BUSY_CYCLES`** (matrix-engine busy → compute-bound proxy), `SQ_INSTS_VALU` /
`SQ_INSTS_MFMA` (op mix), **TCP** hit/miss (L1), **TCC** hit/miss (L2), HBM bytes, and wavefront/occupancy
counters. Many "metrics" you read in rocprof-compute (e.g. `VALUUtilization`, "MFMA busy %", "L2 hit %")
are **derived** from these raw counters — the official reference documents each derivation formula.

## Collect (PMC workflow)
```bash
rocprofv3 -L                                   # list available counters for this gfx target
rocprofv3 --kernel-trace --stats -- ./app      # which kernels, how long
rocprofv3 -i counters.txt -- ./app             # collect the counters listed in counters.txt
```
`counters.txt` holds a `pmc:` line per group. What happens when you request more counters than fit the
HW PMC slots in one pass is **version-dependent**: some rocprofv3 releases **replay** the kernel
(multi-pass — wall time inflates), some **abort** the app (SIGABRT / rc 134, with a traceback that points
at the workload, not the counter list), and some **hang** while holding the GPU. Do not rely on any one
of these: in GEAK every counter pass goes through `kernel_workflow/scripts/kernel_tools/rocprofv3_safe.sh`
(configurable timeout → rc 124 on a hang, mandatory `--kernel-include-regex`, the HIP-inside-ROCR
device guard), and `parse_pmc.collect_pmc_group` bisects a group that aborts or times out, dropping
only the counters that fail alone — the layer is then reported **degraded**, never as zeros. Either
way, timing is meaningless during a counter pass: profile and time in *separate* runs
([`common_pitfalls.md`](common_pitfalls.md)). (The timeout is for kernel-level passes; an e2e serving
run must use `--no-timeout`, because a long startup and finalize/flush are normal there.)

### Collecting through GEAK
```bash
# optional counter layer of the standard profiler entry (groups = parse_pmc.PMC_GROUPS)
bash kernel_workflow/scripts/profile_kernel.sh <gpu_id> "<benchmark cmd>" <out> --pmc   # or --derived
python3 kernel_workflow/scripts/kernel_tools/parse_pmc.py <out>/pmc "" --arch gfx950   # busy + C1 BW
# one ad-hoc pass, under the GPU lock (it locks itself with --gpu; never set HIP_VISIBLE_DEVICES inline)
bash kernel_workflow/scripts/kernel_tools/rocprofv3_safe.sh --gpu <id> --kernel '<regex>' \
     --out <dir> --pmc "TCC_EA0_RDREQ_sum TCC_EA0_WRREQ_sum" --cmd "<benchmark cmd>"
```
Name hygiene: `TCC_EA0_RDREQ_sum` / `TCC_EA0_WRREQ_sum` (all L2→EA requests) are what `parse_pmc.py`
reads; `TCC_EA0_RDREQ_DRAM_sum` is a different, DRAM-destined-only counter — do not substitute one for
the other. `TCC_EA_RDREQ_sum`, `MemUnitBusy`, `L2CacheHit` are absent on gfx942. On RDNA (gfx1201) none
of the CDNA names apply: list the build's counters with `rocprofv3 -L` / `rocprofv3-avail list --pmc`.

## The counters that answer "what's the bottleneck"
| Question | Counter(s) | Derived metric (rocprof-compute) |
|---|---|---|
| Matrix engine saturated? | `SQ_VALU_MFMA_BUSY_CYCLES` / busy cycles | "MFMA busy %", VALU MFMA SoL |
| VALU vs MFMA op mix | `SQ_INSTS_VALU`, `SQ_INSTS_MFMA` | instruction-mix breakdown |
| VALU busy (throughput) | `SQ_ACTIVE_INST_VALU`, busy CU cycles | **`VALUBusy`** / VALU SoL "VALU Utilization" — the bound discriminator |
| VALU lane occupancy | active-thread VALU cycles | `VALUUtilization` / "VALU Active Threads" — a duty-cycle (divergence) reading, **not** throughput |
| L1 (vector) traffic & hit | `TCP` total read/write, hit vs miss-LRU vs miss-evict | L1 cache hit % |
| L2 / channel cache | `TCC` hit/miss | L2 hit % |
| HBM bytes moved | TCC→HBM byte counters | HBM BW (GB/s), used in roofline |
| Occupancy / waves | `SQ_BUSY_CU_CYCLES`, wavefront + level counters | wavefront occupancy, waves/CU |
| LDS pressure | `SQ_INST_LEVEL_LDS`, LDS counters | LDS SoL, bank conflicts |

TCP derivation example from the reference: `TCP_TOTAL_READ = HIT_LRU_READ + MISS_LRU_READ +
MISS_EVICT_READ` — i.e. an L1 hit-rate metric is built from these selectors, not read directly.

## Architectural facts that shape counter reading
- **MI300X = 304 CUs**, partitioned across **XCDs** (Accelerator Complex Dies), each XCD with its own
  lower cache levels; **Infinity Cache** is the shared last-level cache. Cross-XCD L2 traffic shows up
  as elevated TCC miss / HBM bytes — tie to XCD placement in
  [`../hardware/cdna3_mi300/xcd_chiplet.md`](../hardware/cdna3_mi300/xcd_chiplet.md).
- Per-SIMD scoping: counters take a SIMD mask, default `0xF` (all SIMDs), e.g. `SQ_INSTS_VALU:0xF`.

## Thread-trace / streaming SQ counters (advanced)
Beyond PMC, you can *stream* SQ counters into the thread-trace buffer for an activity timeline:
```bash
rocprofv3 --att-perfcounter-ctrl 3 \
  --att-perfcounters "SQ_VALU_MFMA_BUSY_CYCLES SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INST_LEVEL_LDS"
# convenience activity view:
rocprofv3 --att-activity 10
```
On MI300, `--att-perfcounter-ctrl 3` polls every ~120–240 cycles. Note: ATT streaming counters are a
**ROCm 7.0+** (or build-from-source) feature; the `--att-activity` Summary tab (IDLE/ISSUE/STALL +
VALU/VMEM/LDS utilization) is MI200/MI300-only.

## From counters to a verdict
This table feeds the decision tree in
[`reading_a_kernel_bottleneck.md`](reading_a_kernel_bottleneck.md); plot the same numbers on the
roofline in [`roofline_on_mi.md`](roofline_on_mi.md).

## Pitfalls
- `VALUUtilization` etc. are *derived* — quoting a raw selector value as a percentage is wrong.
- Busy vs duty-cycle: classify a compute bound from `VALUBusy` / `MfmaUtil`; `VALUUtilization` can read
  ~100% on a kernel that is not VALU-bound (it only measures lane fill during cycles the VALU did
  issue). See `kernel_workflow/knowledge/profiling_guide.md` → "Busy vs duty-cycle".
- gfx950: FETCH_SIZE and TCC_BUBBLE-derived read bytes **under-count** (rocprof-compute reports the
  same), so a counter-based byte numerator there is a lower bound — carry the independent routes'
  range (`parse_pmc.py`). rocprof-compute `--roof-only` on gfx95x needs >= 3.6.0.
- Counter set is gfx-specific: a counter present on gfx942 may be absent/renamed on gfx950 — always `-L`.
- Asterisked counters in the reference are "validation in progress"; treat with caution.

## Verify
`rocprofv3 -L | grep -i mfma` shows the MFMA busy counter exists for your target before you script it.

## Sources
- Counter block taxonomy (SQ/TCP/TCC/CP), TCP/TCC derivations, `SQ_VALU_MFMA_BUSY_CYCLES`, `SQ_INST_LEVEL_LDS`, ATT streaming + SIMD masks, 304 CUs / XCD / Infinity Cache: ROCm MI300/MI200 performance-counters reference.
- `rocprofv3 -L` / `-i` / `--kernel-trace` / replay-on-overflow workflow: ROCm Blogs profilers intro.
