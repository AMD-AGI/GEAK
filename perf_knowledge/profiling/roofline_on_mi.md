---
title: profiling — building and reading a roofline on MI355X / MI350X (gfx950) and MI300X / MI325X (gfx942)
kind: technique
gens: [gfx950, gfx942]
dtypes: [bf16, fp16, fp8_e4m3, fp8_e4m3_fnuz, fp4_e2m1, fp6, int8, fp32]
updated: 2026-10-07
sources:
  - https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html
  - https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
---

# Roofline on Instinct — the one GEAK method

## TL;DR
A roofline plots **achieved FLOP/s vs arithmetic intensity (FLOP/byte)** under two ceilings: a
sloped **bandwidth roof** (HBM; L2 / Infinity-Cache roofs above it) and a flat **compute roof** per
dtype. Every "% of roofline" in GEAK is a ratio of a **numerator** (bytes or FLOPs the kernel moved,
over its measured time) to a **denominator** (a ceiling), and it must say what both rest on:

| field | values | meaning |
|---|---|---|
| `numerator_basis` | `model` \| `counters` | analytic byte/FLOP model, or measured profiler counters |
| `denominator_basis` | `datasheet` \| `empirical@<tool>-<version>` \| `in-shape probe` | nameplate peak; a measured roof from a named tool+version; a probe at THIS kernel's access shape |

**Datasheet denominators rank only** (which kernel to look at first, a prior for the bound). **Only a
measured numerator over a probed or calibrated denominator may gate or close** a kernel ("at the
ceiling", "done", "negative result"). Everything else is a prior, reported with its basis, never a
verdict. gfx950 (CDNA4) is the main line; gfx942 (CDNA3) is the downgrade.

## The ceilings (datasheet — rank only)
All per-SKU peaks come from one table, `perf_knowledge/hardware/data/sku.json`, rendered for reading
in [`../hardware/cdna4_mi350/peak_tables.md`](../hardware/cdna4_mi350/peak_tables.md) (gfx950) and
[`../hardware/cdna3_mi300/peak_tables.md`](../hardware/cdna3_mi300/peak_tables.md) (gfx942). Pick the
**detected product's** row (`scripts/gpu_identity.py` `sku` → `mi355x` / `mi350x` / `mi300x` / `mi325x` /
`mi308x`): MI350X is not MI355X (2.3 vs 2.5 PF dense FP16 — different clocks), MI325X is not MI300X
(6.0 vs 5.3 TB/s). A dtype the row does not list has **no** compute roof — never substitute FP16's.

| SKU (gfx) | FP16/BF16 | FP8 | FP4/FP6 | FP32 | HBM | FP16 ridge |
|---|---|---|---|---|---|---|
| **MI355X** (gfx950) | 2.5 PF | 5.0 PF | 10 PF | 157.3 TF | 8.0 TB/s | ≈ 312 FLOP/B |
| **MI350X** (gfx950) | 2.3 PF | 4.6 PF | 9.2 PF | 144.2 TF | 8.0 TB/s | ≈ 288 FLOP/B |
| MI300X (gfx942) | 1307.4 TF | 2614.9 TF | — | 163.4 TF | 5.3 TB/s | ≈ 247 FLOP/B |
| MI325X (gfx942) | 1307.4 TF | 2614.9 TF | — | 163.4 TF | 6.0 TB/s | ≈ 218 FLOP/B |

The ridge is **per dtype** (MI355X: ~625 FLOP/B at FP8, ~1250 at FP4, ~20 at FP32) and is computed,
never stored. Caches add roofs above HBM: L2 is 4 MB per XCD; the 256 MB Infinity Cache (MALL) is a
memory-side cache — a footprint inside it rides a higher roof than HBM in a repeated benchmark.

## The ladder (from rank to gate)
GEAK never hard-judges on a datasheet ratio; it climbs a ladder, and each rung says what it licenses:

1. **Analytic budget (rank).** `kernel_workflow/scripts/kernel_tools/hw_budget.py --sku MI355X
   --workload <archetype> --shapes ... --dtype ...` (or `--tensors` for a byte manifest). Output is a
   **bracket** — bytes from footprint to issued traffic, intensity `[lower, upper]` — and both floors
   with `numerator_basis: model`, `denominator_basis: datasheet`, `may_gate: false`. The e2e
   `roofline` analysis skill does the same per head kernel from `sku.json`.
2. **Refusals before any number.** The two-term model does not apply — and the tools refuse rather
   than print a gap — when: fewer workgroups than CUs (`--grid`); the working set fits the memory-side
   LLC in a repeated-iteration benchmark (`--footprint-mb`, memory floor void); the timed region is
   several dispatches (`--dispatches`); the dtype has no rate; the FLOPs issue on VALU, not MFMA
   (`--flops-engine valu`: the matrix peak is not their ceiling).
3. **Empirical roofs (rank, better prior).** `rocprof-compute profile --roof-only` measures the box's
   own roofs → `denominator_basis: empirical@rocprof-compute-<version>`. Record the version.
4. **In-shape probe (gate-eligible denominator).** `kernel_workflow/scripts/kernel_tools/mem_bw_probe.py
   --sku MI355X --runlen ... --stride ... --rw-mix ...` measures the HBM ceiling at THIS kernel's run
   length, stride, read/write mix and program count, as a range; a dense back-to-back MFMA loop at the
   kernel's dtype does the same for compute. Feed them back: `hw_budget.py ... --measured-hbm-tb-s
   lo,hi --measured-from probe.json` → `denominator_basis: in-shape probe`.
5. **Counters numerator.** rocprofv3 PMC routes (`parse_pmc.py`: `TCC_MISS`×line, `TCC_EA0_RDREQ/WRREQ`,
   `FETCH_SIZE+WRITE_SIZE`) → `numerator_basis: counters`; pass every route
   (`--measured-dram-mb a,b,c`) and carry the spread — routes that disagree widen the answer.
6. **Floor probe before a close.** Before calling a kernel at its ceiling, a floor probe (strip the
   non-stream work) shows whether the remaining gap is removable bytes or a rate gap.

Only rungs 4 + 5 together (counters over a probed/calibrated ceiling, `may_gate: true`) may gate a
round or close a kernel. A datasheet ratio near 100% is not "done"; a datasheet ratio above 100% is a
wrong model or an unvalidated peak.

## Classifying the bound (one threshold table)
The cut points live once, in `perf_knowledge/hardware/data/thresholds.json` → `bound_classification`
(read by the e2e `roofline_tools.py`), one ladder on a 0–100 "% of a ceiling" scale:
**≥ 80 saturated, ≥ 60 bound, [40, 60) middle band, < 40 low.** Per metric:

- **roof util** (achieved / roof on each roofline axis): the AI-selected roof binds only at ≥ 0.60;
  below it on every tabulated axis (and above the ~5 µs dispatch floor) the kernel is
  **latency/occupancy-bound** — a small AI alone never makes a kernel memory-bound. A missing compute
  peak is unknown (`compute_util: null`), never 0. ≥ 0.80 on memory: at the wall — cut bytes.
- **SoL pipe %** (rocprof-compute VALU / MFMA / VMEM busy — read busy counters, not duty-cycle
  `VALUUtilization`): one pipe ≥ 60 with the others < 40 → bound on that pipe; all < 40 → latency
  (dependency wait = C1 latency, issue wait = C2 occupancy); all in [40, 60) or no pipe leading by ≥ 15
  points → balanced. See `kernel_workflow/knowledge/profiling_guide.md`.
- **VMEM % of HBM peak ≥ 80** → memory sub-class *bandwidth*: cut bytes, do not add in-flight.

## Build the empirical roof (rocprof-compute)
```bash
rocprof-compute profile --name myrun --roof-only -- python bench.py
# → workloads/myrun/<SOC>/{roofline.csv, empirRoof_gpu-0_FP16.pdf, ...}
rocprof-compute analyze -p workloads/myrun/<SOC>/ --roofline-data-type FP16
```
`--roof-only` collects only roofline counters and runs on-device microbenchmarks to get empirical
roofs (saved in `roofline.csv`), then emits one PDF per dtype; `--kernel-names` labels the points.
Run it through GEAK's profiler entry (`kernel_workflow/scripts/profile_kernel.sh`, under `gpu_lock.sh`)
— never with an inline `HIP_VISIBLE_DEVICES`.

**rocprof-compute caveats that move a denominator or a numerator:**
- **`--roof-only` on gfx95x needs rocprof-compute ≥ 3.6.0.** Older builds produce no usable gfx950
  roof; check `rocprof-compute --version` and put the version in the basis tag
  (`empirical@rocprof-compute-3.6.0`).
- **gfx950 `FETCH_SIZE` / `TCC_BUBBLE`-derived read bytes under-count.** The memory chart's read bytes
  and any `FETCH_SIZE + WRITE_SIZE` numerator read LOW on gfx950, which makes a kernel look further
  from the bandwidth roof than it is. Use the `TCC_EA0_RDREQ*` / `TCC_MISS` routes, carry the range
  (`parse_pmc.py --arch gfx950` stamps the caveat), and cross-check against the model bytes.
- Use the matching `--roofline-data-type`: an FP8 GEMM against the FP32 roof (the tool default) looks
  artificially terrible.

## Reading a point
- **On the sloped roof** → BW-bound. Raise arithmetic intensity (fuse epilogues, larger BLOCK_K, reuse
  in L2 / Infinity Cache), or move fewer bytes (lower precision — a different roof and ridge).
- **On the flat roof** → compute-bound. Only a lower-precision path or a better MFMA-shaped kernel
  helps. MI300X GEMM sustains only ~45–55% of the datasheet flat roof (software ceiling) — a point at
  ~50% of datasheet FP16 may already match the best library kernel.
- **Under both roofs** → occupancy- or latency-bound; counters disambiguate
  ([`reading_a_kernel_bottleneck.md`](reading_a_kernel_bottleneck.md)).
- **Improvement = the point moves up/right toward a roof**, not merely lower wall time.

## Pitfalls
- Reporting a "% of roofline" without its basis pair, or gating on a datasheet denominator.
- Quoting MI355X peaks for an MI350X, or MI300X bandwidth for an MI325X.
- Substituting the FP16 peak for a dtype the SKU does not list (FP4 on gfx942, FP8 on RDNA3).
- Reading a gfx950 `FETCH_SIZE` numerator as the true byte count.
- Forgetting cache roofs: a "BW-bound" verdict against HBM may be L2/MALL-resident.

## Verify
- The record carries `numerator_basis` / `denominator_basis` and, for any gate, `may_gate: true`.
- `roofline.csv` exists, came from rocprof-compute ≥ 3.6.0 on gfx95x, and its empirical compute roof
  is below (not above) the datasheet peak.
- `python3 kernel_workflow/scripts/kernel_tools/extract_sku.py --check` is clean (the peak table).

## Sources
- `--roof-only`, empirical microbench roofs, `roofline.csv`/PDF, `--roofline-data-type`: ROCm Compute
  Profiler profile-mode docs (link above).
- Per-SKU peaks and ridges: `perf_knowledge/hardware/data/sku.json` (single source), citing the AMD
  MI300X / MI325X / MI350X / MI355X data sheets and the CDNA3 / CDNA4 whitepapers.
- Thresholds: `perf_knowledge/hardware/data/thresholds.json` `bound_classification`.
- gfx950 counter caveats and the in-shape probe / floor-probe method: the Gluon pack
  (`perf_knowledge/expert_skills/skills/gluon_authoring/references/hardware/roofline-models.md`,
  `bound-class-signals.md`) and `kernel_workflow/scripts/kernel_tools/{parse_pmc,mem_bw_probe,hw_budget}.py`.
