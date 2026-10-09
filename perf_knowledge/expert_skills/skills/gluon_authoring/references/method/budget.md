# Budget — size the prize before you touch the kernel

**What this stage decides.** The bound-class *prior* (intensity vs the SKU ridge), the MFMA-only
floor and your multiple over it, the closed-form **ideal** budget, the **as-built** budget read off the
artifact, and their **gap** — the headroom, and the input to the stay-plain / hand-off decision
([`entry.md`](entry.md)). It also decides whether a number may be used to **gate or close**, or only
to **rank**.

**When you are here.** Before the first edit. Budget-first is the defining
discipline: do the roofline / budget analysis **before** authoring or editing; do not "optimize to find
out" the budget. You come back here (a) right after the anchor lands, to re-calibrate from its own
trace and IR, and (b) every time a claim needs a "% of roofline" or a "gap = X us".

GEAK's `kernel_workflow` owns the round loop, but the budget still precedes the deep_engineer's first
authored round, and every round's evidence still carries it (the deep_engineer refreshes it when it
re-profiles in its own loop). GEAK's own roofline skill
(`e2e_workflow/knowledge/analysis_skills/roofline/SKILL.md`, `roofline_tools.py`) reads the same peaks
file and keeps its non-fatal degradation ladder; the basis rules below apply to its output too. (If the
deep_engineer keeps the optional `toolctl` bookkeeping, `budget` is also a standing reference on every
stage card.)

**Read order:** `../hardware/atlas.md` table A → `../hardware/roofline-models.md` →
`../hardware/amd-*-skus.md` (pick the chip) → `../hardware/planning-constants.md`. After the first
profile: `../hardware/bound-class-signals.md` → `../hardware/capability-matrix.md`.

---

## 1. Your hardware budget

Before touching the kernel, print the budget. No GPU needed:

```bash
python3 kernel_workflow/scripts/kernel_tools/hw_budget.py --sku MI355X \
    --workload <gemm|attention|attn_bwd|reduction|norm|moe|...> \
    --shapes <k=v,...> --dtype <bf16|fp16|fp8|...>
```

`hw_budget.py` (a GEAK shared kernel tool; the pack path `scripts/hw_budget.py` is a shim) assembles
`perf_knowledge/hardware/data/{sku,hw_constants,workload_models}.json` through `_hwdata`. Shapes are the
model's own variables — GEMM `M,N,K`; attention `b,h,d,s` — see the `--workload` list. A tool that needs
an arch or SKU and was not given one **refuses**; there is no silent default.

**The archetype you name here is load-bearing beyond this page.** `--workload` makes you name the
archetype, and the tool prints the `../workloads/index.md` row next to the ceiling. That row decides
which layers have any surface on this kernel ([`climb.md`](climb.md) `## Layer Backbone`) — read it
now, not when the climb opens.

### Which verdict is robust is per-archetype — read `bound_direction`

There is no single direction for "the byte model": `hw_budget.py` prints a `bound_direction` per
workload, and the two families point opposite ways. `gemm` is `upper` — min-bytes assumes each element
is read once, so intensity is over-stated and the tool leans toward calling things compute-bound;
there, a **memory** verdict is the robust one and a compute verdict needs measurement. `attention` is
`lower` — the model charges every K/V re-read to HBM and excludes L2 reuse, so intensity is
under-stated; there it is the **compute** verdict that is robust. Read the field, do not assume a
direction:

- `bound_direction: upper` ⇒ trust `memory`, measure before trusting `compute`.
- `bound_direction: lower` ⇒ trust `compute`, measure before trusting `memory`.
- `regime: ambiguous (byte routes straddle the ridge)` ⇒ the bracket contains the ridge and **neither**
  verdict is available yet. Do not resolve it in either direction from the model; that is C1's job
  ([`profile.md`](profile.md) `### 3.1 Required evidence — the four dials, every round`).

The bracket itself is the reason: the two byte routes can be far apart (an attention shape can print a
33x spread between `hbm_bytes_min` and `hbm_bytes_model`), and a floor quoted off one end of a 33x
interval is not a floor. `../hardware/roofline-models.md ## Calibration` carries the worked examples.

### Six resources bound every CDNA kernel

Know your number for each before you form a hypothesis. gfx950 (CDNA4, MI350X/MI355X) is the main line;
the gfx942 (CDNA3, MI300X/MI325X) downgrade is noted per row. Take the value from the named key in
`perf_knowledge/hardware/data/hw_constants.json` and pass `--arch` everywhere rather than carrying a
figure across a generation.

| resource | the CDNA-specific fact that bites | gfx942 downgrade |
| --- | --- | --- |
| **MFMA issue** | peak is per-dtype; the matrix unit co-executes with VALU, so VALU is not free but is not additive either. gfx950 adds scaled MFMA / MXFP (fp4/fp6) | no scaled MFMA; fp8 is FNUZ (OCP on gfx950) |
| **Registers** | ArchVGPR **+** AGPR share ONE 512-entry/SIMD file. NVIDIA's 255/thread is per-thread and does not combine. >256 ⇒ **1 wave/SIMD ⇒ zero multi-wave latency hiding** | same |
| **LDS capacity** | per-CU and **arch-specific** — 160 KiB/CU on gfx950; read it from `hw_constants.json` `lds_per_cu_kib` / `hw_budget.py`, do not assume. A second, independent occupancy limiter: a kernel can be register-capped *and* LDS-capped at once | 64 KiB/CU (gfx950 is 2.5×) — a depth or tile CDNA3 reasoning treats as unavailable may simply fit on CDNA4 |
| **LDS bandwidth** | conflict-free `ds_read_b128` sets the steady interval; bank count is **arch-specific** (`hw_constants.json` `lds_banks`; 64 banks on gfx950), 4 B/bank | 32 banks; a swizzle conflict-free on one gen is not known to be on the other |
| **HBM / MALL** | arithmetic intensity vs the SKU ridge decides memory-path-first vs MFMA-continuity-first | lower HBM rate (5.3 / 6.0 TB/s) moves the ridge |
| **L2 / fabric** | re-read patterns can be MALL-resident and cheap, or fabric-bound and not — this one you must **measure**, not derive (C1 in [`profile.md`](profile.md)) | same |

With a workload shape, `hw_budget.py` also prints the **MFMA-only floor in ms and your multiple
over it**. That number sizes the prize: a 1.1× gap and a 4× gap call for different work.

## Roofline method — every percentage carries its basis

One rule, shared with GEAK's roofline skill and `perf_knowledge/profiling/roofline_on_mi.md`:

- **Every "% of roofline" carries two fields**: `numerator_basis` ∈ {`model`, `counters`} and
  `denominator_basis` ∈ {`datasheet`, `empirical@<tool>-<version>`, `in-shape probe`}.
- **A datasheet denominator ranks only.** It may order candidates and set a prior; it may not gate a
  keep, a close or a "nothing left here" claim.
- **Only a measured numerator over a probed or calibrated denominator may gate or close.** This keeps
  GEAK's "do not hard-judge" degradation ladder (a missing input degrades one entry, it does not fail
  the analysis) and adds the pack's interval form, refusal rules and floor probe on top.
- **A ceiling called *calibrated* names the artifact it was measured from.** A hand-typed number and a
  probed one are the same float once they reach the denominator, so a percent-of-ceiling is only as
  credible as the provenance travelling with it: carry the probe's path, or mark the value hand-entered
  and read every percentage off it as an estimate.

**Single source of SKU peaks: `perf_knowledge/hardware/data/sku.json`** (per-SKU rows with `basis`,
`cus`, and per-row `source`). Every consumer reads it — `hw_budget.py`, `calc_perf.py` (its `SKU_PEAKS`
is loaded from the file; a missing dtype is **refused**, never back-filled with fp16/bf16), and GEAK's
`roofline_tools.load_peaks`. Dense datasheet peaks (TF = TFLOP/s), gfx950 first:

| SKU | arch | bf16 = fp16 | fp8 | fp4 | fp32 | HBM | CUs | bf16 ridge (FLOP/B) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MI355X | gfx950 | 2500 | 5000 | 10000 | 157.3 | 8.0 TB/s | 256 | ≈312 |
| MI350X | gfx950 | 2300 | 4600 | 9200 | 144.2 | 8.0 TB/s | 256 | ≈287 |
| MI300X (gfx942 downgrade) | gfx942 | 1307.4 | 2614.9 | — | 163.4 | 5.3 TB/s | 304 | ≈247 |
| MI325X (gfx942 downgrade) | gfx942 | 1307.4 | 2614.9 | — | 163.4 | 6.0 TB/s | 304 | ≈218 |

MI350X is **not** MI355X with a different name: the clock basis differs, so its peaks are lower. MI355X
also carries fp64 78.6. The **ridge is not stored**: it is `peak_tflops[dtype] / peak_hbm_tb_s`, a
different number per dtype (on MI325X: 218 at fp16, 436 at fp8, 27 at fp32), and `hw_budget.py`
computes it per call from the dtype's own peak. A stored single-dtype ridge once put every fp32 kernel
with intensity between 27 and 218 on the wrong side of the crossover. (A cut-down part such as MI308X
— orc-derived row — has a far lower ridge, ~65.)

**rocprof-compute caveats that change a roofline number:**

- `--roof-only` (empirical microbenchmark roofs) on gfx95x needs rocprof-compute **>= 3.6.0**; older
  versions do not produce a valid gfx950 roof, so an "empirical" denominator from them is not one.
  Record the tool version in `denominator_basis` (`empirical@rocprof-compute-<ver>`).
- On gfx950, `FETCH_SIZE` / `TCC_BUBBLE`-based read-byte counts **under-count**. A memory numerator
  built from them alone reads low — the direction that fakes *memory does not bind*. Cross-check against
  `TCC_MISS*128 B` / `TCC_EA0_*` (the route comparison in `### 3.3` below).

Soft discriminator cutoffs (the latency 0.60 cut, SoL 60/40, the 80% saturation line, and the
confirm/discriminate thresholds) live in one table, `perf_knowledge/hardware/data/thresholds.json`;
each entry names its `basis`, and when your measurement disagrees with a cutoff, trust the measurement
and say which cutoff you overrode.

## Ideal budget (compute before editing)

From `BLOCK_M/N/K`, `num_warps`, `num_stages`, dtype, target:

```text
tile/LDS bytes      : A_tile + B_tile, LDS_bytes vs LDS_per_CU   (gfx950 160 KiB; gfx942 64 KiB)
R_acc / R_total     : vs the VGPR/SIMD file -- 512 on CDNA (ArchVGPR+AGPR combined),
                      1536 on RDNA with a 256/wave cap. `probe.py` derives it from --arch
occupancy           : min(by_LDS, by_waves, by_regs)
in-flight bytes     : vs 32 KiB/CU TCP cap
pipeline coverage   : stall_hbm, stall_lds
roofline            : intensity vs ridge (per dtype, from sku.json; bf16 e.g. MI355X ~312,
                      MI350X ~287, MI300X ~247, MI308X ~65) -> bound class
saturation          : grid_tiles / SKU.CUs  ;  waves = ceil(grid_tiles/CUs) ;
                      tail_efficiency = grid_tiles/(waves*CUs)  (<<1 => wave-quantization tail,
                      roofline-models.md ## Saturation / wave quantization) ; layer-6 L2: SKU.L2_MB
feeds-and-speeds    : max(T_mfma, T_lds, T_exp) -> binding resource per iteration
                      (per-resource cycles incl. the LDS-operand-reread term;
                       fused / softmax-between-matmul kernels especially)
```

**`num_stages` on the Gluon path is a budget parameter, not a knob.** In Triton 3.8.0 no pass on the
Gluon path consumes it (`gluon_to_ttgir` does not run the pipeliner), so it enters this budget only as
the champion's recorded depth or as the depth of the LDS ring you plan to **author**
([`climb.md`](climb.md), layer 4). Do not carry plain's value over and treat it as tuning.

## As-built budget (read from the artifact)

| Quantity | Source |
| --- | --- |
| chosen layouts (`#blocked`/`#mma`/`#shared`) + `num_stages` | plain `.ttgir` (`kernel_workflow/scripts/kernel_tools/dump_ir.sh`) |
| actual VGPR / AGPR / spill | `.amdgcn` / compile stats (`probe.py`, `asm_loop_audit.py`) |
| MFMA efficiency, `ds_read` interval, in-flight behavior | ATT (`kernel_workflow/scripts/profile_kernel.sh` with the ATT mode, or `kernel_tools/capture.sh`; rollups via `mfma_efficiency.py` / `calc_perf.py`) — see [`profile.md`](profile.md) |
| achieved DRAM/fabric bytes and rate | `rocprofv3 --pmc` → `kernel_tools/parse_pmc.py` memory block |

On **plain Triton you can only observe aggregates** (total VGPR, total LDS, MFMA eff, `ds_read`
interval); you cannot decompose/control them (R_total split, LDS padding/swizzle, stage independence).
That is enough for the stay-plain / hand-off decision; decomposition + control is a Gluon (Stage-Gluon)
capability. Read the as-built numbers as cycle / utilization counters (clock-insensitive) rather than
absolute time when the box drifts ([`benchmark-hygiene.md`](benchmark-hygiene.md) `## Shared / DVFS GPU
timing`).

## Gap = headroom

```text
gap = ideal_target - as_built_observed   (per bound class)
```

A large gap that needs explicit layout / memory-path / pipeline control ⇒ hand off to the explicit tier
([`entry.md`](entry.md)). A small gap, or a gap owned by wrapper/dispatch ⇒ stay plain.

**A memory gap is only headroom if both sides of that subtraction are real.** An uncalibrated
`ideal_target` divides an analytic byte model by a datasheet peak — two optimistic terms that compound,
and on one measured kernel inflated the gap 3x. Before a memory gap sizes anything: measured DRAM bytes
(`parse_pmc.py`), an in-shape ceiling as a range (`kernel_tools/mem_bw_probe.py`), and a floor probe
confirming memory binds at all. The procedure is `../hardware/roofline-models.md ## Calibration`; the
nine-line check is `### 3.3` below.

**And the two sides have to be independent, which is a different requirement from being real.** Before
using *any* number labelled a ceiling, an upper bound, or an ideal — yours or one a tool printed —
substitute the **ideal value of your own result** into whatever expression produced that number:

> **If the "bound" moves when your result moves, it is a conversion of your result, not a bound on
> it.** It is telling you how much of your result carries through to some other quantity, not how far
> you could go. A real bound is fixed by the hardware, the shape and the algorithm; it does not know
> what you measured.

The check needs no tooling and belongs **before** the number is used, not after a decision is built on
it. Three things make it worth doing every time:

- **The failure is one-directional and expensive.** A conversion read as a ceiling turns "my result is
  poor" into "there is nothing here to take", and those two statements are mathematically unrelated.
  It closes live directions; it never opens dead ones.
- **The label supplies the credibility.** A name containing `ceiling`, or a printed `UPPER bound`, is
  believed on sight — so the reader furthest from the code is the most exposed, and that is usually the
  person making the decision.
- **Check the label against the expression, not against the prose.** One name can carry two different
  quantities (a reachable limit and a discount applied to an optimistic estimate are both "upper"
  something), and a docstring describing one of them does not constrain what the code returns.

This is why the record below carries `hbm_ceiling_source` and the two basis fields: **how a ceiling was
derived is part of the ceiling.** Record the provenance for every bound you report, not just the memory
one.

### 3.3 Before you report a "% of roofline" or a "gap = X us"

Nine lines, each one a way a roofline number has been wrong in practice — and each owned by a dial in
[`profile.md`](profile.md) (C1–C4, trap 5), which is the point: the number is only as good as the
reading under it. Detail + the worked examples: `../hardware/roofline-models.md ## Calibration`.

- [ ] the **numerator** is measured DRAM bytes (`TCC_MISS*128 B`), not the analytic byte model —
      `numerator_basis: counters` (C1)
- [ ] the two byte routes (`TCC_MISS` vs `FETCH_SIZE+WRITE_SIZE`, or `TCC_EA0_*`) agree inside 1% —
      and on gfx950 remember `FETCH_SIZE` / `TCC_BUBBLE` read bytes under-count (C1)
- [ ] the **denominator** is the ceiling probed at THIS access shape — not the datasheet peak, and not
      the read-only peak either — `denominator_basis: in-shape probe` or
      `empirical@<tool>-<version>` (C3)
- [ ] no probe reading exceeded what the device can stream, and the probe's byte accounting is
      injective (a too-high ceiling silently shrinks every gap)
- [ ] the ceiling is stated as a **range**, with its program count and session spread
- [ ] the claimed gap is **larger than that range** (otherwise it is inside the error bar)
- [ ] a **floor probe** confirmed `memory_ideal > non_stream_floor` — i.e. memory binds at all (C4)
- [ ] for a low-precision kernel, the VALU:MFMA op-mix was read first (the FLOP roofline has no axis
      for dequant — [`climb.md`](climb.md) trap 5), and the dtype's real byte width was used (fp4 is
      0.5 B + mx scale bytes, not the 2 B fallback)
- [ ] the gap was weighed by the **served** mix, not by microseconds on one shape
      (`scripts/served_envelope.py` — a win on a shape that carries a negligible share of served time
      is worth a fraction of its own percentage)

A percentage that fails any line may still be reported — labelled with its basis, as a **rank-only**
estimate — but it may not gate a keep or a close.

## Re-calibrate after transcription

Once the Gluon anchor exists, the layouts/stages are explicit, so the as-built budget can be **truly
decomposed** (`R_acc/R_operand/R_prefetch`, exact LDS bytes, stage structure). Re-profile the anchor and
replace planning peaks with measured effective values (`../hardware/roofline-models.md ## Calibration`).
This calibrated budget — not the plain-derived estimate — is the climb's reference. An anchor's
regression versus the roofline / a tuned reference is **expected, not a rejection**; it becomes the
baseline to climb from, after this re-calibration.

Calibrate from **clock-stable** measurements: a budget derived from DVFS-drifting TFLOP/s or from
planning-peak constants mis-sets the bound class ([`profile.md`](profile.md) `## Accuracy problems to
distrust (profile + theory)`, [`benchmark-hygiene.md`](benchmark-hygiene.md) `## Shared / DVFS GPU
timing`). The bound class itself can be mis-read from roofline alone — confirm the binding sub-resource
from the profiler before committing a layer.

## Record

```text
budget/<round>.json:
  bound_class, ideal{lds_bytes,R_total,occupancy,inflight,pipeline_cov,intensity,waves,tail_efficiency},
  as_built{layouts,num_stages,vgpr,agpr,spill,mfma_eff,ds_read_interval,
           achieved_dram_tb_s,dram_mb},                    # parse_pmc.py memory block
  ceiling{hbm_ceiling_tb_s:[lo,hi], hbm_ceiling_source: datasheet|in-shape-probe|empirical@<tool>-<ver>},
  numerator_basis: model|counters,
  denominator_basis: datasheet|empirical@<tool>-<ver>|in-shape probe,
  memory_binds (bool, from the floor probe: non_stream_floor_ms < memory_ideal_ms),
    # i.e. with the streams made cache-resident the kernel finishes SOONER than the memory
    # ideal -> the memory system is what it is waiting on. The other way round the floor is
    # already above the ideal, memory does not bind, and the roofline gap is not a prize.
    # (Same direction the gated packs' floor_probe evidence rule states it in; an earlier
    # revision of this line had the comparison backwards, which reads as "memory binds" on
    # exactly the kernels where it does not.)
  gap, calibrated (bool)   # calibrated = measured bytes AND a probed ceiling, not one of the two
```

`calibrated: false` ⇒ the gap is rank-only. Carry the same fields in the deep_engineer's round log
and `worker_result` notes; the budget file itself lives in its OUTPUT_DIR and is referenced from the
round log (`decision_log.md`, [`records.md`](records.md)).

## Sources

Merged from: `references/phases/budget.md` (all sections); `references/method-reference.md`
`## 1. Your hardware budget` and `### 3.3 Before you report a "% of roofline" or a "gap = X us"`; the
compressed copies of the same two sections in `tile-programming-gluon.md`; the roofline-unification
rule of the gluon-skill-v2 plan (basis fields, sku.json single source, rocprof-compute caveats).

Rewritten / resolved:
- Tool paths: `scripts/hw_budget.py`, `calc_perf.py`, `mem_bw_probe.py`, `parse_pmc.py`, `dump_ir.sh`
  now cited at `kernel_workflow/scripts/kernel_tools/` (pack paths remain as shims); peaks and
  constants at `perf_knowledge/hardware/data/` instead of `references/hardware/*.json`.
- Profiler as-built source: `scripts/profile_kernel.sh` → GEAK `kernel_workflow/scripts/profile_kernel.sh`.
- "escalate (`escalation-gate.md`)" → hand-off decision in `entry.md`.
- The ridge examples (MI350X ~287, MI308X ~65) kept and extended with MI355X / MI300X / MI325X computed
  from the contract's SKU numbers; the ridge is per dtype and never stored.
- `num_stages` in the ideal budget re-scoped as a budget parameter (dead on the Gluon path in 3.8.0).
- gfx950 values first in the six-resource table; gfx942 as a downgrade column.
- Run-mode split (GEAK-embedded vs the upstream `gluon-direction` agent on the `toolctl` spine) removed: one GEAK
  description; the budget file lives in the deep_engineer's OUTPUT_DIR; `toolctl` noted as optional bookkeeping.

Dropped: nothing. The compressed (`tile-programming-gluon.md`) three-line form of §3.3 was a
duplicate of the nine-line form and is subsumed by it.
