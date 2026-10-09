# Planning constants (CDNA / RDNA)

**Numbers only** — plug into `roofline-models.md` and `kernel_workflow/scripts/kernel_tools/calc_perf.py`
(GEAK repo root; this pack's `scripts/calc_perf.py` is a shim). Formulas
live in `roofline-models.md`; ISA mechanism detail in `isa-mechanisms.md`. Treat every
value as **calibratable**; recalibrate from trace per `roofline-models.md ## Calibration`.
The machine forms are single-sourced in `perf_knowledge/hardware/data/`: `hw_constants.json`
(per-arch facts on this page), `sku.json` (per-SKU peaks — **the** peak source; a peak used as a
roofline denominator is `datasheet` basis and ranks only), `thresholds.json` (cutoffs). Where this
page and the JSON disagree, the JSON is current and this page is stale — fix the page.
gfx950 (CDNA4) is the baseline throughout; gfx942 (CDNA3) is the downgrade comparison. GEAK's
per-generation cards carry the datasheet side of the same facts:
`perf_knowledge/hardware/cdna4_mi350/{arch,memory}.md` (gfx950) and
`perf_knowledge/hardware/cdna3_mi300/{arch,memory_hierarchy,occupancy}.md` (gfx942).

Entry: `atlas.md` table A (budget phase). If you know the quantity you want but not what this
pack calls it, `term-index.md` maps search terms to the heading that answers them.

## Terminology

| User / chip | Doc key | Wave | Matrix |
| --- | --- | --- | --- |
| MI350X / MI355X | gfx950 / CDNA4 | 64 | MFMA + scaled |
| MI300X / MI325X / MI308X | gfx942 / CDNA3 (downgrade) | 64 | MFMA |
| MI450 | gfx1250 / CDNA5 | 32* | WMMA + TDM |
| RDNA3 client | gfx11* | **32** (native; WMMA `_w32` planning default) | WMMA |
| RDNA4 client (RX9070 XT / R9700) | gfx120* / gfx1201 | **32** (native; WMMA `_w32` planning default) | WMMA |

\* gfx1250: wave32 in kernel; `get_warp_size` may return 64 — hardcode 32.
**gfx1250 is CDNA5 (MI450), not RDNA4.** RDNA4 client (gfx1201) is a separate
occupancy-first WMMA fork without TDM / named-barrier warpgroups — do not transfer
gfx1250 asynchrony conclusions to RDNA4.
RDNA3 WMMA also has ISA **`_w64`** variants + experimental `mwavefrontsize64`; not the tile-skill
default — see `rdna-fork.md` §Wave size.

## Targets (planning)

| Arch | Chips | Wave | MMA | LDS/CU | LDS banks | LDS alloc / align |
| --- | --- | --- | --- | --- | --- | --- |
| gfx950 (CDNA4) | MI350X/MI355X | 64 | MFMA + scaled | 160 KiB | 64 (4 B) | **1280 B** align |
| gfx942 (CDNA3, downgrade) | MI300X/MI325X/MI308X | 64 | MFMA | 64 KiB | 32 (4 B) | **512 B** min alloc (ISA) |
| gfx1250 (CDNA5) | MI450 | 32 | WMMA + TDM | 320 KiB | — | probe |
| gfx1201 (RDNA4) | R9700 / RX9070 XT | 32 | WMMA | 128 KiB/WGP, ≤64 KiB/WG | 32 (4 B) | probe |

RDNA: `rdna-fork.md`. Do not apply CDNA VGPR formulas on RDNA.

## gfx950 (MI350X / MI355X, CDNA4)

| Resource | Value |
| --- | --- |
| XCD count | 8 |
| Active CUs | 256 (32 per XCD) |
| SIMDs per CU | 4 |
| HBM | 288 GB HBM3E, peak ~8.0 TB/s (both SKUs) |
| Matrix peak (dense, datasheet) | **differs per SKU** — MI355X fp16/bf16 2500 TF, MI350X 2300 TF (lower clock basis); read `sku.json`, never assume MI350X = MI355X |
| LDS peak read | ~256 B/clk (64 banks) |
| TCP (L1 vector) | 32 KiB/CU; in-flight cap 32 KiB/CU |
| L2 | 32 MiB (4 MiB per XCD) |
| MALL (Infinity Cache, memory-side LLC) | 256 MiB — this is **not** the L2 |
| VGPR planning limit | 512 / SIMD (arch + accum combined) |
| Transcendental (exp) | ~¼ full-rate f32 VALU — see `roofline-models.md` feeds-and-speeds |

## gfx942 (MI300X / MI308X, CDNA3) downgrade

| Resource | Value |
| --- | --- |
| Active CUs | **304** (MI300X/MI325X) or **80** (MI308X) — see `amd-cdna3-skus.md` (vs gfx950 256) |
| HBM | MI300X 192 GB, ~5.3 TB/s · MI325X 256 GB, ~6.0 TB/s (vs gfx950 8.0 TB/s) |
| Matrix peak (dense, datasheet) | fp16/bf16 1307.4 TF, fp8 2614.9 TF (MI300X = MI325X) — about half of MI355X; no fp4 |
| LDS / CU | **64 KiB** (vs gfx950 160 KiB — 2.5× less) |
| LDS banks | 32 (vs gfx950 64) |
| LDS peak read | ~128 B/clk (vs gfx950 ~256) |
| VGPR | 512 / SIMD combined budget (same as gfx950) |
| Async direct-to-LDS | **32-bit / 4 B per op** (128-bit is CDNA4-only), clean tiling, swizzled destination with `order=[1,0]`. As on gfx950, a **padded async destination** has failed LLVM translation — build the ring against a swizzled destination first (`../gluon/memory-reference.md`) |
| `ds_read_tr` LDS transpose | **none** (CDNA4-only) |
| fp8 | **FNUZ** (`float8_e4m3fnuz`), not OCP |

Do not assume gfx950 scaled-MFMA / 160 KiB LDS / 16 B async / `ds_read_tr` / OCP fp8 carries to
gfx942. Full CDNA3 page: `cdna3-gfx942.md`.

## gfx1201 (R9700 / RX9070 XT, RDNA4) downgrade

Source: ROCm [GPU arch specs](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
+ `rdna-fork.md`. Numbers align with `calc_perf.py --sku R9700` (`wave_size=32`,
`vgpr_per_wave=256`, `vgpr_per_simd=1536`). Do **not** apply the CDNA
combined 512-VGPR occupancy formula on RDNA (`rdna-fork.md ## Occupancy note`).

**`lds_per_cu_kib` is deliberately absent on every RDNA row and must not be refilled.** It
used to read `64`, which is the per-WGP pool halved. That is arithmetic on the wrong sharing
structure: a WGP is **2 CUs sharing one pool**, so two workgroups that land on the same WGP
contend for a single 128 KiB pool, not for two independent 64s — the per-CU figure is not a
smaller true quantity, it is a quantity that does not exist at that granularity. It is also
*exactly* gfx942's genuine per-CU number, so an RDNA reading was indistinguishable from a
CDNA3 one on inspection. Three quantities, never to be collapsed into one "LDS size":

| Field | Arch | What it is |
| --- | --- | --- |
| `lds_per_cu_kib` | CDNA only | the per-CU pool — **is** the occupancy pool there |
| `lds_per_wgp_kib` | RDNA only | the per-WGP pool (2 CUs) — **this** is the occupancy pool |
| `lds_per_wg_kib` | RDNA only | the per-workgroup **allocation ceiling** (64 KiB) — caps one workgroup, prices **no** occupancy |

Quote a bare LDS number and the reader cannot tell which of the three it is, so name the
scope with it. Where only `lds_per_wg_kib` is recorded for an arch, report occupancy as
**not priced** — do not divide the pool by 2, or the ceiling by anything, to produce a
per-CU figure. `hw_budget.py` emits `lds_capacity.occupancy_pool_scope` and
`occupancy_priced` for exactly this reason.

**Two different VGPR numbers, do not collapse them.** `256` is the **per-wave addressable
cap** (how many VGPRs one wave may name); `1536` is the **register file per SIMD** (what
occupancy divides into). Using 256 as the file — the CDNA habit, where the two happen to
coincide at 512 — under-reports RDNA occupancy by 2-3x. Occupancy on RDNA4 is
`min(cap, file / round_up(vgpr, granule))` with cap 16 waves/SIMD and granule 24; LLVM's own
`; Occupancy:` in the `.s` is the authoritative per-kernel answer **for the register term**
and only for it (`## The emitted ; Occupancy: N is a register-term answer` below). Both the model and
the `.s` reader live in one place — `kernel_workflow/scripts/kernel_tools/amd_occupancy.py` (`waves_by_vgpr` /
`llvm_occupancy`), which `calc_perf.py occ` and the loop audit (`asm_loop_audit.py`;
`isa_loop_audit.py` — flydsl pack) both call.

| Resource | Value |
| --- | --- |
| Wave | **wave32** native (WMMA `_w32` planning default; `_w64` is an ISA/compiler option only) |
| LDS | **128 KiB / WGP** ← the occupancy pool, shared by the WGP's 2 CUs; **≤64 KiB / work-group** is a separate *allocation ceiling* on one workgroup and does **not** price how many fit. There is no per-CU figure on RDNA |
| LDS cacheline | 256 B |
| VGPR | **≤256 / wave** static (wave32) addressable cap, over a **1536 / SIMD** file (granule 24, ≤16 waves/SIMD) — `calc_perf` `vgpr_per_wave=256`, `vgpr_per_simd=1536`. Dynamic VGPR (`S_ALLOC_VGPR`) is **not** an RDNA4 lever — see `rdna-fork.md` §Dynamic VGPR |
| Matrix | WMMA 16×16×16 (`AMDWMMALayout`, **v8-operand** ABI); FP8/BF8 WMMA, **no `mfma_scaled`/blockscale** |
| GMEM | `GLOBAL_LOAD_TR_*` transpose loads (RDNA analog of `ds_read_tr`) |
| Async→LDS / TDM | **none on RDNA4 ISA** (TDM is gfx1250/CDNA5) — hand LDS ping-pong |
| SKU peaks (CUs / L2 / BW / matrix TF) | `amd-rdna4-skus.md` |

Do not transfer gfx950 (160 KiB/CU LDS, wave64, combined-512-VGPR) or gfx1250
(TDM / named-barrier) assumptions to gfx1201.

## LDS throughput / conflict (planning)

```text
gfx950: conflict-free ds_read_b128 steady interval ~16 cyc; full conflict stride 256 B
        (128 B stride = 2-way conflict on gfx950)
gfx942 (downgrade): conflict-free ds_read_b128 steady interval ~16 cyc; full conflict stride 128 B
bank index: gfx950 (addr/4)%64 ; gfx942 (addr/4)%32
```

Machine form: `thresholds.json confirm.ds_read_b128_interval_cyc`; the cross-generation bank
model is GEAK's `perf_knowledge/hardware/shared/memory_model_lds_bank.md`.

## VGPR / occupancy thresholds (CDNA combined budget)

```text
arch_vgpr + accum_vgpr share ONE 512-entry file per SIMD. Allocation granule is 8, cap is 8 waves:
waves/SIMD = min(8, 512 // (8 * ceil(next_free_vgpr / 8)))

<= 64 -> 8 waves ; <= 72 -> 7 ; <= 80 -> 6 ; <= 96 -> 5 ; <= 128 -> 4
<= 168 -> 3      ; <= 256 -> 2 ; <= 512 -> 1 ; > 512 -> SPILL
```

The granule is **8, not 4**, and the ladder has **eight** rungs, not four: an earlier revision of
this block used granule 4 and started at `<= 128 -> 4 waves`, which hides the entire 5-8 wave range
and makes any tile under 128 VGPRs look like it has nothing left to gain. It also printed
`<= 170 -> 3`, a boundary that no granule produces — the real one is **168** (21 granules).
`kernel_workflow/scripts/kernel_tools/amd_occupancy.py` is the implementation and the arbiter
(`waves_by_vgpr`); it has always carried granule 8 / cap 8, so this table was the side that was
wrong. The same ladder is `thresholds.json confirm.vgpr_wave_cliff`. GEAK cards that quote a
**16-VGPR granule** (`perf_knowledge/hardware/cdna3_mi300/occupancy.md`,
`perf_knowledge/hardware/shared/wavefront_simd_vgpr_agpr.md`, `perf_knowledge/hardware/cdna4_mi350/arch.md`)
disagree with this measured ladder; trust `amd_occupancy.py`.

### Read the field the hardware reads, and round it before you compare

`next_free_vgpr` comes from the `.amdgcn` KD — not from profiler `VGPR` meta, and not from a
front-end `n_regs` / `.vgpr_count` print. Those are **reports** of the highest register number
used. A build carrying a `waves_per_eu` hint can pay in AGPRs, so the report stays flat while
the descriptor moves: one measured sweep read `.vgpr_count` **80 / 80 / 80 / 80 / 64** while
`.amdhsa_next_free_vgpr` read **257 / 169 / 129 / 97 / 64** over the same five builds. The knob
was moving the allocator at every setting and the metadata note was blind to it.

**And the allocation is one layer below even that.** What the hardware reads is
`GRANULATED_WORKITEM_VGPR_COUNT` = `COMPUTE_PGM_RSRC1[5:0]` of the 64-byte kernel descriptor,
with `allocated = (granulated + 1) * 8` = `ceil8(ceil4(next_free_vgpr))`. Measured over ~14
builds the round-up is **0 to +7**, never negative, and **zero exactly when `next_free_vgpr` is
already a multiple of 8** — which *measures* granule 8 rather than citing it.

This third layer bites precisely where the first two **agree**. On a default build
`.vgpr_count == .amdhsa_next_free_vgpr` (36 of 36 hint-free builds in one lane) and the
agreement reads as confirmation; neither field is the allocation. One measured cell: printed
**97** says `512/97 = 5` waves/SIMD, allocated **104** says 4. *Two fields agreeing is not two
measurements agreeing when neither one is the quantity.*

**Round before comparing two configurations, not after.** The ladder above is already rounded;
the number the artifact hands you is not, so the natural workflow compares unrounded counts
against rounded rungs. The granule both destroys and creates equalities and the created ones are
invisible: 144 and 140 printed both allocate **144**; 130 and 134 printed both allocate **136**;
and a floor published as "**134** against a ceiling of 128, missing by 6" is really 136, missing
by 8.

## waves/SIMD is not workgroups/CU — convert before spending a round on it

The ladder above is in **waves per SIMD**. The residency that actually feeds bandwidth is
**workgroups per CU**, and a workgroup is indivisible: leftover waves in the register file
cannot be filled by half a workgroup.

```text
wg_per_CU      = min( waves_per_SIMD_by_VGPR * simd_per_cu // num_warps ,
                      LDS_per_CU // lds_bytes_per_wg )
waves_per_SIMD = wg_per_CU * num_warps / simd_per_cu          # the reverse conversion
```

The two forms coincide **only** when `num_warps == simd_per_cu`. On a 4-warp kernel with a
4-SIMD CU the units error can never surface; at `num_warps=2` the true occupancy is **half** the
number written down. Carry `num_warps` as an explicit column on every occupancy table and never
publish the degenerate form. A measured instance of the error inventing a mechanism: two arms
reading `1 wg/CU` and `2 wg/CU` look like a doubling and are **2.00 vs 2.00 waves/SIMD** — the
candidate holds the same eight waves in two half-size workgroups, and the +0.21% it measured was
then "explained" as two effects cancelling rather than read as the expected null.

**The conversion deletes most of the ladder at high warp counts.** At `num_warps=8` on a 4-SIMD
CU the only VGPR thresholds that move `wg/CU` are **64, 128 and 256**; 129-256 is one flat
plateau, and the `<= 168 -> 3` rung is worse than useless there — it allocates a third wave's
registers per SIMD that can never be occupied. Measured: a tile arm that cut `next_free_vgpr`
from **240 to 146** (two rungs up the ladder, −39%) bought **exactly zero** additional
workgroups per CU. Before spending a round on register relief, compute the **next** threshold
that changes `wg/CU` and check the tile can reach it.

**And the VGPR half of a 2-wg/CU entry condition goes vacuous at low warp counts.** The ceiling
is `512 * simd_per_cu / (2 * num_warps)`: 128 at 8 warps, **256 at 4 warps**, 512 at 2. On
gfx950 256 is also the architectural per-thread maximum, so at `num_warps >= 4` the condition
**can never fail** — one configuration reads `vgpr_count: 256`, "passes", and carries **345
spill slots**. The register demand did not shrink, it changed address space. Read
`.vgpr_spill_count` from the same artifact in the same row. (`num_warps` must be a power of two,
so the rungs are 2 / 4 / 8 with nothing between.)

**Neither limiter alone is the occupancy — compute both and take the `min`.** Which one binds is
a property of the `(kernel, configuration, source version)` triple, not of the kernel: in one
four-cell sweep the binding resource read register / register / LDS / LDS, and a single
load-packing change moved it from the register side to the LDS side. Re-determine it after every
structural edit and retire the old answer in place rather than leaving both side by side.

## The emitted `; Occupancy: N` is a register-term answer

LLVM writes `; Occupancy: N` next to every kernel and it is **authoritative for the register
term** — arch-correct by construction, and the right thing to read instead of re-deriving the
file size, granule and cap by hand.

**With one further contamination, on top of the LDS blindness below: it inherits any declared
occupancy hint.** The value is computed from `next_free_vgpr`, and that field is
`max(actual demand, the floor implied by a declared waves-per-EU hint)` — so on a hinted build the
comment reports the *declared* target, not what the code needed. Two arms of one kernel differing by
a **single** register of demand can print different occupancy values, the entire difference being
that one of them declares a hint. And the hint can **cost** waves rather than buy them: a kernel
whose own demand would admit more waves than the hint's floor allows is held down to the floor.
**So: if the build carries a waves-per-EU hint, `; Occupancy: N` and
`next_free_vgpr` describe the request, and only the demand metadata plus a zero spill count
describes the code.**

It is **not** authoritative for a kernel whose LDS is allocated **dynamically at launch**, which
is what the Triton/Gluon-style tile DSLs emit. The emitter prints its own disqualifier four
lines above the value:

```text
; TotalNumVgprs: 80
; ScratchSize: 0
; LDSByteSize: 0 bytes/workgroup (compile time only)
; Occupancy: 6
```

Across **6 retained dumps / 3 distinct VGPR counts / 0 counterexamples**, every emitted value
equals `floor(512 / VGPR)` exactly (64→8, 80→6, 174→2): there is **no LDS term in it at all**.
On the kernel above the real limiter is `163840 / 38144 = 4 wg/CU` = **2 waves/SIMD**, so the
emitted 6 overstates by **3x**; a second lane's independent dump reproduces the shape at 2
against a real 1. So on a dynamic-LDS kernel a hand derivation taking
`min(VGPR-limited, LDS-limited)` is **more correct, not less** — check `LDSByteSize` in the same
block before deferring to the line below it.

**LDS per workgroup is a different field for a JIT kernel than for an AOT one.** A JIT (Triton /
Gluon) kernel's real figure is in the compiled kernel's own metadata (`metadata.shared`); its
ELF `.group_segment_fixed_size` reads **0 — not missing**, which awards it *unlimited*
workgroups per CU. An ahead-of-time binary is the opposite case: `.group_segment_fixed_size` is
the real number there. They are not interchangeable, and a calculator fed the wrong one can
**reverse** a comparison between a JIT kernel and an AOT comparator, silently, with a right
answer whenever the register term happens to bind first.

**Open, and deliberately not generalised:** whether a kernel with **statically** allocated LDS
makes the emitter include the LDS term is **unverified** — no such kernel was among the dumps
above. Do not read this section as settling it in either direction.

**Corollary, about deference rather than about occupancy.** Stamping your own hand derivation
"unverified against the authoritative emitter" is itself a claim that the emitter is in scope,
it carries no evidence, and it points the reader at the number that is 2-3x wrong. Before
deferring to any emitted value: name what it *computes*, not what it is called; check it against
the one thing you can derive independently (does it equal `floor(file / VGPR)` on every dump you
hold? then it is a single-resource bound); and treat a parenthetical like `compile time only`,
`estimated` or `static` in the tool's own output as data, not boilerplate.

## Occupancy is a one-sided criterion

Below some point, too few waves hurts latency hiding. Above that point, more waves does not buy
time. Using an occupancy **loss** to veto a change that spends LDS or VGPR is correct. Using a
computed occupancy **gain** to argue that a change freeing LDS or VGPR will win is not, and this
is the claim with the most measured counter-evidence on CDNA4:

| computed occupancy move | measured outcome |
| --- | --- |
| **+28.6%**, bit-exact, 0 spill / 0 scratch, wider loads | **+0.16%** call-weighted, plus a **−2.9% / −2.4%** regression against an A/A null of 0.24% / 0.05%, at exactly the size where the part should be most occupancy-limited |
| **−50%** (occupancy deliberately halved) | **−2.81%** overall, and *faster* on the large-M buckets |
| **+16.7%** (one extra rung) | **1.038x** on the isolated kernel A/B, **+0.79%** end to end |
| +512 residency slots | lost on precisely the buckets it was supposed to win |

Three points, two directions; the arithmetic was right every time and the prediction wrong every
time, and an out-of-sample follow-up across four distinct VGPR counts reached **7/7**. Two of
those points are confounded in different ways, so **do not average them into an exchange rate —
two contradictory points are not a curve.** The opposite over-reach is worse because it is
self-sealing: *"occupancy is worthless on this part"* must not be written either. The honest
output is labelled points and no slope.

What survives is the **compile-side feasibility** use: does the configuration fit, does it
spill, what does `ceil8(ceil4(next_free_vgpr))` allocate, and what is the next threshold that
actually moves `wg/CU`. What must go is the *justification* — do not motivate a sweep by a
predicted occupancy gain, and do not pre-register a magnitude derived from one. If you time it,
pre-register a direction **and a floor** in the same breath: `+0.2%` is exactly the size of
answer this axis returns when it returns nothing, and with no declared floor it gets written up
as a small win. If you intend to spend on this axis at all, measure your own kernel's exchange
rate (Δtime per Δoccupancy) once first.

## Tail efficiency is one-sided too

Mechanism and formulas: `roofline-models.md ## Saturation / wave quantization` — `waves =
ceil(grid_tiles / CUs)`, `tail_efficiency = grid_tiles / (waves * CUs)`, and the distinction from
the load-imbalance tail. Not repeated here. What belongs on this page is the **label**, because
this is the **second** one-sided metric in the occupancy family, and a reader who has internalised
the section above will reach for this one as the replacement — an occupancy-shaped number that
still looks like it predicts time. It is one-sided in the same way and for a different reason: it
is computed from the launch geometry alone, so it is blind to everything that distinguishes two
configurations at the *same* geometry.

A low tail efficiency is a legitimate statement that a configuration **leaves work on the table**.
It is not a ranking. One measured instance, unusually clean because the usual confound was ruled
out first:

| check | result |
| --- | --- |
| tail efficiency at the losing shape | **−16.3 pts**, against −7.8 / −5.5 / **+0.0** at the other three shapes — the deficit tracks the shape |
| does the comparator share the flaw? | **no** — comparator at **97.8%**, provably clean, so the check that normally kills this story *passed* |
| magnitude | 16.3% × the kernel's ~29% time share ≈ **4.7%**, against an observed **4.1%** deficit — a match to **0.6 points** |
| **the falsifier** | the configuration that **wins by 2.2%** has **identical 81.5%** tail efficiency at that shape, and the delta column is **unchanged at all four shapes** |

Everything above the last row is what a correct diagnosis looks like: right mechanism, correct
arithmetic, confound excluded, magnitude matching to within a point. It was still wrong. **A
quantity that takes the same value for the winner and the loser cannot be what separates them** —
and that one fact refutes the whole chain no matter how well the other rows score.

So: **a magnitude match to within a point is not evidence that a metric can rank.** Agreement in
magnitude is the cheapest thing to obtain — a one-sided metric is derived from the same shape the
deficit is measured at, so it has an excuse to co-vary with it without causing it. Treating the
match as confirmation is the error, and it is the more seductive one here precisely *because* the
comparator check passed: ruling out the flaw-in-common confound feels like it has done the
discriminating work, and it has not.

What survives is the one-sided use: tail efficiency vetoes a **geometry** (this grid does not fill
the device; a smaller tile, split-K, or a persistent kernel would). What must go is the
**comparative** use — do not attribute an A-vs-B deficit to tail efficiency without first checking
its value **for both arms at the same shape**, and if the two arms have the same value, stop. The
cost of that check is one subtraction, and it is the only row in the table that was load-bearing.

## Make an occupancy table falsifiable before you quote it

Every constant on this page is published in the form you **divide by**, and a column computed as
`LDS_per_CU // lds_bytes` cannot corroborate the `LDS_per_CU` it divides by. Reading a threshold
off that column — *"78336 → 2 wg and 87040 → 1 wg, so the budget is 163840, measured"* — is two
divisions by the number under test, and every downstream *"LDS <= 81920 B is a necessary
condition"* sentence inherits the assumption. **An occupancy table earns the word "measured"
only when it predicts a quantity an independent measurement can refute.**

The cheap falsifier: take a streaming kernel whose bandwidth is calibrated at 1 and at 2
workgroups per CU, launch it at 2x the CU count, and vary **only** the size of a dummy LDS
allocation. The residency step is unmissable — one measured sweep shows **1.45x** between two
sizes **512 B apart**, identical in both sweep directions, which is what "163840 B per CU,
inclusive, no reserved region" looks like once it has actually been measured rather than quoted.

**The free falsifier you probably already have: make the compiler refuse.** An over-large tile
raises `OutOfResources`, and the message states the budget outright —
`Required: 174080, Hardware limit: 163840`. That is the property this section is asking for:
the limit is **named by the toolchain, not divided out of the number under test**. It costs one
failed compile, needs no kernel, no timing, and no residency argument.

This is not a hypothetical alternative to the circular reading above — it is the *same cell*.
The worked circular example quotes **87,040 B → 1 wg**, and 87,040 B is what a real `BK=256`,
one-prefetch-stage configuration allocates; carry the second buffer and the request is exactly
**2 × 87,040 = 174,080**, which is the refusal quoted here. The two readings of that cell are
available side by side: one divides the budget by the number under test and calls the quotient
"measured", the other has the budget printed at it. Prefer the one that does not close the loop.

Note also what the refusal is worth as corroboration. This campaign now holds the gfx950 figure
from three sources with **no shared derivation**: the runtime's device query, the driver-level
query, and the compiler's refusal to allocate. Independence is the whole point — three tools
reading one table would be one source counted three times.

**Do not read an `OutOfResources` as a compiler bug.** It is a *capacity* statement, and the
distinction is testable rather than a matter of judgement: capacity failures move when you change
capacity. One lane conflated the two and spent a search on the wrong one. The genuine translation
failure it also hit (an `unrealized_conversion_cast` that survives lowering) persisted across
**all 18 knob combinations**, including one that cuts LDS by **2.5x** — so *"does it survive a
large capacity reduction?"* separates them in a single run: an `OutOfResources` cannot, a real
lowering defect does not care.

Four implementation notes, each of which costs a run if missed:

* **Shared-memory dims must be powers of two.** The violation is a C++ assertion that **aborts
  the process**, not a Python exception, so a per-row `try/except` does not save you — build an
  arbitrary byte count as a sum of power-of-two buffers.
* **Print the compiled shared-memory size next to the requested size.** If the allocator dropped
  or rounded the allocation, the request is not the x-axis.
* **Do not take the last entry of the JIT cache** to identify the kernel you just ran: on a
  second pass every variant is cached and "last" is whatever compiled last.
* **A control must exclude the points at which its own premise fails, in the script.** A
  "the 1-wg/CU control must be flat" gate voided one reading at 27.9% spread entirely from the
  `LDS = 0` row: N workgroups on N CUs does not guarantee one each, and at zero LDS nothing stops
  the dispatcher packing two onto a CU — so zero is the one size at which residency genuinely can
  change, which disqualifies it from a control whose premise is that residency cannot. Narrow the
  gate and record the reason in the artifact, do not waive it in the write-up.

## SKU peaks / ridge / saturation (not here)

Matrix peaks, HBM BW, ridge, `CUs`, and `L2_MB` live in the **SKU layer** (pick your chip):

| ISA class | SKU file |
| --- | --- |
| gfx950 (CDNA4) | `amd-cdna4-skus.md` — MI355X, MI350X |
| gfx942 (CDNA3, downgrade) | `amd-cdna3-skus.md` — MI300X, MI325X, MI308X (80 CU) |
| gfx11* (RDNA3 discrete) | `amd-rdna3-skus.md` |
| gfx1151 (RDNA 3.5 APU) | `amd-rdna35-skus.md` |
| gfx120* (RDNA4) | `amd-rdna4-skus.md` |

`python3 kernel_workflow/scripts/kernel_tools/calc_perf.py roofline --sku <key>` or
`roofline-models.md` §SKU ridge table. Machine single source: `perf_knowledge/hardware/data/sku.json`
(the ridge is derived per dtype, never stored). The `amd-*-skus.md` files are its human view.

Example GEMM intensity M=N=4096, K=8192 FP16: ~1638 ops/byte → compute-bound on MI355X (ridge ~312)
and MI350X (~287); gfx942 downgrade: MI300X (~247), MI325X (~218), MI308X (~65). These are
`datasheet`-basis ridges: they **rank** the regime; a regime call that gates a lever uses a calibrated
ceiling (`roofline-models.md ## Calibration`).

## Extended planning (attention / fused kernels)

- **AGPR ↔ VGPR:** MFMA accumulators in AGPR; VALU ops need `v_accvgpr_read` — affects
  feeds-and-speeds VALU term (`roofline-models.md`). RDNA WMMA has **no** AGPR file.
- **MFMA write→read latency:** ~10+ cycles; `s_nop` in hot loop — count via the static
  ISA loop audit (script name per DSL, below); fix via unroll/occupancy, not reorder alone.
  This is the **completion** hazard; the separate back-to-back **issue interval** and the
  co-execution window it opens are in `isa-mechanisms.md ## MFMA throughput model` and
  `## Matrix/VALU co-execution` (per shape — do not carry one shape's window to another).
- **A deliberate `s_nop` is a different thing from an exposed one.** Padding a stage head to
  shift one wave's LDS burst off the other wave's 3-source VALU is a **phase** lever, not the
  unroll/occupancy fix above. Sweep it; it is not monotone
  (`isa-mechanisms.md ## Instruction-rate facts that flip a lever's sign`).
- **Power / DIDT / PIT:** sustained clock may dip under heavy traffic; calibrate from
  stable-boundary timing (`../method/benchmark-hygiene.md`; GEAK clock facts:
  `perf_knowledge/hardware/cdna4_mi350/clocks_power.md`). Corollary for schedule work: cycles and wall
  time can disagree for a real reason — a denser MFMA stream draws more power and a power cap
  buys back frequency, so judge a scheduling change by cycles and the kernel by both.

## On Gluon

| Item | Gluon |
| --- | --- |
| MFMA layout family (gfx950) | `AMDMFMALayout(version=4)` |
| MFMA layout family (gfx942 downgrade) | `AMDMFMALayout(version=3)` |
| Static ISA loop audit | `kernel_workflow/scripts/kernel_tools/asm_loop_audit.py` (pack `scripts/` shim) |

> **Full CDNA3 capability page: `cdna3-gfx942.md`** (MFMA cannot-select facts, fp8
> `e4m3fnuz`, async-via-`cdna4`-namespace, 64 KiB LDS consequences, cross-arch gate).
> `## gfx942 (MI300X / MI308X, CDNA3) downgrade` in this page is the **planning-number**
> subset only.
