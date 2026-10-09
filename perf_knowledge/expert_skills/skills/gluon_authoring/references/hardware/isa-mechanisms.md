# ISA mechanisms (CDNA3 / CDNA4 / gfx1250)

**Hardware instruction facts only** — no DSL API. Default MFMA shapes, async paths,
LDS transpose, CDNA3↔4 deltas, and RDNA3↔4 deltas. API mapping lives in each skill's
DSL references; capability evidence in `capability-matrix.md`.

Entry: `atlas.md` table B (ISA mechanism column).

Official ISA PDFs: [CDNA3 (MI300)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf),
[CDNA4](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf).

## CDNA3 vs CDNA4 — ISA deltas

gfx950 is the baseline column; gfx942 is the downgrade. GEAK's per-generation ISA cards carry the
datasheet-side view of the same deltas: `perf_knowledge/hardware/cdna4_mi350/isa_notes.md`
("New / changed instruction families vs gfx942"), `perf_knowledge/hardware/cdna4_mi350/matrix_core_blockscale.md`,
and the downgrade `perf_knowledge/hardware/cdna3_mi300/isa_notes.md`. This page keeps the
probe-verified facts those cards do not carry (provenance lines below).

| Category | gfx950 (CDNA4) | gfx942 (CDNA3, downgrade) |
| --- | --- | --- |
| LDS / CU | **160 KiB**; 64 banks; **1280 B** align | 64 KiB; 32 banks; **512 B** alloc |
| LDS offset | **18-bit** M0 | 16-bit M0 |
| MFMA f16/bf16 | **+16×16×32, 32×32×16** | 16×16×16, 32×32×8 |
| MFMA fp8 (ISA) | + **F8F6F4** unified (`CBSZ`/`BLGP`) | 16×16×32, 32×32×16 regular |
| MFMA fp4/fp6 (ISA) | **16×16×128**, **32×32×64** scaled | **none** |
| Block scale | `V_MFMA_SCALE_*`; scale **E8M0**; K-block=32 | **none** |
| Sparse MFMA (SMFMAC) | **K doubles**: `16x16x64` / `32x32x32` | up to `16x16x32` / `32x32x16` |
| LDS→MFMA transpose | `DS_READ_B64_TR_B16/B8/B4`, `B96_TR_B6` | **none** |
| GMEM→LDS | **+DWORDX3/X4**; **16 B/op** | dword `buffer_load_lds` |
| Cross-lane swap | **`V_PERMLANE16_SWAP_B32`** / **`V_PERMLANE32_SWAP_B32`** | **none** |
| Scale-convert helpers | `V_CVT_SCALEF32_*` family (MX dequant), `V_CVT_PK_BF16_F32` | **none** |
| XF32 MFMA | **removed** | present |
| GWS (`DS_GWS`) | **present** — GWS is removed on **RDNA4**, not CDNA4 | present |
| TDM | gfx1250 only | **none** on CDNA3/4 |

> Provenance — the `Sparse MFMA`, `Cross-lane swap`, `Scale-convert`, and `GWS` rows were
> (re)derived by stock-assembler probe on ROCm 7.1 / LLVM 20:
> `echo '<insn>' | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu={gfx942,gfx950}`.
> `ds_gws_init` / `ds_gws_sema_v` assemble on **both** gfx942 and gfx950 and are rejected
> only on gfx1201 (`instruction not supported on this GPU`) — an earlier revision of this
> page wrongly listed GWS as a CDNA4 removal.

gfx942 downgrade note: hardware supports async→LDS, but the direct-to-LDS width is 32-bit
(4 B/op) so it is frequently `s_waitcnt`-bound and net-negative — verify generated ISA
and A/B vs sync staging (the default authored-ring producer on gfx942). On the Gluon path
`num_stages` drives nothing in 3.8.0 on either arch; the ring is authored. The DSL entry point for async→LDS, and the pointer to the full CDNA3
capability page where one ships, are in the per-DSL section at the end of this page.

## Default MFMA shapes (ISA directory)

Authoritative shape list — DSL-specific selectors (`instr_shape`, `mfma_shape`, atoms)
must be consistent with this table.

| dtype | gfx950 (CDNA4) | gfx942 (downgrade) | Notes |
| --- | --- | --- | --- |
| fp16/bf16 | **16×16×32**, 32×32×16 | 16×16×16, 32×32×8 | prefer 16×16×32 on gfx950 |
| fp8 | 16×16×32, 32×32×16 + F8F6F4 | 16×16×32, 32×32×16 | see §ISA vs DSL below |
| int8 | same family | 16×16×32, 32×32×16 | acc often int32 |
| mxfp4 / fp4 | **16×16×128**, **32×32×64** scaled | — | `mfma_scaled` on ISA |

TileLang may also codegen `(16,16,64)` and `(32,32,32)` — valid gfx950 shapes; document
in the `tile-programming-tilelang` skill's gemm-mfma reference, not duplicate full tables here.

## ISA vs DSL lowering (fp8 / fp4 on gfx950)

ISA exposes both **regular fp8 MFMA** (16×16×32) and **scaled** paths (16×16×128). DSLs
lower differently — do not cross-skill copy API paths.

| DSL | gfx950 fp8 matrix path | gfx950 fp4/mxfp4 |
| --- | --- | --- |
| **Gluon** | regular `cdna4.mfma` at 16×16×32 / 32×32×16; `cdna4.mfma_scaled` at 16×16×128 / 32×32×64 | `mfma_scaled` |
| **FlyDSL** | regular `rocdl.mfma.f32.16x16x32.fp8.fp8` | `mfma_scaled` mandatory |
| **TileLang** | regular MFMA (`fp8_e4_4_t` / `fp8_e5_4_t`) | `mfma_scaled` mandatory |

Gluon has no fp8 API gap: `cdna4.__all__` re-exports `*__cdna3_all`, which carries `mfma`, and
that `mfma` is a thin `_semantic.dot` wrapper with no dtype restriction. The fp8 choice is
therefore per `(version, M, N, K)` — the shape table above — not per DSL.

Plain Triton `tl.dot(fp8)` on gfx950 is a separate comparator path (Gluon skill).

## MFMA throughput model (planning cycles)

The matrix pipe runs an MFMA as a fixed number of **passes**, one `PASS` = **4 cycles**, so
`issue interval = PASS x passes` — the earliest cycle the SIMD can issue the *next* MFMA of that
shape. Read the interval as **back-to-back issue spacing**, not as completion latency (that is
the separate write→read hazard, `planning-constants.md ## Extended planning`).

| Path | Instruction / shape | passes | ~cyc / MFMA (issue interval) | Notes |
| --- | --- | --- | --- | --- |
| FP16/BF16 | `mfma_f32_16x16x32` | **4** | ~16 | ~4 MFMA per 64-cyc mem op |
| FP16/BF16 | `mfma_f32_32x32x16` | **8** | ~32 | the shape the co-execution window below is validated on |
| FP8 scaled | `v_mfma_scale_f32_16x16x128_f8f6f4` | **8** | ~32 | BLOCK_K=128 class |
| MXFP4 scaled | `v_mfma_scale_f32_16x16x128` cbsz/blgp=4/4 | **4** | ~16 | scale side path |
| VALU (wave64) | `v_add/mul/fma_f32`, `v_exp_f32`, … | — | ~4 | issue/throughput cost per wave64 VALU op; the cyc/op constant for PMC-free **VALU-busy** estimates |

> Provenance — CDNA4 ISA PDF (`hw_sources.sh pdf <doc> <p0> <p1>`, Tier 3): `PASS = 4 clock
> cycles` and the per-instruction pass count. Derive the interval rather than carrying it: a
> shape absent from this table has its pass count in your own ISA document, and a
> pass count read for one dtype does not transfer to another.

## Matrix/VALU co-execution — the CDNA-only overlap budget

The budget has two independent numbers per matrix op, and they come from different places:
the **interval** (how long until the matrix pipe accepts the next op of that shape) and the
**window** (how much of that interval another pipe can issue into). The interval is derivable
from the ISA; the window is not, and must be measured.

| Arch family | Matrix op | Interval | Co-issuable **window** |
| --- | --- | --- | --- |
| CDNA4 (gfx950) | `v_mfma_f32_32x32x16_f16` / `_bf16` | 32 cyc | **~24** of 32 — a head **read phase of ~8 cyc** takes no co-issue |
| CDNA4 (gfx950) | `v_mfma_f32_16x16x32_f16` / `_bf16` | 16 cyc | `unknown-needs-probe` |
| CDNA4 (gfx950) | `v_mfma_*_16x16x128_f8f6f4` (scaled) | 16 cyc if **both** operands f4, else 32 | `unknown-needs-probe` |
| CDNA4 (gfx950) | `v_mfma_*_32x32x64_f8f6f4` (scaled) | 32 cyc if **both** operands f4, else 64 | `unknown-needs-probe` |
| CDNA3 (gfx942) | `v_mfma_f32_16x16x16_f16` | 16 cyc | **12** of 16 |
| CDNA3 (gfx942) | `v_mfma_f32_32x32x8_f16` | 32 cyc | **28** of 32 |
| RDNA3 (gfx11*) | `v_wmma_f32_16x16x16_f16` | 32 cyc | **0** — no co-execution |
| RDNA4 (gfx120*) | `v_wmma_f32_16x16x16_f16` | 16 cyc | **0** — no co-execution |

**Lever consequence:** hiding VALU work (softmax `exp`/max/scale, dequant, layout fixup)
*underneath* the matrix op is a **CDNA-only** strategy. The identical restructuring on
RDNA yields **zero** overlap — there the budget belongs to occupancy and VOPD dual-issue
instead. Do not port a CDNA "interleave VALU into the MFMA shadow" plan to a WMMA target
and expect it to pay.

**The window is per shape, not per arch, and it is not the interval minus a constant.** Size it
from the MFMA a region actually contains: giving a 16-cyc shape the window of a 32-cyc one
over-fills every one of them. The gfx942 pair above is the counter-example that kills the
subtract-a-constant shortcut — `12 of 16` is not `16 - 8`. A mixed-shape region takes the
**shortest** member's window, so that no window over-fills.

**Read the scaled rows carefully — the interval is operand-dependent.** For the scaled f8f6f4
shapes the cost depends on the operand *formats*, so the same mnemonic is two different budgets:
all-f4 runs at the short interval and anything involving f8 at double it. A low-precision kernel
therefore does not inherit the f16 budget of its tile shape, and mixed-format operands do not
inherit the all-f4 one. Derive it from the formats the region actually emits.

### Calibrating the window for a new shape

Only one row above is calibrated. A shape whose window is `unknown-needs-probe` **must not**
borrow a neighbour's number — a budget built on a borrowed window is arithmetic on a guess, and
it will read as a scheduling failure rather than as a bad input. The window is behaviour, not
documented mechanism, so it is measured the same way the calibrated row was:

1. **Get the interval from the ISA**, not from a trace: pass count x cycles-per-pass for that
   exact mnemonic and dtype (`## MFMA throughput model`).
2. **Take an instruction trace that timestamps every issue** on a loop that actually contains
   the shape, and read where each non-matrix op issued relative to the matrix op ahead of it.
   The window is the span after a matrix op's issue in which another pipe's ops are observed to
   issue; the head span in which none do is the read phase.
3. **Confirm with a class the model already prices.** Fill a region with a known-cost class until
   the measured cadence degrades — the point at which it does gives the window independently of
   step 2's interpretation.
4. **Record it as a new row with its provenance tier**, and re-derive rather than carry it when
   the dtype, the shape, or the target changes.

If a shape cannot be calibrated now, record a **scoped ceiling for the budget on that shape** and
fall back to the levers that need no window arithmetic — the structural ones that reduce demand.
Do not record it as a hardware wall: the overlap mechanism exists across the CDNA family, and an
uncalibrated window is missing knowledge about *this* shape, not absent hardware.

### What fits in the window, and at what price (gfx950, 32-cyc shape)

| Class | Issue cost | Fits per ~24-cyc window |
| --- | --- | --- |
| VALU (`v_fma`, `v_add`, `v_max3`, `v_cvt_pk_*`) | **4 cyc** | **6** |
| TRANS (`v_exp_f32`) | **8 cyc** | **3** — a transcendental is hideable, just at twice the price |
| cross-lane (`v_permlane*_swap`) | **20 cyc** | **1**, plus a single 4-cyc op beside it |
| packed f32 (`v_pk_*`) | 4 cyc per **2** elements *as packed*; **8 cyc** once split | **0 as packed** — it does not co-issue with a matrix op at all. Winning a window means being **scalarized** into two 4-cyc ops (so it costs two slots, not zero); losing one means staying packed and being **exposed**. See the packed row in the next section |

**Lever consequence:** this turns "where does the non-matrix work go" into arithmetic. Per region,
`capacity = (MFMA in region) x window` against `demand = sum of the class costs`; when demand
exceeds capacity no ordering wins and the work itself has to shrink or move. Method:
`../tile-programming/llir-codesign.md`.

**Roll the regions up and you get the loop's ceiling — before touching the GPU:**

```text
per region:    exposed = max(0, demand - capacity)
per loop body: ceiling = mfma_cycles / (mfma_cycles + sum of exposed over all regions)
```

That is the best in-loop matrix efficiency **any** schedule of this kernel can reach, so it
decides which problem you have. A kernel whose regions all fit has `ceiling = 1.0`, and every
point it measures below that is a scheduling or placement failure to chase in
`../tile-programming/llir-codesign.md`. A kernel with exposed work has a lower ceiling and the
gap to it is not recoverable by ordering — the work has to shrink or move to another region
first. Compute this before proposing a lever: it costs no GPU time and it is the difference
between tuning a schedule and tuning a budget. Count a packed op as already exposed, since it
cannot be covered at all.

> Provenance — two different tiers, and they are not interchangeable. The **co-issuable cycle
> counts for CDNA3** are AMD Matrix Instruction Calculator `--detail-instruction`
> (`hw_sources.sh layout <arch> <instr> -d`): the `Can co-execute with VALU` /
> `VALU co-execution cycles possible` fields. CDNA4/gfx950 is **not** covered by the
> calculator at all, so gfx950 is answered from the **in-pack CDNA4 ISA databases** instead
> (`primary-sources.md` — `gfx950_isa.py`, offline, no PDF render and no network; PDF rendering is
> no longer the default gfx950 path). Its 32-cyc interval is ISA-derived there
> (`## MFMA throughput model`).
>
> **The window is a different matter, and the ISA does not answer it.** The CDNA4 document
> publishes no co-issuable-cycles figure — it carries a required-NOP hazard table instead
> (`gfx950_isa.py hazard`), which is a different fact. So the **~8-cyc head read phase and the
> per-class costs above are behaviour read off an instruction trace**, not documented mechanism,
> and no amount of querying the ISA will confirm them. A trace timestamps every issue, so
> re-derive both on your own kernel and shape rather than carrying these: that is the same
> evidence they rest on, and `### Calibrating the window for a new shape` is the procedure.
>
> **Third tier, weaker than both: the scaled-shape intervals** (`f8f6f4`, and their dependence on
> the operand formats) are transcribed from the cost table of a scheduling tool built for this
> target, not from the ISA document or a trace. A tool's cost table is a statement about what
> that tool will price — good enough to plan with, and *not* evidence about the hardware. Treat
> those rows as provisional and confirm the interval from your own ISA document before budgeting
> on them; their windows are unmeasured and are marked so.

## Instruction-rate facts that flip a lever's sign (CDNA3/4)

Whether a VALU-reduction lever helps depends on **issue rate**, not just whether the
instruction exists — and the wrong NVIDIA-trained prior ("exp needs a special unit")
leads to skipping a real win or chasing an impossible one.

| Fact | Arch | Lever consequence |
| --- | --- | --- |
| `v_exp_f32` is a **1/4-rate transcendental** (issues ~every 4 cyc vs 1 for FMA) | CDNA3/4 | **Polynomial exp2** (roundeven + a few full-rate FMAs + bit-assembled `2^n`) *can win* where the loop is **exp-port-bound** — it offloads the transcendental port. Net win only where exp-port-bound (e.g. small grids); on total-VALU-bound loops the extra FMAs lose. Measure per region. There is **no separate MUFU/transcendental port** to "free" like NVIDIA — the win is issue-rate, not port-offload. |
| `v_cvt_pk_bf16_f32` (pack 2×f32→2×bf16) | **CDNA4-only** | bf16 output packing **cannot** be cheapened on gfx942 — the bf16-truncation `and/or/shift` sequence is the only path; `llvm-mc -mcpu=gfx942` rejects the cvt. |
| bf16 MFMA max K = **32×32×8** (no K16) | gfx942 | A K16 bf16 MFMA path is silicon-impossible on gfx942 (assembler-rejected); see `../tile-programming/compiler-contract.md` Tier-B Phase 0 gate. |
| `v_cvt_off_f32_i4` (int4 → f32 offset converter) | **all** (gfx942 / gfx950 / gfx1100 / gfx1201) | ISA-present on **every** target — `v_cvt_off_f32_i4_e32` assembles on all four. A `hasattr(rocdl, 'cvt_off_f32_i4') == False` is therefore a **binding/wrapper gap, not a silicon ceiling**: the fast path stays reachable via inline asm or the raw intrinsic. Do not record it as a hardware `does-not-lower`. |
| **packed f32 (`v_pk_*`) cannot be placed in an MFMA's co-execution window at all**, yet it is the cheapest form per element (one issue slot retires two) | gfx950 | The two halves are not in tension — they apply to *different* work, so packed-vs-scalar is a **per-instruction** decision that depends on whether that instruction won a window slot: work you expect to **cover** should be scalar, work you expect to leave **exposed** should stay packed. Consequence: a blanket "scalarize every packed op in a block containing an MFMA" also splits the exposed remainder, which is a straight loss; and a packed op cannot follow an MFMA back to back, so the *first* uncovered packed op in a region pays a hazard on top of being exposed. |
| The register file sits behind **both** issue ports, so a **3-source VALU** (`v_fma`, `v_maximum3`) and an arriving **LDS return** contend for read/write ports in the same cycle | CDNA3/4 | Issuing in the same cycle is necessary for two instructions to overlap, **not sufficient**. Where a region's 3-source work coincides with the other wave's `ds_read` burst, the pairing the co-execution budget assumed does not materialize. Two independent fixes: remove the third source (fold a scale into an operand upstream so an `fma` becomes a `sub`), or move the burst's **phase** relative to the region head. Count 3-source VALU in the region body to size the first; the second is a phase effect, so sweep it — it is not monotone and bisection misleads. |

Confirm any such instruction with the stock assembler before building a lever around it
(`echo '<insn>' | llvm-mc -mcpu=<arch>`).

## LDS instruction behavior (gfx950)

- `ds_read_b128`: conflict-free steady interval **16 cycles/SIMD** (full CU).
- `ds_read_b64`: ~8-cycle steady interval.
- `ds_read_tr16_b64` / `ds_read_tr8_b64`: MFMA operand transpose paths (gfx950 only).
- `ds_write` may stall ~400 cyc when contending with `buffer_load_to_shared`.
- `GLOBAL_LOAD_LDS` / `buffer_load_to_shared`: up to 128-bit per lane on CDNA4.
- `ds_read_tr` latency tiers: **16 / 32 / 64** cycles — profile before assuming b128 timing.

**Steady state is not the whole model: there is a transient queue in front of it.** Each SIMD pair
has an LDS request FIFO of **8 entries** (`hw_constants.json`
`arch.<gfx>.lds_sp_fifo_entries_per_simd_pair`, a value the data layer has carried with no prose
attached). The consequences a cycle-per-op model misses:

- The first ~8 `ds_read`s of a burst are effectively free **relative to the steady interval**; the
  9th has to wait for the queue to drain, which shows up in a trace as a distinct gap rather than a
  gradual slope; from roughly the 10th the 16-cycle steady rate takes over.
- **`ds_write` shares the same queue.** One slow write at the head can block the reads behind it,
  which is the mechanism under the `ds_write` stall row above.
- So **latency is bought with prefetch, throughput is bought with interleaving** — and confusing them
  is how a kernel ends up issuing *more* LDS traffic to hide latency and manufacturing back-pressure
  instead. Interleaving non-LDS work inside a cluster lengthens the free-burst window; adding LDS
  loads does not.

**Status: registered, not calibrated here.** The 8-entry depth is a machine-readable constant; the
"first 8 free, 9th pays" reading of it is a model, and the per-arch cycle cost of that drain has not
been measured in this repo. Treat a burst longer than 8 as *a thing to measure* (count `ds_read` per
cluster in `asm_loop_audit.py --opcodes`, then A/B an interleave), not as a priced lever. The
placement judgement that follows from it is
`../tile-programming/instruction-scheduling.md ## How long an LDS burst may be before placement starts to matter — a countable precheck`,
which carries the per-cluster count and the same not-measured status. **That page ships only in the
packs whose DSL has an instruction-placement surface (Gluon, plain Triton).** Where it is absent the
count above is still the precheck — count the emitted LDS ops per cluster — but this pack has no
placement lever to route it to, so the finding belongs in the round record, not in a restructure.

## Cache modifiers — REGISTERED GAP, semantics not established here

`missing_target_support` / `missing_lowering_behavior` under the missing-doc protocol (`../method/triage.md`). Recorded
because the gap is large and quiet: production AMD kernels set a per-access cache modifier on the
**majority** of their loads and stores (`.cs`, `.cg`, `.wt`, `.ca` all appear at scale; `.nt` does
not), and this repo has no statement of what each one does to the L1 / L2 / memory-side-LLC path on
CDNA3 or CDNA4.

What IS settled, and is enough for most rounds, is the **polarity** question — a data-reuse argument
rather than an ISA one, and it already has homes: `../workloads/gemm.md` for the two-stream case (an
activation stream and a cache-resident re-read weight want opposite treatment) and
`../workloads/attention.md` for the rule that a bypass is decided by whether the buffer has a **near
consumer**, not by whether the store is the kernel's last write.

What is NOT settled: the per-modifier semantics. Until it is, do not infer a modifier's effect from
its name, and do not carry one over from another kernel because it looked right there:

- treat each modifier as an **A/B with an oracle**, one access site at a time;
- read the emitted instruction to confirm the modifier survived lowering — it is an ISA suffix, and a
  DSL-level argument that does not lower is a silent no-op;
- `hw_sources.sh xml <arch>` and, on gfx950, `gfx950_isa.py` carry the authoritative bit-field
  definitions. That is the tier to escalate to, and a finding belongs back in this section.

## Async global → LDS

| Arch | Width / op | Notes |
| --- | --- | --- |
| gfx950 | **16 B/op** (`dwordx4` class) | `buffer_load_to_shared`; `dwordx3`/`dwordx4` + `lds` are gfx950-only |
| gfx942 (downgrade) | ~4 B/op typical | narrower than CDNA4; only `buffer_load_dword ... lds` assembles |

## gfx1250 (CDNA5 / MI450)

Wave **32**, **WMMA+TDM** — not binary-compatible with gfx950 MFMA or RDNA gfx120*.
**gfx1250 is CDNA5 (MI450 data-center), a separate ISA fork from RDNA4 (gfx1201,
R9700 / RX9070 XT).** RDNA4 client has **no** TDM and **no** named-barrier warpgroups —
it is occupancy-first WMMA, not an asynchrony-first target; do not infer gfx1250
behavior from RDNA4 or vice versa.
See `rdna-fork.md` for client RDNA; gfx1250 WMMA+TDM in each skill's matrix reference.

## RDNA3 vs RDNA4 — ISA deltas

Client RDNA is wave32 WMMA (not CDNA MFMA). This mirrors the CDNA3↔4 table above for
the RDNA fork; API mapping + code anchors live in `rdna-fork.md`. RDNA4 ISA PDF:
[RDNA4](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf).

| Category | gfx11* (RDNA3) | gfx120* (RDNA4) |
| --- | --- | --- |
| Wave | wave32 (client); `_w64` ISA option | wave32; VOPD dual-issue wave32-only |
| WMMA core | 16×16×16 f16/bf16/i8/**i4** | same + **16×16×16 FP8/BF8** (`V_WMMA_F32_16X16X16_FP8_*`) |
| WMMA ABI | **v16-operand**: A/B = **8 VGPR**, replicated across lane halves | **v8-operand**: A/B = **4 VGPR**, no replication (different lane map) |
| WMMA rate (16×16×16 f16) | 32 cyc — 1024 FLOP/WGP/clk | **16 cyc** — 2048 FLOP/WGP/clk |
| Accumulator (D) lane map | row 0 → `v0{0-15}`, row **1** → `v0{16-31}` | row 0 → `v0{0-15}`, row **8** → `v0{16-31}` |
| VALU co-execution | **none** | **none** (see §Matrix/VALU co-execution) |
| Block scale / `mfma_scaled` / OCP FP4 | **none** | **none** — dequant+WMMA or IU4 WMMA |
| Structured sparse | **none** | **SWMMAC** 4:2 |
| GMEM transpose | buffer/global_load + `S_WAITCNT` | **`GLOBAL_LOAD_TR_B64/B128`** transpose loads |
| Fine-grained waitcnt | combined `S_WAITCNT` | split `s_wait_loadcnt` / `s_wait_dscnt` / … (finer) |
| Cache scope / temporal hint | basic | extended scope / temporal hints |
| Software prefetch | — | **`s_prefetch_data`** (ISA addition) |
| Dynamic VGPR | static ≤256/wave | **not an RDNA4 feature** — see below |
| GWS (`DS_GWS`) | present | **removed** |
| LDS | **128 KiB/WGP**; **≤64 KiB/WG** | same |
| Async cp / TDM | **ISA: none** | **ISA: none** (TDM is gfx1250/CDNA5, not RDNA4) |

**Dynamic VGPR is not an RDNA4 lever.** `S_ALLOC_VGPR` is rejected as `invalid
instruction` by the ROCm 7.1 / LLVM 20 assembler for **both** gfx1200 and gfx1201, and
no `+dynamic-vgpr` subtarget feature exists there — it belongs to gfx1250 (CDNA5).
Plan RDNA4 occupancy on the static ≤256 VGPR/wave model only.

The value/exposure of the remaining RDNA4-new levers (SWMMAC, `s_prefetch_data`,
fine-grained waitcnt) via the DSL/compiler stack is `unknown-needs-probe`. WMMA wave32,
no `mfma_scaled`, hand LDS ping-pong: `rdna-fork.md`.

> Provenance — WMMA ABI / rate / accumulator rows: AMD Matrix Instruction Calculator
> (`-a rdna3|rdna4 -i v_wmma_f32_16x16x16_f16 -d` for GPR counts + cycles,
> `-R -A` / `-R -D -w 32` for the lane maps). `S_ALLOC_VGPR`, `GLOBAL_LOAD_TR_*`,
> `s_prefetch_data`, `DS_GWS`, `V_SWMMAC_*` rows: `llvm-mc -mcpu={gfx1100,gfx1201}`
> acceptance probe on ROCm 7.1 / LLVM 20.

## Source provenance — when a distilled fact here is not enough (→ L4 machine-readable ISA)

Every fact on this page is a *distilled* value (probe-then-trust it first). When you need a
fact this page does NOT carry, or must confirm a claim before it gates a correctness/layout
decision, drop to the machine-readable primary sources instead of guessing.

**On gfx950 (CDNA4) do not fetch or render anything — the whole ISA is already in the repo.**
All 13 chapters of the CDNA4 ISA are transcribed into three committed databases in GEAK's shared
hardware data dir `perf_knowledge/hardware/data/` (`gfx950-mfma-layout.json`,
`gfx950-encoding.json.gz`, `gfx950-isa-facts.json.gz`), served by
**`kernel_workflow/scripts/kernel_tools/gfx950_isa.py`** (this pack's `scripts/gfx950_isa.py` is a
shim). It is offline, needs no network and no PDF render, and every
record carries its own page + printed quote. Start with `search`, which queries all three at once:

```
gfx950_isa.py search <anything>            # start here: one query across all three DBs
gfx950_isa.py facts|layout|locate <instr>  # MFMA operand layout: lane/VGPR/byte for A,B,C,D
gfx950_isa.py encoding <instr>             # opcode + operand fields + the ch.12 pseudo-code
gfx950_isa.py errata                       # known errors in the source document — read before
                                           #   hand-assembling or trusting a printed opcode
```

For any OTHER architecture, the three-tier fetch protocol still applies: **`primary-sources.md`**
(same dir) driven by **`kernel_workflow/scripts/kernel_tools/hw_sources.sh`**. Cache-only, nothing committed; on fetch
failure it records a scoped tool-gap and you fall back to the distilled value here.

| You need … | Tier / source | command | cite the result as |
| --- | --- | --- | --- |
| **anything at all on gfx950** — layout, encoding, semantics, registers, wait states, LDS, memory | **committed DB** (offline, no fetch) | `gfx950_isa.py search <term>` | `CDNA4 ISA p<NN>` (the tool prints the page) |
| instruction **encoding** (opcode / bit-fields / operand slots) — e.g. `ds_read_b64_tr_b16`, `v_mfma_scale_*` | 1 · ISA **XML** — but on gfx950 prefer `gfx950_isa.py encoding`, which adds the chapter-12 pseudo-code the XML does not carry | `hw_sources.sh xml <arch>` / `decode <arch> <bytes>` | `<arch> ISA XML <opcode>` |
| operand **data layout** (which VGPR/lane/byte holds A[m][k], B, C/D, scale) on **CDNA1-3 / RDNA3-4** | 2 · **Matrix Calculator** (generates the table) | `hw_sources.sh layout <arch> <instr> -A -B -C -D` | `MI-calc <arch> <instr>` |
| microarch concepts / SKU peaks **not** in the ISA document, any arch | 3 · whitepaper **PDF** (render, don't scrape) | `hw_sources.sh pdf <doc> <p0> <p1>` | `<doc> p<NN>` |
| whether an instruction **exists on a target at all** | 0 · stock **assembler** (cheapest, always available) | `echo '<insn>' \| llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=<arch>` | `llvm-mc <arch> accept/reject` |

> Two traps the databases record and a bare XML/PDF lookup does not. **(1)** The ISA prints the
> VOP3 opcode of 85 VOP1-promoted instructions 64 too high, in *both* §12.11 and the chapter-13
> roster — never hand-assemble from the printed column; `gfx950_isa.py errata` has the resolution.
> **(2)** The four `DS_READ_*_TR_*` transpose loads — the ones that feed MFMA — have **no**
> pseudo-code box anywhere in the document, so their lane→element mapping is not recoverable from
> it. `encoding` reports that explicitly instead of returning a confident-looking partial answer.

The `## Default MFMA shapes`, `## ISA vs DSL lowering`, `## LDS instruction behavior`, and
`## Async global → LDS` sections above are the distilled outputs of exactly this protocol;
when one is stale or missing your target `(arch, dtype, atom)`, re-derive it via the tier that
owns it and write the new fact back here **with its provenance line** (so the next agent trusts
it and never re-runs the tool). Full protocol + the source index: `primary-sources.md`.

## On Gluon

- **gfx950 async→LDS entry point:** `gl.amd.cdna4.async_copy.buffer_load_to_shared` (128-bit or
  32-bit per thread) + `commit_group` / `wait_group` — the producer of the authored LDS ring
  (`../tile-programming/pipeline.md`); build its destination swizzled first.
- **gfx942 downgrade async→LDS entry point:** reachable via the same `cdna4.async_copy` entry — the
  `cdna3` namespace has no async submodule — at 32-bit only, destination `order=[1,0]`. Verify the
  generated ISA; `num_stages` does not pipeline anything on the Gluon path in 3.8.0.
- **Full CDNA3 capability page** (cannot-select MFMA, fp8 `e4m3fnuz`, 64 KiB LDS,
  cross-arch gate): `cdna3-gfx942.md`.
- RDNA4-new lever exposure through the Gluon/compiler stack:
  `capability-matrix.md ## Memory and scheduling matrix`.
