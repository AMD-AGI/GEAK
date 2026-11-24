# ISA mechanisms (CDNA4 gfx950, CDNA3 gfx942 downgrade)

Hardware instruction facts only; Gluon API availability is in `capability-matrix.md`. Any ISA fact
not here: `python3 kernel_workflow/scripts/kernel_tools/amd_isa.py --arch gfx950 search "<words>"`
(then `list`, `facts <V_MFMA_...>`, `layout`, `lds`, `hazard`, `waits`, `waitcnt`, `limits`); does
an instruction exist on a target: `echo '<insn>' | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx950`.
Source errata and trust levels: `primary-sources.md`. RDNA / gfx1250 are out of scope.

## CDNA3 vs CDNA4 — ISA deltas

| Category | gfx950 (CDNA4) | gfx942 (CDNA3, downgrade) |
| --- | --- | --- |
| LDS / CU | **160 KiB**, 64 banks, 18-bit M0 offset | 64 KiB, 32 banks, 16-bit M0 |
| MFMA f16/bf16 | 16x16x16, 32x32x8 **+ 16x16x32, 32x32x16** | 16x16x16, 32x32x8 |
| MFMA int8 | 16x16x32, 32x32x16 **+ 16x16x64, 32x32x32** | 16x16x32, 32x32x16 |
| MFMA fp8 regular | 16x16x32, 32x32x16 | 16x16x32, 32x32x16 (FNUZ) |
| MFMA f8f6f4 / scaled | `V_MFMA_SCALE_F32_16X16X128_F8F6F4`, `..._32X32X64_F8F6F4`; scale **E8M0**, K-block 32 | **none** |
| Sparse (SMFMAC) | K doubles: `16x16x64` / `32x32x32` (f16/bf16) | up to `16x16x32` / `32x32x16` |
| LDS -> MFMA transpose | `DS_READ_B64_TR_B16/B8/B4`, `DS_READ_B96_TR_B6` | **none** |
| GMEM -> LDS | `buffer/global_load_lds` adds **dwordx3 / dwordx4** (12 / 16 B/lane) to the dword and byte/short forms (gfx950 encoding table) | dword (4 B/lane) and byte/short forms only, `buffer_load_* ... lds` and `global_load_lds_*` (CDNA3 encoding table) |
| Cross-lane swap | `V_PERMLANE16_SWAP_B32`, `V_PERMLANE32_SWAP_B32` | **none** |
| Convert helpers | `V_CVT_SCALEF32_*` (MX dequant), **`V_CVT_PK_BF16_F32`** | **none** |
| XF32 MFMA | removed | present |
| GWS (`DS_GWS`) | present (assembles on both; removed only on RDNA4) | present |

Existence rows: `llvm-mc` acceptance probe, ROCm 7.1 / LLVM 20.

## Default MFMA shapes (ISA directory)

| dtype | gfx950 | gfx942 | Gluon path |
| --- | --- | --- | --- |
| fp16/bf16 | **16x16x32**, 32x32x16 (also the K-halved shapes) | 16x16x16, 32x32x8 | `cdna4.mfma` / `cdna3.mfma` |
| fp8 | 16x16x32, 32x32x16 regular; 16x16x128 / 32x32x64 scaled | 16x16x32, 32x32x16 | gfx950: `cdna4.mfma` only at 16x16x32 / 32x32x16; K=128/64 via `mfma_scaled` (scales `None` allowed). gfx942: Gluon fp8 MFMA on FNUZ operands with `version=3` (compile-only; numerics not GPU-checked, `capability-matrix.md`); OCP operands fail to compile (`version=3`); wrong numbers with `version=4` are inferred, not run on gfx942 |
| int8 | see delta table | 16x16x32, 32x32x16 | int32 accumulator |
| fp4 / mxfp4 / fp6 | 16x16x128, 32x32x64 scaled only | none | `cdna4.mfma_scaled` |

The Gluon registry is keyed per `(version, M, N, K)`; details in `capability-matrix.md ## Matrix
operation matrix`. Plain `tl.dot(fp8)` on gfx950 lowers to scale-less `tt.dot_scaled` ->
`v_mfma_f32_32x32x64_f8f6f4`.

## MFMA throughput model (planning cycles)

One `PASS` = 4 cycles; issue interval = `4 x passes` = the earliest cycle the SIMD can issue the
next MFMA of that shape (back-to-back spacing, not completion latency). **Cycles are per instruction
shape: read them with `amd_isa.py --arch gfx950 facts <V_MFMA_...>`** (e.g. `16X16X32_BF16` 16,
`32X32X16_BF16` 32, `16X16X32_FP8_FP8` 16). A pass count read for one dtype does not transfer to
another.

| Path | Instruction | ~cyc / MFMA | Note |
| --- | --- | --- | --- |
| f16/bf16 | `v_mfma_f32_16x16x32_*` | 16 | ~4 MFMA per 64-cyc mem op |
| f16/bf16 | `v_mfma_f32_32x32x16_*` | 32 | the shape the co-execution window is measured on |
| f8f6f4 (scaled or not) | `v_mfma_*_f32_16x16x128_f8f6f4` | 16 if **both** operands f4, else 32 | measured on gfx950 (s_memtime microbench and ATT, fp8 and fp4 operands); the ISA DB prints no cycles |
| f8f6f4 (scaled or not) | `v_mfma_*_f32_32x32x64_f8f6f4` | 32 if both f4, else 64 | same measurement |
| VALU wave64 | `v_add/mul/fma_f32` ... | ~4 | per wave64 op, for PMC-free VALU-busy estimates |

## Matrix/VALU co-execution — the CDNA-only overlap budget

Two numbers per matrix op: the **interval** (ISA-derived) and the **window** (how much of it another
pipe can issue into; measured, not documented).

| Arch | Matrix op | Interval | Co-issuable window |
| --- | --- | --- | --- |
| gfx950 | `v_mfma_f32_32x32x16_f16` / `_bf16` | 32 | **~24 of 32** - a head **read phase of 8 cyc** takes no co-issue; 6 VALU / 3 TRANS fit |
| gfx950 | `v_mfma_f32_16x16x32_f16` / `_bf16` | 16 | head 8; 2 VALU fit |
| gfx950 | `v_mfma_f32_32x32x64_f8f6f4`, fp8 operands | 64 | head 12; 12-13 VALU / 6 TRANS fit |
| gfx950 | `v_mfma_f32_16x16x128_f8f6f4`, fp8 operands | 32 | head 12; 5 VALU / 2 TRANS fit |
| gfx950 | f8f6f4, fp4 operands (both): `32x32x64` / `16x16x128` | 32 / 16 | head 8; 6 VALU / 3 TRANS and 2 VALU / 1 TRANS fit |
| gfx950 | `v_mfma_scale_*` fp8: `32x32x64` / `16x16x128` | 64 / 32 | head 16; 12 VALU / 6 TRANS and 4 VALU / 2 TRANS fit |
| gfx950 | `v_mfma_scale_f32_16x16x128_f8f6f4` fp4 | 16 | head 12; 1 VALU / 0 TRANS fit |
| gfx942 | `v_mfma_f32_16x16x16_f16` | 16 | **12 of 16** (Matrix Instruction Calculator) |
| gfx942 | `v_mfma_f32_32x32x8_f16` | 32 | **28 of 32** (Matrix Instruction Calculator) |

**Overrides KB** (gfx950, Triton 3.8.0): `perf_knowledge/languages/triton_amd/isa_verify.md` ("fp16/bf16 with
`matrix_instr_nonkdim=16` → `v_mfma_f32_16x16x16`", "`v_mfma_f32_32x32x8`") holds for gfx942 only; gfx950 emits
`v_mfma_f32_16x16x32_{f16,bf16}` / `32x32x16_*` (the shapes in the table above; `amd_isa.py --arch gfx950 list`).

gfx950 rows: measured with 1 wave/SIMD and independent accumulators, by s_memtime cycle counts (Triton 3.8.0
image) and ATT issue times (rocprofiler-sdk 1.3.5); VALU counts are for 4-byte encodings (see the next table).

- **The window is per shape *and* operand format, not interval minus a constant** (gfx942's 12/16 is not
  16 - 8). The head tracks how many operand registers the MFMA reads: bf16/f16 and fp4-operand shapes block
  VALU for 2 passes, fp8-operand f8f6f4 for 3, and `v_mfma_scale` adds one pass for the scale operands.
  A neighbour shape's or format's window is not evidence for another. Mixed-shape regions were not traced;
  taking the shortest member's window is the conservative budget.
- Scaled f8f6f4: same mnemonic, several budgets (all-f4 short, any f8 double, scaled head longer);
  derive from the formats the region emits.

### Calibrating the window for a new shape

Get the interval from the ISA (`facts`); take an ATT trace of a loop containing the shape and read
where non-matrix ops issue after each MFMA (the issue-free head span is the read phase); confirm by
filling a region with a known-cost class until cadence degrades. Uncalibrated = missing knowledge
about this shape, not absent hardware: fall back to levers that reduce demand.

### What fits in the window, and at what price (gfx950, 32-cyc shape)

| Class | Issue cost (wave64) | Fits per ~24-cyc window |
| --- | --- | --- |
| VALU, 4-byte encodings (`v_add_f32_e32`, `v_mov`, `v_fmac_f32_e32`) | 4 cyc | 6 |
| VALU, 8-byte VOP3/VOP3P encodings (`v_fma_f32`, `v_maximum3`, `v_cvt_pk_bf16_f32`, `v_pk_mul_f16`, `v_add_f32_e64`) | 4 cyc, plus one lost issue slot about every fifth instruction (instruction size, not operand count) | 5 |
| TRANS (`v_exp_f32`, `v_rcp`, ...) | 8 cyc | 3 - TRANS does issue inside the MFMA shadow |
| cross-lane (`v_permlane16_swap` / `v_permlane32_swap`) | 8 cyc (same as TRANS) | 3 |
| packed f32 (`v_pk_fma/add/mul_f32`) | 4 cyc per 2 elements packed; 8 cyc once split | **0** - the first one after an MFMA issues only once the MFMA has finished plus 3 passes; later ones at the VALU rate. Packed f16/bf16 (`v_pk_mul_f16`) co-issues like scalar VALU |

Measured on gfx950 (s_memtime microbench, Triton 3.8.0 image; ATT issue times, rocprofiler-sdk 1.3.5), both
methods agreeing. Earlier "20 cyc / 1 fit" figures for `v_permlane*_swap` were not reproduced. LLVM's
`isNeverCoissue` (true for TRANS and packed f32 on gfx940/gfx950) matches the hardware for packed f32 but not for
TRANS.

**TRANS rate (gfx950, ATT and throughput microbench):** a TRANS op (`v_exp`, `v_rcp`, `v_log`, `v_sqrt`)
issues once per 8 cycles per SIMD. That is half a single wave's VALU rate, and a quarter of the SIMD's VALU
throughput once two or more waves share the SIMD (two waves co-issue plain VALU; TRANS does not speed up with a
second wave). TRANS never overlaps other VALU, even across waves. Both "half" and "quarter" are right, at
1 and at >= 2 waves/SIMD respectively (`hw_constants.json` gfx950 `exp_rate_vs_valu` 0.25 is the >= 2-wave
figure); budget against the waves/SIMD you run. The ISA document publishes no rate (`amd_isa.py` has none).

```text
per region:    capacity = (MFMA in region) x window ; demand = sum of class costs
               exposed  = max(0, demand - capacity)        # count packed ops as exposed
per loop body: ceiling  = mfma_cycles / (mfma_cycles + sum of exposed)
```

`ceiling = 1.0` => any shortfall is scheduling/placement; `< 1` => within this model the work has to
shrink or move, since no ordering recovers it. Method: `../tile-programming/llir-codesign.md`.

## Instruction-rate facts that flip a lever's sign (CDNA3/4)

| Fact | Arch | Lever consequence |
| --- | --- | --- |
| `v_exp_f32` is a slower-than-VALU transcendental on the **same** VALU issue path (no separate MUFU port; on gfx950 it does not overlap other VALU even across waves, ATT-measured) | CDNA3/4 | Polynomial `exp2` (roundeven + FMAs + bit-assembled `2^n`) trades TRANS issue for FMAs: it is most likely to pay where the loop is exp-issue-bound, and on total-VALU-bound loops the extra FMAs can lose; a cheaper approximation (fewer FMAs, if the tolerance allows) shifts that balance. One recipe losing does not close softmax arithmetic - measure per region |
| `v_cvt_pk_bf16_f32` (2xf32 -> 2xbf16) | **CDNA4-only** | On gfx942 bf16 packing is the `and/or/shift` truncation sequence (`llvm-mc -mcpu=gfx942` rejects the cvt) |
| bf16 MFMA = `16x16x16`, `32x32x8` (no `16x16x32` / `32x32x16`) | gfx942 | K-doubled bf16 is CDNA4-only; `instr_shape=[16, 16, 32]` bf16 on gfx942 fails the pass manager (compile-verified 3.8.0) |
| `v_cvt_off_f32_i4` assembles on gfx942 and gfx950 | all CDNA | `hasattr(rocdl, 'cvt_off_f32_i4') == False` is a binding gap, not silicon: reachable via inline asm |
| packed **f32** (`v_pk_fma/add/mul_f32`) did not co-issue in an MFMA window (ATT- and microbench-measured), yet is cheapest per element; packed f16/bf16 co-issues like scalar VALU | gfx950 | Per-instruction trade-off: scalar form for work expected to be **covered**, packed f32 for what stays **exposed** (a fold measured both ways on different kernels - A/B). The first packed f32 op after an MFMA waits for the MFMA to finish plus 3 passes |
| Inside an MFMA shadow, VALU that reads three operands (`v_fma`, `v_maximum3`, and `v_fmac`, which reads its destination) is slowed while another wave on the same SIMD moves LDS data (`ds_write` more than `ds_read` returns) | gfx950 (ATT-measured) | Two-operand VOP3 ops and VALU-only code (no MFMA) showed no effect. Same-cycle issue is necessary, not sufficient, for overlap. Dropping the third operand (fold a scale upstream: `fma` -> `sub`/`add`) worked; `fma` -> `fmac` did not. Shifting the other wave's LDS burst phase (`s_nop` at stage head) was observed on one kernel and not swept here - treat it as a sweep, non-monotone |
| `s_nop` before DPP / `permlane` / cross-lane ops is an ISA-mandated wait state | CDNA3/4 | Deleting it in hand asm is a correctness bug; counts: `amd_isa.py --arch gfx950 waits` / `hazard` |

## LDS instruction behavior (gfx950)

- `ds_read_b128` conflict-free steady interval **16 cyc/SIMD with all four SIMDs reading** (2-way 32, 4-way
  64); `ds_read_b64` 8 (2-way 16, 4-way 32). The absolute interval depends on load and on burst length (four
  SIMDs in short 16-read bursts measured 8 / 16 / 32 for conflict-free / 2- / 4-way; long streams 12-17
  conflict-free, ATT gfx950): one wave per CU saw a
  conflict-free b128 every 8 cycles and hid 2-way conflicts entirely (it queues about 16 reads before issue
  stalls); SIMD pairs {0,1} and {2,3} share LDS read throughput, so a pair reading alone looks like four SIMDs.
  The diagnostic is the **ratio** to a conflict-free baseline under the same load (double / quadruple), confirmed with
  `SQ_LDS_BANK_CONFLICT` (gfx950, s_memtime microbench and ATT).
- `ds_read_b64_tr_b16` / `_tr_b8`: MFMA operand transpose (gfx950 only). At the same addressing they time
  exactly like `ds_read_b64` (interval and latency; measured, no separate tr tiers) - a slower tr timing
  comes from the access pattern, so read their conflicts the same way.
- **A `ds_write` completes late while `buffer_load_to_shared` data is landing** (ATT, gfx950 / Triton
  3.8.0 / rocprofiler-sdk 1.3.5: about 300-500 cycles to its `lgkmcnt(0)` vs about 60 without DMA; the
  `ds_write` issues at once, so independent work before the wait hides it).
- **No 8-entry LDS request limit is visible on gfx950** (ATT, rocprofiler-sdk 1.3.5): one wave issued 16
  `ds_read_b128` (32 `ds_read_b32`) back to back with no issue stall, stalling first at the 17th; two waves on
  paired SIMDs got about 10-14 each; four SIMDs streaming stall almost at once, from bandwidth. Outstanding
  LDS-DMA (`buffer_load_to_shared`) did not block `ds_read` issue - its cost shows when its data lands (the
  `ds_write` bullet above). `hw_constants.json arch.gfx950.lds_sp_fifo_entries_per_simd_pair` is `null`
  for this reason (no FIFO depth recorded); `ds_read_b128_queued_per_wave_before_stall` (16) records the
  one-wave figure. Latency is bought with prefetch,
  throughput with interleaving; count `ds_read` per cluster (`asm_loop_audit.py --opcodes 12`) and A/B an
  interleave (`../tile-programming/instruction-scheduling.md`).

## Cache modifiers — REGISTERED GAP: residency semantics not established here

The emitted bits are established (compiled on 3.8.0, gfx942 and gfx950: `.cv` / `.wt` -> `sc0 sc1`; `.cg` /
`.cs` -> global `nt`, buffer `sc0 nt`) and so is GPU-checked cross-agent coherence for `.cv` / `.wt`
(`../workloads/collective.md ## Publish and consume`, `../gluon/inline-asm-reference.md`). What is not
established is the per-modifier L1/L2/MALL residency behaviour on CDNA3/4.
A/B one access site at a time and confirm the suffix survived lowering in the ISA; bit-field
definitions: `amd_isa.py --arch gfx950 search <modifier>`.

## Async global → LDS

gfx950: 16 B/lane (`dwordx4`) or 4 B/lane; gfx942: 4 B/lane only. Gluon entry and legality:
`capability-matrix.md ## Direct-to-LDS granularity (per arch)`,
`../gluon/memory-reference.md`.
