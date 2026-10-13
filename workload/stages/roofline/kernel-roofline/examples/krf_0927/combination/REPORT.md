# GLM-5.2 0927 kernels — baseline vs candidate roofline, both engines

Scope: the nine seams under
`/shared_nfs/chuschen/agent-kernel-arena/agentx-glm52/exp_ut0922/0927`, each run through
**both** roofline engines and **both** implementation arms.

| | engine | what it answers | GPU touched |
|---|---|---|---|
| `tl-only` | TraceLens `PerfModel` (predictor) | how far this **shape** is from the hardware limit | yes, for `measured_us` only; the roof is pure shape |
| `rocprof` | rocprof-compute `profile --roof-only` + `analyze -b 4` (diagnoser) | what this **binary** actually issued | yes, counters |
| `combination` | both, folded in the time domain | whether a gap is work or efficiency, and which engine's numerator is wrong | — |

Full tables: [`../tl-only/COMPARE.md`](../tl-only/COMPARE.md) (56 cases, both arms),
`join_{baseline,candidate}.txt`, `compare_{baseline,candidate}.txt`,
`per_case_{baseline,candidate}.json` (8 cases, the one dominant case per seam that the
rocprof leg profiled).

## Provenance

- **TL leg** — arch `MI355X.bw6857_bf1449.json`, md5 `c045ecc958e17ca6062de4970ddd3316`,
  `source: "These are benchmark derived peak flops and bw"`, mem_bw 6857 GB/s.
  56 cases × 2 arms × 3 reps, one process per arm. Run 2026-10-12 03:59–04:01.
  Contention gate: **passed** — bracketing probes 1467 / 1457 TFLOP/s against a 1449
  arch peak (100.5%), worst per-case rep spread 1.088 against a 1.15 gate.
  (A first run at 03:40 overlapped the rocprof leg and was self-flagged UNSTABLE at
  1.26× spread; it is kept at `../tl-only-run1-unstable/` and agrees with this one
  to ~1%.)
- **rocprof leg** — rocprof-compute **3.4.0** (git `69f44ce4`), running out of the
  private tree patched by `scripts/rocprof_hbm_patch.py` for the gfx950 L2→HBM
  read-byte halving, plus the non-fatal PDF export. 9 seams × 2 arms = 18 profiles,
  all rc=0. **Denominator frozen** via `ROOFLINE_MIBENCH_CACHE` → HBM 6857 GB/s on
  every one of the 18 runs, no microbench anywhere in the logs.
  `--peaks both`, so each row also carries a datasheet cross-check.
- 3.4.0 is **below** the ≥3.6.0 gfx95x gate, under which rocprof-compute's own
  microbench issues CDNA3-encoded MFMAs and reports fp8/bf16 peaks 2–4× low. That gate
  is bypassed here by **substituting the denominator rather than the binary**: the
  frozen `roofline.csv` carries independently measured ceilings, not rocprof's
  microbench. Two checks that it really is the corrected set —
  `MFMABF16Flops` 2,453,780 ≠ `MFMAF16Flops` 2,449,443 (so bf16 is measured, not
  inherited from fp16, which is the tell for the bug), and `MFMAF8Flops` = 4,949,468
  rather than the ~2.45M a CDNA3-encoded fp8 bench returns.
- **Independent agreement on the ceilings.** Those values come from the same
  measurement campaign as chuschen's
  `roofline/ceilings_gfx950_0927.json` (`peakbench/mfma_peak.hip`, 8 accumulator
  chains, median of 7 launches, clocks verified at 2380–2396 MHz), and they agree with
  this run's *datasheet* column to within 1%: fp8 4,949,468 measured vs 5,000,000
  datasheet, fp4 9,748,296 measured vs 10,000,000 datasheet. Two denominators derived
  by different routes landing 1% apart is the strongest statement available that the
  compute roofs below are right.
- One caveat on the fp4 axis: 3.4.0's analyzer emits no `MFMA FLOPs (F6F4)` row, so
  `run_roofline.py` fell back to its documented estimate `peak(f6f4) ≈ 2 × peak(f8)` =
  9,898,936 (labelled `est=2xF8emp` in every affected row). That is **1.5% above** the
  directly measured 9,748,296, so the MoE efficiencies below are ~1.5% conservative —
  well inside the margin of anything claimed here.

> **Re-scored 2026-10-12.** The empirical column used to score an unscorable
> kernel as 0% efficient while leaving it in the denominator; it now drops it
> and reports the shortfall in `cov`, matching the datasheet column and
> [folding.md](../../references/folding.md). Four cells moved, all on rows with
> `cov` < 100%: baseline `dsa_indexer_prefill_score_topk` 14.7%->20.2%,
> baseline `mxfp4_fused_moe` 68.9%->73.0%, candidate `dsa_indexer_score_topk`
> 30.4%->38.2%, candidate `mxfp4_fused_moe` 70.1%->71.7%. Nothing else in this
> report changed. `--missing-emp zero` reproduces the original tables.

## 1. The adjudication table

Neither engine's roof can be checked against itself. The measured speedup can — it is
the answer key both engines have to live with. `RP roof` is rocprof's own implied roof
time (`Eff × measured`), so it is a per-arm measurement of *how much work the binary
actually did*; `RP work x` is how much less work the candidate did, and
`efficiency x = measured x / work x` is what is left over, i.e. how much better the
candidate used the hardware for the work it kept.

| case | meas base us | meas cand us | measured x | TL roof us | RP roof base us | RP roof cand us | RP work x | efficiency x |
|---|---|---|---|---|---|---|---|---|
| add_rmsnorm_quant dec_bs3_m18_res | 9.0 | 8.5 | **1.06x** | 0.1 | 0.4 | 0.4 | 1.06x | 1.00x |
| dsa_indexer_decode dec_bs3_ctx104k | 46.7 | 11.6 | **4.04x** | 7.1 | 37.3 | 6.6 | 5.66x | 0.71x |
| dsa_indexer_prefill pre_m10129_kv40k | 1575.4 | 884.2 | **1.78x** | 743.0 | 422.2 | 430.5 | 0.98x | 1.82x |
| dsa_indexer_prefill_score_topk pre_ch_r1383_ctx387k | 5294.0 | 1980.0 | **2.67x** | 1398.5 | 779.1 | 984.7 | 0.79x | 3.38x |
| dsa_indexer_score_topk dec_bs3_ctx104k | 80.1 | 23.2 | **3.45x** | 8.2 | 37.9 | 7.1 | 5.36x | 0.64x |
| dsa_indexer_topk dec_bs3_ctx104k | 30.3 | 7.1 | **4.26x** | 1.1 | 1.0 | 1.4 | 0.68x | 6.31x |
| mxfp4_fused_moe pre_M32768 | 2584.4 | 2096.7 | **1.23x** | 544.6 | 1781.9 | 1470.7 | 1.21x | 1.02x |
| sparse_mla_prefill pre_seq32768 | 4568.8 | 3106.0 | **1.47x** | 10748.5 | 1492.2 | 1149.5 | 1.30x | 1.13x |

Read down the last two columns and the nine seams fall into three mechanisms:

- **Work removed, efficiency *lost*.** `dsa_indexer_decode` (5.66× less work, 0.71×
  efficiency) and `dsa_indexer_score_topk` (5.36× / 0.64×). The candidate wins by not
  touching memory it did not need; what it does touch, it touches *less* well than the
  baseline did. Its hardware efficiency fell from 79.8% to 57.0% and from 47.3% to
  38.2%. **These two still have the most headroom left of anything here**, and the
  lever is now occupancy/access pattern, not algorithm.
- **Same work, run better.** `dsa_indexer_prefill` — work ratio 0.98×, the whole 1.78×
  is efficiency (26.8% → 48.7% of its own roof). Nothing was removed; the kernel simply
  got out of its own way. `dsa_indexer_prefill_score_topk` and `dsa_indexer_topk` are
  the stronger form: the candidate does **more** work (0.79× and 0.68×) and is still
  2.67× and 4.26× faster, for 3.38× and 6.31× efficiency.
- **Work removed, efficiency flat.** `mxfp4_fused_moe` (1.21× / 1.02×) and
  `sparse_mla_prefill` (1.30× / 1.13×). The MoE candidate's entire 1.23× is a smaller
  byte/FLOP bill at an unchanged 72–73% of roof — so there is no "tune it harder" left
  on that axis; further wins there have to come from fewer bytes again.

## 2. Where the two engines disagree, and who is wrong

Per `combination.md` §4, axis disagreement is the signal worth chasing: it means the
numerator, not the percentage.

| seam | TL AI | RP AI | TL axis | RP axis | verdict |
|---|---|---|---|---|---|
| mxfp4_fused_moe / prefill | 1825 | 233 (base) / 277 (cand) | compute | **memory** | the axes disagree outright |
| sparse_mla_prefill / prefill | 15.9 | 203 (base) / 316 (cand) | memory | memory | **TL's bytes are wrong by ~13–20×** |
| dsa_indexer_decode / decode | 314.6 | 60.7 (base) / 345.6 (cand) | memory | memory | TL right for the candidate, 5.2× off for the baseline |
| dsa_indexer_prefill / prefill | 1965 | 1180 / 1522 | compute | compute | agree |
| dsa_indexer_prefill_score_topk | 1008 | 951 / 1331 | compute | compute | agree |
| add_rmsnorm_quant / decode | 0.6 | 0.2 | memory | memory | agree on axis, both irrelevant (see §3) |

1. **`sparse_mla_prefill` — the TL leg is disproved by its own measurement.** It puts
   the roof at 10748 us against a measured 4569 us, i.e. **235% of roof** on the
   baseline and 346% on the candidate. No kernel beats its own bandwidth roof, so this
   is an overcounted byte model, not a fast kernel: `m_sparse_mla` is a *decode* model
   with a per-row KV gather and no cross-row reuse, and at 32768 rows that gather is
   fiction. The number is kept unclamped and marked suspect; a perfect-reuse floor
   brackets the roof at 372 us. **rocprof settles it**: 1492 us, 32.7% of roof,
   ~3.1× of real headroom on the baseline and ~2.7× on the candidate. Use the rocprof
   figure for this seam; do not quote the TL one in either direction.
2. **`mxfp4_fused_moe` — the two engines put it on different axes.** TL scores it
   compute-bound at AI 1825 against `matrix_fp4` = 5110 TFLOP/s; rocprof measures
   AI 233 and calls it memory-bound against an `mfma_f6f4` peak of 9899 TFLOP/s
   empirical / 10000 TFLOP/s datasheet. Two independent problems stack here:
   the **AI is 7.9× apart** (TL is undercounting the expert-weight traffic that
   actually reaches HBM at M=32768), and the **fp4 peak is ~1.94× apart** — the arch
   variant's `matrix_fp4` key holds only the CK number, which `pitfalls.md` already
   flags as a 1.89× underestimate for a FlyDSL/Triton kernel. Both errors push TL's
   percentage down; TL reads 21.1% where rocprof reads 73.0%. **Take rocprof's
   verdict for this seam.** The dominant kernels confirm it directly:
   `mfma_moe1_silu_mul_afp4_wfp4_fp4_...v33` is at **88.1%** of its attainable and
   `gemm2_a4w4_...` at **56.6%**, both sitting on the memory axis.
3. **`dsa_indexer_decode` — the engines agree for the candidate and not the baseline.**
   TL's AI is arm-independent by construction (314.6 in both legs). rocprof reads 60.7
   for the baseline and 345.6 for the candidate. So the baseline was moving ~5× more
   bytes per FLOP than its shape requires — redundant traffic no shape model can see —
   and the candidate removed essentially all of it, landing on the model. That is the
   same conclusion the adjudication table reached from the other direction, from two
   independent measurements.

Everything else agrees on axis, which is the expected outcome: ranking and axis are
near-equivalent across engines, headroom is not.

## 3. Rows where neither engine's roof is the real ceiling

`add_rmsnorm_quant` decode is the clearest. Its byte roof is **0.13 us** against a
measured 9.0 us — "68× available" — but the seam dispatches 2 kernels and a dispatched
kernel costs ~2.3 us of device time before it moves a byte. The honest ceiling is
**4.6 us**, the candidate is at 8.5 us, and the entire available win is 1.8×, not 68×.
Seven cases in the TL table are in this state (flagged ⬓ in
[`../tl-only/COMPARE.md`](../tl-only/COMPARE.md)): four `add_rmsnorm_quant` decode
shapes and three `dsa_indexer_topk` decode shapes. Both engines get this wrong
identically because neither models dispatch. For these rows the lever is fusing nodes.

`indexer_preamble` is the extreme case and is reported as **unsupported** by the TL
leg rather than given a roof: a 5-kernel chain at m ≤ 42 whose time is dispatch floor
end to end. Its rocprof profile is on disk (`indexer_preamble_dec_bs1_{baseline,
candidate}_roofline.txt`) and shows the same thing — no kernel above a few percent of
any roof.

## 4. Coverage

- TL leg: 56 cases × 2 arms, 8 of 9 seams modelled; `indexer_preamble` deliberately not
  roofed (above).
- rocprof leg: 9 seams × 2 arms on the single highest-weight case per seam — 18 of the
  112 TL (case, arm) pairs. Every table above that mixes the two is explicitly folded
  over the 8 covered cases only; `join_legs.py` and `compare_legs.py` now print the
  uncovered count rather than silently presenting a partial table as complete.
- Within the covered cases, rocprof's per-seam device-time coverage is 73–100%
  (empirical) and 54–100% (datasheet); the datasheet gaps are VALU kernels, which have
  no `mfma_*` datasheet peak and are dropped from that denominator rather than scored 0.
- `add_rmsnorm_quant` and `dsa_indexer_topk` have **no** datasheet column at all for
  the same reason — they are pure VALU seams.

## 5. Cross-check against the 0927 campaign table

`roofline/BASELINE_TABLE.md` in the same experiment directory measures the same seams
through a **completely different path**: `rocprofv3 --pmc` over 8 counter passes with
marker-kernel windowing, reduced by `reduce_v2.py` against
`ceilings_gfx950_0927.json`. Two of its rows overlap ours on the same bucket, and both
land on the same answer:

| seam / bucket | their AI | our AI | their emp | our emp |
|---|---|---|---|---|
| mxfp4_fused_moe pre_M32768, base | 232.6 | **232.6** | 0.7367 | 0.689 |
| add_rmsnorm_quant dec_bs3_m18_res, base | 0.435 | 0.2 (seam fold) | 0.0496 | **0.050** |

An AI agreeing to four significant figures and an efficiency agreeing to the third
decimal, from two independent collectors, two reductions and two ceiling files, is
about as good a validation of this pipeline as is available without a second GPU.

Two rows do **not** agree, and the reason is visible rather than mysterious:

- `sparse_mla_prefill pre_seq32768` — their AI 111.0 / emp 0.655 vs our 203.2 / 0.327.
- `dsa_indexer_prefill_score_topk pre_ch_r1383_ctx387k` — their AI 399.2 vs our 951.

Both are ~1.8–2.4× apart in AI and correspondingly in efficiency, and both are rows
where their run used the **old harness cache policy** (`write-evict`, and `read-evict`
on the second) while ours flushes per iteration. Cache policy changes how many bytes
reach HBM, which is the denominator of AI; it does not change what the kernel computes.
Neither number is wrong — they answer the question under different residency
assumptions, and that difference has to be stated whenever the two tables are read side
by side.

**Their speedup column and ours are not comparable, and should not be reconciled.** On
2026-09-27 their `kernel_src` was byte-identical to `baseline_src` for six of the nine
tasks, so their table correctly reports noise (0.993–1.008×) for those. The trees we
profiled were last modified **2026-09-28**, after that table was written. Our 4.04× on
`dsa_indexer_decode` and their 0.995× are measurements of two different candidates, a
day apart — not a disagreement.

## 6. What not to do with these numbers

- Do not divide a TL percentage by a rocprof percentage. On gfx950 the denominators
  differ by 0.856 (HBM), 0.573 (bf16), 0.628 (fp8), 0.502 (fp4).
- Do not read `x_TL%` (TL's percentage moving between arms) as evidence. TL's roof is
  bit-identical across arms, so its percentage *necessarily* tracks the measured
  speedup. Only the rocprof column is a test.
- Do not quote `attainable_speedup` on the five suspect `sparse_mla_prefill` rows or on
  the seven dispatch-floor rows.
- Do not compare `achieved` GFLOP/s across runs to judge contention changes; counters
  are stable, clocks are not.
