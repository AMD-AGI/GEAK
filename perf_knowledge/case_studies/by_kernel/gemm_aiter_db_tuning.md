---
title: aiter per-shape GEMM DB tuning — the +2.23% e2e win and the bias trap that nearly killed it
kind: case_study
operator: dense_gemm
backend: aiter
gens: [gfx942]
dtypes: [bf16]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - GEAK/examples/e2e_workflow/qwen3.5-27b_sglang_gemm-tuning-win/final_report.md
  - GEAK/examples/e2e_workflow/qwen3.5-27b_sglang_isl1024_osl1024_conc64/final_report.md
  - GEAK/e2e_workflow/knowledge/gemm_tuning/aiter_gemm_tuning.md
  - GEAK/e2e_workflow/knowledge/gemm_attention_backends.md
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/tuned_gemm.py
---

# aiter per-shape GEMM DB tuning (the +2.23% win, and the bias trap)

> **All e2e and isolated numbers here are measured-by-us** in the `e2e_workflow` eval dirs
> (Qwen3.5-27B, MI300X gfx942, sglang 0.5.11 / aiter, torch 2.9.1+rocm7.2), via same-session A/B.

## Context
On sglang/vLLM ROCm the dense bf16 GEMM mass (**~79% of GPU time on Qwen3.5-27B**) is dispatched
by **`aiter.tuned_gemm.gemm_a16w16` / `tgemm.mm`**, which picks the fastest of
hipBLASLt/asm/triton/skinny/flydsl **per shape from aiter's own per-shape DB**. The only GEMM
lever that engages this live path is **tuning that DB** — *not* PyTorch TunableOp, *not*
`HIPBLASLT_TUNING_FILE` (both sit under code paths sglang bypasses). Full recipe:
[`../../kernel_workflow/gemm_tuning_workflow.md`](../../kernel_workflow/gemm_tuning_workflow.md). Dispatch
internals: [`../../backends/aiter/tuned_gemm.md`](../../backends/aiter/tuned_gemm.md).

The hot shapes: up/gate prefill **K=5120, N∈{14336,16384,34816}**; down/qkv **N=5120,
K∈{17408,6144}**; M-buckets from the live conc=64 workload (padded, e.g. 16040→16128,
16369→16384, plus the 1024/decode buckets).

## Baseline
- Default aiter DB has **no config for these exact (M,N,K)** → logs `not found tuned config …
  using torch/default solution:0`; untuned aiter ≈ hipBLASLt default.
- e2e reference for the gate: the accepted **`--attention-backend triton`** stack — ref median
  **1548.9 tok/s**.
- Tier-A bake-off (op_bench): hipBLASLt 9.75 ms (correct default); the **aiter op_bench probe
  could not reach the GEMM seam** (62 entrypoints tried, all failed — but the *live*
  `tgemm.mm` path works fine); the op_bench triton entry is a **retired placeholder stub**
  (14.13 ms), not a real impl.

## What we tried

### Attempt 1 (iter1) — synthesized untuned set, guessed `bias=True` → 0 engagement
The untuned set was synthesized from the profile with **`bias=True`** and synthetic prefill
M-buckets {16040, 16369, 1024}. The aiter DB lookup key is the 9-tuple
`(cu_num, padded_M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle)`. sglang issues these
dense GEMMs with **`bias=False`** (bias added separately), and the real M-buckets differ.
**Every tuned row mismatched the live key:**
- `is tuned on cu_num` = **0**, `not found tuned config … using default` = **258**.
- isolated speedup was real (~1.032×) but **never reached the live path** → e2e **0%**.
- It also needed `SGLANG_USE_AITER=1` to route to `tgemm.mm` at all (else
  `UnquantizedLinearMethod` runs `F.linear`/hipBLASLt default).
- **Rejected at the engagement gate** — the TunableOp lesson: a config can look deployed and do
  nothing.

### Attempt 2 (iter2) — bias captured live, full coverage → +2.23%
The fix was to **stop guessing the schema and capture it from the live server**:
1. **Capture (`AITER_TUNE_GEMM=1`)** one warm bench at ISL/OSL/conc=1024/1024/64 → 234 real
   shapes. This **proved 228/234 dense GEMMs are `bias=False`** (only 6 tiny vision shapes are
   `bias=True`); `meta.json` had declared `bias=True` (the oracle) — wrong for the live path.
2. **Bucket-reduce** via `get_padded_m` → 78 unique padded buckets, **sorted FLOPs-DESC** so
   gradlib (processes input order, writes incrementally) tunes the GPU-dominant large-M prefill
   shapes FIRST (gradlib otherwise tunes M-ascending = decode-first = worst ROI).
3. **Tune** `gradlib/gemm_tuner.py --indtype bf16 --mp 8` — races ~1365 hipBLASLt + asm/triton/
   skinny solutions per shape, gates each on **`err_ratio < 0.05`**. N=34816 M=16384 fell to
   **triton sol=0** auto; all other dominant rows → hipBLASLt; **every kept row err_ratio=0.0**
   (parity-safe). 38/78 buckets tuned (partial DB = no regression; uncovered → default).
4. **Deploy by ENV** (reversible): `AITER_CONFIG_GEMM_BF16=<csv> AITER_LOG_TUNED_CONFIG=1`.

## What worked / what didn't
- **Worked:** the bias-correct, FLOPs-DESC, full-coverage tune. Engagement **proven** — all 8
  dominant `bias=False` shapes hit `is tuned on cu_num = 304` via the padded_M match
  (16040→16128, 16369→16384); **246 total engagement hits** in the warm e2e run.
- **Didn't:** the synthesized `bias=True` set (attempt 1, 0 hits), and a hand-authored Triton
  GEMM (iso 1.466× in iter2 / 0.99× in iter1) — the e2e gate was won by the aiter env path, so
  the authored kernel did not enter the stack.

## Final result (numbers, measured-by-us)
Fast 2-launch interleaved A/B, 5 repeats per leg, stacked on `--attention-backend triton`:

| leg | per-repeat tok/s | median | min / max |
|---|---|---|---|
| ref (triton attn) | 1509.4 / 1554.7 / 1544.9 / 1554.4 / 1548.9 | **1548.9** | 1509.4 / 1554.7 |
| cand (triton attn + aiter GEMM tune) | 1586.2 / 1583.5 / 1581.1 / 1573.4 / 1586.4 | **1583.5** | **1573.4** / 1586.4 |

→ **Δ = +2.23%**, distributions **non-overlapping** (cand_min 1573.4 > ref_max 1554.7),
**246 engagement hits** → **ACCEPTED**. Isolated geomean ~1.029× (wins concentrated on N=34816
~1.047×, alt-M N=16384 ~1.052×, down K=17408 ~1.056×; the already-optimal 16040×14336/16384 rows
flat). Stacked with the attention flag: **≈ +6% over the true baseline**. Parity safe — same
bf16 math, `err_ratio=0.0` on every row.

> This **corrected the earlier "GEMM tuning has no benefit" conclusion** — that came from the
> bias-mismatched partial tune in attempt 1, not from GEMM tuning being a bad lever.

## Lessons
1. **Capture `bias` + shapes from the LIVE server, never from `meta.json` or a guessed schema.**
   `bias` is in the DB lookup key; a wrong guess = 0 engagement = silent no-op.
2. **The aiter DB is the ONLY GEMM lever that engages the sglang live path** — TunableOp and
   `HIPBLASLT_TUNING_FILE` are the wrong lever (a prior attempt measured −0.11%/−0.30%).
3. **Prove engagement** (`grep -c 'is tuned on cu_num'` > 0) before believing any e2e delta.
4. **FLOPs-DESC tuning order** banks dominant-shape coverage within a time budget; partial DBs
   never regress (uncovered → default).
5. **Don't pin the tuner to 1 GPU** (`--mp` = all GPUs); **serialize heavy nested tunes** — the
   `rocm_agent_enumerator` fork-storm corrupts e2e timing.
6. **The CSV is build-locked** (`solidx`/`kernelName` are ROCm/hipBLASLt/aiter-specific) —
   regenerate on any upgrade; never ship a hand-copied CSV.

## Cross-links
- The recipe (capture → tune → deploy → gate, both traps): [`../../kernel_workflow/gemm_tuning_workflow.md`](../../kernel_workflow/gemm_tuning_workflow.md)
- Dispatch + key construction: [`../../backends/aiter/tuned_gemm.md`](../../backends/aiter/tuned_gemm.md) · DB: [`../../backends/aiter/configs_db.md`](../../backends/aiter/configs_db.md) · FlyDSL: [`../../backends/aiter/flydsl_path.md`](../../backends/aiter/flydsl_path.md)
- GEMM operator: [`../../operators/dense_gemm/overview.md`](../../operators/dense_gemm/overview.md) · [`../../operators/dense_gemm/tuning.md`](../../operators/dense_gemm/tuning.md)
- The full run this lives in: [`../by_model/qwen3.5-27b_sglang_e2e.md`](../by_model/qwen3.5-27b_sglang_e2e.md)
- e2e flow / gate: [`../../kernel_workflow/optimize_e2e_model.md`](../../kernel_workflow/optimize_e2e_model.md)

## Sources
- The +2.23% win, 246 hits, the A/B table, the bias fix: `GEAK/examples/e2e_workflow/qwen3.5-27b_sglang_gemm-tuning-win/final_report.md`.
- The attempt-1 bias-mismatch reject (0 hits, 258 misses): `GEAK/examples/e2e_workflow/qwen3.5-27b_sglang_isl1024_osl1024_conc64/final_report.md`.
- Recipe, both traps, FLOPs-DESC, capture-bias-from-server rule: `GEAK/e2e_workflow/knowledge/{gemm_tuning/aiter_gemm_tuning.md,gemm_attention_backends.md}`.
- Dispatch + key: `ROCm/aiter@a6bb499:aiter/tuned_gemm.py`, `gradlib/gradlib/gemm_tuner.py`.

<!-- MANIFEST: aiter per-shape GEMM DB tune on Qwen3.5-27B/MI300X — bias-correct live-captured full-coverage tune = +2.23% e2e (246 engagement hits, measured); the bias=True guess gave 0 engagement (the trap). -->
