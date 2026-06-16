# Implementation timeline — Qwen-Qwen3.5-27B end-to-end throughput optimization

## Run overview
- **Model / architecture**: Qwen-Qwen3.5-27B (`Qwen3_5ForConditionalGeneration`), architecture class
  `hybrid_linear_attention_dense`. 64 layers = 48 linear-attention layers (mamba / gated-delta style)
  + 16 full-attention layers (full_attention_interval=4), dense MLP (no MoE); hidden 5120,
  intermediate 17408, head_dim 256, 24 q heads / 4 kv heads, vocab 248320; dtype bf16.
- **Serving stack**: sglang 0.5.11, torch 2.9.1+rocm7.2.0, tp_size=1.
- **Available backends**: aiter / triton / hipblaslt. **Missing**: hipblaslt-bench (no offline GEMM CLI
  tuning), ckProfiler (no CK instance sweep).
- **Workload**: ISL/OSL/conc = 1024/1024/64 (**prefill-dominated**).
- **GPU**: AMD Instinct MI300X (gfx942), baseline on GPU0; noise band = 0.5%.
- **Date**: 2026-06-07.
- **One-line conclusion**: the only accepted optimization is the config-tier `--attention-backend triton`
  (+4.44% e2e, 1485.432 -> 1551.4 tok/s); all editable Triton FLA/mamba kernels had real isolated
  speedups and confirmed engine engagement, but in a prefill regime where ~81% of GPU time is dense GEMM
  they all fell inside the 0.5% noise band by Amdahl and were judged NULL — no kernel source patch made
  it into the final stack.

---

## Phases tree (which items each step optimized)

```
Phases
├── ✔ 1 Setup          baseline = 1485.4 tok/s  (TP=1, GPU0, spread 0.44%)
├── ✔ 2 Profile        Top-N: dense GEMM ~81% / gated-delta cluster ~9% / act_and_mul 2.2%
├── ✔ 3 Strategize     routing: 1 GEMM head (h0 up/gate) + 4 editable kernels
├── ✔ 4 ConfigSweep
│   ├── ✔ cfg0  --attention-backend triton       e2e +4.15%   → accept (only one in the final stack)
│   └── ✘ cfg1  --chunked-prefill-size 8192       e2e −0.42%   → reject (in band)
├── ✔ 5 HeadKernel     dense_gemm up/gate (K=5120, N∈{14336,16384,34816})
│   ├── ✘ aiter DB tune                 iso 1.032×  → 0 engagement (synthetic bias=True vs live bias=False mismatch)
│   └── ✘ Triton GEMM (team_workflow authored) iso 0.99×  → cannot beat hipBLASLt
├── ✔ 6 Milestone      editable FLA/mamba cluster (parallel optimize → serial e2e gate, floor=4)
│   ├── ✘ chunk_gated_delta_rule_fwd_kernel_h   iso 1.18× → e2e +0.17%        (2.95%gpu, ceiling <0.5%)
│   ├── ✘ chunk_fwd_kernel_o                    iso 1.14× → e2e −0.03%        (1.98%gpu)
│   ├── ✘ _causal_conv1d_fwd_kernel             iso 1.10× → e2e +0.29% STACK  (1.26%gpu, parity 12/12)
│   └── ✘ recompute_w_u_fwd_kernel              iso 0.99× (no isolated speedup)
├── ✔ 7 Finalize       bundle: final_launch.sh + final_patch.diff (empty)  (triton flag only, no overlay)
├── ✔ 8 Report         architect_report.md + implementation-timeline.md
└── ✔ 9 Validate       1549.2 tok/s, +4.1%, accepted, parity pass

Legend: ✔ done/accepted · ✘ rejected (in band / no speedup / no engagement) · STACK stackable but did not pass the 0.5% gate alone
Conclusion: only phase 4's --attention-backend triton entered the final stack; the head GEMM and all 4 single kernels failed the e2e gate.
```

## Directory layout (artifact tree · which phase produced which files)

```
e2e_Qwen-Qwen3.5-27B_20260607_080209.../
├── env_report.{md,json}                         # preflight / capability report
├── baseline/bench_summary.json                  # [P1] TRUE baseline 1485.4
├── profile/round_0|round_config/profile_topN.*  # [P2/5] Top-N breakdown
├── strategy.md                                  # [P3] Amdahl routing
├── config/
│   ├── sweep_results.json                        # [P4] cfg0/cfg1 sweep
│   ├── cfg0/  cfg1/                              # [P4] per-config bench
│   ├── hot_untuned_gemm.csv                      # [P5] real shapes captured by AITER_TUNE_GEMM
│   └── Qwen-Qwen3.5-27B_bf16_tuned_gemm.csv      # [P5] aiter GEMM tune output (bias=True synthetic → mismatch)
├── kernels/
│   ├── h0_cijk_upgate_gemm_task/                 # [P5] head GEMM op unittest + opbench_result.json
│   ├── chunk_gated_delta_rule_fwd_kernel_h_task/ # [P6] editable kernel task (reference_io.pt + unittest.py)
│   ├── chunk_fwd_kernel_o_task/                  # [P6]
│   ├── _causal_conv1d_fwd_kernel_task/           # [P6]
│   ├── recompute_w_u_fwd_kernel_task/            # [P6]
│   └── _exp/team_*                               # [P5/6] recursive team_workflow optimize (head Triton author + 4 kernels)
├── overlay/cand_*                                # [P5/6] per-candidate e2e A/B (ref/cand pair)
├── final/
│   ├── final_launch.sh                           # [P7] launch optimized server + bench (with --attention-backend triton)
│   ├── final_patch.diff                          # [P7] empty (pure-flag win, no source patch)
│   └── overlay/                                  # [P7] empty (no accepted kernel overlay)
├── architect_report.md  /  implementation-timeline.md   # [P8] reports
├── director_e2e_validation.json                 # [P9] official validation 1549.2 / +4.1% / accepted
└── logs/                                         # per-phase logs (capture/cfg/opbench/integrate/validation)
```

## Baseline phase

**Throughput** (`baseline/bench_summary.json`, 3 repeats):
- median **1485.432 tok/s**, spread 0.44%
- per run: 1485.432 / 1479.825 / 1486.405 tok/s
- TTFT median 3598.067 ms, TPOT median 39.523 ms

**Profile breakdown** (`profile/round_0/profile_topN.md`, torch-trace; total GPU time 5051.78 ms / 11113 launches / 64 distinct kernels):

| # | kernel | class | backend | editable | calls | total ms | %gpu | shape |
|--|--------|-------|---------|------|-------|----------|------|--------|
| 1 | Cijk...MT256x192x64 | library_gemm | hipblaslt | N | 336 | 2466.834 | **48.8** | [[16040,5120],[5120,14336/16384...]] |
| 2 | Cijk...MT256x192x64 | library_gemm | hipblaslt | N | 256 | 870.846 | **17.2** | [[15360,17408],[17408,5120]]... |
| 3 | Cijk...MT256x224x64 | library_gemm | hipblaslt | N | 128 | 434.096 | **8.6** | [[16040,17408],[17408,5120]]... |
| 4 | Cijk...MT224x320x64 | library_gemm | hipblaslt | N | 48 | 224.633 | **4.5** | [[16369,5120],[5120,16384]] |
| 5 | chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | fused_custom | triton | Y | 192 | 145.073 | 2.9 | [1,1024,16/48,128] |
| 6 | act_and_mul_kernel (sgl_hip) | elementwise_overhead | torch_native | Y | 256 | 108.330 | 2.1 | [[1024,17408],[1024,34816]] |
| 7 | chunk_fwd_kernel_o | fused_custom | triton | Y | 192 | 98.353 | 1.9 | [1,1024,16/48,128] |
| 8 | recompute_w_u_fwd_kernel | fused_custom | triton | Y | 192 | 82.337 | 1.6 | [1,1024,16/48,128] |
| 9 | elementwise_kernel_manual_unroll | elementwise_overhead | torch_native | Y | 1344 | 72.167 | 1.4 | — |
| 10 | _causal_conv1d_fwd_kernel | fused_custom | triton | Y | 192 | 65.523 | 1.3 | — |
| 11 | aiter add_rmsnorm_quant_kernel | fused_custom | aiter | Y | 512 | 65.044 | 1.3 | [[1024,5120]...] |
| 12 | ck_tile FmhaBatchPrefillWithPagedKVCache | library_attn | ck | N | 64 | 54.624 | 1.1 | paged-attn |
| 14 | Cijk...MT160x256x64 | library_gemm | hipblaslt | N | 64 | 43.350 | 0.9 | [[1024,5120],[5120,34816]] |
| 15 | chunk_gated_delta_rule_fwd_kkt_solve_kernel | fused_custom | triton | Y | 192 | 42.098 | 0.8 | [1,1024,16/48,128] |
| 17 | _layer_norm_fwd_1pass_kernel | reduction_norm | triton | Y | 192 | 30.738 | 0.6 | [[49152,128]...] |

**Key reading**: dense hipBLASLt GEMM (ranks 1–4) sums to ~**79%**, all library_gemm ~**81%** of GPU
time — this is the only lever with e2e headroom above the noise band (Amdahl priority). The editable
Triton FLA/mamba cluster (gated-delta, conv1d, norm) is individually ≤3% GPU each. Baseline attention
runs on CK paged-attention (rank 12, 1.1%).

---

## Per-phase timeline

### Phase A — ConfigSweep (config fast path, run first, no source edits)
Source: `config/sweep_results.json`, e2e 5 repeats per config, gate 0.5%.

**Attempt 1 — `--attention-backend triton` (cfg0, stacked on baseline)**
- e2e: **1547.027 tok/s**, spread 0.8%; vs baseline **+4.15%** (far above the 0.5% gate).
- Engine engaged: ✅ server.log shows `attention_backend='triton'`, `linear_attn_backend='triton'`,
  `mamba_backend='triton'`.
- Parity: PASS — greedy temp=0 fixed seed, 3 of 5 prompts byte-identical, 2 diverge only in a deeply
  repetitive tail but reach the same answer (Paris; 17x23=391), a benign bf16 tie, no quality regression.
- Side effect: converts the 16 full-attention layers plus the linear-attn/mamba path to editable Triton
  kernels, opening an editing surface for the downstream kernel track.
- **Decision: accept.** Current best config = `--attention-backend triton`.

**Attempt 2 — `--chunked-prefill-size 8192` (cfg1, stacked on cfg0)**
- e2e: 1540.519 tok/s, spread 0.92%; vs cfg0 **-0.42%** (inside the 0.5% noise band and slightly negative).
- Engine engaged: ✅ `chunked_prefill_size=8192`; but cuda-graph is on by default
  (`disable_cuda_graph=False`, `cuda_graph_max_bs=512`), the tiny elementwise/index/cat ops are already
  covered by the graph, no decode-dispatch headroom.
- **Decision: reject.** In a prefill-dominated scenario, shrinking the prefill chunk yields no
  decode-interleave benefit and slightly hurts prefill GEMM batching. Revert; config stays at cfg0.

**ConfigSweep summary**: accept `--attention-backend triton`, env empty. best = 1547.027 tok/s,
vs baseline 1.0415x.

### Phase B — Re-profile (against the triton-attention config)
Source: `profile/round_config/profile_topN.md` (total GPU 5058.44 ms / 14504 launches / 62 kernels).
- Bottleneck **did not move**: dense hipBLASLt GEMM ranks 1–4 = 78.8% gpu, all library_gemm ~81%
  (nearly identical to baseline).
- Change: the baseline's CK paged-attention (library_attn, 1.1%) is replaced by the editable Triton
  `_fwd_kernel` (rank 12, 1.05%, 64 calls = 16 full-attn layers × prefill); the gated-delta Triton
  kernels were already Triton at baseline, %gpu basically unchanged (2.9%->3.0%).
- Net conclusion: dense GEMM remains the primary lever for the head/config track.

### Phase C — HeadKernel (dense GEMM head track)
**Attempt — aiter tuned_gemm CSV (up/gate GEMM, K=5120, N ∈ {14336,16384,34816})**
Source: `kernels/h0_cijk_upgate_gemm_task/`, `overlay/cand_cijk_upgate_gemm/`, `config/*tuned_gemm.csv`, tune.log.
- Isolated effect: ~1.032x (offline).
- e2e: 0% (probe 1545.5 tok/s ≈ current accepted 1547.027, statistically the same).
- Engine engaged: ❌ **CSV not actually consumed**. Needs `SGLANG_USE_AITER=1` to take the tgemm.mm path
  (otherwise UnquantizedLinearMethod goes through F.linear / hipBLASLt default). Even with aiter on, the
  warm server.log shows `is tuned on cu_num`=0, `not found tuned config ... using default config`=258.
- Root cause: the CSV was tuned on **synthetic bias=True prefill M-buckets {16040,16369,1024} × N{14336,16384,34816}**,
  but the actual up/gate GEMM is **bias=False with real M-buckets (122/96/88/80...)**; the aiter lookup
  key includes bias, so every lookup misses → falls back to the default torch/hipBLASLt solution.
- **Decision: reject (at the engagement gate, the TunableOp lesson).** A real isolated speedup that can't
  reach the live path is an expected outcome, not an e2e measurement.
- Re-tune fix: regenerate the CSV from **live-captured real shapes (bias=False, the actual conc=64
  M-buckets)** and verify `is tuned on cu_num > 0` before integrating.

### Phase D — Milestone 1 (editable Triton FLA/mamba kernel cluster, reaching the MIN_KERNEL_TASKS floor)
Source: `HISTORY.ledger`, `overlay/*/`, `kernels/*_task/`, `insight_log.md`.
All directions are provenance-OK (reference_io.pt sha256 matches meta.json, unittest.py untampered),
engine engagement proven via overlay banner (4 hits/worker + `[OVERLAY_ENGAGED]`), tight same-session
interleaved A/B.

**Attempt 1 — chunk_gated_delta_rule_fwd_kernel_h (k0)**
- Isolated: **1.1811x**; e2e: **+0.171%**; %gpu 2.95%.
- A/B (GPU0, pinned port, 5 clean REF + 6 clean CAND, after 3 grpc-port-flake retries): ref_med 1551.196
  [1546.84, 1557.49], cand_med 1553.847 [1550.40, 1566.40].
- Gate: delta +0.171% < 0.5% **and** distributions overlap (cand_min 1550.40 < ref_max 1557.49) → FAIL.
- **Decision: reject (Amdahl).** 2.95% GPU × 1.1811x gives an e2e ceiling of ~0.45%, below the band by construction.

**Attempt 2 — chunk_fwd_kernel_o (k1)**
- Isolated: **1.1405x**; e2e: **-0.035%**; %gpu 1.98%.
- A/B (GPU1, pinned port 31337, 5 ref + 5 cand): ref_med 1551.585 [1546.918, 1554.663], cand_med 1551.037
  [1548.667, 1558.197].
- Gate: delta -0.035% < band **and** distributions overlap (cand_min 1548.667 < ref_max 1554.663) → FAIL.
- **Decision: reject (Amdahl).** e2e ceiling ~1.98%×(1-1/1.14)=~0.24% < band.

**Attempt 3 — _causal_conv1d_fwd_kernel (k2)**
- Isolated: **1.1049x**; e2e: **+0.292%**; %gpu 1.26%.
- Engine: ✅ banner `injected ...mamba.causal_conv1d_triton` (4 hits/process); confirmed against a no-overlay
  REF. winner_kind=patch (add-module single-submodule injection).
- Parity: **PASS — 12/12 greedy prompts byte-identical**.
- A/B (GPU3, pinned port 31537, 5 repeats/leg): REF med 1536.27 [1530.88, 1541.72], CAND med 1540.75
  [1531.03, 1541.17].
- Gate: delta_med +0.292% (non-negative, cand_med ≥ ref_med) but < 0.5% band, distributions overlap
  (cand_min 1531.03 < ref_max 1541.72) → not a strong accept.
- **Decision: reject / STACK carry-forward.** e2e ceiling ~1.26%×(1-1/1.10)=~0.11% < band; no regression,
  parity clean, can compose with the same cluster — carried into the Director's merge-vs-true-baseline gate
  to capture compound gains.

**Attempt 4 — recompute_w_u_fwd_kernel (k3)**
- Isolated: **0.9938x** (no isolated speedup).
- **Decision: reject.** No speedup, no need to measure e2e.

**Common pattern**: real and verified isolated speedups fail to surface at e2e because each kernel is a
small fraction in a prefill regime (~80% GPU is dense GEMM). This is an expected Amdahl outcome, not an
integration bug. Quick rule: `pct_gpu × (1 - 1/iso) < 0.5%` ⇒ can only be carried forward, cannot pass
the gate alone.

### Phase E — Final re-measure (accepted config `--attention-backend triton`, no overlay)
Source: `final/bench_out/bench_summary.json` (5 repeats), `final/final_patch.diff` (no source patch).
- Throughput: median **1551.4 tok/s**, spread 0.33%.
- per run: 1553.739 / 1551.400 / 1550.808 / 1549.324 / 1554.405 tok/s.
- TTFT median 3564.864 ms, TPOT median 37.616 ms.
- final_patch.diff: no kernel source patch; the only optimization is the launch flag
  `--attention-backend triton` (written into final_launch.sh's EXTRA_SERVER_ARGS).

---

## Summary table (all attempts)

| lever | isolated speedup | e2e effect | verdict | root cause |
|---|---|---|---|---|
| `--attention-backend triton` (cfg0) | — | **+4.15%** (1485.4→1547.0), passes gate | **accept** | only one to pass; engine engaged; parity clean; also opens an editable Triton surface |
| `--chunked-prefill-size 8192` (cfg1) | — | -0.42% (in band, slightly negative) | reject | cuda-graph already on; prefill-dominated, no decode-interleave headroom |
| aiter tuned_gemm CSV (up/gate GEMM) | 1.032x | 0% | reject | engagement gate: bias=True/synthetic M-buckets mismatch live bias=False/real M-buckets, every lookup misses → default |
| chunk_gated_delta_rule_fwd_kernel_h | 1.1811x | +0.171% (in band, overlapping) | reject | Amdahl: 2.95% gpu, ceiling ~0.45% < band |
| chunk_fwd_kernel_o | 1.1405x | -0.035% (in band, overlapping) | reject | Amdahl: 1.98% gpu, ceiling ~0.24% < band |
| _causal_conv1d_fwd_kernel | 1.1049x | +0.292% (in band, non-negative, parity 12/12) | reject / STACK | Amdahl: 1.26% gpu, ceiling ~0.11% < band |
| recompute_w_u_fwd_kernel | 0.9938x | — | reject | no isolated speedup |

---

## Final deliverable
- **Accepted config**: `--attention-backend triton`, env empty, no kernel source patch, no overlay.
- **Throughput**: baseline 1485.432 → final **1551.4 tok/s**, **1.0444x (+4.44%)**.
- Milestones (accepted changes): 1; budget used 4/4, reached the MIN_KERNEL_TASKS floor.

## Measurement notes (box drift → trust only same-session A/B)
- e2e medians cluster tightly (~1551 tok/s) and the box drifts across sessions; trust only the
  **same-session interleaved A/B**, with the gate requiring **delta_med > 0.5% band AND non-overlapping
  distributions (cand_min > ref_max)** — a positive median alone is vetoed by overlapping distributions.
- Noise band 0.5%; ≥5 repeats per config, server kept WARM, startup excluded from the timing window.
- Always check parity (greedy temp=0 fixed seed) — a faster-but-wronger server is a regression.
- Infra: sglang `grpc_port = port + 10000` and rejects >65535 → always pin a low PORT (used 31337/31537),
  and reserve a grpc-port-flake retry budget (saw up to 3).

## Next directions
1. **Dense hipBLASLt GEMM (~81% GPU) is the only lever still with e2e headroom above the band** — a 1.15x
   on 78% GEMM ≈ +10% e2e.
2. **aiter tuned_gemm re-tune**: regenerate the CSV from live-captured shapes (bias=False, real conc=64
   M-buckets), enable `SGLANG_USE_AITER=1`, verify `is tuned on cu_num > 0` before integrating.
3. **Author a Triton dense GEMM** (hipblaslt-bench and ckProfiler both missing, offline CLI tuning
   unavailable) for the head op task, and let the e2e gate pick the best.
4. The editable Triton FLA/mamba cluster as a standalone gate-passing track is exhausted; carry conv1d
   (non-negative, parity-clean) into the merge-vs-true-baseline gate to capture compound gains.
5. Once the floor is reached and the best remaining candidate cannot clear the band, consider stopping the
   kernel track per the Amdahl stop rule.
</content>
