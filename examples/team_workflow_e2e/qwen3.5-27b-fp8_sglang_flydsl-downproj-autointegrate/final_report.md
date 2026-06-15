# Final Report — Qwen3.5-27B-FP8 End-to-End Serving Optimization

## Run overview
- **Model / architecture:** Qwen-Qwen3.5-27B-FP8 — `hybrid_linear_attention_dense_vlm`, 64 layers (48 linear-attention + 16 full-attention, full_attention_interval=4). Weights fp8 a8w8 **blockscale** (weight_block_size [128,128]), bf16 compute.
- **Serving stack:** sglang 0.5.11.
- **Workload:** ISL/OSL/conc = 1024/1024/64 (prefill-dominated capture window; decode/TPOT-bound steady state).
- **GPU:** AMD MI300X, gfx942. available_backends = aiter / triton / hipblaslt (hipblaslt-bench CLI and ckProfiler absent).
- **Serving invariant:** TP=1 on a single GPU (id 0), mem-fraction-static 0.85 — identical for baseline and every trial. Port pinned in 31000-40000.
- **Date:** 2026-06-13/14.
- **Final conclusion:** Stacking `--attention-backend triton` (+2.24%) with an authored FlyDSL fused fp8 a8w8 blockscale Triton core bound capture-safely over the live decode path (+60.09% on that reference) lifts e2e output throughput from **931.593 -> 1559.934 tok/s (1.674x, +67.4%)** with parity preserved.

---

## Phases tree

```
Phases
├── ✔ 1 Setup         baseline = 931.593 tok/s (TP=1, mem-frac 0.85, default flags); TPOT 64.06 ms
├── ✔ 2 Profile(r0)   total GPU 6277 ms; head = fp8 a8w8 blockscale GEMM ~82% (up/gate 57.2%, down 18.7%, qkv/o 6.0%)
├── ✔ 4 ConfigSweep
│   ├── ✔ cfg0 --attention-backend triton   e2e +2.24% (931.6→952.5)  parity pass  → ACCEPTED (STACK)
│   └── ✘ cfg1 --kv-cache-dtype fp8_e4m3     not run: prefill-compute-bound, zero KV headroom + lossy → dropped
├── ✔ 5 HeadKernel
│   ├── ✘ h0 up/gate GEMM (N=34816,K=5120) 57.2%   iso 1.10x → e2e −0.257% (regression, decode stays generic) → REJECTED
│   ├── ✔ h1 down GEMM   (N=5120,K=17408)  18.7%   iso 2.432x → e2e +60.09% (953.1→1525.9) parity pass → ACCEPTED (STACK)
│   └── ✘ h2 qkv/o GEMM  (N=5120,K=6144)    6.0%   iso 1.43x → e2e 0% (win is nested-graph, capture-unsafe; bare core already deployed via h1 seam) → REJECTED
├── ✔ 7 Profile(rhead)  re-profile on accepted stack: _fused_blockscale_kernel (FlyDSL core) now 78.9% — GEMM still head, on the fast core
└── ✔ 9 Validate        final = 1559.934 tok/s, +67.4% vs baseline; TPOT 37.19 ms; accepted, parity pass

Legend: ✔ accepted/passed · ✘ rejected/dropped · STACK = entered the final stack
Conclusion: final stack = [--attention-backend triton] + [FlyDSL fused fp8 a8w8 blockscale core, capture-safe bare bind over all blockscale GEMM shapes].
```

```
Artifacts tree (EVAL_DIR, -L 2, excluding __pycache__/*.pyc/*.so)
.
├── env_report.json / env_report.md / env_info.txt   [P0 preflight] machine ground truth (gfx942, sglang 0.5.11, backends)
├── model_path.txt / bench_e2e.sh / parse_profile.py [P0] harness + profile parser
├── strategy.md                                      [P3 strategize] routing plan (head = fp8 blockscale GEMM)
├── adapters/                                         [P0] sglang.sh / vllm.sh serving adapters
├── preflight_smoke/                                  [P0] smoke bench_summary.json + server.log + profile
├── baseline/                                         [P1 baseline] bench_summary.json (931.593), bench_runs.jsonl, profile/, server.log
├── profile/
│   ├── round_0/                                      [P2] baseline Top-N (profile_topN.md/.json) — GEMM ~82%
│   ├── round_config/                                 [P4] profile after config sweep
│   └── round_head/                                   [P7] post-integration Top-N — FlyDSL core 78.9%
├── config/
│   ├── sweep_results.json                            [P4] cfg0 accepted (+2.24%), cfg1 dropped
│   ├── cfg0/ , parity/                               [P4] cfg0 run dir + parity probes
│   ├── gfx942-GEMM-A8W8_BLOCKSCALE-*.json            [P5] per-(N,K) Triton config overlays (h0/h1/h2 sweep)
│   └── confirm_bn256.py / sweep_dominant.py / verify_isolated*.py [P5] isolation harnesses
├── kernels/
│   ├── h0_..._upgate_task/ h1_..._downproj_task/ h2_..._qkvo_task/  [P5] extracted op tasks (meta.json, unittest.py)
│   └── _exp/team_h0.../ team_h1.../ team_h2.../       [P5] recursive team_workflow authoring runs (FlyDSL)
├── overlay/
│   ├── cand_gemm_upgate_N34816K5120/                 [P5 h0] integrate_result.json — REJECTED (−0.257%)
│   ├── cand_flydsl_downproj_N5120K17408/             [P5 h1] first overlay — crashed at capture (nested graph)
│   └── cand_flydsl_downproj_nocgraph/                [P5 h1] ACCEPTED overlay — bare core, integrate_result.json (+60.09%)
├── final/
│   ├── final_bench/bench_summary.json                [P9 validate] 1559.934 tok/s, TPOT 37.19 ms
│   ├── final_launch.sh / final_patch.diff            [P9] reproduction launch + patch
│   └── overlay/ (gemm_a8w8_blockscale_flydsl.py, sitecustomize.py, integrate_result.json)  [P9] shipped seam
├── logs/                                             [all phases] per-step logs (baseline, cfg, opbench, integrate, parity, profile)
├── architect_report.md                              [P-report] concise summary
└── final_report.md                                  [P-report] this file
```

---

## Baseline phase

- **Throughput:** median **931.593 tok/s**, spread 0.17% over 3 runs: 930.531 / 931.593 / 932.091 tok/s (tight, trustworthy baseline).
- **Latency:** TTFT 4451.45 ms median; **TPOT 64.056 ms** median.
- **Profile (round_0, torch-trace):** total GPU time **6277.48 ms** over 15470 launches, 77 distinct kernels.

### Top-N profile breakdown (round_0, baseline)
The Top-N is dominated by the fp8 a8w8 blockscale dense GEMM (`_gemm_a8w8_blockscale_kernel_...`, hipblaslt, edit=N), split across launch buckets. Deduped by (N,K) family:

| family | (N,K) | role | summed %gpu | ranks | editable |
|---|---|---|---|---|---|
| up/gate | (34816, 5120) | MLP in (gate+up) | **57.2%** | 1,2,3,7,8,9,18,19,20,24 | library (Tier-A/B/C) |
| down | (5120, 17408) | MLP out | **18.7%** | 4,5,6 | library (Tier-A/B/C) |
| qkv/o | (5120, 6144) | qkv + o_proj | **6.0%** | 11,12,13 | library (Tier-A/B/C) |

Total dense fp8 GEMM ≈ **82%** — this is the Amdahl head. Editable tail (each below the 5% head bar):
- `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` — 2.4% gpu, 192 calls, 148.6 ms, triton, editable (gated-delta linear attn).
- `act_and_mul_kernel` (sgl_hip) — 1.7% gpu, 256 calls, 104.6 ms, torch_native, editable (fusion candidate).
- `chunk_fwd_kernel_o` — 1.6% gpu, triton, editable.
- `dynamic_per_group_scaled_quant_kernel` (aiter) — 1.5% gpu, 1024 calls, editable.
- `recompute_w_u_fwd_kernel` — 1.4% gpu, triton; `elementwise_kernel_manual_unroll` — 1.1%; `_causal_conv1d_fwd_kernel` — 1.1% triton; `add_rmsnorm_quant_kernel` (aiter) — 0.9%; full-attn `FmhaBatchPrefill...` (ck) — 0.8%.

Regime: prefill-dominated capture (shapes M≈15360-16384), but steady-state throughput is decode/TPOT-bound — a key fact that decided the h0 outcome.

---

## Per-phase timeline

### Phase: ConfigSweep (config fast path)
**cfg0 — `--attention-backend triton` (axis: attention-backend).**
- Isolated: n/a (service-level flag).
- e2e: 952.451 tok/s vs baseline 931.593 (runs 953.08/952.45/952.33 vs 930.53/931.59/932.09), **delta +2.239%**, spread 0.08%, **non-overlapping** (cfg0 min 952.33 > baseline max 932.09) — passes the 0.5% gate.
- Engagement: server log confirms attention_backend/mamba_backend/linear_attn_backend all = triton; moves the 16 full-attn prefill layers onto the editable Triton `_fwd_kernel`.
- Parity: PASS — 4/6 byte-exact; 2 divergences are benign early-token bf16 tie-breaks with semantically identical correct answers.
- **Decision: ACCEPTED and stacked.** Becomes the reference config for all later A/B.

**cfg1 — `--kv-cache-dtype fp8_e4m3` (axis: kv-cache-dtype).**
- Not run. Live evidence: steady-state decode at conc=64 shows full-token usage 0.13-0.14 (KV pool ~13% used), queue=0 -> workload is prefill-compute-bound, NOT KV-memory-bound; a larger KV pool yields no throughput at fixed conc. Change is lossy by design.
- **Decision: DROPPED** (no upside + accuracy risk).

### Phase: HeadKernel

**h0 — up/gate GEMM (N=34816,K=5120), 57.2% gpu — the primary bet.**
- Attempt: per-(N,K) Triton config overlay on `aiter.ops.triton.gemm_a8w8_blockscale` (BM=256/nw=8/GROUP_M=4 for prefill M; decode M=1/64 stays generic BM=128). Also a FlyDSL author attempt.
- Isolated: **1.1023x** prefill (1.095-1.113x at M=8192/15360/16384), cross_relerr=0.0 (bit-identical, parity-safe). FlyDSL author variant: no isolated speedup (dead end).
- Engagement: PROVEN — after dropping the JSON, `get_gemm_config(...)` returns is_tuned=True with BM=256 for prefill M; live sglang path uses this loader.
- e2e: tight 2-launch A/B on the accepted config, ref_med 953.526 -> cand_med **951.077 => delta −0.257%** (non-overlapping, wrong side). A regression.
- Root cause (Amdahl): the +1.10x lands only on prefill tiles; the throughput-critical decode regime (M=1/64) deliberately stays on generic BM=128, so the tuned tiles never touch steady-state output tok/s at the 1024/1024/conc=64 (decode/TPOT-bound) workload.
- **Decision: REJECTED** (do-no-harm: never stack a regression). Site-packages JSON reverted; install left clean.

**h1 — down GEMM (N=5120,K=17408), 18.7% gpu — authored Tier-C win.**
- Attempt: authored **FlyDSL fused fp8 a8w8 blockscale Triton core**, rebound over `aiter.ops.triton.gemm_a8w8_blockscale` + sglang `fp8_utils` globals via a lazy meta-path `sitecustomize` seam.
- Provenance: OK — unittest.py sha256 == meta.unittest_sha256; synthesized shapes (N=5120,K=17408, fp8_e4m3fnuz->bf16) unchanged; director independently reproduced 2.432x.
- Isolated: **2.432x** geomean.
- Engagement: PROVEN — server log shows "bound BARE FlyDSL core" + "rebound fp8_utils -> BARE FlyDSL core"; 283 decode batches logged "cuda graph: True" => the fused core launches INSIDE sglang's captured decode graph.
- e2e: tight 2-launch A/B, GPU0, TP=1, mem-fraction 0.85, `--attention-backend triton`, ISL/OSL/conc 1024/1024/64. ref_runs 954.349/951.928 (med 953.139); cand_runs 1510.806/1540.907 (med **1525.857**). **delta +60.088%**, non-overlapping (cand_min 1510.8 > ref_max 954.3), >> 0.5% band. TPOT 62.627 -> 37.373 ms. Equal KV budget (both 1,156,449 tokens) -> no KV starvation; fused fp8 core has no bf16 re-materialization (do-no-harm on memory).
- Parity: PASS — 10/12 exact greedy matches; 2 diffs are coherent on-task free-form completions (same benign low-precision class as the accepted triton attention backend).
- Required fix: the authored final_patch shipped a `graph_replay.py` doing its OWN nested `torch.cuda.CUDAGraph` capture+replay — illegal inside sglang's decode capture; the original overlay (`cand_flydsl_downproj_N5120K17408`) crashed ("Cannot prepare for replay during capturing stage", 0 forwards). FIX: stripped the graph_replay wrapper and bound the **bare** fused fp8 core via a minimal `sitecustomize` seam -> capture-safe (Capture cuda graph end 24.62s).
- **Decision: ACCEPTED and stacked** (`overlay/cand_flydsl_downproj_nocgraph`). Caveat: the seam binds the FlyDSL core to ALL blockscale GEMM shapes (down + up/gate + qkv/o), not only the validated down-proj — the +60% reflects the fused fp8 core beating aiter's generic blockscale fallback across the whole decode path, with parity and the non-overlapping A/B confirming it is correct and net-positive.

**h2 — qkv/o GEMM (N=5120,K=6144), 6.0% gpu — marginal head.**
- Attempt: FlyDSL core variant for this shape.
- Isolated: **1.4296x** — but the ENTIRE win is a wrapper-level nested `torch.cuda.CUDAGraph` capture/replay that collapses the decode LAUNCH FLOOR (M=1 1.94x, M=64 1.56x are pure launch-overhead; the bare GEMM core is 0.98x at prefill M=16384 — no compute win).
- e2e: **0%** — the nested graph capture is illegal inside sglang's decode capture (same pattern crashed in h1). The capture-safe bare fp8 core is ALREADY deployed and engaged for this shape via the accepted h1 seam (shape-agnostic rebind of `triton_gemm_a8w8_blockscale`), so there is no remaining capture-safe delta to fold in. Amdahl: head is only ~6.05% gpu (~+0.3% ceiling).
- **Decision: REJECTED** (reason: cuda_graph_capture_unsafe). No A/B run — the sole differentiator (the graph wrapper) is provably capture-unsafe and would either crash or measure identically to the already-deployed bare core. Carry forward unchanged.

### Phase: Re-profile (round_head, on accepted stack)
After the FlyDSL core engages, total GPU time drops to 5085.29 ms; `_fused_blockscale_kernel` (the FlyDSL core) is now **78.9%** of GPU — the fp8 GEMM remains the head but now runs on the fast core. The editable gated-delta/FLA tail re-surfaces (chunk_gated_delta_rule_fwd_h 3.0%, act_and_mul 2.1%, dynamic_per_group_scaled_quant 2.0%, chunk_fwd_kernel_o 2.0%, recompute_w_u 1.6%, causal_conv1d 1.2%) but each is sub-noise solo. (Note: round_head multi-launch spread was 63% — box drift; not used for decisions.)

### Phase: Validate (final)
- Final config: `--attention-backend triton` + FlyDSL fused fp8 blockscale core (capture-safe bare bind), TP=1, mem-fraction 0.85.
- Final throughput: **1559.934 tok/s** median, 3 runs (1560.697 / 1528.215 / 1559.934), spread 2.08%.
- TPOT 37.187 ms; TTFT 3773.07 ms.
- **Speedup vs baseline 931.593: 1.674x (+67.4%).** Parity pass. Reproduction: `final/final_launch.sh`, `final/final_patch.diff`.

---

## Summary table

| Lever | Isolated | e2e | Verdict | Root cause / note |
|---|---|---|---|---|
| cfg0 `--attention-backend triton` | n/a | +2.24% (931.6→952.5) | ✔ accepted | Editable Triton attn/mamba/linear-attn path; parity pass |
| cfg1 `--kv-cache-dtype fp8_e4m3` | n/a | not run | ✘ dropped | Prefill-compute-bound; KV pool ~13% used; lossy, no upside |
| h0 up/gate GEMM (N=34816,K=5120) 57.2% | 1.10x | −0.257% (953.5→951.1) | ✘ rejected | +1.10x on prefill tiles only; decode stays generic → no e2e conversion (Amdahl) |
| h0 up/gate FlyDSL author | ~1.0x | — | ✘ dead end | Author produced no isolated speedup |
| h1 down GEMM (N=5120,K=17408) 18.7% | 2.432x | +60.09% (953.1→1525.9) | ✔ accepted | Fused fp8 core beats aiter generic blockscale fallback across whole decode path; capture-safe bare bind; parity pass |
| h2 qkv/o GEMM (N=5120,K=6144) 6.0% | 1.43x | 0% | ✘ rejected | Win is nested-graph capture (illegal in sglang decode capture); bare core already deployed via h1 seam |
| **Final stack** | — | **+67.4% (931.6→1559.9)** | ✔ validated | TP=1, mem-frac 0.85, parity pass |

---

## Final deliverable + caveats + next directions

**Final deliverable:** `--attention-backend triton` + an authored FlyDSL fused fp8 a8w8 blockscale Triton core, bound capture-safely (bare core, no nested graph) over the single live blockscale GEMM call site so it serves all blockscale GEMM shapes inside sglang's captured decode graph. **931.593 -> 1559.934 tok/s (1.674x, +67.4%)**, TPOT 64.06 -> 37.19 ms, parity preserved. Shipped: `final/final_launch.sh`, `final/final_patch.diff`, `final/overlay/{gemm_a8w8_blockscale_flydsl.py, sitecustomize.py}`.

**Measurement caveats:**
- **Box drift was severe** in the multi-launch profile rounds (round_head spread 63%, round_config 46%). Trust ONLY same-session tight A/B deltas (which every accept/reject decision used); never compare cross-round medians. The accepted +2.24% (cfg0) and +60.09% (h1) were both non-overlapping same-session A/B; the −0.257% h0 regression was likewise same-session and non-overlapping on the wrong side.
- The h1 seam binds the FlyDSL core to ALL blockscale GEMM shapes, not just the validated down-proj; correctness is backed by the 10/12 greedy-parity probe and the non-overlapping A/B, but the win is a whole-decode-path effect, not an isolated down-proj effect.
- fp8 blockscale is inherently approximate (tol 0.06); the 2/12 parity diffs are benign on-task free-form divergences, same class as the accepted triton attention backend.

**Next directions to explore:**
1. A faster fp8 a8w8 blockscale core: round_head shows the FlyDSL core is still 78.9% of GPU. The intrinsic ~21% MFMA-peak ceiling on these tiles (per MEMORY) caps config tuning, so the lever is a kernel rewrite / split path, not config JSON. Re-isolate the FlyDSL core's per-M behavior and target the decode M=1/64 regime specifically.
2. Stack-compound the editable gated-delta/FLA cluster (chunk_gated_delta_rule_fwd_h, chunk_fwd_kernel_o, recompute_w_u, causal_conv1d) — each sub-noise solo, but a combined milestone gated vs the accepted stack could clear the band.
3. Host-overhead fusion (act_and_mul 2.1%, dynamic_per_group_scaled_quant 2.0%, elementwise_unroll) via fusion / wider cuda-graph coverage to collapse dispatch overhead.
