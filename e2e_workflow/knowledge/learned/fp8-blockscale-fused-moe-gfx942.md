---
kernel_class: fused_moe_grouped_gemm
gfx: gfx942
regime: fp8 blockscale g1u1 MoE (DeepSeek-R1 K=7168 inter_per_tp=512 E=257 top-9 per_1x128), sglang+aiter
confidence: 2
confirms: 1
last_seen: 2026-07-08
---

# fp8-blockscale fused-MoE (aiter fused_moe, per_1x128 g1u1) — gfx942 / MI300X

**Live head** = single aiter asm 1-stage kernel `fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256`
(`.co`), dispatched from `sglang...moe_runner.aiter` via `from aiter.fused_moe import fused_moe`.
Rebindable Python seam = `aiter.fused_moe:fused_moe`; constituent gate_up/down GEMMs are NEVER
standalone (no dense-GEMM swap site). 49% GPU time on this DeepSeek-R1 TP=4 run.

**Dispatch is HEURISTIC, not tuned-DB, on this box.** `aiter/configs/tuned_fmoe.csv` (+ model_configs,
incl. `a8w8_blockscale_tuned_fmoe_ds_v3.csv`) ships rows keyed on **`cu_num=80`**; MI300X reports
**`cu_num=304`** → every lookup MISSES → `fused_moe` falls back to `get_2stage_cfgs` DEFAULT heuristic:
for `per_1x128` gfx942 `run_1stage = token>32 and inter_dim%128==0`. So M∈{64,512,2048} run the asm
1-stage kernel, M=1 runs 2-stage CK (block_m16/ksplit7). Confirmed via `run_1stage=…`/`using Nstage
default` server-log lines.

**Tier-B aiter fmoe DB re-tune (env lever `AITER_CONFIG_FMOE=<tuned.csv>`) → FASTER BUT INCORRECT.**
`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py -i <untuned 257/9/per_1x128 @ tokens 1/64/512/2048>`
on cu_num=304 (needs a 1-line guard: its failed-case report does `_, stage, kname, blockM = tag` but
flydsl/asm tags are len>4 → ValueError aborts the whole batch; wrap in len==4 check). It picks
**2-stage CK for every bucket** (the asm 1-stage candidates all `timeout/hang` in the tuner harness, so
they're excluded). Deployed via `AITER_CONFIG_FMOE`, engagement confirmed (`using 2stage … for (304,…)`),
raw ms IS lower (M2048 1.47 vs 1.96 ms ≈1.33x; M1 0.276 vs 0.348 ≈1.26x) — **but it FAILS the frozen
golden**: max_rel_err ≈1.05 (M64) / 1.30 (M512) / 0.98 (M2048); only M1 passes. Root cause: the frozen
captured weights are **shuffled for the asm 1-stage layout**, which the CK 2-stage kernels can't consume
→ garbage. **⇒ the aiter fmoe DB tune is NOT a drop-in win for asm-shuffled fp8-blockscale MoE weights.**
(Note: the unittest's section-3 "PASS" is tuned-cand-vs-tuned-base = both wrong identically; the real
gate is section-1 vs golden.) Restore the tuner file after; it's shared site-packages.

**Takeaway:** the ~1.3x raw gap proves compute headroom exists, but capturing it needs a path that
consumes the asm-shuffled operands correctly. No valid direct env/backend win → **author route**
(kernel_workflow, judged by the immutable unittest). ck absent (no ckProfiler) and CK-swap already
correctness-failed. flydsl (0.2.0, `flydsl_moe_stage1/2`) available but is a 2-stage path → likely the
same shuffle-layout obstacle; triton author is the portable first target. e2e-transfer: UNKNOWN.
