---
id: flydsl_rewrite_quantized_moe
title: Rewrite quantized GEMM / fused-MoE kernels into FlyDSL (Triton->FlyDSL, int4 W4A16 / fp8 blockscale)
kind: expert_skill
authors:
- hongtaom
scope: kernel
match:
  operator: fused_moe_grouped_gemm
  arch_class:
  - '*'
  gens:
  - gfx942
  dtypes:
  - int4_w4a16
  - fp8_e4m3_fnuz
  regimes:
  - prefill
  - decode
  from_backend: triton
  to_backend: flydsl
  profile_signature:
    op_name_regex: ''
    min_pct_gpu: 0.0
expects:
  isolated_speedup_min: 1.5
  e2e_delta_min_pct: 1.0
  parity: required
validation:
  status: validated
  last_verified: '2026-06-17'
  gpu: gfx942/MI300X
  model: Kimi-K2.6-int4-W4A16
  measured:
    isolated: 3.576
    e2e_pct: ''
    parity: pass
  artifact: /wekafs/hongtaom/kimi_k01_fused_moe_flydsl
role: advisory_prior
supersedes: []
---

## When to use
A hot kernel that is an **int4 (W4A16 GPTQ/AWQ) or fp8 (blockscale) quantized GEMM / fused-MoE grouped
GEMM** on MI300X (gfx942), where the live path is a Triton `@triton.jit` body (or a closed `.co`) and
Triton body micro-opts have hit their ceiling. The Triton int4-unpack loop is latency-bound at ~15–20%
HBM, so body-only tuning caps at ~1.1×; a full **FlyDSL rewrite** of the candidate reaches ~60–65% HBM at
decode and ~3.6–6.2× over the Triton golden (speedup grows with M). This is the lever when more
config/body tuning won't move the needle and the win requires a code-level authored core.

## Mechanism
Triton's per-element int4 unpack + dequant loop streams weight bytes inefficiently and stalls latency-bound
(~15–20% HBM), which is why body micro-opts cap ~1.1×. A FlyDSL kernel maps the same math onto a tuned
`compile_*` MFMA primitive (aiter's SOTA DSL on gfx942) that feeds HBM/MFMA far better — 60–65% HBM at
decode, crossing to compute-bound (AI > ridge) at large M. The numeric key that makes the port exact: for
int4 W4A16, FlyDSL int4 is **symmetric** `w_signed·scale` with `w_signed∈[−8,7]`, and GPTQ no-zp is
`(uint4−8)·scale`; since `w_signed = uint4−8`, **folding the constant zp=8 into the weight makes the two
identical**. Per-group zero-points aren't expressible symmetrically, so they split out as a small additive
correction. The win transfers to any quantized grouped-GEMM whose math reconciles onto an existing FlyDSL
primitive.

## Procedure
Deliver the rewrite by **editing `candidate.py`** (a MODIFIABLE file): `make_candidate(inp)` builds +
`flyc.compile(...)`s the FlyDSL kernel once (untimed setup) and returns a zero-arg callable that only
launches it (timed). No `@triton.jit` kernel, no harness edits; golden + inputs are unchanged so
correctness is judged identically.

1. **Locate the target**: dispatch path, golden, harness. Triton kernel → golden = the installed-original
   kernel (Triton-vs-Triton, identical inputs). `.co` kernel → golden = upstream fp32 PyTorch ref (see the
   sibling `rewrite-co-kernel-to-flydsl` skill). Record exact dispatch signature, real shapes per regime
   (decode small-M, prefill large-M), quant format (int4 W4A16 group_size? fp8 blockscale?), zero-point y/n.
2. **Deterministic correctness harness**: reuse the original input builder + golden, fixed seed. Oracle by
   golden type — Triton-vs-Triton → `torch.allclose(rtol=atol=2e-2)` AND cosine ≈ 0.999994; fp32 ref →
   cosine distance < 0.01 (NOT allclose). Mirror the 4-mode contract so the candidate is a drop-in.
3. **Same-session baseline**: measure original and candidate back-to-back in one process, best-of-3 ×
   100-iter (CUDA events / `run_perftest`), min over repeats. `speedup = original_ms / candidate_ms`. Never
   trust stale baselines (GPU drifts ~5% run-to-run).
4. **Map the math onto an existing FlyDSL primitive (the crux — do NOT write from scratch)**: closest
   `compile_*` in `${FLYDSL_ROOT}/kernels/`.
   - int4 W4A16 (GPTQ/AWQ) → `moe_gemm_2stage.compile_moe_gemm2` (`in_dtype="int4_bf16"`,
     `group_size=32`); fold zp=8 into the weight (see Mechanism). Per-group has_zp → split
     `(uint4−zp)·scale = (uint4−8)·scale + (8−zp)·scale` and add the correction as a small
     `[tokens,G]×[E,G,N]` torch grouped-matmul (document as not-fully-fused).
   - fp8 blockscale → `moe_blockscale_2stage.compile_moe_blockscale_gemm{1,2}`.
   - plain grouped GEMM (no SiLU/reduce) → reuse the MoE down-stage as a generic GEMM: `kernel_topk=1`,
     `kernel_tokens=M·top_k`, `doweight=True`, `accumulate=False`, pre-gather `A2[m]=A[m//top_k]`.
   - `mixed_moe_gemm_2stage` is **mxfp4** (symmetric `val·2^scale`), NOT GPTQ — don't use it for W4A16+zp.
5. **Wire layout + compile**: unpack packed int4 (even k = low nibble) → fold zp → int8 → `shuffle_weight`
   → `_pack_shuffled_int8_to_packed_int4_no_perm` (fp8: `shuffle_weight(w, layout=(16,16))`). Transpose
   scales to the kernel's layout (GPTQ `[E,N,G]`→`[E,G,N]` f32). Preallocate all buffers, bind at
   `flyc.compile`, reuse every call; use `torch.cuda.current_stream()` at call time (CUDA-graph safe); bump
   an ABI tag in `module_name` when signatures change so the JIT cache doesn't serve stale binary. If aiter
   is also imported, put the new FlyDSL first on `sys.path` and hide `flydsl` from `find_spec` *during*
   `import aiter`, else aiter can silently drop its HIP ops. Self-bootstrap the import path at the top of
   `candidate.py` from `${FLYDSL_ROOT}` (repo `kernels/`+`tests/`, py3.12 `build-fly/python_packages` — not
   a stale py3.10 build); correctness must PASS (cosine ≈ 0.999994) BEFORE benchmarking.
6. **Validate + bench**: recompute the oracle on ALL regimes after every change (reject regressions), then
   same-session bench decode (M=64) and prefill (M=2048…8192). Expect speedup to grow with M.
7. **Roofline with real HW counters**: `ProfilingAnalyzer(profiling_type="roofline")` (wraps
   `rocprof-compute profile --roof-only` + `analyze -b 4`), parse `hbm_util_emp_pct`, `perf_gflops`,
   `ai_hbm`, `bound`. Trust HW counters, not an analytic byte/FLOP model (analytic gave ~90% where rocprof
   showed ~65%).

## Knobs & pitfalls
- `group_size` (int4) must match the checkpoint (GPTQ/AWQ commonly 32/128). `in_dtype="int4_bf16"` is the
  symmetric path; the zp=8 fold is what makes no-zp GPTQ exact.
- The topk=1 generic-GEMM framing streams more weight bytes than `moe_align` → reported speedup is
  **conservative** vs a fully-fused MoE.
- FlyDSL kernels need the **repo source** checkout (new `value_attrs` API), not just `pip install flydsl`;
  a stale py3.10 build won't load under py3.12.
- ABI/JIT-cache staleness: bump `module_name` on signature change or you'll silently run an old binary.

## Do-no-harm notes
- Quote **same-session** ratios only; never a stale/hardcoded baseline (GPU drifts ~5%). A fast-but-wrong
  candidate scores 0 — correctness gates every change.
- Per-group has_zp correction and one-time pre-gather may be untimed / torch-side — **state which**, and
  that full fusion needs a FlyDSL kernel edit; don't claim a fully-fused number you didn't measure.
- Synthetic uniform routing overstates gains vs the decode-skewed real workload (ordering holds, absolute
  is optimistic) — report it.
- When not triggered (non-quantized op, non-MoE/GEMM, or no FlyDSL primitive reconciles the math), the
  skill is inert — the workflow falls back to the generic Triton path, no regression.

## Sources
- Recipe (canonical author copy in PerfSkills): `workflows/knowledge/rewrite_kernel_to_flydsl.md`
  (PerfSkills `flydsl-rewrite` branch, commit `d7e8df1`); role directions in
  `workflows/roles/engineer.md` (FlyDSL rewrite directions) and `workflows/roles/tech_lead.md`
  (FlyDSL full-rewrite direction).
- Worked example — Triton→FlyDSL int4 **W4A16** GPTQ/AWQ fused-MoE (`fused_moe_kernel_gptq_awq`, Kimi-K2.6):
  `${FLYDSL_EXAMPLES}/int4_w4a16_moe/` — `cand_flydsl_k1.py` (GPTQ→FlyDSL conversion + generic-GEMM trick),
  `kernel/test_harness_flydsl.py`, `run_roofline_k1.py`, `DELIVERY_FLYDSL.md` (3.6–6.2× vs Triton golden).
- Worked example — `.co`→FlyDSL fp8 **blockscale** fused-MoE (`fmoe_fp8_blockscale_g1u1`, Qwen3.5-122B):
  `${FLYDSL_EXAMPLES}/fp8_blockscale_moe/`; deep dive in sibling skill `rewrite-co-kernel-to-flydsl`.
- Related GEAK skill: `flydsl_fp8_gemm_playbook` (e2e down-proj bare-core bind) — this skill is its
  kernel-scope, broader-operator counterpart.
- Validation eval dir (artifact): `/wekafs/hongtaom/kimi_k01_fused_moe_flydsl/` — K1 `fused_moe_kernel_gptq_awq`
  (int4 W4A16, Kimi-K2.6, MI300X gfx942). Deployment-env same-session A/B: **geomean 3.62×** (5 UT cases),
  5.15× prefill, 5/5 correctness PASS (cosine ≈ 0.999994) — `result_flydsl.json`, `DELIVERY_FLYDSL.md`,
  rocprof roofline (FlyDSL 60–65% HBM vs Triton 13–18%) in `roofline_hw/`.
- Independent on-box reproduction (GEAK docker, py3.10 + FlyDSL `build-fly.py310stray`; golden reconstructed
  from the verbatim local `kernel/kernel_jit.orig.py` + a pure-torch `moe_align` shim, no vLLM install):
  5/5 correctness PASS, cosine 0.999994 (exact), candidate ms within ~1.6% of the recorded run; measured
  **geomean 8.75×** against this docker's **triton 3.6.0** golden — `onbox_golden_ab.py` / `onbox_golden_ab.json`
  / `onbox_verify_geak.py` / `VALIDATION_STATUS.md`. The win is robust and direction-consistent across triton
  versions; the *absolute* ratio depends on the Triton baseline, so 3.62× (deployment-representative) is the
  conservative figure recorded in `validation.measured.isolated`.
