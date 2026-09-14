# SKILL — roofline headroom analysis

Turn the standardized Top-N into a per-kernel answer to: **"how much of this hardware's ceiling is this
kernel already getting, and therefore how much is left to win?"**

You (an agent) can execute this skill by reading this file. `roofline_tools.py` next to it holds
mechanical primitives (peak-table parsing, counter parsing, unit math) — use them if they import, do
the arithmetic yourself if they don't. **A broken helper must not disable this skill; a broken skill
must not fail the run.**

---

## 0. The one doctrine that matters

> `roofline_pct` measures **how well the kernel executes its current algorithm's byte/FLOP budget.**
> A kernel that already spends that budget at its class target has **no recoverable time left**, and
> an optimization budget spent on it buys nothing.

So this skill **gates**, it does not merely advise: `roofline_pct >= target_eff` (§7) removes the
kernel from both optimization tracks, however large its `pct_gpu_time`. The measured example in §9 is
exactly this case — a head owning 26.45% of GPU time, at 88% of its roof, whose optimization measured
**−0.064% e2e**. Ranking by `pct_gpu_time` alone spends the run on it every time.

**The gate fires only on a real verdict.** `roofline_pct` is a *model*, and §6/§8 catalogue the ways
that model is wrong. An entry that is `suspect`, `headroom_class: unknown`, or `confidence: low` is
NOT evidence of saturation and must never be skipped — it is re-measured (stage C) or ranked the
ordinary Amdahl way. Deleting the biggest kernel in the profile because a byte model was wrong is a
far worse failure than tuning a saturated one.

---

## 1. Inputs

- `profile/round_<R>/profile_topN.json` — the standardized Top-N (required).
- `env_report.json` — `gfx`, `model_arch_class`, `model_dtype`, `workload` (required).
- The model's `config.json` — layer count, expert count, hidden/intermediate sizes, head counts
  (optional but needed for a good MoE/attention byte model).
- `peaks.md` — hardware denominators, keyed by `gfx`.
- Optionally, captured shapes from the Kernel Extractor and/or rocprofv3 counters (stage B/C, §5).

## 2. Output artifact

Write `profile/round_<R>/profile_roofline.json` and a human-readable `profile_roofline.md` beside it.

```jsonc
{
  "skill": "roofline", "skill_version": "1",
  "gfx": "gfx950",
  "peaks": { "hbm_bw_bytes_s": 8.0e12, "flops": {"fp8": 5.0e15},
             "source": "table|derived", "confidence": "high|low" },
  "stage": "A|B|C",
  "entries": [{
    "name": "...", "short_name": "...",
    "pct_gpu_time": 26.45,              // COPIED verbatim from the Top-N, never recomputed
    "regime": "decode|prefill",
    "op_class": "gemm|moe|attn|elementwise|unknown",
    "modeled": true,
    "t_ms": 0.049471,                   // per-launch time for this regime
    "launches_per_step": 80,
    "bytes_est": 0, "flops_est": 0,
    "achieved_bw_bytes_s": 0, "achieved_flops": 0,
    "hbm_util": 0.88, "compute_util": 0.01,   // achieved/peak on each axis; the pair decides bound_type
    "arithmetic_intensity": 4.6, "ridge_point": 625.0,
    "bound_type": "memory|compute|latency|unknown",  // latency = neither roof near its ceiling
    "roofline_pct": 0.88,               // achieved / peak on the AI-selected roof
    "target_eff": 0.85,
    "attainable_speedup": 1.0,
    "expected_e2e_gain_pct": 0.0,
    "headroom_class": "underperforming|moderate|saturated|unknown",
    "confidence": "low|medium|high",
    "suspect": false,
    "skip_optimization": true,          // roofline_pct >= target_eff -> not optimized at all (§3 step 7)
    "skip_reason": "roofline 0.880 >= target_eff 0.850 -> no recoverable time; ...",
    "notes": "assumptions made, what would sharpen this"
  }],
  "ranking_by_pct": ["..."],            // BOTH rankings are emitted, side by side
  "ranking_by_expected_gain": ["..."],
  "degraded": [ {"name": "...", "reason": "..."} ],
  "skill_errors": []
}
```

`ranking_by_pct` and `ranking_by_expected_gain` are **both** emitted deliberately. The consumer must
see the disagreement rather than a single blended number that hides it.

## 3. Procedure

0. **Scope to the head.** Analyse ONLY entries with `pct_gpu_time >= HEAD_THRESHOLD_PCT` (default 2),
   biggest first, capped at ~8. Below that bar the Amdahl ceiling is under the noise band no matter
   what the roofline says, so a headroom estimate cannot change a decision — modelling those kernels
   only adds failure modes. Skipped entries are **absent** from the artifact; they are NOT `degraded[]`
   (a kernel too small to matter is not a modelling failure and must not read as one).
1. Resolve peaks for `gfx` from `peaks.md`. Not found → derive from device props, set
   `peaks.confidence="low"` (§6 L1).
2. For each selected entry, pick the **e2e-critical regime** — the one carrying the launches
   (`serving.n_decode_steps` vs `n_prefill_steps`; a decode-dominated run means decode). Use that
   regime's `base_latency_ms` as `t_ms`.
3. Classify `op_class` from `name`/`classification`/shapes.
4. Apply the §4 byte/FLOP model for that class. Cannot model it → §6 L2 (degrade this entry only).
5. Compute:
   ```
   achieved_bw    = bytes_est / t
   achieved_flops = flops_est / t
   hbm_util       = achieved_bw    / peak_bw
   compute_util   = achieved_flops / peak_flops
   AI             = flops_est / bytes_est
   ridge_point    = peak_flops / peak_bw

   # AI picks which roof the kernel walks TOWARD; roofline_pct is measured on that roof.
   roof_axis    = "compute" if AI > ridge_point else "memory"
   roofline_pct = compute_util if roof_axis == "compute" else hbm_util

   # But which roof actually BINDS is decided by utilization, not by AI alone. A small AI does NOT
   # by itself mean memory-bound — that is the most common mislabel. If neither roof is near its
   # ceiling (both utils < 0.60) and the launch is above the dispatch floor, the kernel is
   # LATENCY / occupancy-bound, not bandwidth- or compute-bound.
   bound_type   = "latency" if (hbm_util < 0.60 and compute_util < 0.60) else roof_axis

   attainable_speedup    = max(1.0, target_eff / roofline_pct)
   expected_e2e_gain_pct = pct_gpu_time × (1 − 1/attainable_speedup)
   ```
   A latency-bound kernel **still gets a headroom verdict** (its `roofline_pct` on the AI-selected
   roof is real, and `target_eff` already prices in the occupancy penalty for irregular classes like
   paged attention) — so a low-utilization head still ranks by its headroom. What changes is the
   **lever**: latency-bound underperformance is fixed by occupancy / shorter dependency chains /
   fusion, **not** by byte reduction. Byte reduction only helps a genuinely bandwidth-bound
   (high-`hbm_util`) head.
6. Classify headroom, banded against `target_eff` (**not** against the raw roofline — what matters is
   the distance to what a good implementation of this class can realistically reach):
   `roofline_pct ≥ 0.9×target_eff` → **saturated**; `≥ 0.6×target_eff` → **moderate**; else →
   **underperforming**; unmodelled or low-confidence → **unknown**.
   *80% against a 0.85 target is **saturated**, not "nearly there" — tuning has nothing left to give.*
   These three bands are a **description**, deliberately wider than the skip gate in step 7: the band
   `0.9×target ≤ roofline_pct < target` is "saturated but still optimized".

   **Two outcomes must produce NO verdict** (`headroom_class: "unknown"`), because in each the ratio
   is not evidence about the kernel:
   - **Dispatch-bound** — the per-launch time is within launch-overhead scale (~5 µs), so the launch is
     timed by dispatch, not by its transfer or its math. Emit `bound_type: "latency"`; the lever is
     fusion / graph capture, not kernel tuning. Typical of tiny high-call-count kernels. *This is the
     no-verdict sub-case of latency-bound* — distinct from the general latency-bound kernel in step 5
     (low utilization but well above the dispatch floor), which **keeps its verdict** because it is
     doing real work and has recoverable occupancy/dependency headroom.
   - **Infeasible** — `roofline_pct` outside `(0,1]`. That is the byte/FLOP model being wrong, not the
     kernel being at the wall. **A clamped 100% must NEVER be reported as `saturated`** — that turns a
     modelling failure into a routing decision. A compute-axis ratio above 1.0 is most often an
     **unvalidated peak** (the BF16 MFMA microbench commonly reads ~2× low; see §4). See §6 L3.

   `bound_type` is a CLOSED set: `memory | compute | latency | unknown`. If none fits, emit `unknown`
   — never invent a category the consumer has no routing rule for.
7. **Decide the skip gate** — this is the one field the consumer acts on rather than ranks on:
   ```
   skip_optimization = (headroom_class != "unknown") and (not suspect) and (roofline_pct >= target_eff)
   ```
   A kernel at or above its class target has no recoverable time left, so it is dropped from **both**
   optimization tracks (head and kernel) rather than reordered. Populate `skip_reason` when it fires.

   **The two guards are not optional.** Both no-verdict outcomes above already set
   `headroom_class: "unknown"`, and this gate must respect that — otherwise an infeasible model
   clamped to 1.0, or a `confidence: low` derived peak, deletes the largest kernel in the profile on
   the strength of a number §6 says is not evidence. A `low`-confidence entry (stage A, or peaks
   derived from device props) may be **displayed** but must NOT set `skip_optimization`; re-measure at
   stage C first. See §8: a "kernel at 85%" read off an unvalidated peak may really be at 43%.
8. Sanity-check (§6 L3), emit both rankings, write the artifact.

**Per-launch, not aggregate.** Compare bytes for ONE launch against ONE launch's `base_latency_ms`. If
a logical op is split across several launches (e.g. a fused-MoE layer issuing a stage-1 and a stage-2
kernel), sum the launches for one logical unit and compare against the summed time. Getting this
factor wrong is the single most common way to produce a nonsense `roofline_pct` — state in `notes`
which unit you used.

## 4. Byte / FLOP models by op class

Weight bytes use the **weight** dtype (fp8 = 1 B/elem, bf16 = 2 B). Activation bytes use the activation
dtype. Only count HBM traffic — a tensor re-read within one launch and small enough to sit in L2
(`l2_bytes`) counts once.

**Trust the memory axis over the compute axis, especially at decode.** The compute peaks in `peaks.md`
are validated for fp8, but empirical MFMA peaks are not always right — the BF16 microbench commonly
reads ~2× low, which makes a BF16 `compute_util` read ~2× high (and can push `roofline_pct` above 1.0,
where §6 L3 catches it as `suspect`). A decode workload is memory-bound anyway, so prefer `hbm_util`;
only rank on a compute-axis `roofline_pct` after the peak for that dtype has been validated (§8 rule:
BF16 and FP16 MFMA run at the same rate, so those two peaks must be equal — if they are not, the peak
is mis-calibrated and the compute-axis number is not usable).

### dense GEMM `[M,K]×[K,N]`
```
flops = 2·M·N·K
bytes = M·K·a + K·N·w + M·N·a            # A + B + C
```

### MoE / grouped expert GEMM
The decisive question is **how many expert weights are streamed**, which dominates at decode.
```
pairs        = M · top_k
experts_hit  = E · (1 − (1 − 1/E)^pairs)      # expected distinct experts touched
flops = 2 · pairs · (per-expert MAC count for this stage)
bytes ≈ experts_hit · (per-expert weight elems) · w   + activations
```
If the implementation streams **all** `E` experts regardless of routing, use `E` instead of
`experts_hit` — and note that the difference between the two IS an algorithmic lever (§7.1). When
unsure which the kernel does, compute both, report the `experts_hit` figure, and put the all-expert
figure in `notes`.

**Feasibility rule (general, applies to every op class with more than one plausible byte model).**
A byte estimate implying a rate above peak is refuted by the measurement itself. When you have
several candidate models, pick the **largest one that stays feasible** (`bytes ≤ peak_bw × t`) and say
which you used. If *every* candidate is infeasible, the class model is wrong: emit no verdict (§6 L3)
rather than clamping to 100% and calling it saturated.

### attention (paged decode)
KV traffic dominates; Q and the output are negligible at decode.
```
bytes ≈ batch · seq_len · n_kv_heads · head_dim · 2 (K and V) · kv_dtype_bytes
flops ≈ 2 · batch · n_q_heads · seq_len · head_dim · 2 (QK^T and PV)
```
`seq_len` is the *current* average context, not the max — for an isl/osl workload sampled mid-run,
`isl + osl/2` is a reasonable estimate. Say so in `notes`; it is a real source of error.

### elementwise / norm / quant
```
bytes = (input elems · in_dtype) + (output elems · out_dtype)
flops = small — assume memory-bound unless clearly otherwise
```

### unknown
Do not guess. Emit `modeled: false` and degrade this entry (§6 L2).

## 5. Confidence stages — the estimate sharpens as the run proceeds

| stage | source of shapes/bytes | confidence | consumer may |
|---|---|---|---|
| **A** profile time | `est_shape`/`shapes` from the Top-N, model `config.json` | `low` | display + annotate only — **do not rank on it** |
| **B** after extract | the REAL shapes/dtypes the Kernel Extractor captured for the unittest | `medium` | rank as a secondary key |
| **C** after op_bench | rocprofv3 counters on the isolated op (measured bytes/FLOPs) | `high` | rank as a secondary key; may be cited in the report |

Stage A is inherently coarse: at profile time the exact operand shapes have not been captured yet.
**Re-run this skill at stage B/C and overwrite the artifact.** Because the decision to spend the *next*
budget unit happens after the *previous* kernel's extract, refined numbers arrive in time to matter.

### Stage C — counter measurement (rocprofv3)
Measure on the ISOLATED op (the Op Benchmarker already has it isolated — no extra server run):
- bytes: `FETCH_SIZE` + `WRITE_SIZE` (both in **KiB**) → `(FETCH_SIZE + WRITE_SIZE) · 1024`
- FLOPs: `MfmaFlops`, or `MfmaFlopsBF16`/`F16`/`F32`/`F64` per dtype
- **fp8 has no `MfmaFlopsF8`** on current builds — use `SQ_INSTS_VALU_MFMA_MOPS_F8` and convert
  MFMA-ops → FLOPs for the instruction shape in play.
- corroborate `bound_type` with `MemUnitStalled`, `MfmaUtil`, `OccupancyPercent`.

Counter names and availability vary by ROCm build. Probe with `rocprofv3 --list-avail`; a missing
counter degrades to stage A/B (§6 L4), it does not fail the skill.

## 6. Degradation ladder — every level is non-fatal

| level | trigger | behavior |
|---|---|---|
| **L0** | `analysis_skill=none`, skill dir missing/unreadable | Emit nothing. Caller behaves exactly as before this feature existed. |
| **L1** | `gfx` absent from `peaks.md` | Derive peaks from device props; `peaks.confidence="low"` → every entry is `confidence: low` → display-only. |
| **L2** | an op class cannot be modelled | Degrade **that entry only**: `modeled:false`, `headroom_class:"unknown"`, add to `degraded[]`. Other entries are unaffected and the consumer falls back to the pre-skill prior for this one. |
| **L3** | result is impossible: `roofline_pct > 1.0`, or `< 0.001`, or a negative/zero byte count | Clamp for display, set `suspect:true`, **force `headroom_class:"unknown"`** (an infeasible ratio is not a verdict), keep `roofline_pct_raw`, emit `bytes_upper_bound = peak_bw × t` (what the model violated), and flag the entry as a **stage-C counter-measurement candidate**. |
| **L4** | counters unavailable / unstable | Keep the stage-A/B analytic result; do not raise confidence. |
| **L5** | anything else raises | Catch it, append to `skill_errors[]`, write whatever entries succeeded, and continue. **The run never fails because of this skill.** |

`roofline_pct > 1.0` is a real and expected occurrence — it usually means the byte model over-counts
(e.g. assuming all experts are streamed when the kernel skips unrouted ones). Treat it as a signal that
the model needs stage-C measurement, not as a hardware anomaly.

## 7. `target_eff` and how to route on the result

`target_eff` = how close a **well-implemented** kernel of this class gets to its roofline bound. It is
set by **access regularity**, not by how important the kernel is.

It is **also the skip bar** (§3 step 7): at or above it, the kernel is not optimized at all. So these
four numbers decide what a run works on, and moving one moves the whole budget.

| op class | `target_eff` | why |
|---|---|---|
| dense GEMM | **0.85** | regular, dense compute |
| MoE / grouped GEMM | **0.85** | decode weight streaming is essentially a memcpy |
| elementwise / norm / quant | **0.85** | pure streaming |
| attention decode (paged) | **0.60** | irregular paged KV access, occupancy-sensitive |

These are **priors, not constants** — §8 corrects them from observed outcomes.

### Routing table (the actual point of this skill)

| `headroom_class` | `bound_type` | `pct_gpu_time` | route |
|---|---|---|---|
| **any, with `roofline_pct ≥ target_eff`** | any | **any** | **SKIPPED — dropped from both tracks** (§3 step 7). Applies however large `pct_gpu_time` is |
| underperforming | memory / compute | high | **kernel track, top priority** — real micro-optimization headroom |
| underperforming | latency | high | kernel track — but the lever is **occupancy / dependency-chain / fusion / split-K**, not byte reduction |
| saturated (still below target) | memory | high | kernel track, low priority — little left, but not yet at the bar |
| saturated (still below target) | latency | high | occupancy / access pattern is the ceiling; fusion before more tuning |
| any | any | low | low priority (ordinary Amdahl) |
| unknown, `suspect`, or `confidence: low` | any | any | **never skipped** — fall back entirely to `pct_gpu_time` ordering |

**Shippability gate (applies before any of the above).** A head kernel with no editable call site —
a monolithic hand-written assembly `.co` or a precompiled CK `.so` — cannot take an in-kernel rewrite
no matter how much headroom it shows. Its only levers are **host-side**: dispatch/path selection, the
tuning DB (`tuned_fmoe`, `AITER_CONFIG_*`), or a backend swap. When the profiler marks an entry
non-editable, route it to the host-side track and say so; do not dispatch a rewrite that cannot be
integrated. (The `editable` flag comes from the profiler's Top-N, not from this skill.)

### 7.1 What a skipped kernel is NOT

`skip_optimization` means *this run does not spend budget here*. It does not mean the kernel is
optimal in an absolute sense: a kernel at its roofline still moves whatever bytes its **algorithm**
demands, and changing that algorithm (fusing an adjacent op away, not streaming experts routing never
touches, fp8→fp4 weights) can still beat it. That work is out of scope for this skill and for the
optimization tracks it feeds — it is an algorithmic change, not a kernel optimization, and it is
routed by the Architect on its own merits, not by a roofline number.

Record skipped heads in the artifact with their `skip_reason` rather than omitting them, so a run
that skipped its largest kernel says so plainly instead of looking like it found nothing.

**Hard constraints on every lever this skill routes to (do not violate):**
> The **measurement contract and output semantics are fixed.** The workload — `isl`, `osl`,
> **`conc` / batch size — is supplied by the user and must not be changed**. Do **not** introduce
> speculative decoding (MTP or otherwise) as an optimization. A lever that raises throughput by
> changing what is being measured is not a win. Lossy levers (fp4, kv-cache-dtype) must pass the
> accuracy gate before they count.

## 8. Guarding against being wrong

1. **Sanity band** — §6 L3.
2. **Validate the peak before believing a `roofline_pct`.** The peaks are empirical microbench
   results, not spec figures. The load-bearing cross-check: BF16 and FP16 MFMA run at the same rate on
   these parts, so their peaks must be equal — when they are not, the compute-axis number is inflated
   (a "kernel at 85%" may really be at 43%). **This is why the skip gate refuses `confidence: low`
   entries** — 85% is now a hard bar, and an inflated compute axis would skip a kernel with 2× of
   headroom left. Trivial streaming also tops out near ~0.85 of the HBM pin rate, which is why the
   memory `target_eff` is 0.85, not 1.0.
3. **Two noise bands, not one.** An **isolated-kernel** speedup is real only if it clears the
   isolated repeat band (**~3.4%** on identical reruns here — much wider than people assume), while an
   **e2e serving** delta uses the serving band (~0.5%). Do not judge an isolated kernel win against the
   e2e band, and never call a sub-3.4% isolated speedup real.
4. **Contradiction check.** If a kernel squad measures an isolated speedup **larger** than this skill's
   `attainable_speedup`, the model was wrong. Flag it, and prefer the measurement — always.
5. **Self-correction across runs.** Record `predicted vs actual` (predicted `attainable_speedup` and
   `expected_e2e_gain_pct` vs measured isolated speedup and measured e2e delta) into
   `knowledge/backend_playbook.md`, which already grows every run. Systematic error in a class's
   `target_eff` or byte model shows up there and is corrected in the next run.
6. **Never sole authority for ACCEPTING.** This skill prunes (§3 step 7), but it never accepts: no
   result is ever taken as a win because of a roofline number. The e2e gate decides that.

## 9. Worked example (real data — Qwen3.5-35B-A3B-FP8, gfx950, vLLM, TP1, isl/osl 1k, conc 64)

Decode-dominated (2033 decode steps vs 24 prefill). Peaks: 8.0 TB/s HBM, 5.0 PFLOP/s fp8.

**`fused_moe_kernel`** — `pct_gpu_time` 26.45%, decode `t` = 49.47 µs/launch, 80 launches/step over 40
layers ⇒ 2 launches per layer ⇒ one logical layer = 98.9 µs.
E=256, top_k=8, hidden=2048, moe_intermediate=512, fp8 weights. At M=64, `pairs`=512 ⇒
`experts_hit` ≈ 221/256. Layer weight bytes ≈ 697 MB (all-expert: 805 MB).
⇒ achieved 7.04 TB/s = **88% of roofline**, AI ≈ 4.6 ≪ ridge 625 ⇒ **memory-bound**.
⇒ 0.88 ≥ the 0.85 MoE target ⇒ `attainable_speedup` = **1.0×**, `expected_e2e_gain_pct` = **0.00%**,
**saturated**, and **`skip_optimization: true`** — the largest kernel in the profile is not optimized.
*(All-expert bytes give 102% — an L3 `suspect` case, which is therefore **not** skipped: the same
number arrived at through a broken byte model carries no verdict.)*

**`kernel_paged_attention_2d`** — `pct_gpu_time` 8.86%, decode `t` = 141.1 µs, 10 launches/step
(= the 10 full-attention layers of 40, `full_attention_interval`=4).
⇒ achieved ≈ 1.4–2.3 TB/s = **18–29% of roofline** ⇒ against the 0.60 target,
`attainable_speedup` ≈ **2.1–3.4×**, `expected_e2e_gain_pct` ≈ **+4.7–6.2%** → **underperforming**,
not skipped.

**Why this matters:** ranking by `pct_gpu_time` puts MoE first (26.45% vs 8.86%). Ranking by roofline
headroom puts attention first. The run that produced these numbers spent its budget on the MoE and
measured **−0.064% e2e**; the attention kernel it reached later yielded **1.56× isolated**. The gate
encodes that outcome: the MoE is skipped and the budget goes to attention.

**Two honest caveats on these priors** (§8 rule 5 — record predicted vs actual, do not launder it):
- The MoE's isolated speedup was **1.047×**, above the 1.0× a skipped kernel is credited with. By §8
  rule 4 that is the model being wrong, and on an isolated bench it was. It is skipped anyway because
  the number that decides is the e2e one, and that was **−0.064%** — a real isolated win that bought
  nothing served.
- Attention against a 0.60 target predicts **2.1–3.4×** where the squad measured **1.56×**. The prior
  is optimistic here, more so than the 0.50 it replaced. It is used to rank and to gate, not to
  promise a speedup; the isolated bench remains the judge of what was actually achieved.
