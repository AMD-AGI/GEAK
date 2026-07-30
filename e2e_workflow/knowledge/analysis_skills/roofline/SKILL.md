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
> It says **nothing about whether that budget is necessary.**

A kernel at 90% of the memory roofline is not "done" — it is **done with kernel micro-tuning**. The
remaining lever is to move fewer bytes. Never read a high `roofline_pct` as "stop optimizing this
kernel", especially when it is the largest consumer of GPU time. See §7.

Corollary: this skill is **advisory**. It reorders and annotates. `pct_gpu_time` remains the primary
key and the measurement remains the judge. This skill may never prune a candidate.

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
    "arithmetic_intensity": 4.6, "ridge_point": 625.0,
    "bound_type": "memory|compute",
    "roofline_pct": 0.88,               // achieved / peak on the binding axis
    "target_eff": 0.90,
    "attainable_speedup": 1.023,
    "expected_e2e_gain_pct": 0.59,
    "headroom_class": "underperforming|moderate|saturated|unknown",
    "confidence": "low|medium|high",
    "suspect": false,
    "byte_reduction_levers": ["..."],   // populated only when headroom_class=saturated
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

0. **Scope to the head.** Analyse ONLY entries with `pct_gpu_time >= HEAD_THRESHOLD_PCT` (default 5),
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
   AI             = flops_est / bytes_est
   ridge_point    = peak_flops / peak_bw
   bound_type     = "compute" if AI > ridge_point else "memory"
   roofline_pct   = achieved_bw/peak_bw   (memory)   |   achieved_flops/peak_flops (compute)
   attainable_speedup    = max(1.0, target_eff / roofline_pct)
   expected_e2e_gain_pct = pct_gpu_time × (1 − 1/attainable_speedup)
   ```
6. Classify headroom, banded against `target_eff` (**not** against the raw roofline — what matters is
   the distance to what a good implementation of this class can realistically reach):
   `roofline_pct ≥ 0.9×target_eff` → **saturated**; `≥ 0.6×target_eff` → **moderate**; else →
   **underperforming**; unmodelled or low-confidence → **unknown**.
   *88% against a 0.90 target is **saturated**, not "nearly there" — tuning has nothing left to give.*

   **Two outcomes must produce NO verdict** (`headroom_class: "unknown"`), because in each the ratio
   is not evidence about the kernel:
   - **Dispatch-bound** — the per-launch time is within launch-overhead scale (~5 µs), so the launch is
     timed by dispatch, not by its transfer or its math. Emit `bound_type: "latency"`; the lever is
     fusion / graph capture, not kernel tuning. Typical of tiny high-call-count kernels.
   - **Infeasible** — `roofline_pct` outside `(0,1]`. That is the byte/FLOP model being wrong, not the
     kernel being at the wall. **A clamped 100% must NEVER be reported as `saturated`** — that turns a
     modelling failure into a routing decision. See §6 L3.

   `bound_type` is a CLOSED set: `memory | compute | latency | unknown`. If none fits, emit `unknown`
   — never invent a category the consumer has no routing rule for.
7. Sanity-check (§6 L3), emit both rankings, write the artifact.

**Per-launch, not aggregate.** Compare bytes for ONE launch against ONE launch's `base_latency_ms`. If
a logical op is split across several launches (e.g. a fused-MoE layer issuing a stage-1 and a stage-2
kernel), sum the launches for one logical unit and compare against the summed time. Getting this
factor wrong is the single most common way to produce a nonsense `roofline_pct` — state in `notes`
which unit you used.

## 4. Byte / FLOP models by op class

Weight bytes use the **weight** dtype (fp8 = 1 B/elem, bf16 = 2 B). Activation bytes use the activation
dtype. Only count HBM traffic — a tensor re-read within one launch and small enough to sit in L2
(`l2_bytes`) counts once.

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
`experts_hit` — and note that the difference between the two IS a byte-reduction lever (§7). When
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

| op class | `target_eff` | why |
|---|---|---|
| dense GEMM | **0.90** | regular, dense compute |
| MoE / grouped GEMM | **0.90** | decode weight streaming is essentially a memcpy |
| elementwise / norm / quant | 0.85–0.90 | pure streaming |
| attention decode (paged) | **0.50** | irregular paged KV access, occupancy-sensitive |

These are **priors, not constants** — §8 corrects them from observed outcomes.

### Routing table (the actual point of this skill)

| `headroom_class` | `pct_gpu_time` | route |
|---|---|---|
| underperforming | high | **kernel track, top priority** — real micro-optimization headroom |
| **saturated** | **high** | **byte-reduction / algorithmic track — NOT dropped** (§7.1) |
| any | low | low priority (ordinary Amdahl) |
| unknown, or `confidence: low` | any | **fall back entirely to `pct_gpu_time` ordering** |

### 7.1 Saturated + high `pct_gpu_time` → byte-reduction levers

The biggest kernel stays the biggest target; only the *class of fix* changes. Enumerate levers that
make the kernel move **fewer bytes for the same work**:

- **Fuse an adjacent op away.** If a prologue/epilogue kernel (activation quant, silu, a norm) appears
  separately in the Top-N, fusing it into the saturated kernel removes an entire activation round-trip
  — and removes that kernel's own GPU time too.
- **Stop reading what isn't used.** If the kernel streams all `E` experts but routing only touches
  `experts_hit`, the difference is pure waste.
- **Layout / packing.** Remove padding waste, improve coalescing, improve L2 reuse between stages of
  the same logical op.
- **Lower-precision weights** (e.g. fp8 → fp4). Directly cuts bytes. **Lossy — must pass the accuracy
  gate.**

**Hard constraints on every byte-reduction lever (do not violate):**
> The **measurement contract and output semantics are fixed.** The workload — `isl`, `osl`,
> **`conc` / batch size — is supplied by the user and must not be changed**. Do **not** introduce
> speculative decoding (MTP or otherwise) as an optimization. A lever that raises throughput by
> changing what is being measured is not a win. Lossy levers (fp4, kv-cache-dtype) must pass the
> accuracy gate before they count.

## 8. Guarding against being wrong

1. **Sanity band** — §6 L3.
2. **Contradiction check.** If a kernel squad measures an isolated speedup **larger** than this skill's
   `attainable_speedup`, the model was wrong. Flag it, and prefer the measurement — always.
3. **Self-correction across runs.** Record `predicted vs actual` (predicted `attainable_speedup` and
   `expected_e2e_gain_pct` vs measured isolated speedup and measured e2e delta) into
   `knowledge/backend_playbook.md`, which already grows every run. Systematic error in a class's
   `target_eff` or byte model shows up there and is corrected in the next run.
4. **Never sole authority.** No candidate is ever pruned because of this skill; no result is ever
   accepted because of it. The e2e gate decides.

## 9. Worked example (real data — Qwen3.5-35B-A3B-FP8, gfx950, vLLM, TP1, isl/osl 1k, conc 64)

Decode-dominated (2033 decode steps vs 24 prefill). Peaks: 8.0 TB/s HBM, 5.0 PFLOP/s fp8.

**`fused_moe_kernel`** — `pct_gpu_time` 26.45%, decode `t` = 49.47 µs/launch, 80 launches/step over 40
layers ⇒ 2 launches per layer ⇒ one logical layer = 98.9 µs.
E=256, top_k=8, hidden=2048, moe_intermediate=512, fp8 weights. At M=64, `pairs`=512 ⇒
`experts_hit` ≈ 221/256. Layer weight bytes ≈ 697 MB (all-expert: 805 MB).
⇒ achieved 7.04 TB/s = **88% of roofline**, AI ≈ 4.6 ≪ ridge 625 ⇒ **memory-bound**.
⇒ `attainable_speedup` = 0.90/0.88 = **1.023×**, `expected_e2e_gain_pct` = **+0.59%** → **saturated**.
*(All-expert bytes give 102% — an L3 `suspect` case, and evidence for the "stop reading unused
experts" lever.)*

**`kernel_paged_attention_2d`** — `pct_gpu_time` 8.86%, decode `t` = 141.1 µs, 10 launches/step
(= the 10 full-attention layers of 40, `full_attention_interval`=4).
⇒ achieved ≈ 1.4–2.3 TB/s = **18–29% of roofline** ⇒ `attainable_speedup` ≈ **1.7–2.8×**,
`expected_e2e_gain_pct` ≈ **+3.7%** → **underperforming**.

**Why this matters:** ranking by `pct_gpu_time` puts MoE first (26.45% vs 8.86%). Ranking by roofline
headroom puts attention first. The run that produced these numbers spent its budget on the MoE and
measured **−0.064% e2e** (isolated 1.047×, predicted 1.023× ✅); the attention kernel it reached later
yielded **1.56× isolated** (predicted 1.7× ✅). Both predictions were calibrated — and the MoE was
correctly identified as needing a byte-reduction lever, not another tuning pass.
