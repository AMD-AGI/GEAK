# Roofline Playbook — the hardware-anchored objective (approach the ceiling, then stop)

AVO's default scoring is **relative** ("faster than the running best"). That tells
you that you improved, never **how close to the hardware limit** you are, nor
**which ceiling** is holding you back. This playbook makes the objective
**absolute and hardware-anchored** so the lineage deliberately *approaches the
roofline* and stops when there is no real headroom left — the behavior that lets
the agent out-optimize a human expert instead of plateauing early.

Peak numbers come from
[`kernel_knowledge/hardware/cdna3_mi300/peak_tables.md`](../../kernel_knowledge/hardware/cdna3_mi300/peak_tables.md)
and are encoded once in `lib/roofline.mjs` (JS) ⇄ `scripts/avo_lineage.py` (Python).
**Keep those two in sync** if you edit the peak table.

## The model (per shape)

Given per-case telemetry `{latency_ms, flops, bytes, dtype}`:

- **Arithmetic intensity** `AI = flops / bytes` (FLOP/byte).
- **Roofline bound** `= min(peak_compute[dtype], AI × HBM_BW)` (attainable FLOP/s).
- **Achieved** `= flops / latency_s`; **achieved BW** `= bytes / latency_s`.
- **Fraction of roofline** `= achieved / roofline_bound` ∈ (0,1] — the headline number.
- **Ridge point** `AI* = peak_compute / HBM_BW` (fp16 ≈ 245 FLOP/B on MI300X).

MI300X anchors (gfx942): **fp16/bf16 1307 TF**, **fp8/int8 2615 TF**, fp32 163 TF,
**HBM3 5.325 TB/s**. Real kernels sustain ~45% of peak — treat ~45% as "good", and
only the last stretch toward the ceiling is the truly expert regime.

## Regime = the binding constraint (what to attack)

Let `mem_frac = achieved_BW/peak_BW`, `cmp_frac = achieved/peak_compute`.

| Regime | Condition | It means | Lever track |
|---|---|---|---|
| **basement** | `max(mem_frac,cmp_frac) < ~0.35` | far below BOTH ceilings → latency / occupancy / launch bound | `host_runtime` |
| **memory-bound** | not basement, `AI < ridge` | near the BW ceiling | `memory` |
| **compute-bound** | not basement, `AI ≥ ridge` | near the FLOP ceiling | `compute` |
| unknown | no flops/bytes telemetry | relative-only mode | `algorithm` |

The loop computes this for you and passes a `ROOFLINE` steer block
`{regime, fraction_of_roofline, worst_case, focus_case, specialty, levers}` into the
variation/supervisor prompts. **Attack the regime of the worst (most headroom) shape**,
not an already-saturated one.

## Levers per regime (ranked)

- **memory-bound** → fuse to cut HBM round-trips · vectorize+coalesce global
  loads/stores (128 B) · stage & reuse in LDS · XCD-aware tiling for L2 (4 MiB/XCD) +
  Infinity Cache (256 MiB) · reduce bytes (lower precision / recompute vs reload) ·
  async global→LDS copy to overlap.
- **compute-bound** → raise MFMA/matrix-core utilization (right MFMA shape) · cut
  VALU/packing/conversion around the MMA loop · balance register pressure vs occupancy
  (no spills) · software-pipeline to hide MFMA latency · remove LDS bank conflicts
  feeding the cores.
- **basement** → raise wave occupancy (grid/block, LDS & register budget) · collapse
  dispatches / fuse launches · double-buffer to hide load latency · persistent /
  CUDA-graph launch · rebalance work across all CUs / XCDs of this device (tail effects).

These map to the team specialist taxonomy (`algorithm | memory | compute |
host_runtime`), so an `operator=team` round can dispatch one specialist per regime.

## Counting FLOPs & bytes (so the regime is known)

The COMMANDMENT benchmark should emit `flops` and `bytes` per case. If it cannot,
the roofline signal is **inert** and AVO runs in pure relative mode — nothing breaks.

Generic recipes:
- **GEMM `M×N×K`**: `flops = 2·M·N·K`; `bytes ≈ (M·K + K·N + M·N)·dtype_size` (count
  what actually moves through HBM — a cache-resident operand may move once).
- **Attention** (`B,H,S,D`): `flops ≈ 4·B·H·S²·D` (two matmuls); `bytes` = Q/K/V/O
  traffic that misses cache (decode is KV-read-dominated → usually memory-bound).
- **Elementwise / norm / reduction**: `flops` ≈ ops·elements (small); `bytes` =
  in+out tensors → almost always **memory-bound**; AI ≪ ridge.
- **Conv / gather-scatter**: count real MACs ×2 for flops; effective bytes after
  reuse for bytes.

Use the *effective* bytes that cross HBM (post-cache), not the naïve sum, or you'll
mis-classify a cache-friendly kernel as memory-bound. When unsure, prefer
rocprof-measured bytes (see `kernel_knowledge/profiling/rocprofv3_counters.md`) over
analytic estimates.

## Stop rule (diminishing returns)

The loop stops early once `geomean(fraction_of_roofline) ≥ roofline_target`
(default 0.85) **and** `min_commits` is met. Below target it keeps going — there's
real headroom. With no telemetry it never roofline-stops (wall-clock budget governs).

## Anti-cheat (still non-negotiable)

A roofline % is only meaningful if `flops`/`bytes` describe the **real** problem.
Do not shrink shapes, drop precision silently, or skip work to inflate the fraction —
the independent verify re-measures latency and correctness is gated. Quote *achieved*,
never theoretical peak.
