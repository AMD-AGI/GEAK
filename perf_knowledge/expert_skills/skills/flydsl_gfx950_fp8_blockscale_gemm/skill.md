---
id: flydsl_gfx950_fp8_blockscale_gemm
title: Rewrite CK fp8 a8w8 blockscale GEMM to FlyDSL on gfx950 (software fp32 post-MFMA scale, NOT E8M0; shape-dependent perf lever — XCD swizzle on wide-N, 8-wave cluster on narrow-N)
kind: expert_skill
authors:
- yz
scope: kernel
match:
  operator: dense_gemm
  arch_class:
  - '*'
  gens:
  - gfx950
  dtypes:
  - fp8_e4m3
  - fp8_e4m3_fnuz
  regimes:
  - prefill
  - decode
  from_backend: ck
  to_backend: flydsl
  profile_signature:
    op_name_regex: "kernel_gemm_xdl.*blockscale|gemm_a8w8_blockscale"
    min_pct_gpu: 0.0
expects:
  isolated_speedup_min: 1.05
  e2e_delta_min_pct: 1.0
  parity: required
validation:
  status: validated
  last_verified: '2026-07-16 (wide-N q_up), 2026-07-17 (narrow-N qkv)'
  gpu: 'MI355X / gfx950 (device 0x75a3)'
  model: 'claude (kernel_workflow expert-skill verify run)'
  measured:
    isolated: 'wide-N q_up_proj M=4096 N=65536 K=1536 (4-wave preshuffle + xcd8): 1.226-1.237x same-session interleaved A/B vs production CK (gemm_a8w8_blockscale_bpreshuffle), xcd0 ablation 0.973x (loses to CK) confirming the XCD lever. narrow-N qkv_proj M=4096 N=2048 K=7168 (8-wave ping-pong cluster, BLOCK_M=128/BLOCK_N=256): director-verified 1.05x, accepted (interleaved A/B 1.0513/1.0500/1.0462, all >1.0); xcd_swizzle no-op and 4-wave -19% both confirmed for this shape'
    e2e_pct: ''
    parity: 'wide-N: err=0 / cos=1.0 seeds 0-3 vs fp32 dequant oracle (rtol=atol=1e-2). narrow-N: cos=1.0, maxabs_err=0.03125 (identical to CK), checkAllclose pass, 0 elements out of tol'
  artifact: 'wide-N: /sgl-workspace/GEAK_DSV4Pro_Out/geak_exp/verify_skill/q_up_proj/manual_verify/ (RESULT.md, verify_ab.py, fly_loader.py). narrow-N: /sgl-workspace/GEAK_DSV4Pro_Out/geak_exp/verify_skill/qkv_proj/team_ck_gemm_a8w8_blockscale_20260717_041854_1474934_29166/ck_gemm_a8w8_blockscale/ (director_validation.json, tech_lead_report.md, final_patch.diff)'
  notes: 'Cross-machine reproduction, BOTH regimes now verified. wide-N/4-wave+xcd8 (q_up) and narrow-N/8-wave cluster (qkv) are both parity-clean and beat production CK. FlyDSL API drift confirmed as documented (kernels.mma -> kernels.common.mma; xcd_swizzle compile-arg is an uncommitted patch on mainline -> used the byte-frozen snapshot core via a module-alias shim). IMPORTANT install note: the blockscale software-promote cores live in the STANDALONE /sgl-workspace/FlyDSL build (kernels/gemm/fp8_gemm_8wave_blockscale.py::compile_fp8_gemm_8w_blockscale), NOT the aiter-embedded flydsl (aiter/aiter/ops/flydsl = rowscale/epilogue only: cos~0.999 but ~78% elements out of tol, a dead-end, not a blockscale drop-in). Point ports at the standalone build.'
role: advisory_prior
supersedes: []
---

## When to use
Trigger on the **problem signature, not a specific shape**: *any* **CK a8w8 fp8 block-scale** dense GEMM
(`weight_block_size [128,128]`, arbitrary fp32 per-`[128,128]` block scales, bf16 out; the profiled hotspot
name matches `kernel_gemm_xdl*blockscale` / `gemm_a8w8_blockscale`) being rewritten
**CK → FlyDSL** on **MI355X / gfx950 (CDNA4, OCP fp8_e4m3)**, where the FlyDSL **native** fp8 scaled-MFMA
path fails parity against the CK oracle (the symptom that blocks the port) and CK config-JSON / tuned-CSV
tiling is already near its ceiling (~1.0–1.06×). It is **one kernel over arbitrary M/N/K** — whatever shape
hits this signature triggers the recipe; nothing is special-cased or enumerated.

Selection is by **shape, never by layer/function name.** The kernel only sees an `(M, N, K)` + quant
layout; the model-side labels below (`q_up_proj`, `down_proj`, …) are just *where each shape was measured*
and carry **zero** weight in matching or recipe choice — the same `M/N/K` in any other model gets the same
recipe. The rows are illustrative instances seen **so far**, **not** a closed set; they span the regimes,
which is why the tile re-sweep (not the mechanism) matters per shape:

| shape class (drives selection) | M | N | K | seen as (model label — NOT a selector) |
|---|---|---|---|---|
| wide-N, 512 MiB C | 4096 | 65536 | 1536 | `q_up_proj` |
| narrow-N, deep-K | 4096 | 2048 | 7168 | `qkv_proj` |
| narrow-N, deep-K, big-M | 16384 | 1536 | 7168 | `gate_up_proj` |
| K-light, 235 MiB C | 16384 | 7168 | 768 | `down_proj` |

(…and any future / decode `-m`/`-nk` variant of the same operator). Two stages, and only the first is
universal: **(1) the parity fix** — do the block-scale **in software** (never E8M0) — is the *same for
every shape* and is what unblocks the port; **(2) the perf lever is shape-dependent** — wide-N/large-C
shapes win with a 4-wave core + **XCD swizzle**, narrow-N/deep-K shapes win with an **8-wave "cluster"**
core where XCD is a no-op. Three shapes are already tuned end-to-end; use their configs as the per-regime
starting point (see **Per-shape recipes** below) rather than sweeping blind.

## Mechanism
### A. Universal — the parity fix (same for every shape; this is what unblocks the port)
**The descriptive *why* now lives as a default-available fact card** (read by author / tech-lead **without**
`use_expert_skills`): `operators/dense_gemm/numerics.md` → *"fp8 a8w8 block-scale: arbitrary fp32 ≠ E8M0"*,
plus `quantization/block_scaling_mxfp.md`. In one line: CK a8w8 carries **arbitrary fp32** per-`[128,128]`
block scales, but the HW scaled-MFMA scale is **E8M0 (power-of-two only)** and cannot represent them —
folding CK's scale into `mfma_scale_*_f8f6f4` **silently rounds it → parity fails**. It is *representational*,
not a knob, which is why CK→FlyDSL auto-ports on this family fail *unless* they pick the software path below.

**The parity key (the one universal action):** keep the MFMA **unscaled low-precision fp8** and apply the
fp32 block scales **in software after** the MFMA — promote the fp8 partials to fp32, ×per-block fp32 scale,
accumulate. Bit-comparable to CK (**err=0, cos=1.0 across seeds**), and *irreducible* (the scale changes
every 128 in K → one promote per K-block). FlyDSL already ships **two** cores that do exactly this software
promote — pick one by shape (§ Per-shape recipes):
- a **4-wave blockscale-preshuffle** fp8 GEMM core;
- an **8-wave ping-pong** fp8 blockscale GEMM core.

Both pin the E8M0 HW scale to `1.0` and do the fp32 scale in software, so **no target-backend change is
needed** — the fix is choosing a software-scale core over the native scaled-MFMA. Locate the equivalents
**by behaviour** in whatever FlyDSL build is present; do not hardcode their paths or symbols.

### B. Shape-dependent — the perf lever (do NOT assume q_up's answer transfers)
The correctness fix is universal; **which perf lever wins inverts with the shape**, because the two shape
classes are bound by different things. Measured on the three tuned shapes:

- **Wide-N / large-C / cold-tail** (`q_up` N=65536/512 MiB C, `down_proj` N=7168/235 MiB C): bound by **L2 /
  fabric locality**, so **XCD-aware grid rasterization is the lever**. MI355X has **8 XCDs**, each with its
  own L2 slice; remapping CTA `(bx,by)` so CTAs sharing a B/N-panel land on one XCD lifts L2 hit-rate
  (`74.8%→83.0%` on q_up) / cuts fabric read-latency (`1030→803 cyc` on down_proj). Worth **~1.23×** and
  *load-bearing*: `xcd_swizzle=0` → **loses to CK (0.96×)**; `=8` → 1.11–1.36×. Here **8-wave spills**
  (wide-N accumulator blows the VGPR file). GEAK's gate_up auto-win picked the right kernel but **left
  `xcd_swizzle` off → only ~1.02×.**
- **Narrow-N / deep-K / tiny-C** (`qkv` N=2048/16 MiB C/K=7168): grid+LDS-bound at **1 block/CU, 2
  waves/SIMD**, no cold tail → **`xcd_swizzle` is a measured no-op** and the 4-wave core is ~19% slower.
  The win comes from the **8-wave ping-pong** kernel with the **`cluster` schedule** — move all four fp32
  promotes to *after* the last MFMA of the K-iter so none sits on a barrier critical path (MFMA util
  `45.8%→47.9%`). 16-wave (higher occupancy) *loses* (register wall → spill), and `BLOCK_M=256` is 14×
  slower (only 128 CTAs for 256 CUs).

## Per-shape recipes (validated winners)
Three shapes are tuned end-to-end (err=0 / cos=1.0 all). **Selection is by shape class, not by the layer
label** — match your `(M, N, K)` to the nearest row via the decision guide, apply its config, then re-sweep
only the shape-dependent knobs. The `seen as` column is provenance only and never a selector.

**Decision guide (pick the kernel family purely from the shape):**
- **Wide-N (N ≳ 4096) or large C output or a cold-input tail** → **4-wave blockscale-preshuffle core**,
  set **`xcd_swizzle=8`**, tile `t64×256×128 wpe2`. (XCD is the lever; 8-wave will spill.)
- **Narrow-N (N ≲ 2048) + deep-K + tiny C output** → **8-wave ping-pong core with the `cluster` schedule**,
  `BLOCK_M=128, BLOCK_N=256`. (8-wave hides the promote; `xcd_swizzle` is a no-op; don't bother with it.)

| shape M×N×K (drives selection) | bind | winning FlyDSL core | config | measured vs CK | decisive lever | what LOSES here | seen as |
|---|---|---|---|---|---|---|---|
| 4096×65536×1536 (wide-N, 512 MiB C) | L2-locality | 4-wave blockscale-preshuffle | `t64×256×128, wpe2, xcd8` | **1.11× hot / 1.15× cold / 1.18× ev** | XCD swizzle (`xcd0`→`xcd8` = 1.23×) | 8-wave (spills 0.14–0.72×), bigger tile/`wpe≥3` (spill), `fused_promote` (−1.6%) | `q_up_proj` |
| 16384×7168×768 (K-light, 235 MiB C) | cold-tail | 4-wave blockscale-preshuffle | `t64×256×128, wpe2, xcd8, fused_promote` | **1.4541× Director-verified** (`0.187491→0.128937 ms`) | XCD swizzle + `fused_promote`; second run `1.4568×` | `wpe≥3` (spill 4–5×); *8-wave is the top **open** lever here (~1.3× more), port unfinished* | `down_proj` |
| 4096×2048×7168 (narrow-N, deep-K, 16 MiB C) | LDS/grid | 8-wave ping-pong (`cluster`) | `BLOCK_M=128, BLOCK_N=256` (promotes after last MFMA) | **1.045× hot / 1.086–1.10× interleaved** | 8-wave + `cluster` promote-reorder (MFMA util 45.8→47.9%) | **XCD swizzle (no-op)**, 4-wave core (−19%), 16-wave (spill 1.35× slower), `BLOCK_M=256` (14× slower) | `qkv_proj` |

Notes: the narrow-N/deep-K shape (`4096×2048×7168`) is the **marginal-speedup** case — it sits ~1.05–1.10×
because it is already near its promote-free *rowscale* ceiling (~1.37×, a different numeric scheme, not a
drop-in); treat ~1.05× as the **family floor**, not a regression. An untuned shape such as `16384×1536×7168`
(narrow-N + deep-K, happens to be `gate_up_proj`) → start from the **8-wave `cluster`** recipe
**by its shape class**, not from the wide-N one — again a shape decision, not a name decision.

## Bottleneck diagnosis (validated method — think before you sweep)
For a new/untuned shape, work out *what* limits it before touching knobs. These moves were validated across
the three shapes and repeatedly **corrected wrong reads** — they are the transferable part, more than any
single config:

1. **Ablation is ground truth; util% is not.** To test "is X the bottleneck?", *remove X and re-measure* —
   never trust a utilization counter alone. On the wide-N shape rocprof showed VALU≈MFMA (~30%) and *looked*
   promote/VALU-bound, but deleting the entire per-block promote bought only **~1.7%** → it was hidden under
   stalls all along. **util% ≠ boundedness.** Confirm every suspected bottleneck by ablation before optimizing it.
2. **Build the promote-free "ceiling twin".** Compare against an identical-MFMA-work kernel that scales
   cheaply (a rowscale epilogue, or a plain fp8 GEMM). Same `v_mfma` count / same MFMA-busy-cycles but no
   per-block promote isolates the software-scale tax: on narrow-N the blockscale core ran **45.8% MFMA-busy
   vs the twin's 66.9%** at bit-identical MFMA work → the gap *is* the promote-on-critical-path tax (not
   compute, not memory), and the twin's ~1.37× is the true ceiling (different numerics, not a drop-in). The
   plain-GEMM floor gives the shape's intrinsic limit.
3. **Latency- vs throughput-bound: read both roofs at once.** If compute-util **and** HBM-BW-util are *both*
   low simultaneously (e.g. ~25% MFMA + ~22% HBM on the K-light shape), the kernel is **latency-bound** —
   idling, not out of compute or bandwidth (a throughput-bound kernel must saturate *one* roof). The
   FLOP-roofline "headroom" is then fictional; the real ceiling is the occupancy-latency one.
4. **Quantify the reachable ceiling: `MFMA-duty ≈ occupancy × per-wave-MFMA-fraction`.** Split wave lifetime
   with the exact identity `wave_cycles = active + wait_any + wait_inst` (holds to ~0.01%). If occupancy is
   register-capped at 2 waves/SIMD and per-wave MFMA-exec ≈16%, the register-limited MFU ceiling ≈ **~31.6%**
   — and you may already be at ~87% of it, so no in-kernel lever can help much. This tells you *whether a lever
   can even exist* before you spend time on it.
5. **Localize with identical-work counters.** Hold MFMA work fixed and vary one thing. `MemUnitStalled=0`
   kills the memory hypothesis; `AccVGPR=0` / no spill / LDS-bound occupancy kills the register hypothesis;
   equal MFMA-busy-cycles + higher *total* cycles pins the tax to scheduling/stalls, not work.
6. **Phase-decompose via a K-sweep.** Fit `time ≈ fixed + slope·nKtiles`. A dominant mainloop → target it
   (8-wave / scheduling); a fixed cost already near its C-write BW floor → epilog-overlap is low-ROI.
7. **Only levers on the *measured* critical path help.** Narrow-N was promote-on-barrier-path → the win was
   *scheduling* (cluster the promotes off the barrier); memory levers (scale-prefetch) were measured no-ops
   because `MemUnitStalled=0`. Never apply a memory lever to a latency/scheduling problem, or occupancy to a
   register-capped one.
8. **Know when to stop.** Once the winner is within a couple % of the ablation / ceiling-twin number (all
   three shapes are), stop kernel-level tuning — the only lever left is **E2E epilog fusion** (cut the C
   round-trip), which is invisible in a standalone kernel and only bankable in the model graph.

**Honesty caveats.** Some byte counters (e.g. TCC/L2) can deadlock the queue on some ROCm builds → model the
traffic instead and check robustness to ~2× error. GPU drifts ~5% run-to-run → quote **same-session
interleaved A/B only**, and correctness gates every candidate (a fast-but-wrong kernel scores 0).

## Procedure
1. **Locate & oracle.** Dispatch = the CK a8w8 blockscale (b-preshuffle) GEMM being replaced. Immutable
   oracle = fp32 dequant→matmul→bf16, compared with allclose `rtol=atol=1e-2`. Record real shapes per
   regime (prefill large-M, decode small-M); quant = fp8 a8w8 blockscale `[128,128]`.
2. **Pick the FlyDSL core by shape (decision guide above) — do NOT hand-roll, do NOT use scaled-MFMA.**
   Wide-N/large-C → the **4-wave blockscale-preshuffle** core; narrow-N/deep-K → the **8-wave ping-pong**
   core with the `cluster` schedule. Locate them **by behaviour** (a software-scale fp8 blockscale GEMM) in
   the FlyDSL build that is present — never hardcode a path or symbol. Compile once at setup (untimed),
   return a zero-arg launch callable (timed). 4-wave settings: out dtype bf16, scale block-K = 128, async
   global copy on, cshuffle epilog on.
3. **Scale = software fp32 post-MFMA (the parity key).** Both kernels already promote+scale after the MFMA.
   Never switch to `mfma_scale_*_f8f6f4` for arbitrary fp32 block scales.
4. **Apply the shape's perf lever (conditional — do not blind-apply q_up's).**
   - 4-wave path (wide-N): set **`xcd_swizzle=8`** (MI355X = 8 XCDs) — the single most important knob here;
     `0` = loses to CK. This is what the gate_up auto-win missed. Add `fused_promote=True` for K-light
     shapes like `down_proj` (~1.03×).
   - 8-wave path (narrow-N): ensure the **`cluster`** schedule (all four promotes after the last MFMA);
     **skip `xcd_swizzle`** (measured no-op) and do NOT raise occupancy to 16-wave (spills).
5. **Tile / config = start from the nearest recipe row, then re-sweep only shape-dependent knobs.**
   4-wave: `tile_m=64, tile_n=256, tile_k=128, waves_per_eu=2`; only these move the needle —
   `tile_n∈{128,256}` (256 wins, better B reuse), `xcd_swizzle∈{4,8}` (8 wins), `waves_per_eu∈{0,1,2}`
   (≈ equal — occupancy is NOT the lever); everything larger spills (see pitfalls). 8-wave: `BLOCK_M=128,
   BLOCK_N=256` is the only viable tile (`BLOCK_N` is hardwired to 256 by scale alignment; `BLOCK_M=256`
   idles half the CUs).
6. **Feed layout (must match the oracle).** Preshuffle B with the CK-matching 16×16 weight layout; flatten
   the A-scale as transposed contiguous `[scale_k, M]` and the B-scale as contiguous `[scale_n, scale_k]`.
   Preallocate all buffers at compile, reuse them every call, and launch on the current CUDA stream
   (CUDA-graph safe).
7. **Correctness FIRST (err=0 / within `rtol=atol=1e-2` vs the CK oracle), THEN same-session A/B** vs the
   CK baseline (best-of-3 × N-iter, min over repeats). `speedup = ck_ms / flydsl_ms`.

## Knobs & pitfalls
- **Only the *correctness* mechanism is universal — the perf answer is not "one kernel, one config".**
  The software fp32 scale transfers to every shape; the *kernel family, the lever, and the tile* do **not**.
  Wide-N (`q_up`, `down_proj`) → 4-wave `t64×256×128 wpe2` + `xcd8`; narrow-N/deep-K (`qkv`, and
  `gate_up` N=1536) → 8-wave `cluster` `BLOCK_M=128×256`. **Never assume q_up's answer transfers** — on
  `qkv` the winning kernel, the tile, *and* the XCD verdict all flip.
- **`xcd_swizzle` is shape-conditional, NOT universal.** On wide-N/large-C it is the decisive lever: `8`
  wins on MI355X (8 XCDs), `4` close, `0` = off (**loses to CK, 0.96×**), `16` too coarse (0.93×); bijective
  tile remap → bit-exact, always safe to try. On narrow-N/tiny-C (`qkv`, 16 MiB C, LDS/grid-bound, no cold
  tail) it is a **measured no-op** — don't waste a sweep axis on it there.
- **8-wave ping-pong WINS or LOSES depending on N — this is the sharpest shape trap.**
  - **Narrow-N/deep-K (`qkv`): 8-wave `cluster` is the winner** (4-wave is ~19% slower). Keep the `cluster`
    promote-reorder; do NOT go 16-wave (register wall → spill, 1.35× slower).
  - **Wide-N (`q_up`): 8-wave LOSES (0.14–0.72×)** — the 256×256 latency-hiding tile needs the unscaled MFMA
    partial live *simultaneously* with the promoted frag → blows the VGPR budget → MFMA serialises. Use the
    4-wave core instead. (`down_proj` is the exception where 8-wave is a promising *open* lever — its port
    isn't finished, so its shipped winner is still 4-wave+xcd+`fused_promote`.)
- **The 4-wave register-pressure wall is sharp.** Two-level accumulator + register-resident double-buffered
  B leave ~no VGPR headroom at the 2-wave budget. Anything bigger than `t64×256×128 wpe2` — larger
  `tile_m/tile_n/tile_k`, or `waves_per_eu≥3` — **spills to scratch → 3–10× slower** (measured
  `t128×256`=3057µs, `wpe3`=3683µs). `fuse_promote` on wide-N is −1.6% (helps only K-light `down_proj`).
  Don't chase bigger tiles / higher occupancy without HK-style explicit register allocation.
- **FlyDSL API drift**: installed FlyDSL builds sometimes change internal helpers (e.g. a `crd2idx` /
  `.ir_value()` signature), so a small compat shim in the harness may be needed. Prefer a source checkout
  matching the core you author against.
- **fnuz vs OCP fp8**: gfx950 is OCP e4m3; a CK kernel may be templated `f8_fnuz_t` while still passing
  parity — verify the intended OCP path isn't costing accuracy headroom (not a perf lever by itself).

## Do-no-harm notes
- **Never** use the native E8M0 scaled-MFMA (`mfma_scale_*_f8f6f4`) for arbitrary fp32 block scales — it
  *silently* loses precision (parity fail). This is the whole reason the port needs this recipe.
- **Measurement is the final word.** Correctness gates every candidate; a fast-but-wrong kernel scores 0.
  Quote same-session ratios only (GPU drifts ~5% run-to-run).
- **Kernel-level ceiling is shape-set, and it's low — calibrate expectations by shape.** Measured range:
  K-light **1.4541×**, wide-N **~1.1–1.2×** (winner within ~1.7% of the promote-free floor), narrow-N/deep-K
  **~1.05–1.10×** (already near its promote-free *rowscale* ceiling). All are stall/locality-bound at
  ~25–36% of the 5 PFLOP fp8 peak, so the software promote is *not* the bottleneck. The one lever left for
  every shape is **E2E epilog fusion** (cut the C round-trip — invisible in a standalone kernel; on the
  K-light shape, an 8-wave-blockscale port is also an open kernel-level lever). Don't burn budget on
  tile/occupancy/prefetch/VGPR/numerics micro-opts (all measured <2%).
- **Inert when not triggered.** Non-blockscale op, non-gfx950 box, or the FlyDSL blockscale entrypoint
  absent → fall back to the generic path, no regression.

## Sources
The evidence lives in external artifacts (manual FlyDSL-rewrite handoffs + a GEAK auto-run) and is cited for
**provenance only** — GEAK does not depend on those trees, and exact commits / file paths are intentionally
left out. The portable knowledge is the shapes, configs, and measured numbers below.

- **q_up_proj manual rewrite (primary evidence)** — M=4096,N=65536,K=1536; **err=0 / cos=1.0**; **1.114× hot
  / 1.148× cold / 1.179× CUDA-event median**; winner `t64×256×128, wpe2, xcd8`; XCD lever `xcd0`=0.958× vs
  `xcd8`=1.179× (**1.23×**); L2 74.8%→83.0%; kernel-floor ablation puts the winner within ~1.7% of the
  promote-free floor. Core = 4-wave blockscale-preshuffle + `xcd_swizzle`.
- **GEAK gate_up auto-run (independent confirmation of the scale bypass)** — the run chose the 4-wave
  software-scale core at `t64×256×128` (parity clean) but **omitted `xcd_swizzle`** → only ~1.02×. Shows the
  bypass works *and* why XCD swizzle is mandatory to actually win.
- **down_proj workflow validation (Director accepted; artifact case `mlp_down_m16384_n7168_k768`)** —
  **1.4541×** (`0.187491 -> 0.128937 ms`) with a second run at **1.4568×**. Correctness passed:
  FlyDSL `maxabs_err=0.0078`, `cos=1.0`. Winner `t64×256×128, wpe2, xcd8, fused_promote`;
  same 4-wave core as q_up. Open lever: 8-wave-blockscale port (~1.3× more, unfinished).
- **qkv_proj manual rewrite (narrow-N 8-wave `cluster` winner — where the levers invert)** —
  M=4096,N=2048,K=7168; **err=0 / cos=1.0** (24/24 seeds×shapes); **1.045× hot / 1.086–1.10× interleaved**.
  Winner = 8-wave ping-pong core with the **`cluster`** promote-reorder (all 4 promotes after last MFMA →
  MFMA util 45.8%→47.9%). **Proof the levers flip:** `xcd_swizzle` / `waves_per_eu` are no-ops here, the
  4-wave core is 19% slower, 16-wave spills. Rowscale twin = 1.37× ceiling (different numerics, not a drop-in).
- **Target backend (FlyDSL) note** — both cores already implement the software fp32 promote (E8M0 HW scale
  pinned to 1.0) and expose the `xcd_swizzle` / `fused_promote` knobs, so **no FlyDSL change is required**;
  select the right core by behaviour rather than modifying or path-referencing the target backend.
- AMD occupancy blog — XCD-aware swizzling on MI355X:
  https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html
