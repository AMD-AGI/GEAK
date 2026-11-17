---
id: flydsl_decode_op_authoring
title: "Write or optimize ONE decode operator in FlyDSL (GEMV, norm, RoPE / KV write, attention, MLA, router, MoE experts, indexer) at 1-8 rows on MI300X (gfx942)"
kind: expert_skill
authors: [GEAK Team]
scope: kernel
match:
  operator: [skinny_gemv_decode, scaled_quant_gemm, dense_gemm, rmsnorm, fused_add_rmsnorm,
             fused_norm_quant, rope, kv_cache_quant, attention_decode_paged, gqa_mqa_attention,
             sliding_window_attention, mla_attention, sparse_attention_nsa, moe_routing_topk,
             fused_moe_grouped_gemm, grouped_gemm_moe, linear_attention_gated_delta]
  arch_class: ['*']
  gens: [gfx942]
  dtypes: [bf16, fp8_e4m3_fnuz, mxfp4]
  regimes: [decode, graph]     # real task metadata often says regime: graph (CUDA-graph replay of a decode step)
  from_backend: ""
  to_backend: flydsl
expects:
  isolated_speedup_min: 1.0    # the first goal is a correct starting point; the optimize loop then tunes it
  parity: required
validation:
  status: draft
  last_verified: ""
  gpu: ""
  model: ""
  measured: {isolated: "", e2e_pct: "", parity: ""}
  artifact: ""
role: advisory_prior
supersedes: []
---

## When to use
**Applicability test** (all three must hold; judge by what the kernel computes, not by `op_kind` or `regime` strings):

1. **FlyDSL.** The lane runs `mode=author` with `target_language=flydsl`, or the existing source under `kernel_src/` imports `flydsl` and defines a `@flyc.kernel`. A Triton, HIP or library-call kernel fails this line.
2. **One operator.** The kernel computes one decoder-layer operator and `unittest.py` compares that operator's outputs (a projection, a norm, RoPE / KV write, decode attention, MLA, a router, MoE experts, an indexer, a state update). A whole-layer kernel fails this line: its source is a `kernel_src/mono/` tree or takes a layer's whole weight set, and its output is the layer's hidden state.
3. **Decode shape.** The cases have 1 to 8 rows, one new token per row.

If any line is false, this skill does not apply — stop reading. A whole-layer persistent kernel belongs to
`mono_kernel_optimize` (existing kernel) or `mono_kernel_authoring` (new kernel), even when it is written in
FlyDSL; single operators in HIP or Triton are not covered. Use `native_decode_numerics` together with this
skill: it holds the rounding and layout facts that decide parity, and this skill does not repeat them.

**Related skills.** Validated FlyDSL kernel skills already cover part of this operator list:
- On gfx942, when [`flydsl_rewrite_quantized_moe`](../flydsl_rewrite_quantized_moe/skill.md) also matches (porting an int4 W4A16 or fp8 block-scale quantized GEMM / fused-MoE grouped GEMM from Triton to FlyDSL, prefill and decode), it is the high-prior recipe: follow its rule to map the math onto an existing FlyDSL primitive instead of writing from scratch. This skill adds the 1-to-8-row decode design (task split, lane layout) on top.
- On gfx950 this skill does not match (it is gfx942 only); the counterparts there are [`flydsl_decode_moe_stage1_blkmap`](../flydsl_decode_moe_stage1_blkmap/skill.md) (decode MoE stage 1, gate + up), [`flydsl_fp8_blockscale_gemm`](../flydsl_fp8_blockscale_gemm/skill.md) (CK to FlyDSL fp8 block-scale GEMM) and [`flydsl_prefill_moe_stage2_fp8partial`](../flydsl_prefill_moe_stage2_fp8partial/skill.md) (prefill MoE stage 2, down-proj + top-k reduce).

**Maturity, read before trusting a number.** Every design here was gated on gfx942 only, with FlyDSL 0.4.x;
gfx950 is untested. Five of the ten designs have **no timing against the deployed kernel** (GEMV, RMSNorm,
RoPE / KV write, MoE experts, gated delta rule). Several are **slower** than the deployed kernel on some
shapes; the operator map says which. Treat every design as a correct, parallel starting point.

## Mechanism
At 1 to 8 rows an operator is latency-bound, not throughput-bound. Its time is set by the longest chain of
dependent memory round trips, by how many of the 304 CUs one row is spread over, and by fixed launch and
synchronisation costs. *Measured:* kernels written without this view were 8x to about 5400x slower than the
native op. Three choices decide the outcome, and they must be made before the first line of code:

- **The task axis.** Split one row into tasks that each own their outputs exclusively (16 output columns, a
  head, 4 experts, a slice of an expert's intermediate, a run of KV segments, a block of state rows) and map
  task `t` to workgroup `t mod n_blocks`. That is correct for any grid and uses the CUs.
- **The lane axis.** Put the 64 lanes along the axis that is *read* (K of a weight row, the dims of a key),
  so that every byte is read once with 8- or 16-byte loads issued before the arithmetic. Use matrix cores
  only where a wave has 16 positions x up to 16 heads to contract (attention, MLA, indexer); a GEMV at 1 to
  4 rows is bound by the weight stream and FMAs are enough.
- **How partial results meet.** Either nothing crosses workgroups (recompute redundantly), or the last task
  to arrive finishes from a zeroed workspace in a fixed order, or an owner with the highest task index waits
  for lower ones. All three are deadlock-free at any workgroup count.

The register file sets the thread count: a 512-thread workgroup has two waves per SIMD, so 256 registers per
wave; attention-class designs need more and run at 256 threads.

## Procedure
**Author quick card (`author_engineer`; first ten minutes, in order).**
1. Read the unit test: entry point, shapes, widths, tolerance, what it compares besides the outputs (a cache, a state).
2. Read how the native kernel does this op: layout, work split over threads / waves / workgroups, load width, matrix cores. Port the method. [`single_op_shape_rules.md`](references/single_op_shape_rules.md) section 1.
3. Find the op in the operator map, open its reference, and take its task decomposition and lane layout as they are.
4. Note the parity switches the op has from `native_decode_numerics`; do not guess one.

**Optimizer quick card (`tech_lead`, `engineer`, `deep_engineer`; first ten minutes, in order).**
1. Get the op's time split (a profile, or a build of your own that stores the device clock at numbered points) and compare it with "Where the time goes" in the op's reference.
2. Use the table in "Knobs & pitfalls" to see which moves are already applied; skip those.
3. Read the "Measured dead ends" of the op's reference; do not spend a round re-measuring one.
4. Read the ISA once now (registers, scratch, LDS) so every later change has a baseline.

**Steps.** Steps 1 to 4 are the author's; steps 5 and 6 belong to the optimize rounds. Taking the reference's
task split already in step 1 is not chasing performance: it is the structure every later round would otherwise
have to rebuild.
1. *Shape.* Write a device function plus a thin launcher: grid-agnostic tasks, static widths, a launcher name unique per configuration, no host control flow, weights read in place. [`single_op_shape_rules.md`](references/single_op_shape_rules.md).
2. *Correctness on the smallest case first.* Width 1, the shortest context, then every case; run the oracle exactly as the lane's COMMANDMENT file and the task's own `unittest.py` docstring say. If a multi-rank task, hold every GPU of the group in one lease for the whole command.
3. *Grid check.* Give the launcher a workgroup-count override and pass at 1, at 7 and at the full grid.
4. *Resource check.* ISA: registers under the cap for your thread count, scratch 0, LDS under 64 KiB. [`flydsl_idioms_and_resources.md`](references/flydsl_idioms_and_resources.md).
5. *First three speed moves, in this order, one per commit, measured each time:*
   (a) spread one row over at least 64 tasks, up to about the CU count (*measured:* 5x to 11x on the router, 36x on the MoE experts, 2x to 2.6x on the delta rule);
   (b) read each byte once with 8- or 16-byte loads, all loads of a step issued before its arithmetic (*measured:* 1.18x to 1.29x on fp8 GEMVs from halving the K steps);
   (c) remove per-element serial chains: lane = position, one cross-lane reduction per tile, all waves working (*measured:* about 40x on attention).
6. *Stop rule* (*derived* from the fixed-chain measurements in the references). Stop when the op's time is close to an empty launch plus its two or three dependent round trips: what is left is a fixed chain that in-kernel tuning does not shorten. Stop a direction at the first spill. A round is one to two hours: one structural move per round, not three; skip thread-count and unroll tuning until moves (a) to (c) are in.
7. *Report* the design choices, the ISA numbers, the per-width ratio against the oracle's baseline, and which parity switches were taken from source rather than proven by the test.

## Operator map
Speed is the deployed kernel's time over this design's (above 1 = this design is faster), *measured* on gfx942.

| operator (capability ids) | reference | decisions that matter most | speed vs deployed kernel |
|---|---|---|---|
| GEMV / linear (`skinny_gemv_decode`, `scaled_quant_gemm`, `dense_gemm`) | [`gemv_linear.md`](references/gemv_linear.md) | task = 16 output columns; lanes stride over K with 8-byte loads; FMAs, no matrix cores | not measured |
| RMSNorm (`rmsnorm`, `fused_add_rmsnorm`, `fused_norm_quant`) | [`rmsnorm.md`](references/rmsnorm.md) | task = 16-column tile; statistic recomputed per wave (no sync, and the first thing to remove) | not measured |
| RoPE / KV write (`rope`, `kv_cache_quant`) | [`rope_kv_write.md`](references/rope_kv_write.md) | task = query head; read the layer's cos / sin table, never compute angles; the cache is the product | not measured |
| decode attention (`attention_decode_paged`, `gqa_mqa_attention`, `sliding_window_attention`) | [`decode_attention.md`](references/decode_attention.md) | lane = position; matrix-core chains for scores and `P.V`; K / V read once per kv head; the backend's segments split over tasks with an owner merge | 0.67x to 0.83x at 1k context, 1.13x to 1.34x at 20k, 1.48x to 1.85x at 131k; windowed 1.06x; block-sparse 0.47x to 0.60x |
| MLA decode (`mla_attention`) | [`mla_decode.md`](references/mla_decode.md) | absorb / attention / finish tasks; waves split the latent dims; a backend split is never cut | 1.5x to 1.8x vs the Triton path; 0.6x at 1k and on par from 8k vs the assembly path; sparse 0.95x to 0.99x |
| MoE experts (`fused_moe_grouped_gemm`, `grouped_gemm_moe`) | [`moe_experts.md`](references/moe_experts.md) | task = (row, pick, 32 intermediate columns); last arrival finishes; 4-bit decode in software | not measured |
| router (`moe_routing_topk`) | [`router_topk.md`](references/router_topk.md) | 4 experts per task; the last arrival selects in one wave on packed keys | 1.5x to 2.4x with fp32 logits or gate; 0.92x, 0.98x and 0.81x with bf16 logits |
| sparse indexer (`sparse_attention_nsa`) | [`sparse_indexer.md`](references/sparse_indexer.md) | matrix-core scores to 32-bit keys; radix select and ordered compaction in one workgroup | 0.51x to 0.88x; 1.06x only at 131k, token granularity |
| stream mix; gated delta rule (`linear_attention_gated_delta`) | [`other_ops.md`](references/other_ops.md) | partials reduced in a fixed order by one owner; one owner per state row | stream mix 0.32x; delta rule not measured |

## Knobs & pitfalls
| symptom | what to do | already done if | detail |
|---|---|---|---|
| time proportional to one row's bytes, flat in the width | split the row into tasks with exclusive outputs | 64 or more tasks per row, or about the CU count | the op's reference |
| time proportional to context or window, unchanged by the thread count | lane = position, all waves take tiles, matrix cores for position x head | matrix-core instructions in the hot loop; K loaded once per tile | [`decode_attention.md`](references/decode_attention.md), [`mla_decode.md`](references/mla_decode.md) |
| 1- or 2-byte loads in the hot loop | a thread owns a contiguous run; 8- or 16-byte loads | wide loads in the ISA | [`gemv_linear.md`](references/gemv_linear.md), [`rope_kv_write.md`](references/rope_kv_write.md) |
| scratch above 0, or registers at the cap | keep less in flight per round; rows as runtime loops; 256 threads | scratch 0 | [`flydsl_idioms_and_resources.md`](references/flydsl_idioms_and_resources.md) |
| right at the full grid, wrong at 1 or 7 workgroups | fix ownership or wait order | passes at 1, 7 and full | [`single_op_shape_rules.md`](references/single_op_shape_rules.md) |
| wrong only now and then | look for LDS reuse without a barrier, an LDS flag under a divergent branch, a stale workspace | bit-exact over repeats | [`flydsl_idioms_and_resources.md`](references/flydsl_idioms_and_resources.md) |
| an edit changes nothing | the JIT cache returned the old kernel: change the launcher name | the name encodes the configuration | [`flydsl_idioms_and_resources.md`](references/flydsl_idioms_and_resources.md) |
| short-context attention slower than native | more, narrower tasks; then stop | 64 tasks | [`decode_attention.md`](references/decode_attention.md) |

- *Measured dead ends that recur:* matrix cores for a one-row GEMV (about 2x slower); more or fewer waves on a
  latency-bound GEMV; unrolling on top of wider steps (spills); a 512-thread workgroup for a design that needs
  more than 256 registers (2.4x slower); splitting an op whose work already sits in one or two segments.
- A test that compares only outputs misses a cache or state write, and a value test misses a release-protocol
  bug. Compare the whole cache or state, and reason about arrival orders separately.
- Many rounding switches sit below the test tolerance when wrong. A passing test does not prove them.
- The first call of a workspace-based design must run eagerly, outside graph capture.

## Do-no-harm notes
- If the task lets a FlyDSL library kernel serve as the seed and the oracle passes with it, keep it until a
  design here is *measured* faster on the same oracle. For a plain GEMM the library call stays the default.
- Do not present a design whose row in the operator map is below 1 or "not measured" as a win. It is a seed.
- These designs are for 1 to 8 rows. At larger widths and in prefill, matrix-core library kernels win; leave
  that path alone.
- Do not apply the register, LDS or thread-count figures to gfx950: none was measured there.
- Never change the backend's tile, segment or rounding layout to gain speed; that trades parity for time.
- A change that removes instructions but spills is a regression. Check the ISA, not only the timing.
- The MoE design adds its accumulators with atomics and is not bit-reproducible; if the oracle demands
  bit-exact repeats, use a fixed-order sum instead.
- Every ratio here comes from one hardware generation and a few models. The live oracle outranks them.

## Sources
- Design notes and measurements for ten decode operators from an internal mono-kernel generation workflow
  (not public), gated on gfx942 / MI300X with FlyDSL 0.4.x; restated here without code.
- Public GEAK cards: [FlyDSL authoring optimization](../../../languages/flydsl/authoring_optimization.md),
  [tile programming](../../../languages/flydsl/authoring_tile_programming.md),
  [attention levers](../../../languages/flydsl/authoring_attention_levers.md),
  [debugging](../../../languages/flydsl/debugging.md),
  [MI300X architecture](../../../hardware/cdna3_mi300/arch.md),
  [occupancy](../../../hardware/cdna3_mi300/occupancy.md),
  [memory hierarchy](../../../hardware/cdna3_mi300/memory_hierarchy.md).
- Library kernels worth reading for the native method (public FlyDSL repository, `kernels/` tree; file names
  as read at FlyDSL 0.5.0, not re-checked against later versions): `attention/pa_decode_fp8.py`,
  `attention/pa_decode_swa.py`, `attention/fused_rope_cache_kernel.py`, `attention/mla_fwd_decode.py`,
  `gemm/fp8_gemm_4wave.py`, `norm/rmsnorm_kernel.py`, `moe/topk_gating_softmax_kernel.py`.
