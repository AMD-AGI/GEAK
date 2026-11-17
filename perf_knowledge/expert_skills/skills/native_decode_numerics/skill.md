---
id: native_decode_numerics
title: "Keep a decode-op or whole-layer mono kernel numerically consistent with the native vLLM / SGLang ROCm backend it replaces: rounding points, quant maxima, layouts, tile and tie rules (gfx942)"
kind: expert_skill
authors: [GEAK Team]
scope: kernel
match:
  operator: [attention_decode_paged, gqa_mqa_attention, sliding_window_attention, mla_attention, sparse_attention_nsa, fused_moe_grouped_gemm, grouped_gemm_moe, moe_routing_topk, scaled_quant_gemm, dense_gemm, skinny_gemv_decode, rmsnorm, fused_add_rmsnorm, fused_norm_quant, rope, kv_cache_quant, allreduce, fused_allreduce_rmsnorm]
  arch_class: ['*']
  gens: [gfx942]
  dtypes: [bf16, fp8_e4m3_fnuz, mxfp4]
  regimes: [decode, graph]     # real task metadata often says regime: graph (CUDA-graph replay of a decode step)
  from_backend: ""
  to_backend: ""
expects:
  isolated_speedup_min: 1.0    # this skill is a parity aid, not a speed claim: it must not cost speed
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

Applicability test. All three lines must be true; each can be checked in the task directory:

1. **Operator.** The kernel you author or optimize computes, alone or inside a fused or whole-layer kernel, at least one of: paged / GQA / sliding-window / MLA / sparse attention decode, MoE routing or expert GEMMs, an fp8 block-scale or bf16 linear / GEMV, RMSNorm (plain, fused add, fused quant), RoPE or the KV-cache write, a TP all-reduce. Decide from what the device kernel and the baseline source compute, not from `op_kind` / `regime` (free text: `decoder_layer`, `fused_layer`, `norm`, missing; `graph`, `DECODE`, a dict).
2. **Native-derived oracle.** The unit test compares the kernel with numbers that descend from a vLLM or SGLang ROCm serving path: a native op (aiter, vLLM C++ or Triton, SGLang Triton), captures of one, or a frozen baseline kernel or reference built to reproduce one (the deployed package of a whole-layer mono kernel counts). Read what `unittest.py` calls its baseline or reference. False only when the test checks a formula you are free to restate.
3. **Not activation-only.** The kernel is more than an elementwise act-and-mul (`silu_and_mul` and relatives are out).

If any line is false, this skill does not apply: stop reading.

It applies in both lane modes (optimize and author) and is complementary to `mono_kernel_optimize`, `flydsl_decode_op_authoring` and `mono_kernel_authoring`: they say how to make the kernel fast or how to build it, this one says what its numbers must be. It holds no speed lever: it tells you which arithmetic you must not change while you look for one. When the oracle is a frozen baseline kernel, the rows tell you which of its operations are rounding points to keep, not what to change it into.

## Mechanism

- **Native backends round where the signature does not show it**: an fp8 query and softmax numerator, a bf16 store between two GEMMs, a truncating store after a TP sum, a quant maximum of 224 in one kernel and 240 in the next. A kernel that skips such a point is *more accurate* than native and still fails a tolerance calibrated against native; one that passes a single step with the wrong switch flips fp8 codes downstream and drifts over a sequence.
- **The rule depends on the provider the deployment selected at run time** (aiter on or off, eager or compiled, KV-cache dtype, hardware generation, TP size), so the same model has different numerics in two deployments. Identify the provider first, then read the facts.
- **Many switches are invisible to the unit test**: flipped, they stay inside the tolerance (see `references/parity_method.md`). A green test does not confirm them; the reference tables do.
- **Status of what the references carry.** 428 rows distilled from about 560 facts of an internal mono-kernel generation workflow (not public): 74 rows verified on a model capture, 39 by a real-op probe, 315 read from source only. Every row keeps its status. `source only` rows are hypotheses.

## Procedure

**Planner, first ten minutes.** (1) List the operator families inside the kernel; a whole-layer kernel holds most of them. (2) Fill the deployment sheet of step 1. (3) Open only the references of those families and copy the rows whose "Applies to" matches the sheet into the plan as invariants, `source only` rows marked as suspects. (4) Tell the engineer which invariants the planned change touches. A change that touches none (launch geometry, scheduling, memory traffic) needs no numerics work: say so and move on.

**Engineer, first ten minutes.** (1) Find where the current source implements each invariant (search for the constants: 224, 240, 448, the tile, the segment count, the store conversion). (2) Make one numerics-relevant change per oracle run. (3) On a parity failure stop optimizing and walk the top-facts table below in order. (4) Stop when what remains is the floor described in `references/parity_method.md`.

1. **Fill the deployment sheet** from the process that produced the oracle (its recorded environment, launch command and server log, kept in the task's metadata or capture notes), never from the machine you work on.

   | Deployment fact | Where it is read | What it decides |
   |---|---|---|
   | Attention backend | server log `Using <NAME> backend out of potential backends`, or the `--attention-backend` argument | paged, MLA or sparse decode rules; who writes the KV cache |
   | aiter switches | environment: `VLLM_ROCM_USE_AITER` (default off) and its per-op switches (`..._LINEAR`, `..._MOE`, `..._RMSNORM`, `..._CUSTOM_AR`, `..._TRITON_ROPE`); SGLang: `SGLANG_USE_AITER` | provider of linears, MoE, norm, RoPE, all-reduce |
   | fp8 linear kernel | server log `Selected <Kernel> for Fp8LinearMethod` | activation-quant variant (maximum 224 or 240) |
   | MoE backend | server log line naming the MoE backend / experts class, or `--moe-backend` | weight layout, rounding between the two GEMMs, top-k tie rule |
   | All-reduce backends | server log `Using [...] all-reduce backends (in dispatch order) for group 'tp:...'` | truncating or round-to-nearest store of the TP sum |
   | Compile mode | launch command: `--enforce-eager`, or the optimization level and CUDA-graph setting; the custom-op list | number of roundings in RoPE and act-and-mul, RMSNorm provider, fused all-reduce + norm |
   | KV-cache dtype | `--kv-cache-dtype`, or the dtype of the cache tensors in the task | fp8 query / numerator rounding, tile length, write quantisation |
   | Hardware generation | device architecture: gfx942 stores fp8 as e4m3fnuz, gfx950 as OCP e4m3 | 224 / 240 against 448, scale layout, which GEMM runs |
   | RMSNorm provider | startup log line with the IR op priority, for example `rms_norm=['vllm_c', 'native']` | where the norm rounds |
   | TP size and message size | launch command; rows x hidden x bytes per element | which all-reduce kernel, hence which rounding |

   If the task records none of these, read the provider off the native op its unit test or baseline source calls, and treat every row that depends on a missing fact as a suspect.

2. **Open the matching reference** (map below). Start with its "Which ... is running" table, which turns the sheet into one provider per operator.
3. **Apply the rows that match.** For each: where is it implemented in the source you start from? A deployed kernel or a seed usually implements most of them; that list is your invariant set, and the risk is a change that silently removes one (a fused store that drops an intermediate rounding, a retiled loop that moves the running maximum, a reciprocal introduced for speed). When authoring, each matching row is a line of the design.
4. **Treat every `source only` row your change depends on as a suspect** and confirm it with the flip test (`references/parity_method.md`): flipped, a switch that matters moves the error by percents, one that does not moves it by about one unit of the output format.
5. **Run the oracle** exactly as the lane's COMMANDMENT file and the task's own `unittest.py` docstring say; hold every GPU of a multi-rank task in one lease for the whole command. Do not edit the oracle, its tolerance or its captures.
6. **On failure, localise before changing anything**: which output, which step, which rank; a short context or a long one; one layer kind or all. Then use the symptom rows of the top-facts table. One shape or layer kind about 10x worse than the rest is a grid mismatch (tile, segments, window origin), not a precision problem.
7. **Record** in your result which invariants you kept, which `source only` rows you confirmed by a flip test, and which you could not separate.

Budget: sheet plus rows take ten to fifteen minutes. Skip references of families the kernel does not contain; skip `sglang_backends.md` unless the oracle is SGLang; flip-test at most the suspects of the failing step.

## Top facts that most often break parity

Ordered by how often they decided a comparison. "x" is a multiple of the step's tolerance; "invisible" means a flip stays inside it. Row ids point into the references.

| # | What native does | Applies when | Status | If you get it wrong | Rows |
|---|---|---|---|---|---|
| 1 | An fp8 block-scale linear quantises its bf16 input per token and per 128 values to fp8 before the GEMM. Multiplying the bf16 input by the fp8 weight is "too accurate". | vLLM block-fp8 linear, every kernel | capture (1 model) | about 15x; twice the error on a captured MLP | LIN-14 |
| 2 | The quant maximum on gfx942 is 224 in vLLM's own C / Triton quant and 240 in aiter's HIP quant (448 on gfx950). The fused norm + quant ops split the same way, with other amax floors and other rounding before the quant. | the kernel the log line names; aiter on or off | probe | 25x to 33x; 10x and far more on the scale output of a fused norm + quant | LIN-15, NRM-17 to NRM-20 |
| 3 | aiter's quant multiplies by reciprocals: `scale = amax * fp32(1/240)`, `q = fp8(x * fp32(1/scale))`. | aiter HIP quant: linears, MoE inputs, preshuffled linears | probe (linear); source only (MoE) | `amax / 240`: about 2.8x; the quotient form alone is invisible (0.15x to 0.4x) | LIN-16, MOE-82 |
| 4 | Loaded fp8 weights are not the checkpoint's: e4m3fn bytes are reused as e4m3fnuz with doubled scales, rows sit K + 256 bytes apart, and some weights are stored in (16, 16) tiles while still reporting dense strides. | gfx942; `VLLM_ROCM_FP8_PADDING`; aiter on | capture (2 models) for format and padding; capture (1 model) for linear tiles; source only for MoE tiles | every weight halved or doubled; 119x to 616x (linear tiles); 16x to 18x (MoE tiles) | LIN-07, LIN-08, LIN-10, MOE-60 |
| 5 | Paged decode over an fp8 cache: fp8 query, value dot with P rounded to the query dtype against an fp32 denominator, tile 32, 16 split-K segments each with its own running max. | `TRITON_ATTN_DIFFKV`, fp8 KV cache, decode-only batch | capture (1 model) | an unsplit reference is bit-exact only on short contexts; about one unit of the P format on most outputs beyond one segment | ATT-11, ATT-18, ATT-20, ATT-23 |
| 6 | The sliding window masks on the absolute-position grid; segments divide the full context, not the window. | windowed layers on the same backend | source only | about twice the error (0.11 to 0.20 against 0.007 on full-attention layers) | ATT-27 |
| 7 | A sink seeds the running max of segment 0 only (denominator starts at 1) and is read in the dtype the model stores it in. | layers with sinks | capture (1 model) | 2.2x to 38x; max relative error 0.92 when dropped | ATT-29, ATT-32 |
| 8 | The attention call, not RoPE, writes the current token's K / V, and the kernel attends that token as stored: `fp8(fp32(x) / scale)` in the Triton write, a reciprocal multiply in the fused HIP insert. | every paged backend; fp8 cache | source only | stale or unrounded current token: 2x to 170x; scale flipped: 11x to 57x; divide against reciprocal: invisible | ATT-33, ATT-35, RKV-17, RKV-19, RKV-33 |
| 9 | RoPE: a default compiled deployment runs the torch-native code (fp32, one rounding), an eager one the C kernel; the packed row is [Q, K, V] with its own v head dim; a model-side v multiply is `bf16(fp32(v) * fp32(c))`. | all vLLM attention backends | source only; the multiply: probe | wrong split: 100x to 400x; multiply dropped: about 30% | RKV-03, RKV-08, RKV-10 |
| 10 | Router: the bias selects and the unbiased score weighs; logits have the gate's output dtype; tie rule and id order differ per kernel (aiter's Triton routing sends ties to the higher id and returns ids ascending, the other vLLM kernels send ties to the lower id, SGLang's aiter top-k follows a lane-rank order). | the router kernel that runs | source only; near-tie statistics: capture (1 model) | one swapped expert fails the comparison; bf16 against fp32 logits: 1.9x to 619x; another id order: every row | MOE-05, MOE-22, MOE-25, MOE-29, tie table, SGL-43 |
| 11 | Experts: gate / up rows are concatenated in every expert kernel except aiter's Triton w4a16, which interleaves them; which values are rounded to bf16 between and after the two GEMMs differs per kernel. | the experts class the log names | capture (1 model) for both mxfp4 layouts; source only for the rounding points | layout: 31x to 40x; each rounding: 0.05x to 0.2x (invisible) | MOE-59, MOE-63, MOE-65, experts glance table |
| 12 | RMSNorm rounding follows the provider: `native` takes the variance from the unrounded fp32 sum and rounds the normed value before a bf16 weight; `vllm_c` rounds the residual sum first and rounds once. | the IR priority list of the oracle's process | source only | 0.2x to 0.6x on a bf16 output: invisible, so set it from the provider | NRM-04, NRM-05, NRM-08, NRM-09 |
| 13 | TP boundary: each rank's partial is bf16 before the reduce; aiter's custom all-reduce truncates the fp32 sum on gfx942; the fused all-reduce + RMSNorm normalises the unrounded fp32 value. | TP > 1; aiter on; eager or compiled | probe (two ranks) for the stores; source only for the bf16 partials | round-to-nearest matches about 73% of elements; an unfused emulation of the fused kernel about 60% and 45% | TPA-01, TPA-09, TPA-17 |
| 14 | Sparse latent attention: the window cache is e4m3fnuz with maximum 224, the compressed caches are OCP e4m3 with 448 on every arch; the gfx942 indexer reads the OCP keys as FNUZ (half the value; a `0x80` byte becomes NaN and scores 0); the split count comes from buffer widths and the CU count. | the sparse-latent generation before `DeepSeek-V4.1`; re-confirm for `DeepSeek-V4.1` | capture (1 model) | reproduces the captured top-k selection; attention residual 2% to 3% with other split widths against 0.5% to 1% | MLA-46, MLA-59 to MLA-61, MLA-68, MLA-69 |
| 15 | Multi-stream residual mixing: both bf16 stores truncate on gfx942 and the mixes read the unrounded fp32 streams. | aiter fused op; same generation as row 14 | capture (1 model) | mix source: 1.6x to 60x; round-to-nearest misses by up to 1.3e-2 relative | NRM-39, NRM-40 |

## Reference map

| Operators in the kernel | Reference | Holds |
|---|---|---|
| paged, GQA / MQA, sliding-window attention decode | [attention_decode.md](references/attention_decode.md) | KV-cache layout, fp8 query and numerator, tile and split-K segments, window and sink conventions |
| MLA decode, sparse MLA, block-sparse attention, top-k indexer | [mla_and_sparse_attention.md](references/mla_and_sparse_attention.md) | latent cache records, index-key cache dtype, top-k as a set, split over the gathered list, compressor |
| MoE routing and expert GEMMs | [moe_and_routing.md](references/moe_and_routing.md) | logits dtype, selection bias, tie and order rules, weight layouts, rounding between the GEMMs, weighted sum |
| fp8 block-scale and bf16 linears, skinny GEMV, act-and-mul epilogue | [linear_and_gemv.md](references/linear_and_gemv.md) | FNUZ weights and row padding, activation quant variants, preshuffled weights, epilogue roundings |
| RMSNorm, fused add / quant, multi-stream residual mixing | [norm_and_quant.md](references/norm_and_quant.md) | provider priority, where the sum and the normed value round, norm + quant constants, truncating stores |
| RoPE, KV-cache write | [rope_and_kv_write.md](references/rope_and_kv_write.md) | which RoPE runs, table dtype, packed split, write scale and store, fused insert kernels |
| TP all-reduce, fused all-reduce + RMSNorm | [tp_allreduce.md](references/tp_allreduce.md) | dispatch and size gates, bf16 partials, truncation, fused arithmetic |
| any of the above on SGLang | [sglang_backends.md](references/sglang_backends.md) | the SGLang counterparts; almost all read from source only |
| method | [parity_method.md](references/parity_method.md) | deriving a bound, calibrating a tolerance, flip test, negative controls, the floor |

## Knobs & pitfalls

- **More accurate is not closer.** Exact softmax, an fp32 router, an unquantised activation, a round-to-nearest store where native truncates: each is a parity regression.
- **Speed changes that move a rounding point**: fusing two kernels (drops the bf16 store between them), replacing a divide by a reciprocal multiply, retiling an attention loop, changing split-K, summing TP partials in fp32, reading an fp32 table instead of its bf16 cast. Check each against its row before measuring speed.
- **Universal against scenario-specific.** Universal: match the regime that produced the oracle; round where the oracle rounds; take layouts from the loaded tensors, not from the checkpoint. Scenario-specific: every constant and tie rule in the references holds only for the provider named in "Applies to". A deployed kernel may legitimately differ from a row (another provider, another version, a deliberate design choice): then its frozen baseline is the authority, not the row.
- **Eager oracle, compiled deployment.** An oracle captured eagerly holds ops separately that a compiled deployment fuses with other rounding. Match the oracle; report the difference instead of "fixing" it.
- **Near ties are not bugs.** A router token whose k-th and (k+1)-th scores almost tie may route either way; native does so against itself between batch sizes. Do not chase it and do not copy native's ids to hide it.
- **Version drift.** Line numbers are valid only at the commits named in each reference; in another version locate by symbol and re-read the rule.
- **Do not widen, do not fit.** A tolerance comes from a derivation; an unexplained gap is a wrong switch until proven otherwise.

## Do-no-harm notes

- Never edit the oracle, its tolerance, its captures or its negative controls to make a candidate pass.
- With a frozen baseline kernel as oracle, never move the candidate toward a row and away from the baseline: a row explains a rounding point, the baseline defines it.
- Do not apply a row whose "Applies to" does not match the deployment sheet: a 240 rule in a 224 deployment is a bug you introduced.
- Do not spend the round here when the planned change touches no rounding point, quant parameter, layout, grid or tie rule.
- Not for activation-only kernels, prefill-only paths or training; the facts are decode-regime facts on gfx942 (gfx950 notes are `source only` unless a row says otherwise).
- This skill never justifies a slower kernel: if keeping a native rounding costs speed, report the trade instead of dropping the rounding.

## Sources

- Internal mono-kernel generation workflow (not public): per-backend fact tables, per-operator switch studies and parity method, distilled into `references/`. Evidence classes are stated per row.
- vLLM (`vllm-project/vllm`) at commit `f09d5e30c`, aiter (`ROCm/aiter`) at tag `v0.1.26` and commit `285579b6`, SGLang (`sgl-project/sglang`) at commit `22e1cb6d98`: rows cite file, lines and symbol there (compiled libraries without readable source are named without lines).
- fp8 FNUZ on gfx942: [matrix_core.md](../../../hardware/cdna3_mi300/matrix_core.md).
