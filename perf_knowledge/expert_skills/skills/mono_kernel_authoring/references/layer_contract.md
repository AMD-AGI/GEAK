# Layer contract: stages, float64 references, derived bounds, calibration, resources

Open this for step 2 of the procedure in [`../skill.md`](../skill.md). The contract is written before any kernel
code. It decides whether the layer can be written correctly at all, and it fixes the standard for every later check.

**Status.** The method was implemented and exercised in an internal mono-kernel generation workflow (not public) on
three layer kinds of MiMo-V2.6-Flash and four layer kinds of a second model (mixed residual streams, sparse latent
attention with several caches per call). Its checking scripts are not in this repository, so every check is written
as something to do by hand. Figures are relative (error ÷ tolerance, ratios). Backend numerics facts (rounding
points, quantization formats, weight layouts) are in skill `native_decode_numerics`; this page says where they enter.

## 1. What the contract contains

| part | content |
|---|---|
| stage table | per stage: id; input and output ports (name, shape as a function of the decode width, dtype, layout, which rank holds it); the framework weights it reads (per-rank shard); the native op sequence it replaces; `tp` (1, or the TP size only when the stage itself contains a collective); the state it writes |
| float64 references | one function per stage, and the composition of all stages into the whole layer |
| bounds | a derived tolerance per stage output, `exact` for outputs on the framework's own rounding path, and the layer's derived envelope |
| layer tolerance | calibrated from measurement (section 6), then frozen: nobody downstream may loosen it |
| resource plan | CU shares, LDS and register fit per width, the widths that remain feasible |
| refusal list | configurations the kernel will not handle; the adapter sends them to the native path |

## 2. What you need from the oracle first

- Per rank and per captured decode step: the layer inputs and outputs, the attention metadata, and the KV pages the
  step reads and writes, taken after the layer ran (so they already hold the native write of the current token).
- Inputs and outputs of every sub-module boundary. A cut at a captured boundary gets its golden for free.
- The layer's weights as the framework holds them after loading. Post-processing may repack, pre-shuffle or
  interleave them and may move tensors off the module tree; references dequantize from this loaded format, never
  from the checkpoint format.
- The noise of the native layer (error between two runs in separate processes) and the negative controls
  (section 7) with the checks that catch each of them.
- More than one context length: attention cost and error both depend on it.

If your task captured no sub-module boundaries, take them from the native leg with forward hooks in a driver of
your own (outside the immutable files). Cuts without a golden are then checked together up to the next boundary
you do have, at the composed bound. *(Adaptation for a lane task; not exercised.)*

## 3. Cut rules

1. Follow the data dependencies of the native forward. Typical MoE layer: input RMSNorm (+ residual add) → QKV
   projection (+ dequant) → q/k norm + RoPE → KV-cache write → attention core → output projection → all-reduce +
   residual → post-attention RMSNorm → router + top-k → shared and routed experts → all-reduce + residual.
2. **One GEMM-like operation, one reduction, or one hand-off per stage**: small enough that one engineer writes
   it correctly in one sitting. A stage cut too coarse risks failing at authoring.
3. Prefer cuts at captured boundaries. A cut elsewhere needs its golden computed by the references from a captured
   upstream boundary, and is checked together with the stages up to the next captured boundary.
4. A backend that runs two stages in one call is not a reason to merge them: router and experts stay two stages
   even when one native call routes and runs both expert GEMMs. Test the pair at the boundary of that call.
5. **State belongs to exactly one stage.** The KV write of the current token (or a recurrent state) is an output
   port of the stage that writes it. A stage that receives the cache and writes nothing says so and names the
   stage that does. If a call writes caches your stage test cannot express (several caches, packed byte records,
   compressed caches written at a group end), name every cache and state that the layer oracle's multi-step cases
   check them. Never record "writes nothing" for a stage that writes.
6. A module the layer calls several times per step is checked once per call. A boundary that never runs (a norm
   folded into a fused op) has no stage of its own.
7. When a stage cannot be authored, split it at an internal boundary. Mark the old stage as replaced by the new
   parts, which take its place in layer order. **Check the references of the new parts, then build and test only
   those parts; every other stage and its test stays as it is.** The parts keep the old stage's external port
   names; a tensor passed between parts has the same name on both sides. **Under a layout a port is either handed
   off or a layer output, never both (one buffer per port): never split at a layer output of the old stage that a
   later part also reads.**
8. A layout may group adjacent stages on the same blocks. It may not split a stage: that is a contract change.
9. **Write `tp = 1` for every stage**, also for one that runs on every rank on its own weight shard (every stage
   test runs on the whole rank group anyway). **Write the TP size only for a stage that itself contains a
   collective**: those are exactly the stages that reduce in the kernel.

Typical cuts by module shape (structure only; `_ar` = the stage ends with the in-kernel all-reduce):

| module shape | cut |
|---|---|
| fused QKV + RoPE + paged KV | `input_norm` → `qkv_proj` (one packed output) → `qk_rope` (q/k norm folded in when the layer has q/k norm weights; no positions input when the layer has no rotary embedding) → `attn` (current-token KV write + decode) → `o_proj_ar` |
| latent attention (MLA) | `input_norm` → `mla_absorb` (low-rank q/kv projections + RoPE) → `attn` (latent-cache write, absorb, decode) → `o_proj_ar` |
| sparse MLA with an indexer | projections → norm + RoPE → `indexer` (top-k positions as an unordered index set; index-key cache write folded in) → `attn` (reads only the selected positions) → `o_proj_ar` |
| gated-delta-rule linear attention | `input_norm` → packed input projections → `conv` (causal conv update with its state) → `delta_rule` (state write) → `out_proj_ar` |
| dense MLP | `post_norm` → `gate_up` (act-and-mul as the GEMM epilogue when gate and up are one merged weight; otherwise `gate_proj`, `up_proj`, `silu_mul`) → `down_ar` |
| routed MoE | `post_norm` → `router` → optional `shared_gate_up` → `shared_down` → `experts_ar` (the shared output enters as an addend). A hash-routed router has no selection: ids are an exact table lookup of the input ids |
| mixed residual streams | `mix_attn` (one fused call mixing the streams, with the attention input norm folded in) → `attn` → `mix_ffn` → `router` → shared expert → `experts_ar`; no separate norm stages |
| sparse latent attention, several caches | one `attn` stage with all its cache writes; a stage test on one step cannot see a missing cache write (known gap) |

Under SGLang two facts change the cut: the previous layer's FFN all-reduce may be deferred into this layer (the layer
then starts with an input all-reduce stage), and this layer's FFN output may stay a partial sum (the experts stage
then has no all-reduce). Decide both from the framework source, not from the model name.

## 4. Port names are the data flow

- **Resolve every input in this order.** (1) The nearest earlier output of the same name. (2) For a second reader
  of a layer input that an earlier stage updated: the nearest earlier output whose name contains it (the
  post-attention norm's `residual` reads the input norm's `residual1`, not the layer input). (3) An input that no
  output names is a layer input.
- Name an input after the output it reads, not after the native op's argument. Different data, different names.
- The layout and the kernel wire stages by these names, so check every edge: the consumer's test input must equal
  the producer's reference output. *(Exercised: a data check of this kind, first run at the layout step, caught
  a RoPE stage whose test input had already been rotated, so the stage would rotate twice, and an experts input
  named `hidden` that was really the post-norm output.)*
- Common cut-point names: `layer_in`, `core_in`, `core_out`, `attn_partial`, `mlp_in`, `mlp_partial`, `layer_out`.

## 5. Float64 references

- One function per stage, plain float64 tensor math only (matmul, softmax, top-k, RMSNorm). Never the framework's
  kernels or fused ops: the reference must be independent of the native path.
- Reproduce every rounding the native op performs inside the stage, not only the weight format: an activation
  quantized before a GEMM, a softmax numerator rounded before the PV product. *(Measured: leaving out a dynamic
  fp8 activation quantization doubled one MLP reference's error against native.)*
- Write down tie-breaking rules the kernel must match (for example top-k ties go to the lower expert id).
- **Discrete selections** (router top-k, an indexer's top-k) flip where two scores are closer than the native
  rounding chain can resolve. Reproduce that chain first. Where a gap is still below the derived tie bound, adopt
  the captured selection for that entry only and report how many entries were adopted. Never widen a tolerance to
  absorb a flip, and never return the captured rows as the selection.
- Check each reference against the capture: captured inputs in, compare with the captured outputs at the stage's
  bound, on every captured step and rank. The compared tensor must be the captured one, not a copy or a value you
  computed (otherwise the reference is compared with itself). Every stage must be covered by a passing check; a
  stage with a collective sums the per-rank partials itself.
- Compose the stage references into the whole layer and compare with the captured layer outputs.

## 6. Bounds: derived first, then calibrated

**Stage bounds are derived, never chosen.**

- Derive each output's bound from the input and output formats, the accumulation precision and the reduction
  length. Write the derivation next to the number. The internal tooling applied a safety factor of 2.
- A stage with a collective includes the rounding of every rank's stored partial and the backend's rounding of the
  sum, not only the reduce. *(Exercised: a bound that assumed "the all-reduce is an exact sum" failed on the native
  op itself, at 1.03× the bound on random inputs; with the per-rank partial rounding added the bound doubled and
  the native error sat at about half of it.)*
- An output on the framework's own rounding path (the KV-cache write) is `exact`: zero differing elements.
- The layer's derived envelope is the composition along the longest stage chain, from stage bounds alone. Never
  set or adjust it from the measured layer error. An "empirical factor" (measured error ÷ some unit, multiplied
  back in) is the measurement again.
- If a reference check fails, the derivation is missing an error source or the reference is wrong. Find it; do
  not widen the safety factor. Where a chain of re-rounding points gives a loose bound, keep it and write that the
  check is weak there.

**The layer tolerance is calibrated**, because the derived envelope is a worst case and usually too loose to
catch a real bug. *(Measured: a missing all-reduce produced an error only about 1.1× the derived envelope; an
MLP's derived bound was about 16× its measured error.)*

- `measured`: the worst error of the composed layer reference against the native layer output over every
  captured step and rank, in the oracle's own metric.
- `noise`: the error between two runs of the native layer.
- **`layer_tol = min(derived, max(4 × measured, 4 × noise))`**.
- **Every negative control of the task must still fail at least one check at `3 × layer_tol`**, judged by the oracle's own
  pass test. A control the oracle cannot see fails the contract outright. A control caught only by the multi-step
  sequence check counts, but name it in your notes: that risk must stay visible.
- If `3 × layer_tol` is not above the tolerance the controls were recorded at, the margin is already proven (a
  case that fails at a tolerance also fails at any lower one). Otherwise re-run the controls at `3 × layer_tol`.
- Do not compare a recorded "smallest failing error" with the tolerance yourself when the oracle's reported error
  has the tolerance in its denominator: that figure depends on the tolerance it was recorded at.
- A control no check catches at the margin is strengthened: the same bug at a larger magnitude (a larger scale
  factor, more corrupted entries). Never swap it for another bug, delete it, or edit the checks it names. If no
  larger magnitude is meaningful, the contract fails; return the numbers.
- The multipliers 4 and 3 are fixed. Never change them to make a layer pass.
- **Inside a kernel lane the oracle is frozen**: its tolerance and its controls are immutable files. Compute
  `layer_tol` there as a self-check only. If it differs from the oracle's tolerance, or a control would not fail
  at `3 × layer_tol`, report it. Re-running or strengthening controls belongs to whoever freezes the oracle.

*(Measured on three layer kinds in one run: about 6×, 7× and 16× the oracle's first-guess tolerance, the last
capped by the derived envelope; reruns under the revised metric gave about 4× to 6×. A reference that differed
from native by one bf16 ulp after a norm reached 1.5× the first-guess tolerance, because a per-group dynamic fp8
quantization can step on one ulp; a correct kernel with another accumulation order meets the same amplification.)*

Known limits: `measured` covers only the captured decode steps, not random inputs (the factor 4 and the control
margin stand in for them). A tolerance taken from steps that start on a filled cache does not cover the path that
builds the cache from empty, step by step. *(Measured: a kernel within every stage bound reached about 1.2× the
layer tolerance on that path.)* Make sure the oracle has a multi-step case from an empty cache.

## 7. Negative controls

Generic controls wrapped around the adapter. Two must pass: a no-op and a perturbation at one tenth of the
tolerance. Four negative controls must fail, each reported by the check that can see it: an output perturbed at
4× the tolerance, a perturbation on one rank only, a stale output and a skipped KV write. The task's negative
controls are real bugs of this layer: a missing all-reduce, a dropped attention sink, no KV write, a wrong expert
selected, a scaled norm weight, a skipped or wrong rotary embedding, a wrong value scale. Before trusting a
control, confirm the candidate leg really runs the control's code and not the native layer: a failed rebind falls
back to native silently.

## 8. Comparing a decoder layer: residual increment, ulp floor, ties, index sets

For a layer that takes the hidden state and the residual stream and returns `(branch, residual_out)`:

- Compare `residual_out` on what the layer **adds** to the stream. With `Δa = residual_out − stream_in` and
  `stream_in = hidden + residual`: `|d| ≤ tol·(|Δa_ref| + RMS(Δa_ref)) + ulp(res_ref)`.
- Compare the branch as `|d| ≤ tol·(|b_ref| + RMS(b_ref)) + ulp(b_ref + res_ref)`: the next layer adds the two,
  so a difference below one ulp of the sum is invisible.
- Why: the attention branch is only 1–3% of the residual stream, so a whole-output metric dilutes a single-step
  attention bug below any calibrated tolerance and only the multi-step check catches it. *(Measured with the
  whole-output metric: two of eight controls on a dense layer, a scaled input-norm weight and a wrong rotary
  table, were caught only by the sequence check.)*
- The floor is one full ulp of the output format at the value, not half. Both sides round separately, so two
  close exact values can land one ulp apart. *(Measured: a half-ulp floor made rounding flips read as errors
  about 5× to 7× larger.)*
- **Router near-ties.** A token whose native top-k boundary gap is below a tie threshold may be routed either
  way. Compare its branch with the nearer of the native output and an alternative native output re-run with the
  k-th and (k+1)-th choices swapped. Measure the threshold from one-ulp input noise. Where no alternative exists
  (graph replay), the token's branch is not compared; count those tokens. The composed reference has no
  alternative output: the original flow skipped the branch of every near-tie token there and counted them, and
  for layers that carry several streams skipped only tokens whose own top-k ids differ from the captured ones.
  Prefer the stricter rule. *(Measured on one MoE layer: in 5 of 14 captured steps the score gap was small enough
  for the reference to pick other experts; with the measured threshold 10 of the 14 captured tokens were near-ties.)*
- **Index sets.** A selection written into a buffer the parent model supplies is an unordered set. Bring both
  sides to a canonical order (non-negative ascending, padding last) and compare as a set. A kernel that fuses the
  writer away must emit the canonical order itself.

## 9. Resource plan and refusals

- gfx942: 304 CUs on 8 XCDs of 38, 64 KiB LDS per workgroup, 512 VGPRs per SIMD
  ([arch card](../../../../hardware/cdna3_mi300/arch.md),
  [occupancy](../../../../hardware/cdna3_mi300/occupancy.md)). gfx950 has 160 KiB of LDS; this skill was not
  exercised there.
- Draft one persistent workgroup per CU. **CUs of a stage = total CUs × its bytes moved ÷ the sum of bytes moved
  by all stages.** A stage's bytes are what it reads: its weights on this rank, plus the KV pages for attention.
  The share is not proportional to the stage's operation count or to its share of the native layer's time, and it
  is not an even split. Keep stages that hand off to each other on the same XCD where possible.
- The proportional split is for stages that run at the same time. Stages that run one after another can each use
  every CU (the default layout), and a very small stage (a norm, RoPE, a residual add) gets no partition of its
  own.
- For every decode width check that each stage's working set fits its LDS and register budget. **A width that
  does not fit is dropped from the feasible widths, with the reason written down.**
- The plan is an estimate; the layout step replaces it with placements you can verify
  ([`structures_and_layouts.md`](structures_and_layouts.md)).
- Refusals: expert parallelism, LoRA, speculative decoding, other quantization formats, widths outside the
  feasible set, prefill and mixed batches. The adapter routes them to the native path.

## 10. Before you leave the contract

- [ ] Every stage is covered by a passing reference check on every captured step and rank; every edge is checked.
- [ ] Every bound has its derivation next to it; KV writes are `exact`.
- [ ] `layer_tol` follows the rule; every negative control of the task fails at `3 × layer_tol`; controls only
      the sequence check catches are named.
- [ ] If stages will be authored separately: in each stage test the native op passes against float64 at the
      stage's own bound at about 1.0× (for very short stages the timing is noise), and a wrong candidate (float
      outputs scaled by `1 + 4·tol`) fails.
- [ ] After any edit to a reference or a bound, all of the above is re-run.
