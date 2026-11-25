---
id: mono_kernel_optimize
title: "Optimize a whole-decoder-layer persistent kernel with in-kernel TP all-reduce (deployed HIP / FlyDSL mono kernel, or a freshly fused seed) on MI300X"
kind: expert_skill
authors: [GEAK Team]
scope: kernel
match:
  operator: '*'
  arch_class: ['*']
  gens: [gfx942]
  dtypes: [bf16, fp8_e4m3_fnuz, mxfp4]
  regimes: [decode, graph]
  from_backend: [flydsl, hip]
  to_backend: ""
  config: {kernel_class: decoder_layer_monokernel, ranks: multi}
expects:
  isolated_speedup_min: 1.05
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
Applicability test — all three lines must be true, judged from the task directory, not from metadata:
1. The device kernel under test computes a **whole decoder layer for one decode step** (norm, attention and
   FFN / MoE stages, or everything from the attention output on) as one persistent launch per rank, or a small
   fixed number of launches, with the tensor-parallel all-reduce **inside** the kernel.
2. The task's `unittest.py` starts **more than one rank** (one process per GPU) and needs several GPUs.
3. An existing device kernel for the layer is the starting point (as received, or a seed an authoring step already committed), whatever the lane `mode`.

If any line is false, this skill does not apply — stop reading.

`op_kind` and `regime` in task metadata are free text and may be missing: decide from the kernel's header
comment, its stage list and the unit test. Not for single-operator kernels (a norm, an activation, a KV write,
one attention or MoE operator, even a fused one), not for writing a new mono kernel (`mono_kernel_authoring`),
not the `megamoe_*` skills / `expert_workflows/mega_fusion` (mode `mega`: one expert-parallel MoE operator fused
across ranks, gfx950, experimental). Also applies: `native_decode_numerics`. Delivery is never a lane task.

| two task shapes, both first-class; if in doubt it is (a) | (a) deployed mono kernel | (b) freshly fused seed |
|---|---|---|
| tell: the source first, then the baseline leg | sources in a framework package layout, HIP C++ or FlyDSL with its own sync helpers; the baseline leg usually runs the same kernel as extracted (first speedup about 1.0) | an adapter + kernel from an authoring flow with a vendored synchronization runtime, one launch per layer; the baseline leg runs the framework's native layer (first speedup may be below 1.0) |
| launches per layer | as received (one, or front + tail) — never more | exactly one |
| levers | most of the table is applied already, yet one critical-path segment usually still takes several stackable moves: start from the ranked list in planner step 5 | few applied; order of attack: rebalance → hand-offs → all-reduce → per-stage kernels → placement |
| rules ([acceptance.md](references/acceptance.md)) | universal rules | universal + seed-only rules |

## Mechanism
- Decode is memory-bound, not FLOP-bound: each weight byte is read once per step, so layer time ≈ ramps + bytes ÷
  marginal stream rate + hand-off latency (*measured*: an in-kernel 64 MB stream reached 58% of the 5.325 TB/s peak).
- In a tuned kernel waves spent about 57% of their time waiting on dependencies, 26% issuing loads, under 20% computing
  (*measured*): what pays first is a shorter path from a producer's publish to its consumer's first work. "Tuned" is not "nothing left": one critical-path
  segment still gave 1.1× or more in one round on two deployed kernels — four or five stacked steps on one, one operand-role restructuring on the other (*measured in kernel lanes*).
- Safety comes from structure: resident workgroups, acyclic waits, one path on every rank, device-epoch tags, a rank-order fp32 all-reduce.

## Procedure
### Planner quick card (tech lead) — first ten minutes, in order
1. **The task's constraints rank above every lever.** Read the task statement, the COMMANDMENT and the `unittest.py` docstring: memory
   footprint, graph safety, no host-side win, tolerance, case weights, what is timed, the GPUs of the rank group (one lease holds them all).
   Under a stated memory limit an additional or re-laid-out copy of weights or scales is out, unless it replaces the original or the task statement allows it.
2. **Classify the structure.** Layer class: full one-launch · dual launch (front + tail) · partial from the attention output · multi-launch layer · multi-layer per
   launch (outside this lever set). Schedule: every workgroup runs every stage · per-stage task tables · two wave groups. Rows. Shape. → [kernel_structures.md](references/kernel_structures.md)
3. **Inventory before ranking — mandatory.** For every lever you are about to propose, write down the marker you looked for in the source (planner step 5 and the lever table) and what you found. A lever whose marker says "applied" is not a candidate. Judge sub-levers one by one.
4. **Look for an existing trace / timeline switch** before planning instrumentation (HIP: a trace pointer in the argument struct plus a stamp helper; FlyDSL: a timeline build); measurement gets a quarter of the round.
5. **Rank by expected verified gain per engineer-hour, not by stage size.** Effects measured on an already tuned kernel outrank a stage's share of the layer.
   If the timeline shows a compute-bound stage, inspect its arithmetic and MFMA operand roles.
   On a deployed kernel without a timeline, inspect these opportunities in order. The
   [measured rounds](references/measured_rounds.md) give the per-step gains, bundles and repeat counts.

   1. **Move independent work before the wait.** Check whether a stage polls before loading weights,
      converting operands, staging slabs in LDS or preparing masks and addresses. Limit its wait to the tiles it consumes.
   2. **Shorten the path after the wait.** Look for a barrier in each merge round, repeated softmax passes,
      or a trailing stage that only sums. Try one LDS exchange, deferred stores or a fused producer epilogue.
   3. **Separate competing work.** Look for workgroups that own critical tiles of two stages, and for a
      small table every workgroup reads. Move the contending units or try one reader per XCD.
   4. **Check LDS layout.** Look for activations reloaded per tile, misaligned arrays and bank conflicts in
      hot gathers. Stage small inputs once, align arrays to 16 bytes or pad the affected rows.

   The measured rounds also identify low-yield directions on tuned kernels, including deeper prefetch
   while memory is busy, inner-loop tuning of memory-bound stages, isolated XCD remapping and pipelined polls.
6. One direction = one critical-path segment (a stage with the hand-offs into and out of it); shared tables go to one direction.
7. **Stack; do not stop at the first gain.** Inside its direction the engineer adds separately verified steps on the same
   segment until the round ends (steps on different stages multiplied; on one segment re-measure the whole stack after each step).
   Only a schedule change, merging launches or adding rows is a multi-round project. Do not cap the expected gain by this skill's per-lever figures (shares of other kernels' stages).
8. Evidence = a speedup against the baseline **in the same run**, above twice the run-to-run noise measured in this task (the oracle's A/A band).
   The oracle's gates are the contract: bit-identity with the baseline is not required unless the oracle checks it. Three rounds under 2–3% is plateau evidence.

### Engineer quick card — first ten minutes (1–3), then every round (4–8)
1. Read the kernel's header comment and stage list; confirm the class, the applied levers, the task's constraints.
2. Use the existing timeline if there is one (look before building); time-box measurement to a quarter of the round.
3. Run the oracle **exactly as the lane's COMMANDMENT file and the task's `unittest.py` docstring say, holding
   every GPU of the rank group in one lease for the whole command** — no second lease around it, no rank outside it.
4. Measure inside the time box: per-case table → existing timeline → byte floors (an estimate unless you measured). → [measuring.md](references/measuring.md)
5. One lever at a time, local to one stage or hand-off (confirm in the source it is not applied); when it verifies, save and take the next on the same segment.
6. Correctness first, then the benchmark. **Always compare against the baseline measured in the same run** —
   never an earlier run, a different GPU group or a figure from this skill. Inside twice the task's own A/A noise a step is not yet evidence — save it all the same (step 8).
7. Correctness once per step while iterating; on the state you finally save, if it changed synchronization, hand-offs or LDS
   sharing, **repeat it three times** (report the count). The oracle does not check: launch count (kernels per step ÷ layers, after a structural
   change), a hollow kernel (body emptied → correctness must fail), ISA scratch after a prefetch. → [acceptance.md](references/acceptance.md)
8. **Bank each verified step**: as soon as it passes correctness and beats the incumbent, re-save the patch exactly as
   your role file says (the workspace diff). Do not commit inside the workspace unless the role file says so — a commit
   empties that diff. Otherwise revert completely. The lane's verify re-run decides, not your own number.

## Lever table
The levers that built the measured MiMo-V2.6-Flash kernels (step 3 tells which are applied). P90 = serving latency (HIP designs, while being built); stage figures = that stage's own time; "bundle" = with other changes.

| symptom | lever | measured relative effect | already applied if | ref |
|---|---|---|---|---|
| cases with width > 1 are slower per row than width 1 | take every width into the kernel (rows as MFMA columns, or workgroup groups) | largest step in both HIP designs: P90 −59% in one round; in the other design rows arrived inside bundles (−20%, −14%, −12%) | stage functions take a row count; width > 1 cases run the kernel | [structure](references/structure_levers.md) |
| one stage sets the critical path while workgroups idle | rebalance the CU plan by bytes; fold a tiny stage only if it adds no reads | not measured (folding the input norm into QKV cost +14% time in one attempt) | one stage → workgroup table; no idle private partition | [structure](references/structure_levers.md) |
| a stage starts late on busy workgroups; hand-offs costly although balanced | per-stage task tables with staggered bases; producer and consumer on one XCD | bases: only inside bundles; an XCD-local tile map alone was noise in one lane run (move 3 in step 5 is the placement that paid) | `(base + i) mod grid` mapping; a tile map keyed by XCD | [structure](references/structure_levers.md) |
| a long stage stalls short ones; they never overlap | two wave groups per workgroup | not measured as an A/B; the one-sequence design did better in fewer rounds | code branches on the wave index | [structure](references/structure_levers.md) |
| gaps between stages or layers; time inside device-wide barriers; flags cleared per step; slow peer accesses | per-tile (value, tag) hand-offs with no device-wide barrier; one epoch per step; batched polls; scalar peer bases | one barrier ≈ 4–5% of a tuned layer; one epoch per step: layer gap −68% yet P90 only −0.3%; polling with parallel lanes neutral; in one lane run a relaxed memory ordering on the end-of-launch completion counter +1.2% | no grid-wide barrier between stages; epoch advanced once per step; nothing zeroed | [hand-off](references/handoff_levers.md) |
| the all-reduce is a serial block after its GEMM | partials from the GEMM epilogue; sum + residual + next norm in the reduce epilogue | not measured alone (the fused down + final sum is move 2 in step 5) | GEMV epilogues store into peer buffers; no collective call | [hand-off](references/handoff_levers.md) |
| a stage starts with a load ramp | issue this stage's first loads / the first K tile at kernel start | at most 2–3% each on P90 | load-first passes; K tile issued at entry | [hand-off](references/handoff_levers.md) |
| a GEMV stage far above its byte floor | wide coalesced loads, dequant in registers, bytes in flight | gate/up −23% (bundle: XCD tile map, cached scales, lane redistribution, depth-2 loop, one-wave epilogue); a scale copy in read order −12% P90 (bundle; an extra copy: step 1) | 16-byte loads before the math; scale after block accumulation | [stage](references/stage_levers.md) |
| attention grows with context; few workgroups busy | split-KV with an in-kernel combine; long tasks read K once | K read once −15% P90 (the −16% bundle inside existing split-KV is move 1) | tasks per (request, key chunk) plus a combine | [stage](references/stage_levers.md) |
| attention span grows faster than its KV bytes | wave-local softmax, K straight into MFMA, KV write after the loop | KV write after the loop + cooperative q quant −5% P90; K into MFMA + Vᵀ swizzle neutral; wave-local softmax flat on P90 (median −10%, bundle) | no barrier inside the key loop | [stage](references/stage_levers.md) |
| router / top-k / experts visible in the timeline | wave-level top-k, FP4 decode in registers, several accumulators | FP4 decode −14% P90 (bundle with more rows) | permute decode, no LDS table | [stage](references/stage_levers.md) |

## Knobs & pitfalls
- Floors (*measured*): a stage's floor is ramp + bytes ÷ marginal stream rate, never bytes ÷ peak nor bytes ÷ 85% of
  peak. Effective fractions of peak: 58% for a tuned in-kernel 64 MB stream, 31% for a generic 64 MB read, 75% at 1 GiB.
  The role files' roofline uses 0.7–0.85× nameplate bandwidth: report that figure, but steer decode-sized stages by this floor.
- Numbers: the oracle's same-run speedup decides; kernel-trace time is what the kernel itself costs (the oracle's
  ratio carries a fixed graph-launch part in both legs); the timeline tells which stage. A lane gain is not a serving gain.
- Registers (public constants): 512 threads per workgroup leave 256 registers per wave (architected VGPRs plus accumulator
  file); a prefetch or double buffer can spill — check the ISA for scratch and accumulator moves. LDS is 64 KiB per workgroup.
- Stale builds: a JIT / build cache may miss an edit in a helper module — rename the kernel or clear the cache.
- Never read the target model's own finished mono kernel from any source but the task directory; a different model's
  is a structure reference only. Statements without a status mark are read from source.

## Do-no-harm notes
- The task statement's constraints (memory footprint, graph safety, no host-side win) rank above every lever. No
  extra or re-laid-out weight / scale copy under a stated memory limit unless in place or allowed by the task.
- Never loosen the oracle tolerance or edit an oracle file. Never change the all-reduce order (fp32 sum in rank order); every rank takes the same path every step.
- No host synchronization on the steady decode path; nothing per-step from the host; no per-call allocation.
- Do not add launches, and do not fall back to the native path for any eligible width, to look faster; the host
  side computes nothing new. A deployed kernel keeps the launch count it arrived with (one, or front + tail).
- Never remove a bound from a wait. A new wait uses the kernel's existing wait helper, never a bare spin loop;
  giving deadlines to a kernel that has none is a separate, declared change. The timeline build is never installed.
- Out of budget mid-change: revert to the last saved patch. "No verified gain" is a valid result; never claim a step inside the A/A noise as a gain.

## Sources
- An internal mono-kernel generation workflow (not public): its lever cards, optimize-phase rules, timing and
  bandwidth notes, and measured rounds on three MiMo-V2.6-Flash decode-layer kernels (two HIP C++, one FlyDSL
  optimized by a GEAK kernel lane), MI300X, four ranks — as ratios only; peak from [arch.md](../../../hardware/cdna3_mi300/arch.md).
- Public code (symbol names and docstrings read at these commits; behaviour not re-verified): FlyDSL
  `kernels/monokernel/`, `docs/ktrace_guide.md` at `9b70d465`; sgl-amd-mono `sgl_amd_mono/common/` at `572ddf18`.
- GEAK kernel-lane runs, one round each, on two deployed vLLM mono kernels (HIP for MiMo-V2.6-Flash, FlyDSL for DeepSeek-V4.1), independently re-measured: the figures marked *measured in kernel lanes*.
- STATUS: draft — the skill itself is not yet validated in a GEAK kernel lane. Nothing here replaces the oracle.
