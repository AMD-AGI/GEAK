# Persistent regions and compute/communication research

Use the relevant section when your own source and measurements raise a question
about execution boundaries, data movement or scheduling. These primary references
offer alternative implementations, not a prescribed target, expected speedup or
required reading sequence. Publication claims are not qualification on this task.
Keep the supplied edit boundary, numerical/state contract, resource lease and
engineering budget. A broader idea needing absent state or authority is a concrete
request to the TechLead, not permission to change the frozen task.

## Establish the complete replacement and its oracle

Map required outputs/state, producers/consumers, ownership, materialized values,
lifetimes and synchronization across the proposed region. Distinguish mathematical
requirements from the baseline's kernel boundaries. A symbol-duration ranking is
an investigation index; overlap, other ranks and work outside the captured scope
prevent interpreting it as removable end-to-end time.

[ForgeMegakernel, §4](https://arxiv.org/html/2609.12379v1) develops toolchain and
operator correctness through a layer, stack and real-checkpoint decode before
persistent scheduling. Its independent float64/logit and per-layer KV-state checks
localize errors. Transfer that principle: use the task's independent oracle and
request additional intermediate-state visibility where needed; do not edit the
frozen oracle or copy the paper's precision/drift tolerances. Its long-run M9 is
optional in the paper; this task's actual repeated-state contract governs checks.
Missing reference support remains missing, not a pass.

## Search boundaries and resource lifetimes

[One Megakernel Is Not Enough, §IV–VI](https://arxiv.org/html/2610.35336v1) separates
region composition, tile dependencies and execution events. It compares ordinary
operations, segmented persistent regions and broader megakernels under serial and
shared-resource schedules. Its predictor requires
source-declared geometry/storage/synchronization and compatible service measurements.
Its reported selection accuracy is limited to built, qualified validation pools;
it does not infer arbitrary serving latency from a kernel list.

Keep useful library operations and phase boundaries available. Compare the complete
replacement under its real neighbors: a faster isolated producer may occupy
resources longer or deprive its consumer. Check launch count, intermediate traffic,
resource allocation/lifetime and the full unprofiled timer. A single launch is an
implementation property, not the objective. Choose the smallest next change that
distinguishes a suspected launch, movement or interference cost.

## Persistent dependencies and storage are design choices

[Hazy's interpreter and counters](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
and Forge M6–M8 illustrate per-SM instruction streams, dependency counters and
shared-memory pooling. [Cohere's serving engine](https://cohere.com/blog/megakernels)
instead reports abandoning shared-memory paging in favor of fixed per-operation
pipelines, stage rings and carried pipeline phases; it combines static schedules
with local work stealing and uses ordinary kernels for prefill. These are different
organizations, not a mandatory combined recipe.

For a proposed persistent region, establish which producer publishes each value,
which consumer observes it, counter reset/epoch rules, storage reuse lifetime,
memory ordering and forward progress. Exercise the admitted shapes, raggedness and
repeated calls. Inspect whether compilation and dispatch realize the intended
organization. A correct but slower realization can reveal a cost worth revising;
follow the workflow's retention rules and keep it separate from the measured best.

## Communication needs receiver-ready completion

[Weave, §4](https://arxiv.org/html/2609.21483v3) uses routing-dependent worker/chunk
allocation, a shared work counter and role changes within a persistent MoE region.
It is an H100 EP4 implementation. Its broader timing combines separately measured
components, so it is not a full serving result. Static and adaptive splits are
alternatives whose cost depends on the current routing and hardware.

[X-Stage, §3 and §5](https://arxiv.org/html/2607.23264v2) distinguishes accepted
remote-store issue, local-source reuse and remote visibility. Its finite-decoupling
model is software-visible and calibrated to a particular path/producer; do not
assume a physical queue size or carry calibration across devices. A sender-only
timing improvement does not establish receiver readiness. Preserve all-rank
inputs, ownership, buffers, synchronization and the task's complete rank-max or
other declared completion timer. Verify real overlap, not merely concurrent issue.

[TileRT's AMD design](https://www.tilert.ai/blog/tilert-amd-agentx.html) describes
streaming handshakes, prefetch and pushes into peer symmetric buffers followed by
ordered local reduction. Its bundled low-batch engine result does not isolate
persistence or communication. Its [pinned README](https://raw.githubusercontent.com/tile-ai/TileRT/0ea19371f120977761bfa0cf6a2d08415d8fce0e/README.md)
documents distinct CUDA/ROCm backends; availability there is not installation here.

## Use the existing native tools and choose an actual backend

`SKILL_DIR` is supplied in the engineer prompt. Start with existing raw evidence;
collect only what resolves the next implementation question.

| Need | Existing path and boundary |
|---|---|
| Correctness, compile and unprofiled timing | From `KERNEL_PATH`, use `bash "$SKILL_DIR/scripts/gpu_lock.sh" "$GPU_ID" <supplied command>` and the immutable `COMMANDMENT`. Use the supplied lease; a comma pool means one selected GPU, while an explicit `set:` lease uses the existing all-device route. Never infer additional ranks from spare GPUs. |
| Dispatch/counter evidence | `bash "$SKILL_DIR/scripts/profile_kernel.sh" "$GPU_ID" "<supplied benchmark command entering KERNEL_PATH>" "$OUTPUT_DIR/profile_<label>"`; read `profile_report.txt` and the listed raw artifacts. `SKILL_DIR/knowledge/profiling_guide.md` documents failures and available modes. This wrapper supplies no dedicated disassembly mode. |
| Dataflow, IR/ISA interpretation | `SKILL_DIR/knowledge/kernel_hardware_evidence.md`, plus actual compiler artifacts or supplied, qualified tools. Static ISA, compiler resource requests and observed execution are different evidence. |
| Preserve source | `SKILL_DIR/scripts/save_candidate_patch.sh`, with the workflow-supplied base commit and editable source paths, including new backend/build dependencies. Follow existing custody and return rules. |

Choose among the task's available HIP/C++, Triton/Gluon, FlyDSL, CK, TileLang or
assembly paths when they fit the hypothesis. Read the target backend's existing
`KERNEL_KNOWLEDGE_DIR/languages/` and operator cards; source-language choice does not
qualify a target toolchain. Forge, Hazy and Cohere report NVIDIA implementations. Their SM/warp scheduling, barriers and memory scopes
need an AMD-specific realization with the actual CU/wave/XCD, LDS/register and peer
semantics. Preserve public entrypoints and reproducible builds. State in existing
notes what changed, what executed, what the independent oracle/timer established
and what remains untested; full serving gain still requires serving evaluation.
