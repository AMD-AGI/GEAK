# UT request tools

Task boundaries -- which kernels form one UT, its entry callable, family and cases --
are decided between the profile and UT stages, by whoever plans the campaign. These
tools turn a profile's `kernels.json` and a reviewed declaration into the UT stage's
only input, `ut_request.json`. Neither stage imports them.

| Tool | Does |
| --- | --- |
| `draft_task_decls.py` | Drafts `task_decls.json` from profile evidence, every decision marked `needs_review` |
| `plan_ut_cases.py` | Point, bucket and envelope cases with their call weights, from measured points |
| `build_ut_request.py` | Verifies `kernels.json`, lints the declaration, writes `ut_request.json` and its files |

## Declare complete task boundaries


Read the UT stage's [delivery contract](../../stages/ut/kernel-ut/references/delivery-contract.md).
Draft from a profile's evidence directory (the `profile_root` of its `kernels.json`):

```bash
python3 "$WORKLOAD_ROOT/brain/tools/draft_task_decls.py" --eval-dir "$PROFILE_DIR" \
    --out "$PLAN_DIR/task_decls.json"
```

`task_decls.json` is drafted
from the evidence: one task per selected operator with all its member kernels and its
other phase branch; its `target_callable` is the operator's entry (or the branches'
innermost common frame); its `ut_family` is the family whose `match` patterns fit (the
module class breaks ties); a collective is a task on the deployment's world size
(`required_gpu_count` = tp) whose lane builds the multi-rank harness, and only device
copies are excluded with a stated reason; each phase gets one `joint_phases` row per
distinct set of the task's kernels launched together (two rows in one phase are dispatch
branches). Every decision carries its basis and a `needs_review` note. Review it:

- **Boundary.** One owner per complete math boundary. Keep required companions (a
  dispatch-branch sibling or a mandatory chain member below the threshold) under
  `companions` with a reason. Mandatory chains, including split-K plus reduce, are one
  task; mutually exclusive branches are separate `joint_phases` rows, never demanded in
  every invocation. Whole fused MoE keeps routing and the complete operation.
- **Mixed dispatch.** A branch can itself be a mandatory chain. Declare
  `selection_scope: "dispatch"` and keep each branch's `required_kernels` together.
  Verify chain rows with `--dispatch-chain`: every call containing any member must
  contain all members; keep other calls in the evidence as inapplicable. Shared
  companions leave this scope ambiguous: keep it blocked and refine the evidence.
  Never discard partial calls or split a chain into independent leaves.
- Selected indexer score/top-k is one task covering both served prefill and decode
  branches ([joint indexer UT](../../stages/ut/kernel-ut/references/joint-indexer-ut.md)); collectives follow
  [multi-rank UT](../../stages/ut/kernel-ut/references/collective-ut.md). The [GLM example](glm-agentx-example.md)
  only for matching features; it supplies no default shapes or identities.
- **Family and axes.** Confirm `ut_family` (or `custom` with `ut_family_why`), pin the
  family's const `axes`/`attrs` and write `case_axes` per phase from the host-op dims in
  `evidence/shapes` (expressions in `shape` and `context`). `build_ut_request.py --check`
  lists exactly what is still missing.
- Empty each `needs_review`, then:

  ```bash
  python3 "$WORKLOAD_ROOT/brain/tools/build_ut_request.py" --kernels "$PROFILE_DIR/kernels.json" \
      --decls "$PLAN_DIR/task_decls.json" --out "$PLAN_DIR/request"
  ```

  It verifies `kernels.json` against the profile evidence, lints the declaration (every
  selected identity owned once, joint rows consistent with where kernels launched, nothing
  left unreviewed) and writes `ut_request.json`: per task the delivery-contract KERNEL
  object, the family/scaffold arguments and the concrete case manifest -- point cases plus
  bucket cases for the bands no point covers, without double-weighting, each with its axes
  and `effective_calls` -- with `tasks/<label>/run/ut_case_plan.json`. It also slices the
  compile evidence per task into `tasks/<label>/run/server_variants.json` and
  `server_calls.json` (structural operands inline; bulk operands as seeded descriptors --
  `server_calls/*.pt` only for a call recorded before this, or above
  COMPILE_WATCH_INLINE_CAP), and records `server_preimports` (decls `server_preimports`;
  sglang defaults to `sglang.srt.utils.common`) and the launches each phase must show.

Weights count the operation that `cases.call` times. When that differs from GPU
launches (chains, repeated launches or shared symbols), provide observed
`invocation_counts` in the declaration; see [call weights](../../stages/profile/kernel-profile/references/dynamic-workloads.md#call-weights).
After changing a declaration, rebuild the request; the UT stage installs a request as a
whole and keeps the earlier cohort.

The UT stage reads only `ut_request.json` and the files it lists: it does not know which
config, or how many, its cases came from. To pool several configs into one UT, write one
request whose cases carry the pooled `effective_calls` (and an optional `branch` label);
the UT treats every weight as one basis.
