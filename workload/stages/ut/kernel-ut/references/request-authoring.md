# Writing a UT request

This skill's only input is `ut_request.json`. Whoever calls UT writes it; in a campaign
that is Brain. This file is the author's contract: which upstream evidence a request may be
built from, the boundary decisions it must record, the request format and the checks to run
before installing it. The UT stage itself reads no profile evidence and cannot tell how
many configs a case's weight pools.

No tool drafts or assembles the request. The author reads the evidence and the installed
source and writes the request; every decision in it keeps its basis on record.

## Inputs

- `kernels.json` from each profile (`stages/profile/kernels.py`). Run `kernels.py verify`
  first: it proves the file still describes unchanged evidence.
- The profile's evidence directory (`profile_root` in `kernels.json`):
  - `selection_targets.json`: the selected identities and their operators;
  - `evidence/source_map.json`: launch frames, operator entry and module class;
  - `evidence/shapes/<point>.json`: per point, the launched kernels, launches per instance
    and host-op operand dims. `<point>@eager` and `<point>@serial` are evidence windows:
    they supply frames only, never costs;
  - `evidence/adjacency.json`: kernels launched back to back;
  - `sampling_points_final.json`: each point's phase, achieved shape and context, and the
    instances it represents, plus the bands no point covers;
  - `evidence/compile_variants.json`: what the server compiled, and the calls that built it.
- The serving image's installed source, read in a CPU-only container. A frame proves a call
  reached a file; it does not prove the boundary. Read the code.

## Decisions

Every selected identity ends up in exactly one place: a task, a blocked identity with its
reason, or a companion of a task with its reason.

1. **Boundary.** Each task is one complete math operation with a callable entry. Prefer the
   kernel library's own entry (an aiter or `amd_kernels` function) to the model code that
   calls it. Keep a whole fused op together: a fused MoE keeps routing, sort and finalize.
2. **Chain or dispatch.** If every call launches all of the members, it is a chain:
   `joint_ut`. If each call launches exactly one of them, they are branches: `dispatch_ut`.
   Two kernels launching at the same profile points does not make them a chain. Check
   per call, from the source dispatch (row count, N/K, request count) or a strict-launch
   trace. Getting this wrong is the commonest blocker
   (`kernel_selection.py --strict-launch` fails `joint_member_missing_in_call`).
3. **Kernels without frames.** These are kernels that only graph replay or a micro-batch
   thread launches, such as graph-capture all-reduce tiers and DBO-only hand-offs. Assign them
   from source to the entry that launches them, and say so. The UT seam trace must then
   confirm the binding. Do not guess from names alone.
4. **Family.** Choose a family from `ut_contracts/families/` only if it
   describes the whole operation. Otherwise use `custom` with a reason; the lane then writes a
   contract. A whole-layer monokernel is `mono_layer`: fill it from `references/mono_models.json`
   ([mono_layer requests](mono-layer-ut.md#0-requesting-a-mono_layer-task)).
5. **GPUs.** A collective, or an entry built on a TP group, runs on the deployment's world
   size (`required_gpu_count`); see [multi-rank UT](collective-ut.md).
6. **Cases and weights.**
   - Make one point case per measured point the task launched at.
   - For each band no point covers (`sampling_points_final.json`), add a bucket case at a
     representative observed shape. Weight it with that band's instances only, never twice.
   - `effective_calls = instances represented × calls per instance`.
   - Count calls of the operation the UT times. If one call launches several kernels, name
     a leaf that launches once per call as its invocation anchor, or give the observed call
     counts (`stages/profile/kernel-profile/references/dynamic-workloads.md`, Call weights). Summing leaf launches
     overstates calls by the kernels per call.
   - Check conservation: in every phase a task runs, its cases' instances add up to the
     phase's observed total.
7. **Case axes.** Derive each axis from the host-op dims in `evidence/shapes`, as an
   expression in the phase shape and context. Under speculation, decode rows are
   requests × tokens per step. Kernels launched straight through `hipModuleLaunchKernel` (FlyDSL,
   `amd_kernels`) have no host op. Take their shapes from the phase shape and the
   library's documented layout, and say so.
8. **Server environment.**
   - `server_preimports`: the modules the server imported that rebind library functions
     the kernel compiles against (sglang: `sglang.srt.utils.common`).
   - The task's slice of `compile_variants.json` goes into `server_variants.json` and
     `server_calls.json`, so the UT replays what the server compiled.

## Output: `ut_request.json`

`scripts/scaffold_tasks.py --check` validates the request, and
`../../tests/test_scaffold_tasks.py` holds a minimal one.

- `contract`: `"workload/ut_request/v1"`.
- `tasks[]`, one per task:
  - `label`: a plain directory name, unique.
  - `requested_kernel`: the delivery gate's KERNEL object
    ([delivery contract](delivery-contract.md)). It contains:
    - `entity_kind: "gpu_kernel"`;
    - `selected_identities`, `selected_device_kernels`, the anchor `device_kernel`, and
      `required_companions` with reasons;
    - `target_callable`, with the file's installed path and SHA-256;
    - `seam_candidates` with a numeric `depth`, a `role` and the `device_kernels` each must
      launch;
    - `joint_ut` or `dispatch_ut` (never both), one row per regime;
    - `phases`, `required_gpu_count`, `invocation_anchors` or `invocation_count_basis`,
      and `source_delivery_required: true`.
  - `scaffold`: `family` (or `family_file`), `target_callable`, the family's const `axes`
    and `attrs`, and `phase_regimes` (`{"DECODE": {"cuda_graph": true}}` when decode is
    graph-launched). Optional: `bind_module` / `bind_file` (a whole-package bind) and
    `contract_overrides` (`[{path, value, why}]`).
  - `manifest.cases[]`: a unique `sig`, `regime` (phase or branch), `axes`, and a finite,
    nonnegative `effective_calls`. Also `kind` (`point` or `bucket`) and the evidence it
    came from. `manifest.replay_sigs` lists the graph-phase cases to replay.
  - `server_preimports`, `server_evidence` (`run/server_variants.json`,
    `run/server_calls.json`), `expected_launches` per phase, and
    `compile_variant_waivers` with a reason for each.
  - Optional `launch_waivers` (`[{pattern, why}]`, expanded over `expected_launches`) and
    `negative_controls` (`[{leaf, category, find, replace, expect}]`, replacing the scaffold's
    stubs): a `mono_layer` model's verified sets in `references/mono_models.json`.
- `blocked_identities[]`: `{kernel, blocker}` for each selected identity no task owns. Examples
  are a device copy, a HIP runtime kernel, or a symbol shared by several collectives.
- `files`: `{relative path: {"sha256": ...}}` for every file the request ships, under
  `tasks/<label>/run/`: `ut_case_plan.json`, `server_variants.json`, `server_calls.json`.

## Before handing it to UT

- Run `scaffold_tasks.py --request <dir>/ut_request.json --check`: contract, file hashes,
  labels, families and weights.
- Check by hand: every selected identity is owned exactly once, cases conserve instances per
  phase, and no case ID repeats.
- If a UT lane blocks on a boundary (it writes `run/lane_blocked.json` with the verdict),
  correct the request and reinstall it. Without `--force`, existing UTs are kept. Then resume
  the wave for that task. Do not edit a verdict into a pass.
