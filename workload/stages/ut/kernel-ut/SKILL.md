---
name: kernel-ut
description: Build and qualify standalone GPU kernel UTs from a ut_request.json -- install and scaffold each requested task from its contract family, extract its source, write and qualify the UT in parallel lanes with command-scoped GPU leases. Correctness only; timing belongs to optimization. Reads no profile evidence.
---

# Kernel UT generation

The input is one `ut_request.json`, written by its caller (in a campaign, Brain) under
[request authoring](references/request-authoring.md). It names each task's kernels, entry callable, contract
family, scaffold arguments and concrete cases with their weights. This skill writes and
qualifies those UTs. It does not launch serving, profile, rediscover shapes, decide
which kernels form a task, or know how many configs a case's weight pools: every
`effective_calls` is one basis. Missing evidence is a task blocker, reported with what
would unblock it.

Set `SKILL_ROOT` to this directory, `SHARED_TOOLS` to `$SKILL_ROOT/shared` and `EVAL_DIR`
to this invocation's absolute output path (`$OUT`) before task construction; relative
exported paths change meaning under a container's working directory. Read
[setup](references/setup.md), [UT contracts](ut_contracts/README.md),
[source extraction](references/source-extraction.md),
[task authoring](references/task-authoring.md) and
[source delivery records](references/source-delivery-records.md); in the pipeline also
[pipeline handoff](references/pipeline-handoff.md).

## 1. Install the requested tasks

The request is `params.request`, else `$OUT/ut_request.json`, else `$IN/ut_request.json`.
With none, report `blocked`: task boundaries are not drafted here.

```bash
python3 "$SKILL_ROOT/scripts/scaffold_tasks.py" --request <ut_request.json> --check
python3 "$SKILL_ROOT/scripts/scaffold_tasks.py" --request <ut_request.json> --out "$OUT"
```

`--check` verifies the contract and every listed file's SHA-256. Installing writes
`task_plan.json` and, per task, `tasks/<label>/run/` (the request's case plan and server
evidence), `run/requested_kernel.json` (the delivery gate's KERNEL object), the family
scaffold (`cases.py`, `meta.json`, `unittest.py`, `run/ut_cases.json`,
`run/ut_contract.json`) and `meta.server_preimports`, `meta.server_evidence`,
`meta.expected_launches` and compile-variant waivers. An existing `unittest.py` is never
overwritten without `--force`. A different request needs a fresh output directory.
When a Roofline call follows, pass the profile's shares with
`--selection <selection_targets.json>` (recorded in `task_plan.json`); UT itself needs none.

## 2. Verify the boundary and run the UT wave

The source map's frames prove a call reached a file on this deployment; they do not prove
a binding. Verify the live callable with `$SHARED_TOOLS/seam_trace.py` and
`python3 "$SHARED_TOOLS/kernel_selection.py" --strict-launch` (`--help` for its inputs),
keeping complete raw verdicts. If the requested boundary is wrong, report the task
`blocked` with the verdict so the request can be corrected; do not edit a failed verdict
into a pass or redraw the boundary here.

UT receipts and source delivery use the same native `sourceSnapshot` owner.
An independent driver digest made repeated qualification incompatible; scaffold
the driver with its owner and freeze the helper set before running the UT.

The CPU half (source resolution inside the serving container, scaffold, CPU lint) needs
no GPU; run GPU gates only on the allocated devices, with no probe server holding them.

**Then write the UTs in parallel, not one after another.** Once the tasks are installed,
hand them to the UT wave; do not author them yourself in this session:

```bash
python3 "$SKILL_ROOT/scripts/ut_wave.py" plan  --eval-dir "$EVAL_DIR" \
    --serving <deployment serving.json> --gpus <this call's GPUs>      # lanes and GPUs
python3 "$SKILL_ROOT/scripts/ut_wave.py" start --eval-dir "$EVAL_DIR" --runner "$RUNNER" \
    --serving <deployment serving.json> --gpus <this call's GPUs> \
    --backend docker --image <the serving image>
python3 "$SKILL_ROOT/scripts/ut_wave.py" wait  --eval-dir "$EVAL_DIR" --timeout 540   # repeat
```

One CPU agent lane per task writes and fixes its own task directory
([UT lane](references/ut-lane.md)); `--author-workers` controls author concurrency.
Run a task's entry in the pinned runtime with
`python3 "$WORKLOAD_ROOT/engine/campaign/run_task_entry.py" <task-dir> unittest.py ...`, so
the task's `unittest.py` cannot shadow Python's standard library; the lane guidance carries
this command. Lane progress and reports stay under the task's `run/`; administrative
outputs must not make tested inputs stale.
The wave leases GPUs only for each `ut_wave.py gpu -- <command>`.
The call passes its resources and environment explicitly: `--gpus` is this call's
allocation and `--serving` the deployment's `serving.json`, whose `env` and
`pythonpath_prepend` every lane inherits (kernel dispatch reads them) and whose `gpus` the
pool may only narrow. Nothing is looked up beside the output directory. Omitting
`--serving` is refused; `--no-serving-env` runs the lanes without the deployment
environment and is recorded in `wave.json`, only for tasks that do not depend on it.
Release the probe
server and any outer campaign GPU lock first. Each command leases `required_gpu_count`
cards together (normally 1), drains its workers, then returns them to the pool. CPU
writing/lint holds no GPU. The wave qualifies all non-timing
gates with `unittest.py --validation-mode correctness --reuse`; no timing tail runs.
Preserve weighted shapes and timing support for the optimizer.
`wait` returns 0 when every task is `passed`, `blocked` or `failed`; read `ut_wave/summary.json`
and the lane logs it names, fix nothing in a task yourself, and carry blocked and failed
tasks into the report with their reasons. `start` again resumes an interrupted wave.

## 3. Extract source and build the standalone UT

Read [source extraction](references/source-extraction.md) and
[task authoring](references/task-authoring.md). Default to fixed-seed generated inputs
at shapes/dtypes/layouts established from existing workload evidence. Run one such
baseline case per phase, then expand cases and editable leaves. The UT reads its
frozen case manifest; it never launches serving, profiles, or captures shapes.
Each `tasks/<name>/` contains its own `kernel_src/`, `cases.py`, `meta.json`,
`unittest.py`, baseline manifest/overlay and `run/` evidence. Preserve minimal structural fixtures only where needed.
Copy the shared harness/leg/overlay helpers into the task; freeze them once authored.

Build each task from its UT contract family: read [UT contracts](ut_contracts/README.md),
create the task with `ut_contracts/scaffold_ut.py new`, and fill only what the family
leaves open (the live `call`, the family's hooks, the case manifest, a replay bundle for
graph phases, negative controls). The family fixes operand distributions, layout facts and
each output's error bound; `unittest.py` runs every gate through `ut_driver.py`. Run the
task's CPU lint (`python3 ut_contract.py <task>`) before any GPU session. Depart from a
family only by an override with a stated reason; for an operator no family covers, write a
contract in the same format (`--family-file`) rather than a free-form harness.

Deliver actual native/JIT/generated computation source for every claimed leaf,
with build resources and version/hash provenance. A copied dispatcher that
imports installed computation is not extraction of those leaves. Prove actual
leaf invocation through the candidate overlay, independently of correct output
or import location. Preserve deployed signatures, layouts, quantization,
routing semantics, cache/page state and complete outputs.

Run every GPU command from its task directory under the lease tools in
[setup](references/setup.md) — `gpu_lock.sh` for one device, `gpu_lock_set.py` for the
whole allocated set outside a wave. Inside an on-demand UT lane use `ut_wave.py gpu --
<command>`; its host broker owns the lease and returns GPUs after worker cleanup.
Never nest the old lock wrappers inside it. `GPU_IDS` is populated only in GPU commands.

The UT must cover baseline/candidate binding, frozen-oracle and random-value
correctness at served shapes, phase-specific graph/compile behavior and input
refresh, each editable leaf's numerical negative control (or a
`leaf_control_waivers` entry with its why, for a whole-package bind's helpers; such a bind also
declares every file that differs from the installed package, as a leaf or a `frozen_differs` why), restored-source
integrity. UT extraction does not measure timing: use the explicit correctness
validation mode, retain every weighted signature, and report performance as unmeasured.
Optimization later measures both timing legs for every positive-mass signature. Compilation
failure is not a successful numerical negative control. Report missing cases
and mass; do not renormalize survivors into full coverage.
Generated UTs default to one isolated timing process per side and round, streaming
all shapes without a scheduling flag. Keep metadata-only `timing_specs` and a
one-shape `make_timing_case` factory; never preallocate all shapes' tensors.
Changing the harness, driver or case factory requires fresh qualification under
the updated helper/source identity, even if an older snapshot format omitted them.

The UT must also cover what the server compiled, in the server's import environment:

- **server_preimports.** Both legs import `meta.server_preimports` before the cases. A
  serving framework can rebind library functions at import (sglang replaces
  `triton.next_power_of_2`, a constexpr function, with a plain one); a kernel calling it
  in a jit body is green in a bare UT and fails in the server. The lint rejects an
  sglang/vLLM entry without preimports.
- **server_calls gate.** Every recorded entry call is replayed in both legs: integer and
  small operands (sequence bookkeeping, page tables, scales) exactly as the server passed
  them, bulk payloads from a seed. The candidate must run every call the baseline runs and
  match it. This is where a branch only warmup reaches (one token, no cache) is tested.
  Retained replay calls have distinct IDs within the task and preserve their compilation-variant links.
- **compile_variants gate.** Each Triton variant the server built for the task's kernels
  must also be built by the baseline legs, and the legs' Triton/torch namespaces must match
  the server's. A missing variant is reported with the fields that differ from the nearest
  UT variant; fix it with a case (the server call is in `run/server_calls.json`) or waive it
  in `meta.ut_contract.compile_variant_waivers` with a reason. Optional arguments in
  generated cases (descale, sinks, None-ness, index dtypes) come from these server calls,
  not from defaults. For a `mono_layer` task, the request waives, one id at a
  time with its why, each variant the runner's fallback branch compiled: the frame the
  entry called is a `forward` outside the entry's package, and the variant was eager and
  outside graph capture (Muse Spark: `MuseMonoRunner.launch` -> `MuseSpark1xDecoderLayer.forward`
  for the bf16-expert layers). The monokernel path's own variants are never waived.

Before reporting a verdict, verify deterministic input construction, baseline
self-noise and that numerical bad-source controls actually fail. For generated
cases, obtain reference outputs from the frozen baseline in a separate process
using the same seeds; never use the candidate to generate its own expected output.
Use the fixed driver's bounded temporary output batches. See
[task authoring](references/task-authoring.md).

Randomize ordinary numerical payloads, not the workload definition. Keep observed
shape/layout/quantization, phase and call weights. Routing, indices, page tables and
sequence metadata must retain valid semantics and the observed work/access pattern:
reuse or reconstruct them from existing metadata where reliable; capture only the
missing structural state. Synthetic structural cases may extend correctness coverage,
but do not claim they reproduce serving performance without evidence. Missing structural evidence is a blocker for the request author, who may ask the profile stage for it.

Use `h.measure_legs()` and `h.serving_weighted_speedup()` from the shared
harness (the driver's timing gate does). Set `workload.kind="agentx_trace_replay"`, `weighting="observed"`, and
`serving_weight_model.source="observed_workload"` with an evidence path.
Each case has a unique `sig`, observed `regime` and finite nonnegative
`effective_calls` on one consistent basis. Aggregate as
`sum(baseline_ms * calls) / sum(candidate_ms * calls)`; no fixed-load fallback
or weighted geometric mean. A near-1x unmodified extraction is expected.

## 4. Validate and deliver

Read [source records](references/source-delivery-records.md), freeze the task
snapshot before GPU validation and generate records from actual outcomes.
After real GPU tests, preflight the exact return with the shared delivery gate:

```bash
node "$SHARED_TOOLS/kernel_extraction_delivery.mjs" \
  --kernel "$TASK_DIR/run/requested_kernel.json" \
  --result "$TASK_DIR/run/draft_return.json" --phase extract \
  --out "$TASK_DIR/run/RETURN.json"
```

Use `extract_op` for the whole-op return form defined in the contract. Both
forms require actual source delivery in this skill. The gate reads workflow
schemas and hashed evidence; it does not execute or independently certify GPU
tests. Keep failures visible and continue independent tasks when possible.
`ut_wave.py` marks a task passed only after both its fresh UT and native delivery
verification pass. A resume includes tasks with passing correctness but missing,
stale or invalid delivery; the task owner completes those records, reusing unchanged
UT gates. Docker authors and GPU workers run as the host UID/GID with a writable
per-task home so the host broker can finish their queue requests. Keep build outputs
in task-local writable trees; installed image sources remain the frozen baseline.

Write `REPORT.md` and `results.json` with all requested identities, task/source
locations, actual commands and devices, per-gate status, declared and achieved
coverage, uncovered mass and limitations. Separate inventory completion, task
completion, GPU UT status and delivery consistency. A CPU-only check or a
subset of passing tasks is not full acceptance. Do not report E2E improvement
without a separately authorized and measured E2E result.
