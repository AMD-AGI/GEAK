# GEAK integration and execution contracts

## Claude Code Workflow discovery

GEAK runs deterministic JavaScript Workflows inside Claude Code. This pack is an in-repository
expert-skill resource (`skill.md`), not a separately installed `.claude/skills` plugin. No global
Claude settings, extra agent or duplicate skill installation is required.

The call chain is `e2e_workflow.js` → `kernel_workflow.js` → `kernel_lane.js`, or a direct call to
`kernel_workflow.js`. The dispatcher forwards `use_gluon_skill` to normal and bakeoff lanes; the e2e
workflow forwards it through its central lane-argument builder. The interface `run_e2e.py` also
preserves an explicit handoff value. After Setup resolves the actual GPU, the lane appends
`gluonSkillBlock` to tech_lead, author_engineer, engineer, deep_engineer and profile_engineer prompts,
including the inline Optimize prompt. Each role reads the entry and only the current stage reference.

| Condition | Behavior |
| --- | --- |
| `use_gluon_skill=auto` (default), gfx942/gfx950, Triton/Gluon target | Automatically load the method as language guidance |
| `use_gluon_skill=false` | No method injection, including nested/bakeoff lanes |
| Other backend or architecture | No Gluon method injection |
| Triton target | Keep Triton; inspect champion/config and applicable diagnostic guidance; migrate only when explicitly selected |
| Gluon target with Triton source | Port admission, tuned champion, transcription, recovery, climb |
| Gluon target with Gluon source | Incumbent baseline, skip transcription and parity gate, optimize in place |
| Fresh authoring | API/layout guidance; do not invent champion or parity evidence |

Inspect actual imports and kernel bodies to identify the source backend; a filename is insufficient.
The method is currently `draft`: structural/script checks do not prove its expected speedup.
`use_expert_skills=true` remains a separate opt-in for validated recipe selection. Read the registry's
status and selector, honor wildcard/list matches, and never promote a draft or bypass measured gates.

## Calling from Claude Code

```javascript
Workflow({
  scriptPath: "<REPO>/kernel_workflow/kernel_workflow.js",
  args: { kernel_path: "<TASK>", workflow_dir: "<REPO>/kernel_workflow",
          mode: "optimize", target_language: "gluon", use_gluon_skill: "auto",
          gpu_ids: "<allocated-pool>" }
})
```

For a port, set the recovery budget/candidate-floor arguments described in [entry.md](entry.md).
The default discovery does not rewrite those launch arguments. GEAK owns planning, scheduling,
acceptance and the immutable oracle. `toolctl` is optional bookkeeping inside the existing role.

## Script boundaries

| Capability | Entry | Contract |
| --- | --- | --- |
| Static hardware/ISA/layout | shared `hw_budget.py`, `calc_perf.py`, `amd_isa.py --arch ...`, `amd_resources.py`, `amd_occupancy.py` | Explicit arch/SKU; CPU planning, no GPU lease; local models are not native roofline evidence |
| Recovery / context / records | pack `scripts/toolctl.py`, `recover_gluon.py`, `recordctl.py` | Workspace artifacts; stage-specific refs; no independent agent loop |
| Search timing | shared `ab_bench.py` | Screening only; GEAK harness/oracle decides acceptance |
| Trace / ATT / raw PMC | GEAK `profile_kernel.sh`, shared `capture.sh` | Supplementary diagnostics, explicit extra passes; GPU lease/source guard |
| Roofline / full counters | shared `roofline.sh` | Delegate to GEAK native `kernel-roofline`; preserve engine output and provenance |
| Existing diagnostic analyzer output | `rocprof_compute_probe.sh --workload DIR` | Offline analyzer in its original tool environment; no GPU lease |

Pack shims call the shared tools; there is one implementation. Use `--help` for CLI syntax and
`--selftest` where supported. Native availability is tested before execution; missing tools are
reported explicitly. Do not install optional profilers for CPU-only or existing-trace work.

Dependency setup belongs to the shared tools: see
[matrix asset preparation](../../../../../../kernel_workflow/scripts/kernel_tools/dependencies.md)
and [profiler/ATT environment checks](../../../../../../kernel_workflow/scripts/kernel_tools/profiling-dependencies.md).
`matrix_dependency.py` checks the selected calculator interpreter and actual target queries;
`profiler_preflight.sh` reports the tools inside the execution locus. Consume these reports
before requesting the corresponding capability. Setup commands are explicit environment
preparation, not runtime skill tools. A CLI or decoder load check is not GPU validation.

## Native TraceLens handoff

`$KT` below means `kernel_workflow/scripts/kernel_tools`. The native skill is discovered under
GEAK's e2e analysis-skill index as `kernel-roofline`. Check its current installation and runner
guidance; a checkout without it returns unavailable (adapter exit 3), with no substitute calculator.

```bash
# Offline reporting from an existing measurement: no GPU lock.
bash "$KT/roofline.sh" -- --measurement /abs/measurement.json \
  --tl-source /abs/TraceLens --tl-commit <frozen-commit> --arch /abs/arch.json --output /abs/tl-report
# New UT measurement: native case_runner_template.py arguments follow --.
bash "$KT/roofline.sh" --engine measure --gpu <allocated-id> --workspace /abs/workspace -- \
  --adapter /abs/adapter.py --task task --case case --arm candidate --capture --output /abs/measurement
# Full counters: CMD must satisfy the native measured-window marker/manifest contract.
ROOFLINE_MIBENCH_CACHE=/abs/frozen.csv bash "$KT/roofline.sh" --engine rocprof \
  --gpu <allocated-id> --workspace /abs/workspace -- --workdir /abs/workspace \
  --cmd '<native measured-window command>' --mode full --output /abs/counters
```

Run in the source workspace so GEAK's source guard covers the measured implementation. Within an
existing lease omit `--gpu`; never pass a device override to the native runner after `--`.
Offline `tl-only` uses `GEAK_TL_PYTHON` (default python3). GPU modes use `GEAK_GPU_PYTHON` in the
configured locus. For a separate container, mount the native runner and inputs and pass native
arguments using container-visible paths. The frozen CSV environment value is mapped with
`TILE_KERNEL_HOST_WORKDIR` / `TILE_KERNEL_CONTAINER_WORKDIR` and checked for readability.

The report owner produces `geak.kernel_roofline.v1`: preserve mode, engine, native report links,
coverage and limitations in the round evidence. Do not synthesize legacy speedup/headroom fields,
reorder measured Top-N, equate local SKU specifications with native calibration, or fill missing
coverage with invented metadata. A low efficiency is diagnostic context; measured A/B decides gains.

## Shared implementation ownership

The language pack contains Gluon API/layout recovery, pipeline authoring, parity
decisions, stage references and recipe/context bookkeeping. Shared GEAK tools
contain hardware data access, ISA providers, static resource parsing, IR capture,
assembly analysis, benchmark helpers and profiling adapters. The compatibility
entry for each shared tool ships with the PR that introduces that tool.

Probe, assembly audit, ATT hotspot diagnostics and parity consume one resource
model. An AMDHSA `.vgpr_count` is a combined allocation, not an architectural
VGPR-only count; do not add AGPR again. LLVM architectural counts stay separate.
Numeric zero stays zero; missing registers, target, launch geometry or an
unsupported wave mode remain unknown. Compiler max-flat is not an actual launch.
ATT excerpts cannot establish allocated register counts; supply a matched CSV
and explicit `--arch` to `hotspot_analyzer.py`.

The complete GEAK tool tree must be mounted in an external workload container
for `capture.sh` to run IR collection. `locus.sh` is shared and retains the
`TILE_KERNEL_*` variable contract. A compiler-only capture checks the container
without requiring an installed profiler.

## Payload and stage execution

All six stage cards expose `amd_isa` alongside the compatibility ISA entry. Use
`toolctl exec ... amd_isa -- --arch gfx942 ...` with the actual target. A command
receipt records success or failure; a failed receipt cannot satisfy a required
transition artifact. Choosing a stage reference is a record view, not proof of
measured admission or completion of that stage.

`capture.sh --payload profile_payload.json` calls the pack's
`scripts/profile_payload.py` adapter, which invokes shared positional capture.
The adapter applies `execution.argv`, `execution.env_allowlist` and the declared
host/container working directories. Relative source paths resolve against the
host working directory. Configure `TILE_KERNEL_CONTAINER` for a container
payload; a mismatch with `locus.kind` is an error. Payload environment entries
cannot override GPU lease/device or locus controls.

The optional `capture.mode` is `full` (default), `lite` or `isa-only`. Full mode
requests diagnostic capture; native TraceLens roofline remains a separate call.
The actual source hash in the execution locus must match `source.sha256` before
capture runs, and `capture.json` records the payload hash. Device BDF and image
inventory fields remain caller-supplied declarations, not hardware attestation.
Choose a device-kernel substring that also identifies a single IR compilation;
the profiler accepts regex selectors, but the static IR collector uses a literal
substring. A trace with no matching kernel refuses capture and lists candidates;
it does not silently select another kernel.

## Offline flow smoke

From the repository root:

```bash
node kernel_workflow/scripts/test_gluon_flow_smoke.js
node kernel_workflow/scripts/test_gluon_skill_routing.js
python3 -m pytest -q perf_knowledge/expert_skills/_contribute/test_gluon_flow_smoke.py
```

The JavaScript smoke executes the real workflow with synthetic role responses:
Setup, optional Author, Analyze, Benchmark, Profile, Engineer, Verify, Commit,
Reprofile, Report and Director validation. Author/verification failures stop
adoption; only `committed=true` updates the incumbent and triggers reprofile.
The Python smoke runs real stage/context/tool CLIs and payload capture with CPU
fixtures. These checks establish control flow and artifact handoff, not model
behavior, numerical correctness, GPU speedup or promotion of this draft method.
