# GEAK integration and execution contracts

## How GEAK runs this pack

GEAK runs deterministic JavaScript Workflows inside Claude Code. This pack is an in-repository
expert-skill resource (`skill.md`), not a separately installed `.claude/skills` plugin. GEAK runs it
through one entry point, the **gluon_authoring expert workflow**
([`expert_workflows/gluon_authoring/workflow.js`](../../../../expert_workflows/gluon_authoring/README.md)),
launched directly. Nothing in `kernel_workflow/` or `e2e_workflow/` loads this pack: their default
runs are unchanged, and `use_expert_skills=true` still selects only `validated` recipes, so this
`draft` is never applied there.

The workflow is the stage machine; this pack is the method it runs:

| workflow stage | role | reads (opened at the top of its prompt) |
| --- | --- | --- |
| Setup, Benchmark | GEAK `director`, `benchmark_engineer` | their own role files (frozen baseline, `COMMANDMENT`) |
| Profile | GEAK `profile_engineer` | [profile.md](profile.md) `## 3.1` |
| Entry | `gluon_lead` | [`skill.md`](../../skill.md), [entry.md](entry.md) (modes, Stage-Entry, bundle) |
| Budget | `gluon_lead` | [budget.md](budget.md), `../workloads/index.md` |
| Transcribe / Author | `gluon_engineer` | [transcribe.md](transcribe.md) / `../gluon/index.md` |
| Recover | `gluon_engineer` | [recover.md](recover.md), `../tile-programming/pipeline.md`, the repaid layer's files |
| Climb | `gluon_engineer` | [climb.md](climb.md), [profile.md](profile.md), [benchmark-hygiene.md](benchmark-hygiene.md), the layer's files |
| (every round) | `evidence_auditor` | [index.md](index.md) `### The four gates` |
| Measure | GEAK `verify_engineer` | its own role file |
| Close, Validate | `gluon_lead`; GEAK `director` | [close.md](close.md) |

Entry modes are settled from the bodies, not the names (a filename is insufficient):

| source / frozen baseline | mode | what runs |
| --- | --- | --- |
| tuned plain Triton | port | champion bundle + G1, transcribe, recover to parity (G2), climb |
| Gluon transcription, plain Triton golden copy | port (`source_is_anchor`) | champion measured by verify, provided anchor verified, recover, climb |
| explicit Gluon, nothing converted | incumbent | incumbent bundle + G1, climb; no parity gate |
| no Triton champion (torch / HIP / library) | fresh | author the first Gluon kernel from the API references, climb; no champion or parity evidence is invented |

The method is currently `draft`: structural/script checks do not prove its expected speedup.

## Calling from Claude Code

```javascript
Workflow({
  scriptPath: "<REPO>/perf_knowledge/expert_skills/expert_workflows/gluon_authoring/workflow.js",
  args: { kernel_path: "<TASK>", workflow_dir: "<REPO>/kernel_workflow",
          gpu_ids: "<allocated-pool>", budget: 6, entry_mode: "auto", target_arch: "gfx950",
          deadline_epoch: 1794830000 /* optional */ }
})
```

The workflow sets the loop shape (G0) from the entry mode; there are no port launch args to pass.
GEAK owns setup, the immutable oracle, GPU locking, measurement and acceptance; the workflow owns the
stage order and refuses a round whose gate evidence is missing. `toolctl` is optional bookkeeping
inside the engineer's round.

## Script boundaries

| Capability | Entry | Contract |
| --- | --- | --- |
| Static hardware/ISA/layout | shared `hw_budget.py`, `calc_perf.py`, `amd_isa.py --arch ...`, `amd_resources.py`, `amd_occupancy.py` | Explicit arch/SKU; CPU planning, no GPU lease; local models are not native roofline evidence |
| Recovery / context / records | pack `scripts/toolctl.py`, `recover_gluon.py`, `recordctl.py` | Workspace artifacts; stage-specific refs; no independent agent loop |
| Search timing | shared `ab_bench.py` | Screening only; GEAK harness/oracle decides acceptance |
| Trace / ATT / raw PMC | GEAK `profile_kernel.sh`, shared `capture.sh` | Supplementary diagnostics, explicit extra passes; GPU lease/source guard |
| Roofline / full counters | shared `roofline.sh` | Optional external `kernel-roofline` runner; preserve its output and provenance |
| Existing diagnostic analyzer output | `rocprof_compute_probe.sh --workload DIR` | Offline analyzer in its original tool environment; no GPU lease |

Run the canonical `profile_kernel.sh` inside the workload environment. Its existing command-string
contract does not implement remote container orchestration. The task-level `capture.sh` and explicit
compute diagnostic adapter support the configured host/container locus; keep those outputs' identity
and measurement scope explicit.

Pack shims call the shared tools; there is one implementation for each promoted tool. Use `--help` for CLI syntax and
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

`$KT` below means `kernel_workflow/scripts/kernel_tools`. This checkout does not include the
native `kernel-roofline` runner. To use the adapter, install that external skill and set
`GEAK_KERNEL_ROOFLINE_DIR` to its directory; check its version and runner guidance. Without it,
the adapter returns unavailable (exit 3).

GEAK's existing e2e `roofline` analysis remains the default, and the e2e profiler can separately
consume matching upstream TraceLens artifacts. Neither route installs the native runner. Keep
legacy advisory results, local planning estimates and native calibration/report provenance distinct;
missing native coverage does not disable GEAK's public profiler or its measured Top-N.

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

`capture.sh --payload profile_payload.json` calls the shared
`kernel_workflow/scripts/kernel_tools/profile_payload.py` validator and execution adapter.
The pack's `scripts/profile_payload.py` remains a compatibility entry point.
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
for t in perf_knowledge/expert_skills/expert_workflows/gluon_authoring/tests/test_*.js; do node "$t"; done
python3 -m pytest -q perf_knowledge/expert_skills/_contribute/test_gluon_flow_smoke.py
```

The JavaScript tests execute the real expert workflow with synthetic role responses: stage order per
entry mode, the required references at the top of each prompt, evidence re-ask/reject, acceptance by
verify and the commit gate, and the parity / G1 / deadline / budget stops. The Python smoke runs real
stage/context/tool CLIs and payload capture with CPU fixtures. These checks establish control flow and
artifact handoff, not model behavior, numerical correctness, GPU speedup or promotion of this draft
method.
