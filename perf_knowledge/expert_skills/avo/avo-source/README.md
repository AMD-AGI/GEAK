# AVO source-level workflow (`avo-source`)

AVO is a long-horizon source optimizer. It repeatedly proposes one source
change, checks correctness and performance, and commits only independently
verified improvements into an auditable `avo-vN` lineage.

Use AVO for a high-value kernel when a longer, commit-gated campaign is more
appropriate than the bounded multi-direction optimization in
`kernel_workflow/`.

## Where this lives

```text
perf_knowledge/expert_skills/avo/         AVO agents (one directory per agent)
  avo-source/                             this agent: source-level optimization
    README.md                             this file
    examples/                             one portable example task
    avo-source-impl/                      implementation (run_geak_avo.py, run_avo.mjs,
                                          avo_workflow.js, lib/, roles/, kernel_base/, tests/)
```

`avo-source` is a standalone workflow with its own entry point. You start it
yourself; nothing in `kernel_workflow/` or `e2e_workflow/` refers to it, and
it is not registered in `index.yaml`. Below, `<GEAK>` is the checkout root and
`<IMPL>` is `<GEAK>/perf_knowledge/expert_skills/avo/avo-source/avo-source-impl`.

### What it reuses from GEAK

AVO owns its workflow, runner, gates, and every role file. It reuses four GEAK
assets by path, so it must run from inside a complete GEAK checkout:

| GEAK asset | How AVO reaches it |
| --- | --- |
| `perf_knowledge/` reference library | passed to the roles as `KERNEL_KNOWLEDGE_DIR` |
| `kernel_workflow/scripts/gpu_lock.sh` | `kernel_base/scripts/gpu_lock.sh` is a three-line forwarder |
| ten files in `kernel_workflow/knowledge/` | `kernel_base/knowledge/` holds one symlink per file |
| `examples/tasks/knn` | used by the Layer-2 smoke test in `TESTING.md` |

`kernel_base/` is the directory AVO's kernel roles resolve `roles/`, `scripts/`
and `knowledge/` from. AVO's own files there are the four kernel roles,
`scripts/profile_kernel.sh`, and `knowledge/amd_instinct.md`. GEAK's
`kernel_workflow/knowledge/learned/` card store is deliberately not linked.

The linked knowledge files are written for GEAK's `kernel_workflow/`. Where
they say `$SKILL_DIR/../...`, that path does not resolve under `kernel_base/`.
Those references are all in the RDNA/Ryzen sections; AVO has been run on AMD
Instinct (gfx942/gfx950) only.

Keep the kernel directory you optimize **outside** the GEAK checkout. Campaign
output defaults to `<GEAK>-runs/avo-source`, next to the checkout.

## Scope

This integration supports source-level optimization only:

- Triton
- FlyDSL
- HIP/C++
- CK
- TileLang
- Python or C++ wrapper code allowed by the task contract

Standalone AMDGCN assembly, LLVM IR, TTGIR, and lower-level IR feedback are
rejected before an agent starts.

AVO never edits the input task directly unless `apply_to_original` is explicitly
enabled. Normal campaigns operate on an isolated copy and validate the final
result against the frozen original.

## Choose a launch method

All supported methods converge on the same execution core:

```text
Claude Code natural language ─┐
portable example helper ──────┼─> run_geak_avo.py ─> run_avo.mjs ─> avo_workflow.js
handoff JSON ─────────────────┘
external runtime config ────────────────────────────> run_avo.mjs ─> avo_workflow.js
```

Use one of four entry methods:

1. **Claude Code natural language** — recommended for interactive use.
2. **GEAK handoff wrapper** — recommended for scripts, CI, and reproducible
   campaigns.
3. **Portable examples** — use for smoke tests and as configuration templates.
4. **Node shim** — advanced interface for an external orchestrator that already
   owns full runtime configuration.

### 1. Claude Code natural language

Start Claude Code, point it at this directory, and describe the campaign:

```text
Use <GEAK>/perf_knowledge/expert_skills/avo/avo-source to optimize ./kernel for
four hours on GPUs 0 and 1. Keep the original unchanged, use the default Team
multi-start mode, and validate before launch.
```

The agent then follows this procedure:

1. Resolve the kernel directory and the output directory to absolute paths.
   Ask only when the kernel or a materially important setting cannot be inferred.
2. Write a handoff JSON. Keep explicit user choices. Otherwise use the engine
   defaults: `operator: "team"`, `start_policy: "diverse"`,
   `start_k: operator_directions`, `apply_to_original: "false"`.
3. Map the request directly: hours/minutes to `budget_s`, the GPU list to
   `gpu_ids`, the requested language to `target_language`, the optimization
   focus and constraints to `task`. `single` selects `operator: "single"`;
   `best start` or `linear chain` selects `start_policy: "best"`.
4. Always validate first with `--dry-run` (see below).
5. Launch only when execution was requested.
6. Report the result path and the campaign output directory. Do not claim
   success from an agent-reported speedup; use the Director validation and the
   normalized result.

Do not run `avo_workflow.js` with the native Workflow tool; that runtime cannot
load AVO's Node modules. Reject requests for assembly, LLVM IR, or TTGIR
optimization. FlyDSL tasks need their environment script (`AVO_FLYDSL_ENV`)
before GPU execution.

Natural language is an intent/configuration layer, not a separate optimizer.

### 2. GEAK handoff wrapper

`run_geak_avo.py` is the canonical production API. It validates source-only
mode, normalizes defaults, checks the GEAK checkout it lives in, injects the
checkout-local dependencies, locates Node and
Claude, invokes the Node runtime, and writes normalized result JSON.

Minimal `handoff.json`:

```json
{
  "kernel_path": "<existing-kernel-directory>",
  "exp_root": "<campaign-output-directory>",
  "budget_s": 3600,
  "gpu_ids": "0",
  "target_language": "triton",
  "task": "Optimize the source implementation without changing its API.",
  "apply_to_original": "false"
}
```

Omitted operator fields normalize to `operator="team"`,
`start_policy="diverse"`, and `start_k=operator_directions`.
`kernel_path` must be an absolute path to an existing directory; a relative
path is rejected. A relative `exp_root` is resolved against the current
directory; give an absolute path outside `<GEAK>`, or omit it to use
`<GEAK>-runs/avo-source`. `root_dir` (the agents' working directory) defaults to
the common parent of `<GEAK>` and `kernel_path`. When the two are on different
top-level directories that common parent would be `/`; the wrapper then stops
and asks for an explicit `root_dir`. Set it explicitly whenever you want a
narrower or a specific directory.

Validate without agents or a GPU:

```bash
python3 <IMPL>/run_geak_avo.py handoff.json result.json --dry-run
```

Launch:

```bash
python3 <IMPL>/run_geak_avo.py handoff.json result.json
```

Inspect `result.json` for `status`, `validation_status`, `final_speedup`,
`best_version`, and `final_patch`. A completed no-gain campaign is not a process
failure.

### 3. Portable examples

Examples keep versioned paths relative and resolve them before delegating to the
canonical wrapper:

```bash
cd <GEAK>/perf_knowledge/expert_skills/avo/avo-source/examples
python3 run_example.py flydsl-gemm-a16-w16 --dry-run
```

Remove `--dry-run` to launch. A real run copies the case's `kernel/` to the
output directory and optimizes that copy. Select an idle GPU with `--gpu-ids`. The
example is a FlyDSL task, so a real run also requires `AVO_FLYDSL_ENV`.

### 4. Node shim

`run_avo.mjs` is the low-level runtime behind the wrapper. It accepts a fully
expanded config containing `scriptPath`, CLI/runtime settings, and complete
workflow, knowledge, kernel, and output paths:

```bash
node <IMPL>/run_avo.mjs avo_run_config.json --validate-only
node <IMPL>/run_avo.mjs avo_run_config.json
```

The config must set `root_dir` (a directory outside `<GEAK>` that contains the
kernel) and `claude_home`. Without `root_dir` the agents start in the GEAK
checkout root, which is not a supported setup.

Use it only when an external orchestrator owns that expanded contract. Normal
callers should not duplicate the path injection and normalization already
provided by `run_geak_avo.py`.

### Unsupported invocations

- `kernel_workflow` has no AVO mode; start AVO through the entry points above.
- Direct native-Workflow execution of `avo_workflow.js` is unsupported because
  that runtime cannot load AVO's Node modules and filesystem helpers.

## Operator choices

Launch path and operator are independent choices.

- `operator="team"` is the default: several orthogonal variations are proposed
  and verified in each step; the best passing candidate may enter the same
  commit-gated lineage.
- `start_policy="diverse"` is the default: Team workers start from the global
  best plus strategy-diverse committed versions when the lineage contains
  enough alternatives. `start_k` defaults to `operator_directions`.
- `operator="single"`: one focused variation per step. It must be requested
  explicitly and always evolves from the current best.
- `start_policy="best"`: all Team workers use the current best. It must also be
  requested explicitly when a Team campaign should remain a linear chain.

To reproduce the former single/best behavior:

```json
{
  "operator": "single",
  "start_policy": "best"
}
```

## Commit topology: chain and tree

AVO already supports a **tree-shaped champion lineage**:

- `start_policy="best"` starts every candidate from the global best and produces
  a linear chain.
- `topk`, `diverse`, and `elites` may start a worker from an older committed
  `avo-vN` tag.
- A winning off-branch candidate is committed with that actual start tag as its
  Git parent. The same edge is stored as `parent_id` in
  `avo_state/lineage.json`.
- Parent identity is checked after commit; a tag whose Git parent does not match
  the selected start is treated as a fatal lineage-integrity failure.
- The global best advances only when the new node beats the current global
  commit gate. Final output is synthesized as `git diff avo-v0 <best-tag>`, so
  it remains correct even when the best lies on an off-branch path.

This is not a general archive that commits every locally useful candidate.
Non-global candidates remain attempts or workspace artifacts. The committed
tree contains only globally accepted champions, possibly branching from older
champions. Supporting a population tree of locally improving but
not-yet-global nodes would require a separate archive gate and must not weaken
the global-best gate used for final delivery.

Useful controls:

- `budget_s`: wall-clock campaign budget, measured from workflow start.
- `final_phase_reserve_s`: portion of `budget_s` reserved for Report, final
  Director validation, artifact materialization, and Cleanup. The default is
  10% with a 300-second floor and 900-second cap (clamped for short campaigns).
- `gpu_ids`: comma-separated GPU IDs.
- `max_steps`: hard step cap.
- `min_commits`: minimum desired accepted versions.
- `verify_repeats`: repeated measurements during verification.
- `start_policy`: `diverse` by default; alternatives are `best`, `topk`, and
  `elites`.
- `use_profiling`: enable staged profiling.
- `profiling_after_step`: first step eligible for profiling.
- `early_stop`: permit roofline-based early completion.
- `apply_to_original`: apply the final validated patch to the input task when
  set to `"true"`; the default is `"false"`.

## Kernel task contract

The kernel directory should contain its editable source and a reproducible
correctness/performance harness. Existing PerfSkills task packages normally
provide:

```text
config.yaml
source/
source_golden/
scripts/task_runner.py
test_cases.json
```

During setup, AVO freezes the original and creates a `COMMANDMENT.md` describing
the exact compile, correctness, benchmark, profile, and editable-file contract.
Candidates that fail correctness or the deterministic performance gate are not
committed.

## Runtime requirements

- a complete GEAK checkout (see *What it reuses from GEAK*)
- Python 3
- Node.js 18 or newer
- a working Claude CLI
- the kernel's compiler/runtime dependencies
- an available AMD GPU for correctness and performance execution

Optional environment overrides:

```bash
export GEAK_NODE_BIN="<node-executable>"
export GEAK_CLAUDE_BIN="<claude-executable>"
export GEAK_CLAUDE_HOME="<claude-home>"
export GEAK_CLAUDE_MODEL="<model>"
```

Kernel-specific setup can be passed with `kernel_env_script`. The FlyDSL example
uses `AVO_FLYDSL_ENV` in its harness configuration.

## Outputs

Each campaign creates an isolated evaluation directory below `exp_root`.
Important artifacts include:

- `avo_repo/`: git-backed working lineage with `avo-vN` tags.
- `avo_state/lineage.json`: committed versions and verified scores.
- `avo_state/attempts.jsonl`: attempted variations and gate outcomes.
- `avo_state/evolution_log/`: bounded per-step causal records.
- `baseline_timing.json`: frozen baseline measurements.
- `trajectory.json` and `strategy_ledger.md`: campaign history.
- `final_patch.diff`: cumulative selected source change.
- `director_validation.json`: final independent validation when the campaign
  reaches director finalization.
- `_shim_logs/result.json`: raw headless runtime result.

The wrapper writes the caller-facing normalized result to the requested
`result.json`.

## Resume

AVO can continue an interrupted evaluation directory when its runtime config
uses the same `eval_dir`. State is reconstructed from the git tags,
`avo_state/lineage.json`, baseline state, and supervisor log.

Put `eval_dir` at the top level of the handoff (or under `args`), or invoke the
Node shim with a fully expanded config.

## Tests

Run the workflow test suite:

```bash
bash <IMPL>/run_tests.sh        # needs pytest and Node.js 18 or newer
```

Set `GEAK_PYTHON_BIN` to choose the python that has pytest.

These tests are not part of GEAK's CI; run them by hand after a change here.

For integration-only checks without a GPU or agent:

```bash
python3 <IMPL>/run_geak_avo.py handoff.json result.json --dry-run
```

Detailed design notes are under `avo-source-impl/docs/`; testing internals are
documented in `avo-source-impl/TESTING.md`.

## Provenance

Imported from `PerfSkills-avo`, branch `GEAK-v4-integrated`, commit `8a57517`.
