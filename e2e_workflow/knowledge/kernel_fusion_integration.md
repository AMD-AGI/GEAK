# KernelFusion — how it is integrated (GEAK side)

KernelFusion is **off by default**. Switched on (`kernel_fusion=true`; `interface/run_e2e.py` forwards its
switch), it is not a phase of its own: it is broken down into the stages GEAK already runs.

- **Profile** gets a second analysis of the same trace, the *fusion profile*: which kernels run back to
  back as one region, and the one Python seam that launches each (Opportunity Analysis, the
  `fusion-analysis` skill).
- **Strategize** routes the regions worth a fused kernel into the kernel track, as kernel candidates
  that name a `fusion_id` (the Feasibility Review, now the System Architect's).
- **The Milestone loop** builds, optimizes and measures each routed region like any kernel candidate:
  a region UT (extract), a fused kernel (`kernel_lane.js`), then the integrate A/B on the overlay
  accepted so far. An accepted region's overlay is checked on disk before it is banked.
- **The Finalize-gate, Finalize and Validate** are GEAK's own and unchanged. A region's A/B that did not
  finish in the loop is finished at the Finalize-gate like any other.

Switched off, nothing changes: no prompt, lane, log line or result field carries any trace of it.

## The shape of the integration

```
e2e_workflow/e2e_workflow.js                    KF_ON-gated hooks in Profile, Strategize, the Milestone loop and the integrate A/B
e2e_workflow/roles/fusion_analyst.md            the fusion profile (PHASE=analyze: Opportunity Analysis)
e2e_workflow/roles/_fragments/fusion_strategy.md
                                                the Feasibility Review and routing rules the System Architect follows
e2e_workflow/roles/system_architect.md          strategize step 1e, plan_milestone step 6: only when FUSION_STRATEGY is an input
e2e_workflow/knowledge/kernel_fusion_skills/fusion-analysis/
                                                the fusion-analysis skill (SKILL.md, references/), its trace digest
                                                (scripts/kf_trace_digest.py) and the digest's tests
e2e_workflow/roles/_fragments/fusion_ut_contract.md
                                                read by kernel_extractor (or the kernel-extract skill) for a region UT
e2e_workflow/roles/_fragments/fused_kernel_task.md
                                                read by the kernel_lane.js run that writes the fused kernel
e2e_workflow/scripts/fusion_overlay_check.py    the on-disk check of an accepted region's overlay (run through one agent)
interface/kernel_fusion_args.py                 the switch and knobs -> workflow args; the only KernelFusion code run_e2e.py calls
```

```
GEAK  Setup ─► Profile ─────────────► Strategize ─────────────► ConfigSweep ─► Tuning ─► HeadKernel ─► Milestone loop ─────────────────► Finalize-gate ─► Finalize ─► Validate
KF             + fusion profile        + fusion regions as                                            a region is a kernel candidate:
               (Opportunity Analysis   kernel_candidates                                              extract (region UT) ─► kernel_lane.js
               on Profile's trace)     (Feasibility Review)                                           (fused kernel) ─► integrate A/B
                                                                                                      + overlay check ─► bank
```

| What the KernelFusion PR did | Where it runs now | GEAK code it reuses |
|---|---|---|
| capture a serving trace | nothing of its own: the Profile round's trace (AgentX: the kernel-extract skill's sampling captures); `trace_dir` overrides | Profile / KernelExtract |
| Opportunity Analysis (`topk`) | the fusion profile, right after `profiler:baseline` (AgentX: after the baseline preparation) | `roles/fusion_analyst.md` PHASE=analyze; `kf_trace_digest.py`, `parse_profile.py` |
| Feasibility Review | the System Architect, at Strategize and every `plan_milestone`: it routes regions as `kernel_candidates` `{short_name: "kf_<id>", fusion_id, pct_gpu_time, ...}` | `system_architect.md` step 1e + `fusion_strategy.md`; the strategy and plan schemas as they are |
| region UT | the Milestone loop's extract for that candidate. AgentX: the kernel-extract skill prepares one UT per recommended region at KernelExtract (one invocation each, scoped by `FUSION_SCOPE`), and a passing one joins the prepared inventory by its GPU-time share | `kernel_extractor` PHASE=extract / the kernel-extract skill PHASE=prepare, under `fusion_ut_contract.md`; `reusePreparedUT` |
| fused kernel | the candidate's kernel lane | `kernel_lane.js` (`reserveBoundedWorkflow`), with `fused_kernel_task.md` |
| Integration Review + combined A/B (`apply`) | each region's own integrate A/B on the overlay accepted so far, in the order the kernel track dispatches. Regions that rebind the same callable or overlap are routed one at a time, and one that does not land leaves its `fallback` eligible | `runIntegrateBothLegs` (finish retries, speedup review, checkpoint), corrective re-author, the Finalize-gate replay; `roles/e2e_integrator.md` PHASE=integrate as it is |
| overlay validation | `fusion_overlay_check.py` on every accepted region overlay before it is banked: it compiles, the kernels it loads exist, every file of the overlay it was built on is kept, and it is the overlay the A/B measured. A failed or unrun check turns the accept into a final rejection | inside `runIntegrateBothLegs` |
| re-validation | GEAK's own re-profile after each milestone, Finalize and Validate, on the whole stack | unchanged |

## The fusion profile beside GEAK's profile

GEAK's profile (`parse_profile.py`) ranks kernels by their share of GPU time. The fusion profile reads the
same trace and adds what a per-kernel ranking cannot show: which kernels run back to back as one region in
every step kind, the seam that launches the region, the tensors that cross its boundary, the launches a
fused kernel saves, and an estimated e2e gain. It takes no capture of its own and leaves `profile` (the
Top-N every other stage reads) as it is; the regions reach the rest of the run only as the extra inputs of
Strategize and `plan_milestone` (`FUSION_*`), and only once there are regions.

GEAK's profiler captures without Python stacks by default (they make a multi-hundred-MB trace whose flush
blocks the serving scheduler). The digest still has every kernel sequence and the `cpu_op` that launched
each kernel; it then has no "Live model code" table, and the analyst confirms the live module from those
cpu_ops and the source (`roles/fusion_analyst.md`). `trace_dir` can point the analysis at traces captured
with stacks instead.

## Accuracy, time and modes

A fused kernel reorders floating-point work, so a region's A/B is held to task accuracy, not byte-exact
parity: the integrator's `ACCURACY_GATE` (the run's `accuracy_gate`, gsm8k unless it is `none`). It travels
in the A/B's inputs, so a corrective re-author and the Finalize-gate replay keep it. Once a region is
banked, every later A/B and Validate get the same gate: no later candidate can pass byte parity against the
true baseline.

What runs ahead of the search gets a bounded share of it. Under `time_budget_s`, the fusion profile and
(AgentX) the region UTs together have a quarter of the search window left when the fusion profile starts.
The analysis reads a trace and source on no GPU, so it gets two attempts of at most an hour each. Each region
UT call makes one attempt, plus one retry that is told the first attempt's outcome. A failed fusion profile
is recorded and the run goes on exactly as with the switch off. Everything in the Milestone loop runs under
GEAK's own bounds (`budget`, `milestone_min_pct`, the lane reserve, the final reserve).

Fast and deep mode run no Milestone loop, so KernelFusion records `skipped` there.

## Switch

Precedence: CLI > env > handoff.

```
run_e2e.py handoff.json result.json                        # off (default)
run_e2e.py handoff.json result.json --kernel-fusion        # on for this run
run_e2e.py handoff.json result.json --no-kernel-fusion     # off for this run, whatever env / handoff say
GEAK_KERNEL_FUSION=1|0                                     # env
"kernel_fusion": true                                      # handoff: on with defaults, or an object that
"kernel_fusion": {"prompt": "/path/deploy.txt", "trace_dir": "...", ...}   # switches it on and configures it
"kernel_fusion": {"enabled": false, ...}                   # configured but off
```

`interface/kernel_fusion_args.py` turns this into workflow args: `kernel_fusion=true` plus the `kf_*` knobs
below; off, it adds nothing. A knob that cannot be used (a path that does not exist, a `topk.json` without
candidates) is recorded as a KernelFusion `error` and no fusion profile runs; the run goes on without it.

| key (handoff) | env | default | meaning |
|---|---|---|---|
| `strategy` | `GEAK_KF_STRATEGY` | `knowledge/kernel_fusion_skills/fusion-analysis/SKILL.md` | the analysis method (fusion levels L1–L4, requirements incl. the boundary ledger, procedure, estimation, report contents); another file replaces it, `false` drops it. Output formats are defined by `roles/fusion_analyst.md` |
| `prompt` | `GEAK_KF_PROMPT` | — | the run's own prompt(s), comma-separated, read after the strategy (e.g. the deployment / workload description `muse_agentx_con16.txt`) |
| `trace_dir` | `GEAK_KF_TRACE_DIR` | the run's Profile trace | existing traces (any depth; one dir per sampling point) to analyze instead |
| `workload_summary` | `GEAK_KF_WORKLOAD_SUMMARY` | the run's profiled workload | per-point weights for the analysis and the region UTs |
| `topk_json` | `GEAK_KF_TOPK_JSON` | — | a finished `topk.json`; the analysis is not re-run |
| `source_dirs` | `GEAK_KF_SOURCE_DIRS` | installed packages | framework/op-library source for the analyst when it is not importable where the run is |
| `implementable_levels` | `GEAK_KF_IMPLEMENTABLE_LEVELS` | `L1,L2,L3` | fusion levels the analysis may recommend and the architect may route (all levels are still ranked) |
| `unsupported_patterns` | `GEAK_KF_UNSUPPORTED_PATTERNS` | L3 MegaKernel (e.g. MegaMoE), L4 MonoKernel | patterns inside those levels that are ranked but not recommended or routed |
| `task` | `GEAK_KF_TASK` | — | operator constraints appended to every KernelFusion prompt (e.g. directories it must not read) |

Synthetic speculative acceptance (`SGLANG_SIMULATE_ACC_*`, `TILERT_SIMULATE_ACC_*`,
`TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`, as InferenceX AgentX throughput runs set) stays on for
throughput legs; every parity / gsm8k check runs on servers without it (the run's `correctness_unset_env`,
as for every integrate).

## Outputs

`<eval_dir>/kernel_fusion/analysis/` (digest, `topk.json`, `REPORT.md`, `MODEL_NOTES.md`); a region's UT
is a task dir like any kernel candidate's (AgentX: from `<eval_dir>/kernel_extraction/round_kf_<id>/`), its
lane runs under `<eval_dir>/kernels/_exp/`, and its candidate overlay is `<eval_dir>/overlay/cand_kf_<id>`.
A banked region is in `accepted_kernels` as `kind: "patch"` (the knowledge-base replay reads it like any
patch) with `op_kind: "fusion"` and its `fusion_id`. The workflow return, `state` and `result.json` gain a
`kernel_fusion` record only when the switch is on:

| field | meaning |
|---|---|
| `status` | `accepted` (a region was banked) · `profiled` (the fusion profile ran, nothing banked) · `skipped` (fast / deep mode) · `error` (config or analysis error; the run went on) · `not_run` (the search window had closed) |
| `reason` | why it is `skipped` or `error` |
| `trace_dir` / `topk_json` / `report_path` | what the fusion profile read and wrote |
| `candidates` / `recommended` | the fusion profile: every region and the analyst's non-conflicting picks |
| `routed` | the regions the architect routed into the kernel track |
| `unprepared` | AgentX: recommended regions that got no passing region UT, with the reason |
| `accepted` | the banked regions: `short_name`, `fusion_id`, `isolated_speedup`, `e2e_delta_pct` |
| `outcomes` | the kernel track's verdict on every routed region (from `history.ledger`) |
| `soft_parity` | a region is banked: later A/Bs and Validate carry the accuracy gate |

The record is carried in `state`, so a resumed run reuses its fusion profile.

## Running pieces on their own

```
python3 e2e_workflow/knowledge/kernel_fusion_skills/fusion-analysis/scripts/kf_trace_digest.py \
  --trace-dir <traces> --out /tmp/digest                                                 # digest only
python3 e2e_workflow/scripts/fusion_overlay_check.py --overlay <cand dir> --initial <overlay> --root <eval_dir>
# e2e_workflow.js with args kernel_fusion: "true", phases: "setup": Setup, Profile + the fusion profile, Strategize
python3 -m pytest interface/test_run_e2e_kernel_fusion.py e2e_workflow/scripts/tests/test_fusion_overlay_check.py \
  e2e_workflow/knowledge/kernel_fusion_skills/fusion-analysis/tests                      # tests
node e2e_workflow/scripts/test_kernel_fusion.js     # the whole workflow on CPU, switched off and on
```
