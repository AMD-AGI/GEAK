# KernelFusion — optional stage before the GEAK e2e workflow

A separate module, **on by default**: every external `interface/run_e2e.py` call runs it to
completion **before** the main e2e workflow (baseline → trace → Opportunity Analysis → Feasibility Review +
author → Integration Review + apply back → the e2e workflow). When it accepts a fused overlay, the main workflow starts from that overlay (its own
baseline is measured on the fused stack). Switched off, nothing changes.

```
trace ──► Opportunity ──► Feasibility Review ──► Integration Review
(GEAK     Analysis       + author             + apply
profiler) (fusion-       (UT: GEAK;           (GEAK e2e_integrator
          analysis       kernel: agent)        A/B, compounding
          skill)                               overlay)
```

Step keys (`kf_steps`) keep their short names: `trace`, `topk` (Opportunity Analysis), `author`, `apply`.

| Step | What runs | Reused GEAK code |
|---|---|---|
| Setup | eval dir + baseline throughput (one fresh server, one timed round, no warmup round) | `roles/director.md` PHASE=setup, `bench_e2e.sh` |
| `trace` | AgentX: the kernel-extract skill's workload measurement + sampling capture (points planned from the baseline log, one bounded probe each with Python stacks, `kernel_workload.json` weights); other workloads: one warm serving trace. Skipped when `trace_dir` is given | `knowledge/kernel_extraction_skills/kernel-extract` (stages 1-2), else `roles/profiler.md` |
| `topk` — Opportunity Analysis | `kf_trace_digest.py` (deterministic: per-step kernel sequences, plus the live model classes and files read from the eager Python stacks, where every seam must live), then the Opportunity Analysis agent follows the **`fusion-analysis` skill** (`knowledge/kernel_fusion_skills/fusion-analysis/SKILL.md`) plus the run's prompt(s) → `analysis/topk.json` + the report the skill asks for | `parse_profile.py` |
| `author` — Feasibility Review | the Feasibility Review agent reads every opportunity and picks, by its own judgement, the patterns worth a kernel (no cap, no gain floor; the one hard line is a Python seam); each: region UT, then a fused kernel | AgentX: kernel-extract skill `PHASE=prepare` (one joint-boundary task per region); else `roles/kernel_extractor.md`. The fused kernel itself: one agent per pattern, given the region and the UT, no GEAK kernel layer |
| `apply` — Integration Review | once, after every kernel is written: the Integration Review agent reads the UT results and picks which kernels go on (no speedup floor; one kernel per seam / overlapping region); they go in together as ONE combined overlay on ONE A/B vs the reference (gsm8k on the whole stack first, then validation-length legs). Only if the stack fails accuracy or engagement is each kernel checked alone (correctness only) and the A/B re-run without the failing ones; only a measured e2e win is handed to the main workflow | `roles/e2e_integrator.md` PHASE=integrate |

## Switch

Precedence: CLI > env > handoff.

```
run_e2e.py handoff.json result.json                        # on (default)
run_e2e.py handoff.json result.json --no-kernel-fusion     # off for this run
GEAK_KERNEL_FUSION=0|1                                     # env
"kernel_fusion": false                                     # handoff, or an object that configures it:
"kernel_fusion": {"prompt": "/path/deploy.txt", "trace_dir": "...", ...}
```

| key (handoff) | env | default | meaning |
|---|---|---|---|
| `strategy` | `GEAK_KF_STRATEGY` | `knowledge/kernel_fusion_skills/fusion-analysis/SKILL.md` | the analysis method (fusion levels L1–L4, requirements incl. the boundary ledger, procedure, estimation, report contents); another file replaces it, `false` drops it. Output formats are defined by `kf_workflow.js`, and the implementable scope is injected (`kf_implementable_levels`, default `L1,L2,L3`; `kf_unsupported_patterns`, default L3 MegaKernel and L4) |
| `prompt` | `GEAK_KF_PROMPT` | — | the run's own prompt(s), comma-separated, read after the strategy (e.g. the deployment / workload description `muse_agentx_con16.txt`) |
| `trace_dir` | `GEAK_KF_TRACE_DIR` | — | existing traces (any depth; one dir per sampling point); skips the capture |
| `workload_summary` | `GEAK_KF_WORKLOAD_SUMMARY` | — | per-point weights for the analysis |
| `topk_json` | `GEAK_KF_TOPK_JSON` | — | a finished `topk.json` (REPORT.md / MODEL_NOTES.md beside it are carried over); the analysis is not re-run |
| `reuse_journal` | `GEAK_KF_REUSE_JOURNAL` | — | an earlier run's Workflow `journal.jsonl`: its finished UT-prepare and author agents are reused, not re-run |
| `baseline_dir` | `GEAK_KF_BASELINE_DIR` | — | a measured baseline dir of this same stack (`bench_summary.json`, `server.log`); copied, not re-measured |
| `source_dirs` | `GEAK_KF_SOURCE_DIRS` | installed packages | framework/op-library source for the analyst when it is not importable where the stage runs |
| `steps` | `GEAK_KF_STEPS` | `trace,topk,author,apply` | subset to run; `topk` alone with `trace_dir` needs no GPU |
| `author_k` | `GEAK_KF_AUTHOR_K` | 0 (no cap) | optional cap on the patterns the Feasibility Review sends to authoring |
| `baseline_mode` | `GEAK_KF_BASELINE_MODE` | `isolated_server` | lifecycle of the stage's baseline: one fresh server, one timed round, no discarded warmup round |
| `implementable_levels` | `GEAK_KF_IMPLEMENTABLE_LEVELS` | `L1,L2,L3` | fusion levels the analysis may recommend and the Feasibility Review may author (all levels are still ranked) |
| `unsupported_patterns` | `GEAK_KF_UNSUPPORTED_PATTERNS` | L3 MegaKernel (e.g. MegaMoE), L4 MonoKernel | patterns inside those levels that are ranked but not recommended or authored |
| `accuracy_confirm_limit` | `GEAK_KF_ACCURACY_CONFIRM_LIMIT` | 1319 (full gsm8k test set) | when the fused stack misses the gsm8k gate on the 200-question sample, both sides are re-scored on this many questions and that result decides; the per-kernel screen (only if it still fails) uses it too |
| `min_gain_pct` | `GEAK_KF_MIN_GAIN_PCT` | 0.5 | the combined A/B hands the overlay on only when its gain exceeds max(measured noise band, this) |
| `validation_mode` / `validation_replicas` | `GEAK_KF_VALIDATION_MODE` / `_REPLICAS` | `isolated_server` / 1 | measurement mode / replicas of the combined A/B (one A/B) |
| `task` | `GEAK_KF_TASK` | — | operator constraints appended to every agent prompt (e.g. directories it must not read) |
| `budget_s` | `GEAK_KF_BUDGET_S` | min(12 h, ⅓ of the run) | stage wall clock; the main workflow gets the rest |
| `kf_dir` | `GEAK_KF_DIR` | `<exp_root>/kf_<main eval dir suffix>` | stage output dir |

Synthetic speculative acceptance (`SGLANG_SIMULATE_ACC_*`, `TILERT_SIMULATE_ACC_*`,
`TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`, as InferenceX AgentX throughput runs set) stays on for
throughput legs; every parity / gsm8k check of the apply step runs on servers without it.

## Outputs

`<kf_dir>/analysis/` (digest, `topk.json`, the prompt's report), `kernels/` (task dirs + kernel-layer
runs), `overlay/` (integrator candidates), `kf_return.json`. `result.json` gains a `kernel_fusion`
section with the stage A/B (`throughput_speedup`), and `combined_speedup_vs_original` = stage ×
main workflow speedup. A finished `kf_return.json` is reused when the same eval dir is resumed.

## Running pieces on their own

```
python3 kf_trace_digest.py --trace-dir <traces> --out /tmp/digest           # digest only
node interface/runtime/engine/run_workflow.mjs e2e_workflow/kernel_fusion/kf_workflow.js --agent claude \
  --args '{"workflow_dir":"<repo>/e2e_workflow","kf_dir":"/tmp/kf","kf_steps":"topk",
           "kf_trace_dir":"<traces>","kf_prompt":"<prompt>","model_path":"<model>"}'   # analysis only
python3 -m pytest e2e_workflow/kernel_fusion/tests                          # tests (node optional)
```
