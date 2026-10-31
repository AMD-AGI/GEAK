# KernelFusion — optional stage before the GEAK e2e workflow

A separate module, **on by default**: every external `interface/run_e2e.py` call runs it to
completion **before** the main e2e workflow (baseline → trace → topk → author → apply back → the
e2e workflow). When it accepts a fused overlay, the main workflow starts from that overlay (its own
baseline is measured on the fused stack). Switched off, nothing changes.

```
trace ──► topk ──► author ──► apply
(GEAK     (user    (GEAK      (GEAK e2e_integrator
profiler)  prompt)  kernel     A/B, compounding
                    layer)     overlay)
```

| Step | What runs | Reused GEAK code |
|---|---|---|
| Setup | eval dir + baseline throughput | `roles/director.md` PHASE=setup, `bench_e2e.sh` |
| `trace` | AgentX: the kernel-extract skill's workload measurement + sampling capture (points planned from the baseline log, one bounded probe each with Python stacks, `kernel_workload.json` weights); other workloads: one warm serving trace. Skipped when `trace_dir` is given | `knowledge/kernel_extraction_skills/kernel-extract` (stages 1-2), else `roles/profiler.md` |
| `topk` | `kf_trace_digest.py` (deterministic), then the analyst agent applies **your prompt** → `analysis/topk.json` + the report the prompt asks for | `parse_profile.py` |
| `author` | per recommended candidate **and its fallback** (the analyst's `fallback`, else the lowest-difficulty exclusive alternative of a difficulty ≥ 3 candidate): region UT, then a fused kernel | AgentX: kernel-extract skill `PHASE=prepare` (one joint-boundary task per region); else `roles/kernel_extractor.md`; then `kernel_workflow/kernel_lane.js` `mode=author` |
| `apply` | per slot, serially: A/B the primary, else its fallback; `accepted`/`stack` overlays compound; then one combined confirmation of the whole stack vs the original baseline — only `validated_win` is handed to the main workflow | `roles/e2e_integrator.md` PHASE=integrate, `roles/director.md` PHASE=validate |

## Switch

Precedence: CLI > env > handoff.

```
run_e2e.py handoff.json result.json                        # on (default)
run_e2e.py handoff.json result.json --no-kernel-fusion     # off for this run
GEAK_KERNEL_FUSION=0|1                                     # env
"kernel_fusion": false                                     # handoff, or an object that configures it:
"kernel_fusion": {"prompt": "/path/prompt.txt,/path/deploy.txt", "trace_dir": "...", ...}
```

| key (handoff) | env | default | meaning |
|---|---|---|---|
| `prompt` | `GEAK_KF_PROMPT` | `default_prompt.md` | the TopK analysis prompt(s), comma-separated, all read (e.g. `agentx/prompt.txt,agentx/muse_agentx_con16.txt`) |
| `trace_dir` | `GEAK_KF_TRACE_DIR` | — | existing traces (any depth; one dir per sampling point); skips the capture |
| `workload_summary` | `GEAK_KF_WORKLOAD_SUMMARY` | — | per-point weights for the analysis |
| `topk_json` | `GEAK_KF_TOPK_JSON` | — | a finished `topk.json` (REPORT.md / MODEL_NOTES.md beside it are carried over); the analysis is not re-run |
| `baseline_dir` | `GEAK_KF_BASELINE_DIR` | — | a measured baseline dir of this same stack (`bench_summary.json`, `server.log`); copied, not re-measured |
| `source_dirs` | `GEAK_KF_SOURCE_DIRS` | installed packages | framework/op-library source for the analyst when it is not importable where the stage runs |
| `steps` | `GEAK_KF_STEPS` | `trace,topk,author,apply` | subset to run; `topk` alone with `trace_dir` needs no GPU |
| `author_k` | `GEAK_KF_AUTHOR_K` | 3 | recommended candidates to implement |
| `kernel_budget` | `GEAK_KF_KERNEL_BUDGET` | 4 | kernel-layer rounds per candidate |
| `validation_mode` / `validation_replicas` | `GEAK_KF_VALIDATION_MODE` / `_REPLICAS` | `isolated_server` / 2 | the director's combined confirmation of the whole fused stack |
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
