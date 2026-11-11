# Advanced kernel modes

Use the public `kernel_workflow.js` dispatcher for every standalone mode.
These examples are additions/changes to the args in `SKILL.md`.

## Author a new implementation

Use `mode: "author"` and `target_language: "triton"`, `"flydsl"`, `"hip"`
or `"ck"` supported by the execution image. `kernel_path` must contain an
immutable task oracle, normally `meta.json`, `unittest.py`, helpers and a real
frozen reference such as `baseline_src/` or `baseline_overlay/`. Preserve
captured `reference_io.pt` when present; GEAK can share it read-only.

`op_spec` may supply `op_kind`, `shapes`, `dtype`, `math_contract` and `regime`.
These describe the operation; they do not replace a runnable independent oracle.
Record shape/dtype coverage and the real performance denominator. Do not time
the candidate against a naive implementation authored during the same run.

The author engineer builds a correct seed, then the normal optimization loop
improves it. `authored:false` / `author_failed` means no valid authored seed;
do not present a partial source file as a successful implementation.

## Race backend languages

```json
{
  "mode": "bakeoff",
  "backends": ["hip", "triton", "flydsl"],
  "gpu_ids": "0,1,2",
  "budget": 6,
  "apply_to_original": "false"
}
```

Merge these fields into complete args with `kernel_path`/`workflow_dir`.
`backends` adds rewrite candidates; the input language always participates as
an optimize lane. An empty list permits discovery. Check installed languages;
an unavailable backend may be dropped with an advisory.

Freeze → Discover → per-language lanes → Report compares against the **same
frozen input baseline**. The budget above is six directions per lane, not six
for the entire race. Discovery can also produce an env/config tuning winner.
Only accepted candidates faster than baseline can win. A lane winner carries
its patch; an env winner carries `apply_env` and a tuning artifact. Preserve
those deployment inputs, and report `winner=null` if no candidate qualifies.

Bakeoff gives each lane one GPU. It rejects `gpus_per_job > 1` or any nonempty
`job_gpu_ids`; multi-GPU operations must use optimize/author mode instead.
With one GPU, lanes serialize their measurements through the GPU lock.

## Align scoring with a real workload

Generate a workload spec from the relevant trace when the user wants
production-shape weighting:

```bash
"$GEAK_PYTHON" "$GEAK_ROOT/e2e_workflow/scripts/parse_profile.py" \
  --torch-trace /work/traces/profile.json \
  --workload-out "$RUN_DIR/workload.json" --target kernel_symbol
```

Resolve the trace and exact target symbol first. Pass the generated absolute
path as `workload_spec_path`. The primary speedup is
`sum(call_count * baseline_latency) / sum(call_count * candidate_latency)`;
unweighted geometric mean remains secondary. Correctness stays on the frozen
oracle. Do not claim coverage for shapes missing from the trace/task.

## Operations that need several GPUs

Set sufficient `gpu_ids` and the task's actual `gpus_per_job` for collectives,
EP MoE or multi-rank tests. `job_gpu_ids` can pin one fixed group within the
pool. GEAK also detects requirements from test/benchmark code, but ambiguous
inference can fall back to one GPU: verify the resolved group before launch.
Do not infer runtime rank count from extraction/profile metadata alone.

Let GEAK's group lease keep all required GPUs locked for a measurement. This
does not change the user's scheduler CPU allocation policy.

## Research and recovery

`dra_enabled: "true"` opts into web research before optimization; it needs
WebSearch/WebFetch to be allowed in the Claude session running the workflow.
Check the session's permissions before enabling research. Leave it off when the
request does not call for it.

The standalone kernel entry has no documented generic phase-resume flag.
Do not reuse `eval_dir` and assume execution resumes where it stopped. Inspect
the original process, current patch, `director_validation.json`, baseline and
COMMANDMENT. If a continuation is needed, preserve the old run and use a fresh
run directory with a documented validated warm start supported by the checkout.
Keep the original performance denominator and revalidate; do not report a
ratio against the warm start as cumulative speedup over the original input.
