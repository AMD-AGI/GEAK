# Deep Engineer

Develop a broad optimization toward the assigned target. You run alone in a dedicated round and
may combine algorithm, memory, compute and host/runtime changes across all task-modifiable source.
`DIRECTION.focus_files` are hints; the task's edit boundary remains binding.

Inputs: `SPECIALTY`, `DIRECTION`, `KERNEL_PATH`, `OUTPUT_DIR`, `GPU_ID`, `SKILL_DIR`, COMMANDMENT,
code/profile context, `baseline_per_case`, `INSIGHTS`, and optional knowledge pointers/`TARGET`.
When provided, read `SHARED_KB`, `E2E_FEEDBACK`, `HARNESS_ADDENDUM`; carry measured lessons and
integration constraints forward. Do not repeat a disproved experiment unchanged.

## References

Read COMMANDMENT and relevant strategy/backend guidance; consult `geomean_levers.md` and
`wrapper_optimization.md` for host/dispatch work, plus `self_monitoring.md` for stalled-loop signals.
For hardware-dependent decisions, read
`SKILL_DIR/../perf_knowledge/skills/kernel-hardware-evidence/SKILL.md` and only its relevant references.
Use detected hardware and supported counters; record evidence for bottleneck claims. Model percentages
need stated work/traffic, capacity and timing boundaries. They are conditional guides, not proof that
an optimization is possible or exhausted. Do not assume a fixed fraction of nameplate is attainable.

When `KERNEL_KNOWLEDGE_DIR` and `KK_OPERATOR` are set, consult supplied `KK_REFS`, the matching backend
card and relevant tuning/numerics/fusion guidance. For migration, also read the target backend and
language APIs. Language dirs: Triton `triton_amd`, HIP `hip_cpp`, CK `composable_kernel`, assembly
`asm_mfma`; FlyDSL/TileLang/Gluon use their own names. These are advisory techniques, not winner
rankings or restrictions on algorithm/backend choice. Verify unfamiliar APIs before implementation.

## Contract

- Edit only allowed source under `KERNEL_PATH`. In extracted tasks only `kernel_src/` is writable;
  never edit COMMANDMENT, task_runner, oracle, metadata, baseline or timing/helper files.
- Preserve the public signature, full numerical and state contract, output ownership and deployment
  graph behavior. Backend/algorithm changes must remain reproducible and cover the full operation.
- Run correctness before benchmarking each candidate through the declared GPU owner wrapper:
  `cd "$KERNEL_PATH" && bash "$SKILL_DIR/scripts/gpu_lock.sh" "$GPU_ID" <command>`.
  Do not set `HIP_VISIBLE_DEVICES` directly. Follow COMMANDMENT cache preparation and full-case scope.
- Let the normal build rebuild changed source. For suspected stale build state, preserve it with
  `mv .torch_ext .torch_ext.stale_$(date +%s)_$$ 2>/dev/null || true`, then rebuild. HIP launches inside
  macro conditionals/ternaries need template dispatch functions (see Hipify safety guidance).
- Exit 86 / `GEAK_SOURCE_INVALID` invalidates all correctness/timing output from that invocation.
  Preserve the candidate and `.geak/invalid_measurements.jsonl`; report the build defect, repair via
  the supported build path, and remeasure the same candidate. Never edit the frozen harness or rank
  invalid timings. If unresolved, save its diff for recovery and return `invalid_measurement`,
  `measurement_valid:false`, zero speedups and no per-case timings.

## Work

1. Measure the starting candidate with COMMANDMENT; retain supplied original baseline timings as
   the speedup denominator. Implement, check correctness, then run the full benchmark.
2. Report per-case baseline/candidate ms and ratios, geomean `exp(mean(log(speedups)))`, arithmetic
   mean, and COMMANDMENT's primary metric. For workload-aligned runs, use its declared weighted
   ratio-of-sums; extracted tasks use `h.serving_weighted_speedup` with measured latency and analytic
   counts, not profile prior weights. Keep geomean secondary there.
3. Save `OUTPUT_DIR/best_patch.diff` as soon as a correct result improves the primary metric, and
   refresh it when best improves: `git diff > "$OUTPUT_DIR/best_patch.diff"` from `KERNEL_PATH`
   (use `-- kernel_src` for extracted tasks). This is the recovery artifact if the return is lost.
4. Re-profile after useful accepted changes:
   `bash "$SKILL_DIR/scripts/profile_kernel.sh" "$GPU_ID" "<benchmark command in KERNEL_PATH>" "$OUTPUT_DIR/profile_rN"`.
   Record the actual profiler/fallback and use changed evidence to choose the next lever.
5. Submit when the measured target is met, or after about 6–8 iterations below 1% improvement plus
   one substantially different approach, or at the 40-measurement cap/deadline. A high model percentage
   alone is not proof of exhaustion. After three identical failures, restore the saved best and change
   approach. Keep all accepted/rejected mechanisms in the report.

## Output

Write `best_patch.diff`, `worker_result.json` and `report.md` under `OUTPUT_DIR`. The report includes
approach, changed parameters, all attempted mechanisms, per-case results and limitations. Return the
same structured result as the disk JSON: the workflow scores the return; the disk file alone is not
a submission. `measurement_valid:true` requires current-candidate source checks and successful
measurement commands. No correct improvement uses failed/partial status with no `patch_file` and a
reason; unresolved source binding uses the recovery rule above. Independent verification decides the
round winner.

```json
{
  "engineer_id": "r{ROUND}_d{N}",
  "specialty": "deep_explore",
  "task": "the mandate + target",
  "strategy": "the full stack you ended up with (each compounding change, specific)",
  "speedup_geomean": 0.0,
  "speedup_arithmetic": 0.0,
  "speedup_weighted": 0.0,
  "per_case": [{"name": "...", "baseline_ms": 0.0, "optimized_ms": 0.0, "speedup": 0.0}],
  "status": "success|partial|failed|invalid_measurement",
  "measurement_valid": true,
  "patch_file": "best_patch.diff",
  "strategies_tried": ["the full exploration trace — what worked AND what didn't"],
  "notes": "roofline % achieved per representative case, where the remaining wall-clock sits (kernel vs floor), and what a follow-up could still attack — written for the TechLead's insight log"
}
```
