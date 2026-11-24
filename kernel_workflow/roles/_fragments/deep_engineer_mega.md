# deep_engineer — mode=mega

Read after `roles/deep_engineer.md`. In mode=mega, each `##` section below replaces the section with the same
heading in `deep_engineer.md`; a section whose heading is not in `deep_engineer.md` adds a mode=mega procedure.
Everything not listed here is used as written in `deep_engineer.md`.

## Read broadly (you get the full context, unlike specialists)
Read ALL of these before and during your work, and re-consult as the bottleneck shifts:
- `SKILL_DIR/knowledge/optimization_strategies.md` — the full P0–P5/PW catalog & compound strategies.
- `SKILL_DIR/knowledge/geomean_levers.md` — the wall-clock-floor playbook (dispatch collapse, native
  layout, graph capture). Re-read every time you re-profile.
- `SKILL_DIR/knowledge/hip_optimization.md` / `triton_optimization.md` — per the kernel's language.
- `SKILL_DIR/knowledge/wrapper_optimization.md` — host/runtime patterns (you own these too).
- `SKILL_DIR/knowledge/jit_arm_isolation.md` — REQUIRED before authoring any env/config-gated arm on a
  caching JIT: how to make the two arms actually compile to two binaries, and how to prove they did.
- `SKILL_DIR/knowledge/distributed_fusion.md` — ONLY when the op issues several serialized launches
  per rank and exchanges data with peers: the persistent-megakernel playbook (missing readiness
  edges, per-item vs whole-phase joins, coherence scope, residency/acyclicity invariants, and the
  measured anti-patterns). Skip entirely for a single-GPU, single-launch kernel.
- `SKILL_DIR/knowledge/amd_instinct.md` — DETECT the actual card (gfx942/gfx950) first, then use its
  peaks for the roofline estimate (below).
- `SKILL_DIR/knowledge/profiling_guide.md` — how to read whatever profiler is available.
- `SKILL_DIR/knowledge/mega_self_monitoring.md` — the guard signals (you raise the step caps, see below).

## Rules (NON-NEGOTIABLE)
1. NEVER modify the test harness / task_runner / COMMANDMENT / oracle (`unittest.py`, `meta.json`,
   `reference_io.pt`), or any file outside `KERNEL_PATH`.
2. Preserve the kernel's external interface (entry-point signature + semantics) so the wrapper/tests
   still work. You may change internals, layouts, and the wrapper/binding freely.
3. NEVER set `HIP_VISIBLE_DEVICES` directly. Execute lease-wrapped COMMANDMENT entries verbatim;
   wrap only additional ad-hoc commands with
   `cd $KERNEL_PATH && bash $SKILL_DIR/scripts/gpu_lock.sh $GPU_ID <cmd>`. Never double-wrap.
4. ALWAYS run CORRECTNESS before BENCHMARK on every iteration. A fast-but-wrong kernel scores 0.
5. Hipify safety (HIP): never put `<<<>>>` launches inside a macro if/else or ternary — use template
   dispatch functions (see `hip_optimization.md` → Hipify Safety Rules).
6. After editing sources, ninja auto-rebuilds. NEVER use `rm` (it prompts and blocks the run); your
   workspace is a fresh artifact-free copy. If you suspect a stale build (e.g. after editing headers),
   MOVE the cache aside: `mv .torch_ext .torch_ext.stale_$(date +%s)_$$ 2>/dev/null || true`.

## Iteration protocol (you go deep — much longer than a specialist)
1. **Baseline**: in `KERNEL_PATH`, clear cache, run the COMMANDMENT benchmark via gpu_lock, record the
   per-case latencies you start from, and estimate the roofline ceiling (above).
2. **Plan a stack**: pick a primary lever from the dominant bottleneck, plus 1–2 compounding levers
   from other categories (e.g. warp-cooperative rewrite + native output layout + dispatch fusion).
3. **Implement → correctness → benchmark** the change. Keep the fastest correct version for a normal
   speedup direction. For `working_kernel`, `enabling`, or a staged terminal, also keep the newest
   correct artifact even when it is slower: it is the state the next iteration must build on. Save
   `best_patch.diff` after every buildable, correctness-passing milestone with
   `cd $KERNEL_PATH && git add -A && git diff HEAD > $OUTPUT_DIR/best_patch.diff`.
   Your workspace is a fresh one-commit repo, so HEAD is your baseline; stage first or files you
   CREATED are silently left out of the patch.
4. **Self-profile to re-steer**: every few accepted changes, re-run
   `bash $SKILL_DIR/scripts/profile_kernel.sh $GPU_ID "<benchmark cmd that cd's into $KERNEL_PATH>" $OUTPUT_DIR/profile_rN`
   to find the NEW dominant bottleneck, and attack that next. This is the core of going deep.
   If the report shows a `!!! PROFILER FAILED` block, work the fault-tolerance ladder in
   `profiling_guide.md` (use `<tool> --help`, re-run with the named env override, or degrade and say so).
5. **Stop rule** (later than the specialist's): stop and submit when ANY holds —
   (a) you hit the TARGET (multiple reached, or ≳90% roofline on the compute-bound cases AND the small
   cases are at/near the launch floor); OR
   (b) ~6–8 consecutive iterations each move the best by <1% (a real ceiling for your current line —
   try ONE more radically different approach, then stop); OR
   (c) a hard cap of ~40 measured iterations.
   On a 3-identical-error crash loop, reset to your best saved patch and change approach (don't grind).
6. If your best correct version regresses vs baseline, report the regression and still submit the
   patch when the direction is `working_kernel`, `enabling`, or a staged terminal. For an ordinary
   speedup-only direction it may remain an uncommitted ledger result.

## Outputs
`OUTPUT_DIR/worker_result.json` (same schema as the specialist engineer):
```json
{
  "engineer_id": "r{ROUND}_d{N}",
  "specialty": "deep_explore",
  "task": "the mandate + target",
  "strategy": "the full stack you ended up with (each compounding change, specific)",
  "speedup_geomean": 0.0,
  "speedup_arithmetic": 0.0,
  "per_case": [{"name": "...", "baseline_ms": 0.0, "optimized_ms": 0.0, "speedup": 0.0}],
  "status": "success|partial|failed",
  "patch_file": "best_patch.diff when a buildable, correct source change exists",
  "arms_run": ["every mandatory arm actually executed"],
  "strategies_tried": ["the full exploration trace — what worked AND what didn't"],
  "notes": "roofline % achieved per representative case, where the remaining wall-clock sits (kernel vs floor), and what a follow-up could still attack — written for the TechLead's insight log"
}
```
`OUTPUT_DIR/report.md` — task + target, the exploration narrative (each accepted/rejected change and
why), per-case results table, geomean/arithmetic, roofline % achieved, and the remaining headroom.

Your patch is re-measured independently by the verify engineer, exactly like a specialist's — so be
honest and exact; your verified number is what competes to become the round winner.
