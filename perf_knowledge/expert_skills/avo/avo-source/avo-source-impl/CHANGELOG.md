# Changelog — avo-workflow

Feature history of the AVO kernel-evolution engine. Newest first. Dates are
`YYYY-MM-DD`. Categories: **Added** (new capability), **Changed** (behavior of an
existing capability), **Fixed** (bug/regression), **Docs** (documentation only).

This log tracks *engine features*, not every edit — pair each entry with its design
doc under `docs/` and its tests.

---

## 2026-11-16 — End an agent call on its final answer

- **Fixed:** `run_avo.mjs` ended an agent call only when the `claude` process
  closed or the 40-minute timer fired. `claude --print` stays alive while the
  agent still holds a background Bash task or a pending `ScheduleWakeup`, so an
  agent that had already returned its final JSON kept the orchestrator waiting,
  up to the timeout; the log then showed `✓ <label> (2400s, ...)`.
- A `result` event whose whole text is one JSON object with every required key is
  the final answer (`lib/agent_contract.mjs`). If the process is still alive
  3 seconds after it, the shim stops the agent's process group and every process
  that carries its tag, waits until none is left (`waitTagGone` in
  `lib/agent_reaper.mjs`), and returns. Progress line:
  `✂ final answer received; process still alive, stopping N process(es)`.
- An agent that exits by itself within the grace takes the previous path
  unchanged. Results that are not a final answer keep the previous path (process
  close or timeout).
- The prompt sent to agents is unchanged. Each `_shim_logs` record gains
  `early_finish`.
- Tests: `tests/agent_contract.test.mjs`, `tests/agent_early_finish.test.mjs`,
  two cases in `tests/agent_reaper.test.mjs`.

## 2026-09-20 — Task-selectable weighted commit score

- **Added:** `args.commit_score_field` accepts `verified_geomean` (default) or
  `verified_weighted`. The selected score now drives candidate ranking, commit
  gates, lineage scores, and cumulative best state.
- **Why:** weighted harnesses can legitimately improve their declared primary
  metric while regressing an unweighted mean over buckets. The old hard-wired
  geomean rejected such candidates even when the independently verified weighted
  result cleared the current weighted champion.
- Existing campaigns remain byte-for-byte compatible because the default is
  still `verified_geomean`.
- **Added:** optional `args.commit_calibrated_margin_cap` caps the Director's
  measured A/A margin for campaign-specific micro-gain decisions without
  disabling the independent Director precommit replay. The raw measured margin
  remains in the Director report for audit.

## 2026-09-17 — Restore harness `_runs.jsonl` before lineage checkout

- **Fixed:** `checkoutAvoRepo` restores tracked `_runs.jsonl` (GLM 0917 unittest
  appends one JSON line per run) in addition to `_cand_overlay/` /
  `_priv_overlay*`. Director pre-commit could accept a candidate, then commit
  failed with `avo_repo has tracked changes before checkout: M _runs.jsonl`.
- **Changed:** fresh `avo_repo` `.gitignore` also lists `_runs.jsonl` so new
  lineages do not git-add the append-only log.

## 2026-09-12 — Source scoring, GPU pool, gfx950, verify apply-check

- **Fixed:** lineage `_geomean` skips `included: false` and, when `weight`/`calls` plus latencies are present, uses the time-weighted ratio of sums. `lineage_engineer` now passes `--verified-geomean` so resume `cumulative` matches the commit gate.
- **Fixed:** source-mode final reproduction tolerance defaults to **10%** (ASM stays 3%). A `reproduction_mismatch` still reports the director geomean (and does not apply to original); the wrapper no longer maps it to `status=error` / `final_speedup=0`.
- **Fixed:** director / benchmark / profile / final validate lock from `GPU_POOL` (full `gpu_ids`) instead of `GPU_LIST[0]`.
- **Added:** gfx950 / MI355X peak table (FP16 2.5 PF, FP8 5 PF, FP4/FP6 10 PF, HBM 8 TB/s) in `roofline.mjs` and `avo_lineage.py`. Pass `hw` from the handoff.
- **Fixed:** skip the verify agent when `git apply --check` fails on the BASE_REF worktree (`PATCH_DID_NOT_APPLY`).

---

- **Fixed:** `checkoutAvoRepo` restores tracked `_cand_overlay/` / `_priv_overlay*`
  (rewritten on every unittest from `kernel_src/`) before the clean-worktree
  guard. Director pre-commit could pass, then commit failed with
  `avo_repo has tracked changes before checkout`.
- **Fixed:** source-level `avo_repo` import gitignores and tar-excludes those
  overlay trees so new lineages never track measurement byproducts.

## 2026-08-17 — Default to Team + diverse multi-start

- **Changed (default flip):** omitted `operator` now selects `team` instead of
  `single`; omitted `start_policy` now selects `diverse` instead of `best`.
- **Multi-start sizing:** `start_k` continues to default to
  `operator_directions`, so each Team round exploits the global best and uses
  strategy-diverse committed starts when the lineage has enough alternatives.
- **Explicit legacy behavior:** callers that need the paper-faithful linear
  chain must now set `operator=single` and `start_policy=best`.
- The independent verifier, Director pre-commit gate, and global-best commit
  invariant are unchanged.

## 2026-06-27 — Lower the ASM commit significance margin to 0.2%

- **Changed**: the ASM-mode default `commit_significance_margin` is now **0.002 (0.2%)**
  (was 0.003 / 0.3%); HIP/Triton stays 0.5%. ASM micro-opts (waitcnt scheduling, register
  reuse, dual-issue packing) yield genuine but tiny wins; the lower floor lets a clean
  sub-0.3% gain commit instead of being floored out.
- **Safety unchanged**: the gate stays variance-aware — `margin = max(significanceMargin,
  noiseK*cv)` — so a within-noise "win" is still rejected no matter how low this floor
  goes; only CLEAN (low-cv) sub-0.3% gains newly qualify. Override per-run via
  `args.commit_significance_margin`.

## 2026-06-27 — Run to full budget by DEFAULT; `early_stop` is now opt-in

- **Changed (default flip)**: the Evolve loop now uses the **FULL wall-clock budget by
  default**. The intelligent EARLY exits — the convergence stop (supervisor
  `converge=true`, barren-cycles, repeated-diagnosis) and the roofline early-stop — are
  **DISABLED by default** and re-enabled only with `args.early_stop=true` (aliases
  `smart_stop` / `converge_early`). The supervisor still re-plans on stall regardless, so
  new directions keep flowing; the flag only controls whether the run may STOP early.
  (Hard safety caps `max_steps` and persistent-infra-failure abort are unchanged.)
- **Why**: a `converge=true` on a hard kernel is a heuristic; spending the remaining
  budget probing is rarely worse and can surface a late win, so run-to-budget is the
  safer default. Opt into `early_stop` when banking unused budget matters more.
- Logged at Evolve start (`early_stop OFF (default): run to the full budget …`); when
  `early_stop` is off, a convergence/roofline signal prints `… noted but NOT stopping`.

## 2026-06-25 — Workflow-aware commit gate: lower significance margin for ASM

- **Changed**: `commit_significance_margin` default is now workflow-aware —
  **0.003 (0.3%) for ASM runs** vs **0.005 (0.5%)** for HIP/Triton. ASM optimization
  happens at the instruction level (waitcnt scheduling, register reuse, dual-issue /
  clause packing), where a genuine win is typically sub-1% — far smaller than the
  source/IR-level gains HIP/Triton kernels yield. The old uniform 0.5% margin floored
  out real ASM micro-gains.
- **Detection**: `target_language=="asm"` (the signal asm run configs already carry,
  e.g. `avo_asm_sigmoid.json`), with `kernel_language=="asm"` and an explicit
  `asm_mode==true` accepted as aliases. The chosen margin is logged at Setup.
- **Safety**: the gate stays variance-aware — `margin = max(significanceMargin,
  noiseK*cv)` — so a within-noise "win" is still rejected regardless of workflow type.
  Lowering the floor only lets *clean* (low-cv) sub-0.5% gains qualify, which is
  exactly when an ASM micro-gain is real given kerncap's min-of-N + median-of-repeats
  noise floor. `min_commit_speedup` (1.0) and `commit_noise_k` (0.85) are unchanged.
- **Override**: pass `commit_significance_margin` to pin any value per run.
- Tests: `tests/commit_gate.test.mjs` gains an ASM-margin case (clean +0.4% commits
  under 0.3% but not 0.5%; noisy +0.4% rejected under both). `lib/commit_gate.mjs`
  itself is unchanged (pure function); only the avo_workflow.js default differs.

## 2026-06-23 — Deterministic per-step IR capture (close the coverage gap)

Phase 2 wired IR through lineage/resume/pre-screen, but capture was best-effort: the
variation_engineer dumped IR only "when it felt like it", so most steps emitted no
`ir.json` — `score.ir` was sparse, the pre-screen never fired, and `best_ir` stayed
cold on resume. This makes capture deterministic and required (still read-only; fully
inert when `ir_feedback` is off).

**Added**
- `scripts/ir_probe.py` `combine` subcommand: merge a TTGIR file + an AMDGPU asm file
  into ONE candidate vector — high-level lowering signals (`convert_layout`,
  `num_stages`, `num_warps`, `mma_used`) plus lowered occupancy cost (`vgpr`, `sgpr`,
  `agpr`, `spill_bytes`, `lds_bytes`). AMDGCN ground truth wins on overlapping keys
  (`lds_bytes`, `mma_used`); either side may be absent (degrades to `ok=false`).
- `scripts/ir_dump_triton.sh` **cache-scrape** mode: given `IR_KERNEL_NAME`
  (+ optional `IR_TRITON_CACHE`, default `~/.triton/cache`) it finds the NEWEST
  `<kernel>.ttgir` (matching THIS candidate's compile) and the sibling `.amdgcn`, then
  runs `combine` → `$STEP_DIR/ir.json`. No agent hand-dumping. Sets
  `TRITON_ALWAYS_COMPILE=1` so cached IR matches the current source.
- `tests/test_ir_probe.py`: 7 new cases for `combine` (merge, AMDGCN override,
  TTGIR-only, both-missing, CLI happy/missing paths).

**Changed**
- `roles/variation_engineer.md` §6b: IR dump is now **REQUIRED when `ir_feedback`** —
  emit `$STEP_DIR/ir.json` and return `ir` on EVERY exit (win OR lose), produced AFTER
  the correctness compile via the cache-scrape helper. New inputs `IR_KERNEL_NAME`,
  `IR_TRITON_CACHE`.
- `avo_workflow.js`: new args `ir_kernel_name` (default `_attn_fwd`) + `ir_triton_cache`,
  threaded into both variation prompts (single + team fan-out). `logAttempt` now carries
  a compact `ir` brief (`{ok,tier,metrics}`) on every attempt record, so the IR
  trajectory — not just commits — is durable in `attempts.jsonl`.
- `run_avo_fmha_prefill_team_ir.sh`: injects `ir_kernel_name`/`ir_triton_cache` into the
  temp config (env-overridable) and echoes them in the run banner.

**Why**: turns IR capture from "looks like the agent sometimes did it" into "every step
has IR by construction" — the precondition for the pre-screen and IR-steer to actually
bite on hard kernels.

---

## 2026-06-22 — Design: Skills Discovery (cross-run, device-bound, tiered) — ASM-first

**Docs**
- New design doc [`docs/avo_skills_discovery_design.md`](docs/avo_skills_discovery_design.md)
  (**pending approval, no code yet**): distill a long-horizon run's accumulated
  experience (`attempts.jsonl` / `lineage.json` / `evolution_log/` + roofline `regime`)
  into a persistent, **per-arch** skills library that tiers each normalized strategy
  **SOLID / PROMISING / UNSTABLE / UNRELIABLE**, then seeds future runs. Closes the
  cross-run-memory gap tracked in `avo_memory_design.md` §7.3. Scoped **ASM-on-gfx942
  first** (closed, hardware-keyed lever vocabulary); deterministic distill/seed core in
  `lib/` + `scripts/skills_db.py`, keyword-then-LLM strategy normalization, graceful
  no-DB degradation. Phased plan P0 (this doc) → P1 (discover) → P2 (apply) → P3 (gfx942
  validation + generalize to HIP/Triton/Team).

---


## 2026-06-22 — IR for HIP/AMDGPU kernels (parser + rig dump helper)

Bring HIP/AMDGPU IR analysis to parity with Triton. IR stays read-only evidence;
fully inert when `ir_feedback` is off or the toolchain is absent.

**Added**
- `scripts/ir_dump_hip.sh`: HIP analogue of `ir_dump_triton.sh`. Three inputs (first set
  wins): **`IR_HIP_FILE`** (existing `.s`/asm, no toolchain), **`IR_CODE_OBJECT`**
  (`.hsaco`/`.co` → `llvm-objdump -d --mcpu=$HIP_ARCH`), **`IR_HIP_SRC`** (HIP source →
  `hipcc -S --offload-arch=$HIP_ARCH`). Degrades to an `ok=false` vector (loop IR-inert).
- `tests/fixtures/sample_hip.s` + 5 pytest cases: full `hipcc -S` fixture, `.amdhsa_*`
  directive-only path, `probe_text`/CLI `--lang hip`, and the `ir_dump_hip.sh` passthrough.

**Changed**
- `scripts/ir_probe.py` `parse_amdgcn`: robust multi-form extraction — `TotalNumVgprs`/
  `NumVgprs`/`.amdhsa_next_free_vgpr`/`.vgpr_count`, `NumSgprs`/`.amdhsa_next_free_sgpr`,
  `ScratchSize`/`.amdhsa_private_segment_fixed_size`, **new** `lds_bytes` (`LDSByteSize`/
  `.amdhsa_group_segment_fixed_size`) + **`agpr`**; `mma_used` also matches `v_wmma`.

**Docs**
- `roles/variation_engineer.md` §6b (HIP helper modes), `docs/avo_ir_feedback_design.md`
  (HIP parse+dump landed; LLVM-IR/MIR extractor + IR-diff still deferred), `README.md`,
  `TESTING.md`.
- New `docs/avo_ir_gpu_guide.md`: practical reference for GPU IR levels (Triton TTGIR /
  HIP-AMDGPU asm), reading each level, optimizing with IR (source-knob vs hand-asm),
  dumping IR, and exporting a runnable `.co`/`.hsaco`. Linked from README + IR design doc.

> pytest `test_ir_probe.py`: 16 passed (was 11).

---

## 2026-06-22 — IR Layer 1: lineage `score.ir`, resume `best_ir`, `ir_prescreen`, Triton helper

**Added**
- `avo_lineage.py commit --ir-file`: optional IR JSON persisted under `committed[].score.ir`.
- `avo_lineage.py resume`: **`best_ir`** from the current best lineage node.
- `avo_workflow.js`: on resume, merge **`best_ir`** from `avo_lineage.py resume`; restore
  `bestIr` for `irSteer`; write `._commit_ir_vN.json` before `lineage_engineer` commit;
  optional **`ir_prescreen`** (`predictRegression` high-confidence skip before verify,
  single + team fan-out).
- `scripts/ir_dump_triton.sh`: TTGIR path → `ir_probe.py parse` → `IR_OUT` JSON (rig contract).
- `roles/lineage_engineer.md` / `roles/director.md` / `README.md` / tests updated.

**Changed** (L1.3 report UX, same release)
- Before Report, when **`ir_feedback`** is on, writes **`EVAL_DIR/ir_report_context.json`**
  (`best_ir`, `irSteer`, roofline snapshot) and passes **`IR_REPORT_CONTEXT`** +
  **`IR_FEEDBACK`** into **`tech_lead:report`**. **`profile_engineer`** baseline/reprofile
  prompts receive **`IR_FEEDBACK`**; **`roles/tech_lead.md`** and
  **`workflows/roles/profile_engineer.md`** describe optional IR sections in Markdown outputs.

---

## 2026-06-22 — Orchestrator: resume `prior_steps` from `avo_lineage.py`

**Fixed**
- `avo_workflow.js`: on `resume`, **overwrite** `setup.prior_steps` with
  `avo_lineage.py resume` output so Evolve step numbering stays correct even if the
  Director returns an inflated `wc -l`-style count.

**Changed**
- `run_avo_fmha_prefill_team_resume.sh`: optional **SIGTERM** of an in-flight duplicate
  resume (`SKIP_STOP_FMHA_RESUME=1` to skip); prints **`prior_steps` probe** before
  `node run_avo.mjs`.

---

## 2026-06-20 — Resume `prior_steps`: max evolve step index (not directory count)

**Fixed**
- `avo_lineage.py resume` now emits **`prior_steps`**: max `NNNN` from
  `variation_NNNN*` / `escalate_NNNN*` under `EVAL_DIR`. Team mode previously made
  `ls variation_* | wc -l` in `director.md` inflate the step (e.g. step 60 → 174).
- `roles/director.md`: resume path must pass through `prior_steps` from
  `avo_lineage.py resume`; removed the `wc -l` recipe.

**Tests**
- `tests/test_avo_lineage.py`: `_max_completed_evolve_step` + resume JSON
  `prior_steps` coverage.

---

## 2026-06-18 — Vendored `run_avo.mjs` (canonical path in avo-workflow)

**Changed**
- **Canonical copy:** `run_avo.mjs` now lives in `PerfSkills/avo-workflow/` next to
  `avo_workflow.js` so the headless shim versions with the engine.
- **Monorepo forwarder:** `avo_projects/run_avo.mjs` (outside PerfSkills git) is a
  thin `spawnSync` delegate so existing `run_avo_*.sh` launchers keep using
  `node run_avo.mjs <config.json>` from the avo_projects root.

**Docs**
- README *Headless runs (`run_avo.mjs`)* section.

---

## 2026-06-18 — Supervisor normalization + orchestrator heartbeat (shim)

**Added**
- `lib/supervisor_directive.mjs` + `tests/supervisor_directive.test.mjs`: normalize
  supervisor REDIRECT JSON (every `new_directions[]` entry gets `name`/`expected`/`prompt`;
  repair insights for bad rows).
- `run_avo.mjs`: `syncAvoHeartbeat(stateDir, fields)` merges authoritative progress into
  `avo_state/heartbeat.json` (`step` = Evolve loop counter).

**Changed**
- `avo_workflow.js`: after each supervisor success, call **`recordSupervision`** (when
  present) with the **normalized** directive; ESCALATE uses normalized directions;
  **`pulseOrchestratorHeartbeat`** at end of each Evolve iteration and on supervisor /
  escalate / converge_stop. Pass **`SYNC_HEARTBEAT`** to `variation_engineer` when the
  shim provides `syncAvoHeartbeat` so role heartbeat bash does not fight the loop.

**Docs**
- `docs/avo_memory_design.md` (heartbeat vs attempts), `docs/avo_supervisor_design.md` §4,
  `roles/supervisor.md` (audit when `recordSupervision` exists), `roles/variation_engineer.md`.

---

## 2026-06-18 — Roadmap doc (pending approval)

**Docs**
- `docs/avo_workflow_roadmap.md`: prioritized delivery phases (supervisor hardening,
  commit objectives, roofline gates, heartbeat, nudge/crossover/adaptive verify,
  RAG/E2E/sync matrix) + approval block; README See also link.

---

## 2026-06-18 — Commit gate defaults slightly relaxed (B1 / C1)

**Changed**
- Default `commit_significance_margin` **0.01 → 0.005** (fixed floor when `cv` is absent).
- Default `commit_noise_k` **1.0 → 0.85** (scales the variance-aware band when `cv`
  is present). Per-run overrides via workflow `args` unchanged.

---

## 2026-06-18 — Supervisor convergence + empty-direction backstop

**Added**
- Deterministic root-cause exit in `avo_workflow.js`: optional supervisor `converge`,
  `converge_barren_cycles` (default 4), `converge_dx_repeat` + `dx_sim_threshold`
  (default 3 / 0.55 token–Jaccard on consecutive diagnoses). Stops Evolve early and
  banks budget instead of endless REDIRECT/ESCALATE churn.
- Supervisor prompt input `REDIRECTS_SINCE_COMMIT`.

**Fixed**
- Orchestrator fallback when `new_directions` is empty or all names are already in
  `FAILED_DIRECTIONS` (injected escape steer + insight line).

**Docs**
- `docs/avo_supervisor_design.md` §5b + tunables; `roles/supervisor.md`: correct
  `supervisor_log.jsonl` append recipe (`ts` = numeric epoch, not calendar strings).

---

## 2026-06-16 — Docs: start-point selection design

**Docs**
- `docs/avo_start_points_design.md` (new): standalone design for start-point selection
  — why a single-best chain collapses to one local optimum; the `best | topk | diverse
  | elites` policies and where each applies (`operator=single` chain vs `operator=team`
  / ESCALATE fan-out vs supervisor backtrack); the exploit+explore split (best always a
  start); and the **no-regress invariant** (global best advances only on a verified
  global improvement; off-branch commits parent on their own ref → tree, never demote).
  Links the roofline `regime` (elites cells) + the Integrator merge. README See also.

---

## 2026-06-16 — Accuracy & rigor: measured bytes (A2), variance-aware gate (C1), autotune sweep (A4)

Harden the roofline objective so it can be trusted over a long lineage and push the
last mile toward the ceiling. All deterministic cores unit-tested; every path
gracefully inert without the new signals.

**Added**
- `lib/measure.mjs` (C1, pure): `median`/`mean`/`stddev`/`mad`/`cv`, `dropOutliers`
  (MAD rule), `summarizeRepeats`, `aggregateCV`. + `tests/measure.test.mjs` (11).
- `lib/autotune.mjs` (A4, pure): `expandConfigSpace` (cartesian product,
  deterministically capped + even-subsampled, default kept), `selectBestConfig` (best
  verified geomean, tie → closest to roofline), `configKey`, `excludeFailed`. +
  `tests/autotune.test.mjs` (8).
- `scripts/rocprof_probe.py` (A2, parse half): rocprof counter CSV → measured
  `hbm_bytes` + occupancy/MFMA-active vector; `parse`/`collect` CLI (`collect`
  rig-only/inert). Fixture `tests/fixtures/sample_rocprof.csv` + `tests/test_rocprof_probe.py` (9).
- Args: `commit_noise_k` (C1), `autotune` + `autotune_max_configs` (A4).

**Changed**
- `lib/commit_gate.mjs` (C1): variance-aware margin — `max(significanceMargin,
  noiseK·ver.cv)`; reads the verifier's measured `cv`. Backward compatible (no `cv` →
  legacy fixed margin). +4 gate tests.
- `lib/roofline.mjs` + `scripts/avo_lineage.py` (A2): `analyzeCase`/`_roofline_case`
  prefer `measured_bytes` over analytic `bytes`, tag `bytes_source`, carry rocprof
  `counters`. +3 roofline tests.
- `avo_workflow.js`: gate `noiseK` wired; `VERIFY_SCHEMA` gains `cv`; per-case schema
  gains `measured_bytes` + `counters`; `AUTOTUNE` hint (`{enabled, max_configs}`)
  threaded into both variation prompts.
- `roles/avo_verify.md`: robust aggregation (outlier drop + median) + report `cv`;
  optional rocprof `measured_bytes`/`counters`. `roles/variation_engineer.md`: §5b
  bounded autotune sweep (rules from `lib/autotune.mjs`).
- `run_tests.sh`: runs `measure` + `autotune` node suites.

**Docs**
- `docs/avo_roofline_design.md`: new §13 (A2/C1/A4) + updated follow-ups. README args +
  Files list. (A2/IR rig `collect`/`dump` remain rig-only; expert bar B1 still pending.)

> JS suites 8 (+measure 11, +autotune 8, +gate 4, +roofline 3); pytest 46 (+rocprof 9).

---

## 2026-06-15 — IR feedback Phase 1 (steer enrichment, behind a switch)

Thread IR evidence into the operator's steer (read-only), gated by a master switch.
Makes a roofline regime *actionable* at the source line; fully inert when off.

**Added**
- Master switch `ir_feedback` (default **false**, opt-in) + `ir_lang`
  (`auto|triton|hip|off`) + `ir_tier` (`auto|ttgir|amdgcn`) args.

**Changed**
- `avo_workflow.js`: `steerFrom` now adds an `ir` field via `irSteer(regime, bestIr)`,
  threaded into the `ROOFLINE` block for `variation_engineer` (single + fan-out) /
  `supervisor` / team planner; `bestIr` (current best's IR vector) captured on commit;
  `VARIATION_SCHEMA` gains an optional `ir` vector; the IR switch is passed to workers.
  All inert when `ir_feedback=false` or `ir.ok=false`.
- `roles/variation_engineer.md`: use `ROOFLINE.ir.hints` to pick the *specific* lever;
  optional §6b step to dump+parse IR via `ir_probe.py` and report the `ir` vector.

**Docs**
- `docs/avo_ir_feedback_design.md`: Phase 1 marked ✅ LANDED; `ir_feedback` default
  corrected to `false` (opt-in until the rig dump is wired). README args + See also.

> Next (Phase 2): per-step IR *dump* on the rig + store `ir` in `lineage.json`
> (`avo_lineage.py`) so steering survives resume and the report shows it.

---

## 2026-06-15 — IR feedback Phase 0 (deterministic probe + evidence core)

Land Phase 0 of `docs/avo_ir_feedback_design.md`: lower-level IR as a read-only
feedback signal that turns a roofline *regime* into a *specific source lever*. Pure,
fixture-tested core only — no loop change yet; fully inert when IR can't be produced.

**Added**
- `lib/ir_evidence.mjs` (pure): `irSteer(regime, ir)` combines regime + IR metrics into
  ranked, specific levers (e.g. memory-bound + `convert_layout>0` → "remove the layout
  conversions"; compute-bound + `mma_used=false` → "hit an MFMA path"); and
  `predictRegression(irBase, irCand)` — a high-confidence pre-verify negative screen
  (spill appeared / matrix cores dropped / `convert_layout` grew), config-aligned, that
  can only *skip* a likely loser, never accept.
- `scripts/ir_probe.py` (deterministic parse half): `parse_ttgir` (convert_layout,
  num_warps, num_stages, lds_bytes, mma_used) + `parse_amdgcn` (vgpr, sgpr, spill_bytes,
  mma_used) → bounded `ir` metric vector; `parse`/`dump` CLI (`dump` is rig-only → inert
  without the toolchain). Graceful `ok=false` on unsupported/empty input.
- Fixtures `tests/fixtures/sample.{ttgir,amdgcn}`; tests `tests/ir_evidence.test.mjs`
  (12) + `tests/test_ir_probe.py` (10); `run_tests.sh` runs `ir_evidence`.

**Docs**
- `docs/avo_ir_feedback_design.md`: Phase 0 marked ✅ LANDED.

> Design: [`docs/avo_ir_feedback_design.md`](docs/avo_ir_feedback_design.md). Next
> (Phase 1): thread an `IR_EVIDENCE` block into the steer (read-only) + a
> `variation_engineer §4` bullet; then Phase 2 stores the `ir` vector in `lineage.json`.
> The actual per-step IR `dump` is a rig task (toolchain-dependent).

---

## 2026-06-15 — AVO-on-Team completion: regime→specialist dispatch + Integrator merge

Land §7 of `docs/avo_on_team.md`: the Team operator now puts the right specialist on
the binding ceiling and **merges** complementary winners instead of only picking the
best — fully reusing Team's hierarchy under the roofline objective. All deterministic
cores unit-tested; graceful when there's no regime telemetry.

**Added**
- `lib/roofline.mjs` `assignSpecialists(regime, k)` — K specialist tracks,
  dominant-ceiling first then orthogonal cover (cycles >4); `unknown`→algorithm-first.
- `lib/archive.mjs` `selectMergeGroup(passers)` (complementary, **same-base**,
  best-per-regime) + `mergeImproves(mergedSpeedup, members)` (merge must beat the best
  individual it combined). Both pure + unit-tested.
- `roles/avo_integrator.md` — base-ref-aware integrator (derived from Team's
  `integrator`, parallel to `avo_verify`): worktrees the shared base, stacks/hand-merges
  the winners, emits one incremental merged patch.
- Tests: `tests/roofline.test.mjs` +3 (assignSpecialists), `tests/archive.test.mjs` +7
  (selectMergeGroup / mergeImproves).

**Changed**
- `avo_workflow.js`: `planTeamDirections` passes `SPECIALISTS` (regime-ordered) to the
  Team `plan_round` and backfills each worker's `specialty`; `fanOutEvolve` now returns
  all gated `passers` (with per-candidate regime); new shared `commitRoundResult` tries
  an Integrator merge (`selectMergeGroup` → `avo_integrator` → independent `avo_verify`
  → commit iff `passesCommitGate` **and** `mergeImproves`, else commit best). Used by
  both `operator=team` and ESCALATE. `INTEGRATE_SCHEMA` added.
- `roles/variation_engineer.md`: honor `DIRECTION.specialty` (team round) and load only
  that specialist's knowledge, so K workers cover distinct ceilings and stay mergeable.

**Docs**
- `docs/avo_on_team.md`: §7 marked **landed** (was designed-pending); mapping table +
  §6 status updated (Integrator + specialist dispatch now landed; P3/P4 remain pending).
- README Files list: `roles/avo_integrator.md`.

> Design: [`docs/avo_on_team.md`](docs/avo_on_team.md) §7. Outer-loop invariants
> unchanged (deterministic gate, independent verify, wall-clock budget, lineage); the
> merge is "just another gated candidate", never a new trust path.

---

## 2026-06-15 — AVO-on-Team design completion + IR feedback + telemetry guard

Land the IR-feedback design, harden the roofline telemetry, and **complete** the
"AVO uses Team" design so it reflects the shipped `operator=team` and the remaining
Integrator/specialist work.

**Added (design only)**
- `docs/avo_ir_feedback_design.md` — design for lower-level **IR as a read-only
  feedback signal** (Triton TTGIR first, HIP LLVM-IR/MIR later) to turn a roofline
  *regime* into a *specific source lever*. Proposes deterministic `scripts/ir_probe.py`
  (dump+parse) + `lib/ir_evidence.mjs` (`irSteer`, `predictRegression`), a per-version
  `ir` vector in `lineage.json`, and a phased landing route (Phase 0–4), all fully
  graceful (`ir.ok=false` → inert). No loop code landed yet (doc + plan).

**Changed**
- `lib/roofline.mjs` + `scripts/avo_lineage.py`: **telemetry sanity gate** — a case
  whose `fraction_of_roofline > 1 (+tol)` has untrustworthy flops/bytes; it is now
  treated as INVALID (degraded like missing telemetry, excluded from the signal) and
  surfaced via `rooflineSummary.n_invalid` + a loop warning, so a bad FLOP/byte recipe
  can neither fire the early-stop nor distort steering. Tests added.

**Docs**
- `docs/avo_on_team.md` **completed & corrected**: marked `operator=team` as **landed**
  (was stale "next step"); described the round as actually implemented (reused Team
  `tech_lead` `plan_round` under the `ROOFLINE` steer → `variation_engineer` fan-out →
  independent verify → commit best); updated the component-mapping table
  (roofline/archive/IR rows + Integrator "designed-not-wired"); added **§7** completing
  the design — **regime→specialist dispatch** (`recommendLevers().specialty`,
  `assignSpecialists`) and **Integrator merge of complementary winners** (via
  `archive.crossoverPairs` + Team `integrator`, gated by the same commit gate) — plus a
  landed-vs-pending status section (§6) and acceptance criteria.

---

## 2026-06-15 — Roofline-anchored objective

Turn AVO's objective from purely relative ("beat the last version") into a
hardware-anchored one ("approach the MI300X roofline, then stop"), so the engine
deliberately closes the gap to the limit instead of plateauing early. Fully
graceful: inert (pure relative mode) when a kernel has no FLOP/byte telemetry.

**Added**
- `lib/roofline.mjs` — pure roofline model: arithmetic intensity, roofline bound,
  % of peak, regime classification (`memory-bound` / `compute-bound` / `basement` /
  `unknown`), and a regime→lever recommender. Peaks from
  `kernel_knowledge/hardware/cdna3_mi300/peak_tables.md`.
- `lib/headroom.mjs` — roofline early-stop (`shouldStopRoofline`, default target
  0.85 after `min_commits`) + Amdahl attention allocation (`allocateAttention`).
- `lib/archive.mjs` — MAP-Elites quality-diversity archive keyed by
  (strategy × regime), `selectElites` for diverse fan-out, `crossoverPairs` for
  complementary-regime parents.
- `start_policy="elites"` in `lib/start_select.mjs`.
- `scripts/avo_lineage.py`: roofline computation (mirror of `lib/roofline.mjs`),
  stored as `score.roofline` in `lineage.json`; new `roofline` subcommand; `--hw`
  flag on `seed`/`commit`; `resume` now restores roofline fraction/regime.
- New args: `hw` (`gfx942` | `gfx942_mi325`), `roofline_target`.
- Telemetry contract: per-case `flops` / `bytes` / `dtype` (emitted by
  `avo_verify` / `variation_engineer`, optional).
- `knowledge/roofline_playbook.md` (regime→lever map + per-op FLOP/byte recipes +
  stop rule); `docs/avo_roofline_design.md`.
- Tests: `tests/roofline.test.mjs` (14), `tests/headroom.test.mjs` (8),
  `tests/archive.test.mjs` (8); `tests/test_avo_lineage.py` +7 (total 24).

**Changed**
- `avo_workflow.js`: computes the current best's roofline after each commit, injects
  a `ROOFLINE` steer block into `variation_engineer` / `supervisor` / team planner,
  applies the roofline early-stop in the Evolve loop, stores per-version `regime`,
  and surfaces `roofline_fraction` / `roofline_regime` in the report + return value.
- `roles/{variation_engineer,supervisor,avo_verify,lineage_engineer}.md` consume /
  emit the roofline signals.
- `run_tests.sh` runs the three new node suites.

**Fixed**
- Root `.gitignore`'s generic `lib/` (Python template) rule recursively ignored
  `avo-workflow/lib/`, silently keeping the pure-JS source modules
  (`commit_gate`, `start_select`, roofline/headroom/archive) out of git — a clean
  clone crashed at the `import` in `avo_workflow.js`, and the node unit tests failed.
  Added a negation (`!avo-workflow/lib/` + `!avo-workflow/lib/**`) with an inline
  comment; documented in README ("Repo note") + TESTING.md.

**Docs**
- README: roofline objective section, new args, updated Files list, repo note.
- PLAN.md: Step 5 (roofline-anchored objective). TESTING.md: Layer-1 entries 1d–1f.

> Design: [`docs/avo_roofline_design.md`](docs/avo_roofline_design.md).
> Status: deterministic logic landed + unit-tested (design-only). Real-machine
> rocprof telemetry, an expert-library reference bar, and a crossover/autotune
> variation primitive are tracked follow-ups (see the design doc §13).

---

## (prior) — Step 2 native deterministic loop (baseline of this changelog)

State before this changelog began, for context (not exhaustively dated):

- Native `avo_workflow.js` controller: wall-clock budget loop, reset-to-best,
  deterministic commit gate (`lib/commit_gate.mjs`), git-tagged lineage
  (`avo-v{N}`), stagnation detector + LLM supervisor (REDIRECT/ESCALATE),
  Diverse Fan-out start points (`best`/`topk`/`diverse`), `operator=single|team`,
  delayed profiling, resume from durable state, end-of-run cleanup.
- `scripts/avo_lineage.py` (seed/commit/context/resume/heartbeat),
  `scripts/avo_lineage_report.py`; roles + knowledge + `docs/` design set.
- Same I/O contract as `workflows/team_workflow.js` (drop-in kernel engine).
