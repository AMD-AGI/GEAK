# Experiment records — ledgers, the experiment contract, and the delivery record

**What this chapter decides.** What a Triton-family run writes down, when, and in which shape, so
that every number a gate acts on is auditable after the run and a resumed run reads one history.
You are here at four moments: when you pin the task contract (before the first measurement),
after every round (the round ledger), whenever a bounded side-experiment or an anomaly check is
run (sections 9 and 9a), and at close (the delivery record that mirrors `final_report.json`).

The templates are copyable and serve both sides of the champion handoff
(`tile-programming-triton` on the plain tier, `tile-programming-gluon` on the deep dig): the
ledger shape is the same, so a resumed run reads one history. Policy lives in
`../../skill.md` (the pack entry) and `index.md`;
execution order and self-monitoring in `climb.md`; harness and timing fields in
`benchmark-hygiene.md`; acceptance and the `final_report.json` contract in `close.md`.

**Who writes these, and whose numbers they carry.** The deep_engineer running the `deep_explore`
direction keeps these records in its OUTPUT_DIR; GEAK owns the round loop and acceptance. The
accepted latency numbers are the ones `verify_engineer` / `measure_legs`
(`e2e_workflow/scripts/harness_lib.py`: CUDA events with per-sample sync, read-evict flush,
median, a fresh process per leg, same-window baseline) produced, and a commit needs GEAK's
`MIN_IMPROVE` = 2 %. The records below carry those numbers **by reference** plus the pack's own
evidence (budget, profile, IR); a pack-side measured noise band may only make a verdict STRICTER,
never admit a sub-2 % change. The `recordctl` journal and the canonical record tools below are
optional bookkeeping the engineer may use (`orchestration.md`, `## Optional bookkeeping tools`).

## Canonical optimization journal

The durable process record is `optimization_journal.jsonl`, written through the stage-scoped
`recordctl` facade (`scripts/recordctl.py`). One schema covers three work-unit grains:
`sweep_batch` (one batch, not one row per point), `branch_arm` (one arm result), and
`climb_round` (one candidate round). Legacy `round_record` v3 rows remain readable; new
work-kind rows use the v4 journal envelope.

Every v4 row carries identity hashes for body, layout, comparator, measurement boundary/method,
environment, toolchain, skill build, and policy; the hypothesis, bounded alternatives, change
reference, normalized comparison and measurement summary, oracle, disposition, evidence
references, invalidations, and next action. Raw profile/IR/ISA/log bodies stay in their files
and enter the journal only by reference.

Before each new CLIMB hypothesis, any identity change, or BRANCH re-entry, run
`recordctl recall` and attach its hashed receipt to the next append. The first SWEEP batch has
an explicit `first_sweep` exemption and empty recall is valid. A declared valid measurement must
be appended before the next work unit or stage transition. These checks do not require
classification, a bound class, a candidate stack, or a particular lever. `run_round.sh`,
`classify.py`, and `gate.py` are not prerequisites for this journal contract.

These records are modeled on the two-tier flow: **two baselines** (`plain_baseline` target line
+ `gluon_anchor` working baseline; `recover.md`), **three-evidence closure** (budget + profile +
IR; `climb.md`), the **layer backbone**, and the bound-class escalation gate (`entry.md`). There
is no `optimization_mode`; layer states are `open | active | closed | blocked | out_of_scope`.

## 1. Task Contract

The **full expansion** of the canonical contract pin (production boundary, comparator / budget,
correctness oracle, shape set, toolchain and the bounded round budget — pinned before the first edit,
`entry.md`; the short pinned form is written first): fields duplicated below are the SAME pin — do not re-invent them. Extra
fields here are the host_overhead decomposition, config ridges, timing_method, and the
escalation/comparator artifacts. The per-round evidence record and the toolchain-pin stamp are
inherited from that pin.

```text
repo / kernel file / kernel name / public wrapper / launch site
geak: { round, direction_id, gpu_id, output_dir }   # GEAK round loop + acceptance
measured boundary: kernel-only | wrapper+kernel | full operator
production boundary: eager | cuda_graph        # which wins are real (launch-fusion law)
timing_method:
  acceptance: harness_lib.time_op (CUDA events, per-sample sync, read-evict flush, median) /
              GEAK verify + measure_legs (fresh process per leg, same-window baseline)
  graph-served: harness_lib.time_op(graph=True) -- captured-graph replay, same event+flush method
  search/screening only (ab_bench): cudagraph_amortized_replay | do_bench_cudagraph | batched_wallclock
baseline_identity: live production path @ production boundary (NOT a strawman)
host_overhead_us: wrapper/dispatch/launch share of the wall time (decomposed)
target arch: gfx950 (default, main line) | gfx942 (downgrade)
toolchain pin (SINGLE source of truth; stamp every result/report with it):
  ROCm / Triton tag (+ LLVM or commit hash if a custom build) / PyTorch / docker image / GPU arch
  # any artifact/result produced on a different pin must be flagged + re-validated before it counts
  # (benchmark-hygiene.md ## Toolchain pin)
workload class: gemm | attention | reduction/elementwise | other   (../workloads/intake.md)
dtype + FP8/FP4 gate: gfx950 native OCP e4m3 vs gfx942 e4m3fnuz upcast risk
correctness oracle + tolerance:
config set (seed / expanded served range + ridges):   # anti-overfit
shape stream (quick / full):
fallback / dispatch policy:
noise / repeat policy: commit gate = GEAK MIN_IMPROVE 2%; measured noise band may only tighten it
parity threshold (declared by the run; default 0.95, parity_gate --threshold):
primary metric BY BOUND CLASS: MFMA-eff | achieved HBM BW | ds_read interval |
                               VGPR/spill | pipeline coverage
secondary metric:
escalation-gate inputs: bound_class + ideal-vs-as-built budget gap
comparator / ceiling evidence: stretch/reference kernel (or "none in contract"); a
  stay-plain / "at ceiling" close OWES gap_decomposition + asm_loop_audit vs it (## 2, ## 8)
per-round evidence record: the raw artifacts (att/mfma_eff.txt + ir/asm_audit.txt: ATT rollup +
  static asm_loop_audit) + your round-log entry (## 1c below); see profile.md ## Output
round budget N / timeout policy:
stop condition:   (close.md ## Stop conditions)
```

## 1b. Kernel register

One record for the kernel under optimization; the inner records (sections 2-8) fill it in. (The
multi-direction register that ran one row per config-bucket direction was retired with that
regime.)

```text
bound_class: compute | hbm | lds | register | latency | occupancy
dominant_stage (floor-probe):
plain_ceiling (vs production baseline):
escalation_decision: escalate | stay_plain
parity_status: reached | parity_unreached     # parity_unreached is carried on EVERY later number
gluon_result (vs plain target; win | no_win):
llvm_codesign_result (only if sanctioned; win | no_win | n/a):
winning_tier: plain | gluon | gluon+llvm
baseline_status: ran | crashed | oom | unsupported | missing   # did the production baseline run?
win_type: speedup | unblock | tie_kept    # unblock = baseline did not run; any correct candidate is the win (vs_baseline null)
per_bucket_fallback (kept production baseline for which shape regions, via dispatch):
fallbacks_tried (mechanisms/layers attempted before block/defer):
error_class (if not closed): retryable | scoped_ceiling_needs_user | global_blocker
resume_hint (if deferred_needs_user; what to re-run first after the user resolves it):
inherited_from (if warm-started from a prior deliverable: prior run/build identity + re-validated y/n; else n/a):
rounds_spent / round_budget_N:
state: open | active | closed | blocked | deferred_needs_user
```

`state = deferred_needs_user` is a scoped ceiling parked for the user (it does NOT stop the run);
keep the best in-scope result as fallback and re-optimize on resume
(`## 1d. Residual / Deferred-Task & Resume ledger`).

## 1c. Per-round log entry — written as you go, not reconstructed

The v4 journal row is the reviewable entry: it stores the bounded hypothesis, alternatives,
change reference, result summaries, disposition, next action, and typed evidence references. A
separate `decision_log.md` is optional commentary and is never the only carrier for an
optimization result.

Rules that keep the log honest:

- **A round whose edit did not land is not a negative result for the lever.** Record it as "the
  change did not take effect" — a different finding, and a cheap one to check with
  `kernel_workflow/scripts/kernel_tools/probe.py`.
- **Record the rejects too, with their numbers.** A lever you measured and dropped is the most
  reusable thing in the log; an unrecorded reject gets re-tried by the next run.
- **Label injected numbers.** A number measured on a build that re-injected plain's
  auto-pipeliner (`gluon_swp` / `patch_reinject` / `patch_async_reinject`) is recorded with the
  label `injected`: it is a diagnostic of the `lost_pipeline` debt below the parity gate or a
  last-resort parity build, never a win, and never on an incumbent (already-Gluon) kernel
  (`recover.md`).
- **A path in the record is a reference, not a guarantee that the artifact came from the run it
  is filed under.** A file left untouched by a failed write keeps a valid path and stale
  contents, and the log records it either way
  (`benchmark-hygiene.md ## Provenance: every gate acts on a number you already have`).
- **A run discarded because the *instrument* was unsound is VOID; a run that completed on a sound
  instrument and showed no improvement is NEGATIVE.** Only the second is evidence about the
  lever — the first carries none. Misfiling a void run as a negative is **permanent**: the lever
  goes into the log as "tested, did not help", it is never retried, and nothing in the record
  distinguishes the two afterwards. So write the disposition **at the time, as a named field,
  carrying the reason** — not as prose a later reader has to infer. **This is not the "edit did
  not land" rule above**: same disposition, unrelated cause. That one is detected by re-checking
  that the edit took effect; this one is invisible to that check, because the edit *did* land
  and the kernel *did* run — what failed was the instrument. In one measured instance a batch was
  voided by its own null control firing while a separate memory-side exclusivity gate passed the
  same device; in another, a timing round was voided because the contamination was **additive**,
  which makes a *negative* specifically uninformative rather than refuting
  (`benchmark-hygiene.md ### The additive case: the control gets greener as the measurement gets dirtier`).
- **Cite a field the record carries, never an ordinal position in a view.** In one measured
  instance a summary and its source journal numbered their rows from different origins, so a
  citation was off by one with **both documents internally consistent and the wrong row
  well-formed** — a class no contradiction-check can detect, because there is no contradiction to
  find. This is the software form of the hardware case in
  `benchmark-hygiene.md ### Device identity: a wrong index returns a plausible number, never an error`.

## 1d. Residual / Deferred-Task & Resume ledger

One row per `deferred_needs_user` task (scoped ceiling parked for the user). It is the resume
input: a follow-up run re-enters it, reuses the records above, re-optimizes, and merges/updates
the final report in place. The handoff block it is filled from is
`triage.md ### needs_user handoff (for a scoped ceiling, B)`.

```text
task_id (-> Kernel register):
error_class: scoped_ceiling_needs_user
what_failed (one line) + evidence (IR/asm/compile/correctness/env):
falsifying_probe (command + exact output that promoted A -> B; triage.md):
user_action_required: sanction LLVM co-tuning | switch build/version | provide oracle/data |
                      fix scoped env | other
best_kept_for_this_bucket (fallback held = production baseline or best in-scope result):
resume_entry (what to re-run first once resolved):
status: awaiting_user | resolved | resumed_done
```

A tile retune is not a deferred task of this pack: it is a `resweep_request` returned in the
deep_engineer's result for tech_lead to hand to GEAK's plain tuning (changing the tile re-recovers
every layout).

## 2. Escalation Gate Record

Replaces the old mode comparison. The gate runs on plain aggregates (no transcription); feeds
`final_report.escalation`. See `entry.md` (escalation gate).

```text
bound_class: compute | hbm | lds | register | latency
ideal_budget (relevant class):       # budget.md; every % carries numerator_basis + denominator_basis
as_built_observed (relevant class):
gap:
decision: escalate | stay_plain | unproven   # unproven = stay_plain claimed without comparator evidence
reason:
# if stay_plain: comparator gap-decomposition is REQUIRED (else -> unproven)
comparator_identity:              # the stretch/reference kernel named in the contract (or "none in contract")
gap_decomposition_artifact:       # path: same-boundary A/B latency gap vs comparator + bound-class attribution
asm_audit_artifact:               # path: asm_loop_audit.py output for BOTH this kernel and the comparator
plain_directions_tried (Stage-Plain best-of-N, 0-15 ladder):
plain_best_result (the target line):
# if escalate:
explicit_control_needed: lds_layout | async_pipeline | mfma_layout | scaled_mfma |
                         register_slicing | llir_scheduling
```

## 3. Transcribe / Anchor + Calibration Record

The first Stage-Gluon step; **not** optimization. Regression vs plain is expected and is not a
reject. See `transcribe.md` (transcription) and `recover.md` (parity gate). The
`plain_ttgir_layouts_recovered` block is produced by `scripts/ttgir_bridge.py recover --arch gfx950`
and assembled/auto-filled into the anchor record by `scripts/recover_gluon.py --record`; the
`layout_equivalence_vs_plain` line comes from `scripts/ttgir_bridge.py verify --plain … --anchor …
--arch gfx950` (LinearLayout normal-form diff) and the `correctness` line from the oracle. Gluon
pack, since this is the first Stage-Gluon step; from the triton pack the record is filled after
the champion handoff.

```text
plain_ttgir_layouts_recovered:
  #blocked -> gl.BlockedLayout(...)
  #mma / #amd_mfma -> gl.amd.AMDMFMALayout(...)
  #shared -> Padded / SwizzledSharedLayout(...)
  #dot_operand -> gl.DotOperandLayout(...)
  convert_layout placements
  num_stages -> champion record + budget parameter ONLY (no Gluon pass consumes it in 3.8.0);
               it sizes the AUTHORED pipeline depth, it is not a knob to carry over
unrecoverable (if any): probe build -> re-dump at ns=1 -> forced divergence (ledger) | structure_suspect
equivalence: layout diff (ttgir_bridge verify) | oracle @ tol | determinism (~40 launches) | asm parity
correctness == plain: pass | fail
perf_delta_vs_plain: (regression expected, NOT a reject)
residual attribution (suspects, exactly): lost_pipeline | lost_layout | lost_RA
                       # lost vectorization is a lost_layout; fix = re-recover, not hand-patching
parity: threshold (declared; default 0.95) / champion_ms / anchor_ms / status: reached | parity_unreached
injected diagnostic (optional, labelled `injected`, never a win): lost_pipeline debt measured
gluon_anchor_metrics.json: written
tile_op_cards_emitted: one anchor-state card per tile-op (load A/B, MFMA, LDS rd/wr,
  epilogue); Handoff = whatever was authored/transcribed -- on the Gluon path no pass pipelines
  (3.8.0), so `auto` describes only plain Triton or a labelled injected build
  (../tile-programming/tile-op-contract.md)
# calibration (now layouts/stages are explicit, the budget can be decomposed):
R_acc / R_operand / R_prefetch:
exact LDS bytes:          # gfx950: 160 KiB / 64 banks; gfx942 downgrade: 64 KiB / 32 banks
stage structure:
effective peaks from anchor trace:
budget/anchor.json: calibrated = true   (budget.md)
```

## 4. Layer State Table (over the backbone)

```text
layer: anchor | memory-path | lds-layout | pipeline | slicing | beyond-hot-loop | low-precision-side
state: open | active | closed | blocked | out_of_scope
best_candidate:
evidence (three): budget_ok | profile_delta | ir_signal
remaining_headroom (vs CALIBRATED budget):
next_action:
```

## 5. Per-Layer Round Ledger (core)

> Append this as a `climb_round` through `recordctl`; the block below is a human checklist for
> constructing its bounded fields and evidence references.

Fill once per round; the deep_engineer advances this without user input until termination. A
layer closes ONLY on three-evidence (`climb.md`). Re-profile every round (`profile.md`).

```text
round_id:
layer (backbone):
profile_source: gluon_anchor | last-kept
bound_class_picked:
pick_rationale (highest headroom vs CALIBRATED budget):
mechanism (ONE named):
card_cell_changed: Scope | Layout | Dispatch | Handoff   # which tile-op card axis this round mutates (tile-op-contract.md)
card_before -> card_after:                               # the single-cell diff (old value -> new value)
rewrite_delta:
compiler_contract_active: base | +scheduling | +agpr_hint | +authored_pass
compile_status:
correctness (vs oracle):
--- three-evidence (ALL required to close) ---
budget   : ideal vs as_built -> gap (consistent? y/n)
profile  : mfma_eff / ds_read_interval / achieved_bw / vgpr / spill / occupancy (delta +?)
ir_signal: expected IR/asm signal present? (per memory-path / pipeline / slicing / low-precision)
--- two baselines ---
vs_gluon_anchor (must NOT regress, UNLESS step_type == enabling_step_transient):
vs_plain_target (progress toward beating):
labels: parity_unreached? | injected?        # carried onto the number, never dropped
acceptance_source: verify_engineer / measure_legs (accepted) | ab_bench screen (screening only)
step_type: kept_win | enabling_step_transient | rejected   # transient = provisionally accepted toward a coupled combination
vs_checkpoint: delta vs the last net-win-confirmed state (the combine/delivery basis)
regression_budget_remaining: consecutive net-negative steps left before backtrack
layer_state_after: open | active | closed | blocked | out_of_scope
open_layers_remaining:
budget_rounds_remaining:
continue | stop (reason): all_closed | budget_exhausted | hard_constraint | broken_correctness
next_round_target:
```

A `kept_win` needs the GEAK commit gate (`MIN_IMPROVE` = 2 %) on the accepted numbers; a
measured noise band wider than 2 % raises the bar, a narrower one never lowers it.

## 6. Self-Monitoring / Insight Buffer

Counters tracked every round:

```text
best_vs_gluon_anchor:
best_vs_plain_target:
steps_since_improve:
current_bound_class:
layers_closed:
consecutive_regression:
checkpoint:   (last net-win-confirmed state; the combine/delivery basis)
```

Thresholds:

```text
stall      : 10 steps no improve -> consider closing/blocking the layer;
             15 -> try a different mechanism for this layer; 20 -> stop the layer, mark blocked
ceiling    : 3 profiles within 1% AND >=3 tuning steps -> stop tuning the layer
crash-loop : same compile/correctness error 3x -> re-read source, change edit strategy
budget     : 70% rounds -> warn ; 85% -> wrap up ; 95% -> emit timeout report
diversity  : within a layer, do not repeat the same parameter sweep; switch the
             mechanism (layout vs config vs schedule)
regression : accept a transient net-negative step on a DECLARED coupled (enabling)
             path; >=2-3 consecutive with no gain over the checkpoint -> backtrack to
             the checkpoint + switch direction; reset on any step that net-beats it
```

Rolling insight buffer (last ~15 observations), each tagged WIN | FAIL | OK | WARN with step,
mechanism, and metric. The same counters drive the triage buckets (`triage.md`) and the budget
burn-down in the delivered report (`close.md`).

## 7. Negative Result / Toolchain Ceiling / Timeout

Negative result (revert to plain):

```text
layer / bound_class / mechanism tested:
why blocked or no-extra-mechanism:
residual budget gap:
one named overhead (if slow-correct): conversion | padding | mask | launch |
                                      shared-staging | scheduling-mismatch | wrapper
search_scaffolding_removed: yes
decision: revert_to_plain (the gate over-estimated headroom)
```

Toolchain ceiling (scoped blocker):

```text
required feature: scheduler knob (sched_barrier) | scaled MFMA | async copy | ...
build identity (Triton tag / ROCm / install policy):
scoped blocker (target/dtype/version-scoped):
falsifying probe (command + exact error / hasattr==False on the target arch):   (triage.md)
fallback direction:
needs_compiler_change: yes | no          # yes = Scenario B, out-of-scope here
compiler_change_handoff (only if needs_compiler_change == yes):
  which_pass: interleave rule | RA/AGPR policy | post-asm peephole | other
  ir_profile_evidence: IR/asm shows the existing pass clumps / spills / leaves a gap
  proposed_change_one_line + expected_effect:
fallback_taken: non-scheduled latency hiding | switch layer | revert plain
```

**Mis-attribution guard:** set `needs_compiler_change: yes` only after the IR confirms the
*kernel* structure is correct — prefetched MFMA regions present, no unhoisted hot-loop
`convert_layout`, `R_total` fits. If the structure is wrong it is a kernel bug to fix here,
**not** a compiler ceiling. The build is read-only; never edit the compiler — the handoff note is
the deliverable (`../tile-programming/compiler-contract.md ## Compiler scope`).

Timeout (feeds `final_report` `outcome = timeout`):

```text
rounds_used / round_budget:
timeout_reason:
remaining_open_layers:
highest_expected_next_gain:
recommended_next_round:
```

## 8. Final Delivery Record (1:1 with final_report.json)

The delivered `final_report.json` follows `runtime/final-report-v3.schema.json`
(`schema: "kernel_opt.final_report/3"`); it is written with
`scripts/canonical_record.py write-final-report` and checked with `scripts/report_lint.py`, which
is stricter than the schema (`close.md`). The v3 required top-level fields are `schema`,
`run_id`, `generation`, `lifecycle_profile` (`front-end | deep-consumer | direct-owner |
blocked`), `stage` (`anchor | search | branch | converge | resweep | finalization`), `outcome`,
`arbitration`, `status`, `refs`, `audit`, `provenance` (`receipts[]`), `integrity`
(`source_schema`, `normalization_status`, `legacy_unverified`, `eligible_clean`), `caveats[]`,
`deferred[]`, `known_wrong[]`; `measurement`, `headline`, `served_range` are optional in the
schema and the lint requires `measurement`.

**Outcome enum** (`canonical_record.OUTCOMES`): `win | partial | negative_keep_baseline |
negative_revert_plain | timeout`. **Status enum** (`RUN_STATUSES`): `closed |
deferred_needs_user | blocked | partial_time_limit`. The legacy `result` field of earlier
revisions maps onto `outcome` (`win | partial | negative_revert_plain | timeout`;
`negative_keep_baseline` is the incumbent-kernel negative, where the baseline is itself Gluon).

The content the report must carry, as the round records feed it:

```text
skill: tile-programming-triton | tile-programming-gluon    # the pack that produced THIS record
geak: { round, direction_id, output_dir }           # where GEAK's record of this track lives
kernel_name / kernel_path / target_arch / workload_class:
environment: { rocm, triton_tag, llvm_or_commit, pytorch, docker_image, gpu, pin_consistent }   # the toolchain pin, stamped
correctness: { oracle, tol, coverage: { full_oracle_cells, proxy_cells, proxy_method } }   # expanded-cell coverage
escalation: { decision, gap, reason }
plain_baseline: { per_case[], geomean_latency_ms }        # the target line
gluon_anchor:   { per_case[], delta_vs_plain }
parity: { threshold, status: reached | parity_unreached }   # parity_unreached rides on every number below
layers[]: { name, state, mechanism, budget_ok, profile_delta, ir_signal }
best_gluon: { per_case[], verified_geomean_vs_plain, verified_arith_vs_plain }
                # the per_case numbers are verify_engineer / measure_legs outputs (by reference)
                # injected-labelled numbers never appear here as the win
per_shape map + per-shape regressions:
per_direction: { winning_tier, baseline_status, win_type, vs_baseline (null if unblock), rounds_spent }
ir_acceptance: each closed layer's IR/asm signal on file
                # ALSO required for a STRUCTURAL stay-plain win (num_stages / waves_per_eu / PRELOAD_V /
                # pipeline-flipping BLOCK_*): the dumped TTGIR/asm confirming the claimed structure
                # (entry.md, escalation gate "Stay plain"). Pure launch/dispatch/shape wins are exempt.
orchestration: { mode: single_deep_explore, gpu_id, rounds }   # one deep_engineer, one GPU (orchestration.md)
outcome: win | partial | negative_keep_baseline | negative_revert_plain | timeout
status: closed | deferred_needs_user | blocked | partial_time_limit
budget: { N, spent, remaining, allocation_rule, llvm_co_tuning_sanctioned }   # burn-down; timeout <=> remaining==0
caveats[]: # any acceptance artifact (budget JSON / IR dump) genuinely unavailable + the exact reason
close_audit: { verified_by: self | fleet, findings, high_severity_count, audit_ref }
                # REQUIRED. Who checked this closure and what the check found (scripts/close_audit.py).
                # A self-check and an independent check leave identical artifacts, so without
                # verified_by a reader cannot tell a cross-examined result from an unexamined one --
                # the same distinction stay_plain_basis (configured|measured) already draws.
skeptic_verdict: credible | weak | contradicted | not_run
skeptic_items[]: { claim, verdict, disposition: accepted | rebutted, evidence_ref }
                # REQUIRED when the closure self-review ran (close.md ## Closure challenge; the
                # schema field names are upstream's, the review is closure_review.md); not_run is a real
                # answer and differs from an absent field. Itemize the challenges: a verdict
                # recorded without dispositions reads exactly like agreement, so `contradicted` and
                # `credible` become indistinguishable in the one file a supervisor is allowed to
                # read. `evidence_ref` on a rebuttal is the on-disk reason the challenge does not apply.
deferred[]: { task_id, error_class, user_action_required, resume_entry }  # if partial
open_layers[]:
toolchain: { knobs }   # active compiler knobs; env pin is in `environment` above
minimal best_patch.diff (only if outcome == win):
kept / rejected changes:
```

**Record gate (a closed direction must be auditable).** A direction is marked `closed` ONLY if
its acceptance artifacts are on disk in `work_root`: the calibrated ideal-vs-as-built `budget/`
JSON; the closing round's `round_<n>/record.json` (6-group) + its raw artifacts kept on disk
(optional SOL rc_metrics + ATT mfma_efficiency rollup + static asm_loop_audit); and — for every
closed tile layer AND every structural stay-plain win (above) — the `ir/` dump that confirms the
accepted signal. If an artifact is genuinely unavailable (profiler-degraded mode, missing tool),
record the exact reason in `caveats[]`; never leave it silently empty. Cost is one budget JSON +
one round record (+ figures) + one IR dump per *closed* layer/win, not per probe.

**Stay-plain / ceiling closure additionally owes comparator evidence.** A direction closed as
`stay_plain` (or any "plain is the ceiling" verdict) is auditable ONLY if, on disk:
`gap_decomposition_artifact` (same-boundary A/B vs the contract's comparator + bound-class
attribution) AND `asm_audit_artifact` (`kernel_workflow/scripts/kernel_tools/asm_loop_audit.py`
of this kernel and the comparator). Without them the verdict is `unproven`, not `closed`
(`entry.md`, escalation gate "Stay plain"). If the contract names no comparator, record that
explicitly — the closure then rests on the budget gap alone and that limitation is named in
`caveats[]`. Also required for a load-imbalanced kernel: the P1 scheduling lever resolved
(`closed | out_of_scope` with reason).

Output: `optimized/<best gluon source> + best_patch.diff` (only on `win`), plus
`final_report.json + summary.md`. On `negative_revert_plain`, deliver the plain baseline and
record why Gluon did not pay off (which layers blocked, residual gap).

## 9. Experiment / hypothesis-test contract

A self-contained, **agent-agnostic** contract for one bounded micro-experiment (a benchmark sweep,
a knob/tile probe, an anomaly check). The deep_engineer fills it in and runs the experiment
itself, on its own GPU under `gpu_lock.sh`, inside its own loop — no other role or agent runs it
(`orchestration.md`); the block is written out in full so the record is self-contained for whoever
reads it later. Used by the Tier-0 diagnose flow (`entry.md`, Stage-Diagnose)
and by `## 9a. Anomaly validation` below.

```text
hypothesis:            # one falsifiable sentence (what changes what, expected direction)
env:                   # container/image + the GPU id handed to kernel_workflow/scripts/gpu_lock.sh
                       #   <gpu_id> <cmd> (never an inline HIP_VISIBLE_DEVICES) + busy GPUs to avoid
code_under_test:       # file + entry symbol + the overlay/PYTHONPATH that selects it
harness_reuse:         # which existing worker/timing fn to reuse VERBATIM (do not rewrite timing);
                       #   acceptance-grade timing = harness_lib.time_op / GEAK measure_legs
baseline_identity:     # baseline differs from candidate ONLY in the variable under test;
                       # runtime path-confirm method (assert no silent fallback to another path)
timing_method:         # production boundary (CUDA-graph kernel-only via time_op(graph=True), or
                       #   eager) + repeat/interleave structure; say acceptance vs screening
traffic_roofline:      # essential-traffic / AI / %peak formula GIVEN VERBATIM (no re-derivation),
                       #   with numerator_basis + denominator_basis (budget.md)
oracle:                # correctness reference + tol; every cell checked, out-of-tol flagged
comparison_protocol:   # same-session interleaved A/B pairs; MEDIAN statistic (min reported
                       #   alongside for search/screening only, ab_bench); read-evict flush;
                       #   noise-floor gate that may only tighten the 2% commit gate
                       #   (benchmark-hygiene.md ## Repeatability + measurement order)
sweep_grid:            # case x config x size/token x variable-under-test
deliverables:          # output files + the ## Experiment result schema below
failure_handling:      # on compile/OOM/GPU-busy/import error: record exact cause, run the subset, continue
```

### Experiment result

How any experiment reports back into the deep_engineer's records:

```text
hypothesis:
result: supported | not_supported | inconclusive
evidence_table:        # per cell: baseline / candidate / delta / achieved %peak (+ bases)
noise_band:            # measured repeat spread; deltas inside it are "within noise"
correctness:           # oracle pass/fail per cell
regressions[]:         # configs the change made worse (with magnitude)
recommendation:        # ship | drop | needs-more; + the EXACT gate if shipping
runtime:               # GPU time spent
caveats[]:             # cross-run/stored-baseline mixing, model assumptions, screening-only
                       #   statistics, injected labels, etc.
```

A `ship` recommendation from a side-experiment is a screening result: it becomes accepted only
through `verify_engineer` / `measure_legs` (the acceptance-grade `harness_lib` run in
`benchmark-hygiene.md`).

### Experiment robustness defaults (always on)

Apply to every experiment the deep_engineer runs, without being re-stated in each contract:

- measurement-validity: same-session interleave + median over repeats (min only as a screening
  supplement) + noise-floor gate that may only make the verdict stricter
  (`benchmark-hygiene.md ## Repeatability + measurement order`);
- baseline-identity: only the variable-under-test differs; no stored cross-run baseline
  (`benchmark-hygiene.md ## Same-knob baseline`);
- correctness oracle gate on every cell;
- profiler-degraded-mode classification when counters are unavailable (`entry.md`, Stage-Diagnose
  "Degraded-mode classification (no profiler)"; RDNA4 PMC-blind is a degrade, not a failure);
- GPU pinned through `kernel_workflow/scripts/gpu_lock.sh`;
- structured return via the `### Experiment result` schema above.

## 9a. Anomaly validation (before an anomaly consumes optimization budget)

Classification and bound-class routing run on measured/profiled aggregates. If an "interesting"
point is a **measurement or model artifact**, you can chase an optimization for a problem that
does not exist. So any **anomaly** must be validated before it earns optimization budget — run as
an experiment under the `## 9` contract (the deep_engineer's own "A1-A4 anomaly check").

An anomaly is either a **trend anomaly** — a non-monotonic dip/spike in the
size/token/parallelism sweep curve (e.g. a speedup trough at one token count) — or an
**absolute outlier** — a point that deviates sharply from its like-class neighbors (e.g. 1.0x
where neighbors are 1.3x). Gate it:

- **A1 — re-measure same-session, interleaved.** Re-run the anomalous cell with baseline and
  candidate interleaved back-to-back, N repeats, reading the median (the min may be quoted as the
  floor for screening) (`benchmark-hygiene.md ## Repeatability + measurement order`). Do **not**
  compare against a stored cross-run number (`benchmark-hygiene.md ## Same-knob baseline`).
- **A2 — reproduce above the noise band.** The deviation must exceed the measured noise band AND
  reproduce with the same sign across **2 clean sessions / seeds**. A deviation inside the noise
  band is `measurement-explained` — drop it. **The axis is sessions, not buckets**: same-sign
  reproduction across sessions is what tests drift, whereas sign agreement across buckets/shapes
  *inside one run* is not, because drift is common-mode over them
  (`benchmark-hygiene.md ## A pre-registered criterion fixes the threshold, not the statistic's power`)
  — and a cross-session claim is quoted against the cross-session spread, not the in-window
  repeat (`benchmark-hygiene.md ## Repeatability + measurement order`, noise-floor gate).
- **A3 — attribute.** Classify the surviving anomaly as **kernel** (real: occupancy cliff,
  tile/padding granularity, bank conflict, scheduling), **harness** (cross-run drift,
  cache/autotune residue, cold compile, clock), or **model** (traffic/AI/roofline formula wrong →
  the anomaly is in the *analysis*, not the run).
- **A4 — gate.** Only a **kernel-attributed, reproduced** anomaly earns optimization budget. A
  `harness`-attributed one → fix the measurement and re-classify; a `model`-attributed one → fix
  the formula; record either as `measurement-explained` / `model-explained` and move on.

Validation is minutes; optimization is GPU-hours plus integration/validation, so validating first
is near-always positive-EV. (In practice a decode speedup "trough" survived A1 only partially — it
was largely cross-run host-timing drift, and the kernel-attributed residual was inside the noise
band, so it never warranted the work.)

## Supporting records (slim)

### Prior-Experiment / Previous-Version Control

```text
prior_winner_config_or_path:
current_harness_result / current_winner_result:
difference_class: gpu_or_toolchain | harness_bug | shape_stream | search_gap | boundary_change
scope_decision: final_replacement | scoped
```

Do not close the config layer until the current winner beats or explains known prior winners on
the same hardware and benchmark boundary.

### Non-Deterministic Correctness (attention / reduction)

```text
determinism_class:
baseline_self_variance:
reference_oracle (deterministic / high-precision):
candidate_vs_reference / candidate_vs_baseline:
accepted_tolerance:
```

### MFMA Guard (gfx950 v4 / gfx942 v3)

```text
target family + AMDMFMALayout version (4 = gfx950 / 3 = gfx942 downgrade):
instr_shape:
operand dtype / accumulator dtype:
result layout / operand layouts / k_width:
store layout:
scale format/layout (if scaled MFMA; gfx950 only):
correctness oracle:
```

Full per-target support / evidence -> `../hardware/capability-matrix.md`.

### Layout Map (multi-layout Gluon patch)

```text
expression | parent layout | slice axis | expand direction | expected rank | owner tensor | consumer
```

### Split / Partition (Stage-Plain)

```text
split_or_partition_knob:
program_count_before_after / inner_loop_work_before_after:
main_kernel_latency / reduce_or_combine_latency / temporary_buffer_cost:
correctness oracle + tolerance / precision-sensitive buckets:
winner_buckets / decision:
```

## failure_class enum (trimmed)

```text
direction_unproven
environment_toolchain_blocker
backend_or_lowering_failure
api_or_layout_failure
no_extra_gluon_mechanism
correct_but_slower
shape_split_winner
performance_noise
bandwidth_or_launch_ceiling
wrapper_overhead_dominates
timeout_budget_exhausted
negative_result_stop
```

The triage buckets map onto it (`triage.md ## Retryable vs scoped-ceiling vs global (continue / defer / halt)`).

## Sources

Merged: `references/experiment-records.md` (all sections), `references/anomaly-validation.md`
(whole file → `## 9a. Anomaly validation`).

Rewritten by the contract rules (nothing dropped as content):
- Task contract `timing_method`: `cudagraph_amortized_replay | do_bench_cudagraph |
  batched_wallclock` kept but labelled **search/screening only (ab_bench)**; acceptance timing is
  `harness_lib.time_op` / GEAK verify + `measure_legs` (GEAK owns timing).
- Experiment contract / robustness defaults / A1: "min over repeats" → median (min kept as a
  screening supplement); noise-floor gate restated as "may only tighten the GEAK
  `MIN_IMPROVE` 2 % gate".
- Experiment `env`: "GPU id (HIP_VISIBLE_DEVICES)" → GPU id passed to
  `kernel_workflow/scripts/gpu_lock.sh` (no inline `HIP_VISIBLE_DEVICES`).
- Anchor record: `num_stages -> starting pipeline depth` → `num_stages` is champion record +
  budget parameter only (dead on the Gluon path in 3.8.0); tile-op card "Handoff = none=auto
  where the compiler still pipelines" → no Gluon pass pipelines, `auto` only for plain or an
  `injected` build. Layout recovery/verification attributed to `ttgir_bridge` (recover/verify),
  `recover_gluon --record` = anchor assembly.
- Added fields required by the contract: run mode, parity threshold/status (`parity_unreached`
  carried), the three suspects (`lost_pipeline / lost_layout / lost_RA`), `injected` label,
  acceptance source, falsifying-probe stamp, v3 `outcome`/`status` enums (legacy `result` →
  `outcome`; adds `negative_keep_baseline`).
- Section renamed: second `## 1c. Residual / Deferred-Task & Resume ledger` → `## 1d. …`;
  `## 9. … (agent-agnostic)` → `## 9. Experiment / hypothesis-test contract` (pinned);
  `### Experiment result (how any experiment reports back)` → `### Experiment result` (pinned);
  the old "`## 3a. Anomaly validation`" self-reference now resolves to `## 9a. Anomaly validation`.
- Run-mode split (GEAK-embedded vs the pack's `toolctl` spine) removed: `run mode` / `run_mode`
  fields → `geak: {round, direction_id, …}`; `orchestration.mode: parallel_subagents |
  sequential_fallback` (+ `max_parallel_subagents`) → `single_deep_explore`; experiments delegated
  to the upstream `gluon-bench` agent → run by the deep_engineer itself; the closure-skeptic agent
  reference → `close.md ## Closure challenge`; the `orchestration.md ## Runtime contract` pin
  reference → `entry.md`.
- Pointers translated to the new method files (`phases/*`, `escalation-gate.md`, `diagnose.md`,
  `failure-triage.md`, `orchestration.md`).
