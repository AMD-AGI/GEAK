# gluon_authoring: the Gluon method as an expert workflow

Takes a kernel to **Gluon** (`triton.experimental.gluon`) on AMD CDNA — gfx950 and gfx942 — by
running the [`gluon_authoring` skill pack](../../skills/gluon_authoring/skill.md) as a stage machine:
entry → budget → transcribe → recover → climb → close. Recover and climb use the kernel workflow's
search (`kernel_workflow/kernel_lane.js`): `tech_lead` plans 2–3 directions per round, engineers carry
them in parallel, an integrate step stacks the verified ones, and the budget counts directions. The
Gluon method and its evidence gates live inside each direction; see `## Relation to kernel-workflow`. The method is multi-stage, gated and
tool-driven, which a prompt appendix cannot carry (measured: appended to GEAK's role prompts, the
stage references were never opened and none of the gate tools ran). Here the stages, the gates and
the evidence each round owes are **code**; the skill pack stays the single source of the method, and
this directory adds no copy of it.

This directory is self-contained: `workflow.js`, its roles and tests. Nothing in `kernel_workflow/` or
`e2e_workflow/` refers to it. It reads from `kernel_workflow/` the GEAK roles that own correctness,
acceptance and the search (`director`, `benchmark_engineer`, `verify_engineer`, `tech_lead` for
plan_round and update_memory, `integrator.md` as the integrate strategy), the GPU lock
(`scripts/gpu_lock.sh`), `scripts/materialize_workspace.sh` and the shared tools in
`scripts/kernel_tools/`; and from the skill pack its references and scripts.

Profiling is the pack's own: the baseline and every round profile through
`skills/gluon_authoring/scripts/gluon_profile.py`, not `kernel_workflow/scripts/profile_kernel.sh`
(which is unchanged and still serves the kernel workflow). One call collects every dial of
`profile.md ## 3.1` — kernel trace (D), PMC groups bisected on abort/timeout (C), ATT with the
instruction readers (B), IR with both occupancy limiters (A) — puts a kernel filter and a timeout on
every profiler pass, runs the workload from a generated `cmd.sh` (no profiler re-splits a quoted
command), and records each layer's state in `profile.json`; `--compare <prev>` writes the profile
delta a climb round lands on.

## Launch

```
Workflow({ scriptPath: "<GEAK>/perf_knowledge/expert_skills/expert_workflows/gluon_authoring/workflow.js", args: {
  kernel_path: "<task dir>",                 // required: the GEAK kernel task (config.yaml + scripts/ + source/)
  workflow_dir: "<GEAK>/kernel_workflow",    // required: shared roles, gpu_lock.sh, kernel_tools
  exp_root: "<dir>",                         // default <GEAK>/exp; the run's EVAL_DIR is created under it
  gpu_ids: "6,7",                            // GPU pool; every GPU command takes one GPU via gpu_lock.sh
  budget: 6,                                 // directions, as in kernel-workflow (transcribe/author 1, each recover/climb direction 1)
  deadline_epoch: 1794830000,                // optional unix seconds; no round starts inside the close reserve
  target_arch: "gfx942",                     // optional gfx942|gfx950; checked against the detected GPU
  entry_mode: "auto",                        // auto|port|incumbent|fresh (auto: gluon_lead reads the bodies)
  task: "...",                               // optional free text for the director/lead
}})
```

Other args, with defaults: `min_improve` 0.02 (commit gate), `deep_cost` 2 (a deep_explore direction,
alone in its round), `max_directions` 0 (no per-round cap beyond the budget; 1 gives the old
one-engineer track), `no_stop_s` 900 and `max_forced_replans` 6 (with a deadline, a tech_lead stop is
refused while time is left), `parity_threshold` 0.95, `max_recover_rounds` ceil(budget/2),
`max_no_improve` (port 6, else 2; climb rounds without progress), `progress_delta` (port −0.05, else
`min_improve`; how far over the best candidate seen a round's winner must land to count as progress),
`enabling_delta` (port −0.05, else 0; how far an `enabling_step` may sit below the track best and still
be kept), `prior_log` (an earlier run's insight log or final report when this run continues from its kept line),
`warm_start` on, `update_experience` on, `box_quiet` true, `held_out` false, `kb_mode` store, `kb_remote` auto,
`warm_start_min_speedup` 1.05, `perf_knowledge_dir`, `kb_artifacts_dir`, `kb_store_dir`, `storage_reclaim` on
(kernel-workflow's knowledge loop, same defaults; an A/B run sets `warm_start=off` and `held_out=true` so neither arm
reads or teaches the KB), `use_learned_kb` true, `kb_dir_cap` 1 and `kb_cold_direction` true (kernel-workflow's learned
cards for the planner, with its budget),
`deadline_reserve_s` 1800, `round_time_cap_s` 3600, `evidence_reasks` 1, `evidence_audit` true,
`token_reserve` 150000, `agent_timeout_ms` 3600000, `engineer_timeout_ms` 7200000, `eval_dir`,
`apply_to_original` false, `skill_dir` / `expert_workflow_dir` (default: the in-tree pack and this
directory, derived from `workflow_dir`). Single-GPU kernels only (`gpus_per_job` must be 1).

## Stage machine

| # | stage (phase) | role | what the code decides from its return |
| --- | --- | --- | --- |
| 1 | Setup | `director` (kernel_workflow, PHASE=setup, MODE=optimize) | EVAL_DIR, canonical workspace, GPU identity; refuses anything but gfx942/gfx950, a `target_arch` mismatch, or no frozen baseline |
| 2 | Benchmark | `benchmark_engineer` (kernel_workflow) | `COMMANDMENT.md` and `BASELINE_PER_CASE` — the frozen baseline every speedup is measured against |
| 3 | Profile | `gluon_profiler` PHASE=baseline (`gluon_profile.py`) + `profile.md` | the baseline dials (`EVAL_DIR/gluon/profile_baseline`), handed to entry and round 1 as `LAST_PROFILE`; every layer's state stated (`trace`, `pmc`, `att`, `ir`), a non-`ok` one with `degraded_reason`; otherwise re-asked, then a `baseline_profile_incomplete` caveat |
| 4 | Entry | `gluon_lead` PHASE=entry | entry mode (`port` / `incumbent` / `fresh`); champion bundle; **G1** `champion_gate.py` — exit ≠ 0 blocks the run before any edit; the track workspace `EVAL_DIR/gluon/track` tagged `geak_base`; `champion_patch` when the champion is not the frozen baseline |
| 4b | Entry | `verify_engineer` on `champion_patch` | the champion's speedup vs the frozen baseline (`champion_vs_baseline`), so every later number also carries `vs_champion` |
| 1b | Analyze | `tech_lead` PHASE=analyze (kernel_workflow) | roadmap, `codebase_context.md`, `KK_OPERATOR` / `KK_REFS` for the planner and engineers |
| 5 | Budget | `gluon_lead` PHASE=budget | `hw_budget.py` ran (exit 0, json written); otherwise re-asked, then a `budget_missing` caveat |
| 6 | Transcribe (port) / Author (fresh) | `gluon_engineer`, one direction | the four equivalence checks (layout diff PASS/RECONCILED, numeric, determinism, asm parity) and G3; a failed equivalence or an incorrect anchor stops the run — no recovery on a wrong anchor |
| 7 | Recover (port) | `tech_lead` plan_round (one direction per debt suspect), `gluon_engineer` per direction in parallel, integrate | **G2** on the round's kept candidate: parity clears only when its `parity_gate.py` exits 0 **and** verify measures the track at ≥ `parity_threshold` of the champion; until then the loop does not climb; after `max_recover_rounds` rounds the run closes `parity_unreached` |
| 8 | Climb | `tech_lead` plan_round (~2–3 directions, each a Gluon layer), `gluon_engineer` per direction in parallel, integrate | G3/G4 (+ATT) evidence per direction; verify's number; track keep, commit gate, stall count |
| 8b | Integrate | `gluon_engineer` PHASE=integrate (integrator.md's strategy on the track tags), `verify_engineer`; `gluon_profiler` PHASE=reprofile when the integrated candidate is kept | when ≥ 2 directions verified above the kept line (or enabling); kept only if verified above the best individual |
| 8c | Memory | `tech_lead` PHASE=update_memory | insights and the hypothesis ledger, read by the next plan and every engineer |
| 5b | WarmStart | resolver, `verify_engineer`, adopt, `kb:attest` | experience KB (language gluon): an incumbent adopts the first stored patch that passes verify; port/fresh read it as reference |
| 9 | Close | `gluon_lead` PHASE=close | the report, `final_report.json`, `closure_review.md` for any non-win (re-asked if missing); the outcome and the delivered patch are decided by the code and passed in |
| 10 | Validate | `director` (kernel_workflow, PHASE=validate) | only when a patch is delivered; its verdict is the headline |
| 11 | Write-back | `update_experience`, `kb:cite`, `kb:write` (kernel_workflow) | a learned card from a director-accepted win; the citation ledger for every non-held-out run; the store entry for a verified win |

Entry modes follow `references/method/entry.md ## 4`: **port** runs 6→7→8, **incumbent** (already
explicit Gluon, nothing converted) goes straight to 8 with no parity gate, **fresh** (no Triton
champion to transcribe) authors the first Gluon kernel at 6 and climbs. A task that ships a Gluon
transcription as `source/` with the plain kernel as its golden copy is a **port** with
`source_is_anchor: true`: the frozen baseline is the anchor, the champion is measured separately (4b)
and the transcribe round verifies the provided anchor instead of re-transcribing it.

G0 (the loop shape) is enforced by construction: the stop rules (`max_no_improve`,
`progress_delta`) follow the entry mode, and a port's below-comparator anchor is kept on the track
without ever being accepted.

Stop conditions, all in code: direction budget; `tech_lead` stop (refused while more than `no_stop_s`
are left before a deadline); `deadline_epoch` (a low-effort agent reads `date +%s`
before each round — a workflow has no clock — and no round starts with less than
`deadline_reserve_s` left; each engineer gets `ROUND_DEADLINE_EPOCH`; an unreadable clock never
truncates a run); token budget (`budget.remaining() < token_reserve`); `max_no_improve` climb rounds
without progress (kernel_lane.js's rule: neither accepted nor `progress_delta` over the best candidate
seen); G1 failure; transcribe/author failure; `parity_unreached`.

## Rounds, directions and the gating order

A recover or climb round is: `tech_lead` PHASE=plan_round (GEAK's role, with an appended Gluon
addendum: each direction names a `layer`, a recover direction a `suspect`, and `paired: true` marks a
prerequisite + the lever it makes legal) → the script's filters → one `gluon_engineer` per direction in
parallel, each with its evidence gates, audit and `verify_engineer` → integrate → keep / accept →
`tech_lead` PHASE=update_memory. The filters, in kernel_lane.js's order: cut to the remaining budget
(and `max_directions`); a `deep_explore` direction runs alone and costs `deep_cost`; then the gating
order of `climb.md ## Layer Backbone` inside the round — 1.5 does not share a round with 4/4+/5 or a
second 1.5 model, 3 not with 4, 4 not with 4+ — the later direction of such a pair is deferred, not
dispatched and not charged. A coupled chain whose halves do not pay alone is one paired direction; it
owes `pair_screen` (the prerequisite alone and the pair, each a permuted screen vs `TRACK_START`).
Every direction returns a `lesson`; `HISTORY` (all directions of the last rounds, null and rejected
ones included) and tech_lead's `INSIGHTS` go to the next plan and to every engineer.

## The track, the commit gate and acceptance

`EVAL_DIR/gluon/track` is the run's git store of kept lines. Each direction materializes its own copy
(`round_N/d<i>/ws`, with the track's `.git`), resets it to the last tag the workflow kept
(`TRACK_START`), commits its candidate, tags it `r<N>_d<i>` (an integration `r<N>_int`, a transcription
`r<N>`), fetches the tag back into the track and writes `best_patch.diff` = `git diff geak_base <tag>` —
always relative to the frozen baseline, so `verify_engineer` applies it to a clean copy of the canonical
workspace, which never advances. Parallel directions therefore never share a working tree.

- **Measurement** is `verify_engineer` (kernel_workflow, unchanged) on every candidate, against
  `BASELINE_PER_CASE`, with the frozen COMMANDMENT. The engineer's own timings (`ab_bench.py`) are
  screening evidence only.
- **Track keep**: verified correct and faster than the track best (or an `enabling_step` within
  `progress_delta` of it). A port's anchor below the champion is kept here.
- **Commit gate (accepted)**: verified correct and > accepted best × (1 + `min_improve`), starting from
  the frozen baseline (1.0×). Verify's weighted metric is primary when the oracle provides it.
- **Outcome** (`close.md` enum): `win` when there is an accepted candidate — for a port, only when it
  also beats the champion by `min_improve`; otherwise `timeout` (deadline), `negative_keep_baseline`
  (incumbent / fresh) or `negative_revert_plain` (port). A port that does not beat the champion
  delivers the champion itself when the champion is faster than the frozen baseline (the anchor), and
  nothing otherwise.
- **Validation**: `director` PHASE=validate on the delivered patch, against the true baseline.

GPU access is only through `kernel_workflow/scripts/gpu_lock.sh` (pool `gpu_ids`): parallel
directions queue on the pool, so give it as many GPUs as directions per round. The skill stays advisory about tactics; its gates and
records are this workflow's contract.

## Evidence contract (what a round must return)

Every method-stage prompt opens with **Required reading**: the absolute paths and section headings
of the skill references for that stage (`STAGE_REFS` in `workflow.js`), plus, for recover and climb,
the references of each layer (`LAYER_REFS`) — the ones for the layer the round works on are required.
The return is `ROUND_SCHEMA`; `missingEvidence()` checks it per stage:

| field | required for | check |
| --- | --- | --- |
| `references_read` | every stage | covers every required path (layer references for the round's layer) |
| `track_tag`, `patch_file` | every candidate | present (an anchor the workspace already ships: `anchor_provided`, tag `geak_base`; verify still measures it as it is, because the frozen baseline may be the plain champion, and it sets the track line, never a deliverable) |
| `equivalence` | transcribe | `layout_diff` ∈ PASS/RECONCILED/FAIL from `ttgir_bridge.py verify`; `numeric`, `determinism`, `asm_parity` recorded |
| `g3_occupancy` (G3) | transcribe, author, recover repayment, climb | `probe.py measure` command, exit 0, `ir_dir`, both limiters (`vgpr_wg_per_cu`, `lds_wg_per_cu`; `null` only with `missing_reason`), `binding` |
| `g4_screen` (G4) | recover repayment, climb | `ab_bench.py ... --permute`, exit code, `json_path`, `resolved` |
| `att` | layers 1.5, 4, 4+ and a `lost_pipeline` repayment | the round's `gluon_profile.py` (att layer on), `capture.sh` without `--no-att/--lite/--isa-only`, or `rocprofv3_safe.sh --att`, with `output_dir` and `state: ok` (another state needs a reason); an `output_dir` inside the round's `profile_dir` stands for itself when `profile_command` collected ATT; or `status: unavailable` with a reason (≥ 20 chars) |
| `edge_shapes` | transcribe (baseline: the champion), author; recover/climb when masks, indexing, a tile or loop bounds change | `edge_shapes.py` command and exit code: 1 (fails a boundary shape the baseline passes) is a gap — for a transcription, not equivalent; 3 (the baseline fails it too) becomes a `boundary_limitation` caveat |
| `pair_screen` | a paired direction | both permuted screens (prerequisite alone, pair) vs `TRACK_START` and `outcome` (pair / prerequisite / neither: the half the candidate is) |
| `parity_gate` (G2) | recover | `parity_gate.py` command, exit code, `mode: gate`; on an incumbent only `mode: diagnostic` is allowed |
| `layer`, `profile_dir`, `profile_command` | climb | layer ∈ 1.5/2/3/4/4+/5/6/7; this round's re-profile by `gluon_profile.py` (`--compare LAST_PROFILE` gives the profile delta) |
| `suspect` | recover repayment | `lost_pipeline` / `lost_layout` / `lost_RA` (repaid at layer 4 / 3 / 5) |
| `disassembly_ref` | inline-asm rounds | present |

A gap is sent back to the engineer by name (`evidence_reasks`, default once: "complete exactly the
MISSING items for the candidate already at TRACK"). If it is still missing, the round is **rejected**:
no verify, the track stays at the last kept tag, the round still costs budget. A complete return then
goes to the `evidence_auditor` (low effort, no GPU), which checks the cited artifacts exist, the tag
and patch agree, re-runs `probe.py measure` and `parity_gate.py` (CPU-only) and reads the
`ab_bench.py` json for its permuted window; a finding is handled like a missing field. Entry (G1
command, `champion_patch` when needed, `track_ready`), budget (`hw_budget.py`) and close
(`closure_review.md` for a non-win) are checked the same way.

The gate numbers are the method's (`references/method/index.md ### The four gates`): G0 loop shape,
G1 champion, G2 parity, G3 occupancy (both limiters), G4 attribution (permuted screen; acceptance is
GEAK verify + the 2% commit gate). Equivalence has no number; it is the transcription round's gate.

## Result

`{ workflow, entry_mode, outcome, blocked, stopped_by, deadline_hit, parity, caveats, requests,
rounds[], rounds_used, budget, budget_used, insights, accepted, champion_vs_baseline, final_patch,
reverted_to_champion, report_path, closure_review_path, validation_status,
director_verified_speedup_geomean }`. Each `rounds[]` row is one direction or integration: round, id,
stage, specialty, layer, claim, lesson, verify status, `vs_baseline`, `vs_champion`, kept, accepted,
the integration's `combos` (orthogonal / rival / conflict), and the missing items or audit findings of
a rejected one. On disk: `EVAL_DIR/gluon/{entry,budget,round_N/{d<i>,integrate},track}`,
`report.md`, `final_report.json`, `closure_review.md`, and `EVAL_DIR/final_patch.diff`.

## What changed in GEAK for this

The earlier integration appended a text block to role prompts (`use_gluon_skill`, `gluonSkillBlock`
in `kernel_lane.js`, forwarded by `kernel_workflow.js`, `e2e_workflow.js` and `interface/run_e2e.py`,
plus a paragraph in `author_engineer.md`). All of it is removed: default kernel and e2e workflows are
byte-identical to main. Kept, because they are not injection: the skill pack itself; the shared
`kernel_tools/`; and the `.tile-runtime/` exclusions in `materialize_workspace.sh`, the director's
workspace `.gitignore`, tech_lead's state sync comment and `docs/eval_archive_contract.md` — that
directory is written by the shared `kernel_tools/dump_ir.sh` and the pack's runtime scripts into
whichever workspace runs them, so any workflow using those tools needs it kept out of patches and
copies.

## Layout and tests

```
workflow.js   the stage machine (Setup -> ... -> Validate)
roles/        gluon_lead (entry, budget, close), gluon_profiler (baseline profile, reprofile), gluon_engineer
              (transcribe/author/recover/climb/integrate), evidence_auditor (artifact audit);
              GEAK roles are read from kernel_workflow/roles/
tests/        harness.js (runs workflow.js with synthetic agent responses) and test_*.js:
              stage order per entry mode, required reading, evidence re-ask/reject (G3, G4, ATT state,
              gluon_profile re-profile, baseline profile layers, audit, pair screens),
              acceptance by verify and the commit gate, parity/G1/deadline/token/budget stops,
              the parallel search (plan, directions, integrate, budget, deep_explore, gating, memory,
              no-early-stop), the knowledge loop (analyze, KK cards, warm start, write-back, reclaim),
              alignment with kernel_lane.js (defaults, schemas, commands, rules read from it),
              launch contract (args, arch, reference paths, size, no injection left in kernel/e2e)
```

Run: `for t in tests/test_*.js; do node "$t"; done`. CPU-only; no agent, GPU or file is touched.

## Relation to kernel-workflow

Relative to `kernel_workflow/kernel_lane.js` this workflow is the same search with Gluon-specific
parts swapped in; `tests/test_main_alignment.js` reads kernel_lane.js and fails when its defaults,
planner schema or rules move.

| | kernel_lane.js | here |
| --- | --- | --- |
| budget, pacing | directions; tech_lead ~2–3 per round; deep_explore alone, `deep_cost` 2 | the same (transcribe/author is one direction) |
| planner, memory | `tech_lead` plan_round / update_memory | the same role and phases, plus a Gluon addendum (layers, suspects, pairs, gating) |
| engineer | `engineer` / `deep_engineer` per specialty | `gluon_engineer` per direction, with G3/G4/ATT gates, re-ask and the artifact audit |
| integrate | `integrator` when ≥ 2 verified; its own number | integrator.md's strategy on the track tags with permuted screens; the integrated candidate is verified |
| profile | `profile_kernel.sh` (`profile_engineer`) | `gluon_profile.py`: every direction's round profile, `--compare` the last kept one |
| stall, commit, stop | `max_no_improve` 2, `progress_delta` = `min_improve`, `min_improve` 0.02; no early stop with a deadline | the same; a port gives ground (6, −0.05) |
| added | — | entry modes, G1 champion gate, transcribe with equivalence, recover with G2, layer gating inside a round, paired directions |
| learned KB | planner reads `knowledge/learned/` with a per-round budget | the same switch, cap and control direction |
| analyze | `tech_lead` PHASE=analyze before Benchmark: roadmap, `codebase_context.md`, perf_knowledge cards (`KK_*`) | the same role and phase; `KK_*`, `CODEBASE_CONTEXT` and `SELF_MONITORING` go to the planner and every engineer |
| warm start | experience KB read by kernel id, verify, adopt, attest | the same commands keyed by language gluon; an incumbent adopts (the tag `warm_start` is the kept line), a port or fresh run reads it as reference |
| write-back | `update_experience` card, `kb.py cite` ledger, `experience_store.py write-remote` | the same roles, commands and gates, language gluon |
| storage reclaim | `reclaim_eval_artifacts.sh` after every round | the same script on `EVAL_DIR/gluon` |
| not wired | deep research, EVO, routing (all off by default in kernel-workflow) | — |

## Open points

- `validation.status` of the skill stays `draft`; a run of this workflow is the natural validation
  harness (`expects.isolated_speedup_min` over the whole track), but `--record` is not wired here.
- A hung agent is resolved to null by the guard but not killed; directions work in their own copies,
  so a late writer can only re-tag its own tag in the track.
- The pack's record tools (`round_record.py`, `recordctl.py`, `close_audit.py`, `toolctl.py`) remain
  optional for the engineer; the workflow's own ledger is the `rounds[]` it returns and the per-round
  `round.md`.
