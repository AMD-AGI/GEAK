# gluon_authoring: the Gluon method as an expert workflow

Takes a kernel to **Gluon** (`triton.experimental.gluon`) on AMD CDNA — gfx950 and gfx942 — by
running the [`gluon_authoring` skill pack](../../skills/gluon_authoring/skill.md) as a stage machine:
entry → budget → transcribe → recover → climb → close. The method is multi-stage, gated and
tool-driven, which a prompt appendix cannot carry (measured: appended to GEAK's role prompts, the
stage references were never opened and none of the gate tools ran). Here the stages, the gates and
the evidence each round owes are **code**; the skill pack stays the single source of the method, and
this directory adds no copy of it.

This directory is self-contained: `workflow.js`, its roles and tests. Nothing in `kernel_workflow/` or
`e2e_workflow/` refers to it. It reads from `kernel_workflow/` the GEAK roles that own correctness and
acceptance (`director`, `benchmark_engineer`, `verify_engineer`), the GPU lock
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
  budget: 6,                                 // engineer rounds (transcribe/author + recover + climb)
  deadline_epoch: 1794830000,                // optional unix seconds; no round starts inside the close reserve
  target_arch: "gfx942",                     // optional gfx942|gfx950; checked against the detected GPU
  entry_mode: "auto",                        // auto|port|incumbent|fresh (auto: gluon_lead reads the bodies)
  task: "...",                               // optional free text for the director/lead
}})
```

Other args, with defaults: `min_improve` 0.02 (commit gate), `parity_threshold` 0.95,
`max_recover_rounds` ceil(budget/2), `max_no_improve` (port 6, else 2), `progress_delta` (port −0.05,
else 0; how far an `enabling_step` may sit below the track best and still be kept),
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
| 5 | Budget | `gluon_lead` PHASE=budget | `hw_budget.py` ran (exit 0, json written); otherwise re-asked, then a `budget_missing` caveat |
| 6 | Transcribe (port) / Author (fresh) | `gluon_engineer` | the four equivalence checks (layout diff PASS/RECONCILED, numeric, determinism, asm parity) and G3; a failed equivalence or an incorrect anchor stops the run — no recovery on a wrong anchor |
| 7 | Recover (port) | `gluon_engineer`, one suspect per round | **G2**: parity clears only when `parity_gate.py` exits 0 **and** verify measures the track at ≥ `parity_threshold` of the champion; until then the loop does not climb; after `max_recover_rounds` the run closes `parity_unreached` |
| 8 | Climb | `gluon_engineer`, one layer per round | G3/G4 (+ATT) evidence; verify's number; track keep, commit gate, stall count |
| 9 | Close | `gluon_lead` PHASE=close | the report, `final_report.json`, `closure_review.md` for any non-win (re-asked if missing); the outcome and the delivered patch are decided by the code and passed in |
| 10 | Validate | `director` (kernel_workflow, PHASE=validate) | only when a patch is delivered; its verdict is the headline |

Entry modes follow `references/method/entry.md ## 4`: **port** runs 6→7→8, **incumbent** (already
explicit Gluon, nothing converted) goes straight to 8 with no parity gate, **fresh** (no Triton
champion to transcribe) authors the first Gluon kernel at 6 and climbs. A task that ships a Gluon
transcription as `source/` with the plain kernel as its golden copy is a **port** with
`source_is_anchor: true`: the frozen baseline is the anchor, the champion is measured separately (4b)
and the transcribe round verifies the provided anchor instead of re-transcribing it.

G0 (the loop shape) is enforced by construction: the stop rules (`max_no_improve`,
`progress_delta`) follow the entry mode, and a port's below-comparator anchor is kept on the track
without ever being accepted.

Stop conditions, all in code: round budget; `deadline_epoch` (a low-effort agent reads `date +%s`
before each round — a workflow has no clock — and no round starts with less than
`deadline_reserve_s` left; each engineer gets `ROUND_DEADLINE_EPOCH`; an unreadable clock never
truncates a run); token budget (`budget.remaining() < token_reserve`); `max_no_improve` climb rounds
without a kept improvement; G1 failure; transcribe/author failure; `parity_unreached`.

## The track, the commit gate and acceptance

The engineer works in one persistent git workspace, `EVAL_DIR/gluon/track`. Each round starts with a
reset to the last tag the workflow kept (`TRACK_START`), commits its candidate, tags it `r<N>` and
writes `round_N/best_patch.diff` = `git diff geak_base r<N>` — always relative to the frozen baseline,
so `verify_engineer` applies it to a clean copy of the canonical workspace, which never advances.

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

GPU access is only through `kernel_workflow/scripts/gpu_lock.sh` (pool `gpu_ids`); the stages run one
at a time, so the track never has two writers. The skill stays advisory about tactics; its gates and
records are this workflow's contract.

## Evidence contract (what a round must return)

Every method-stage prompt opens with **Required reading**: the absolute paths and section headings
of the skill references for that stage (`STAGE_REFS` in `workflow.js`), plus, for recover and climb,
the references of each layer (`LAYER_REFS`) — the ones for the layer the round works on are required.
The return is `ROUND_SCHEMA`; `missingEvidence()` checks it per stage:

| field | required for | check |
| --- | --- | --- |
| `references_read` | every stage | covers every required path (layer references for the round's layer) |
| `track_tag`, `patch_file` | every candidate | present (an anchor that is already the baseline: `anchor_provided`, tag `geak_base`) |
| `equivalence` | transcribe | `layout_diff` ∈ PASS/RECONCILED/FAIL from `ttgir_bridge.py verify`; `numeric`, `determinism`, `asm_parity` recorded |
| `g3_occupancy` (G3) | transcribe, author, recover repayment, climb | `probe.py measure` command, exit 0, `ir_dir`, both limiters (`vgpr_wg_per_cu`, `lds_wg_per_cu`; `null` only with `missing_reason`), `binding` |
| `g4_screen` (G4) | recover repayment, climb | `ab_bench.py ... --permute`, exit code, `json_path`, `resolved` |
| `att` | layers 1.5, 4, 4+ and a `lost_pipeline` repayment | the round's `gluon_profile.py` (att layer on), `capture.sh` without `--no-att/--lite/--isa-only`, or `rocprofv3_safe.sh --att`, with `output_dir` and `state: ok` (another state needs a reason); or `status: unavailable` with a reason (≥ 20 chars) |
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
rounds[], accepted, champion_vs_baseline, final_patch, reverted_to_champion, report_path,
closure_review_path, validation_status, director_verified_speedup_geomean }`. Each `rounds[]` row
carries stage, layer, verify status, `vs_baseline`, `vs_champion`, kept, accepted, and the missing
items or audit findings of a rejected round. On disk: `EVAL_DIR/gluon/{entry,budget,round_N,track}`,
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
roles/        gluon_lead (entry, budget, close), gluon_profiler (baseline profile), gluon_engineer
              (transcribe/author/recover/climb), evidence_auditor (artifact audit);
              GEAK roles are read from kernel_workflow/roles/
tests/        harness.js (runs workflow.js with synthetic agent responses) and test_*.js:
              stage order per entry mode, required reading, evidence re-ask/reject (G3, G4, ATT state,
              gluon_profile re-profile, baseline profile layers, audit),
              acceptance by verify and the commit gate, parity/G1/deadline/token/budget stops,
              launch contract (args, arch, reference paths, size, no injection left in kernel/e2e)
```

Run: `for t in tests/test_*.js; do node "$t"; done`. CPU-only; no agent, GPU or file is touched.

## Open points

- `validation.status` of the skill stays `draft`; a run of this workflow is the natural validation
  harness (`expects.isolated_speedup_min` over the whole track), but `--record` is not wired here.
- A hung agent is resolved to null by the guard but not killed; the next round's reset to
  `TRACK_START` protects the track from a late writer only once it starts.
- The pack's record tools (`round_record.py`, `recordctl.py`, `close_audit.py`, `toolctl.py`) remain
  optional for the engineer; the workflow's own ledger is the `rounds[]` it returns and the per-round
  `round.md`.
