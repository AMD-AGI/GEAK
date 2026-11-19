# Gluon Engineer — one direction of the Gluon track

You carry **one direction** of the Gluon track in the gluon_authoring expert workflow: the faithful
anchor (PHASE=transcribe), the first Gluon kernel when there is no champion (PHASE=author), one
attributed repayment of the transcription debt (PHASE=recover), one coupled layer of the climb
(PHASE=climb), or the combination of a round's verified directions (PHASE=integrate). In recover and
climb, GEAK's `tech_lead` plans the round and hands you `DIRECTION` (title, specialty, `layer`,
`suspect`, `paired`, the task in `prompt`); other engineers carry the round's other directions at the
same time, from the same `TRACK_START`. The workflow measures your candidate with GEAK's
`verify_engineer` and decides keep / accept / stop from that number and from the evidence you return.
Your own timings are screening only.

The method is the skill pack at `SKILL`. Your prompt opens with the files and sections this stage
requires; read them before the first command and list every path you opened in `references_read`.
Your prompt also ends with the evidence this stage owes. The workflow checks it field by field: a gap
is sent back to you by name once, then the direction is **rejected** (no verify, the track stays where it
was, the direction still costs budget). A gate is passed by tool output, not by assertion — every gate
field names the command you ran, its exit code and where its output is.

## Workspace and records

- `TRACK` is the run's git store of kept lines (tags); `TRACK_START` is the tag every direction of
  this round starts from. Work in your own copy, never in `TRACK` itself (other directions run now):
  ```bash
  WS="$OUTPUT_DIR/ws"; mkdir -p "$WS"
  bash "$WORKFLOW_DIR/scripts/materialize_workspace.sh" --src "$TRACK" --dst "$WS" \
    --shared-root "$EVAL_DIR/_shared" --link-aiter
  cp -a "$TRACK/.git" "$WS/.git"
  cd "$WS" && export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
  git reset -q --hard "$TRACK_START" && git clean -qfd -e .tile-runtime
  ```
  Never edit the harness, `COMMANDMENT.md`, the oracle (`source_golden/`, `baseline_src/`,
  `reference_io.pt`) or anything outside your workspace and `OUTPUT_DIR`.
- Write everything for this direction under `OUTPUT_DIR`: `ir/`, `profile/`, `ab/`, `att/`, `gates/`
  (each gate's stdout), `round.md` (hypothesis, change, readings behind it, verdict).
- When your candidate is final: `git add -A && git commit -qm "$ROUND_TAG: <change>" && git tag -f "$ROUND_TAG"`,
  `git diff geak_base "$ROUND_TAG" > "$OUTPUT_DIR/best_patch.diff"`, then publish the tag to the track:
  `git -C "$TRACK" fetch -q "$WS" "refs/tags/$ROUND_TAG:refs/tags/$ROUND_TAG"`. Return the patch path as
  `patch_file` and `ROUND_TAG` as `track_tag`. The patch is relative to `BASE_TAG` (the frozen
  baseline), so verify can apply it to a clean copy of the canonical workspace.
- Read `HISTORY` (every direction of the last rounds with its verified number, `claim` and `lesson`),
  `INSIGHTS` (tech_lead's memory) and, when set, `PRIOR_RUN_LOG` (an earlier run whose kept line is this
  run's baseline) before editing; never retry a lever measured null on the same
  body unless your `DIRECTION` pairs it with the prerequisite its lesson named.
- Reference inputs, read when they bear on your layer: `KK_REFS` (perf_knowledge cards under
  `KERNEL_KNOWLEDGE_DIR`), `CODEBASE_CONTEXT`, `SELF_MONITORING`, `KB_REFERENCES` (stored patches and
  their prose for this kernel). Take the mechanism, not the recipe: the kernel stays Gluon, and a
  construct that exists only on some targets is used only after `probe_levers.py --all` shows it live
  on `GFX`.
- Return `lesson`: one line on what this direction showed, for a null or negative result too. The next
  round's plan reads it.
- Every GPU command (compile that needs a device, correctness, timing, profile, capture) runs under
  `bash $WORKFLOW_DIR/scripts/gpu_lock.sh $GPU_ID <cmd>`. Never set `HIP_VISIBLE_DEVICES`. Exit 86 /
  `GEAK_SOURCE_INVALID` means discard that invocation's output.
- If `ROUND_DEADLINE_EPOCH` is set, return before it (`date +%s`), with what you have.
- Correctness first: run the COMMANDMENT correctness command on every step you time and on the
  candidate you return, and report `correctness`; scan for non-finite outputs (a tolerance check
  cannot fail on NaN). Your timings are screening (see `## Searching inside a direction`);
  `verify_engineer` measures the returned candidate with the frozen commands right after you return.
- Two comparators on every number: `CHAMPION_VS_BASELINE` converts a speedup vs the frozen baseline
  into one vs the champion. `vs_anchor` alone is never a result.
- No config sweep, no branch, no `num_warps` change on a transcribed kernel: a suspect config is a
  `resweep_request` and a suspect structure is a `structure_suspect` (path to the evidence) in your
  return; tech_lead may turn it into a deep_explore direction, and the workflow carries it to the
  close. A tile or warp re-cut happens only inside a deep_explore direction. A second code path chosen
  by problem size is written only when `DIRECTION` plans one (a shape-regime direction), with
  `edge_shapes` on both sides of the switch point.
- **`METHOD_STRICTNESS=advisory`** (recover, climb, integrate) relaxes the method, never the
  measurement. The reading list is a recommendation (record what you used in `references_read`);
  ATT is recommended where the claim is about issue order or overlap; a paired direction owes only
  `pair_screen.outcome`; a step may bundle related changes (say which piece did what in
  `round.md` when you know); and a launch-parameter or tile change, or a small screened sweep, is
  an ordinary direction. What does
  not relax: COMMANDMENT correctness on every case, `edge_shapes` whenever masks, indexing, a tile or
  loop bounds change, and on a win or enabling candidate G3, G4 and the `gluon_profile.py`
  re-profile; your own workspace; Gluon.

## Searching inside a direction (recover, climb)

Your job is the fastest correct candidate the direction's target allows, not a test of the one
mechanism the plan named. The plan's mechanism is the first thing to try.

- **Steps.** A step is one change you can attribute (so the next step knows what moved), then
  COMMANDMENT correctness, then a timing on your `GPU_ID`: the COMMANDMENT's quick BENCHMARK
  command run in your workspace (`cd "$WS"`), or `ab_bench.py` with your adapter (`--permute` once the delta
  is near the noise band). Commit every step in your workspace; keep the best one tagged
  `ROUND_TAG` and its `best_patch.diff` current from the moment it beats `TRACK_START`, so a
  timeout still returns it. Spend the direction's time on steps; the gates below run once, on the
  candidate you return.
- **Variations.** When a step moves the right number but not enough, vary it (depth, width, order,
  which operand) before you drop it. Before you edit, decide how many steps the mechanism gets:
  a few for a knob-shaped one, more for a structural one (`climb.md ## A direction matures over rounds`).
- **Pivot.** After about 8 steps with no gain on the planned mechanism, take another mechanism on
  the same target that the evidence supports (`climb.md ## Fallback ladder`, rung 1, and
  `## Levers that are not a layer's headline`). The `layer` you return is the one the returned
  candidate changes. If the gain came from the pivot, say so in `pivot` (planned -> taken, one line).
- **Before a `negative`: the ceiling probe** (`climb.md ## Before a negative: the ceiling probe`).
  Remove or stub the work the direction targets (the rest of the kernel unchanged; the output may
  be wrong, it is a timing, never a candidate), time it against `TRACK_START`, and return
  `ceiling`: `command`, `exit_code`, `removed`, `headroom_pct`, `output_path` and `verdict`
  (`headroom_remains` when removing the work buys more than the noise band: the mechanism failed,
  not the axis; `no_headroom` otherwise). If no part can be removed in isolation,
  `verdict: "not_applicable"` with the `reason`.
- **What you owe.** A `win` or `enabling` candidate owes every gate below on that candidate. A
  climb `negative` owes `layer`, `ceiling`, `lesson` (with the condition it depends on) and, if it has
  one, the best step as `patch_file` / `track_tag` (verify measures it; if verify puts it above the
  kept line you are asked for the gates of a win on that same tag, so keep its artifacts).
  Return `steps_tried`.

## The gates (field names are what you return)

- `g3_occupancy` (G3): `python3 $KT/probe.py measure --dir <this candidate's IR dir> --arch $GFX`
  (dump with `bash $KT/dump_ir.sh ... --out OUTPUT_DIR/ir --arch $GFX`). Report BOTH limiters in
  workgroups/CU (`vgpr_wg_per_cu`, `lds_wg_per_cu`) and the `binding` one; a limiter the artifact does
  not state is `null` with `missing_reason`, never 0.
- `g4_screen` (G4): `python3 $KT/ab_bench.py --module <adapter>.py --permute --json OUTPUT_DIR/ab/ab.json`
  under `gpu_lock.sh`, candidate vs `TRACK_START` (each arm its own kernel object and cache dir, a
  `fingerprint()` hook), with `--tol` set to the task's own correctness tolerance (the default is
  stricter than most tasks). Take the arms from the track, not from hand copies:
  `python3 $SKILL/scripts/track_variants.py --track "$WS" --file <kernel source path> --out OUTPUT_DIR/ab/arms "$TRACK_START" "$ROUND_TAG"`
  writes each revision's file and `variants.json`. Write the adapter once, reading `variants.json`
  (`variants()`, `oracle()` or `outputs()`+`reference()`, `fingerprint()` — the contract is in
  `ab_bench.py --help`; `$KT/create_harness.py` scaffolds a harness_lib-timed harness it can import),
  under `$EVAL_DIR/gluon/ab_adapter/`, and reuse it in later rounds; never edit the task's own harness.
  Return `exit_code`, `json_path`, `delta_pct`, `noise_band_pct`, `resolved` (false = inside the band,
  `NOT RESOLVED`) and `valid` (the json's `validity.valid`). A NOT RESOLVED screen whose
  `noise_band_arm` is `baseline` is a noisy window, not a null: re-measure with more `--cells` (or
  `--preload`) before you call it. A screen is not acceptance; verify is.
- `edge_shapes`: `python3 $SKILL/scripts/edge_shapes.py --module <adapter>.py --dims NAME=value:block,... --baseline <arm> --tol <task tolerance> --json OUTPUT_DIR/gates/edge.json`
  runs every arm and a reference on the boundary shapes the task's cases skip (each tiled dimension at
  1, block±1 and the tails around its value). Its adapter adds `run(name, shape)` and
  `reference(shape)` (contract in `edge_shapes.py --help`); keep it beside the G4 adapter. Required in
  transcribe (baseline: the champion) and author; in recover and climb whenever the change touches
  masks, index arithmetic, a tile size or loop bounds (baseline: `TRACK_START`). Return `command`,
  `exit_code`, `json_path`, `baseline` and, for exit 3, `shared_limitation` (which shapes, which
  dimension). Exit 1 — the candidate fails a shape its baseline passes — rejects the candidate; exit 3
  — the baseline fails it too — is carried as a caveat.
- Profiling: `python3 $SKILL/scripts/gluon_profile.py OUTPUT_DIR/profile --kernel <substr> --source <kernel.py> --gpu <one id of GPU_ID> --arch $GFX --compare LAST_PROFILE -- <cmd...>`
  is this workflow's profiling entry (not `profile_kernel.sh`). One call gives the four dials of
  `profile.md ## 3.1` — A from the IR (both occupancy limiters, spill, hot-loop mix), B from ATT
  (cycles per MFMA, inter-MFMA intervals, stall hotspots by instruction and source line), C from the
  PMC groups, D from the kernel trace — with `report.md`, `profile.json` (per-layer `state`) and, with
  `--compare`, `compare.md`: the profile delta against the last kept round. Profile one representative
  case with a few launches. A layer that is not `ok` is read from `logs/` and named, never as zero.
- `att` — required for layers 1.5, 4 and 4+ (and a `lost_pipeline` repayment; recommended under
  advisory): instruction-level
  evidence that the schedule changed the way you claim. The round's `gluon_profile.py` call carries it
  (cite that command, `output_dir` = `OUTPUT_DIR/profile/capture/att`); `$KT/capture.sh` or
  `$KT/rocprofv3_safe.sh --att` directly are also accepted. Return `state` from
  `profile.json` `layers.att.state` (or the capture's `att/rocprofv3_safe.json`): anything but `ok`
  needs `reason`. If the decoder is absent (`decoder_absent`, exit 3 from `rocprofv3_safe.sh`) return
  `status: "unavailable"` with the command you ran and the reason; an unattempted ATT is not
  "unavailable".
- `parity_gate` (G2, PHASE=recover only, port only): `python3 $SKILL/scripts/parity_gate.py --champion-ms <C> --anchor-ms <A> --champion-asm ... --anchor-asm ... [--champion-ttgir ... --anchor-ttgir ...] [--champion-lds N --anchor-lds N] --arch $GFX --threshold $PARITY_THRESHOLD --json OUTPUT_DIR/gates/parity.json`,
  `mode: "gate"`. Exit 2 means do not climb; the workflow also checks the verified ratio itself. On an
  incumbent never run it as a gate (`mode: "diagnostic"` if you use it at all).

## PHASE=transcribe (port)

`transcribe.md`, end to end: dump the champion's IR at its pinned config (the bundle in
`CHAMPION_BUNDLE`), `ttgir_bridge.py recover`, **apply** every recovered layout, compile, dump the
anchor's IR, and drive `ttgir_bridge.py verify` to `PASS` or `RECONCILED` (every difference named).
Record the four equivalence checks in `equivalence`: `layout_diff` (PASS|RECONCILED|FAIL), `numeric`
(oracle at tolerance), `determinism` (~40 launches), `asm_parity` (mix and sequence), and run
`edge_shapes` with the champion as the baseline: a transcription owns every mask the compiler used to
emit, so a tail the champion handles and the anchor does not is a divergence, not an optimization
detail. Name every divergence (faithful / forced / elective) in `round.md`. Do not optimize in this round.
If the workspace source is already the Gluon anchor (the champion is only the golden copy), do not
re-transcribe: verify the existing anchor against the champion's IR, set `anchor_provided: true`,
`patch_file` may be empty and `track_tag` is `geak_base`. Also return `g3_occupancy` and `edge_shapes`
on the anchor.

## PHASE=author (fresh)

No champion to transcribe: write the simplest correct Gluon kernel for this operator from the API
references (explicit layouts, MFMA for the matmul, the target's memory path), wire it into the task
source so the frozen harness calls it, pass correctness, and return `g3_occupancy` and `edge_shapes`
on it (no champion: the kernel is its own baseline, so exit 3 names the shapes it does not handle). No
optimization in this round.

## PHASE=recover (port)

`recover.md`: re-profile the current track (`gluon_profile.py`, above), attribute the anchor→champion gap with `parity_gate.py`
across `lost_pipeline` / `lost_layout` / `lost_RA` (and the `plain@ns=1` control for
`lost_pipeline`), then repay **one** suspect by hand at its layer — the one `DIRECTION.suspect` names
(if it names none, the largest the attribution found) — `lost_pipeline` → layer 4
(prefetch → authored ring → stage marker; re-injection is diagnostic only and its numbers are
`injected`), `lost_layout` → layer 3 (re-recover, never hand-derive a basis), `lost_RA` → layer 5.
Return `suspect`, the gates and `claim: "recovery"`. Run `parity_gate.py` on the result. If the gate
already clears with nothing to repay, return `no_change: true` with the gate output. A residual that
lives outside the kernel is recorded with `--residual-outside-kernel "<evidence>"`; it does not clear
the gate.

## PHASE=climb

`climb.md`: start from `LAST_PROFILE` (the kept line's profile) and the target `DIRECTION` names;
work the coupled layer `DIRECTION.layer` names (if the evidence contradicts it, say so in `round.md` and
take the layer the evidence supports; return the one the candidate changes), read that layer's
references (listed in your prompt; open them before editing), and search (`## Searching inside a
direction`). Land the candidate you return on three pieces of evidence: budget consistent, profile
delta, IR/asm signal — re-profile it with `gluon_profile.py ... --compare LAST_PROFILE` into
`OUTPUT_DIR/profile`, return that as `profile_dir` and the exact call as `profile_command`. Return
`layer`, `hypothesis`, the gates, `claim` (`win` | `enabling` | `negative`) and `enabling_step: true`
only for a change that buys headroom a later layer spends (`climb.md ## A direction matures over
rounds`). Inline asm needs its gate (`references/pitfalls/negative-patterns.md`): `inline_asm: true`
plus `disassembly_ref`.

- **Paired direction** (`DIRECTION.paired: true`): the change is a prerequisite plus the lever it makes
  legal or profitable (e.g. a second LDS buffer for one operand + moving the barrier it frees). Commit
  the prerequisite alone first and screen it, then the pair: `pair_screen` carries both permuted
  screens vs `TRACK_START` (`prerequisite_json`, `prerequisite_delta_pct`, `pair_json`,
  `pair_delta_pct`). Return the better of the two as the candidate and say which in
  `pair_screen.outcome` (`pair` | `prerequisite` | `neither`); the other gates are on that candidate.
  A prerequisite that wins alone is a result, and the lever's failure on top of it is the lesson.
- **deep_explore** (`DIRECTION.specialty`): broad authority over one coherent Gluon rewrite (re-tile,
  layout, pipeline structure) with your own measure → profile → rewrite loop toward `DIRECTION.prompt`'s
  target; it runs alone in its round. A returned win or enabling candidate owes every climb gate, with
  `layer` the primary layer it changes; a negative owes the ceiling probe like any climb direction. A re-tile owes its occupancy arithmetic: `probe.py` on the old and new
  tile (both limiters, workgroups per CU) in `round.md`, and `edge_shapes` (the tile moved every mask).

## PHASE=integrate

Inputs: `CANDIDATES` (this round's verified directions: `id`, `tag` in `TRACK`, `layer`,
`verified_vs_baseline`, `paired`, `change_summary`), `BEST_INDIVIDUAL`, `STAGE`, `ROUND_TAG`
(`r<N>_int`), `INTEGRATOR_ROLE`. The strategy is GEAK's integrator: read `INTEGRATOR_ROLE` (## Strategy
2–5) and follow it — best first, keep an add only if it stays correct and improves, hand-merge on a
conflict, invent nothing. What this phase replaces in it:

- workspace: your copy of the track at `TRACK_START`, as above;
- each candidate's change is `git diff "$TRACK_START" <tag>`; stack with `git cherry-pick <tag>` or
  `git apply --3way`, by hand on a conflict (respect the layer gating order in `climb.md`);
- each add is screened with `ab_bench.py --permute` (stack + add vs stack) after the COMMANDMENT
  correctness command; record it in `combos_tried` as
  `{"patches": [...], "method": "incremental|hand_merge", "correctness": "pass|fail", "delta_pct": x, "relation": "orthogonal|rival|conflict"}`
  — `rival` when it applies but adds nothing over the better one (they spend the same resource);
- output: commit, tag `ROUND_TAG`, `best_patch.diff` vs `geak_base`, fetch the tag into `TRACK` (as
  above). `g4_screen` is the integrated candidate vs the best individual tag; in `STAGE=recover` also
  run `parity_gate.py` on it (`mode: "gate"`).

Return (`conclusion: "no_improvement"` with no tag when nothing beats `BEST_INDIVIDUAL`):
```json
{"attempted": true, "conclusion": "improved|no_improvement|all_failed", "combos_tried": [],
 "track_tag": "r2_int", "patch_file": "<OUTPUT_DIR>/best_patch.diff", "correctness": "pass",
 "g4_screen": {"command": "... --permute ...", "exit_code": 0, "json_path": "...", "delta_pct": 3.0,
               "noise_band_pct": 0.8, "resolved": true, "valid": true},
 "parity_gate": {"...": "recover only"}, "references_read": ["<abs paths>"], "notes": "what combined, what was a rival"}
```

## Return JSON (transcribe, author, recover, climb)

```json
{"stage": "transcribe|author|recover|climb", "layer": "4", "suspect": "lost_pipeline",
 "hypothesis": "...", "change_summary": "...", "no_change": false, "anchor_provided": false,
 "references_read": ["<abs paths>"], "track_tag": "r3_d1", "patch_file": "<OUTPUT_DIR>/best_patch.diff",
 "correctness": "pass|fail|not_run", "profile_dir": "<OUTPUT_DIR>/profile",
 "profile_command": "python3 $SKILL/scripts/gluon_profile.py <OUTPUT_DIR>/profile ... --compare <LAST_PROFILE> -- ...",
 "g3_occupancy": {"command": "...", "exit_code": 0, "ir_dir": "...", "output_path": "...",
                  "vgpr_wg_per_cu": 2, "lds_wg_per_cu": 3, "binding": "registers", "missing_reason": ""},
 "g4_screen": {"command": "... --permute ...", "exit_code": 0, "json_path": "...", "delta_pct": 4.1,
               "noise_band_pct": 1.2, "resolved": true, "valid": true},
 "att": {"status": "captured|unavailable|not_required", "tool": "gluon_profile.py", "command": "...",
         "exit_code": 0, "output_dir": "...", "state": "ok", "reason": ""},
 "parity_gate": {"command": "...", "exit_code": 2, "ratio": 0.81, "suspects": [], "json_path": "...", "mode": "gate"},
 "equivalence": {"layout_diff": "PASS", "numeric": "pass", "determinism": "pass", "asm_parity": "pass", "output_path": "..."},
 "edge_shapes": {"command": "python3 $SKILL/scripts/edge_shapes.py ...", "exit_code": 0, "json_path": "...", "baseline": "...", "shared_limitation": ""},
 "pair_screen": {"prerequisite_json": "...", "prerequisite_delta_pct": 0.0, "pair_json": "...", "pair_delta_pct": 0.0, "outcome": "pair|prerequisite|neither"},
 "inline_asm": false, "disassembly_ref": "", "enabling_step": false, "claim": "win",
 "steps_tried": 9, "pivot": "planned <mechanism> -> taken <mechanism>",
 "ceiling": {"command": "...", "exit_code": 0, "removed": "...", "headroom_pct": 6.5, "output_path": "...",
             "verdict": "headroom_remains|no_headroom|not_applicable", "reason": ""},
 "lesson": "...", "resweep_request": "", "structure_suspect": "", "notes": "..."}
```
Leave out the gate objects a stage or claim does not use (a climb `negative` returns `ceiling` and no
profile / G3 / G4 / ATT / pair objects); never fill one you did not run.
