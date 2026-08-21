# Tuning skillset — how it is integrated (GEAK side)

This file documents the **wiring**. It deliberately contains **no tuning method**: the method lives in
the vendored skillset at `<repo>/tuning_skillset/` and is read from there. If you find yourself wanting
to copy a procedure out of the skillset into this file, that is the thing this design exists to prevent.

## The shape of the integration

```
<repo>/tuning_skillset/                      the skillset, VENDORED WHOLE and UNMODIFIED (47 files)
e2e_workflow/roles/tuning_specialist.md      the ONLY adapter — a thin role that delegates into it
e2e_workflow/e2e_workflow.js                 one gated phase block: want('tune'), after config, before head
e2e_workflow/knowledge/tuning_skillset.manifest.sha256    hash pin for the vendored tree
e2e_workflow/scripts/tuning_skillset_sync.py --verify|--update|--sync
```

That is the whole surface. No other role's prompt receives the skillset, and no part of its method has
been rewritten into GEAK's `knowledge/` tree.

## Why vendored whole

> **The skillset is validated standalone.** Its claims are executable (`validate/claims.py`, 37 of them),
> re-run per image and per architecture. That validation is evidence about a *specific tree*. Paraphrase
> it into six GEAK files and the evidence no longer describes what GEAK runs — you inherit the prose and
> lose the guarantee, and nothing tells you when the two have diverged.

Keeping it whole also keeps the loop cheap in both directions: the skillset continues to be developed and
re-verified standalone, and adopting a new version is `--sync` plus a manifest bump rather than a
re-scatter across GEAK.

Consequences, which are load-bearing:

- **Never edit anything under `tuning_skillset/`.** Fixes go upstream, then come back via `--sync`.
  `--verify` fails the moment a file inside the tree changes, is added, or goes missing.
- GEAK-side artifacts belong in `EVAL_DIR/tuning/`, never inside the vendored tree.
- The tree is invoked through **its own** entry points (`README.md` routes; each `tuning-*/SKILL.md` is
  independently invocable; `tuning-core/` is the shared foundation the others assume).

## Why its own phase, before HeadKernel

Position: `Setup → Profile → Strategize → ConfigSweep → **TuningSkillset** → HeadKernel → Milestone → Finalize → Report → Validate`.

**Standalone**, because the skillset is a complete six-step loop — scope, baseline, search, correctness
gate, deploy, engagement verify. Folded into the head-kernel bake-off it collapses into "try a tuned
config alongside the other candidates": the per-op tuners, the deploy paths and the engagement checks
never run, which is most of the skillset. Owning a phase is what makes it run end to end.

**Before HeadKernel**, for three reasons:

1. Tuning is the cheap lever and it **reshapes the profile**. Run it first and the (expensive) head
   budget goes to the ops that are still hot *after* tuning. An accepted tuning therefore triggers a
   re-profile + re-strategize, exactly as an accepted config sweep does.
2. Its accepted env/flags fold into the carried config, so every later measurement — head A/B reference
   legs, Finalize, the Director's Validate — happens **on top of** a tuned stack. Head deltas stay honest
   because tuning is already in their reference leg. (Corollary for readers of the report: the tuning
   delta and the head deltas must not be summed.)
3. **Attribution.** The phase takes its own in-session interleaved pre/post A/B, so the report can state
   what tuning bought, in tok/s and as a share of the run's total gain. Run it inside another phase and
   that number stops existing.

## What the orchestrator enforces (rather than trusts)

The role can return `gate:"accepted"` and still not be banked. `e2e_workflow.js` withholds the accept
unless **all** of these hold, and logs loudly when it does:

| bar | why |
| --- | --- |
| `engagement_verified === true` | the skillset's central thesis — an artifact the runtime never loads is not a win, and it fails *silently*, with plausible-looking numbers |
| `ab_complete !== false` | a post-only number is not a measurement (same rule as the head track) |
| `correctness_gate !== 'fail'` | a faster wrong server is a regression |
| `post > pre > 0` | both legs real, and the direction is actually a win |

A withheld accept is downgraded to `no_win` with the reason appended, and the run continues on the
pre-tuning config. This matters more here than elsewhere: an unproven tuning artifact would sit in the
reference leg of every downstream A/B.

## Args

| arg | default | meaning |
| --- | --- | --- |
| `tuning_skillset` | `"true"` | phase on/off. `"false"` injects nothing anywhere → byte-identical to a build without the feature |
| `tuning_skillset_dir` | `<repo>/tuning_skillset` | override to point at an upstream checkout (e.g. to re-verify standalone) |
| `tuning_kb` | `"true"` | consult `tuning-kb/`, the per-model **answer key**. Set `"false"` for blind evaluation runs — the skillset says so itself |
| `phases` | `all` | the phase key is `tune`, e.g. `phases:"tune"` to run only this phase against carried `state` |

There is deliberately **no op budget**. The head track caps its ops because each one spends a recursive
kernel-authoring run; tuning ops are cheap by comparison and their value is cumulative, so a cap would
only leave measurable wins on the table. The role decides where the returns stop and reports where.

Mode interaction: **fast** mode skips `tune` (its contract is head-kernel-only inside a wall-clock cap).
**deep** mode keeps it, for the same reason it keeps ConfigSweep — cheap, and a tuned stack is a better
starting point for the co-optimization lanes.

## Verifying, and keeping it verifiable

```bash
# integrity: is the vendored tree still the tree that was validated?
python3 e2e_workflow/scripts/tuning_skillset_sync.py --verify

# adopt a new upstream version (re-record the hashes in the same step)
python3 e2e_workflow/scripts/tuning_skillset_sync.py --sync /path/to/tuning_skillset

# the skillset's OWN standalone validation, run from inside GEAK, unchanged:
cd tuning_skillset && python3 validate/claims.py --json /tmp/claims.json
```

The last command is the point of the whole arrangement: the standalone verification loop keeps working
verbatim on the copy GEAK ships. `e2e_workflow/scripts/test_tuning_skillset_phase.js` guards the
structural invariants (vendored whole, not scattered, phase ordering, additive-when-off), and
`e2e_workflow/scripts/tests/test_tuning_skillset_sync.py` guards the integrity tool.

## How a tuning win reaches production

This is the part that is easy to get wrong, because the failure is silent: the run reports a gain, the
bundle ships, and the gain does not reproduce.

GEAK's deliverable is `EVAL_DIR/final/` — `overlay/`, `final_patch.diff`, `final_launch.sh` — and
`final_launch.sh` is contractually self-contained (it bakes `OVERLAY_PYTHONPATH`, the accepted
flags/env, `BACKEND`, `TP` and delegates to `bench_e2e.sh`). Callers reuse that script directly, so
**anything not expressible through it does not exist downstream.**

The overlay is a `PYTHONPATH` mechanism: `sitecustomize.py` injects patched modules and rebinds
attributes. It carries **code**. A tuning win typically is not code — it is a config table a library
reads from inside its own installed package directory, plus a derived cache that must be dropped or the
new rows are ignored while everything still looks healthy. Neither survives an overlay.

### Two halves, two routes

A tuning win can have a **code** half and a **data** half, and conflating them is how it breaks.

The code half is a dispatch-path change that makes the tuned artifact bind — a routing switch, or a
wrapper fix. This is squarely in scope for the phase (authoring kernels is not), and it is not
hypothetical: `tuning-aiter/SKILL.md` §2b measures a correctly-tuned row deployed on a wrapper that
drops `kernelName` running **85.7% slower than doing nothing** (−6.48% e2e) while every engagement gate
passes, and the same CSV on a wrapper that honours the selection giving **+23.88%**. It travels as a
reversible **overlay** (`apply_overlay` → carried `curOverlay` → `final/overlay`), exactly like a head
env-winner's routing patch. Never as a live-tree `.py` edit: that lands in both legs of every later A/B,
so the comparison silently measures the same thing twice.

The data half is the tuned table itself. Env-pointed artifacts (`AITER_CONFIG_*=<csv>`,
`PYTORCH_TUNABLEOP_FILENAME=<dat>`, `VLLM_TUNED_CONFIG_FOLDER=<dir>`) already work through `apply_env` →
carried `curEnv` → `accepted_config.env` → `FINAL_ENV`, which is why the head track's env winners need
nothing special. The gap is everything else. So the tuning phase writes a **deploy bundle**:

```
EVAL_DIR/tuning/deploy/
  MANIFEST.json       repo, base_sha, target_files, apply, rebuild, cache_invalidation,
                      extra_env, engagement_check, notes
  tuning_patch.diff   git apply-able, paths relative to `repo`
  files/              the artifacts, under destination-relative paths
  deploy.sh           idempotent installer: place files, invalidate caches, fail loudly
```

and Finalize (`roles/e2e_integrator.md` `PHASE=finalize`, step 1b) folds it into the existing handles
rather than inventing a new one:

| | |
| --- | --- |
| `final/tuning/` | the bundle, copied so `final/` stays self-contained |
| `final_patch.diff` | tuning diff concatenated under a `# --- tuning skillset ---` banner |
| `final_launch.sh` | runs `final/tuning/deploy.sh` **before** the server launch, exits non-zero if it fails |
| `FINAL_ENV` | `apply_env`, already carried in `ACCEPTED_ENV` |

Ordering is not cosmetic: a config applied to a running server does nothing, and graph-captured decode
paths read it at capture time, so the deploy must precede launch and a restart is mandatory.

### The live-tree carve-out

Installing a config table means writing **inside an installed package** (`/sgl-workspace/aiter/aiter/configs/...`),
and the Integrator has a hard rule against exactly that: it asserts `git status --porcelain` is empty in
the installed tree before and after every A/B leg, and restores anything dirty. Left alone, the head
track would delete the banked tuning mid-run — and then measure every later delta against a reference
leg that had quietly lost it, inflating all of them.

So the phase declares `live_tree_files` (also `MANIFEST.target_files`), the orchestrator hands it to
**every** integrate leg via `tuningIntegrateInputs()`, and the rule becomes "clean **apart from this
list**". The carve-out is narrow on purpose: any other dirty path is still a hard failure, a listed path
found missing is reported rather than measured past, and it covers **data only** — a `.py` listed there
is a contract violation, not a carve-out, because the code half has the overlay. It is sound because the
rule's rationale — unrecorded, irreversible, contaminates the baseline leg — inverts for these paths:
they are recorded, reproducible from the deploy bundle, and *supposed* to be in both legs, exactly like
accepted env.

An undeclared live-tree write is therefore the one way a tuning win can pass every gate in this phase and
still evaporate later in the run.

Finalize then **re-checks engagement** on the assembled bundle using the manifest's `engagement_check`
and returns `tuning_in_bundle` + `tuning_engagement_recheck`. A bundle that silently drops the tuning is
reported as broken rather than shipped as complete.

## Reporting

The phase result flows out three ways:

- `wfReturn.tuning_skillset` — machine-readable, including `tuning_delta_pct`, `tuning_speedup`,
  `share_of_total_gain_pct` (null when the run had no net gain to apportion), and the deploy fields.
- `result.json` → **`tuning_skillset`**, added by `interface/run_e2e.py`. This is **purely additive**:
  every pre-existing key keeps its name, type and meaning, and a run without the phase produces a
  byte-identical `result.json`. The block carries a prose `explanation` plus `reaches_production_via`,
  which tells a consumer that `final_patch` and `final_launch_script` — the handles it already uses —
  now also carry the tuning. Note that the tuning gain is **inside** the headline
  `throughput_speedup`, not additional to it; the block attributes part of the headline rather than
  adding to it.
- `final_report.md` **§2b "Tuning skillset — attributable contribution"** — mandatory whenever the phase
  ran; see `roles/system_architect.md` `PHASE=report`. Its pre/post pair is the tuning phase's own
  same-session A/B and is **not** overwritten by the Director's headline reconciliation.
