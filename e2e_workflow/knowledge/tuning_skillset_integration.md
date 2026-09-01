# Tuning skillset — how it is integrated (GEAK side)

This file documents the **wiring**. It deliberately contains **no tuning method**: the method lives in
the vendored skillset at `<repo>/perf_knowledge/expert_skills/tuning/` and is read from there. If you find yourself wanting
to copy a procedure out of the skillset into this file, that is the thing this design exists to prevent.

## The shape of the integration

```
<repo>/perf_knowledge/expert_skills/tuning/  the skillset, VENDORED WHOLE and UNMODIFIED (47 files)
e2e_workflow/roles/quick_tune.md             the cheap standalone pass — one head op, bare, bounded
e2e_workflow/roles/op_benchmarker.md         Tier A/B — the wider per-op search, same op, same oracle
e2e_workflow/e2e_workflow.js                 no phase: quickTune() + headTuningInputs() inside HeadKernel
e2e_workflow/knowledge/tuning_skillset.manifest.sha256    hash pin for the vendored tree
e2e_workflow/scripts/tuning_skillset_sync.py --verify|--update|--sync
```

That is the whole surface. No other role's prompt receives the skillset, and no part of its method has
been rewritten into GEAK's `knowledge/` tree. The two roles that do receive it are the two that already
own the op being tuned — which is the point of the change described in the next section.

## Why vendored whole

> **The skillset is validated standalone.** Its claims are executable (`validate/claims.py`, 37 of them),
> re-run per image and per architecture. That validation is evidence about a *specific tree*. Paraphrase
> it into six GEAK files and the evidence no longer describes what GEAK runs — you inherit the prose and
> lose the guarantee, and nothing tells you when the two have diverged.

Keeping it whole also keeps the loop cheap in both directions: the skillset continues to be developed and
re-verified standalone, and adopting a new version is `--sync` plus a manifest bump rather than a
re-scatter across GEAK.

Consequences, which are load-bearing:

- **Never edit anything under `perf_knowledge/expert_skills/tuning/`.** Fixes go upstream, then come back via `--sync`.
  `--verify` fails the moment a file inside the tree changes, is added, or goes missing.
- GEAK-side artifacts belong in `EVAL_DIR/tuning/`, never inside the vendored tree.
- The tree is invoked through **its own** entry points (`README.md` routes; each `tuning-*/SKILL.md` is
  independently invocable; `tuning-core/` is the shared foundation the others assume).

## Why it is NOT its own phase (it used to be)

Position: `Setup → Profile → Strategize → ConfigSweep → **HeadKernel (extract → quick_tune → bake-off →
lanes → e2e gate)** → Milestone → Finalize → Report → Validate`.

It used to be a phase of its own, `TuningSkillset`, between ConfigSweep and HeadKernel, run by one
`tuning_specialist` role, taking its own interleaved pre/post server A/B. The argument was that the
skillset is a complete six-step loop — scope, baseline, search, correctness gate, deploy, engagement
verify — that collapses into "try a tuned config alongside the other candidates" when it is folded into
a bake-off, and that owning a phase is what makes it run end to end. The second argument was
attribution: its own A/B is what let the report state what tuning bought as a separate number.

Both were real. Neither survived contact with what the two tracks were actually doing:

- **They were tuning the same ops from opposite ends.** GEMM and attention are exactly what the skillset
  has tuners for and exactly what HeadKernel optimizes. With a phase boundary between them, the head
  track could not see which backend tuning had already proven fastest, and the tuning track could not
  see the roofline or the extracted op oracle the head track was about to build. Each re-derived the
  other's findings, on different harnesses, and the two numbers were not comparable.
- **The standalone A/B was the most expensive measurement in the run, taken at the worst moment** —
  a full server pre/post before HeadKernel had established anything, on a stack HeadKernel would then
  change underneath it.

So the loop now runs **per head op**, on that op's extracted unittest — the same immutable oracle the
deep lanes are scored on — as the two cheapest rungs of the head ladder:

1. **`quick_tune`** (`roles/quick_tune.md`): a bare, bounded standalone pass, one op, no server. It
   recalls a tuned table from the per-op store if one exists, tries the env/dispatch knobs and a small
   per-shape sweep, proves engagement, writes the deploy bundle, and hands `handoff_note` forward. Bare
   rather than a `roleAgent` on purpose: it never launches a server, so the serving-config invariant
   every role prompt carries is noise to it, and the point of this rung is to be the cheapest thing on
   the ladder.
2. **`op_benchmarker` Tier A/B**: the wider cross-backend bake-off and per-backend tune, told what rung 1
   already timed so it does not re-search it.

Then, and only then, the expensive `kernel_workflow` lanes run, on top of whatever tuning found.

The six-step loop is intact — it is scoped to one op instead of to the whole model, which is the scope
its own per-op tuners work at anyway. What is gone is the separate phase, the separate role, and the
separate server A/B.

**What that costs, stated plainly.** The attribution number is gone in the form it had:
`tuning_delta_pct`, `tuning_speedup` and `share_of_total_gain_pct` are now `null`, which means NOT
MEASURED and never zero. What replaces it is per op and arguably better evidence —
`ops_tuned[].isolated_speedup`, each measured on an immutable oracle rather than inferred from a server
delta with a drifting baseline — but it does not answer "what % of the run's gain was tuning", and a
consumer that needs that number will not find it. Accepted env/flags/overlay are folded into the carried
config at the end of the head track, **before** the post-head re-profile, so the Milestone loop ranks
kernels on the tuned stack and every later A/B measures on top of it. The tuning gain is inside the head
track's reference leg: tuning and head deltas must not be summed.

## What the orchestrator enforces (rather than trusts)

The role can return `gate:"accepted"` and still not be banked. `e2e_workflow.js` withholds the accept
unless **all** of these hold, and logs loudly when it does:

`finalizeHeadTuning()` banks an op only when **all** of these hold, and logs loudly when it drops one:

| bar | why |
| --- | --- |
| `gate === 'accepted'` | the role's own verdict, which is necessary and not sufficient |
| `engagement_verified === true` | the skillset's central thesis — an artifact the runtime never loads is not a win, and it fails *silently*, with plausible-looking numbers |
| `isolated_speedup > 1.0` | measured on the op's immutable oracle, against the frozen live kernel |

An op that fails any bar is not folded into `curEnv`/`curFlags`/`curOverlay` and is not banked as an
accepted kernel; the run continues on the untuned config for that op. This matters more here than
elsewhere: an unproven tuning artifact would sit in the reference leg of every downstream A/B. Its
`handoff_note` still reaches the bake-off — a dead end that is explained is worth passing on.

`correctness_gate` is enforced one level down, by the oracle itself: the unittest checks against
`meta.tol`, so a numerically-wrong candidate never produces a speedup to bank in the first place. That
is stricter than the old phase's server-level accuracy check and it is why there is no separate row for
it here.

## Args

| arg | default | meaning |
| --- | --- | --- |
| `tuning_skillset` | `"true"` | tuning on/off. `"false"` injects nothing anywhere → byte-identical to a build without the feature |
| `quick_tune_minutes` | `25` | the budget STATED to the cheap standalone pass, per head op. `0` disables that rung; `op_benchmarker` Tier A/B still reads the skillset |
| `tuning_skillset_dir` | `<repo>/perf_knowledge/expert_skills/tuning` | override to point at an upstream checkout (e.g. to re-verify standalone) |
| `tuning_kb` | `"true"` | consult `tuning-kb/`, the per-model **answer key**. Set `"false"` for blind evaluation runs — the skillset says so itself |
| `phases` | `all` | there is no `tune` phase key any more — tuning runs wherever `head` runs. The token is still recognised (and still skipped in fast mode) so an older caller passing `phases:"tune"` resolves to false rather than throwing |

There is deliberately **no separate op budget**: tuning follows the head budget, because it now tunes
exactly the ops the head track selected and no others. Within one op, `quick_tune_minutes` is stated to
the agent rather than enforced as a shorter wall clock — cutting an in-flight tuning pass short loses
the measurement it was about to return, so the rung stays cheap because of what it is told to do, not
because it gets killed.

Mode interaction: tuning rides the head track, so it runs wherever HeadKernel runs — including **fast**
mode, which it could not before. **deep** mode gets it on every one of its head ops, which is the case
the old phase served worst: the co-optimization lanes now start from a tuned op.

## Verifying, and keeping it verifiable

```bash
# integrity: is the vendored tree still the tree that was validated?
python3 e2e_workflow/scripts/tuning_skillset_sync.py --verify

# adopt a new upstream version (re-record the hashes in the same step)
python3 e2e_workflow/scripts/tuning_skillset_sync.py --sync /path/to/tuning_skillset

# the skillset's OWN standalone validation, run from inside GEAK, unchanged:
cd perf_knowledge/expert_skills/tuning && python3 validate/claims.py --json /tmp/claims.json
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

- `wfReturn.tuning_skillset` — machine-readable: `ops_tuned[]` (per op: `isolated_speedup`, `engaged`,
  `artifact`) is the attribution, plus the deploy fields. `tuning_delta_pct`, `tuning_speedup` and
  `share_of_total_gain_pct` are kept in the shape for consumers written against the old phase and are
  always `null` — NOT MEASURED, never zero.
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
