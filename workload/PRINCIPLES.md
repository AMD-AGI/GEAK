# Principles

The charter for everything under `workload/`. `README.md` explains how the pipeline is
built and why; this file states what any change to it must still be true of afterwards.
Read it before editing code here. When a rule and a convenience disagree, the rule wins —
and if a rule is genuinely wrong, change *it*, in this file, with the reason, rather than
working around it in one place.

## 1. The mandate

Add an optimization pipeline for **real agentic inference workloads** to GEAK, in five
steps:

1. **baseline** — measure the workload reproducibly.
2. **profile** — find where GPU time actually goes under that traffic, and turn the hot
   kernels into standalone optimization tasks; report each kernel's share and its
   roofline.
3. **optimize** — spend a fixed compute budget making those kernels faster, in parallel,
   one task per GPU, default budget 48 h. GEAK's `kernel_workflow.js` is the default
   method; running an agent directly, without GEAK, must remain available as an
   alternative.
4. **applyback** — put the survivors back into the serving stack.
5. **e2e** — re-measure and report the end-to-end result.

Four standing requirements came with it, and they are the reason most of the rules below
exist:

- **It is a skill, and it lives in the repo.** The code goes under `workload/` in the GEAK
  repository. Nothing it runs may depend on a task-local absolute path.
- **It is modular.** Swapping the profiling method, the roofline, the optimization method
  or the agent CLI must be a matter of naming a different implementation — not an edit.
- **It is effective.** The pipeline exists to produce a real speedup on a real deployment,
  not to produce a report about one.
- **It is robust.** It runs unattended for days, across node failures, and resumes.

## 2. Rules

### Scope and placement

**R1 — Everything lives in the repo; nothing points at a developer's machine.**
All paths come from config and resolve against the repository root. A scenario is a JSON
file in `config/`. If a campaign needs node-local storage — and it does, because traces
truncate on NFS — that path is declared in its config with the reason, not hard-coded.
*Enforced by:* `stage.py` path resolution; `workload/tests`.

**R2 — No secrets in the tree, ever.**
`runners/claude.env.example` is a committed template; a filled-in copy never is. Gateway
keys, tokens and subscription headers live outside the repository, in a file the scenario
config points at via `runner_env.env_file`, and the runner sources it itself. No code path
prints a credential into a log or a `result.json`.
*Enforced by:* `.gitignore` (`*.env`, with the template whitelisted); the repo's
`.gitleaks.toml`; the runner reading credentials rather than being handed them.

### Modularity

**R3 — Modularity is two contracts, not a framework.**
`stages/CONTRACT.md` (read `$IN` and `$CONFIG`, write `$OUT`, finish with
`$OUT/result.json`) and `runners/CONTRACT.md` (`runner.sh <prompt> <workdir> <log>`; `0`
success, `78` unmet precondition, anything else retryable). Do not add a plugin registry,
a DSL, a pipeline description language or a dynamic workflow engine. A new implementation
is a directory with a `SKILL.md`, named in the config. That is the whole mechanism, and
keeping it that small is what makes it survive being swapped.

**R4 — Agent execution goes through the runner, never through a named CLI.**
No stage script, scheduler or helper may invoke `claude` — or any other agent binary —
directly. `runners/stub.sh` calls no agent at all and exists as executable proof that the
scheduler is agent-agnostic; if a change breaks it, the change coupled the pipeline to a
vendor.

**R5 — Vendored skills are byte-identical to what was handed over.**
The upstream profile and roofline skills are installed unmodified, down to their own
`scripts/` and `tests/`, under a hash manifest. Fix their defects in the slot around them,
or upstream — never by editing the vendored tree. A 48-hour campaign must not be running a
skill that no longer matches its `MANIFEST`.
*Enforced by:* `scripts/skill_manifest.py verify-all`.

**R6 — The slot owns the criterion; the skill owns the work.**
How artifacts map to `result.json`, and what counts as ranked or passing, belong to the
stage — in `finalize.py`/`rank.py` where it is mechanism, and in `stages/<stage>/README.md`
where it is judgement. Replacing an implementation must never require rewriting the
mapping.

### Honesty of numbers

These rules are not caution for its own sake. Each one is here because the failure it
prevents has already happened on this workload.

**R7 — An operator win is not a system win. Only `applyback` may claim e2e.**
A kernel that is unambiguously faster in isolation routinely leaves the serving number
flat or slightly worse, and on agentic traffic that is the normal outcome, not the
exception. Kernels are therefore ranked by expected end-to-end gain —
`pct_gpu_time × (1 − 1/attainable_speedup)` — not by achievable speedup. Operator and
end-to-end numbers are written side by side so that neither can stand in for the other.

**R8 — No speedup counts until it survives a cold rerun.**
Every keep is re-measured from a fresh process and scored on the **minimum** of N runs.
Hot measurement is not enough: the unit-test leg allocates from a virgin allocator, so an
output that is never written reads back as zeros and compares equal — a kernel can pass
every correctness check while hot and fail from cold. The rerun is the task's own unit
test; the slot reads its verdict and does not re-derive one. A wrapper that re-implements
the judgement eventually disagrees with the test it wraps, and then wins the disagreement.

**R9 — Refuse to publish rather than publish something unfounded.**
Every stage separates `blocked` (a missing precondition — no GPU, no benchmark client, no
receipt) from `fail` (attempted, did not succeed), and says what was missing. `applyback`
declines to report an A/B at all if the overlay did not demonstrably shadow its targets,
or if the two sides used different metric bases: overlay injection fails open, so "no
gain" is otherwise indistinguishable from "the patch never loaded".

**R10 — A run must be able to say what it optimizes.**
Every scenario config declares a `target` block — model, image, component commits. The
config filename and `run_root` must carry a recognisable token of the declared model,
checked before a GPU hour is spent, and the declaration is cross-checked at every
`validate` against what the stages actually observed. A campaign once ran end to end under
the name `agentx_dsv4` against a GLM-5.2-MXFP4 stack: every number was right and every
label was wrong, which is the harder error, because a wrong number invites doubt and a
wrong name invites confidence.
*Enforced by:* `scripts/target.py`.

**R11 — Thresholds are calibrated, not assumed.**
The keep threshold, the attempt cap, the noise band, the re-verification repeat count:
each is a knob in `config/default.json` with the measurement that produced it recorded
next to it. The noise band is `1.045` because four kernels whose source md5 was identical
to the original re-measured as high as `1.0453` — not because a document said `1.02`.
Adding a new threshold means adding the measurement, or saying plainly that it is a guess.

### Robustness

**R12 — Unattended means repairing and continuing, not stopping.**
The pipeline is expected to run for days without a human. A node that dies, a container
that will not start, a stalled attempt: each costs the current attempt and nothing more.
Credit and adoption state live on disk, so a restart resumes rather than re-dispatches,
and the same kernel is never handed to two workers. A supervisor restart must not be able
to reset a counter that a deliverable is read from.

**R13 — Never delete measurement data.**
Run artifacts, receipts, traces and campaign evidence are inputs to conclusions that have
not been drawn yet. Move them, archive them, gitignore them — `runs/` and `campaigns/`
already are — but do not remove them to reclaim space.

### Hygiene

**R14 — Repo prose is English.**
Docs, comments and commit messages. Comments explain *why*, next to the code the reason
applies to; a comment restating what the line does is noise.

**R15 — The self-check is CPU-only and targets silent fakery.**
`python3 -m pytest workload/tests -q` must pass with no GPU, no tokens and no network, and
it covers the places where a mistake silently produces a plausible wrong number — config
merging and `--set`, `result.json` validation, the ranking formula, scheduler dedup and
blocked branches, applyback's refuse-to-publish gates, vendored-skill integrity. New
machinery in any of those places arrives with a test.

## 3. Non-goals

- A general workflow engine. Five named slots, swappable implementations, nothing more.
- A UI, a service, or a database. Stages communicate only through files on disk.
- Chasing microbenchmark numbers. A faster kernel that does not move the serving number is
  a negative result, and gets reported as one.
- Supporting every model and backend in code. New scenarios are config, not branches.

## 4. Before you commit

- [ ] No absolute path outside `config/`; no credential anywhere.
- [ ] `python3 -m pytest workload/tests -q` passes.
- [ ] `python3 workload/scripts/skill_manifest.py verify-all` passes.
- [ ] `python3 workload/scripts/stage.py --config <scenario> --dry-run` still resolves.
- [ ] A new threshold carries its measurement; a new stage separates `blocked` from `fail`.
- [ ] If a rule above got in the way, this file changed too — with the reason.
