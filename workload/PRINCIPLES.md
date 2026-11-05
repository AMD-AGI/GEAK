# Principles

This is a **skill**, not a product and not a framework: configured stages that an agent
executes, held together by as little machinery as possible. `README.md` says how it works.
This file says how it is designed and maintained, and is what a PR here is read against.

Profiling and UT generation may be separate stages (`profile` → `ut`) so completed
dispatch evidence can be reused without repeating GPU captures. Their boundary is a
versioned artifact contract; the original combined profile skill remains compatible.
If a rule is wrong, change *it* here, with the reason.

## Design

**R1 — Where prose will do, do not write code.**
A `SKILL.md`, a `README.md`, or a JSON field beats a module. Code is for what must be
checked mechanically; everything else is instructions to an agent.

**R2 — Modularity is contracts, not a framework.**
`stages/CONTRACT.md` and `runners/CONTRACT.md` are the whole mechanism. A new stage,
runner or skill is a directory named in a config. No registry, no DSL, no workflow engine,
no base classes.

**R3 — The glue stays thin.**
Each module does one thing and says so in one file; the layer that connects them only
passes paths and reads `result.json`. If the connector starts knowing about a specific
stage, that knowledge belongs in the stage.

**R4 — Nothing hardcoded.**
Model names, shapes, paths, thresholds, hosts, counts — all config. A change that only
works for the scenario in front of you is not finished; a new scenario must be a new JSON
file, never a new branch in code.

**R5 — The smallest change that states the rule.**
Add the general rule, not the special case. A fix that needs a chain of conditions is a
design mistake; fix the design instead. Prefer deleting over adding.

**R6 — The slot owns the criterion; the skill owns the work.**
What counts as passing belongs to the stage, not to the implementation plugged into it.
Swapping an implementation must not require rewriting the criterion — and the criterion
includes whether the configuration actually took effect.

**R7 — Vendored skills stay byte-identical.**
Fix their defects in the slot around them, or upstream. *Enforced by:*
`scripts/skill_manifest.py verify-all`.

## Honesty

**R8 — A run states what it was configured to do, and whether it did it.**
Config is policy, not suggestion. `blocked` (precondition missing) is never written as
`fail`, and neither is ever written as `ok`.

**R9 — No secrets and no developer-machine paths in the tree.**
Credentials live outside the repo, reached through `runner_env.env_file`. Paths resolve
from the repo root or from config.

**R10 — Never delete measurement data.**
Move it, archive it, gitignore it.

**R11 — English prose; checks that cost nothing to run.**
`scripts/skill_manifest.py verify-all` (vendored trees untouched) and
`stage.py --config <scenario> --dry-run` (the config resolves) are the self-check: no GPU,
no token, no network. The `smoke_stub` scenario drives the whole chain the same way when
given a `STUB_SCRIPT`. A check that needs a cluster is not a check anybody runs.

## Non-goals

A workflow engine. A UI, a service, a database — stages talk through files. Code support
for every model and backend. Chasing microbenchmarks that do not move the serving number.

## Before you open a PR

- [ ] The diff is as small as the rule allows, and adds no special case.
- [ ] Nothing hardcoded; no secret; no absolute path outside `config/`.
- [ ] `skill_manifest.py verify-all` passes and `stage.py --config <scenario> --dry-run`
      still resolves.
- [ ] If a rule above got in the way, this file changed too — with the reason.
