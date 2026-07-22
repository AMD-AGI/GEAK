# Contributing to GEAK

This document describes best practices for contributing to GEAK. It covers the branching model, pull request workflow, release process, code quality standards, and CI expectations. Following these guidelines keeps the codebase healthy and makes reviews faster.

---

## Branch Strategy

We use a **main + release** model with short-lived feature branches.

```text
feat/xxx ──► main (active development) ──► release/vX.Y (release stabilization) ──► tag vX.Y.0
fix/xxx  ──┘                                   hotfix ────────────────────────────┘
```

| Branch | Purpose | Who merges |
|--------|---------|------------|
| `main` | Protected default branch for day-to-day development. All reviewed features, fixes, and docs changes land here via PR. | Maintainers via PR review |
| `release/vX.Y` | Cut from `main` when preparing a release. Only bug fixes and release prep (version bump, changelog) go here. Once ready, tag the release and merge any stabilization fixes back into `main`. | Maintainers |
| `feat/<topic>` | New features, enhancements. Branch from `main`, merge back to `main` via PR. | Any contributor |
| `fix/<topic>` | Bug fixes. Branch from `main`, merge back to `main` via PR. | Any contributor |
| `hotfix/<topic>` | Critical fixes. Branch from the active `release/*` branch when patching a release; otherwise branch from `main`. Merge back into the source branch, then propagate to `main` if needed. | Maintainers only |
| `docs/<topic>` | Documentation-only changes. Branch from `main`, merge back to `main`. | Any contributor |
| `ci/<topic>` | CI/CD pipeline and harness changes (`ci/`, `.github/workflows/`). Branch from `main`, merge back via PR. Self-tested on the self-hosted L1 runner before merge. | Maintainers only |

### Rules

- **Never push directly to `main`.** All changes go through pull requests.
- Day-to-day development targets **`main`**.
- `release/vX.Y` branches are cut from `main` for release stabilization only.
- Keep feature branches **short-lived** (< 2 weeks). Rebase onto `main` frequently to avoid painful merge conflicts.
- Delete your branch after the PR is merged.

---

## Pull Request Workflow

### 1. Before you start

- Check existing issues and PRs to avoid duplicate work.
- For any non-documentation changes, please open an issue first to describe the current problem. For larger changes, we should discuss the design before implementation.

### 2. Fork and clone

External contributors do **not** have push access to `AMD-AGI/GEAK`. Fork the repository first, then work on your fork:

```bash
# 1. Fork on GitHub: click "Fork" on https://github.com/AMD-AGI/GEAK
# 2. Clone your fork locally
git clone https://github.com/<your-username>/GEAK.git
cd GEAK

# 3. Add the upstream remote so you can sync later
git remote add upstream https://github.com/AMD-AGI/GEAK.git
```

> **Maintainers** with write access can skip forking and push branches directly to `AMD-AGI/GEAK`.

### 3. Create your branch

Keep your `main` branch in sync with upstream before branching:

```bash
git fetch upstream
git checkout main && git merge upstream/main
git checkout -b feat/my-new-feature
```

### 4. Develop

- Write code following the [Code Standards](#code-standards) below.
- Add or update tests for any behavioral change.
- Run the linter and tests locally before pushing (see [Local Checks](#local-checks)).

### 5. Commit messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <short summary>

<optional body>
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`

Examples:

```
feat(profiler): add rocprofiler-compute v2 support
fix(discovery): handle missing CMakeLists in kernel repos
docs(readme): update parallel optimization examples
refactor(tools): extract common harness validation logic
```

- Keep the subject line under **72 characters**.
- Use imperative mood ("add", not "added" or "adds").
- Reference issue numbers where applicable: `Fixes #123`.

### 6. Push and open a pull request

Push your branch to **your fork**, then open a PR against the upstream repository:

```bash
git push origin feat/my-new-feature
# Then on GitHub: open a Pull Request from
#   <your-username>/GEAK:feat/my-new-feature  →  AMD-AGI/GEAK:main
```

- Target branch: **`main`** on `AMD-AGI/GEAK` (unless it's a hotfix to an active release branch).
- Fill in the PR template:
  - **Summary** — what and why (1–3 bullets).
  - **Test plan** — how you verified correctness.
  - **Related issues** — link to issues.
- Add **exactly one** type label to describe the PR's primary intent (see [Labels](#labels) below). A PR should be focused — if you find yourself needing two type labels, split it into separate PRs.

### 7. Draft PRs

Use GitHub **Draft PRs** when your work is not yet ready for formal review:

- **When to use**: You want early feedback on an approach, need CI to run against your changes, or want to signal to the team that you're working on something.
- **How to create**: Click "Create pull request" ▸ select **"Create draft pull request"** from the dropdown.
- **Behavior**: Draft PRs cannot be merged. Reviewers can leave comments but the PR won't enter the formal review queue.
- **When ready**: Click **"Ready for review"** to convert it to a regular PR and notify reviewers.

> **Tip**: Opening a Draft PR early is encouraged — it's better to get feedback on the direction before investing days of work.

### 8. Code review

- **At least 2 approval** from a maintainer is required to merge.
- Address review comments with new commits (do not force-push during review so reviewers can see incremental changes).
- Once approved, the **author** squash-merges via GitHub.

### 9. After merge

- Delete the feature branch.
- If the change needs a release note, add an entry to the changelog (see [Releases](#release-process)).

## Code standards

- Follow existing patterns in the package (`geak/`, `interface/`, `e2e_workflow/`) — naming, typing, error handling.
- Run **Ruff** before pushing; fix new lint issues in touched files.
- Prefer small, reviewable PRs; avoid drive-by refactors outside the stated goal.

## Local checks

Approximate what L0 CI runs (GPU-free) locally before you push:

```bash
# Lint (advisory today; keep touched files clean)
ruff check .

# The stdlib-only unit tests L0 runs (no GPU / no claude_agent_sdk)
python -m pytest -q \
  interface/test_run_e2e_recovery.py \
  e2e_workflow/scripts/tests/test_workload_alignment.py

# Node regression + handoff->args mapping (also run by L0)
node e2e_workflow/scripts/test_expert_skills_off_identical.js
python interface/run_e2e.py ci/fixtures/handoff.dry.json /tmp/l0_result.json --dry-run
```

The GPU end-to-end suite (L1) runs on the self-hosted SPUR runner, not locally.
See [CI/CD](#cicd) for the full picture, and [`ci/README.md`](ci/README.md).

---

## CI/CD

GEAK CI is split into two tiers by cost. **L0** is cheap and gates everything;
**L1** is the expensive GPU end-to-end and is human-gated. Full details live in
[`ci/README.md`](ci/README.md).

### L0 — checks (every push / PR to `main`)

Runs on GitHub-hosted runners — no GPU, no dataset, no secrets
(`.github/workflows/ci-l0-checks.yml`):

1. **Lint** — `ruff check .` (advisory today; surfaces findings without blocking while the tree is cleaned up).
2. **Python unit tests** — stdlib-only tests (`interface/test_run_e2e_recovery.py`, `e2e_workflow/scripts/tests/test_workload_alignment.py`); no GPU / no `claude_agent_sdk`.
3. **Node regression** — proves the `use_expert_skills=OFF` path injects nothing (byte-identical).
4. **Dry-run mapping** — validates the `handoff.json → e2e_workflow.js` arg mapping against `ci/fixtures/handoff.dry.json`.

### L1 — SPUR e2e (label-gated + manual dispatch)

The real GEAK e2e on the SPUR (SLURM) GPU cluster, driven from the self-hosted
runner (`.github/workflows/ci-l1-smoke-e2e.yml`). Because it burns exclusive GPU
time, a **human applying a label is the cost gate**:

| Trigger | Tier | Scope |
|---------|------|-------|
| `l1-ci-smoke` label | **smoke** | one SPUR job for the smoke-tier model |
| `l1-ci-full` label | **verify** | one SPUR job per enrolled model, matrix waited-on + aggregated (red if any model fails) |
| `workflow_dispatch` (Run workflow) | `probe` / `smoke` / `verify` | same tiers on demand; `probe` is an infra-only harness check (real SPUR/docker/GPU/weights, stops before the e2e) |

> A PR cannot merge unless L0 passes. L1 is run deliberately (label/dispatch) by a
> maintainer, not on every PR. To enroll a model or add an image for L1, see
> [`ci/ONBOARDING.md`](ci/ONBOARDING.md).

---

## Release Process

GEAK follows [Semantic Versioning](https://semver.org/): `vMAJOR.MINOR.PATCH`.

| Bump | When |
|------|------|
| **MAJOR** | Breaking API / config changes |
| **MINOR** | New features, backward-compatible |
| **PATCH** | Bug fixes, docs |

> **Current release line: `v4`** (the `v4` line lives on `main`, the default branch).

### Steps (maintainers only)

1. **Cut a release branch** from `main`:
   ```bash
   git checkout -b release/v4.1 main
   ```
2. **Bump the version** in `pyproject.toml` and `geak/__init__.py` (keep them in sync).
3. **Update CHANGELOG.md** — move "Unreleased" items under the new version heading.
4. Stabilize on the release branch — only bug fixes allowed, no new features.
5. **Tag on the release branch** once stabilization is complete:
   ```bash
   git checkout release/v4.1
   git tag -a v4.1.0 -m "Release v4.1.0"
   git push origin v4.1.0
   ```
6. **Merge back into `main`** via PR: `release/v4.1 → main` (to bring in any release-branch fixes). Get review + merge.

### Hotfixes

Hotfixes always go through the corresponding **release branch**, keeping the tag workflow consistent with normal releases.

- **Latest release** (e.g., current release is `v4.0`):
  1. Branch `hotfix/<topic>` from `release/v4.0`, apply the fix.
  2. Merge hotfix back into `release/v4.0`.
  3. Tag on `release/v4.0` (e.g., `v4.0.1`).
  4. Merge `release/v4.0` back into `main`.

- **Older release** (e.g., `v3.9` needs a patch while latest is `v4.0`):
  1. Branch `hotfix/<topic>` from `release/v3.9`, apply the fix.
  2. Merge hotfix back into `release/v3.9`.
  3. Tag on `release/v3.9` (e.g., `v3.9.1`).
  4. Merge back into `main` only if the fix should also apply to ongoing development.

### Changelog format

```markdown
## [v4.0.0] — 2026-07-15

### Added
- Profiler: rocprofiler-compute v2 integration (#142)

### Fixed
- Discovery: crash on repos without CMakeLists (#138)

### Changed
- Config: `geak.yaml` now accepts `tools.profiling_type` (#145)
```

---

## Issue & Project Management

- Use **GitHub Issues** for bugs, feature requests, and tasks.
- Use **GitHub Milestones** to group issues by release (e.g., `v4.0`).
- Use labels consistently. There are two categories: **type labels** (exactly one per PR / issue) and **meta labels** (zero or more).

### Labels

#### Type labels (mutually exclusive — pick one)

Every PR and issue must carry exactly one type label. This keeps each PR focused on a single purpose and simplifies changelog generation.

| Label | When to use |
|-------|-------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `refactor` | Code restructuring with no behavior change |
| `docs` | Documentation only |
| `defaults` | Changes to built-in default values (hyperparameters, thresholds, etc)|
| `test` | Adding or updating tests only |
| `ci` | CI/CD pipeline changes |
| `chore` | Build, tooling, or dependency updates |

> **Rule of thumb**: If a PR would need two type labels, it should be split into two PRs. For example, a bug-fix PR should not also introduce a new feature — submit them separately so each can be reviewed, reverted, and release-noted independently.

---

## Security & Secrets

- **Never commit** API keys, tokens, or credentials.
- Use environment variables for secrets.
- If you accidentally commit a secret, rotate it immediately and notify the maintainers.
- The `.pre-commit-config.yaml` includes `detect-private-key` to catch common mistakes.
- **Customer IP**: Do not add customer-specific kernels, proprietary code, or results or benchmarks tied to them to this repository, or discuss them in issues, discussions, pull requests, or other project channels.
- **Confidential roadmap**: Do not mention non-public project plans, internal milestones, or internal codenames that are not yet publicly announced (e.g. `PRISM`) in the repository, issues, discussions, or elsewhere until those names or plans are officially public.

---

## License

All contributions must be compatible with the project's [Apache-2.0 license](https://github.com/AMD-AGI/GEAK/blob/main/LICENSE.md). By opening a pull request, you agree that your contribution is licensed under the same terms.

Every new source file should include the SPDX header:

```python
# Copyright (c) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
```

---

## Contributing to a specific part

The PR/branch/review rules above apply everywhere, but several subsystems have
their **own** local contribution contract (schema, validation gate, generator, or
onboarding steps). Read the relevant guide **before** opening a PR that touches
that area — each has rules that CI or a maintainer will hold you to.

| Area | What you're contributing | Start here |
|------|--------------------------|------------|
| **Perf knowledge base** (`perf_knowledge/`) | SOTA operator × backend cards, hardware/language/backend notes | [`perf_knowledge/README.md`](perf_knowledge/README.md) → conventions in [`perf_knowledge/index/conventions.md`](perf_knowledge/index/conventions.md), sourcing in [`perf_knowledge/index/sourcing_rules.md`](perf_knowledge/index/sourcing_rules.md). Every content file ends with `## Sources`; the matrix/registry are **generated** — after editing cards run `python3 index/_gen_registry.py`. |
| **Expert skills** (`perf_knowledge/expert_skills/`) | A human-authored, e2e-validated optimization recipe | [`perf_knowledge/expert_skills/_contribute/SKILL.md`](perf_knowledge/expert_skills/_contribute/SKILL.md) — scaffold → fill → **validate (efficacy + do-no-harm gate)** → `make_pr.sh`. Only `validated` skills auto-apply; branch is `expert-skill/<slug>`. |
| **Learned experience cards** (`e2e_workflow/knowledge/learned/`) | Distilled, advisory priors from a run | [`e2e_workflow/knowledge/learned/README.md`](e2e_workflow/knowledge/learned/README.md) — curate, never blind-append; a card must have an `INDEX.md` line and a `source:`; budget ≤ 40 cards. |
| **e2e workflow** (`e2e_workflow/`) | System-layer roles, knowledge, pipeline | [`e2e_workflow/README.md`](e2e_workflow/README.md) (+ `roles/`, `knowledge/`). Keep the L0 node regression (`use_expert_skills=OFF` byte-identical) green. |
| **Kernel workflow** (`kernel_workflow/`) | Single-kernel optimizer roles/knowledge | [`kernel_workflow/README.md`](kernel_workflow/README.md) (+ `roles/`, `knowledge/`). |
| **CI harness** (`ci/`, `.github/workflows/`) | Runner scripts, matrix, monitors | [`ci/README.md`](ci/README.md). Use a `ci/<topic>` branch (maintainers only) and self-test on the L1 runner. |
| **Enroll a model / Docker image for L1** | A new SPUR model or arch image | [`ci/ONBOARDING.md`](ci/ONBOARDING.md) — weights staging, handoff layout, `exp_root=geak`, TraceLens priors, `models.tsv`, `docker_default.json`. |
| **run_e2e contract / API** | The `handoff.json → e2e_workflow` interface | [`docs/reference/run-e2e-contract.md`](docs/reference/run-e2e-contract.md) · [`docs/reference/api-reference.md`](docs/reference/api-reference.md). The L0 dry-run guards this mapping. |

> If your change spans several of these, still keep the PR focused (one type label)
> — and follow the strictest local contract among the areas you touch.

---

## Quick Reference

```text
 Fork AMD-AGI/GEAK ──► Clone ──► Branch from main ──► Develop ──► Pre-commit + Tests
      ──► Push to fork ──► Open PR (fork → AMD-AGI/GEAK:main) ──► Review ──► Squash-merge

 Release flow:  main ──► release/vX.Y ──► tag vX.Y.0 ──► merge back to main
```

Thank you for contributing to GEAK!
