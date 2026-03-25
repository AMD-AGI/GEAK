# Development Guidelines

This document describes best practices for contributing to GEAK. It covers the branching model, pull request workflow, release process, code quality standards, and CI expectations. Following these guidelines keeps the codebase healthy and makes reviews faster.

---

## Branch Strategy

We use a **dev + main** model with short-lived feature branches.

```text
feature/xxx ──► dev (daily development) ──► release/vX.Y ──► main (stable releases)
fix/xxx     ──┘                              hotfix ──────────┘
```

| Branch | Purpose | Who merges |
|--------|---------|------------|
| `main` | Always points to the **latest release** commit. Protected — no direct pushes allowed. | Maintainers only (via release PR) |
| `dev` | **active development branch.** All features and fixes are merged here. | Maintainers via PR review |
| `release/vX.Y` | Cut from `dev` when preparing a release. Only bug fixes and release prep (version bump, changelog) go here. Once ready, merged into both `main` and back into `dev`. | Maintainers |
| `feature/<topic>` | New features, enhancements. Branch from `dev`, merge back to `dev` via PR. | Any contributor |
| `fix/<topic>` | Bug fixes. Branch from `dev`, merge back to `dev` via PR. | Any contributor |
| `hotfix/<topic>` | Critical fixes. If branched from `main`, merge into both `main` and `dev`. If branched from `release/*`, merge back into that release branch only. | Maintainers only |
| `docs/<topic>` | Documentation-only changes. Branch from `dev`, merge back to `dev`. | Any contributor |

### Rules

- **Never push directly to `main` or `dev`.** All changes go through pull requests.
- `main` is **release-only** — it only receives merges from `release/vX.Y` or `hotfix/*` branches.
- Day-to-day development targets **`dev`**.
- Keep feature branches **short-lived** (< 2 weeks). Rebase onto `dev` frequently to avoid painful merge conflicts.
- Delete your branch after the PR is merged.

---

## Pull Request Workflow

### 1. Before you start

- Check existing issues and PRs to avoid duplicate work.
- For large changes, **open an issue first** to discuss the design.

### 2. Create your branch

```bash
git checkout dev && git pull origin dev
git checkout -b feature/my-new-feature
```

### 3. Develop

- Write code following the [Code Standards](#code-standards) below.
- Add or update tests for any behavioral change.
- Run the linter and tests locally before pushing (see [Local Checks](#local-checks)).

### 4. Commit messages

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

### 5. Open a pull request

- Target branch: **`dev`** (unless it's a hotfix).
- Fill in the PR template:
  - **Summary** — what and why (1–3 bullets).
  - **Test plan** — how you verified correctness.
  - **Related issues** — link to issues.
- Add **exactly one** type label to describe the PR's primary intent (see [Labels](#labels) below). A PR should be focused — if you find yourself needing two type labels, split it into separate PRs.

### 6. Draft PRs

Use GitHub **Draft PRs** when your work is not yet ready for formal review:

- **When to use**: You want early feedback on an approach, need CI to run against your changes, or want to signal to the team that you're working on something.
- **How to create**: Click "Create pull request" ▸ select **"Create draft pull request"** from the dropdown.
- **Behavior**: Draft PRs cannot be merged. Reviewers can leave comments but the PR won't enter the formal review queue.
- **When ready**: Click **"Ready for review"** to convert it to a regular PR and notify reviewers.

> **Tip**: Opening a Draft PR early is encouraged — it's better to get feedback on the direction before investing days of work.

### 7. Code review

- **At least 1 approval** from a maintainer is required to merge.
- Address review comments with new commits (do not force-push during review so reviewers can see incremental changes).
- Once approved, the **author** squash-merges via GitHub.

### 8. After merge

- Delete the feature branch.
- If the change needs a release note, add an entry to the changelog (see [Releases](#release-process)).


---

## CI/CD

### PR CI (triggered on every PR to `dev`)

Every PR automatically runs the following checks:

1. **Lint & format** — `ruff check` and `ruff format --check`.
2. **Correctness tests** — `pytest` runs the full test suite to verify functional correctness.
3. **Benchmark tests** (non-blocking) — only triggered when the PR carries the `feat` label **and** touches performance-sensitive code (kernel implementations, optimization passes, etc.). Compares benchmark results against the `dev` baseline and posts a summary in the PR comments. **Does not block merge** — maintainers review the benchmark results and decide whether to merge.

> A PR cannot be merged unless lint and correctness checks pass. Benchmark results are advisory — maintainers use them to make informed merge decisions.

### Scheduled CI (nightly / weekly)

A scheduled pipeline runs periodically on the `dev` branch to catch performance regressions early:

| Job | Frequency | Purpose |
|-----|-----------|---------|
| Correctness tests | Weekly | Catch flaky tests or regressions from merged PRs |
| Benchmark suite | Biweekly | Full benchmark run against the latest `dev`; results are tracked over time to detect gradual performance degradation |

If the scheduled benchmark detects a regression beyond a defined threshold, it automatically opens a GitHub Issue with the regression details for the team to investigate.

---

## Release Process

GEAK follows [Semantic Versioning](https://semver.org/): `vMAJOR.MINOR.PATCH`.

| Bump | When |
|------|------|
| **MAJOR** | Breaking API / config changes |
| **MINOR** | New features, backward-compatible |
| **PATCH** | Bug fixes, docs |

### Steps (maintainers only)

1. **Cut a release branch** from `dev`:
   ```bash
   git checkout -b release/v3.2 dev
   ```
2. **Bump the version** in `src/minisweagent/__init__.py`.
3. **Update CHANGELOG.md** — move "Unreleased" items under the new version heading.
4. Stabilize on the release branch — only bug fixes allowed, no new features.
5. **Tag on the release branch** once stabilization is complete:
   ```bash
   git checkout release/v3.2
   git tag -a v3.2.0 -m "Release v3.2.0"
   git push origin v3.2.0
   ```
6. **Merge into `main`** via PR: `release/v3.2 → main`. Get review + merge.
7. **Merge back into `dev`** via PR: `release/v3.2 → dev` (to bring in any release-branch fixes).
8. GitHub Actions (or manual) publishes the release artifacts.

### Hotfixes

Hotfixes always go through the corresponding **release branch**, keeping the tag workflow consistent with normal releases.

- **Latest release** (e.g., current release is `v3.2`):
  1. Branch `hotfix/<topic>` from `release/v3.2`, apply the fix.
  2. Merge hotfix back into `release/v3.2`.
  3. Tag on `release/v3.2` (e.g., `v3.2.1`).
  4. Merge `release/v3.2` into `main` and `dev`.

- **Older release** (e.g., `v3.1` needs a patch while latest is `v3.3`):
  1. Branch `hotfix/<topic>` from `release/v3.1`, apply the fix.
  2. Merge hotfix back into `release/v3.1`.
  3. Tag on `release/v3.1` (e.g., `v3.1.1`).
  4. No need to merge into `main` or `dev` — they are already ahead.

### Changelog format

```markdown
## [v3.2.0] — 2026-03-15

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
- Use **GitHub Milestones** to group issues by release (e.g., `v3.2`).
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
- Use environment variables (`AMD_LLM_API_KEY`, etc.) for secrets.
- If you accidentally commit a secret, rotate it immediately and notify the maintainers.
- The `.pre-commit-config.yaml` includes `detect-private-key` to catch common mistakes.
- **Customer IP**: Do not add customer-specific kernels, proprietary code, or results or benchmarks tied to them to this repository, or discuss them in issues, discussions, pull requests, or other project channels.
- **Confidential roadmap**: Do not mention non-public project plans, internal milestones, or internal codenames that are not yet publicly announced (e.g. `PRISM`) in the repository, issues, discussions, or elsewhere until those names or plans are officially public.

---

## License

All contributions must be compatible with the project's [MIT License](../LICENSE.md). By opening a pull request, you agree that your contribution is licensed under the same terms.

Every new source file should include the SPDX header:

```python
# Copyright (c) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

---

## Quick Reference

```text
 Branch from dev ──► Develop ──► Pre-commit + Tests ──► Open PR (→ dev) ──► Review ──► Squash-merge
                                                                               │
                                                                         Fix comments
                                                                         Push new commits

 Release flow:  dev ──► release/vX.Y ──► main (tag vX.Y.0) ──► merge back to dev
```

Thank you for contributing to GEAK!
