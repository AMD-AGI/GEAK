# Contributing to GEAK

Thank you for your interest in contributing to GEAK! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository and clone your fork locally.
2. Follow the installation instructions in the [README](README.md).
3. Create a new branch from `main` for your changes.

## Development Workflow

1. **Create a branch** — use a descriptive name (e.g., `fix/kernel-race-condition`, `feat/triton-autotuner`).
2. **Make your changes** — keep commits focused and atomic.
3. **Test your changes** — ensure existing tests pass and add new tests where appropriate.
4. **Submit a pull request** — target the `main` branch.

## Pull Request Process

- PRs require **2 approving reviews** before merging.
- Link any related issues (e.g., `Fixes #123`).
- Keep PRs focused — one logical change per PR.

## Code Style

- Follow the existing code style and conventions in the repository.
- Use meaningful variable and function names.
- Keep functions focused and reasonably sized.

## Reporting Issues

- Search existing issues before creating a new one.
- When opening an issue, include enough detail to reproduce the problem (environment, steps, logs).

## License

All contributions must be compatible with the project's [Apache-2.0 license](https://github.com/AMD-AGI/GEAK/blob/main/LICENSE.md). By opening a pull request, you agree that your contribution is licensed under the same terms.

Every new source file should include the SPDX header:

```python
# Copyright (c) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
