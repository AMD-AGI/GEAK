# GEAK documentation

GEAK is an AI-driven GPU kernel optimization framework. The main user-facing overview is **`README.md` at the repository root** (not duplicated here).

## In this folder

- **[Development guidelines](development_guidelines.md)** — branches, PR workflow, CI, coding standards.
- **[ROCm environment reference](env_install.md)** — ROCm layout and library source paths useful for kernel work.
- **[RAG filter sub-agent](subagent_guide.md)** — optional RAG filtering utilities in the codebase.

To build these pages locally (optional):

```bash
pip install mkdocs-material
mkdocs serve
```

**Published site:** after [.github/workflows/pages.yaml](https://github.com/AMD-AGI/GEAK/blob/main/.github/workflows/pages.yaml) runs, the site is at **https://amd-agi.github.io/GEAK/** .  
In the repository, set **Settings → Pages → Build and deployment → Source: GitHub Actions** (once per repo).
