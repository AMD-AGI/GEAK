# GEAK v4 documentation

GEAK v4 is a multi-agent GPU optimizer for **AMD Instinct MI GPUs** (CDNA, gfx942 / gfx950), driven by
**Claude Code** and orchestrated by deterministic JS **Workflows**:

- **`e2e_workflow`** ⭐ — end-to-end **sglang / vLLM serving throughput** of a whole LLM.
- **`kernel_workflow`** — optimize or author a single AMD GPU kernel (Triton, HIP, CK, FlyDSL).

The main overview is `README.md` at the repository root.

## In this folder

- **[Quick start](quick_start.md)** — launch Claude Code and run your first workflow.
- **[Installation](installation.md)** — prerequisites, setup, verification.
- **[API reference](api_reference.md)** — the `Workflow` tool, arg surfaces, helper scripts, and the
  external-orchestrator contract.
- **[Compatibility matrix](compatibility.md)** — verified hardware/software/backend combinations.

## Related in-repo references

- [`perf_knowledge/`](../perf_knowledge/) — AMD operator × backend knowledge base (reference only).
- [`e2e_workflow/README.md`](../e2e_workflow/README.md) · [`kernel_workflow/README.md`](../kernel_workflow/README.md) · [`interface/run_e2e.md`](../interface/run_e2e.md)

Preview locally (optional): `pip install mkdocs-material && mkdocs serve`.
