# AVO Knowledge Base (K / RAG) & Profiling — Design

Alignment + implementation for two paper requirements in `avo-workflow`: the
**knowledge base `K`** (the RAG analogue, §3.1) and **profiling-guided
optimization** (§3.2 / §4.4). Grounded in the paper (arXiv:2603.24517) and the
avo-workflow code.

> Companion: [`avo_memory_design.md`](avo_memory_design.md) (incl. how prior
> implementations are surfaced), [`avo_supervisor_design.md`](avo_supervisor_design.md).

---

## Part A — Knowledge base `K` (RAG)

### What the paper assumes
> §3.1: "`K` is a domain-specific knowledge base [containing] **CUDA programming
> guides, PTX ISA documentation, Blackwell architecture specifications, and
> existing kernel implementations including FlashAttention-4 source code**."
> §4.1: the agent has **documentation retrieval**.

So `K` = curated docs + reference kernels the operator consults on demand. The
paper does not require a vector DB; it requires the agent to be able to *consult
deep domain knowledge*.

### avo-workflow realization
`K` is wired to **`kernel_knowledge/`** — PerfSkills' curated AMD MI300X knowledge
base (the MI300X analogue of the paper's CUDA/PTX/Blackwell `K`):

- `kernel_knowledge/operators/<op>/…` — per-op overviews, numerics, tuning, and
  **per-backend reference notes** (triton / hip / ck / aiter / flydsl / asm) — the
  "existing kernel implementations" analogue.
- `kernel_knowledge/optimization/`, `profiling/`, `reference/` — occupancy &
  registers, rocprof counters, ISA / ROCm ecosystem.

Wiring (deterministic JS → roles):
- `avo_workflow.js` resolves `KERNEL_KNOWLEDGE_DIR` (arg `kernel_knowledge_dir`,
  default sibling `kernel_knowledge/`) and passes it to the **variation_engineer**
  (`§3 Knowledge base K`) and the **supervisor** (propose only directions `K`
  supports for this op + bottleneck).
- Plus the static `avo-workflow/knowledge/` (playbook, contract) and the reused
  `workflows/knowledge/` (per-class optimization menus).

**Rule:** `K` is *reference only, never a decision* — every choice is still
verified by measuring (the commit gate).

### Alignment status
✅ Aligned in substance (deep curated `K` consulted on demand).
**Deviation / future:** retrieval is *static file reading*, not a retrieval tool
(no `query`/`optimize` MCP like GEAK-AVO's RAG). For a large `K`, a retrieval tool
would scale better — tracked as future work. The paper's `K` is likewise "consult
docs", so static `K` is faithful; the tool is an efficiency upgrade.

---

## Part B — Profiling

### What the paper assumes
> §3.2: "early steps may focus on **structural changes** informed by reference
> implementations in `K`, while later steps can shift toward **micro-architectural
> tuning guided by profiling feedback** … and patterns observed across the
> accumulated lineage."
> §4.4 / §5: the headline optimizations (register spilling, branchless rescaling,
> pipeline overlap) were **discovered via profiling**, and the **bottleneck moves**
> as the kernel evolves.

So: (1) delay profiler-driven micro-tuning until structure is sound; (2) use
profiling to find the *current* bottleneck, which shifts across versions.

### avo-workflow realization
1. **Baseline profile** (Profile phase): the reused `profile_engineer` classifies
   the baseline bottleneck into `profileSummary`.
2. **Delayed / staged profiling** (`currentStage()` in `avo_workflow.js`):
   - `structural` for the first `profiling_after_step` steps (default 3) **or until
     the first commit lands** — the variation_engineer establishes structure and
     does **not** lean on the profiler (avoids premature local optima, CuTeGen).
   - `profiling-guided` afterward — the variation_engineer uses `PROFILE_SUMMARY`
     (and, if `use_profiling`, the profiler) for micro-arch tuning.
   - Forced `structural` when `use_profiling=false`.
   - `STAGE` is passed into every variation step (single + team + escalate) and
     honored by `variation_engineer.md §4`.
3. **Re-profile after every commit** (`reprofileBest()`): when a version commits,
   the new best (HEAD of `avo_repo`) is re-profiled and `profileSummary` is
   updated — so later steps target the **current** bottleneck, matching §4.4's
   "bottleneck moves" observation. (Mirrors team_workflow re-profiling on
   improvement.)

Knobs: `profiling_after_step` (stage switch), `use_profiling` (master off →
keeps the verified FULL_BENCHMARK for the gate but drops profiling).

### Alignment status
✅ Aligned: structural-first → profiling-guided staging + per-commit re-profile +
profiler-guided micro-tuning in the later stage.
**Deviation / future:** the depth of profiler data depends on the reused
`profile_engineer` + `profile_kernel.sh` (rocprof-compute on MI300X) — counter
coverage is a harness concern, not an AVO-loop concern. Per-shape bottleneck
attribution (which shape is limited by what) is a possible future enrichment.

---

## Summary

| Requirement | Paper | avo-workflow | Status |
|-------------|-------|--------------|--------|
| Examine multiple prior implementations (§3.2) | agent inspects `P_t` | `avo_lineage.py context` injects top-k prior diffs + per-shape; agent also `git show` | ✅ |
| Knowledge base `K` (§3.1) | CUDA/PTX/arch docs + ref kernels | `kernel_knowledge/` wired to variation + supervisor | ✅ (static; retrieval tool = future) |
| Delayed, profiling-guided tuning (§3.2) | structural-first → profiler-guided | `currentStage()` + `STAGE` honored | ✅ |
| Bottleneck moves (§4.4) | re-find bottleneck as kernel evolves | `reprofileBest()` after each commit | ✅ |
