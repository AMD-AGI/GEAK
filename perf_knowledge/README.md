# perf_knowledge — AMD Instinct (MI-series) Kernel-Optimization Knowledge Base


> **Consumption contract (read this).** This base is **reference material** — it may be stale,
> incomplete, or wrong. It provides *facts and how-to* (APIs, code skeletons, knobs, pitfalls, which
> backends exist), **not decisions**. Consumers (kernel_workflow / e2e_workflow agents) must form their own
> judgment and **decide by on-box measurement** — never treat anything here (including `status`/TFLOPS,
> which are dated evidence) as a verdict. Machine consumers enumerate candidates from
> `index/capability_index.yaml` (no ranking) and then measure. Adding this base can only *seed/locate*
> candidates faster; it must never reduce an agent's result below its measured baseline.
A sourced, machine-queryable knowledge base for **writing, selecting, and tuning the fastest kernel**
for any operator on any AMD Instinct MI GPU — CDNA1 (MI100) · CDNA2 (MI200/250X) · CDNA3 (MI300A/X/325X)
· CDNA4 (MI350X/355X).

It has two jobs:
1. **Teach optimization** — hardware, programming languages, library backends, techniques, profiling.
2. **Be a SOTA registry** — for **every `operator × backend` cell**, point at the *best known
   implementation(s)* with source, measured performance, applicability, knobs, and pitfalls.

> **~586 docs · ~43,900 lines · 0 broken links · every content file ends with `## Sources`.**
> 14× the file count of the original `perf_knowledge`. Built P0→P4 + hot-path enrichment, 2026-06-08
> (see [`index/changelog.md`](index/changelog.md)).

---

## Quick start

| I want to… | Go to |
|---|---|
| **Pick the best backend for an operator** | [`index/sota_matrix.md`](index/sota_matrix.md) (human) · [`index/sota_registry.yaml`](index/sota_registry.yaml) (machine) → the operator's [`operators/<op>/backends/<backend>.md`](operators/) card |
| **Decide what to even try** | [`index/decision_trees.md`](index/decision_trees.md) |
| **Understand an operator** | [`operators/<op>/overview.md`](operators/) (+ `tuning` / `numerics` / `fusion`) |
| **Learn the hardware / a language / a library** | [`hardware/`](hardware/) · [`languages/`](languages/) · [`backends/`](backends/) |
| **Apply a cross-cutting technique** | [`optimization/`](optimization/) · [`quantization/`](quantization/) |
| **Profile / triage a kernel** | [`profiling/`](profiling/) |
| **Run an end-to-end optimization** | [`kernel_workflow/`](kernel_workflow/) |
| **See a real war story with numbers** | [`case_studies/`](case_studies/) |
| **Use / contribute a human expert recipe** | [`expert_skills/`](expert_skills/) — validated, advisory optimization skills the workflows consult when enabled (opt-in: `use_expert_skills=true`, default OFF); add one via [`expert_skills/_contribute/SKILL.md`](expert_skills/_contribute/SKILL.md) |
| **See what to borrow from the ecosystem** | [`landscape/`](landscape/) (multi-backend libs · DSLs · AI kernel agents · autotuning · AMD SOTA 2026 · serving registries) |
| **Look up an env var / pinned repo** | [`reference/env_vars.md`](reference/env_vars.md) · [`reference/repo_index.md`](reference/repo_index.md) |

---

## Layout

```
perf_knowledge/
├── index/         (10)  navigation + SOTA registry + taxonomy + sourcing rules + templates + generator
├── hardware/      (27)  CDNA1–4 deep dives + shared (matrix core, memory, numerics)
├── languages/     (42)  triton · flydsl · hip · ck · asm · tilelang · rocwmma · hipkittens · mojo · cutlass
├── backends/      (35)  aiter · hipblaslt · ck_lib · rocblas/tunableop · fa_rocm · mori/rccl · miopen
│                        · pytorch_inductor · sglang · vllm
├── operators/    (420)  ★ THE CARTESIAN CORE — 54 operators ×
│                        {overview, tuning, numerics, fusion} + backends/<backend>.md (204 SOTA cards)
├── optimization/  (12)  occupancy · LDS/bank-conflicts · MFMA scheduling · pipelining · vectorization
│                        · XCD/L2 locality · grid sizing · autotuning · roofline · fusion · numerics
├── quantization/  (10)  formats · FNUZ-vs-OCP · MXFP block-scaling · scaling strategies · Quark
│                        · accuracy gates · HW support matrix · KV quant · deployment recipes
├── profiling/     (10)  rocprofv3 / rocprof-compute / rocprof-sys · counters · bottleneck triage
│                        · roofline · traces · benchmarking · engagement verification · pitfalls
├── kernel_workflow/      (9)  single-kernel ladder · e2e model flow · GEMM tuning recipe · backend selection
│                        · kernel integration · GEAK authoring · backend choice · bring-up checklist
├── case_studies/   (8)  by_model (Qwen3.5-27B, DeepSeek-MLA, Llama-fp8) + by_kernel
├── landscape/      (7)  ecosystem survey + "what to borrow" (multi-backend libs, DSLs, AI agents,
│                        autotuning, AMD SOTA 2026, serving registries)
└── reference/      (3)  repo_index · env_vars dictionary · benchmarks_and_agents
```

`★ operators/` is the heart: a Cartesian map of **54 operators × backends**. The most relevant SOTA
cells are linked in [`index/sota_matrix.md`](index/sota_matrix.md); the full backend set per operator
lives in each `operators/<op>/backends/`.

---

## How the SOTA registry works

[`index/sota_registry.yaml`](index/sota_registry.yaml) is the machine-readable mirror of the matrix.
A workflow (e.g. `e2e_workflow`'s architect / op_benchmarker) answers *"best backend for operator
X on gen Y, dtype Z, regime R?"* by filtering `entries[]` → `status: sota` → following the `card:` path.

**The registry and matrix are GENERATED, not hand-maintained.** The single source of truth is each
card's YAML frontmatter (`operator`, `backend`, `status`, `gens`, `dtypes`, `regimes`, `sources`).
After editing cards, regenerate both:

```bash
cd perf_knowledge && python3 index/_gen_registry.py
# -> rewrites index/sota_registry.yaml + index/sota_matrix.md from operators/*/backends/*.md frontmatter
```

Status badges: `🟢 sota` · `🟡 competitive` · `🧪 experimental` · `🟤 legacy` · `⚪ na` (with reason).
Backend card filenames are the canonical taxonomy ids (`ck`, `fa_rocm`, `mori`, `rccl`, …) so every
matrix badge links correctly.

---

## Non-negotiables (see [`index/sourcing_rules.md`](index/sourcing_rules.md))

- **Every content file ends with `## Sources`** citing primary sources — ROCm docs/blogs, CDNA ISA §,
  GitHub `repo@commit:path`, arXiv, vendor benchmarks.
- **Every performance number is measured** and tagged `value @ hardware, ROCm/lib version, date`, or
  explicitly labeled **vendor-reported**. No theoretical peak presented as achievable (MI300X commonly
  sustains ~45% of peak).
- **Cartesian completeness**: a missing `operator × backend` cell is written as `na` *with a reason*
  and the path to use instead — never left blank.
- Heavy on-box grounding: code excerpts are pulled from the installed `ROCm/aiter@a6bb4993` (and vLLM
  csrc, ROCm/mori), with file paths cited.

---

## Conventions & contributing

- Frontmatter schema, section order, perf format, and status badges: [`index/conventions.md`](index/conventions.md).
- Card / overview skeletons: [`index/_templates/`](index/_templates/).
- Controlled vocabulary (operator / backend / dtype / gen ids): [`index/taxonomy.md`](index/taxonomy.md).
- To add a backend card: copy `index/_templates/sota_card_template.md`, fill it (frontmatter + Sources
  required), drop it in `operators/<op>/backends/<backend>.md`, then re-run `index/_gen_registry.py`.

## Sources
- ROCm "AMD Instinct MI300X workload optimization": https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Matrix Cores on CDNA3/CDNA4: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- Performance reality (≈45% of peak on MI300X): https://arxiv.org/pdf/2510.27583
- Build provenance and per-phase log: [`index/changelog.md`](index/changelog.md).
