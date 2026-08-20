# kernel_workflow — Dynamic Workflow for Kernel / Model Inference Optimization

A deterministic **Workflow** (JS-orchestrated multi-agent pipeline) that optimizes the inference
speed of a GPU kernel directory — a single kernel, several kernels fused together, or an end-to-end
vLLM / SGLang model — on AMD Instinct MI-series accelerators (MI300X / MI300A / MI308X / MI325X on
CDNA3 gfx942, and MI350X / MI355X on CDNA4 gfx950 — the card is detected on-box, not assumed). The
budget loop, round fan-out, and verification are **JS control flow**, while every judgement call is made
by an agent returning **structured JSON**.

## Key properties
1. **Deterministic orchestration** — the budget loop / parallelism / verification live in
   `kernel_workflow.js`, not in LLM-interpreted prose. The TechLead returns structured decisions.
2. **Independent verification of every claimed speedup** — each engineer's patch is re-benchmarked
   by a separate `verify_engineer` in a clean workspace *as soon as it finishes* (pipelined). The
   script trusts only verified, absolute-latency numbers → the winner is genuinely the fastest.
3. **Specialist engineers (A)** — `algorithm | memory | compute | host_runtime`; each loads only its
   relevant knowledge → focused context, sharper results, naturally orthogonal & mergeable. Plus a
   fifth **`deep_explore`** track: an open-ended deep optimizer the TechLead hands a high target (Nx
   and/or ~90% roofline) with minimal steering — broad authority (kernel + wrapper + binding), its own
   long measure→self-profile→rewrite loop. It costs `deep_cost` budget (default 2) and always runs in a
   dedicated round on its own (its ground-up rewrite isn't expected to merge with specialist patches).
4. **Host/Runtime as a first-class track (B)** — attacks the wall-clock floor (dispatch collapse,
   native layouts, CUDA graph, wrapper overhead). This is where the last 1.5–3x of geomean lives.
5. **Cross-round memory (C)** — an insight blackboard + hypothesis ledger threads what was learned
   into the next round's engineer prompts; dead-ends are not retried.
6. **Integrator (E)** — combines the round's winning ideas (stack compatible patches OR hand-merge
   conflicting ones into a coherent best implementation). Does not consume budget.
7. **Director arbitration (H)** — independently validates the final patch against the TRUE original
   baseline and can flag / request a corrective round.

## Roles → workflow mapping
- **Director** = the script's orchestration + a setup agent + a final validation/arbitration agent.
- **TechLead** = agent for analyze/roadmap, per-round planning (orthogonal directions + stop), the
  cross-round memory, and the final report.
- **Engineers** = parallel specialist agents (optimize), plus `benchmark_engineer`, `profile_engineer`,
  `verify_engineer`, and `integrator`.

## Pipeline
`Setup → Analyze+Roadmap → Benchmark(COMMANDMENT+baseline) → Baseline Profile → [Research (opt-in)] →`
`LOOP[ Plan round → (Optimize ‖ Verify, pipelined) → Integrate → Commit winner → Re-profile → Update memory ] →`
`Final Report → Director Validation`.

Each round's winner is committed into the canonical workspace, so the next round builds on the
cumulative best. Speedup is always measured in **absolute latency vs the true baseline**:
`geomean( baseline_ms / optimized_ms )`.

## Budget
`budget` = the **total number of optimization directions** the TechLead may dispatch to engineers
across all rounds. Only optimization-direction engineers count; benchmark / profile / verify /
integrate / commit / validate do **not** consume budget. The script hard-caps each round to the
remaining budget; the TechLead may also stop early (`stop=true`) when further directions won't pay.
Example (budget=6): round 1 = 3 directions, round 2 = 3; or 4 then 2; or stop after 4.

## Invocation
This is a Workflow, run via the `Workflow` tool with `scriptPath` and `args`. **No paths are
hard-coded in the script** — it is portable to any install location. Set `scriptPath` to wherever
this folder lives and pass that same folder as `args.workflow_dir`:

> **IMPORTANT:** pass `args` as a real JSON **object** (a mapping), **not** as a JSON-encoded
> string. Do not wrap it in quotes or `json.dumps()` it. If `args` arrives as a string the
> workflow cannot read `args.workflow_dir` / `args.kernel_path` and aborts immediately.

```
Workflow({
  scriptPath: "<WF_DIR>/kernel_workflow.js",   // <WF_DIR> = absolute path to THIS kernel_workflow/ folder
  args: {
    kernel_path: "/abs/path/to/kernel_or_model_dir",  // REQUIRED
    workflow_dir: "<WF_DIR>",  // REQUIRED: same folder as scriptPath (holds roles/ knowledge/ scripts/);
                               //           a JS workflow can't read its own path, so the caller passes it
    budget: 6,                 // optional, default 6
    min_improve: 0.02,         // optional, default 0.02 (2%): min verified geomean gain over the
                               //           cumulative best for a round winner to be committed
    deep_cost: 2,              // optional, default 2: budget cost of one deep_explore direction
                               //           (heavyweight; always runs in its own dedicated round)
    gpu_ids: "0",              // optional, comma-separated, default "0"
    task: "focus on ...",      // optional natural-language steer
    exp_root: "",              // optional, output root; default = sibling "exp/" next to workflow_dir
    eval_dir: "",              // optional, override the output dir for this single run
    apply_to_original: "false",// optional; if "true", write the validated patch back to kernel_path
    // --- mode dispatch (kernel_workflow.js is the single ENTRY POINT / dispatcher) ---
    mode: "optimize",          // optional: "optimize" (default, edit an existing kernel) | "author"
                               //   | "bakeoff" (try several backend languages, keep the fastest)
    target_language: "triton", // author mode: triton (always) | flydsl | hip | ck — the language to write
    backends: ["hip","triton","flydsl"], // bakeoff: EXTRA rewrite languages to race (empty = auto-discover).
                               //   ADDITIVE only: the incumbent (input kernel's own) language ALWAYS
                               //   competes as an in-place optimize lane and cannot be dropped here.
    op_spec: {},               // author mode: {op_kind, shapes, dtype, math_contract, regime} for the op
    perf_knowledge_dir: "",  // optional: AMD authoring knowledge base the author_engineer reads
    // --- workload alignment (optional; aligns the PERF harness with the real workload) ---
    workload_spec_path: "",    // optional: path to a workload-v1 json (parse_profile.py --workload-out).
                               //   The benchmark harness then times the EXACT (shape,dtype) cases the
                               //   workload hits, weighted by each case's total-time contribution, and
                               //   the PRIMARY metric becomes the time-weighted ratio-of-sums (the
                               //   unweighted geomean is kept as a secondary diagnostic). Correctness is
                               //   unaffected (it stays on the frozen immutable oracle).
                               //   Also accepted as op_spec.workload_path, or op_spec.workload (inline).
    // --- Deep Research Agent (DRA) — opt-in web-grounded research phase before the optimize loop ---
    dra_enabled: "false",      // optional, default "false" (OFF → behavior byte-identical). "true" runs
                               //   the Research phase after Profile / before the optimize loop.
    dra_max_questions: 8,      // optional, default 8: max research questions fanned out in parallel
    dra_blindspot: "false",    // optional, default "false": run an extra blindspot-critique + 2nd
                               //   parallel research wave (Stage 5/6) — budget-permitting
    dra_max_blindspots: 4      // optional, default 4: cap on blindspots / 2nd-wave follow-ups
  }
})
```

### Workload alignment (NEW)
By default the harness benchmarks small/medium/large cases unweighted. Pass a **workload spec** to
instead benchmark the shapes/dtypes the kernel actually sees in production, weighted by how much
wall-clock each contributes (`weight = call_count × baseline_latency`). Generate one from a profiler
trace with `python3 e2e_workflow/scripts/parse_profile.py --torch-trace <trace> --workload-out
workload.json [--target <kernel_name>]`, then pass `workload_spec_path: ".../workload.json"`. The
optimization target becomes the **time-weighted ratio-of-sums**
`Σ count·baseline / Σ count·optimized` (true wall-clock speedup of the kernel's total workload
contribution); the unweighted geomean is still reported. The perf **baseline is the original/extracted
implementation**, never an LLM naive reimplementation. When invoked from the e2e layer this is wired
automatically (profiler → extractor → `op_spec.workload_path`).

### Author mode (NEW)
`mode="author"` is for when there is **no existing source to optimize** — a hot op (e.g. a library
GEMM/attention) needs a fresh implementation. Here `kernel_path` is an **op task dir** holding the
IMMUTABLE oracle (`meta.json` + `unittest.py` + frozen `baseline_src/`, plus a `reference_io.pt` only if
the dir came from e2e's `kernel_extractor` — that golden is ~1 GB and immutable, so every workspace
shares the single original via a read-only symlink rather than copying it). The `author_engineer`
writes the simplest correct implementation in `target_language` (correctness-judged against the
oracle), commits it as the baseline, and then the **same optimize loop** improves it. Returns
`authored:false` / `validation_status:"author_failed"` if no correct baseline can be produced (the
caller drops that language). `mode="optimize"` (default) is unchanged and fully backward compatible.

### Deep Research Agent / Research phase (NEW, opt-in)
`dra_enabled="true"` inserts a **`Research` phase AFTER Profile and BEFORE the optimize loop** (so the
COMMANDMENT + baseline profile + analysis already exist). It lives in the `kernel_lane.js` worker
alongside the rest of the pipeline, and the dispatcher forwards `dra_*` through unchanged. The
**`researcher`** persona (`roles/researcher.md`) runs a v4-native deep-research pass:
1. **Stage 0 + 1/2** (`research_plan`, one agent): extract facts from the kernel source +
   `profiling_summary.md` + `analysis.json` + the `COMMANDMENT`, then generate & rank research
   QUESTIONS spanning BOTH grounded bottleneck questions AND design-space / "is there a fundamentally
   faster algorithm or execution strategy?" questions.
2. **Stages 3/4** (`research_question`, fanned out in **parallel** — one agent per question): each
   researches its question on the live web via **native `WebSearch`/`WebFetch`** and synthesizes one
   judgment. Every research agent is wrapped in the `agentT()` hang-guard, so a hung research agent
   resolves to `null` and the parallel round-barrier still proceeds (it cannot wedge the run).
3. **Stage 5/6** (`research_blindspot`, optional, `dra_blindspot="true"`): a blindspot critique + a
   second parallel research wave on the follow-ups.
4. **Stage 7** (`research_synthesize`, one agent): a ranked **portfolio of optimization directions**,
   written as `deep_search.md` (full evidence), `deep_search_brief.md` (compact, ~2-4 KB ranked
   directions only — what the planner reads), and `deep_search.json` (structured).

The TechLead's `plan_round` then **Reads `EVAL_DIR/deep_search_brief.md` (if present)** and seeds
`directions[]` from the ranked DRA directions — **diversifying** them across parallel engineers (with
≥1 free explorer slot, never anchoring all engineers on one theme) and treating **high-ceiling
rewrites (raw-HIP/`load_inline`, HIP/CUDA graph capture, algorithmic reformulation) as first-class**,
not secondary. The brief is a prior, never a cage: profile/per-case data and measurement still rule.

**Web tools:** the research agents need `WebSearch`/`WebFetch`. They are on the e2e allowlist
(`interface/run_e2e.py` `ALLOWED_TOOLS`). For a standalone `claude -p` invocation of this workflow
with `dra_enabled`, pass them on the allowlist too (`--allowed-tools Workflow,Bash,Read,Write,WebSearch,WebFetch`).
With `dra_enabled` off (the default) nothing opts into the web tools and behavior is unchanged.

### Bake-off mode (NEW) — one kernel, many backend languages, keep the fastest
`kernel_workflow.js` is now the single **ENTRY POINT / dispatcher**; the single-language pipeline lives
in the sibling **`kernel_lane.js` worker**:

- `mode="optimize" | "author"` → the dispatcher **passes straight through** to one `kernel_lane` worker
  (byte-compatible with the old behavior; the worker is `kernel_lane.js`, unchanged).
- `mode="bakeoff"` → the dispatcher **is the bake-off orchestrator** (pass `backends`, or leave empty to auto-discover):
  1. **Freeze** (`roles/oracle_freezer.md`): freeze ONE immutable oracle — frozen `baseline_src/` +
     immutable `unittest.py` + a `meta.cases[]` shape/seed manifest — no server and **no recorded golden
     tensors** (the task dir stays a few MB, and every lane tar-copies it). The correctness truth source
     and the speedup denominator are BOTH the input kernel's own behavior, and both are the same artifact:
     `baseline_src/`, re-run live on every parity draw.
  2. **Discover**: reuse e2e's `op_benchmarker` role **in place and UNCHANGED** to probe per-language
     existing impls, decide the `author_plan`, AND run the per-backend GEMM/quant tune. The role's Tier-B
     tune is written for a live server; since a standalone run has none, the dispatcher's prompt redirects
     it to run **offline** — the tune shapes come from the frozen oracle's `meta.json` (real M/N/K/`bias`/
     dtype, so no server capture and no bias guessing), and engagement is verified in the **isolated**
     `unittest`/`op_bench` (`AITER_LOG_TUNED_CONFIG`, not `server.log`). Only pure server-flag levers
     (`--attention-backend` swap, serving-only playbook probes) are skipped. **The e2e `op_benchmarker.md`
     file is never edited** — all standalone behavior is carried by the dispatcher prompt.
  3. **Bake-off**: run one **unchanged `kernel_lane` worker per backend language** in parallel over the
     GPU pool (1 GPU/lane; `gpu_ids` with >1 id runs lanes concurrently, a single id serializes them).
     The **incumbent (input) language ALWAYS runs as an in-place `optimize` lane** — it is force-included
     even when `backends` lists only other languages, so a bake-off can always win by simply optimizing
     the original, not just by rewriting. `backends` is additive (extra rewrites), never a replacement.
  4. **Report**: rank **all candidates** on the **SAME frozen baseline** (the anti-cheating invariant)
     and pick the fastest — three candidate classes: the input-language *optimize* lane (always present),
     each *author* lane, and the *tuned env backend* (aiter/CK) from Discover. A candidate only wins if it
     actually beat the frozen baseline (speedup > 1.0x); if none did, `winner=null` and the ORIGINAL kernel
     is kept. Optional `apply_to_original` (a lane winner applies its patch; an env winner records its
     `apply_env` + tuning artifact).
  5. **UpdateExperience** (`roles/update_experience.md`): on a measured win only, the TechLead distills
     **at most one** reusable principle into `knowledge/learned/` (curate/merge, never blind-append; see
     `knowledge/learned/README.md`). In bake-off this runs **once, centrally** — the lanes are launched
     with `update_experience: 'off'`, and the lesson worth keeping is the cross-language routing outcome
     the dispatcher alone can see.

### Learned knowledge (every run, not just bake-off)
`kernel_lane.js` curates a card at the end of **every** lane run that earned a measured win — standalone,
dispatcher passthrough (`mode=optimize|author`), or a lane opened by `e2e_workflow`. The sink is always
`<this workflow>/knowledge/learned/`, derived from `workflow_dir`, so an e2e-driven lane writes its
kernel-level lesson **here**, never into `e2e_workflow/knowledge/learned/` (that sink is e2e-gated and
owned by e2e's own `system_architect`; it cites these cards instead of copying them). The cards are read
back as **advisory priors** by `tech_lead` and `author_engineer` — ADD-only, always overruled by on-box
measurement. Disable with `update_experience: 'off'`.

Each card is **self-describing**: it opens with a skill-style discovery header (`name`, `description`,
`keywords`, `kernels`, `platforms`, `kernel_class`, `regime`, `confidence`).
`knowledge/learned/INDEX.md` is a **generated** projection of those headers — rebuild it with
`python3 kernel_workflow/scripts/kb.py --kb-dir kernel_workflow/knowledge/learned index` (`--check` fails when it is stale). Nothing appends to the index by
hand, which is also why concurrent lanes can no longer drop each other's entries.

Retrieval is **semantic and done by the reading role**, not by a matcher: the index is ≤40 cards, each
line already carries the description + kernel symbols + keywords, so the role reads it and judges
relevance by meaning (a `split-k on skinny-M GEMM` card is worth opening for a tall-K GEMM). `grep` is a
shortcut for an exact kernel symbol, never the lookup path. Keyword drift (`split-k`/`split_k`/`splitk`)
is contained three ways: the reader is semantic so a synonym costs ranking not retrieval; the generator
normalizes spelling mechanically; and the index publishes a `## keyword vocabulary` appendix (every term
in use + counts) that curators pick from, with surviving near-duplicates flagged for a human call rather
than auto-merged.

A card's `key:` is **one line of plain English** (`MXFP8 E8M0 dense linear, decode-bound · gfx950`), not a
rigid `class · gfx · regime` triple — the triple collapses genuinely different cards (vLLM MXFP8 vs sglang
bf16) onto one merge target. The machine-readable slots are the discovery-header fields. `e2e_workflow`'s
`learned/` uses the same contract; its cards predate the discovery header, so its index stays hand-kept
until they are backfilled (see that folder's README).

Card content is sanitized: **relative numbers only** (speedup ratios, percent deltas, % of achievable
peak, roofline bound class) — never wall-clock `ms` or absolute `TFLOP/s`/`GB/s`, which vary by box and
stay in `EVAL_DIR`. A card also records the **pitfalls actually hit** (`symptom → root cause → fix`) and,
when several directions compounded, a `stack:` block giving the **total first, then each direction**.

Available backend languages: `triton` (always) · `flydsl` (SOTA GEMM DSL) · `hip` · `ck`, plus the
skeletons under `../perf_knowledge/languages/` — a language absent on the image is dropped with an
advisory, never a hard fail. Cost ≈ N languages × one single-lane run.

**Nesting:** the dispatcher runs each lane via `workflow(kernel_lane.js)` at exactly ONE level
(dispatcher=0 → worker=1). `e2e_workflow` and `kernel_workflow_bmk` therefore call the **worker
(`kernel_lane.js`) directly**, never the dispatcher, so they never exceed one level.

```
Workflow({ scriptPath: "<WF_DIR>/kernel_workflow.js", args: {
  kernel_path: "/abs/path/to/my_hip_kernel", workflow_dir: "<WF_DIR>",
  mode: "bakeoff", backends: ["hip","triton","flydsl"], gpu_ids: "0,1,2", budget: 6,
}})
```

`<WF_DIR>` is the only location-specific value and it is supplied at call time (it is just the
dirname of `scriptPath`). Everything else is derived: `exp_root` defaults to `<parent of WF_DIR>/exp`.

The user-facing prompt stays minimal & generic, e.g.:
- `optimize /xxx/xxx/knn`
- `optimize /xxx/xxx/knn, budget 6, focus on wrapper overhead`
These map to `kernel_path` (+ optional `budget` / `task`). No repo URL needed.

## Output
Everything lands under `<exp_root>/team_<kernel>_<timestamp>/<kernel>/` (default `exp_root` =
the `exp/` folder sibling to `workflow_dir`):
- `COMMANDMENT.md`, `baseline_timing.json`, `analysis.json`, `codebase_context.md`, `roadmap.md`
- `baseline_metrics.json`, `profiling_summary.md`
- (DRA, when `dra_enabled`) `deep_search.md` (full research), `deep_search_brief.md` (compact ranked
  directions — the planner's input), `deep_search.json` (structured portfolio), and
  `research/{facts.json, questions.json, answers/<id>.json, blindspots.json}` (the research trail)
- `round_N/engineer_i/{worker_result.json, report.md, best_patch.diff}` — each engineer's mini-report
- `round_N/integrate/`, `insight_log.md`, `current_best.diff`
- `tech_lead_report.md` — round-by-round narrative + final per-case table (the TechLead summary)
- `final_patch.diff`, `optimized/`, `director_validation.json` — the official verified result

## Generality (single kernel ↔ e2e model)
The script never branches on kernel type or single-vs-e2e. Everything flows through the
**COMMANDMENT** discovered/built at the Benchmark phase (setup / correctness / benchmark / profile
commands + a parse hint). For a vLLM/SGLang model the only difference is what those commands contain
(launch the server, run a throughput/latency benchmark, define output-parity correctness); the
Director/TechLead/Engineer orchestration is identical.

## Files
```
kernel_workflow.js     ENTRY POINT / dispatcher (mode=optimize|author -> kernel_lane; mode=bakeoff -> multi-language bake-off)
kernel_lane.js         single-language WORKER (the deterministic optimize/author pipeline; called per lane)
kernel_workflow_bmk.js batch orchestrator (runs kernel_lane on a list of kernels, one batch per GPU)
roles/               director, tech_lead, engineer, deep_engineer (deep_explore),
                     author_engineer, benchmark_engineer, profile_engineer,
                     verify_engineer, integrator, oracle_freezer (bake-off freeze),
                     update_experience (learned-card curation, every run),
                     researcher (DRA, opt-in)
knowledge/           optimization_strategies, hip/triton/wrapper, profiling_guide,
                     amd_instinct (multi-card: gfx942/gfx950), self_monitoring, geomean_levers
knowledge/learned/   distilled experience cards (ADVISORY priors; each card self-describing via its
                     discovery header, INDEX.md GENERATED from them; written by the
                     TechLead update_experience step at the end of EVERY run). This sink is
                     kernel-gated (frozen-baseline isolated A/B); e2e-gated lessons stay in
                     e2e_workflow/knowledge/learned/ and cite these cards -- see learned/README.md
scripts/             gpu_lock.sh, profile_kernel.sh,
                     kb.py ... index      (regenerate a learned/INDEX.md from the cards' discovery
                     frontmatter; sink-agnostic -- takes the dir, so it also serves
                     e2e_workflow/knowledge/learned; `--check` for CI), test_learned_index.js (its guard),
                     test_mode_dispatch.js (regression guard: mode dispatch + bake-off lane
                     routing; stubs the runtime, no GPU/agent — `node scripts/test_mode_dispatch.js`)
```
The bake-off references e2e's `op_benchmarker` role + `harness_lib.py` **in place** at
`../e2e_workflow/` (single source, no copy).
