## Highlights

### Multi-backend optimization in kernel_workflow

- Adds a `bakeoff` mode alongside `optimize` and `author`: it freezes the input kernel into an immutable baseline, discovers what could compete, runs one lane per candidate in parallel across the GPU pool, and reports every candidate with its speedup, validation status, and reproduction recipe.
- Three candidate classes compete: the incumbent language's in-place optimize lane, always included so there is a control; author lanes across Triton, HIP, CK, FlyDSL, aiter, ASM, TileLang, and Gluon; and environment-tuned backends that need no source change at all.
- Every candidate is scored against the same frozen baseline, and the oracle rejects any lane that alters the baseline, the harness, or the test, so cross-language numbers are directly comparable. `optimize` and `author` runs behave exactly as before.

### Gluon authoring and optimization

- Adds an expert skill (`gluon_authoring`) that ports a tuned plain-Triton champion to Gluon, faithfully recovering layouts and optionally re-injecting the pipeline, enabling further Gluon optimization.
- Applies to MI325X (gfx942) and MI355X (gfx950) on Triton 3.6–3.8, covering attention, block-scaled GEMM, and more, boosting performance over their tuned Triton champion.
- To enable, turn on expert skills (`use_expert_skills=true`). Future work: promotion out of draft, a wider measured set, e2e workflow support, and LLVM/LLIR co-design.

### Enhanced FlyDSL integration

- Adds an expert skill that builds FlyDSL on demand and lands a high-performance kernel for int4 (group-quantized) MoE into a live vLLM server; also fixes a load-time weight-conversion bug that duplicated memory and starved the KV cache (OOM).
- Applies to int4 group-quantized MoE models on MI300X under vLLM, covering both prefill and decode from one kernel through per-M-bucket tiles, with an acceptance gate that picks its metric from the live ISL/OSL rather than assuming decode.
- To enable, turn on expert skills. Matching offers the skill as an advisory candidate rather than a fixed choice, so new skills and backends can be added freely, and the FlyDSL build runs only when a skill matches.

### Roofline-guided routing in e2e_workflow

- Adds a pluggable analysis-skill slot to the e2e knowledge layer, with roofline as its first skill: per kernel it estimates how much of the hardware ceiling is already reached and the expected end-to-end gain, classifying each kernel on memory and compute utilization at once.
- Routing reports two orderings, by share of GPU time and by expected gain, rather than silently combining them, and a saturated kernel is rerouted toward moving fewer bytes rather than dropped.
- The skill is advisory and may never prune a candidate or override measured GPU time. With the analysis skill off, behavior is byte-identical to v4.0.

---

## CI/CD infrastructure

The sections above are what GEAK can optimize; this one is how GEAK itself is developed. A change to an agent workflow is now validated by running it, not by reading it.

- A GPU-free tier runs on every change: lint, unit tests behind a coverage gate, secret scanning, code analysis, workflow control-flow regressions, and a dry-run check of the end-to-end entry point.
- A self-hosted tier runs the real end-to-end workflow per model on ROCm hardware, gating on GPU health before it commits an allocation, killing a wedged run instead of letting it burn to the wall-clock cap, and judging the outcome on exit code and result status, including whether the measured baseline is real.
- The agent model and the per-model budget are configured through repository variables, and an onboarding guide covers enrolling a new model or image. No secrets or endpoints live in the repository.

---

## What's New

### Added

- Bake-off mode in `kernel_workflow` for multi-backend kernel optimization, with frozen-baseline scoring so candidates from different backends are directly comparable.
- Gluon authoring support, including the Triton-to-Gluon port path and Gluon knowledge.
- FlyDSL build-on-demand provisioning and int4 MoE apply-back into live vLLM serving.
- A pluggable analysis-skill slot in `e2e_workflow`, with roofline as its first skill, adding headroom-aware kernel routing.
- A two-tier CI harness covering both static checks and full end-to-end runs on GPU hardware.
- Learned optimization knowledge from real campaigns across MoE, quantization, routing, and decode kernels.

### Improved

- Higher speedups on kernels whose best implementation is not in their original language.
- Better use of optimization budget through headroom-aware routing.
- More trustworthy measurement and stricter acceptance, so a reported winner is one that beat the frozen baseline.
- Clearer end-to-end reporting of baseline alignment, backend provenance, and the final verdict.
- Fixes for a memory-duplication bug in the FlyDSL MoE apply-back and an import shadowing issue in the operator benchmark.

---

## Requirements

- AMD Instinct MI GPU, such as gfx942 or gfx950
- ROCm 6+
- ROCm profiler support, such as `rocprof-compute`, `rocprofv3`, or `rocprof`
- Python 3.8+
- Claude Code 2.1.177 or newer for dynamic Workflow support
- For end-to-end optimization: runnable sglang or vLLM backend and local model weights
- Kernel toolchains remain your responsibility, with the exception of FlyDSL, which GEAK can now build on demand
