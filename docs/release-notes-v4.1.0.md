## Highlights

### Multi-backend optimization in kernel_workflow

- Adds a `bakeoff` mode alongside `optimize` and `author`: it freezes the input kernel into an immutable baseline, discovers what could compete, runs one lane per candidate in parallel across the GPU pool, and reports every candidate with its speedup, validation status, and reproduction recipe.
- Various candidate classes compete: the incumbent language's in-place optimize lane, always included so there is a control; author lanes across Triton, HIP, CK, FlyDSL, aiter, ASM, TileLang, and Gluon; and environment-tuned backends that need no source change at all.
- Every candidate is scored against the same frozen baseline, and the oracle rejects any lane that alters the baseline, the harness, or the test — so a lane cannot move its own denominator, and cross-language numbers are directly comparable. Dispatch overhead can no longer be reported as kernel speedup on short kernels.
- To enable, ask for a bake-off by mode; `optimize` and `author` runs behave exactly as before.

### Gluon authoring and optimization

- Adds a Gluon authoring expert skill: the API surface, a do-not-write list of constructs that compile and then silently cost performance, and a procedure with the scripts to execute it.
- The Triton to Gluon port recovers layout and pipeline faithfully rather than translating them. Layouts come from the champion's own compiler IR through the compiler's own conversion, each with a round-trip proof, so no mapping table can fall behind Triton and an unsupported layout is reported rather than guessed at.
- Applies to gfx942 and gfx950, entered either from a tuned Triton kernel or from an existing Gluon one. Because a port measured against an unoptimized kernel measures the config sweep rather than Gluon, the plain source must already be at its own best config before the port begins.
- To enable, turn on expert skills. The skill is advisory, never overrides a measured A/B, and ships as a draft, so nothing selects it until it is validated.

### Enhanced FlyDSL integration

- Adds an expert skill that builds FlyDSL on demand and lands a high-performance kernel for int4 (group-quantized) MoE into a live vLLM server; also fixes a load-time weight-conversion bug that duplicated memory and starved the KV cache (OOM).
- Applies to int4 group-quantized MoE models on MI300X under vLLM, covering both prefill and decode from one kernel through per-M-bucket tiles, with an acceptance gate that picks its metric from the live ISL/OSL rather than assuming decode.
- To enable, turn on expert skills. For extensibility, matching provides the skill as an advisory candidate rather than a fixed choice — new skills and backends can be added freely.

### Roofline-guided routing in e2e_workflow

- Adds a pluggable analysis-skill slot to the e2e knowledge layer, with roofline as its first skill: per kernel it estimates how much of the hardware ceiling is already reached, the attainable speedup, and the expected end-to-end gain, classifying each kernel on memory and compute utilization at once so that latency-bound kernels are not mistaken for either.
- Routing reports two orderings — by share of GPU time and by expected gain — rather than silently combining them. A kernel that dominates GPU time while sitting near its roofline has little left to give, while a smaller kernel far below its roofline may be the real win.
- A saturated kernel is rerouted, not dropped. Since a roofline percentage says how well a kernel executes its current byte and FLOP budget, not whether that budget is necessary, the remaining lever is to move fewer bytes: fusing adjacent operations, skipping unrouted experts, layout and packing changes, lower precision.

---

## CI/CD infrastructure

The sections above are what GEAK can optimize; this one is how GEAK itself is developed. A change to an agent workflow is now validated by running it, not by reading it.

- A GPU-free tier runs on every change: lint, Python unit tests behind a coverage gate, secret scanning, code analysis, workflow control-flow regressions covering mode dispatch and the guarantee that disabling expert skills leaves prompt assembly byte-identical, and a dry-run check of the end-to-end entry point's argument mapping.
- A self-hosted tier runs the real end-to-end workflow per model on ROCm hardware through the cluster scheduler, as a matrix with a single-model smoke tier and a full verification tier driven by a model registry. Each run resolves the container image and weights, gates on GPU health before committing an allocation, brings up the serving container, installs the agent toolchain inside it, runs the workflow, and judges the outcome on exit code and result status — including whether the measured baseline is real.
- Runs are protected rather than merely launched: an independent runner per model, a host-side liveness monitor that kills a wedged run instead of letting it burn to the wall-clock cap, a container watchdog, architecture auto-detection for image selection, and GPU restriction.
- The agent model and the per-model budget are configured through repository variables, so a model bump is a settings change rather than a code change. No secrets or endpoints live in the repository.
- To enroll a new model or image, follow the onboarding guide in the CI directory.

---

## What's New

### Added

- Bake-off mode in `kernel_workflow` for multi-backend kernel optimization.
- Frozen-baseline scoring, so candidates from different backends are directly comparable.
- Gluon authoring support, including the Triton-to-Gluon port path and Gluon knowledge.
- FlyDSL build-on-demand provisioning and int4 MoE apply-back into live vLLM serving.
- A pluggable analysis-skill slot in `e2e_workflow`, with roofline as its first skill.
- Headroom-aware kernel routing alongside ranking by share of GPU time.
- A two-tier CI harness covering both static checks and full end-to-end runs on GPU hardware.
- Learned optimization knowledge from real campaigns across MoE, quantization, routing, and decode kernels.

### Improved

- Higher speedups on kernels whose best implementation is not in their original language.
- Better use of optimization budget through headroom-aware routing.
- More trustworthy measurement, especially on kernels short enough for launch overhead to distort the result.
- Stricter acceptance, so a reported winner is one that beat the frozen baseline.
- Clearer end-to-end reporting of baseline alignment, backend provenance, and the final verdict.
- Lower per-run overhead in the kernel workflow.
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
