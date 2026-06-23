# GEAK v4 — Design Document

**Author:** An, Zihao `<zihaoan2@amd.com>`
**System:** Autonomous GPU-kernel & end-to-end inference optimization agent.

---

## 1. Overview

GEAK v4 is a **general optimization agent** that makes GPU kernels and full model-serving stacks run faster — automatically. Point it at a single kernel, or at a live model served by an inference engine, and it runs the whole loop on its own: find what's slow, write and tune better kernels across multiple backends, fold the winners back into the running system, and prove the speedup end-to-end.

It is general by construction: it **auto-detects the target hardware** and tunes against the real device, and it drives any serving engine through a small **one-file adapter** — so the same system applies across GPU architectures, models, and inference stacks without per-target rewiring.

Internally GEAK is a **two-layer hierarchical agent system** with a deterministic control plane:

- The **outer (end-to-end) layer** optimizes real serving throughput (tokens/sec). It profiles, decides what to optimize, integrates results back into the live server, and validates the gain.
- The **inner (kernel) layer** is a reusable single-kernel optimizer, judged on isolated speedup. The outer layer calls it recursively whenever a kernel needs to be written or tuned.

The control flow (phases, parallel fan-out, budgets, gates) is plain code — reproducible. Every *judgment* (strategy, kernel authoring, deciding what counts as a real win) is handled by a specialized LLM agent.

---

## 2. Core Design

### 2.1 Hierarchical multi-agent + Profiler + Knowledge + dynamic workflow
GEAK is a **deterministic orchestrator driving a team of specialized agents**, each with one job:

- **Outer team:** *Director* (sets up the isolated environment, owns the final validated number), *System Architect* (the strategist — decides what to optimize), *Profiler* (measures what's actually hot), *Config Tuner* (cheap config-only wins), *Kernel Extractor* (turns a live kernel into a standalone test), *Op Benchmarker* (optimizes the heaviest kernels), *Integrator* (folds wins back in and runs the end-to-end gate).
- **Inner team (per kernel):** a *Tech Lead* (plans), a *Director* (sets up + independently validates), parallel *engineers* in four lanes (algorithm / memory / compute / host-runtime), plus *author*, *deep-explore*, *benchmark*, *profile*, and *verify* engineers.

Two things keep it honest and effective: a **Profiler** that grounds every decision in measured data (nothing is optimized unless it's first shown to matter), and a **Knowledge** subsystem that feeds prior experience in and writes new lessons out.

### 2.2 Three optimization modes: fast, default, deep
GEAK runs at three effort levels from one codebase, so the user trades time for thoroughness:

- **Fast** — a quick, time-boxed pass that goes straight for the few hottest kernels. Best for a rapid win or a smoke test.
- **Default** — the full pipeline: config tuning → heavy kernels → a sweep of all editable kernels, each integrated and validated end-to-end.
- **Deep** — long-horizon mode for the last drops of performance. It runs to a large time/compute budget (e.g. 24h), optimizes **many kernels and backends in parallel**, revives stalled-but-promising directions instead of giving up, and re-profiles to chase the bottleneck as it moves.

The fast and deep switches are pure add-ons: with both off, a default run is **byte-identical** to the original pipeline — no behavior drift.

### 2.3 Kernel backend change, applied back to the model
The backend is a **search dimension**, not a fixed choice. GEAK can take a live kernel and **re-write it in a different backend** — Triton, FlyDSL, HIP, CK, TileLang — then pick the fastest *correct* version by comparing independently-verified results. A directed migration (e.g. rewrite a Triton kernel to FlyDSL) can be guided by optional human-authored migration recipes. The winning kernel is then **overlaid back into the running model** through a reversible hook, **without ever editing the installed framework/source tree**.

### 2.4 Automatic head-kernel selection + workload-driven unit tests
GEAK picks **what to optimize by impact, automatically.** The System Architect ranks profiled kernels by their true end-to-end leverage (time-share × achievable speedup) and promotes the heaviest ones ("head kernels"). The Kernel Extractor then builds a **standalone unit test straight from the real workload** — capturing the actual input/output as a correctness oracle and the exact shapes and dtypes the model uses. Crucially, it always covers **both the decode and prefill regimes** (real serving is decode-bound, so a prefill-only test would chase the wrong target). This makes optimization **more controllable and more accurate**: kernels are tuned and verified on exactly the cases production runs, and the oracle is frozen so no agent can "cheat" the test.

---

## 3. Pipeline

```
        ┌──────────────────── Dynamic workflow (deterministic control plane) ────────────────────┐
 model  │  Setup → Profile → Strategize → ConfigSweep → HeadKernel → Milestone → Finalize → Report → Validate │
   or  ─▶│ Director  Profiler  Architect   ConfigTuner   Extractor→     Architect→   Integrator  Architect  Director│
 kernel │                                                OpBench/⟲kernel  ⟲kernel                                  │
        └──────────────────────── Knowledge in (learned cards) / out (curation) ──────────────────────────┘
```

| Phase | Owner | What happens |
|---|---|---|
| **Setup** | Director | Isolated working dir; launch a warm server; record the **true baseline** throughput + noise band. Model weights and packages stay read-only. |
| **Profile** | Profiler | Trace the warm server under the real workload → one canonical, Amdahl-ranked hot-kernel list. |
| **Strategize** | System Architect | Route each hot kernel to the right track: **config / head / kernel / host overhead**. |
| **ConfigSweep** | Config Tuner | Cheap wins first — one server flag/env/backend at a time; keep what helps. |
| **HeadKernel** | Extractor → Op Benchmarker → ⟲kernel layer → Integrator | The heaviest GEMM/attention kernels: extract a test → optimize across backends → gate end-to-end. |
| **Milestone** | Architect → Extractor → ⟲kernel layer → Integrator | Sweep the remaining editable kernels above a threshold; integrate; re-profile; grow the knowledge base. |
| **Finalize / Report / Validate** | Integrator / Architect / Director | Assemble the final overlay + patch + launch bundle; write the report; **independently re-measure** the combined result and arbitrate the official number. |

In **deep mode**, the HeadKernel step becomes a single global optimizer that runs many kernel×backend lanes in parallel to a long budget, with periodic re-profiling and an end-to-end gate.

---

## 4. Components

### 4.1 Orchestrator
The deterministic control plane that sequences phases, fans work out in parallel, enforces budgets, and selects the **fast / default / deep** mode. The mode switches are designed so that with them off, the default behavior is unchanged.

### 4.2 Profiler & Amdahl triage
Produces the one hot-kernel list everything else routes on. Beyond raw profiling, it **cleans the data**: it discounts busy-wait/communication time, ignores one-time warmup/JIT spikes, and splits a kernel that serves both prefill and decode into separate entries — then recomputes the percentages so the ranking reflects *real, optimizable* compute. (A multi-GPU trace missing its expected collective kernels is rejected as invalid rather than guessed.) This is what stops GEAK from wasting effort on a kernel that only *looks* hot.

### 4.3 System Architect (the strategist)
The "brain." It reads the profile and applies Amdahl reasoning — *a 1.15× on a kernel that's 78% of the time is ~+10% end-to-end, while a 5× on a 2% kernel is invisible* — to decide where effort pays off. It routes each kernel to config tuning, the head track, the kernel track, or host-overhead, and it curates GEAK's long-term knowledge.

### 4.4 Config Tuner (cheapest wins first)
Raises throughput by changing **configuration only** — attention backend, CUDA-graph/compile, scheduling and memory knobs, backend toggles, optional lower precision — one change at a time, each measured and kept only if it helps (and accuracy-checked when it could affect outputs). No source code is touched here.

### 4.5 Kernel Extractor
Turns a live kernel into a **self-contained, frozen test** the kernel layer can optimize against: it captures (or, for GEMMs, synthesizes) the correctness oracle, records the real shapes/dtypes, includes both decode and prefill cases, and notes the exact hook point used to swap the kernel back in later.

### 4.6 Op Benchmarker (heavy-kernel optimizer)
Optimizes the heaviest GEMM/attention kernels using a **cheapest-first ladder**, escalating only if a cheaper step doesn't win — and every step is gated on correctness:

1. **Pick the best existing backend** for this kernel.
2. **Tune that backend's parameters** (e.g. per-shape GEMM configs).
3. **Write a brand-new kernel from scratch** via the kernel layer (Triton always; FlyDSL first for GEMM/FP8).
4. **Try lower precision** (accuracy-gated).

The advantage: GEAK doesn't pay for an expensive rewrite when a config tweak already wins, yet it still **guarantees a dominant kernel is never skipped** — it clearly distinguishes "couldn't measure" from "genuinely no win," and produces the best version of *each* candidate backend so they can be compared fairly.

### 4.7 Kernel optimization engine (the inner layer)
A reusable single-kernel optimizer. Each round, the Tech Lead plans a few **independent** directions, specialist engineers implement them in parallel in private workspaces, and each result is **independently re-measured** the moment it finishes. Winners are combined, and a change is only kept if it genuinely beats the current best. The correctness oracle is **immutable** (read-only + checksum-verified), and the Director **independently re-validates the final result against the true original baseline** — so the reported speedup is a verified number, not a self-reported claim. It can also write a correct kernel **from scratch** in any target language, and run a dedicated open-ended "deep-explore" rewrite round.

### 4.8 Backend abstraction
One interface over **Triton, FlyDSL, HIP, CK, TileLang, aiter** (plus vendor GEMM libraries and runtime auto-tuners). Each kernel can be authored/tuned in any of them, and the bake-off simply compares their verified results — making "which backend" an outcome of measurement, not a guess.

### 4.9 Serving adapters (any inference engine)
A backend-agnostic dispatcher owns the server lifecycle and the timed benchmark; the actual engine is selected by name and loaded from a small adapter file that defines four functions (launch / health / bench / default-port). **Adding a new serving engine is just writing one adapter** — vLLM and sglang ship today. The timing client can also be swapped independently without changing the engine.

### 4.10 Integrator & gates (the trust layer)
This is what makes GEAK's numbers believable. It overlays a winning kernel back into the live server **reversibly** (the installed package tree must be clean before and after), then accepts it only if it passes a strict gate:

1. the isolated speedup is real and the oracle wasn't tampered with,
2. the kernel **actually runs live** (including inside multi-process/TP workers and under CUDA-graph capture),
3. the end-to-end gain clears the noise band under a **tight, back-to-back A/B** (so machine drift can't fake a win), and
4. **output quality is preserved** — for quantized kernels, task accuracy (e.g. GSM8K) is checked against the *true* baseline.

It also enforces **do-no-harm**: e.g. a faster kernel whose larger memory footprint shrinks the KV-cache and nets a regression is rejected. Verdicts are `accepted`, `stack` (a small but real gain, carried so it can compound), or `rejected`. If nothing genuinely helps, the baseline is kept.

### 4.11 Knowledge system
GEAK deliberately keeps several **distinct** knowledge sources rather than one pile:

- **Learned cards — persistent, written by GEAK.** GEAK's own accumulated experience: one validated principle per card (keyed by kernel-class · architecture · regime), each with a confidence rating, the measured effect, and a citation to the run that proved it. A bounded index keeps only the highest-value cards. At the end of a run, strong findings are curated in and weak/contradicted ones dropped. *This is what makes GEAK faster the more it runs.*
- **perf_knowledge — external reference, read-only.** A curated reference base of state-of-the-art implementations per operator×backend. GEAK reads it as a starting-point pointer ("which backend/algorithm is known-best for this op"), but **never writes to it and never treats it as ground truth** — the on-box A/B is always the judge.
- **Skills / guides — judgment playbooks.** Human-written reasoning guides (preflight checks, profile parsing, backend selection, …) the agents follow: knowledge encoded as procedure, not rigid scripts.
- **Run-scoped blackboards — ephemeral, within one run.** Live scratchpads (per-op and run-global) that let the many parallel lanes in a *single* run share what's working right now. They exist only inside that run's working directory and are discarded afterward.
- **Expert skills — opt-in.** Human-authored backend-migration recipes; advisory only.

**Learned cards vs. run-scoped blackboards are not the same thing:** learned cards are *persistent, curated, evidence-cited* memory that survives across runs and is size-bounded; the blackboards are *ephemeral, mutable* scratchpads used only for live coordination within one run. The bridge between them: at run end, blackboard findings that meet the evidence bar get distilled into learned cards.

### 4.12 Resilience & isolation
Every GPU command runs through a lock that serializes access and isolates each workspace's build cache; the device architecture is auto-detected and pinned to avoid vendor-JIT enumeration storms; agents are wrapped with timeouts and retries to survive hangs and transient API faults; and the environment preflight **degrades rather than blocks** on missing optional tools — so GEAK runs across varied setups.

---

## 5. Summary

GEAK v4 turns the scarce, slow work of GPU-kernel and inference optimization into an automated, trustworthy loop: it profiles what actually matters, optimizes kernels across multiple backends — in parallel and over long horizons when asked — integrates the winners back into the live serving stack, and proves every gain end-to-end with tight A/B and accuracy gates, doing no harm when a change doesn't genuinely help. It is general by construction: hardware is auto-detected, any serving engine plugs in through a one-file adapter, and the knowledge it accumulates makes it faster the more it is used.
