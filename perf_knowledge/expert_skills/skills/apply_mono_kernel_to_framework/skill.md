---
id: apply_mono_kernel_to_framework
title: "Deliver an existing whole-decoder-layer mono kernel into vLLM (specialized_decode layer) or SGLang (sgl-amd-mono-style plugin) and measure it end to end, switch on versus off, on MI300X (gfx942)"
kind: expert_skill
authors: [GEAK Team]
scope: e2e
match:
  operator: '*'
  arch_class: ['*']
  gens: [gfx942]
  dtypes: [bf16, fp8_e4m3_fnuz, mxfp4]
  regimes: [decode, graph]
  from_backend: ""
  to_backend: ""
expects:
  e2e_delta_min_pct: 1.0
  parity: required
validation:
  status: draft
  last_verified: ""
  gpu: ""
  model: ""
  measured: {isolated: "", e2e_pct: "", parity: ""}
  artifact: ""
role: advisory_prior
supersedes: []
---

## When to use
Applicability test. All three lines must be true, and each is something you can observe:
1. You are an **e2e role** (routing or integration) working against a live vLLM or SGLang server. You are
   **not** inside a kernel lane: your work item is not a task directory whose `unittest.py` oracle you must beat.
2. A **mono kernel already exists** for the layer kind to replace (one persistent kernel per rank that computes
   a whole decoder layer for a decode step, with the tensor-parallel all-reduce inside), and its own whole-layer
   oracle reports it correct and faster than the native layer at the widths it was built for.
3. What is left is to make the engine launch it behind a default-off switch and to measure the **same server
   with the switch on versus off**.

If any line is false, this skill does not apply — stop reading. Writing the kernel, making it faster or fixing
its numerics is kernel-lane work (`mono_kernel_authoring`, `mono_kernel_optimize`, `native_decode_numerics`).

> **Status of the claims.** The delivery phase this skill is ported from (an internal mono-kernel generation
> workflow, not public) was **designed but never exercised end to end**: procedure and acceptance checks are a
> *design prior*. Rows marked *observed* were hit in hand-driven integrations of whole-layer kernels into vLLM.
> Switches, paths and conventions are *read from upstream source* at the snapshots named in Sources and change
> between versions: verify each at the exact commit you target, follow the commit, and record the difference.

### Quick card: routing role (first ten minutes)
1. Run the test above, then read the kernel's contract: layer kinds and layer indices covered, decode widths,
   TP size, device, weight formats as loaded, refusal list, launches per layer, extra memory per rank.
2. Estimate the share of the workload's decode steps that the kernel covers (step kind and rows). That share times
   the share of layer time replaced bounds the delta; if it is small, say so instead of scheduling a long A/B.
3. Plan ONE candidate (`source: expert_skill:apply_mono_kernel_to_framework (advisory)`), switch-gated, baseline =
   the identical server with the switch off, engagement check = *launches grow on every rank, 0 with the switch off*.
4. Point the integrator at [vLLM](references/vllm_specialized_decode.md) or [SGLang](references/sglang_plugin.md).

### Quick card: integration role (first ten minutes)
1. Pin the framework commit and read the hook it already has (step 1). Write no code before that.
2. List every framework-internal name the adapter will read; check on CPU that each exists at the commit.
3. Build the `{layer index: kind}` table from the kinds that pass their oracle; everything else stays stock.
4. Start the patched server with the switch **off** first: it must behave as stock (step 9).
5. Only then switch on, read the counter line, and spend GPU time on the timed A/B.

## Mechanism
- **The kernel only pays inside the engine's captured decode graphs.** It replaces a layer's many small
  dependent launches, their gaps and the all-reduce with one resident kernel per rank, so the launch must be in
  the graphs the engine replays; per-step host work in the glue (metadata, counter reads, a compile) gives it back.
- **Routing is a per-step decision made from data every rank shares.** Each rank decides alone, but the kernel's
  cross-rank waits assume every rank launched: decide only from the batch descriptor or graph key (forward mode,
  row count, tokens per request), never from tensor values or per-rank state, so that it replays as captured.
- **Any disagreement between ranks is a hang until the deadline.** One rank on the native path while another
  launches leaves both waiting. That is why fail-stop is part of the deliverable: every wait is bounded, a
  sticky error word is broadcast to the peers, and the host decodes it and aborts with the cause.
- **Refusal is per run, routing is per step.** An unsupported configuration is refused for the whole run, on
  every rank, with every reason stated. An ineligible step takes the native path and is counted by reason.
  A silent fallback makes "on" equal "off" and the A/B meaningless, so both must be visible in the log.

## Procedure
Common to both engines. Switch and layout change between versions, so read them at your commit. As read from
source: vLLM at public commit `f09d5e30c`, a `specialized_decode` package behind `--monokernel`; SGLang at upstream
main on 2026-11-23, an `sgl-amd-mono`-style plugin behind `--enable-amd-monokernel` (older, now removed name: `--monokernel-backend amd`).

1. **Pin and read.** Read the mono-kernel hook in the framework source at the exact commit you target (switch,
   spec or decode abstraction, another model's package). Use what exists; if there is none, add the smallest
   hook and list it separately. Serve with the configuration and image the kernel's oracle was frozen under.
2. **Carry the kernel unchanged.** Kernel sources and their runtime support library go in byte for byte (keep
   the license header). Only the glue is rewritten; it builds the launch arguments (weight views, KV and
   metadata pointers) as the layer oracle's adapter does. A kernel bug found here goes back to a kernel lane.
3. **Touch only the covered layers.** Use the commit's hook (current vLLM: the layer loop asks the step for a
   kernel per layer index, stock modules stay; current SGLang: the plugin's forward replaces the model's for
   covered steps). Without one, replace only the table's instances, never the layer class: other kinds share it.
4. **Route per step,** by step kind and rows (rows = requests × tokens per request of the step). Pure decode at
   a row count in the kernel's widths takes the kernel; a verify step only if the kernel's oracle and width list
   cover it. Prefill, mixed batches, other widths and piecewise-graph steps take the native path, counted by reason.
5. **Refuse explicitly, in three stages:** configuration before load; model after load (formats, shapes, dtypes
   as actually loaded); cache at every KV-cache bind (layout, block size, KV dtype, slot-mapping form). Each stage
   lists every reason at once and reaches the same verdict on every rank; never a silent fallback. HOW it is
   expressed follows the framework at your commit: for an explicit opt-in flag, current vLLM (`KernelConfig.monokernel`
   in `vllm/config/kernel.py`, `MonoDecodeLayer._create`) and SGLang (`handle_monokernel` in `binding.py`) fail at
   startup with an error, in the newer code one error listing every unsupported setting. An ineligible step is not an error (step 4).
   Exception: SGLang's binding at that commit has no per-bind stage, so state KV-layout conditions in the spec's `requires`.
6. **Build at load, before KV sizing and outside any graph capture:** JIT every width, allocate the peer
   buffers, the control region and any second weight copy, and create the cross-rank peer group over the TP group.
7. **Count and log.** Per rank: launches (read from the control region at log intervals, never per step) and
   skips by reason, in one fixed log line.
8. **Wire fail-stop.** On a decoded error the engine aborts with the cause; the failed step emits no token.
9. **Run the CPU checks and three-server smoke** on one full TP group. Use the same fixed prompts on the
   unpatched server and on the patched server with the switch off and on. Check switch-off parity, inspect
   the first greedy divergence per prompt, and verify the expected launch count on every rank.
   The [smoke procedure](references/smoke_and_acceptance.md) defines the prompt set and pass conditions.
10. **Time switch-on against switch-off.** For final acceptance, compare the candidate with a fresh unpatched
    server under the [e2e integrator's parity and throughput gates](../../../../e2e_workflow/roles/e2e_integrator.md#the-gate-a-change-enters-e2e-only-if-all-hold).
    Report coverage, counters and commands for reproducing the result.
11. **Partial delivery.** One patch per model; only kinds that pass enter its table. If the smoke fails, bisect
    by kind, drop the offending kind and rerun the whole smoke; if nothing passes, deliver nothing and say so.

## Knobs & pitfalls
| Symptom | Cause | What to do | Status |
|---|---|---|---|
| Wrong outputs in the server although the layer oracle passes | The engine reallocated the KV cache after its profiling run; the kernel kept pointers from bind time | Re-read the KV pointers every step | observed |
| Memory fault on every rank at a later step, typically a short prefill after a prefix-cache hit | Mixed capture mode: the engine also captures piecewise graphs with small dummy decode batches; they took the kernel path, so the model's own graphs of those sizes were never captured | Piecewise-mode steps, capture or replay, take the native path; test with a forced-decode run, not only the benchmark | observed |
| All ranks hang (until the deadline, if the kernel has one) | Ranks chose different paths for one step | Same path on every rank, decided from shape, attention metadata and graph mode | observed |
| Start-up breaks during graph capture with the switch on | A host watchdog thread reads the error word while a graph is captured in global capture-error mode | Capture in relaxed mode (`capture_error_mode` of `torch.cuda.graph`) while that thread runs | observed |
| GPU memory stays held after a failed start | A traceback keeps a frame alive, and with it a whole model runner | Drop the references on failure | observed |
| Weight layout as loaded is not what the kernel reads | Format or layout converted at load | Refuse at stage 2 with the same verdict on every rank, expressed as step 5 says; one rank failing or falling back alone hangs the rest | observed |
| KV capacity shrinks, or start-up runs out of memory | Second weight copy or JIT allocated after KV sizing | Step 6; run both legs at the same memory fraction | design |
| Capture hangs or the first steps time out | A width compiled inside capture | JIT every width at load. Compile time is seconds per width, so many variants slow start-up | design |

- **Shared resources per kind.** One peer group, control region and launcher set per kind, shared by its
  instances, only if the kernel was verified with instances sharing them; otherwise per instance (count the memory).
- **Formats are the loaded ones.** fp8 on gfx942 is FNUZ, so the engine may have converted and rescaled a
  checkpoint at load ([matrix_core.md](../../../hardware/cdna3_mi300/matrix_core.md)). A format with no matrix
  instruction on the generation (MXFP4 on gfx942) needs software dequant in the kernel; refuse it otherwise.
- **Widths are per device.** A gfx942 workgroup has 64 KiB of LDS ([occupancy.md](../../../hardware/cdna3_mi300/occupancy.md)), so fewer widths fit than on gfx950.

### Benchmark hygiene
- Nothing else on the measured GPUs. Stop only processes you started, by id, never by name pattern.
- A fresh server per leg (no prefix cache carried over); legs alternated off and on; same-session ratios only.
- A layer gain does not show at the serving metric while the steps the kernel does not take (other widths,
  prefill-mixed) dominate it: read the skip counters first, and measure at the concurrency you will serve.
  *Measured* (hand-driven MiMo-V2.6-Flash integrations into vLLM; smoke reference, section 10): concurrency-1 P90
  inter-token latency 1.4–1.5× better with one-row steps only, 3.0–3.7× once multi-row decode steps were covered.
- Time to first token can look worse only because the faster run reaches later, longer trace turns: compare on the same completed requests.

## Do-no-harm notes
- **Switch off must be inert.** Compare against the unpatched server's reproducibility and check for zero launches on every rank.
- **Refuse explicitly, never fall back silently:** every reason at once, the same verdict on every rank,
  expressed as the framework does at your commit (current vLLM and SGLang: one startup error for an opt-in flag).
- **Only the steps the kernel's oracle covers:** pure decode at its widths, verify steps only if covered. Every other step stays native.
- **Never emit the token of a failed step.** Abort with the decoded cause.
- Do not edit the kernel, widen a tolerance, lower the candidate leg's memory fraction or edit the installed
  package to obtain a pass. No number from a run whose effect proof failed; an accuracy run is never speed evidence.

## Sources
- Upstream, public, read at the snapshots named. vLLM: [RFC 60904](https://github.com/vllm-project/vllm/issues/60904), pull requests
  [61311](https://github.com/vllm-project/vllm/pull/61311), [62256](https://github.com/vllm-project/vllm/pull/62256), [63486](https://github.com/vllm-project/vllm/pull/63486), [60397](https://github.com/vllm-project/vllm/pull/60397), [61082](https://github.com/vllm-project/vllm/pull/61082), [61662](https://github.com/vllm-project/vllm/pull/61662); source at commit [`f09d5e30c`](https://github.com/vllm-project/vllm/commit/f09d5e30c35862c19a17dd1706147d29d1f7c4f4).
  SGLang: [RFC 43949](https://github.com/sgl-project/sglang/issues/43949), pull requests [44353](https://github.com/sgl-project/sglang/pull/44353), [45389](https://github.com/sgl-project/sglang/pull/45389); source at main [`38ae3402`](https://github.com/sgl-project/sglang/commit/38ae340221b4b65859c68e85d9e8ffb6b9c6f167) on 2026-11-23; the plugin [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono). File and symbol citations are in the two framework references.
- This repository: [e2e_integrator.md](../../../../e2e_workflow/roles/e2e_integrator.md) (gate, overlay, capture safety),
  [system_architect.md](../../../../e2e_workflow/roles/system_architect.md) (`engagement_check`), [e2e_optimization.md](../../../../e2e_workflow/knowledge/e2e_optimization.md).
- Origin: an internal mono-kernel generation workflow (not public): its framework delivery guides, integration role,
  designed acceptance checks, risk list and a pitfalls card from hand-driven integrations. Not validated by GEAK's e2e gate; status `draft`.
