---
id: apply_mono_kernel_to_framework
title: "Ship a mono kernel in vLLM (specialized_decode layer) or SGLang (sgl-amd-mono plugin): routing, refusals, build, fail-stop, smoke (decode, TP, gfx942)"
kind: expert_skill
authors: [GEAK Team]
scope: e2e               # e2e: validated by e2e_workflow (Director same-session A/B), consumed by routing / integration
# ---- selector: the workflow matches these against the live bottleneck ----
match:
  # Not operator-gated, for the same reason as mono_kernel_layer: the trigger is a decoder layer
  # delivered as one persistent kernel, which no single operator name describes.
  operator: '*'
  arch_class: ['*']
  gens: [gfx942]
  dtypes: [bf16, fp8_e4m3_fnuz, mxfp4]
  regimes: [decode]
  from_backend: ""
  to_backend: flydsl
# ---- expected effect: the validation gate's pass criteria ----
# e2e_delta_min_pct is the repo-conventional sanity bar for an e2e-scope skill, NOT a measured claim:
# nothing in this skill has been validated on a box yet (status: draft).
expects:
  e2e_delta_min_pct: 1.0
  parity: required
# ---- validation: AUTO-FILLED by validate_skill.py — do NOT hand-edit ----
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

A mono kernel for one decoder-layer kind already exists (see
[`mono_kernel_layer`](../mono_kernel_layer/skill.md)) and has to be delivered into a serving engine on
gfx942 with tensor parallelism, so that it is measured end to end against the same server with the switch
off:

- **vLLM**: a `specialized_decode` layer, switch `--monokernel`
  ([references/vllm_specialized_decode.md](references/vllm_specialized_decode.md));
- **SGLang**: an `sgl-amd-mono`-style plugin, switch `--monokernel-backend amd`
  ([references/sglang_sgl_amd_mono.md](references/sglang_sgl_amd_mono.md)).

**Terms.**
- `WIDTHS`: the decode row counts the kernel is built for (device-dependent through LDS).
- `PHYSICAL_CU_COUNT`: the number of resident blocks, at most one persistent workgroup per CU.
- `REFUSALS`: the layer kind's list of conditions under which the kernel must be refused.
- `NUMERICS`: the layer kind's numerics statement (the KV write is exact; activations at the derived
  relative bound); `TOLERANCE`: the per-output bounds taken from it. `BITWISE` is claimed only if the
  layer unittest shows max_abs = 0 for every output.
- `MonoSpec` (SGLang): the spec type the plugin registers with SGLang's mono-kernel framework.

## Mechanism

A mono kernel pays only when it is what the engine actually launches inside its captured decode graphs;
that is where its speedup comes from. So the delivery is mostly a question of *when the engine routes a
step to it* and *what happens when it cannot*: routing is a per-step decision made from data every rank
shares (the batch descriptor in vLLM, the CUDA-graph key in SGLang), it must replay exactly as captured,
and every case the kernel does not cover has to keep the stock path (vLLM: or refuse, which switches the
kernel off for the run with every reason named). Because ranks wait on each other inside the kernel, any
disagreement between ranks is a hang until the deadline, not a wrong answer; this is why the failure
handling (bounded waits, one sticky error word, abort with the decoded cause) is part of the deliverable.

## Procedure

Steps that both frameworks share. Where a rule is stated for one framework only, it is marked; the full
shape of each delivery is in the two reference pages.

1. **Read the framework source at the exact commit you target before writing anything.** It defines the
   hook you plug into. vLLM: use whatever that commit actually has (e.g. `KernelConfig.monokernel`, the
   `specialized_decode/` package of another model); where this guide and the commit disagree, follow the
   commit and say so in your notes. SGLang: the `monokernels` package defines the spec type, the
   registration hook and the selection key; use exactly those and record the `api_version` you target.
2. **Keep the framework footprint minimal; the switch is off by default.** The code lives in the model's
   (vLLM) or the plugin's (SGLang) own package. vLLM: no changes to the core beyond what the commit already
   provides; if it has no hook at all, add the smallest possible one (a flag and one call site at model
   load) and list it separately in the patch. SGLang: the patch adds the package under the plugin's layout
   and leaves the model file untouched.
3. **Route only pure decode at widths in `WIDTHS`** to the mono kernel; everything else keeps the stock
   path (vLLM names prefill, mixed batches, speculative verify and any width outside `WIDTHS`). The
   decision depends only on data every rank shares and replays under CUDA graphs exactly as it was
   captured.
4. **Declare what the kernel cannot handle instead of failing at run time.** vLLM refuses in three stages
   (config before load, model after load, cache at each KV-cache bind), each listing every reason at once;
   a refusal disables the mono kernel for the run with a log line that names every reason and is not an
   exception. SGLang expresses every `REFUSALS` entry as a requirement in the spec (`requires`).
5. **Build before KV sizing.** JIT every width at load, before the KV-cache sizing, so the sizing sees the
   mono kernel's memory (peer buffers, control region, any weight repack). The `PeerGroup` is created over
   the TP group.
6. **Counters.** Per rank: launches and skips by reason, readable from a log line (vLLM: or the metrics
   endpoint if the commit has one for specialized decode).
7. **Fail-stop.** Every wait is bounded; the kernel records an error in its sticky word and broadcasts it
   to the peers; decode it on the host and abort the engine with the cause; never emit the token of a
   failed step.
8. **Smoke test.** The server with the switch on versus the same command with it off, installed through
   the workflow's overlay setup, never by editing site-packages. CUDA graphs stay on (SGLang's smoke says
   so explicitly; the vLLM routing likewise replays under CUDA graphs). A launch counter of 0 with the
   switch on is a failed effect proof even if the outputs match, because it means routing sent everything
   away (stated for vLLM's `plan()`). Serving-smoke and GSM8K results are correctness gates here, not
   speed evidence.
9. **Measure end to end** against the switch-off run with the workflow's A/B, following the benchmark
   hygiene below.

## Knobs & pitfalls

Knobs: the switch (`--monokernel` in vLLM, `--monokernel-backend amd` in SGLang), `WIDTHS`, the supported
TP sizes, the resident-block count (`PHYSICAL_CU_COUNT`), and, in SGLang, the kernel's wait deadline (half
the watchdog, clamped to 5–60 s).

**Integration with the serving engine.** Most of these are integration or robustness issues that a stage's
own checks do not see, so test them with the engine.
- **KV cache can be reallocated after profiling.** A kernel that captures KV pointers at bind time reads
  the old cache; re-read the KV pointers every step.
- **Graph-capture modes.** With mixed capture modes (e.g. vLLM FULL_AND_PIECEWISE) the engine also
  captures PIECEWISE graphs with small dummy decode batches. If those take the kernel path, the compiled
  model's own graphs of those sizes are never captured, and a later replay of the missing graph faults.
  Steps in PIECEWISE mode (capture or replay) must take the native path; test with a forced-decode run,
  not only the benchmark.
- **Same path on every rank**, decided from shape, attention metadata and graph mode; a mismatch hangs the
  system-scope waits.
- **Watchdog thread vs graph capture.** A host thread reading the error flag in global capture mode breaks
  capture at start-up; run it in relaxed capture mode.
- **Leaked runners.** A traceback that keeps a frame alive can keep a whole model runner alive; drop
  references on failure.
- **Weight layout mismatch** → disable the kernel on every rank consistently, do not crash.
- Runtime-level pitfalls that also bite here (RCCL under graph capture, divergent fallbacks, the FlyDSL
  compile cache): [runtime_primitives.md](../mono_kernel_layer/references/runtime_primitives.md#pitfalls-seen-so-far).

**Serving-benchmark hygiene.**
- Do not run anything else on the measured GPUs.
- Restart the server before every serving benchmark (no prefix cache carried over); alternate native /
  kernel runs.
- Never pattern-kill processes: you may kill another user's client.
- Standalone bench gains often do not show at the serving metric while the steps the kernel does not take
  (widths > 1, prefill-mixed) dominate it; cover those first.
- TTFT can look worse only because the faster run reaches later, longer trace turns; compare on the same
  set of completed requests.

## Do-no-harm notes

- **Switch off must be inert.** With the switch off, the patched server runs the stock path; the vLLM
  smoke checks it as a bit-for-bit match with the unpatched server.
- vLLM: refuse instead of silently falling back; a refusal names every reason in a log line and is not an
  exception. SGLang: every `REFUSALS` entry is expressed as a `requires` condition of the spec. In both, a
  kernel-path error aborts the engine with the decoded cause.
- Pure decode at the kernel's widths only. Prefill, mixed batches, speculative verify, PIECEWISE graph
  modes and any unsupported configuration stay on the native path; vLLM's config-stage refusals are PP,
  DP, EP, LoRA, speculative decoding, KV connectors, quantization formats other than the loaded one, a TP
  size the kernel was not built for and the layer kind's `REFUSALS`, and models using
  `@support_torch_compile` are refused.
- Advisory prior, status draft: a matched skill is a candidate to reproduce, never a mandate, and the
  engine-level A/B decides.

## Sources

- Upstream vLLM and SGLang issues and pull requests are linked in the two reference pages.
- Companion kernel skill: [`mono_kernel_layer`](../mono_kernel_layer/skill.md).
