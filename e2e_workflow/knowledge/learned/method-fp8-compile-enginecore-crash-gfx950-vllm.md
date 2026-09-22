---
key: fp8 serving launch · gfx950 · vLLM 0.29.0 — intermittent EngineCore startup loss; --enforce-eager workaround
type: method
confidence: ★★
effect: intermittent EngineCore startup loss at conc=64 (isl4096/osl32), seen in BOTH eager and compile launches; the eager workaround produced a completing 1728 tok/s baseline (320/320), but eager also failed and compile also succeeded — no per-mode fix
confirms_cited: 0
confirms_blind: 0
attempts: 8
last_seen: 2026-09-22
name: method-fp8-compile-enginecore-crash-gfx950-vllm
description: intermittent EngineCore startup failure on gfx950 vLLM 0.29.0 fp8 TP1, in BOTH eager and compile; --enforce-eager is an observed (not guaranteed) workaround
keywords: [fp8, enforce-eager, enginecore, startup-failure, intermittent, triton-fp8-blockscale-mm, torch-compile, gfx950, vllm]
platforms: [gfx950]
kernel_class: method
regime: n/a
lifecycle: active
---
# fp8 intermittent EngineCore startup failure on gfx950 (vLLM 0.29.0) — try `--enforce-eager` as a workaround
- lever: on gfx950 / vLLM-0.29.0 fp8 (a8w8 block-scale) TP1, EngineCore can fail during startup (Python `RuntimeError: Engine core initialization failed`, after real device work e.g. a FillFunctor hipLaunchKernel) — retrying with `--enforce-eager` has brought the server up, including a measured baseline; treat it as a workaround to try, not a diagnosis.
- apply: in e2e_workflow pass `initial_extra_server_args: "--enforce-eager"` — it seeds baseline/config/validation launches (INIT_FLAGS→curFlags→EXTRA_SERVER_ARGS) and auto-drops the CUDA-graph deploy requirement (e2e_workflow.js:739); it does NOT cover the bare preflight parity smoke, which launches compiled and has also come up fine.
- verify: eager and compile both select the same TritonFp8BlockScaledMM kernel, so the kernel is equivalent but the whole path is not; confirm /health=200 per launch and keep the mode consistent across reference vs candidate legs (eager disables torch.compile fusions AND HIP graphs, a different operating point).
- caution: also verify the failure still reproduces before assuming it — cause is unresolved and the signature is intermittent: eager launches have ALSO failed and compiled launches have ALSO succeeded, so re-probe compiled mode when it works, preserve failure artifacts for diagnosis, and read this as a per-launch retry rather than a compile-only or GPU-independent fix.
- source: 2026-09-22 run logs — eager fail vllm_launch_debug_default.log (enforce_eager=True/NONE → EngineCore init failed) and vllm_launch_debug_attn_triton.log (device FillFunctor before failure); compile OK preflight_smoke/server.log (enforce_eager=False → Application startup complete); eager baseline 1728 tok/s (isl4096/osl32/conc64, 320/320); Astra peer-review 20260922.
