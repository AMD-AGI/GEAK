---
key: fp8 serving launch · gfx950 · vLLM 0.29.0 torch.compile · enforce-eager workaround
type: method
confidence: ★★★
effect: a GPU-INDEPENDENT server-launch killer — fp8 model init C++-aborts EngineCore before any bench; swapping GPUs never helps; `--enforce-eager` clears it
confirms: 4
last_seen: 2026-09-22
---
# fp8 + torch.compile aborts EngineCore at model load on gfx950 (vLLM 0.29.0) — launch `--enforce-eager` to escape
- symptom: `vllm serve` of an fp8 (a8w8 block-scale) model on gfx950 / vLLM 0.29.0 dies during EngineCore
  init with a **C++-level abort and NO Python traceback**, immediately after the line
  `Selected TritonFp8BlockScaledMMKernel for Fp8LinearMethod`. Server never becomes healthy → preflight
  blocks → ZERO bench samples. Model used in the confirms: `Qwen3-14B-FP8` (TP1).
- **it is NOT what it looks like.** Ruled out on clean idle GPUs, pristine env, fresh JIT caches, stock
  `/usr/local/bin/vllm`: NOT the container/image (stock serve launches the SAME fp8 model fine in eager),
  NOT the GPU (reproduces across GPUs; a compile-path abort at init is GPU-independent by construction —
  it happens before real device work, so **moving to a fresh GPU changes nothing**), NOT the torch
  profiler (`--profiler-config` in eager launches fine), NOT OTel (`otlp_traces_endpoint=None` in the
  crash; the `vllm/tracing/otel.py` frames are an always-applied passthrough decorator, not active
  tracing), NOT the osl/decode operating point.
- root cause: a vLLM torch.compile bug triggered ONLY when `enforce_eager=False` — i.e. the fp8 compile
  path is live (`compilation_config.custom_ops:['+quant_fp8', ...]`, `pass_config.fuse_norm_quant=True`,
  `fuse_act_quant=True`, `CompilationMode.VLLM_COMPILE`, `FULL_AND_PIECEWISE` cudagraph). Eager
  (`CompilationMode.NONE`) selects the **same** TritonFp8BlockScaledMM kernel but skips the crashing
  compile/fusion build.
- decisive isolation (back-to-back, same box/model/session — only `enforce_eager` differs):
  · `vllm serve` (compile, `enforce_eager=False`)         → log dead-ends at `Selected TritonFp8BlockScaledMMKernel`, abort
  · `vllm serve --enforce-eager` (`enforce_eager=True`)    → `Application startup complete.`, /health=200
  · `vllm serve --enforce-eager` + harness torch profiler  → `Application startup complete.`, /health=200
- apply (workaround): launch the server `--enforce-eager`. In the e2e_workflow, pass the first-class arg
  `initial_extra_server_args: "--enforce-eager"` — it seeds `EXTRA_SERVER_ARGS` on EVERY launch
  (baseline / sweep / head-kernel / validation), and the workflow is eager-aware (it auto-disables the
  CUDA-graph deploy requirement when the flag is present, so no graph-safety requirement is injected into
  kernel tasks). Confirmed carrying a full run: baseline cleared at ~1728 tok/s, then Profile.
- **caution, don't foreclose (per README rule 3):** eager DISABLES torch.compile fusions, so the run
  optimizes the **eager fp8 path** — a valid, COMPLETING run, but it forgoes the compile tuning surface.
  It is a workaround, not a fix. Also **verify** compile mode still aborts before assuming it: the crash
  is **FLAKY** — the same model's bare (non-eager) preflight smoke has launched fine at least once. So if
  a run genuinely needs the compile path, re-probe a bare launch rather than treating the abort as
  deterministic; the durable fix is either the operator patching vLLM or a workflow auto-retry that
  detects this signature and relaunches `--enforce-eager`.
- source: 2026-09-22 controlled smokes on idle GPUs (`/tmp/fp8_stock_smoke.sh` eager=OK,
  `/tmp/fp8_compile_smoke.sh` non-eager=CRASH, `/tmp/fp8_prof_smoke.sh` eager+profiler=OK) + three
  independent sessions with the same signature: `Qwen3-14B-FP8-vllm/...045621` (`server_launch_failure.json`
  verdict=block), `Qwen3-14B-FP8-routing-vllm/...110329` (smoke_default2 abort vs smoke_eager `Application
  startup complete.`), `Qwen3-14B-FP8-compaction-vllm`. Relaxes the OTel/profiler/image false leads.
