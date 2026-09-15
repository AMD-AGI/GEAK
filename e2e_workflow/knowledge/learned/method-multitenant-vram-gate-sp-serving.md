---
key: e2e A/B capacity planning · any gfx · SP/TP serving on a SHARED box
type: method
confidence: ★★
effect: prevents burning a whole milestone on unlaunchable A/Bs; 4 candidates in one run reached
  "overlay built + offline-verified" but 0 e2e legs, ~45 failed launches, all at component LOAD
confirms: 1
last_seen: 2026-07-31
---
# Gate the e2e A/B on per-rank free VRAM BEFORE building the overlay

- **fact that drives everything: `TP=N` in sglang-DIFFUSION is Ulysses SEQUENCE parallelism, so model
  weights are REPLICATED per rank, not sharded.** FLUX.1-dev at TP=8 needs the WHOLE model on every
  card (transformer 22.2 GiB + T5 ~9.5 + CLIP + VAE). Do not reason from LLM tensor-parallel intuition
  (`per_rank ≈ weights/TP`) — here it is `per_rank ≈ weights`.
- lever: **derive the per-rank requirement from a SUCCESSFUL run's own server log, then poll
  `torch.cuda.mem_get_info` on EVERY card of the mandated set and only then launch.** On this box the
  real figure is ~30–32 GiB/rank (`config/cfg3d/server.log`: load delta 227.83→197.88 GB = 29.95 GB;
  max "Peak memory usage" over 204 samples = 28762 MB). Sampling free memory *during* our own launch
  gave 48 GiB/rank — that overshoot is transient/caching, and an over-tight gate (52 GiB) hides
  marginal windows that would actually have worked. Both numbers, and the correction, are in-run.
- apply: `min(free over the TP set) >= NEED + margin` with a 10 s poll; arm a resume driver that runs
  ref → cand → engagement grep → parity legs unattended when the gate opens.
- verify: a squeezed card fails at `fsdp_load.load_model_from_full_model_state_dict` ("0 bytes is
  free") or at NCCL init ("Failed to CUDA calloc … HIP failure: out of memory" → `ncclUnhandledCudaError`).
  Both are *component-load* faults, so shrinking `E2E_REPEATS`/`CONC` cannot help — the failure is
  before the first timed repeat. Also reap orphaned `sgl_diffusion::scheduler_*` workers (~30 GiB each)
  before every retry; KFD PIDs outside the container's PID namespace are NOT reapable — treat those as
  hard external occupancy.
- caution: **also verify that a "fix" does not void comparability.** Lowering TP, dropping a card,
  `--dit-cpu-offload/--text-encoder-cpu-offload True`, or FSDP-sharded inference all fit the model in
  less VRAM but break the run-wide serving invariant, so every delta becomes incomparable to the
  baseline — report `incomplete`, never a number measured under a changed config. Two knobs that ARE
  invariant-safe and did move the failure point: `SGLANG_USE_RUNAI_MODEL_STREAMER=0` (the streamer's
  eager `dist.new_group()` NCCL connect fails a 6 MB calloc on a squeezed card and aborts the
  transformer load; disabling it is load-path-only and identical in both legs). `expandable_segments`
  is a no-op on this ROCm build, and `--mem-fraction-static` is inapplicable (diffusion has no KV cache).
- **sequencing consequence:** on a shared box, do all offline-verifiable work first (build the overlay,
  prove the seam binds in every rank, prove bit-exactness/1-ULP vs the frozen oracle on ONE
  optimization-pool card) so the blocked step is only the A/B — a verified isolated win with a pending
  e2e leg is `incomplete`, NOT `rejected`.
- source: exp/e2e_FLUX-1-dev_20260729_132053_288743_23791 (h0/h1/k0/k1, 2026-07-30 → 07-31)
