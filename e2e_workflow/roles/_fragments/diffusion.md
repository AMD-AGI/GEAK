# DIFFUSION MODE — read this before your role file's instructions bite

This run's target is a **text-to-image diffusion model** served by **sglang-diffusion**, not an
LLM. Your role file (`roles/<role>.md`) is written for token serving. Everything in it still
applies *structurally* — TRUE baseline, warm repeated medians, Amdahl routing, reversible overlays,
gate-before-accept — but four of its assumptions are wrong here, and this fragment overrides them.

Read `knowledge/diffusion_sglang.md` too. It has the flag surface and the numbers already measured
on this box.

## Override 1 — the metric is img/s

`bench_e2e.sh` is metric-neutral: it medians whatever `output_throughput` the adapter reports, and
the JSON keys keep their `..._tok_s` names. **For this run every one of those numbers is images per
second.** Never write "tok/s" in a report, a strategy note, or a playbook entry for this run.
`ttft_ms` is time to the single output image; `tpot_ms` is per-denoise-step time. `ISL`/`OSL` are
the image **width/height** (1024/1024 → 1024×1024), and `CONC` is the number of timed images per
repeat, not in-flight requests.

## Override 2 — there is no prefill/decode split

The pipeline is text-encode → **denoise × N steps** → VAE decode. The denoise loop runs the
identical DiT graph N times (N=28) and dominates. So:

- A kernel's share of the denoise loop **is** its share of e2e — Amdahl here is unusually direct,
  and `pct_gpu_time` from the Top-N can be trusted more than it can for an LLM.
- There is one shape regime, not two. Do not look for prefill-vs-decode variants of a kernel.
- There is **no KV cache**. `--mem-fraction-static`, KV-layout and paged-attention reasoning are
  all inapplicable.
- VAE decode runs once per image, the DiT runs 28 times. Weight the two accordingly before
  spending budget.

## Override 3 — correctness is images, not tokens

Greedy token parity does not exist here. The gate is, in order of cost:

1. **Image parity — every candidate.** `scripts/image_parity.py` compares the candidate's 15 fixed
   prompt images (fixed seed) against the baseline's own images: worst-case LPIPS ≤ 0.15,
   SSIM ≥ 0.85, MSE ≤ 0.006. The bench writes the candidate set when you export
   `GEAK_PARITY_REF=<baseline ref template>` (and the baseline writes it with
   `GEAK_PARITY_REF_WRITE=<template>`). It fails closed. Report `parity_kind:"image_parity"`.
   On this box an identical config reproduces the reference to ~1e-10 MSE, so a real failure is
   unambiguous — do not rationalise one.
2. **GenEval2 prompt fidelity — endpoints, and every approximation.** `scripts/geneval2_gate.py`
   scores Soft-TIFA with Qwen3-VL. Pixel metrics only compare a candidate to the baseline's own
   images; they cannot see that the image stopped matching the *prompt*. Run it at the TRUE
   baseline and on the final winner.

**When image parity is the WRONG gate.** Some accelerations change the output *on purpose* —
cache-dit (step caching) and quantization are the two live ones here. For those, an image-parity
failure carries no information, exactly as byte-parity failure carries none for an MXFP8 kernel.
Gate them on **GenEval2 with `--limit 0` (all 800 prompts)** instead, with steps/resolution/
guidance/seed held identical, and report `parity_kind:"accuracy"`. Do NOT gate on the 64-prompt
subset: the same config at seed 42 vs 43 scores 55.44 vs 51.04, so a 4.4-point swing is pure noise
there. At 800 prompts the measurement is stable enough to decide (base 23.21 vs cache-dit 23.59).

**A throughput win that degrades the image is not a win.** Fewer denoise steps, smaller resolution,
a lossier VAE, or a shorter schedule all raise img/s a lot and are all fraud in this context. If a
candidate's `num_inference_steps`, `width`, `height`, `guidance_scale` or `seed` differ from the
baseline's, reject it on those grounds alone — before even looking at the number. The canary is
known: 8 steps instead of 28 posts +200% and fails every image metric.

## Override 4 — two measurement modes, and which one to use

The adapter has an `SGLD_MODE` switch. **Use the same mode for every number you compare.**

- **`SGLD_MODE=serve` (default, and what the TRUE baseline uses).** A real `sglang serve` process
  plus sglang's own diffusion `bench_serving`, driving **CONC concurrent image requests**. This is
  the throughput protocol. Warmup is `--warmup-requests`.
- **`SGLD_MODE=local`.** Server-less: one in-process engine generating images back-to-back.
  Concurrency is 1 by construction, so this measures **per-image latency**, not throughput. Use it
  for profiling and as a cross-check, never mixed into a throughput comparison.

**Profiling always runs through the local engine** (`SGLD_MODE=local PROFILE=1`): the diffusion
server exposes no profiler hook, and a single-request trace of the denoise loop is exactly what the
Top-N needs. If you set `PROFILE=1` in serve mode the adapter stops the serve process first, so the
two engines never contend for the same GPUs — but prefer a dedicated `SGLD_MODE=local PROFILE=1`
invocation. sglang writes a stock `torch.profiler` `.trace.json.gz` with `record_shapes=True` into
`PROFILE_DIR`, which `scripts/parse_profile.py` parses unchanged. With SP>1 only rank 0 dumps —
read percentages, not absolute totals, and remember the NCCL all-to-all is a real cost the
single-rank view under-represents.

## Do not sweep data parallelism — it is unimplemented, and the flag lies

`--dp-size` / `--data-parallel-size` / `--dp` are **silently dropped** by sglang 0.5.12 (argparse
dest `data_parallel_size` vs dataclass field `dp_size`), so a "DP variant" launched that way is
really running SP and will post a ~0% delta that means nothing. If you force the value through
(`--config '{"dp_size":N}'`), the engine raises `ValueError: DP is not yet supported`. Both were
verified on this box. Spend those slots on cache-dit, the attention backend, the ring/ulysses split,
torch.compile, the VAE fast paths, quantization, or the kernel track instead.

## Preflight notes specific to this target

There is **no `config.json`** — a diffusers-style checkout describes itself in **`model_index.json`**
(plus per-component configs under `transformer/`, `vae/`, `text_encoder{,_2}/`, `scheduler/`). Read
that for the arch class; do not treat its absence as a failed preflight. FLUX.1-dev is a ~12B
rectified-flow MMDiT: 19 dual-stream + 38 single-stream blocks, bf16, ~24 GB of weights, with T5 +
CLIP text encoders and a separate VAE — several distinct components, not one transformer stack, so
say *which* component a hot kernel belongs to.

## Non-negotiable launch flags

Every bench, every variant, every hand-run command:
`--model-id FLUX.1-dev --dit-cpu-offload False --text-encoder-cpu-offload False`.
Without `--model-id`, sglang cannot resolve the native pipeline from a path spelled `FLUX-1-dev`
and **silently falls back to the diffusers backend — 17.2× slower**. The adapter injects these as
`SGLD_BASE_ARGS` before your `EXTRA_SERVER_ARGS`, so a variant can still override one deliberately;
if you launch anything by hand, include them. If any log shows *"Falling back to diffusers
backend"*, the number and the trace are void — fix it, do not report it.
