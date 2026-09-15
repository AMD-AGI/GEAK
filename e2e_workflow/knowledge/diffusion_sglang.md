# sglang-diffusion (multimodal_gen) on AMD Instinct — what the e2e layer needs to know

Read this whenever `BACKEND=sglang_diffusion`. It covers what is structurally different about a
diffusion target versus an LLM, the tunable surface, and the priors already measured on this box.
Everything here is falsifiable by measurement — treat the priors as a starting order, not truth.

## 1. The model is not an LLM. What that changes.

| | LLM (sglang/vllm) | diffusion (sglang-diffusion) |
|---|---|---|
| metric | output tok/s | **img/s** (`output_throughput` still carries it) |
| phases | prefill (compute) + decode (memory) | text-encode → **denoise × N steps** → VAE decode |
| batching | continuous, KV cache dominates | one request = N identical DiT forwards; no KV cache |
| the hot loop | varies by phase | **the denoise loop** — same graph N times |
| correctness | greedy token parity | image parity (LPIPS/SSIM/MSE) + prompt fidelity (GenEval2) |
| ISL/OSL | sequence lengths | **image width / height** |

Consequences that matter for strategy:

- **Amdahl is unusually clean.** ~28 identical DiT forwards dominate; a kernel's share of the
  denoise loop *is* its share of e2e. There is no prefill/decode regime split to reason about, and
  no shape variety across steps — one shape regime, sampled 28 times per image.
- **There is no KV cache**, so none of the usual memory-bound decode tricks apply, and
  `--mem-fraction-static` is meaningless here.
- **A "speedup" can be fraud.** Fewer denoise steps, lower resolution, or a lossier VAE all raise
  img/s enormously and produce a worse image. This is why the gate is mandatory, not advisory:
  8 steps instead of 28 measured **+200% throughput** on this box and failed every image metric.

## 2. Mandatory config hygiene (not tuning — correctness of the measurement)

1. **`--model-id FLUX.1-dev`.** sglang resolves its native pipeline by matching the substring
   `flux.1` against the model path (`registry.py`). The local checkout is `FLUX-1-dev` — dot became
   dash — so resolution fails and sglang logs *"Could not resolve native configuration … Falling
   back to diffusers backend"* and keeps going. Measured cost of that fallback on this box
   (512², 4 steps): 5459 ms/img vs 318 ms/img = **17.2× slower**. A run missing this flag is not
   benchmarking the engine you think it is. With the flag, sglang also correctly picks Ulysses SP
   over CFG-parallel — FLUX.1-dev is guidance-distilled, so CFG-parallel is wasted work.
2. **`--dit-cpu-offload False --text-encoder-cpu-offload False`.** sglang defaults both to `true`
   for *any* image model regardless of VRAM (`server_args.py`); FLUX.1-dev is ~24 GB bf16 against
   288 GB HBM.
3. **Never set `HIP_VISIBLE_DEVICES`** on this ROCm stack — it can make `torch.cuda.is_available()`
   return false. Pin GPUs with `ROCR_VISIBLE_DEVICES`.
4. **Ports collide by default.** `--master-port` defaults to a fixed 30005, so two concurrent
   engines kill each other (`EOFError` / connection refused). `diffusion_bench.py` derives
   `31000 + 200×<first GPU>` and probes; keep that if you launch anything by hand.
5. **Hard stop on the fallback marker.** If a log says *Falling back to diffusers backend* /
   *Using diffusers backend* / *Loaded diffusers pipeline*, discard the number and the trace.

## 3. Tunable surface (Tier-0, free — sweep this before touching a kernel)

**Parallel layout, within the fixed 8-GPU budget.** `--num-gpus` is the engine's GPU count;
`--dp-size` / `--sp-degree` / `--ulysses-degree` / `--ring-degree` decide how they are used. SP
splits one image across GPUs (lower latency/image); DP runs independent images (higher aggregate
img/s for a saturating batch). For a 64-image workload these trade off directly, and which wins is
an empirical question — sweep it, do not assume. Keep the total GPU count fixed at 8 or the numbers
are not comparable.

**Attention.** On ROCm the valid `--attention-backend` values are `fa` (default), `aiter`,
`aiter_sage`, `torch_sdpa` (`runtime/platforms/rocm.py`). `--component-attention-backends` sets
them per component (e.g. `text_encoder=torch_sdpa,transformer=aiter`) — the text encoder and the
DiT have very different shapes, so the best backend need not be the same for both.

**Numerics / precision.** `--vae-precision bf16` (+3.2% measured, see §4). `--quantization` for the
transformer (fp8 on gfx950 is the interesting one) — parity-breaking, so it must go through the
image gate and GenEval2, not just the throughput gate. `--disable-autocast`.

**Compilation.** `--enable-torch-compile` — worth trying, but TorchInductor compiles FLUX's **57
transformer blocks**, which took longer than a 40-minute per-variant cap in the previous run and
was killed before rendering a single image. Budget it explicitly (≥90 min) or it will look like a
failure when it is a timeout.

**Cache-DiT — the biggest unbanked lever on this box. Read §4a before touching it.** Enabled by
**env, not by the flag**: `SGLANG_CACHE_DIT_ENABLED=true` (`--cache-dit-config` is the
diffusers-backend path and is `null` on the native pipeline). It skips redundant DiT computation
between denoise steps and has its own quality/speed frontier:
`SGLANG_CACHE_DIT_RDT` (residual-difference threshold, default 0.24 — the main dial),
`SGLANG_CACHE_DIT_MC` (max continuous cached steps, 3), `SGLANG_CACHE_DIT_FN`/`_BN` (first/last
blocks always computed, 1/0), `SGLANG_CACHE_DIT_WARMUP` (4), `SGLANG_CACHE_DIT_TAYLORSEER` +
`_TS_ORDER`, and the `_SCM_*` policy/preset knobs.

**Other env levers worth a sweep slot:** `SGLANG_USE_ROCM_VAE`, `SGLANG_USE_ROCM_VAE_CONV2D`,
`SGLANG_USE_ROCM_VAE_CONV2D_BF16` (the ROCm VAE fast paths — related to the `--vae-precision bf16`
win), `SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D`, `SGLANG_USE_ROCM_CUDNN_BENCHMARK`,
`SGLANG_DIFFUSION_USE_PRECOMPILED`, `SGLANG_USE_RUNAI_MODEL_STREAMER` (load time only — it does not
affect the timed pass), `SGLANG_DIFFUSION_ATTENTION_BACKEND` (env twin of the flag).

**Offload/memory (leave off, but they exist):** `--dit-layerwise-offload`,
`--dit-offload-prefetch-size`, `--vae-cpu-offload`, `--image-encoder-cpu-offload`,
`--pin-cpu-memory`, `--use-fsdp-inference`. **Presets:** `--performance-mode {manual,auto,speed,memory}`.
**Warmup:** `--warmup`, `--warmup-steps`, `--warmup-resolutions`.

**Env:** the usual aiter levers (`SGLANG_USE_AITER`, `GPU_ARCHS`), plus
`SGLANG_DIFFUSION_TORCH_PROFILER_DIR`. `envs.py:232` documents an override to force a specific
attention backend.

## 4. Measured priors on THIS box (MI350X/gfx950, sglang 0.5.12, ROCm 7.2.0)

Workload: FLUX.1-dev, 1024², 28 steps, guidance 3.5, seed 42, bf16.

| fact | value | source |
|---|---|---|
| **SP=8 (8 GPUs), local serial** | **0.475 img/s**, 2106 ms/img, 75.2 ms/step | this run, 64 images × 3 repeats, spread **0.6%** |
| **SP=8 (8 GPUs), serve @ conc=64** | **0.443 img/s** | this run — concurrency adds nothing to ONE SP engine; the two protocols agree |
| baseline, SP=2 (2 GPUs) | 0.3201 img/s, 3124 ms/img, ~0.125 s/step | prior run, 64 timed images |
| run-to-run spread | 0.12% (std 3.6 ms) | prior run — a 10% win is ~28σ, so single runs are decisive |
| same config, different GPU pair | 0.3214 img/s, MSE 2.8e-10, SSIM 0.9999998 | determinism is real |
| **image parity, identical config** | **LPIPS 0.0 / SSIM 1.0 / MSE 0.0** (bit-exact, 15 prompts) | this run — the gate's noise floor is ZERO on the same GPU set |
| **image parity, 8-step canary** | **LPIPS 0.671 / SSIM 0.402 / MSE 0.0548 → FAIL** | this run — the gate bites; reproduces the prior run's 0.659/0.412/0.046 |
| **`--vae-precision bf16`** | **+3.2%** (0.3203 → 0.3305 img/s) | the only accepted win of the prior run |
| 8 denoising steps | +200% img/s, LPIPS 0.659 / SSIM 0.412 / MSE 0.046 | the gate rejects it — keep it as a canary |
| diffusers fallback | 17.2× slower | see §2.1 |
| GenEval2 Soft-TIFA, 64 prompts | **55.44 / 100** | Qwen3-VL-8B-Instruct scorer, ~37 s to score |
| engine load | ~23 s | inside every bench call (excluded from the timed pass) |

The previous run explored config only and banked +3.2%. **The kernel track is unexplored.**

## 4a. Cache-DiT: strong evidence, and the gate it needs

Measured in the previous run (SP=2, 1024², 28 steps, seed 42, `SGLANG_CACHE_DIT_ENABLED=true`,
otherwise identical flags):

| | baseline | cache-dit | |
|---|---|---|---|
| generation, 800 GenEval2 prompts | 2733 s | **1317 s** | **2.07× faster** |
| GenEval2 Soft-TIFA, 800 prompts | 23.21 | **23.59** | no fidelity loss (slightly higher) |
| generation, 64 prompts | 220 s | 109 s | 2.02× faster |

It is by far the largest known lever here, and **it was never banked into a recipe** — test it
early. Its knobs (`_RDT`, `_MC`, `_FN`/`_BN`, `_WARMUP`, TaylorSeer) trade quality for speed
continuously, so there is a frontier to search, not one on/off decision. Note the prior evidence
is at the DEFAULT settings (F=1, B=0, W=4, R=0.24, MC=3) — a more aggressive R/MC will eventually
break fidelity, which is exactly what the gate is for.

**Cache-DiT is an approximation, so image parity is the WRONG gate for it.** It deliberately
changes the output — LPIPS/SSIM vs the baseline's own images will fail, and that failure means
nothing. It is the diffusion analogue of a quantized kernel: the right bar is *task* quality.
Gate it (and quantization, and anything else that intentionally perturbs numerics globally) on
**GenEval2 at 800 prompts**, with `num_inference_steps` / resolution / guidance / seed unchanged.

**Scorer trap (already handled by `scripts/geneval2_gate.py`, but know it).** `sglang serve` returns
**JPEG** bytes, which `evaluate_flux.py` saves under `.png` names. The GenEval2 scorer loads images
via transformers → torchvision `decode_image`, and this container's torchvision has **no libjpeg**,
so scoring dies with `decode_jpeg: torchvision not compiled with libjpeg support` — *after* the full
800-image generation pass has been paid for. The gate script therefore generates and scores as two
calls with a PNG re-encode in between. If you drive `evaluate_flux.py` yourself, do the same (or use
`--gen-backend cli`, which writes real PNGs).

**GenEval2 noise — do NOT gate on the 64-prompt subset.** Same config, seed 42 → 55.44; seed 43 →
51.04. That is a **4.4-point swing from the seed alone**, far larger than any effect you are trying
to detect. The 64-prompt subset is a smoke test (does the pipeline still make sensible images?).
Comparisons that decide accept/reject must use the full 800 (~22 min of generation at SP=2; less on
8 GPUs), where base 23.21 vs cache-dit 23.59 was resolvable.

## 4b. Sequence parallelism scales badly — measured, and it sets the strategy

| layout | img/s | s/image | GPUs | img/s per GPU |
|---|---|---|---|---|
| SP=2 (`--num-gpus 2`) | 0.3201 | 3.12 | 2 | 0.160 |
| SP=8 (`--num-gpus 8`) | **0.475** | 2.10 | 8 | **0.059** |

Four times the GPUs buys **1.49×** throughput. A profile at SP=2 shows why: **34% of GPU time is
`ncclDevKernel_Generic_1`** — the Ulysses all-to-all. SP buys latency, and pays for it in
communication that grows with the degree.

### Data parallelism is NOT available in sglang 0.5.12 — do not spend sweep slots on it

Two separate facts, both verified on this box, and the first one is a trap:

1. **`--dp-size` / `--data-parallel-size` / `--dp` are silently ignored.** All three land in the
   argparse dest `data_parallel_size`, but the `ServerArgs` field is `dp_size`, and
   `ServerArgs.from_dict()` copies namespace keys **by field name** — so the value is dropped.
   Launching with `--dp-size 8` resolved to `dp_size: 1, sp_degree: 8`: you get SP, and the log
   looks perfectly healthy. A variant that "tests DP" this way is re-testing SP and will report a
   ~0% delta. **If you ever need the field to actually take, use `sglang serve --config <json>`
   with `{"dp_size": N}`** (`load_config_file()`'s dict is merged by field name) or set the
   attribute directly (what `diffusion_bench.py` does for `SGLD_DP_SIZE`).
2. **Once the value does take, the engine rejects it:** `ValueError: DP is not yet supported`
   (`server_args.py:_validate_parallelism`). It is unimplemented, not misconfigured.

So on this stack a single engine can only spend its GPUs on **sequence parallelism**, whose scaling
is the poor curve above. Real replica parallelism means **K independent engines/servers** at the
deployment layer (K × 2 GPUs at 0.16 img/s each ≈ 1.28 img/s for 8 GPUs ≈ 2.7× the SP=8 number).
That is a deployment-topology recommendation, outside what one `sglang serve` process can be
configured to do — measure it and report it, but the in-engine optimization target stays the
single-engine configuration.

**Where the single-engine headroom therefore is:** cache-dit (§4a), attention backend choice, the
ring/ulysses split of SP, torch.compile, VAE fast paths, quantization, and the kernel track.

## 5a. Two measurement modes

`SGLD_MODE=serve` (default) = `sglang serve` + sglang's diffusion `bench_serving` at
`--max-concurrency CONC`: the **throughput** protocol, and the only one in which DP is visible.
`SGLD_MODE=local` = one in-process engine, images back-to-back: a **latency** protocol
(concurrency 1), used for profiling and cross-checks. Never mix the two in one comparison.

## 5. Profiling

There is no live `/start_profile` on the diffusion HTTP server. Profiling is in-process: pass
`profile=True` in the sampling params (the adapter does this for `PROF=1`) and
`SGLDiffusionProfiler` writes `<request_id>-<mode>-global-rank<N>.trace.json.gz` into
`SGLANG_DIFFUSION_TORCH_PROFILER_DIR`. It is a stock `torch.profiler` trace with
**`record_shapes=True`**, so `scripts/parse_profile.py` ingests it unchanged (it already handles
`.json.gz`) and the Top-N carries shapes for extraction.

`num_profiled_timesteps` bounds the capture (default 5 in the adapter). More steps do not add
information — every denoise step runs the identical graph — they only grow the trace, and
`with_stack=True` is on, so traces get large fast. Use `profile_all_stages` only when you
specifically need text-encode or VAE-decode in frame; the default (denoise only) is the right
default because that is where the time is.

**Rank note:** only rank 0 dumps by default. With SP>1 the per-rank work is a shard of the image,
so the trace shows one rank's share — read percentages, not absolute totals, and be alert to
collective ops (all-to-all in Ulysses attention) appearing as "communication" time that a
single-rank view under-reports.

## 6. Where the time actually goes (expected, verify with the profile)

FLUX.1-dev is a 12B rectified-flow transformer: 19 dual-stream (MMDiT) blocks + 38 single-stream
blocks = the 57 blocks torch.compile struggles with. Per denoise step, expect: QKV/proj/MLP GEMMs,
attention (aiter `fmha_v3_fwd` on gfx950 — visible in the smoke log), RMSNorm/QK-norm, the
modulation (AdaLN) elementwise chain, and RoPE. The smoke run also logged *"FlashInfer not
available, using Triton fallback for RoPE"* — a Triton kernel on the hot path with no tuned
alternative is exactly the shape of an editable-kernel target. VAE decode is once per image, not
per step, so its Amdahl share is ~1/28 of a denoise step's — do not spend the budget there unless
the profile says otherwise.

## 7. References

- Skill: `/sgl-workspace/sglang/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/`
  (`benchmark-and-profile.md`, `existing-fast-paths.md`) — **check `existing-fast-paths.md` before
  declaring any hotspot "new"**: fused QK-norm+RoPE, packed QKV, Ulysses/USP overlap and several
  other families already exist upstream.
- `docs.sglang.io/docs/sglang-diffusion/compatibility_matrix` — what is supported per platform.
- `rocm.blogs.amd.com/artificial-intelligence/sglang-diffusion/README.html` — AMD's own tuning notes.
