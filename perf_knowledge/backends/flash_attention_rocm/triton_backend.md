---
title: FlashAttention-ROCm — Triton backend (aiter kernels)
kind: backend
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp32, fp8_e4m3]
regimes: [prefill, decode, training]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/Dao-AILab/flash-attention
  - https://github.com/ROCm/aiter
  - https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
---

# FlashAttention-ROCm — Triton backend

## TL;DR
The Triton backend implements FlashAttention-2 with kernels from **`ROCm/aiter`** (`third_party/aiter`
submodule). It is **more feature-rich than CK**: fp16/bf16/**fp32** + **fp8** (FA-v3 interface),
**arbitrary head dims**, and causal / varlen / MQA-GQA / dropout / rotary / **ALiBi** / paged attention. In
**vLLM it is the default on ROCm**. Enable it at the FA level with
`FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"` (install **and** runtime). The main gaps: **sliding-window is
WIP**, and AMD Triton can hit register/shared-mem lowering issues (e.g. Phi3V compile failure → fall back
to CK). For pure decode/MLA, the newer AITER FA backends usually beat it.

## Concepts
- **Kernels from aiter.** Unlike the CK backend (composable_kernel), the Triton backend's FA kernels live
  in `ROCm/aiter`, vendored at `third_party/aiter` and auto-installed.
- **Two-flag enable:** `FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"` must be set **at install** (so the path
  builds) **and at runtime** (to select it). Without it you get CK.
- **vLLM default on ROCm.** `VLLM_USE_TRITON_FLASH_ATTN` defaults to Triton FA on AMD; vLLM also falls back
  to Triton for fp32 (CK FA has no fp32).

## Capability envelope
| feature | Triton backend |
|---|---|
| dtypes | fp16, bf16, **fp32**, **fp8** (FA-v3 interface) |
| head dim | **arbitrary** (no 256 cap) |
| backward | yes (fwd + bwd) |
| extras | causal masking, variable/arbitrary seqlen, MQA/GQA, dropout, rotary, **ALiBi**, paged attention |
| sliding window | **WIP** (not yet supported) |
| GPUs | CDNA (MI200/MI300) + RDNA |

## The levers
- **Enable:** `FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pip install --no-build-isolation .`
  (pin aiter: `cd third_party/aiter && git checkout <sha>` before building).
- **Autotune:** `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE"` searches kernel configs (one-time warmup
  cost); without it a deterministic default config is used.
- **Pin a config (no autotune):** `FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON`, e.g.
  `'{"BLOCK_M":128,"BLOCK_N":64,"waves_per_eu":1,"PRE_LOAD_V":false,"num_stages":1,"num_warps":8}'`.
- **vLLM:** leave `VLLM_USE_TRITON_FLASH_ATTN` unset (default Triton) or `=1`; build with `BUILD_TRITON=1`.
- **AMD Triton GEMM/FA ISA hygiene** (general): `AMDGCN_ENABLE_DUMP=1` to confirm `global_load_dwordx4` /
  LDS `_b128`; `matrix_instr_nonkdim=16`, `waves_per_eu`, `kpack=2` (cf. `languages/triton_amd/`).

## When to use Triton
- **fp8** attention (FA-v3 interface) or **arbitrary head dim** (> 256) — CK can't.
- **ALiBi / rotary / dropout / paged / varlen** features beyond core FA-2.
- The vLLM **default** path on ROCm for general fp16/bf16 attention.
- **fp32** attention (CK FA has none → Triton/SDPA).

## Pitfalls
- **Sliding-window WIP** — for SWA models (Mistral/Mixtral/Qwen2) at half precision, switch to **CK**
  (`VLLM_USE_TRITON_FLASH_ATTN=0`).
- **Compile failures from shared-mem/register pressure** — e.g. Phi3VForCausalLM overflows shared memory in
  ROCm Triton FA → disable Triton (→ CK). General AMD-Triton register-lifetime weakness (HipKittens paper).
- **vLLM V1 may report Triton even when you asked for CK** — select the backend explicitly on V1.
- **Decode/MLA:** the dedicated AITER FA / MLA backends report **1.2–4.4× higher TPS** than generic FA;
  prefer them for decode-heavy serving. `ROCM_ATTN` can be 2.7–4.4× *slower* when KV head size is
  unsupported by HIP paged attention (falls back to Triton decode).
- Two-flag enable: forgetting it at **install** time means the kernels aren't built (runtime flag alone
  won't help).

## Verify
- `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pytest tests/test_flash_attn_triton_amd.py` (full suite = hours).
- Confirm active backend from logs (don't trust env flag alone on vLLM V1).
- Micro-bench Triton vs CK vs AITER FA at your `(B,H,S,D,causal,dtype)`, esp. for decode.

## Alternatives / cross-links
[overview.md](overview.md) · [ck_backend.md](ck_backend.md) · `languages/triton_amd/` ·
`backends/aiter/` · attention operators (`operators/attention_prefill_fmha/`,
`operators/attention_decode_paged/`, `operators/mla_attention/`).

## Sources
- FlashAttention ROCm README (Triton backend = aiter kernels; `FLASH_ATTENTION_TRITON_AMD_ENABLE`;
  fp32/fp8; arbitrary head dim; feature list; SWA WIP; autotune & config-json env vars):
  https://github.com/Dao-AILab/flash-attention
- aiter (Triton FA kernels source): https://github.com/ROCm/aiter
- vLLM ROCm attention backends (Triton default, AITER 1.2–4.4×, ROCM_ATTN fallback cliff, V1 selection):
  https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html
- AMD-Triton register-lifetime weakness: HipKittens https://arxiv.org/html/2511.08083v1 (see
  `languages/hipkittens/perf_findings.md`)
