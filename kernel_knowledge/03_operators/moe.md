# Mixture-of-Experts (MoE) Compute on AMD MI300X (CDNA3 / gfx942)

> Scope: the full **fused MoE** layer on MI300X — gate → top-k → align/sort/permute → grouped/expert
> GEMM → activation → unpermute → combine. AMD-only (gfx942). This is the dominant op in modern MoE LLMs
> (DeepSeek-V3/R1, Mixtral, Qwen-MoE, Kimi-K2, GLM-MoE): in profiles it routinely accounts for
> **80–90% of GPU time** (Kimi-K2.5: fused_moe was 88–90% of total GPU time). It is the #1 Amdahl lever.
> See `grouped_gemm.md` (expert GEMM core), `moe_routing_ep.md` (routing math + expert parallelism),
> and `gemm.md` (MFMA inner loop).

---

## 1. The fused_moe pipeline

For `T` tokens, hidden `H`, `E` experts, top-`k`, intermediate `I`, the MoE layer computes per token:
`y = sum_{e in topk(x)} g_e · down_e( act( gate_e(x) ) ⊙ up_e(x) )`.

The stages (and where each runs on MI300X):

```
x:[T,H]
 │ 1. GATE GEMM        x · W_gate -> logits[T,E]            (small dense GEMM)
 │ 2. TOP-K + SOFTMAX  pick k experts/token, normalize      -> topk_ids[T,k], topk_w[T,k]
 │ 3. ALIGN & SORT     sort tokens by expert, pad to block  -> sorted_token_ids, expert_ids(per-block),
 │                                                              num_tokens_post_pad
 │ 4. PERMUTE/GATHER   gather x rows into expert-contiguous  -> x_sorted[T*k(+pad), H]
 │ 5. EXPERT GEMM 1    grouped GEMM up/gate per expert       -> h[ , 2I]   (W1:[E,H,2I])
 │ 6. ACT (SwiGLU)     SiLU(gate) ⊙ up [+ fp8 quant]         -> a[ , I]
 │ 7. EXPERT GEMM 2    grouped GEMM down per expert          -> o[ , H]    (W2:[E,I,H])
 │ 8. UNPERMUTE+COMBINE scatter back, weight by g_e, sum k   -> y[T,H]
 └ (+ SHARED EXPERT: a dense FFN every token also passes through, added to y)
```

The "fused" in fused_moe means stages 4–8 (or subsets) are merged to avoid materializing the huge
`[T·k, *]` intermediates in HBM. The hot kernels are stages **3 (align/sort)**, **5+7 (grouped GEMM)**,
and **8 (combine)**.

---

## 2. The Triton fused_moe kernel core (vLLM/SGLang)

vLLM's `fused_moe` Triton kernel (originally from DeepSeek, refined by AnyScale + SGLang) does the expert
GEMM with the **sorted/aligned token layout** so each `BLOCK_M` tile belongs to exactly one expert. The
key trick: `expert_ids[pid_m]` tells the block which expert weight panel to use, and `sorted_token_ids`
gives the (gathered) row indices — the permute (stage 4) is fused into the GEMM's A-load via gather.

```python
@triton.jit
def fused_moe_kernel(
        A, B, C,                              # A:[T,H] tokens (unsorted), B:[E,N,K] weights, C:[T*k,N] out
        sorted_token_ids,                     # [num_tokens_post_pad] token row per slot (padded)
        expert_ids,                           # [num_blocks] expert id per BLOCK_M tile
        num_tokens_post_padded,
        topk_weights,                         # [T*k] routing weight per slot (for MUL_ROUTED_WEIGHT)
        N, K, EM, num_valid_tokens,
        stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn,
        MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    # ---- grouped tile ordering for L2 reuse (same as dense GEMM) ----
    num_pid_m = tl.cdiv(EM, BLOCK_M); num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # early-out for padding tiles beyond the real token count
    if pid_m * BLOCK_M >= tl.load(num_tokens_post_padded):
        return

    # ---- the gather: A rows come from sorted_token_ids (permute fused into the load) ----
    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids + offs_token_id)
    token_mask = offs_token < num_valid_tokens               # padding slots masked off
    offs_k = tl.arange(0, BLOCK_K)
    # A row = real token index = offs_token // top_k  (k replicas share a token row)
    a_ptrs = A + (offs_token // top_k)[:, None] * stride_am + offs_k[None, :] * stride_ak

    # ---- B: pick THIS tile's expert weight panel ----
    off_experts = tl.load(expert_ids + pid_m)                # which expert this BLOCK_M serves
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_ptrs = B + off_experts * stride_be + offs_k[:, None]*stride_bk + offs_bn[None, :]*stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)     # fp32 accumulate in VGPRs
    for k in range(0, tl.cdiv(K, BLOCK_K)):                  # MFMA inner loop -> v_mfma_f32_16x16x16_*
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k*BLOCK_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k*BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if MUL_ROUTED_WEIGHT:                                    # fold routing weight g_e into the output
        w = tl.load(topk_weights + offs_token, mask=token_mask, other=0.0)
        acc = acc * w[:, None]
    c = acc.to(C.dtype.element_ty)
    c_ptrs = C + offs_token[:, None]*stride_cm + offs_bn[None, :]*stride_cn
    tl.store(c_ptrs, c, mask=token_mask[:, None] & (offs_bn[None, :] < N))
```

Notes:
- This kernel is called **twice** — once for the up/gate GEMM (`B=W1`, N=2I), once for down (`B=W2`,
  N=H), with the SwiGLU activation (stage 6) between them (its own fused kernel or folded into the
  down-GEMM A-load).
- The **gather is the permute**: `a_ptrs` reads token rows in expert-sorted order without a separate
  scatter buffer. `offs_token // top_k` recovers the original token row (k slots share one input row).
- `MUL_ROUTED_WEIGHT` folds the combine weight `g_e` into stage 7's epilogue, deleting a separate
  multiply in stage 8.
- The `expert_ids`/`sorted_token_ids`/`num_tokens_post_padded` triple comes from **MoE align & sort**
  (next section) — the kernel is meaningless without it.

---

## 3. MoE align & sort (stage 3) on MI300X — the hidden bottleneck

Sorts tokens by expert id **and pads each expert's group to a multiple of `block_size`** so every GEMM
tile maps to one expert. This is *not* a plain radix sort: `cub::DeviceRadixSort` can't do the alignment
padding that makes the grouped GEMM efficient. SGLang's custom CUDA/HIP kernel beats the Triton baseline
by **7× on MI300X** (10× on MI100).

Output triple:
- `sorted_token_ids[num_tokens_post_pad]` — token indices, expert-ordered, each expert region padded to
  `block_size`, padding filled with a sentinel (≥ num_tokens) that the GEMM masks.
- `expert_ids[num_blocks]` — one expert id **per BLOCK_M block** (what the GEMM reads as `off_experts`).
- `num_tokens_post_pad` — scalar total after padding = `sum_e ceil(M_e/block_size)*block_size`.

Algorithm (SGLang D-C-P, multi-block): **D**istributed counting → **C**umsum (local → block-reduce →
align to block_size → store) → **P**arallel placement. Replaces the old single-block alignment kernel
(serialized at ~33W cycles) with a multi-block scheme (~20W total).

**MI300X-specific wins (from the SGLang work):**

| Aspect | Value / rule |
|---|---|
| LDS used | ~5 KB; bank-conflict rate only 6.8% |
| Registers | 52 VGPR / 48 SGPR, **zero spills** |
| Max experts | `MAX_EXPERT_NUMBER == 256` (concurrent multi-block) |
| Active CUs | 66 (new) vs 39 (old) |
| vL1D hit rate | 61% (new) vs ~0% (old k1) |
| **XCD awareness** | grid size = **multiple of #XCD (8)**; or use lowest-speed die (XCD7) when grid < 8 |
| Cross-die sync | expensive — avoid `hipCooperativeLaunch` patterns that raise die-die L2 traffic |
| load/compute split | CDNA3: **1 warp load / 1 warp compute** (64-lane SIMD) vs NV 2/1 |

> Surprising finding: for this **memory-bound** op, **MI100 beats MI300X** — the multi-die XCD
> interconnect overhead hurts a bandwidth-bound sort. Lesson for any MoE bookkeeping kernel on gfx942:
> minimize cross-XCD communication; keep work XCD-local; size grids to multiples of 8.

---

## 4. Activation (stage 6): fused SwiGLU + fp8 quant

Between the two expert GEMMs, gated MLPs need `a = SiLU(h[:, :I]) ⊙ h[:, I:]`, often followed by **fp8
quantization** so the down-GEMM runs in fp8. CUDA has a fused `persistent_masked_m_silu_mul_quant`
kernel; **on ROCm that C++ kernel is `#ifndef USE_ROCM`-guarded → a no-op, so vLLM falls back to a Triton
kernel** matching its precision (correct but less tuned than a dedicated HIP kernel — an open gap, ROCm
aiter issue #2420). The fused op does, in one launch:

```
SiLU(gate_half) ⊙ up_half            # over a [E, T, 2I] (or masked-M) tensor
 → per-group fp8 quant (group_size=128)   # block-scale, multiple scale layouts
```

On MI300X prefer **aiter**'s fused SiLU+Mul+quant where available; else the Triton fallback. Folding the
fp8 quant here means stage 7 reads fp8 weights *and* fp8 activations → 2614 vs 1307 TFLOPS and half the
HBM traffic.

---

## 5. Shared experts

DeepSeek-style MoE has a **shared expert**: a dense FFN every token passes through, added to the routed
output. Two scheduling choices on MI300X:
- **Separate dense GEMM** (simplest): run the shared FFN as a normal dense GEMM (`gemm.md`) in parallel
  with the routed path, add at the end. Easy; one extra kernel.
- **Overlapped**: with async EP backends (DeepEP), overlap the shared-expert compute with the
  dispatch/combine all-to-all (the comm has spare compute). vLLM exposes shared-expert overlap with the
  async all2all backends.

---

## 6. fp8 / fp4 / int MoE

| dtype | Peak | MoE notes (gfx942) |
|---|---|---|
| bf16/fp16 | 1307 TFLOPS | baseline; per-token route weights |
| **fp8 (block-scale G128)** | **2614** | DeepSeek-V3 native; aiter block-scale fused MoE = **up to 3×**; best accuracy/speed balance |
| fp8 per-tensor | 2614 | cheapest scaling; small accuracy cost |
| int8 / int4 | 2614 / — | aiter FP8/INT4 per-tensor fused MoE; W-only int4 dequant path |
| **mxfp4 (W4A16/W4A4)** | gfx950 native; gfx942 dequant | Kimi-K2.5 used W4A16+BF16 fused MoE (FlyDSL) to replace the Triton path; vLLM MXFP4 w4a4 MoE on CDNA4 |

Quant formats (the `vllm` MoE feature matrix): **G** grouped, **G(128)** block-scale 128, **A** per-token,
**T** per-tensor. The Triton expert kernel supports all of G/A/T; aiter rocm MoE supports
mxfp4 + fp8 with G(32)/G(128)/A/T.

---

## 7. The fused_moe modular design (vLLM)

vLLM factors MoE into **prepare → experts → finalize** so any all-to-all dispatch backend can pair with
any expert kernel:

```
FusedMoEPrepareAndFinalize  (dispatch/combine; quant may happen here)
    │  prepare()   -> (possibly quantized, possibly batched) activations
FusedMoEExperts             (permute → grouped/expert GEMM → act → unpermute)
    │  apply()
    └  finalize() -> combine, weight by g_e, return standard-format activations
```

Expert-kernel × dispatch-backend menu:

| Expert kernel | Input fmt | Quant | Notes |
|---|---|---|---|
| **triton** | standard | all (G/A/T) | the §2 kernel; editable; default fallback |
| **triton batched** | batched | all | pairs with DeepEP-low-latency |
| **deep_gemm** | std/batched | fp8 G(128) | block-scale; strong prefill |
| **cutlass_fp8/fp4** | std/batched | fp8 / nvfp4 | NV-leaning |
| **rocm aiter moe** | standard | mxfp4, fp8 (G32/G128/A/T) | the AMD tuned path; **first try on gfx942** |
| **marlin** | std/batched | uint4/8, fp8, fp4 | weight-only quant |

| Dispatch (`--all2all-backend`) | Output | Quant | Async |
|---|---|---|---|
| naive | standard | all | N |
| **deepep_high_throughput** | standard | fp8 G(128)/A/T | Y (prefill EP) |
| **deepep_low_latency** | batched | fp8 G(128)/A/T | Y (decode EP) |
| flashinfer_nvlink | standard | nvfp4/fp8/bf16 | N |

(See `moe_routing_ep.md` for the dispatch/combine kernels themselves.)

---

## 8. Backend ladder (MI300X MoE)

| Tier | Mechanism | Edit? |
|---|---|---|
| A — backend select | bench aiter vs CK vs Triton fused_moe on the routed shapes | no |
| B — tune | autotune the chosen path (Triton configs / CK instance / aiter env) | no |
| C — rewrite | edit the Triton fused_moe / align-sort / act kernels | yes |
| D — quant | fp8 / mxfp4 experts + kv | flag → accuracy gate |

| Backend | Enable / notes |
|---|---|
| **aiter fused_moe** | `VLLM_USE_AITER_MOE=1`; SGLang `SGLANG_ROCM_AITER_BLOCK_MOE=1` + `CK_BLOCK_GEMM=1`; block-scale fp8 = up to **3×**; first try on gfx942 |
| **CK / ck_tile fused-MoE** | hand-asm inner loop; strong but rigid; tune instance |
| **Triton fused_moe** (sglang/vllm) | default; needs tuning on MI300X to reach peak; the editable Tier-C path; ship a tuned config json per (E,H,I,topk,dtype) |
| **FlyDSL** (emerging) | used for Kimi-K2.5 W4A16 fused MoE replacing Triton on MI300X |

> Coverage caveat: out-of-the-box SGLang/vLLM uses the **Triton** fused_moe, which is *correct but
> under-tuned* on MI300X. The big wins are (1) flip to **aiter** for covered shapes, (2) supply a tuned
> Triton config for uncovered ones, (3) fp8 block-scale. Always Tier-A bench — aiter may fall back to
> generic Triton for an uncovered (E,H,I) combo.

---

## 9. Tuning-knob table (Triton fused_moe on gfx942)

| Knob | Range | Effect |
|---|---|---|
| `BLOCK_M` | 16/32/**64** | **must equal** MoE-align `block_size`; small for decode skew |
| `BLOCK_N` | 64–256 | N = 2I (up/gate) or H (down) tile |
| `BLOCK_K` | 32–128 | K = H or I; larger for skinny-M decode |
| `GROUP_M` | 1–8 | L2 reuse of expert weight panels (256 MB Infinity Cache) |
| `matrix_instr_nonkdim` | **16**, 32 | MFMA; 16 wins (see `gemm.md`) |
| `num_warps` | 4, 8 | 8 prefill, 4 decode |
| `num_stages` | 1, **2** | LDS pipeline |
| `waves_per_eu` | 1–4 | hit occupancy band |
| `kpack` | 1, **2** | LDS→MFMA feed |
| align `block_size` | 32/64 | XCD-aware; pairs with BLOCK_M |
| quant | bf16/fp8-G128/mxfp4 | Tier-D; accuracy gate |
| `--all2all-backend` | deepep_ht/_ll | EP regime (prefill vs decode) |

Ship the winning Triton config as a JSON keyed by `(num_experts, H, I, topk, dtype)` (the
`E=…,N=…,device_name=…` config files in vLLM/SGLang) — these are the per-shape tuned dicts that close
most of the gap to aiter.

---

## 10. Profiling checklist

- Confirm fused_moe really is the top kernel (it usually is, ~80–90%) before spending budget elsewhere.
- align/sort: grid = multiple of 8 (XCD); check zero VGPR spills, LDS conflict < ~7%, active CUs ≥ 60.
- expert GEMM: all 304 CUs busy despite skew (else split-K the hot expert / raise GROUP_M sharing).
- act: confirm the fused SiLU+Mul(+quant) path is taken, not a 3-kernel fallback (ROCm gap — check for
  the Triton fallback vs a dedicated kernel).
- fp8: block-128 scales applied in the fp32 epilogue; run an **accuracy gate** (gsm8k / task eval), not
  byte parity — cross-backend bf16/fp8 reductions are not byte-identical.

---

## Sources
- vLLM Fused MoE Kernel Features (modular prepare/experts/finalize, backend × quant matrix): https://docs.vllm.ai/en/latest/design/moe_kernel_features/
- AITER: AI Tensor Engine for ROCm (fused_moe, block-scale fp8 3×, quant formats): https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
- Efficient MoE Align & Sort design in SGLang (D-C-P, 7× MI300X, XCD awareness, VGPR/LDS): https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang
- Accelerating Kimi-K2.5 on MI300X: optimizing fused MoE with FlyDSL (88–90% GPU time, W4A16): https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
- ROCm/aiter issue #2420 — fused SiLU+Mul+FP8-Quantize gap on ROCm: https://github.com/ROCm/aiter/issues/2420
- The vLLM MoE Playbook: TP, DP, PP, EP on AMD (ROCm Blogs): https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html
- vLLM fused_moe Triton kernel source (sorted_token_ids/expert_ids gather): https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
- Accelerated LLM Inference on AMD Instinct with vLLM 0.9.x and ROCm (fp8/MoE): https://rocm.blogs.amd.com/software-tools-optimization/vllm-0.9.x-rocm/README.html
