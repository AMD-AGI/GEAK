# Kernel Fusion Patterns for LLM Inference — AMD Instinct MI300X (CDNA3 / gfx942)

> Scope: AMD only. Target MI300X (gfx942, CDNA3); gfx950 (CDNA4) notes inline. Fusion is the single highest-leverage software optimization for LLM inference on MI300X because the decode path is **HBM-bound**: most non-GEMM ops (norms, RoPE, activations, quant) are pure memory traffic. Fusing them into a neighbor collapses HBM round-trips.
>
> **The Amdahl framing:** if a memory-bound op is X% of GPU time and fusing it into a GEMM makes it ~free, you remove X% directly. A model is a chain of memory-bound neighbors around compute-bound GEMMs — fusion turns the chain into a few fat kernels, and the un-fused tail (`1 − Σ fused%`) bounds your max speedup. Profile first (`profiling_roofline.md`), fuse the biggest memory-bound contributors.

---

## Why fusion wins on MI300X (the numbers)

A `y = act(x @ W + b)` MLP layer, unfused, does:
1. GEMM: read A,B → write `tmp` (HBM).
2. bias+act: read `tmp` → write `tmp2` (HBM).
3. (quant): read `tmp2` → write `q` + scales (HBM).

Each pointwise pass is a full HBM write **and** read of the activation. At ~5.3 TB/s, for memory-bound layers these passes can cost **more than the GEMM itself**. Fusing them into the GEMM **epilogue** removes 2× full activation round-trips and turns 3 kernel launches into 1. The transform happens **in registers** before the single store.

---

## Fusion opportunity decision table

| Pattern | Ops collapsed | Where to fuse | Typical win | Notes |
|---|---|---|---|---|
| **GEMM epilogue: bias + act** | matmul → +bias → GELU/SiLU | GEMM epilogue | removes 1–2 HBM round-trips | hipBLASLt/CK epilogue, Triton |
| **GEMM epilogue: + quant** | matmul → scale → fp8/int8 cast | GEMM epilogue | removes quant pass + keeps output small | per-token scale reduce in epilogue |
| **RMSNorm + residual (+ quant)** | add residual → RMSNorm → fp8 cast | one elementwise kernel | removes 2–3 passes | AITER fused gated RMSNorm+quant |
| **RoPE + qk-norm** | apply RoPE → q/k RMSNorm | attention prologue | removes a pass over Q,K | fuse into attention input load |
| **Attention out + o_proj** | softmax·V → o_proj GEMM | attention epilogue / next GEMM | removes attention-output round-trip | shape permitting |
| **act_and_mul (SwiGLU)** | SiLU(gate) * up | MLP, fused with gate/up GEMM | removes intermediate buffer | "fusion = eliminating buffers" |
| **Fused MoE** | router → permute → grouped GEMM → combine | one (or few) kernels | removes permute/combine traffic | AITER/CK/Triton grouped GEMM |

> Heuristic: **fuse a memory-bound op into its compute-bound neighbor's epilogue/prologue.** Two compute-bound GEMMs only fuse if a producer-consumer tile dependency lets the second start from registers/LDS (rare; attention's QKᵀ→PV is the main case).

---

## 1. GEMM epilogue fusion (bias / activation / quant)

The GEMM computes the accumulator `acc` in fp32 (AGPRs); the epilogue applies bias, activation, scaling, and an optional output cast **before the single global store**.

### Triton fused GEMM + bias + SiLU + fp8-quant epilogue

```python
@triton.jit
def gemm_bias_act_quant(a_ptr, b_ptr, bias_ptr, c_ptr, cscale_ptr,
                        M, N, K, ..., FP8_MAX: tl.constexpr):
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        acc += tl.dot(load_a(k), load_b(k))          # mfma_16x16, fp32 accum
    # ---- epilogue (in registers, no HBM round-trip) ----
    acc += tl.load(bias_ptr + offs_n)[None, :]        # bias
    acc = acc * tl.sigmoid(acc)                        # SiLU
    scale = tl.max(tl.abs(acc)) / FP8_MAX              # per-tile dynamic scale
    q = (acc / scale).to(tl.float8e4nv)               # cast to fp8 e4m3
    tl.store(c_ptr + offs, q)
    tl.store(cscale_ptr + pid, scale)
```

### hipBLASLt / CK native epilogues
hipBLASLt exposes epilogue enums: `HIPBLASLT_EPILOGUE_BIAS`, `..._GELU`, `..._RELU`, `..._GELU_BIAS`, `..._GELU_AUX` (write pre-activation for backward), and ROCm 7.2's `..._CLAMP_EXT` / `..._CLAMP_BIAS_EXT`. CK expresses fusion via the **epilogue** stage of CK-Tile: `E = α·(A·B) + β·D` with arbitrary suffix ops. Set `OPTIMIZE_EPILOGUE=1` on MI300X to store the MFMA-layout accumulator directly and skip a reblock (cost: lower store-vector width, usually net win for fused epilogues). ROCm 7.2 CK adds **GEMM+GEMM fusion**.

---

## 2. RMSNorm + residual (+ quant) fusion

RMSNorm followed by an fp8 quant for the next GEMM is 3 passes over the hidden state if unfused. Fuse them into one kernel that reads the residual + input once, normalizes, applies weight, and emits the quantized output + per-token scale in a single store.

```python
@triton.jit
def fused_add_rmsnorm_quant(x_ptr, res_ptr, w_ptr, out_ptr, scale_ptr,
                            N, eps, FP8_MAX: tl.constexpr):
    cols = tl.arange(0, BLOCK_N)
    x   = tl.load(x_ptr + row*N + cols, mask=cols<N).to(tl.float32)
    res = tl.load(res_ptr + row*N + cols, mask=cols<N).to(tl.float32)
    h   = x + res                                   # fused residual add
    tl.store(res_ptr + row*N + cols, h, mask=cols<N)  # update residual stream
    var = tl.sum(h*h, axis=0) / N
    h   = h * tl.rsqrt(var + eps) * tl.load(w_ptr + cols, mask=cols<N)
    scale = tl.max(tl.abs(h)) / FP8_MAX             # per-token (per-row) scale
    tl.store(out_ptr + row*N + cols, (h/scale).to(tl.float8e4nv), mask=cols<N)
    tl.store(scale_ptr + row, scale)
```

**AITER** ships this as a production fused kernel (v0.1.12: **fused gated RMSNorm + group quantization**). Enable AITER fused norm/quant in vLLM via `VLLM_ROCM_USE_AITER=1`.

**MI300X note:** the reduction (`sum(h*h)`, `max(|h|)`) runs on VALU; keep `BLOCK_N` covering the full hidden dim per row so the reduction stays in one workgroup (no cross-block atomics). For very large hidden dims, split the row across waves and reduce via LDS.

---

## 3. RoPE + qk-norm fusion (attention prologue)

Apply rotary embeddings and (for Qwen/Gemma-style) per-head q/k RMSNorm as the attention kernel **loads** Q and K, instead of two separate passes that re-read Q,K from HBM.

```text
attention_prologue(q_tile, k_tile, cos, sin, qnorm_w, knorm_w):
    q = rmsnorm(q_tile, qnorm_w)        # qk-norm fused in
    k = rmsnorm(k_tile, knorm_w)
    q = rope(q, cos, sin)               # rotary, in registers
    k = rope(k, cos, sin)
    # q,k stay in registers/LDS → feed directly into QKᵀ MFMA
```

**When:** any model with RoPE (most). Saves a full read/write of Q and K. **Tradeoff:** adds VALU work to the attention prologue (sin/cos, rsqrt) competing with MFMA — but it replaces HBM traffic, a net win in the HBM-bound decode regime.

---

## 4. Attention output + o_proj fusion

The attention output `O = softmax(QKᵀ)·V` is immediately consumed by the output projection `o_proj` GEMM. If tiling lines up, feed `O` tiles from registers/LDS into the `o_proj` GEMM without a HBM round-trip of `O`.

**When it helps:** decode, where `O` is small but the round-trip still costs at M=1. **Tradeoff:** requires the attention and o_proj tile shapes to be compatible and enough register/LDS budget to hold the bridge tile; often only partially fusable. In practice many stacks keep them separate and instead fuse o_proj's epilogue (bias/quant).

---

## 5. act_and_mul fusion in the MLP (SwiGLU)

SwiGLU: `down( SiLU(x·W_gate) * (x·W_up) )`. The `SiLU(gate) * up` step is a classic fusion — never materialize separate `gate` and `up` buffers.

**Best practice (TritonMoE finding):** *"fusion is about eliminating buffers, not reducing kernel launches."* Keeping the gate+up intermediate **in registers** (fused gate+up projection) gave **35% memory savings** and 89–131% of CUDA Megablocks throughput. So:
- Fuse gate and up into **one grouped GEMM** producing both halves, then apply `SiLU(gate)*up` in the epilogue, keeping the product in registers for the down-proj input.

```python
# gate||up produced together; activate+multiply in epilogue, no intermediate HBM buffer
acc_gate, acc_up = grouped_gemm_two_outputs(x, W_gate, W_up)   # one launch
h = (acc_gate * tl.sigmoid(acc_gate)) * acc_up                 # SiLU(gate)*up in regs
store(h_ptr, h)   # or feed straight into down-proj if tiles allow
```

---

## 6. Fused MoE

MoE forward = router scoring → token permutation (gather by expert) → per-expert GEMMs → combine (scatter+weighted-sum). Unfused, the permute and combine are large gather/scatter HBM passes.

**Fuse into a grouped GEMM + fused align/sort:**
- **MoE align & sort** computes the expert-sorted token order efficiently; SGLang's optimized version gives **7× on MI300X / MI300A** over the Triton baseline (3× A100, 3× H200, 10× MI100).
- **Grouped/block-scheduled GEMM** runs all experts' GEMMs in one persistent kernel, with the gather folded into the A-load and the combine into the epilogue scatter — eliminating the standalone permute/combine buffers.
- **AITER FusedMoE** is the default on MI300X (`VLLM_ROCM_USE_AITER=1`); supports FlyDSL kernels for mixed-precision MoE (A4W4) and **falls back to CK** when FlyDSL isn't installed. vLLM ships MI300X-tuned Triton MoE configs (e.g., Qwen3-235B-A22B → +16.83% request throughput).

```text
fused_moe(x, router, experts_w):
    topk_ids, topk_w = router(x)                 # routing
    sorted_tok, expert_off = align_and_sort(topk_ids)   # fused align&sort (7x on MI300X)
    # one persistent grouped GEMM over experts; gather in A-load, combine in epilogue
    out = grouped_gemm_scatter(x[sorted_tok], experts_w, expert_off, topk_w)
    return out
```

**MI300X tradeoffs:** load-balance experts across the 8 XCDs (a hot expert on few CUs idles the rest — use Stream-K-style work assignment); for A4W4/MXFP4 MoE, watch for dequant-fallback paths that lose native FP4 GEMM throughput (see quantization file). The `device_gemm does not support this GEMM problem` AITER-CK error on odd MoE shapes (SGLang #16025) is a reminder to validate the CK instance covers your expert dims.

---

## How to express fusions: framework map

| Surface | How to fuse | Best for |
|---|---|---|
| **Triton** | write the epilogue/prologue inline in the `@triton.jit` kernel; autotune (`matrix_instr_nonkdim=16`) | custom MLP/MoE/attention, novel dtype |
| **hipBLASLt** | `hipblasLtMatmulDescSetAttribute` epilogue enum (BIAS/GELU/CLAMP/AUX) + scale ptrs | standard dense GEMM + bias/act/quant |
| **CK / CK-Tile** | epilogue stage of the pipeline; `OPTIMIZE_EPILOGUE=1`; GEMM+GEMM fusion (ROCm 7.2) | C++ kernels, max control |
| **AITER** | prebuilt fused kernels (RMSNorm+quant, FusedMoE, attention); `VLLM_ROCM_USE_AITER=1` | production vLLM/SGLang default path |
| **torch.compile (Inductor)** | automatic pointwise fusion + Triton codegen; `TORCHINDUCTOR_MAX_AUTOTUNE=1` | quick wins on PyTorch graphs |

---

## Fusion checklist (do this in order)

1. Profile → Top-N kernels by `gpu_time` (see `profiling_roofline.md`).
2. Identify **memory-bound** neighbors of GEMMs (norm, RoPE, act, quant, residual add).
3. Fuse the biggest memory-bound contributor into its GEMM epilogue/prologue first (Amdahl).
4. Prefer **eliminating intermediate buffers** over merely merging launches (registers/LDS, not HBM).
5. On MI300X set `OPTIMIZE_EPILOGUE=1` for fused GEMM epilogues; use `matrix_instr_nonkdim=16` in Triton.
6. Watch register pressure — a heavy epilogue can spill (AGPR/VGPR) and tank occupancy; check with `rocprof-compute`.
7. Re-profile; confirm the fused kernel's arithmetic intensity moved off the HBM roof and the neighbor kernel vanished from the Top-N.

---

## Sources

- AITER (AI Tensor Engine for ROCm — fused RMSNorm+quant, FusedMoE, attention): <https://github.com/ROCm/aiter>
- Optimizing with Composable Kernel (epilogue fusion, CK-Tile): <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html>
- CK-Tile GEMM hands-on (epilogue, OPTIMIZE_EPILOGUE): <https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html>
- vLLM V1 performance optimization on ROCm (AITER fused GEMM/RMSNorm/MoE, VLLM_ROCM_USE_AITER): <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html>
- TritonMoE — fused MoE dispatch + fused RMSNorm/SwiGLU, validated on MI300X (arXiv 2605.23911 / repo): <https://arxiv.org/html/2605.23911v1> · <https://github.com/bassrehab/triton-kernels>
- SGLang efficient MoE align & sort (7× on MI300X): <https://github.com/yiakwy-xpu-ml-framework-team/HPC-2025/blob/main/2025-3/Efficient%20MoE%20Align%20&%20Sort%20in%20SGLagn%20Fused%20MoE/Design%20Efficient%20MoE%20Align%20&%20Sort%20in%20SGLang%20Fused%20MoE.md>
- MI300X workload optimization (OPTIMIZE_EPILOGUE, mfma_16x16): <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html>
