---
name: flash-mla-optimization
description: "Expert optimization roadmap for Flash MLA sparse attention decode Triton kernel on AMD MI355X. P0: FP8 in-kernel dequant (fused gather+dequant+attention). P1: multi-head packing, accumulator splitting, BF16 MFMA, and shape-gated Split-K (conditional, never global) WITH its companion pieces (specialized split_k=2/4/8 combine, SplitKBufferPool, dual-scope Split-K). P2: dual-scope fusion, autotuning+config-pruning+bucketing, QK hoisting. NOTE: the MLA kernel is ~60% of e2e GPU time but can look like ~1% in the unit-test trace (data-gen artifact) — gate final decisions on e2e throughput, not isolated geomean, and don't stop on an isolated plateau."
---

## ⚠️ CRITICAL: THE #1 PRIORITY IS FP8 IN-KERNEL DEQUANT

**Before doing ANYTHING else, read this section.**

The production serving stack (sglang on AMD MI355X) stores KV cache as **FP8** (MODEL1_FP8Sparse layout). The live call chain is:

```
sglang engine → flash_mla_with_kvcache_entrypoint(k_cache=FP8, indices=physical, ...)
  → tilelang kernel reads FP8 directly, dequants inside kernel, outputs bf16
```

A triton kernel that only reads pre-dequantized bf16 **CANNOT replace tilelang in production** because:
1. Python-side dequant adds a separate GPU kernel launch + full HBM read+write (3x traffic)
2. The dequant intermediate buffer doubles VRAM usage
3. Even a 16x-faster triton kernel on bf16 loses all gains to the dequant overhead at e2e

**The triton kernel MUST implement `triton_sparse_attn_decode_fp8()` — a fused kernel that reads
raw FP8 bytes from the KV cache, dequantizes with E8M0 scales, reconstructs RoPE byte-pairs,
and computes attention, ALL in one kernel launch with ZERO intermediate buffers.**

Test with: `python test_triton_decode.py --quick`

### FP8 Implementation Checklist (do these FIRST)

1. Read Strategy 1 in the **Full Implementation Guide** (Part 2 below, same file) — complete code snippets
2. Understand MODEL1 KV cache byte layout (576 bytes/token data + 8 bytes/token scales)
3. Implement `fused_gather_attn_decode_dsv4()` in `triton_flash_mla_decode.py`
4. Key operations inside the kernel:
   - **Address computation**: `block_idx = indices // block_size`, use int64 pointers
   - **FP8 nope tile load**: 7 tiles × 64 bytes, `bitcast=True` to float8e4m3
   - **E8M0 scale dequant**: `scale = exp2(uint8_value - 127.0)`, then `bf16 = fp8 * scale`
   - **RoPE byte-pair reconstruction**: load lo/hi bytes, `uint16 = lo | (hi << 8)`, bitcast to bf16
   - **NaN clamping**: `clamp(where(isnan(v), 0, v), -65504, +65504)` after dequant
   - **Batch-load all tiles before compute** for memory latency hiding
5. Use `other=127` for scale loads (not `other=0`!) — 127 gives neutral scale=1.0 for invalid positions
6. Run `--mode fp8` correctness tests (tolerance: atol=2e-2, rtol=2e-2)

### Common FP8 Mistakes (will cause silent wrong results)

| # | Mistake | Fix |
|---|---------|-----|
| 1 | `scale = x / 127` | `scale = exp2(x - 127.0)` — E8M0 is power-of-2 |
| 2 | RoPE byte order reversed | `lo \| (hi << 8)` — low byte first |
| 3 | `to(float8, bitcast=False)` | Must use `bitcast=True` |
| 4 | Scale `other=0` for invalid | `other=127` → scale=1.0 (neutral) |
| 5 | int32 pointer overflow | Use int64 for all KV cache pointer arithmetic |
| 6 | Forgot NaN clamping | FP8 dequant can produce NaN → clamp immediately |

---

## ⚠️⚠️ CRITICAL #0: CUDA-GRAPH CAPTURE SAFETY (precompile before capture)

**This gates e2e above everything else. A kernel that is correct AND 3× in isolation scores
`e2e_delta=null, engagement_hits=0` ("hung on first capture batch, never healthy, 0 forwards" →
REJECT) if it is not capture-safe.** This is the #1 reason an isolated win fails to convert.

**Root cause (measured on DS_v4 MI355X TP=8):** sglang captures the decode path into a HIP cuda-graph,
**largest batch first (bs=256)**. A Triton `@triton.jit` kernel JIT-compiles on its FIRST call. The
HIP backend's raw→full metadata upgrade runs *inside* capture (`SGLANG_PREP_IN_CUDA_GRAPH`), so
sglang's own eager warmup forwards do NOT reach the triton branch — the kernel's **first real call
lands inside the capture region**. Under TP=8 that in-capture compile stalls one rank → the NCCL
collectives in the captured region desync → heartbeat timeout → **server never becomes healthy**.
(Single-GPU capture-of-cold-compile does NOT hang — it is specifically the TP + capture interaction.)

**THE KERNEL MODULE MUST EXPOSE `ensure_warmed(device)`** — precompiles every serving specialization
with tiny SELF-CONTAINED synthetic tensors (no real q/k_cache needed; Triton caches by constexpr
signature + arg dtypes, so a b=1 tiny-topk call compiles the SAME binary later used at bs=256). The
integration seam calls it ONCE before capture (wrap `on_after_cuda_graph_warmup`; see
e2e_integrator role). **NEVER rely on lazy first-call JIT.** Reference implementation (put in
`triton_flash_mla_decode.py`, adapt the constexpr combos to your kernel):

```python
_BYTES_PER_TOKEN = 584   # MODEL1_FP8Sparse: 448 nope + 128 rope + 7 e8m0 + 1 pad
_WARMED = set()

class _WNS:
    def __init__(self, **kw): self.__dict__.update(kw)

def _dummy_scope(block_size, topk, has_topk_length, device):
    nb = 4
    kv = torch.zeros((nb, block_size, 1, _BYTES_PER_TOKEN), dtype=torch.uint8, device=device)
    idx = torch.arange(topk, dtype=torch.int32, device=device).remainder(nb*block_size).view(1, topk)
    tlen = torch.full((1,), topk, dtype=torch.int32, device=device) if has_topk_length else None
    return _WNS(blocked_k_quantized=kv, block_size=block_size,
                indices_in_kvcache=idx, topk_length=tlen)

def ensure_warmed(device=None, h_q=128, d_v=512, d_qk=512):
    """Compile all serving specializations BEFORE capture. Idempotent; cheap after first call.
    Call from model init / on_after_cuda_graph_warmup. MUST NOT run during graph capture."""
    if device is None: device = torch.cuda.current_device()
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    combos = [  # serving is dual-scope + attn_sink; topk_length present or not (+ single-scope robustness)
        dict(has_extra=True,  has_sink=True, htl_m=True,  htl_e=True),
        dict(has_extra=True,  has_sink=True, htl_m=False, htl_e=False),
        dict(has_extra=False, has_sink=True, htl_m=True,  htl_e=False),
    ]
    s = torch.cuda.Stream(device=dev); s.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(s):
        for c in combos:
            key = (h_q, c["has_extra"], c["has_sink"], c["htl_m"], c["htl_e"])
            if key in _WARMED: continue
            q = torch.zeros((1, 1, h_q, d_qk), dtype=torch.bfloat16, device=dev)
            attn_sink = torch.zeros((h_q,), dtype=torch.float32, device=dev) if c["has_sink"] else None
            p = _WNS(decode=_WNS(b=1), h_q=h_q, h_kv=1, d_qk=d_qk, d_v=d_v, s_q=1)
            t = _WNS(q=q, attn_sink=attn_sink, sm_scale=float(d_qk**-0.5),
                     kv_scope=_dummy_scope(128, 64, c["htl_m"], dev),
                     extra_kv_scope=(_dummy_scope(256, 64, c["htl_e"], dev) if c["has_extra"] else None))
            try:
                run_triton_decode(p, t); _WARMED.add(key)
            except Exception as e:
                print(f"[warm] specialization {key} failed: {e}", flush=True)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    return len(_WARMED)
```

**Capture-safety rules for the kernel + wrapper (all mandatory):**
1. **No JIT/compile/dynamic-alloc inside the captured region.** `ensure_warmed` handles compile; also
   avoid host syncs (`.item()`, `.cpu()`, `torch.cuda.synchronize()`, Python branch on a GPU scalar)
   and shape-varying `.reshape().contiguous()` copies in the per-call hot path.
2. **If `@triton.autotune` is used, it is DOUBLY dangerous** (autotune benchmarks each config with a
   GPU launch + sync → guaranteed capture deadlock). Either warm ALL autotune configs in
   `ensure_warmed`, or use a FIXED config for the serving specialization (no autotune on the hot path).
3. **Verify with the capture probe in `test_triton_decode.py`** (warmup → cold capture → replay must be
   compile-free and correct) BEFORE claiming the kernel is done — isolated correctness alone does NOT
   prove capture safety.

---

## SPEC IDs (STRICT enforcement — the machine-checkable optimization plan)

When this skill matches in `enforcement.mode: strict` (see skill.md), the optimization plan is a
**mandate**: the authored kernel MUST implement every `mandatory_specs` id below, and the module MUST
export a self-report dict that the unittest reads and prints into `UNITTEST_RESULT`:

```python
# in triton_flash_mla_decode.py — one bool per SPEC id; the unittest asserts the mandatory ones.
SPECS_IMPLEMENTED = {
    "fp8_fused_dequant":            True,   # P0: single fused FP8 in-kernel dequant, zero intermediate buffer
    "capture_safety_ensure_warmed": True,   # #0: ensure_warmed precompile-before-capture
    "dual_scope_fused":             True,   # SPEC 8: main+extra in one kernel, shared online softmax
    "autotune_or_bucket_dispatch":  True,   # SPEC 1: @triton.autotune OR an explicit shape-bucket dispatch
    "shape_specialized_constexpr":  True,   # SPEC 9: constexpr topk/block_size/h_q for the fixed serving shape
    # optional/large-topk-only:
    "split_k":                      False,  # SPEC 2/3/6 — only if it actually fires (topk>=8192)
}
# If you skip a MANDATORY spec, you MUST also export a measured justification, or the round is REJECTED:
SPEC_SKIP_JUSTIFICATION = {
    # "split_k": "topk=128+1024=1152 < 8192 gate -> never activates in production; measured no-op",
}
```

| SPEC id | maps to | mandatory? |
|---|---|---|
| `fp8_fused_dequant` | §Strategy 1 (P0) | **yes** |
| `capture_safety_ensure_warmed` | §CRITICAL #0 | **yes** |
| `dual_scope_fused` | SPEC 8 | **yes** |
| `autotune_or_bucket_dispatch` | SPEC 1 (autotune) or an equivalent deterministic shape-bucket dispatch | **yes** |
| `shape_specialized_constexpr` | SPEC 9 | **yes** |
| `split_k` (+combine/buffer-pool) | SPEC 2/3/6 | only if it fires (topk≥8192) — else skip is auto-justified |

Rule: a mandatory SPEC that is `False` in `SPECS_IMPLEMENTED` **without** a matching
`SPEC_SKIP_JUSTIFICATION` entry (backed by a measured benchmark) fails the strict gate. "Cleaner code"
or "seemed unnecessary" is not a justification.

---

## Implementation Guide (MUST READ)

Before implementing any optimization, read the complete implementation guide:

  **Part 2: Full Implementation Guide** (inlined at the bottom of this same file, 1900+ lines)

This guide contains:
- **Strategy 1: Fused Gather + Dequant + Attention** — the most important section, with complete code
- FP8 KV cache memory layout with byte-level ASCII diagrams
- E8M0 scale dequantization, RoPE byte-pair reconstruction
- Multi-head packing, accumulator splitting, Q preloading
- Split-K kernel + combine kernel + dispatch heuristics
- 20-entry common mistakes table with symptoms and fixes
- OOM troubleshooting (VRAM vs register pressure)

---

# Optimize Sparse Attention Decode Triton Kernel

High-level optimization roadmap for the MLA sparse attention decode kernel on AMD MI355X (CDNA4, 256 CUs, 8 TB/s HBM).

## Real DS_v4 Serving Shapes (optimize for THESE)

The kernel is used in DS_v4 decode serving. Every optimization must target the **actual production
shapes**, not artificial large-topk configs that never occur in real serving:

| Parameter | Production value | Source |
|-----------|-----------------|--------|
| h_q | **128** | `num_attention_heads=128` |
| h_kv | 1 | `num_key_value_heads=1` |
| d_qk = d_v | **512** | `head_dim=512` |
| d_rope | 64 | `qk_rope_head_dim=64` |
| s_q | 1 | decode (single token/step) |
| Dual-scope | **always** | main SWA + extra c4 sparse |
| Main topk | **128** | `SWA_WINDOW=128` |
| Main block_size | **128** | `swa_page_size=128` |
| Extra topk | **1024** | `index_topk=1024` (c4_sparse_topk) |
| Extra block_size | **256** | `page_size=256` |
| Decode batch | 32–256 | c32/c64/c128/c256 concurrency |
| FP8 | MODEL1_FP8Sparse (E8M0) | `is_fp8_kvcache=True` |

**Key implication**: the max real topk is 128+1024=1152 (total across scopes). `topk=16384` does
not exist in production. Split-K (gated at topk≥8192) will **never activate** in real serving. Do
NOT optimize for topk=16384 at the expense of the real dual-scope shapes.

## Architecture Context

MLA stores K and V concatenated: `kv[..., :d_v]` is V, the full `kv[..., :d_qk]` is used for QK score.

| | MODEL1 (d_qk=512) | V3.2 (d_qk=576) |
|---|---|---|
| d_nope | 448 | 512 |
| d_rope | 64 | 64 |
| d_v | 512 | 512 |
| h_q | 64 or 128 | 128 |
| Dual scope | Yes (main + extra KV cache) | No (single scope only) |

Decode supports **dual KV scopes**: a main scope and an optional extra scope with independent KV caches, block sizes, topk values, and topk_lengths. Results must be combined via online softmax.

## Priority-Ordered Strategy Summary

| Priority | Strategy | Impact | Why |
|----------|----------|--------|-----|
| **P0** | **Fused FP8 gather+dequant+attention** | **HIGHEST — production blocker** | Without this, kernel cannot replace tilelang. Eliminates Python dequant + intermediate buffer. Halves HBM traffic vs bf16. |
| P1 | Multi-head packing (BLOCK_H) + tl.dot MFMA | Very high | 2x tensor core throughput, KV data reuse across heads |
| P1 | Accumulator splitting (8×64 tiles) | Very high | Prevents register spill on 512-wide d_v |
| P1 | BF16 compute for QK and PV dot products | High | 2x MFMA throughput vs float32 |
| **P1** | **Shape-gated Split-K for large-topk + small-batch** | **Very high (on the cases that dominate geomean)** | When `total_tokens` is small (decode b≤~256) a single CTA per token serializes the whole topk loop and under-fills the 256 CUs. Split-K exposes inter-SM parallelism and is the ONLY lever for the large-topk anchors (topk≥8192) that dominate the geomean. **MUST be conditionally dispatched** (see below) — applying it globally regresses small-topk/large-batch and the launch-floor cases, which LOSES on geomean. |
| **P1** | **Split-K companion pieces (NOT optional once you do Split-K)** | **Very high — Split-K underperforms or regresses without them** | Split-K is only as good as its combine + buffering. Three pieces are part of the P1 Split-K work, not later polish: (a) **specialized unrolled combine kernels** for split_k=2/4/8 (`_combine_splitk_kernel_2/_8`) — a generic runtime-loop combine causes ITL tail-latency spikes (measured: e2e P95/P99 ITL 2–3× worse with a generic combine vs the no-Split-K path); (b) **`SplitKBufferPool`** — cache the f32 partial buffers instead of `torch.empty` per call (~15% on small batch); (c) **dual-scope Split-K** (`_fused_..._dual_scope_splitk_kernel`) so small-batch dual-scope also parallelizes, not just single-scope. |
| P2 | Dual-scope fusion in kernel | High | Eliminates torch.cat + extra buffer |
| P2 | QK hoisting out of d_v tile loop | High | Halves K traffic when d_v has multiple tiles |
| **P2** | **Autotuning (`@triton.autotune`) + config pruning + total-tokens bucketing** | **High on a real serving mix (was under-rated as P3)** | A fixed config leaves throughput on the table across the 100× shape spread a live server issues. `@triton.autotune` keyed on `(total_tokens_bucket, h_q, topk)` + `_prune_configs` (BLOCK_H≤16 when h_q≤64) + `_bucket_total_tokens` (avoid re-trigger per batch size) is the main remaining lever once P0/P1 are done. Promoted from P3: on isolated geomean its impact looks "medium", but the kernel is ~60% of e2e GPU time so adaptive configs move the headline more than the isolated number suggests. |
| P2 | Grid swap (heads fast, tokens slow) | Medium | L1/L2 KV cache locality |
| P3 | exp2 fast math | Medium | Faster softmax on AMD |
| P3 | topk_length early exit | Low | Skip empty blocks |

**Implementation order**: P0 first (FP8 fusion), then P1 (MFMA + tiling + shape-gated Split-K **with
its companion combine/buffer-pool/dual-scope pieces**), then P2 (dual-scope fusion, autotuning, QK
hoisting), then P3. A kernel without P0 cannot be deployed even if P1-P3 give 16x isolated speedup on bf16.

### Why Split-K is P1, not P3 (and the conditional-dispatch rule)
Split-K was previously ranked P3 ("medium, large topk"). In practice the large-topk + small-batch
shapes (e.g. `topk=16384, b=148`) are ~50× above the small-case launch floor and **dominate the
geometric-mean score**, so a win there moves the headline more than almost anything else — Split-K is
P1. BUT it is a double-edged lever: its combine kernel adds an extra launch + partial buffers, which
is a NET LOSS on cases that don't need it (small/medium topk, large batch, the b≤2 launch-floor
cases). Because geomean is a geometric mean, a single regressed small case hurts more than a big-topk
win helps. So Split-K **MUST be shape-gated**, never global:
- **Take the Split-K path ONLY** when the single-CTA path under-fills the GPU: single-scope
  `topk ≥ 8192` (and `BLOCK_H == h_q` so each split still reads every KV entry exactly once — zero
  per-split KV reload); dual-scope only for small `total_tokens` with large `total_topk`
  (`_select_split_k` / `_decide_splitk_dual_scope`).
- **Everything else stays on the unmodified no-Split-K fused path.**
- **Cheap combine**: store partials as **BF16** (half the HBM traffic — precision-safe because each
  split's partial output is already finalized; only the final combine does one f32 online-softmax
  reduction over the few splits), keep `PartM/PartL` f32, allocate with `torch.empty` (never zeros).
  `SPLIT_K=4` is a good default for b≈148 (sweep 2/4/8; 8 over-splits, 2 under-fills).
- **Verify no regression**: per-case ms for the non-Split-K shapes must stay ≥ their no-Split-K values
  (within noise); only the large-topk anchors should improve. If any small case regressed, the gate
  threshold is too loose — tighten it. (A global Split-K once scored geomean 2.39 vs 3.05 for the same
  kernel without it; the shape-gated version scored 3.69 — same code, gating is the whole difference.)

## ⚠️ Don't stop on an isolated-geomean plateau — the real judge is e2e

The unit-test harness generates its FP8 KV data with PyTorch every case (random gen, topk sort, FP8
quant, ref attention). In a profiler trace of the unit test, those data-gen/ref kernels dominate and
the MLA kernel can look like ~1% of GPU time — a **measurement artifact**, not reality. Once the kernel
is fast in isolation, the isolated geomean flattens into measurement noise and an optimizer that stops
on "no isolated improvement" will quit with P2/P3 work (autotune, combine specialization, buffer pool,
dual-scope Split-K) still unimplemented.

But in the live serving path this MLA decode kernel is **~60% of e2e GPU time** (measured on DS_v4,
TP=8). So:
- **Do not declare done on an isolated-geomean plateau.** After P0+P1 land, keep implementing the P2
  pieces (autotune, dual-scope Split-K, the Split-K companion combine/buffer-pool) — their payoff shows
  up at e2e even when isolated geomean barely moves.
- **Gate the final decision on an e2e throughput measurement** (overlay the kernel into the server,
  `SGLANG_HACK_FLASHMLA_BACKEND=triton`, bench tok/s) whenever possible — not on isolated geomean alone.
- Reference e2e results (c32, ISL=8192, OSL=1024, TP=8):
  - Tilelang baseline: **333 tok/s**
  - Expert triton kernel (from sglang amd/deepseek_v4 branch): **614 tok/s (1.84x)**
  - Our best kernel (P0+P1, no autotune, 1228 lines): **647 tok/s (1.94x)** — **beats expert by +5.4%!**
  - Our autotune_v2 kernel (constexpr-reduced, 1240 lines): 640 tok/s (1.92x)
  
  **Our kernel already surpasses the expert at c32.** The expert's split_k=4 path at c32 adds combine
  overhead that exceeds the CU utilization benefit. Our non-split-K direct path is faster here.
  
  The expert may still be faster at very small batch (c2-c8) where split-K parallelism helps more.
  Further optimization should focus on higher concurrency (c64/c128) and reducing remaining gaps.

## Detailed Implementation Specs (MUST-DO — expert-level reference)

These are the **specific techniques from the expert reference implementation** that the kernel MUST
implement to reach production quality. Each item includes exact code patterns and constraints.
**ALL 8 SPECs are mandatory. Do NOT skip any. Implement them ALL.**

### SPEC 1: `@triton.autotune` (P1 — HIGHEST PRIORITY, repeatedly skipped by agents)

**WHY agents keep skipping it**: They try "simpler" optimizations (NaN cleanup, wrapper tweaks) that
look like lower-hanging fruit but deliver <2% on the real shapes. Autotune is the SINGLE BIGGEST
remaining lever — it lets the compiler find optimal BLOCK_H/BLOCK_N/num_warps per shape instead of
our hardcoded 4-branch if/elif.

**EXACT implementation required** (use EXACTLY these 10 configs — DO NOT ADD MORE):
```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16,  "BLOCK_N": 32},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16,  "BLOCK_N": 64},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16,  "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64,  "BLOCK_N": 32},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64,  "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 64},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64,  "BLOCK_N": 64},  num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 64},  num_warps=8, num_stages=1),
    ],
    key=["total_tokens_bucket", "h_q", "topk"],
)
```

**⚠️ CRITICAL: DO NOT add configs beyond these 10.** In particular:
- **NEVER add `BLOCK_H=128, BLOCK_N=128, num_warps=8`** — causes VGPR overflow → GPU SIGABRT
  in TP=8 serving (tested: server crashes with scheduler exit code -6, NCCL heartbeat timeout)
- **NEVER add `BLOCK_H=32, num_warps=8`** or `BLOCK_H=16, BLOCK_N=128, num_warps=8` — same risk
- These exact 10 configs are from the expert reference implementation and are proven safe in
  production TP=8 serving on AMD MI355X. Adding "extra" configs causes the autotune
  compilation phase to trigger GPU faults that crash all ranks.

- **`num_stages=1` always** (>1 causes register spill with no benefit — expert-confirmed).
- **`key` includes `total_tokens_bucket`** — NOT raw `total_tokens`. Bucket with:
  ```python
  def _bucket_total_tokens(total_tokens: int) -> int:
      if total_tokens <= 0: return 1
      n = 1
      while n < total_tokens: n <<= 1
      return n
  ```
  This prevents autotune re-triggering for every unique batch size.
- **Grid must use `meta`**: `grid = lambda meta: (cdiv(h_q, meta["BLOCK_H"]), total_tokens)`
- **Config pruning** (add `prune_configs_by` to `@triton.autotune`):
  ```python
  def _prune_configs(configs, named_args, **kwargs):
      h_q = named_args.get("h_q", 128)
      if h_q <= 64:
          pruned = [c for c in configs if c.kwargs.get("BLOCK_H", 16) <= 16]
      else:
          pruned = [c for c in configs if c.kwargs.get("BLOCK_H", 16) <= h_q]
      return pruned if pruned else configs
  ```
  This prevents BLOCK_H >= 32 when h_q <= 64 (causes MFMA precision issues on AMD).
- **Remove the manual BLOCK_H if/elif branches** in the wrapper — let autotune decide.

### SPEC 1b: Minimize constexpr parameters (P1 — REQUIRED for autotune to work in TP=8 serving)

**ROOT CAUSE of autotune crashing in TP=8 serving**: Every constexpr bool parameter DOUBLES the
number of kernel binaries Triton must compile. Original: 7 bool constexprs + 9 configs = 288 kernels.
The expert has only 2 bool constexprs + 10 configs = 40 kernels per autotuned kernel.
**288 compilations block the server main thread for minutes → NCCL heartbeat timeout → crash.**

**DONE so far**: S_Q → runtime, HAS_TOPK_LEN0+1 merged → HAS_TOPK_LENGTH.
Current: 5 constexpr (HAS_EXTRA, HAS_ATTN_SINK, HAS_TOPK_LENGTH, BLOCK_H, BLOCK_N) × 9 configs = **72** compilations. Server starts successfully at ~280s (no NCCL crash).

**Remaining**: Remove HAS_EXTRA via SPEC 8 (separate single/dual scope kernels) → 4 constexpr × 9 = **36** per kernel variant. Expert's dual-scope autotuned kernel has `HAS_TOPK_LENGTH_MAIN + HAS_TOPK_LENGTH_EXTRA + HAS_ATTN_SINK + BLOCK_H + BLOCK_N` but uses separate kernels for single vs dual scope.

```
KEEP as constexpr:
  - HAS_ATTN_SINK: tl.constexpr    # controls attn_sink finalize (hot path)
  - HAS_TOPK_LENGTH: tl.constexpr  # controls early-exit in loop (hot path)
  - BLOCK_H: tl.constexpr          # autotune tile size
  - BLOCK_N: tl.constexpr          # autotune tile size

ALREADY DONE:
  - S_Q → runtime int ✓
  - HAS_TOPK_LEN0/1 → merged to single HAS_TOPK_LENGTH ✓

TODO:
  - HAS_EXTRA → REMOVE (use SPEC 8: separate single/dual scope kernels)
```

### SPEC 2: Dual-Scope Split-K (P1 — critical for real serving shape)

**WHY it matters**: Real serving has main_topk=128 + extra_topk=1024 = **1152 total topk**. At
small batch (b=2..32, total_tokens=2..32), a single CTA serializes all 1152 iterations — severely
under-filling the 256 CUs. The expert uses split-K for dual-scope with MUCH lower thresholds.

**Expert's `_decide_splitk_dual_scope` heuristic** (from expert source `triton_mla_kernels_decode_fused.py`):

Expert uses named threshold constants (source: lines 44-55):
```python
DUAL_SCOPE_SPLITK_TOPK_THRESHOLD = 2048
NOSPLITK_TOKEN_THRESHOLD_LOW_TOPK = 64
SMALL_BATCH_TOKEN_THRESHOLD = 8
SPLITK_HIGH_TOPK_THRESHOLD = 512

def _decide_splitk_dual_scope(total_tokens, h_q, total_topk):
    # Step 1: decide WHETHER to use split-K (4 boolean conditions)
    use_splitk_for_small_bs = (total_tokens <= 8 and (h_q >= 128 or total_topk >= 1024))
    use_splitk_for_h64_large_topk = (h_q <= 64 and total_topk >= 1024
                                      and total_tokens > 8 and total_tokens <= 128)
    use_splitk_for_large_topk = (total_tokens > 64 and total_topk >= 2048)
    use_splitk_for_large_hq = (h_q > 64 and total_tokens > 8 and total_topk >= 256)

    if not (use_splitk_for_small_bs or use_splitk_for_h64_large_topk
            or use_splitk_for_large_topk or use_splitk_for_large_hq):
        return 0  # no split-K

    # Step 2: select split_k VALUE
    if total_tokens <= 8:
        if total_topk >= 512 and total_tokens <= 4:
            return 8
        return 4
    elif use_splitk_for_large_hq:
        if total_topk >= 512:
            return 4    # <-- THIS is the c32 path: split_k=4
        return 2
    elif use_splitk_for_h64_large_topk:
        return 2
    else:
        return _select_split_k(total_topk, h_q, total_tokens)
```

**DS_v4 c32 walkthrough**: total_tokens=32, h_q=128 (>64), total_topk=1152 (>=256):
- `use_splitk_for_large_hq = True` → enters step 2 "large_hq" branch
- total_topk=1152 >= 512 → **split_k=4** (128 CTAs vs 32 CTAs without split-K)

**Expert's `_should_use_fused_splitk`** (dispatch in `triton_mla_kernels_decode_optimized.py`):
This function decides whether to route to the "low_overhead" split-K path vs the regular non-splitk path:
```python
def _should_use_fused_splitk(total_tokens, h_q, total_topk):
    if total_tokens <= 4:
        return True
    if h_q <= 64 and total_topk <= 800:
        return total_tokens <= 256
    if h_q <= 64 and total_topk >= 1024:
        return total_tokens <= 128
    if h_q > 64:
        if total_topk >= 400:
            return total_tokens <= 32      # <-- c32 activates here
        else:
            return total_tokens <= 128
    return True
```

**⚠️ IMPORTANT: dual-scope split-K threshold tuning is subtle.** E2e measurements show:
- `total_tokens <= 8` (conservative, our current): e2e 637 tok/s at c32
- `total_tokens <= 32` with `split_k=8` and generic combine: e2e **612 tok/s** at c32 (**WORSE**)

Root cause: expert uses **split_k=4** (not 8) at c32 + **specialized combine kernel** (SPEC 3).
Our test used split_k=8 + generic loop combine → combine overhead > CU utilization gain.

**Recommended approach**: implement SPEC 3 (specialized combine) FIRST, then widen to expert thresholds.
The expert's low-overhead path (`fused_gather_attn_decode_dsv4_dual_scope_low_overhead`) bundles:
- `_decide_splitk_dual_scope` for value selection
- `SplitKBufferPool` for pre-allocated buffers (SPEC 6)
- Specialized combine dispatch (SPEC 3): k=2 → `_combine_splitk_kernel_2`, k=4 → `_combine_splitk_kernel`, k=8 → `_combine_splitk_kernel_8_optimized`

**Dual-scope split-K kernel**: The split-K kernel processes the COMBINED range `[0, topk_main +
topk_extra)`. Each split's `[k_start, k_end)` may cross the main/extra scope boundary:
```python
# Inside the split-K kernel:
k_start = pid_k * k_per_split
k_end = min(k_start + k_per_split, total_topk)

for n_start in range(k_start, k_end, BLOCK_N):
    offs_n = n_start + tl.arange(0, BLOCK_N)

    # Which scope does this block belong to?
    in_main = offs_n < topk_main
    in_extra = ~in_main & (offs_n < total_topk)

    # Process main scope portion (if any tokens in this block are main)
    if tl.sum(in_main.to(tl.int32)) > 0:
        main_indices = tl.load(IDX_main_ptr + pid_t * topk_main + offs_n,
                               mask=in_main, other=0)
        # ... load KV from main cache, dequant, QK, softmax, PV ...

    # Process extra scope portion (if any tokens in this block are extra)
    if tl.sum(in_extra.to(tl.int32)) > 0:
        extra_offs = offs_n - topk_main
        extra_indices = tl.load(IDX_extra_ptr + pid_t * topk_extra + extra_offs,
                                mask=in_extra, other=0)
        # ... load KV from extra cache, dequant, QK, softmax, PV ...
```

**WHY condition 3 is critical for DS_v4**: At c32 serving, total_tokens=32, h_q=128 (>64),
total_topk=1152 (>=400). Condition 3 matches: `h_q > 64 and total_topk >= 400 and
total_tokens <= 32 → split_k=4`. This means grid goes from (1, 32) = 32 CTAs to
(1, 32, 4) = 128 CTAs, much better utilization of the 256 CUs. Without this, c32 runs at
~50% CU utilization.

### SPEC 3: Specialized Combine Kernels (P1 companion — CRITICAL for split-K to work)

Expert has **3 separate combine kernels** (from `triton_mla_kernels_decode_fused.py`):
- `_combine_splitk_kernel_2`: for split_k=2, fixed BLOCK_H=16, BLOCK_D=128
- `_combine_splitk_kernel`: for split_k=4, fixed BLOCK_H=16, BLOCK_D=128
- `_combine_splitk_kernel_8_optimized`: for split_k=8, **autotuned** BLOCK_H in {16,32,64,128}

Dispatch in `fused_gather_attn_decode_dsv4_dual_scope_low_overhead`:
```python
if split_k == 8:
    _combine_splitk_kernel_8_optimized[grid_combine](...)  # autotuned
elif split_k == 2:
    _combine_splitk_kernel_2[grid_combine](...)
elif split_k == 4:
    _combine_splitk_kernel[grid_combine](...)  # k=4
```

Each uses the online softmax merge (single-pass, NOT two-pass find-max-then-accumulate):
```python
# Merge partial (out_a, lse_a) with (out_b, lse_b):
max_lse = tl.maximum(lse_a, lse_b)
w_a = tl.math.exp2((lse_a - max_lse) * LOG2E)
w_b = tl.math.exp2((lse_b - max_lse) * LOG2E)
out = (out_a * w_a[:, None] + out_b * w_b[:, None]) / (w_a + w_b)[:, None]
lse = max_lse + tl.math.log2(w_a + w_b) / LOG2E
```

**WHY this matters**: Our current combine kernel uses a two-pass loop (pass 1: find global max LSE,
pass 2: accumulate weighted outputs). Expert's kernels are fully unrolled single-pass with online merge.
E2e evidence: widening split-K to c32 with our generic combine → **612 tok/s** (WORSE than no split-K 637).
With specialized combine, expert achieves higher throughput at c32 via split_k=4.

A generic loop-based combine causes **P95/P99 ITL tail-latency spikes** (measured 2-3x worse).

### SPEC 4: AMD buffer_ops Safety Guard (P0 correctness — MUST ADD)

**This is a correctness bug, not a performance optimization.** When KV cache exceeds ~2GB, AMD's
`buffer_ops` optimization uses int32 address arithmetic that **overflows silently → data corruption**.

```python
BUFFER_OPS_DISABLE_THRESHOLD = 2 * 1024 * 1024 * 1024  # 2 GB

def _launch_kernel_with_buffer_guard(kernel, kv_cache, grid, *args, **kwargs):
    kv_cache_size = kv_cache.untyped_storage().nbytes()
    if kv_cache_size > BUFFER_OPS_DISABLE_THRESHOLD:
        with triton.knobs.amd.scope(use_buffer_ops=False):
            kernel[grid](*args, **kwargs)
    else:
        kernel[grid](*args, **kwargs)
```

Add this check in the Python wrapper around EVERY kernel launch that touches k_cache.

### SPEC 5: Explicit `.to(tl.float32)` After Every `tl.dot()` (P1 precision)

AMD MFMA may accumulate in mixed precision. The expert casts explicitly after every dot:
```python
# WRONG (implicit promotion, may lose precision on AMD):
qk += tl.dot(q_tile, tl.trans(kv_tile))

# CORRECT:
qk += tl.dot(q_tile, tl.trans(kv_tile)).to(tl.float32)
```
Apply to ALL 16 `tl.dot` calls (8 for QK, 8 for P@V) in the inner loop.

### SPEC 6: `SplitKBufferPool` (P2 — small batch speedup)

Expert's version (from `triton_mla_kernels_decode_fused.py` line 2736) caches **both tensors AND
pre-computed strides** to avoid repeated Python method calls:

```python
class SplitKBufferPool:
    _buffers = {}
    @classmethod
    def get_buffers(cls, split_k, total_tokens, h_q, d_v, device):
        key = (split_k, total_tokens, h_q, d_v, device)
        if key not in cls._buffers:
            po = torch.empty(split_k, total_tokens, h_q, d_v,
                             dtype=torch.float32, device=device)
            plse = torch.empty(split_k, total_tokens, h_q,
                               dtype=torch.float32, device=device)
            cls._buffers[key] = {
                "partial_output": po,
                "partial_lse": plse,
                "stride_po": po.stride(),     # cached strides!
                "stride_plse": plse.stride(),  # cached strides!
            }
        return cls._buffers[key]
    @classmethod
    def clear(cls):
        cls._buffers.clear()
```
Use cached `stride_po`/`stride_plse` in kernel launches instead of calling `.stride()` per call.
~15% on small batch due to avoiding torch.empty() allocation + stride() method overhead.

### SPEC 7: Factored `_process_kv_block_aggressive` Helper (P2 — code quality + correctness)

Expert's kernel (line 128 in `triton_mla_kernels_decode_fused.py`) factors the ~120-line inner loop
body into `_process_kv_block_aggressive`, called by ALL 4 kernel variants (single-scope, dual-scope,
split-K single, split-K dual). This:
- Ensures FP8 dequant, RoPE reconstruction, and softmax are always consistent
- Prevents copy-paste bugs when modifying the inner loop
- Makes it easy to add `.to(tl.float32)` after tl.dot in one place (SPEC 5)

Expert signature (takes/returns ALL 8 accumulator tiles individually):
```python
@triton.jit
def _process_kv_block_aggressive(
    kv_block_base, nope_rope_offset, scale_base_offset,
    valid, valid_2d,
    q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7,  # 8 query tiles
    acc_0, acc_1, ..., acc_7,                     # 8 accumulator tiles
    m_i, l_i, sm_scale_log2e,
    BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_SIZE: tl.constexpr, D_NOPE: tl.constexpr, ...
):
    # Load 7 nope FP8 tiles + 7 E8M0 scales + RoPE byte-pairs
    # Dequant, NaN clamp, QK dot (8×), online softmax, P@V dot (8×)
    return acc_0, ..., acc_7, m_i, l_i
```

### SPEC 8: Separate Single-Scope vs Dual-Scope Kernels (P2)

Instead of one kernel with `HAS_EXTRA: tl.constexpr`, write TWO separate `@triton.jit` kernels:
- `_fused_mla_decode_single_scope_kernel` (no extra scope args → compiler eliminates dead branches)
- `_fused_mla_decode_dual_scope_kernel` (main + extra loops with shared softmax state)

The compiler generates better code when it can statically eliminate the extra-scope branches at
compile time, rather than relying on constexpr dead-code elimination.

---

## Beyond Expert: Production-Specific Optimizations (SPEC 9-12)

These go BEYOND the expert reference implementation. The expert must support multiple models and
arbitrary shapes; we can specialize for the EXACT DS_v4 serving workload.

### SPEC 9: Compile-Time Shape Specialization (highest potential — expert CAN'T do this)

The expert kernel uses runtime parameters for topk, block_size, h_q because it must support
arbitrary shapes. Our serving shape is FIXED:
- main_topk=128, block_size=128, extra_topk=1024, block_size=256, h_q=128, d_qk=512

Hardcode these as `tl.constexpr` in a specialized kernel variant:
```python
@triton.jit
def _fused_mla_decode_dsv4_specialized(
    # ... same args but with constexpr shapes:
    TOPK0: tl.constexpr = 128,
    BS0: tl.constexpr = 128,
    TOPK1: tl.constexpr = 1024,
    BS1: tl.constexpr = 256,
    H_Q: tl.constexpr = 128,
    # ...
):
```
This lets the compiler:
- **Unroll the topk loop** (`range(0, 128, 64)` → 2 iterations, fully unrolled)
- **Eliminate all bounds checks** (`if n_start < topk0` → always True for first 2 iterations)
- **Optimize index math** (constant division/modulo → bit shifts)
- **Reduce register pressure** (compiler knows exact tile counts, no dynamic allocation)

The wrapper dispatches to the specialized kernel when the runtime shape matches, falling back to the
generic autotuned kernel otherwise. Expected: **+10-20%** on the exact production shape.

### SPEC 10: Triton Cache Pre-Warming (solves autotune TP=8 crash)

**Problem**: Autotune compiles ~40 kernel binaries on first call, blocking the server main thread
for minutes → NCCL heartbeat timeout → TP=8 crash.

**Solution**: After server startup, BEFORE accepting requests, run a warmup pass that triggers all
autotune compilations with dummy tensors matching the production shapes:
```python
def warmup_triton_mla_kernels(device, h_q=128, d_qk=512):
    """Pre-compile all autotune configs by running the kernel with dummy data."""
    for total_tokens in [2, 32, 64, 128, 256]:  # covers all serving batch sizes
        for has_extra in [True, False]:
            q = torch.randn(total_tokens, h_q, d_qk, dtype=torch.bfloat16, device=device)
            # ... build minimal dummy KV scope matching production layout ...
            run_triton_decode(dummy_p, dummy_t)  # triggers autotune compilation
    torch.cuda.synchronize()
```
Insert this call in sglang's server initialization, after model loading but before the health
endpoint goes live. All subsequent serving requests hit triton cache → zero compilation delay.

This can be added to `sglang/srt/layers/attention/hip_flash_mla.py` or the backend's `__init__`.

### SPEC 11: LDS (Shared Memory) KV Tile Reuse Across Head Groups

When `BLOCK_H < h_q` (e.g., BLOCK_H=64, h_q=128 → 2 CTAs per token), two CTAs load the
**exact same KV data** from HBM independently. With h_kv=1, KV is identical across all heads.

Load KV tiles to LDS once per CTA group, then reuse across the BLOCK_H heads:
```python
# Conceptual (Triton LDS is implicit via tl.load with cache hints):
# Use tl.load with eviction_policy="evict_last" for KV tiles (keep in L1/L2)
# Ensure grid ordering puts adjacent head-groups on the same CU for L1 sharing
kv_tile = tl.load(kv_ptr, eviction_policy="evict_last")  # hint: keep in cache
```
The grid swap (heads fast, tokens slow) already helps with L2 locality. But explicit LDS sharing
via cooperative groups (if Triton supports `tl.cooperative_groups`) could halve HBM bandwidth.

Expected: **+5-15%** on memory-bound large-batch cases.

### SPEC 12: AMD-Specific Hardware Optimizations

- **`eviction_policy` hints**: Use `"evict_last"` for Q tiles (reused across all topk iterations)
  and `"evict_first"` for KV tiles (streamed once, not reused)
- **`waves_per_eu` control**: If Triton exposes it, set occupancy target explicitly (occupancy=2
  may beat occupancy=1 by hiding memory latency, even with more register spill)
- **Vectorized FP8 loads**: Load 16 bytes at once (`tl.load` with appropriate pointer alignment)
  instead of per-element — the FP8 KV cache layout (576 bytes/token, 64-byte aligned) supports this

## Design Philosophy

1. **FP8 end-to-end is non-negotiable.** The kernel MUST read FP8 from KV cache and produce bf16 output in one kernel launch. No Python-side dequant. No intermediate buffers. This is what tilelang does and what the production path requires.

2. **Fuse everything possible into one kernel.** Decode is memory-bandwidth-bound. Every intermediate buffer materialized in global memory costs a full read+write pass.

3. **Separate MODEL1 and V3.2 code paths.** Different FP8 layouts (E8M0 vs float32 scales, different tile sizes). One kernel per model variant.

4. **Adaptive dispatch over fixed configs.** Workloads vary by 100x. Use autotuning with key bucketing.

5. **Split-K for parallelism on large topk.** Partition topk across CTA groups for GPU utilization.

---

## Optimization Roadmap

The kernel architecture has these components:

```
triton_sparse_attn_decode(q, kv_scope, extra_kv_scope, ...)     [dispatch]
  ├── fused_gather_attn_decode_dsv4(q, kv_fp8, indices, ...)    [single scope]
  │     ├── _fused_gather_attn_dsv4_kernel                       [main kernel]
  │     └── _fused_gather_attn_dsv4_splitk_kernel + _combine     [large topk]
  ├── fused_gather_attn_decode_dsv4_dual_scope(...)              [dual scope, large batch]
  │     └── _fused_gather_attn_dsv4_dual_scope_kernel            [main + extra sequential]
  └── fused_gather_attn_decode_dsv4_dual_scope_low_overhead(...) [dual scope, small batch]
        └── _fused_gather_attn_dsv4_dual_scope_splitk_kernel     [split-K + combine]
```

All kernels read FP8 KV cache directly. There is NO bf16 input path.

Follow these stages in order. Each stage is independently verifiable — run `python test_triton_decode.py --quick` after each step.

**Critical**: Large-topk workloads can trigger GPU OOM and wedge the device. See [OOM Troubleshooting](#oom-troubleshooting) in Part 2.

---

### Stage 1: Single-Scope Fused Kernel (get correctness first)

Build `fused_gather_attn_decode_dsv4()` and the underlying `_fused_gather_attn_dsv4_kernel`.

#### Phase 1: Core Fused Gather+Dequant+Attention Kernel

Write `_fused_gather_attn_dsv4_kernel` — the single-scope fused kernel. This is the core building block.

The kernel does ALL of the following in one pass (zero intermediate buffers):
1. **Grid**: `(cdiv(h_q, BLOCK_H), total_tokens)` — heads fast, tokens slow (grid swap)
2. **Q preload**: Load all 8 Q tiles `[BLOCK_H, 64]` bf16 ONCE outside the topk loop
3. **Topk loop** (`for n_start in range(0, topk, BLOCK_N)`):
   a. Load indices, compute `block_idx = indices // block_size`, `offset = indices % block_size`
   b. Compute KV cache addresses with **int64** pointer arithmetic
   c. **Batch-load** all 7 FP8 nope tiles + RoPE bytes + 7 E8M0 scales
   d. **Dequant**: `scale = exp2(uint8 - 127.0)`, `kv = fp8.to(float8e4nv, bitcast=True).to(bf16) * scale`
   e. **RoPE reconstruct**: `lo | (hi << 8)` → bitcast to bf16
   f. **QK dot**: `qk += tl.dot(q_i, tl.trans(kv_i))` for all 8 tiles (= full d_qk=512)
   g. **Online softmax**: exp2 form, `alpha = exp2((m_old - m_new) * LOG2E)`
   h. **PV accumulate**: `acc_i = acc_i * alpha + tl.dot(p_bf16, kv_i)` for all 8 tiles
4. **Finalize**: attn_sink (decode sigmoid formula), lonely query handling, store output bf16 + LSE float32

Key implementation details from expert code:
- `_process_kv_block_aggressive()` — extracted helper for the inner loop body (shared by all kernel variants)
- 8 × `[BLOCK_H, 64]` tile accumulators in float32 (prevents register spill)
- `other=127` for scale loads (neutral scale=1.0 for invalid positions)
- `topk_length` early exit: skip entire BLOCK_N blocks when `n_start >= topk_len`
- All pointer arithmetic in int64 (KV cache can exceed 2GB)

**Autotune**: `@triton.autotune` with `key=["total_tokens_bucket", "h_q", "topk"]`:
```python
configs=[
    triton.Config({"BLOCK_H": 16,  "BLOCK_N": 32},  num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 16,  "BLOCK_N": 64},  num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 16,  "BLOCK_N": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 64,  "BLOCK_N": 32},  num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 64,  "BLOCK_N": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 128, "BLOCK_N": 64},  num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_H": 64,  "BLOCK_N": 64},  num_warps=8, num_stages=1),
    triton.Config({"BLOCK_H": 128, "BLOCK_N": 64},  num_warps=8, num_stages=1),
]
```
Always `num_stages=1` (avoids register spill from prefetch). Bucket `total_tokens` to nearest power-of-2.

**Verification**: `python test_triton_decode.py --quick` — single-scope cases must pass (atol=2e-2).

**Why**: This single kernel replaces the entire bf16 pipeline (Python dequant + gather + torch.cat + separate attention kernel). Zero intermediate buffers = half the HBM traffic.

#### Phase 2: Python Wrapper (`fused_gather_attn_decode_dsv4`)

The Python entry point for single-scope attention:
- Reshape `kv_cache` to `uint8` flat view: `kv_uint8.reshape(num_blocks, -1)`
- Check `kv_cache_size > 2GB` → disable AMD `buffer_ops` if needed
- For `topk < 8192`: launch `_fused_gather_attn_dsv4_kernel` directly
- For `topk >= 8192`: use Split-K (see Stage 3)
- Handle `topk_length` and `attn_sink` as optional dummy tensors when None

**Verification**: Single-scope test cases pass.

---

### Stage 2: Dual-Scope Fusion

DS_v4 uses dual KV scopes (main + extra). Build the dual-scope kernel variants.

#### Phase 3: Dual-Scope No-SplitK Kernel

Write `_fused_gather_attn_dsv4_dual_scope_kernel` — processes main scope then extra scope sequentially, sharing the same online softmax state (m_i, l_i, acc_0..acc_7).

Key differences from single-scope:
- Two separate topk loops: first over main indices/KV, then over extra indices/KV
- Each scope has its own `block_size`, `topk`, `topk_length`, `KV_Cache` pointer
- Shared softmax state carries across scopes (no reset between main and extra)
- Autotune key includes both `topk_main` and `topk_extra`
- Config pruning: `BLOCK_H >= 32` only when `h_q >= 64`

**Verification**: Dual-scope test cases pass.

#### Phase 4: Dual-Scope Split-K Kernel

Write `_fused_gather_attn_dsv4_dual_scope_splitk_kernel` — for small batch sizes where Split-K provides better GPU utilization.

- Third grid dimension `pid_k` partitions the combined `topk_main + topk_extra` across splits
- Each split's `[k_start, k_end)` may cross the main/extra boundary
- Partial output stored as float32 `[split_k, total_tokens, h_q, d_v]`
- Combine kernel merges partials via online softmax (NOT simple averaging)
- Attn_sink fused into the combine step

**Verification**: Small-batch dual-scope cases pass.

#### Phase 5: Dispatch Logic

Write `fused_gather_attn_decode_dsv4_dual_scope()` and `_dual_scope_low_overhead()`:

```python
def _should_use_fused_splitk(total_tokens, h_q, total_topk):
    if total_tokens <= 4: return True
    if h_q <= 64 and total_topk <= 800: return total_tokens <= 256
    if h_q <= 64 and total_topk >= 1024: return total_tokens <= 128
    if h_q > 64:
        if total_topk >= 400: return total_tokens <= 32
        else: return total_tokens <= 128
    return True
```

- Small batch → `fused_gather_attn_decode_dsv4_dual_scope_low_overhead` (split-K)
- Large batch → `fused_gather_attn_decode_dsv4_dual_scope` (no split-K)

**Verification**: All dual-scope test cases pass with correct dispatch.

---

### Stage 3: Split-K for Large TopK

Handle `topk >= 8192` for single-scope via Split-K.

#### Phase 6: Single-Scope Split-K Kernel + Combine

- `_fused_gather_attn_dsv4_splitk_kernel`: third grid dim `pid_k`, each split processes `topk_per_split` tokens
- Partial output in float32 (NOT bf16 — combine needs precision)
- `_combine_splitk_kernel`: merges partials via online softmax
- Specialized combine kernels for split_k=2, 4, 8 (manual unroll, no runtime loop)
- `_select_split_k()`: heuristic based on topk, h_q, total_tokens

#### Phase 7: SplitKBufferPool

Cache split-K intermediate buffers to avoid repeated `torch.empty()` overhead:
```python
class SplitKBufferPool:
    _buffers = {}
    @classmethod
    def get_buffers(cls, split_k, total_tokens, h_q, d_v, device): ...
```
Especially impactful for small batch sizes where allocator overhead dominates (~15% speedup for batch=2).

**Verification**: Large topk=16384 cases pass. No OOM.

---

### Stage 4: Safety & Production Hardening

#### Phase 8: AMD Buffer Ops Safety Guard

When KV cache exceeds ~2GB, AMD's `buffer_ops` optimization uses int32 address arithmetic that overflows → silent data corruption. Detect and disable at runtime:

```python
BUFFER_OPS_DISABLE_THRESHOLD = 2 * 1024 * 1024 * 1024
if kv_cache_size > BUFFER_OPS_DISABLE_THRESHOLD:
    # disable buffer_ops for this kernel launch
```

#### Phase 9: Autotune Config Pruning

Reduce startup compilation time by pruning configs that waste resources:
- `BLOCK_H >= 32` only when `h_q >= 64` (avoid MFMA precision issues on AMD)
- Remove near-duplicate configs that produce identical performance
- Target: ~10 configs per kernel variant (from original 143 → 44 → 10)

**Verification**: All test cases still pass. Startup time reduced.

---

### Summary: What to Implement

| File | What to write |
|------|---------------|
| `_process_kv_block_aggressive()` | Shared helper: FP8 load → dequant → QK → softmax → PV |
| `_fused_gather_attn_dsv4_kernel` | Single-scope fused kernel |
| `fused_gather_attn_decode_dsv4()` | Single-scope Python wrapper + split-K dispatch |
| `_fused_gather_attn_dsv4_dual_scope_kernel` | Dual-scope no-splitK kernel |
| `_fused_gather_attn_dsv4_dual_scope_splitk_kernel` | Dual-scope split-K kernel |
| `_combine_splitk_kernel` / `_2` / `_8` | Split-K combine kernels |
| `fused_gather_attn_decode_dsv4_dual_scope()` | Dual-scope dispatch |
| `fused_gather_attn_decode_dsv4_dual_scope_low_overhead()` | Small-batch split-K dispatch |
| `SplitKBufferPool` | Buffer pool for split-K intermediates |

Start with Phase 1 (single-scope fused kernel) — get correctness passing first, then add dual-scope and split-K.

See **Part 2: Full Implementation Guide** below (Strategy 1 "Complete Fused Kernel Code") for full implementation.

---

_The old Phase 16/17 notes (autotune reduction + always-fused dispatch) are already incorporated above — the new architecture has no 2-phase fallback and uses a compact config set._

---

## Debugging Strategies

When correctness tests fail during optimization, use these strategies to isolate the issue.

### Numerical Debugging

1. **Compare against reference**: The reference implementation in `ref.py` is the ground truth. Always compare output AND lse values, not just output.
2. **Check tolerance thresholds**: bf16 dot products introduce rounding. Use `atol=1e-2, rtol=1e-2` for bf16 output, tighter for float32 intermediates.
3. **Isolate the failing dimension**: If output differs, check whether the error is in the nope region (FP8 dequant issue) or rope region (byte reconstruction issue).
4. **NaN propagation**: NaN in KV data propagates through softmax and corrupts all heads for that token. Always clamp after dequantization.
5. **Online softmax bugs**: The most common mistake is forgetting to rescale old accumulators when the running max changes. Verify `acc *= exp2((old_max - new_max) * LOG2E)` is applied to ALL accumulators.

### Performance Debugging

1. **Roofline analysis**: Decode is bandwidth-bound. Calculate theoretical bandwidth utilization = (bytes read + bytes written) / kernel_time. If utilization < 50%, there's likely redundant memory traffic.
2. **Autotune config inspection**: Print which config autotune selected. If it picks an unexpected config, the workload categorization may be wrong.
3. **Kernel launch overhead**: For small batches, time the Python wrapper separately from the kernel. If Python overhead > kernel time, you need buffer pools and fixed configs.
4. **Register pressure**: If the compiler spills to local memory, reduce BLOCK_H or the number of accumulators. Check `triton.compiler` output for register counts.

---

# Part 2: Full Implementation Guide

_The roadmap above (Part 1) gives the priority order and SPECs. Everything below is the complete
~1900-line implementation guide with full code, memory layouts, and reference formulas._

# Flash MLA Sparse Attention Decode: Triton Kernel Optimization Guide for AMD MI355X

A complete, hands-on implementation guide for optimizing the Flash MLA sparse attention decode Triton kernel on AMD MI355X (CDNA4, 256 CUs, 8 TB/s HBM bandwidth). This document covers 12 optimization strategies with full code snippets, starting from a naive baseline and progressing to a production-grade fused kernel.

**Target audience**: Engineers writing or optimizing Triton kernel code by hand.

**Target hardware**: AMD Instinct MI355X (CDNA4 architecture).

**Model focus**: DeepSeek-V4 MODEL1 variant (d_qk=512, d_nope=448, d_rope=64, d_v=512, h_q=64 or 128).

---

## Table of Contents

1. [Architecture Context](#1-architecture-context)
2. [Strategy 1: Fused Gather + Dequant + Attention](#strategy-1-fused-gather--dequant--attention)
3. [Strategy 2: Accumulator Splitting](#strategy-2-accumulator-splitting)
4. [Strategy 3: Q Preloading](#strategy-3-q-preloading)
5. [Strategy 4: Batched KV Loading](#strategy-4-batched-kv-loading)
6. [Strategy 5: Autotuning](#strategy-5-autotuning)
7. [Strategy 6: Split-K Parallelization](#strategy-6-split-k-parallelization)
8. [Strategy 7: Grid Swap](#strategy-7-grid-swap)
9. [Strategy 8: topk_length Early Exit](#strategy-8-topk_length-early-exit)
10. [Strategy 9: Memory Safety](#strategy-9-memory-safety)
11. [Strategy 10: Attention Sink](#strategy-10-attention-sink)
12. [Strategy 11: Python-side Low Overhead](#strategy-11-python-side-low-overhead)
13. [Strategy 12: Online Softmax Numerics](#strategy-12-online-softmax-numerics)
14. [Common Mistakes Table](#common-mistakes-table)
15. [OOM Troubleshooting](#oom-troubleshooting)
16. [Recommended Implementation Order](#recommended-implementation-order)
17. [Reference Formulas](#reference-formulas)

---

## 1. Architecture Context

### MLA (Multi-head Latent Attention) Basics

MLA stores K and V in a shared latent representation. The KV cache stores a single vector per token of dimension `d_qk`, where:

- `kv[..., :d_v]` is used as V (the value projection)
- the full `kv[..., :d_qk]` is used for the QK dot product

For MODEL1 (d_qk=512):

| Parameter | Value |
|-----------|-------|
| d_nope    | 448   |
| d_rope    | 64    |
| d_qk      | 512 (= d_nope + d_rope) |
| d_v       | 512   |
| h_q       | 64 or 128 |
| h_kv      | 1     |
| Dual scope | Yes (main + extra KV cache) |

### Decode Characteristics

Decode attention is **memory-bandwidth-bound**: each query attends to a sparse set of KV tokens (selected by topk). The kernel reads KV data once and performs a small amount of compute (dot product + softmax + PV accumulation). Every intermediate buffer materialized in global memory costs a full read+write pass that directly degrades performance.

### Baseline Inefficiency

The naive baseline works like this:

```
Python side:
  1. index_select gather from blocked KV cache -> BF16 intermediate [b, s_q, topk, d_qk]
  2. torch.cat for dual scope (main + extra)
  3. .float() conversion
  4. .contiguous() copies
  5. NaN replacement pass
  6. Launch Triton kernel (reads BF16 intermediate)
  7. Python-side attn_sink post-processing
  8. Python-side lonely query handling
```

Problems:
- The BF16 intermediate buffer can be enormous: batch=256, topk=1024, d_qk=512 = 256 MB
- Each Python operation is a separate kernel launch + memory allocation + global memory pass
- The Triton kernel uses float32 dot products (half the tensor core throughput)
- Fixed block sizes, no autotuning

---

## Strategy 1: Fused Gather + Dequant + Attention

**Impact: HIGHEST. This is the single most important optimization.**

### The Problem

The baseline performs gather and dequantization in Python before the attention kernel:

```python
# BASELINE: Python-side gather + dequant (SLOW)
def triton_sparse_attn_decode_entry(q, kv_scope, ...):
    # Step 1: Dequantize FP8 -> BF16 (separate kernel or CPU code)
    blocked_k = dequantize_k_cache(kv_scope.blocked_k_quantized, layout)
    # blocked_k is now BF16: [num_blocks, block_size, 1, d_qk]

    # Step 2: Flatten and gather
    kv_flat = blocked_k.reshape(-1, d_qk)  # BF16, full size in global memory

    # Step 3: For dual scope, concatenate
    if extra_kv_scope is not None:
        extra_flat = extra_blocked_k.reshape(-1, d_qk)
        combined_kv = torch.cat([kv_flat, extra_flat], dim=0)  # ANOTHER allocation

    # Step 4: Launch kernel that reads this huge BF16 buffer
    _sparse_attn_decode_kernel[grid](q, combined_kv, indices, ...)
```

This produces a `[total_tokens, d_qk]` BF16 intermediate that is read once by the kernel. For FP8 KV caches, the data is also 2x larger than necessary (BF16 vs FP8).

### The Solution: Fuse Everything

The optimized kernel reads raw FP8 bytes directly from the KV cache, dequantizes on-the-fly, and performs attention -- all in a single kernel with zero intermediate buffers.

### MODEL1 KV Cache Memory Layout (d_qk=512)

Understanding the exact byte layout is essential for writing the in-kernel gather.

```
KV Cache: uint8 tensor, organized into blocks.
Each block contains `block_size` tokens.

Block layout (byte offsets):
+=============================================================+
|  Token 0 data (576 bytes)                                   |
|  +-- Bytes   0 -  63: nope tile 0 (64 x FP8 e4m3)          |
|  +-- Bytes  64 - 127: nope tile 1 (64 x FP8 e4m3)          |
|  +-- Bytes 128 - 191: nope tile 2 (64 x FP8 e4m3)          |
|  +-- Bytes 192 - 255: nope tile 3 (64 x FP8 e4m3)          |
|  +-- Bytes 256 - 319: nope tile 4 (64 x FP8 e4m3)          |
|  +-- Bytes 320 - 383: nope tile 5 (64 x FP8 e4m3)          |
|  +-- Bytes 384 - 447: nope tile 6 (64 x FP8 e4m3)          |
|  +-- Bytes 448 - 575: RoPE data  (64 x BF16 as 128 raw bytes)|
|                                              Total: 576 bytes |
+=============================================================+
|  Token 1 data (576 bytes)                                   |
|  ... (same layout)                                          |
+=============================================================+
|  ...                                                        |
+=============================================================+
|  Token (block_size-1) data (576 bytes)                      |
+=============================================================+
|  ---- Scale region starts at offset: block_size * 576 ----  |
+=============================================================+
|  Token 0 scales (8 bytes)                                   |
|  +-- Bytes 0-6: 7 x E8M0 uint8 scales (one per nope tile)  |
|  +-- Byte  7:   padding (unused)                            |
+=============================================================+
|  Token 1 scales (8 bytes)                                   |
|  ... (same layout)                                          |
+=============================================================+

Constants:
  BYTES_PER_TOKEN_DATA  = 576    # 7*64 + 128
  BYTES_PER_TOKEN_SCALE = 8      # 7 + 1 padding
  TILE_SIZE             = 64     # elements per nope tile
  D_NOPE                = 448    # 7 * 64
  D_ROPE                = 64
  NUM_NOPE_TILES        = 7
```

### Complete Fused Kernel Code

```python
import triton
import triton.language as tl

# Constants for MODEL1 layout
BYTES_PER_TOKEN_DATA: tl.constexpr = 576
BYTES_PER_TOKEN_SCALE: tl.constexpr = 8
TILE_SIZE: tl.constexpr = 64
D_NOPE: tl.constexpr = 448
LOG2E: tl.constexpr = 1.4426950408889634


@triton.jit
def _fused_gather_dequant_attn_kernel(
    # Pointers
    Q_ptr,              # [total_tokens, h_q, d_qk] bf16
    KV_Cache,           # [num_blocks, bytes_per_block] uint8 (raw FP8 KV cache)
    Indices_ptr,        # [total_tokens, topk] int32 (indices into flattened KV)
    Out_ptr,            # [total_tokens, h_q, d_v] bf16 output
    LSE_ptr,            # [total_tokens, h_q] float32 log-sum-exp
    AttnSink_ptr,       # [h_q] float32 attention sink values
    TopkLength_ptr,     # [total_tokens] int32 topk lengths
    # Scalars
    sm_scale,           # float32 softmax scale
    total_tokens,       # int32
    h_q,                # int32
    topk,               # int32
    block_size,         # int32 (KV cache block size, e.g. 64)
    stride_kv_block,    # int64 stride between blocks in bytes
    stride_q_t,         # stride of Q along token dim
    stride_q_h,         # stride of Q along head dim
    stride_q_d,         # stride of Q along d dim
    stride_o_t,         # stride of Out along token dim
    stride_o_h,         # stride of Out along head dim
    stride_o_d,         # stride of Out along d dim
    # Constexpr
    HAS_ATTN_SINK: tl.constexpr,
    HAS_TOPK_LENGTH: tl.constexpr,
    BLOCK_H: tl.constexpr,     # heads per CTA
    BLOCK_N: tl.constexpr,     # KV tokens per loop iteration
):
    # ----------------------------------------------------------------
    # Grid: (num_h_groups, total_tokens)
    # pid(0) = head group (fast-varying for L2 locality)
    # pid(1) = token index (slow-varying)
    # ----------------------------------------------------------------
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)

    # Head offsets within this CTA
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    # Tile offsets (64 elements per tile)
    offs_tile = tl.arange(0, TILE_SIZE)

    # ================================================================
    # Step A: Load Q tiles (hoisted out of KV loop -- see Strategy 3)
    # ================================================================
    q_base = Q_ptr + pid_t * stride_q_t
    # Load all 8 Q tiles: 7 nope + 1 rope
    q_0 = tl.load(q_base + offs_h[:, None] * stride_q_h + (0 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
                  mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_1 = tl.load(q_base + offs_h[:, None] * stride_q_h + (1 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
                  mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_2 = tl.load(q_base + offs_h[:, None] * stride_q_h + (2 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
                  mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_3 = tl.load(q_base + offs_h[:, None] * stride_q_h + (3 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
                  mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_4 = tl.load(q_base + offs_h[:, None] * stride_q_h + (4 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
                  mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_5 = tl.load(q_base + offs_h[:, None] * stride_q_h + (5 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
                  mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_6 = tl.load(q_base + offs_h[:, None] * stride_q_h + (6 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
                  mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_7 = tl.load(q_base + offs_h[:, None] * stride_q_h + (7 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
                  mask=mask_h[:, None], other=0.0).to(tl.bfloat16)

    # ================================================================
    # Step B: Initialize accumulators (8 x [BLOCK_H, 64] -- see Strategy 2)
    # ================================================================
    acc_0 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_4 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_5 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_6 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_7 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)

    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    # ================================================================
    # Step C: Load topk_length once (see Strategy 8)
    # ================================================================
    if HAS_TOPK_LENGTH:
        topk_len = tl.load(TopkLength_ptr + pid_t)
    else:
        topk_len = topk

    # ================================================================
    # Step D: Main KV loop
    # ================================================================
    for n_start in range(0, topk, BLOCK_N):
        # --- Early exit if past topk_length ---
        if HAS_TOPK_LENGTH:
            if n_start >= topk_len:
                break

        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < topk
        if HAS_TOPK_LENGTH:
            mask_n = mask_n & (offs_n < topk_len)

        # ============================================================
        # Step D.1: Load indices and compute block addresses
        # ============================================================
        idx_ptrs = Indices_ptr + pid_t * topk + offs_n
        indices = tl.load(idx_ptrs, mask=mask_n, other=-1)

        is_invalid = indices == -1
        valid = mask_n & ~is_invalid
        indices_clamped = tl.maximum(indices, 0)

        block_idx = indices_clamped // block_size
        offset_in_block = indices_clamped % block_size

        # MUST use int64 for pointer arithmetic (KV cache can exceed 2GB)
        block_idx_64 = block_idx.to(tl.int64)
        offset_in_block_64 = offset_in_block.to(tl.int64)
        stride_kv_block_64 = tl.cast(stride_kv_block, tl.int64)

        kv_block_base = KV_Cache + block_idx_64 * stride_kv_block_64
        nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA
        scale_base_offset = (block_size * BYTES_PER_TOKEN_DATA
                             + offset_in_block_64 * BYTES_PER_TOKEN_SCALE)

        # ============================================================
        # Step D.2: Batch-load all scales (see Strategy 4)
        # ============================================================
        scale_ptrs = kv_block_base + scale_base_offset
        # Load 7 E8M0 uint8 scale bytes; other=127 -> exp2(0) = 1.0 (neutral)
        scale_uint8_0 = tl.load(scale_ptrs + 0, mask=valid, other=127).to(tl.uint8)
        scale_uint8_1 = tl.load(scale_ptrs + 1, mask=valid, other=127).to(tl.uint8)
        scale_uint8_2 = tl.load(scale_ptrs + 2, mask=valid, other=127).to(tl.uint8)
        scale_uint8_3 = tl.load(scale_ptrs + 3, mask=valid, other=127).to(tl.uint8)
        scale_uint8_4 = tl.load(scale_ptrs + 4, mask=valid, other=127).to(tl.uint8)
        scale_uint8_5 = tl.load(scale_ptrs + 5, mask=valid, other=127).to(tl.uint8)
        scale_uint8_6 = tl.load(scale_ptrs + 6, mask=valid, other=127).to(tl.uint8)

        # E8M0 dequant: scale = 2^(uint8_value - 127)
        scale_bf16_0 = tl.math.exp2(scale_uint8_0.to(tl.float32) - 127.0).to(tl.bfloat16)
        scale_bf16_1 = tl.math.exp2(scale_uint8_1.to(tl.float32) - 127.0).to(tl.bfloat16)
        scale_bf16_2 = tl.math.exp2(scale_uint8_2.to(tl.float32) - 127.0).to(tl.bfloat16)
        scale_bf16_3 = tl.math.exp2(scale_uint8_3.to(tl.float32) - 127.0).to(tl.bfloat16)
        scale_bf16_4 = tl.math.exp2(scale_uint8_4.to(tl.float32) - 127.0).to(tl.bfloat16)
        scale_bf16_5 = tl.math.exp2(scale_uint8_5.to(tl.float32) - 127.0).to(tl.bfloat16)
        scale_bf16_6 = tl.math.exp2(scale_uint8_6.to(tl.float32) - 127.0).to(tl.bfloat16)

        valid_2d = valid[:, None] & (offs_tile[None, :] < TILE_SIZE)

        # ============================================================
        # Step D.3: Batch-load all 7 nope FP8 tiles (see Strategy 4)
        # ============================================================
        tile_base = kv_block_base[:, None] + nope_rope_offset[:, None]

        # Issue ALL loads before any compute (ILP optimization)
        nope_uint8_0 = tl.load(tile_base + 0 * TILE_SIZE + offs_tile[None, :],
                               mask=valid_2d, other=0)
        nope_uint8_1 = tl.load(tile_base + 1 * TILE_SIZE + offs_tile[None, :],
                               mask=valid_2d, other=0)
        nope_uint8_2 = tl.load(tile_base + 2 * TILE_SIZE + offs_tile[None, :],
                               mask=valid_2d, other=0)
        nope_uint8_3 = tl.load(tile_base + 3 * TILE_SIZE + offs_tile[None, :],
                               mask=valid_2d, other=0)
        nope_uint8_4 = tl.load(tile_base + 4 * TILE_SIZE + offs_tile[None, :],
                               mask=valid_2d, other=0)
        nope_uint8_5 = tl.load(tile_base + 5 * TILE_SIZE + offs_tile[None, :],
                               mask=valid_2d, other=0)
        nope_uint8_6 = tl.load(tile_base + 6 * TILE_SIZE + offs_tile[None, :],
                               mask=valid_2d, other=0)

        # ============================================================
        # Step D.4: Load RoPE bytes
        # ============================================================
        # RoPE is stored as raw BF16 bytes at offset D_NOPE (448) from token start.
        # Each BF16 value = 2 bytes, so 64 BF16 values = 128 bytes.
        rope_ptrs = tile_base + D_NOPE + offs_tile[None, :] * 2  # *2: each bf16 = 2 bytes
        rope_lo = tl.load(rope_ptrs,     mask=valid_2d, other=0).to(tl.uint16)
        rope_hi = tl.load(rope_ptrs + 1, mask=valid_2d, other=0).to(tl.uint16)
        # Reconstruct BF16: lo = low byte, hi = high byte
        kv_7 = (rope_lo | (rope_hi << 8)).to(tl.bfloat16, bitcast=True)
        kv_7 = tl.where(valid_2d, kv_7, 0.0)

        # ============================================================
        # Step D.5: Dequantize FP8 tiles
        # ============================================================
        # uint8 -> FP8 (bitcast, NOT value conversion) -> BF16 -> multiply scale
        nope_fp8_0 = nope_uint8_0.to(tl.float8e4nv, bitcast=True)
        kv_0 = (nope_fp8_0.to(tl.bfloat16) * scale_bf16_0[:, None]).to(tl.bfloat16)
        kv_0 = tl.where(valid_2d, kv_0, 0.0)

        nope_fp8_1 = nope_uint8_1.to(tl.float8e4nv, bitcast=True)
        kv_1 = (nope_fp8_1.to(tl.bfloat16) * scale_bf16_1[:, None]).to(tl.bfloat16)
        kv_1 = tl.where(valid_2d, kv_1, 0.0)

        nope_fp8_2 = nope_uint8_2.to(tl.float8e4nv, bitcast=True)
        kv_2 = (nope_fp8_2.to(tl.bfloat16) * scale_bf16_2[:, None]).to(tl.bfloat16)
        kv_2 = tl.where(valid_2d, kv_2, 0.0)

        nope_fp8_3 = nope_uint8_3.to(tl.float8e4nv, bitcast=True)
        kv_3 = (nope_fp8_3.to(tl.bfloat16) * scale_bf16_3[:, None]).to(tl.bfloat16)
        kv_3 = tl.where(valid_2d, kv_3, 0.0)

        nope_fp8_4 = nope_uint8_4.to(tl.float8e4nv, bitcast=True)
        kv_4 = (nope_fp8_4.to(tl.bfloat16) * scale_bf16_4[:, None]).to(tl.bfloat16)
        kv_4 = tl.where(valid_2d, kv_4, 0.0)

        nope_fp8_5 = nope_uint8_5.to(tl.float8e4nv, bitcast=True)
        kv_5 = (nope_fp8_5.to(tl.bfloat16) * scale_bf16_5[:, None]).to(tl.bfloat16)
        kv_5 = tl.where(valid_2d, kv_5, 0.0)

        nope_fp8_6 = nope_uint8_6.to(tl.float8e4nv, bitcast=True)
        kv_6 = (nope_fp8_6.to(tl.bfloat16) * scale_bf16_6[:, None]).to(tl.bfloat16)
        kv_6 = tl.where(valid_2d, kv_6, 0.0)

        # ============================================================
        # Step D.6: Compute QK scores (tiled dot product)
        # ============================================================
        # QK = sum over d_qk of Q * K^T
        # Each tile contributes: qk += dot(q_i, kv_i^T)  -- shapes [BLOCK_H,64] x [64,BLOCK_N]
        qk = tl.dot(q_0, tl.trans(kv_0)).to(tl.float32)
        qk += tl.dot(q_1, tl.trans(kv_1)).to(tl.float32)
        qk += tl.dot(q_2, tl.trans(kv_2)).to(tl.float32)
        qk += tl.dot(q_3, tl.trans(kv_3)).to(tl.float32)
        qk += tl.dot(q_4, tl.trans(kv_4)).to(tl.float32)
        qk += tl.dot(q_5, tl.trans(kv_5)).to(tl.float32)
        qk += tl.dot(q_6, tl.trans(kv_6)).to(tl.float32)
        qk += tl.dot(q_7, tl.trans(kv_7)).to(tl.float32)
        # qk shape: [BLOCK_H, BLOCK_N]

        qk = qk * sm_scale
        qk = tl.where(valid[None, :], qk, float("-inf"))

        # ============================================================
        # Step D.7: Online softmax update (see Strategy 12)
        # ============================================================
        m_ij = tl.max(qk, axis=1)       # [BLOCK_H]
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.where(m_i == float("-inf"), 0.0,
                         tl.math.exp2((m_i - m_new) * LOG2E))
        p = tl.where(qk == float("-inf"), 0.0,
                     tl.math.exp2((qk - m_new[:, None]) * LOG2E))
        l_new = alpha * l_i + tl.sum(p, axis=1)

        # ============================================================
        # Step D.8: PV accumulation (reuses same kv_i tiles)
        # ============================================================
        p_bf16 = p.to(tl.bfloat16)
        # kv_i is [BLOCK_N, 64]; p_bf16 is [BLOCK_H, BLOCK_N]
        # dot(p_bf16, kv_i) -> [BLOCK_H, 64]
        acc_0 = acc_0 * alpha[:, None] + tl.dot(p_bf16, kv_0).to(tl.float32)
        acc_1 = acc_1 * alpha[:, None] + tl.dot(p_bf16, kv_1).to(tl.float32)
        acc_2 = acc_2 * alpha[:, None] + tl.dot(p_bf16, kv_2).to(tl.float32)
        acc_3 = acc_3 * alpha[:, None] + tl.dot(p_bf16, kv_3).to(tl.float32)
        acc_4 = acc_4 * alpha[:, None] + tl.dot(p_bf16, kv_4).to(tl.float32)
        acc_5 = acc_5 * alpha[:, None] + tl.dot(p_bf16, kv_5).to(tl.float32)
        acc_6 = acc_6 * alpha[:, None] + tl.dot(p_bf16, kv_6).to(tl.float32)
        acc_7 = acc_7 * alpha[:, None] + tl.dot(p_bf16, kv_7).to(tl.float32)

        m_i = m_new
        l_i = l_new

    # ================================================================
    # Step E: Finalize output (attn_sink + lonely query)
    # ================================================================
    # See Strategy 10 for attn_sink details
    if HAS_ATTN_SINK:
        attn_sink_vals = tl.load(AttnSink_ptr + offs_h, mask=mask_h, other=0.0)
        exp_attn_sink_minus_m = tl.math.exp2((attn_sink_vals - m_i) * LOG2E)
        denominator = l_i + exp_attn_sink_minus_m
        denominator = tl.where(denominator == 0.0, 1.0, denominator)
        output_scale = 1.0 / denominator
    else:
        output_scale = tl.where(l_i == 0.0, 0.0, 1.0 / l_i)

    is_lonely_q = l_i == 0.0

    acc_0 = tl.where(is_lonely_q[:, None], 0.0, acc_0 * output_scale[:, None])
    acc_1 = tl.where(is_lonely_q[:, None], 0.0, acc_1 * output_scale[:, None])
    acc_2 = tl.where(is_lonely_q[:, None], 0.0, acc_2 * output_scale[:, None])
    acc_3 = tl.where(is_lonely_q[:, None], 0.0, acc_3 * output_scale[:, None])
    acc_4 = tl.where(is_lonely_q[:, None], 0.0, acc_4 * output_scale[:, None])
    acc_5 = tl.where(is_lonely_q[:, None], 0.0, acc_5 * output_scale[:, None])
    acc_6 = tl.where(is_lonely_q[:, None], 0.0, acc_6 * output_scale[:, None])
    acc_7 = tl.where(is_lonely_q[:, None], 0.0, acc_7 * output_scale[:, None])

    # Compute LSE
    lse = tl.where(is_lonely_q, float("+inf"),
                   m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E)
    tl.store(LSE_ptr + pid_t * h_q + offs_h, lse, mask=mask_h)

    # ================================================================
    # Step F: Store output tiles
    # ================================================================
    out_base = Out_ptr + pid_t * stride_o_t
    for tile_idx in range(8):
        acc_tile = (acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7)[tile_idx]
        out_offs = out_base + offs_h[:, None] * stride_o_h + (tile_idx * TILE_SIZE + offs_tile[None, :]) * stride_o_d
        tl.store(out_offs, acc_tile.to(tl.bfloat16), mask=mask_h[:, None])
```

Note: The loop over `tile_idx` at the end is pseudocode -- Triton does not support tuple indexing at runtime. In practice, write 8 separate `tl.store` calls, one per accumulator.

### Python Entry Point Change

```python
def fused_gather_attn_decode_dsv4(q, kv_cache, indices, block_size, sm_scale, d_v,
                                   attn_sink=None, topk_length=None):
    """
    EXPERT entry point: passes raw FP8 KV cache to kernel.
    Zero intermediate buffers.
    """
    total_tokens, h_q, d_qk = q.shape[0], q.shape[-2], q.shape[-1]
    topk = indices.shape[-1]

    # Zero-copy view of KV cache as uint8
    kv_uint8 = kv_cache.view(torch.uint8)
    kv_flat = kv_uint8.reshape(kv_cache.shape[0], -1)  # [num_blocks, bytes_per_block]

    out = torch.empty((total_tokens, h_q, d_v), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((total_tokens, h_q), dtype=torch.float32, device=q.device)

    BLOCK_H = 16  # or autotuned
    BLOCK_N = 64  # or autotuned

    grid = (triton.cdiv(h_q, BLOCK_H), total_tokens)

    _fused_gather_dequant_attn_kernel[grid](
        q, kv_flat, indices.contiguous(), out, lse,
        attn_sink, topk_length,
        sm_scale, total_tokens, h_q, topk, block_size,
        kv_flat.stride(0),
        q.stride(-3), q.stride(-2), q.stride(-1),
        out.stride(0), out.stride(1), out.stride(2),
        HAS_ATTN_SINK=(attn_sink is not None),
        HAS_TOPK_LENGTH=(topk_length is not None),
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
    )
    return out, lse
```

---

## Strategy 2: Accumulator Splitting

**Impact: Critical for avoiding register spill.**

### The Problem

The baseline uses a single `[BLOCK_H, 512]` accumulator:

```python
# BASELINE: single wide accumulator
acc = tl.zeros([BLOCK_H, 512], dtype=tl.float32)
# Register cost:
#   BLOCK_H=16:  16 x 512 x 4 bytes = 32 KB
#   BLOCK_H=64:  64 x 512 x 4 bytes = 128 KB
#   BLOCK_H=128: 128 x 512 x 4 bytes = 256 KB = ENTIRE REGISTER FILE!
```

When the accumulator alone approaches the 256 KB register file, the compiler spills to local memory (VRAM-backed scratch space), which is orders of magnitude slower.

### The Solution: 8 x 64-element Tile Accumulators

```python
# EXPERT: 8 separate tile accumulators
acc_0 = tl.zeros([BLOCK_H, 64], dtype=tl.float32)  # 4 KB at BLOCK_H=16
acc_1 = tl.zeros([BLOCK_H, 64], dtype=tl.float32)
acc_2 = tl.zeros([BLOCK_H, 64], dtype=tl.float32)
acc_3 = tl.zeros([BLOCK_H, 64], dtype=tl.float32)
acc_4 = tl.zeros([BLOCK_H, 64], dtype=tl.float32)
acc_5 = tl.zeros([BLOCK_H, 64], dtype=tl.float32)
acc_6 = tl.zeros([BLOCK_H, 64], dtype=tl.float32)
acc_7 = tl.zeros([BLOCK_H, 64], dtype=tl.float32)
# Total: 8 x 4 KB = 32 KB at BLOCK_H=16 (same memory, but compiler can manage tiles independently)
```

### Why This Works

The compiler schedules register usage per tile. With 8 independent accumulators:
- Only one tile's registers need to be live at a time during PV accumulation
- The compiler can reuse registers across tiles
- KV tiles loaded during QK phase are naturally 64-wide, matching accumulator width

### PV Accumulation Per-tile

```python
# Inside the KV loop, after computing attention weights p:
p_bf16 = p.to(tl.bfloat16)  # [BLOCK_H, BLOCK_N]

# Each tile uses the SAME kv_i loaded during the QK phase
# QK phase:  qk += dot(q_i, trans(kv_i))   -- [BLOCK_H,64] x [64,BLOCK_N] -> [BLOCK_H,BLOCK_N]
# PV phase:  acc_i += dot(p, kv_i)          -- [BLOCK_H,BLOCK_N] x [BLOCK_N,64] -> [BLOCK_H,64]
# One load serves BOTH phases. Zero extra memory traffic.

acc_0 = acc_0 * alpha[:, None] + tl.dot(p_bf16, kv_0).to(tl.float32)
acc_1 = acc_1 * alpha[:, None] + tl.dot(p_bf16, kv_1).to(tl.float32)
acc_2 = acc_2 * alpha[:, None] + tl.dot(p_bf16, kv_2).to(tl.float32)
acc_3 = acc_3 * alpha[:, None] + tl.dot(p_bf16, kv_3).to(tl.float32)
acc_4 = acc_4 * alpha[:, None] + tl.dot(p_bf16, kv_4).to(tl.float32)
acc_5 = acc_5 * alpha[:, None] + tl.dot(p_bf16, kv_5).to(tl.float32)
acc_6 = acc_6 * alpha[:, None] + tl.dot(p_bf16, kv_6).to(tl.float32)
acc_7 = acc_7 * alpha[:, None] + tl.dot(p_bf16, kv_7).to(tl.float32)
```

### Per-tile Output Store

```python
# After finalization, store each tile separately:
out_base = Out_ptr + pid_t * stride_o_t

tl.store(out_base + offs_h[:, None] * stride_o_h + (0 * 64 + offs_tile[None, :]) * stride_o_d,
         acc_0.to(tl.bfloat16), mask=mask_h[:, None])
tl.store(out_base + offs_h[:, None] * stride_o_h + (1 * 64 + offs_tile[None, :]) * stride_o_d,
         acc_1.to(tl.bfloat16), mask=mask_h[:, None])
tl.store(out_base + offs_h[:, None] * stride_o_h + (2 * 64 + offs_tile[None, :]) * stride_o_d,
         acc_2.to(tl.bfloat16), mask=mask_h[:, None])
tl.store(out_base + offs_h[:, None] * stride_o_h + (3 * 64 + offs_tile[None, :]) * stride_o_d,
         acc_3.to(tl.bfloat16), mask=mask_h[:, None])
tl.store(out_base + offs_h[:, None] * stride_o_h + (4 * 64 + offs_tile[None, :]) * stride_o_d,
         acc_4.to(tl.bfloat16), mask=mask_h[:, None])
tl.store(out_base + offs_h[:, None] * stride_o_h + (5 * 64 + offs_tile[None, :]) * stride_o_d,
         acc_5.to(tl.bfloat16), mask=mask_h[:, None])
tl.store(out_base + offs_h[:, None] * stride_o_h + (6 * 64 + offs_tile[None, :]) * stride_o_d,
         acc_6.to(tl.bfloat16), mask=mask_h[:, None])
tl.store(out_base + offs_h[:, None] * stride_o_h + (7 * 64 + offs_tile[None, :]) * stride_o_d,
         acc_7.to(tl.bfloat16), mask=mask_h[:, None])
```

---

## Strategy 3: Q Preloading

**Impact: Eliminates O(topk/BLOCK_N) redundant Q loads.**

### The Problem

Q is the same for every KV block iteration. Loading Q inside the KV loop re-reads from global memory each iteration:

```python
# BAD: Q loaded inside the loop
for n_start in range(0, topk, BLOCK_N):
    # This load happens topk/BLOCK_N times!
    q_tile = tl.load(Q_ptr + ..., mask=..., other=0.0)
    # ... use q_tile for QK dot product ...
```

With topk=1024 and BLOCK_N=64, Q is loaded 16 times -- 15 of those are redundant.

### The Solution: Hoist All Q Loads Before the Loop

```python
# EXPERT: Load all 8 Q tiles ONCE before any loop iteration
q_base = Q_ptr + pid_t * stride_q_t

# 7 nope tiles + 1 rope tile
q_0 = tl.load(q_base + offs_h[:, None] * stride_q_h
               + (0 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
              mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
q_1 = tl.load(q_base + offs_h[:, None] * stride_q_h
               + (1 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
              mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
q_2 = tl.load(q_base + offs_h[:, None] * stride_q_h
               + (2 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
              mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
q_3 = tl.load(q_base + offs_h[:, None] * stride_q_h
               + (3 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
              mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
q_4 = tl.load(q_base + offs_h[:, None] * stride_q_h
               + (4 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
              mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
q_5 = tl.load(q_base + offs_h[:, None] * stride_q_h
               + (5 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
              mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
q_6 = tl.load(q_base + offs_h[:, None] * stride_q_h
               + (6 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
              mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
q_7 = tl.load(q_base + offs_h[:, None] * stride_q_h
               + (7 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
              mask=mask_h[:, None], other=0.0).to(tl.bfloat16)

# Now the main loop uses q_0..q_7 without any Q loads:
for n_start in range(0, topk, BLOCK_N):
    # ... load KV, dequant ...
    qk = tl.dot(q_0, tl.trans(kv_0)).to(tl.float32)
    qk += tl.dot(q_1, tl.trans(kv_1)).to(tl.float32)
    # ... etc ...
```

### Register Cost of Q Preloading

Each Q tile is `[BLOCK_H, 64]` in BF16:
- BLOCK_H=16: 8 tiles x 16 x 64 x 2 bytes = 16 KB (acceptable)
- BLOCK_H=64: 8 tiles x 64 x 64 x 2 bytes = 64 KB (still fine, register file is 256 KB)
- BLOCK_H=128: 8 tiles x 128 x 64 x 2 bytes = 128 KB (marginal -- may trigger spill with large BLOCK_N)

---

## Strategy 4: Batched KV Loading

**Impact: Hides memory latency through instruction-level parallelism (ILP).**

### Concept

The GPU memory system can track dozens of in-flight load requests simultaneously. By issuing ALL loads before doing ANY compute, the loads overlap with each other and with the subsequent compute.

### Anti-pattern: Interleaved Load-Compute

```python
# BAD: Load one tile, process it, load next tile, process it...
# The GPU stalls waiting for each load to complete before the next one starts.
for tile_idx in range(7):
    nope_data = tl.load(tile_ptr + tile_idx * 64 + offs, mask=valid_2d, other=0)
    scale = tl.load(scale_ptr + tile_idx, mask=valid, other=127)
    kv_tile = dequant(nope_data, scale)  # GPU waits for load to finish
    qk += tl.dot(q_tile, tl.trans(kv_tile))
```

### Expert Pattern: Batch All Loads, Then Process

```python
# EXPERT: Issue ALL loads first (GPU tracks them all in parallel)
# --- Phase 1: Issue all loads ---
nope_0 = tl.load(tile_base + 0 * 64 + offs_tile[None, :], mask=valid_2d, other=0)
nope_1 = tl.load(tile_base + 1 * 64 + offs_tile[None, :], mask=valid_2d, other=0)
nope_2 = tl.load(tile_base + 2 * 64 + offs_tile[None, :], mask=valid_2d, other=0)
nope_3 = tl.load(tile_base + 3 * 64 + offs_tile[None, :], mask=valid_2d, other=0)
nope_4 = tl.load(tile_base + 4 * 64 + offs_tile[None, :], mask=valid_2d, other=0)
nope_5 = tl.load(tile_base + 5 * 64 + offs_tile[None, :], mask=valid_2d, other=0)
nope_6 = tl.load(tile_base + 6 * 64 + offs_tile[None, :], mask=valid_2d, other=0)

rope_lo = tl.load(rope_ptrs,     mask=valid_2d, other=0)
rope_hi = tl.load(rope_ptrs + 1, mask=valid_2d, other=0)

scale_0 = tl.load(scale_ptrs + 0, mask=valid, other=127)
scale_1 = tl.load(scale_ptrs + 1, mask=valid, other=127)
scale_2 = tl.load(scale_ptrs + 2, mask=valid, other=127)
scale_3 = tl.load(scale_ptrs + 3, mask=valid, other=127)
scale_4 = tl.load(scale_ptrs + 4, mask=valid, other=127)
scale_5 = tl.load(scale_ptrs + 5, mask=valid, other=127)
scale_6 = tl.load(scale_ptrs + 6, mask=valid, other=127)

# --- Phase 2: All data is now available, compute without stalls ---
scale_bf16_0 = tl.math.exp2(scale_0.to(tl.float32) - 127.0).to(tl.bfloat16)
kv_0 = (nope_0.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16) * scale_bf16_0[:, None]).to(tl.bfloat16)
# ... process all tiles ...
```

This pattern separates the memory-bound phase (all loads) from the compute-bound phase (all dequant + dot products), maximizing throughput on both.

---

## Strategy 5: Autotuning

**Impact: 2-5x speedup from finding the right tile sizes per workload.**

### @triton.autotune Setup

```python
@triton.autotune(
    configs=[
        # -- warps=4 (good for compute-bound) --
        triton.Config({"BLOCK_H": 16,  "BLOCK_N": 32},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16,  "BLOCK_N": 64},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 16,  "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64,  "BLOCK_N": 32},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 64,  "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 64},  num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        # -- warps=8 (good for memory-bound, large BLOCK_N) --
        triton.Config({"BLOCK_H": 64,  "BLOCK_N": 64},  num_warps=8, num_stages=1),
        triton.Config({"BLOCK_H": 128, "BLOCK_N": 64},  num_warps=8, num_stages=1),
    ],
    key=["total_tokens_bucket", "h_q", "topk"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _fused_gather_dequant_attn_kernel(...):
    ...
```

### Key Bucketing

Without bucketing, autotune re-triggers for every unique `total_tokens` value, causing massive recompilation:

```python
def _bucket_total_tokens(total_tokens: int) -> int:
    """Round up to nearest power of 2 so autotune cache is reusable."""
    if total_tokens <= 0:
        return 1
    n = 1
    while n < total_tokens:
        n <<= 1
    return n

# Usage in Python wrapper:
total_tokens_bucket = _bucket_total_tokens(total_tokens)
```

### Config Pruning

Eliminate configs that cause precision issues on AMD hardware:

```python
def _prune_configs(configs, named_args, **kwargs):
    """Remove configs that cause MFMA precision issues or waste resources."""
    h_q = named_args.get("h_q", 128)
    pruned = []
    for c in configs:
        block_h = c.kwargs.get("BLOCK_H", 16)
        # CRITICAL: BLOCK_H >= 32 with h_q <= 64 causes MFMA reduction
        # order changes on AMD, introducing numerical inconsistencies
        if h_q <= 64 and block_h >= 32:
            continue
        # Don't use BLOCK_H larger than h_q
        if block_h > h_q:
            continue
        pruned.append(c)
    return pruned if pruned else configs
```

### Why `num_stages=1` Everywhere

Expert kernels use `num_stages=1` universally. `num_stages>=2` tells Triton to software-pipeline loads (prefetch next iteration's data while computing current iteration). This doubles register pressure from prefetched data, causing spills. Since the expert kernel already hides latency through batched loading (Strategy 4), software pipelining provides no benefit.

### Fixed Config for Tiny Workloads

For very small batches (batch <= 4), autotuning overhead dominates. Create a separate, non-autotuned kernel copy:

```python
@triton.jit
def _fused_gather_dequant_attn_kernel_fixed(
    # Same signature, but BLOCK_H and BLOCK_N are hardcoded
    ...,
    BLOCK_H: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 64,
):
    ...  # Same kernel body

# Dispatch logic:
if total_tokens <= 4:
    _fused_gather_dequant_attn_kernel_fixed[grid](...)  # No autotuning overhead
else:
    _fused_gather_dequant_attn_kernel[grid](...)         # Autotuned
```

---

## Strategy 6: Split-K Parallelization

**Impact: Critical for large topk (>= 512). Exposes inter-SM parallelism.**

### The Problem

Without Split-K, a single CTA processes ALL topk tokens sequentially. With topk=4096 and BLOCK_N=64, that is 64 loop iterations per CTA. If total_tokens is small (e.g., batch=2), only a few CTAs are active, leaving most of the 256 CUs idle.

### The Solution: 3D Grid with Split-K

Split the topk dimension across `split_k` CTAs. Each CTA processes a chunk of topk and writes partial results. A separate combine kernel merges them.

### Main Kernel with Split-K

```python
@triton.jit
def _fused_gather_dequant_attn_splitk_kernel(
    # ... same args as before, plus:
    PartialOut_ptr,   # [split_k, total_tokens, h_q, d_v] float32
    PartialLSE_ptr,   # [split_k, total_tokens, h_q] float32
    split_k,          # number of splits
    topk_per_split,   # = ceil(topk / split_k)
    stride_po_k, stride_po_t, stride_po_h, stride_po_d,
    stride_pl_k, stride_pl_t, stride_pl_h,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)  # <--- Split-K dimension

    # Compute this split's topk range
    k_start = pid_k * topk_per_split
    k_end = tl.minimum(k_start + topk_per_split, topk)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    # ... load Q tiles (same as before) ...

    # Initialize accumulators
    acc_0 = tl.zeros([BLOCK_H, 64], dtype=tl.float32)
    # ... acc_1 through acc_7 ...
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Loop ONLY over this split's range [k_start, k_end)
    for n_start in range(k_start, k_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < k_end  # Note: k_end, not topk
        # ... same gather + dequant + QK + softmax + PV as before ...

    # Write PARTIAL results (float32, NOT bf16!)
    # LSE = m_i + log(l_i) -- but DON'T apply attn_sink here
    partial_lse = tl.where(l_i == 0.0, float("+inf"),
                           m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E)

    # Store partial output (unnormalized: acc / l_i, without attn_sink)
    output_scale = tl.where(l_i == 0.0, 0.0, 1.0 / l_i)
    partial_out_0 = acc_0 * output_scale[:, None]
    # ... all 8 tiles ...

    # Write to partial buffers
    po_base = PartialOut_ptr + pid_k * stride_po_k + pid_t * stride_po_t
    tl.store(po_base + offs_h[:, None] * stride_po_h
             + (0 * 64 + tl.arange(0, 64)[None, :]) * stride_po_d,
             partial_out_0, mask=mask_h[:, None])
    # ... store all 8 tiles ...

    pl_base = PartialLSE_ptr + pid_k * stride_pl_k + pid_t * stride_pl_t
    tl.store(pl_base + offs_h * stride_pl_h, partial_lse, mask=mask_h)
```

### Combine Kernel (split_k=2)

Write separate combine kernels for common split_k values (2, 4, 8) because Triton cannot unroll `for i in range(split_k)` at compile time when split_k is not constexpr.

```python
@triton.jit
def _combine_splitk2_kernel(
    PartialOut_ptr,     # [2, total_tokens, h_q, d_v] float32
    PartialLSE_ptr,     # [2, total_tokens, h_q] float32
    Out_ptr,            # [total_tokens, h_q, d_v] bf16
    LSE_ptr,            # [total_tokens, h_q] float32
    AttnSink_ptr,
    total_tokens, h_q, d_v,
    stride_po_k, stride_po_t, stride_po_h, stride_po_d,
    stride_pl_k, stride_pl_t, stride_pl_h,
    stride_o_t, stride_o_h, stride_o_d,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    INF_THRESHOLD: tl.constexpr = 1e30

    # Load LSE from both splits
    lse_0 = tl.load(PartialLSE_ptr + 0 * stride_pl_k + pid_t * stride_pl_t + pid_h * stride_pl_h)
    lse_1 = tl.load(PartialLSE_ptr + 1 * stride_pl_k + pid_t * stride_pl_t + pid_h * stride_pl_h)

    # Handle infinity: partials with LSE = +inf had no valid tokens
    lse_0_valid = tl.abs(lse_0) < INF_THRESHOLD
    lse_1_valid = tl.abs(lse_1) < INF_THRESHOLD
    lse_0_safe = tl.where(lse_0_valid, lse_0, float("-inf"))
    lse_1_safe = tl.where(lse_1_valid, lse_1, float("-inf"))

    # Online softmax merge
    max_lse = tl.maximum(lse_0_safe, lse_1_safe)
    exp_0 = tl.where(lse_0_valid, tl.math.exp2((lse_0_safe - max_lse) * LOG2E), 0.0)
    exp_1 = tl.where(lse_1_valid, tl.math.exp2((lse_1_safe - max_lse) * LOG2E), 0.0)
    sum_exp = exp_0 + exp_1
    sum_exp_safe = tl.where(sum_exp == 0.0, 1.0, sum_exp)

    scale_0 = exp_0 / sum_exp_safe
    scale_1 = exp_1 / sum_exp_safe

    # Combined LSE
    combined_lse = tl.where(
        sum_exp == 0.0, float("+inf"),
        max_lse + tl.math.log2(sum_exp) / LOG2E
    )

    # Apply attn_sink in the combine step (NOT in the main kernel)
    if HAS_ATTN_SINK:
        attn_sink_val = tl.load(AttnSink_ptr + pid_h)
        diff_clamped = tl.minimum(tl.maximum(attn_sink_val - combined_lse, -100.0), 100.0)
        exp_diff = tl.math.exp2(diff_clamped * LOG2E)
        sink_scale = 1.0 / (1.0 + exp_diff)
        scale_0 = scale_0 * sink_scale
        scale_1 = scale_1 * sink_scale

    # Combine output tiles
    offs_d = tl.arange(0, BLOCK_D)
    for d_start in range(0, d_v, BLOCK_D):
        d_offs = d_start + offs_d
        d_mask = d_offs < d_v

        out_0 = tl.load(PartialOut_ptr + 0 * stride_po_k + pid_t * stride_po_t
                        + pid_h * stride_po_h + d_offs * stride_po_d, mask=d_mask, other=0.0)
        out_1 = tl.load(PartialOut_ptr + 1 * stride_po_k + pid_t * stride_po_t
                        + pid_h * stride_po_h + d_offs * stride_po_d, mask=d_mask, other=0.0)

        combined = out_0 * scale_0 + out_1 * scale_1

        # Handle lonely queries
        is_lonely = sum_exp == 0.0
        combined = tl.where(is_lonely, 0.0, combined)

        tl.store(Out_ptr + pid_t * stride_o_t + pid_h * stride_o_h + d_offs * stride_o_d,
                 combined.to(tl.bfloat16), mask=d_mask)

    # Store final LSE
    final_lse = tl.where(sum_exp == 0.0, float("+inf"), combined_lse)
    tl.store(LSE_ptr + pid_t * h_q + pid_h, final_lse)
```

### Dual-Scope Split-K

For dual-scope (main + extra KV cache), each split's range may cross the boundary:

```python
# Each split covers a range of the COMBINED topk:
total_topk = topk_main + topk_extra
topk_per_split = (total_topk + split_k - 1) // split_k

# Inside kernel:
k_start = pid_k * topk_per_split
k_end = tl.minimum(k_start + topk_per_split, total_topk)

# Process main scope tokens in [k_start, min(k_end, topk_main))
main_end = tl.minimum(k_end, topk_main)
for n_start in range(k_start, main_end, BLOCK_N):
    # Use KV_Cache_Main, Indices_Main
    offs_n_local = n_start + tl.arange(0, BLOCK_N)
    # ... gather from main cache ...

# Process extra scope tokens in [max(k_start, topk_main), k_end)
extra_start = tl.maximum(k_start, topk_main)
for n_global in range(extra_start, k_end, BLOCK_N):
    # Use KV_Cache_Extra, Indices_Extra
    offs_n_local = (n_global - topk_main) + tl.arange(0, BLOCK_N)
    # ... gather from extra cache ...
```

### Dispatch Heuristics

```python
SMALL_BATCH_TOKEN_THRESHOLD = 8
DUAL_SCOPE_SPLITK_TOPK_THRESHOLD = 2048
SPLITK_HIGH_TOPK_THRESHOLD = 512

def _decide_splitk_dual_scope(total_tokens, h_q, total_topk):
    """Decide whether and how much Split-K to use."""
    use_splitk_for_small_bs = (
        total_tokens <= 8 and (h_q >= 128 or total_topk >= 1024))
    use_splitk_for_h64_large_topk = (
        h_q <= 64 and total_topk >= 1024
        and 8 < total_tokens <= 128)
    use_splitk_for_large_topk = (
        total_tokens > 64 and total_topk >= 2048)
    use_splitk_for_large_hq = (
        h_q > 64 and total_tokens > 8 and total_topk >= 256)

    if not any([use_splitk_for_small_bs, use_splitk_for_h64_large_topk,
                use_splitk_for_large_topk, use_splitk_for_large_hq]):
        return 0  # No split-K

    if total_tokens <= 8:
        return 8 if (total_topk >= 512 and total_tokens <= 4) else 4
    elif use_splitk_for_large_hq:
        return 4 if total_topk >= 512 else 2
    elif use_splitk_for_h64_large_topk:
        return 2
    else:
        return 4 if total_topk >= 8192 else 2
```

### Buffer Pool for Split-K Intermediates

```python
class SplitKBufferPool:
    """Cache intermediate split-K buffers to avoid repeated allocation.
    ~15% speedup for batch=2 by eliminating torch.empty() overhead."""
    _buffers = {}

    @classmethod
    def get_buffers(cls, split_k, total_tokens, h_q, d_v, device):
        key = (split_k, total_tokens, h_q, d_v, device)
        if key not in cls._buffers:
            cls._buffers[key] = {
                "partial_output": torch.empty(
                    split_k, total_tokens, h_q, d_v,
                    dtype=torch.float32, device=device),
                "partial_lse": torch.empty(
                    split_k, total_tokens, h_q,
                    dtype=torch.float32, device=device),
            }
        buf = cls._buffers[key]
        # Return buffers AND pre-computed strides
        po = buf["partial_output"]
        pl = buf["partial_lse"]
        return po, pl, po.stride(0), po.stride(1), po.stride(2), po.stride(3), \
               pl.stride(0), pl.stride(1), pl.stride(2)

    @classmethod
    def clear(cls):
        """Call when VRAM pressure is high. Cached buffers are NOT reclaimed by GC."""
        cls._buffers.clear()
```

---

## Strategy 7: Grid Swap

**Impact: ~20% speedup from improved L2 cache hit rate.**

### The Problem

The baseline grid has tokens as `program_id(0)` (fast-varying):

```python
# BASELINE grid ordering:
grid = (total_tokens, triton.cdiv(h_q, BLOCK_H))
# In kernel:
pid_t = tl.program_id(0)   # token (fast-varying)
pid_h = tl.program_id(1)   # head group (slow-varying)
```

On AMD CDNA GPUs, `program_id(0)` is the fastest-varying dimension. Consecutive CTAs in this dimension are dispatched first. With the baseline ordering, CTA 0 and CTA 1 process different tokens -- they read completely different KV data, thrashing the L2 cache.

### The Solution: Heads Fast, Tokens Slow

```python
# EXPERT grid ordering:
grid = (triton.cdiv(h_q, BLOCK_H), total_tokens)
# In kernel:
pid_h = tl.program_id(0)   # head group (fast-varying)
pid_t = tl.program_id(1)   # token (slow-varying)
```

Now CTA 0 and CTA 1 process different head groups but the SAME token. They read from the same KV cache locations, which will already be in L2 from the first CTA's loads.

### Grid Swap for Split-K

The 3D grid with Split-K extends naturally:

```python
# 3D grid: (heads, tokens, split_k)
grid = (triton.cdiv(h_q, BLOCK_H), total_tokens, split_k)
pid_h = tl.program_id(0)   # heads fast
pid_t = tl.program_id(1)   # tokens slow
pid_k = tl.program_id(2)   # split-K slowest
```

---

## Strategy 8: topk_length Early Exit

**Impact: Proportional to sparsity -- skips entire loop iterations.**

### The Problem

Some tokens have fewer valid KV entries than `topk`. Without early exit, the kernel loops through all `topk` entries, loading and masking out invalid ones.

### The Solution: Break When Past topk_length

```python
# Load topk_len ONCE before the loop
if HAS_TOPK_LENGTH:
    topk_len = tl.load(TopkLength_ptr + pid_t)
else:
    topk_len = topk

for n_start in range(0, topk, BLOCK_N):
    # Early exit: skip ALL remaining blocks
    if HAS_TOPK_LENGTH:
        if n_start >= topk_len:
            break

    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < topk
    if HAS_TOPK_LENGTH:
        mask_n = mask_n & (offs_n < topk_len)

    # ... rest of loop body ...
```

### Performance Impact Example

With topk=1024, topk_length=100, BLOCK_N=64:
- Baseline: 16 loop iterations (processes 1024 tokens, masks out 924)
- With early exit: 2 loop iterations (processes 128 tokens, skips 14 iterations entirely)

### Using `tl.constexpr` for Dead Code Elimination

Declare `HAS_TOPK_LENGTH` as `tl.constexpr` so the compiler eliminates the topk_length code path entirely when not needed:

```python
def _fused_kernel(
    ...,
    HAS_TOPK_LENGTH: tl.constexpr,  # True or False, known at compile time
):
    # When HAS_TOPK_LENGTH is False at compile time, the compiler
    # removes all topk_length-related code, saving registers and branches.
    if HAS_TOPK_LENGTH:
        topk_len = tl.load(TopkLength_ptr + pid_t)
    # ...
```

---

## Strategy 9: Memory Safety

**Impact: Prevents silent data corruption and GPU faults.**

### int64 Pointer Arithmetic for >2GB Tensors

The KV cache can easily exceed 2 GB (e.g., 4096 blocks x 64 tokens/block x 576 bytes/token = 150 MB for a single scope -- but in practice, production caches are much larger). When any dimension multiplied by a stride exceeds 2^31 (2,147,483,648), int32 arithmetic silently overflows.

```python
# BAD: int32 overflow
block_offset = block_idx * stride_kv_block  # int32 overflow if stride > 2^31 / max_block_idx

# CORRECT: use int64 for all potentially large pointer computations
block_idx_64 = block_idx.to(tl.int64)
stride_kv_block_64 = tl.cast(stride_kv_block, tl.int64)
offset_in_block_64 = offset_in_block.to(tl.int64)

kv_block_base = KV_Cache + block_idx_64 * stride_kv_block_64
nope_rope_offset = offset_in_block_64 * tl.cast(BYTES_PER_TOKEN_DATA, tl.int64)
```

### AMD buffer_ops Disable for >2GB KV Cache

AMD's `buffer_ops` optimization uses int32 address arithmetic internally. For KV caches exceeding ~2 GB, this causes silent data corruption.

```python
BUFFER_OPS_DISABLE_THRESHOLD = 2 * 1024 * 1024 * 1024  # 2 GB in bytes

def _check_and_disable_buffer_ops(kv_cache, extra_kv_cache=None):
    """Disable AMD buffer_ops when KV cache exceeds 2GB."""
    kv_size = kv_cache.numel() * kv_cache.element_size()
    needs_disable = kv_size > BUFFER_OPS_DISABLE_THRESHOLD

    if extra_kv_cache is not None:
        extra_size = extra_kv_cache.numel() * extra_kv_cache.element_size()
        needs_disable = needs_disable or (extra_size > BUFFER_OPS_DISABLE_THRESHOLD)

    return needs_disable

# In the Python dispatch function:
needs_disable = _check_and_disable_buffer_ops(kv_cache, extra_kv_cache)

if needs_disable:
    # Approach 1: Use triton.knobs (if available in your Triton version)
    with triton.knobs.amd.scope():
        triton.knobs.amd.use_buffer_ops = False
        _fused_gather_dequant_attn_kernel[grid](...)
else:
    _fused_gather_dequant_attn_kernel[grid](...)
```

### Cast Strides to int64 Inside Kernels

Even when the Python-side value fits in int32, the product of a stride and an index can overflow:

```python
# Inside kernel: always cast strides before multiplication
stride_q_t_64 = tl.cast(stride_q_t, tl.int64)
pid_t_64 = tl.cast(pid_t, tl.int64)
q_offset = pid_t_64 * stride_q_t_64  # safe even for large pid_t
```

---

## Strategy 10: Attention Sink

**Impact: Correctness-critical. Eliminates 2 Python-side kernel launches.**

### Decode Formula (Sigmoid, NOT Logsumexp)

Decode uses a multiplicative sigmoid-like correction that is fundamentally different from prefill:

```
Decode:  output *= 1 / (1 + exp(attn_sink - lse))
         equivalently: output *= l_i / (l_i + exp(attn_sink - m_i))

Prefill: new_lse = log(exp(kernel_lse) + exp(attn_sink))
         output *= exp(kernel_lse - new_lse)
```

Using the prefill formula for decode (or vice versa) produces subtly wrong results that may pass loose tolerance checks but will fail tight correctness tests.

### In-Kernel Decode Implementation

```python
# After the main KV loop, when m_i and l_i are finalized:

if HAS_ATTN_SINK:
    # Load per-head attn_sink value
    attn_sink_vals = tl.load(AttnSink_ptr + offs_h, mask=mask_h, other=0.0)

    # Compute exp(attn_sink - m_i) -- the denominator correction
    exp_attn_sink_minus_m = tl.math.exp2((attn_sink_vals - m_i) * LOG2E)

    # denominator = l_i + exp(attn_sink - m_i)
    denominator = l_i + exp_attn_sink_minus_m

    # Protect against 0/0 (lonely query with attn_sink = -inf)
    denominator = tl.where(denominator == 0.0, 1.0, denominator)

    # Scale factor: l_i / denominator
    # But we already have acc = sum(p_i * v_i), and l_i = sum(p_i)
    # So final output = acc / denominator (not acc / l_i * l_i / denominator)
    output_scale = 1.0 / denominator
else:
    # No attn_sink: standard normalization by l_i
    output_scale = tl.where(l_i == 0.0, 0.0, 1.0 / l_i)

# Handle lonely queries (no valid KV tokens)
is_lonely_q = l_i == 0.0

# Apply to all 8 accumulators
acc_0 = tl.where(is_lonely_q[:, None], 0.0, acc_0 * output_scale[:, None])
acc_1 = tl.where(is_lonely_q[:, None], 0.0, acc_1 * output_scale[:, None])
acc_2 = tl.where(is_lonely_q[:, None], 0.0, acc_2 * output_scale[:, None])
acc_3 = tl.where(is_lonely_q[:, None], 0.0, acc_3 * output_scale[:, None])
acc_4 = tl.where(is_lonely_q[:, None], 0.0, acc_4 * output_scale[:, None])
acc_5 = tl.where(is_lonely_q[:, None], 0.0, acc_5 * output_scale[:, None])
acc_6 = tl.where(is_lonely_q[:, None], 0.0, acc_6 * output_scale[:, None])
acc_7 = tl.where(is_lonely_q[:, None], 0.0, acc_7 * output_scale[:, None])

# LSE: report the raw LSE (before attn_sink), set +inf for lonely queries
lse = tl.where(is_lonely_q, float("+inf"),
               m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E)
```

### Combine Kernel Sink Handling with Clamping

When attn_sink is applied in the combine kernel (Split-K path), clamping is required to prevent float32 overflow:

```python
# In combine kernel, after computing combined_lse:
if HAS_ATTN_SINK:
    attn_sink_val = tl.load(AttnSink_ptr + pid_h)

    # CRITICAL: clamp the difference to prevent exp overflow
    # exp(100) ~ 2.7e43 (fits in float32); exp(200) = inf
    diff_clamped = tl.minimum(tl.maximum(attn_sink_val - combined_lse, -100.0), 100.0)
    exp_diff = tl.math.exp2(diff_clamped * LOG2E)
    sink_scale = 1.0 / (1.0 + exp_diff)

    # Apply sink scaling to all partial output scales
    scale_0 = scale_0 * sink_scale
    scale_1 = scale_1 * sink_scale
    # ... for all split_k partials ...
```

---

## Strategy 11: Python-side Low Overhead

**Impact: Significant for small batches where Python overhead dominates kernel time.**

### Buffer Pool

Avoid repeated `torch.empty()` calls for output and intermediate tensors:

```python
class DecodeBufferPool:
    """Reuse output buffers across decode calls."""
    _cache = {}

    @classmethod
    def get_output(cls, total_tokens, h_q, d_v, device):
        key = ("out", total_tokens, h_q, d_v, device)
        if key not in cls._cache:
            cls._cache[key] = torch.empty(
                total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device)
        return cls._cache[key]

    @classmethod
    def get_lse(cls, total_tokens, h_q, device):
        key = ("lse", total_tokens, h_q, device)
        if key not in cls._cache:
            cls._cache[key] = torch.empty(
                total_tokens, h_q, dtype=torch.float32, device=device)
        return cls._cache[key]
```

### Stride Precomputation

Pre-compute and cache strides to avoid Python-side `tensor.stride()` calls on every invocation:

```python
class StrideCache:
    """Cache stride values alongside buffer pool entries."""
    _strides = {}

    @classmethod
    def get_strides(cls, tensor, key_prefix):
        key = (key_prefix, tensor.data_ptr(), tensor.shape)
        if key not in cls._strides:
            cls._strides[key] = tuple(tensor.stride())
        return cls._strides[key]
```

### Dispatch Separation

Separate the "decide what to do" logic from the "do it" logic. Pre-compute all dispatch decisions before any kernel launch:

```python
def triton_sparse_attn_decode_optimized(q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink):
    """Optimized dispatch with minimal Python overhead."""
    b, s_q, h_q, d_qk = q.shape
    total_tokens = b * s_q

    # --- Phase 1: All dispatch decisions (pure Python, no GPU ops) ---
    has_extra = extra_kv_scope is not None
    total_topk = kv_scope.topk + (extra_kv_scope.topk if has_extra else 0)
    split_k = _decide_splitk_dual_scope(total_tokens, h_q, total_topk)
    use_fused = True  # Always fused after Phase 17

    # --- Phase 2: Get buffers (cached, no allocation) ---
    out = DecodeBufferPool.get_output(total_tokens, h_q, d_v, q.device)
    lse = DecodeBufferPool.get_lse(total_tokens, h_q, q.device)

    if split_k > 0:
        po, pl, *strides = SplitKBufferPool.get_buffers(
            split_k, total_tokens, h_q, d_v, q.device)

    # --- Phase 3: Launch kernel(s) ---
    if split_k > 0:
        grid = (triton.cdiv(h_q, BLOCK_H), total_tokens, split_k)
        _fused_gather_dequant_attn_splitk_kernel[grid](...)
        _combine_kernel[combine_grid](...)
    else:
        grid = (triton.cdiv(h_q, BLOCK_H), total_tokens)
        _fused_gather_dequant_attn_kernel[grid](...)

    return out, lse
```

### Reuse Dummy Tensors

When kernel arguments are optional (e.g., extra KV cache, attn_sink), avoid allocating empty tensors:

```python
# Create once at module level
_DUMMY_TENSOR = torch.zeros(1, dtype=torch.float32, device="cuda")

# In dispatch:
attn_sink_ptr = attn_sink if attn_sink is not None else _DUMMY_TENSOR
```

---

## Strategy 12: Online Softmax Numerics

**Impact: Correctness-critical. Must be implemented exactly right.**

### Complete Update Loop

```python
# Initialize before the loop
m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
acc_0 = tl.zeros([BLOCK_H, 64], dtype=tl.float32)
# ... acc_1 through acc_7 ...

LOG2E: tl.constexpr = 1.4426950408889634

for n_start in range(0, topk, BLOCK_N):
    # ... load KV tiles, compute kv_0..kv_7 ...

    # --- Step 1: Compute QK scores ---
    qk = tl.dot(q_0, tl.trans(kv_0)).to(tl.float32)
    qk += tl.dot(q_1, tl.trans(kv_1)).to(tl.float32)
    qk += tl.dot(q_2, tl.trans(kv_2)).to(tl.float32)
    qk += tl.dot(q_3, tl.trans(kv_3)).to(tl.float32)
    qk += tl.dot(q_4, tl.trans(kv_4)).to(tl.float32)
    qk += tl.dot(q_5, tl.trans(kv_5)).to(tl.float32)
    qk += tl.dot(q_6, tl.trans(kv_6)).to(tl.float32)
    qk += tl.dot(q_7, tl.trans(kv_7)).to(tl.float32)
    # qk shape: [BLOCK_H, BLOCK_N]

    # --- Step 2: Scale and mask ---
    qk = qk * sm_scale
    qk = tl.where(valid[None, :], qk, float("-inf"))

    # --- Step 3: Compute block-local max ---
    m_ij = tl.max(qk, axis=1)     # [BLOCK_H]
    m_new = tl.maximum(m_i, m_ij)  # [BLOCK_H]

    # --- Step 4: Compute rescale factor (alpha) ---
    # CRITICAL: when m_i == -inf (first iteration with valid tokens):
    #   m_i - m_new = -inf - m_new = -inf
    #   exp2(-inf) = 0
    #   alpha = 0 -> correctly zeros old accumulator
    # DO NOT use alpha=1.0 when m_i==-inf; that preserves garbage from uninitialized state.
    alpha = tl.where(m_i == float("-inf"), 0.0,
                     tl.math.exp2((m_i - m_new) * LOG2E))
    # alpha shape: [BLOCK_H]

    # --- Step 5: Compute attention weights ---
    # CRITICAL: when qk == -inf (invalid position):
    #   qk - m_new = -inf - m_new = -inf
    #   exp2(-inf) = 0
    #   p = 0 -> correctly ignores invalid positions
    p = tl.where(qk == float("-inf"), 0.0,
                 tl.math.exp2((qk - m_new[:, None]) * LOG2E))
    # p shape: [BLOCK_H, BLOCK_N]

    # --- Step 6: Update running sum ---
    l_new = alpha * l_i + tl.sum(p, axis=1)
    # l_new shape: [BLOCK_H]

    # --- Step 7: Rescale old accumulators and add new contribution ---
    p_bf16 = p.to(tl.bfloat16)

    # CRITICAL: alpha must be applied to ALL 8 accumulators
    # Missing any one produces wrong output for the corresponding d_v tile
    acc_0 = acc_0 * alpha[:, None] + tl.dot(p_bf16, kv_0).to(tl.float32)
    acc_1 = acc_1 * alpha[:, None] + tl.dot(p_bf16, kv_1).to(tl.float32)
    acc_2 = acc_2 * alpha[:, None] + tl.dot(p_bf16, kv_2).to(tl.float32)
    acc_3 = acc_3 * alpha[:, None] + tl.dot(p_bf16, kv_3).to(tl.float32)
    acc_4 = acc_4 * alpha[:, None] + tl.dot(p_bf16, kv_4).to(tl.float32)
    acc_5 = acc_5 * alpha[:, None] + tl.dot(p_bf16, kv_5).to(tl.float32)
    acc_6 = acc_6 * alpha[:, None] + tl.dot(p_bf16, kv_6).to(tl.float32)
    acc_7 = acc_7 * alpha[:, None] + tl.dot(p_bf16, kv_7).to(tl.float32)

    # --- Step 8: Update state ---
    m_i = m_new
    l_i = l_new

# --- After the loop: Compute LSE ---
# LSE = m_i + log(l_i) = m_i + log2(l_i) / LOG2E
# Handle l_i == 0 (lonely query) to avoid log(0) = -inf
lse = tl.where(
    l_i == 0.0,
    float("+inf"),
    m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E
)
```

### Why `exp2` Instead of `exp`

On AMD CDNA hardware, `exp2` maps directly to a hardware instruction (v_exp_f32). `exp(x)` is computed as `exp2(x * LOG2E)` by the compiler anyway, but making it explicit:
1. Avoids compiler-dependent optimization
2. Makes the `LOG2E` factor visible for correctness review
3. Matches AMD hardware instruction semantics

### Edge Case: First Iteration

When the kernel encounters its first block of valid tokens (m_i was -inf before):

```
m_i = -inf
m_ij = max(qk) = some_value
m_new = max(-inf, some_value) = some_value

alpha = exp2((-inf - some_value) * LOG2E) = exp2(-inf) = 0

acc *= alpha   ->  acc *= 0  ->  acc = 0  (CORRECT: zeros old accumulator)
l_i *= alpha   ->  l_i *= 0  ->  l_i = 0  (CORRECT: resets running sum)
```

If alpha were set to 1.0 instead (as in some buggy implementations), the old zero-initialized accumulator would be preserved, which is semantically wrong -- it only works by accident because acc was zero-initialized.

---

## Common Mistakes Table

| # | Mistake | Symptom | Root Cause | Fix |
|---|---------|---------|------------|-----|
| 1 | Scale uses `x / 127` instead of `exp2(x - 127)` | Completely wrong dequantized values | Confusing E8M0 with linear scaling | `tl.math.exp2(scale.to(tl.float32) - 127.0).to(tl.bfloat16)` |
| 2 | RoPE byte order reversed (hi first, lo second) | Wrong RoPE values, attention scores off | Little-endian byte order assumed incorrectly | `lo \| (hi << 8)` -- low byte at even offset, high byte at odd offset |
| 3 | FP8 conversion uses `bitcast=False` | Values interpreted as integers, not FP8 bit patterns | `.to(float8)` does VALUE conversion by default | Must use `.to(tl.float8e4nv, bitcast=True)` |
| 4 | Scale `other=0` for invalid positions | Invalid positions get scale = exp2(-127) ~ 5.9e-39 | Zero is wrong neutral value for E8M0 | `other=127` -> scale = exp2(0) = 1.0 (neutral multiplier) |
| 5 | int32 pointer overflow for >2GB tensors | Garbage output, silent wrong results, GPU fault | `block_idx * stride` exceeds 2^31 | Cast to int64: `block_idx.to(tl.int64) * tl.cast(stride, tl.int64)` |
| 6 | AMD buffer_ops enabled with KV >2GB | Silent data corruption | AMD buffer_ops uses int32 address arithmetic | Detect size > 2GB, set `triton.knobs.amd.use_buffer_ops = False` |
| 7 | Split-K combine uses simple averaging | Wrong results -- each partial has different normalization | Ignoring per-partial LSE | Use online softmax merge: `w_i = exp(lse_i - max_lse)`, weighted sum |
| 8 | BLOCK_H >= 32 with h_q <= 64 | Numerical inconsistencies, results differ per run | AMD MFMA reduction order changes | Prune configs: only BLOCK_H <= 16 when h_q <= 64 |
| 9 | `num_stages > 1` in autotune configs | Register spill, slower than num_stages=1 | Prefetch doubles register pressure | Always use `num_stages=1` |
| 10 | Partial Split-K output stored as BF16 | Precision loss in combine step | BF16 truncation before softmax merge | Partial output MUST be float32 |
| 11 | Combine kernel ignores +inf LSE from lonely partials | NaN propagation from `exp(+inf - max_lse)` | +inf in LSE means "no valid tokens" | Use `INF_THRESHOLD = 1e30`; treat `abs(lse) >= threshold` as invalid |
| 12 | No lonely query handling | Divide-by-zero NaN (1/l_i when l_i=0) | Query with zero valid KV tokens | Check `is_lonely = l_i == 0.0`; output 0, LSE +inf |
| 13 | Attn_sink exponent not clamped before exp() | float32 overflow to inf | `exp(attn_sink - lse)` can be > 1e38 | `diff_clamped = clamp(attn_sink - lse, -100, 100)` before `exp2()` |
| 14 | Data/scale offset miscalculated in KV cache | Loads garbage data -- nope tiles read scale bytes or vice versa | Off-by-one in offset formulas | data offset: `token_idx * 576`; scale offset: `block_size * 576 + token_idx * 8` |
| 15 | `tl.dot` operand precision mismatch | Compilation error or silent wrong results | Dot product requires matching types | QK: both sides BF16; result `.to(tl.float32)` |
| 16 | Using prefill attn_sink formula for decode | Subtly wrong results that pass loose tolerance | Different mathematical formulations | Decode: sigmoid `1/(1+exp(sink-lse))`. Prefill: logsumexp merge |
| 17 | Grid dimension ordering wrong (tokens fast) | ~2x performance loss (correctness unaffected) | Cache thrashing on AMD CDNA | `program_id(0)` = head dim (fast), `program_id(1)` = token dim (slow) |
| 18 | alpha = 1.0 when m_i == -inf | Fragile correctness (works only if acc initialized to 0) | Semantic error: old acc should be ZEROED | `alpha = tl.where(m_i == float("-inf"), 0.0, exp2(...))` |
| 19 | Forgetting to rescale ANY accumulator | Wrong output in the tile that was skipped | All 8 acc tiles need `*= alpha` | Double-check: acc_0 through acc_7 ALL rescaled |
| 20 | Missing `tl.where(valid_2d, kv, 0.0)` after dequant | NaN from invalid FP8 values propagates | FP8 garbage bits may decode to NaN | Always mask invalid positions to 0.0 after dequant |

---

## OOM Troubleshooting

### Two Types of OOM

OOM in GPU kernels manifests in two fundamentally different ways. Confusing them leads to the wrong fix.

#### Type 1: Global Memory (VRAM) OOM

**Symptom**: `torch.cuda.OutOfMemoryError` or GPU memory fault that wedges the device (on AMD MI355X, all subsequent operations hang until GPU reset).

**Cause**: Allocating tensors that exceed available VRAM.

| Cause | Size Formula | Solution |
|-------|-------------|----------|
| Baseline's BF16 gather buffer | `batch x topk x d_qk x 2` bytes | Fused kernel (zero extra VRAM) |
| Split-K partial output buffers | `split_k x total_tokens x h_q x d_v x 4` bytes (f32) | Reduce split_k; use buffer pool; skip split-K for small topk |
| `.contiguous()` hidden copy | `total_tokens x h_q x d_qk x 2` bytes | Pass strides to kernel instead |
| Dual-scope `torch.cat` | `(main_topk + extra_topk) x d_qk x 2` bytes | Dual-scope fused kernel (process scopes sequentially) |
| Float32 fallback matmul | `batch x topk x d_qk x 4` bytes (f32!) | DELETE the fallback path entirely; always use Triton |

**Anti-pattern**: NEVER fall back to PyTorch float32 matmul for large topk. Example: b=148, topk=16384, d_qk=576 -> the f32 buffer alone is ~9.66 GB.

**Diagnostic**:

```python
def print_gpu_mem(tag=""):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[{tag}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
```

#### Type 2: Register / Per-SM Resource OOM

**Symptom**: Kernel launch failure (error at compile time or launch time), or extreme slowness from register spill to local memory (VRAM-backed scratch space). NOT `torch.cuda.OutOfMemoryError`.

**Cause**: Kernel requires more registers than the 256 KB register file per CU.

| Cause | Register Impact | Solution |
|-------|----------------|----------|
| Single `[BLOCK_H, 512]` accumulator | BLOCK_H=16: 32KB; BLOCK_H=128: 256KB | Split into 8 x `[BLOCK_H, 64]` tile accumulators |
| BLOCK_H too large (128) | All data structures scale with BLOCK_H | Cap BLOCK_H in config pruning; use BLOCK_H=16 or 64 |
| BLOCK_N too large (128) with batch loading | 8 KV tiles x BLOCK_N x 64 x 2 bytes = 128KB | Reduce BLOCK_N or drop batch-loading for this config |
| `num_stages > 1` | Doubles register pressure from prefetch buffers | Always use `num_stages=1` |
| All autotune configs too large | Every config triggers spill | Include a safe fallback: `BLOCK_H=16, BLOCK_N=32, num_warps=4` |

**Register usage estimator**:

```python
def estimate_registers(BLOCK_H, BLOCK_N, TILE_SIZE=64, num_tiles=8):
    """Estimate total register file usage for a given config."""
    regs = 0
    regs += num_tiles * BLOCK_H * TILE_SIZE * 2   # Q tiles (bf16, preloaded)
    regs += num_tiles * BLOCK_H * TILE_SIZE * 4   # Accumulators (f32)
    regs += num_tiles * BLOCK_N * TILE_SIZE * 2   # KV tiles (bf16, batch-loaded)
    regs += BLOCK_H * BLOCK_N * 4                  # QK scores (f32)
    regs += BLOCK_H * 4 * 3                        # m_i, l_i, alpha (f32 scalars)
    regs += BLOCK_H * BLOCK_N * 4                  # p (f32, attention weights)
    regs += 7 * BLOCK_N * 2                        # Scales (bf16)
    regs = int(regs * 1.3)                         # ~30% overhead for temps/spills
    status = ("EXCEEDS 256KB!" if regs > 256 * 1024
              else "High (occupancy=1)" if regs > 200 * 1024
              else "OK (occupancy>=2)")
    print(f"  BLOCK_H={BLOCK_H:3d}, BLOCK_N={BLOCK_N:3d}: {regs/1024:.0f} KB -- {status}")

# Example outputs:
# BLOCK_H= 16, BLOCK_N= 32:   82 KB -- OK (occupancy>=2)
# BLOCK_H= 16, BLOCK_N= 64:  157 KB -- OK (occupancy>=2)
# BLOCK_H= 64, BLOCK_N= 64:  413 KB -- EXCEEDS 256KB!
# BLOCK_H=128, BLOCK_N=128:  836 KB -- EXCEEDS 256KB!
```

### Defense Layers

Apply these layered defenses throughout:

| Layer | Strategy | When to Apply | Effect |
|-------|----------|---------------|--------|
| Fused kernel | Eliminate intermediate buffers | Always | Zero extra VRAM for KV data |
| BF16 intermediates | Never use float32 for intermediate buffers | Always | 2x less memory than float32 |
| Token chunking | Split along token dim when buffer > 2 GB | Before allocation | Each chunk fits in memory |
| Split-K with buffer pool | Reuse partial buffers | Large topk with split-K | Avoids repeated allocation |
| No float32 fallback | Delete PyTorch matmul path | Always | Prevents the 9+ GB OOM |

---

## Recommended Implementation Order

The strategies above are listed by category, not by implementation order. Follow this sequence for the smoothest development experience.

**⚠️ CRITICAL: FP8 in-kernel dequant (Strategy 1) is the #1 production requirement.** Without it, the kernel cannot replace tilelang in the live serving stack — Python-side dequant adds a separate GPU kernel + 3x HBM traffic that negates all other optimizations. Even a 16x-faster bf16 kernel loses at e2e because of the dequant overhead.

### Phase 1: FP8 Fused Kernel (MUST DO FIRST — Production Blocker)

1. **Strategy 12: Online Softmax Numerics** -- Get the math right first. Use `exp2` with `LOG2E`, handle `m_i == -inf` correctly with `alpha = 0`.
2. **Strategy 2: Accumulator Splitting** -- Split `[BLOCK_H, 512]` into 8 x `[BLOCK_H, 64]`. This unblocks the fused kernel.
3. **Strategy 1: Fused Gather + Dequant + Attention** -- THE MOST IMPORTANT OPTIMIZATION. Read FP8 bytes directly from KV cache, dequant with E8M0 scales, reconstruct RoPE byte-pairs, compute attention — all in one kernel, zero intermediate buffers. Implement for MODEL1 (d_qk=512) first.
4. **Strategy 9: Memory Safety** -- Add int64 pointer arithmetic and AMD buffer_ops guard. Essential: KV cache can exceed 2GB.
5. **Strategy 10: Attention Sink** -- Fuse attn_sink into the kernel using the decode (sigmoid) formula. Required for correctness.

**Verification**: `python test_triton_decode.py --mode fp8` must pass. The kernel reads FP8 KV cache directly and produces correct bf16 output. Compare against `ref_sparse_attn_decode()` with tolerance atol=2e-2, rtol=2e-2.

### Phase 2: MFMA + Multi-Head Packing (Performance)

6. **Strategy 3: Q Preloading** -- Hoist Q loads out of the loop. Trivial change, immediate benefit.
7. **Multi-head packing (BLOCK_H)** -- Pack 16+ heads per program, use 2D `tl.dot` for QK and PV. This activates MFMA tensor cores for 2x throughput.
8. **Strategy 4: Batched KV Loading** -- Issue all tile loads before compute within the fused kernel.

**Verification**: Confirm correctness still passes. Measure latency reduction vs Phase 1.

### Phase 3: Performance Tuning

9. **Strategy 5: Autotuning** -- Add @triton.autotune with pruning. Find the best configs for your workload shapes.
10. **Strategy 7: Grid Swap** -- Swap grid dimensions. One-line change in grid definition, large impact.
11. **Strategy 8: topk_length Early Exit** -- Add the early break. Simple conditional.
12. **Strategy 11: Python-side Low Overhead** -- Buffer pool, stride cache, dispatch separation.

### Phase 4: Scale-Out

13. **Strategy 6: Split-K Parallelization** -- Implement the Split-K main kernel, combine kernel, and dispatch heuristics. This is the most complex strategy; save it for last.

**Verification**: Test with large topk (4096+) and small batch (2-8). Confirm no OOM and correct results.

---

## Reference Formulas

### E8M0 Dequantization

```
scale = 2^(uint8_value - 127)
dequantized_bf16 = fp8_value.to(bf16) * scale

In Triton:
  scale = tl.math.exp2(uint8_val.to(tl.float32) - 127.0).to(tl.bfloat16)
  kv_bf16 = (fp8_bytes.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16) * scale[:, None]).to(tl.bfloat16)
```

E8M0 is a pure power-of-2 format: 8-bit exponent, 0-bit mantissa. Each uint8 byte encodes one scale. The bias is 127, matching the IEEE 754 single-precision exponent bias.

### RoPE Byte Reconstruction

```
RoPE data: 64 BF16 values stored as 128 raw bytes (not typed FP8).

For each BF16 element at position i:
  lo_byte = load(base + 2*i)       # Low byte (even offset)
  hi_byte = load(base + 2*i + 1)   # High byte (odd offset)
  uint16_val = lo_byte.to(uint16) | (hi_byte.to(uint16) << 8)
  bf16_val = uint16_val.to(bf16, bitcast=True)

In Triton:
  rope_lo = tl.load(rope_ptrs, mask=valid_2d, other=0).to(tl.uint16)
  rope_hi = tl.load(rope_ptrs + 1, mask=valid_2d, other=0).to(tl.uint16)
  kv_rope = (rope_lo | (rope_hi << 8)).to(tl.bfloat16, bitcast=True)
```

### Split-K Combine (Online Softmax Merge)

To merge N partial results `{(out_i, lse_i)}`:

```
max_lse = max(lse_0, lse_1, ..., lse_{N-1})
w_i = exp(lse_i - max_lse)   for each i (0 if lse_i was +inf / invalid)
sum_w = sum(w_i)
scale_i = w_i / sum_w

combined_output = sum(out_i * scale_i)
combined_lse = max_lse + log(sum_w)

In exp2 form (for AMD):
  w_i = exp2((lse_i - max_lse) * LOG2E)
```

This is NOT a simple average. Each partial has its own normalization state (m_i, l_i) that must be properly reconciled.

### Attn_Sink: Decode vs Prefill

**Decode formula** (sigmoid-like, multiplicative):

```
output *= l_i / (l_i + exp(attn_sink - m_i))
       = 1 / (1 + exp(attn_sink - lse))

When l_i == 0 (lonely query): output = 0, lse = +inf
```

**Prefill formula** (logsumexp merge, additive):

```
new_lse = log(exp(kernel_lse) + exp(attn_sink))
output *= exp(kernel_lse - new_lse)

Edge cases:
  attn_sink == +inf: new_lse = +inf, output *= 0
  attn_sink == -inf: new_lse = kernel_lse, output unchanged
  kernel_lse == -inf: new_lse = attn_sink, output *= 0
  both -inf: new_lse = -inf, output = 0
```

### Online Softmax (exp2 form)

```
LOG2E = 1.4426950408889634

# Standard exp(x) -> exp2(x * LOG2E) replacement:
exp(x) = exp2(x * LOG2E)

# Softmax rescaling:
alpha = exp2((m_old - m_new) * LOG2E)

# Attention weights:
p = exp2((qk - m) * LOG2E)

# LSE computation:
lse = m + log(l) = m + log2(l) / LOG2E
```

### AMD CDNA Grid Dispatch Order

```
program_id(0) = fastest-varying dimension (dispatched first within a wave)
program_id(1) = next fastest
program_id(2) = slowest (if 3D grid)

For cache locality: make heads fast (pid(0)) and tokens slow (pid(1))
  grid = (cdiv(h_q, BLOCK_H), total_tokens)
  grid_3d = (cdiv(h_q, BLOCK_H), total_tokens, split_k)

This ensures consecutive CTAs process different heads of the SAME token,
sharing KV data in L1/L2 cache.
```

### MODEL1 KV Cache Address Computation

```
Given:
  token_index     = flat index into the KV cache
  block_size      = number of tokens per block
  stride_kv_block = bytes per block (= block_size * 576 + block_size * 8)

block_idx        = token_index // block_size
offset_in_block  = token_index % block_size

data_offset  = block_idx * stride_kv_block + offset_in_block * 576
scale_offset = block_idx * stride_kv_block + block_size * 576 + offset_in_block * 8

nope_tile_i  = data_offset + i * 64        (for i in 0..6)
rope_data    = data_offset + 448            (128 bytes = 64 BF16 values)
scale_i      = scale_offset + i             (for i in 0..6, uint8 each)
```

---

## Appendix: Baseline Kernel for Reference

For comparison, here is the original baseline kernel that the above strategies optimize:

```python
@triton.jit
def _sparse_attn_decode_kernel(
    Q_ptr, KV_ptr, Indices_ptr, Out_ptr, LSE_ptr,
    AttnSink_ptr, TopkLen_ptr,
    sm_scale,
    b, s_q, h_q, topk, total_kv_tokens,
    d_qk: tl.constexpr, d_v: tl.constexpr,
    PADDED_D_QK: tl.constexpr, PADDED_D_V: tl.constexpr,
    stride_q_b, stride_q_sq, stride_q_hq, stride_q_d,
    stride_kv_flat, stride_kv_d,
    stride_idx_b, stride_idx_sq, stride_idx_topk,
    stride_o_b, stride_o_sq, stride_o_hq, stride_o_d,
    HAS_ATTN_SINK: tl.constexpr,
    HAS_TOPK_LENGTH: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    pid_bsq = tl.program_id(0)
    pid_hq = tl.program_id(1)

    batch_idx = pid_bsq // s_q
    sq_idx = pid_bsq % s_q

    # Processes ONE head at a time (no BLOCK_H tiling)
    # Uses float32 throughout (no BF16 tensor cores)
    # Reads from pre-dequantized BF16 KV (not raw FP8)
    # Single [PADDED_D_V] accumulator (no tiling)
    # Uses tl.exp instead of tl.math.exp2
    # Grid: (b*s_q, h_q) -- tokens fast, heads slow (wrong for AMD cache locality)

    d_offsets = tl.arange(0, PADDED_D_QK)
    d_mask = d_offsets < d_qk

    q = tl.load(Q_ptr + ..., mask=d_mask, other=0.0).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([PADDED_D_V], dtype=tl.float32)

    for topk_start in range(0, topk, BLOCK_TOPK):
        # ... loads K and V separately, uses element-wise multiply for QK ...
        qk = tl.sum(q[None, :] * kv, axis=1) * sm_scale
        # ... online softmax with tl.exp ...
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

    # ... post-processing ...
```

Key differences from the optimized kernel:
- No FP8 gather/dequant (reads pre-dequantized BF16)
- Single head per CTA (no BLOCK_H dimension)
- Float32 compute (no BF16 tensor cores)
- Element-wise multiply instead of tl.dot (no MFMA utilization)
- Single 512-wide accumulator (register pressure)
- Tokens as fast-varying grid dimension (poor cache locality on AMD)
- No Q preloading (redundant loads)
- No batched KV loading (serial latency)
- No Split-K (poor SM utilization for large topk)
- Python-side gather/dequant/post-processing (extra memory traffic)

---

## Appendix: Final Dispatch Architecture

After all optimizations are applied and the 2-phase fallback is removed:

```
triton_sparse_attn_decode(q, kv_scope, extra_kv_scope, ...)
  |
  +-- d_qk == 512 (MODEL1)
  |     +-- single scope:
  |     |     +-- topk < threshold: fused gather+attn kernel
  |     |     +-- topk >= threshold: fused gather+attn + split-K + combine
  |     +-- dual scope:
  |           +-- _decide_splitk_dual_scope() -> split_k?
  |           +-- split_k > 0: fused split-K dual-scope kernel + combine
  |           +-- split_k == 0: fused no-splitk dual-scope kernel
  |
  +-- d_qk == 576 (V3.2)
        +-- fused gather + attention (+ split-K if topk large)
```

All paths use the fused kernel. The 2-phase (separate gather + attention) path has been deleted because E2E benchmarks showed it was slower at high concurrency despite appearing faster in kernel-level microbenchmarks.

**Design lesson**: Always validate kernel dispatch heuristics with end-to-end benchmarks, not just kernel-level microbenchmarks. Python-side overhead (buffer allocation, extra kernel launches, inability to fuse) is invisible to kernel profiling but dominates real serving performance.
