---
name: flash-mla-unit-test
description: >
  Unit test infrastructure for Flash MLA sparse attention Triton kernel (FP8 in-kernel dequant).
  Provides test data generation (FP8 KV cache + indices), PyTorch reference for golden output,
  and correctness/perf test driver. Use when building the kernel task dir for MLA optimization.
---

# Flash MLA Unit Test Infrastructure

Test harness for the fused gather+dequant+attention Triton kernel on DeepSeek V4 (MODEL1, d_qk=512).
The kernel reads FP8 KV cache directly and dequantizes in-kernel — matching the production path.

Copy `scripts/` to the kernel task dir as `kernel_src/`. The workflow's kernel_extractor or
engineer writes the actual kernel implementation in `triton_flash_mla_decode.py` (the stub
functions raise `NotImplementedError` until implemented).

## Directory Layout

```
flash_mla_tilelang_to_triton/
  skill.md                          # skill entry point
  docs/
    unit-test.md                    # this file
    optimize.md                     # optimization roadmap + full implementation guide
  scripts/                          # copy to kernel task dir as kernel_src/
    test_triton_decode.py           # test driver (correctness + perf)
    lib.py                          # TestParam, KVScope, data generation (FP8 quant/dequant)
    ref.py                          # PyTorch reference (golden output from dequanted bf16)
    quant.py                        # FP8 quantization/dequantization (MODEL1_FP8Sparse)
    kernelkit/                      # test utilities (bench, compare, precision)
```

---

## Critical: reference_io.pt Data Requirements

When `kernel_extractor` generates `reference_io.pt` for a task dir, it **MUST** include the
following data per case. Missing any of these blocks one or more optimization phases.

### Per-scope data (main + extra when dual-scope)

| Field | Dtype | Purpose | Required |
|-------|-------|---------|----------|
| `blocked_k_quantized` | FP8/uint8 (raw bytes) | **FP8 KV cache — kernel reads this directly** | **YES — this is the input** |
| `blocked_k` | BF16 `[num_blocks, block_size, 1, d_qk]` | Dequanted KV for ref.py golden output | YES (for golden comparison) |
| `indices_in_kvcache` | int32 `[b, s_q, topk]` | Physical KV cache indices | YES |
| `block_size` | int | Tokens per KV cache block | YES (for in-kernel address computation) |
| `topk_length` | int32 `[b]` or None | Valid topk count per batch | YES (early exit, masking) |
| `block_table` | int32 `[b, max_blocks_per_seq]` | Logical-to-physical block mapping | Available via KVScope |
| `abs_indices` | int32 `[b, s_q, topk]` | Logical token indices | Available via KVScope |
| `cache_seqlens` | int32 `[b]` | Per-batch sequence lengths | Available via KVScope |

### Top-level case data

| Field | Dtype | Purpose |
|-------|-------|---------|
| `q` | BF16 `[b, s_q, h_q, d_qk]` | Query tensor |
| `sm_scale` | float | Softmax scale factor |
| `attn_sink` | float32 `[h_q]` or None | Attention sink values (may contain +inf/-inf) |
| `has_extra` | bool | Whether dual-scope |
| `d_qk`, `d_v`, `b`, `h_q`, `topk` | int | Shape params for dispatch |
| `golden_out` | BF16 `[b, s_q, h_q, d_v]` | Reference output |
| `golden_lse` | float32 `[b, h_q, s_q]` | Reference LSE |

### FP8 KV Cache Layout (MODEL1, d_qk=512)

The `blocked_k_quantized` tensor must match the actual sglang FP8 layout. For MODEL1:

```
Per-token data (576 bytes):
  Bytes 0-447:   FP8 nope data (7 tiles x 64 elements, E4M3 format)
  Bytes 448-575: RoPE data (64 bf16 values stored as 128 raw bytes, lo|hi byte pairs)

Per-token scales (8 bytes):
  7 x E8M0 uint8 scales + 1 padding byte
  Located at offset: block_size * 576 bytes from block start

Dequant formula: bf16_value = fp8_value.to(bf16) * exp2(uint8_scale - 127.0)
RoPE reconstruction: uint16 = lo_byte | (hi_byte << 8); bf16 = bitcast(uint16)
```

For V3.2 (d_qk=576): different layout (4 tiles x 128, float32 inline scales). See `quant.py`.

### Data Generation Pipeline

Use `lib.py` `generate_testcase_for_decode()` which does:
1. Generate random Q, KV data (clamped to [-1, 1])
2. Generate random block_table (shuffled block indices)
3. Generate random sparse indices via `_randperm_batch()`
4. **FP8 quant → dequant** via `kv_scope.quant_and_dequant_()`:
   - Quantize to FP8 → stored in `blocked_k_quantized`
   - Dequantize back → overwrites `blocked_k` with BF16
   - Both are now available: FP8 original + BF16 roundtripped
5. **NaN mask unused KV** — tokens not referenced by any index get `blocked_k[...] = NaN`
   (tests kernel's robustness to NaN in unreferenced memory)
6. **Non-contiguousify** all tensors via `kk.non_contiguousify()` (tests stride handling)

---

## geak_unittest.py Contract for Optimization Workflows

When `kernel_extractor` writes `geak_unittest.py` for a task dir, it MUST follow these rules:

### Rule 1: Call the FP8 entry point, pass KVScope objects

```python
# RIGHT — triton_sparse_attn_decode reads kv_scope.blocked_k_quantized (FP8)
from triton_flash_mla_decode import triton_sparse_attn_decode
out, lse = triton_sparse_attn_decode(q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink)

# or via the test harness wrapper:
from triton_flash_mla_decode import run_triton_decode
out, lse = run_triton_decode(p, t)
```

### Rule 2: Do NOT pre-concatenate or pre-dequant dual-scope data

```python
# WRONG — dequants FP8 to bf16 in Python, then passes bf16 to kernel
blocked_k = dequantize_k_cache(kv_scope.blocked_k_quantized, layout)
out, lse = kernel(q, blocked_k, ...)

# WRONG — torch.cat combines scopes, kernel never sees separate FP8 caches
combined_kv = torch.cat([main_kv, extra_kv], dim=0)

# RIGHT — pass KVScope objects, kernel reads FP8 directly and handles dual-scope
out, lse = triton_sparse_attn_decode(q, kv_scope, extra_kv_scope, ...)
```

### Rule 3: Use lib.py for dynamic test data — no static reference_io.pt needed

```python
# lib.generate_testcase_for_decode(p) does everything:
#   1. Generate random Q, KV data
#   2. Quantize to FP8 → kv_scope.blocked_k_quantized
#   3. Dequantize back → kv_scope.blocked_k (bf16, for ref golden output)
#   4. Both available: FP8 for the kernel, bf16 for ref.ref_sparse_attn_decode()
t = lib.generate_testcase_for_decode(p)
# t.kv_scope.blocked_k_quantized — FP8 raw bytes (kernel reads this)
# t.kv_scope.blocked_k — bf16 dequanted (ref.py uses this for golden output)
# t.kv_scope.block_size — tokens per KV cache block
# t.kv_scope.indices_in_kvcache — physical indices
```

### Rule 4: Include a self-contained reference function

The unittest imports `ref.py` which computes golden outputs from the dequanted bf16 KV.
The kernel is tested against this golden output with FP8-appropriate tolerance.

### Rule 5: Test shapes = REAL DS_v4 serving shapes

The test cases use **actual production shapes**, not a cartesian sweep of artificial values.
DS_v4 real decode serving: h_q=128, h_kv=1, d_qk=d_v=512, d_rope=64, s_q=1, FP8 MODEL1.
**Always dual-scope**: main (SWA) topk=128 block_size=128 + extra (c4 sparse) topk=1024
block_size=256. Batch sizes: b=2,32,64,128,256 (matching real concurrency c2..c256).

| Dimension | Production value | Test coverage |
|-----------|-----------------|---------------|
| **d_qk** | 512 | 512 |
| **h_q** | 128 | 128 (production only) |
| **main topk** | 128 (SWA_WINDOW) | 128 |
| **main block_size** | 128 (swa_page_size) | 128 |
| **extra topk** | 1024 (index_topk) | 1024 |
| **extra block_size** | 256 (page_size) | 256 |
| **batch size** | 2, 32, 64, 128, 256 | 2, 4, 32, 64, 128, 256 |
| **s_q** | 1 (decode) | 1, 2 (robustness) |
| **dual-scope** | always | yes + single-scope for robustness |
| **attn_sink** | always present | True + False (corner) |
| **topk_length** | clamped | True, False |

Total: **35 cases** (10 correctness + 10 corner + 15 perf). Lean and production-focused.

Corner cases (all must PASS):
- **all-invalid indices**: all indices set to invalid values
- **zero seqlens**: some batch entries have seqlen=0
- **lonely queries**: no valid KV tokens → output=0, LSE=+inf
- **NaN in KV**: unreferenced KV positions contain NaN (kernel must not read them)
- **non-contiguous inputs**: all inputs may be non-contiguous (stride-based access)

### Rule 6: Tolerances (FP8 in-kernel dequant)

Golden outputs come from ref.py using dequanted bf16. FP8 in-kernel dequant adds
quantization error, so tolerances are looser than exact bf16 comparison:

| | out abs_tol | out rel_tol | cos_diff_tol | lse abs_tol | lse rel_tol |
|------|------------|------------|-------------|------------|------------|
| FP8 (in-kernel dequant) | 2e-2 | 2e-2 | 1e-4 | 1e-2 | 1e-2 |
| FP8 + Split-K combine | 2e-2 | 2e-2 | 5e-6 | 1e-4 | 1e-4 |

### Rule 7: Output format for workflow compatibility

The unittest must print machine-readable results in this exact format:

```
Case N/M: <case_id>
  baseline_ms=<float>  optimized_ms=<float>  speedup=<float>x

UNITTEST_RESULT correctness=N/M geomean_speedup=X.XXXX
```

Exit code 0 = all pass, 1 = any failure.

---

## File Descriptions

### `test_triton_decode.py` — Test Driver

```bash
python test_triton_decode.py --quick              # Quick correctness (12 cases)
python test_triton_decode.py --benchmark-only      # Perf only, skip correctness
```

FP8-only — no `--mode` flag. Matches the production sglang path where
`SGLANG_HACK_FLASHMLA_BACKEND=triton` receives raw FP8 `k_cache`.
Golden outputs from ref.py (dequanted bf16). Tolerance: `atol=2e-2, rtol=2e-2`.

### `lib.py` — Test Data Generation

`generate_testcase_for_decode(p)` creates:
- `kv_scope.blocked_k_quantized` — FP8 raw bytes (kernel reads this)
- `kv_scope.blocked_k` — bf16 dequanted (ref.py golden output)
- `kv_scope.block_size` — tokens per KV cache block
- `kv_scope.indices_in_kvcache` — physical indices (int32)
- `kv_scope.topk_length` — valid topk per batch (optional)

### `ref.py` — PyTorch Reference

`ref_sparse_attn_decode(p, t)` computes golden output from dequanted bf16 KV.

### `quant.py` — FP8 Quantization

`FP8KVCacheLayout.MODEL1_FP8Sparse` — MODEL1 layout (d_qk=512):
- Per-token: 7×64 FP8 nope + 128 RoPE bytes = 576 bytes
- Per-token scales: 7× E8M0 uint8 + 1 padding = 8 bytes

### `kernelkit/` — Test Utilities

`check_is_allclose()`, `bench_by_cuda_events()`, `non_contiguousify()`, etc.

---

## Key Constraints

- Use `lib.py` for data generation — do NOT reimplement FP8 cache logic
- Use `ref.py` for golden outputs — do NOT write a simplified reference
- Use `kernelkit` for correctness checks and benchmarking
- NaN in unreferenced KV positions must be preserved in test data
- All inputs tested as non-contiguous (via `kk.non_contiguousify()`)

---

## Common Mistakes That Block Optimization

1. **Adding a Python-side dequant before the kernel** — The whole point of this kernel
   is FP8 in-kernel dequant. Any Python-side `dequantize_k_cache()` call negates the
   performance benefit (adds a separate GPU kernel + 3x HBM traffic).

2. **Pre-concatenating dual-scope KV with torch.cat** — Must pass separate KVScope
   objects. The kernel handles dual-scope internally with shared online softmax state.

3. **Reading `kv_scope.blocked_k` (bf16) instead of `kv_scope.blocked_k_quantized` (FP8)**
   — The kernel must read FP8 directly. `blocked_k` exists only for ref.py golden output.

4. **Max topk=2048** — Must include topk=16384 for Split-K and chunked attention testing.

5. **No corner cases** — Must include all-invalid indices, zero seqlens, attn_sink
   with +inf/-inf, lonely queries. Missing these causes silent correctness bugs.

6. **No NaN in KV data** — lib.py masks unreferenced KV as NaN to catch accidental reads.
   A kernel that reads garbage from unreferenced positions may pass tests but corrupt
   outputs in production.

7. **Using bf16 tolerance for FP8** — FP8 has quantization error. Use atol=2e-2, rtol=2e-2.
   Using tighter tolerance causes false failures.
