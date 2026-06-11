---
title: paged_kv_copy — numerics
kind: technique
operator: paged_kv_copy
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, int8]
regimes: [both]
updated: 2026-06-08
sources:
  - https://vllm.ai/blog/2026-04-22-fp8-kvcache
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/cache.py
---

# paged_kv_copy — numerics

## Plain copy = byte-exact
Non-quant `reshape_and_cache` / `copy_blocks` / `swap_blocks` move bytes exactly. Oracle
`torch.equal` after gathering the written slots. A delta = a `slot_mapping`/layout bug, not precision.

## FP8 / int8 KV quant: the real numeric surface
The quant variants (`reshape_and_cache_with_pertoken_quant`, `_with_block_quant`) introduce error from the
**scale granularity**:
- **per-token** scale (one scale per token) is more accurate than **per-tensor**; **per-head** (FA3-style
  array of scales, one per KV-head) is finer still — vLLM expanded `reshape_and_cache_flash` to take an array
  of scales for exactly this.
- gate **task accuracy** (greedy/temp=0 eval), not byte parity. FP8 KV is a known accuracy lever — validate.

## FNUZ vs OCP must match the read
gfx942 stores **FNUZ** fp8 KV, gfx950 **OCP**; vLLM uses **e4m3** on AMD (vs e5m2 on NVIDIA). The cache write
dialect must match what the attention kernel reads — a wrong-dialect read is a silent **2×** error (exponent
bias off by one), not a crash. The shuffled-layout path and `pa_fwd_asm` assume the matching dialect; confirm
the write and read agree ([[operators/kv_cache_quant/numerics.md]] if present, [[hardware/shared/dtype_numerics.md]]).

## Shuffled layout is loss-free
The KV-cache shuffle is a **layout** change (byte relocation), not a value change — byte-exact given a correct
inverse in the reader. A numeric delta after enabling `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` means the
attention reader's layout assumption is wrong, not a precision artifact.

## Verify
Greedy/temp=0 e2e parity after enabling FP8 KV or the shuffled layout; isolated `allclose` on the written
slots (exact non-quant, FP8 within atol/rtol). A small eval for any FP8 KV change.

## Sources
- FP8 KV per-head scales, e4m3 on AMD, accuracy considerations: https://vllm.ai/blog/2026-04-22-fp8-kvcache
- aiter quant cache ops: ROCm/aiter@a6bb49937:aiter/ops/cache.py.
- FNUZ/OCP 2× hazard: [[hardware/shared/dtype_numerics.md]], [[operators/kv_cache_quant/overview.md]].
