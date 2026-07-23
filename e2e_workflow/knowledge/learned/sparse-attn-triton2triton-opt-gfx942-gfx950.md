---
key: block-sparse GQA attention (sparse-fwd prefill + decode index-score) · gfx942 & gfx950 · vLLM
type: reference
confidence: ★★★
effect: MERGED upstream win; gfx950 TP8 +42% output tok/s / −50% TTFT / −30% TPOT; gfx950 c64 +6.4% tok/s / −10.5% TTFT; low-bit (w4) +10% tok/s at conc256; gsm8k unchanged
last_seen: 2026-06-25
---
# Block-sparse GQA attention → the concrete Triton→Triton rewrite that lands the win (both archs)
Kernel type: **Triton→Triton optimization (route=rewrite / mode=optimize)** — the live path is an editable
in-tree Triton block-sparse flash kernel with NO CK/aiter library equivalent to swap to, so there is no
env/flag/backend bake-off winner; the ONLY op-level lever is rewriting the existing Triton kernel. This card
records what actually won: an arch-specialized rewrite of the two hot callables, gated on constexpr
`_IS_MI3XX` / `_IS_ROCM_MI3XX` (= `on_gfx942() or on_gfx950()`). CDNA2 (gfx90a) and other archs fall through
to the dense path unchanged.

- **prefill lever — sparse-fwd kernel KV-block sub-tiling** (ROCm-only import): each 128-token KV block is
  sub-tiled by a constexpr `SUB_K` to right-size the per-block QK/PV MFMAs. `SUB_K` and the MFMA launch params
  are chosen **per-arch** (gfx950 ≠ gfx942 constants). The AMD-specific module carries ONLY the specialized
  prefill kernel and reuses the generic `common.ops` for the decode kernels, the fp8 dtype set, and the block
  size. Take-away: the win came from KV-block sub-tiling (SUB_K), NOT from packing more query tokens per program.
- **decode lever — index-score GEMV collapse**: in the decode index-score path one query token
  (`BLOCK_SIZE_HQ == 1`) is scored against `[N, D]` keys, so `tl.dot([N,D] x [D,1])` is a degenerate
  matrix-vector product. On MI3xx, replace the `tl.dot` with an equivalent **fp32 multiply + reduce**, gated on
  `_IS_ROCM_MI3XX and BLOCK_SIZE_HQ == 1`. Small enough to stay inline in `common.ops` (no separate AMD copy).
- **overlap / correctness note** (top-k index path): keep BOTH branches — the vectorized MI3xx GEMV branch AND
  `out_dtype=tl.float32` accumulation on the `tl.dot` fallback (the fp32 accum is REQUIRED for the fp8 index
  cache). Do not drop the fallback's fp32-accum when adding the vectorized path.
- **apply**: for a block-sparse / NSA-style GQA attention run on gfx942 or gfx950, Tier-C Triton
  route=rewrite — (1) sub-tile the 128-tok KV block by an arch-specific `SUB_K` + per-arch MFMA launch params
  in the prefill fwd kernel; (2) collapse the degenerate `BLOCK_SIZE_HQ==1` decode GEMV to fp32 mul+reduce.
  Prefill and decode are SEPARATE callables — tune each independently.
- **verify**: judge against the immutable oracle (bf16 rtol=atol=2e-2). Accuracy held clean: gsm8k 5-shot
  flexible 0.9143→0.9136 / strict 0.9136→0.9128 (within noise; independent runs 0.9545→0.9575 and 0.9416 flat).
  This is a REAL e2e transfer, not iso-only — the isolated head is modest (~5.6% GPU) but arch-specialized
  MFMA + GEMV collapse compounds across prefill (TTFT ~2× on TP8) and decode (TPOT −30%). Gains SCALE with
  concurrency (low-bit w4: +10% tok/s at conc256).
- **caution**: SUB_K and MFMA params are per-arch — a value tuned for gfx950 will not be optimal on gfx942
  (ship different constants). Keep the `_IS_MI3XX` gate so non-MI3xx archs stay on the dense fallback. Some
  arch-check redundancy vs `platforms.rocm` is harmless.
- source: merged upstream vLLM block-sparse attention optimization, 2026-06-25, gfx950 (MI350) + gfx942 (MI300).
  Perf: gfx950 c64 8k/1k +6.4% tok/s / −10.5% TTFT / −6.0% TPOT; TP8 8k/1k c64 +42.2% tok/s / −50.6% TTFT /
  −29.5% TPOT; TP4 c64 ~+7.75% tok/s / −13.4% TTFT.
