# Dense GEMM tuning on gfx1151 (RDNA3.5 / Strix Halo, vLLM)

> The CDNA recipes in this directory (`aiter_gemm_tuning.md`, `fp8_gemm_tuning_sglang_aiter.md`,
> `moe_int4_tuning.md`) **do not apply here**. They tune aiter's per-shape C++/CK/hipBLASLt DB via
> `AITER_TUNE_GEMM` → `gradlib` → `AITER_CONFIG_GEMM_BF16`; that whole path is CDNA-only. Two
> different levers work on gfx1151, and **both have been measured end-to-end**.

Hardware background: `perf_knowledge/hardware/rdna35_gfx1151/` (peaks, memory hierarchy, noise floor).

## What is NOT available (check first, do not plan around it)
`[measured]` On the Strix Halo serving image used for these runs:

| rung | status on gfx1151 |
|---|---|
| aiter C++/CK GEMM, `AITER_TUNE_GEMM` capture → gradlib → `AITER_CONFIG_GEMM_BF16` | **unavailable** — the CDNA aiter GEMM path |
| `ckProfiler` / CK instance sweep | **binary absent on the image** |
| `hipblaslt-bench` offline Tensile tune | **binary absent on the image** |
| FlyDSL (`flydsl_hgemm`, `flydsl_preshuffle_gemm_a8`) | CDNA gfx942/950 only |
| fp8 / MXFP4 / block-scaled GEMM | **no hardware path** — see `matrix_core_wmma.md` |
| **PyTorch TunableOp (rocBLAS + hipBLASLt per-shape race)** | **works — lever 1** |
| **aiter *Triton* ops (`aiter.ops.triton`)** | **works — lever 2** |

Note the absences are *image provisioning*, not architecture, for `ckProfiler`/`hipblaslt-bench`.
Record them as `degrade` per `preflight.md`; do not conclude "no win available".

## Lever 1 — per-shape BLAS selection (TunableOp)

**The structural finding, and it is the important part:** on this part the two vendor BLAS libraries
win *different shapes*, and the gap between routing per-shape and forcing one globally is enormous.

`[measured]` E2E, same model/workload:

| policy | e2e |
|---|---|
| **per-shape mix** (small/skinny → hipBLASLt, large → rocBLAS) | **+14.3%** (warm_server harness) |
| force hipBLASLt globally | **−8.2%** |

A 22-point spread. **Never ship a global `PYTORCH_TUNABLEOP`/BLAS override**; the win is entirely in
the per-shape table.

`[measured]` The same table re-measured on a **fresh server** gives **+10.93%**, reproducible, and is
written into `current_setting.sh` automatically. The warm_server harness overestimates by ~25–30% on
this part, which reconciles 14.3 → ~10.9. **Quote the fresh-server number**; treat warm_server as a
screening tool only.

### The gotcha that silently costs you the win
Use the **`installed/` cold-state table, not the raw hot intermediate table** that the tuning pass
leaves behind. The raw table is written while the process is warm and encodes a different winner per
shape; deploying it reproduces a fraction of the gain and looks like noise.

### Default backend is rocBLAS, not hipBLASLt
```python
torch.backends.cuda.preferred_blas_library()   # -> _BlasBackend.Cublas  == rocBLAS
```
`Cublaslt` would mean hipBLASLt. So a `torch.mm` baseline on this box characterises **rocBLAS**, and
any claim about "the vendor library" must say which one. ROCm 7.2 does ship gfx1151-tuned hipBLASLt
kernels — they are simply not what you get by default.

## Lever 2 — reroute the live vLLM seam to aiter's Triton `gemm_a16w16`

`[measured]` **+13.85% e2e** Director-verified (133.811 → 152.345 tok/s, 1.139×, non-overlapping
3-repeat same-session A/B, byte-exact serving-greedy parity).

- **Seam:** on gfx1151 the live vLLM dense bf16 GEMM path is
  `vllm...layers.utils:rocm_unquantized_gemm_impl` (`aten::linear` → rocBLAS unquantized).
- **Route:** onto `aiter.ops.triton` `gemm_a16w16` (`_gemm_a16_w16_kernel`, split-K) backed by a
  **gfx1151-tuned per-shape tile table**. Covers qkv_proj / o_proj / down_proj together (decode M=8;
  largest single head o_proj N=2560 K=4096 at +13.588% attributed).
- **Mechanism:** pure overlay — CUDA-graph-safe, byte-exact, no package edit. Replays directly from
  the e2e KB overlay artifact when the store holds a bindable overlay for the same identity.
- **This is the route CDNA gates OFF.** `use_aiter_triton_gemm()` disables it on gfx942/gfx950 vLLM.
  On gfx1151 it engages and wins. **Re-check the live dispatch per arch rather than inheriting the
  CDNA verdict** — this is the single clearest example in this KB of a CDNA prior being backwards.

### Verify engagement, not file presence
One ENGAGED banner per rerouted Linear **plus live hit counters** (this run: qkv_proj / o_proj /
down_proj = 100 / 100 / 100) and a CUDA-graph-safe line in the server log. **A tuned table that never
binds fails silently** — the hit counters are the gate.

## Sizing the opportunity: iso-GEMM × is NOT the e2e number
`[measured]` Lever 2's isolated GEMM speedup is only **~1.31×**, yet e2e moved **+13.85%**. The win is
**decode launch/dispatch**, not GEMM FLOPs. Corollary in both directions:
- Do not reject a candidate because its isolated × looks small at decode.
- Do not project e2e from an isolated × either. **Gate on e2e.**

## Where there is no headroom
`[measured]` `lm_head` is already at the **233 GB/s measured DRAM wall** — it is bandwidth-bound and
has nothing to give. In one authoring campaign all three candidate heads lost their bake-off to the
vendor path. That is a legitimate outcome on this part, not a harness failure: **the vendor path wins
3 of 4 GEMM regimes here** (see `perf_knowledge/hardware/rdna35_gfx1151/peak_tables.md`). The regime
where a generated kernel does win is **skinny/decode-shaped** — which is where lever 2 lives.

## A result that did NOT survive its own parity gate
`[measured]` A linear stride-padding change measured **+12.20%** with two zero-overlap arms — but the
accompanying "bit-identical" claim **does not hold** (2 of 8 controls diverged). It remains a research
candidate, **not shippable**, and is recorded here so nobody re-derives it and ships it. Parity is a
gate, not a formality.

## Measurement hygiene on this part
- E2E noise floor is **not** the 0.4% kernel-microbenchmark floor. Use the run's own measured
  significance threshold and require non-overlapping repeats.
- The CPU shares power and bandwidth with the GPU; pin the governor. `[measured]` a `powersave`
  governor produced 16.7% / 34.0% round-to-round spread — enough to manufacture any of the wins above.
  See `perf_knowledge/hardware/rdna35_gfx1151/clocks_power.md`.

## Sources
- `learned/dense-gemm-bf16-gfx1151-vllm.md` (lever 2, Director-validated, KB overlay replay).
- `perf_knowledge/hardware/rdna35_gfx1151/{peak_tables,memory,clocks_power,matrix_core_wmma}.md`.
- gfx1151 TunableOp E2E campaign, fresh-server verified, 2026-09.
