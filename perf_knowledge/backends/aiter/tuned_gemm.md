---
title: aiter tuned_gemm — the live dense-GEMM dispatch + per-shape tuning recipe
kind: backend
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: sota
updated: 2026-06-05
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/tuned_gemm.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:gradlib/gradlib/gemm_tuner.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:gradlib/gradlib/GemmTuner.py
---

# aiter tuned_gemm — dispatch + tuning recipe

## TL;DR
`aiter.tuned_gemm.gemm_a16w16` (exposed as `tgemm.mm`) is **the live dense bf16/fp16 GEMM path on
sglang/vLLM**. On every call it looks up a per-shape config DB and dispatches to the fastest of
`hipblaslt / asm / skinny / triton / flydsl / torch`. To make a serving GEMM faster you **tune aiter's
DB** with `gradlib` — capture real shapes, race candidates per shape, deploy the winning CSV by env. This
is the only GEMM lever that engages the serving path; `HIPBLASLT_TUNING_OVERRIDE_FILE` and PyTorch
TunableOp do **not** (aiter bypasses the torch BLAS dispatch).

## The dispatch path (`aiter/tuned_gemm.py`)

### Entry + key construction
`gemm_a16w16(A, B, bias, otype, scale_a, scale_b, scale_c)` (decorated with `@torch_compile_guard`):
- flattens `A` to 2-D `[m, k]`, `n = B.shape[0]`;
- detects `bpreshuffle` from `B.is_shuffled`;
- builds the lookup via `get_GEMM_A16W16_config(M=m, N=n, K=k, bias=use_bias, dtype=str(A.dtype),
  otype=str(otype), scaleAB=(scale_a or scale_b is not None), bpreshuffle=...)`.

The **DB index is the 9-tuple**:
`(cu_num, padded_M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle)`
(`get_GEMM_A16W16_config_` sets the index on the CSV; `cu_num` from `get_cu_num()`). Every component must
match the live call — **`bias` and `cu_num` are the usual mismatch traps.**

### padded_M bucketing
The lookup is tried for three `padded_M` variants in order `[None, 0, 1]` where each non-None value calls
`get_padded_m(M, N, K, gl)` (`aiter/ops/gemm_op_common.py`). This lets the tuner bucket large prefill M
into a bounded set of rows so the DB stays small and decode/prefill M both hit an entry. First non-None
hit wins.

### libtype dispatch (`solMap`)
```python
solMap = {"torch": torch_gemm, "hipblaslt": hipb_gemm, "skinny": skinny_gemm,
          "asm": asm_gemm, "triton": triton_gemm}
```
- `hipblaslt` → `hipb_mm(inp, weights.t(), solidx, bias, otype, scale_a, scale_b, scale_c, bpreshuffle)`
  — calls a specific Tensile **solution index** directly (creates the hipBLASLt ext on first use).
- `asm` → `gemm_a16w16_asm(inp, weights, out, bias, splitK, kernelName, bpreshuffle)` (bf16 in, fp32/bf16
  out only).
- `skinny` → `wvSpltK` (solidx 0), `LLMM1` (1), `wv_splitk_small_fp16_bf16` (2); gated by
  `is_skinny_default_shape` (M≤16 with N/K bounds, K%8==0).
- `flydsl` → handled before `solMap`: re-resolves kernel params and calls `flydsl_hgemm` (see
  [flydsl_path.md](flydsl_path.md)). If FlyDSL isn't importable the config is dropped and lookup falls
  through.
- `triton` → `aiter.ops.triton.gemm.basic.gemm_a16w16` (no scaling, no bpreshuffle).

### No-hit defaults (`get_GEMM_A16W16_config`)
When nothing matches: `gfx12*` → `torch`; `bpreshuffle` on gfx942 → `hipblaslt solidx=-1` (let heuristic
pick), on gfx950 bf16 with N%64==K%64==0 → `asm`; else if `is_skinny_default_shape` → `skinny solidx=2`;
otherwise `torch`. A `logger.info(... not found tuned config ... will use default ...)` line is emitted.

## Capture → tune → deploy

### 1. Capture (live shapes, correct bias)
Launch a warm server with `AITER_TUNE_GEMM=1`. `save_shapes()` appends each distinct
`(M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle)` (deduped) to
`aiter/configs/bf16_untuned_gemm.csv`. **`bias` is recorded as `bias is not None`** — capturing live is
the only way to get the real bias flag right.

### 2. Tune (`gradlib`)
```bash
# all GPUs visible; writes incrementally to the tuned CSV
python3 gradlib/gradlib/gemm_tuner.py --indtype bf16 --mp <ngpus> \
        --input_file aiter/configs/bf16_untuned_gemm.csv \
        --tuned_file my_bf16_tuned.csv
```
The tuning ladder (`gradlib/gradlib/GemmTuner.py`):
1. `--libtype` selects which families to race (default `all` = `hipblaslt,asm,triton,flydsl,torch,skinny`;
   validated list in `libtype_list`).
2. **hipBLASLt**: `find_hipblas_sols` enumerates the solution pool, a fast pass times all, then top-N
   (`topn=20`) are re-timed accurately.
3. **asm**: reads the asm kernel list (`bf16gemm/bf16gemm_fp32bf16.csv`), each tile is a candidate.
4. **flydsl/skinny/triton/torch**: each contributes its candidate set (flydsl only if a catalog kernel
   exists for the shape; skinny only if `is_skinny_default_shape`).
5. All candidates run on a multi-process pool across `--mp` GPUs; each is timed and checked vs a torch
   reference with `checkAllclose` → `err_ratio`. Candidates above `--errRatio` (default **0.05**) are
   filtered (`df[df["err_ratio"] < args.errRatio]`). `--shape_grouped` pins all candidates for one shape
   to one GPU to remove cross-GPU timing variance.
6. The fastest passing candidate per key is written as a row with full provenance columns
   (`libtype, solidx, splitK, us, kernelName, err_ratio, tflops, bw`).
- `gemm_tuner.py` runs the tuner in a **spawned subprocess with retry** (up to 30) so a single
  GPU-fault SIGABRT/SIGSEGV is recovered without losing already-written rows.
- `--model_dir <hf_model>` + `--tp` auto-derives the four canonical Linear shapes (qkv, o-proj, gate+up,
  down) × an N-set `[1,512,1024,2048,3072,4096,8192,16384]×batch` if you have no captured file.

### 3. Deploy (no package edit)
```bash
export AITER_CONFIG_GEMM_BF16=/abs/path/my_bf16_tuned.csv   # ':'-mergeable with the shipped CSV
export AITER_LOG_TUNED_CONFIG=1                              # log every DB hit
```
`AITER_CONFIG_GEMM_BF16` is read by `AITER_CONFIG.AITER_CONFIG_GEMM_BF16_FILE`; a `:`-joined list is merged
(`update_config_files`) so your tuned rows overlay the shipped `bf16_tuned_gemm.csv` without editing
site-packages. See [configs_db.md](configs_db.md) for schema + merge semantics.

## Config space / knobs
- **DB key**: `(cu_num, padded_M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle)` — match all 9.
- **Tuner**: `--libtype`, `--errRatio` (0.05), `--mp`, `--warmup`/`--iters`, `--shape_grouped`,
  `--all_bias` (tune both bias/non-bias regardless of capture), `--all` (retune existing rows).
- **Per-libtype**: hipBLASLt `solidx` + `HIPBLASLT_TUNING_USER_MAX_WORKSPACE` (workspace cap); asm
  `kernelName` + `splitK`; flydsl tiling (see [flydsl_path.md](flydsl_path.md)).

## Numerics / parity
Same-math bf16 algorithm/solution swap → parity-safe. The tuner gates every candidate on
`err_ratio < errRatio` (default 0.05; bf16 atol/rtol 5e-2, else 1e-2) vs a torch reference, so a numerically
divergent solution is never written.

## Pitfalls
- ⚠ **bias mismatch = 0 engagement**: tuning synthesized `bias=true` shapes while live calls are
  `bias=false` → every lookup misses (verified failure, 2026-06-07). Always capture live with
  `AITER_TUNE_GEMM=1`.
- ⚠ `cu_num` is in the key — a DB tuned on a different CU count (different GPU/partition) won't hit.
- TunableOp / `HIPBLASLT_TUNING_OVERRIDE_FILE` hook the PyTorch BLAS dispatch, which aiter bypasses → 0
  engagement on serving.
- Tuning large prefill shapes is slow (racing ~hundreds–thousands of hipBLASLt sols/shape) and can
  fork-storm the host (`rocm_agent_enumerator`) — bucket-reduce big M (`get_padded_m`) and use
  `--shape_grouped`.

## How to verify
1. `grep -c 'is tuned on cu_num' server.log` (with `AITER_LOG_TUNED_CONFIG=1`) > 0.
2. Same-session 2-launch A/B: ref = current, cand = `+AITER_CONFIG_GEMM_BF16=...`; accept iff
   `delta > 0.5% AND cand_min > ref_max` AND parity holds (measured +2.23%, 246 hits — see overview).

## Alternatives / cross-links
SOTA card: [`operators/dense_gemm/backends/aiter.md`](../../operators/dense_gemm/backends/aiter.md).
Executed kernels: [`operators/dense_gemm/backends/hipblaslt.md`](../../operators/dense_gemm/backends/hipblaslt.md)
· [flydsl_path.md](flydsl_path.md) · [configs_db.md](configs_db.md) · [integration.md](integration.md).

## Sources
- On-box: `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0`:
  `aiter/tuned_gemm.py` (dispatch, key, save_shapes, solMap),
  `gradlib/gradlib/gemm_tuner.py` (driver, retry, model_dir),
  `gradlib/gradlib/GemmTuner.py` (ladder, libtype_list, err_ratio gate),
  `aiter/utility/base_tuner.py` (`--mp`, `--errRatio`, `--shape_grouped` args),
  `aiter/jit/core.py` (`AITER_CONFIG_GEMM_BF16`, merge).
- +2.23% e2e / 246 hits: perf_knowledge run `e2e_Qwen-Qwen3.5-27B_20260607_193315`, 2026-06-08.
