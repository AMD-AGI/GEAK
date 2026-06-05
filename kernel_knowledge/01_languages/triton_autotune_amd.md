# Triton Autotuning on AMD — MI300X / CDNA3 / gfx942 Deep Dive

> Scope: the **full set of AMD-specific Triton tuning knobs**, what each does, valid ranges, when it
> matters, a complete `@triton.autotune` config space, how to bake the winning config, and the env
> vars for inspecting/controlling autotuning. Companion to `triton_amd.md`. All knob names verified
> against `third_party/amd/backend/compiler.py` `HIPOptions` (upstream `triton-lang/triton`).

---

## 0. The knob landscape (one table)

Triton config knobs split into three groups on AMD. **Standard** knobs exist on all backends;
**AMD-only** knobs are passed inside the `triton.Config({...})` kwargs dict (they map to `HIPOptions`
fields); **env/global** knobs are process-wide.

| Knob | Group | Type / range | Default | When it matters most |
|---|---|---|---|---|
| `BLOCK_M`,`BLOCK_N`,`BLOCK_K` | constexpr | powers of 2, 16…256 | — | every GEMM/attn — primary lever |
| `GROUP_SIZE_M` | constexpr | 1,4,**8**,16 | — | GEMM L2 reuse (use ×XCD=8) |
| `SPLIT_K` | constexpr | 1,2,4,8,16 | 1 | skinny/decode GEMM (small M·N, large K) |
| `num_warps` | standard | 1,2,**4**,8 | 4 | occupancy vs VGPR spill (wave64!) |
| `num_stages` | standard | **1**,2,(3) | 2 | loop pipelining depth |
| `matrix_instr_nonkdim` | **AMD** | 0,**16**,32 | 0 (auto) | MFMA tile size — GEMM/attn |
| `kpack` | **AMD** | 1,**2** | 1 | LDS read width / K-packing |
| `waves_per_eu` | **AMD** | 0–8 | 0 | force occupancy by trimming VGPRs |
| `schedule_hint` | **AMD** | `none`/`attention`/`memory-bound-attention` | `none` | attention sched pipeline |
| `instruction_sched_variant` | **AMD (legacy alias)** | see §7 | `none` | older ROCm/triton forks |
| `OPTIMIZE_EPILOGUE` | env | 0/1 | 0 | drop epilogue convert_layout (GEMM → 1) |
| `waves_per_eu` (`maxnreg`) | standard | int | None | hard VGPR cap (rarely needed) |

> **Critical:** AMD knobs only take effect when placed in the **Config kwargs dict**, e.g.
> `triton.Config({"BLOCK_M":128, ..., "matrix_instr_nonkdim":16, "kpack":2, "waves_per_eu":2}, num_warps=4, num_stages=2)`.
> Setting them as Python variables does nothing.

---

## 1. `matrix_instr_nonkdim` — MFMA instruction size

Selects the non-K dimension of the MFMA matrix instruction the AMD backend emits for `tl.dot`.

| Value | MFMA shape (fp16/bf16) | Notes |
|---|---|---|
| `0` | backend auto-picks | default; usually fine but not always optimal |
| `16` | `v_mfma_f32_16x16x16` | **recommended for GEMM on MI300X** |
| `32` | `v_mfma_f32_32x32x8` | larger output tile per instr, coarser scheduling |

- **AMD guidance:** for GEMM, `mfma_16x16` typically beats `mfma_32x32` even at large tiles —
  better power efficiency and finer-grained scheduling overlap (more chances to hide latency).
- `mfma_32x32` produces a bigger per-wave accumulator → **more AGPR/VGPR pressure** → can force
  occupancy down or spill. Pick 16 unless 32 measurably wins for your shape.
- For fp8/int8 the K-dim of the instruction is larger (32 for fp8 nonkdim=16); pair with
  `BLOCK_K ≥ 64`.
- Interacts with `BLOCK_M/BLOCK_N`: tiles must be divisible by the MFMA M/N. `nonkdim=32` requires
  `BLOCK_M,BLOCK_N` divisible by 32.

---

## 2. `waves_per_eu` — occupancy hint via register trimming

`waves_per_eu=n` asks the LLVM AMDGPU backend to **reduce VGPR usage** so that `n` wavefronts can be
resident per Execution Unit (SIMD). It emits the `amdgpu-waves-per-eu` function attribute.

Hardware: **512 VGPRs/EU**, allocated in blocks of **16**.

```
round_up_16(vgpr_used) * waves_per_eu  must be ≤ 512  to be achievable
```

| vgpr_used | rounds to | max waves/EU |
|---|---|---|
| ≤ 64 | 64 | 8 |
| 128 | 128 | 4 |
| 170 | 176 | 2  (176×3 = 528 > 512) |
| 256 | 256 | 2 |

- Use when you are **just above an occupancy boundary** (e.g. VGPR=176 → set `waves_per_eu=3` and
  LLVM may shave it under 170 to fit 3 waves). Going too aggressive forces **spills** — counterproductive.
- `0` = no hint (compiler decides). Typical tuned values: **2 or 3** for GEMM, **3–4** for
  memory-bound elementwise/norm kernels.
- Verify: `AMDGCN_ENABLE_DUMP=1 ... | grep .vgpr_count`; compute occupancy with `occ.sh`.

---

## 3. `kpack` — K-dimension packing for LDS reads

`kpack` controls how many K-slices of the operand are packed together when reading from LDS, which
governs the **LDS access width**.

| Value | Effect | When |
|---|---|---|
| `1` | one K-block per LDS read | default; small `BLOCK_K` |
| `2` | pack 2 → wider `ds_read_b128` | **fp16/bf16 GEMM, BLOCK_K≥64** — usually a win |

- `kpack=2` lets the backend emit **128-bit `ds_read_b128`** (vs two `b64`), halving LDS instruction
  count and improving LDS bandwidth utilization. Confirm `ds_read_b128` appears in the ISA.
- Costs extra VGPRs (holds 2 slices) → may reduce occupancy; tune jointly with `waves_per_eu`.
- **CDNA4 (gfx950):** `kpack` is **deprecated** — the backend warns and forces it to `1`. Only set
  `kpack=2` for gfx942.

---

## 4. `num_warps` — wave64 reality and spill avoidance

On AMD a warp = **64 lanes**. `num_warps=N` → `N×64` threads/block.

| num_warps | threads | waves on 1 SIMD (if 2 wg/CU) | VGPR/wave budget |
|---|---|---|---|
| 4 | 256 | 1–2 | up to ~512/256 |
| 8 | 512 | 2 | ~256 each |

- **The #1 AMD perf bug:** carrying `num_warps=8` from an NVIDIA config. 8 warps → 2 waves share a
  SIMD → each gets only ~256 VGPRs → spill to scratch (HBM) → **3–5× slowdown**. Cutting warps to
  eliminate spills is "the lowest-hanging fruit."
- Start GEMM at `num_warps=4`; only go to 8 if the kernel is VGPR-light and occupancy-bound.
- Memory-bound kernels (softmax, norm) often prefer `2` or `4`.

---

## 5. `num_stages` — software pipelining depth (AMD semantics differ)

On AMD, `num_stages` drives the **stream pipeliner** (`add_schedule_loops` / `add_pipeline`), not
NVIDIA's `cp.async` mbarrier pipeline. Defaults to **2** (HIPOptions). AMD-specific recommendations:

| Kernel pattern | `num_stages` |
|---|---|
| single GEMM | **2** |
| two fused GEMMs (Flash-Attention) | **1** |
| GEMM + non-GEMM epilogue op | 2 |
| no-GEMM (elementwise / reduction) | 1 |

- Higher stages = more in-flight global loads buffered in LDS → **more LDS + VGPRs** → can crush
  occupancy on the 64 KB-LDS MI300X. `num_stages=3+` usually *hurts* a single GEMM on AMD.
- `num_stages>1` is required to enable **block ping-pong** scheduling (`knobs.amd.use_block_pingpong`).

---

## 6. `GROUP_SIZE_M` and `SPLIT_K` — grid shaping for MI300X

### `GROUP_SIZE_M` (L2 block swizzle)
Reorders block scheduling so neighboring blocks reuse the same B columns in L2.
- Use **multiples of the XCD count (8)**: `8`, `16`. `GROUP_SIZE_M=8` is a strong default.
- Bigger groups → more L2 reuse but worse load balance for small grids.

### `SPLIT_K` (K-dimension parallelism)
Splits the K reduction across multiple programs that atomically/`tl.atomic_add` accumulate.
- **Use for skinny GEMM / decode**: small M·N (few output tiles) but large K leaves CUs idle.
  `SPLIT_K=4/8/16` spreads K across more of the 304 CUs → hits the **≥1024 program** target.
- Costs an atomic accumulate or a separate reduction kernel + zero-init of C.
- Not needed when M·N already yields ≥1024 tiles.

```python
# decode-shaped GEMM: M=64, N=4096, K=14336 -> only ~few tiles without split-k
configs = [triton.Config(
    {"BLOCK_M":64,"BLOCK_N":128,"BLOCK_K":64,"GROUP_SIZE_M":8,"SPLIT_K":sk,
     "matrix_instr_nonkdim":16,"kpack":2,"waves_per_eu":3},
    num_warps=4, num_stages=2) for sk in (1,2,4,8,16)]
```

---

## 7. `schedule_hint` / `instruction_sched_variant` — instruction scheduling

The AMD backend exposes a scheduling hint to control how MFMA / VMEM / DS / valu instructions are
interleaved in the loop (built on LLVM's `sched_group_barrier` / IGLP machinery).

**Current upstream field: `schedule_hint`** (HIPOptions, default `'none'`):

| Value | Meaning |
|---|---|
| `none` | preserve default AMDGPU backend scheduling (safe baseline) |
| `attention` | scheduling pipeline tuned for FA-style chained dots |
| `memory-bound-attention` | variant for memory-bound attention (e.g. decode) |

- Multiple values may be comma-separated. **Experimental** — the field name and accepted values have
  changed across ROCm/triton releases (older forks used `instruction_sched_variant` with values like
  `default` / `iglp0` / `iglp1`). Always `grep schedule_hint third_party/amd/backend/compiler.py` for
  the version you run.
- For raw control, `llvm_fn_attrs` can pass LLVM sched strategies, e.g.
  `llvm_fn_attrs="amdgpu-sched-strategy=iterative-ilp"`.
- Leave at `none` unless you are tuning attention; benefits are kernel-specific and small for GEMM.

---

## 8. AMD env / global knobs (process-wide)

| Variable / `knobs.amd.*` | Effect | Recommendation |
|---|---|---|
| `OPTIMIZE_EPILOGUE=1` | remove epilogue `convert_layout` (LDS round-trip) | **ON for GEMM** |
| `TRITON_PRINT_AUTOTUNING=1` | print winning config + timing | ON while tuning |
| `AMDGCN_ENABLE_DUMP=1` / `knobs.amd.dump_amdgcn` | dump final ISA | inspect `*_dwordx4`, `ds_*_b128` |
| `MLIR_ENABLE_DUMP=1` | dump TTGIR / TritonAMDGPU IR | check MFMA layout, LDS bytes |
| `knobs.amd.use_buffer_ops` | use `buffer_load/store` (bounds-checked, faster OOB) | ON for masked loads |
| `knobs.amd.use_async_copy` | async global→LDS copy (default gfx950) | gfx950; experimental gfx942 |
| `knobs.amd.use_block_pingpong` | ping-pong two warp-groups (needs num_stages>1) | try for GEMM |
| `knobs.amd.use_in_thread_transpose` | in-thread transpose for some layouts | layout-dependent |
| `TRITON_ALWAYS_COMPILE=1` | bypass kernel cache | force re-tune |

`supported_fp8_dtypes` on AMD = `("fp8e4nv","fp8e5","fp8e5b16","fp8e4b8")`. The **fnuz** types for
MFMA are `fp8e4b8` (E4M3 fnuz) and `fp8e5b16` (E5M2 fnuz) — see `triton_amd.md` §6.

---

## 9. Complete `@triton.autotune` config space for MI300X GEMM

A production-grade, AMD-aware config space + a pruning hook to skip illegal/oversized tiles.

```python
import triton, triton.language as tl

def _mi300x_gemm_space():
    space = []
    block_mn = [(128,128),(128,256),(256,128),(256,256),(128,64),(64,128)]
    block_k  = [32, 64, 128]
    for (BM, BN) in block_mn:
        for BK in block_k:
            for nonkdim in (16, 32):                # MFMA size
                if nonkdim == 32 and (BM % 32 or BN % 32):
                    continue                        # tile must divide MFMA M/N
                for kpack in (1, 2):                # gfx942 only; 1 on gfx950
                    for nw in (4, 8):               # wave64: 4=>256 threads
                        for we in (0, 2, 3):        # waves_per_eu hint
                            space.append(triton.Config(
                                {"BLOCK_M":BM, "BLOCK_N":BN, "BLOCK_K":BK,
                                 "GROUP_SIZE_M":8, "SPLIT_K":1,
                                 "matrix_instr_nonkdim":nonkdim,
                                 "kpack":kpack, "waves_per_eu":we},
                                num_warps=nw, num_stages=2))
    return space

def _prune(configs, named_args, **kw):
    M, N, K = named_args["M"], named_args["N"], named_args["K"]
    out = []
    for c in configs:
        k = c.kwargs
        # LDS budget guard: (BM*BK + BK*BN)*2B*num_stages must fit 64KB
        lds = (k["BLOCK_M"]*k["BLOCK_K"] + k["BLOCK_K"]*k["BLOCK_N"]) * 2 * c.num_stages
        if lds > 64*1024:
            continue
        # skip configs whose tiles vastly exceed the problem (wasted work)
        if k["BLOCK_M"] > 2*M or k["BLOCK_N"] > 2*N:
            continue
        out.append(c)
    return out or configs[:1]

@triton.autotune(configs=_mi300x_gemm_space(),
                 key=["M","N","K"],
                 prune_configs_by={"early_config_prune": _prune},
                 warmup=25, rep=100)        # more reps -> less noisy timing
@triton.jit
def gemm_kernel(...):
    ...   # body as in triton_amd.md §8
```

Run with `TRITON_PRINT_AUTOTUNING=1` to see the winner, e.g.:
```
Triton autotuning for function gemm_kernel finished after 4.21s;
best config selected: BLOCK_M:128, BLOCK_N:256, BLOCK_K:64, GROUP_SIZE_M:8, SPLIT_K:1,
matrix_instr_nonkdim:16, kpack:2, waves_per_eu:2, num_warps:4, num_stages:2;
```

---

## 10. Baking the winning config (drop autotune overhead in production)

Once tuned, **freeze** the config so production never re-searches (autotune adds first-call latency
and is non-deterministic across runs).

**Option A — single hard-coded config (still using `@triton.autotune` with one entry):**
```python
WINNER = triton.Config(
    {"BLOCK_M":128,"BLOCK_N":256,"BLOCK_K":64,"GROUP_SIZE_M":8,"SPLIT_K":1,
     "matrix_instr_nonkdim":16,"kpack":2,"waves_per_eu":2},
    num_warps=4, num_stages=2)

@triton.autotune(configs=[WINNER], key=["M","N","K"])
@triton.jit
def gemm_kernel(...): ...
```

**Option B — shape→config dispatch table** (what vLLM/SGLang ship: per-shape JSON of winners):
```python
import json, functools
TUNED = json.load(open("MI300X_gemm_configs.json"))   # {"4096,4096,4096": {...}, ...}

@functools.lru_cache
def pick(M, N, K):
    key = f"{M},{N},{K}"
    c = TUNED.get(key, TUNED["default"])
    return triton.Config({k:c[k] for k in (
        "BLOCK_M","BLOCK_N","BLOCK_K","GROUP_SIZE_M","SPLIT_K",
        "matrix_instr_nonkdim","kpack","waves_per_eu")},
        num_warps=c["num_warps"], num_stages=c["num_stages"])
```
SGLang/vLLM keep these as committed per-GPU JSON files (e.g. `E=…,N=…,device_name=MI300X.json` for
fused MoE), generated by a `tuning_*.py` sweep and loaded at startup — no runtime autotune.

**Option C — `triton.compile` AOT** for the exact frozen specialization (advanced; ships an HSACO).

---

## 11. Tuning workflow (recipe)

1. `OPTIMIZE_EPILOGUE=1` for GEMM; ensure grid ≥ 1024 programs (else add `SPLIT_K`).
2. Autotune the §9 space with `TRITON_PRINT_AUTOTUNING=1`, `rep≥100`.
3. Take the winner; `AMDGCN_ENABLE_DUMP=1` → confirm `global_load_dwordx4` + `ds_read/write_b128`
   in the inner loop and **no** scratch spill (`scratch_size: 0`, no `buffer_store ... scratch`).
4. If VGPR just over a boundary (`grep .vgpr_count`), nudge `waves_per_eu` up by 1.
5. Check `matrix_instr_nonkdim=16` vs `32` — keep whichever wins; 16 usually.
6. Pad leading dims if multiple of 512 B (Tagram hotspot, TN GEMM).
7. Freeze (§10) into a per-shape table; never autotune in the serving hot path.

---

## Sources

1. AMD Instinct MI300X workload optimization — Triton tuning (matrix_instr_nonkdim, waves_per_eu, num_stages, occupancy, split-K, ≥1024 grid, Tagram): <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html>
2. Optimizing Triton kernels (OPTIMIZE_EPILOGUE, AMDGCN_ENABLE_DUMP, ds_read_b128, global_load_dwordx4) — ROCm Documentation: <https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html>
3. Triton AMD backend `HIPOptions` (matrix_instr_nonkdim, kpack, waves_per_eu, schedule_hint, num_stages, supported_fp8_dtypes, knobs.amd.*) — triton-lang/triton: <https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py>
4. ROCm-specific GEMM tuning params (waves_per_eu, kpack, matrix_instr_nonkdim in TorchInductor) — pytorch/pytorch PR #143286: <https://github.com/pytorch/pytorch/pull/143286>
5. `triton.Config` / `triton.autotune` API (num_warps, num_stages, prune_configs_by) — Triton docs: <https://triton-lang.org/main/python-api/generated/triton.Config.html>
6. Enabling vLLM V1 on AMD GPUs With Triton (num_warps spill, per-shape tuned configs) — PyTorch blog: <https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/>
7. Matmul performance / matrix_instr_nonkdim & kpack on MI300X — triton-lang/triton issue #4959: <https://github.com/triton-lang/triton/issues/4959>
