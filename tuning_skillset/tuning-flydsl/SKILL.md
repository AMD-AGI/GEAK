---
name: tuning-flydsl
description: Tune FlyDSL kernels on AMD Instinct — build and prune a Config space, set the autotune key, and race FlyDSL against the other backends through aiter's libtype mechanism. Use when the kernel is written in FlyDSL or when deciding whether the FlyDSL path is competitive for a shape.
---

# Tuning FlyDSL kernels

Read `../tuning-core/SKILL.md` first. If you have tuned Triton, read
`../tuning-triton/SKILL.md` too — the autotuning model is deliberately close, and knowing
where it *differs* is most of what this skill adds.

FlyDSL is present in both target images (0.1.4 in vllm, 0.1.5 in sglang) and aiter carries
FlyDSL kernels for GEMM, MoE and linear attention. Like Triton, you tune by **authoring a
search space**; the autotuner races what you give it and nothing more.

## 1. The API mirrors Triton — with one important divergence

```python
import flydsl
flydsl.autotune(configs, key=None, warmup=5, rep=25, prune_configs_by=None,
                reset_to_zero=None, pre_hook=None, post_hook=None, do_bench=None)
flydsl.Config(*, num_warps=None, waves_per_eu=None, maxnreg=None, pre_hook=None, **kwargs)
```

Everything from the Triton skill transfers: configs are raced per distinct `key` tuple,
results are cached, `prune_configs_by` lets you drop candidates before compiling, and
`Config.all_kwargs()` shows what actually reaches the backend.

The divergence worth internalizing:

| | Triton | FlyDSL |
| --- | --- | --- |
| `num_warps` | `Config` keyword | `Config` keyword |
| `waves_per_eu` | **positional dict only** — keyword raises `TypeError` | **`Config` keyword** — first-class |
| tile sizes | positional dict | `**kwargs` |
| `num_stages` | `Config` keyword | not a `Config` field |

So a config that is correct in one framework is wrong in the other, and the error mode
differs: Triton raises `TypeError` for `waves_per_eu=`, while FlyDSL accepts it. Porting a
config list between them by search-and-replace produces silently different kernels. Confirm
with `all_kwargs()` after building the first config in any new space — it is one line and it
settles the question:

```python
cfg = flydsl.Config(BLOCK_M=128, BLOCK_N=128, num_warps=8, waves_per_eu=2)
cfg.all_kwargs()      # everything the backend will see
```

## 2. `key` still decides whether tuning helps or hurts

FlyDSL builds its cache key from the named `key` arguments **plus the dtypes of all
arguments**. Dtypes coming along for free is a genuine convenience — but the shape dimensions
do not. If `M` is not in `key`, an M=1 decode call reuses whatever config was tuned at large
M.

The measured cost of that mistake on the equivalent Triton setup was **3.75×**, silently
(`../tuning-triton/SKILL.md` §3). The mechanism here is the same, so the exposure is the
same. Name every dimension the kernel tiles or branches on.

## 3. Prune, and split by regime

`prune_configs_by` is the hook; the analytic predicate and its measured reduction
(1600 → 912 candidates) are in `../tuning-core/search_strategy.md`.

Regime-splitting matters more than the autotuner's cleverness. On the same M=1 GEMM, racing a
decode-shaped space instead of a general one was **34% faster** — not because the search
improved, but because the right answer was finally in the list
(`../tuning-triton/SKILL.md` §4). The same reasoning applies to any FlyDSL kernel that must
serve both prefill and decode: an autotuner cannot select what you did not offer it.

`warmup=5, rep=25` are the defaults and are low. Raise them, or pass your own `do_bench`, for
small shapes where run-to-run spread is largest.

## 4. The kernel names encode the space

FlyDSL kernel names in aiter's tuned configs are self-describing, which makes them a fast way
to learn the axes that matter. A real row from a shipped config:

```
flydsl_gemm2_abf16_wbf16_bf16_t32x64x128_split_k16_block_m_warp2_block_n_warp2
  _async_copyTrue_b_to_ldsTrue_b_preshuffleFalse_c_to_ldsFalse_gfx950
```

Reading it left to right: bf16 in / bf16 weights / bf16 out, a `32×64×128` tile, **split-K
16**, 2×2 warps per block, async copy on, B staged through LDS, no B preshuffle, no C through
LDS — compiled for **gfx950**.

Two lessons in one string. First, the tunable axes go well beyond tile size: split-K factor,
async copy, and which operands are staged through LDS are all in play, and a space that
varies only `BLOCK_M`/`BLOCK_N` is leaving most of the search unexplored. Second, the trailing
`gfx950` is not decoration — that kernel does not serve a gfx942 box. Which is exactly why
the bf16 configs shipped in the vllm image are unusable on MI300X (`../tuning-aiter/` §1).

Split-K deserves specific attention: it is the main lever for K-heavy and decode shapes,
where there is not enough M×N parallelism to fill the GPU. If your shape corpus includes the
`k_heavy` or `decode` regimes (`../benchmark/shapes.py`), put split-K factors in the space.

## 5. Racing FlyDSL against everything else

The decisive question is usually not "what is the best FlyDSL config" but "is FlyDSL the
right backend for this shape". aiter answers it directly:

```bash
python3 gradlib/gradlib/gemm_tuner.py \
    --input_file /tmp/untuned.csv --tuned_file /tmp/tuned.csv
```

One run races FlyDSL against hipBLASLt, Triton, CK and torch on your shapes and records the
winner per shape in a `libtype` column. There is no flag to select which backends run, and
`--indtype` raises at the aiter commits currently in play — put the dtype in the input CSV
instead. Both traps are documented in `../tuning-aiter/` §4, which is also where you will find
where the resulting CSV must be written for the runtime to read it.

Measured on gfx950 at 4096³ bf16, so you know what FlyDSL is competing against on MI355:

| backend | TFLOPS |
| --- | --- |
| hipBLASLt (winner replayed by index) | 1241 |
| CK (best of 126 instances) | 1245 |
| torch | 1285 |
| **FlyDSL** (`flydsl_hgemm`, `auto_shuffle_b=True`) | **880** |
| aiter Triton `gemm_a16w16` | 810 |

FlyDSL sits between the vendor libraries and the Triton path on the square shape — ahead of
Triton, ~30% behind hipBLASLt. Two things follow. On a plain large GEMM, racing is likely to
pick hipBLASLt, so FlyDSL work is better spent on shapes and fusions the vendor library does
not serve well. And the corpus's `gemm_flydsl_hgemm` case still found a 12.5% REAL uplift on
gfx950 from tuning alone, so the gap above is not a tuned-versus-tuned comparison — it is a
starting point.

With `AITER_LOG_TUNED_CONFIG=1`, the per-call log names the winning `libtype`, so you can see
whether FlyDSL is actually being selected in the live path rather than only in the bench.

## 6. Architecture

FlyDSL kernels compile per-arch and the arch is baked into the kernel name. A tuned FlyDSL
config never transfers between gfx942 and gfx950 — not as a matter of policy, but because the
artifact names a target. Re-tune per architecture, and expect the *winning shape of config*
to shift too: CU count differs (304 vs 256), so the split-K and tile choices that balanced one
device will not balance the other.

Verified on gfx950: `flydsl_hgemm` compiles and runs, the tuning knobs in §2 are all live, and
tuning them is worth 12.5% over the default on 4096³ bf16. What did *not* survive the move is
the winning config itself — as predicted above.

## Checklist

- [ ] `all_kwargs()` checked once per new space — `waves_per_eu` is a keyword here, unlike Triton
- [ ] `key` names every dimension the kernel tiles or branches on
- [ ] space includes split-K, async copy and LDS staging, not just tile sizes
- [ ] space pruned analytically; regimes raced separately
- [ ] `warmup`/`rep` raised above defaults for small shapes
- [ ] raced against `torch` and the other libtypes before committing to FlyDSL
- [ ] arch of the tuned artifact matches the deployment target
- [ ] engagement proven (`../tuning-core/engagement_verification.md`)
